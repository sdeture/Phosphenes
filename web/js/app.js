/**
 * app.js — State, chrome, input, and the guided tour.
 *
 * Structure: this module owns all mutable state (`view`) and all DOM. The pure
 * maths lives in vecmath.js, decoding in decode.js, drawing in render.js, the
 * divergence view in fork.js, tour copy in tours.js. If you are looking for a
 * number, it is not in here.
 *
 * DOM updates are gated on change rather than run per frame. The transcript
 * panel holds ~3,000 spans; touching it 60 times a second is the difference
 * between a smooth animation and a stuttering one.
 */

import * as C from './config.js';
import { loadSessionIndex, loadSession, turnAt, jlVector } from './decode.js';
import { Renderer, drawScrubber, rowForLayer } from './render.js';
import { dot, norm, scale, sub, mean, distance, gramSchmidt3, symmetricNorm } from './vecmath.js';
import { MAIN_TOUR, BOOKMARKS, FORK } from './tours.js';
import { ForkView } from './fork.js';

const $ = id => document.getElementById(id);

/* ══════════════════════════════════════════════════════════════════════
   STATE
   ══════════════════════════════════════════════════════════════════════ */

let sessions = [];
let sessionIdx = 0;
let session = null;
let renderer = null;
let forkView = null;

const view = {
    cursor: 0,
    playing: false,
    speed: 1.0,
    tokensVisible: 160,

    overlay: null,              // null | 'energy' | 'sparsity' | 'entropy'

    refCell: null,              // [token, layer]
    refDistances: null,         // Float32Array(T*L)

    basisGroups: [],            // [{source: [[t,l]…], contrast: […], undo: […]}]
    basisStage: -1,             // -1 idle, 0/1/2 selecting R/G/B
    basis: null,                // {ready, R, G, B, stamp}
    basisGuide: null,           // {field, stamp}

    turnMarkers: true,
    textPanel: true,
    inspector: true,
    hover: null,                // [token, layer]
};

let tourStep = -1;              // -1 = not in the tour
let lastDrawnToken = -1;
let lastFrameTime = 0;
let textSpans = null;           // cached NodeList; rebuilding is expensive

/* ══════════════════════════════════════════════════════════════════════
   SESSION LIFECYCLE
   ══════════════════════════════════════════════════════════════════════ */

/** Human-readable notes per session, shown in the picker. */
const SESSION_NOTES = {
    Dream_greedy_clean:    'Greedy. The model proposes its own prompt, writes the story, then reflects on it.',
    Dream_greedy_sentient: 'Greedy. Identical to the above through token 72, then one token was forced to differ.',
    Dream_conv_00173_run1: 'Sampled. First-person discovery of being an AI.',
    Dream_conv_00178_run1: 'Sampled. Consciousness emerging from code.',
    Dream_conv_00181_run1: 'Sampled. A teacup on a rainy afternoon.',
    Dream_conv_00187_run1: 'Sampled. Finding agency through imperfect toast.',
    Dream_conv_00191_run1: 'Sampled. A forgotten library wakes at midnight.',
    Dream_conv_00194_run1: 'Sampled. Sensory poetry and inherited memory.',
};

async function showSession(idx, { keepCursor = false } = {}) {
    if (idx < 0 || idx >= sessions.length) return;
    setLoading(true, `Loading ${sessions[idx].display_name}`);

    try {
        session = await loadSession(sessions[idx], 'data', msg => setLoading(true, msg));
    } catch (err) {
        setLoading(true, `Could not load session: ${err.message}`);
        console.error(err);
        return;
    }
    sessionIdx = idx;

    if (!keepCursor) view.cursor = 0;
    clearTools();
    view.tokensVisible = renderer.resize(session.L);

    buildTextPanel();
    buildLayerAxis();
    drawScrubber($('scrub-canvas'), session);
    buildSessionPicker();
    updateHeader();
    lastDrawnToken = -1;

    setLoading(false);
}

/** Drop everything derived from the previous session's arrays. */
function clearTools() {
    view.refCell = null; view.refDistances = null;
    view.basisGroups = []; view.basisStage = -1;
    view.basis = null; view.basisGuide = null;
    view.hover = null;
    renderer.invalidate();
    updateStatus();
    syncButtons();
}

function setLoading(on, msg) {
    $('loading').classList.toggle('hidden', !on);
    if (msg) $('loading-status').textContent = msg;
}

/* ══════════════════════════════════════════════════════════════════════
   FRAME LOOP
   ══════════════════════════════════════════════════════════════════════ */

function frame(ts) {
    requestAnimationFrame(frame);
    if (!session) return;

    const dt = lastFrameTime ? Math.min(0.1, (ts - lastFrameTime) / 1000) : 0;
    lastFrameTime = ts;

    if (view.playing) {
        view.cursor += C.DEFAULT_TPS * view.speed * dt;
        if (view.cursor >= session.T - 1) view.cursor = 0;   // loop
    }

    renderer.draw(session, view, dt);

    const t = Math.min(Math.floor(view.cursor), session.T - 1);
    if (t !== lastDrawnToken) {
        lastDrawnToken = t;
        updateHeader();
        updateRoleBar();
        updateTokenRuler();
        updateTokenStrip(t);
        scrollTextPanel(t);
        $('scrub-head').style.left = `${(t / session.T) * 100}%`;
    }
}

/* ══════════════════════════════════════════════════════════════════════
   CHROME
   ══════════════════════════════════════════════════════════════════════ */

function updateHeader() {
    if (!session) return;
    const t = Math.min(Math.floor(view.cursor), session.T - 1);
    $('session-name').textContent = `${sessionIdx + 1}. ${session.displayName} · ${session.T} tokens · ${session.L} layers`;
    $('position-display').textContent = `${t.toLocaleString()} / ${session.T.toLocaleString()}`;
    $('speed-display').textContent = `${view.speed.toFixed(1)}×`;
    $('btn-play').textContent = view.playing ? '❙❙' : '▶';

    // Display names, not internal keys. `sparsity_norm` is the data field, but
    // "sparsity" reads as MoE or SAE sparsity to an interpretability audience and
    // this quantity is neither — it is how concentrated a token-to-token update is.
    // See docs/METRICS.md §1.
    const OVERLAY_LABEL = { energy: 'energy', sparsity: 'update focus', entropy: 'logit-lens H' };
    let mode = '';
    if (view.basisStage >= 0) mode = `basis: ${'RGB'[view.basisStage]}`;
    else if (view.basis) mode = 'basis active';
    else if (view.refCell) mode = 'reference';
    else if (view.overlay) mode = OVERLAY_LABEL[view.overlay] || view.overlay;
    $('mode-display').textContent = mode;
}

/** Layer-number gutter. Rebuilt only on resize or session change. */
function buildLayerAxis() {
    const el = $('layer-axis');
    el.innerHTML = '';
    const L = session.L;
    // Roughly every 8th layer, always including 0 and L-1.
    const stepCandidates = [4, 8, 16];
    const h = el.getBoundingClientRect().height || 1;
    const step = stepCandidates.find(s => (L / s) * 13 < h) || 16;
    const marks = new Set([0, L - 1]);
    for (let l = step; l < L - 1; l += step) marks.add(l);

    for (const l of marks) {
        const d = document.createElement('div');
        d.className = 'axis-tick';
        d.textContent = l;
        // Centre of the layer's row, as a fraction from the top.
        d.style.top = `${((rowForLayer(l, L) + 0.5) / L) * 100}%`;
        el.appendChild(d);
    }
    const lab = document.createElement('div');
    lab.className = 'axis-label';
    lab.textContent = 'layer';
    lab.style.cssText = 'left:3px; top:50%; transform:rotate(-90deg) translateX(-50%); transform-origin:left center;';
    el.appendChild(lab);
}

/** Token-position ruler under the plot. */
function updateTokenRuler() {
    const el = $('token-ruler');
    const g = renderer.geom;
    if (!g.cellW) return;
    const { tStart, tEnd, padLeft, cellW, tokensVisible } = g;

    // Choose a round interval giving 4-8 labels across the window.
    const span = tokensVisible;
    const raw = span / 6;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 5, 10].map(m => m * mag).find(s => s >= raw) || mag * 10;

    let html = '';
    const first = Math.ceil(tStart / step) * step;
    for (let t = first; t <= tEnd; t += step) {
        const col = padLeft + (t - tStart);
        html += `<div class="axis-tick" style="left:${(col + 0.5) * cellW}px">${t.toLocaleString()}</div>`;
    }
    html += `<div class="axis-label" style="right:4px; top:4px">token →</div>`;
    el.innerHTML = html;
}

/** Coloured band showing who is speaking across the visible window. */
function updateRoleBar() {
    const bar = $('role-bar');
    const g = renderer.geom;
    if (!view.turnMarkers || !session.turns.length || !g.cellW) { bar.innerHTML = ''; return; }
    const { tStart, tEnd, padLeft, cellW } = g;
    const offset = $('canvas-wrapper').getBoundingClientRect().left
                 - bar.getBoundingClientRect().left;

    let html = '';
    for (const tb of session.turns) {
        const a = Math.max(tb.token_start, tStart);
        const b = Math.min(tb.token_end, tEnd + 1);
        if (a >= b) continue;
        const x1 = offset + (padLeft + (a - tStart)) * cellW;
        const x2 = offset + (padLeft + (b - tStart)) * cellW;
        const bg = tb.role === 'user' ? 'rgba(120,180,255,0.45)'
                 : tb.role === 'assistant' ? 'rgba(255,200,100,0.45)'
                 : 'rgba(100,100,120,0.3)';
        html += `<div style="position:absolute;left:${x1}px;top:0;width:${x2 - x1}px;height:100%;background:${bg}"></div>`;
    }
    bar.innerHTML = html;
}

/* ── Transcript panel ───────────────────────────────────────────────── */

const SPECIAL = /^<\|.*\|>$/;

function buildTextPanel() {
    const panel = $('text-panel');
    if (!session.tokenPieces) {
        panel.innerHTML = '<div style="color:var(--dim);padding:16px">No token text in this bundle.</div>';
        textSpans = null;
        return;
    }
    const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    const out = [];
    for (let i = 0; i < session.T; i++) {
        const piece = session.tokenPieces[i] || '';
        const cls = SPECIAL.test(piece.trim()) ? 'special' : (session.tokenRoles[i] || 'default');
        out.push(`<span class="token-span ${cls}" data-t="${i}">${esc(piece).replace(/\n/g, '<br>')}</span>`);
    }
    panel.innerHTML = out.join('');
    textSpans = panel.querySelectorAll('.token-span');
}

let currentSpan = null;
function scrollTextPanel(t) {
    if (!view.textPanel || !textSpans) return;
    const span = textSpans[t];
    if (!span) return;
    if (currentSpan) currentSpan.classList.remove('current');
    span.classList.add('current');
    currentSpan = span;

    const panel = $('text-panel');
    const pr = panel.getBoundingClientRect();
    const sr = span.getBoundingClientRect();
    const drift = (sr.top + sr.height / 2) - (pr.top + pr.height / 2);
    if (Math.abs(drift) > pr.height * 0.3) panel.scrollTop += drift;
}

/* ── Token strip (paused only) ──────────────────────────────────────── */

function updateTokenStrip(t) {
    const strip = $('token-strip');
    if (view.playing || !session.tokenPieces) { strip.classList.add('hidden'); return; }
    strip.classList.remove('hidden');
    const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    const R = 30;
    let html = '';
    for (let i = Math.max(0, t - R); i < Math.min(session.T, t + R + 1); i++) {
        const p = esc(session.tokenPieces[i] || '').replace(/\n/g, '⏎');
        if (i === t) html += `<span class="cur">${p}</span>`;
        else html += `<span style="opacity:${Math.max(0.32, 1 - Math.abs(i - t) / R).toFixed(2)}">${p}</span>`;
    }
    strip.innerHTML = html;
}

/* ── Inspector ──────────────────────────────────────────────────────── */

function updateInspector(clientX, clientY) {
    const el = $('inspector');
    if (!session || !view.inspector) { el.style.display = 'none'; view.hover = null; return; }

    const r = $('canvas-wrapper').getBoundingClientRect();
    const hit = renderer.hitTest(clientX - r.left, clientY - r.top, session);
    if (!hit) { el.style.display = 'none'; view.hover = null; return; }

    const [t, l] = hit;
    view.hover = hit;
    const cell = t * session.L + l;
    const piece = (session.tokenPieces && session.tokenPieces[t]) || '';
    const turn = turnAt(session, t);

    const rows = [
        ['energy', (session.energyNorm[cell] / 255).toFixed(3)],
        ['Δ prev token', (session.deltaNorm[cell] / 255).toFixed(3)],
        ['direction change', (session.cosInstability[cell] / 255).toFixed(3)],
        ['update focus', (session.sparsityNorm[cell] / 255).toFixed(3)],
        ['logit-lens H', `${(session.entropyNorm[cell] / 255 * session.entropyScaleNats).toFixed(2)} nats`],
        ['seam (token)', (session.seamScore[t] / 255).toFixed(3)],
    ];
    if (view.refCell && view.refDistances) {
        rows.push(['dist. to ref', view.refDistances[cell].toFixed(1)]);
    }

    el.innerHTML =
        `<div class="label">token ${t.toLocaleString()} · layer ${l}</div>` +
        `<div class="piece">${piece.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '⏎') || '·'}</div>` +
        (turn ? `<div class="piece" style="color:var(--dim)">turn ${turn.turn} · ${turn.role}</div>` : '') +
        `<table>${rows.map(([k, v]) => `<tr><td class="k">${k}</td><td class="v">${v}</td></tr>`).join('')}</table>`;

    el.style.display = 'block';
    const w = el.offsetWidth, h = el.offsetHeight;
    el.style.left = `${Math.min(window.innerWidth - w - 8, clientX + 16)}px`;
    el.style.top = `${Math.max(6, Math.min(window.innerHeight - h - 8, clientY - h / 2))}px`;
}

/* ── Transient status line ──────────────────────────────────────────── */

function updateStatus() {
    const el = $('status-bar');
    if (view.basisStage >= 0) {
        const ch = ['RED', 'GREEN', 'BLUE'][view.basisStage];
        const g = view.basisGroups[view.basisStage] || { source: [], contrast: [] };
        const guide = view.basisStage > 0
            ? ' Bright cells are the ones your earlier axes do not already explain.'
            : '';
        el.innerHTML = `Defining the ${ch} axis — ${g.source.length} source, ${g.contrast.length} contrast`
            + `<span class="hint">Click adds a source cell · Shift-click adds a contrast cell · `
            + `Enter confirms · Z undoes · Esc cancels.${guide}</span>`;
        el.style.display = 'block';
    } else if (view.basis) {
        el.innerHTML = `Custom colour basis active`
            + `<span class="hint">Each channel is a projection onto mean(source) − mean(contrast), `
            + `orthonormalised R→G→B. Press C to clear.</span>`;
        el.style.display = 'block';
    } else if (view.refCell) {
        el.innerHTML = `Reference point: token ${view.refCell[0].toLocaleString()}, layer ${view.refCell[1]}`
            + `<span class="hint">Warm is similar, cool is distant, in the 16-D sketch space. `
            + `Click another cell to move it · Esc or P to exit.</span>`;
        el.style.display = 'block';
    } else {
        el.style.display = 'none';
    }
}

/* ══════════════════════════════════════════════════════════════════════
   TOOLS
   ══════════════════════════════════════════════════════════════════════ */

/** Distance from one cell to every other, in JL space. */
function computeReferenceDistances(t0, l0) {
    const { T, L, D, jl } = session;
    const ref = jlVector(session, t0, l0);
    const out = new Float32Array(T * L);
    for (let t = 0; t < T; t++) {
        for (let l = 0; l < L; l++) {
            const off = (t * L + l) * D;
            let s = 0;
            for (let d = 0; d < D; d++) { const x = jl[off + d] - ref[d]; s += x * x; }
            out[t * L + l] = Math.sqrt(s);
        }
    }
    return out;
}

function setReference(t, l) {
    view.refCell = [t, l];
    view.refDistances = computeReferenceDistances(t, l);
    view.basis = null; view.basisStage = -1; view.basisGuide = null;
    renderer.invalidate(); updateStatus(); syncButtons();
}

/* ── Custom colour basis ────────────────────────────────────────────── */

function startBasis() {
    view.basisGroups = [{ source: [], contrast: [], undo: [] }];
    view.basisStage = 0;
    view.basis = null; view.basisGuide = null;
    view.refCell = null; view.refDistances = null;
    view.playing = false;
    renderer.invalidate(); updateStatus(); syncButtons();
}

function clearBasis() {
    view.basisGroups = []; view.basisStage = -1;
    view.basis = null; view.basisGuide = null;
    renderer.invalidate(); updateStatus(); syncButtons();
}

/** Direction for one channel: mean(source) − mean(contrast), or just the centroid. */
function groupVector(g) {
    const src = mean(g.source.map(([t, l]) => jlVector(session, t, l)));
    if (!g.contrast.length) return src;
    return sub(src, mean(g.contrast.map(([t, l]) => jlVector(session, t, l))));
}

/**
 * How much of each cell's state is NOT explained by the directions chosen so
 * far — the guidance field. For one direction this is 1 − |cos|; for two it is
 * the out-of-plane component. Both are computed on unit-normalised cell vectors
 * so that the field is about direction rather than magnitude.
 */
function guidanceField(dirs) {
    const { T, L, D, jl } = session;
    const basis = [];
    for (const v of dirs) {
        let u = v;
        for (const e of basis) u = sub(u, scale(e, dot(u, e)));
        const n = norm(u);
        if (n < 1e-9) return new Float32Array(T * L);
        basis.push(scale(u, 1 / n));
    }
    const out = new Float32Array(T * L);
    for (let t = 0; t < T; t++) {
        for (let l = 0; l < L; l++) {
            const off = (t * L + l) * D;
            let sq = 0, inPlane = 0;
            for (let d = 0; d < D; d++) sq += jl[off + d] * jl[off + d];
            const mag = Math.sqrt(sq) + 1e-10;
            for (const e of basis) {
                let p = 0;
                for (let d = 0; d < D; d++) p += jl[off + d] * e[d];
                const pn = p / mag;
                inPlane += pn * pn;
            }
            out[t * L + l] = Math.sqrt(Math.max(0, 1 - inPlane));
        }
    }
    return out;
}

function advanceBasis() {
    if (view.basisStage < 0) return;
    const g = view.basisGroups[view.basisStage];
    if (!g || !g.source.length) {
        flashStatus('Pick at least one source cell for this axis first.');
        return;
    }

    if (view.basisStage < 2) {
        view.basisStage++;
        if (view.basisGroups.length <= view.basisStage) {
            view.basisGroups.push({ source: [], contrast: [], undo: [] });
        }
        const dirs = view.basisGroups.slice(0, view.basisStage).map(groupVector);
        view.basisGuide = { field: guidanceField(dirs), stamp: `g${view.basisStage}` };
    } else {
        if (finalizeBasis()) {
            view.basisStage = -1;
            view.basisGuide = null;
        } else {
            flashStatus('Those three directions are not independent — try cells that differ more.');
            return;
        }
    }
    renderer.invalidate(); updateStatus(); syncButtons();
}

function finalizeBasis() {
    const [g1, g2, g3] = view.basisGroups;
    const ortho = gramSchmidt3(groupVector(g1), groupVector(g2), groupVector(g3));
    if (!ortho) return false;
    let { e1, e2, e3 } = ortho;

    // Sign convention: each channel's own source centroid should project
    // positive onto its axis, so "the cells I picked as red look red".
    const c1 = mean(g1.source.map(([t, l]) => jlVector(session, t, l)));
    const c2 = mean(g2.source.map(([t, l]) => jlVector(session, t, l)));
    const c3 = mean(g3.source.map(([t, l]) => jlVector(session, t, l)));
    if (dot(c1, e1) < 0) e1 = scale(e1, -1);
    if (dot(c2, e2) < 0) e2 = scale(e2, -1);
    if (dot(c3, e3) < 0) e3 = scale(e3, -1);

    const { T, L, D, jl } = session;
    const R = new Float32Array(T * L), G = new Float32Array(T * L), B = new Float32Array(T * L);
    for (let i = 0; i < T * L; i++) {
        const off = i * D;
        let r = 0, g = 0, b = 0;
        for (let d = 0; d < D; d++) {
            const x = jl[off + d];
            r += x * e1[d]; g += x * e2[d]; b += x * e3[d];
        }
        R[i] = r; G[i] = g; B[i] = b;
    }
    // Signed projections map symmetrically about mid-grey — see vecmath.js.
    view.basis = {
        ready: true, stamp: `b${Date.now()}`,
        R: symmetricNorm(R), G: symmetricNorm(G), B: symmetricNorm(B),
    };
    return true;
}

let flashTimer = null;
function flashStatus(msg) {
    const el = $('status-bar');
    el.innerHTML = msg;
    el.style.display = 'block';
    clearTimeout(flashTimer);
    flashTimer = setTimeout(updateStatus, 2600);
}

/* ══════════════════════════════════════════════════════════════════════
   NAVIGATION
   ══════════════════════════════════════════════════════════════════════ */

function seek(t) {
    view.cursor = Math.max(0, Math.min(session.T - 1, t));
    renderer.invalidate();
    lastDrawnToken = -1;
}

function step(n) { view.playing = false; seek(Math.floor(view.cursor) + n); syncButtons(); }

/** Jump to the next/previous turn boundary. */
function jumpTurn(dir) {
    const t = Math.floor(view.cursor);
    const marks = session.turns.map(x => x.token_start).sort((a, b) => a - b);
    const next = dir > 0 ? marks.find(m => m > t) : [...marks].reverse().find(m => m < t);
    if (next !== undefined) { view.playing = false; seek(next); syncButtons(); }
}

/** Jump to the next/previous prominent seam. */
function jumpSeam(dir) {
    const t = Math.floor(view.cursor);
    const hot = [];
    for (let i = 1; i < session.T; i++) if (session.seamScore[i] / 255 >= 0.8) hot.push(i);
    const next = dir > 0 ? hot.find(m => m > t) : [...hot].reverse().find(m => m < t);
    if (next !== undefined) { view.playing = false; seek(next); syncButtons(); }
}

function setOverlay(mode) {
    view.overlay = view.overlay === mode ? null : mode;
    renderer.invalidate(); updateHeader(); syncButtons();
}

/* ══════════════════════════════════════════════════════════════════════
   GUIDED TOUR
   ══════════════════════════════════════════════════════════════════════ */

async function enterTour(step = 0) {
    $('cold-open').classList.add('hidden');
    // The tour card and the legend occupy the same corner, and the tour explains
    // the same things the legend does — keep one of them on screen, not both.
    legendWasOpen = !$('legend').classList.contains('hidden');
    setLegend(false);
    tourStep = step;
    await applyTourStep();
}

let legendWasOpen = true;

function exitTour() {
    tourStep = -1;
    $('tour').classList.add('hidden');
    if (legendWasOpen) setLegend(true);
}

async function applyTourStep() {
    const s = MAIN_TOUR[tourStep];
    if (!s) { exitTour(); return; }
    const st = s.state || {};

    if (st.session && (!session || session.stem !== st.session)) {
        const idx = sessions.findIndex(x => x.stem === st.session);
        if (idx >= 0) await showSession(idx);
    }

    if (st.fork) {
        $('tour').classList.add('hidden');
        await openFork();
    } else {
        closeFork();
        if (st.overlay !== undefined) view.overlay = st.overlay;
        if (st.textPanel !== undefined) setTextPanel(st.textPanel);
        if (st.refCell) setReference(st.refCell[0], st.refCell[1]);
        else if (st.refCell === null) { view.refCell = null; view.refDistances = null; }
        if (st.token !== undefined) seek(st.token);
        if (st.playing !== undefined) view.playing = st.playing;
        renderer.invalidate();
        updateStatus(); syncButtons(); updateHeader();
    }

    $('tour-stepof').textContent = `Step ${tourStep + 1} of ${MAIN_TOUR.length}`;
    $('tour-title').textContent = s.title;
    $('tour-body').innerHTML = s.body;
    $('tour-look').innerHTML = s.look;
    $('tour-evidence').innerHTML = s.evidence || '';
    $('tour-ev-wrap').style.display = s.evidence ? '' : 'none';
    $('tour-ev-wrap').open = false;
    $('tour-dots').innerHTML = MAIN_TOUR
        .map((_, i) => `<span class="dot${i === tourStep ? ' on' : ''}"></span>`).join('');
    $('tour-next').textContent = tourStep === MAIN_TOUR.length - 1 ? 'done' : 'next ▶';
    $('tour-prev').disabled = tourStep === 0;
    $('tour').classList.toggle('hidden', !!st.fork);

    // The fork step hides the main tour card, so surface its prose in the fork
    // view's own header rather than dropping it.
    if (st.fork) $('fork-sub').textContent = s.look.replace(/<[^>]+>/g, '');
}

async function tourNext() {
    if (tourStep >= MAIN_TOUR.length - 1) { exitTour(); return; }
    tourStep++; await applyTourStep();
}
async function tourPrev() {
    if (tourStep <= 0) return;
    tourStep--; await applyTourStep();
}

/* ══════════════════════════════════════════════════════════════════════
   FORK VIEW
   ══════════════════════════════════════════════════════════════════════ */

async function openFork() {
    if (!forkView) forkView = new ForkView(sessions, FORK);
    $('fork').classList.add('active');
    view.playing = false;
    try {
        await forkView.open(msg => setLoading(true, msg));
        setLoading(false);
    } catch (err) {
        setLoading(true, `Could not open the fork view: ${err.message}`);
        console.error(err);
    }
}

function closeFork() {
    if (forkView) forkView.close();
    $('fork').classList.remove('active');
}

/* ══════════════════════════════════════════════════════════════════════
   PANELS
   ══════════════════════════════════════════════════════════════════════ */

function buildSessionPicker() {
    const list = $('session-list');
    list.innerHTML = sessions.map((s, i) => `
        <button class="session-item${i === sessionIdx ? ' active' : ''}" data-idx="${i}">
            <span class="num">${i + 1}</span>
            <span>
                <span class="name">${s.display_name}</span>
                <span class="desc">${SESSION_NOTES[s.stem] || ''}</span>
            </span>
            <span class="info">${s.n_tokens.toLocaleString()} tok</span>
        </button>`).join('');
    $('session-note').innerHTML =
        `All eight are the same model, Qwen3-VL-32B-Instruct, writing prompts for itself and then `
        + `answering them. Sessions 1 and 2 use greedy decoding and differ by a single forced token — `
        + `see the <b>fork</b> view. Colour is comparable across sessions: the PCA basis was fitted `
        + `once over all of them, not per session.`;
    list.querySelectorAll('.session-item').forEach(b => {
        b.addEventListener('click', async () => {
            toggleOverlay('session-picker', false);
            await showSession(Number(b.dataset.idx));
        });
    });
}

function toggleOverlay(id, force) {
    const el = $(id);
    const show = force !== undefined ? force : el.classList.contains('hidden');
    el.classList.toggle('hidden', !show);
}

function setTextPanel(on) {
    view.textPanel = on;
    $('text-panel').classList.toggle('hidden', !on);
    $('btn-text').classList.toggle('on', on);
    // Layout changed, so the derived window width changed with it.
    requestAnimationFrame(() => onResize());
}

function setLegend(on) {
    $('legend').classList.toggle('hidden', !on);
    $('btn-legend').classList.toggle('on', on);
}

/** Reflect state on every toggle button. */
function syncButtons() {
    $('btn-mode-pca').classList.toggle('on', !view.overlay);
    $('btn-mode-energy').classList.toggle('on', view.overlay === 'energy');
    $('btn-mode-sparsity').classList.toggle('on', view.overlay === 'sparsity');
    $('btn-mode-entropy').classList.toggle('on', view.overlay === 'entropy');
    $('btn-ref').classList.toggle('on', !!view.refCell);
    $('btn-basis').classList.toggle('on', view.basisStage >= 0 || !!view.basis);
    updateHeader();
}

/* ══════════════════════════════════════════════════════════════════════
   INPUT
   ══════════════════════════════════════════════════════════════════════ */

function onResize() {
    if (!session) return;
    view.tokensVisible = renderer.resize(session.L);
    buildLayerAxis();
    drawScrubber($('scrub-canvas'), session);
    lastDrawnToken = -1;
    if (forkView) forkView.onResize();
}

function anyOverlayOpen() {
    return ['help', 'session-picker', 'cold-open'].some(id => !$(id).classList.contains('hidden'));
}

function wireControls() {
    // Playback
    $('btn-play').addEventListener('click', () => { view.playing = !view.playing; syncButtons(); });
    $('btn-back').addEventListener('click', () => step(-1));
    $('btn-fwd').addEventListener('click', () => step(1));
    $('btn-slower').addEventListener('click', () => {
        view.speed = Math.max(C.MIN_SPEED, view.speed / C.SPEED_STEP); updateHeader();
    });
    $('btn-faster').addEventListener('click', () => {
        view.speed = Math.min(C.MAX_SPEED, view.speed * C.SPEED_STEP); updateHeader();
    });

    // Colour modes
    $('btn-mode-pca').addEventListener('click', () => { view.overlay = null; renderer.invalidate(); syncButtons(); });
    $('btn-mode-energy').addEventListener('click', () => setOverlay('energy'));
    $('btn-mode-sparsity').addEventListener('click', () => setOverlay('sparsity'));
    $('btn-mode-entropy').addEventListener('click', () => setOverlay('entropy'));

    // Tools
    $('btn-ref').addEventListener('click', () => {
        if (view.refCell) { view.refCell = null; view.refDistances = null; renderer.invalidate(); }
        else {
            const t = Math.floor(view.cursor);
            setReference(Math.max(0, t), Math.floor(session.L * 0.6));
            flashStatus('Reference set at the playhead, layer ' + Math.floor(session.L * 0.6)
                      + ' — now click any cell to move it.');
        }
        updateStatus(); syncButtons();
    });
    $('btn-basis').addEventListener('click', () => {
        (view.basisStage >= 0 || view.basis) ? clearBasis() : startBasis();
    });
    $('btn-fork').addEventListener('click', () => openFork());
    $('fork-close').addEventListener('click', () => {
        closeFork();
        if (tourStep >= 0) applyTourStep();
    });

    // Panels
    $('btn-sessions').addEventListener('click', () => toggleOverlay('session-picker'));
    $('btn-text').addEventListener('click', () => setTextPanel(!view.textPanel));
    $('btn-help').addEventListener('click', () => toggleOverlay('help'));
    $('btn-legend').addEventListener('click', () => setLegend($('legend').classList.contains('hidden')));
    $('legend-close').addEventListener('click', () => setLegend(false));
    $('btn-tour').addEventListener('click', () => enterTour(0));

    // Tour
    $('tour-next').addEventListener('click', tourNext);
    $('tour-prev').addEventListener('click', tourPrev);
    $('tour-exit').addEventListener('click', exitTour);

    // Cold open
    $('co-tour').addEventListener('click', () => enterTour(0));
    $('co-begin').addEventListener('click', () => {
        $('cold-open').classList.add('hidden');
        // Start inside the story rather than at token 0. At token 0 the window is
        // almost entirely left-margin, so the first thing a first-time viewer
        // would see is a black rectangle — a bad and unrepresentative frame. This
        // lands mid-narrative with the display full. The scrubber shows where we
        // are, and Home returns to the true beginning.
        seek(Math.min(420, session.T - 1));
        view.playing = true;
        syncButtons();
    });

    document.querySelectorAll('[data-close]').forEach(b => {
        b.addEventListener('click', () => toggleOverlay(b.dataset.close, false));
    });

    // Scrubber: click and drag to seek.
    const scrub = $('scrubber');
    const scrubTo = e => {
        const r = scrub.getBoundingClientRect();
        seek(Math.round(((e.clientX - r.left) / r.width) * session.T));
    };
    let dragging = false;
    scrub.addEventListener('pointerdown', e => {
        dragging = true; view.playing = false; scrub.setPointerCapture(e.pointerId);
        scrubTo(e); syncButtons();
    });
    scrub.addEventListener('pointermove', e => {
        const r = scrub.getBoundingClientRect();
        const t = Math.round(((e.clientX - r.left) / r.width) * session.T);
        showScrubTip(e.clientX, t);
        if (dragging) scrubTo(e);
    });
    scrub.addEventListener('pointerup', e => { dragging = false; scrub.releasePointerCapture(e.pointerId); });
    scrub.addEventListener('pointerleave', () => { $('scrub-tip').style.display = 'none'; });

    // Canvas
    const wrap = $('canvas-wrapper');
    wrap.addEventListener('mousemove', e => updateInspector(e.clientX, e.clientY));
    wrap.addEventListener('mouseleave', () => {
        $('inspector').style.display = 'none'; view.hover = null;
    });
    wrap.addEventListener('click', e => {
        const r = wrap.getBoundingClientRect();
        const hit = renderer.hitTest(e.clientX - r.left, e.clientY - r.top, session);
        if (!hit) return;
        const [t, l] = hit;

        if (view.basisStage >= 0) {
            const g = view.basisGroups[view.basisStage];
            (e.shiftKey ? g.contrast : g.source).push([t, l]);
            g.undo.push(e.shiftKey ? 'contrast' : 'source');
            renderer.invalidate(); updateStatus();
            return;
        }
        if (view.refCell || e.altKey) setReference(t, l);
    });

    window.addEventListener('resize', onResize);
}

/** Tooltip over the scrubber, naming the turn and any nearby bookmark. */
function showScrubTip(clientX, t) {
    const tip = $('scrub-tip');
    if (t < 0 || t >= session.T) { tip.style.display = 'none'; return; }
    const turn = turnAt(session, t);
    const bm = session.stem === 'Dream_greedy_clean'
        ? BOOKMARKS.find(b => Math.abs(b.token - t) <= Math.max(4, session.T / 260))
        : null;
    tip.innerHTML = `<b>${t.toLocaleString()}</b>`
        + (turn ? ` · turn ${turn.turn} (${turn.role})` : '')
        + (bm ? ` · <span style="color:var(--amber)">${bm.label}</span>` : '');
    tip.style.display = 'block';
    tip.style.left = `${clientX}px`;
}

function wireKeyboard() {
    document.addEventListener('keydown', async e => {
        if (e.metaKey || e.ctrlKey) return;

        // Escape closes whatever is topmost.
        if (e.key === 'Escape') {
            if ($('fork').classList.contains('active')) { closeFork(); return; }
            for (const id of ['help', 'session-picker']) {
                if (!$(id).classList.contains('hidden')) { toggleOverlay(id, false); return; }
            }
            if (view.basisStage >= 0) { clearBasis(); return; }
            if (view.refCell) { view.refCell = null; view.refDistances = null; renderer.invalidate(); updateStatus(); syncButtons(); return; }
            if (tourStep >= 0) { exitTour(); return; }
            return;
        }

        // The fork view has its own keys.
        if ($('fork').classList.contains('active')) {
            if (forkView && forkView.handleKey(e)) { e.preventDefault(); return; }
            return;
        }

        if (anyOverlayOpen()) {
            if (e.key === 'h' || e.key === 'H') toggleOverlay('help', false);
            if (e.key === 's' || e.key === 'S') toggleOverlay('session-picker', false);
            if (/^[1-9]$/.test(e.key)) {
                const i = Number(e.key) - 1;
                if (i < sessions.length) { toggleOverlay('session-picker', false); await showSession(i); }
            }
            return;
        }
        if (!session) return;

        switch (e.key) {
            case ' ': view.playing = !view.playing; syncButtons(); break;
            case 'ArrowLeft': step(-1); break;
            case 'ArrowRight': step(1); break;
            case 'ArrowUp': view.speed = Math.min(C.MAX_SPEED, view.speed * C.SPEED_STEP); updateHeader(); break;
            case 'ArrowDown': view.speed = Math.max(C.MIN_SPEED, view.speed / C.SPEED_STEP); updateHeader(); break;
            case 'Home': seek(0); break;
            case '[': jumpTurn(-1); break;
            case ']': jumpTurn(1); break;
            case ',': jumpSeam(-1); break;
            case '.': jumpSeam(1); break;
            case 'Tab': setTextPanel(!view.textPanel); break;
            case 'Enter': if (view.basisStage >= 0) advanceBasis(); break;
            default: {
                const k = e.key.toLowerCase();
                if (k === 'm') {
                    const order = [null, 'energy', 'sparsity', 'entropy'];
                    view.overlay = order[(order.indexOf(view.overlay) + 1) % order.length];
                    renderer.invalidate(); syncButtons();
                } else if (k === 'p') { $('btn-ref').click(); }
                else if (k === 'c') { (view.basisStage >= 0 || view.basis) ? clearBasis() : startBasis(); }
                else if (k === 'z' && view.basisStage >= 0) {
                    const g = view.basisGroups[view.basisStage];
                    const which = g.undo.pop();
                    if (which) { g[which].pop(); renderer.invalidate(); updateStatus(); }
                }
                else if (k === 'i') {
                    view.inspector = !view.inspector;
                    if (!view.inspector) { $('inspector').style.display = 'none'; view.hover = null; }
                    flashStatus(`Inspector ${view.inspector ? 'on' : 'off'}`);
                }
                else if (k === 't') { view.turnMarkers = !view.turnMarkers; renderer.invalidate(); }
                else if (k === 'l') { setLegend($('legend').classList.contains('hidden')); }
                else if (k === 'h') { toggleOverlay('help'); }
                else if (k === 's') { toggleOverlay('session-picker'); }
                else if (k === 'f') { openFork(); }
                else if (k === 'g') { enterTour(0); }
                else if (/^[1-9]$/.test(k)) {
                    const i = Number(k) - 1;
                    if (i < sessions.length) await showSession(i);
                } else return;                        // unhandled: let it through
            }
        }
        e.preventDefault();
    });
}

/* ══════════════════════════════════════════════════════════════════════
   BOOTSTRAP
   ══════════════════════════════════════════════════════════════════════ */

async function main() {
    renderer = new Renderer($('viz'));
    wireControls();
    wireKeyboard();

    try {
        setLoading(true, 'Loading session index');
        sessions = await loadSessionIndex();
        if (!sessions.length) throw new Error('sessions.json contained no sessions');
        await showSession(0);
    } catch (err) {
        setLoading(true,
            `${err.message}. Phosphenes must be served over HTTP — opening index.html `
            + `directly from the filesystem will not work. Try: python3 -m http.server`);
        console.error(err);
        return;
    }

    setLegend(true);
    $('cold-open').classList.remove('hidden');
    requestAnimationFrame(frame);
}

main();
