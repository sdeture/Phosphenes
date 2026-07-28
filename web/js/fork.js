/**
 * fork.js — The one-token fork: two runs side by side, and what separates them.
 *
 * ── The experiment ────────────────────────────────────────────────────
 *
 * Two runs of Qwen3-VL-32B-Instruct on the same prompt with greedy (argmax)
 * decoding, so both are deterministic and neither carries sampling noise. The
 * model had been asked what it would like to be asked, and was in the middle of
 * writing its own prompt — "Tell me a story about a ___". In one run it wrote
 * "library"; in the other, that single token was forced to "sentient". Nothing
 * else differs.
 *
 * The activations for tokens 0-72 are therefore *bit-identical* between the two
 * runs, which this view exploits: the two panes render from separate data and
 * are visibly the same picture until the fork. That is a check on the instrument
 * as much as a display of the result — if the panes differed before token 73,
 * something in the pipeline would be wrong.
 *
 * ── What the divergence strip shows, and what it does not ─────────────
 *
 * The strip plots, per position and per layer, the Euclidean distance between
 * the two runs' JL sketch vectors at that (position, layer).
 *
 * Before the fork this is exactly zero. After it, an honest caveat applies and
 * is stated in the UI: the two runs are no longer processing the same words, so
 * the distance is not "how differently the model handled this word". It is how
 * differently the model is *configured* at the same point in its own output. The
 * two sequences never realign — they are different stories of different lengths
 * — so no token-level alignment is possible or claimed.
 *
 * What survives that caveat, and is the point of the view, is the *shape* of the
 * divergence: it is small in early layers and very large in late ones. A
 * different word barely changes what the early layers represent, and completely
 * changes what the late layers predict.
 */

import { loadSession } from './decode.js';
import { Renderer, rowForLayer } from './render.js';
import * as C from './config.js';

const $ = id => document.getElementById(id);

export class ForkView {
    /**
     * @param {Array<object>} sessionIndex The full session index.
     * @param {object} spec The FORK descriptor from tours.js.
     */
    constructor(sessionIndex, spec) {
        this.index = sessionIndex;
        this.spec = spec;
        this.left = null;
        this.right = null;
        this.divergence = null;     // Float32Array(minT * L)
        this.divMax = 1;
        this.loaded = false;

        this.rendererL = new Renderer($('fork-canvas-l'));
        this.rendererR = new Renderer($('fork-canvas-r'));
        this.traceCanvas = $('fork-trace-canvas');

        this.cursor = 0;
        this.playing = false;
        this.raf = null;
        this.lastTs = 0;
        this.lastToken = -1;

        this._wire();
    }

    /* ────────────────────────────────────────────────────────────────
       LOADING
       ──────────────────────────────────────────────────────────────── */

    async open(onProgress = () => {}) {
        if (!this.loaded) {
            const findInfo = stem => {
                const info = this.index.find(s => s.stem === stem);
                if (!info) throw new Error(`session "${stem}" is not in sessions.json`);
                return info;
            };
            onProgress('Loading both runs of the fork');
            // Sequential rather than parallel: two 6 MB bundles decoding at once
            // spikes memory and the progress message becomes a lie.
            this.left = await loadSession(findInfo(this.spec.left.stem), 'data', onProgress);
            this.right = await loadSession(findInfo(this.spec.right.stem), 'data', onProgress);

            onProgress('Measuring divergence');
            this._computeDivergence();
            this._verifyPrefix();
            this._fillCopy();
            this.loaded = true;
        }

        this.lastToken = -1;
        this.onResize();
        // Start at the first position where the window is completely full. The
        // fork is at token 73, so any earlier cursor leaves most of both panes as
        // empty left margin — and the whole point is to compare two pictures,
        // which needs both of them to have pixels in. At this position the
        // identical prefix occupies the left of the panes and the divergence the
        // right, both visible at once.
        this.cursor = Math.max(this.spec.forkAt, this.tokensVisible - 1);
        this._drawTrace();
        this._drawScrub();
        if (!this.raf) this.raf = requestAnimationFrame(ts => this._frame(ts));
    }

    close() {
        this.playing = false;
        if (this.raf) { cancelAnimationFrame(this.raf); this.raf = null; }
    }

    /**
     * Per-(position, layer) distance between the two runs in JL space.
     *
     * Only positions present in both runs are compared; the longer run's tail has
     * nothing to be compared against and is left out of the strip rather than
     * padded, because padding would draw a value that does not exist.
     */
    _computeDivergence() {
        const L = this.left.L, D = this.left.D;
        if (this.right.L !== L || this.right.D !== D) {
            throw new Error('the two runs have different layer counts; they are not comparable');
        }
        const n = Math.min(this.left.T, this.right.T);
        const out = new Float32Array(n * L);
        const a = this.left.jl, b = this.right.jl;

        let max = 0;
        for (let t = 0; t < n; t++) {
            for (let l = 0; l < L; l++) {
                const off = (t * L + l) * D;
                let s = 0;
                for (let d = 0; d < D; d++) { const x = a[off + d] - b[off + d]; s += x * x; }
                const v = Math.sqrt(s);
                out[t * L + l] = v;
                if (v > max) max = v;
            }
        }
        this.divergence = out;
        this.divN = n;
        this.divMax = max || 1;

        // Display scale for the strip: the 99th percentile of POST-FORK values,
        // not the maximum. The maximum sits at the deepest layers, and dividing
        // by it — then compressing with a square root, as an earlier version did
        // — flattened the very thing worth seeing, which is that divergence is
        // small in early layers and enormous in late ones. A linear ramp against
        // a high percentile keeps that gradient legible and lets the handful of
        // values above the percentile clip, which costs nothing here because they
        // are all in the same place.
        const post = out.subarray(this.spec.forkAt * L);
        const sorted = Float32Array.from(post).sort();
        this.divScale = sorted[Math.floor(sorted.length * 0.99)] || max || 1;

        // Summary figures for the header, measured rather than asserted.
        const forkAt = this.spec.forkAt;
        let atFork = 0;
        for (let l = 0; l < L; l++) atFork += out[forkAt * L + l];
        this.statForkMean = atFork / L;

        let magSum = 0;
        for (let t = 0; t < this.left.T; t++) {
            for (let l = 0; l < L; l++) {
                const off = (t * L + l) * D;
                let s = 0;
                for (let d = 0; d < D; d++) s += a[off + d] * a[off + d];
                magSum += Math.sqrt(s);
            }
        }
        this.statTypicalMag = magSum / (this.left.T * L);
    }

    /**
     * Check that the shared prefix really is identical, and say so in the UI.
     *
     * This is a self-test with a visible result. If it ever fails, the claim the
     * whole view rests on is false, and the view should say that rather than
     * quietly showing two pictures that look similar.
     */
    _verifyPrefix() {
        const L = this.left.L;
        let worst = 0;
        for (let t = 0; t <= this.spec.sharedThrough; t++) {
            for (let l = 0; l < L; l++) worst = Math.max(worst, this.divergence[t * L + l]);
        }
        this.prefixIdentical = worst === 0;
        this.prefixWorst = worst;
    }

    _fillCopy() {
        const s = this.spec;
        // Tokens are shown untrimmed. The leading space is part of the token —
        // trimming it produced "about alibrary", which misrepresents both the
        // tokenisation and the sentence.
        $('fork-prefix').textContent = s.prefixTail;
        $('fork-tokL').textContent = s.left.token;
        $('fork-tokR').textContent = s.right.token;
        $('fork-tag-l').textContent = `“…a ${s.left.label}…”  —  ${this.left.T.toLocaleString()} tokens`;
        $('fork-tag-r').textContent = `“…a ${s.right.label}…”  —  ${this.right.T.toLocaleString()} tokens`;

        $('fork-conts').innerHTML =
            `<span style="color:#7fe0a0">▸</span> <span style="color:var(--dim)">${esc(s.continuations.left)}</span><br>`
            + `<span style="color:#ff9ec4">▸</span> <span style="color:var(--dim)">${esc(s.continuations.right)}</span>`;

        $('fork-stat-shared').textContent = (s.sharedThrough + 1).toLocaleString();
        $('fork-stat-jump').textContent = this.statForkMean.toFixed(0);
        $('fork-stat-scale').textContent = this.statTypicalMag.toFixed(0);

        $('fork-sub').innerHTML = this.prefixIdentical
            ? `tokens 0–${s.sharedThrough} verified bit-identical`
            : `<span style="color:#ff8080">warning: shared prefix differs by up to `
              + `${this.prefixWorst.toFixed(3)} — the panes should be identical here</span>`;
    }

    /* ────────────────────────────────────────────────────────────────
       RENDERING
       ──────────────────────────────────────────────────────────────── */

    onResize() {
        if (!this.loaded) return;
        // Both panes must share one window width so the position axes line up.
        const wL = this.rendererL.resize(this.left.L);
        const wR = this.rendererR.resize(this.right.L);
        this.tokensVisible = Math.min(wL, wR);
        this._buildAxis($('fork-axis-l'), this.left.L);
        this._buildAxis($('fork-axis-r'), this.right.L);
        this._drawTrace();
        this._drawScrub();
        this.lastToken = -1;
    }

    _buildAxis(el, L) {
        const h = el.getBoundingClientRect().height || 1;
        const step = h > 220 ? 16 : 32;
        const marks = new Set([0, L - 1]);
        for (let l = step; l < L - 1; l += step) marks.add(l);
        el.innerHTML = [...marks].map(l =>
            `<div class="axis-tick" style="top:${((rowForLayer(l, L) + 0.5) / L) * 100}%">${l}</div>`
        ).join('');
    }

    _frame(ts) {
        this.raf = requestAnimationFrame(t => this._frame(t));
        if (!this.loaded) return;

        const dt = this.lastTs ? Math.min(0.1, (ts - this.lastTs) / 1000) : 0;
        this.lastTs = ts;
        if (this.playing) {
            this.cursor += C.DEFAULT_TPS * dt;
            if (this.cursor >= this.divN - 1) this.cursor = 0;
        }

        const base = {
            tokensVisible: this.tokensVisible, overlay: null,
            refCell: null, refDistances: null,
            basis: null, basisGroups: null, basisGuide: null,
            turnMarkers: false, hover: null,
        };
        this.rendererL.draw(this.left, { ...base, cursor: this.cursor }, dt);
        this.rendererR.draw(this.right, { ...base, cursor: this.cursor }, dt);

        const t = Math.floor(this.cursor);
        if (t !== this.lastToken) {
            this.lastToken = t;
            this._drawTrace();
            $('fork-scrub-head').style.left = `${(t / this.divN) * 100}%`;
            $('fork-position').textContent = `${t.toLocaleString()} / ${this.divN.toLocaleString()}`;
            $('fork-play').textContent = this.playing ? '❙❙' : '▶';
        }
    }

    /**
     * The divergence strip: layers on y, position on x, black where identical.
     *
     * A perceptually monotonic black-to-amber ramp is used rather than a
     * rainbow: the quantity is a magnitude with a meaningful zero, and a
     * sequential scale is the honest encoding for that. Zero is pure black so
     * the shared prefix reads unmistakably as "nothing here".
     */
    _drawTrace() {
        const canvas = this.traceCanvas;
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.max(1, Math.floor(rect.width * dpr));
        canvas.height = Math.max(1, Math.floor(rect.height * dpr));
        const ctx = canvas.getContext('2d');

        const L = this.left.L;
        const W = this.tokensVisible || 160;
        const tEnd = Math.min(Math.floor(this.cursor), this.divN - 1);
        const tStart = Math.max(0, tEnd - W + 1);
        const padLeft = W - (tEnd - tStart + 1);

        const img = ctx.createImageData(W, L);
        const px = img.data;
        for (let col = 0; col < W; col++) {
            const t = tStart + (col - padLeft);
            for (let row = 0; row < L; row++) {
                const p = (row * W + col) * 4;
                if (t < 0 || t >= this.divN) { px[p + 3] = 0; continue; }
                const layer = L - 1 - row;
                const v = Math.min(1, this.divergence[t * L + layer] / this.divScale);
                px[p]     = Math.min(255, v * 296) | 0;
                px[p + 1] = Math.min(255, v * 208) | 0;
                px[p + 2] = Math.min(255, v * 96) | 0;
                px[p + 3] = 255;
            }
        }
        const off = document.createElement('canvas');
        off.width = W; off.height = L;
        off.getContext('2d').putImageData(img, 0, 0);

        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(off, 0, 0, W, L, 0, 0, canvas.width, canvas.height);

        // Mark the fork if it is on screen.
        const forkCol = padLeft + (this.spec.forkAt - tStart);
        if (forkCol >= 0 && forkCol < W) {
            const x = (forkCol / W) * canvas.width;
            ctx.strokeStyle = 'rgba(255,255,255,0.75)';
            ctx.lineWidth = 1 * dpr;
            ctx.setLineDash([4 * dpr, 3 * dpr]);
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
            ctx.setLineDash([]);
        }
    }

    /** Scrubber for the fork view: just the fork marker and position. */
    _drawScrub() {
        const canvas = $('fork-scrub-canvas');
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.max(1, Math.floor(rect.width * dpr));
        canvas.height = Math.max(1, Math.floor(rect.height * dpr));
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        const w = rect.width, h = rect.height;
        ctx.clearRect(0, 0, w, h);

        // Mean divergence over layers, as a filled profile.
        const L = this.left.L;
        ctx.fillStyle = 'rgba(255,200,100,0.32)';
        const cols = Math.floor(w);
        for (let c = 0; c < cols; c++) {
            const t = Math.floor((c / cols) * this.divN);
            let s = 0;
            for (let l = 0; l < L; l++) s += this.divergence[t * L + l];
            const v = (s / L) / this.divMax;
            const bh = Math.max(1, v * h * 2.6);
            ctx.fillRect(c, h - bh, 1, bh);
        }
        // Shared prefix, in a distinct colour: this region is exactly zero.
        const fx = (this.spec.forkAt / this.divN) * w;
        ctx.fillStyle = 'rgba(120,255,170,0.30)';
        ctx.fillRect(0, 0, Math.max(1.5, fx), h);
        ctx.strokeStyle = 'rgba(255,255,255,0.7)';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(fx, 0); ctx.lineTo(fx, h); ctx.stroke();
    }

    /* ────────────────────────────────────────────────────────────────
       INPUT
       ──────────────────────────────────────────────────────────────── */

    _wire() {
        $('fork-play').addEventListener('click', () => { this.playing = !this.playing; this.lastToken = -1; });
        $('fork-back').addEventListener('click', () => this._step(-1));
        $('fork-fwd').addEventListener('click', () => this._step(1));
        $('fork-goto').addEventListener('click', () => { this.playing = false; this._seek(this.spec.forkAt); });

        const scrub = $('fork-scrubber');
        const to = e => {
            const r = scrub.getBoundingClientRect();
            this._seek(Math.round(((e.clientX - r.left) / r.width) * this.divN));
        };
        let dragging = false;
        scrub.addEventListener('pointerdown', e => {
            dragging = true; this.playing = false;
            scrub.setPointerCapture(e.pointerId); to(e);
        });
        scrub.addEventListener('pointermove', e => { if (dragging) to(e); });
        scrub.addEventListener('pointerup', e => { dragging = false; scrub.releasePointerCapture(e.pointerId); });
    }

    _seek(t) {
        this.cursor = Math.max(0, Math.min(this.divN - 1, t));
        this.rendererL.invalidate(); this.rendererR.invalidate();
        this.lastToken = -1;
    }
    _step(n) { this.playing = false; this._seek(Math.floor(this.cursor) + n); }

    /** @returns {boolean} true if the key was consumed. */
    handleKey(e) {
        switch (e.key) {
            case ' ': this.playing = !this.playing; this.lastToken = -1; return true;
            case 'ArrowLeft': this._step(-1); return true;
            case 'ArrowRight': this._step(1); return true;
            case 'Home': this._seek(0); return true;
            case 'f': case 'F': this._seek(this.spec.forkAt); return true;
            default: return false;
        }
    }
}

function esc(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
