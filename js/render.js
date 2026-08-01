/**
 * render.js — The visualisation renderer.
 *
 * ── What is drawn ──────────────────────────────────────────────────────
 *
 * A scrolling window of the (token x layer) grid. Columns are tokens, rows are
 * transformer layers with layer 0 at the BOTTOM, matching every depth-vs-time
 * plot in neuroscience and the convention stated in the README. The renderer
 * flips the layer axis when writing pixels; nothing downstream of `rowForLayer`
 * needs to think about it.
 *
 * ── How it stays fast ─────────────────────────────────────────────────
 *
 * Everything is cached on (window position, colour mode, tool state): PCA or
 * custom-basis colour, energy brightness, seam glow, overlay tints. Recomputed
 * when the cache key changes, which while playing is once per token rather than
 * once per frame.
 *
 * All of it happens at cell resolution — a grid of (visibleTokens x layers),
 * typically about 200x64 = 12,800 cells — and is then scaled up to the canvas by
 * the GPU via drawImage with smoothing on. Compositing at display resolution
 * instead would be roughly two orders of magnitude more work for an identical
 * result, because every effect here is per-cell.
 */

import * as C from './config.js';
import { quantile } from './vecmath.js';

/** Screen row for a layer index. Layer 0 at the bottom. */
export const rowForLayer = (layer, L) => L - 1 - layer;
/** Layer index for a screen row. Inverse of rowForLayer. */
export const layerForRow = (row, L) => L - 1 - row;

export class Renderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d', { alpha: false });

        /** Offscreen cell-resolution buffer, scaled up on draw. */
        this.off = document.createElement('canvas');
        this.offCtx = this.off.getContext('2d', { alpha: false });
        this.offImage = null;

        this.cacheKey = '';
        this.cachedRGB = null;

        /** CSS-pixel geometry of the last frame, for hit-testing. */
        this.geom = { w: 0, h: 0, cellW: 0, cellH: 0, tStart: 0, tEnd: 0, padLeft: 0, tokensVisible: 0 };
    }

    /**
     * Resize backing stores to the canvas's current CSS size.
     *
     * Returns the number of token columns that now fit. The caller needs it
     * because the window width is derived from the canvas width rather than
     * fixed — that is what lets the plot fill the viewport instead of being
     * letterboxed to a fixed aspect ratio.
     *
     * @param {number} layers
     * @returns {number} tokensVisible
     */
    resize(layers) {
        const rect = this.canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = Math.max(1, Math.floor(rect.width * dpr));
        this.canvas.height = Math.max(1, Math.floor(rect.height * dpr));
        this.cssW = rect.width;
        this.cssH = rect.height;

        const tokensVisible = C.computeTokensVisible(rect.width);

        if (this.off.width !== tokensVisible || this.off.height !== layers) {
            this.off.width = tokensVisible;
            this.off.height = layers;
            this.offImage = null;
        }
        this.invalidate();
        return tokensVisible;
    }

    /** Force the cached tier to be recomputed on the next frame. */
    invalidate() { this.cacheKey = ''; this.cachedRGB = null; }

    /**
     * Draw one frame.
     *
     * @param {object} s Decoded session.
     * @param {object} v View state — see app.js `view`.
     * @param {number} dt Seconds since the previous frame, for animation phases.
     */
    draw(s, v, dt) {
        const { L } = s;
        const W = v.tokensVisible;

        const tEnd = Math.min(Math.floor(v.cursor), s.T - 1);
        const tStart = Math.max(0, tEnd - W + 1);
        const padLeft = W - (tEnd - tStart + 1);

        // ── Cached base colour ──────────────────────────────────────────
        const key = [
            tEnd, W, L, v.overlay, v.refCell ? v.refCell.join(',') : '-',
            v.basis ? v.basis.stamp : '-', v.basisGuide ? v.basisGuide.stamp : '-',
        ].join('|');

        let rgb;
        if (key === this.cacheKey && this.cachedRGB) {
            rgb = this.cachedRGB.slice();
        } else {
            rgb = this._baseRGB(s, v, tStart, tEnd, padLeft, W, L);
            this.cachedRGB = rgb.slice();
            this.cacheKey = key;
        }

        // ── Cell buffer -> pixels ───────────────────────────────────────
        if (!this.offImage) this.offImage = this.offCtx.createImageData(W, L);
        const px = this.offImage.data;
        for (let i = 0, p = 0; i < W * L; i++, p += 4) {
            const j = i * 3;
            px[p]     = Math.max(0, Math.min(255, rgb[j] * 255)) | 0;
            px[p + 1] = Math.max(0, Math.min(255, rgb[j + 1] * 255)) | 0;
            px[p + 2] = Math.max(0, Math.min(255, rgb[j + 2] * 255)) | 0;
            px[p + 3] = 255;
        }
        this.offCtx.putImageData(this.offImage, 0, 0);

        // ── Scale up and overlay vector graphics ────────────────────────
        const ctx = this.ctx;
        const dpr = window.devicePixelRatio || 1;
        const dw = this.cssW, dh = this.cssH;

        ctx.save();
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.fillStyle = `rgb(${C.BG[0]},${C.BG[1]},${C.BG[2]})`;
        ctx.fillRect(0, 0, dw, dh);
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(this.off, 0, 0, W, L, 0, 0, dw, dh);

        const cellW = dw / W, cellH = dh / L;
        this.geom = { w: dw, h: dh, cellW, cellH, tStart, tEnd, padLeft, tokensVisible: W };

        this._playhead(ctx, padLeft + (tEnd - tStart), cellW, dh);
        if (v.turnMarkers) this._turnMarkers(ctx, s, tStart, tEnd, padLeft, cellW, dh);
        if (v.basisGroups && v.basisGroups.length) {
            this._basisMarkers(ctx, v.basisGroups, tStart, tEnd, padLeft, cellW, cellH, L);
        }
        if (v.hover) this._crosshair(ctx, v.hover, tStart, padLeft, cellW, cellH, L, dw, dh);

        ctx.restore();
    }

    /* ────────────────────────────────────────────────────────────────
       TIER 1
       ──────────────────────────────────────────────────────────────── */

    /**
     * Base colour for every visible cell.
     *
     * Order matters: PCA (or custom basis) sets hue, energy scales brightness,
     * the seam glow is added, and finally a single-metric overlay — if one is
     * active — replaces hue entirely while keeping a fraction of the luminance.
     */
    _baseRGB(s, v, tStart, tEnd, padLeft, W, L) {
        const { T } = s;
        const rgb = new Float32Array(W * L * 3);

        const basis = (v.basis && v.basis.ready) ? v.basis : null;

        for (let col = 0; col < W; col++) {
            const t = tStart + (col - padLeft);
            if (t < 0 || t >= T) continue;                 // left margin before the sequence
            for (let layer = 0; layer < L; layer++) {
                const row = rowForLayer(layer, L);
                const dst = (row * W + col) * 3;
                const cell = t * L + layer;

                let r, g, b;
                if (basis) {
                    r = basis.R[cell]; g = basis.G[cell]; b = basis.B[cell];
                } else {
                    const src = cell * 3;
                    r = s.rgb[src] / 255; g = s.rgb[src + 1] / 255; b = s.rgb[src + 2] / 255;
                }

                const bright = C.ENERGY_FLOOR
                    + (C.ENERGY_CEIL - C.ENERGY_FLOOR) * (s.energyNorm[cell] / 255);
                rgb[dst] = r * bright; rgb[dst + 1] = g * bright; rgb[dst + 2] = b * bright;
            }
        }

        // Seam glow: warm, centred on the middle layers, one value per token.
        if (C.SEAM_GLOW_INTENSITY > 0) {
            const sig = C.SEAM_GLOW_SIGMA;
            for (let col = 0; col < W; col++) {
                const t = tStart + (col - padLeft);
                if (t < 0 || t >= T) continue;
                const seam = s.seamScore[t] / 255;
                if (seam <= 0) continue;
                for (let row = 0; row < L; row++) {
                    const y = row / L - 0.5;
                    const glow = Math.exp(-(y / sig) * (y / sig)) * seam * C.SEAM_GLOW_INTENSITY;
                    const i = (row * W + col) * 3;
                    rgb[i]     += glow * C.SEAM_GLOW_COLOR[0];
                    rgb[i + 1] += glow * C.SEAM_GLOW_COLOR[1];
                    rgb[i + 2] += glow * C.SEAM_GLOW_COLOR[2];
                }
            }
        }

        if (v.overlay) this._overlay(s, v, rgb, tStart, padLeft, W, L);
        if (v.refCell && v.refDistances) this._reference(s, v, rgb, tStart, padLeft, W, L);
        else if (v.basisGuide) this._guidance(v, rgb, tStart, padLeft, W, L, s.T, L);

        return rgb;
    }

    /**
     * Replace hue with a single-metric heat map.
     *
     * A fraction of the original luminance is retained (OVERLAY_LUMA_KEEP) so
     * that the structure you were just looking at does not vanish when you
     * switch modes — losing your place is worse than a slightly impure scale.
     */
    _overlay(s, v, rgb, tStart, padLeft, W, L) {
        const src = {
            energy: s.energyNorm, sparsity: s.sparsityNorm, entropy: s.entropyNorm,
        }[v.overlay];
        if (!src) return;
        const tint = C.OVERLAY_TINTS[v.overlay] || C.OVERLAY_TINTS.energy;
        const [lr, lg, lb] = C.LUMA_WEIGHTS;
        const keep = C.OVERLAY_LUMA_KEEP;

        for (let col = 0; col < W; col++) {
            const t = tStart + (col - padLeft);
            if (t < 0 || t >= s.T) continue;
            for (let layer = 0; layer < L; layer++) {
                const row = rowForLayer(layer, L);
                const i = (row * W + col) * 3;
                const lum = lr * rgb[i] + lg * rgb[i + 1] + lb * rgb[i + 2];
                const m = src[t * L + layer] / 255;
                // Ramp the tint's own contribution with m as well as scaling by
                // it, so that low values read as dark rather than as dark-tinted.
                const gain = 0.7 * (0.3 + 0.7 * m) * m;
                rgb[i]     = lum * keep + tint[0] * gain;
                rgb[i + 1] = lum * keep + tint[1] * gain;
                rgb[i + 2] = lum * keep + tint[2] * gain;
            }
        }
    }

    /**
     * Recolour by distance from a reference cell.
     *
     * The colour scale is set from the visible cells only, so it adapts as you
     * scroll — the question being answered is "what here resembles the
     * reference", not "what in the whole session does". Padding columns are
     * excluded from the quantiles; including them would drag the low end to zero
     * and wash out the first screenful of every session.
     */
    _reference(s, v, rgb, tStart, padLeft, W, L) {
        const d = v.refDistances;
        const vis = [];
        for (let col = 0; col < W; col++) {
            const t = tStart + (col - padLeft);
            if (t < 0 || t >= s.T) continue;
            for (let layer = 0; layer < L; layer++) vis.push(d[t * L + layer]);
        }
        if (!vis.length) return;
        const lo = quantile(vis, C.REF_Q_LO);
        const hi = quantile(vis, C.REF_Q_HI);
        const inv = 1 / (hi - lo + 1e-10);
        const [nr, ng, nb] = C.REF_NEAR_COLOR;
        const [fr, fg, fb] = C.REF_FAR_COLOR;
        const [rt, rl] = v.refCell;

        for (let col = 0; col < W; col++) {
            const t = tStart + (col - padLeft);
            if (t < 0 || t >= s.T) continue;
            for (let layer = 0; layer < L; layer++) {
                const row = rowForLayer(layer, L);
                const i = (row * W + col) * 3;
                const sim = 1 - Math.max(0, Math.min(1, (d[t * L + layer] - lo) * inv));
                const bright = 0.3 + 0.7 * sim;
                rgb[i]     = (nr * sim + fr * (1 - sim)) * bright;
                rgb[i + 1] = (ng * sim + fg * (1 - sim)) * bright;
                rgb[i + 2] = (nb * sim + fb * (1 - sim)) * bright;
                if (t === rt && layer === rl) { rgb[i] = 1; rgb[i + 1] = 1; rgb[i + 2] = 1; }
            }
        }
    }

    /**
     * Greyscale field shown while picking the 2nd and 3rd custom-basis axes:
     * how much of each cell's state is NOT yet explained by the axes chosen so
     * far. Bright means "still unspoken for" — click there to get an axis that
     * carries new information rather than a restatement of the last one.
     */
    _guidance(v, rgb, tStart, padLeft, W, L, T) {
        const g = v.basisGuide.field;
        const vis = [];
        for (let col = 0; col < W; col++) {
            const t = tStart + (col - padLeft);
            if (t < 0 || t >= T) continue;
            for (let layer = 0; layer < L; layer++) vis.push(g[t * L + layer]);
        }
        if (!vis.length) return;
        const lo = quantile(vis, 0.05), hi = quantile(vis, 0.95);
        const inv = 1 / (hi - lo + 1e-10);
        for (let col = 0; col < W; col++) {
            const t = tStart + (col - padLeft);
            if (t < 0 || t >= T) continue;
            for (let layer = 0; layer < L; layer++) {
                const row = rowForLayer(layer, L);
                const i = (row * W + col) * 3;
                const n = Math.max(0, Math.min(1, (g[t * L + layer] - lo) * inv));
                rgb[i] = n; rgb[i + 1] = n; rgb[i + 2] = n;
            }
        }
    }

    /* ────────────────────────────────────────────────────────────────
       VECTOR OVERLAYS
       ──────────────────────────────────────────────────────────────── */

    _playhead(ctx, col, cellW, h) {
        const x = col * cellW;
        ctx.strokeStyle = C.COLORS.playhead;
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }

    _turnMarkers(ctx, s, tStart, tEnd, padLeft, cellW, h) {
        ctx.font = '10px ui-monospace, monospace';
        for (const tb of s.turns) {
            if (tb.token_start < tStart || tb.token_start > tEnd + 1) continue;
            const x = (padLeft + (tb.token_start - tStart)) * cellW;
            const col = tb.role === 'user' ? 'rgba(120,180,255,0.55)'
                      : tb.role === 'assistant' ? 'rgba(255,200,100,0.55)'
                      : 'rgba(100,100,120,0.55)';
            ctx.strokeStyle = col;
            ctx.lineWidth = 1;
            ctx.setLineDash(tb.role === 'user' ? [7, 4] : []);
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = col;
            ctx.fillText(`T${tb.turn}`, x + 3, 12);
        }
    }

    _basisMarkers(ctx, groups, tStart, tEnd, padLeft, cellW, cellH, L) {
        for (let gi = 0; gi < groups.length && gi < 3; gi++) {
            const g = groups[gi];
            const mark = (cells, color, minus) => {
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                for (const [t, l] of cells) {
                    if (t < tStart || t > tEnd) continue;
                    const x = (padLeft + (t - tStart)) * cellW;
                    const y = rowForLayer(l, L) * cellH;
                    ctx.strokeRect(x, y, cellW, cellH);
                    const cx = x + cellW / 2, cy = y + cellH / 2;
                    const r = Math.min(cellW, cellH) / 4;
                    ctx.beginPath();
                    ctx.moveTo(cx - r, cy); ctx.lineTo(cx + r, cy);
                    if (!minus) { ctx.moveTo(cx, cy - r); ctx.lineTo(cx, cy + r); }
                    ctx.stroke();
                }
            };
            mark(g.source, C.COLORS.basisBright[gi], false);
            mark(g.contrast, C.COLORS.basisDim[gi], true);
        }
    }

    _crosshair(ctx, hover, tStart, padLeft, cellW, cellH, L, w, h) {
        const [t, l] = hover;
        const x = (padLeft + (t - tStart) + 0.5) * cellW;
        const y = (rowForLayer(l, L) + 0.5) * cellH;
        ctx.strokeStyle = 'rgba(255,200,100,0.38)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, 0); ctx.lineTo(x, h);
        ctx.moveTo(0, y); ctx.lineTo(w, y);
        ctx.stroke();
    }

    /**
     * Map a point in canvas CSS coordinates to a (token, layer) cell.
     *
     * @returns {?[number, number]} null outside the data area or in the left margin.
     */
    hitTest(x, y, s) {
        const { w, h, cellW, cellH, tStart, padLeft, tokensVisible } = this.geom;
        if (!w || x < 0 || y < 0 || x >= w || y >= h) return null;
        const col = Math.floor(x / cellW);
        if (col < padLeft) return null;                     // before the sequence began
        const t = Math.min(s.T - 1, tStart + (col - padLeft));
        const row = Math.max(0, Math.min(s.L - 1, Math.floor(y / cellH)));
        return [t, layerForRow(row, s.L)];
    }
}

/**
 * Draw the scrubber: turn structure as bands, seams as ticks.
 *
 * This is the one piece of chrome that earns its space several times over. It
 * shows, at a glance and without playing anything, how the conversation is
 * shaped — who spoke when, for how long, and where the model's state jumped.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {object} s Decoded session.
 */
export function drawScrubber(canvas, s) {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const w = rect.width, h = rect.height;
    ctx.clearRect(0, 0, w, h);

    // Turn bands
    for (const tb of s.turns) {
        const x1 = (tb.token_start / s.T) * w;
        const x2 = (tb.token_end / s.T) * w;
        ctx.fillStyle = tb.role === 'user' ? 'rgba(120,180,255,0.28)'
                      : tb.role === 'assistant' ? 'rgba(255,200,100,0.22)'
                      : 'rgba(100,100,120,0.18)';
        ctx.fillRect(x1, 0, Math.max(1, x2 - x1), h);
    }

    // Seam ticks. Drawn above a threshold only: every token has a seam score,
    // and drawing all of them would produce a solid block rather than a map.
    ctx.strokeStyle = 'rgba(255,225,170,0.75)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let t = 1; t < s.T; t++) {
        const v = s.seamScore[t] / 255;
        if (v < 0.55) continue;
        const x = Math.round((t / s.T) * w) + 0.5;
        const len = h * (0.32 + 0.6 * v);
        ctx.moveTo(x, h); ctx.lineTo(x, h - len);
    }
    ctx.stroke();
}
