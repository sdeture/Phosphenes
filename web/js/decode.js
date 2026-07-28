/**
 * decode.js — Loading and decoding session bundles.
 *
 * A session bundle is a single JSON file produced by convert_for_web.py. It
 * carries several large numeric arrays base64-encoded inside JSON, plus the
 * decoded token strings and turn boundaries.
 *
 * ── Why the arrays are quantised to uint8 ──────────────────────────────
 *
 * The full object being visualised, for one conversation, is:
 *
 *     2,990 tokens x 64 layers x 5,120 dimensions = 979,251,200 numbers
 *
 * At bfloat16 that is about 1.96 GB for a single conversation. That number is
 * the reason this tool exists in the form it does, and the reason there are so
 * few interpretability interactives that span a whole conversation rather than
 * a single prompt.
 *
 * Two compressions get it into a browser:
 *
 *   1. Johnson-Lindenstrauss projection, 5,120 -> 16 dimensions, applied at
 *      extraction time. JL gives a distortion bound on pairwise distances that
 *      depends on the number of points and the target dimension, not on the
 *      source dimension. Distances survive; individual coordinates do not.
 *      (See docs/METRICS.md for what this does and does not license.)
 *
 *   2. Per-dimension uint8 quantisation of the JL vectors, and uint8
 *      quantisation of the [0,1]-normalised scalar metrics, applied here.
 *
 * Result: about 5.8 MB per session, from 1.96 GB. That is a 340x reduction, and
 * it is lossy in ways worth being explicit about — 8 bits per JL dimension
 * means roughly 0.4% relative precision per coordinate. Fine for a display
 * whose output is 8-bit colour; not fine for downstream numerics. Anything
 * quantitative should be computed from the .npz files in data/, not from these
 * bundles.
 */

/**
 * Decode a base64 string to bytes.
 *
 * atob + a manual copy is used rather than fetch/Blob tricks because it is
 * synchronous and predictable; these payloads are a few megabytes and decode in
 * a few milliseconds.
 *
 * @param {string} b64
 * @returns {Uint8Array}
 */
export function base64ToBytes(b64) {
    const binary = atob(b64);
    const out = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
    return out;
}

/**
 * Undo the uint8 quantisation of the JL vectors.
 *
 * Bounds are stored per (layer, dimension), flattened row-major as L*D. Using
 * one range per dimension pooled across layers — which an earlier version of
 * the encoder did — makes early-layer vectors unusable, because residual-stream
 * magnitude grows by roughly 78x from layer 0 to layer 60 and the pooled range
 * sets the step size from the deepest layers. See quantize_jl in
 * convert_for_web.py.
 *
 * @param {Uint8Array} q Quantised values, length T*L*D, ordered (token, layer, dim).
 * @param {Float32Array} lo Minimums, length L*D, ordered (layer, dim).
 * @param {Float32Array} hi Maximums, length L*D, ordered (layer, dim).
 * @param {number} T Token count.
 * @param {number} L Layer count.
 * @param {number} D JL dimension.
 * @returns {Float32Array} Dequantised values, length T*L*D, same ordering.
 */
export function dequantizeJL(q, lo, hi, T, L, D) {
    // Precompute per-(layer, dim) step so the inner loop is two multiplies.
    const step = new Float32Array(L * D);
    for (let i = 0; i < L * D; i++) step[i] = (hi[i] - lo[i]) / 255.0;

    const out = new Float32Array(T * L * D);
    for (let t = 0; t < T; t++) {
        for (let l = 0; l < L; l++) {
            const off = (t * L + l) * D;
            const b = l * D;
            for (let d = 0; d < D; d++) {
                out[off + d] = q[off + d] * step[b + d] + lo[b + d];
            }
        }
    }
    return out;
}

/**
 * Fetch the session index.
 *
 * @param {string} base Directory containing sessions.json.
 * @returns {Promise<Array<object>>}
 */
export async function loadSessionIndex(base = 'data') {
    const resp = await fetch(`${base}/sessions.json`);
    if (!resp.ok) throw new Error(`sessions.json: HTTP ${resp.status}`);
    const idx = await resp.json();
    return idx.sessions;
}

/**
 * Fetch and decode one session bundle into the shape the renderer wants.
 *
 * Array layout convention, used everywhere downstream: scalar per-cell metrics
 * are flat with stride L, indexed `t * L + layer`. Layer 0 is the earliest
 * transformer layer. The renderer flips this when drawing so that layer 0
 * appears at the bottom — the convention from EEG and from every depth-vs-time
 * plot in neuroscience, where the signal travels upward.
 *
 * @param {object} info An entry from the session index.
 * @param {string} base Directory containing the bundles.
 * @param {(msg: string) => void} [onProgress]
 * @returns {Promise<object>} Decoded session.
 */
export async function loadSession(info, base = 'data', onProgress = () => {}) {
    onProgress(`Fetching ${info.display_name}`);
    const resp = await fetch(`${base}/${info.file}`);
    if (!resp.ok) throw new Error(`${info.file}: HTTP ${resp.status}`);
    const raw = await resp.json();

    onProgress('Decoding activations');
    const T = raw.n_tokens;
    const L = raw.n_layers;
    const D = raw.jl_dim;

    // Refuse a bundle in the old per-dimension bounds format rather than
    // dequantising it wrongly and showing plausible nonsense. Regenerate with
    // `python convert_for_web.py`.
    if (!raw.jl_bounds_shape || raw.jl_min.length !== L * D) {
        throw new Error(
            `${info.file}: JL bounds are ${raw.jl_min.length} values, expected ${L * D} ` +
            `(per layer x dimension). This bundle predates the per-layer quantiser — ` +
            `regenerate it with: python convert_for_web.py`,
        );
    }

    const jl = dequantizeJL(
        base64ToBytes(raw.jl),
        new Float32Array(raw.jl_min),
        new Float32Array(raw.jl_max),
        T, L, D,
    );

    return {
        stem: raw.stem,
        displayName: raw.display_name,
        modelId: raw.model_id || '',
        T, L, D,

        /** Pre-computed PCA colour, uint8, length T*L*3, ordered (token, layer, channel). */
        rgb: base64ToBytes(raw.rgb),
        /** Dequantised JL sketch vectors, Float32, length T*L*D. */
        jl,

        /** Per-cell scalars in [0,255]; divide by 255 to recover [0,1]. */
        energyNorm:     base64ToBytes(raw.energy_norm),
        deltaNorm:      base64ToBytes(raw.delta_norm),
        cosInstability: base64ToBytes(raw.cos_instability),
        sparsityNorm:   base64ToBytes(raw.sparsity_norm),
        /** Logit-lens entropy on an absolute scale: 255 = uniform over vocabulary. */
        entropyNorm:    base64ToBytes(raw.entropy_norm),
        /** Nats corresponding to entropyNorm = 255, for the inspector readout. */
        entropyScaleNats: raw.entropy_scale_nats || Math.log(151936),

        /** Per-token scalar in [0,255]. */
        seamScore: base64ToBytes(raw.seam_score),

        /** Decoded token strings, length T. Includes chat special tokens. */
        tokenPieces: raw.token_pieces || null,
        /** '', 'user', 'assistant' or 'system' per token, length T. */
        tokenRoles: raw.token_roles || [],
        /** [{turn, role, token_start, token_end}], token_end exclusive. */
        turns: raw.turns || [],
    };
}

/**
 * Locate the turn containing a token.
 *
 * @param {object} session
 * @param {number} t
 * @returns {?object}
 */
export function turnAt(session, t) {
    return session.turns.find(tb => t >= tb.token_start && t < tb.token_end) || null;
}

/**
 * Read one cell's JL vector out of the flat store.
 *
 * @param {object} session
 * @param {number} t Token index.
 * @param {number} l Layer index.
 * @returns {Float32Array} Length D. A copy, safe to keep.
 */
export function jlVector(session, t, l) {
    const { L, D, jl } = session;
    const off = (t * L + l) * D;
    return jl.slice(off, off + D);
}
