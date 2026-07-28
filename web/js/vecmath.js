/**
 * vecmath.js — Small pure linear-algebra and normalisation helpers.
 *
 * Everything here operates on Float32Array and returns fresh arrays. No module
 * state, no DOM. This is the file to read if you want to check the maths
 * without reading the renderer.
 *
 * Vectors are points in the 16-dimensional Johnson-Lindenstrauss sketch space,
 * not in the model's native 5,120-dimensional residual stream. The distinction
 * matters for interpretation and is discussed in docs/METRICS.md: JL projection
 * approximately preserves pairwise Euclidean distances, so *distances* and
 * *distance-derived* quantities are meaningful; individual coordinates are not.
 */

/* ────────────────────────────────────────────────────────────────
   ELEMENTWISE AND INNER-PRODUCT OPERATIONS
   ──────────────────────────────────────────────────────────────── */

/** Inner product. */
export function dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
}

/** Euclidean (L2) norm. */
export function norm(v) {
    return Math.sqrt(dot(v, v));
}

/** Scalar multiple, as a new array. */
export function scale(v, s) {
    const r = new Float32Array(v.length);
    for (let i = 0; i < v.length; i++) r[i] = v[i] * s;
    return r;
}

/** Difference a - b, as a new array. */
export function sub(a, b) {
    const r = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) r[i] = a[i] - b[i];
    return r;
}

/** Sum a + b, as a new array. */
export function add(a, b) {
    const r = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) r[i] = a[i] + b[i];
    return r;
}

/** Centroid of a list of equal-length vectors. */
export function mean(vectors) {
    const D = vectors[0].length;
    const m = new Float32Array(D);
    for (const v of vectors) {
        for (let d = 0; d < D; d++) m[d] += v[d];
    }
    for (let d = 0; d < D; d++) m[d] /= vectors.length;
    return m;
}

/** Euclidean distance between two vectors, without allocating a difference. */
export function distance(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) {
        const d = a[i] - b[i];
        s += d * d;
    }
    return Math.sqrt(s);
}

/* ────────────────────────────────────────────────────────────────
   ORTHONORMALISATION
   ──────────────────────────────────────────────────────────────── */

/**
 * Gram-Schmidt orthonormalisation of exactly three vectors.
 *
 * Used by the custom colour-basis feature. The user picks three directions in
 * JL space by clicking cells; those directions are almost never orthogonal, and
 * projecting colour channels onto non-orthogonal axes makes the channels
 * correlated — two "different" axes end up showing the same thing, and the
 * display looks informative while carrying less information than it appears to.
 *
 * Orthonormalising fixes that, at a cost the user should know about: the second
 * and third axes are no longer exactly what was asked for. They are the
 * components of those requests that are *independent of* the earlier ones. The
 * viewer communicates this by shading, during selection, the cells whose state
 * is already well explained by the axes chosen so far — so you can see which
 * part of the space is still unspoken for.
 *
 * @param {Float32Array} v1
 * @param {Float32Array} v2
 * @param {Float32Array} v3
 * @param {number} eps Degeneracy threshold; a residual norm below this means
 *        the requested direction was (nearly) inside the span of the previous
 *        ones, and no independent axis exists.
 * @returns {?{e1: Float32Array, e2: Float32Array, e3: Float32Array}} null if
 *          the three inputs are degenerate.
 */
export function gramSchmidt3(v1, v2, v3, eps = 1e-6) {
    const n1 = norm(v1);
    if (n1 < eps) return null;
    const e1 = scale(v1, 1 / n1);

    const r2 = sub(v2, scale(e1, dot(v2, e1)));
    const n2 = norm(r2);
    if (n2 < eps) return null;
    const e2 = scale(r2, 1 / n2);

    let r3 = sub(v3, scale(e1, dot(v3, e1)));
    r3 = sub(r3, scale(e2, dot(r3, e2)));
    const n3 = norm(r3);
    if (n3 < eps) return null;
    const e3 = scale(r3, 1 / n3);

    return { e1, e2, e3 };
}

/* ────────────────────────────────────────────────────────────────
   ROBUST NORMALISATION
   ──────────────────────────────────────────────────────────────── */

/**
 * Quantile of a numeric array. Sorts a copy; fine for the array sizes here
 * (at most tokens x layers ~ 200k, done once per mode change, not per frame).
 *
 * @param {ArrayLike<number>} arr
 * @param {number} q In [0, 1].
 */
export function quantile(arr, q) {
    const sorted = Float64Array.from(arr).sort();
    if (sorted.length === 0) return 0;
    const i = Math.floor(sorted.length * q);
    return sorted[Math.min(sorted.length - 1, Math.max(0, i))];
}

/**
 * Map an array to [0, 1] by clipping at the given quantiles.
 *
 * Why quantiles rather than min/max: activation-derived quantities have heavy
 * tails. A single outlier token — often the very first token of a sequence,
 * whose "change since the previous token" is undefined and whose norm is
 * unusual — compresses everything else into the bottom few percent of the
 * range, and the display goes flat. Clipping at the 5th and 95th percentiles
 * costs the extremes and buys usable contrast across the body of the data.
 *
 * The cost is real and should be stated: values outside the clip range are
 * *indistinguishable* after normalisation. The inspector reports pre-clip
 * values so that nothing is only visible through the clipped view.
 *
 * @param {ArrayLike<number>} arr
 * @param {number} qLo
 * @param {number} qHi
 * @returns {Float32Array} Same length, values in [0, 1].
 */
export function quantileNorm(arr, qLo = 0.05, qHi = 0.95) {
    const lo = quantile(arr, qLo);
    const hi = quantile(arr, qHi);
    const out = new Float32Array(arr.length);
    if (hi <= lo) return out;
    const inv = 1 / (hi - lo);
    for (let i = 0; i < arr.length; i++) {
        out[i] = Math.max(0, Math.min(1, (arr[i] - lo) * inv));
    }
    return out;
}

/**
 * Symmetric variant used for signed projections onto a custom basis.
 *
 * A projection onto a contrast direction (mean(A) - mean(B)) is signed and
 * roughly centred, so it should map to colour symmetrically about mid-grey:
 * 0.5 means "on the boundary", not "low". quantileNorm would instead put the
 * 5th percentile at black, which throws away the sign information that makes a
 * contrast axis worth defining.
 *
 * @param {ArrayLike<number>} arr
 * @param {number} qLo
 * @param {number} qHi
 * @returns {Float32Array} Values in [0, 1], with the midpoint of the clipped
 *          range mapped to 0.5.
 */
export function symmetricNorm(arr, qLo = 0.02, qHi = 0.98) {
    const lo = quantile(arr, qLo);
    const hi = quantile(arr, qHi);
    const mid = (lo + hi) / 2;
    const halfSpread = (hi - lo) / 2 + 1e-10;
    const out = new Float32Array(arr.length);
    for (let i = 0; i < arr.length; i++) {
        out[i] = Math.max(0, Math.min(1, ((arr[i] - mid) / halfSpread) * 0.5 + 0.5));
    }
    return out;
}
