/**
 * noise.js — Pre-generated noise fields for the turbulence and grain effects.
 *
 * Both effects need a spatially coherent random field that changes over time.
 * Generating one per frame would cost more than the rest of the renderer put
 * together, so a small ring of fields is generated once per session and
 * cross-faded. Cross-fading between smoothed fields produces motion that reads
 * as flow rather than as flicker.
 *
 * The PRNG is seeded so that a given session always looks the same. That is a
 * requirement, not a nicety: if the shimmer differed between two viewings, a
 * viewer could not trust that a texture they noticed was in the data.
 */

/**
 * mulberry32 — a small, fast, well-distributed 32-bit PRNG.
 *
 * Chosen over Math.random because it is seedable. Not cryptographic; does not
 * need to be.
 *
 * @param {number} seed
 * @returns {() => number} Generator producing values in [0, 1).
 */
export function seededRandom(seed) {
    return function () {
        seed |= 0;
        seed = (seed + 0x6D2B79F5) | 0;
        let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

/**
 * Standard normal sample via the polar (Marsaglia) form of Box-Muller.
 *
 * @param {() => number} rng
 * @returns {number}
 */
export function gaussianRandom(rng) {
    let u, v, s;
    do {
        u = rng() * 2 - 1;
        v = rng() * 2 - 1;
        s = u * u + v * v;
    } while (s >= 1 || s === 0);
    return u * Math.sqrt((-2 * Math.log(s)) / s);
}

/**
 * Separable box blur, two passes, edge-clamped by shrinking the window.
 *
 * Three box passes approximate a Gaussian; one is used here because the field
 * is already random and the goal is spatial coherence rather than a particular
 * kernel shape.
 *
 * @param {Float32Array} arr Input, length rows*cols, row-major.
 * @param {number} rows
 * @param {number} cols
 * @param {number} radius
 * @returns {Float32Array} Blurred copy.
 */
export function boxBlur2D(arr, rows, cols, radius) {
    const tmp = new Float32Array(arr.length);
    const out = new Float32Array(arr.length);

    for (let r = 0; r < rows; r++) {
        const rowOff = r * cols;
        for (let c = 0; c < cols; c++) {
            let sum = 0, n = 0;
            for (let dc = -radius; dc <= radius; dc++) {
                const cc = c + dc;
                if (cc >= 0 && cc < cols) { sum += arr[rowOff + cc]; n++; }
            }
            tmp[rowOff + c] = sum / n;
        }
    }
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            let sum = 0, n = 0;
            for (let dr = -radius; dr <= radius; dr++) {
                const rr = r + dr;
                if (rr >= 0 && rr < rows) { sum += tmp[rr * cols + c]; n++; }
            }
            out[r * cols + c] = sum / n;
        }
    }
    return out;
}

/** Rescale in place to zero mean and unit variance. */
function standardise(a) {
    let sum = 0, sumSq = 0;
    for (let i = 0; i < a.length; i++) { sum += a[i]; sumSq += a[i] * a[i]; }
    const mean = sum / a.length;
    const std = Math.sqrt(Math.max(0, sumSq / a.length - mean * mean)) + 1e-6;
    for (let i = 0; i < a.length; i++) a[i] = (a[i] - mean) / std;
    return a;
}

/**
 * Build the two field rings.
 *
 * Fields are indexed in *window* space (row, col) rather than in token space.
 * That is deliberate: the noise stays fixed relative to the screen while the
 * data scrolls through it, so the shimmer reads as a property of the display
 * surface — like grain on film — rather than as a property of a token. If it
 * were locked to tokens, a viewer would see it as another data channel, which
 * it is not. Only its *amplitude* is data.
 *
 * @param {number} rows Layer count.
 * @param {number} cols Visible token count.
 * @param {number} nTurbulence
 * @param {number} nGrain
 * @returns {{turbulence: Float32Array[], grain: Float32Array[]}}
 */
export function generateNoiseFields(rows, cols, nTurbulence = 4, nGrain = 8) {
    const turbulence = [];
    const grain = [];

    let rng = seededRandom(7);
    for (let n = 0; n < nTurbulence; n++) {
        const f = new Float32Array(rows * cols);
        for (let i = 0; i < f.length; i++) f[i] = gaussianRandom(rng);
        turbulence.push(standardise(boxBlur2D(f, rows, cols, 2)));
    }

    rng = seededRandom(13);
    for (let n = 0; n < nGrain; n++) {
        const f = new Float32Array(rows * cols);
        for (let i = 0; i < f.length; i++) f[i] = gaussianRandom(rng);
        grain.push(f);
    }

    return { turbulence, grain };
}

/**
 * Cross-fade sample from a field ring.
 *
 * @param {Float32Array[]} fields
 * @param {number} phase Monotonically increasing; integer part selects the
 *        field, fractional part is the blend weight.
 * @param {number} i Flat index into the field.
 * @returns {number}
 */
export function sampleRing(fields, phase, i) {
    const a = Math.floor(phase) % fields.length;
    const b = (a + 1) % fields.length;
    const f = phase - Math.floor(phase);
    return fields[a][i] * (1 - f) + fields[b][i] * f;
}
