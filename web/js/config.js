/**
 * config.js — Every tunable constant in Phosphenes, in one place, with its reason.
 *
 * Design rule for this file: a number lives here if changing it changes what the
 * viewer looks like or how it performs. Numbers that are forced by the data
 * (layer counts, JL dimensionality) are read from the session bundle instead —
 * they are facts, not choices.
 *
 * Where a value was chosen by eye rather than derived, this file says so. That
 * distinction matters: the perceptual mappings below are deliberately tuned for
 * human vision, and calling them "arbitrary" would be wrong, but so would
 * calling them principled.
 */

/* ────────────────────────────────────────────────────────────────
   GEOMETRY
   ──────────────────────────────────────────────────────────────── */

/**
 * Target on-screen width of one token column, in CSS pixels.
 *
 * The number of visible tokens is derived from this and the window width
 * (see computeTokensVisible), rather than fixed, so that the display fills
 * whatever window it is given instead of letterboxing. 7px is about the
 * narrowest column at which a single bright token still reads as a distinct
 * vertical stripe on a standard-density display.
 */
export const TARGET_CELL_PX = 7;

/** Clamp on derived column count. Below ~60 the temporal texture is lost; above
 *  ~420 the per-frame overlay cost starts to show on integrated graphics. */
export const MIN_TOKENS_VISIBLE = 60;
export const MAX_TOKENS_VISIBLE = 420;

/** Gutter reserved for the layer-number axis, in CSS pixels. */
export const AXIS_W = 34;
/** Gutter reserved for the token-position ruler, in CSS pixels. */
export const AXIS_H = 18;

/* ────────────────────────────────────────────────────────────────
   PLAYBACK
   ──────────────────────────────────────────────────────────────── */

/**
 * Default playback rate, tokens per second.
 *
 * 24 tok/s is roughly 5x an average human silent-reading rate (~4 words/s).
 * Chosen so that a 3,000-token session runs about two minutes: long enough to
 * see structure develop, short enough that a first-time viewer watches it end.
 */
export const DEFAULT_TPS = 24.0;
export const MIN_SPEED = 0.1;
export const MAX_SPEED = 16.0;
/** Multiplicative step for the speed up/down controls. */
export const SPEED_STEP = 1.5;

/* ────────────────────────────────────────────────────────────────
   COLOUR AND EFFECT MAPPING
   ──────────────────────────────────────────────────────────────── */

/** Page background, RGB 0-255. Also the canvas clear colour. */
export const BG = [11, 11, 18];

/**
 * Brightness floor for a cell at zero normalised energy.
 *
 * Not 0: a cell with low energy is still a real measurement, and rendering it
 * as pure black would make "low activation" visually indistinguishable from
 * "no data" (the left margin before the sequence starts). 0.15 keeps colour
 * identity legible while still reading as dim.
 */
export const ENERGY_FLOOR = 0.15;
export const ENERGY_CEIL = 1.0;

/**
 * Warm glow applied at tokens with high seam score, falling off vertically
 * from the middle layers.
 *
 * Empirically the seam score is 2.85x higher within +/-2 tokens of a
 * conversational turn boundary than elsewhere (measured on the
 * Well-Read-Library session; asserted in analysis/verify_tour_claims.py).
 * The ratio is sensitive to the window: 3.77x at +/-0, 2.31x at +/-3.
 * The glow is what makes that difference preattentive — you see the
 * boundary before you read it.
 */
export const SEAM_GLOW_INTENSITY = 0.55;
export const SEAM_GLOW_COLOR = [1.0, 0.88, 0.65];
/** Vertical spread of the glow, as a fraction of display height (Gaussian sigma). */
export const SEAM_GLOW_SIGMA = 0.28;

/**
 * Colour turbulence amplitude, driven per-cell by delta_norm (how much the
 * hidden state moved since the previous token).
 *
 * This is the one effect that is animation rather than data: the *amount* of
 * shimmer is the measurement, the shimmer itself is a carrier. Human vision
 * detects coherent motion far below the contrast threshold for static
 * patterns, so motion is a cheap extra channel that does not compete with
 * colour for the same perceptual bandwidth.
 */
export const TURBULENCE_AMP = 0.16;
/** Rate at which the turbulence noise field is cross-faded, in fields/second. */
export const TURBULENCE_SPEED = 2.5;
/** Number of pre-generated smoothed noise fields cycled for turbulence. */
export const TURBULENCE_FIELDS = 4;

/**
 * Grain amplitude, driven per-cell by cos_instability (1 - cosine similarity
 * to the previous token's state at the same layer).
 *
 * Reads as "static" or "noise" — which is the intended semantics. A cell where
 * the model's representational *direction* is changing, not just its
 * magnitude, looks unstable.
 */
export const GRAIN_MAX = 0.12;
export const GRAIN_SPEED = 3.0;
export const GRAIN_FIELDS = 8;

/** Saturation boost applied to PCA colours during data preparation, mirrored
 *  here for the custom-basis path so both look consistent. */
export const SATURATION_BOOST = 1.3;

/* ────────────────────────────────────────────────────────────────
   METRIC OVERLAY TINTS
   ──────────────────────────────────────────────────────────────── */

/**
 * Overlay modes replace PCA colour with a single-metric heat map, keeping a
 * fraction of the original luminance so the underlying structure stays visible.
 * Tints are cyan-ish for energy and green-ish for sparsity: distinguishable
 * for the most common form of colour-vision deficiency, since they differ in
 * blue as well as in red/green.
 */
export const OVERLAY_TINTS = {
    energy:   [0.3, 0.8, 1.0],
    sparsity: [0.2, 1.0, 0.4],
};
/** How much of the original luminance survives under an overlay. */
export const OVERLAY_LUMA_KEEP = 0.3;

/** Rec.601 luma weights, used to collapse PCA colour to brightness under overlays. */
export const LUMA_WEIGHTS = [0.299, 0.587, 0.114];

/* ────────────────────────────────────────────────────────────────
   REFERENCE-POINT MODE
   ──────────────────────────────────────────────────────────────── */

/** Warm end (identical to the reference cell) and cool end (most distant). */
export const REF_NEAR_COLOR = [1.0, 0.9, 0.5];
export const REF_FAR_COLOR  = [0.1, 0.2, 0.5];
/** Quantiles used to set the distance colour scale, over visible cells only. */
export const REF_Q_LO = 0.01;
export const REF_Q_HI = 0.95;

/* ────────────────────────────────────────────────────────────────
   UI PALETTE  (kept in sync with css/phosphenes.css :root)
   ──────────────────────────────────────────────────────────────── */

export const COLORS = {
    amber:     'rgb(255, 200, 100)',
    user:      'rgb(120, 180, 255)',
    assistant: 'rgb(255, 200, 100)',
    system:    'rgb(100, 100, 120)',
    dim:       'rgb(120, 120, 140)',
    playhead:  'rgba(255, 200, 100, 0.5)',
    axis:      'rgba(120, 120, 140, 0.55)',
    /** Marker colours for the three custom-basis channels, bright = source,
     *  dim = contrast. */
    basisBright: ['rgb(255,70,70)', 'rgb(70,255,70)', 'rgb(90,120,255)'],
    basisDim:    ['rgb(150,45,45)', 'rgb(45,150,45)', 'rgb(55,70,150)'],
};

/* ────────────────────────────────────────────────────────────────
   DERIVED GEOMETRY
   ──────────────────────────────────────────────────────────────── */

/**
 * Choose how many token columns to show, given the available width.
 *
 * Deriving this instead of fixing it is what lets the display fill the window.
 * The alternative — a fixed column count letterboxed to preserve cell aspect
 * ratio — wasted up to a third of the viewport on background.
 *
 * @param {number} availW Width available for the data area, CSS pixels.
 * @returns {number} Column count, clamped and integral.
 */
export function computeTokensVisible(availW) {
    const raw = Math.round(availW / TARGET_CELL_PX);
    return Math.max(MIN_TOKENS_VISIBLE, Math.min(MAX_TOKENS_VISIBLE, raw));
}
