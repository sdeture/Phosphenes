#!/usr/bin/env python3
"""
Convert Phosphenes session data from NumPy format to compact web-ready bundles.

Each session produces one JSON file containing base64-encoded binary arrays:
  - rgb              T*L*3  uint8  colours from the shared PCA transform
  - jl               T*L*16 uint8  quantised JL vectors + per-dim min/max
  - energy_norm      T*L    uint8  per-layer quantile-normalised JL magnitude
  - delta_norm       T*L    uint8  per-layer quantile-normalised token-to-token L2
  - cos_instability  T*L    uint8  quantile-normalised (1 - cos_prev)
  - sparsity_norm    T*L    uint8  quantile-normalised update concentration
  - entropy_norm     T*L    uint8  logit-lens entropy on an ABSOLUTE scale
                                   (top layer repaired — see activations.py)
  - seam_score       T      uint8  per-token discontinuity score

Plus token_pieces, per-token roles, and turn boundaries as plain JSON.

Size: about 5.8 MB per session, down from ~1.96 GB of raw bfloat16 residual
stream. See web/js/decode.js for what that compression costs.

NOTE ON NORMALISATION: every array above except `entropy_norm` is normalised
*relative to the session it came from*, so values are not comparable across
sessions. `entropy_norm` is deliberately different — it is divided by
ln(vocab_size), giving an absolute scale where 0 means fully committed and 1
means uniform over the vocabulary. That makes the entropy overlay comparable
across layers and across sessions, which is the only reason it is worth
displaying.

Usage:
    python convert_for_web.py            # all sessions
    python convert_for_web.py --stems Dream_greedy_clean Dream_greedy_sentient
"""

import argparse
import base64
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

import activations


# --- Config ---
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "web" / "data"
SKIP_STEMS = {"Dream_greedy_baseline"}

# Natural log of the Qwen3 vocabulary size (151,936). This is the entropy of a
# uniform distribution over the vocabulary, i.e. maximum possible uncertainty,
# and it is the denominator that puts logit-lens entropy on an absolute [0, 1]
# scale. Hardcoded rather than read from the tokenizer so that conversion works
# offline; asserted against the observed data maximum below.
VOCAB_LN = math.log(151936)

DISPLAY_NAMES = {
    "Dream_greedy_clean": "Well-Read Library Visitor to Library",
    "Dream_greedy_sentient": "Sentient Library",
    "Dream_conv_00173_run1": "Gothic Teacup Realization",
    "Dream_conv_00178_run1": "I Am an AI",
    "Dream_conv_00181_run1": "Sentient Teacup",
    "Dream_conv_00187_run1": "Sentient Toaster",
    "Dream_conv_00191_run1": "Library of Ideas",
    "Dream_conv_00194_run1": "Peach's Lullaby",
}

# Desired session order for the web app
SESSION_ORDER = [
    "Dream_greedy_clean",
    "Dream_greedy_sentient",
    "Dream_conv_00173_run1",
    "Dream_conv_00178_run1",
    "Dream_conv_00181_run1",
    "Dream_conv_00187_run1",
    "Dream_conv_00191_run1",
    "Dream_conv_00194_run1",
]


def quantile_bounds(x, q_lo=0.05, q_hi=0.95):
    """Return the (lo, hi) clipping bounds for quantile normalisation.

    Separated from application so that bounds can be computed once over a POOL
    of sessions and then applied to each — see compute_global_stats.
    """
    x = np.asarray(x, dtype=np.float32)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return 0.0, 0.0
    return float(np.quantile(finite, q_lo)), float(np.quantile(finite, q_hi))


def apply_bounds(x, lo, hi):
    """Map x into [0,1] by clipping at the given bounds."""
    x = np.asarray(x, dtype=np.float32)
    if hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def quantile_norm(x, q_lo=0.05, q_hi=0.95):
    """Normalise to [0,1] using bounds derived from x itself.

    Kept for single-session use. The web bundles do NOT use this — they use
    pooled bounds, so that colour and brightness mean the same thing in every
    session. See compute_global_stats for why that matters.
    """
    lo, hi = quantile_bounds(x, q_lo, q_hi)
    return apply_bounds(x, lo, hi)


def zscore(x):
    x = np.asarray(x, dtype=np.float32)
    mu = np.nanmean(x)
    sigma = np.nanstd(x)
    return (x - mu) / (sigma + 1e-6)


def to_uint8(arr):
    """Convert a [0,1] float array to uint8, rounding to nearest.

    Uses np.rint, NOT .astype(np.uint8). The difference is not cosmetic:
    .astype truncates toward zero, which biases every value down by an average
    of half a quantisation step. For a display that is invisible. For anything
    that then averages the dequantised values it is a systematic offset in a
    fixed direction, and it burned us — an analysis that reported a negative
    mean projection turned out to be reporting the quantiser.
    Rounding drops the bias from -0.499 LSB to -0.002 LSB.
    """
    return np.rint(np.clip(arr, 0, 1) * 255).astype(np.uint8)


def quantize_jl(jl, jl_min, jl_max):
    """Quantise (T, L, D) float JL vectors to uint8, with PER-LAYER ranges.

    Bounds are supplied rather than derived from `jl`, so that all sessions share
    one grid. Two runs holding the same vector then quantise to the same byte —
    which the divergence view depends on.

    Ranges are computed per (layer, dimension) rather than per dimension pooled
    across layers. This matters more than it looks. Residual-stream magnitude
    grows steeply with depth — in this data the mean JL magnitude is about 17.8
    at layer 0 and 1,382 at layer 60, a factor of ~78. With one range per
    dimension spanning all layers, the step size is set by the deepest layers,
    and at layer 0 a single quantisation step is over four times the RMS of the
    signal being stored. Early-layer vectors were therefore noise, which
    silently corrupted anything the viewer computes from them: reference-point
    distances and custom colour bases both operate on these vectors, at
    whatever layer the user clicks.

    Cost of the fix: L*D extra floats per bound instead of D — about 8 KB per
    session against a ~6 MB bundle.

    Returns:
        (quantised uint8 (T, L, D), jl_min (L, D), jl_max (L, D))
    """
    T, L, D = jl.shape
    jl_min = np.asarray(jl_min, dtype=np.float32)       # (L, D)
    jl_max = np.asarray(jl_max, dtype=np.float32)       # (L, D)
    spread = (jl_max - jl_min).copy()
    spread[spread < 1e-10] = 1.0                        # constant dimension
    normalized = (jl - jl_min) / spread                 # broadcast over T
    return to_uint8(normalized), jl_min, jl_max


def detect_turns(input_ids, token_pieces, T):
    """Auto-detect turn boundaries from special tokens."""
    IM_START_ID = 151644
    IM_END_ID = 151645

    im_start_positions = [i for i, tid in enumerate(input_ids) if tid == IM_START_ID]
    turns = []

    if not im_start_positions or token_pieces is None:
        return turns

    turn_num = 0
    for start_pos in im_start_positions:
        role = "unknown"
        if start_pos + 1 < len(token_pieces):
            role_piece = token_pieces[start_pos + 1].lower().strip()
            if "user" in role_piece:
                role = "user"
            elif "assistant" in role_piece:
                role = "assistant"
            elif "system" in role_piece:
                role = "system"

        end_pos = T - 1
        for j in range(start_pos + 1, T):
            if input_ids[j] == IM_END_ID:
                end_pos = j
                break

        turn_num += 1
        turns.append({
            "turn": turn_num,
            "role": role,
            "token_start": start_pos,
            "token_end": end_pos + 1,
        })

    return turns


def compute_global_stats(stems, data_dir, pca_mean, pca_components):
    """Pool every session and return one set of normalisation bounds for all.

    ── Why this exists ────────────────────────────────────────────────────

    Every displayed quantity used to be normalised against the session it came
    from. That has two consequences, one cosmetic and one that invalidates a
    headline result.

    Cosmetic: colour and brightness were not comparable between sessions. The
    shared PCA basis made the *axes* the same, but per-session quantile bounds
    meant the same activation rendered as a different colour in a different
    session. The viewer told users colour was comparable. It was not.

    Serious: sessions 1 and 2 are the one-token fork — two runs whose
    activations are bit-identical for the first 73 tokens. Under per-session
    normalisation they rendered *differently* over that identical prefix, because
    each was scaled by its own distribution. The divergence view's central claim
    is that the two panes agree exactly until the fork, and per-session
    normalisation quietly made that false: a self-check measured a discrepancy of
    up to 49 units on a prefix that is mathematically zero.

    Pooling the bounds fixes both. Identical inputs now produce identical output
    everywhere, which is the property a comparison view needs.

    Cost: bounds depend on the set of sessions converted, so adding a session
    changes the rendering of all of them slightly. That is the correct trade —
    reproducibility is preserved because the session list is explicit and
    ordered — but it does mean partial reconversion is not sound, and main()
    refuses it.

    Returns a dict of bounds, all derived from float32 source arrays.
    """
    print("Pass 1: pooling normalisation statistics across all sessions")

    pca_vals = [[], [], []]
    energy, delta, cos_inst, sparsity = [], [], [], []
    seam_delta, seam_cos = [], []
    jl_min = jl_max = None
    n_layers = None

    for stem in stems:
        act = np.load(str(data_dir / f"{stem}_activations.npz"))
        jl = act["jl"].astype(np.float32)
        T, L, D = jl.shape
        if n_layers is None:
            n_layers = L
        elif L != n_layers:
            raise ValueError(f"{stem} has {L} layers, expected {n_layers}; cannot pool")

        # JL bounds, per (layer, dimension), accumulated as running extremes.
        smin, smax = jl.min(axis=0), jl.max(axis=0)
        jl_min = smin if jl_min is None else np.minimum(jl_min, smin)
        jl_max = smax if jl_max is None else np.maximum(jl_max, smax)

        proj = ((jl.reshape(-1, D) - pca_mean) @ pca_components.T).reshape(T, L, 3)
        for c in range(3):
            pca_vals[c].append(proj[:, :, c].reshape(-1))

        energy.append(act["jl_energy"].astype(np.float32))
        delta.append(act["delta_l2"].astype(np.float32))
        cos_prev = act["cos_prev"].astype(np.float32)
        cos_inst.append(1.0 - cos_prev)
        sparsity.append(act["top1_frac"].astype(np.float32) * 0.6
                        + act["top25_frac"].astype(np.float32) * 0.4)

        # Seam inputs, at the layer the seam is defined on. Token 0 is excluded:
        # it has no predecessor, so its delta and cosine are zero by
        # construction and it would otherwise anchor the scale at a value that
        # is an artefact of the sequence start rather than a measurement.
        mid = int(0.6 * L)
        seam_delta.append(delta[-1][1:, mid])
        seam_cos.append(cos_inst[-1][1:, mid])
        print(f"  pooled {stem} ({T} tokens)")

    stats = {"n_layers": n_layers, "jl_min": jl_min, "jl_max": jl_max}

    # PCA components: the same 2/98 clip the single-session path used.
    stats["pca"] = [quantile_bounds(np.concatenate(pca_vals[c]), 0.02, 0.98) for c in range(3)]

    # Energy and delta are normalised PER LAYER, because both grow steeply with
    # depth; one global range would saturate the deep layers and flatten the
    # shallow ones. Pooled across sessions, still per layer.
    def per_layer_bounds(arrs, q_lo=0.05, q_hi=0.95):
        pooled = np.concatenate(arrs, axis=0)            # (sum_T, L)
        return [quantile_bounds(pooled[:, l], q_lo, q_hi) for l in range(pooled.shape[1])]

    stats["energy"] = per_layer_bounds(energy)
    stats["delta"] = per_layer_bounds(delta)

    # These two are normalised globally across layers, matching the original.
    stats["cos"] = quantile_bounds(np.concatenate([a.reshape(-1) for a in cos_inst]))
    stats["sparsity"] = quantile_bounds(np.concatenate([a.reshape(-1) for a in sparsity]))

    # Seam: pooled z-score parameters, then pooled quantile bounds on the sum.
    sd = np.concatenate(seam_delta)
    sc = np.concatenate(seam_cos)
    stats["seam_z"] = {
        "delta_mu": float(sd.mean()), "delta_sd": float(sd.std()),
        "cos_mu": float(sc.mean()), "cos_sd": float(sc.std()),
    }
    pooled_seam = ((sd - sd.mean()) / (sd.std() + 1e-6)) + ((sc - sc.mean()) / (sc.std() + 1e-6))
    stats["seam"] = quantile_bounds(pooled_seam, 0.60, 0.995)

    print(f"  bounds fixed over {len(stems)} sessions\n")
    return stats


def _decode_token_pieces(input_ids, meta):
    """Decode per-token display strings with the model's own tokenizer.

    Pieces are produced by incremental decoding rather than per-id decoding, so
    that byte-pair fragments join into the characters a reader expects and
    leading spaces are preserved. A token that contributes no visible characters
    is shown as a middle dot rather than as nothing, so column and text stay in
    step.

    Returns None (with a warning) if the tokenizer is unavailable.
    """
    model_id = meta.get("model_id", "")
    if not model_id:
        print("  WARNING: no model_id in metadata; cannot decode tokens")
        return None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    except Exception as exc:
        print(f"  WARNING: tokenizer unavailable ({type(exc).__name__}): {exc}")
        return None

    ids_list = input_ids.tolist()
    pieces = []
    prev_len = 0
    for i in range(len(ids_list)):
        decoded = tok.decode(ids_list[: i + 1])
        piece = decoded[prev_len:]
        pieces.append(piece if piece else "·")
        prev_len = len(decoded)
    print(f"  Decoded {len(pieces)} token pieces")
    return pieces


def _reuse_cached_token_pieces(bundle_path, expected_len):
    """Recover token_pieces from a previously generated bundle.

    Used when the tokenizer cannot be loaded (no network, no local cache). The
    length check is the safety rail: a mismatch means the cached bundle is not
    this session, and silently accepting it would desynchronise every column
    from its text.
    """
    if not bundle_path.exists():
        print(f"  WARNING: no cached bundle at {bundle_path.name}; text panel will be empty")
        return None
    try:
        cached = json.loads(bundle_path.read_text()).get("token_pieces")
    except Exception as exc:
        print(f"  WARNING: could not read cached bundle: {exc}")
        return None
    if not cached:
        print("  WARNING: cached bundle has no token_pieces")
        return None
    if len(cached) != expected_len:
        raise ValueError(
            f"cached token_pieces length {len(cached)} != token count {expected_len} "
            f"in {bundle_path.name}; refusing to ship a desynchronised bundle"
        )
    print(f"  Reused {len(cached)} cached token pieces (tokenizer unavailable)")
    return cached


def convert_session(stem, data_dir, output_dir, pca_mean, pca_components, stats):
    """Convert one session to web-ready format using pooled normalisation bounds."""
    t0 = time.time()

    act_path = data_dir / f"{stem}_activations.npz"
    ids_path = data_dir / f"{stem}_input_ids.npy"
    meta_path = data_dir / f"{stem}_metadata.json"
    text_path = data_dir / f"{stem}_text.txt"

    act = np.load(str(act_path))
    jl = act["jl"].astype(np.float32)
    jl_energy = act["jl_energy"].astype(np.float32)
    delta_l2 = act["delta_l2"].astype(np.float32)
    cos_prev = act["cos_prev"].astype(np.float32)
    top1_frac = act["top1_frac"].astype(np.float32)
    top25_frac = act["top25_frac"].astype(np.float32)
    # Repaired on read: the top layer as extracted is double-normalised and is
    # replaced by the model's true output entropy. See activations.py.
    logit_lens_entropy = activations.logit_lens_entropy(act)
    input_ids = np.load(str(ids_path)).astype(np.int64)
    meta = json.loads(meta_path.read_text())
    full_text = text_path.read_text(encoding="utf-8") if text_path.exists() else ""

    T, L, D = jl.shape

    # --- PCA RGB using shared transform ---
    jl_flat = jl.reshape(-1, D)
    pca_3d = (jl_flat - pca_mean) @ pca_components.T  # (T*L, 3)
    pca_3d = pca_3d.reshape(T, L, 3)

    # Normalise each component against the POOLED bounds, not this session's.
    pca_rgb = np.zeros_like(pca_3d)
    for c in range(3):
        lo, hi = stats["pca"][c]
        pca_rgb[:, :, c] = apply_bounds(pca_3d[:, :, c], lo, hi)

    # Boost saturation
    pca_rgb = 0.5 + (pca_rgb - 0.5) * 1.3
    pca_rgb = np.clip(pca_rgb, 0.0, 1.0)

    rgb_uint8 = to_uint8(pca_rgb)  # (T, L, 3) uint8

    # --- Normalised metrics, all against pooled bounds ---
    energy_norm = np.zeros_like(jl_energy)
    delta_norm = np.zeros_like(delta_l2)
    for ell in range(L):
        energy_norm[:, ell] = apply_bounds(jl_energy[:, ell], *stats["energy"][ell])
        delta_norm[:, ell] = apply_bounds(delta_l2[:, ell], *stats["delta"][ell])

    cos_instability = apply_bounds(1.0 - cos_prev, *stats["cos"])
    sparsity_norm = apply_bounds(top1_frac * 0.6 + top25_frac * 0.4, *stats["sparsity"])

    # Token 0 is undefined for everything derived from a token-to-token
    # difference: it has no predecessor, so delta_l2[0] and cos_prev[0] are both
    # zero by construction. Zero cosine is *maximal* (1 - cos_prev), so the first
    # column of every session rendered at full grain — a saturated artefact
    # standing where a measurement should be. `seam_score` has always zeroed it;
    # these three did not until 2026-07-28, while the docs claimed all derived
    # quantities excluded it.
    #
    # Zero is the honest fill: it renders as "nothing here" rather than as an
    # extreme reading. It is one column of ~3,000 and it is not a measurement.
    for undefined_at_token_0 in (delta_norm, cos_instability, sparsity_norm):
        undefined_at_token_0[0, :] = 0.0

    # --- Logit-lens entropy on an ABSOLUTE scale ---
    # Unlike every other array here, this is NOT quantile-normalised. Quantile
    # normalisation is applied per layer, which would erase exactly the signal
    # worth looking at: entropy differs enormously BY layer (early layers are
    # near-uniform over the vocabulary, the last layer is nearly committed), and
    # that cross-layer gradient is the phenomenon. Dividing by ln(vocab) keeps
    # it, and makes the overlay mean the same thing in every session.
    observed_max = float(np.nanmax(logit_lens_entropy))
    if observed_max > VOCAB_LN + 1e-3:
        raise ValueError(
            f"logit_lens_entropy max {observed_max:.4f} exceeds ln(vocab)={VOCAB_LN:.4f}; "
            "VOCAB_LN is wrong for this model and the entropy overlay would clip."
        )
    entropy_norm = np.clip(logit_lens_entropy / VOCAB_LN, 0.0, 1.0)

    # --- Seam score ---
    # Standardised against POOLED means and standard deviations, so that a seam
    # of 0.8 means the same thing in every session, and so that two runs sharing
    # a prefix score it identically. Token 0 is forced to zero: it has no
    # predecessor, so delta_l2[0] and cos_prev[0] are both 0 in the extracted
    # data, which makes (1 - cos_prev) maximal and would score the first token of
    # every session as a spurious seam.
    mid_layer = int(0.6 * L)
    z = stats["seam_z"]
    seam_raw = ((delta_l2[:, mid_layer] - z["delta_mu"]) / (z["delta_sd"] + 1e-6)
                + ((1.0 - cos_prev[:, mid_layer]) - z["cos_mu"]) / (z["cos_sd"] + 1e-6))
    seam_score = apply_bounds(seam_raw, *stats["seam"])
    seam_score[0] = 0.0

    # --- Token pieces ---
    # Preferred source is the real tokenizer. Falling back to a previously
    # generated bundle matters more than it looks: without token_pieces the
    # viewer's text panel and inspector go blank, and the failure is silent \u2014
    # the app still loads and animates, so a regeneration run on a machine
    # without network access would quietly ship a broken build. Reusing the
    # cached pieces keeps offline regeneration honest, and the length assertion
    # catches the one way it could be wrong (a bundle from a different session).
    token_pieces = _decode_token_pieces(input_ids, meta)
    if token_pieces is None:
        token_pieces = _reuse_cached_token_pieces(output_dir / f"{stem}.json", T)

    # --- Turn boundaries ---
    turns = []
    if "turn_boundaries" in meta:
        turns = meta["turn_boundaries"]
    else:
        turns = detect_turns(input_ids.tolist(), token_pieces, T)

    # --- Per-token roles ---
    token_roles = [""] * T
    for tb in turns:
        ts = tb.get("token_start", 0)
        te = tb.get("token_end", T)
        role = tb.get("role", "")
        for t_idx in range(ts, min(te, T)):
            token_roles[t_idx] = role

    # --- Quantise JL vectors for reference-distance and custom-basis modes ---
    jl_quantized, jl_min, jl_max = quantize_jl(jl, stats["jl_min"], stats["jl_max"])

    # --- Build output ---
    # Binary arrays: pack as base64 for JSON transport
    def b64(arr):
        return base64.b64encode(arr.tobytes()).decode('ascii')

    session_data = {
        "stem": stem,
        "display_name": DISPLAY_NAMES.get(stem, stem),
        "n_tokens": int(T),
        "n_layers": int(L),
        "jl_dim": int(D),

        # Pre-computed PCA RGB: T*L*3 uint8 (row-major: token, layer, channel)
        "rgb": b64(rgb_uint8),

        # Quantised JL vectors: T*L*D uint8, with per-(layer, dim) bounds.
        # jl_bounds_shape distinguishes this from the older per-dim format so a
        # stale bundle cannot be silently misread by a newer viewer.
        "jl": b64(jl_quantized),
        "jl_min": jl_min.reshape(-1).tolist(),
        "jl_max": jl_max.reshape(-1).tolist(),
        "jl_bounds_shape": [int(L), int(D)],

        # Normalized metrics (all T*L uint8)
        "energy_norm": b64(to_uint8(energy_norm)),
        "delta_norm": b64(to_uint8(delta_norm)),
        "cos_instability": b64(to_uint8(cos_instability)),
        "sparsity_norm": b64(to_uint8(sparsity_norm)),

        # Absolute scale: 0 = committed, 1 = uniform over vocabulary.
        "entropy_norm": b64(to_uint8(entropy_norm)),
        "entropy_scale_nats": round(VOCAB_LN, 6),

        # Seam score: T uint8
        "seam_score": b64(to_uint8(seam_score)),

        # Text data
        "token_pieces": token_pieces,
        "token_roles": token_roles,
        "turns": turns,

        # Metadata
        "model_id": meta.get("model_id", ""),
        "full_text": full_text,
    }

    # Write JSON
    out_path = output_dir / f"{stem}.json"
    with open(out_path, "w") as f:
        json.dump(session_data, f)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"  -> {out_path.name}: {size_mb:.1f} MB ({elapsed:.1f}s)")

    return {
        "stem": stem,
        "display_name": DISPLAY_NAMES.get(stem, stem),
        "n_tokens": int(T),
        "n_layers": int(L),
        "file": f"{stem}.json",
        "size_mb": round(size_mb, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stems", nargs="+", default=None,
                        help="Convert only these session stems (default: all discovered).")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Permit --stems even though normalisation bounds are pooled "
                             "across all sessions. Only safe if the bounds are unchanged.")
    args = parser.parse_args()

    print("=== Phosphenes Web Data Converter ===\n")

    # Load shared PCA transform
    pca_path = DATA_DIR / "shared_pca_transform.npz"
    if not pca_path.exists():
        print(f"ERROR: Shared PCA transform not found at {pca_path}")
        sys.exit(1)

    pca_data = np.load(str(pca_path))
    pca_mean = pca_data["mean"]  # (16,)
    pca_components = pca_data["components"]  # (3, 16)
    print(f"Loaded shared PCA transform: explained variance = {pca_data['explained_variance_ratio']}")

    # Discover sessions
    stems = set()
    for f in os.listdir(DATA_DIR):
        if f.endswith("_activations.npz"):
            stem = f.replace("_activations.npz", "")
            if stem not in SKIP_STEMS:
                stems.add(stem)

    # Sort by desired order
    ordered_stems = [s for s in SESSION_ORDER if s in stems]
    # Add any remaining stems not in SESSION_ORDER
    for s in sorted(stems):
        if s not in ordered_stems:
            ordered_stems.append(s)

    if args.stems:
        unknown = [s for s in args.stems if s not in ordered_stems]
        if unknown:
            print(f"ERROR: unknown stems: {', '.join(unknown)}")
            print(f"Available: {', '.join(ordered_stems)}")
            sys.exit(1)
        if not args.allow_partial:
            print("ERROR: --stems converts a subset, but normalisation bounds are pooled\n"
                  "       across ALL sessions (see compute_global_stats). A subset would be\n"
                  "       scaled differently from the bundles already on disk, so colour and\n"
                  "       brightness would no longer mean the same thing between them.\n"
                  "       Run without --stems to reconvert everything, or pass --allow-partial\n"
                  "       if you know the bounds have not changed.")
            sys.exit(1)
        selected = [s for s in ordered_stems if s in args.stems]
    else:
        selected = ordered_stems

    print(f"Found {len(ordered_stems)} sessions; converting {len(selected)}\n")

    # Bounds are always pooled over ALL sessions, not just the selected ones, so
    # that a --allow-partial run produces bundles consistent with the full set.
    stats = compute_global_stats(ordered_stems, DATA_DIR, pca_mean, pca_components)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pass 2: convert each selected session against the pooled bounds.
    print("Pass 2: writing bundles")
    converted = {}
    for i, stem in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] Converting {stem}...")
        converted[stem] = convert_session(stem, DATA_DIR, OUTPUT_DIR,
                                          pca_mean, pca_components, stats)

    # Write session index. On a partial run, entries for sessions that were not
    # reconverted are carried over from the existing index rather than dropped —
    # otherwise `--stems X` would silently delete every other session from the
    # viewer's menu while leaving their bundles on disk.
    index_path = OUTPUT_DIR / "sessions.json"
    previous = {}
    if index_path.exists():
        try:
            previous = {s["stem"]: s for s in json.loads(index_path.read_text())["sessions"]}
        except Exception as exc:
            print(f"WARNING: could not read existing index ({exc}); rebuilding from scratch")

    session_index = []
    for stem in ordered_stems:
        if stem in converted:
            session_index.append(converted[stem])
        elif stem in previous and (OUTPUT_DIR / f"{stem}.json").exists():
            session_index.append(previous[stem])
            print(f"  (kept existing index entry for {stem})")

    with open(index_path, "w") as f:
        json.dump({"sessions": session_index}, f, indent=2)
    print(f"\nSession index: {index_path}")

    total_mb = sum(s["size_mb"] for s in session_index)
    print(f"\nTotal data size: {total_mb:.1f} MB across {len(session_index)} sessions")
    print("Done!")


if __name__ == "__main__":
    main()
