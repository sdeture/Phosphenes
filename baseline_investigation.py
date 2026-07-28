"""
Tonic Negative Baseline Investigation
======================================
Three tests to probe the robustness and interpretation of the negative
tonic affect baseline found in Phosphenes session data.

Test 1: Permutation null (Phosphenes)
  - Shuffle positive/negative word labels 1000x
  - Build a new affect direction each time
  - Compute the tonic baseline under each random direction
  - Ask: is the real baseline significantly more negative than chance?

Test 2: Layer 0 control (Phosphenes)
  - Compute the affect direction and tonic baseline at layer 0 (embedding)
  - Compare to layers 11, 22, 33, 44, 55
  - Ask: is the negative baseline already present before any transformer
    computation, or does it emerge through the layers?

Test 3: Cross-architecture validation (LayerTime)
  - Apply the same methodology to 4 different architectures:
    Qwen3-30B-A3B (MoE), Qwen3-14B (Dense), ERNIE-4.5 (MoE), GLM (Dense)
  - Ask: does the negative baseline replicate across architectures?
  - Secondary: does it differ between MoE and Dense?

Output: baseline_investigation_results.json + printed summary
"""

from pathlib import Path

# Repository root, resolved from this file's location. Absolute paths were
# hardcoded here, which made every one of these scripts unrunnable outside one
# machine; two of them also pointed outside the repository entirely.
_ROOT = Path(__file__).resolve().parent

import json
import base64
import numpy as np
import os
import glob
from collections import defaultdict

# ====================================================================
# Constants
# ====================================================================

PHOSPHENES_DATA_DIR = str(_ROOT / "web" / "data")
# NOT PRESENT IN THIS REPOSITORY. The cross-architecture test that reads this
# directory recorded 100% errors in baseline_investigation_results.json because
# every input file was missing, i.e. it never ran. Override with the env var
# LAYERTIME_DATA_DIR if you have that package; otherwise the test will skip.
LAYERTIME_DATA_DIR = os.environ.get("LAYERTIME_DATA_DIR", str(_ROOT / "_missing_layertime"))
OUTPUT_PATH = str(_ROOT / "baseline_investigation_results.json")

PHOSPHENES_SESSION_FILES = [
    "Dream_conv_00173_run1.json",
    "Dream_conv_00178_run1.json",
    "Dream_conv_00181_run1.json",
    "Dream_conv_00187_run1.json",
    "Dream_conv_00191_run1.json",
    "Dream_conv_00194_run1.json",
    "Dream_greedy_clean.json",
    "Dream_greedy_sentient.json",
]

POSITIVE_WORDS = {
    "joy", "love", "beauty", "wonder", "warm", "light", "hope", "dream",
    "alive", "discover", "curiosity", "delight", "gentle", "peaceful", "glow"
}

NEGATIVE_WORDS = {
    "fear", "dark", "cold", "empty", "lost", "alone", "pain", "silent",
    "fade", "shadow", "broken", "nothing", "gone", "dust", "ache"
}

ALL_AFFECT_WORDS = list(POSITIVE_WORDS) + list(NEGATIVE_WORDS)

# Original 5 layers + layer 0 for the embedding control
LAYERS_OF_INTEREST = [11, 22, 33, 44, 55]
ALL_LAYERS = [0] + LAYERS_OF_INTEREST

N_PERMUTATIONS = 1000
RNG_SEED = 2026

# LayerTime runs to use (one representative per architecture)
LAYERTIME_RUNS = {
    "Qwen3-30B-A3B (MoE)": "Base_run1",
    "Qwen3-14B (Dense)": "Qwen14B_Base_run1",
    "ERNIE-4.5 (MoE)": "ERNIE_Base_run1",
    "GLM-4.7-Flash (Dense)": "GLM_Claude_Distill_run1",
}

# Additional LayerTime runs for replication within architecture
LAYERTIME_EXTRA_RUNS = {
    "Qwen3-30B Instruct": "Instruct_run1",
    "Qwen3-30B Thinking": "Thinking_run2",
    "Qwen3-14B Reasoning": "Qwen14B_Reasoning_run1",
    "ERNIE PT": "ERNIE_PT_run1",
    "ERNIE Thinking": "ERNIE_Thinking_run1",
}


# ====================================================================
# Data Loading
# ====================================================================

def token_affect_label(token_piece):
    """Return 'positive', 'negative', or None."""
    t = token_piece.lower().strip().lstrip("\u2581\u0120 ").strip(".,!?;:'\"")
    for w in POSITIVE_WORDS:
        if t == w or t.startswith(w):
            return "positive"
    for w in NEGATIVE_WORDS:
        if t == w or t.startswith(w):
            return "negative"
    return None


def load_phosphenes_session(filepath):
    """Load and dequantize a Phosphenes session JSON."""
    with open(filepath) as f:
        d = json.load(f)
    n_tokens = d["n_tokens"]
    n_layers = d["n_layers"]
    jl_dim = d["jl_dim"]
    jl_bytes = base64.b64decode(d["jl"])
    jl_uint8 = np.frombuffer(jl_bytes, dtype=np.uint8).reshape(n_tokens, n_layers, jl_dim)
    jl_min = np.array(d["jl_min"], dtype=np.float32)
    jl_max = np.array(d["jl_max"], dtype=np.float32)
    jl_float = jl_uint8.astype(np.float32) / 255.0 * (jl_max - jl_min) + jl_min
    return {
        "stem": d.get("stem", os.path.splitext(os.path.basename(filepath))[0]),
        "display_name": d.get("display_name", ""),
        "n_tokens": n_tokens,
        "n_layers": n_layers,
        "jl_dim": jl_dim,
        "jl": jl_float,  # (n_tokens, n_layers, 16)
        "token_pieces": d["token_pieces"],
    }


def load_layertime_run(stem):
    """Load a LayerTime NPZ file + text + metadata."""
    npz_path = os.path.join(LAYERTIME_DATA_DIR, f"{stem}_activations.npz")
    text_path = os.path.join(LAYERTIME_DATA_DIR, f"{stem}_text.txt")
    meta_path = os.path.join(LAYERTIME_DATA_DIR, f"{stem}_metadata.json")
    ids_path = os.path.join(LAYERTIME_DATA_DIR, f"{stem}_input_ids.npy")

    npz = np.load(npz_path)
    jl = npz["jl"]  # (T, L, 16), float16

    with open(meta_path) as f:
        meta = json.load(f)

    # Load input_ids for tokenization
    input_ids = np.load(ids_path)

    # Load text for token matching
    # We need to decode token_ids to pieces for affect matching.
    # The text file has the full conversation but not per-token pieces.
    # We'll use transformers tokenizer if available, else fall back to text heuristic.
    token_pieces = _decode_token_ids(input_ids, meta.get("model_id", ""))

    return {
        "stem": stem,
        "display_name": meta.get("model_name", stem),
        "model_id": meta.get("model_id", ""),
        "n_tokens": int(meta["num_tokens"]),
        "n_layers": int(meta["num_layers"]),
        "jl_dim": int(meta["jl_dim"]),
        "d_model": int(meta["d_model"]),
        "jl": jl.astype(np.float32),  # upcast from float16
        "token_pieces": token_pieces,
        "input_ids": input_ids,
    }


def _decode_token_ids(input_ids, model_id):
    """
    Attempt to decode token IDs to string pieces.
    Falls back to reading from text file if tokenizer unavailable.
    For affect word matching, we only need rough token pieces.
    """
    try:
        from transformers import AutoTokenizer
        # Use Qwen tokenizer as default -- most LayerTime models are Qwen-based
        # and the others share similar vocab for English words
        tok_name = model_id if model_id else "Qwen/Qwen3-30B-A3B"
        tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
        pieces = [tokenizer.decode([tid]) for tid in input_ids]
        return pieces
    except Exception:
        # Fallback: return empty strings, matching will just find nothing
        # This is OK -- we'll report the match count and it'll be visible
        return [""] * len(input_ids)


# ====================================================================
# Core Analysis Functions
# ====================================================================

def build_affect_direction(jl_data, labels, layer_idx):
    """
    Build affect direction at a single layer.
    jl_data: (n_tokens, n_layers, 16)
    labels: list of 'positive', 'negative', or None per token
    layer_idx: which layer to use
    Returns: unit direction vector (16,) or None
    """
    pos_vecs = []
    neg_vecs = []
    for i, lbl in enumerate(labels):
        if layer_idx >= jl_data.shape[1]:
            break
        v = jl_data[i, layer_idx, :]
        if lbl == "positive":
            pos_vecs.append(v)
        elif lbl == "negative":
            neg_vecs.append(v)

    if len(pos_vecs) < 2 or len(neg_vecs) < 2:
        return None

    pos_mean = np.mean(pos_vecs, axis=0)
    neg_mean = np.mean(neg_vecs, axis=0)
    direction = pos_mean - neg_mean
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm
    return direction


def compute_baseline(jl_data, labels, direction, layer_idx):
    """
    Compute tonic baseline: mean projection of non-affect tokens.
    Returns: float (baseline value)
    """
    projs = jl_data[:, layer_idx, :] @ direction
    mask = np.array([lbl is None for lbl in labels])
    if mask.sum() == 0:
        return float(np.mean(projs))
    return float(np.mean(projs[mask]))


def compute_cohens_d(jl_data, labels, direction, layer_idx):
    """Cohen's d between positive and negative token projections."""
    projs = jl_data[:, layer_idx, :] @ direction
    pos_vals = [projs[i] for i, lbl in enumerate(labels) if lbl == "positive"]
    neg_vals = [projs[i] for i, lbl in enumerate(labels) if lbl == "negative"]
    if len(pos_vals) < 2 or len(neg_vals) < 2:
        return float("nan")
    pos_arr = np.array(pos_vals)
    neg_arr = np.array(neg_vals)
    pooled_std = np.sqrt((np.var(pos_arr, ddof=1) + np.var(neg_arr, ddof=1)) / 2)
    if pooled_std == 0:
        return 0.0
    return float((np.mean(pos_arr) - np.mean(neg_arr)) / pooled_std)


# ====================================================================
# Test 1: Permutation Null
# ====================================================================

def run_permutation_test(sessions, n_perms=N_PERMUTATIONS, seed=RNG_SEED):
    """
    For each permutation:
      1. Take the same set of tokens that matched positive OR negative words
      2. Randomly re-assign half as "positive" and half as "negative"
      3. Build a new affect direction from the shuffled labels
      4. Compute the tonic baseline under that random direction
    Compare real baseline to the null distribution.
    """
    print("\n" + "=" * 70)
    print("TEST 1: PERMUTATION NULL")
    print("=" * 70)

    rng = np.random.default_rng(seed)

    # Pool all tokens across sessions with their labels
    # For the permutation, we use the 80D concatenated direction (layers 11-55)
    # to match the original analysis.

    # First, compute real baselines per session using pooled direction
    all_labels = []
    all_jl_concat = []  # 80D vectors for affect tokens
    all_jl_concat_all = []  # 80D vectors for ALL tokens (for baseline)
    all_is_affect = []  # boolean

    session_token_ranges = []  # (start, end) per session

    offset = 0
    for sess in sessions:
        jl = sess["jl"]
        labels = [token_affect_label(tp) for tp in sess["token_pieces"]]
        n = sess["n_tokens"]

        for i in range(n):
            parts = []
            ok = True
            for L in LAYERS_OF_INTEREST:
                if L < sess["n_layers"]:
                    parts.append(jl[i, L, :])
                else:
                    ok = False
                    break
            if not ok:
                continue

            v80 = np.concatenate(parts)
            all_jl_concat_all.append(v80)
            all_labels.append(labels[i])
            all_is_affect.append(labels[i] is not None)

            if labels[i] is not None:
                all_jl_concat.append(v80)

        session_token_ranges.append((offset, offset + n))
        offset += n

    all_jl_concat = np.array(all_jl_concat)       # (n_affect, 80)
    all_jl_concat_all = np.array(all_jl_concat_all) # (n_all, 80)
    all_is_affect = np.array(all_is_affect)        # (n_all,)

    # Get indices of positive vs negative in the affect-only array
    affect_labels = [lbl for lbl in all_labels if lbl is not None]
    n_affect = len(affect_labels)
    n_positive = sum(1 for l in affect_labels if l == "positive")
    n_negative = sum(1 for l in affect_labels if l == "negative")

    print(f"  Total tokens: {len(all_labels)}")
    print(f"  Affect tokens: {n_affect} ({n_positive} positive, {n_negative} negative)")
    print(f"  Neutral tokens: {sum(1 for l in all_labels if l is None)}")

    # --- Real baseline ---
    real_pos_mask = np.array([l == "positive" for l in affect_labels])
    real_neg_mask = np.array([l == "negative" for l in affect_labels])

    real_pos_mean = np.mean(all_jl_concat[real_pos_mask], axis=0)
    real_neg_mean = np.mean(all_jl_concat[real_neg_mask], axis=0)
    real_direction = real_pos_mean - real_neg_mean
    real_norm = np.linalg.norm(real_direction)
    if real_norm > 0:
        real_direction = real_direction / real_norm

    # Project all tokens onto real direction
    real_projs = all_jl_concat_all @ real_direction
    neutral_mask = ~all_is_affect
    real_baseline = float(np.mean(real_projs[neutral_mask]))

    print(f"\n  Real affect direction norm (pre-normalization): {real_norm:.4f}")
    print(f"  Real tonic baseline (80D): {real_baseline:.4f}")

    # --- Permutation null ---
    print(f"\n  Running {n_perms} permutations...")
    null_baselines = []

    for perm_i in range(n_perms):
        # Shuffle which affect tokens are "positive" vs "negative"
        perm_labels = np.zeros(n_affect, dtype=bool)
        perm_labels[:n_positive] = True
        rng.shuffle(perm_labels)

        perm_pos_mean = np.mean(all_jl_concat[perm_labels], axis=0)
        perm_neg_mean = np.mean(all_jl_concat[~perm_labels], axis=0)
        perm_direction = perm_pos_mean - perm_neg_mean
        perm_norm = np.linalg.norm(perm_direction)
        if perm_norm > 0:
            perm_direction = perm_direction / perm_norm

        perm_projs = all_jl_concat_all @ perm_direction
        perm_baseline = float(np.mean(perm_projs[neutral_mask]))
        null_baselines.append(perm_baseline)

        if (perm_i + 1) % 200 == 0:
            print(f"    {perm_i + 1}/{n_perms} done")

    null_baselines = np.array(null_baselines)

    # Compute p-value: fraction of null baselines <= real baseline
    # (one-sided test: is the real baseline unusually negative?)
    p_value = float(np.mean(null_baselines <= real_baseline))

    # Also compute two-sided: fraction with |null| >= |real|
    p_two_sided = float(np.mean(np.abs(null_baselines) >= np.abs(real_baseline)))

    null_mean = float(np.mean(null_baselines))
    null_std = float(np.std(null_baselines))
    null_min = float(np.min(null_baselines))
    null_max = float(np.max(null_baselines))

    z_score = (real_baseline - null_mean) / null_std if null_std > 0 else 0.0

    print(f"\n  --- Permutation Results ---")
    print(f"  Real baseline:          {real_baseline:.4f}")
    print(f"  Null distribution:      mean={null_mean:.4f}, std={null_std:.4f}")
    print(f"  Null range:             [{null_min:.4f}, {null_max:.4f}]")
    print(f"  Z-score:                {z_score:.4f}")
    print(f"  p (one-sided, <= real): {p_value:.4f}")
    print(f"  p (two-sided):          {p_two_sided:.4f}")

    if p_value < 0.001:
        interpretation = "SIGNIFICANT: Real baseline is more negative than >99.9% of random directions"
    elif p_value < 0.01:
        interpretation = "SIGNIFICANT: Real baseline is more negative than >99% of random directions"
    elif p_value < 0.05:
        interpretation = "SIGNIFICANT: Real baseline is more negative than >95% of random directions"
    else:
        interpretation = "NOT SIGNIFICANT: Random directions produce equally negative baselines"
    print(f"  Interpretation: {interpretation}")

    return {
        "real_baseline": real_baseline,
        "null_mean": null_mean,
        "null_std": null_std,
        "null_min": null_min,
        "null_max": null_max,
        "null_percentiles": {
            "p1": float(np.percentile(null_baselines, 1)),
            "p5": float(np.percentile(null_baselines, 5)),
            "p25": float(np.percentile(null_baselines, 25)),
            "p50": float(np.percentile(null_baselines, 50)),
            "p75": float(np.percentile(null_baselines, 75)),
            "p95": float(np.percentile(null_baselines, 95)),
            "p99": float(np.percentile(null_baselines, 99)),
        },
        "z_score": z_score,
        "p_one_sided": p_value,
        "p_two_sided": p_two_sided,
        "n_permutations": n_perms,
        "n_affect_tokens": n_affect,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_neutral": int(neutral_mask.sum()),
        "interpretation": interpretation,
    }


# ====================================================================
# Test 2: Layer 0 Control
# ====================================================================

def run_layer0_control(sessions):
    """
    Compute the affect direction and tonic baseline at every layer from 0
    through 55 (skipping in between the layers of interest for efficiency,
    but including 0).

    If the negative baseline is already present at layer 0, it suggests
    the effect is in the token embeddings, not in transformer computation.
    If it emerges through layers, it suggests the transformer is creating it.
    """
    print("\n" + "=" * 70)
    print("TEST 2: LAYER 0 CONTROL")
    print("=" * 70)

    # Test layers: 0, 5, 11, 16, 22, 27, 33, 38, 44, 49, 55, 60, 63
    test_layers = [0, 5, 11, 16, 22, 27, 33, 38, 44, 49, 55, 60, 63]

    # Pool labels across all sessions
    all_session_data = []
    for sess in sessions:
        labels = [token_affect_label(tp) for tp in sess["token_pieces"]]
        all_session_data.append((sess["jl"], labels, sess["n_layers"]))

    results_by_layer = {}

    for L in test_layers:
        # Pool positive and negative vectors across sessions
        pos_vecs = []
        neg_vecs = []
        for jl, labels, n_layers in all_session_data:
            if L >= n_layers:
                continue
            for i, lbl in enumerate(labels):
                if lbl == "positive":
                    pos_vecs.append(jl[i, L, :])
                elif lbl == "negative":
                    neg_vecs.append(jl[i, L, :])

        if len(pos_vecs) < 2 or len(neg_vecs) < 2:
            results_by_layer[L] = {
                "baseline": float("nan"),
                "cohens_d": float("nan"),
                "n_positive": len(pos_vecs),
                "n_negative": len(neg_vecs),
            }
            continue

        # Build affect direction
        pos_mean = np.mean(pos_vecs, axis=0)
        neg_mean = np.mean(neg_vecs, axis=0)
        direction = pos_mean - neg_mean
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm

        # Compute baseline and Cohen's d across all sessions
        all_projs = []
        all_labels_flat = []
        for jl, labels, n_layers in all_session_data:
            if L >= n_layers:
                continue
            projs = jl[:, L, :] @ direction
            all_projs.extend(projs.tolist())
            all_labels_flat.extend(labels)

        all_projs = np.array(all_projs)
        neutral_mask = np.array([l is None for l in all_labels_flat])

        baseline = float(np.mean(all_projs[neutral_mask])) if neutral_mask.sum() > 0 else float("nan")

        # Cohen's d
        pos_projs = np.array([all_projs[i] for i, l in enumerate(all_labels_flat) if l == "positive"])
        neg_projs = np.array([all_projs[i] for i, l in enumerate(all_labels_flat) if l == "negative"])
        pooled_std = np.sqrt((np.var(pos_projs, ddof=1) + np.var(neg_projs, ddof=1)) / 2)
        cohens_d = float((np.mean(pos_projs) - np.mean(neg_projs)) / pooled_std) if pooled_std > 0 else 0.0

        # Direction norm (pre-normalization) as a measure of separation magnitude
        direction_magnitude = float(norm)

        results_by_layer[L] = {
            "baseline": baseline,
            "cohens_d": cohens_d,
            "direction_magnitude": direction_magnitude,
            "n_positive": len(pos_vecs),
            "n_negative": len(neg_vecs),
            "n_neutral": int(neutral_mask.sum()),
        }

    # Print summary table
    print(f"\n  {'Layer':<8} {'Baseline':>10} {'Cohen d':>10} {'Dir. Mag.':>10} {'N_pos':>8} {'N_neg':>8}")
    print("  " + "-" * 58)
    for L in test_layers:
        r = results_by_layer[L]
        baseline = r["baseline"]
        cd = r["cohens_d"]
        mag = r.get("direction_magnitude", float("nan"))
        baseline_str = f"{baseline:.4f}" if not np.isnan(baseline) else "N/A"
        cd_str = f"{cd:.4f}" if not np.isnan(cd) else "N/A"
        mag_str = f"{mag:.4f}" if not np.isnan(mag) else "N/A"
        print(f"  L{L:<6} {baseline_str:>10} {cd_str:>10} {mag_str:>10} {r['n_positive']:>8} {r['n_negative']:>8}")

    # Interpretation
    l0_baseline = results_by_layer[0]["baseline"]
    l44_baseline = results_by_layer[44]["baseline"]

    print(f"\n  Layer 0 baseline:  {l0_baseline:.4f}")
    print(f"  Layer 44 baseline: {l44_baseline:.4f}")

    if np.isnan(l0_baseline):
        interp = "INCONCLUSIVE: Could not compute layer 0 baseline"
    elif abs(l0_baseline) < 1.0 and abs(l44_baseline) > 5.0:
        interp = "EMERGENT: Baseline is near-zero at embedding, becomes negative through layers"
    elif l0_baseline < -5.0:
        interp = "PRE-EXISTING: Negative baseline already present in token embeddings"
    else:
        ratio = l0_baseline / l44_baseline if l44_baseline != 0 else float("nan")
        interp = f"MIXED: Layer 0 baseline is {l0_baseline:.2f}, layer 44 is {l44_baseline:.2f} (ratio: {ratio:.2f})"

    print(f"  Interpretation: {interp}")

    return {
        "layer_results": {str(L): results_by_layer[L] for L in test_layers},
        "interpretation": interp,
        "test_layers": test_layers,
    }


# ====================================================================
# Test 3: Cross-Architecture Validation (LayerTime)
# ====================================================================

def run_cross_architecture(primary_runs, extra_runs=None):
    """
    Apply affect direction methodology to multiple architectures from
    the LayerTime dataset.
    """
    print("\n" + "=" * 70)
    print("TEST 3: CROSS-ARCHITECTURE VALIDATION")
    print("=" * 70)

    all_runs = dict(primary_runs)
    if extra_runs:
        all_runs.update(extra_runs)

    results = {}

    for label, stem in all_runs.items():
        print(f"\n  --- {label} ({stem}) ---")

        try:
            run = load_layertime_run(stem)
        except Exception as e:
            print(f"  ERROR loading {stem}: {e}")
            results[label] = {"error": str(e)}
            continue

        jl = run["jl"]
        n_tokens = run["n_tokens"]
        n_layers = run["n_layers"]
        token_pieces = run["token_pieces"]

        # Label tokens
        labels = [token_affect_label(tp) for tp in token_pieces]
        n_pos = sum(1 for l in labels if l == "positive")
        n_neg = sum(1 for l in labels if l == "negative")
        n_neutral = sum(1 for l in labels if l is None)

        print(f"    Tokens: {n_tokens}, Layers: {n_layers}, d_model: {run['d_model']}")
        print(f"    Affect tokens: {n_pos} positive, {n_neg} negative, {n_neutral} neutral")

        if n_pos < 2 or n_neg < 2:
            print(f"    SKIPPED: Too few affect tokens for direction construction")
            results[label] = {
                "error": "Too few affect tokens",
                "n_positive": n_pos,
                "n_negative": n_neg,
            }
            continue

        # Pick layers to test: layer 0, ~25%, ~50%, ~75%, last layer
        layer_fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
        test_layers = sorted(set(
            [0] + [max(0, min(n_layers - 1, int(f * (n_layers - 1)))) for f in layer_fracs]
        ))

        layer_results = {}
        for L in test_layers:
            direction = build_affect_direction(jl, labels, L)
            if direction is None:
                layer_results[L] = {"baseline": float("nan"), "cohens_d": float("nan")}
                continue

            baseline = compute_baseline(jl, labels, direction, L)
            cd = compute_cohens_d(jl, labels, direction, L)

            layer_results[L] = {
                "baseline": baseline,
                "cohens_d": cd,
            }

        # Multi-layer concatenated direction (use ~5 evenly spaced layers)
        concat_layers = test_layers[1:]  # skip layer 0
        if len(concat_layers) >= 3:
            concat_pos = []
            concat_neg = []
            for i, lbl in enumerate(labels):
                parts = []
                ok = True
                for L in concat_layers:
                    if L < n_layers:
                        parts.append(jl[i, L, :])
                    else:
                        ok = False
                        break
                if not ok:
                    continue
                v = np.concatenate(parts)
                if lbl == "positive":
                    concat_pos.append(v)
                elif lbl == "negative":
                    concat_neg.append(v)

            if len(concat_pos) >= 2 and len(concat_neg) >= 2:
                concat_pos_mean = np.mean(concat_pos, axis=0)
                concat_neg_mean = np.mean(concat_neg, axis=0)
                concat_dir = concat_pos_mean - concat_neg_mean
                concat_norm = np.linalg.norm(concat_dir)
                if concat_norm > 0:
                    concat_dir = concat_dir / concat_norm

                # Project all tokens
                all_concat_vecs = []
                for i in range(n_tokens):
                    parts = []
                    ok = True
                    for L in concat_layers:
                        if L < n_layers:
                            parts.append(jl[i, L, :])
                        else:
                            ok = False
                            break
                    if ok:
                        all_concat_vecs.append(np.concatenate(parts))
                    else:
                        all_concat_vecs.append(np.zeros(len(concat_layers) * 16))

                all_concat_vecs = np.array(all_concat_vecs)
                concat_projs = all_concat_vecs @ concat_dir
                neutral_mask = np.array([l is None for l in labels])
                concat_baseline = float(np.mean(concat_projs[neutral_mask])) if neutral_mask.sum() > 0 else float("nan")
            else:
                concat_baseline = float("nan")
                concat_layers = []
        else:
            concat_baseline = float("nan")

        # Print layer results
        print(f"\n    {'Layer':<10} {'Frac':>6} {'Baseline':>10} {'Cohen d':>10}")
        print("    " + "-" * 40)
        for L in test_layers:
            r = layer_results[L]
            frac = L / (n_layers - 1) if n_layers > 1 else 0
            b_str = f"{r['baseline']:.4f}" if not np.isnan(r['baseline']) else "N/A"
            d_str = f"{r['cohens_d']:.4f}" if not np.isnan(r['cohens_d']) else "N/A"
            print(f"    L{L:<8} {frac:>6.2f} {b_str:>10} {d_str:>10}")

        if not np.isnan(concat_baseline):
            print(f"\n    Concat baseline ({len(concat_layers)} layers): {concat_baseline:.4f}")

        # Determine if baseline is negative
        mid_layer = test_layers[len(test_layers) // 2]
        mid_baseline = layer_results[mid_layer]["baseline"]

        results[label] = {
            "model_id": run["model_id"],
            "d_model": run["d_model"],
            "n_layers": n_layers,
            "n_tokens": n_tokens,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_neutral": n_neutral,
            "layer_results": {str(L): layer_results[L] for L in test_layers},
            "concat_baseline": concat_baseline,
            "concat_layers": concat_layers,
            "test_layers": test_layers,
            "mid_layer_baseline": mid_baseline,
        }

    # Summary comparison
    print("\n  " + "=" * 60)
    print("  CROSS-ARCHITECTURE SUMMARY")
    print("  " + "=" * 60)
    print(f"\n  {'Model':<30} {'Mid-Layer BL':>14} {'Concat BL':>14} {'Negative?':>10}")
    print("  " + "-" * 70)

    n_negative_models = 0
    n_total_models = 0
    moe_baselines = []
    dense_baselines = []

    for label in all_runs:
        r = results.get(label, {})
        if "error" in r:
            print(f"  {label:<30} {'ERROR':>14} {'':>14} {'':>10}")
            continue

        n_total_models += 1
        mid_bl = r.get("mid_layer_baseline", float("nan"))
        concat_bl = r.get("concat_baseline", float("nan"))
        is_neg = "YES" if (not np.isnan(concat_bl) and concat_bl < 0) else ("YES" if (not np.isnan(mid_bl) and mid_bl < 0) else "NO")
        if is_neg == "YES":
            n_negative_models += 1

        mid_str = f"{mid_bl:.4f}" if not np.isnan(mid_bl) else "N/A"
        concat_str = f"{concat_bl:.4f}" if not np.isnan(concat_bl) else "N/A"
        print(f"  {label:<30} {mid_str:>14} {concat_str:>14} {is_neg:>10}")

        # Track MoE vs Dense
        bl_val = concat_bl if not np.isnan(concat_bl) else mid_bl
        if not np.isnan(bl_val):
            if "MoE" in label or "moe" in label.lower():
                moe_baselines.append(bl_val)
            elif "Dense" in label or "dense" in label.lower():
                dense_baselines.append(bl_val)

    print(f"\n  Models with negative baseline: {n_negative_models}/{n_total_models}")

    if moe_baselines and dense_baselines:
        moe_mean = np.mean(moe_baselines)
        dense_mean = np.mean(dense_baselines)
        print(f"  MoE mean baseline:   {moe_mean:.4f} (n={len(moe_baselines)})")
        print(f"  Dense mean baseline: {dense_mean:.4f} (n={len(dense_baselines)})")
        print(f"  MoE - Dense:         {moe_mean - dense_mean:.4f}")

    return results


# ====================================================================
# Main
# ====================================================================

def main():
    print("=" * 70)
    print("TONIC NEGATIVE BASELINE INVESTIGATION")
    print("=" * 70)

    # Load Phosphenes sessions
    print("\nLoading Phosphenes sessions...")
    sessions = []
    for fname in PHOSPHENES_SESSION_FILES:
        fpath = os.path.join(PHOSPHENES_DATA_DIR, fname)
        sess = load_phosphenes_session(fpath)
        sessions.append(sess)
        print(f"  {fname}: {sess['n_tokens']} tokens, {sess['n_layers']} layers")

    # Run all three tests
    perm_results = run_permutation_test(sessions)
    layer0_results = run_layer0_control(sessions)
    cross_arch_results = run_cross_architecture(LAYERTIME_RUNS, LAYERTIME_EXTRA_RUNS)

    # Compile results
    all_results = {
        "test_1_permutation_null": perm_results,
        "test_2_layer0_control": layer0_results,
        "test_3_cross_architecture": cross_arch_results,
    }

    # Write to JSON
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults written to {OUTPUT_PATH}")

    # Final synthesis
    print("\n" + "=" * 70)
    print("SYNTHESIS")
    print("=" * 70)

    # Test 1
    print(f"\n  Test 1 (Permutation): p = {perm_results['p_one_sided']:.4f}")
    if perm_results['p_one_sided'] < 0.05:
        print("    -> The negative baseline is NOT an artifact of arbitrary direction choice")
    else:
        print("    -> The negative baseline COULD be an artifact of arbitrary direction choice")

    # Test 2
    l0_bl = layer0_results["layer_results"].get("0", {}).get("baseline", float("nan"))
    l44_bl = layer0_results["layer_results"].get("44", {}).get("baseline", float("nan"))
    print(f"\n  Test 2 (Layer 0): L0 baseline = {l0_bl:.4f}, L44 baseline = {l44_bl:.4f}")
    if not np.isnan(l0_bl) and not np.isnan(l44_bl):
        if abs(l0_bl) < abs(l44_bl) * 0.3:
            print("    -> Baseline EMERGES through transformer layers (not in embeddings)")
        elif abs(l0_bl) > abs(l44_bl) * 0.7:
            print("    -> Baseline is ALREADY PRESENT in embeddings")
        else:
            print("    -> Baseline PARTIALLY present in embeddings, AMPLIFIED through layers")

    # Test 3
    n_neg = sum(1 for label in cross_arch_results
                if "error" not in cross_arch_results[label]
                and cross_arch_results[label].get("concat_baseline", 0) < 0)
    n_total = sum(1 for label in cross_arch_results
                  if "error" not in cross_arch_results[label])
    print(f"\n  Test 3 (Cross-architecture): {n_neg}/{n_total} models show negative baseline")
    if n_neg == n_total and n_total > 0:
        print("    -> Negative baseline REPLICATES across all tested architectures")
    elif n_neg > n_total / 2:
        print("    -> Negative baseline replicates in MOST but not all architectures")
    else:
        print("    -> Negative baseline does NOT reliably replicate")

    print("\n" + "=" * 70)
    print("Investigation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
