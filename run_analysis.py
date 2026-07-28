"""
Phosphenes Affect Analysis
Analyzes 8 session JSON files for tonic affect baseline, turn-level affect structure,
and layer-wise affect emergence.
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
import sys

DATA_DIR = str(_ROOT / "web" / "data")
OUTPUT_PATH = str(_ROOT / "analysis_results.json")

SESSION_FILES = [
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

LAYERS_OF_INTEREST = [11, 22, 33, 44, 55]


def token_affect_label(token_piece):
    """Return 'positive', 'negative', or None."""
    t = token_piece.lower().strip().lstrip("▁Ġ ").strip(".,!?;:'\"")
    for w in POSITIVE_WORDS:
        if t == w or t.startswith(w):
            return "positive"
    for w in NEGATIVE_WORDS:
        if t == w or t.startswith(w):
            return "negative"
    return None


def load_session(filepath):
    """Load and dequantize a session JSON."""
    with open(filepath) as f:
        d = json.load(f)

    n_tokens = d["n_tokens"]
    n_layers = d["n_layers"]
    jl_dim = d["jl_dim"]

    jl_bytes = base64.b64decode(d["jl"])
    jl_uint8 = np.frombuffer(jl_bytes, dtype=np.uint8).reshape(n_tokens, n_layers, jl_dim)

    jl_min = np.array(d["jl_min"], dtype=np.float32)  # shape (16,)
    jl_max = np.array(d["jl_max"], dtype=np.float32)  # shape (16,)

    # Dequantize: shape (n_tokens, n_layers, 16)
    jl_float = jl_uint8.astype(np.float32) / 255.0 * (jl_max - jl_min) + jl_min

    return {
        "stem": d.get("stem", os.path.splitext(os.path.basename(filepath))[0]),
        "display_name": d.get("display_name", os.path.splitext(os.path.basename(filepath))[0]),
        "n_tokens": n_tokens,
        "n_layers": n_layers,
        "jl_dim": jl_dim,
        "jl_float": jl_float,  # (n_tokens, n_layers, 16)
        "token_pieces": d["token_pieces"],
        "token_roles": d.get("token_roles", []),
        "turns": d.get("turns", []),
    }


def compute_affect_labels(token_pieces):
    """Return array of 'positive', 'negative', or None for each token."""
    return [token_affect_label(tp) for tp in token_pieces]


def main():
    print("=" * 70)
    print("PHOSPHENES AFFECT ANALYSIS")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # Step 1: Load all sessions
    # ------------------------------------------------------------------ #
    print("\nLoading sessions...")
    sessions = []
    for fname in SESSION_FILES:
        fpath = os.path.join(DATA_DIR, fname)
        sess = load_session(fpath)
        sess["labels"] = compute_affect_labels(sess["token_pieces"])
        sessions.append(sess)
        print(f"  Loaded {fname}: {sess['n_tokens']} tokens, {sess['n_layers']} layers")

    # ------------------------------------------------------------------ #
    # Step 2: Build shared affect direction using layers of interest
    # Pool positive and negative vectors across all sessions at each layer
    # ------------------------------------------------------------------ #
    print("\nBuilding shared affect direction...")

    # We'll build a separate affect direction per layer, then also a
    # concatenated 80D direction for multi-layer analysis.
    layer_pos_vecs = {L: [] for L in LAYERS_OF_INTEREST}
    layer_neg_vecs = {L: [] for L in LAYERS_OF_INTEREST}

    for sess in sessions:
        jl = sess["jl_float"]  # (n_tokens, n_layers, 16)
        labels = sess["labels"]
        for i, lbl in enumerate(labels):
            for L in LAYERS_OF_INTEREST:
                if L < sess["n_layers"]:
                    v = jl[i, L, :]
                    if lbl == "positive":
                        layer_pos_vecs[L].append(v)
                    elif lbl == "negative":
                        layer_neg_vecs[L].append(v)

    # Compute affect direction per layer (unit vector)
    affect_directions = {}
    for L in LAYERS_OF_INTEREST:
        if len(layer_pos_vecs[L]) == 0 or len(layer_neg_vecs[L]) == 0:
            print(f"  WARNING: Layer {L} has no positive or negative tokens!")
            affect_directions[L] = None
            continue
        pos_mean = np.mean(layer_pos_vecs[L], axis=0)
        neg_mean = np.mean(layer_neg_vecs[L], axis=0)
        direction = pos_mean - neg_mean
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        affect_directions[L] = direction
        print(f"  Layer {L}: {len(layer_pos_vecs[L])} positive, {len(layer_neg_vecs[L])} negative tokens")

    # 80D concatenated direction for multi-layer analysis
    concat_pos_vecs = []
    concat_neg_vecs = []
    for sess in sessions:
        jl = sess["jl_float"]
        labels = sess["labels"]
        for i, lbl in enumerate(labels):
            # Concatenate vectors from all 5 layers
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
            v80 = np.concatenate(parts)  # 80D
            if lbl == "positive":
                concat_pos_vecs.append(v80)
            elif lbl == "negative":
                concat_neg_vecs.append(v80)

    concat_pos_mean = np.mean(concat_pos_vecs, axis=0) if concat_pos_vecs else None
    concat_neg_mean = np.mean(concat_neg_vecs, axis=0) if concat_neg_vecs else None
    if concat_pos_mean is not None and concat_neg_mean is not None:
        concat_direction = concat_pos_mean - concat_neg_mean
        cnorm = np.linalg.norm(concat_direction)
        if cnorm > 0:
            concat_direction = concat_direction / cnorm
    else:
        concat_direction = None

    # ------------------------------------------------------------------ #
    # Step 3: Project all tokens onto affect direction at each layer
    # ------------------------------------------------------------------ #
    print("\nProjecting tokens onto affect direction...")

    for sess in sessions:
        jl = sess["jl_float"]   # (n_tokens, n_layers, 16)
        n_tokens = sess["n_tokens"]

        # Per-layer projections dict
        layer_projections = {}
        for L in LAYERS_OF_INTEREST:
            if L >= sess["n_layers"] or affect_directions[L] is None:
                continue
            d_vec = affect_directions[L]  # (16,)
            # project each token's vector at this layer
            projs = jl[:, L, :] @ d_vec  # (n_tokens,)
            layer_projections[L] = projs

        sess["layer_projections"] = layer_projections  # {layer: (n_tokens,) array}

        # 80D projection using concat direction
        if concat_direction is not None:
            valid_layers = [L for L in LAYERS_OF_INTEREST if L < sess["n_layers"]]
            if len(valid_layers) == len(LAYERS_OF_INTEREST):
                vecs_80 = np.concatenate([jl[:, L, :] for L in LAYERS_OF_INTEREST], axis=1)  # (n_tokens, 80)
                proj_80 = vecs_80 @ concat_direction
                sess["proj_80"] = proj_80
            else:
                sess["proj_80"] = None
        else:
            sess["proj_80"] = None

    # ------------------------------------------------------------------ #
    # Step 4: Tonic Affect Baseline (per session, using 80D projection)
    # ------------------------------------------------------------------ #
    print("\nComputing tonic affect baselines...")
    per_session_baseline = {}
    session_display_names = {}

    for sess in sessions:
        stem = sess["stem"]
        display_name = sess["display_name"]
        session_display_names[stem] = display_name
        labels = sess["labels"]
        proj = sess.get("proj_80")
        if proj is None:
            # fallback: average across all available layers
            available = list(sess["layer_projections"].keys())
            if not available:
                per_session_baseline[stem] = float("nan")
                continue
            proj = np.mean(
                np.stack([sess["layer_projections"][L] for L in available], axis=1), axis=1
            )

        # Baseline = mean projection of tokens NOT in pos/neg sets
        baseline_mask = np.array([lbl is None for lbl in labels])
        if baseline_mask.sum() == 0:
            per_session_baseline[stem] = float(np.mean(proj))
        else:
            per_session_baseline[stem] = float(np.mean(proj[baseline_mask]))

    # ------------------------------------------------------------------ #
    # Step 5: Turn-Level Affect Structure (80D projection, per session)
    # ------------------------------------------------------------------ #
    print("\nComputing turn-level affect structure...")
    per_session_turn_affect = {}

    for sess in sessions:
        stem = sess["stem"]
        labels = sess["labels"]
        proj = sess.get("proj_80")
        if proj is None:
            available = list(sess["layer_projections"].keys())
            if not available:
                per_session_turn_affect[stem] = {}
                continue
            proj = np.mean(
                np.stack([sess["layer_projections"][L] for L in available], axis=1), axis=1
            )

        turns_data = sess["turns"]
        turn_affect = {}
        for turn_info in turns_data:
            turn_num = turn_info["turn"]
            if turn_num > 6:
                continue
            ts = turn_info["token_start"]
            te = turn_info["token_end"]
            if ts >= te:
                continue
            turn_proj = proj[ts:te]
            turn_affect[str(turn_num)] = float(np.mean(turn_proj))

        per_session_turn_affect[stem] = turn_affect

    # ------------------------------------------------------------------ #
    # Step 6: Layer-Wise Cohen's d (positive vs negative at each layer)
    # ------------------------------------------------------------------ #
    print("\nComputing layer-wise Cohen's d...")
    layer_cohens_d = {}

    for L in LAYERS_OF_INTEREST:
        if affect_directions[L] is None:
            layer_cohens_d[str(L)] = float("nan")
            continue

        pos_projs = []
        neg_projs = []
        for sess in sessions:
            if L not in sess["layer_projections"]:
                continue
            projs = sess["layer_projections"][L]
            labels = sess["labels"]
            for i, lbl in enumerate(labels):
                if lbl == "positive":
                    pos_projs.append(projs[i])
                elif lbl == "negative":
                    neg_projs.append(projs[i])

        if not pos_projs or not neg_projs:
            layer_cohens_d[str(L)] = float("nan")
            continue

        pos_arr = np.array(pos_projs)
        neg_arr = np.array(neg_projs)
        mean_diff = np.mean(pos_arr) - np.mean(neg_arr)
        pooled_std = np.sqrt((np.var(pos_arr, ddof=1) + np.var(neg_arr, ddof=1)) / 2)
        if pooled_std > 0:
            d_val = mean_diff / pooled_std
        else:
            d_val = 0.0
        layer_cohens_d[str(L)] = float(d_val)

    # ------------------------------------------------------------------ #
    # Step 7: Per-session affect by turn AND layer
    # ------------------------------------------------------------------ #
    print("\nComputing per-session affect by turn and layer...")
    per_session_affect_by_turn_and_layer = {}

    for sess in sessions:
        stem = sess["stem"]
        turns_data = sess["turns"]
        by_layer = {}
        for L in LAYERS_OF_INTEREST:
            if L not in sess["layer_projections"]:
                continue
            projs = sess["layer_projections"][L]
            by_turn = {}
            for turn_info in turns_data:
                turn_num = turn_info["turn"]
                if turn_num > 6:
                    continue
                ts = turn_info["token_start"]
                te = turn_info["token_end"]
                if ts >= te:
                    continue
                by_turn[str(turn_num)] = float(np.mean(projs[ts:te]))
            by_layer[str(L)] = by_turn
        per_session_affect_by_turn_and_layer[stem] = by_layer

    # ------------------------------------------------------------------ #
    # Step 8: Write results to JSON
    # ------------------------------------------------------------------ #
    print("\nWriting results to JSON...")
    results = {
        "per_session_baseline": per_session_baseline,
        "per_session_turn_affect": per_session_turn_affect,
        "layer_cohens_d": layer_cohens_d,
        "per_session_affect_by_turn_and_layer": per_session_affect_by_turn_and_layer,
        "session_display_names": session_display_names,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {OUTPUT_PATH}")

    # ------------------------------------------------------------------ #
    # Step 9: Print formatted summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS SUMMARY")
    print("=" * 70)

    print("\n--- 1. TONIC AFFECT BASELINE (per session) ---")
    print(f"{'Session':<40} {'Baseline':>12}")
    print("-" * 55)
    baselines = []
    for stem, val in per_session_baseline.items():
        display = session_display_names.get(stem, stem)
        print(f"{display:<40} {val:>12.6f}")
        if not np.isnan(val):
            baselines.append(val)
    if baselines:
        print(f"\n  Grand mean baseline: {np.mean(baselines):.6f}")
        print(f"  All negative?       {all(b < 0 for b in baselines)}")

    print("\n--- 2. TURN-LEVEL AFFECT STRUCTURE (pooled across sessions) ---")
    turn_pool = {}
    for stem, turn_dict in per_session_turn_affect.items():
        for tn, val in turn_dict.items():
            turn_pool.setdefault(tn, []).append(val)

    print(f"{'Turn':<10} {'Role':<25} {'Mean Affect':>12} {'N Sessions':>12}")
    print("-" * 62)
    turn_roles = {
        "1": "User (setup)",
        "2": "Assistant (response)",
        "3": "User (deepening)",
        "4": "Assistant (creative dream)",
        "5": "User (reflection prompt)",
        "6": "Assistant (reflection)",
    }
    for tn in sorted(turn_pool.keys(), key=int):
        vals = turn_pool[tn]
        role = turn_roles.get(tn, "unknown")
        print(f"Turn {tn:<5} {role:<25} {np.mean(vals):>12.6f} {len(vals):>12}")

    print("\n--- 3. LAYER-WISE AFFECT EMERGENCE (Cohen's d) ---")
    cohens_header = "Cohen's d"
    print(f"{'Layer':<10} {cohens_header:>12} {'Interpretation':<25}")
    print("-" * 50)
    for L in LAYERS_OF_INTEREST:
        d_val = layer_cohens_d.get(str(L), float("nan"))
        if np.isnan(d_val):
            interp = "N/A"
        elif abs(d_val) < 0.2:
            interp = "negligible"
        elif abs(d_val) < 0.5:
            interp = "small"
        elif abs(d_val) < 0.8:
            interp = "medium"
        else:
            interp = "large"
        print(f"Layer {L:<5} {d_val:>12.4f} {interp:<25}")

    print("\n--- 4. TURN 4 vs TURN 6 AFFECT (creative dream vs reflection) ---")
    t4_vals = turn_pool.get("4", [])
    t6_vals = turn_pool.get("6", [])
    if t4_vals and t6_vals:
        t4_mean = np.mean(t4_vals)
        t6_mean = np.mean(t6_vals)
        print(f"  Turn 4 (creative dream) mean affect: {t4_mean:.6f}")
        print(f"  Turn 6 (reflection)     mean affect: {t6_mean:.6f}")
        print(f"  Drop from Turn 4 to 6:               {t6_mean - t4_mean:.6f}")
        print(f"  Turn 4 > Turn 6?                     {t4_mean > t6_mean}")

    print("\n--- 5. PER-SESSION AFFECT BY TURN (80D projection) ---")
    for stem, turn_dict in per_session_turn_affect.items():
        display = session_display_names.get(stem, stem)
        print(f"\n  {display}:")
        for tn in sorted(turn_dict.keys(), key=int):
            val = turn_dict[tn]
            print(f"    Turn {tn}: {val:>10.6f}")

    print("\n--- 6. LAYER-WISE TURN 4 AFFECT (positive divergence point) ---")
    print(f"{'Layer':<10}", end="")
    for stem in per_session_affect_by_turn_and_layer:
        display = session_display_names.get(stem, stem)
        print(f" {display[:12]:>14}", end="")
    print()
    print("-" * (10 + 14 * len(per_session_affect_by_turn_and_layer)))

    for L in LAYERS_OF_INTEREST:
        print(f"Layer {L:<5}", end="")
        for stem, by_layer in per_session_affect_by_turn_and_layer.items():
            val = by_layer.get(str(L), {}).get("4", float("nan"))
            if np.isnan(val):
                print(f" {'N/A':>14}", end="")
            else:
                print(f" {val:>14.6f}", end="")
        print()

    print("\n" + "=" * 70)
    print("Analysis complete.")
    print(f"Full results at: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
