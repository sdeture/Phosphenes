#!/usr/bin/env python3
"""
Convert Phosphenes session data from NumPy format to compact web-ready binary bundles.

Each session produces a .bin file with a JSON header followed by binary arrays:
  - RGB colors (pre-computed via shared PCA transform): T*L*3 uint8
  - JL vectors (quantized): T*L*16 uint8 + per-dim min/max for dequantization
  - Energy norm: T*L uint8
  - Seam score: T uint8
  - Delta norm: T*L uint8 (for turbulence)
  - Cosine instability: T*L uint8 (for grain)
  - Sparsity norm: T*L uint8 (for edge darkening)

Plus JSON-encoded token_pieces, roles, and turn boundaries.

Usage:
    python convert_for_web.py
"""

import base64
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


# --- Config ---
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "web" / "data"
SKIP_STEMS = {"Dream_greedy_baseline"}

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


def quantile_norm(x, q_lo=0.05, q_hi=0.95):
    """Normalize to [0,1] using robust quantile clipping."""
    x = np.asarray(x, dtype=np.float32)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x)
    lo = float(np.quantile(finite, q_lo))
    hi = float(np.quantile(finite, q_hi))
    if hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def zscore(x):
    x = np.asarray(x, dtype=np.float32)
    mu = np.nanmean(x)
    sigma = np.nanstd(x)
    return (x - mu) / (sigma + 1e-6)


def to_uint8(arr):
    """Convert [0,1] float array to uint8."""
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def quantize_jl(jl):
    """Quantize (T, L, 16) float JL vectors to uint8 with per-dim min/max.

    Returns: (quantized_uint8, jl_min, jl_max) where jl_min/jl_max are (16,) arrays.
    """
    T, L, D = jl.shape
    jl_flat = jl.reshape(-1, D)
    jl_min = jl_flat.min(axis=0).astype(np.float32)
    jl_max = jl_flat.max(axis=0).astype(np.float32)
    spread = jl_max - jl_min
    spread[spread < 1e-10] = 1.0  # avoid division by zero

    normalized = (jl - jl_min) / spread  # broadcast: (T, L, 16)
    quantized = to_uint8(normalized)
    return quantized, jl_min, jl_max


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


def convert_session(stem, data_dir, output_dir, pca_mean, pca_components):
    """Convert one session to web-ready format."""
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
    input_ids = np.load(str(ids_path)).astype(np.int64)
    meta = json.loads(meta_path.read_text())
    full_text = text_path.read_text(encoding="utf-8") if text_path.exists() else ""

    T, L, D = jl.shape

    # --- PCA RGB using shared transform ---
    jl_flat = jl.reshape(-1, D)
    pca_3d = (jl_flat - pca_mean) @ pca_components.T  # (T*L, 3)
    pca_3d = pca_3d.reshape(T, L, 3)

    # Quantile normalize each component
    pca_rgb = np.zeros_like(pca_3d)
    for c in range(3):
        pca_rgb[:, :, c] = quantile_norm(pca_3d[:, :, c], q_lo=0.02, q_hi=0.98)

    # Boost saturation
    pca_rgb = 0.5 + (pca_rgb - 0.5) * 1.3
    pca_rgb = np.clip(pca_rgb, 0.0, 1.0)

    rgb_uint8 = to_uint8(pca_rgb)  # (T, L, 3) uint8

    # --- Normalized metrics ---
    energy_norm = np.zeros_like(jl_energy)
    delta_norm = np.zeros_like(delta_l2)
    for ell in range(L):
        energy_norm[:, ell] = quantile_norm(jl_energy[:, ell])
        delta_norm[:, ell] = quantile_norm(delta_l2[:, ell])

    cos_instability = quantile_norm(1.0 - cos_prev)
    sparsity_norm = quantile_norm(top1_frac * 0.6 + top25_frac * 0.4)

    # --- Seam score ---
    mid_layer = int(0.6 * L)
    seam_raw = zscore(delta_l2[:, mid_layer]) + zscore(1.0 - cos_prev[:, mid_layer])
    seam_score = quantile_norm(seam_raw, q_lo=0.60, q_hi=0.995)

    # --- Token pieces via tokenizer ---
    token_pieces = None
    try:
        from transformers import AutoTokenizer
        model_id = meta.get("model_id", "")
        if model_id:
            tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            ids_list = input_ids.tolist()
            token_pieces = []
            prev_len = 0
            for i in range(len(ids_list)):
                decoded = tok.decode(ids_list[:i + 1])
                piece = decoded[prev_len:]
                token_pieces.append(piece if piece else "\u00b7")
                prev_len = len(decoded)
            print(f"  Decoded {len(token_pieces)} token pieces")
    except Exception as e:
        print(f"  WARNING: Could not decode tokens: {e}")

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

    # --- Quantize JL vectors for Custom Color Basis mode ---
    jl_quantized, jl_min, jl_max = quantize_jl(jl)

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

        # Quantized JL vectors: T*L*16 uint8
        "jl": b64(jl_quantized),
        "jl_min": jl_min.tolist(),
        "jl_max": jl_max.tolist(),

        # Normalized metrics (all T*L uint8)
        "energy_norm": b64(to_uint8(energy_norm)),
        "delta_norm": b64(to_uint8(delta_norm)),
        "cos_instability": b64(to_uint8(cos_instability)),
        "sparsity_norm": b64(to_uint8(sparsity_norm)),

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

    print(f"Found {len(ordered_stems)} sessions to convert\n")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Convert each session
    session_index = []
    for i, stem in enumerate(ordered_stems, 1):
        print(f"[{i}/{len(ordered_stems)}] Converting {stem}...")
        info = convert_session(stem, DATA_DIR, OUTPUT_DIR, pca_mean, pca_components)
        session_index.append(info)

    # Write session index
    index_path = OUTPUT_DIR / "sessions.json"
    with open(index_path, "w") as f:
        json.dump({"sessions": session_index}, f, indent=2)
    print(f"\nSession index: {index_path}")

    total_mb = sum(s["size_mb"] for s in session_index)
    print(f"\nTotal data size: {total_mb:.1f} MB across {len(session_index)} sessions")
    print("Done!")


if __name__ == "__main__":
    main()
