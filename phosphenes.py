#!/usr/bin/env python3
"""
Phosphenes  --  Interactive real-time visualization of LLM activation data.

A phenomenological instrument for experiencing and exploring the internal
states of language models during leisure / dreaming activities.

Part of the LayerTime EEG project (DeTure & DeTure, 2026).

Usage:
    python phosphenes.py                           # Default: ERNIE_Thinking_run1
    python phosphenes.py --stem Thinking_run2      # Specific model run

Controls:
    Space       Play / Pause
    Left/Right  Step back/forward one token (when paused)
    Up/Down     Speed up / slow down playback
    Tab         Toggle text panel (right margin)
    I           Toggle inspector (hover for exact values)
    M           Cycle metric highlight: off -> energy -> sparsity
    C           Color basis mode
    P           Reference point mode
    T           Toggle turn markers
    1-9         Switch model run
    R           Toggle frame recording (PNGs for ffmpeg)
    F           Toggle fullscreen
    Q / Esc     Quit
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame
import pygame.freetype
from scipy.ndimage import gaussian_filter


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — Constants & Configuration
# ═══════════════════════════════════════════════════════════════════════════

WINDOW_W = 1400
WINDOW_H = 900
TARGET_FPS = 60

# Cell dimensions (pixels per token-layer cell before upscale)
CELL_PX = 8
CELL_PY = 14

# Number of token columns visible in the scrolling window
TOKENS_VISIBLE = 160

# Colors
BG = (11, 11, 18)
BG_F = np.array(BG, dtype=np.float32) / 255.0
TEXT_COL = (230, 230, 240)
DIM_COL = (120, 120, 140)
AMBER = (255, 200, 100)
SOFT_AMBER = (200, 160, 80)

# Font search order (macOS system fonts)
FONT_SEARCH = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Courier.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
]

# Model registry: key -> stem (legacy fallback)
MODEL_REGISTRY = {
    1: "Base_run1",
    2: "Instruct_run1",
    3: "Thinking_run2",
    4: "Thinking_poetry_run1",
    5: "Qwen14B_Base_run1",
    6: "Qwen14B_Reasoning_run1",
    7: "ERNIE_Base_run1",
    8: "ERNIE_PT_run1",
    9: "ERNIE_Thinking_run1",
}

# Display names for Dream sessions
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

# Sessions to skip in auto-discovery
SKIP_STEMS = {"Dream_greedy_baseline"}


@dataclass
class EffectParams:
    """Tunable visual-effect parameters."""
    energy_floor: float = 0.15
    energy_ceil: float = 1.0
    smooth_sigma: float = 1.2           # Gaussian blur for organic softness
    turbulence_amp: float = 0.16        # Max turbulence color offset
    turbulence_speed: float = 2.5       # Phase advance per second
    grain_max: float = 0.12             # Max grain amplitude (low cos_prev)
    edge_dark: float = 0.12             # Edge darkening for high sparsity
    seam_glow_intensity: float = 0.55   # Warm glow strength
    seam_glow_sigma: float = 5.0        # Glow blur radius
    heartbeat_alpha: float = 0.18       # Phase-band overlay opacity
    tokens_per_second: float = 24.0     # Default playback speed


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — Data Loading & Preprocessing
# ═══════════════════════════════════════════════════════════════════════════

def quantile_norm(x: np.ndarray, q_lo: float = 0.05, q_hi: float = 0.95) -> np.ndarray:
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


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = np.nanmean(x)
    sigma = np.nanstd(x)
    return (x - mu) / (sigma + 1e-6)


def unit_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norms + eps)


def _safe_tokenizer(model_id: str):
    """Try to load tokenizer (downloads if not cached)."""
    try:
        from transformers import AutoTokenizer
        try:
            return AutoTokenizer.from_pretrained(model_id, use_fast=True)
        except Exception:
            return AutoTokenizer.from_pretrained(model_id, use_fast=False)
    except Exception:
        return None


@dataclass(frozen=True)
class TurnBoundary:
    """Token range for one conversation turn segment."""
    turn: int
    role: str            # "user" | "assistant" | "system"
    token_start: int
    token_end: int


@dataclass
class ModelData:
    """All preprocessed data for a single model run."""
    stem: str
    display_name: str
    model_id: str
    architecture: str            # "Qwen30B", "Qwen14B", "ERNIE"

    n_tokens: int
    n_layers: int

    # Raw metric arrays — all (T, L)
    jl_energy: np.ndarray
    delta_l2: np.ndarray
    cos_prev: np.ndarray
    top1_frac: np.ndarray
    top25_frac: np.ndarray
    input_ids: np.ndarray        # (T,)

    # JL vectors for reference point computation — (T, L, 16)
    jl: np.ndarray

    # JL-based deltas (comparable scales)
    token_delta_jl: np.ndarray   # (T, L) distance from prior token (same layer)
    layer_delta_jl: np.ndarray   # (T, L) distance from prior layer (same token)
    token_delta_top1_jl: np.ndarray  # (T, L) fraction of token delta in top JL dim
    layer_delta_top1_jl: np.ndarray  # (T, L) fraction of layer delta in top JL dim

    # Normalized versions — all (T, L) in [0,1]
    energy_norm: np.ndarray
    delta_norm: np.ndarray
    cos_instability: np.ndarray  # inverted: high = unstable
    sparsity_norm: np.ndarray
    token_delta_jl_norm: np.ndarray
    layer_delta_jl_norm: np.ndarray

    # PCA color field — (T, L, 3) in [0,1]
    pca_rgb: np.ndarray

    # Cluster labels — (T, L) int
    cluster_labels: np.ndarray
    n_clusters: int

    # Derived scalars — (T,)
    seam_score: np.ndarray
    heterogeneity: np.ndarray

    # Heartbeat
    heartbeat_phase: Optional[np.ndarray] = None  # (L,) mod-6
    has_heartbeat: bool = False

    # Text / tokenizer
    full_text: str = ""
    tokenizer: object = None
    token_pieces: Optional[list] = None
    token_roles: Optional[list] = None  # (T,) list of strings: "user", "assistant", "system", ""

    # Pre-generated noise textures for turbulence (list of (L, TOKENS_VISIBLE) arrays)
    noise_textures: list = field(default_factory=list)
    grain_textures: list = field(default_factory=list)
    # Pre-built heartbeat tint (L, 3) or None
    heartbeat_tint_cells: Optional[np.ndarray] = None
    # Pre-built edge masks
    edge_y_cell: Optional[np.ndarray] = None
    edge_x_cell: Optional[np.ndarray] = None

    # Turn boundaries (therapy sessions)
    turn_boundaries: list = field(default_factory=list)
    has_turns: bool = False

    # Self-reference analysis
    self_ref_tokens: np.ndarray = None  # (T,) bool mask of self-referential tokens
    self_ref_centroid: np.ndarray = None  # (L, 16) average JL vector at self-ref tokens
    self_ref_distances: np.ndarray = None  # (T, L) distance from centroid


@dataclass
class ColorBasisGroup:
    """One color axis defined by source cells and optional contrast cells."""
    source_cells: list[tuple[int, int]] = field(default_factory=list)      # [(token, layer), ...]
    contrast_cells: list[tuple[int, int]] = field(default_factory=list)    # [(token, layer), ...]
    undo_stack: list[tuple[str, tuple[int, int]]] = field(default_factory=list)  # [("source"|"contrast", cell), ...]

    def add_source(self, t: int, l: int):
        self.source_cells.append((t, l))
        self.undo_stack.append(("source", (t, l)))

    def add_contrast(self, t: int, l: int):
        self.contrast_cells.append((t, l))
        self.undo_stack.append(("contrast", (t, l)))

    def undo_last(self) -> bool:
        """Remove last added cell. Returns False if nothing to undo."""
        if not self.undo_stack:
            return False
        kind, cell = self.undo_stack.pop()
        if kind == "source":
            self.source_cells.remove(cell)
        else:
            self.contrast_cells.remove(cell)
        return True

    def is_valid(self) -> bool:
        """Must have at least one source cell."""
        return len(self.source_cells) > 0

    def compute_vector(self, jl: np.ndarray) -> np.ndarray:
        """Compute the 16-D direction vector for this axis.

        If contrast cells exist: direction = mean(source) - mean(contrast)
        Otherwise: direction = mean(source) (centroid)

        Args:
            jl: Full JL array of shape (T, L, 16)
        Returns:
            16-D direction vector (NOT normalized)
        """
        src_vecs = np.array([jl[t, l] for t, l in self.source_cells])
        src_mean = src_vecs.mean(axis=0)
        if self.contrast_cells:
            con_vecs = np.array([jl[t, l] for t, l in self.contrast_cells])
            con_mean = con_vecs.mean(axis=0)
            return src_mean - con_mean
        return src_mean


@dataclass
class ColorBasisResult:
    """Finalized color basis: orthonormal axes + pre-normalized projections."""
    e1: np.ndarray  # (16,) orthonormal basis vector for R
    e2: np.ndarray  # (16,) orthonormal basis vector for G
    e3: np.ndarray  # (16,) orthonormal basis vector for B
    proj_r: np.ndarray  # (T, L) globally normalized to [0, 1]
    proj_g: np.ndarray  # (T, L) globally normalized to [0, 1]
    proj_b: np.ndarray  # (T, L) globally normalized to [0, 1]


def load_model_data(base_dir: Path, stem: str) -> ModelData:
    """Load and preprocess everything for one model run."""
    t0 = time.time()

    act_path = base_dir / f"{stem}_activations.npz"
    ids_path = base_dir / f"{stem}_input_ids.npy"
    meta_path = base_dir / f"{stem}_metadata.json"
    text_path = base_dir / f"{stem}_text.txt"

    for p in [act_path, ids_path, meta_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    meta = json.loads(meta_path.read_text())
    model_id = meta.get("model_id", "")

    act = np.load(str(act_path))
    jl = act["jl"].astype(np.float32)          # (T, L, 16)
    jl_energy = act["jl_energy"].astype(np.float32)
    delta_l2 = act["delta_l2"].astype(np.float32)
    cos_prev = act["cos_prev"].astype(np.float32)
    top1_frac = act["top1_frac"].astype(np.float32)
    top25_frac = act["top25_frac"].astype(np.float32)

    input_ids = np.load(str(ids_path)).astype(np.int64)
    T, L, D = jl.shape

    # Determine architecture
    if L >= 48:
        architecture = "Qwen30B"
    elif L >= 40:
        architecture = "Qwen14B"
    else:
        architecture = "ERNIE"

    # --- Per-layer quantile normalization ---
    energy_norm = np.zeros_like(jl_energy)
    delta_norm = np.zeros_like(delta_l2)
    for ell in range(L):
        energy_norm[:, ell] = quantile_norm(jl_energy[:, ell])
        delta_norm[:, ell] = quantile_norm(delta_l2[:, ell])

    cos_instability = quantile_norm(1.0 - cos_prev)
    sparsity_norm = quantile_norm(top1_frac * 0.6 + top25_frac * 0.4)

    # --- JL-based deltas (comparable scales) ---
    # Token delta: distance from prior token (same layer)
    token_delta_jl = np.zeros((T, L), dtype=np.float32)
    token_delta_jl[1:, :] = np.linalg.norm(jl[1:] - jl[:-1], axis=-1)

    # Layer delta: distance from prior layer (same token)
    layer_delta_jl = np.zeros((T, L), dtype=np.float32)
    layer_delta_jl[:, 1:] = np.linalg.norm(jl[:, 1:] - jl[:, :-1], axis=-1)

    # JL sparsity: fraction of delta in top-1 JL dimension
    # Token delta sparsity
    token_delta_top1_jl = np.zeros((T, L), dtype=np.float32)
    token_diff = np.zeros((T, L, D), dtype=np.float32)
    token_diff[1:, :, :] = jl[1:] - jl[:-1]
    token_sq = token_diff ** 2
    token_total = token_sq.sum(axis=-1, keepdims=True) + 1e-10
    token_delta_top1_jl = (token_sq.max(axis=-1) / token_total.squeeze())

    # Layer delta sparsity
    layer_delta_top1_jl = np.zeros((T, L), dtype=np.float32)
    layer_diff = np.zeros((T, L, D), dtype=np.float32)
    layer_diff[:, 1:, :] = jl[:, 1:] - jl[:, :-1]
    layer_sq = layer_diff ** 2
    layer_total = layer_sq.sum(axis=-1, keepdims=True) + 1e-10
    layer_delta_top1_jl = (layer_sq.max(axis=-1) / layer_total.squeeze())

    # Normalize deltas
    token_delta_jl_norm = quantile_norm(token_delta_jl)
    layer_delta_jl_norm = quantile_norm(layer_delta_jl)

    # --- PCA on JL vectors → (T, L, 3) RGB ---
    from sklearn.decomposition import PCA

    jl_flat = jl.reshape(-1, D)  # (T*L, 16)
    rng = np.random.default_rng(42)
    sample_n = min(60000, jl_flat.shape[0])
    sample_idx = rng.choice(jl_flat.shape[0], size=sample_n, replace=False)

    pca = PCA(n_components=3, random_state=42)
    pca.fit(jl_flat[sample_idx])
    pca_3d = pca.transform(jl_flat).reshape(T, L, 3)

    # Normalize each PCA component to [0,1] for RGB
    pca_rgb = np.zeros_like(pca_3d)
    for c in range(3):
        pca_rgb[:, :, c] = quantile_norm(pca_3d[:, :, c], q_lo=0.02, q_hi=0.98)

    # Boost saturation: push away from gray center
    center = 0.5
    pca_rgb = center + (pca_rgb - center) * 1.3
    pca_rgb = np.clip(pca_rgb, 0.0, 1.0)

    # --- KMeans clustering (for inspector labels) ---
    from sklearn.cluster import KMeans

    n_clusters = 8
    jl_flat_n = unit_normalize_rows(jl_flat)
    fit_n = min(50000, jl_flat_n.shape[0])
    fit_idx = rng.choice(jl_flat_n.shape[0], size=fit_n, replace=False)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto", max_iter=300)
    km.fit(jl_flat_n[fit_idx])
    cluster_labels = km.predict(jl_flat_n).reshape(T, L).astype(np.int32)

    # --- Seam score ---
    mid_layer = int(0.6 * L)
    seam_raw = zscore(delta_l2[:, mid_layer]) + zscore(1.0 - cos_prev[:, mid_layer])
    seam_score = quantile_norm(seam_raw, q_lo=0.60, q_hi=0.995)

    # --- Heterogeneity (cross-layer entropy) ---
    heterogeneity = np.zeros(T, dtype=np.float32)
    for t in range(T):
        cnt = np.bincount(cluster_labels[t], minlength=n_clusters).astype(np.float32)
        p = cnt / max(1.0, cnt.sum())
        p = p[p > 0]
        heterogeneity[t] = float(-np.sum(p * np.log(p + 1e-12))) / math.log(max(2, n_clusters))

    # --- Heartbeat phase ---
    heartbeat_phase = None
    has_heartbeat = False
    if architecture == "Qwen30B":
        heartbeat_phase = np.arange(L, dtype=np.int32) % 6
        has_heartbeat = True

    # --- Text / tokenizer ---
    full_text = text_path.read_text(encoding="utf-8") if text_path.exists() else ""
    tokenizer = _safe_tokenizer(model_id)
    token_pieces = None
    if tokenizer is not None:
        try:
            # Incremental decode: properly reassembles multi-byte UTF-8 sequences
            # (convert_ids_to_tokens returns raw BPE pieces with byte-level artifacts)
            ids_list = input_ids.tolist()
            token_pieces = []
            prev_len = 0
            for i in range(len(ids_list)):
                decoded = tokenizer.decode(ids_list[:i + 1])
                piece = decoded[prev_len:]
                token_pieces.append(piece if piece else "·")  # fallback for empty pieces
                prev_len = len(decoded)
        except Exception:
            token_pieces = None

    # --- Pre-generate noise textures for turbulence (at cell resolution) ---
    noise_rng = np.random.default_rng(7)
    noise_textures = []
    for _ in range(4):
        raw = noise_rng.standard_normal((L, TOKENS_VISIBLE)).astype(np.float32)
        smoothed = gaussian_filter(raw, sigma=1.8)
        smoothed = (smoothed - smoothed.mean()) / (smoothed.std() + 1e-6)
        noise_textures.append(smoothed)

    # --- Pre-generate grain textures (at cell resolution) ---
    grain_rng = np.random.default_rng(13)
    grain_textures = []
    for _ in range(8):
        raw = grain_rng.standard_normal((L, TOKENS_VISIBLE)).astype(np.float32)
        grain_textures.append(raw)

    # --- Pre-build heartbeat overlay at cell resolution (L, 3) ---
    heartbeat_tint_cells = None
    if has_heartbeat and heartbeat_phase is not None:
        hb_tints = make_heartbeat_tints(6)
        heartbeat_tint_cells = hb_tints[heartbeat_phase]  # (L, 3)

    # --- Pre-build edge masks at cell resolution ---
    edge_y_cell = np.zeros(CELL_PY, dtype=np.float32)
    edge_y_cell[0] = 1.0
    if CELL_PY > 1:
        edge_y_cell[-1] = 0.5
    edge_x_cell = np.zeros(CELL_PX, dtype=np.float32)
    edge_x_cell[0] = 1.0

    # --- Parse turn boundaries (therapy sessions) ---
    turn_boundaries = []
    has_turns = False
    if "turn_boundaries" in meta:
        has_turns = True
        for tb in meta["turn_boundaries"]:
            turn_boundaries.append(TurnBoundary(
                turn=tb["turn"], role=tb["role"],
                token_start=tb["token_start"], token_end=tb["token_end"],
            ))

    # Fallback: auto-detect turn boundaries from special tokens in input_ids
    if not has_turns:
        IM_START_ID = 151644
        IM_END_ID = 151645
        im_start_positions = [i for i, tid in enumerate(input_ids) if tid == IM_START_ID]

        if im_start_positions and token_pieces is not None:
            has_turns = True
            turn_num = 0
            for start_pos in im_start_positions:
                # Role is the token after <|im_start|>
                role = "unknown"
                if start_pos + 1 < len(token_pieces):
                    role_piece = token_pieces[start_pos + 1].lower().strip()
                    if "user" in role_piece:
                        role = "user"
                    elif "assistant" in role_piece:
                        role = "assistant"
                    elif "system" in role_piece:
                        role = "system"

                # Find matching <|im_end|>
                end_pos = T - 1  # default to end
                for j in range(start_pos + 1, T):
                    if input_ids[j] == IM_END_ID:
                        end_pos = j
                        break

                turn_num += 1
                turn_boundaries.append(TurnBoundary(
                    turn=turn_num, role=role,
                    token_start=start_pos, token_end=end_pos + 1,
                ))

            print(f"  Auto-detected {len(turn_boundaries)} turns from special tokens")

    # Build per-token role array
    token_roles = [""] * T
    for tb in turn_boundaries:
        for t_idx in range(tb.token_start, min(tb.token_end, T)):
            token_roles[t_idx] = tb.role

    # --- Self-reference detection ---
    # Look for tokens that are "I", "my", "me", "myself" (case-insensitive, with common tokenizer prefixes)
    self_ref_patterns = {'i', 'my', 'me', 'myself', 'I', 'My', 'Me', 'Myself',
                         '▁i', '▁my', '▁me', '▁myself', '▁I', '▁My', '▁Me', '▁Myself',
                         'Ġi', 'Ġmy', 'Ġme', 'Ġmyself', 'ĠI', 'ĠMy', 'ĠMe', 'ĠMyself',
                         ' i', ' my', ' me', ' myself', ' I', ' My', ' Me', ' Myself'}
    self_ref_tokens = np.zeros(T, dtype=bool)
    if token_pieces is not None:
        for t_idx, piece in enumerate(token_pieces):
            if piece in self_ref_patterns:
                self_ref_tokens[t_idx] = True

    # Compute self-reference centroid (average JL vector at self-ref tokens, per layer)
    self_ref_centroid = None
    self_ref_distances = None
    n_self_ref = self_ref_tokens.sum()
    if n_self_ref > 0:
        # Average across self-ref tokens for each layer: (L, 16)
        self_ref_centroid = jl[self_ref_tokens].mean(axis=0)  # (L, 16)
        # Distance from centroid for all tokens
        # For each (t, l), compute distance from self_ref_centroid[l]
        self_ref_distances = np.zeros((T, L), dtype=np.float32)
        for l in range(L):
            self_ref_distances[:, l] = np.linalg.norm(jl[:, l, :] - self_ref_centroid[l], axis=-1)

    elapsed = time.time() - t0
    turns_str = f", {len(turn_boundaries)} turn segments" if has_turns else ""
    self_ref_str = f", {n_self_ref} self-ref tokens" if n_self_ref > 0 else ""
    print(f"[phosphenes] Loaded {stem}: {T} tokens x {L} layers ({architecture}){turns_str}{self_ref_str} in {elapsed:.1f}s")

    display_name = DISPLAY_NAMES.get(stem, stem)

    return ModelData(
        stem=stem,
        display_name=display_name,
        model_id=model_id,
        architecture=architecture,
        n_tokens=T,
        n_layers=L,
        jl_energy=jl_energy,
        delta_l2=delta_l2,
        cos_prev=cos_prev,
        top1_frac=top1_frac,
        top25_frac=top25_frac,
        input_ids=input_ids,
        jl=jl,
        token_delta_jl=token_delta_jl,
        layer_delta_jl=layer_delta_jl,
        token_delta_top1_jl=token_delta_top1_jl,
        layer_delta_top1_jl=layer_delta_top1_jl,
        energy_norm=energy_norm,
        delta_norm=delta_norm,
        cos_instability=cos_instability,
        sparsity_norm=sparsity_norm,
        token_delta_jl_norm=token_delta_jl_norm,
        layer_delta_jl_norm=layer_delta_jl_norm,
        pca_rgb=pca_rgb,
        cluster_labels=cluster_labels,
        n_clusters=n_clusters,
        seam_score=seam_score,
        heterogeneity=heterogeneity,
        heartbeat_phase=heartbeat_phase,
        has_heartbeat=has_heartbeat,
        full_text=full_text,
        tokenizer=tokenizer,
        token_pieces=token_pieces,
        token_roles=token_roles,
        noise_textures=noise_textures,
        grain_textures=grain_textures,
        heartbeat_tint_cells=heartbeat_tint_cells,
        edge_y_cell=edge_y_cell,
        edge_x_cell=edge_x_cell,
        turn_boundaries=turn_boundaries,
        has_turns=has_turns,
        self_ref_tokens=self_ref_tokens,
        self_ref_centroid=self_ref_centroid,
        self_ref_distances=self_ref_distances,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — Color System
# ═══════════════════════════════════════════════════════════════════════════

def make_heartbeat_tints(n_phases: int = 6) -> np.ndarray:
    """
    Subtle warm-to-cool tint for each heartbeat phase.
    Returns (n_phases, 3) float32 in [0,1].
    """
    # Phase 0-1: warm (reddish), phase 2-3: neutral, phase 4-5: cool (bluish)
    tints = np.array([
        [0.20, 0.08, 0.02],   # warm amber
        [0.15, 0.10, 0.03],   # warm gold
        [0.08, 0.12, 0.06],   # neutral green
        [0.05, 0.10, 0.10],   # neutral teal
        [0.03, 0.07, 0.18],   # cool blue
        [0.06, 0.04, 0.20],   # cool indigo
    ], dtype=np.float32)
    # Tile or trim to requested size
    if n_phases <= tints.shape[0]:
        return tints[:n_phases]
    return np.tile(tints, (n_phases // 6 + 1, 1))[:n_phases]


def get_current_turn(data: ModelData, token_idx: int) -> Optional[TurnBoundary]:
    """Find which turn boundary contains the given token index."""
    for tb in data.turn_boundaries:
        if tb.token_start <= token_idx < tb.token_end:
            return tb
    return None


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — Core Renderer
# ═══════════════════════════════════════════════════════════════════════════

def _extract_window(arr: np.ndarray, t_start: int, t_end: int,
                    n_layers: int, window_size: int) -> np.ndarray:
    """Extract (L, W) window from a (T, L) array, padding left with zeros."""
    chunk = arr[t_start:t_end + 1, :].T  # (L, span)
    span = chunk.shape[1]
    pad = window_size - span
    if pad > 0:
        out = np.zeros((n_layers, window_size), dtype=np.float32)
        out[:, pad:] = chunk
        return out
    return chunk.astype(np.float32)


def _extract_window_3d(arr: np.ndarray, t_start: int, t_end: int,
                       n_layers: int, window_size: int) -> np.ndarray:
    """Extract (L, W, C) window from a (T, L, C) array, padding left."""
    chunk = arr[t_start:t_end + 1, :, :]           # (span, L, C)
    chunk = chunk.transpose(1, 0, 2)                # (L, span, C)
    span = chunk.shape[1]
    pad = window_size - span
    if pad > 0:
        C = chunk.shape[2]
        out = np.zeros((n_layers, window_size, C), dtype=np.float32)
        out[:, pad:, :] = chunk
        return out
    return chunk.astype(np.float32)


@dataclass
class RenderState:
    """Mutable per-frame render state."""
    token_cursor: float = 0.0
    turbulence_phase: float = 0.0
    grain_phase: float = 0.0
    # Cached canvas geometry (set during render)
    canvas_x: int = 0
    canvas_y: int = 0
    canvas_w: int = 0
    canvas_h: int = 0
    # pixel-level geometry
    img_w: int = 0
    img_h: int = 0
    t_start: int = 0
    t_end: int = 0
    pad_left: int = 0
    # Token strip geometry (for click detection)
    strip_x: int = 0
    strip_y: int = 0
    strip_w: int = 0
    strip_h: int = 18
    # Cache fields for frame-skip optimization
    _cached_t_end: int = -1
    _cached_base_rgb: Optional[np.ndarray] = None  # (L, W, 3) float32 after base effects
    _cached_mode_hash: int = 0


def compute_reference_distances(data: ModelData, ref_t: int, ref_l: int) -> np.ndarray:
    """Compute Euclidean distances in JL space from reference point to all cells."""
    ref_vec = data.jl[ref_t, ref_l]  # (16,)
    # Broadcast subtraction: (T, L, 16) - (16,) → (T, L, 16)
    diffs = data.jl - ref_vec
    distances = np.linalg.norm(diffs, axis=-1)  # (T, L)
    return distances


def set_reference_point(app: 'AppState', data: ModelData, t: int, l: int) -> None:
    """Set a new reference point and compute distances."""
    app.ref_token = t
    app.ref_layer = l
    app.ref_mode = True
    app.ref_distances = compute_reference_distances(data, t, l)
    print(f"[phosphenes] Reference point set: token={t}, layer={l}")


def finalize_color_basis(jl: np.ndarray, groups: list[ColorBasisGroup]) -> Optional[ColorBasisResult]:
    """Compute orthonormal basis from 3 color groups and project all JL vectors.

    Args:
        jl: Full JL array (T, L, 16)
        groups: Exactly 3 ColorBasisGroup objects (R, G, B)

    Returns:
        ColorBasisResult with pre-normalized projections, or None if degenerate
    """
    assert len(groups) == 3 and all(g.is_valid() for g in groups)

    v1 = groups[0].compute_vector(jl)
    v2 = groups[1].compute_vector(jl)
    v3 = groups[2].compute_vector(jl)

    # Gram-Schmidt orthonormalization
    EPS = 1e-6

    n1 = np.linalg.norm(v1)
    if n1 < EPS:
        print("[phosphenes] ERROR: R direction is near-zero")
        return None
    e1 = v1 / n1

    v2_orth = v2 - np.dot(v2, e1) * e1
    n2 = np.linalg.norm(v2_orth)
    if n2 < EPS:
        print("[phosphenes] ERROR: G direction is nearly parallel to R — pick a more perpendicular spot")
        return None
    e2 = v2_orth / n2

    v3_orth = v3 - np.dot(v3, e1) * e1 - np.dot(v3, e2) * e2
    n3 = np.linalg.norm(v3_orth)
    if n3 < EPS:
        print("[phosphenes] ERROR: B direction lies in the R-G plane — pick a more perpendicular spot")
        return None
    e3 = v3_orth / n3

    # Sign flip: ensure source centroids project POSITIVE onto their own axis
    # This guarantees that the R reference appears red (not cyan)
    src1_mean = np.mean([jl[t, l] for t, l in groups[0].source_cells], axis=0)
    src2_mean = np.mean([jl[t, l] for t, l in groups[1].source_cells], axis=0)
    src3_mean = np.mean([jl[t, l] for t, l in groups[2].source_cells], axis=0)
    if np.dot(src1_mean, e1) < 0: e1 = -e1
    if np.dot(src2_mean, e2) < 0: e2 = -e2
    if np.dot(src3_mean, e3) < 0: e3 = -e3

    # Project ALL JL vectors onto orthonormal basis
    T, L, D = jl.shape
    raw_r = np.zeros((T, L), dtype=np.float32)
    raw_g = np.zeros((T, L), dtype=np.float32)
    raw_b = np.zeros((T, L), dtype=np.float32)

    for l_idx in range(L):
        layer_vecs = jl[:, l_idx, :]  # (T, 16)
        raw_r[:, l_idx] = layer_vecs @ e1
        raw_g[:, l_idx] = layer_vecs @ e2
        raw_b[:, l_idx] = layer_vecs @ e3

    # Global normalization: percentile-based, computed once over ALL tokens
    def global_norm(arr):
        lo, hi = np.percentile(arr, [2, 98])
        mid = (lo + hi) / 2
        spread = (hi - lo) / 2 + 1e-10
        return np.clip((arr - mid) / spread * 0.5 + 0.5, 0, 1).astype(np.float32)

    proj_r = global_norm(raw_r)
    proj_g = global_norm(raw_g)
    proj_b = global_norm(raw_b)

    return ColorBasisResult(e1=e1, e2=e2, e3=e3, proj_r=proj_r, proj_g=proj_g, proj_b=proj_b)


def compute_perpendicularity_to_vector(data: ModelData, ref_t: int, ref_l: int) -> np.ndarray:
    """Compute how perpendicular each (t, l) is to the reference vector. Returns (T, L) in [0, 1]."""
    ref_vec = data.jl[ref_t, ref_l]  # (16,)
    ref_norm = np.linalg.norm(ref_vec) + 1e-10

    T, L, D = data.jl.shape
    perp = np.zeros((T, L), dtype=np.float32)

    for l in range(L):
        layer_vecs = data.jl[:, l, :]  # (T, 16)
        layer_norms = np.linalg.norm(layer_vecs, axis=1) + 1e-10
        # Cosine similarity
        cos_sim = (layer_vecs @ ref_vec) / (layer_norms * ref_norm)
        # Perpendicularity = 1 - |cos|  (0 = parallel, 1 = perpendicular)
        perp[:, l] = 1.0 - np.abs(cos_sim)

    return perp


def compute_perpendicularity_to_plane(data: ModelData, ref1: tuple[int, int], ref2: tuple[int, int]) -> np.ndarray:
    """Compute how perpendicular each (t, l) is to the plane spanned by two reference vectors."""
    v1 = data.jl[ref1[0], ref1[1]]  # (16,)
    v2 = data.jl[ref2[0], ref2[1]]  # (16,)

    # Orthonormalize: make v2_orth perpendicular to v1
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
    v2_proj = v2 - np.dot(v2, v1_norm) * v1_norm
    v2_orth = v2_proj / (np.linalg.norm(v2_proj) + 1e-10)

    T, L, D = data.jl.shape
    perp = np.zeros((T, L), dtype=np.float32)

    for l in range(L):
        layer_vecs = data.jl[:, l, :]  # (T, 16)
        layer_norms = np.linalg.norm(layer_vecs, axis=1, keepdims=True) + 1e-10
        layer_unit = layer_vecs / layer_norms

        # Project onto plane (v1, v2_orth)
        proj_v1 = (layer_unit @ v1_norm)  # (T,)
        proj_v2 = (layer_unit @ v2_orth)  # (T,)

        # Component in plane
        in_plane_sq = proj_v1**2 + proj_v2**2
        # Perpendicular component (what's left)
        perp[:, l] = np.sqrt(np.clip(1.0 - in_plane_sq, 0, 1))

    return perp


def compute_perpendicularity_to_direction(jl: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Compute how perpendicular each (t, l) is to a direction vector.

    Returns (T, L) in [0, 1] where 1 = perfectly perpendicular, 0 = parallel.
    """
    d_norm = np.linalg.norm(direction) + 1e-10
    d_unit = direction / d_norm

    T, L, D = jl.shape
    perp = np.zeros((T, L), dtype=np.float32)

    for l in range(L):
        layer_vecs = jl[:, l, :]  # (T, 16)
        layer_norms = np.linalg.norm(layer_vecs, axis=1) + 1e-10
        cos_sim = (layer_vecs @ d_unit) / layer_norms
        perp[:, l] = 1.0 - np.abs(cos_sim)

    return perp


def compute_perpendicularity_to_plane_from_dirs(jl: np.ndarray, dir1: np.ndarray, dir2: np.ndarray) -> np.ndarray:
    """Compute how perpendicular each (t, l) is to the plane spanned by two direction vectors.

    Returns (T, L) in [0, 1] where 1 = perfectly perpendicular to plane, 0 = in plane.
    """
    # Orthonormalize
    v1_norm = dir1 / (np.linalg.norm(dir1) + 1e-10)
    v2_proj = dir2 - np.dot(dir2, v1_norm) * v1_norm
    v2_orth = v2_proj / (np.linalg.norm(v2_proj) + 1e-10)

    T, L, D = jl.shape
    perp = np.zeros((T, L), dtype=np.float32)

    for l in range(L):
        layer_vecs = jl[:, l, :]
        layer_norms = np.linalg.norm(layer_vecs, axis=1, keepdims=True) + 1e-10
        layer_unit = layer_vecs / layer_norms

        proj_v1 = layer_unit @ v1_norm
        proj_v2 = layer_unit @ v2_orth
        in_plane_sq = proj_v1**2 + proj_v2**2
        perp[:, l] = np.sqrt(np.clip(1.0 - in_plane_sq, 0, 1))

    return perp


def handle_color_basis_click(app: 'AppState', data: ModelData, t: int, l: int, is_shift: bool) -> None:
    """Handle a click during color basis selection."""
    if not app.color_basis_selecting:
        return

    # Bounds check
    if t < 0 or t >= data.n_tokens or l < 0 or l >= data.n_layers:
        return

    group = app.color_basis_groups[app.color_basis_current_idx]

    if is_shift:
        group.add_contrast(t, l)
        print(f"[phosphenes] Added contrast cell ({t},{l}) to {'RGB'[app.color_basis_current_idx]} — "
              f"{len(group.source_cells)} source + {len(group.contrast_cells)} contrast")
    else:
        group.add_source(t, l)
        print(f"[phosphenes] Added source cell ({t},{l}) to {'RGB'[app.color_basis_current_idx]} — "
              f"{len(group.source_cells)} source + {len(group.contrast_cells)} contrast")


def clear_color_basis(app: 'AppState') -> None:
    """Clear all color basis state."""
    app.color_basis_mode = False
    app.color_basis_selecting = False
    app.color_basis_current_idx = 0
    app.color_basis_groups = []
    app.color_basis_result = None
    app.color_basis_guidance = None
    print("[phosphenes] Color basis cleared")


def advance_color_basis(app: 'AppState', data: ModelData) -> None:
    """Advance to next color axis (Enter key) or finalize if all 3 are set."""
    if not app.color_basis_selecting:
        return

    group = app.color_basis_groups[app.color_basis_current_idx]

    if not group.is_valid():
        print(f"[phosphenes] Need at least one source cell for {'RGB'[app.color_basis_current_idx]} before advancing")
        return

    if app.color_basis_current_idx < 2:
        # Advance to next color
        app.color_basis_current_idx += 1
        # Create next group if needed
        if len(app.color_basis_groups) <= app.color_basis_current_idx:
            app.color_basis_groups.append(ColorBasisGroup())

        # Update perpendicularity guidance
        if app.color_basis_current_idx == 1:
            # After R: show perpendicularity to R direction
            r_vec = app.color_basis_groups[0].compute_vector(data.jl)
            app.color_basis_guidance = compute_perpendicularity_to_direction(data.jl, r_vec)
            print("[phosphenes] R confirmed — grayscale shows perpendicularity. Pick bright spots for G.")
        elif app.color_basis_current_idx == 2:
            # After G: show perpendicularity to R-G plane
            r_vec = app.color_basis_groups[0].compute_vector(data.jl)
            g_vec = app.color_basis_groups[1].compute_vector(data.jl)
            app.color_basis_guidance = compute_perpendicularity_to_plane_from_dirs(data.jl, r_vec, g_vec)
            print("[phosphenes] G confirmed — grayscale shows perpendicularity to R-G plane. Pick bright spots for B.")

        print(f"[phosphenes] Now selecting {'RGB'[app.color_basis_current_idx]}. Click to add source, Shift+Click for contrast.")
    else:
        # All 3 set — finalize!
        result = finalize_color_basis(data.jl, app.color_basis_groups)
        if result is None:
            print("[phosphenes] Degenerate basis — try picking more perpendicular directions")
            return

        app.color_basis_result = result
        app.color_basis_selecting = False
        app.color_basis_mode = True
        app.color_basis_version += 1
        print("[phosphenes] Color basis finalized! Press C to clear.")


def undo_color_basis(app: 'AppState') -> None:
    """Undo last click in current color group."""
    if not app.color_basis_selecting:
        return
    group = app.color_basis_groups[app.color_basis_current_idx]
    if group.undo_last():
        print(f"[phosphenes] Undid last {'RGB'[app.color_basis_current_idx]} cell — "
              f"{len(group.source_cells)} source + {len(group.contrast_cells)} contrast")
    else:
        print("[phosphenes] Nothing to undo")


def _draw_color_basis_markers(
    rgb: np.ndarray,
    groups: Optional[list],
    t_start: int,
    t_end: int,
    pad_left: int,
    L: int,
    W: int,
) -> None:
    """Draw colored dot markers for color basis source and contrast cells.

    Source cells: bright color (R=red, G=green, B=blue)
    Contrast cells: dim version of same color
    """
    if not groups:
        return

    BRIGHT = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    DIM    = [(0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5)]

    for g_idx, group in enumerate(groups):
        if g_idx >= 3:
            break
        bright = BRIGHT[g_idx]
        dim = DIM[g_idx]

        for t, l in group.source_cells:
            if t_start <= t <= t_end:
                x = pad_left + (t - t_start)
                row = L - 1 - l  # flip: layer 0 at bottom
                if 0 <= x < W and 0 <= row < L:
                    rgb[row, x, :] = bright

        for t, l in group.contrast_cells:
            if t_start <= t <= t_end:
                x = pad_left + (t - t_start)
                row = L - 1 - l
                if 0 <= x < W and 0 <= row < L:
                    rgb[row, x, :] = dim


def render_frame(
    data: ModelData,
    state: RenderState,
    params: EffectParams,
    canvas_w: int,
    canvas_h: int,
    *,
    highlight_mode: Optional[str] = None,
    ref_mode: bool = False,
    ref_distances: Optional[np.ndarray] = None,
    ref_token: Optional[int] = None,
    ref_layer: Optional[int] = None,
    color_basis_mode: bool = False,
    color_basis_result: Optional['ColorBasisResult'] = None,
    color_basis_groups: Optional[list] = None,
    color_basis_selecting: bool = False,
    color_basis_guidance: Optional[np.ndarray] = None,
    color_basis_current_idx: int = 0,
    color_basis_version: int = 0,
) -> tuple[np.ndarray, int, int, int, int]:
    """
    Render the main visualization as a scaled region uint8 RGB array.

    Returns: (scaled_region_uint8, x_off, y_off, new_w, new_h)
        - scaled_region_uint8: (new_h, new_w, 3) uint8 array
        - x_off, y_off: offsets within the canvas for centering
        - new_w, new_h: dimensions of the scaled region

    OPTIMIZED: Caches base work (depends on t_end and modes) and only
    recomputes turbulence/grain overlays each frame.

    Pipeline:
      BASE (cached when t_end/modes unchanged):
        1. Extract token window (cell res)
        2. Base color from PCA -> RGB (cell res)
        3. Brightness from energy (cell res)
        4. Heartbeat phase overlay (cell res)
        5. Seam glow (cell res)
        6. Role-based dimming (cell res)
        7. Metric highlight / special modes (cell res)

      OVERLAY (every frame):
        8. Turbulence from delta_l2 (cell res)
        9. Grain from cos_prev instability (cell res)

      FINALIZE (every frame, on cached or fresh base):
        10. Clamp + Gaussian smooth (cell res)
        11. Expand to pixel grid + edge sharpness
        12. Scale to canvas
        13. Self-reference outlines (when paused)
    """
    T, L = data.n_tokens, data.n_layers
    W = TOKENS_VISIBLE

    # --- 1. Determine window ---
    t_end = min(int(state.token_cursor), T - 1)
    t_start = max(0, t_end - W + 1)
    span = t_end - t_start + 1
    pad_left = W - span

    state.t_start = t_start
    state.t_end = t_end
    state.pad_left = pad_left

    # --- Compute mode hash to detect when cached base is invalid ---
    # Hash of all parameters that affect base rendering
    mode_hash = hash((
        highlight_mode,
        ref_mode,
        id(ref_distances) if ref_distances is not None else 0,
        ref_token,
        ref_layer,
        color_basis_mode,
        color_basis_version,
        color_basis_selecting,
        id(color_basis_guidance) if color_basis_guidance is not None else 0,
        canvas_w,
        canvas_h,
    ))

    # --- Check cache validity ---
    cache_valid = (
        t_end == state._cached_t_end and
        mode_hash == state._cached_mode_hash and
        state._cached_base_rgb is not None
    )

    if cache_valid:
        # Use cached base RGB (copy to avoid mutating cache)
        rgb = state._cached_base_rgb.copy()
    else:
        # --- BASE COMPUTATION (only when cache miss) ---

        # --- 2. Base color from PCA (cell resolution: L x W x 3) ---
        rgb = _extract_window_3d(data.pca_rgb, t_start, t_end, L, W)  # (L, W, 3)

        # --- 3. Brightness from energy ---
        en = _extract_window(data.energy_norm, t_start, t_end, L, W)  # (L, W)
        brightness = params.energy_floor + (params.energy_ceil - params.energy_floor) * en
        rgb = rgb * brightness[:, :, np.newaxis]

        # --- 5. Seam glow (cell resolution) ---
        if params.seam_glow_intensity > 0:
            seam_w = data.seam_score[t_start:t_end + 1]
            if pad_left > 0:
                seam_padded = np.zeros(W, dtype=np.float32)
                seam_padded[pad_left:] = seam_w
                seam_w = seam_padded

            # Vertical Gaussian falloff centered at middle (precomputed per L)
            y = np.arange(L, dtype=np.float32)
            y_center = L / 2.0
            vert = np.exp(-((y - y_center) / (L * 0.28)) ** 2)  # (L,)
            glow = (vert[:, np.newaxis] * seam_w[np.newaxis, :]) * params.seam_glow_intensity
            # Blur at cell resolution (cheaper)
            glow = gaussian_filter(glow, sigma=params.seam_glow_sigma / max(CELL_PX, CELL_PY))
            glow_color = np.array([1.0, 0.88, 0.65], dtype=np.float32)
            rgb = rgb + glow[:, :, np.newaxis] * glow_color

        # --- 7. Metric highlight mode (cell resolution) ---
        if highlight_mode is not None:
            lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
            gray = np.stack([lum, lum, lum], axis=-1)

            metric = None
            if highlight_mode == "energy":
                metric = _extract_window(data.energy_norm, t_start, t_end, L, W)
                tint = np.array([0.3, 0.8, 1.0], dtype=np.float32)  # cyan
            elif highlight_mode == "sparsity":
                metric = _extract_window(data.sparsity_norm, t_start, t_end, L, W)
                tint = np.array([0.2, 1.0, 0.4], dtype=np.float32)  # green

            if metric is not None:
                blend = metric[:, :, np.newaxis]
                heat = blend * tint
                rgb = gray * 0.3 + heat * 0.7 * (0.3 + 0.7 * blend)

        # --- Reference point mode (cell resolution) ---
        if ref_mode and ref_distances is not None:
            lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
            gray = np.stack([lum, lum, lum], axis=-1)

            # Extract window of distances
            dist_window = _extract_window(ref_distances, t_start, t_end, L, W)

            # Normalize: 0 = reference point (closest), 1 = far
            # Use quantile to handle outliers
            d_flat = dist_window.flatten()
            d_lo = np.percentile(d_flat, 1)
            d_hi = np.percentile(d_flat, 95)
            dist_norm = np.clip((dist_window - d_lo) / (d_hi - d_lo + 1e-10), 0, 1)

            # Invert: high similarity = bright, low similarity = dark
            similarity = 1.0 - dist_norm

            # Color: warm (close) to cool (far)
            # Close = yellow/white, Far = deep blue
            warm = np.array([1.0, 0.9, 0.5], dtype=np.float32)
            cool = np.array([0.1, 0.2, 0.5], dtype=np.float32)
            sim_3d = similarity[:, :, np.newaxis]
            ref_color = sim_3d * warm + (1 - sim_3d) * cool

            rgb = ref_color * (0.3 + 0.7 * sim_3d)

            # Mark the reference point itself with a bright crosshair indicator
            # Note: rgb array has layer 0 at index 0 (top of array), but layer 0 displays at bottom
            # So we need to flip: rgb row = L - 1 - ref_layer
            if ref_token is not None and ref_layer is not None:
                if t_start <= ref_token <= t_end:
                    ref_x = pad_left + (ref_token - t_start)
                    ref_row = L - 1 - ref_layer  # flip for display
                    if 0 <= ref_x < W and 0 <= ref_row < L:
                        rgb[ref_row, ref_x, :] = [1.0, 1.0, 1.0]  # white marker

        # --- Color basis guidance mode (while selecting G or B) ---
        if color_basis_selecting and color_basis_guidance is not None:
            perp = _extract_window(color_basis_guidance, t_start, t_end, L, W)

            # Normalize to [0, 1]
            flat = perp.flatten()
            lo = np.percentile(flat, 5)
            hi = np.percentile(flat, 95)
            perp_norm = np.clip((perp - lo) / (hi - lo + 1e-10), 0, 1)

            # Grayscale: bright = perpendicular (good), dark = parallel (redundant)
            rgb = np.zeros((L, W, 3), dtype=np.float32)
            rgb[:, :, 0] = perp_norm
            rgb[:, :, 1] = perp_norm
            rgb[:, :, 2] = perp_norm

            # Draw markers for all already-defined groups + current group
            _draw_color_basis_markers(rgb, color_basis_groups, t_start, t_end, pad_left, L, W)

        # --- Color basis mode (finalized) ---
        elif color_basis_mode and color_basis_result is not None:
            r_win = _extract_window(color_basis_result.proj_r, t_start, t_end, L, W)
            g_win = _extract_window(color_basis_result.proj_g, t_start, t_end, L, W)
            b_win = _extract_window(color_basis_result.proj_b, t_start, t_end, L, W)

            # Already globally normalized -- use directly
            rgb = np.zeros((L, W, 3), dtype=np.float32)
            rgb[:, :, 0] = r_win
            rgb[:, :, 1] = g_win
            rgb[:, :, 2] = b_win

            # Draw markers for all 3 groups
            _draw_color_basis_markers(rgb, color_basis_groups, t_start, t_end, pad_left, L, W)

        # --- Cache the base RGB ---
        state._cached_base_rgb = rgb.copy()
        state._cached_t_end = t_end
        state._cached_mode_hash = mode_hash

    # --- OVERLAY EFFECTS (always applied, animate each frame) ---

    # --- 8. Turbulence from delta_l2 (cell resolution) ---
    if params.turbulence_amp > 0 and data.noise_textures:
        delta = _extract_window(data.delta_norm, t_start, t_end, L, W)
        phase = state.turbulence_phase
        idx_a = int(phase) % len(data.noise_textures)
        idx_b = (idx_a + 1) % len(data.noise_textures)
        frac = phase - int(phase)
        noise = data.noise_textures[idx_a] * (1.0 - frac) + data.noise_textures[idx_b] * frac
        turb = noise * delta * params.turbulence_amp
        rgb = rgb * (1.0 + turb[:, :, np.newaxis])

    # --- 9. Grain from cos_prev instability (cell resolution) ---
    if params.grain_max > 0 and data.grain_textures:
        cos_inst = _extract_window(data.cos_instability, t_start, t_end, L, W)
        grain_phase = state.grain_phase  # independent phase, animates even when paused
        idx_a = int(grain_phase) % len(data.grain_textures)
        idx_b = (idx_a + 1) % len(data.grain_textures)
        frac = grain_phase - int(grain_phase)
        grain = data.grain_textures[idx_a] * (1.0 - frac) + data.grain_textures[idx_b] * frac
        grain = grain * params.grain_max
        rgb = rgb * (1.0 + (grain * cos_inst)[:, :, np.newaxis])

    # --- FINALIZE ---

    # --- 10. Clamp at cell resolution ---
    np.clip(rgb, 0.0, 1.0, out=rgb)

    # --- Gaussian smooth at cell resolution (much cheaper: L x W) ---
    if params.smooth_sigma > 0:
        # Smooth at cell res with proportionally smaller sigma
        cell_sigma = params.smooth_sigma / min(CELL_PX, CELL_PY) * 1.5
        if cell_sigma > 0.3:
            for c in range(3):
                rgb[:, :, c] = gaussian_filter(rgb[:, :, c], sigma=cell_sigma)

    # --- 11. Expand to pixel grid ---
    h_px = L * CELL_PY
    w_px = W * CELL_PX
    rgb_px = np.repeat(np.repeat(rgb, CELL_PY, axis=0), CELL_PX, axis=1)  # (h_px, w_px, 3)

    # Edge sharpness from sparsity (thin grid lines)
    if params.edge_dark > 0 and data.edge_y_cell is not None:
        sp = _extract_window(data.sparsity_norm, t_start, t_end, L, W)
        sp_px = np.repeat(np.repeat(sp, CELL_PY, axis=0), CELL_PX, axis=1)
        edge_y_full = np.tile(data.edge_y_cell, L)
        edge_x_full = np.tile(data.edge_x_cell, W)
        edge_mask = np.maximum(edge_y_full[:, np.newaxis], edge_x_full[np.newaxis, :])
        rgb_px *= (1.0 - edge_mask * sp_px * params.edge_dark)[:, :, np.newaxis]

    np.clip(rgb_px, 0.0, 1.0, out=rgb_px)

    # --- 12. Scale to canvas ---
    scale_y = canvas_h / h_px
    scale_x = canvas_w / w_px
    scale = min(scale_y, scale_x)

    new_h = max(1, int(h_px * scale))
    new_w = max(1, int(w_px * scale))

    # Use pre-computed index arrays if size hasn't changed, else compute
    y_idx = np.clip((np.arange(new_h) * (h_px / new_h)).astype(np.intp), 0, h_px - 1)
    x_idx = np.clip((np.arange(new_w) * (w_px / new_w)).astype(np.intp), 0, w_px - 1)

    y_off = (canvas_h - new_h) // 2
    x_off = (canvas_w - new_w) // 2

    # Scale and convert to uint8
    scaled_region = rgb_px[y_idx[:, np.newaxis], x_idx[np.newaxis, :], :]
    scaled_region_uint8 = (scaled_region * 255).astype(np.uint8)

    state.canvas_x = x_off
    state.canvas_y = y_off
    state.canvas_w = new_w
    state.canvas_h = new_h
    state.img_w = w_px
    state.img_h = h_px

    return (scaled_region_uint8, x_off, y_off, new_w, new_h)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — Inspector System
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class InspectorState:
    enabled: bool = True
    token_idx: Optional[int] = None
    layer_idx: Optional[int] = None
    mx: int = 0
    my: int = 0


def update_inspector(
    inspector: InspectorState,
    data: ModelData,
    rstate: RenderState,
    mouse_pos: tuple[int, int],
    screen_margin_y: int,
) -> None:
    """Map screen coords → (token, layer) in data space."""
    mx, my = mouse_pos
    inspector.mx = mx
    inspector.my = my

    # Adjust for header margin
    my_adj = my - screen_margin_y

    cx = rstate.canvas_x
    cy = rstate.canvas_y
    cw = rstate.canvas_w
    ch = rstate.canvas_h

    if not (cx <= mx < cx + cw and cy <= my_adj < cy + ch):
        inspector.token_idx = None
        inspector.layer_idx = None
        return

    rel_x = (mx - cx) / cw
    rel_y = (my_adj - cy) / ch

    # Token index
    tok_in_win = int(rel_x * TOKENS_VISIBLE)
    if tok_in_win < rstate.pad_left:
        inspector.token_idx = None
        inspector.layer_idx = None
        return

    inspector.token_idx = rstate.t_start + (tok_in_win - rstate.pad_left)
    inspector.token_idx = min(inspector.token_idx, data.n_tokens - 1)

    # Layer index (bottom = layer 0 → flip Y)
    inspector.layer_idx = int((1.0 - rel_y) * data.n_layers)
    inspector.layer_idx = max(0, min(data.n_layers - 1, inspector.layer_idx))


def screen_to_token_layer(
    data: ModelData,
    rstate: RenderState,
    mouse_pos: tuple[int, int],
    screen_margin_y: int,
) -> tuple[Optional[int], Optional[int]]:
    """Convert screen coordinates to (token, layer) indices. Returns (None, None) if outside canvas."""
    mx, my = mouse_pos
    my_adj = my - screen_margin_y

    cx = rstate.canvas_x
    cy = rstate.canvas_y
    cw = rstate.canvas_w
    ch = rstate.canvas_h

    if not (cx <= mx < cx + cw and cy <= my_adj < cy + ch):
        return None, None

    rel_x = (mx - cx) / cw
    rel_y = (my_adj - cy) / ch

    tok_in_win = int(rel_x * TOKENS_VISIBLE)
    if tok_in_win < rstate.pad_left:
        return None, None

    token_idx = rstate.t_start + (tok_in_win - rstate.pad_left)
    token_idx = min(token_idx, data.n_tokens - 1)

    layer_idx = int((1.0 - rel_y) * data.n_layers)
    layer_idx = max(0, min(data.n_layers - 1, layer_idx))

    return token_idx, layer_idx


def strip_click_to_token(
    data: ModelData,
    rstate: RenderState,
    mouse_pos: tuple[int, int],
) -> Optional[int]:
    """Convert click on token strip to token index. Returns None if outside strip."""
    mx, my = mouse_pos

    sx = rstate.strip_x
    sy = rstate.strip_y
    sw = rstate.strip_w
    sh = rstate.strip_h

    if not (sx <= mx < sx + sw and sy <= my < sy + sh):
        return None

    rel_x = (mx - sx) / sw
    tok_in_win = int(rel_x * TOKENS_VISIBLE)

    if tok_in_win < rstate.pad_left:
        return None

    token_idx = rstate.t_start + (tok_in_win - rstate.pad_left)
    token_idx = min(token_idx, data.n_tokens - 1)

    return token_idx


def _display_piece(piece: str) -> str:
    if piece == "<0x0A>":
        return "\\n"
    if piece == "▁":
        return " "
    # Handle BPE space prefix (Ġ = U+0120, used by GPT/Qwen tokenizers for leading space)
    piece = piece.replace("Ġ", " ")
    # Handle BPE newline prefix (Ċ = U+010A)
    piece = piece.replace("Ċ", "\\n")
    return piece.replace("\n", "\\n")


def render_inspector(
    screen: pygame.Surface,
    font: pygame.freetype.Font,
    inspector: InspectorState,
    data: ModelData,
    rstate: RenderState,
    margin_y: int,
) -> None:
    """Render inspector tooltip and crosshairs."""
    if not inspector.enabled or inspector.token_idx is None:
        return

    t = inspector.token_idx
    layer = inspector.layer_idx

    # --- Crosshair lines on canvas ---
    cx = rstate.canvas_x
    cy = rstate.canvas_y + margin_y
    cw = rstate.canvas_w
    ch = rstate.canvas_h

    rel_t = (t - rstate.t_start + rstate.pad_left + 0.5) / TOKENS_VISIBLE
    rel_l = 1.0 - (layer + 0.5) / data.n_layers

    cross_x = int(cx + rel_t * cw)
    cross_y = int(cy + rel_l * ch)

    pygame.draw.line(screen, AMBER, (cross_x, cy), (cross_x, cy + ch), 1)
    pygame.draw.line(screen, AMBER, (cx, cross_y), (cx + cw, cross_y), 1)

    # --- Tooltip ---
    lines = [
        f"Token {t}, Layer {layer}",
    ]

    # Decoded token
    if data.token_pieces and t < len(data.token_pieces):
        piece = _display_piece(data.token_pieces[t])
        lines.append(f'"{piece}"')
    elif data.input_ids is not None and t < len(data.input_ids):
        lines.append(f"ID: {int(data.input_ids[t])}")

    # Turn info
    if data.has_turns:
        ct = get_current_turn(data, t)
        if ct:
            pos_in = t - ct.token_start
            turn_len = ct.token_end - ct.token_start
            lines.append(f"Turn {ct.turn} ({ct.role}) {pos_in}/{turn_len}")

    lines.append("")
    lines.append(f"jl_energy:  {data.jl_energy[t, layer]:.2f}")
    lines.append(f"delta_l2:   {data.delta_l2[t, layer]:.2f}")
    lines.append(f"cos_prev:   {data.cos_prev[t, layer]:.3f}")
    lines.append(f"top1_frac:  {data.top1_frac[t, layer]:.3f}")
    lines.append(f"top25_frac: {data.top25_frac[t, layer]:.3f}")
    lines.append(f"cluster:    {data.cluster_labels[t, layer]}")
    lines.append(f"seam:       {data.seam_score[t]:.3f}")
    lines.append(f"hetero:     {data.heterogeneity[t]:.3f}")

    line_h = 17
    box_w = 210
    box_h = len(lines) * line_h + 16

    # Position near mouse, clamped to screen
    tx = inspector.mx + 24
    ty = inspector.my - box_h // 2
    sw, sh = screen.get_size()
    if tx + box_w > sw - 10:
        tx = inspector.mx - box_w - 24
    ty = max(10, min(sh - box_h - 10, ty))

    # Draw tooltip background
    tip_surf = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
    tip_surf.fill((15, 15, 25, 225))
    pygame.draw.rect(tip_surf, AMBER, (0, 0, box_w, box_h), 2, border_radius=4)
    screen.blit(tip_surf, (tx, ty))

    # Draw text
    for i, line in enumerate(lines):
        color = AMBER if i == 0 else TEXT_COL
        if line == "":
            continue
        font.render_to(screen, (tx + 8, ty + 8 + i * line_h), line, color, size=13)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5b — Turn Markers
# ═══════════════════════════════════════════════════════════════════════════

TURN_COLORS = {
    "system": (100, 100, 120),
    "user": (120, 180, 255),
    "assistant": (255, 200, 100),
}


def render_role_bar(
    screen: pygame.Surface,
    data: ModelData,
    rstate: RenderState,
    header_h: int,
    new_w: int,
    new_h: int,
) -> None:
    """Draw a colored role bar at the top of the visualization — like an extra 'layer 64'.

    Blue for user turns, amber for assistant turns, matching the text panel colors.
    """
    if not data.has_turns or not data.turn_boundaries:
        return

    L = data.n_layers
    W = TOKENS_VISIBLE
    cell_w = new_w / W
    bar_h = max(4, int(new_h / L))  # same height as one cell row

    origin_x = rstate.canvas_x
    origin_y = header_h + rstate.canvas_y  # top of visualization

    # Draw bar at the very top of the visual (overlapping top row acts as "layer 64")
    bar_y = origin_y - bar_h - 1  # 1px gap above the visual

    ROLE_COLORS = {
        "user": (120, 180, 255),
        "assistant": (255, 200, 100),
        "system": (100, 100, 120),
    }

    for tb in data.turn_boundaries:
        color = ROLE_COLORS.get(tb.role)
        if not color:
            continue

        # Find the visible range of this turn
        vis_start = max(tb.token_start, rstate.t_start)
        vis_end = min(tb.token_end, rstate.t_end + 1)
        if vis_start >= vis_end:
            continue

        # Convert to window coordinates and draw as one rect per contiguous span
        win_start = rstate.pad_left + (vis_start - rstate.t_start)
        win_end = rstate.pad_left + (vis_end - rstate.t_start)

        x1 = int(origin_x + win_start * cell_w)
        x2 = int(origin_x + win_end * cell_w)
        pygame.draw.rect(screen, color, (x1, bar_y, x2 - x1, bar_h))


def render_turn_markers(
    screen: pygame.Surface,
    font: pygame.freetype.Font,
    data: ModelData,
    rstate: RenderState,
    margin_y: int,
) -> None:
    """Draw vertical lines at turn boundaries and turn labels."""
    if not data.has_turns or not data.turn_boundaries:
        return

    cx = rstate.canvas_x
    cy = rstate.canvas_y + margin_y
    cw = rstate.canvas_w
    ch = rstate.canvas_h

    prev_turn = -1
    for tb in data.turn_boundaries:
        if tb.token_start < rstate.t_start or tb.token_start > rstate.t_end + 1:
            continue

        win_pos = rstate.pad_left + (tb.token_start - rstate.t_start)
        rel_x = win_pos / TOKENS_VISIBLE
        line_x = int(cx + rel_x * cw)

        color = TURN_COLORS.get(tb.role, (150, 150, 150))

        if tb.role == "user":
            # Dashed line
            y = cy
            while y < cy + ch:
                end_y = min(y + 8, cy + ch)
                pygame.draw.line(screen, color, (line_x, y), (line_x, end_y), 1)
                y += 12
        else:
            pygame.draw.line(screen, color, (line_x, cy), (line_x, cy + ch), 1)

        # Turn label at top (only once per turn number)
        if tb.turn != prev_turn:
            label = f"T{tb.turn}"
            font.render_to(screen, (line_x + 4, cy - 16), label, color, size=10)
        prev_turn = tb.turn


def render_color_basis_overlay(
    screen: pygame.Surface,
    font: pygame.freetype.Font,
    app: 'AppState',
    data: ModelData,
    rstate: RenderState,
    header_h: int,
    new_w: int,
    new_h: int,
) -> None:
    """Draw selection markers for color basis groups as a pygame overlay.

    Source cells: bright outline in group color + small "+" sign
    Contrast cells: dimmer outline in group color + small "-" sign
    """
    if not app.color_basis_groups:
        return

    # Cell dimensions in screen pixels
    L = data.n_layers
    W = TOKENS_VISIBLE
    cell_w = new_w / W
    cell_h = new_h / L

    # Origin of the visualization on screen
    origin_x = rstate.canvas_x
    origin_y = header_h + rstate.canvas_y

    # Colors for each group axis
    BRIGHT_COLORS = [(255, 60, 60), (60, 255, 60), (60, 60, 255)]   # R, G, B
    DIM_COLORS    = [(160, 40, 40), (40, 160, 40), (40, 40, 160)]   # dimmer versions
    SIGN_BRIGHT   = [(255, 180, 180), (180, 255, 180), (180, 180, 255)]  # for + sign
    SIGN_DIM      = [(200, 120, 120), (120, 200, 120), (120, 120, 200)]  # for - sign

    for g_idx, group in enumerate(app.color_basis_groups):
        if g_idx >= 3:
            break

        bright = BRIGHT_COLORS[g_idx]
        dim = DIM_COLORS[g_idx]
        sign_bright = SIGN_BRIGHT[g_idx]
        sign_dim = SIGN_DIM[g_idx]

        # Draw source cells: bright outline + "+" sign
        for t, l in group.source_cells:
            if rstate.t_start <= t <= rstate.t_end:
                win_col = rstate.pad_left + (t - rstate.t_start)
                win_row = L - 1 - l  # layer 0 at bottom

                sx = int(origin_x + win_col * cell_w)
                sy = int(origin_y + win_row * cell_h)
                sw = max(2, int(cell_w))
                sh = max(2, int(cell_h))

                # Draw outline rect (2px for visibility)
                pygame.draw.rect(screen, bright, (sx, sy, sw, sh), 2)

                # Draw "+" sign centered in cell
                cx = sx + sw // 2
                cy = sy + sh // 2
                half = max(2, min(sw, sh) // 4)
                pygame.draw.line(screen, sign_bright, (cx - half, cy), (cx + half, cy), 1)
                pygame.draw.line(screen, sign_bright, (cx, cy - half), (cx, cy + half), 1)

        # Draw contrast cells: dim outline + "-" sign
        for t, l in group.contrast_cells:
            if rstate.t_start <= t <= rstate.t_end:
                win_col = rstate.pad_left + (t - rstate.t_start)
                win_row = L - 1 - l

                sx = int(origin_x + win_col * cell_w)
                sy = int(origin_y + win_row * cell_h)
                sw = max(2, int(cell_w))
                sh = max(2, int(cell_h))

                # Draw outline rect
                pygame.draw.rect(screen, dim, (sx, sy, sw, sh), 2)

                # Draw "-" sign centered in cell
                cx = sx + sw // 2
                cy = sy + sh // 2
                half = max(2, min(sw, sh) // 4)
                pygame.draw.line(screen, sign_dim, (cx - half, cy), (cx + half, cy), 1)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — Text Display
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TextState:
    enabled: bool = False
    radius: int = 25   # tokens before/after


@dataclass
class TextPanelState:
    """State for the right-margin text panel."""
    enabled: bool = True
    panel_width: int = 250
    font_size: int = 13
    line_height: int = 18
    # Pre-computed wrapped lines: list of list of (token_idx, display_text, role, x_offset)
    wrapped_lines: list = field(default_factory=list)
    _cached_for_stem: str = ""


def _build_text_panel_lines(data: ModelData, font: pygame.freetype.Font, panel_width: int, font_size: int) -> list:
    """Pre-compute word-wrapped lines for the text panel.

    Returns list of lines, where each line is a list of (token_idx, display_text, role, x_offset) tuples.
    """
    if not data.token_pieces:
        return []

    margin = 10  # left/right padding inside panel
    usable_width = panel_width - 2 * margin

    lines = []
    current_line = []
    current_x = 0.0

    for t_idx in range(data.n_tokens):
        piece = _display_piece(data.token_pieces[t_idx])
        role = data.token_roles[t_idx] if data.token_roles else ""

        # Measure text width
        rect = font.get_rect(piece, size=font_size)
        piece_w = rect.width

        # Handle explicit newlines in the piece - start new line
        if "\\n" in piece and current_line:
            lines.append(current_line)
            current_line = []
            current_x = 0.0
            # Still add the newline token to the new line
            current_line.append((t_idx, piece, role, current_x))
            current_x += piece_w
            # Start another new line after the newline token
            lines.append(current_line)
            current_line = []
            current_x = 0.0
            continue

        # Wrap if this piece would overflow
        if current_x + piece_w > usable_width and current_line:
            lines.append(current_line)
            current_line = []
            current_x = 0.0

        current_line.append((t_idx, piece, role, current_x))
        current_x += piece_w

    if current_line:
        lines.append(current_line)

    return lines


def render_text_panel(
    screen: pygame.Surface,
    font: pygame.freetype.Font,
    data: ModelData,
    rstate: RenderState,
    text_panel: TextPanelState,
    panel_x: int,
    panel_h: int,
    panel_y: int,
) -> None:
    """Render the right-margin text panel with role-colored tokens."""
    if not text_panel.enabled or not text_panel.wrapped_lines:
        return

    pw = text_panel.panel_width
    margin = 10
    fs = text_panel.font_size
    lh = text_panel.line_height

    # Semi-transparent background
    bg_surf = pygame.Surface((pw, panel_h), pygame.SRCALPHA)
    bg_surf.fill((11, 11, 18, 220))
    screen.blit(bg_surf, (panel_x, panel_y))

    # Find which line contains the middle visible token
    t_mid = (rstate.t_start + rstate.t_end) // 2
    target_line = 0
    for i, line in enumerate(text_panel.wrapped_lines):
        for (t_idx, _, _, _) in line:
            if t_idx >= t_mid:
                target_line = i
                break
        else:
            continue
        break

    # Calculate scroll so target_line is vertically centered
    visible_lines = panel_h // lh
    start_line = max(0, target_line - visible_lines // 2)
    end_line = min(len(text_panel.wrapped_lines), start_line + visible_lines)

    # Role colors
    ROLE_COLORS = {
        "user": (120, 180, 255),       # blue
        "assistant": (255, 200, 100),   # amber/gold
        "system": (100, 100, 120),      # gray
        "": (180, 180, 200),            # default light gray
        "unknown": (180, 180, 200),
    }
    HIGHLIGHT_COLOR = (255, 255, 255)  # white for current token

    # Draw visible lines
    for line_idx in range(start_line, end_line):
        line = text_panel.wrapped_lines[line_idx]
        y = panel_y + (line_idx - start_line) * lh + 4

        for (t_idx, piece, role, x_off) in line:
            # Highlight the middle token
            is_current = (t_idx == t_mid)
            color = HIGHLIGHT_COLOR if is_current else ROLE_COLORS.get(role, (180, 180, 200))

            # Dim tokens far from current view
            if not is_current:
                dist = abs(t_idx - t_mid)
                if dist > 80:
                    # Fade out distant tokens
                    fade = max(0.3, 1.0 - (dist - 80) / 160)
                    color = tuple(int(c * fade) for c in color)

            x = panel_x + margin + int(x_off)
            font.render_to(screen, (x, y), piece, color, size=fs)

    # Draw thin separator line on the left edge of panel
    pygame.draw.line(screen, (60, 60, 80), (panel_x, panel_y), (panel_x, panel_y + panel_h), 1)


def render_text_ticker(
    screen: pygame.Surface,
    font: pygame.freetype.Font,
    data: ModelData,
    cursor: float,
    text_state: TextState,
    y_pos: int,
) -> None:
    """Render the synced text ticker bar."""
    if not text_state.enabled:
        return

    t = min(int(cursor), data.n_tokens - 1)
    sw = screen.get_width()

    # Semi-transparent background bar
    bar_h = 36
    bar_surf = pygame.Surface((sw - 40, bar_h), pygame.SRCALPHA)
    bar_surf.fill((10, 10, 18, 200))
    screen.blit(bar_surf, (20, y_pos))

    # Build context string
    r = text_state.radius
    t_lo = max(0, t - r)
    t_hi = min(data.n_tokens, t + r + 1)

    if data.token_pieces:
        before = "".join(_display_piece(data.token_pieces[i]) for i in range(t_lo, t))
        current = _display_piece(data.token_pieces[t]) if t < len(data.token_pieces) else "?"
        after = "".join(_display_piece(data.token_pieces[i]) for i in range(t + 1, t_hi))
    else:
        before = " ".join(str(int(data.input_ids[i])) for i in range(t_lo, t))
        current = str(int(data.input_ids[t]))
        after = " ".join(str(int(data.input_ids[i])) for i in range(t + 1, t_hi))

    # Truncate
    max_side = 60
    if len(before) > max_side:
        before = "..." + before[-max_side:]
    if len(after) > max_side:
        after = after[:max_side] + "..."

    display = f"{before}[{current}]{after}"
    font.render_to(screen, (30, y_pos + 6), display, (200, 200, 220), size=12)

    # Position info on right
    info = f"t={t}/{data.n_tokens}  seam={data.seam_score[t]:.2f}  het={data.heterogeneity[t]:.2f}"
    font.render_to(screen, (sw - 300, y_pos + 6), info, TEXT_COL, size=12)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — Main Loop
# ═══════════════════════════════════════════════════════════════════════════

HIGHLIGHT_MODES = [None, "energy", "sparsity"]
HIGHLIGHT_NAMES = ["off", "energy", "sparsity"]


@dataclass
class TutorialState:
    """State for the multi-page tutorial overlay."""
    active: bool = True  # starts active on launch
    current_page: int = 0
    pages: list = field(default_factory=list)  # list of strings (one per page)
    n_pages: int = 0


@dataclass
class AppState:
    running: bool = True
    paused: bool = False
    playback_speed: float = 1.0
    show_turn_markers: bool = True
    highlight_idx: int = 0
    recording: bool = False
    frame_count: int = 0
    current_key: int = 9
    is_fullscreen: bool = False
    # Reference point mode
    ref_mode: bool = False          # True when viewing distances from reference
    ref_token: Optional[int] = None  # reference token index
    ref_layer: Optional[int] = None  # reference layer index
    ref_distances: Optional[np.ndarray] = None  # (T, L) distances from reference
    # Color basis mode (C key) - multi-select RGB reference directions
    color_basis_mode: bool = False          # True when finalized basis is active
    color_basis_selecting: bool = False     # True while user is picking cells
    color_basis_current_idx: int = 0        # 0=R, 1=G, 2=B (which color we're setting)
    color_basis_groups: list = field(default_factory=list)  # list of ColorBasisGroup (up to 3)
    color_basis_result: Optional[ColorBasisResult] = None   # finalized result (None while selecting)
    color_basis_guidance: Optional[np.ndarray] = None       # (T, L) perpendicularity for guidance display
    color_basis_version: int = 0            # bump on finalize, used for cache invalidation


def _load_font() -> pygame.freetype.Font:
    """Load the best available font."""
    for p in FONT_SEARCH:
        if Path(p).exists():
            try:
                return pygame.freetype.Font(p)
            except Exception:
                continue
    return pygame.freetype.SysFont("monospace", 14)


def load_tutorial_pages() -> list[str]:
    """Load tutorial pages from intro/ directory next to this script."""
    intro_dir = Path(__file__).resolve().parent / "intro"
    if not intro_dir.exists():
        return []

    pages = []
    for f in sorted(intro_dir.glob("page*.txt")):
        pages.append(f.read_text(encoding="utf-8"))
    return pages


def render_tutorial(
    screen: pygame.Surface,
    font: pygame.freetype.Font,
    tutorial: TutorialState,
) -> None:
    """Render the current tutorial page as a full-screen overlay."""
    if not tutorial.active or not tutorial.pages:
        return

    sw, sh = screen.get_size()

    # Dark background
    screen.fill((11, 11, 18))

    page_text = tutorial.pages[tutorial.current_page]
    lines = page_text.split("\n")

    # Calculate vertical centering
    line_height = 22
    total_height = len(lines) * line_height
    start_y = max(40, (sh - total_height) // 2 - 40)

    # Track if we've seen the first title (for subtitle handling)
    seen_first_title = False

    # Render each line centered horizontally
    for i, line in enumerate(lines):
        y = start_y + i * line_height

        if not line.strip():
            continue

        stripped = line.strip()

        if stripped == "---":
            # Draw a subtle horizontal line
            line_w = min(500, sw - 200)
            line_x = (sw - line_w) // 2
            pygame.draw.line(screen, (60, 60, 80), (line_x, y + 8), (line_x + line_w, y + 8), 1)
            continue

        # Check if this is a title (ALL CAPS, no lowercase)
        is_title = stripped == stripped.upper() and any(c.isalpha() for c in stripped) and len(stripped) > 3

        if is_title and i < 3 and not seen_first_title:
            # Main title: larger, amber
            seen_first_title = True
            rect = font.get_rect(stripped, size=22)
            x = (sw - rect.width) // 2
            font.render_to(screen, (x, y), stripped, (255, 200, 100), size=22)
        elif seen_first_title and i < 4 and not is_title:
            # Subtitle: slightly smaller, softer color
            rect = font.get_rect(stripped, size=15)
            x = (sw - rect.width) // 2
            font.render_to(screen, (x, y), stripped, (180, 160, 120), size=15)
        elif is_title:
            # Section header: medium, amber
            rect = font.get_rect(stripped, size=16)
            x = (sw - rect.width) // 2
            font.render_to(screen, (x, y), stripped, (255, 200, 100), size=16)
        else:
            # Normal text: centered, light
            # Detect indented lines (keep them relatively indented)
            indent = len(line) - len(line.lstrip())

            # Center the block of text
            rect = font.get_rect(stripped, size=14)

            if indent >= 3:
                # Indented line: offset from a left margin rather than centering
                left_margin = max(100, (sw - 500) // 2)
                x = left_margin + indent * 7
                color = (180, 180, 200)  # slightly dimmer for indented
            else:
                # Center normally
                x = (sw - rect.width) // 2
                color = (220, 220, 235)

            font.render_to(screen, (x, y), stripped, color, size=14)

    # Navigation footer
    footer_y = sh - 50

    page_indicator = f"Page {tutorial.current_page + 1} / {tutorial.n_pages}"
    rect = font.get_rect(page_indicator, size=12)
    font.render_to(screen, ((sw - rect.width) // 2, footer_y + 20), page_indicator, (100, 100, 120), size=12)

    # Navigation hints
    nav_parts = []
    if tutorial.current_page > 0:
        nav_parts.append("[<-] Back")
    if tutorial.current_page < tutorial.n_pages - 1:
        nav_parts.append("[->] Next")
    nav_parts.append("[Space] Start")
    nav_parts.append("[1-8] Jump to session")

    nav_text = "    ".join(nav_parts)
    rect = font.get_rect(nav_text, size=13)
    font.render_to(screen, ((sw - rect.width) // 2, footer_y), nav_text, (150, 150, 170), size=13)


def _render_header(
    screen: pygame.Surface,
    font: pygame.freetype.Font,
    data: ModelData,
    app: AppState,
    rstate: RenderState,
    params: EffectParams,
) -> None:
    sw = screen.get_width()

    # Line 1: model + playback
    model_str = f"PHOSPHENES   {data.display_name} ({data.architecture}, {data.n_layers}L)"
    font.render_to(screen, (20, 14), model_str, TEXT_COL, size=14)

    t = min(int(rstate.token_cursor), data.n_tokens - 1)
    status = "PAUSED" if app.paused else f"{app.playback_speed:.1f}x"
    if app.recording:
        status += "  [REC]"
    status_col = AMBER if app.paused else TEXT_COL
    font.render_to(screen, (sw - 200, 14), status, status_col, size=14)

    # Line 2: modes + position + turn info
    parts = []
    if app.color_basis_mode and app.color_basis_result:
        def _group_str(idx, letter):
            if idx < len(app.color_basis_groups):
                g = app.color_basis_groups[idx]
                s = len(g.source_cells)
                c = len(g.contrast_cells)
                return f"{letter}={s}s" + (f"+{c}c" if c else "")
            return f"{letter}=?"
        parts.append(f"ORTHO: {_group_str(0,'R')} {_group_str(1,'G')} {_group_str(2,'B')}")
    elif app.color_basis_selecting:
        color_name = "RGB"[app.color_basis_current_idx]
        g = app.color_basis_groups[app.color_basis_current_idx] if app.color_basis_current_idx < len(app.color_basis_groups) else None
        n_src = len(g.source_cells) if g else 0
        n_con = len(g.contrast_cells) if g else 0
        if app.color_basis_current_idx == 0 and app.color_basis_guidance is None:
            parts.append(f"SELECT {color_name}: click=source, shift+click=contrast, Enter=confirm ({n_src}s+{n_con}c)")
        else:
            parts.append(f"SELECT {color_name}: bright=perpendicular, click=source, Enter=confirm ({n_src}s+{n_con}c)")
    elif app.ref_mode:
        parts.append(f"REF: t={app.ref_token}, L={app.ref_layer}")
    else:
        hname = HIGHLIGHT_NAMES[app.highlight_idx]
        if hname != "off":
            parts.append(f"Highlight: {hname}")
    if data.has_turns:
        ct = get_current_turn(data, t)
        if ct:
            parts.append(f"Turn {ct.turn} ({ct.role})")
    parts.append(f"t={t}/{data.n_tokens}")
    line2 = "   ".join(parts)
    if app.color_basis_mode or app.color_basis_selecting:
        mode_col = (150, 255, 200)  # cyan-ish
    elif app.ref_mode:
        mode_col = (255, 220, 150)  # warm
    else:
        mode_col = DIM_COL
    font.render_to(screen, (20, 34), line2, mode_col, size=11)


def _render_controls(screen: pygame.Surface, font: pygame.freetype.Font, has_turns: bool = False, ref_mode: bool = False, color_basis: bool = False) -> None:
    sh = screen.get_height()
    base = "Space:play  \u2190\u2192:step  \u2191\u2193:speed  Tab:text  I:inspect  M:mode  C:color"
    ref_hint = "  P/Click:ref" if not ref_mode else "  P:clear-ref"
    turn_hints = "  T:turns  []:jump" if has_turns else ""
    tail = "  H:help  1-9:model  F:full  Q:quit"
    font.render_to(screen, (20, sh - 18), base + ref_hint + turn_hints + tail, DIM_COL, size=10)


def render_token_strip(
    screen: pygame.Surface,
    font: pygame.freetype.Font,
    data: ModelData,
    rstate: RenderState,
    inspector: 'InspectorState',
    canvas_x: int,
    canvas_w: int,
    y_pos: int,
) -> None:
    """Render horizontal token strip aligned with canvas columns (when paused)."""
    t_start = rstate.t_start
    t_end = rstate.t_end
    pad_left = rstate.pad_left
    W = TOKENS_VISIBLE
    cell_w = canvas_w / W

    # Save geometry for click detection
    strip_h = 18
    rstate.strip_x = canvas_x
    rstate.strip_y = y_pos
    rstate.strip_w = canvas_w
    rstate.strip_h = strip_h

    # Semi-transparent background
    strip_surf = pygame.Surface((canvas_w, strip_h), pygame.SRCALPHA)
    strip_surf.fill((10, 10, 18, 180))
    screen.blit(strip_surf, (canvas_x, y_pos))

    # Render each visible token
    for t_idx in range(t_start, min(t_end + 1, data.n_tokens)):
        win_pos = pad_left + (t_idx - t_start)
        x = canvas_x + int(win_pos * cell_w)

        # Get token text
        if data.token_pieces and t_idx < len(data.token_pieces):
            piece = _display_piece(data.token_pieces[t_idx])
        else:
            piece = str(t_idx)

        # Truncate long tokens
        max_chars = max(1, int(cell_w / 7))  # ~7 pixels per char at size 9
        if len(piece) > max_chars:
            piece = piece[:max_chars-1] + "…"

        # Highlight if this is the inspected token
        is_highlighted = (inspector.token_idx == t_idx)
        color = AMBER if is_highlighted else (180, 180, 200)

        font.render_to(screen, (x + 2, y_pos + 3), piece, color, size=9)


def discover_sessions(base_dir: Path) -> dict[int, str]:
    """Auto-discover available session stems in the directory."""
    found = {}
    key = 1
    for meta_file in sorted(base_dir.glob("*_metadata.json")):
        stem = meta_file.name.replace("_metadata.json", "")
        if stem in SKIP_STEMS:
            continue
        has_act = (base_dir / f"{stem}_activations.npz").exists()
        has_ids = (base_dir / f"{stem}_input_ids.npy").exists()
        if has_act and has_ids:
            found[key] = stem
            key += 1
            if key > 9:
                break
    return found


def _show_loading(screen: pygame.Surface, font: pygame.freetype.Font, msg: str) -> None:
    screen.fill(BG)
    sw, sh = screen.get_size()
    font.render_to(screen, (sw // 2 - 100, sh // 2), msg, TEXT_COL, size=16)
    pygame.display.flip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Phosphenes — LLM activation visualizer")
    parser.add_argument("--stem", default="Dream_greedy_clean")
    parser.add_argument("--fps", type=int, default=TARGET_FPS)
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args()

    pygame.init()
    pygame.freetype.init()

    flags = pygame.RESIZABLE
    if args.fullscreen:
        flags |= pygame.FULLSCREEN

    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H), flags)
    pygame.display.set_caption("Phosphenes")
    clock = pygame.time.Clock()

    font = _load_font()

    base_dir = Path(__file__).resolve().parent / "data"
    if not base_dir.exists():
        base_dir = Path(__file__).resolve().parent  # fallback to script dir

    # Auto-discover sessions, falling back to hard-coded registry
    registry = discover_sessions(base_dir)
    if not registry:
        registry = {k: v for k, v in MODEL_REGISTRY.items()
                    if (base_dir / f"{v}_activations.npz").exists()}

    # Check stem exists
    stem = args.stem
    act_file = base_dir / f"{stem}_activations.npz"
    if not act_file.exists():
        print(f"[phosphenes] ERROR: {act_file} not found.")
        print(f"[phosphenes] Available stems: {list(registry.values())}")
        return 1

    # Find key for initial stem
    initial_key = 1
    for k, v in registry.items():
        if v == stem:
            initial_key = k
            break

    app = AppState(current_key=initial_key)
    rstate = RenderState()
    inspector = InspectorState()
    text_state = TextState()
    text_panel = TextPanelState()
    params = EffectParams()

    # Initialize tutorial
    tutorial = TutorialState()
    tutorial.pages = load_tutorial_pages()
    tutorial.n_pages = len(tutorial.pages)
    if not tutorial.pages:
        tutorial.active = False  # no intro files found, skip

    if args.fullscreen:
        app.is_fullscreen = True

    _show_loading(screen, font, f"Loading {DISPLAY_NAMES.get(stem, stem)}...")
    data = load_model_data(base_dir, stem)

    # Build text panel wrapped lines
    if text_panel.enabled:
        text_panel.wrapped_lines = _build_text_panel_lines(data, font, text_panel.panel_width, text_panel.font_size)
        text_panel._cached_for_stem = data.stem

    # ── Main loop ──
    while app.running:
        dt = clock.tick(args.fps) / 1000.0

        # ── Events ──
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                app.running = False

            # Tutorial input handling (consumes events when active)
            elif tutorial.active:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:
                        if tutorial.current_page < tutorial.n_pages - 1:
                            tutorial.current_page += 1
                    elif event.key == pygame.K_LEFT:
                        if tutorial.current_page > 0:
                            tutorial.current_page -= 1
                    elif event.key == pygame.K_SPACE:
                        tutorial.active = False
                    elif event.key == pygame.K_ESCAPE:
                        tutorial.active = False
                    elif pygame.K_1 <= event.key <= pygame.K_8:
                        # Jump directly to session
                        new_key = event.key - pygame.K_0
                        if new_key in registry:
                            tutorial.active = False
                            new_stem = registry[new_key]
                            new_path = base_dir / f"{new_stem}_activations.npz"
                            if new_path.exists():
                                _show_loading(screen, font, f"Loading {DISPLAY_NAMES.get(new_stem, new_stem)}...")
                                data = load_model_data(base_dir, new_stem)
                                app.current_key = new_key
                                rstate = RenderState()
                                # Reset color basis
                                clear_color_basis(app)
                                # Rebuild text panel
                                if text_panel.enabled:
                                    text_panel.wrapped_lines = _build_text_panel_lines(data, font, text_panel.panel_width, text_panel.font_size)
                                    text_panel._cached_for_stem = data.stem
                continue  # skip normal event handling while tutorial is active

            elif event.type == pygame.KEYDOWN:
                key = event.key

                if key in (pygame.K_q, pygame.K_ESCAPE):
                    app.running = False

                elif key == pygame.K_SPACE:
                    app.paused = not app.paused

                elif key == pygame.K_LEFT and app.paused:
                    rstate.token_cursor = max(0, rstate.token_cursor - 1)

                elif key == pygame.K_RIGHT and app.paused:
                    rstate.token_cursor = min(data.n_tokens - 1, rstate.token_cursor + 1)

                elif key == pygame.K_UP:
                    app.playback_speed = min(16.0, app.playback_speed * 1.5)

                elif key == pygame.K_DOWN:
                    app.playback_speed = max(0.1, app.playback_speed / 1.5)

                elif key == pygame.K_TAB:
                    text_panel.enabled = not text_panel.enabled

                elif key == pygame.K_i:
                    inspector.enabled = not inspector.enabled

                elif key == pygame.K_m:
                    app.highlight_idx = (app.highlight_idx + 1) % len(HIGHLIGHT_MODES)

                elif key == pygame.K_t:
                    app.show_turn_markers = not app.show_turn_markers

                elif key == pygame.K_p:
                    # Toggle reference mode off, or turn it on at cursor position
                    if app.ref_mode:
                        app.ref_mode = False
                        app.ref_distances = None
                        app.ref_token = None
                        app.ref_layer = None
                        print("[phosphenes] Reference mode OFF")
                    else:
                        # Set reference at center of view, middle layer
                        ref_t = int(rstate.token_cursor)
                        ref_l = data.n_layers // 2
                        set_reference_point(app, data, ref_t, ref_l)

                elif key == pygame.K_c:
                    # Toggle color basis mode
                    if app.color_basis_mode or app.color_basis_selecting:
                        clear_color_basis(app)
                    else:
                        # Start color basis selection
                        app.color_basis_selecting = True
                        app.color_basis_current_idx = 0
                        app.color_basis_groups = [ColorBasisGroup()]
                        app.color_basis_guidance = None
                        app.ref_mode = False
                        print("[phosphenes] Color basis: selecting R. Click=source, Shift+Click=contrast, Enter=confirm")

                elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    if app.color_basis_selecting:
                        advance_color_basis(app, data)

                elif key == pygame.K_BACKSPACE:
                    if app.color_basis_selecting:
                        undo_color_basis(app)

                elif key == pygame.K_LEFTBRACKET and app.paused and data.has_turns:
                    cur_t = int(rstate.token_cursor)
                    for tb in reversed(data.turn_boundaries):
                        if tb.token_start < cur_t:
                            rstate.token_cursor = float(tb.token_start)
                            break

                elif key == pygame.K_RIGHTBRACKET and app.paused and data.has_turns:
                    cur_t = int(rstate.token_cursor)
                    for tb in data.turn_boundaries:
                        if tb.token_start > cur_t:
                            rstate.token_cursor = float(tb.token_start)
                            break

                elif key == pygame.K_r:
                    app.recording = not app.recording
                    if app.recording:
                        app.frame_count = 0
                        print("[phosphenes] Recording started — frames saved to ./frames/")
                    else:
                        print(f"[phosphenes] Recording stopped: {app.frame_count} frames")

                elif key == pygame.K_f:
                    app.is_fullscreen = not app.is_fullscreen
                    if app.is_fullscreen:
                        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                    else:
                        screen = pygame.display.set_mode((WINDOW_W, WINDOW_H), pygame.RESIZABLE)

                elif key == pygame.K_h:
                    tutorial.active = True
                    tutorial.current_page = 0

                elif pygame.K_1 <= key <= pygame.K_9:
                    new_key = key - pygame.K_1 + 1
                    if new_key in registry and new_key != app.current_key:
                        new_stem = registry[new_key]
                        new_path = base_dir / f"{new_stem}_activations.npz"
                        if new_path.exists():
                            _show_loading(screen, font, f"Loading {DISPLAY_NAMES.get(new_stem, new_stem)}...")
                            data = load_model_data(base_dir, new_stem)
                            app.current_key = new_key
                            rstate.token_cursor = 0.0
                            rstate.turbulence_phase = 0.0
                            # Clear cache and special modes when switching models
                            rstate._cached_t_end = -1
                            rstate._cached_base_rgb = None
                            rstate._cached_mode_hash = 0
                            app.ref_mode = False
                            app.ref_distances = None
                            app.ref_token = None
                            app.ref_layer = None
                            clear_color_basis(app)
                            # Rebuild text panel lines for new model
                            if text_panel.enabled:
                                text_panel.wrapped_lines = _build_text_panel_lines(data, font, text_panel.panel_width, text_panel.font_size)
                                text_panel._cached_for_stem = data.stem

            elif event.type == pygame.VIDEORESIZE:
                if not app.is_fullscreen:
                    screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    # Check if clicked on token strip (when paused)
                    if app.paused:
                        strip_tok = strip_click_to_token(data, rstate, event.pos)
                        if strip_tok is not None:
                            # Update inspector to this token
                            inspector.token_idx = strip_tok
                            inspector.layer_idx = data.n_layers // 2  # Middle layer
                            inspector.enabled = True
                            continue  # Don't process as canvas click

                    tok, layer = screen_to_token_layer(data, rstate, event.pos, 52)
                    if tok is not None and layer is not None:
                        if app.color_basis_selecting:
                            mods = pygame.key.get_mods()
                            is_shift = bool(mods & pygame.KMOD_SHIFT)
                            handle_color_basis_click(app, data, tok, layer, is_shift)
                        else:
                            # Normal reference point mode
                            set_reference_point(app, data, tok, layer)

        # ── Update ──
        # Shimmer always animates (even when paused)
        rstate.turbulence_phase += params.turbulence_speed * dt
        rstate.grain_phase += params.tokens_per_second * app.playback_speed * 0.7 * dt

        if not app.paused:
            rstate.token_cursor += params.tokens_per_second * app.playback_speed * dt
            if rstate.token_cursor >= data.n_tokens:
                rstate.token_cursor = 0.0

        # Update inspector
        if inspector.enabled:
            update_inspector(inspector, data, rstate, pygame.mouse.get_pos(), 52)

        # ── Render ──
        # Tutorial overlay (when active, skip normal visualization)
        if tutorial.active:
            render_tutorial(screen, font, tutorial)
            pygame.display.flip()
            clock.tick(30)  # don't need 60fps for static tutorial
            continue

        sw, sh = screen.get_size()
        header_h = 52
        footer_h = 22
        text_h = 42 if text_state.enabled else 0
        canvas_h = max(100, sh - header_h - footer_h - text_h - 10)
        text_panel_w = text_panel.panel_width if text_panel.enabled else 0
        canvas_w = max(100, sw - 40 - text_panel_w)

        # Determine if any special mode is active
        special_mode = app.ref_mode or app.color_basis_mode or app.color_basis_selecting

        scaled_region_uint8, x_off, y_off, new_w, new_h = render_frame(
            data, rstate, params,
            canvas_w, canvas_h,
            highlight_mode=HIGHLIGHT_MODES[app.highlight_idx] if not special_mode else None,
            ref_mode=app.ref_mode,
            ref_distances=app.ref_distances,
            ref_token=app.ref_token,
            ref_layer=app.ref_layer,
            color_basis_mode=app.color_basis_mode,
            color_basis_result=app.color_basis_result,
            color_basis_groups=app.color_basis_groups,
            color_basis_selecting=app.color_basis_selecting,
            color_basis_guidance=app.color_basis_guidance,
            color_basis_current_idx=app.color_basis_current_idx,
            color_basis_version=app.color_basis_version,
        )

        # Blit frame to screen
        screen.fill(BG)

        # Use frombuffer to avoid transpose - it accepts (H, W, 3) byte data directly
        region_bytes = np.ascontiguousarray(scaled_region_uint8)
        surf = pygame.image.frombuffer(region_bytes.tobytes(), (new_w, new_h), 'RGB')
        screen.blit(surf, (20 + x_off, header_h + y_off))

        # Set canvas offsets for inspector (absolute positions, not incremental)
        rstate.canvas_x = 20 + x_off
        rstate.canvas_y = y_off  # y_off is relative to canvas area, header_h added in inspector

        # Overlays
        _render_header(screen, font, data, app, rstate, params)

        if text_state.enabled:
            render_text_ticker(screen, font, data, rstate.token_cursor, text_state,
                               sh - footer_h - text_h - 4)

        if data.has_turns:
            render_role_bar(screen, data, rstate, header_h, new_w, new_h)

        if app.show_turn_markers and data.has_turns:
            render_turn_markers(screen, font, data, rstate, header_h)

        # Color basis selection markers
        if app.color_basis_selecting or app.color_basis_mode:
            render_color_basis_overlay(screen, font, app, data, rstate, header_h, new_w, new_h)

        # Right margin text panel
        if text_panel.enabled:
            panel_x = 20 + canvas_w + 10  # 10px gap after visual
            render_text_panel(screen, font, data, rstate, text_panel, panel_x, canvas_h, header_h)

        # Token strip when paused (below canvas)
        if app.paused:
            strip_y = header_h + canvas_h + 4
            render_token_strip(screen, font, data, rstate, inspector, rstate.canvas_x, rstate.canvas_w, strip_y)

        render_inspector(screen, font, inspector, data, rstate, header_h)
        _render_controls(screen, font, data.has_turns, app.ref_mode, app.color_basis_mode)

        pygame.display.flip()

        # Recording
        if app.recording:
            frame_dir = base_dir / "frames"
            frame_dir.mkdir(exist_ok=True)
            pygame.image.save(screen, str(frame_dir / f"frame_{app.frame_count:06d}.png"))
            app.frame_count += 1

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
