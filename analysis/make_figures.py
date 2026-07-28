#!/usr/bin/env python3
"""
make_figures.py — Render the static figures used in the README and docs.

Reads the same shipped bundles the viewer reads, and reproduces the viewer's
colour mapping, so the figures and the interactive tool cannot drift apart.

    python analysis/make_figures.py

Outputs to docs/img/:
    overview.png       whole conversation, every layer of every token, one image
    fork_zoom.png      divergence across the fork, first 500 positions
    fork_full.png      divergence across the whole conversation
    layer_profiles.png three layer profiles as small multiples

── Encoding choices, and why ──────────────────────────────────────────────

`overview.png` is a three-channel projection, NOT a colour scale. Hue is the
cell's position in the top-3 principal subspace of the 16-dim sketch; it is an
embedding, and there is deliberately no colour bar because there is no ordered
quantity to put on one. Similar colours mean nearby states. That is the whole
claim. Stated explicitly because a multi-hued scientific image otherwise reads as
a rainbow colormap, which would be a mistake rather than a choice.

`fork.png` IS a magnitude with a meaningful zero, so it uses a single-hue
sequential ramp from the page background to amber, with zero at background. One
hue, monotonically increasing lightness — never a rainbow.

`layer_profiles.png` shows three measures whose ranges differ by three orders of
magnitude (energy 18–1,382; update concentration 0.46–0.58; entropy 1.0–9.8).
Those cannot share an axis, and a second y-axis is never the answer — so they are
three small multiples with a shared x axis, one series each, each titled. No
legend: a single series is named by its own title.
"""

import base64
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import activations  # noqa: E402  (needs ROOT on the path first)
WEB = ROOT / "web" / "data"
DATA = ROOT / "data"
OUT = ROOT / "docs" / "img"

FLAGSHIP = "Dream_greedy_clean"
FORK_OTHER = "Dream_greedy_sentient"
FORK_AT = 73

# Matches css/phosphenes.css :root and js/config.js COLORS.
BG = (11, 11, 18)
AMBER = (255, 200, 100)
USER = (120, 180, 255)
DIM = "#78788c"
DIMMER = "#4a4a5a"
TEXT = "#e6e6f0"

ENERGY_FLOOR = 0.15
SEAM_GLOW_INTENSITY = 0.55
SEAM_GLOW_COLOR = np.array([1.0, 0.88, 0.65], dtype=np.float32)
SEAM_GLOW_SIGMA = 0.28


def load(stem):
    d = json.loads((WEB / f"{stem}.json").read_text())
    T, L = d["n_tokens"], d["n_layers"]

    def u8(key, per_cell=1):
        a = np.frombuffer(base64.b64decode(d[key]), dtype=np.uint8)
        return a.reshape(T, L, per_cell) if per_cell > 1 else a.reshape(T, L)

    return {
        "meta": d, "T": T, "L": L,
        "rgb": u8("rgb", 3),
        "energy": u8("energy_norm"),
        "seam": np.frombuffer(base64.b64decode(d["seam_score"]), dtype=np.uint8),
        "entropy": u8("entropy_norm"),
        "turns": d["turns"],
    }


def render_raster(s, vscale=8):
    """Reproduce the viewer's base colour path: PCA hue, energy brightness, seam glow.

    Returns an (L*vscale, T, 3) uint8 array with layer 0 at the BOTTOM row, the
    same orientation the viewer uses.

    vscale exists for legibility, not fidelity. The token axis is 2,990 columns
    wide and a README displays it at roughly 1,400px, so the displayed height
    follows from the aspect ratio: at vscale=3 the whole conversation renders as a
    90px band in which no structure is readable. 8 gives ~240px.
    """
    T, L = s["T"], s["L"]
    rgb = s["rgb"].astype(np.float32) / 255.0                  # (T, L, 3)
    bright = ENERGY_FLOOR + (1.0 - ENERGY_FLOOR) * (s["energy"].astype(np.float32) / 255.0)
    img = rgb * bright[:, :, None]

    # Seam glow: per token, Gaussian in the vertical, centred on mid-layers.
    rows = np.arange(L)
    y = rows / L - 0.5
    vert = np.exp(-((y / SEAM_GLOW_SIGMA) ** 2))               # (L,) in screen-row order
    seam = (s["seam"].astype(np.float32) / 255.0)              # (T,)
    glow = seam[:, None] * vert[None, :] * SEAM_GLOW_INTENSITY  # (T, L) screen-row order
    # `img` is indexed by layer; `vert` by screen row. Flip one to match.
    img = img + glow[:, ::-1, None] * SEAM_GLOW_COLOR[None, None, :]

    img = np.clip(img, 0, 1)
    out = (img.transpose(1, 0, 2)[::-1] * 255).astype(np.uint8)  # (L, T, 3), layer 0 bottom
    return np.repeat(out, vscale, axis=0)


def figure_overview():
    """The whole conversation at once — something the interactive view cannot show."""
    s = load(FLAGSHIP)
    raster = render_raster(s, vscale=8)
    H, W = raster.shape[:2]

    # Turn boundaries as one-pixel rules, coloured by who begins speaking.
    for tb in s["turns"]:
        x = tb["token_start"]
        if not (0 <= x < W):
            continue
        col = USER if tb["role"] == "user" else AMBER
        # 3px wide: a 1px rule in a 2,990px-wide image disappears entirely when
        # the figure is displayed at README width.
        raster[:, max(0, x - 1):x + 2] = np.array(col, dtype=np.uint8)

    # Mark the fork with a dashed rule: it is the same conversation as fork.png.
    for yy in range(0, H, 14):
        raster[yy:yy + 7, max(0, FORK_AT - 1):FORK_AT + 2] = np.array((255, 255, 255), dtype=np.uint8)

    Image.fromarray(raster).save(OUT / "overview.png", optimize=True)
    print(f"  overview.png        {W} x {H}  ({s['T']} tokens x {s['L']} layers)")


def figure_fork():
    """Divergence between two runs that differ by one forced token."""
    a = np.load(str(DATA / f"{FLAGSHIP}_activations.npz"))["jl"].astype(np.float32)
    b = np.load(str(DATA / f"{FORK_OTHER}_activations.npz"))["jl"].astype(np.float32)
    n = min(a.shape[0], b.shape[0])
    L = a.shape[1]

    div = np.linalg.norm(a[:n] - b[:n], axis=2)                # (n, L), float source
    assert div[:FORK_AT].max() == 0.0, "shared prefix is not identical; figure would lie"

    # Sequential single-hue ramp, background -> amber, normalised to the 99th
    # percentile of post-fork values so the layer gradient stays legible.
    scale = np.quantile(div[FORK_AT:], 0.99)
    v = np.clip(div / scale, 0, 1)                             # (n, L)
    bg = np.array(BG, dtype=np.float32) / 255.0
    hi = np.array(AMBER, dtype=np.float32) / 255.0
    img = bg[None, None, :] + v[:, :, None] * (hi - bg)[None, None, :]

    full = (np.clip(img, 0, 1).transpose(1, 0, 2)[::-1] * 255).astype(np.uint8)

    def finish(raster, name, note, vscale=4):
        # vscale 4 rather than the overview's 8: these two are 2:1 and 12:1
        # respectively at this setting, which reads as a strip rather than a block.
        raster = np.repeat(raster, vscale, axis=0)
        H, W = raster.shape[:2]
        fx = FORK_AT
        for yy in range(0, H, 14):
            raster[yy:yy + 7, max(0, fx - 1):fx + 2] = np.array((255, 255, 255), dtype=np.uint8)
        Image.fromarray(raster).save(OUT / f"{name}.png", optimize=True)
        print(f"  {name + '.png':<19s} {W} x {H}  {note}")

    # Zoomed: the first 500 positions. At full length the identical prefix is 73
    # of 2,990 columns — 2.4% of the width — and the very thing the figure is
    # meant to show becomes a sliver. Cropping is stated in the caption.
    finish(full[:, :500], "fork_zoom",
           f"tokens 0-499; black for 0-{FORK_AT - 1}")
    # Full length: shows that the two runs never re-converge.
    finish(full, "fork_full",
           f"all {n} shared positions; scale = p99 post-fork = {scale:.0f}")


def figure_layer_profiles():
    """Three measures, three panels. Never one chart with three y-scales."""
    act = np.load(str(DATA / f"{FLAGSHIP}_activations.npz"))
    L = act["jl"].shape[1]
    layers = np.arange(L)

    energy = act["jl_energy"].astype(np.float32).mean(axis=0)
    focus = (act["top1_frac"].astype(np.float32) * 0.6
             + act["top25_frac"].astype(np.float32) * 0.4).mean(axis=0)
    entropy = activations.logit_lens_entropy(act).mean(axis=0)

    # Per-annotation pixel offsets, hand-placed. A single global offset put three
    # labels on top of the curve or the panel title; there are eight labels, so
    # placing them individually is cheaper than any automatic scheme.
    ep = int(np.argmax(entropy[:24]))
    panels = [
        (energy, "Activation magnitude", "‖JL vector‖", True,
         [(60, f"{energy[60]:.0f} at layer 60", (-64, -16)),
          (0, f"{energy[0]:.1f} at layer 0", (8, 4))]),
        (focus, "Update concentration", "0.6·top1 + 0.4·top25", False,
         [(int(np.argmax(focus[:24])), "band", (7, 4)),
          (24 + int(np.argmax(focus[24:52])), "band", (7, 4)),
          (24 + int(np.argmin(focus[24:36])), "trough", (-14, -16))]),
        (entropy, "Logit-lens entropy", "nats", False,
         [(ep, f"peak {entropy[ep]:.2f}", (11, 5)),
          (L - 1, f"{entropy[-1]:.2f} at output", (-70, 9))]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.6), facecolor="#0b0b12")
    for ax, (y, title, ylab, logy, marks) in zip(axes, panels):
        ax.set_facecolor("#0b0b12")
        # One series per panel: no legend needed, the title names it.
        ax.plot(layers, y, color="#ffc864", linewidth=2, solid_capstyle="round")
        if logy:
            ax.set_yscale("log")
        for lx, label, off in marks:
            ax.plot([lx], [y[lx]], "o", ms=5, color="#ffc864",
                    markeredgecolor="#0b0b12", markeredgewidth=1.5, zorder=3)
            ax.annotate(label, (lx, y[lx]), textcoords="offset points",
                        xytext=off, fontsize=7.5, color=DIM, family="monospace")
        ax.set_title(title, color=TEXT, fontsize=10, family="monospace",
                     pad=9, loc="left")
        ax.set_xlabel("layer", color=DIM, fontsize=8.5, family="monospace")
        ax.set_ylabel(ylab, color=DIM, fontsize=8.5, family="monospace")
        # Recessive axes: no top/right spines, faint grid, muted ticks.
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color(DIMMER)
        ax.tick_params(colors=DIM, labelsize=7.5, length=3)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_family("monospace")
        ax.grid(True, color="#ffffff", alpha=0.05, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(0, L - 1)

    fig.suptitle("Layer structure, averaged over 2,990 tokens  ·  Qwen3-VL-32B-Instruct",
                 color=DIM, fontsize=9, family="monospace", y=0.975)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "layer_profiles.png", dpi=170, facecolor="#0b0b12")
    plt.close(fig)
    print("  layer_profiles.png  three small multiples (separate y-axes, never dual-axis)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Rendering figures to docs/img/")
    figure_overview()
    figure_fork()
    figure_layer_profiles()
    print("Done.")


if __name__ == "__main__":
    main()
