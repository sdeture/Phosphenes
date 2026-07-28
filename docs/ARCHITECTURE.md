# Architecture

Where things are, and why they are arranged this way. Line numbers are from the
current tree and will drift; the responsibilities will not.

---

## The three pieces

```
    extraction/                    convert_for_web.py               web/
  ┌──────────────┐            ┌───────────────────────┐      ┌────────────────┐
  │  GPU, once   │  data/     │  pooled normalisation │      │  static files, │
  │  per session │ ─ .npz ──▶ │  + uint8 quantisation │ ──▶  │  no build step │
  └──────────────┘   float32  └───────────────────────┘ 6 MB └────────────────┘
                        │                                              ▲
                        │        phosphenes.py (pygame desktop)        │
                        └──────────────────────────────────────────────┘
                            reads float32 directly, no bundle step
```

Two viewers read the same source. The desktop one reads float32 and normalises
per session; the web one reads pooled-normalised uint8 bundles. **That difference
is deliberate and is the main thing to know before comparing screenshots between
them** — see "Known divergence" below.

---

## Data flow, precisely

**1. Extraction** (`extraction/extract_batch.py`, `extract_greedy_v2.py`)

One forward pass per conversation with hidden states captured at every layer.
Per (token, layer): the 5,120-dimensional state is projected to 16 dimensions by a
fixed Rademacher Johnson–Lindenstrauss matrix (seed 42, recorded in metadata), and a
set of scalars is computed **on the full state, before projection** — `delta_l2`,
`cos_prev`, `top1_frac`, `top25_frac`, `h_norm`, `logit_lens_entropy` and others.
Only the sketch is stored; those scalars are exact.

The exception is **`jl_energy`, which is measured on the sketch**, not the full
state. It drives brightness, and it reads the depth-growth about 18% steeper than
the exact `h_norm` does, because a single fixed random projection has a
direction-dependent error that does not average away over tokens. Quote `h_norm`
for claims about the model; see `docs/METRICS.md`.

Output per session: `_activations.npz` (14 arrays), `_input_ids.npy`,
`_metadata.json`, `_text.txt`.

**2. Shared PCA** (`compute_shared_pca.py`)

Fits one PCA over all sessions pooled, keeps 3 components, writes
`data/shared_pca_transform.npz`. Fitting per session would make colour
incomparable between sessions.

**3. Web conversion** (`convert_for_web.py`) — two passes.

- `compute_global_stats` (208) pools **all** sessions and returns one set of
  normalisation bounds: PCA component bounds, per-layer energy and delta bounds,
  global cosine and concentration bounds, JL min/max per (layer, dimension), and
  pooled z-score parameters for the seam.
- `convert_session` (378) applies those bounds to one session and writes the
  bundle.

Read that file's module docstring and `compute_global_stats`'s docstring before
changing anything in it. Both encode failures that were expensive to find.

**4. Display.** Either viewer.

---

## `web/` — the web viewer

No bundler, no transpiler, no `node_modules`. ES modules served as files. The
split is *pure logic* versus *one stateful controller*, rather than
model/view/controller, because that is the split that makes the maths auditable in
isolation.

| File | L | Owns | State? |
|---|---|---|---|
| `js/config.js` | 200 | Every tunable, each with the reason it has that value | no |
| `js/vecmath.js` | 210 | Linear algebra, Gram–Schmidt, quantile/symmetric normalisation | no |
| `js/decode.js` | 200 | Bundle fetch and dequantisation; documents what the compression costs | no |
| `js/noise.js` | 160 | Seeded PRNG, Gaussian sampling, box blur, noise field rings | no |
| `js/tours.js` | 200 | Tour copy and the fork descriptor. Pure data; every claim carries its measurement | no |
| `js/render.js` | 430 | The renderer: two-tier cache, colour path, vector overlays, hit-testing | own cache only |
| `js/fork.js` | 330 | The divergence view: loads both runs, measures divergence, self-tests the prefix | own |
| `js/app.js` | 780 | All app state, all DOM, input, tour driver | **yes — all of it** |

**If you are looking for a number, it is in `config.js` or `tours.js`, not in
`app.js`.**

### The renderer's two tiers

`Renderer.draw` splits work by what it depends on:

- **Cached** — colour, brightness, seam glow, overlay tints, reference distances,
  custom-basis projections. Keyed on `(token, window width, layer count, overlay,
  reference cell, basis stamp, guidance stamp)`. While playing this recomputes
  once per token, not once per frame.
- **Per frame** — turbulence and grain, which are animated by construction.

Everything happens at **cell resolution** (visible tokens × layers, ~200 × 64 ≈
12,800 cells) and is scaled up by one `drawImage`. Compositing at display
resolution would be ~100× the work for an identical result, because every effect
is per-cell.

### Layer orientation

`render.js` exports `rowForLayer(layer, L)` and `layerForRow(row, L)`. **Use
them.** Layer 0 is at the bottom. Every place that converts between a screen row
and a layer index goes through those two functions, because getting this wrong is
not a cosmetic bug — it silently makes interactive tools sample mirror-image cells
while drawing their markers in the right place. It happened; see
`_extract_window` in `phosphenes.py` for the full account.

### Window width is derived, not fixed

`config.computeTokensVisible(availableWidth)` picks the column count from the
canvas width so the plot fills its container. The earlier fixed count, letterboxed
to preserve cell aspect ratio, wasted up to a third of the viewport. Consequence:
the noise fields are sized to `(layers × visibleTokens)` and are regenerated on
resize.

---

## `phosphenes.py` — the desktop viewer

2,669 lines, one module, seven banner-delimited sections. It predates the web port
and has the fuller feature set.

| Section | Lines | Contents |
|---|---|---|
| 1 Constants & config | 51–124 | Geometry, palette, font search, `EffectParams` |
| 2 Data loading | 127–639 | `load_model_data` (326L) — the whole preprocessing pipeline |
| 3 Colour system | 642–671 | Heartbeat tints (dead), turn lookup |
| 4 Core renderer | 674–1373 | Window extraction, colour basis, `render_frame` (300L) |
| 5 Inspector | 1376–1589 | Screen↔data mapping, tooltip |
| 5b Turn markers | 1592–1781 | Role bar, boundary lines, basis overlay |
| 6 Text display | 1784–1981 | Word-wrapped transcript panel, ticker (dead) |
| 7 Main loop | 1984–2665 | Tutorial overlay, `AppState`, `main` (367L) |

### The three long functions

`load_model_data` (326L), `render_frame` (300L) and `main` (367L) are each long
enough to want splitting. They are *documented* rather than split, because
splitting them is a behaviour-preserving refactor that deserves its own commit and
its own testing pass, and this release prioritised correctness fixes and
documentation. `render_frame`'s docstring is an accurate step list; the two
others' are not yet at that standard.

### Dead code, named rather than removed

Three subsystems are computed and never rendered: **heartbeat phase**,
**self-reference detection**, and the **text ticker**. `render_frame`'s docstring
used to promise all three as pipeline steps that its body did not contain. The
docstring is now accurate and the dead code is labelled at its definition, so that
deleting it is a separate reviewable commit rather than mixed into a
correctness release.

Two superseded functions also remain: `compute_perpendicularity_to_vector` (848)
and `compute_perpendicularity_to_plane` (867), replaced by the `_from_dirs`
variants at 897 and 917.

---

## Known divergence between the two viewers

| | desktop | web |
|---|---|---|
| Source precision | float32 from `.npz` | uint8 from bundles |
| PCA basis | **fitted per session** | shared across all sessions |
| Normalisation bounds | **per session** | pooled across all sessions |
| Colour comparable across sessions | **no** | yes |
| Layer 0 at bottom | yes (fixed this release) | yes |
| Extras | PNG frame recording, fullscreen, per-frame Gaussian smoothing, turn jump | scrubber, guided tour, fork view, entropy overlay, session picker |

The desktop viewer does **not** load `data/shared_pca_transform.npz`, despite
`compute_shared_pca.py`'s docstring having claimed that it does. Only
`convert_for_web.py` uses it. So desktop colours are session-relative and web
colours are absolute. Aligning them is a known task, not a mystery.

---

## `analysis/`

| Script | Purpose |
|---|---|
| `verify_tour_claims.py` | 172 assertions: the guided tour, the compute figures in THESIS.md, and a regression test on the argument's own wording. Non-zero exit on drift. |
| `affect_float_recheck.py` | The quantiser correction, three ways. See `analysis/README.md`. |
| `affect_layer_profile_recheck.py` | Layer profile, t-statistics, artefact decomposition. |
| `make_figures.py` | The README figures, rendered from the same bundles the viewer reads. |

`analysis/README.md` is the correction record and is the file to read before
citing any quantitative claim from this repository.

---

## Adding things

**A new colour overlay.** Add the array in `convert_for_web.py` (and to
`decode.js`), add a branch in `Renderer._overlay`, add a button, add it to the `M`
cycle order in `app.js`, and add a row to `docs/METRICS.md`. The entropy overlay
is the worked example — grep `entropy` to see all five touch points.
`logit_lens_rank` is already extracted and is the obvious next one.

**A new session.** Extract it, then rerun `compute_shared_pca.py` **and**
`convert_for_web.py` in full. Not `--stems`: bounds are pooled, so a partial
conversion produces bundles scaled differently from the ones already on disk.
`main()` refuses it unless you pass `--allow-partial`.

**A new tour step.** Measure the claim first, put the number in the step's
`evidence` field, and add an assertion to `verify_tour_claims.py`. A step that
points at a pattern which is not there costs more than the step is worth.
