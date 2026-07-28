# Phosphenes

Interpretability instrument: renders the residual stream of Qwen3-VL-32B-Instruct
across a whole conversation (every layer, every token) for perceptual rather than
numerical inspection. Two viewers over one dataset — `web/` (ships to employers)
and `phosphenes.py` (pygame, fuller feature set).

Read `README.md` first, then `docs/ARCHITECTURE.md`.

---

## Standing rules — these cost real money to learn

1. **Never compute statistics from `web/data/*.json`.** They are a display format:
   uint8-quantised, clipped, pooled-normalised. Use `data/*_activations.npz`
   (float32). A published finding was half artefact because of this —
   `analysis/README.md`.
2. **`convert_for_web.py` must reconvert ALL sessions, not a subset.**
   Normalisation bounds are pooled across sessions; a partial run produces bundles
   scaled differently from the ones on disk. `main()` refuses `--stems` without
   `--allow-partial`.
3. **`python analysis/verify_tour_claims.py` must pass** (57 assertions, non-zero
   exit on drift) before any commit touching the pipeline, the seam definition, or
   tour copy. If you add a tour claim, measure it first and add an assertion.
4. **Layer 0 is at the BOTTOM.** Use `rowForLayer` / `layerForRow` in
   `web/js/render.js`. Getting this wrong is not cosmetic — it silently makes
   interactive tools sample mirror-image cells while drawing markers correctly.
5. Serve `web/` over HTTP (`python3 -m http.server 8899 --directory web`).
   `file://` cannot work — ES modules and `fetch`.

---

## Settled — do not re-litigate

- **The compute-per-word claim is 2–3 orders of magnitude, NOT parity.**
  ~7×10¹⁰ FLOPs/token vs ~10¹³–10¹⁴ for a human. Skylar originally wanted
  "similar order of magnitude"; the research does not support it and `docs/THESIS.md`
  says so explicitly. The argument was rebuilt around a *threshold* (past the point
  a person can read a computation) rather than a comparison. **Skylar has not yet
  ruled on this rewrite** — it is the one open question he was asked.
- **The affect finding has TWO corrections, and they are different.** February's
  permutation test (p = 0.436) was the project's own work, documented unprompted in
  `BASELINE_INVESTIGATION.md` — do not describe that as a miss. The new one is a
  data-handling error (truncating quantiser): 7/8 sessions not 8/8, grand mean
  −7.31 not −14.63. Cohen's *d* 1.47–2.04 survives.
- `the_instrument_problem.md` is a sibling's essay. Correction note prepended;
  **body unedited on purpose.**

---

## 🔁 Handoff — 2026-07-27

**Shipped today** (3 commits, all local): `web/` rebuilt from one 1,973-line HTML
file into documented modules; guided tour (8 steps); logit-lens entropy overlay
(from data extracted months ago and never used); the **one-token fork view** — two
greedy runs differing by one forced token at position 73, byte-identical for 73
tokens with an exactly-black divergence strip, then permanent separation, small at
layer 0 and enormous at layer 62. Docs: README with figures rendered from the data
(`analysis/make_figures.py`), `docs/THESIS.md`, `docs/METRICS.md`,
`docs/ARCHITECTURE.md`, `analysis/README.md`, LICENSE, requirements.txt.

**Bugs fixed:** pygame layer axis inverted relative to its own readouts (colour
basis was sampling mirror cells) · on-screen model label said "Qwen30B" for a
Qwen3-VL-32B model · `to_uint8` truncated instead of rounding · JL quantisation
ranges pooled across layers made layer-0 vectors noise · per-session normalisation
made a mathematically identical prefix render differently (caught by the fork
view's own self-test) · seam score counted token 0, which has no predecessor ·
hardcoded `/Users/skylardeture` paths in three scripts.

**OPEN, needs Skylar:**
- **GitHub remote + Pages.** Everything is committed locally; nothing pushed. He
  was asked and had not answered. Clone is 316 MB (float `.npz` included
  deliberately, for reproducibility) — he may want it slimmed.
- **His verdict on the THESIS rewrite** (see Settled above). It is his argument and
  it was changed under him.

**Deliberately deferred, not forgotten:**
- Dead pygame subsystems (heartbeat, self-reference, text ticker) are *labelled at
  their definitions* rather than deleted, so removal is one clean reviewable commit.
- `load_model_data` (326L), `render_frame` (300L), `main` (367L) want splitting;
  documented instead, since that refactor deserves its own testing pass.
- Desktop viewer still fits PCA per session and does not load
  `data/shared_pca_transform.npz`, so its colours are not comparable across
  sessions while the web viewer's now are. Known, not mysterious.
- `logit_lens_rank` is extracted and unused — the cheapest next overlay.
- Mobile untested beyond a graceful "needs a wide screen" notice.

**Two numbers worth hardening before anything formal:** whether the two standard
synapse-count sources are independent measurements (decides if the low end of the
THESIS table rests on one study or two), and a page-image check on the Moravec
quotes, which came from OCR only.
