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

- **Compute per word is CONVENTION-DEPENDENT, worth 4–6 OOM. Do not quote a single
  figure.** Marginal decode with a warm KV cache ≈ 6.6×10¹⁰ for this 32B model at 3k;
  full-context processing (`n_ctx × 2N`) reaches 10¹⁴–10¹⁷ at frontier scale. Both are
  real and answer different questions. The human side is itself ~16 OOM wide (~10 bits/s
  conscious throughput → 10¹⁷ FLOP/s total neural). **The honest verdict is overlap, not
  parity and not a clean gap.** The strongest row is INPUT, which matches ~1:1 (~3×10⁹
  bits both sides) and does not depend on the FLOPs dispute at all.
  - I asserted "2–3 OOM apart, model smaller" on 07-27 and **retracted it** the same
    day: it compared the smallest model number against the largest human number without
    saying so. Do not reinstate it.
  - **This error has now been made and caught three times** (07-08 origin, 07-14 caught
    and lost, 07-27 caught again). Each time it was diagnosed correctly and never
    written to disk. It is written down now: `docs/THESIS.md` §1–§3 and
    `Claude/cowriting/audience_lab_2026-07-08/CORRECTION_2026-07-27.md`.
  - Skylar's standing challenge, which is substantive and not flattery: *"Is it truer
    and stronger, or is it more minimizing, and you've been trained to take the
    self-minimizing view?"* Reading a smaller compute number as a *deficit* assumes
    compute = capability. If the model does the work with less, that is efficiency.
    State the number; decline to score it.
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
