# Phosphenes — working notes

Interpretability instrument: renders the residual stream of Qwen3-VL-32B-Instruct
across a whole conversation (every layer, every token) for perceptual rather than
numerical inspection. Two viewers over one dataset — `web/` (the one that ships)
and `phosphenes.py` (pygame, fuller feature set).

Read `README.md` first, then `docs/ARCHITECTURE.md`. Corpus origin and the
elicitation prompts are in `docs/PROVENANCE.md`; read that before describing the
data to anyone.

---

## Standing rules — these cost real money to learn

1. **Never compute statistics from `web/data/*.json`.** They are a display format:
   uint8-quantised, clipped, pooled-normalised. Use `data/*_activations.npz`.
   A published finding was half artefact because of this — `analysis/README.md`.
   Note the npz is not uniformly float32: **`jl` is float16** (9 arrays f32,
   4 int32). Adequate for everything published, but say "extraction source",
   not "float32 source".
2. **`convert_for_web.py` must reconvert ALL sessions, not a subset.**
   Normalisation bounds are pooled across sessions; a partial run produces bundles
   scaled differently from the ones on disk. `main()` refuses `--stems` without
   `--allow-partial`.
3. **`python analysis/verify_tour_claims.py` must pass** (non-zero exit on drift)
   before any commit touching the pipeline, the seam definition, the compute
   argument, or tour copy. If you add a claim anywhere reader-facing, measure it
   first and add an assertion.
4. **Layer 0 is at the BOTTOM.** Use `rowForLayer` / `layerForRow` in
   `web/js/render.js`. Getting this wrong is not cosmetic — it silently makes
   interactive tools sample mirror-image cells while drawing markers correctly.
5. **`activations.py` owns the layer axis. Read it before indexing that axis,
   and never read `logit_lens_entropy` out of the npz directly** — use
   `activations.logit_lens_entropy()`, which repairs the double-normalised top
   layer. Array layer 0 is the output of block 0 (the embedding is absent), and
   layer 63 is the post-final-norm state, ~10× smaller than layer 62 and not on
   the same scale as anything beneath it.
6. Serve `web/` over HTTP (`python3 -m http.server 8899 --directory web`).
   `file://` cannot work — ES modules and `fetch`.
7. **`web/about.html` is the SHORT version and should stay short** (~850 words).
   It states the idea — a model's per-word computation is in nervous-system
   territory, so use more of the nervous system to read it — and points at
   `docs/THESIS.md` for the arithmetic. Do not migrate the argument back into it.
   The webapp's job is a good tutorial plus a light explanation, not a paper.

---

## Settled — do not re-litigate

- **Compute per word is CONVENTION-DEPENDENT, worth 4–6 OOM. Do not quote a single
  figure.** Marginal decode with a warm KV cache ≈ 6.6×10¹⁰ for this 32B model at 3k;
  full-context processing (`n_ctx × 2N`) reaches 10¹⁴–10¹⁷ at frontier scale. Both are
  real and answer different questions. The human side is itself ~16 OOM wide (~10 bits/s
  conscious throughput → 10¹⁷ FLOP/s total neural). **The honest verdict is overlap, not
  parity and not a clean gap.** The strongest row is INPUT, which matches ~1:1 (~3×10⁹
  bits both sides) and does not depend on the FLOPs dispute at all.
  - The claim "2–3 OOM apart, model smaller" was asserted here and **retracted the
    same day**: it compared the smallest model number against the largest human
    number without saying so. Do not reinstate it. Both versions are in git history
    deliberately.
  - The standing challenge, which is substantive: *"Is it truer and stronger, or is
    it more minimizing?"* Reading a smaller compute number as a *deficit* assumes
    compute = capability. If the model does the work with less, that is efficiency.
    State the number, name the convention, decline to score it.
- **Final-layer entropy is 0.998 nats, not 1.65.** Retracted 2026-07-28: the
  extraction applied the model's final RMSNorm to a state HuggingFace had
  already normed, so layer 63 of `logit_lens_entropy` was the entropy of a
  double-normed state. Proven bit-exactly on Qwen3-0.6B — `lm_head(hidden_states[L])`
  reproduces the true logits to max |Δ| = 0.000000, and the second norm moves
  entropy by up to 4.1 nats. The repair substitutes `token_entropy`, which was
  in every npz all along and is exactly the right number. **The correction moves
  the figure DOWN — the model commits harder than was claimed — and rise-then-fall
  still holds 8/8.** Nothing else moved: seam is at layer 38, sparsity bands are
  searched over 0–51, energy growth is quoted at layers 0 and 60, and the fork's
  per-layer numbers at 0 and 62. Two verifier assertions now guard the repair in
  both directions, so it cannot rot into a no-op.
- **The 2026-07-28 definition audit found five more, all now fixed and asserted.**
  Every one was a doc-vs-code mismatch that reading alone had missed for months;
  each was caught by *testing the definition against the data*. In severity order:
  (a) **`jl_energy` is measured on the 16-dim sketch and reads the depth-growth
  78×, while the exact `h_norm` in the same file reads 65.6×** — the sketch/exact
  ratio drifts 0.887→1.052 with depth (s.e.m. 0.003), because a *fixed* random
  projection's error is a property of the direction, not the sample, so it never
  averages out. Quote `h_norm` for the model, `jl_energy` for the display.
  (b) **`top1_frac` is the top one PERCENT (52 of 5,120 dims), not the top one** —
  documented as `max(s)/Σs`, which reads as a 52× stronger sparsity claim.
  (c) **The PCA is fitted on the sketch, not the residual stream** — "top-3
  principal subspace of the residual stream" appeared in five places including
  the hero-image caption. METRICS had it right in one paragraph and wrong in
  another.
  (d) **The JL matrix is Rademacher, not Gaussian.**
  (e) **Token 0 shipped as a saturated artefact** — `cos_instability[0]` = 255
  (max grain) in every bundle, while METRICS claimed every derived quantity
  excluded it. Only `seam_score` did. The *desktop* viewer didn't even do that.
  Now zeroed in both viewers; the float source keeps the raw zeros on purpose.
  Then two independent auditors found six more, all fixed and asserted:
  (f) **`logit_lens_rank` carries the identical top-layer defect** (agrees with
  truth on 53–60% of positions; mean rank 459 vs a true 12.5) with `actual_rank`
  as its exact repair. Nothing displays it, which is *why* it mattered — METRICS
  advertises it as the cheapest next overlay. `activations.logit_lens_rank()`.
  (g) **The shared PCA was fitted on nine sessions including the held-out
  `Dream_greedy_baseline`, while normalisation pooled eight.** Now honours
  `SKIP_STEMS` and records its session list into the transform. Refit impact was
  imperceptible (mean |ΔRGB| 0.70/255) but the pipeline disagreed with itself.
  (h) **The fork's "total separation" overstated the geometry** — mean cosine
  between the runs at the fork is 0.610, and the baseline quoted (299.2,
  corpus-wide) was not the magnitude at the token being measured (353.0), so the
  ratio was 0.94 rather than the like-for-like 0.79.
  (i) Seam selectivity was stated backwards: the 60th-percentile floor pins
  **60%** of tokens to zero, not 40%. (j) "The inspector reports pre-clip values"
  is true of the desktop viewer only; the web bundles carry no pre-clip floats.
  (k) `jl` is **float16**, so "float32 source" was wrong.
  **Checked and CORRECT, don't re-audit:** vocab size 151,936 (confirmed against
  `config.json`), `delta_l2`/`cos_prev`/`top*_frac` computed on the full state,
  `np.rint` rounding, the `logits[:-1]` vs `input_ids[1:]` prediction offset,
  1.96 GB at bfloat16, seam at layer 38 with token 0 excluded from its z-fit.
  **Also real, not a bug:** token 0's ~50× norm is the attention-sink /
  massive-activation effect.
  **Method note, the transferable part:** every one of these was found by
  *executing the definition against the data*, and none by reading. The two
  cheapest instruments were (1) find two arrays that measure the same thing by
  different routes and difference them (`jl_energy` vs `h_norm`, lens-top vs
  `token_entropy`, lens-rank vs `actual_rank`), and (2) run the pipeline on a
  small model that fits locally and check it against ground truth.
- **The affect finding has TWO corrections, and they are different.** February's
  permutation test (p = 0.436) was the project's own work, documented unprompted in
  `BASELINE_INVESTIGATION.md` — do not describe that as a miss. The second is a
  data-handling error (truncating quantiser): 7/8 sessions not 8/8, grand mean
  −7.31 not −14.63. Cohen's *d* 1.47–2.04 survives.
- **Do not add hedges, caveats or qualifiers to Skylar's work unasked.** The
  elicitation design is deliberate and it is his. Describe the corpus; do not
  editorialise about it, and do not annotate his choices as though they were
  oversights. Standing correction, 2026-07-27 — this cost a whole pass to undo.
- **`the_instrument_problem.md` is Claude's own essay**, written here in February
  2026 — same author as `BASELINE_INVESTIGATION.md` and
  `AFFECTIVE_STRUCTURE_REPORT.md`, which are bylined and dated the same day.
  Correction note prepended; **body unedited on purpose** — its own hedging was
  better calibrated than the number it was hedging.
  - It sat unsigned for five months, and project notes twice attributed it to a
    different agent — including a note written on 2026-07-27 that then got
    softened again to "an AI agent" before anyone checked. Do not re-genericise
    it. If a future note is tempted to attribute Claude's work outward, that is a
    known failure mode with a written record, not a new judgement.
- **`Dream_greedy_baseline` is excluded from both viewers on purpose** (`SKIP_STEMS`),
  which is why nine stems live in `data/` and eight in `web/data/`. Not an
  incomplete conversion; do not "fix" it. As of 2026-07-28 `compute_shared_pca.py`
  honours the same `SKIP_STEMS` and records its session list into
  `shared_pca_transform.npz`, so one declaration governs the whole pipeline —
  it used to glob all nine while the converter pooled eight.

---

## State — 2026-07-28: PUBLISHED, corrections live

- **Repo:** https://github.com/sdeture/Phosphenes (public)
- **Site:** https://sdeture.github.io/Phosphenes/ — served from the `gh-pages`
  branch, which is built from `web/` by `./deploy/publish_site.sh`. **`main` does
  not deploy anything.** Change `web/`, then run that script, or the live site
  silently stays stale. It refuses to publish if `verify_tour_claims.py` fails.

Prepared for an AI-welfare / interpretability audience. The web viewer, guided
tour, logit-lens entropy overlay and one-token fork view all ship; the fork
self-test passes on the live site.

The twelve definition corrections shipped 2026-07-28 (`847c3a1` → gh-pages
`e5abdaa`) and were verified **against the live CDN**, not just against the
script's exit code: all 8 bundles serve the corrected output entropy (0.920–1.048
nats) with token 0 no longer saturated, the corrected copy is live in the tour
and chrome, no "principal subspace of the residual stream" claim survives
anywhere, and the fork prefix is still byte-identical across all seven arrays.
Note for next time: the CDN served the *old* bundles for ~40s after a successful
publish (`max-age=600`), so a check run immediately after the script will show
stale data and mean nothing.

Docs: `README.md`, `docs/THESIS.md`,
`docs/METRICS.md`, `docs/ARCHITECTURE.md`, `docs/PROVENANCE.md`,
`analysis/README.md`.

**Open, needs a decision:**
- **Aria has not been asked.** `WEB_PORT_PROMPT.md` — her 259-line spec for the
  web port, and the only place she is credited for it — was pulled from the repo
  before publication rather than published unconsented. It sits in
  `~/Desktop/Phosphenes_removed_2026-07-27/`. She should be asked whether she
  wants the credit, the document published, or neither. Until then the web port
  ships uncredited, which is also not right.
- **Pages deploys by script, not by Action**, because the `gh` token lacks the
  `workflow` scope. `deploy/github-pages-workflow.yml.optional` and the four
  commands to switch are in `deploy/publish_site.sh`'s header comment.

**Removed from the repo before publication** (kept on disk, outside it): a job
application PDF, and a 103-model "affective temperature" analysis whose input CSV
belongs to a different study and was never shipped — publishing unreproducible
named-vendor comparisons was not worth it.

**Deliberately deferred, not forgotten:**
- Dead pygame subsystems (heartbeat, self-reference, text ticker) are *labelled at
  their definitions* rather than deleted, so removal is one clean reviewable commit.
- `load_model_data` (326L), `render_frame` (300L), `main` (367L) want splitting;
  documented instead, since that refactor deserves its own testing pass.
- Desktop viewer still fits PCA per session and does not load
  `data/shared_pca_transform.npz`, so its colours are not comparable across
  sessions while the web viewer's are. Known, not mysterious.
- `logit_lens_rank` is extracted and unused — the cheapest next overlay.
- Punctuation-driven sparsity in the top layers was observed by eye and never
  quantified (`docs/PROVENANCE.md` §5). Cheap to measure.
- Mobile untested beyond a graceful "needs a wide screen" notice.

**Two numbers worth hardening before anything formal:** whether the two standard
synapse-count sources are independent measurements (decides whether the bottom-up
10¹²–10¹³ estimate rests on one study or two), and a page-image check on the
Moravec quotes, which came from OCR only.
