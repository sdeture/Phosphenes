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
   uint8-quantised, clipped, pooled-normalised. Use `data/*_activations.npz`
   (float32). A published finding was half artefact because of this —
   `analysis/README.md`.
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
5. Serve `web/` over HTTP (`python3 -m http.server 8899 --directory web`).
   `file://` cannot work — ES modules and `fetch`.
6. **`web/about.html` is the SHORT version and should stay short** (~850 words).
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
  incomplete conversion; do not "fix" it.

---

## State — 2026-07-27: PUBLISHED

- **Repo:** https://github.com/sdeture/Phosphenes (public)
- **Site:** https://sdeture.github.io/Phosphenes/ — served from the `gh-pages`
  branch, which is built from `web/` by `./deploy/publish_site.sh`. **`main` does
  not deploy anything.** Change `web/`, then run that script, or the live site
  silently stays stale. It refuses to publish if `verify_tour_claims.py` fails.

Prepared for an AI-welfare / interpretability audience. The web viewer, guided
tour, logit-lens entropy overlay and one-token fork view all ship; the fork
self-test passes on the live site. Docs: `README.md`, `docs/THESIS.md`,
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
