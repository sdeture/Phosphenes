# Corrections and audits

This directory holds the scripts that check this project's own claims, and the
record of what they found. It exists because one of the findings was wrong.

Everything below is reproducible from the checked-in scripts. Where a number
came from an audit rather than from my own hand-verification, this file says so.

---

## 1. The affect baseline was substantially a quantiser artefact

**Status: superseded. Do not cite the original numbers.**

### The original claim

`run_analysis.py` builds an "affect direction" in the 16-dimensional JL space as
`mean(vectors at positive-affect-word tokens) − mean(vectors at negative-affect-word
tokens)`, then reports the mean projection of all other tokens onto it — a
"tonic baseline". `AFFECTIVE_STRUCTURE_REPORT.md` and `the_instrument_problem.md`
report that **all eight sessions** have a negative baseline, grand mean
**−14.63**.

### What was wrong

`run_analysis.py` reads `web/data/*.json`, whose JL vectors were uint8-quantised
for display. The quantiser used

```python
(np.clip(arr, 0, 1) * 255).astype(np.uint8)     # truncates toward zero
```

`.astype` truncates rather than rounds, biasing every dimension down by
**−0.4988 LSB** (theoretical −0.5). A *difference* of means cancels that bias,
so the affect **direction** was fine. A **mean projection** is an absolute
quantity, and it did not cancel: the bias contributed **−7.3 to −8.1** to every
session's baseline, in the same direction as the reported finding.

### Corrected result

Recomputed from the float32 `data/*_activations.npz` arrays:

| | published (quantised) | float32 (correct) |
|---|---|---|
| sessions with negative baseline | **8 / 8** | **7 / 8** |
| grand mean | **−14.63** | **−7.31** |
| sessions with \|t\| > 2 | 7 / 8 | **3 / 8** |
| session-level t(7) | −4.89 | **−2.42** |

**Exactly 50.0% of the reported grand-mean magnitude was the quantiser.**

The session that flips is `Dream_conv_00191_run1` ("Library of Ideas"):
**−1.28 → +6.24**. It is the session the original report already flagged as
"nearly neutral". On float data it is not significantly positive either; the
honest description is *indistinguishable from zero*.

Fixing the quantiser to round does **not** restore 8/8. With `np.rint` the
per-dimension bias falls to −0.0018 LSB (a 233× reduction) and the answer
converges on the float answer, which is 7/8.

### What survives

- **The separation between positive- and negative-affect word tokens.** Cohen's
  *d* = 1.47–2.04 by layer, essentially unchanged between float and quantised
  (layer 44: 2.08 float vs 2.04 quantised). The direction itself is stable —
  cosine similarity 0.9994 between the float and quantised estimates.
- **The qualitative layer profile** — separation deepening through the middle
  layers and reversing near the output. Layer 63 flips sign (−3.39 → +10.08);
  the shape does not.

### What does not survive, and never did

- **"Eight out of eight."** It is seven, and three are individually
  distinguishable from zero.
- **The magnitude −14.63.**
- **The layer-0 control.** `baseline_investigation.py` reads the same quantised
  bundles. Under the *old* per-dimension quantiser, one quantisation step at
  layer 0 was **4.26× the RMS of the signal there** — layer-0 numbers were noise.
  That control was the load-bearing evidence for "the effect is created by the
  transformer, not by the embeddings", and it did not support it. (The current
  per-layer quantiser reduces the layer-0 step to 0.025× RMS, so a rerun is now
  meaningful. It has not been done.)
- **The cross-architecture replication.** `baseline_investigation_results.json`
  records `test_3_cross_architecture` as **100% errors** — every input file was
  missing. That test never ran. Any draft citing it cites nothing.
- **Statistical significance — but this the project had already caught and
  published.** A 1,000-iteration permutation test over affect labels gave
  **p = 0.436**, three days after the original report.
  `BASELINE_INVESTIGATION.md` is a document devoted to falsifying the project's
  own headline finding and it says so plainly: *"The negative sign is an artifact
  of which 30 words were chosen."* `AFFECTIVE_STRUCTURE_REPORT.md` carries that
  correction inline, with the superseded sentence struck through.

  **This audit adds a different failure, not a repeat of that one.** The February
  correction was statistical: the *sign* is not robust to which words define the
  axis. What nobody caught is that the *descriptive* number was also wrong, for a
  data-handling reason — the analysis read a display format. Both corrections are
  needed, and only the second one is new.

### Why it was fragile in the first place

The affect direction sits **0.13° to 2.07° from exactly orthogonal** to the mean
activation vector, whose norm is ~520–550 while the baseline is ~1–19. The
reported quantity is a near-orthogonal residual of a much larger vector.
Quantisation rotates the direction by about 2°, which is by itself enough to move
the baseline by up to ~18 units. The sign of this quantity is not a robust
measurement; it is sensitive to a 2° rotation of a direction estimated from 149
positive and 64 negative tokens.

Two further weaknesses in the construction, worth recording:

- The word lists match by prefix (`t.startswith(w)`), which pulls in `lightning`
  for "light" and `lover` for "love".
- **`dream`/`dreams`/`dreaming` account for 23 of the 149 positive instances** in
  a corpus of dream narratives. The positive class is partly measuring the task
  word.
- The classes are unbalanced, 149 : 64.

### The defensible statement

> Across eight conversations, seven of eight have a negative point-estimate tonic
> baseline on the affect direction (grand mean −7.31, t(7) = −2.42), three
> individually distinguishable from zero. The result does not survive a
> permutation test over affect labels (p = 0.436). The separation between
> positive- and negative-affect tokens is large and robust (Cohen's *d* 1.47–2.04)
> and is a different claim.

`the_instrument_problem.md` — the essay left in this directory in February —
hedges the number heavily: *"it might mean nothing… the ruler might be crooked…
The map is not the territory."* Those hedges were better calibrated than the
number they hedged. The essay is kept as written, with a correction note, because
retro-fitting it would destroy the more interesting record: what it is like to
believe a result before checking the encoder.

### Reproducing

```bash
python analysis/affect_float_recheck.py          # three-way table
python analysis/affect_layer_profile_recheck.py  # layer profile, t-stats, decomposition
```

**Reproducibility caveat.** The "quantised" column reproduced the bundles *as
they shipped in February*. The encoder has since been fixed (rounding, and
per-layer ranges), so re-running against the current `web/data/*.json` will not
reproduce the -14.63 figure — which is the point. `affect_float_recheck.py`'s
round-trip column applies the quantiser in software and is the path to inspect.

### Provenance

The three-way recomputation was performed by a delegated audit; the scripts are
checked in and self-contained. I independently verified the two mechanisms it
identified: the truncation bias (`.astype` vs `np.rint`) and the layer-0
step-size-to-signal ratio under pooled per-dimension ranges. I did not
independently re-derive the per-session baselines.

---

## 2. What was fixed in the pipeline

| Fix | Where | Effect |
|---|---|---|
| Round instead of truncate | `to_uint8` | per-dim bias −0.499 → −0.002 LSB |
| Per-(layer, dim) quantisation ranges | `quantize_jl` | layer-0 step/RMS 4.26 → 0.025 |
| Pooled normalisation across sessions | `compute_global_stats` | colour and brightness now genuinely comparable between sessions; two runs sharing a prefix now render **byte-identically** |
| Token 0 excluded from the seam score | `convert_for_web.py` | removes a spurious maximal seam at the first token of every session, caused by `delta_l2[0] = cos_prev[0] = 0` |
| Refuse partial reconversion | `main()` | pooled bounds make `--stems` unsound without `--allow-partial` |

The pooled-normalisation change was forced by a self-test, not found by
inspection. The divergence view asserts that two runs differing by one token
render identically before the fork; it measures that assertion on load and
reports the result in its own header. Under per-session normalisation it measured
a discrepancy of up to 49 units on a prefix that is mathematically zero, and said
so on screen.

---

## 3. Standing checks

```bash
python analysis/verify_tour_claims.py    # 113 assertions; non-zero exit on drift
```

Every number the guided tour states is restated there as an assertion with a
tolerance and recomputed from source. Claims about the model are checked against
float `.npz` data; claims about what a viewer will *see* are checked against the
shipped bundles, because a claim true of the float data can still be invisible in
the build.

---

## 4. The general lesson, stated once

**Do not compute statistics on a display format.** The bundles in `web/data/`
exist to be drawn on a screen at 8 bits per channel. Every property that makes
them good for that — quantisation, clipping, per-session scaling — is a property
that biases an average. `data/*.npz` is four files away and is the float source.

The corollary is the one that actually cost something here: a difference of means
is robust to a constant bias, so the *direction* was fine and the analysis looked
healthy. It was the one absolute quantity in the pipeline that carried the
artefact, and it was the one being reported.
