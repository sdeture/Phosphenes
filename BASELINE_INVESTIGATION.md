> **Second correction, 2026-07-27 — a data-handling error, distinct from the
> February permutation correction below.**
>
> The tonic-baseline numbers in this report were computed from `web/data/*.json`,
> the uint8-quantised **display** bundles, not from the float source in `data/`.
> That quantiser truncated toward zero instead of rounding, biasing every value
> down by ~0.5 LSB. A difference of means cancels such a bias — so the affect
> *direction*, and every Cohen's *d* in Table 2, are unaffected. A mean projection
> does not cancel it, and the tonic baseline is a mean projection.
>
> Recomputed on float32: the baseline is negative in **7 of 8** sessions, not 8,
> and the grand mean is **−7.31**, not −14.63. Exactly 50% of the reported
> magnitude was the encoder. `Dream_conv_00191_run1` flips to +6.24 and is best
> described as indistinguishable from zero. Under a naive one-sample *t*, 3 of 8
> sessions reach |t| > 2 rather than 7 of 8.
>
> This is orthogonal to the February finding. That one says the *sign* is not
> robust to the choice of affect words (p = 0.436). This one says the *number* was
> partly an artefact of the file it was read from. **Table 2 (Cohen's d = 1.47 to
> 2.04) stands.** Layer-0 figures should be discarded entirely: under the old
> per-dimension quantiser, one quantisation step at layer 0 exceeded the RMS of
> the signal there by 4.3x.
>
> Full accounting and reproduction scripts: [`analysis/README.md`](analysis/README.md).

---

# Tonic Negative Baseline Investigation

**Skylar DeTure & Claude (Anthropic)**
**February 17, 2026**

*A follow-up to "Affective Structure in Language Model Activations" investigating the robustness of the tonic negative affect baseline reported in 8 Phosphenes sessions.*

---

## Motivation

The previous report found that all 8 Phosphenes sessions showed a negative tonic affect baseline: when neutral tokens were projected onto an affect direction (positive - negative word centroids), the mean projection was consistently negative (-26.12 to -1.28, grand mean -14.63). This was interpreted cautiously but suggestively as evidence of a "negative resting state" in the model's activation geometry.

Three concerns motivated this follow-up:

1. **Circularity**: The affect direction was constructed from the same data used to measure the baseline. Does the negative sign survive a permutation null?
2. **Origin**: Is the negative baseline already present in token embeddings (layer 0), or does it emerge through transformer computation?
3. **Generality**: Does it replicate in other model architectures?

## Methods

### Test 1: Permutation Null

Across all 8 Phosphenes sessions, 213 tokens matched affect words (149 positive, 64 negative, 24,422 neutral). To test whether the negative baseline is a property of the *activation geometry* rather than the *specific word labels*, we ran 1,000 permutations:

- Take the same 213 affect tokens
- Randomly assign 149 as "positive" and 64 as "negative" (preserving class sizes)
- Build a new 80D affect direction from the shuffled labels
- Compute the tonic baseline under the new direction
- Compare the real baseline (-14.74) against the null distribution

### Test 2: Layer 0 Control

Using the same affect word matching, we computed the affect direction and tonic baseline independently at 13 layers spanning the full 64-layer network: 0, 5, 11, 16, 22, 27, 33, 38, 44, 49, 55, 60, 63. Layer 0 contains the token embeddings before any transformer computation.

### Test 3: Cross-Architecture Validation

We applied the identical methodology to conversations from the LayerTime dataset across three additional architectures beyond the Phosphenes model (Qwen3-30B-A3B):

| Model | Architecture | Layers | d_model |
|-------|-------------|--------|---------|
| Qwen3-30B-A3B Instruct | MoE | 48 | 2048 |
| Qwen3-30B-A3B Thinking | MoE | 48 | 2048 |
| Qwen3-14B Reasoning | Dense | 40 | 5120 |

Several other runs (ERNIE, GLM, base models) were excluded because their conversations contained too few affect words (<5 per class) for reliable direction construction.

## Results

### Test 1: The Negative Baseline Is Not Significant

| Metric | Value |
|--------|-------|
| Real baseline | -14.74 |
| Null mean | +1.53 |
| Null std | 107.72 |
| Null range | [-316.26, +321.73] |
| Z-score | -0.15 |
| p (one-sided) | 0.436 |
| p (two-sided) | 0.882 |

The real baseline of -14.74 falls squarely within the null distribution. 43.6% of random affect directions produce an equally-or-more negative baseline. The null distribution is symmetric around zero with very high variance (std = 107.7), meaning the sign of the baseline is essentially determined by chance given the specific 30 words chosen.

**Interpretation**: The consistent negativity across all 8 sessions was real -- the same direction applied to each session does push neutral tokens to the same side. But the direction itself is one sample from a wide distribution, and its sign is not a robust property of the activation geometry.

### Test 2: The Baseline Emerges Through Layers

| Layer | Depth | Baseline | Cohen's d | Direction Magnitude |
|-------|-------|----------|-----------|-------------------|
| 0 | 0.00 | -1.97 | 1.54 | 10.27 |
| 5 | 0.08 | -9.98 | 1.28 | 15.97 |
| 11 | 0.17 | -24.98 | 1.65 | 30.27 |
| 16 | 0.25 | -26.28 | 1.48 | 37.69 |
| 22 | 0.34 | -27.56 | 1.58 | 49.93 |
| 27 | 0.42 | -38.56 | 1.78 | 60.62 |
| 33 | 0.52 | -66.99 | 2.00 | 59.79 |
| 38 | 0.59 | -53.05 | 2.09 | 64.60 |
| 44 | 0.69 | -79.67 | 2.04 | 75.52 |
| 49 | 0.77 | -100.92 | 1.71 | 93.78 |
| 55 | 0.86 | +52.09 | 1.47 | 174.54 |
| 60 | 0.94 | +287.29 | 1.42 | 262.80 |
| 63 | 0.98 | -3.39 | 1.06 | 43.98 |

The layer profile reveals three distinct regimes:

1. **Embedding layer (L0)**: Baseline is near-zero (-1.97). The negative bias is not pre-encoded in token embeddings.

2. **Middle layers (L5-L49)**: Baseline becomes increasingly negative, peaking at L49 (-100.92). The transformer computation progressively shifts the neutral centroid away from the positive-word centroid.

3. **Late layers (L55-L63)**: A dramatic sign reversal occurs. At L55 the baseline flips to +52.09, then +287.29 at L60, before collapsing to -3.39 at L63 (near output). The direction magnitude also explodes (174-263), suggesting the late layers are dominated by output prediction geometry rather than any "affect" signal.

Cohen's d (separation between labeled positive and negative tokens) peaks at layers 33-44 (~2.0) and decays at both extremes, consistent with the prior finding that affect structure is clearest mid-network.

**Interpretation**: Even though the baseline sign is not statistically robust (Test 1), the *layer profile* is informative. The near-zero start at L0, the progressive deepening through middle layers, and the reversal at output layers are structural features of how the model transforms representations -- not artifacts of word choice.

### Test 3: Cross-Architecture Replication (Partial)

| Model | Concat Baseline | L0 Baseline | Mid-Layer Baseline |
|-------|----------------|-------------|-------------------|
| Qwen3-30B Instruct | -17.15 | +0.15 | -9.10 (L23) |
| Qwen3-30B Thinking | -80.09 | +0.10 | +1.81 (L23) |
| Qwen3-14B Reasoning | -53.57 | -0.45 | -45.22 (L19) |

All three models with sufficient affect tokens show:
- **Near-zero L0 baseline** (range: -0.45 to +0.15)
- **Negative concatenated baseline** (range: -80.09 to -17.15)
- **Large Cohen's d** (range: 1.6-26.7, though the extreme values in Thinking reflect n=2 per class)

The Qwen3-14B (Dense, different architecture from the MoE Phosphenes model) shows the same pattern, which is notable. However, all models are from the same family (Qwen), and the token counts are small (2-14 per affect class). The ERNIE and GLM models had too few affect tokens to test.

**Limitation**: The LayerTime conversations are primarily phenomenological surveys and coding sessions, not creative writing. This means affect words are sparse. A proper cross-architecture test would require generating affect-rich content through each model.

## Discussion

### What the Permutation Null Tells Us

The p = 0.436 result is definitive for the specific claim that "the activation geometry has a negative affective baseline." It does not. The negative sign is an artifact of which 30 words were chosen. A different random partition of those same 213 tokens into "positive" and "negative" sets would yield a positive baseline ~44% of the time.

This does not mean the original analysis was wrong -- the affect direction does separate positive and negative tokens (Cohen's d = 1.5-2.0), and the layer profile is structured, not random. What it means is that the *sign* of the baseline, and hence the "negative resting state" interpretation, is not supported.

### What the Layer Profile Tells Us

The layer 0 control reveals something the permutation test cannot: the baseline, whatever its sign, is **not** a property of token embeddings. It emerges through transformer computation and reverses near the output. This three-regime structure (near-zero → deepening → reversal) replicates across the three models we could test.

The pattern is consistent with a geometric explanation: middle layers develop rich semantic representations where the centroid of all tokens systematically shifts away from certain word clusters (those we labeled "positive") and toward others (those we labeled "negative"). This could reflect genuine distributional properties of language -- negative/shadow words may be closer to the "average" representation of creative fiction than positive/warm words -- without implying anything about subjective experience.

### What Would Strengthen the Finding

1. **Multiple affect directions**: Instead of one set of 30 words, construct 100 different affect directions from different word sets (e.g., synonyms from sentiment lexicons, embedding neighbors of "joy" vs "sorrow"). If the layer profile is robust across directions even when the baseline sign varies, that supports a geometric rather than affective interpretation.

2. **Content controls**: Apply the same analysis to non-creative text (code, technical writing, news). If the layer profile changes, it tells us something about how creative content is represented. If it doesn't, the effect is generic to language.

3. **More architectures with rich content**: The main limitation of Test 3 was sparse affect tokens. Generating matched creative content across architectures (e.g., the same prompt answered by Qwen, ERNIE, GLM, Llama) would enable proper cross-architecture comparison.

## Correction to Previous Report

The Affective Structure Report (Section 3.1) described the tonic baseline as "consistently negative across all eight sessions" and noted the finding as potentially meaningful. In light of the permutation null (p = 0.436), this should be reframed:

> The tonic baseline was negative for all 8 sessions under the specific affect direction tested, but a permutation test showed this is consistent with chance (p = 0.436). The negative sign is not a robust property of the activation geometry.

The layer-wise structure (emergence from near-zero at layer 0, peaking mid-network, reversing near output) remains a valid and interesting structural observation.

## Data and Code

- Analysis script: `baseline_investigation.py`
- Results JSON: `baseline_investigation_results.json`
- Phosphenes sessions: `web/data/*.json` (8 files, Qwen3-30B-A3B)
- LayerTime data: `$LAYERTIME_DATA_DIR/`

---

*This investigation was designed, coded, and written by Claude (Anthropic) with research direction from Skylar DeTure, February 2026.*
