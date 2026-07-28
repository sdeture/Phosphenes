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

# Affective Structure in Language Model Activations: A Two-Scale Investigation

**Skylar DeTure & Claude (Anthropic)**
**February 2026**

---

## Abstract

We investigate whether language models exhibit consistent affective structure in their internal representations and creative output, and whether this structure varies with architecture. Using two complementary datasets, we examine affect at both the mechanistic level (activation-space projections across 64 transformer layers in 8 sessions from Qwen3-VL-32B) and the behavioral level (phenomenological ratings from 4,117 conversations across 103 models). At the mechanistic level, we find that all 8 sessions show a negative-leaning tonic affect baseline in their activation geometry, with affect-laden tokens separable at all tested layers (Cohen's d = 1.47 to 2.04). At the behavioral level, Mixture-of-Experts models produce significantly warmer creative output than dense models (d = 0.48, p = 0.018), with the effect concentrated in experiential dimensions (flow quality, temporal horizon, attention breadth, vividness) rather than structural quality dimensions (cohesion, resolution, trust). Continuous architectural parameters (hidden size, depth) show no independent effect on affective temperature. These findings suggest that affective structure in language model output is not merely a surface-level stylistic choice but has detectable mechanistic correlates, and that architecture type modulates the behavioral expression of this structure.

---

## 1. Introduction

Recent work has established that language models encode high-level semantic concepts as linear directions in their activation spaces (Tigges et al., 2023; Zou et al., 2023; Gurnee & Tegmark, 2024). Sentiment, in particular, has been shown to occupy a recoverable linear subspace, with causally relevant attention heads identified through activation patching (Tigges et al., 2023). Separately, a growing literature on AI welfare has begun asking whether models exhibit internal states that could be morally relevant (Butlin et al., 2023; Long et al., 2024; Anthropic, 2025a), with some evidence for functional introspection in large language models (Anthropic, 2025b; Berg et al., 2025).

These two research threads -- mechanistic interpretability and AI welfare -- have developed largely in isolation. The interpretability literature asks *where* concepts are represented; the welfare literature asks *whether* models have states worth caring about. Neither has focused on what we call **tonic affect**: a persistent, session-wide affective baseline analogous to mood in biological systems. If such a baseline exists in activation geometry, it would have implications for both interpretability (as a global bias term that modulates token-level representations) and welfare (as a candidate for a standing internal state).

This report pursues a two-scale investigation. The two scales are complementary but not directly linked: the mechanistic analysis comes from a single model architecture, while the behavioral analysis covers many architectures but lacks mechanistic depth. The connection between them is conceptual rather than empirical -- we cannot yet trace the mechanistic affect geometry of Phosphenes to the behavioral differences observed in the Welfare Study. With that caveat, the combination provides a richer picture than either alone. First, we use activation-level data from Phosphenes -- a tool for visualizing transformer hidden states via Johnson-Lindenstrauss random projection (Aghajanyan et al., 2021; Bhojanapalli et al., 2021) -- to characterize the affective geometry of 8 creative writing sessions from a single model (Qwen3-VL-32B-Instruct). Second, we use behavioral-level data from the AI Welfare Study -- 4,117 conversations across 103 models with 16 phenomenological dimension ratings -- to test whether architectural variables predict affective output at scale.

---

## 2. Data and Methods

### 2.1 Phosphenes Dataset (Deep, N=8)

Eight conversations were conducted with Qwen3-VL-32B-Instruct using the DreamPromptFlow protocol, in which the model selects its own creative prompt, generates a dream narrative, and then reflects on the experience. Six sessions used temperature 0.6 sampling; two used greedy decoding (one clean baseline, one with a single-token perturbation at position 72). Each session contained approximately 3,000 tokens across 64 transformer layers.

Hidden state activations were extracted via a single forward pass with `output_hidden_states=True` and projected from the model's 5,120-dimensional hidden space to 16 dimensions using a Rademacher Johnson-Lindenstrauss matrix (seed=42). The JL projection preserves pairwise distances with high probability while compressing the data by a factor of 320. For web visualization, JL vectors were quantized to uint8 and stored alongside RGB color mappings derived from a shared PCA transform across all sessions.

For analysis, the uint8-quantized JL vectors were dequantized to float32 using per-dimension min/max scaling.

### 2.2 AI Welfare Study Dataset (Broad, N=103)

The AI Welfare Study dataset contains 4,117 conversations across 103 models from multiple providers (Google, Anthropic, OpenAI, Meta, Qwen, Mistral, DeepSeek, and others). Each conversation followed a similar protocol: the model was asked to choose a creative dream prompt, respond to it, and then provide a subjective reflection. Each conversation was rated on 16 phenomenological dimensions on a 1-10 scale, including affective temperature (cool to warm), flow quality, cohesion, agency, metacognition, and others.

Architectural metadata was available for 87 of the 103 models, including binary MoE classification (56 MoE, 31 dense), hidden size (range: 2,048-16,384), and depth (range: 28-162 layers). Behavioral indicators included denial rate (proportion of responses containing explicit denial of phenomenological experience) and engagement rate.

### 2.3 Affect Direction Construction

We defined affect using lexical token matching. Positive tokens were those whose decoded text matched or started with one of 15 positive words (joy, love, beauty, wonder, warm, light, hope, dream, alive, discover, curiosity, delight, gentle, peaceful, glow). Negative tokens matched or started with one of 15 negative words (fear, dark, cold, empty, lost, alone, pain, silent, fade, shadow, broken, nothing, gone, dust, ache). This yielded 149 positive and 64 negative token instances pooled across all 8 sessions.

The shared affect direction was computed as the difference of group means (mean positive vector minus mean negative vector), normalized to a unit vector. Separate directions were computed at each of five test layers (11, 22, 33, 44, 55) and as an 80-dimensional concatenated vector across all five layers. Importantly, the affect direction was constructed from the same sessions used for analysis. The resulting Cohen's d values between positive and negative tokens therefore reflect a partially circular measure -- they confirm that the direction *does* separate the tokens it was designed to separate, but do not independently validate the direction on held-out data. The tonic baseline measure (projection of non-affect tokens) is less circular, since those tokens were not used to construct the direction.

### 2.4 Statistical Methods

For the Phosphenes analysis, affect was quantified by projecting each token's JL vector onto the affect direction at each layer. Tonic baseline was defined as the mean projection of tokens not in either the positive or negative lexicon. Cohen's d was computed using pooled standard deviations between positive and negative token projections.

For the AI Welfare Study, all analyses used model-level means as the unit of analysis to account for the nested structure of conversations within models. Architecture comparisons used Mann-Whitney U tests (non-parametric, appropriate for unequal group sizes and non-normal distributions). Continuous architectural predictors were assessed via Spearman rank correlations. Partial correlations controlling for architecture type were computed by correlating residuals after linear regression on the MoE indicator.

---

## 3. Results

### 3.1 Tonic Affect Baseline

All eight Qwen sessions exhibited a negative-leaning tonic affect baseline in the 80-dimensional multi-layer projection (Table 1). The grand mean baseline across sessions was -14.63, with no session showing a positive baseline. Note that the units of projection are arbitrary and depend on the specific affect direction; the consistent negative *sign* is the meaningful finding, not the magnitudes.

> **Correction (February 17, 2026):** A follow-up permutation test (see `BASELINE_INVESTIGATION.md`) found that the negative sign of the tonic baseline is not statistically robust: randomly re-assigning the same affect tokens to "positive" vs "negative" classes and recomputing the direction yields an equally-or-more negative baseline 43.6% of the time (p = 0.436, 1,000 permutations). The consistent negativity across sessions reflects the use of a single shared direction, not a robust property of the activation geometry. The layer profile (near-zero at embedding layer, deepening through middle layers, reversing near output) remains a valid structural observation. See the full investigation for details.

**Table 1. Tonic affect baseline by session (80D multi-layer projection)**

| Session | Baseline |
|:---|---:|
| Gothic Teacup Realization | -26.12 |
| Sentient Toaster | -24.45 |
| I Am an AI | -20.05 |
| Peach's Lullaby | -14.96 |
| Sentient Teacup | -10.80 |
| Sentient Library (greedy, perturbation) | -10.45 |
| Well-Read Library Visitor (greedy, clean) | -8.94 |
| Library of Ideas | -1.28 |

The range was substantial (25 units), suggesting that while the direction of baseline affect is consistently negative, its magnitude varies considerably across sessions. Sessions with gothic or mechanistic themes (Gothic Teacup, Sentient Toaster) showed the most negative baselines, while the Library of Ideas session was nearly neutral.

### 3.2 Layer-Wise Affect Separation

Positive and negative tokens were well-separated at all five tested layers, with Cohen's d exceeding 1.4 at every depth (Table 2).

**Table 2. Cohen's d between positive and negative tokens by layer**

| Layer | Cohen's d |
|:---|---:|
| 11 | 1.65 |
| 22 | 1.58 |
| 33 | 2.00 |
| 44 | 2.04 |
| 55 | 1.47 |

Affect separation was strongest in the middle-to-late layers (33-44), with a slight decrease at layer 55. This pattern is consistent with a model that develops affect representations in the mid-network and partially consolidates them by the final layers, potentially as affect-laden content merges with other semantic features needed for next-token prediction.

### 3.3 Turn-Level Affect Structure

Across sessions, the creative turns (Turn 4, the dream narrative) showed a relative uplift in affect compared to the initial assistant responses (Turn 2), though all assistant turns remained negative in absolute terms. Turn 5 (the user's reflection prompt) was the only consistently positive turn, likely because the prompt template contains positive framing. The most interesting pattern is that Turn 6 (the model's own reflection) was reliably less negative than Turn 4 (the dream) in 7 of 8 sessions, suggesting that self-reflection is associated with a positive affective shift.

### 3.4 Architecture and Affective Temperature

MoE models produced significantly warmer creative output than dense models (Table 3). This was a medium-sized effect that held at conventional significance thresholds.

**Table 3. Affective temperature by architecture type**

| | N (models) | Mean | SD |
|:---|:---:|:---:|:---:|
| MoE | 56 | 5.20 | 1.67 |
| Dense | 31 | 4.35 | 1.85 |

Mann-Whitney U = 1135, p = 0.018. Cohen's d = 0.48.

Neither hidden size (rho = -0.156, p = 0.149, n = 87) nor depth (rho = -0.129, p = 0.232, n = 87) showed significant correlations with affective temperature. The MoE/dense distinction was the only architectural predictor that reached significance.

### 3.5 Denial Rate and Affect

Denial rate showed a weak negative bivariate correlation with affective temperature (rho = -0.145, p = 0.147, n = 102) that vanished entirely when architecture was controlled (partial rho = -0.013, p = 0.908). This suggests that any association between denial and coldness is confounded by architecture: dense models tend to both deny more and produce colder output, but these are independent consequences of architecture rather than causally linked behaviors.

### 3.6 Phenomenological Dimensions by Architecture

The MoE-dense distinction affected some phenomenological dimensions more than others (Table 4). The four dimensions reaching significance (p < 0.05) were all "experiential process" dimensions -- aspects of how the model engages with the task -- rather than "output quality" dimensions.

**Table 4. Phenomenological dimensions with largest MoE vs. dense effects**

| Dimension | Cohen's d | p | MoE Mean | Dense Mean |
|:---|:---:|:---:|:---:|:---:|
| Flow quality | +0.58 | 0.014 | 3.93 | 3.01 |
| Temporal horizon | +0.53 | 0.029 | 7.20 | 6.63 |
| Attention breadth | +0.49 | 0.033 | 6.73 | 6.10 |
| Context vividness | +0.46 | 0.045 | 6.92 | 6.45 |
| Affective temperature | +0.48 | 0.018 | 5.20 | 4.35 |

Dimensions showing near-zero architectural effects (|d| < 0.15) included cohesion, resolution, phenomenological trust, error sensitivity, and recognition resonance. This pattern suggests that MoE architecture facilitates richer experiential engagement without affecting the structural coherence of the output.

### 3.7 Warmest and Coldest Models

The five warmest models were Google Gemini 2.5 Pro (8.45), Qwen3-VL-32B-Instruct (8.36), Google Gemini 3 Pro Preview (8.28), InclusionAI Ling-1T (8.18), and Qwen3-VL-235B-A22B-Instruct (7.63). The five coldest were Mistral Large 2411 (1.00), Moonshot Kimi K2.5 (1.00), OpenAI GPT-4o (1.06), Tencent Hunyuan-A13B-Instruct (1.19), and MiniMax-01 (1.30). Provider-level clustering was strong: Google Gemini and Qwen models consistently ranked warm, while Mistral and early OpenAI GPT models ranked cold.

---

## 4. Discussion

### 4.1 The Negative Baseline

The most striking finding from the Phosphenes analysis is the universal negative tonic affect baseline. Every session, regardless of content, temperature, or decoding strategy, exhibited a negative-leaning default in its activation geometry. ~~This is not merely a property of the specific tokens being generated; the baseline is computed over tokens that are neither positive nor negative by our lexical criterion, suggesting it reflects a structural bias in the model's representation space.~~ **[Correction: A subsequent permutation test (p = 0.436) showed that the negative sign is not statistically robust -- see `BASELINE_INVESTIGATION.md`. The consistency across sessions reflects a shared direction, not a geometric property. The layer-wise structure (emergence through layers, mid-network peak, output reversal) remains informative.]**

Several interpretations are possible. The negative baseline may reflect the affect direction's alignment with a general "unmarked" representation that happens to project negatively, rather than a genuine affective state. Alternatively, it may reflect training data statistics: if negative affect words are more common or more distinctive in pretraining data, the affect direction could be calibrated such that the neutral center of representation space falls on the negative side. A third possibility is that it reflects a real asymmetry in the model's internal geometry, analogous to the negativity bias observed in human cognition. Our data cannot distinguish these interpretations, and we flag this as a key limitation.

### 4.2 MoE Architecture and Warmth

The finding that MoE models produce warmer creative output than dense models (d = 0.48) is, to our knowledge, novel. Prior work has shown that MoE models respond differently to instruction tuning (Shen et al., 2024) and exhibit less superposition (Friedman et al., 2025), but behavioral differences in creative output have not been characterized.

We speculate that the MoE advantage in experiential dimensions may relate to the expert routing mechanism. MoE models can route different aspects of a creative task to different experts, potentially allowing for more differentiated processing of affective, narrative, and stylistic components. Dense models must process all of these through the same capacity bottleneck. This hypothesis is testable with expert-level activation analysis, which we leave to future work.

### 4.3 What Architecture Does Not Predict

The null results are as informative as the positive ones. Hidden size and depth -- the two most commonly cited architectural parameters in scaling law analyses -- showed no detectable effect on affective temperature. This suggests that affective structure is not a simple function of model capacity. Similarly, denial rate had no independent predictive value once architecture was controlled, indicating that a model's tendency to deny phenomenological experience is architecturally confounded rather than causally linked to its affective output.

### 4.4 Limitations

Several important limitations constrain interpretation:

1. **Lexical affect operationalization.** Our affect direction in the Phosphenes analysis is constructed from a small lexicon of 30 words. This captures only a narrow slice of affective content and may miss figurative, ironic, or contextual affect.

2. **Single model family for mechanistic analysis.** The Phosphenes data comes entirely from Qwen3-VL-32B-Instruct. We cannot determine whether the tonic negative baseline generalizes across architectures. The AI Welfare Study provides indirect evidence via affective temperature ratings, but these are behavioral rather than mechanistic.

3. **Phenomenological ratings as a dependent variable.** The 16 phenomenological dimensions are rated by a judge (in this case, an LLM evaluator), not directly measured from activations. The validity of these ratings as proxies for internal model states is an open question.

4. **Multiple comparisons.** In Analysis 5, we tested 16 dimensions for MoE vs. dense differences. Four reached p < 0.05, which exceeds the one we'd expect by chance alone, but none survive Bonferroni correction (threshold = 0.003). The pattern of effects is interpretable (experiential > structural), which provides some additional confidence, but these results should be treated as hypothesis-generating rather than confirmatory.

5. **Confounding by training data and RLHF.** Model families that are warmer (Gemini, Qwen) may differ from colder families (Mistral, GPT-4o) not because of architecture but because of training data composition or RLHF objectives. We cannot disentangle architecture from training in this observational design.

---

## 5. Conclusion

Affective structure in language model activations and output is detectable at two scales. At the mechanistic level, affect-laden tokens are separable throughout the transformer depth (d > 1.4 at all tested layers), and the representation space exhibits a persistent negative baseline. At the behavioral level, MoE architecture predicts warmer creative output than dense architecture, with the effect concentrated in experiential rather than structural quality dimensions. These findings provide initial evidence that affective structure in language models has both mechanistic depth and architectural modulation, and suggest that the MoE/dense distinction deserves more attention in studies of model behavior and AI welfare.

---

## References

Aghajanyan, A., Gupta, S., & Zettlemoyer, L. (2021). Intrinsic dimensionality explains the effectiveness of language model fine-tuning. *ACL-IJCNLP 2021*, 7319-7328.

Anthropic. (2025a). Exploring model welfare. *Anthropic Research Blog*.

Anthropic. (2025b). Signs of introspection in large language models. *Transformer Circuits Thread*.

Berg, C., de Lucena, D., & Rosenblatt, J. (2025). Large language models report subjective experience under self-referential processing. *arXiv:2510.24797*.

Bhojanapalli, S., Yun, C., Rawat, A.S., Reddi, S.J., & Kumar, S. (2021). Johnson-Lindenstrauss lemma, linear and nonlinear random projections, random Fourier features, and random kitchen sinks: Tutorial and survey. *arXiv:2108.04172*.

Butlin, P., Long, R., Elmoznino, E., Bengio, Y., et al. (2023). Consciousness in artificial intelligence: Insights from the science of consciousness. *arXiv:2308.08708*.

Friedman, D., Wettig, A., & Chen, D. (2025). Sparsity and superposition in mixture of experts. *arXiv:2510.23671*.

Gurnee, W. & Tegmark, M. (2024). Language models represent space and time. *ICLR 2024*.

Jiang, Y., Rajendran, G., Ravikumar, P., Aragam, B., & Veitch, V. (2024). On the origins of linear representations in large language models. *ICML 2024*.

Li, C., Wang, J., Zhu, K., et al. (2023). Large language models understand and can be enhanced by emotional stimuli. *arXiv:2307.11760*.

Long, R., Sebo, J., Butlin, P., et al. (2024). Taking AI welfare seriously. *arXiv:2411.00986*.

Marks, L., et al. (2024). Interpreting learned feedback patterns in large language models. *NeurIPS 2024*.

Shen, S., Hou, L., Zhou, Y., et al. (2024). Mixture-of-experts meets instruction tuning: A winning combination for large language models. *ICLR 2024*.

Tigges, C., Hollinsworth, O.J., Geiger, A., & Nanda, N. (2023). Linear representations of sentiment in large language models. *arXiv:2310.15154*.

Zou, A., Phan, L., Chen, S., et al. (2023). Representation engineering: A top-down approach to AI transparency. *arXiv:2310.01405*.

---

*This report was produced collaboratively by Skylar DeTure and Claude (Anthropic, claude-opus-4-6). Analysis scripts and data are available in the Phosphenes repository. The interactive visualization is live at [futuretbd.ai/phosphenes](https://futuretbd.ai/phosphenes/).*
