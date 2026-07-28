# Metrics reference

Every quantity Phosphenes computes, what it means, and what it does not mean.

All numbers quoted below are for the flagship session (`Dream_greedy_clean`,
Qwen3-VL-32B-Instruct, 2,990 tokens, 64 layers, d_model 5,120) unless stated
otherwise, and are reproduced by `analysis/verify_tour_claims.py`.

---

## 0. The object being measured

For one conversation the model's residual stream is

```
2,990 tokens × 64 layers × 5,120 dimensions = 979,763,200 numbers
```

about **1.96 GB** at bfloat16. Two compressions make it displayable.

### Johnson–Lindenstrauss projection, 5,120 → 16

Applied once at extraction time (`extraction/extract_batch.py`), with a fixed
random **Rademacher** matrix — entries ±1 scaled by `1/√k`, seed 42, recorded in
each session's metadata as `jl_seed`. (Described here as Gaussian until
2026-07-28; it never was.) The JL lemma bounds the distortion of *pairwise
Euclidean distances* in terms of the number of points and the target dimension.

**What this licenses.** Distances, and quantities derived from distances:
`jl_energy`, reference-point distance, cosine similarity between cells,
projections onto directions defined by differences of cell groups.

**What it does not license.** Any statement about an individual coordinate of
the 16-dimensional vector. The coordinates are random mixtures of the 5,120
original dimensions and correspond to nothing. A "dimension 7 is high here"
claim is meaningless. The PCA colour mapping is safe because PCA is computed
*within* JL space and is a rotation of it — the colour is a summary of position,
not of any particular feature.

At 16 dimensions the JL distortion is loose in theory. Treat the sketch as a
faithful guide to *relative* geometry and a poor guide to absolute magnitudes.

### uint8 quantisation

Applied in `convert_for_web.py` for the web bundles only. Two properties of the
implementation matter and were both bugs at one point:

- **Rounding, not truncation.** `np.rint`, not `.astype(np.uint8)`. Truncation
  biases every value down by ~0.5 quantisation steps. That is invisible in a
  display and fatal in an average — see `analysis/README.md`.
- **Per-(layer, dimension) ranges, not per-dimension.** Residual-stream
  magnitude, as the sketch measures it, grows about 78× from layer 0 to layer 60
  (65.6× on the exact norm). A single range per dimension pooled over all layers
  puts the step size at the deep layers' scale,
  and at layer 0 one step then exceeds the RMS of the signal — early-layer
  vectors become noise, which silently corrupts reference distances and custom
  bases at whatever layer the user clicks.

**Anything quantitative should be computed from `data/*.npz`, not from
`web/data/*.json`.** The bundles are a display format.

### Normalisation is pooled across sessions

Every displayed quantity is normalised against bounds computed over **all eight
sessions at once** (`compute_global_stats`). This is why colour and brightness
mean the same thing in every session, and why two runs sharing a prefix render
identically. Per-session normalisation — the earlier behaviour — broke both.

Consequence: adding or removing a session changes every bundle slightly.
`convert_for_web.py` therefore refuses `--stems` without `--allow-partial`.

---

## 1. Raw per-cell metrics

Computed at extraction, stored in `data/*_activations.npz`. Shape `(T, L)`
unless noted. `H[t, l]` is the full 5,120-dimensional hidden state;
`Y[t, l]` is its 16-dimensional JL sketch.

### `jl_energy` — magnitude, as estimated from the sketch
```
jl_energy[t, l] = ‖Y[t, l]‖₂          ← the 16-dim sketch, NOT the full state
```
Drives **brightness**. Grows steeply with depth: layer mean 17.8 at layer 0,
1,381.7 at layer 60, a factor of 77.7. That growth is a well-known property of
residual streams; here you see it without plotting anything.

### `h_norm` — magnitude, exactly
```
h_norm[t, l] = ‖H[t, l]‖₂             ← the full 5,120-dim state
```
The same quantity without the sketch, and it disagrees: **19.9 at layer 0,
1,308.8 at layer 60, a factor of 65.6**, against the sketch's 77.7.

> **Quote `h_norm` for claims about the model, `jl_energy` for claims about the
> display.** The gap is not sampling noise. The ratio `jl_energy / h_norm` drifts
> monotonically with depth — 0.887 at layer 0 to 1.052 at layer 60, with a
> standard error of 0.003 — and `1.052 / 0.887 = 1.185` accounts for the whole
> `77.7 / 65.6 = 1.183` discrepancy.
>
> The cause is that the projection matrix is *fixed*. A random projection's error
> is a property of the **direction** being projected, not of the sample, so
> tokens that share a dominant direction share the same error. Residual streams
> are strongly anisotropic and their dominant direction rotates with depth, so
> the error is systematic per layer and does not shrink as tokens are averaged.
> The per-cell spread, 15.7%, does match JL theory for k = 16 (17.7%) — it is the
> *mean* that is biased, not the scatter.

Both are in every `.npz`. Nothing but brightness needs the sketch version.

### `delta_l2` — how far the state moved since the previous token
```
delta_l2[t, l] = ‖H[t, l] − H[t−1, l]‖₂        delta_l2[0, l] = 0
```
Computed on the **full** hidden state, not the sketch. Drives **shimmer**.

### `cos_prev` — did it change direction, or only size?
```
cos_prev[t, l] = cos(H[t, l], H[t−1, l])       cos_prev[0, l] = 0
```
Layer means run 0.489–0.816, median 0.731; 56 of 64 layers fall in 0.63–0.77.
`1 − cos_prev` drives **grain**.

> **Token 0 is an artefact.** It has no predecessor, so `delta_l2[0]` and
> `cos_prev[0]` are both zero *by construction*. Zero cosine means maximal
> `1 − cos_prev`, so token 0 scores as maximally unstable.
>
> In the **float source** those zeros are still there — that is the raw record,
> and any analysis must drop row 0 itself. In the **web bundles**,
> `seam_score`, `delta_norm`, `cos_instability` and `sparsity_norm` are all
> forced to 0 at token 0. Only `seam_score` was, until 2026-07-28; the other
> three shipped with `cos_instability[0]` saturated at 255, so the first column
> of every session rendered at maximum grain. This paragraph asserted the fix
> before the fix existed.
>
> Note that token 0's very large `jl_energy` and `h_norm` are **not** part of
> this: they need no predecessor. The first token's norm is ~50× the session mean
> at every layer, which is the well-documented attention-sink / massive-activation
> effect and a real property of the model.

### `top1_frac`, `top25_frac` — how concentrated was the update?
```
s = (H[t, l] − H[t−1, l])²
top1_frac[t, l]  = Σ top-k(s) / Σs,   k = ⌈0.01 · d_model⌉ = 52
top25_frac[t, l] = Σ top-k(s) / Σs,   k = ⌈0.25 · d_model⌉ = 1,280
```

> **`top1_frac` is the top one *percent*, not the top one dimension.** The `1`
> in the name is a percentage. Documented here as `max(s) / Σs` until
> 2026-07-28, which was wrong by a factor of 52 in what it counts — and read as
> a far stronger claim than the data supports, since the measured mean is 0.29
> and "one dimension out of 5,120 carries 29% of the change" is not true.
> Measured: `top1_frac` mean 0.29, `top25_frac` mean 0.83.

> **Naming warning.** The UI calls the combination "focus" and the data field is
> `sparsity_norm`. Neither is MoE sparsity nor SAE feature sparsity. It is the
> concentration of the *token-to-token change* across residual dimensions. An
> interpretability reader will assume otherwise unless told.

### `logit_lens_entropy` — uncertainty at each layer
Shannon entropy, in nats, of the next-token distribution obtained by applying
the final norm and then the unembedding to the layer-`l` state (the "logit
lens"). Shape `(T, L)`.

Layer means: 8.84 nats at layer 0, rising to a peak of **9.84 at layer 9**, then
falling to **0.998 at the output**. A uniform distribution over the 151,936-token
vocabulary is 11.93 nats.

The **rise before the fall** is present in all eight sessions. The model becomes
*more* uncertain through its early layers before committing — consistent with
early layers building representations rather than predicting.

> **Read this array through `activations.logit_lens_entropy()`, not directly.**
> The extraction double-normalised the top layer, so raw column 63 in
> `data/*.npz` is wrong. The loader replaces it with `token_entropy`, the same
> quantity computed from the model's own output logits. The substitution is
> exact rather than approximate: the logit lens at the top of the stack *is* the
> output distribution. Layers 0–62 are unaffected.
>
> The published figure was **1.65 nats**, retracted 2026-07-28. The correction
> moves the number *down* — the model commits harder than was claimed — and the
> rise-then-fall still holds in 8 of 8 sessions. Full account, including the
> bit-exact check on a smaller model of the same family, is the module docstring
> of `activations.py`.

### What a layer index means

Two things about the layer axis are not obvious, and one of them cost a
published number:

- **Layer 0 is the output of the first transformer block, not the embedding.**
  The embedding layer is not represented in any array here. The `L = 64` columns
  cover blocks 0–63.
- **The top column is the post-final-norm state.** HuggingFace appends the last
  hidden state *after* the model's final RMSNorm, because that is the tensor the
  unembedding consumes. So layer 63 of `jl`, `jl_energy`, `h_norm`, `delta_l2`,
  `cos_prev` and the two `*_frac` arrays holds a normalised state — about 10×
  smaller in magnitude than layer 62, and not on the same scale as the 63
  columns beneath it. It is a real quantity, and it is what produces the output,
  but a layer-to-layer comparison spanning it compares two different things.

Every other measurement in this repository is taken below that boundary: the
seam score at layer 38, the sparsity bands searched over layers 0–51, the
activation-growth figures quoted at layers 0 and 60, and the fork's per-layer
divergence quoted at layers 0 and 62.

### Also extracted, not yet displayed
`logit_lens_rank`, `top3_ids`, `top3_probs`, `actual_rank`, `actual_prob`. See
"Unused data" below. (`h_norm` and `token_entropy` were on this list until
2026-07-28; `token_entropy` now supplies the top layer of the entropy overlay,
and `h_norm` supplies the activation-growth figure the README quotes.)

> **`actual_rank` breaks ties optimistically.** It counts strictly-greater
> logits, so exact ties all report rank 0. Measured on the flagship: 1.02% of
> rank-0 positions (25 of 2,461) are ties where the realised token is not the
> argmax. Any top-1 hit rate computed from it is high by about a point. The same
> applies to `logit_lens_rank`.

---

## 2. Normalised per-cell metrics

Mapped to `[0, 1]` by clipping at quantiles of the pooled distribution, then to
uint8. Quantiles rather than min/max because these quantities have heavy tails
and a single outlier flattens everything else.

| Field | Source | Scope of normalisation |
|---|---|---|
| `energy_norm` | `jl_energy` | **per layer**, pooled across sessions |
| `delta_norm` | `delta_l2` | **per layer**, pooled across sessions |
| `cos_instability` | `1 − cos_prev` | global, pooled |
| `sparsity_norm` | `0.6·top1_frac + 0.4·top25_frac` | global, pooled |
| `entropy_norm` | `logit_lens_entropy / ln(151936)` | **absolute** — no quantiles |

Energy and delta are normalised **per layer** because both grow steeply with
depth; one global range would saturate the deep layers and flatten the shallow
ones. The others are normalised globally across layers because their layer
dependence *is* the signal (the two sparsity bands, the entropy gradient).

`entropy_norm` is the only field on an absolute scale: 0 means fully committed,
1 means uniform over the vocabulary. That is deliberate — quantile-normalising
it per layer would erase exactly the cross-layer gradient worth looking at.

**Clipping costs information.** Values outside the clip range are
indistinguishable after normalisation — about 10% of cells at layer 38 sit
outside the 5/95 range. The **desktop** inspector reports pre-clip values; the
**web** inspector cannot, because the bundles carry only the quantised arrays.
This paragraph claimed otherwise, without the distinction, until 2026-07-28.

---

## 3. `seam_score` — per-token discontinuity

```
mid   = ⌊0.6 · L⌋ = 38
raw   = z(delta_l2[:, mid]) + z(1 − cos_prev[:, mid])
seam  = clip to [0,1] at the 60th and 99.5th percentiles of pooled raw
seam[0] = 0
```
`z(·)` uses **pooled** means and standard deviations across all sessions, with
token 0 excluded from the fit, so a seam of 0.8 means the same thing everywhere.

The 60th-percentile floor makes this a *detector* rather than a continuous
measure: **60.1% of tokens are pinned to zero** by construction (range
56.4–63.4% across the eight sessions). It answers "is this token unusual", not
"how unusual is every token". Stated here as "roughly 40%" until 2026-07-28 —
the percentile was read as the survivor fraction rather than the floor.

### What it actually detects, measured

Across all eight sessions:

| Location | Mean seam | Ratio to elsewhere |
|---|---|---|
| `<\|im_end\|>` (a turn **ending**) | **0.780** | **7.1×** (range 6.2–7.8×) |
| everywhere else | 0.110 | 1× |
| `<\|im_start\|>` (a turn **beginning**) | **0.004** | 0.04× |

`im_end > im_start` in **8 of 8** sessions.

The asymmetry is the interesting part and it is explicable: at `<|im_end|>` the
prediction problem changes completely — the model must stop continuing this
speaker and hand over. By `<|im_start|>` that handover is already committed, and
the token is almost perfectly predicted.

Seams also fire on within-turn structure: a colon introducing a list (token 445,
seam 1.00), a topic pivot (token 1,347, "Transformer", 1.00), the boundaries of
enumerated items. **These were not labelled.** The score is computed from
activation dynamics alone and the correspondence to text structure is a finding,
not an input.

**Limits.** One layer (38), one linear combination, one session-independent
threshold. It has not been evaluated against a labelled boundary set on held-out
data, and no precision/recall figure is claimed. Treat it as a usable heuristic
with a measured effect size, not a validated detector.

---

## 4. Colour

### Default: PCA within JL space
A single PCA is fitted **once over all sessions pooled** (`compute_shared_pca.py`,
`data/shared_pca_transform.npz`), reduced to 3 components, and each component is
mapped to one of R, G, B by clipping at the pooled 2nd and 98th percentiles.
Saturation is then boosted by `0.5 + (x − 0.5)·1.3` — cosmetic, and the only
purely cosmetic step in the colour path.

Because the basis *and* the bounds are shared, similar colours mean similar
positions in the sketch's principal subspace, **across sessions**.

### Custom basis: contrast directions you choose
Click cells to define each channel as `mean(source) − mean(contrast)`. The three
directions are Gram–Schmidt orthonormalised in the order R, G, B, and signs are
fixed so each channel's own source centroid projects positive onto its axis.

Two consequences to be aware of:

- **Order matters.** R is privileged; G carries only the part of your second
  request independent of R, and B only the part independent of both.
- **Signed projections are mapped symmetrically about mid-grey** (0.5 = on the
  boundary), not from black. A contrast axis is signed, and mapping it like a
  magnitude would discard the sign.

While selecting the 2nd and 3rd axes the display shows `1 − ‖in-plane
component‖` on unit-normalised cell vectors: how much of each cell is *not yet*
explained by the axes chosen so far. Bright cells are where an informative next
axis is available.

### Effect channels

| Channel | Driven by | Note |
|---|---|---|
| brightness | `energy_norm` | floor 0.15 so low energy stays distinguishable from no-data |
| shimmer | `delta_norm` | amplitude is data; the noise field is a carrier |
| grain | `cos_instability` | ditto |
| warm glow | `seam_score` | Gaussian in the vertical, centred on mid-layers |

The noise fields are seeded (`noise.js`) so a session always looks the same. If
shimmer differed between viewings you could not trust a texture you noticed.

---

## 5. Divergence, in the fork view

For the two greedy runs that differ by one forced token at position 73:

```
divergence[t, l] = ‖Y_left[t, l] − Y_right[t, l]‖₂
```

Verified properties:

- Token IDs are identical for `t < 73`; the first difference is at exactly 73.
- JL vectors are **bit-identical** over `t ∈ [0, 72]` in the float source
  (max absolute difference 0.0), and the shipped uint8 bundles are byte-identical
  over the same range for every array including colour and seam.
- Mean over layers at the fork: **280.5**. Two baselines, and they differ:
  the magnitude of the state at *that same token* is **353.0**, giving a ratio of
  **0.79**; the corpus-wide mean cell magnitude is **299.2**, giving 0.94. The
  like-for-like comparison is the first one. Until 2026-07-28 only the second was
  quoted, under the phrase "near-total separation".
- **Not orthogonal.** Mean cosine between the two runs at the fork is **0.610**
  (0.253 at layer 0, 0.728 at layer 62). Two unrelated states would give 0, and
  `d/‖v‖` would be √2 ≈ 1.414 rather than 0.79. The divergence is large and
  permanent; it is not separation into unrelated states.
- It does not decay: 264.4 averaged over tokens 1,000–2,900.
- At the fork it is **36.5 at layer 0** and **1,286.6 at layer 62**.

**The honest caveat, also stated in the UI.** After the fork the two runs are no
longer processing the same words, so this is *not* "how differently the model
handled this word". It is how differently the model is configured at the same
point in its own output. The sequences never realign — different stories, 2,990
vs 3,379 tokens — so no token-level alignment is possible or claimed.

What survives that caveat is the **shape**: divergence small in early layers,
enormous in late ones. A different word barely changes what early layers
represent and completely changes what late layers predict.

---

## 6. Unused data, and what it is for

The `.npz` files ship 14 arrays. The viewer displays 7. Already extracted and
not yet surfaced:

- `logit_lens_rank` — rank of the eventually-chosen token at each layer. A
  per-layer "when did it decide" trace, sharper than entropy. **Read it through
  `activations.logit_lens_rank()`**: its top column carries the same
  double-normalisation defect as the entropy did, and the same exact repair
  (`actual_rank`). Note also that it is shape `(T-1, L)` — row `i` is the
  prediction made at position `i` for position `i+1` — so it is offset by one
  from every other per-layer array.
- `actual_rank`, `actual_prob`, `top3_ids`, `top3_probs` — the realised
  prediction and its competitors. Would let the display mark tokens the model
  found surprising *in its own terms*.
- `token_entropy` — final-layer entropy per token.
- `h_norm` — full-space magnitude, for checking JL magnitude fidelity.

The most useful addition would be `logit_lens_rank` as a fifth colour mode, at
no extraction cost.

---

## 7. Known limits

1. **One model, eight conversations.** Every layer-structure claim here is about
   Qwen3-VL-32B-Instruct. Nothing establishes that the sparsity bands or the
   entropy rise generalise across architectures. The cross-architecture test in
   `baseline_investigation.py` **did not run** — its input files were missing —
   so any claim resting on it rests on nothing.
2. **The seam score is a heuristic**, not a validated detector (§3).
3. **JL at 16 dimensions** is a loose approximation (§0).
4. **The corpus is not a random sample.** These conversations were selected
   because the model chose consciousness-adjacent themes, which makes the corpus
   useful as a stress case and useless as a base rate.
5. **The affect analysis in `run_analysis.py` is superseded.** See
   `analysis/README.md`: its headline claim was substantially an artefact of the
   quantiser it read from, and the corrected result is weaker than published.
