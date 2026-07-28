# Metrics reference

Every quantity Phosphenes computes, what it means, and what it does not mean.

All numbers quoted below are for the flagship session (`Dream_greedy_clean`,
Qwen3-VL-32B-Instruct, 2,990 tokens, 64 layers, d_model 5,120) unless stated
otherwise, and are reproduced by `analysis/verify_tour_claims.py`.

---

## 0. The object being measured

For one conversation the model's residual stream is

```
2,990 tokens × 64 layers × 5,120 dimensions = 979,251,200 numbers
```

about **1.96 GB** at bfloat16. Two compressions make it displayable.

### Johnson–Lindenstrauss projection, 5,120 → 16

Applied once at extraction time (`extraction/extract_batch.py`), with a fixed
random Gaussian matrix, seed 42, recorded in each session's metadata as
`jl_seed`. The JL lemma bounds the distortion of *pairwise Euclidean distances*
in terms of the number of points and the target dimension.

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
  magnitude grows about 78× from layer 0 to layer 60. A single range per
  dimension pooled over all layers puts the step size at the deep layers' scale,
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

### `jl_energy` — magnitude
```
jl_energy[t, l] = ‖Y[t, l]‖₂
```
Drives **brightness**. Grows steeply with depth: layer mean 17.8 at layer 0,
1,381.7 at layer 60, a factor of 77.7. That growth is a well-known property of
residual streams; here you see it without plotting anything.

### `delta_l2` — how far the state moved since the previous token
```
delta_l2[t, l] = ‖H[t, l] − H[t−1, l]‖₂        delta_l2[0, l] = 0
```
Computed on the **full** hidden state, not the sketch. Drives **shimmer**.

### `cos_prev` — did it change direction, or only size?
```
cos_prev[t, l] = cos(H[t, l], H[t−1, l])       cos_prev[0, l] = 0
```
Typically 0.63–0.77 by layer. `1 − cos_prev` drives **grain**.

> **Token 0 is an artefact.** It has no predecessor, so `delta_l2[0]` and
> `cos_prev[0]` are both zero *by construction*. Zero cosine means maximal
> `1 − cos_prev`, so token 0 scores as maximally unstable. Every derived quantity
> must exclude it. The seam score does; an earlier version did not, and reported
> the first token of every session as a large discontinuity.

### `top1_frac`, `top25_frac` — how concentrated was the update?
```
s = (H[t, l] − H[t−1, l])²
top1_frac[t, l]  = max(s) / Σs
top25_frac[t, l] = Σ top-k(s) / Σs,   k = ⌈0.25 · d_model⌉
```

> **Naming warning.** The UI calls the combination "focus" and the data field is
> `sparsity_norm`. Neither is MoE sparsity nor SAE feature sparsity. It is the
> concentration of the *token-to-token change* across residual dimensions. An
> interpretability reader will assume otherwise unless told.

### `logit_lens_entropy` — uncertainty at each layer
Shannon entropy, in nats, of the next-token distribution obtained by applying
the unembedding to the layer-`l` state (the "logit lens"). Shape `(T, L)`.

Layer means: 8.84 nats at layer 0, rising to a peak of **9.84 at layer 9**, then
falling to **1.65 at the final layer**. A uniform distribution over the 151,936-token
vocabulary is 11.93 nats.

The **rise before the fall** is present in all eight sessions. The model becomes
*more* uncertain through its early layers before committing — consistent with
early layers building representations rather than predicting.

### Also extracted, not yet displayed
`h_norm`, `logit_lens_rank`, `token_entropy`, `top3_ids`, `top3_probs`,
`actual_rank`, `actual_prob`. See "Unused data" below.

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
indistinguishable after normalisation. The inspector reports pre-clip values so
nothing is visible only through the clipped view.

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
measure: roughly 40% of tokens are pinned to zero by construction. It answers
"is this token unusual", not "how unusual is every token".

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
positions in the residual stream's principal subspace, **across sessions**.

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
- Mean over layers at the fork: **280.5**, against a typical cell magnitude of
  **299.2** — one token produces near-total separation immediately.
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
  per-layer "when did it decide" trace, sharper than entropy.
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
