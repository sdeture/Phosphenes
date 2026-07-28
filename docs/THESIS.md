# Why point a nervous system at a language model

The argument behind Phosphenes, with the numbers worked through and the
objections stated.

**Summary of what follows.** A 32-billion-parameter transformer spends about
7 × 10¹⁰ floating-point operations producing one word. A human brain, on the
best available estimates, spends something like 10¹³–10¹⁴ operation-equivalents
per word. Those are **two to three orders of magnitude apart, and the model is
the smaller one.** They are not comparable, and this document does not claim they
are. What it claims is that the conclusion does not require them to be.

---

## 1. The model's number

Qwen3-VL-32B-Instruct is **dense** — not a mixture of experts, so every parameter
participates in every token. Its configuration gives hidden size 5,120 and 64
layers, which agrees with what I measured directly from the extracted activations.

The standard estimate of forward-pass cost per token is Kaplan et al. (2020),
Table 1:

```
C_forward ≈ 2N + 2 · n_layer · n_ctx · d_attn
```

The first term is the two FLOPs (a multiply and an add) each parameter
contributes; the second is attention over the context. At the ~3,000-token
contexts in this repository the attention term is **5–9%** of the total — real,
but not what dominates. For N = 32 × 10⁹ this gives

> **≈ 7 × 10¹⁰ FLOPs per token.**

Seventy billion arithmetic operations for one word. Two other framings of the
same fact, which happen to be more useful for what follows:

- The **state** the model moves through, for one 2,990-token conversation, is
  2,990 × 64 × 5,120 = **979,251,200 numbers**, about 1.96 GB at the precision it
  was computed in. That is the object Phosphenes displays.
- 7 × 10¹⁰ FLOPs is roughly what the fastest computer in the world could do in
  one second in the early 1990s. The model spends it on one word, then discards
  it.

---

## 2. The brain's number, and why it is not really a number

The reference work here is Joe Carlsmith's 2020 report for Open Philanthropy,
*How Much Computational Power Does It Take to Match the Human Brain?* Its
mechanistic-method range is

> **10¹³ – 10¹⁷ FLOP/s, with a median around 10¹⁵**, and Carlsmith's stated
> position that this is *"more likely than not… enough"*, with less than 10%
> credence on requirements above 10²¹ FLOP/s.

Two things about that number matter more than its value.

**It is a sufficiency budget, not a measurement.** Carlsmith is asking how much
computation would be *enough to match* the brain's task-performance, not how much
the brain *performs*. Those are different questions, and no amount of care makes
the second one answerable with current neuroscience. Anyone treating 10¹⁵ FLOP/s
as a measured property of brains has misread the source.

**The historical anchors disagree by nine orders of magnitude.** Moravec's
scale-up from retinal processing gives **10¹⁴** — and it should be cited as
*1998*, not 1988: the earlier *Mind Children* figure was 10¹³, and the two differ
because Moravec silently changed a scaling assumption from a "compromise value" of
10,000× to a volume-based 100,000×. (Both standard secondary sources get this
wrong and propagate the later number onto the earlier book.) Kurzweil's headline
**10¹⁶** is his own deliberate round-up from underlying estimates of 10¹⁴–10¹⁵.
Sandberg & Bostrom's whole-brain-emulation roadmap spans **10¹⁵** for an analog
population model, **10¹⁸** for a spiking network, **10²⁵** at the metabolome
level, and **10⁴³** for molecular simulation — depending entirely on what you
decide counts as the relevant level of description.

That spread is not sloppiness. It is the honest consequence of a question that has
no level-independent answer.

For the per-word figure, human silent reading averages **238 words per minute =
3.97 words/s** (Brysbaert 2019, meta-analysis of 190 studies, 95% CI 230–246).
Dividing a whole-brain rate by a reading rate is crude — most of the brain is not
reading — but it is the comparison people reach for, so here it is explicitly.

---

## 3. The comparison, stated honestly

| | per word |
|---|---|
| Qwen3-VL-32B-Instruct, dense, 3k context | **≈ 7 × 10¹⁰ FLOPs** |
| Human, Carlsmith median (10¹⁵ FLOP/s ÷ 3.97 words/s) | **≈ 2.5 × 10¹⁴** |
| Human, from synapse counts × measured cortical firing rates | **≈ 10¹²–10¹³** |

**The best-supported gap is 2–3 orders of magnitude — roughly 72× to 1,810×.**
Using Carlsmith's median rather than the firing-rate route gives about **3.4
orders of magnitude**. Taking the full defensible band on both sides gives
**1.4 to 6.4 orders of magnitude**.

So: **the model does 100 to 1,000 times less computation per word than a human
brain, on the estimates that are least unfavourable to the comparison, and
possibly a million times less.**

I want to be blunt about this, because the version of this argument that says
"comparable" or "the same order of magnitude" is the version I set out to write,
and the numbers did not support it. They do not support it. If you see that claim
made, including by me, it is wrong.

---

## 4. Why the conclusion survives anyway

The conclusion was never actually about parity. It is about which side of a
threshold we are on, and the threshold is not the brain — it is **the point past
which a person cannot follow a computation by reading it.**

Put the three quantities on a log scale together with something we do understand:

```
  10¹        10²        10⁶         10¹⁰·⁸        10¹³ – 10¹⁴
   │          │          │             │              │
 a hash    a regex   a JPEG       ONE WORD OF      one word of
 lookup     match    decode      A 32B MODEL       a human brain
   └──────────┴──────────┘             └──────────────┘
        legible by reading             not legible by reading
```

The transformer sits about **nine orders of magnitude above** the largest
computation a person can actually trace, and two or three below a brain. On this
scale it is overwhelmingly nearer the brain end, and — the operative point —
**it is on the far side of the line that matters.** Nobody reads 7 × 10¹⁰
operations. Nobody reads 979 million numbers.

Neuroscience has been on that side of the line since it started, and its response
was not to give up or to read faster. It was to build instruments that hand the
data to the visual system: EEG traces, spike rasters, fMRI heat maps,
spectrograms. Those are not simplifications for the public. They are how
professionals work, because a trained eye extracts structure from a raster that no
one extracts from the underlying table.

Interpretability is in the same position and mostly has not built that layer. It
has excellent tools for asking narrow questions with high precision — activation
patching, sparse autoencoders, circuit analysis — and comparatively little for the
question *"what is the shape of this whole episode, and where in it should I be
looking?"* That is a perception problem, and perception is the thing humans are
unreasonably good at and currently not using.

**The claim, then, in the form it can bear:**

> Per-word computation in current language models is far past the point of being
> readable, and within a few orders of magnitude of biological scale. In every
> other domain where that has been true, progress required instruments that
> engage the whole sensory apparatus — vision first, then binding across
> modalities, then trained intuition. Interpretability has barely started on
> that, and the fact that it is unbuilt is a bigger opportunity than any
> individual technique.

That claim does not need parity. It needs the threshold, and the threshold is
not in dispute.

---

## 5. Objections

**"FLOPs and synaptic events are not the same unit."** Correct, and this is the
deepest problem with the whole comparison. A floating-point multiply-add is a
well-defined operation; a synaptic event is an analog physical process whose
information content depends on what you think the neuron is computing. Carlsmith
spends a large part of his report on exactly this, which is why his output is a
range spanning four orders of magnitude rather than a number. **Treat the table in
§3 as an order-of-magnitude orientation, not a measurement.**

**"Most of the brain is not processing the current word."** Also correct. The
per-word figure divides total brain compute by a reading rate, and includes
homeostasis, vision, motor control and everything else running concurrently.
Restricting to cortex would cut it — Azevedo et al. put **19% of the brain's
neurons in the cerebral cortex** — and restricting to language-selective regions
would cut it much further, though published "language network is ~10% of cortex"
figures are localizer-threshold artefacts and should not be quoted as anatomy.
Every honest restriction moves the human number **down**, which narrows the gap
and helps the argument. I have not applied those corrections, because doing so
selectively is how one manufactures a favourable number.

**"One synapse may be worth many operations."** Dendritic computation, spike
timing, and neuromodulation could each make a single synapse worth far more than
one FLOP. This moves the human number **up**, and widens the gap. It is the
strongest objection in the direction that hurts.

**"Per-token cost excludes training."** Yes. Training a frontier model is
~10²³–10²⁵ FLOPs, and a human's development is not free either. Neither figure
belongs in a per-word comparison, and including one without the other would be
the error.

**"Serial depth differs enormously."** A 64-layer transformer performs 64
sequential transformations per token. A brain doing the same task has perhaps
10–20 synaptic steps available within a reaction time, but operates with vastly
more parallel width. Two systems can spend similar totals with completely
different computational structure, and structure is what interpretability is
actually about.

**"Cortical neurons fire far more slowly than people assume."** True, and it is
the main reason the firing-rate estimate lands so much lower than Carlsmith's
median. Average rates in the fractions-of-a-hertz-to-few-hertz band, rather than
the tens of hertz often assumed, follow from energy-budget arguments. The
re-derivation combining the lowest published rate estimates with corrected
per-spike energy costs **has not been done in the literature**, so the bottom of
the range is softer than it looks.

**"This is an argument for prettier pictures."** It is an argument for
instruments, which are judged by whether they let you find things you would not
otherwise find and then check them. So Phosphenes ships with the checks: the
guided tour states a measurement for every pattern it points at, 57 of those are
re-verified from source by `analysis/verify_tour_claims.py`, and the divergence
view runs a self-test on load and reports failure on screen. A visualisation that
cannot be checked is decoration, and the difference is testability, not taste.

---

## 6. Sources

- **Kaplan et al. (2020)**, *Scaling Laws for Neural Language Models*, Table 1 —
  forward-pass FLOP accounting. arXiv:2001.08361
- **Hoffmann et al. (2022)**, *Training Compute-Optimal Large Language Models* —
  the 2N convention in wide use. arXiv:2203.15556
- **Carlsmith (2020)**, *How Much Computational Power Does It Take to Match the
  Human Brain?*, Open Philanthropy — 10¹³–10¹⁷ FLOP/s, median 10¹⁵, and the
  sufficiency-versus-measurement distinction that most citations of it drop.
- **Brysbaert (2019)**, *How many words do we read per minute? A review and
  meta-analysis of reading rate*, *Journal of Memory and Language* — 238 wpm,
  190 studies, CI 230–246.
- **Moravec (1998)** — 10¹⁴. Note that *Mind Children* (1988, pp. 57–60) gives
  **10¹³** via a different scaling assumption. Do not cite the two as one figure.
- **Kurzweil (2005)**, *The Singularity Is Near* — 10¹⁶ cps, his own round-up
  from 10¹⁴–10¹⁵.
- **Sandberg & Bostrom (2008)**, *Whole Brain Emulation: A Roadmap*, FHI — the
  level-dependence, 10¹⁵ to 10⁴³.
- **Azevedo et al. (2009)** — neuron counts by region; 19% of neurons in cortex.
- **Attwell & Laughlin (2001)**; **Lennie (2003)**, *The cost of cortical
  computation* — energy budgets constraining mean firing rates.

### Provenance, stated because it should be

Numbers 1–4 above were verified against primary sources. Two caveats I am
carrying forward rather than hiding:

- The *Mind Children* page attributions and quotations came from **OCR text
  only**; page images were unreachable. Attribution was cross-checked two
  independent ways (running heads and the book's own index) and is
  high-confidence, but a stray OCR error inside a quoted sentence cannot be
  excluded. Verify against a physical copy before quoting verbatim in print.
- Several published criticisms of the brain-compute genre — by Hofstadter, Modis,
  and others — are referred to here in substance but **their specific wordings
  were not verified in this pass**, so none is quoted.

Two numbers would most repay hardening before this goes anywhere formal: whether
the two standard synapse-count sources are actually independent measurements (it
decides whether the low end of §3 rests on one study or two), and a page-image
check on Moravec.
