# Why point a nervous system at a language model

The argument behind Phosphenes, with the numbers worked through and the
objections stated.

**Summary of what follows.** "Computation per word" is not one number on either
side. For a language model it depends on an accounting convention worth **four to
six orders of magnitude** — whether you count the marginal cost of generating one
more token with a warm cache (~10¹¹ FLOPs for a 32B model) or the cost of
processing the whole context that word was conditioned on (up to ~10¹⁷ at frontier
scale and deep context). For a human it depends on what you count as computation,
and those candidates span **sixteen** orders of magnitude, from ~10 bits/s of
conscious throughput to 10¹⁷ FLOP/s of total neural activity.

The honest result is therefore **overlap, not equality and not a clean gap** — and
which end of each range applies depends on the question. This document states the
conventions explicitly, because doing so is most of the work, and because a version
of this argument that quietly picks one number from each side can be made to come
out any way you like.

**Correction, and it is mine.** An earlier draft of this file asserted flatly that
per-word computation is "two to three orders of magnitude apart, model smaller,"
and told the reader that the comparable-scale claim was wrong wherever they saw it.
That was overconfident. It compared a *marginal-with-cache* model figure against a
*total-neural-activity* human figure — the smallest available number on one side
against the largest on the other — without saying so. Retracted below.

**On the corpus.** This document argues about computation, not about the eight
recordings. Where they came from, the prompts, and the decoding conditions are in
[`PROVENANCE.md`](PROVENANCE.md).

---

## 1. The model's number, and the convention it depends on

Qwen3-VL-32B-Instruct is **dense** — not a mixture of experts, so every parameter
participates in every token. Its configuration gives hidden size 5,120 and 64
layers, which agrees with what I measured directly from the extracted activations.

Forward-pass cost per token, Kaplan et al. (2020), Table 1:

```
C_forward ≈ 2N + 2 · n_layer · n_ctx · d_attn
```

The first term is the two FLOPs — a multiply and an add — each parameter
contributes; the second is attention over the context. At the ~3,000-token
conversations here the attention term is only **3%** of the total.

That gives one figure. It is not the only one.

### The two conventions

**(a) Marginal, cache warm.** Producing one more token, given that the context has
already been processed and its keys and values cached, costs `2N` plus attention
against the cache. For this model: **≈ 6.6 × 10¹⁰ FLOPs per token.** This is the
right number for the Phosphenes sessions, where the model writes most of a
2,990-token conversation and each token is computed once.

**(b) Full context.** The cost of the forward pass over the entire context a word
was conditioned on, is `n_ctx × 2N`. At frontier scale and deep context this is
enormous:

| model / context | full-context cost per word |
|---|---|
| ~25 B params, 10k context | 5.0 × 10¹⁴ (0.5 PFLOP) |
| ~70 B, 32k | 4.5 × 10¹⁵ |
| ~350 B, 70k | 4.9 × 10¹⁶ (~50 PFLOP) |
| ~400 B, 200k | 1.6 × 10¹⁷ |

**Both are real quantities and neither is a mistake.** They answer different
questions: (a) is what it costs to emit one more word, (b) is how much computation
stands behind that word. The gap between them is exactly the context length —
**four to six orders of magnitude.**

The situation decides which applies. A model reading a 200,000-token document and
answering in ten words spends ~1.3 × 10¹⁵ FLOPs per output word, because almost all
the work is the reading. A model writing a 3,000-token conversation largely by
itself spends ~7 × 10¹⁰ per token, because nothing is recomputed. Same architecture,
five orders of magnitude apart, no arithmetic error anywhere.

> Anyone quoting a single "FLOPs per word" figure for language models without
> saying which convention they used has left out the most important part. I did
> exactly that in the first version of this document.

### Two other framings of the same fact

- The **state** the model moves through, for one 2,990-token conversation, is
  2,990 × 64 × 5,120 = **979,763,200 numbers**, about 1.96 GB at the precision it
  was computed in. That is the object Phosphenes displays, and it is
  convention-free.
- The **state accessible per word** — residual-stream width × context — is about
  **3.2 × 10⁹ bits** at a 25k context. Hold onto that; §3 compares it to something.

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

## 3. The comparison: two ranges that overlap

Both sides are ranges. Setting them out at once is the only honest presentation,
because any single pair of numbers can be chosen to give any answer.

| per word (~1 s) | | order |
|---|---|---|
| **Human** — conscious / behavioural throughput | ~10 bits/s (Zheng & Meister) | 10¹ bits |
| **Human** — afferent sensory bandwidth, 30M sensory neurons | ~3 × 10⁹ bits/s | **10⁹ bits** |
| **Human** — total neural computation | 10¹³–10¹⁷ FLOP/s (Carlsmith) | 10¹³–10¹⁷ |
| **LLM** — state accessible per word, 25k context | residual width × context | **10⁹ bits** |
| **LLM** — marginal generation, 32B, cache warm | 2N + attention | 10¹¹ |
| **LLM** — full-context processing, frontier scale, deep context | n_ctx × 2N | 10¹⁴–10¹⁷ |

Three things fall out, and only the third is contestable.

**The information rows match, closely, and this is the strongest correspondence
available.** Human afferent bandwidth ~3 × 10⁹ bits/s against ~3.2 × 10⁹ bits of
state the model can reach per word — about **1 : 1**, and it does not depend on the
FLOPs convention at all. Both are channel-capacity measures. (Derivation on both
sides: *Comparing Human Sensory Bandwidth to LLM Input Bandwidth*, December 2025.)

**The human side is itself about sixteen orders of magnitude wide.** Conscious
throughput is ~10 bits/s; afferent bandwidth is ~3 × 10⁹ bits/s; total neural
activity is 10¹³–10¹⁷ FLOP/s. So "how much computation does a human do per word"
has no single answer either, and an argument that reaches for the *largest* human
number while using the *smallest* model number is not measuring anything — it is
choosing an outcome. That is what my first draft did.

**On compute, the ranges overlap at the top and separate at the bottom.**
Full-context processing at frontier scale (10¹⁴–10¹⁷) sits **inside** the human
neural band (10¹³–10¹⁷). Marginal generation for a 32B model (10¹¹) sits about two
orders **below** it. Both comparisons are valid; they are about different things.

For **this repository specifically**, convention (a) applies — 32B model, 3k
conversations, the model writing most of it — so ~7 × 10¹⁰ FLOPs per token is the
correct figure for what Phosphenes displays, and it is at the small end of the
model range. That is worth saying plainly rather than borrowing a frontier-scale
number to make the artifact look bigger.

### A framing error in the deflationary direction

There is one more thing wrong with "the model does 1,000× less computation, so the
comparison fails," beyond the mismatched conventions: **it treats compute as a
proxy for capability, and then reads a smaller number as a deficit.**

But the model is doing the task. It writes coherent prose, holds a 3,000-token
thread, and produces a structured account of its own processing. If it does that
with two or three orders of magnitude less computation per word than a brain
spends, the honest description of that gap is **efficiency, not shortfall.** "Does
comparable work with less compute" is not a weaker claim than parity — for some
purposes it is a stronger one.

Nothing here settles whether the work really is comparable; that is a question
about behaviour, not about FLOPs, and this document does not address it. The point
is narrower: the *direction* in which one reads a compute difference is an
interpretive choice, not a measurement, and the deflationary reading has no better
claim on neutrality than the inflationary one. Both need arguing.

**And for the argument this document is actually making, all three readings give
the same answer.** Whether per-word computation is 10¹¹ or 10¹⁵, whether it reads
as efficient or deficient, it is at minimum nine orders of magnitude past the point
where a person can follow it by reading. That is the premise the instrument rests
on, and it is the one part of this that no convention choice can move.

---

## 4. Why the conclusion survives anyway

The conclusion was never actually about parity. It is about which side of a
threshold we are on, and the threshold is not the brain — it is **the point past
which a person cannot follow a computation by reading it.**

Put the three quantities on a log scale together with something we do understand:

```
   10¹        10²        10⁶        10¹¹     10¹⁴     10¹⁷
    │          │          │          │        │        │
 a hash     a regex    a JPEG        ├─────────────────┤   a language model
 lookup      match     decode         marginal → full context
                                           ├───────────┤   a human brain
    └──────────┴──────────┘          └──────────────────┘
       legible by reading              not legible by reading
```

Both bands sit **at least nine orders of magnitude above** the largest computation
a person can actually trace, and they overlap each other. Where exactly the model
falls inside its band depends on the convention, and §3 is where that gets argued —
but it does not need settling here. On a log scale that starts at a hash lookup,
every defensible reading of the model's number and every defensible reading of the
brain's lands on the far side of the same line. **That line, not parity, is what
the instrument rests on.** Nobody reads 7 × 10¹⁰ operations. Nobody reads 979
million numbers.

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
Every honest restriction moves the human number **down**. I have not applied those
corrections, because doing so selectively is how one manufactures a favourable
number.

**"One synapse may be worth many operations."** Dendritic computation, spike
timing, and neuromodulation could each make a single synapse worth far more than
one FLOP. This moves the human number **up**, and is the strongest objection in
the direction that hurts a comparability claim.

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
the main reason bottom-up estimates built from synapse counts × measured firing
rates land around **10¹²–10¹³ FLOP/s** — well below Carlsmith's 10¹⁵ median, and
below the bottom of the range in §3. Average rates in the
fractions-of-a-hertz-to-few-hertz band, rather than
the tens of hertz often assumed, follow from energy-budget arguments. The
re-derivation combining the lowest published rate estimates with corrected
per-spike energy costs **has not been done in the literature**, so the bottom of
the range is softer than it looks.

**"This is an argument for prettier pictures."** It is an argument for
instruments, which are judged by whether they let you find things you would not
otherwise find and then check them. So Phosphenes ships with the checks: the
guided tour states a measurement for every pattern it points at;
`analysis/verify_tour_claims.py` recomputes **172 assertions** from source —
the tour's numbers, every compute figure in this document, and a regression test
on the argument's own wording, added after a retracted claim survived in one
surface and not the other. The divergence view runs a self-test on load and
reports failure on screen. A visualisation that cannot be checked is decoration,
and the difference is testability, not taste.

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
decides whether the bottom-up 10¹²–10¹³ estimate in §5 rests on one study or
two), and a page-image check on Moravec.
