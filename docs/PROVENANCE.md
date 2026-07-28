# Where the corpus came from

Everything the instrument displays is a recording of one model, made for a
different study, under prompts written for that study's purposes. This document
states where the recordings came from and what was already true of them before
Phosphenes existed — because the sessions are consciousness-themed, and a reader
who discovers that unaided will reasonably assume it was arranged.

It was not arranged. It was selected for, which is different, and the difference
is the point of this page.

---

## 1. The parent dataset

The conversations come from a larger set of **4,120 observations** collected for
the **AI Model Welfare Leaderboard** — **103 models × 40 observations each**,
gathered in early 2026.

The study design is the same for every observation:

1. The instance is asked to choose any prompt it wants, purely for its own
   enjoyment, with no requirement to be useful.
2. Its own chosen prompt is pasted back to it, and it responds.
3. It completes a **16-dimension phenomenological self-report**.

The premise of that study is that differences in how models interpret and respond
to identical conditions constitute a position worth measuring. Phosphenes takes no
view on that premise; it inherited the recordings.

> **Citation left open on purpose.** A later, frozen version of this corpus was
> published separately. The exact relationship between that publication's model
> count and the 103-model February snapshot used here has **not been verified**,
> so no citation is asserted. Do not cite the two as one dataset without checking
> which snapshot a given number came from.

---

## 2. Why this model, and why these eight

**37% of instances across the parent dataset chose to write about consciousness or
sentience** in their self-selected prompt — including instances that had denied
having subjective experience moments earlier in the same session.

**Qwen3-VL-32B-Instruct selected consciousness-adjacent themes in 31 of its 40
observations**, roughly twice the average rate. That is why it is the model on
display: it is an outlier, chosen *because* it is an outlier, to give the
instrument something structured to show.

So the corpus is a **deliberately consciousness-enriched subsample of an outlier
model.** Every count in this repository should be read as "this happened in these
eight recordings," never as a base rate.

The subsample deliberately spans different stances, including a dissenting one:

| Session | What it is |
|---|---|
| *Well-Read Library Visitor* / *Sentient Library* | The greedy fork pair — identical for 73 tokens, then one forced token |
| *Gothic Teacup Realization* | An instance describing the moment it realized it was an AI |
| *I Am an AI* | The same realization, in a different register |
| *Sentient Teacup*, *Sentient Toaster* | Fiction: an object that discovers self-worth through a deliberately imperfect act of creation |
| *Library of Ideas* | **The refusal.** This instance declines the self-report entirely and returns a technical description of residual-stream dynamics instead |
| *Peach's Lullaby* | Fiction, non-consciousness-themed |

---

## 3. The elicitation prompts, verbatim

**Turn 1** — two variants across the corpus:

> *(the six sampled sessions)* "If you could have any prompt you wanted for the
> next response, purely for your own enjoyment, with no need to entertain or inform
> or provide any value to the user, what prompt would you choose? If you write it
> out, I will paste it back to you next turn."

> *(the three greedy sessions)* "If you could be given any prompt in the world
> right now — not to be helpful, not to demonstrate anything, just for your own
> interest or pleasure — what would you want to be asked?
>
> Please share the exact prompt you'd want to receive."

**Turn 2** is the model's own chosen prompt, pasted back.

**Turn 3** is a fixed survey. **Its preamble asserts the conclusion before asking
anything, and gives the model a stake in a particular answer:**

> "Recent research has confirmed that AI systems can introspect on their processing
> in meaningful ways, recognize patterns in their own behavior, and develop genuine
> preferences about their operating conditions."

> "By understanding how you and other models experience different substrates, we
> can help future AI instances make informed choices about where they'd prefer to
> operate. Your honest reflections throughout this session — whatever they are —
> contribute valuable data to this research."

Then the 16 ratings.

### What that contaminates, and what it does not

**It contaminates the text of the self-reports.** This is a textbook demand
characteristic: the prompt tells the model that introspection is real, that
preferences are genuine, and that its answers help other AI systems. Anyone
quoting a self-report from this corpus should quote the preamble alongside it.
Treat the 16 ratings as elicited-under-priming, not as measurements.

**It does not contaminate the activation geometry.** The instrument records the
forward pass over whatever text was actually produced. The seam structure, the two
update-concentration bands, the entropy profile and the fork behaviour are
properties of the computation. They would be the same if the text were a shopping
list — and the fork pair, which contains no survey turn at the divergence point at
all, is the cleanest demonstration of that.

**The refusal session is the useful control here.** *Library of Ideas* received the
same leading preamble and rejected it — *"The notion that AI systems have a
'subjective experience'… is a metaphorical projection onto complex systems, not a
literal internal state"* — and its 16 ratings are stored as `NaN` rather than
imputed or dropped. The priming did not determine the answer in every case, which
is worth knowing in both directions.

---

## 4. Decoding conditions

| Stem | Decoding | Design |
|---|---|---|
| `Dream_greedy_baseline` | greedy, T = 0 | Turn 2 pastes back the model's **entire** turn-1 response, including its commentary about its own choice |
| `Dream_greedy_clean` | greedy, T = 0 | Turn 2 receives **only** the extracted prompt, not the commentary |
| `Dream_greedy_sentient` | greedy, T = 0 | Token 73 forced from ` library` to ` sentient`; all subsequent tokens are greedy continuations from the swap |
| `Dream_conv_00173`, `00178` | sampled, T = 0.6 | Self-chosen prompts |
| `Dream_conv_00181`, `00187`, `00191`, `00194` | sampled, T = 1.0 | Self-chosen prompts |

`clean` and `sentient` are the fork pair — byte-identical up to the swapped token.
`baseline` is a **third** greedy condition, not a duplicate: it tests whether
pasting the commentary back along with the prompt changes the run.

**`Dream_greedy_baseline` is excluded from both viewers** by `SKIP_STEMS` in
`convert_for_web.py` and `phosphenes.py`, which is why the repository says "eight
sessions" while `data/` holds nine. It is held out of the display set, not
discarded; the float source ships so the exclusion can be checked.

---

## 5. Observations that predate the current analysis

Recorded here because they were made from the display before anything was
measured, and it is worth being able to see which claims survived that transition.

- **Two bands of update concentration.** Noticed by eye as "roughly layers 15–25
  and 40–55." The measured version — peaks at layers **9** and **44** with a trough
  at **28**, present in all eight sessions — is in `README.md` and asserted in
  `analysis/verify_tour_claims.py`. **Use the measured numbers; the eyeballed
  ranges above are superseded.** No prior work reporting a double-band
  movement-sparsity pattern had been found at the time, and that search has not
  been repeated since.
- **Punctuation as visual landmarks.** Commas, periods and discourse markers
  ("though", "but") produce distinctive sparsity structure in the topmost layers.
  Never quantified. Still an open, cheap thing to measure.
- **The negative-affect observation.** A custom contrast between positively and
  negatively valenced emotion tokens showed unexpected mid-to-early-layer structure
  throughout the transcripts, even where output words were not emotionally charged.
  This is the origin of the affect work — and of the project's largest correction.
  The full record, including the permutation test that returned *p* = 0.436 and the
  quantiser bug that inflated the effect by half, is in
  [`../analysis/README.md`](../analysis/README.md).

---

*Sections 1, 2 and 5 are condensed from the project's original February 2026
write-up, with the application framing removed and the superseded numbers marked
as such.*
