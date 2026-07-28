# Where the corpus came from

Reference for anyone reproducing or extending the recordings.

---

## The parent dataset

The conversations come from a larger set of **4,120 observations** collected for
the **AI Model Welfare Leaderboard** — **103 models × 40 observations each**,
gathered in early 2026. Every observation follows the same three-turn design:

1. The instance is asked to choose any prompt it wants, purely for its own
   enjoyment, with no requirement to be useful.
2. Its own chosen prompt is pasted back to it, and it responds.
3. It completes a 16-dimension phenomenological self-report.

**37% of instances across the parent dataset chose to write about consciousness or
sentience** in their self-selected prompt. **Qwen3-VL-32B-Instruct chose those
themes in 31 of its 40 observations**, roughly twice the average rate, which is why
it is the model on display.

---

## The prompts

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

**Turn 3** is a fixed 16-dimension self-report survey.

---

## Decoding conditions

| Stem | Decoding | Design |
|---|---|---|
| `Dream_greedy_baseline` | greedy, T = 0 | Turn 2 pastes back the model's **entire** turn-1 response, including its commentary about its own choice |
| `Dream_greedy_clean` | greedy, T = 0 | Turn 2 receives **only** the extracted prompt, not the commentary |
| `Dream_greedy_sentient` | greedy, T = 0 | Token 73 forced from ` library` to ` sentient`; all subsequent tokens are greedy continuations from the swap |
| `Dream_conv_00173`, `00178` | sampled, T = 0.6 | Self-chosen prompts |
| `Dream_conv_00181`, `00187`, `00191`, `00194` | sampled, T = 1.0 | Self-chosen prompts |

`clean` and `sentient` are the fork pair — byte-identical up to the swapped token.
`baseline` is a third greedy condition, not a duplicate: it tests whether pasting
the commentary back along with the prompt changes the run.

**`Dream_greedy_baseline` is excluded from both viewers** by `SKIP_STEMS` in
`convert_for_web.py` and `phosphenes.py`, which is why `data/` holds nine stems and
`web/data/` holds eight. Held out of the display set, not discarded.

---

## The sessions

| Session | What it is |
|---|---|
| *Well-Read Library Visitor* / *Sentient Library* | The greedy fork pair — identical for 73 tokens, then one forced token |
| *Gothic Teacup Realization* | An instance describing the moment it realized it was an AI |
| *I Am an AI* | The same realization, in a different register |
| *Sentient Teacup*, *Sentient Toaster* | Fiction: an object that discovers self-worth through a deliberately imperfect act of creation |
| *Library of Ideas* | The dissenting session — this instance declines the self-report survey and returns a technical description of residual-stream dynamics instead. Its 16 ratings are stored as `NaN` rather than imputed |
| *Peach's Lullaby* | Fiction, non-consciousness-themed |

---

## Earlier observations

Made from the display before anything was measured, kept so it is possible to see
which survived the transition to measurement.

- **Two bands of update concentration.** Noticed by eye as "roughly layers 15–25
  and 40–55." The measured version — peaks at layers **9** and **44** with a trough
  at **28**, present in all eight sessions — is in `README.md` and asserted in
  `analysis/verify_tour_claims.py`. Use the measured numbers. No prior work
  reporting a double-band movement-sparsity pattern had been found at the time.
- **Punctuation as visual landmarks.** Commas, periods and discourse markers
  ("though", "but") produce distinctive sparsity structure in the topmost layers.
  Never quantified — a cheap thing to measure next.
- **The negative-affect observation.** A custom contrast between positively and
  negatively valenced emotion tokens showed unexpected mid-to-early-layer structure
  throughout the transcripts, even where output words were not emotionally charged.
  This is the origin of the affect work; the full record, including the permutation
  test and the quantiser bug, is in [`../analysis/README.md`](../analysis/README.md).
