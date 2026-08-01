# Phosphenes

A prototype of a mechanistic interpretability visualization tool for
observational science and hypothesis development.

![The whole conversation: 2,990 tokens across 64 layers of Qwen3-VL-32B-Instruct](docs/img/overview.png)

*One conversation, complete. Each column is a token, each row a transformer
layer, layer 0 at the bottom. Vertical rules are turn boundaries — blue where the
human begins, amber where the model does. Colour is the cell's position in the
top-3 principal subspace of its 16-dimensional sketch: it is an **embedding, not
a scale**, so similar colours mean nearby internal states and there is no colour
bar to give. Brightness is activation magnitude.*

---

**The argument, with the compute comparison worked through and its
counterarguments stated: [the about page](web/about.html).**

---

## Try it

The web viewer needs no dependencies beyond a local HTTP server. It will **not**
work from `file://` — it fetches session data, and ES modules and `fetch` both
require a real origin.

```bash
git clone <this repo> && cd Phosphenes
python3 -m http.server 8899 --directory web
# open http://127.0.0.1:8899
```

It opens on a one-line description, then offers a **guided tour** — seven steps, each one a
specific thing to look at with the measurement behind it one click away. Take the
tour first; the tool is dense and the tour is the intended entry point.

The desktop version has a few extras the web port lacks (PNG frame recording for
video, fullscreen, per-frame Gaussian smoothing):

```bash
pip install -r requirements.txt
python phosphenes.py                       # or --stem Dream_conv_00181_run1
```

First launch downloads the Qwen3-VL-32B-Instruct **tokenizer only** (~30s, no
weights).

---

## What the instrument shows that is checkable

Every number here is recomputed from source by
[`analysis/verify_tour_claims.py`](analysis/verify_tour_claims.py) (153
assertions, non-zero exit on drift).

![Three layer profiles](docs/img/layer_profiles.png)

**A turn *ending* is violent; a turn *beginning* is nothing.** The seam score —
built from mid-network movement and direction change, with no knowledge of the
text — averages **0.780** at `<|im_end|>` tokens against **0.110** elsewhere
(**7.1×**, range 6.2–7.8×), and **0.004** at `<|im_start|>`. `im_end > im_start`
in **8 of 8** sessions. At `<|im_end|>` the prediction problem changes completely;
by `<|im_start|>` the handover is already committed. Seams also land on
within-turn structure — a colon introducing a list, a topic pivot — none of which
was labelled.

**Depth has horizontal structure.** Update concentration has two bands, at layers
**9** and **44**, with a trough at **28**. Present in all eight sessions.

**The model opens the question before it closes it.** Logit-lens entropy *rises*
from 8.84 nats at layer 0 to a peak of **9.84 at layer 9**, then falls to **0.998**
at the output — against 11.93 for a uniform guess over the vocabulary. Present in
all eight sessions.

**Activation magnitude grows ~66×** with depth, 19.9 → 1,308.8, measured on the
full 5,120-dimensional state. Visible as brightness, with nothing plotted —
though the brightness channel is driven by the 16-dimensional sketch, which reads
the growth ~18% steeper (77.7×) because a single fixed random projection has a
direction-dependent error that does not average away.

---

## Reproducing

```bash
python analysis/verify_tour_claims.py      # 153 assertions: tour, compute, prose
python analysis/make_figures.py            # the figures in this README
python compute_shared_pca.py               # refit the shared PCA basis
python convert_for_web.py                  # rebuild the web bundles
```

---

## The sessions

Eight recordings of Qwen3-VL-32B-Instruct asked what prompt it would want "purely
for its own enjoyment, with no need to entertain or inform or provide any value to
the user," then given its own answer back, then asked to reflect. Across the
larger study these were drawn from, this model chose consciousness-related themes
in **31 of 40** self-directed conversations, roughly twice the rate of the average
model in that study.

Sessions 1–2 are the greedy fork pair (T = 0). Sessions 3–4 use T = 0.6, sessions
5–8 use T = 1.0. A **ninth** recording, `Dream_greedy_baseline`, ships in `data/`
but is deliberately excluded from both viewers (`SKIP_STEMS`): it is a third greedy
condition testing whether pasting the model's *commentary* back along with its
chosen prompt changes the run. Held out of the display set, not discarded.

The eight sessions span different stances, including a dissenting one: in *Library
of Ideas* the model declines the self-report survey and returns a technical
description of residual-stream dynamics instead.

---

## Credits

**DeTure & DeTure, 2026.** Part of the LayerTime EEG research programme.

MIT licensed — see [LICENSE](LICENSE).
