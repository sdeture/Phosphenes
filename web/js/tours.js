/**
 * tours.js — Guided walkthrough definitions.
 *
 * Pure data. The app reads a step, applies its `state`, and shows its prose.
 *
 * ── Editorial rule for this file ────────────────────────────────────────
 *
 * Every quantitative claim in a caption has been measured on the actual
 * session data, and the measurement is recorded next to it in a `evidence`
 * field. If you change a caption to assert something new, measure it first and
 * put the number in `evidence`. The reproduction script for all of these is
 * `analysis/verify_tour_claims.py`; it prints each number this file cites and
 * exits non-zero if any has drifted.
 *
 * The reason for that discipline: this tool asks a viewer to trust their eyes
 * on a display whose whole premise is that pattern-recognition sees things
 * numerical readouts miss. If any caption points at a pattern that is not
 * really there, the premise collapses. Better to say less.
 *
 * Step fields:
 *   title     Short heading.
 *   body      HTML. One idea. Two sentences is usually plenty.
 *   look      What to actually look at, imperative. Rendered separately, in amber.
 *   evidence  Provenance of any number in `body` or `look`. Shown on demand.
 *   state     Applied before the step is shown:
 *               session   Session stem to load, if different.
 *               token     Playhead position.
 *               playing   Boolean.
 *               overlay   null | 'energy' | 'sparsity' | 'entropy'
 *               refCell   [token, layer] for reference-point mode, or null.
 *               textPanel Boolean.
 */

export const MAIN_TOUR = [
    {
        title: '',
        body: `The image represents the residual stream for every token and every layer of a
               recorded LLM transcript. Each vertical <b>column is a token</b> and each
               horizontal <b>row is a layer</b>, starting with layer 0 at the bottom.
               Brightness represents the magnitude of the residual stream at that token
               and layer.`,
        look: `Brightness rises as you look upward. The residual stream grows as it goes deeper.`,
        evidence: `Brightness is not the raw magnitude. Each layer is normalised to its own
                   5th-95th percentile range, pooled across the eight displayed sessions,
                   because the residual stream grows steeply with depth and a single global
                   scale would saturate the top of the image and flatten the bottom. What
                   survives that normalisation is a gentler climb: mean brightness in this
                   session runs 103 of 255 at layer 0 to 151 of 255 at layer 60.
                   <br><br>
                   The underlying growth, measured exactly on the full 5,120-dim state
                   (h_norm), Well-Read-Library session: 19.9 at layer 0, 1,308.8 at layer 60
                   — a 66x increase. The magnitude actually drawn is the 16-dim sketch's,
                   which reads 17.8 to 1,381.7, or 78x: a single fixed random projection has a
                   direction-dependent error that does not average away over tokens.`,
        state: { session: 'Dream_greedy_clean', token: 380, playing: false, overlay: null, refCell: null, textPanel: true },
    },
    {
        title: '',
        body: `This online prototype uses 16-dimensional JL sketches of the residual stream,
               but researchers with sufficient compute can easily adapt it to work with the
               full dimensional vector. By default, color represents the projection of the
               residual stream (or in this case the JL sketch) onto its top 3 principal
               components, calculated across all tokens, layers, and conversations.`,
        look: ``,
        state: { token: 193, playing: false, overlay: null, refCell: null, textPanel: true },
    },
    {
        title: 'Depth has horizontal structure',
        body: `Switch to the <b>focus</b> overlay: how concentrated each token-to-token
               update is across the 5,120 residual dimensions. Bright means the update was
               concentrated in a few of them; dark means it was spread thin across all of
               them.`,
        look: `Two horizontal bands, one low and one high, with a quiet trough between them.`,
        evidence: `Layer-mean sparsity (0.6*top1_frac + 0.4*top25_frac) has local maxima at
                   layer 9 (0.548) and layer 44 (0.575), with a trough at layer 28 (0.463).
                   The banding is a property of the network, not of this conversation:
                   it appears in all eight sessions.`,
        state: { token: 620, playing: false, overlay: 'sparsity', refCell: null, textPanel: false },
    },
    {
        title: '',
        body: `The <b>entropy</b> overlay applies a logit lens at every layer: if you decoded
               the next token from this layer's state, how uncertain would the answer be?
               Dark is committed, bright is undecided.`,
        look: `Uncertainty <i>rises</i> through the early layers before it falls. The model
               opens the question up before it closes it.`,
        evidence: `Layer-mean logit-lens entropy: 8.84 nats at layer 0, peaking at 9.84 nats at
                   layer 9, then falling to 0.998 nats at the output. A uniform
                   distribution over the 151,936-token vocabulary would be 11.93 nats.`,
        state: { token: 620, playing: false, overlay: 'entropy', refCell: null, textPanel: false },
    },
    {
        title: '',
        body: `Observe the dashed vertical lines demarcating turn boundaries. Turn boundaries
               are read from the chat template's im_start/im_end tokens.`,
        look: ``,
        state: { token: 2431, playing: false, overlay: null, refCell: null, textPanel: true },
    },
    {
        title: 'Where else does it do this?',
        body: `Reference-point mode recolours everything by <b>distance in JL space</b> from
               one cell you pick. Warm is similar, cool is different. The white cell is the
               reference.`,
        look: `Bands of warmth far from the reference — other moments, other layers, where
               the model's state resembles this one.`,
        evidence: `Distance is Euclidean in the 16-dimensional JL sketch, which approximately
                   preserves distances from the full 5,120-dimensional space. The colour
                   scale is set by the 1st and 95th percentiles of visible distances, so it
                   is relative to what is on screen.`,
        state: { token: 700, playing: false, overlay: null, refCell: [640, 38], textPanel: false },
    },
    {
        title: 'Now define your own axes',
        body: `Press <b>C</b>, then click cells to build each colour channel: click adds to
               the channel's <i>source</i> group, shift-click adds to its <i>contrast</i>
               group. Each channel becomes a projection onto mean(source) − mean(contrast).`,
        look: `Try red = cells inside the human's turn, contrasted against cells inside the
               model's. Then ask which of the model's cells still come out red.`,
        evidence: `Channels are Gram-Schmidt orthonormalised in the order R, G, B, so the
                   second and third axes carry only the part of your request that is
                   independent of the earlier ones. While you select, the display shades
                   cells by how much of their state is <i>not yet</i> explained by the axes
                   you have already chosen.`,
        state: { token: 700, playing: false, overlay: null, refCell: null, textPanel: false },
    },
];

/**
 * Bookmarks — jump targets offered outside the tour, in the scrubber's context menu.
 * Each is a moment where something is visible and checkable in the text.
 *
 * Token indices refer to the Well-Read-Library session (`Dream_greedy_clean`).
 */
export const BOOKMARKS = [
    { token: 9,    label: 'seam on "prompt"',   note: 'First content word of the human\'s opening question. Seam 1.00.' },
    { token: 73,   label: 'the fork',           note: 'The model writes "library". The other run was forced to write "sentient".' },
    { token: 193,  label: 'turn ends',          note: 'im_end closing the model\'s proposal. Seam 0.98.' },
    { token: 239,  label: 'the story begins',   note: 'Turn 4 opens: "In the quiet hours between sleep and waking..." Seam 0.00.' },
    { token: 445,  label: 'prose to list',      note: 'The colon before the invented book titles. Seam 1.00.' },
    { token: 1347, label: '"Transformer"',      note: 'The human starts explaining the model\'s own architecture to it. Seam 1.00.' },
    { token: 2431, label: 'instruction ends',   note: 'Last token of the longest human turn. Seam 1.00.' },
    { token: 2433, label: 'the self-report',    note: 'Turn 6: the model writes about what the story-writing was like.' },
];
