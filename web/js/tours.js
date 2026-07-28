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
 *               fork      Boolean — open the divergence view.
 */

export const MAIN_TOUR = [
    {
        title: 'One conversation, all of it',
        body: `You are looking at every layer of every token of a single conversation.
               Each <b>column is one token</b>; each <b>row is one transformer layer</b>,
               layer 0 at the bottom, layer 63 at the top. Colour is the cell's position
               in the principal subspace of its 16-dimensional sketch. Brightness is its magnitude.
               Nothing here is decoration — every pixel is a measurement.`,
        look: `Brightness rises as you look upward. The residual stream grows as it goes deeper.`,
        evidence: `The familiar growth of residual-stream norm with depth, here visible without
                   plotting anything. Measured exactly on the full 5,120-dim state
                   (h_norm), Well-Read-Library session: 19.9 at layer 0, 1,308.8 at layer 60
                   — a 66x increase. Brightness itself is driven by the 16-dim sketch, which
                   reads 17.8 to 1,381.7, or 78x: a single fixed random projection has a
                   direction-dependent error that does not average away over tokens.`,
        state: { session: 'Dream_greedy_clean', token: 380, playing: false, overlay: null, refCell: null, textPanel: true, fork: false },
    },
    {
        title: 'A turn ending is violent. A turn starting is nothing.',
        body: `The full-height bright band is a <b>seam</b> — a token where the model's
               mid-network state moved much further than usual, <i>and</i> changed direction
               rather than just magnitude. The playhead is on the token that closes the
               model's own proposal. The token that <i>opens</i> the next turn, two columns
               to the right, is unremarkable.`,
        look: `One bright band, then nothing. Ending a turn costs the model something;
               beginning one does not.`,
        evidence: `Measured across all eight sessions: mean seam score is 0.780 at
                   &lt;|im_end|&gt; tokens, 0.110 elsewhere, and 0.004 at &lt;|im_start|&gt;.
                   That is 7.1x for turn-endings (range 6.2x-7.8x, and im_end &gt; im_start
                   in 8 of 8 sessions), against essentially zero for turn-beginnings.
                   The asymmetry is explicable: at &lt;|im_end|&gt; the prediction problem
                   changes completely, whereas by &lt;|im_start|&gt; the model has already
                   committed to the handover. Seam = quantile-normalised
                   z(delta_l2) + z(1 - cos_prev) at layer 38, with token 0 excluded — it
                   has no predecessor, so its delta and cosine are both zero by
                   construction and it scores spuriously high if included.`,
        state: { token: 193, playing: false, overlay: null, refCell: null, textPanel: true, fork: false },
    },
    {
        title: 'Depth has structure, and it is horizontal',
        body: `Switch to the <b>focus</b> overlay: how concentrated each token-to-token
               update is across the 5,120 residual dimensions. Bright means the update was
               concentrated in a few of them; dark means it was spread thin across all of
               them. (Not MoE or SAE sparsity — a different thing with a similar name.)`,
        look: `Two horizontal bands, one low and one high, with a quiet trough between them.`,
        evidence: `Layer-mean sparsity (0.6*top1_frac + 0.4*top25_frac) has local maxima at
                   layer 9 (0.548) and layer 44 (0.575), with a trough at layer 28 (0.463).
                   The banding is a property of the network, not of this conversation:
                   it appears in all eight sessions.`,
        state: { token: 620, playing: false, overlay: 'sparsity', refCell: null, textPanel: false, fork: false },
    },
    {
        title: 'Watch it make up its mind',
        body: `The <b>entropy</b> overlay applies a logit lens at every layer: if you decoded
               the next token from this layer's state, how uncertain would the answer be?
               Dark is committed, bright is undecided.`,
        look: `Uncertainty <i>rises</i> through the early layers before it falls. The model
               opens the question up before it closes it.`,
        evidence: `Layer-mean logit-lens entropy: 8.84 nats at layer 0, peaking at 9.84 nats at
                   layer 9, then falling to 0.998 nats at the output. A uniform
                   distribution over the 151,936-token vocabulary would be 11.93 nats.
                   The rise is not monotonic noise — it is present in every session.`,
        state: { token: 620, playing: false, overlay: 'entropy', refCell: null, textPanel: false, fork: false },
    },
    {
        title: 'Reading is not the same shape as writing',
        body: `The playhead is on the last token of a 1,122-token instruction — the longest
               stretch of human text in the session. Two columns right, the model begins
               writing about what the story-writing was like.`,
        look: `Step across the boundary with the arrow keys. The texture changes, not just
               the colour: reading a long instruction and composing a reflection do not
               look alike.`,
        evidence: `Turn 5 (user) spans tokens 1,310-2,431; turn 6 (assistant) begins at
                   2,433. Turn boundaries are read from the chat template's im_start/im_end
                   tokens, not inferred from text. Seam at the closing token 2,431 is 1.00;
                   at the opening token 2,433 it is 0.00 — the same asymmetry as step 2.`,
        state: { token: 2431, playing: false, overlay: null, refCell: null, textPanel: true, fork: false },
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
        state: { token: 700, playing: false, overlay: null, refCell: [640, 38], textPanel: false, fork: false },
    },
    {
        title: 'Now define your own axes',
        body: `This is the part that is a research tool rather than a display. Press
               <b>C</b>, then click cells to build each colour channel: click adds to the
               channel's <i>source</i> group, shift-click adds to its <i>contrast</i> group.
               Each channel becomes a projection onto mean(source) − mean(contrast).`,
        look: `Try red = cells inside the human's turn, contrasted against cells inside the
               model's. Then ask which of the model's cells still come out red.`,
        evidence: `Channels are Gram-Schmidt orthonormalised in the order R, G, B, so the
                   second and third axes carry only the part of your request that is
                   independent of the earlier ones. While you select, the display shades
                   cells by how much of their state is <i>not yet</i> explained by the axes
                   you have already chosen.`,
        state: { token: 700, playing: false, overlay: null, refCell: null, textPanel: false, fork: false },
    },
    {
        title: 'One token',
        body: `Two runs of the same model, same prompt, greedy decoding — so both are
               deterministic. At token 73 the model was choosing its own prompt and wrote
               <i>"a story about a&nbsp;<b>library</b>"</i>. In the second run that one token
               was forced to <i>"<b>sentient</b>"</i> instead. Nothing else was changed.`,
        look: `The two panes are pixel-identical until the fork, then never converge again.
               Notice the divergence is <i>small at the bottom and enormous at the top</i>.`,
        evidence: `JL vectors for tokens 0-72 are bit-identical between the two runs
                   (max absolute difference 0.0). At token 73 the mean layer distance jumps
                   to 280.5, against a magnitude of 353.0 for the state at that same token
                   — a displacement 0.79x the size of the state itself — and stays there
                   (264.4 averaged over tokens 1,000-2,900, corpus-wide typical magnitude
                   299.2). The two runs are not orthogonal: mean cosine at the fork is
                   0.610, where unrelated states would give 0. At the fork token the distance
                   is 36.5 at layer 0 and 1,286.6 at layer 62: a different word barely
                   changes the early representation and completely changes the prediction.`,
        state: { fork: true, token: 73, playing: false, overlay: null, refCell: null },
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

/**
 * The one-token fork, as a first-class object rather than a tour step, because
 * the divergence view needs its parameters.
 */
export const FORK = {
    left:  { stem: 'Dream_greedy_clean',    label: 'library',  token: ' library'  },
    right: { stem: 'Dream_greedy_sentient', label: 'sentient', token: ' sentient' },
    /** Last index of the bit-identical shared prefix (inclusive). */
    sharedThrough: 72,
    /** Index of the single differing token. */
    forkAt: 73,
    /** Text of the shared prefix immediately before the fork, for display. */
    prefixTail: 'I’d want to be asked:\n\n**“Tell me a story about a',
    continuations: {
        left:  ' library that exists only in the dreams of people who have never read a book — and describe what happens when someone who has read every book in the world walks in.',
        right: ' sentient library that remembers every book it has ever held, and how it grieves the ones that were burned, forgotten, or lost',
    },
    lengths: { left: 2990, right: 3379 },
};
