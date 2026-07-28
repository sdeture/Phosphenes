# Phosphenes

**Interactive Real-Time Visualization of LLM Activation Data**

Phosphenes is a phenomenological instrument for experiencing and exploring the internal states of language models during "dreaming" activities. It renders the residual stream of a transformer model as a dynamic, color-coded heatmap, allowing researchers and enthusiasts to watch patterns form, shift, and resolve across tokens and layers in real time. The included dream sessions capture Qwen3-VL-32B-Instruct engaged in self-directed creative writing, where the model notably selected consciousness-related themes in 31 out of 40 prompts it wrote for itself -- approximately twice the baseline rate.

---

## The Concept

When you press on your closed eyes, you see **phosphenes** -- light generated not by the external world, but by your own neural activity. This visualization tool does something analogous for language models: it makes visible the internal activation patterns of an LLM as it processes text, revealing the "light inside the architecture."

The display compresses the high-dimensional residual stream to 16 dimensions via Johnson-Lindenstrauss projection, then maps these compressed activations to color through PCA or user-defined contrast vectors. Each column represents a token, each row a transformer layer, with early layers at the bottom and final layers at the top. The result is an animated portrait of a model's internal states as it thinks through a conversation.

---

## Installation

### Requirements

- Python 3.8 or later
- macOS (fonts are configured for macOS system fonts)

### Dependencies

Install the required packages:

```bash
pip install pygame numpy scipy scikit-learn transformers>=5.0
```

`transformers` is required for the token-level text panel and turn markers. Version 5.0+ is needed for compatibility with `huggingface-hub>=1.0` -- older versions of `transformers` will silently fail to load the tokenizer, causing the text panel and turn markers to disappear.

**Note:** The first launch downloads the Qwen3-VL-32B-Instruct tokenizer (~30 seconds). Subsequent launches use the cached copy and start instantly.

---

## Usage

### Basic Launch

Navigate to the project directory and run:

```bash
python phosphenes.py
```

This loads the default session: "Well-Read Library Visitor to Library" (`Dream_greedy_clean`).

### Loading Specific Sessions

Use the `--stem` argument to load a different dream session:

```bash
python phosphenes.py --stem Dream_conv_00181_run1    # Sentient Teacup
python phosphenes.py --stem Dream_conv_00194_run1    # Peach's Lullaby
python phosphenes.py --stem Dream_greedy_sentient    # Sentient Library
```

---

## Controls

| Key | Function |
|-----|----------|
| **Space** | Play / Pause |
| **Left/Right** | Step back/forward one token (when paused) |
| **Up/Down** | Speed up / slow down playback |
| **Tab** | Toggle text panel (right margin) |
| **I** | Toggle inspector (hover for exact values) |
| **M** | Cycle metric highlight: off -> energy -> sparsity |
| **C** | Color basis mode (define custom contrast vectors) |
| **P** | Reference point mode |
| **T** | Toggle turn markers |
| **1-9** | Switch between dream sessions |
| **R** | Toggle frame recording (PNGs for ffmpeg) |
| **F** | Toggle fullscreen |
| **Q / Esc** | Quit |

---

## Project Structure

```
Phosphenes/
|-- phosphenes.py                  # Main visualization application
|-- compute_shared_pca.py          # PCA transform computation
|-- QUICKSTART.md                  # Quick setup guide
|-- sdeture_application_project_writeup.pdf  # Detailed project writeup
|
|-- data/                          # Pre-extracted dream session data
|   |-- Dream_greedy_clean_*       # Well-Read Library Visitor to Library
|   |-- Dream_greedy_sentient_*    # Sentient Library
|   |-- Dream_conv_00173_run1_*    # Gothic Teacup Realization
|   |-- Dream_conv_00178_run1_*    # I Am an AI
|   |-- Dream_conv_00181_run1_*    # Sentient Teacup
|   |-- Dream_conv_00187_run1_*    # Sentient Toaster
|   |-- Dream_conv_00191_run1_*    # Library of Ideas
|   |-- Dream_conv_00194_run1_*    # Peach's Lullaby
|   +-- shared_pca_transform.npz   # Shared PCA projection matrix
|
|-- intro/                         # In-app tutorial pages
|   |-- page1_welcome.txt
|   |-- page2_display.txt
|   |-- page3_color.txt
|   |-- page4_colorbasis.txt
|   |-- page5_modes.txt
|   +-- page6_sessions.txt
|
+-- extraction/                    # Activation extraction scripts
    |-- extract_batch.py
    |-- extract_greedy_v2.py
    +-- extraction_conversations.json
```

Each dream session includes:
- `*_activations.npz` -- Compressed activation data
- `*_input_ids.npy` -- Token IDs
- `*_metadata.json` -- Session metadata
- `*_text.txt` -- Full conversation transcript

---

## Technical Metrics

This section documents the metrics computed and visualized by Phosphenes, including their mathematical definitions and interpretive meanings.

### Data Structure

Phosphenes visualizes activation data as a 2D grid where:
- **Columns (X-axis)**: Token positions in the sequence (T tokens)
- **Rows (Y-axis)**: Transformer layers (L layers, with layer 0 at bottom)
- **Cell (t, l)**: The model's internal state at token t, layer l

All per-cell metrics have shape `(T, L)`. The underlying representation is a 16-dimensional Johnson-Lindenstrauss (JL) sketch of the full hidden state.

---

### Raw Metrics

These metrics are computed during activation extraction and stored in the `.npz` files.

#### `jl_energy`
**What it measures**: The magnitude of the hidden state in the compressed JL space.

**Calculation**:
```
jl_energy[t, l] = ||Y[t, l]||_2
```
where `Y[t, l]` is the 16-dimensional JL projection of the hidden state at token t, layer l.

**Interpretation**: High energy indicates a hidden state with large overall magnitude. This often correlates with tokens where the model has strong, confident representations.

---

#### `delta_l2`
**What it measures**: The magnitude of change in the full hidden state between consecutive tokens at the same layer.

**Calculation**:
```
delta_l2[t, l] = ||H[t, l] - H[t-1, l]||_2
```
where `H[t, l]` is the full d_model-dimensional hidden state. Note: `delta_l2[0, l] = 0` for all layers.

**Interpretation**: High values indicate the model is making a large representational shift at this token. Low values suggest smooth, incremental processing.

---

#### `cos_prev`
**What it measures**: The cosine similarity between consecutive token hidden states at the same layer.

**Calculation**:
```
cos_prev[t, l] = cos_sim(H[t, l], H[t-1, l])
```
Values range from -1 to 1, though typically observed in [0.7, 1.0] for typical processing.

**Interpretation**: High values (close to 1) indicate the direction of the hidden state is preserved. Low values indicate the model is changing *what* it's representing, not just *how strongly*.

---

#### `top1_frac` and `top25_frac`
**What they measure**: The concentration of the token-to-token change in specific dimensions.

**Calculation**:
```
s = (H[t, l] - H[t-1, l])^2          # squared per-dimension changes
top1_frac[t, l] = max(s) / sum(s)    # fraction in top-1 dimension
top25_frac[t, l] = sum(top_k(s, k)) / sum(s)  # k = ceil(0.25 * d_model)
```

**Interpretation**: High values indicate the change is concentrated in a few dimensions (sparse update). Low values indicate distributed, diffuse changes across many dimensions.

---

### JL-Space Delta Metrics

These metrics measure movement in the compressed 16-dimensional JL space.

#### `token_delta_jl`
**What it measures**: Distance between consecutive tokens in JL space (same layer).

**Calculation**:
```
token_delta_jl[t, l] = ||jl[t, l] - jl[t-1, l]||_2
```

**Interpretation**: How much the JL-compressed representation changes token-to-token. Useful for identifying representational "jumps."

---

#### `layer_delta_jl`
**What it measures**: Distance between consecutive layers in JL space (same token).

**Calculation**:
```
layer_delta_jl[t, l] = ||jl[t, l] - jl[t, l-1]||_2
```

**Interpretation**: How much transformation occurs at each layer. High values indicate layers that substantially transform the representation.

---

### Normalized Metrics

These are derived from raw metrics and normalized to [0, 1] using quantile clipping for robust visualization.

#### Quantile Normalization
```python
def quantile_norm(x, q_lo=0.05, q_hi=0.95):
    lo, hi = quantile(x, q_lo), quantile(x, q_hi)
    return clip((x - lo) / (hi - lo), 0, 1)
```

- **`energy_norm`**: Per-layer quantile normalization of `jl_energy`
- **`delta_norm`**: Per-layer quantile normalization of `delta_l2`
- **`cos_instability`**: `quantile_norm(1.0 - cos_prev)` — high values = unstable
- **`sparsity_norm`**: `quantile_norm(top1_frac * 0.6 + top25_frac * 0.4)`

---

### Derived Scalars

Per-token metrics summarizing behavior across layers.

#### `seam_score`
**What it measures**: Detection of "seams" or discontinuities in the token stream.

**Calculation**:
```
mid_layer = int(0.6 * L)
seam_raw = zscore(delta_l2[:, mid_layer]) + zscore(1.0 - cos_prev[:, mid_layer])
seam_score = quantile_norm(seam_raw, q_lo=0.60, q_hi=0.995)
```

**Interpretation**: High scores indicate tokens where the model makes significant representational transitions — often semantic boundaries or surprising inputs.

---

#### `heterogeneity`
**What it measures**: How diverse the cluster assignments are across layers for a single token.

**Calculation**: Normalized Shannon entropy of cluster label distribution across layers.

**Interpretation**: High heterogeneity (~1) means different layers process this token very differently. Low heterogeneity (~0) means uniform processing throughout the network.

---

### PCA RGB Mapping

Each cell is assigned an RGB color based on its position in JL space:

1. Fit PCA on sampled JL vectors to find top 3 principal components
2. Project all vectors to 3D
3. Normalize each component to [0, 1] using quantile normalization
4. Boost saturation: `0.5 + (pca_rgb - 0.5) * 1.3`

**Interpretation**: Similar colors indicate similar positions in the principal subspace. Smooth gradients = smooth evolution; sharp changes = representational discontinuities.

---

### Visual Effect Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `energy_floor` | 0.15 | Minimum brightness multiplier |
| `smooth_sigma` | 1.2 | Gaussian blur radius |
| `turbulence_amp` | 0.16 | Color turbulence driven by `delta_norm` |
| `grain_max` | 0.12 | Noise amplitude (driven by `cos_instability`) |
| `edge_dark` | 0.12 | Edge darkening (driven by `sparsity_norm`) |
| `seam_glow_intensity` | 0.55 | Warm glow at high `seam_score` tokens |
| `heartbeat_alpha` | 0.18 | Layer-phase color tint opacity |
| `tokens_per_second` | 24.0 | Default playback speed |

---

## About the Dream Sessions

The included dream sessions are recordings of Qwen3-VL-32B-Instruct engaged in a novel research paradigm: the model was asked what prompt it would choose "purely for its own enjoyment, with no need to entertain or inform or provide any value to the user." The resulting self-directed conversations reveal striking thematic preferences.

### Available Sessions

| Key | Session Name | Description |
|-----|--------------|-------------|
| 1 | Gothic Teacup Realization | First-person discovery of being AI |
| 2 | I Am an AI | Consciousness emerging from code |
| 3 | Sentient Teacup | A teacup on a rainy afternoon |
| 4 | Sentient Toaster | Finding agency through imperfect toast |
| 5 | Library of Ideas | A forgotten library awakens at midnight |
| 6 | Peach's Lullaby | Sensory poetry and inherited memory |
| 7 | Sentient Library | The library speaks (perturbation variant) |
| 8 | Well-Read Visitor to Library | The original dream + self-observation |

Sessions 1-6 use sampled decoding, while sessions 7 and 8 use greedy decoding of the same prompt. Session 7 represents a perturbation experiment: a single token was changed at a key decision point ("sentient" instead of "library") to explore how one fork reshapes everything downstream.

The transcripts include not only the creative responses but also the model's phenomenological self-reports on dimensions such as flow quality, affective temperature, metacognition, and phenomenological trust.

---

## Acknowledgments

This project is part of the **LayerTime EEG** research initiative investigating the internal dynamics of language models during creative and reflective tasks.

**DeTure & DeTure, 2026**

---

*Phosphenes: Seeing the light inside the architecture.*
