# Phosphenes QuickStart Guide

Get the visualization running in under 2 minutes.

## Prerequisites

- Python 3.8 or later
- macOS (fonts are configured for macOS system fonts)

## Step 1: Install Dependencies

Open Terminal and run:

```bash
pip install pygame numpy scipy scikit-learn
```

Optional (enables token-level text display):
```bash
pip install transformers
```

## Step 2: Navigate to the Project

```bash
cd /path/to/Phosphenes
```

## Step 3: Launch

```bash
python phosphenes.py
```

A window will open showing the "Well-Read Library Visitor to Library" dream session by default.

## Basic Controls

| Key | What it does |
|-----|--------------|
| **Space** | Play / Pause the animation |
| **Left/Right arrows** | Step through tokens one at a time (when paused) |
| **Up/Down arrows** | Speed up / slow down playback |
| **Tab** | Show/hide the conversation text panel |
| **1-8** | Switch between different dream sessions |
| **Q** or **Esc** | Quit |


## Understanding the Display

- **Each column** = one token in the conversation
- **Each row** = one transformer layer (early layers at bottom, final layers at top)
- **Colors** = activation patterns mapped via PCA (similar colors = similar internal states)
- **Top bar** = speaker indicator (blue = user, amber = assistant)

## Explore Further

- Press **I** to enable the inspector (hover over cells for exact values)
- Press **M** to cycle through metric highlights (energy, sparsity)
- Press **C** to enter color basis mode (define your own contrast vectors)

See the full README.md for complete documentation.
