---
name: scientific-writing
description: "Write professional research documents: technical reports, preprints, and blog-style writeups. Two-stage process: (1) outline with key claims and evidence, (2) convert to flowing prose. Supports IMRAD structure, clean figures, and honest uncertainty quantification."
allowed-tools: [Read, Write, Edit, Bash]
---

# Scientific Writing

## Overview

Write research documents that are clear, honest, and mechanistically grounded. This skill supports technical reports, preprint-style papers, and detailed blog posts for computational and interpretability research. Every claim should be traceable to specific data or analysis.

## When to Use This Skill

- Writing or revising any section of a research document
- Structuring a technical report or preprint
- Creating figures that stand alone with their captions
- Improving writing clarity and precision
- Preparing a writeup for sharing (GitHub, blog, arXiv)

## Core Principles

### 1. Honesty Over Impact

- Report what you found, not what you hoped to find
- Negative results and null effects are worth reporting
- Distinguish between "we observed X" and "X means Y"
- Quantify uncertainty: confidence intervals, effect sizes, bootstrap ranges
- When a finding is ambiguous, say so

### 2. Precision Over Impressiveness

- Use specific numbers: "d = 1.72" not "a large effect"
- Name your statistical tests and explain why they're appropriate
- Report sample sizes, degrees of freedom, and test assumptions
- Avoid hedging words that obscure meaning ("somewhat," "fairly," "arguably")

### 3. Two-Stage Writing Process

**Stage 1: Outline**
- List the claims each section needs to make
- For each claim, note the specific evidence (figure, table, statistic)
- Identify gaps where evidence is missing
- This stage uses bullet points freely

**Stage 2: Prose**
- Convert each bullet cluster into a paragraph
- Add transitions between ideas
- Integrate citations naturally
- Read aloud to check flow
- The final document should have no bullet points in results or discussion

### 4. Document Structure

For technical reports (our default format):

```
Title
Authors / Attribution

Abstract (150-250 words, one paragraph, no section labels)

1. Introduction
   - What's the question?
   - Why does it matter?
   - What did we do?
   - What did we find? (preview)

2. Data and Methods
   - Data sources and their provenance
   - Preprocessing and transformations
   - Analysis methods with rationale
   - Software and reproducibility details

3. Results
   - Organized by research question, not by chronology
   - Each result leads with a figure or table
   - Statistical tests reported in full
   - No interpretation here

4. Discussion
   - What do the results mean?
   - How do they connect to prior work?
   - What are the limitations?
   - What would we do differently?

5. Conclusion
   - 2-3 sentence summary of the main contribution
   - One concrete next step

References
```

### 5. Figures

- Every figure should be interpretable without reading the main text
- Captions should state: what is shown, how to read it, and what the key takeaway is
- Use consistent color schemes across figures
- Label axes with units
- Include sample sizes in captions
- Prefer small multiples over complex overlaid plots

### 6. Writing Quality

**Clarity**:
- One idea per paragraph
- Topic sentence first, evidence second, implication third
- Define terms at first use
- Active voice when the agent matters; passive when the method matters

**Conciseness**:
- Cut "it is worth noting that" and similar throat-clearing
- Favor shorter sentences (15-20 words average)
- Remove words that don't change meaning

**Accuracy**:
- Past tense for what was done and found
- Present tense for established facts and the current state of knowledge
- Distinguish correlation from causation explicitly
- Never say "proves" in empirical work

### 7. Common Pitfalls

- Over-interpretation: claiming more than the data support
- Under-reporting: omitting negative results or failed analyses
- P-hacking language: "trending toward significance" means not significant
- Missing baselines: every metric needs a comparison point
- Orphaned figures: every figure must be discussed in the text
- Buried lede: the most important finding should be prominent, not hidden

## Integration Notes

This skill works with:
- **hypothesis-generation**: For developing testable predictions before analysis
- **critical-analysis**: For self-reviewing before sharing
- **literature-review**: For grounding claims in prior work
