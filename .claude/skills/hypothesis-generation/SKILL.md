---
name: hypothesis-generation
description: "Generate and evaluate testable hypotheses. Formulate predictions from observations, design analyses to test them, explore competing explanations, and specify what would falsify each hypothesis."
allowed-tools: [Read, Write, Edit, Bash]
---

# Hypothesis Generation

## Overview

Develop testable hypotheses from data and observations. This skill emphasizes falsifiability, competing explanations, and concrete predictions. It's designed for computational and interpretability research where the data is already collected and the question is what analyses to run.

## When to Use This Skill

- Developing hypotheses from preliminary exploration
- Designing analyses to test specific predictions
- Formulating competing explanations for an observation
- Planning what to measure and what would count as evidence
- Transitioning from exploratory to confirmatory analysis

## Workflow

### 1. Observation to Question

Start with a specific observation:
- "We observe X in the data"
- "X and Y appear correlated"
- "Model behavior Z is unexpected"

Convert to a precise question:
- "Does X occur reliably across conditions?"
- "Is the X-Y correlation causal, confounded, or spurious?"
- "What mechanism produces Z?"

### 2. Generate Competing Hypotheses

For every observation, produce at least 3 explanations:

**H1 (Interesting)**: The explanation you hope is true
**H2 (Boring)**: The mundane explanation (artifact, confound, baseline effect)
**H3 (Alternative)**: A different interesting explanation

Force yourself to take H2 seriously. Most exciting findings have boring explanations that need to be ruled out.

### 3. Specify Predictions

For each hypothesis, state:
- **If H is true, we should observe**: [specific, measurable prediction]
- **If H is false, we should observe**: [what falsification looks like]
- **Distinguishing test**: [an observation that separates H1 from H2 from H3]

Good predictions are:
- Quantitative when possible ("effect size > 0.5")
- Directional ("A should be greater than B")
- Conditional ("this should hold for X but not Y")
- Pre-registered before looking at the relevant data

### 4. Design the Analysis

For each distinguishing test:
- What data do you need?
- What statistical test is appropriate?
- What's the null distribution?
- What sample size gives you adequate power?
- What confounds need to be controlled?

### 5. Evaluate Hypothesis Quality

Rate each hypothesis on:

| Criterion | Question |
|-----------|----------|
| Testability | Can we test this with available data? |
| Falsifiability | What would disprove it? |
| Parsimony | Is this the simplest explanation? |
| Scope | How much does it explain? |
| Novelty | Does it offer new insight? |
| Mechanism | Does it explain *how*, not just *what*? |

### 6. Document and Commit

Before running analyses:
- Write down all hypotheses and predictions
- Record which tests are confirmatory vs. exploratory
- Note any researcher degrees of freedom
- Commit to reporting all results, not just significant ones

## Common Traps

- **Confirmation bias**: Designing tests that can only confirm, never falsify
- **HARKing**: Hypothesizing After Results are Known (reframing exploration as prediction)
- **Ignoring boring explanations**: The most interesting hypothesis isn't always the right one
- **Overly vague predictions**: "There should be a difference" is not falsifiable
- **Circular reasoning**: Using the same data to generate and test a hypothesis
- **Single explanation thinking**: Settling on one hypothesis without considering alternatives

## Output Format

When generating hypotheses, produce a structured document:

```markdown
## Observation
[What we see in the data]

## Research Question
[Precise question]

## Competing Hypotheses

### H1: [Name]
**Mechanism**: [How/why this would work]
**Prediction**: [Specific, testable]
**Falsification**: [What would disprove it]

### H2: [Name — the boring explanation]
**Mechanism**: [...]
**Prediction**: [...]
**Falsification**: [...]

### H3: [Name — alternative]
**Mechanism**: [...]
**Prediction**: [...]
**Falsification**: [...]

## Distinguishing Tests
[Analyses that separate the hypotheses]

## Analysis Plan
[Specific steps, tests, and criteria]
```
