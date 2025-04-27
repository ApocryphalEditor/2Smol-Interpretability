# 2Smol Interpretability Suite: SRM, BCR, and Beyond

This repository forms a living collection of tools, experiments, and frameworks designed for machine learning interpretability. It includes:

*   **Semantic Resonance Mapping (SRM):** A method for projecting and measuring activation structures within latent space.
*   **Baseline Collapse Recovery (BCR):** A specific study focusing on semantic collapse during generation and the potential for intervention-driven recovery.
*   **Interactive Chat Interfaces:** GUI tools for real-time exploration of neuron interventions.

At its heart, this project asks:
_Can we map the hidden geometries of machine cognition and intervene meaningfully — even without formal ML training?_

---

## Core Techniques

### Semantic Resonance Mapping (SRM)

SRM, originally inspired by the Spotlight Resonance Method developed by George Bird (see: [Spotlight-Resonance-Method](https://github.com/gwb/Spotlight-Resonance-Method) - *[Optional: Link if relevant]*), explores activation spaces through three main stages:

1.  **Defining a Basis Plane:** Selecting two basis vectors (often representing specific neurons or averaged activations from prompt groups) to define a 2D plane within the model's high-dimensional activation space. This plane acts as an *interpretive frame*.
2.  **Projection:** Projecting other activation vectors (captured during model processing) onto this 2D basis plane.
3.  **Measurement:** Analyzing the angle (directional meaning) and magnitude (*resonance* or alignment strength) of these projections relative to the basis vectors, often by sweeping a probe vector through the plane.

SRM can reveal activation geometries, directional drifts (like the [**Grey Vector**](https://github.com/ApocryphalEditor/2Smol-Interpretability/blob/main/Documentation/SRM%20Suite/A%20Multi-Basis%20Projection%20Framework%20for%20Latent%20Alignment%20Analysis%20in%20Language%20Models.pdf)), and hidden semantic tendencies relative to the chosen basis.

SRM leverages two key experimental concepts developed in this project:

*   **The Raccoon Schemas:** A set of six structured experimental designs (`SRM Schemas 01-06.pdf`) that systematically test how meaning shifts depending on neuron focus, prompt framing, clamp intensity, and basis choice. These schemas range from probing simple neuron influence (Schema 01) to sophisticated basis-invariant meaning persistence tests (Schema 06, the **"Bat Country Protocol"**). They frame SRM not just as a measurement, but as an *interpretive act*. See: [Racoon Schemas v1](https://github.com/ApocryphalEditor/2Smol-Interpretability/blob/main/Documentation/SRM%20Suite/Racoon%20Schema_full_images.pdf)

*   **The Epistemic Matrix:** A structured, reusable prompt set (`epistemic_certainty_prompt_grid_template.txt`) that varies core semantic ideas along axes of rhetorical type (e.g., declarative, rhetorical, observational) and certainty level (1–5). It allows controlled, factorial testing of how the model’s internal activations respond not only to *what* is said, but *how* it is said. The Matrix ensures experiments are *epistemically grounded* and enables programmatic generation of prompt-derived basis vectors. See: [Discussion of "Matrix" and "Grid" concepts](https://github.com/ApocryphalEditor/2Smol-Interpretability/blob/main/Documentation/SRM%20Suite/Epistemic%20Matrix.pdf)

**Important Note on Flexibility:** While the provided Epistemic Matrix uses declarative/rhetorical/observational axes, *any* semantic contrast could define a basis (e.g., serious vs. comedic, mathematical vs. spiritual). The power lies in realizing how flexible interpretive axes can be when treated as dynamic projections, not only fixed neuron indices.

### Baseline Collapse Recovery (BCR)

BCR investigates specific failure modes and intervention potentials:

*   How semantic collapse (e.g., repetition, incoherence) manifests during baseline text generation.
*   How targeted neuron interventions (clamping activation values) can potentially restore semantic coherence.

The full workflow — from running intervention sweeps using `capture_intervened_activations.py` to detecting recovery events with `analyze_textv2.py` — is detailed in:

➡️ **[`BCR Paper and Reproduction Guidelines`](https://github.com/ApocryphalEditor/2Smol-Interpretability/tree/main/Documentation/BCR)** ⬅️

---

## Experimental Structure & Codebase

The repository is organized around the following functionalities:

#### Data Capture
*   `capture_baseline_activations.py`: Captures activations/text for baseline runs.
*   `capture_intervened_activations.py`: Captures activations/text during neuron intervention sweeps.
*   `find_lonely_neurons.py`: Calculates neuron activation statistics (mean, stddev) across prompts.

#### Basis Construction
*   `generate_basis_vectors.py`: Creates `.npz` basis files from neuron pairs (one-hot) or averaged activations (single-plane/ensemble).

#### SRM Analysis & Comparison
*   `analyze_srm_sweep.py`: Performs the core SRM projection and resonance calculation onto basis planes.
*   `compare_srm_runs.py`: Compares results from two SRM analyses (e.g., baseline vs. intervention) to calculate deltas.
*   `add_grey_vector.py`: Calculates the average baseline projection (Grey Vector) and adds it to analysis metadata.

#### Text Analysis (BCR Workflow)
*   `extract_text_from_logs.py`: Consolidates generated text from log files into a TSV.
*   `analyze_textv2.py`: Analyzes text for collapse/recovery heuristics.

#### Interactive Tools & Utilities
*   `GPT2All.py` / `GPT2Last.py`: GUI tools for real-time neuron intervention exploration.
*   `visualize_srm_multi_basis_v721_global.py`: An older script for overlaying multiple SRM results.
*   `utils.py`: Shared helper functions for file I/O, data processing, calculations, etc.

---

## Broader Context

This suite emerged from a meta-experiment exploring:
_Can non-specialists, working collaboratively with LLMs, build functional research-grade interpretability pipelines?_

Through intuition, *vibe coding*, and careful epistemic scaffolding, we aim to map the *hidden interiors* of machine cognition. Everything here — from the Grey Vector concept to the Epistemic Matrix structure — is an artifact of that ongoing attempt.

We hope it serves as both map and invitation for further exploration.

---