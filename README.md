# SA-ROC: The Safety-Aware ROC Framework

![License](https://img.shields.io/badge/License-Protected-blue)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)

This repository contains the official Python implementation of the **Safety-Aware ROC (SA-ROC) Framework**, a novel methodology for quantifying trust and operational safety in clinical AI.

This code is the official implementation for the manuscript:
> **Quantifying Trust in Clinical AI: The Safety-Aware ROC (SA-ROC) Framework** <br>
> *Young-Tak Kim, Hyunji Kim, et al.* <br>
> (Currently under review at *Nature Biomedical Engineering*)

---

### Overview

Conventional metrics like AUC fail to answer the clinician's critical question: "When is it safe to trust this AI?" The SA-ROC framework bridges this gap by reframing AI evaluation around clinician-defined safety policies. It partitions a model’s predictions into explicit **Safe Zones** for reliable automation and a **Gray Zone** for essential human review, providing a direct, quantitative blueprint for safe clinical automation.

![SA-ROC Conceptual Diagram](figure1_conceptual_diagram.png)
*Fig. 1: The conceptual workflow of the SA-ROC framework.*

### Key Features

- **Visualize Operational Safety:** Augments the traditional ROC curve with a visual map of a model's safety landscape.
- **Policy-Driven Partitioning:** Divides AI predictions into Rule-in Safe, Rule-out Safe, and Gray Zones based on explicit clinical reliability targets (α⁺ and α⁻).
- **Quantify Uncertainty:** Introduces the Gray Zone Area (Γ_Area) to measure the "cost of indecision" and non-automated workload.
- **Design and Compare Policies:** Enables the design of custom automation policies and facilitates head-to-head comparisons of AI models under real-world safety constraints.

### Getting Started

#### Prerequisites
- Python 3.9 or higher
- Standard scientific computing libraries (NumPy, Matplotlib, Scikit-learn, etc.)

#### Installation
1.  Clone the repository:
    ```bash
    git clone [https://github.com/MGH-LMIC/SA-ROC.git](https://github.com/MGH-LMIC/SA-ROC.git)
    cd SA-ROC
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Quickstart Example

Here is a simple example of how to use the `saroc` library to analyze a model's predictions.

```python
# Import the main class from the library
from saroc import SA_ROC

# 1. Load your model's prediction scores and the ground truth labels
# (scores should be continuous values, e.g., between 0 and 1)
scores = [...]  # Array of your model's prediction scores
labels = [...]  # Array of the true labels (0 or 1)

# 2. Initialize the SA-ROC analyzer with your data
analyzer = SA_ROC(scores=scores, labels=labels)

# 3. Define a clinical safety policy
# e.g., require 99% NPV for rule-out and 95% PPV for rule-in
policy = {'alpha_minus': 0.99, 'alpha_plus': 0.95}

# 4. Analyze the model under the specified policy
results = analyzer.analyze(policy)
print(results)

# 5. Plot the SA-ROC curve and the safety profile
analyzer.plot_sa_roc_curve()
analyzer.plot_safety_profile()
