# SA-ROC: The Safety-Aware ROC Framework

This repository contains the official Python implementation for the manuscript **"Quantifying Trust in Clinical AI: The Safety-Aware ROC (SA-ROC) Framework"**.

---

## Overview

The **SA-ROC framework** advances clinical AI evaluation by providing a direct blueprint for safe automation. Unlike traditional metrics, SA-ROC reframes AI assessment around **clinician-defined safety policies**, enabling transparent and policy-driven clinical decision support.

<div align="center">
<img src="SA-ROC_Overview.png" alt="SA-ROC Framework Overview" width="80%">
</div>

> **(a) Score Partitioning:** Based on a clinician's policy (e.g., "a negative prediction must be 100% reliable"), the framework partitions the model's raw risk scores into three zones. The **Rule-out Safe Zone** and **Rule-in Safe Zone** represent predictions reliable enough for autonomous action, while the **Gray Zone** contains uncertain cases mandating human review.
>
> **(b) Safety Level Dynamics:** The size of these zones dynamically changes with the required safety level (α). As the demand for reliability increases, the Safe Zones shrink and the **Gray Zone** expands, quantifying the trade-off between safety and the human workload.
>
> **(c) The SA-ROC Curve:** This entire safety landscape is visualized on the standard ROC curve. The curve segments are color-coded, providing an integrated view of a model's discrimination and its operational safety. The **Gray Zone Area (Γ_Area)** quantifies the model's overall operational uncertainty.

---

## Key Features

- **Visual Safety Mapping:** Augments traditional ROC curves with an operational safety visualization, using color-coded segments for immediate insight into model reliability.

- **Policy-Driven Design:** Allows users to define custom automation policies based on explicit clinical requirements, such as "99% NPV for rule-out decisions."

- **Uncertainty Quantification:** Introduces the **Gray Zone Area (Γ_Area)** metric to quantify the "cost of indecision," measuring the non-automated workload and operational efficiency.

- **Model Comparison:** Provides a framework for head-to-head comparisons of different AI models under consistent, clinically relevant safety constraints.

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone [https://github.com/MGH-LMIC/SA-ROC.git](https://github.com/MGH-LMIC/SA-ROC.git)
cd SA-ROC

# Install dependencies
pip install -r requirements.txt
```
### Data Format

Your data must be prepared in a CSV file containing the following three columns:

-   `ID`: A unique identifier for each case (e.g., `case_001`).
-   `Score`: The AI model's continuous risk score, typically between 0 and 1 (e.g., `0.85`).
-   `Label`: The ground truth, where `1` is for the positive class and `0` is for the negative class (e.g., `1`).

---

## Quick Start Example

```python
import pandas as pd
from saroc import SA_ROC

# Load your data from a CSV file
data = pd.read_csv("your_data.csv")
scores = data['Score'].values
labels = data['Label'].values

# Initialize the SA-ROC analyzer
analyzer = SA_ROC(scores=scores, labels=labels)

# Define a clinical safety policy
policy = {
    'alpha_minus': 0.99,  # 99% NPV requirement for rule-out
    'alpha_plus': 0.95    # 95% PPV requirement for rule-in
}

# Analyze and visualize the results
results = analyzer.analyze(policy)
analyzer.plot()  # Generates the SA-ROC visualization
```

## Associated Manuscript

> **Quantifying Trust in Clinical AI: The Safety-Aware ROC (SA-ROC) Framework**
>
> **Authors:** Young-Tak Kim¹, Hyunji Kim¹, Manisha Bahl¹, Michael H. Lev¹, Ramon Gilberto González¹,²,³, Michael S. Gee¹, Synho Do¹,⁴,⁵*
>
> **Affiliations:**
> - ¹ Department of Radiology, Massachusetts General Hospital, Harvard Medical School
> - ² Data Science Office, Massachusetts General Brigham
> - ³ Athinoula A. Martinos Center for Biomedical Imaging, MGH
> - ⁴ Kempner Institute, Harvard University
> - ⁵ KU-KIST Graduate School of Converging Science and Technology, Korea University
>
> **Corresponding Author:** sdo@mgh.harvard.edu

## License

Please note that the intellectual property described in this work is subject to a pending patent application. The code in this repository is provided to editors and reviewers to facilitate the peer review of our manuscript. For inquiries regarding other uses, including commercial licensing, please contact the corresponding author.
