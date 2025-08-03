# SA-ROC: The Safety-Aware ROC Framework

![License](https://img.shields.io/badge/License-For_Review_Only-lightgrey)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)

This repository contains the official Python implementation for the manuscript **"Quantifying Trust in Clinical AI: The Safety-Aware ROC (SA-ROC) Framework"**.

---

### Associated Manuscript

> ### Quantifying Trust in Clinical AI: The Safety-Aware ROC (SA-ROC) Framework
>
> **Young-Tak Kim¹**, **Hyunji Kim¹**, **Manisha Bahl¹**, **Michael H. Lev¹**, **Ramon Gilberto González¹,²,³**, **Michael S. Gee¹**, **Synho Do¹,⁴,⁵***
>
> ¹ *Department of Radiology, Massachusetts General Hospital, Harvard Medical School, Boston, MA, USA*<br>
> ² *Data Science Office, Massachusetts General Brigham, Boston, MA, USA*<br>
> ³ *Athinoula A. Martinos Center for Biomedical Imaging, Massachusetts General Hospital, Boston, MA, USA*<br>
> ⁴ *Kempner Institute, Harvard University, Boston, MA, USA*<br>
> ⁵ *KU-KIST Graduate School of Converging Science and Technology, Korea University, Seoul, Republic of Korea*
>
> **\*Corresponding Author: sdo@mgh.harvard.edu**

---

### Overview of the SA-ROC Framework

The SA-ROC framework provides a direct blueprint for safe clinical automation by reframing AI evaluation around clinician-defined safety policies.

![Overview of the SA-ROC Framework](SA-ROC_Overview.png)

> **(a) Score Partitioning:** Based on a clinician's policy (e.g., "a negative prediction must be 100% reliable"), the framework partitions the model's raw risk scores into three zones. The **Rule-out Safe Zone** and **Rule-in Safe Zone** represent predictions reliable enough for autonomous action, while the **Gray Zone** contains uncertain cases mandating human review.
>
> **(b) Safety Level Dynamics:** The size of these zones dynamically changes with the required safety level (α). As the demand for reliability increases, the Safe Zones shrink and the **Gray Zone** expands, quantifying the trade-off between safety and the human workload.
>
> **(c) The SA-ROC Curve:** This entire safety landscape is visualized on the standard ROC curve. The curve segments are color-coded, providing an integrated view of a model's discrimination and its operational safety. The **Gray Zone Area (Γ_Area)** quantifies the model's overall operational uncertainty.

### Key Features

- **Visualize Operational Safety:** Augments the traditional ROC curve with a visual map of a model's safety landscape.
- **Policy-Driven Partitioning:** Divides AI predictions into Safe Zones and a Gray Zone based on explicit clinical reliability targets.
- **Quantify Uncertainty:** Introduces the Gray Zone Area (Γ_Area) to measure the "cost of indecision" and non-automated workload.
- **Design and Compare Policies:** Enables the design of custom automation policies and facilitates head-to-head comparisons of AI models.

---

### Getting Started

#### Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/MGH-LMIC/SA-ROC.git](https://github.com/MGH-LMIC/SA-ROC.git)
   cd SA-ROC
Install the required dependencies:

Bash

pip install -r requirements.txt
Data Preparation
[!NOTE]
To use the SA-ROC framework, prepare your data in a CSV file with the following three columns:

ID: A unique identifier for each case.

Score: The continuous risk score output from your AI model (e.g., between 0 and 1).

Label: The ground truth label, where 1 represents the positive class and 0 represents the negative class.

Quickstart Example
Python

import pandas as pd
from saroc import SA_ROC

# 1. Load data from your CSV file
data = pd.read_csv("your_data.csv")
scores = data['Score'].values
labels = data['Label'].values

# 2. Initialize the analyzer
analyzer = SA_ROC(scores=scores, labels=labels)

# 3. Define a clinical safety policy (e.g., 99% NPV, 95% PPV)
policy = {'alpha_minus': 0.99, 'alpha_plus': 0.95}

# 4. Analyze and plot the results
results = analyzer.analyze(policy)
analyzer.plot() # Generates the SA-ROC curve visualization
Citation
If you use the SA-ROC framework or our code in your research, please cite our manuscript.

코드 스니펫

@article{Kim2025_SAROC,
  author    = {Kim, Young-Tak and Kim, Hyunji and Bahl, Manisha and Lev, Michael H. and Gonz\'{a}lez, Ramon Gilberto and Gee, Michael S. and Do, Synho},
  title     = {Quantifying Trust in Clinical AI: The Safety-Aware ROC (SA-ROC) Framework},
  journal   = {Manuscript under review},
  year      = {2025}
}
License
The intellectual property embodied in this software is subject to a pending patent application. The source code is made available to editors and reviewers for the sole purpose of facilitating the evaluation of our manuscript. For any inquiries regarding other uses, including commercial licensing, please contact the corresponding author.
