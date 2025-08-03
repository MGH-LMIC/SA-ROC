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
