# SA-ROC: The Safety-Aware ROC Framework

![License](https://img.shields.io/badge/License-Protected-blue)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)

This repository contains the official Python implementation for the manuscript **"Quantifying Trust in Clinical AI: The Safety-Aware ROC (SA-ROC) Framework"**.

---

### Associated Manuscript

> ### Quantifying Trust in Clinical AI: The Safety-Aware ROC (SA-ROC) Framework
>
> **Young-Tak Kim¹**, **Hyunji Kim¹**, **Manisha Bahl¹**, **Michael H. Lev¹**, **Ramon Gilberto González¹,²,³**, **Michael S. Gee¹**, **Synho Do¹,⁴,⁵***
>
> ¹ *Department of Radiology, Massachusetts General Hospital, Harvard Medical School, Boston, MA, USA*
> ² *Data Science Office, Massachusetts General Brigham, Boston, MA, USA*
> ³ *Athinoula A. Martinos Center for Biomedical Imaging, Massachusetts General Hospital, Boston, MA, USA*
> ⁴ *Kempner Institute, Harvard University, Boston, MA, USA*
> ⁵ *KU-KIST Graduate School of Converging Science and Technology, Korea University, Seoul, Republic of Korea*
>
> **\*Corresponding Author: sdo@mgh.harvard.edu**

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

---

### Getting Started

#### Prerequisites
- Python 3.9 or higher
- Standard scientific computing libraries (NumPy, Matplotlib, Scikit-learn, etc.)

#### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/MGH-LMIC/SA-ROC.git](https://github.com/MGH-LMIC/SA-ROC.git)
   cd SA-ROC
