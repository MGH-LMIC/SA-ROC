# SA-ROC: The Safety-Aware ROC Framework

This repository contains the official Python implementation for the manuscript **"Quantifying Trust in Clinical AI: The Safety-Aware ROC (SA-ROC) Framework"**.

---

## Overview

The **SA-ROC framework** advances clinical AI evaluation by providing a direct blueprint for safe automation. Unlike traditional metrics, SA-ROC reframes AI assessment around **clinician-defined safety policies**, enabling transparent and policy-driven clinical decision support.

<div align="center">
<img src="https://github.com/user-attachments/assets/e555e93c-a34b-4f6f-b463-edbd71763f6c" alt="SA-ROC Framework Overview" width="100%">
</div>
<br>

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

## Requirements & Setup

### Python
- **Python 3.9+** (tested on 3.9.x)

### Installation

```bash
# Clone the repository
git clone https://github.com/MGH-LMIC/SA-ROC.git
cd SA-ROC

# Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# Install dependencies (pinned to a known-good stack)
pip install -r requirements.txt
```

---

## Data Format

Provide a CSV with the following columns:

- `ID` — Unique identifier per case (e.g., `case_001`)
- `Score` — Model score or calibrated probability in `[0, 1]`
- `Label` — Ground truth, `1` (positive) or `0` (negative)

**Example**


| ID | Score | Label |
| :--- | :---: | :---: |
| case_0001 | 0.83 | 1 |
| case_0002 | 0.14 | 0 |


Place your file under `data/` (e.g., `data/example_data.csv`).

## Quick Start (Minimal Example)

```python
import os
import numpy as np
import pandas as pd

from saroc.analyzer import SA_ROC_Analyzer

# 1) Load data
df = pd.read_csv("data/example_data.csv")  # columns: ID, Score, Label
scores = df["Score"].to_numpy(dtype=float)
labels = df["Label"].to_numpy(dtype=int)

# 2) Initialize analyzer
analyzer = SA_ROC_Analyzer(scores=scores, labels=labels, verbose=True)

# 3) Define a clinical safety policy
policy = {
    "alpha_minus": 0.99,  # NPV target (rule-out)
    "alpha_plus":  0.95,  # PPV target (rule-in)
}

# 4) Choose an operational threshold (example: median score)
op_threshold = float(np.median(scores))
print(f"Using operational threshold: {op_threshold:.3f}")

# 5) Render SA-ROC (with light CI for speed) and save figure
os.makedirs("figs", exist_ok=True)
_ = analyzer.plot_sa_roc(
    policy,
    op_threshold=op_threshold,
    show_ci=True,
    n_bootstraps=100,            # increase (e.g., 2000) for publication-grade CIs
    title="SA-ROC with 95% CI",
    save_path="figs/sa_roc_with_ci.png",
)
print("Saved: figs/sa_roc_with_ci.png")
```

---

## Core Diagnostics (Recommended Plots)

```python
# Jitter + Distributions (class-wise score landscape)
_ = analyzer.plot_jitter_and_distributions(
    policy,
    op_threshold=op_threshold,
    save_path="figs/jitter_distributions.png",
)
print("Saved: figs/jitter_distributions.png")

# Safety Profile (Gamma_Area and Gray Zone % vs safety level α)
_ = analyzer.plot_safety_profile(
    op_threshold=op_threshold,
    save_path="figs/safety_profile.png",
    title="Safety Profile Curve",
)
print("Saved: figs/safety_profile.png")
```

---

## Policy Recommendation Routines

We provide utilities to convert clinical targets into operational thresholds `(τ-, τ+)`.
Each routine returns `(policy_dict | None, reason)`, where `policy_dict = {"tau_minus": ..., "tau_plus": ...}`.

```python
import os
from IPython.display import Image, display

os.makedirs("figs", exist_ok=True)

# (1) Dual-Purity: NPV ≥ α-, PPV ≥ α+, and τ- < τ+
rec, reason = analyzer.recommend_policy_dual_safety(alpha_minus=0.99, alpha_plus=0.90)
print("[Dual-Purity]", reason, rec)
if rec:
    _ = analyzer.plot_policy_overview(
        rec,
        title="Policy: Dual Purity",
        subtitle=f"NPV≥0.99, PPV≥0.90",
        save_path="figs/policy_dual_purity_overview.png",
    )
    print("Saved: figs/policy_dual_purity_overview.png")

# (2) Capped Gray (Rule-in): fix τ+ with PPV ≥ α+, then choose τ- to grow gray zone under a cohort cap
rec, reason = analyzer.recommend_policy_capped_gray_rule_in(max_gray_pct=20.0, alpha_plus=0.90)
print("[Capped Gray (Rule-in)]", reason, rec)
if rec:
    _ = analyzer.plot_policy_overview(
        rec,
        title="Policy: Capped Gray + Rule-in",
        subtitle="PPV≥0.90, Gray≤20%",
        save_path="figs/policy_capped_gray_rulein_overview.png",
    )
    print("Saved: figs/policy_capped_gray_rulein_overview.png")

# (3) Capped Gray (Rule-out): fix τ- with NPV ≥ α-, then choose τ+ to grow gray zone under a cap
rec, reason = analyzer.recommend_policy_capped_gray_rule_out(max_gray_pct=20.0, alpha_minus=0.99)
print("[Capped Gray (Rule-out)]", reason, rec)
if rec:
    _ = analyzer.plot_policy_overview(
        rec,
        title="Policy: Capped Gray + Rule-out",
        subtitle="NPV≥0.99, Gray≤20%",
        save_path="figs/policy_capped_gray_ruleout_overview.png",
    )
    print("Saved: figs/policy_capped_gray_ruleout_overview.png")

# (4) Utility-Maximizing: search (τ-, τ+) for maximal total expected utility
utility = {"U_TN": +1.0, "U_FP": -10.0, "U_TP": +20.0, "U_FN": -50.0, "cost_gray": -0.2}
best, U, thr = analyzer.recommend_policy_max_utility(utility)
print("[Utility-Maximizing] Best policy:", best)

# Visualize the chosen policy
_ = analyzer.plot_policy_overview(
    {"tau_minus": best["tau_minus"], "tau_plus": best["tau_plus"]},
    title="Policy: Utility-Optimized",
    subtitle=f"U={utility}",
    save_path="figs/policy_utility_overview.png",
)
print("Saved: figs/policy_utility_overview.png")

# Heatmap + breakdown
_ = analyzer.plot_utility_heatmap_and_breakdown(
    U, thr, best, utility,
    title="Utility Analysis (Heatmap & Breakdown)",
    save_path="figs/policy_utility_heatmap_breakdown.png",
)
print("Saved: figs/policy_utility_heatmap_breakdown.png")
```

---

## Tutorial Notebook

A step-by-step tutorial is provided in **`tutorial.ipynb`**. It demonstrates:

1. Loading example data and basic validation  
2. Defining safety policies and choosing an operating threshold  
3. Rendering core diagnostics: SA-ROC, jitter + distributions, safety profile  
4. Running the four policy recommendation routines  
5. Visualizing policy outcomes (overview + utility heatmap/bars)

Launch:
```bash
jupyter lab  # or: jupyter notebook
```

---

## Project Structure

```
SA-ROC/
├─ saroc/
│ └─ analyzer.py # SA_ROC_Analyzer: core logic, plots, policy routines
├─ data/
│ └─ example_data.csv # example dataset (ID, Score, Label)
├─ figs/ # figures saved by examples/tutorial (auto-created)
├─ tutorial.ipynb # end-to-end tutorial
├─ requirements.txt # pinned stack
└─ README.md
```

## Reproducibility & Notes

- **Backend & Warnings**: The analyzer sets Matplotlib's backend to `Agg` and suppresses Python warnings for clean, deterministic output (useful in headless/CI).
- **Bootstrapping**: Confidence intervals are computed via bootstrap. Use small `n_bootstraps` (e.g., 100–200) for exploration and increase for final reporting.
- **Input Validation**: Scores must be calibrated within `[0, 1]`. Labels must be binary `{0, 1}` with both classes present.
- **Figure Outputs**: Plots are saved under `figs/` by default; ensure the directory is writable on your system.

## Associated Manuscript

> **Quantifying Trust in Clinical AI: The Safety-Aware ROC (SA-ROC) Framework**
>
> **Authors:**<br>
> Young-Tak Kim¹, Hyunji Kim¹, Manisha Bahl¹, Michael H. Lev¹, Ramon Gilberto González¹,²,³, Michael S. Gee¹, Synho Do¹,⁴,⁵*
>
> **Affiliations:**<br>
> ¹ *Department of Radiology, Massachusetts General Hospital, Harvard Medical School*<br>
> ² *Data Science Office, Massachusetts General Brigham*<br>
> ³ *Athinoula A. Martinos Center for Biomedical Imaging, Massachusetts General Hospital*<br>
> ⁴ *Kempner Institute, Harvard University*<br>
> ⁵ *KU-KIST Graduate School of Converging Science and Technology, Korea University*
>
> **\*Corresponding Author:** sdo@mgh.harvard.edu

## License

Please note that the intellectual property described in this work is subject to a pending patent application. The code in this repository is provided to editors and reviewers to facilitate the peer review of our manuscript. For inquiries regarding other uses, including commercial licensing, please contact the corresponding author.
