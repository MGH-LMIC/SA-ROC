# SA-ROC: Safety-Aware ROC Framework

<div align="center">

[![License](https://img.shields.io/badge/License-For_Review_Only-red.svg?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-3776ab.svg?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![arXiv](https://img.shields.io/badge/arXiv-Under_Review-b31b1b.svg?style=flat-square)](https://arxiv.org)

**Quantifying Trust in Clinical AI Through Safety-Aware Evaluation**

[📚 Documentation](#getting-started) • [🚀 Quick Start](#quickstart-example) • [📖 Citation](#citation) • [🔬 Research](#associated-manuscript)

</div>

---

## 🎯 Overview

The **SA-ROC framework** revolutionizes clinical AI evaluation by providing a direct blueprint for safe automation. Unlike traditional metrics, SA-ROC reframes AI assessment around **clinician-defined safety policies**, enabling transparent and policy-driven clinical decision support.

### 🔑 Key Innovation

<div align="center">
<img src="SA-ROC_Overview.png" alt="SA-ROC Framework Overview" width="80%">
</div>

> **Transform uncertainty into actionable insights** by partitioning AI predictions into **Safe Zones** (autonomous action) and **Gray Zones** (human review required).

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🎨 **Visual Safety Mapping**
- Augments traditional ROC curves with operational safety visualization
- Color-coded segments for immediate insight into model reliability

### 📋 **Policy-Driven Design**
- Define custom automation policies based on clinical requirements
- Explicit reliability targets (e.g., "99% NPV for rule-out decisions")

</td>
<td width="50%">

### 📊 **Uncertainty Quantification**
- **Gray Zone Area (Γ_Area)** metric quantifies "cost of indecision"
- Measures non-automated workload and operational efficiency

### 🔬 **Model Comparison**
- Head-to-head comparisons across different AI models
- Policy-agnostic evaluation framework

</td>
</tr>
</table>

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/MGH-LMIC/SA-ROC.git
cd SA-ROC

# Install dependencies
pip install -r requirements.txt
```

### Data Format

> **📋 Required CSV Structure**

Your data should contain exactly three columns:

| Column | Description | Example |
|--------|-------------|---------|
| `ID` | Unique case identifier | `case_001` |
| `Score` | AI model risk score (0-1) | `0.85` |
| `Label` | Ground truth (0=negative, 1=positive) | `1` |

---

## 💡 Quick Start

```python
import pandas as pd
from saroc import SA_ROC

# Load your data
data = pd.read_csv("your_data.csv")
scores = data['Score'].values
labels = data['Label'].values

# Initialize SA-ROC analyzer
analyzer = SA_ROC(scores=scores, labels=labels)

# Define clinical safety policy
policy = {
    'alpha_minus': 0.99,  # 99% NPV requirement
    'alpha_plus': 0.95    # 95% PPV requirement
}

# Analyze and visualize
results = analyzer.analyze(policy)
analyzer.plot()  # Generates SA-ROC visualization
```

### Expected Output

The framework will generate:
- **SA-ROC curve** with color-coded safety zones
- **Quantitative metrics** including Gray Zone Area
- **Policy compliance** assessment

---

## 📚 Research

### Associated Manuscript

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

---

## 📄 Citation

If you use SA-ROC in your research, please cite:

```bibtex
@article{Kim2025_SAROC,
  author    = {Kim, Young-Tak and Kim, Hyunji and Bahl, Manisha and 
               Lev, Michael H. and Gonz\'{a}lez, Ramon Gilberto and 
               Gee, Michael S. and Do, Synho},
  title     = {Quantifying Trust in Clinical AI: The Safety-Aware ROC 
               (SA-ROC) Framework},
  journal   = {Manuscript under review},
  year      = {2025}
}
```

---

## ⚖️ License

> **🔒 Patent Pending**
> 
> This software embodies intellectual property subject to a pending patent application. Source code is provided to editors and reviewers for manuscript evaluation purposes only.
> 
> For commercial licensing inquiries, contact: **sdo@mgh.harvard.edu**

---

<div align="center">

**Built with ❤️ by the [MGH Laboratory for Medical Informatics and Computing](https://github.com/MGH-LMIC)**

[⭐ Star this repo](https://github.com/MGH-LMIC/SA-ROC) • [🐛 Report Issues](https://github.com/MGH-LMIC/SA-ROC/issues) • [💬 Discussions](https://github.com/MGH-LMIC/SA-ROC/discussions)

</div>
