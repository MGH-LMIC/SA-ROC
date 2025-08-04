# saroc/analyzer.py
# -*- coding: utf-8 -*-
"""
Safety-Aware ROC (SA-ROC) Analyzer

This module provides the SA_ROC_Analyzer class to:
  1) Compute a performance profile across all unique score thresholds
  2) Analyze safety policies (rule-out / rule-in) at a given operational threshold
  3) Estimate 95% CIs for AUC and Γ_Area via bootstrapping
  4) Plot SA-ROC curves, combined jitter/distribution plots, and Safety Profile curves
  5) Recommend policy thresholds (τ_minus, τ_plus) under multiple paradigms:
       - Dual safety (NPV ≥ α-, PPV ≥ α+)
       - Capped gray-zone with rule-in safety
       - Capped gray-zone with rule-out safety
       - Utility-maximizing policy over (τ-, τ+)
  6) Visualize policy overviews (2×2 panels) and utility heatmaps/breakdowns

Notes
-----
- Scores are assumed to be in [0, 1]. Labels must be binary {0, 1} and both classes present.
- Zone definition used consistently across methods:
    Rule-out zone: score < τ_minus
    Rule-in  zone: score ≥ τ_plus
    Gray zone   : τ_minus ≤ score < τ_plus
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

# IMPORTANT: Set non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore")

class SA_ROC_Analyzer:
    """
    Perform Safety-Aware ROC (SA-ROC) analysis on AI model predictions.

    Parameters
    ----------
    scores : array-like of shape (n_samples,)
        Model scores (probabilities or calibrated risk scores) in [0, 1].
    labels : array-like of shape (n_samples,)
        Ground-truth binary labels {0, 1}.
    verbose : bool, default=True
        If True, prints progress messages & shows tqdm in bootstrapping.

    Attributes
    ----------
    scores : np.ndarray
        Copy of the input scores (float), validated to be in [0, 1].
    labels : np.ndarray
        Copy of the input labels (int), validated to be {0, 1}.
    df_metrics : pd.DataFrame
        Per-threshold metrics including:
            Threshold, FPR, TPR, PPV, NPV, TP, FP, TN, FN
        Thresholds are unique score values sorted in descending order.
    roc_auc : float
        AUC computed from sklearn's roc_curve.
    """

    # ----------------------------- Initialization -----------------------------

    def __init__(self, scores, labels, verbose: bool = True) -> None:
        self.scores = np.asarray(scores, dtype=float).reshape(-1)
        self.labels = np.asarray(labels, dtype=int).reshape(-1)
        self.verbose = bool(verbose)

        self._validate_inputs()

        self.df_metrics: Optional[pd.DataFrame] = None
        self.roc_auc: Optional[float] = None

        # Compute once at construction; downstream plots reuse this.
        self._calculate_performance_profile()

    # ------------------------------ Validation --------------------------------

    def _validate_inputs(self) -> None:
        """Validate inputs: length, value ranges, and class presence."""
        if self.scores.shape[0] != self.labels.shape[0]:
            raise ValueError(
                f"Length mismatch: scores={self.scores.shape[0]}, labels={self.labels.shape[0]}"
            )

        if self.scores.ndim != 1 or self.labels.ndim != 1:
            raise ValueError("scores and labels must be 1-D arrays.")

        if np.any(np.isnan(self.scores)):
            raise ValueError("scores contain NaN.")
        if np.any((self.scores < 0.0) | (self.scores > 1.0)):
            raise ValueError("scores must lie in [0, 1].")

        unique_labels = np.unique(self.labels)
        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError("labels must be binary in {0, 1}.")

        if unique_labels.size < 2:
            raise ValueError("labels must contain both classes 0 and 1.")

    # ------------------------- Core Profile Computation ------------------------

    def _calculate_performance_profile(self) -> None:
        """
        Calculate a detailed performance profile across all unique score thresholds.
        Populates self.df_metrics and self.roc_auc.
        """
        if self.verbose:
            print("-> Calculating model performance profile...")

        fpr_full, tpr_full, _ = roc_curve(self.labels, self.scores)
        self.roc_auc = float(auc(fpr_full, tpr_full))

        performance_data: List[Dict[str, float]] = []
        unique_thresholds = np.unique(self.scores)

        # Evaluate at each unique score as a threshold (descending)
        for t in np.sort(unique_thresholds)[::-1]:
            pred = (self.scores >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(self.labels, pred, labels=[0, 1]).ravel()

            # Precision (PPV) and NPV
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 1.0

            # Rates
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            performance_data.append(
                {
                    "Threshold": float(t),
                    "FPR": fpr, "TPR": tpr, "PPV": ppv, "NPV": npv,
                    "TP": float(tp), "FP": float(fp), "TN": float(tn), "FN": float(fn),
                }
            )

        self.df_metrics = pd.DataFrame(
            performance_data,
            columns=["Threshold", "FPR", "TPR", "PPV", "NPV", "TP", "FP", "TN", "FN"],
        )
        if self.verbose:
            print(f"-> Performance profile created. Base AUC: {self.roc_auc:.3f}")

    # ---------------------------- Policy Analysis -----------------------------

    def analyze_policy(self, policy: Dict[str, float], op_threshold: float) -> Dict[str, float]:
        """
        Analyze model performance under a safety policy at a given operational threshold.

        Symmetric policy uses α for rule-out (NPV) and rule-in (PPV):
          policy = {'alpha_minus': α, 'alpha_plus': α}

        Definitions
        -----------
        - Rule-out zone (left):  scores in [0, τ_minus), requires NPV ≥ α_minus
        - Rule-in  zone (right): scores in [τ_plus, 1], requires PPV ≥ α_plus
        - Gray zone: [τ_minus, τ_plus), where policy is not satisfied

        Γ_Area (Gamma Area) on ROC space:
            Γ_Area = FPR(τ_minus) * (1 - TPR(τ_plus))
        If no valid separation (τ_minus < τ_plus) exists, Γ_Area := 1.0.

        Parameters
        ----------
        policy : dict
            Must contain 'alpha_minus' (float) and 'alpha_plus' (float), ∈ (0, 1].
        op_threshold : float
            Operational threshold separating left (rule-out) and right (rule-in).

        Returns
        -------
        dict
            Keys: 'gamma_area', 'is_valid_policy', 't_rule_out', 't_rule_in',
                  'fpr_boundary', 'tpr_boundary'
        """
        if self.df_metrics is None:
            raise RuntimeError("Performance profile not computed.")

        alpha_minus = float(policy.get("alpha_minus", 1.0))
        alpha_plus = float(policy.get("alpha_plus", 1.0))

        # Rows that satisfy policy on each side of the operational threshold
        rule_out_rows = self.df_metrics[
            (self.df_metrics["Threshold"] <= op_threshold) & (self.df_metrics["NPV"] >= alpha_minus)
        ]
        rule_in_rows = self.df_metrics[
            (self.df_metrics["Threshold"] > op_threshold) & (self.df_metrics["PPV"] >= alpha_plus)
        ]

        # Extreme thresholds that still satisfy the policy
        t_rule_out = float(rule_out_rows["Threshold"].max()) if not rule_out_rows.empty else 0.0
        t_rule_in = float(rule_in_rows["Threshold"].min()) if not rule_in_rows.empty else 1.0

        is_valid = t_rule_out < t_rule_in

        # Find nearest rows to the chosen thresholds (for FPR/TPR bounds)
        row_ro = self.df_metrics.iloc[(self.df_metrics["Threshold"] - t_rule_out).abs().argsort()[:1]]
        row_ri = self.df_metrics.iloc[(self.df_metrics["Threshold"] - t_rule_in).abs().argsort()[:1]]

        fpr_boundary = float(row_ro["FPR"].iloc[0]) if not row_ro.empty else 0.0
        tpr_boundary = float(row_ri["TPR"].iloc[0]) if not row_ri.empty else 1.0

        gamma_area = (fpr_boundary * (1.0 - tpr_boundary)) if is_valid else 1.0

        return {
            "gamma_area": float(gamma_area),
            "is_valid_policy": bool(is_valid),
            "t_rule_out": float(t_rule_out),
            "t_rule_in": float(t_rule_in),
            "fpr_boundary": float(fpr_boundary),
            "tpr_boundary": float(tpr_boundary),
        }

    # -------------------------- Confidence Intervals --------------------------

    def calculate_confidence_intervals(
        self,
        policy: Dict[str, float],
        op_threshold: float,
        n_bootstraps: int = 2000,
        random_seed: int = 42,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate 95% CIs for AUC and Γ_Area by bootstrap resampling.

        Parameters
        ----------
        policy : dict
            Safety policy (see analyze_policy).
        op_threshold : float
            Operational threshold separating the two policy sides.
        n_bootstraps : int, default=2000
            Number of bootstrap samples.
        random_seed : int, default=42
            Seed for reproducible resampling via NumPy Generator.

        Returns
        -------
        dict with keys:
            'auc_ci' : (low, high)
            'gamma_area_ci' : (low, high)
        """
        rng = np.random.default_rng(random_seed)
        n_samples = len(self.labels)
        boot_aucs: List[float] = []
        boot_gammas: List[float] = []

        if self.verbose:
            print(f"\n-> Starting bootstrap for CI calculation ({n_bootstraps} iterations)...")

        for _ in tqdm(range(n_bootstraps), disable=not self.verbose):
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            labels_boot = self.labels[indices]
            scores_boot = self.scores[indices]

            if np.unique(labels_boot).size < 2:
                continue  # skip degenerate bootstrap with single class

            # Compute on bootstrap sample (silent)
            temp_analyzer = SA_ROC_Analyzer(scores=scores_boot, labels=labels_boot, verbose=False)
            boot_aucs.append(float(temp_analyzer.roc_auc))

            analysis_boot = temp_analyzer.analyze_policy(policy, op_threshold)
            boot_gammas.append(float(analysis_boot["gamma_area"]))

        if len(boot_aucs) == 0 or len(boot_gammas) == 0:
            raise RuntimeError(
                "Bootstrapping failed to produce valid samples with both classes. "
                "Consider increasing n_bootstraps or check class balance."
            )

        auc_ci = tuple(np.percentile(boot_aucs, [2.5, 97.5]).astype(float))
        gamma_ci = tuple(np.percentile(boot_gammas, [2.5, 97.5]).astype(float))

        if self.verbose:
            print("-> Bootstrap CI calculation complete.")
        return {"auc_ci": auc_ci, "gamma_area_ci": gamma_ci}

    # --------------------------------- Plots ----------------------------------

    @staticmethod
    def _ensure_parent_dir(path: str) -> None:
        """Create parent directory for a file path if it does not exist."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    @staticmethod
    def _save_figure(fig: plt.Figure, save_path: Optional[str], verbose: bool) -> None:
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            if verbose:
                print(f"-> Plot saved to {save_path}")

    def plot_sa_roc(
        self,
        policy: Dict[str, float],
        op_threshold: float,
        show_ci: bool = False,
        n_bootstraps: int = 2000,
        title: str = "SA-ROC Analysis",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Generate the SA-ROC curve visualization.

        The ROC polyline is colored by policy satisfaction:
        - Blue:  Rule-out side (Threshold ≤ op_threshold) with NPV ≥ alpha_minus
        - Red:   Rule-in  side (Threshold >  op_threshold) with PPV ≥ alpha_plus
        - Gray:  Neither satisfied (gray zone contribution)

        Legend (lower-right) additionally shows:
        - τ- (rule-out threshold)
        - τ+ (rule-in threshold)
        - Γ_Area (gray zone area)
        """
        from matplotlib.lines import Line2D  # for text-only legend entries

        if self.verbose:
            print(f"-> Plotting SA-ROC curve for policy: {policy}")

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect("equal", "box")

        # Color the ROC polyline segments based on the policy
        for i in range(len(self.df_metrics) - 1):
            segment = self.df_metrics.iloc[i : i + 2]
            t_start = segment["Threshold"].iloc[0]

            if (t_start <= op_threshold) and (self.df_metrics.iloc[i]["NPV"] >= policy.get("alpha_minus", 1.01)):
                color = "#3498db"  # Blue: rule-out satisfied
            elif (t_start > op_threshold) and (self.df_metrics.iloc[i]["PPV"] >= policy.get("alpha_plus", 1.01)):
                color = "#e74c3c"  # Red: rule-in satisfied
            else:
                color = "#bdc3c7"  # Gray: neither

            ax.plot(segment["FPR"], segment["TPR"], color=color, linewidth=5, solid_capstyle="round")

        # Diagonal reference
        ax.plot([0, 1], [0, 1], color="black", ls="--", lw=1)

        # Policy analysis & optional CI
        analysis_results = self.analyze_policy(policy, op_threshold)
        gamma_area_val = analysis_results["gamma_area"]
        is_valid = analysis_results["is_valid_policy"]

        # thresholds to show in legend
        if is_valid:
            tau_minus = analysis_results["t_rule_out"]
            tau_plus  = analysis_results["t_rule_in"]
            tau_minus_str = f"{tau_minus:.3f}"
            tau_plus_str  = f"{tau_plus:.3f}"
            gamma_area_str = f"{gamma_area_val:.3f}"
        else:
            tau_minus_str = "N/A"
            tau_plus_str  = "N/A"
            gamma_area_str = f"{gamma_area_val:.3f}"  # stays 1.000 if invalid per analyze_policy

        auc_text = f"AUC = {self.roc_auc:.3f}"
        gamma_text = f"Gamma_Area = {gamma_area_val:.3f}"

        if show_ci:
            ci_results = self.calculate_confidence_intervals(
                policy, op_threshold, n_bootstraps=n_bootstraps
            )
            auc_ci = ci_results["auc_ci"]
            gamma_ci = ci_results["gamma_area_ci"]
            auc_text += f" (95% CI: {auc_ci[0]:.3f}-{auc_ci[1]:.3f})"
            gamma_text += f" (95% CI: {gamma_ci[0]:.3f}-{gamma_ci[1]:.3f})"

        plot_title = f"{title}\n{auc_text} | {gamma_text}"
        ax.set_title(plot_title, fontsize=14, weight="bold")

        # Shade Gamma_Area rectangle if policy valid
        if is_valid:
            ax.add_patch(
                plt.Rectangle(
                    (0, analysis_results["tpr_boundary"]),
                    analysis_results["fpr_boundary"],
                    1 - analysis_results["tpr_boundary"],
                    facecolor="gray",
                    alpha=0.2,
                    zorder=0,
                )
            )

        # Mark the operational threshold point on ROC
        op_idx = int(np.argmin(np.abs(self.df_metrics["Threshold"] - op_threshold)))
        op_row = self.df_metrics.iloc[op_idx]
        op_marker, = ax.plot(
            op_row["FPR"], op_row["TPR"], "o",
            mfc="none", mec="k", mew=2, ms=9,
            label=f"Op. Threshold ({op_threshold:.2f})"
        )

        # Axes cosmetics
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

        # Legend (lower-right) — include tau-, tau+, Gamma_Area as text-only entries
        legend_entries = [
            mpatches.Patch(color="#e74c3c", label=f"Rule-in (PPV ≥ {policy.get('alpha_plus', 'N/A')})"),
            mpatches.Patch(color="#3498db", label=f"Rule-out (NPV ≥ {policy.get('alpha_minus', 'N/A')})"),
            mpatches.Patch(color="#bdc3c7", label="Gray Zone"),
            op_marker,
            Line2D([], [], linestyle="None", label=f"τ- = {tau_minus_str}"),
            Line2D([], [], linestyle="None", label=f"τ+ = {tau_plus_str}"),
            Line2D([], [], linestyle="None", label=f"Γ_Area = {gamma_area_str}"),
        ]
        ax.legend(handles=legend_entries, loc="lower right", fontsize=10)

        # NOTE: Removed the bottom-left "Policy analysis" callout box as requested.

        self._save_figure(fig, save_path, self.verbose)
        return fig


    def plot_jitter_and_distributions(
        self,
        policy: Dict[str, float],
        op_threshold: float,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Combined plot: (top) jittered individual scores by class,
        (bottom) class-conditional score distributions. Safe zones are shaded.
        """
        if self.verbose:
            print("-> Plotting combined jitter and distribution plot...")

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

        scores_neg = self.scores[self.labels == 0]
        scores_pos = self.scores[self.labels == 1]

        # --- Top: Jitter Plot ---
        axs[0].scatter(
            scores_neg, np.random.normal(0.75, 0.05, len(scores_neg)),
            s=15, color="#3498db", alpha=0.5, edgecolor="k", linewidth=0.5, label="Negative"
        )
        axs[0].scatter(
            scores_pos, np.random.normal(0.25, 0.05, len(scores_pos)),
            s=15, color="#e74c3c", alpha=0.5, edgecolor="k", linewidth=0.5, label="Positive"
        )

        analysis_results = self.analyze_policy(policy, op_threshold)
        if analysis_results["is_valid_policy"]:
            t_ro = analysis_results["t_rule_out"]
            t_ri = analysis_results["t_rule_in"]
            axs[0].axvspan(0, t_ro, color="#3498db", alpha=0.2, zorder=0, label=f"Rule-out Safe (NPV≥{policy.get('alpha_minus', 'N/A')})")
            axs[0].axvspan(t_ri, 1, color="#e74c3c", alpha=0.2, zorder=0, label=f"Rule-in Safe (PPV≥{policy.get('alpha_plus', 'N/A')})")

        axs[0].axvline(op_threshold, color="black", ls="--")
        axs[0].set_title("Individual Scores and Safe Zones", fontsize=16, weight="bold")
        axs[0].set_yticks([])
        axs[0].legend(loc="lower left")
        axs[0].set_xlim(-0.02, 1.02)

        # --- Bottom: Distributions ---
        sns.histplot(scores_neg, bins=50, kde=True, color="#3498db", stat="density", ax=axs[1], label="Negative Class")
        sns.histplot(scores_pos, bins=50, kde=True, color="#e74c3c", stat="density", ax=axs[1], label="Positive Class")
        axs[1].axvline(op_threshold, color="black", ls="--")
        axs[1].set_xlabel("Model Score", fontsize=12)
        axs[1].set_ylabel("Density", fontsize=12)

        if analysis_results["is_valid_policy"]:
            axs[1].axvspan(0, analysis_results["t_rule_out"], color="#3498db", alpha=0.15, zorder=0)
            axs[1].axvspan(analysis_results["t_rule_in"], 1, color="#e74c3c", alpha=0.15, zorder=0)

        fig.tight_layout()
        self._save_figure(fig, save_path, self.verbose)
        return fig

    def plot_safety_profile(
        self,
        op_threshold: float,
        save_path: Optional[str] = None,
        title: str = "Safety Profile Curve",
    ) -> plt.Figure:
        """
        Safety Profile curve with dual y-axes:
          - Left Y: Γ_Area (Operational Uncertainty) vs. Safety Level α
          - Right Y: % of cohort in the gray zone vs. α

        For each α in [min(min(PPV), min(NPV)), 1.0], compute:
          Γ_Area(α) from analyze_policy({α, α}, op_threshold)
          Gray-zone %: fraction with τ_minus ≤ score < τ_plus (if valid), else 0.
        """
        if self.verbose:
            print("-> Plotting safety profile curve...")

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax1 = plt.subplots(figsize=(8, 6))

        # Safety α range
        min_safety = float(min(self.df_metrics["PPV"].min(), self.df_metrics["NPV"].min()))
        alphas = np.linspace(min_safety, 1.0, 100)

        gamma_profile: List[float] = []
        gray_zone_cohort_pct: List[float] = []

        for alpha in alphas:
            policy = {"alpha_minus": float(alpha), "alpha_plus": float(alpha)}
            results = self.analyze_policy(policy, op_threshold)
            gamma_profile.append(float(results["gamma_area"]))

            if results["is_valid_policy"]:
                t_left, t_right = results["t_rule_out"], results["t_rule_in"]
                # Gray zone: [t_left, t_right)
                gray_zone_count = int(np.sum((self.scores >= t_left) & (self.scores < t_right)))
                gray_zone_cohort_pct.append(100.0 * gray_zone_count / len(self.labels))
            else:
                gray_zone_cohort_pct.append(0.0)

        # Left Y-axis for Γ_Area
        color1 = "#c0392b"
        ax1.set_xlabel("Safety Level (α)", fontsize=12)
        ax1.set_ylabel("Operational Uncertainty (Γ_Area)", fontsize=12, color=color1)
        line1, = ax1.plot(alphas, gamma_profile, color=color1, lw=2.5, label="Γ_Area")
        ax1.tick_params(axis="y", labelcolor=color1)

        max_gamma = max(gamma_profile) if len(gamma_profile) else 0.0
        upper_gamma = float(np.ceil(max(0.1, max_gamma) * 10) / 10)
        ax1.set_ylim(0, upper_gamma)
        ax1.yaxis.set_major_locator(MultipleLocator(0.1))

        # Right Y-axis for % Cohort in Gray Zone
        ax2 = ax1.twinx()
        color2 = "#8e44ad"
        ax2.set_ylabel("% of Cohort in Gray Zone", fontsize=12, color=color2)
        line2, = ax2.plot(alphas, gray_zone_cohort_pct, color=color2, lw=2.5, ls="--", label="% in Gray Zone")
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_ylim(0, 100)
        ax2.yaxis.set_major_locator(MultipleLocator(20))

        ax1.set_title(title, fontsize=16, weight="bold")
        ax1.legend(handles=[line1, line2], loc="upper left")

        fig.tight_layout()
        self._save_figure(fig, save_path, self.verbose)
        return fig

    # ---------------------- Policy Recommendation Methods ---------------------

    def recommend_policy_dual_safety(
        self, alpha_minus: float, alpha_plus: float
    ) -> Tuple[Optional[Dict[str, float]], str]:
        """
        Recommend τ_minus and τ_plus such that:
            NPV(τ_minus) ≥ alpha_minus, PPV(τ_plus) ≥ alpha_plus, and τ_minus < τ_plus.

        Returns
        -------
        (policy_dict | None, reason: str)
            policy_dict = {'tau_minus': float, 'tau_plus': float} on success, else None and reason.
        """
        df = self.df_metrics
        valid_npv = df[df["NPV"] >= alpha_minus]
        if valid_npv.empty:
            return None, f"NPV constraint unmet (Max NPV={df['NPV'].max():.3f})"
        tau_minus = float(valid_npv["Threshold"].max())

        valid_ppv = df[df["PPV"] >= alpha_plus]
        if valid_ppv.empty:
            return None, f"PPV constraint unmet (Max PPV={df['PPV'].max():.3f})"
        tau_plus = float(valid_ppv["Threshold"].min())

        if tau_minus >= tau_plus:
            return None, "Invalid: τ_minus ≥ τ_plus (zone overlap)"
        return {"tau_minus": tau_minus, "tau_plus": tau_plus}, "Success"

    def recommend_policy_capped_gray_rule_in(
        self, max_gray_pct: float, alpha_plus: float
    ) -> Tuple[Optional[Dict[str, float]], str]:
        """
        PPV-based rule-in safety with workload cap:
          1) Fix τ_plus = min threshold with PPV ≥ alpha_plus
          2) Choose τ_minus < τ_plus to maximize gray-zone size, subject to gray-zone ≤ max_gray_pct.

        Returns
        -------
        (policy_dict | None, reason: str)
        """
        df = self.df_metrics
        scores = self.scores
        n = len(scores)

        valid_ppv = df[df["PPV"] >= alpha_plus]
        if valid_ppv.empty:
            return None, f"PPV constraint unmet (Max PPV={df['PPV'].max():.4f})"
        tau_plus = float(valid_ppv["Threshold"].min())

        # For each candidate τ_minus < τ_plus, count gray-zone samples: τ_minus ≤ score < τ_plus
        thresholds = df["Threshold"].values
        n_gray_if_tau_minus = np.array([
            int(np.sum((scores >= t) & (scores < tau_plus))) for t in thresholds
        ])

        max_allowed = int(np.floor(n * max_gray_pct / 100.0))
        # Valid candidates: τ_minus < τ_plus and gray count ≤ cap
        mask_valid = (thresholds < tau_plus) & (n_gray_if_tau_minus <= max_allowed)

        if not np.any(mask_valid):
            return None, "No τ_minus yields gray-zone within cap."

        # Pick τ_minus that maximizes gray-zone count (within cap)
        idx_best = np.argmax(n_gray_if_tau_minus * mask_valid)
        tau_minus = float(thresholds[idx_best])

        if tau_minus >= tau_plus:
            return None, "Invalid: τ_minus ≥ τ_plus after selection."
        return {"tau_minus": tau_minus, "tau_plus": tau_plus}, "Success"

    def recommend_policy_capped_gray_rule_out(
        self, max_gray_pct: float, alpha_minus: float
    ) -> Tuple[Optional[Dict[str, float]], str]:
        """
        NPV-based rule-out safety with workload cap:
          1) Fix τ_minus = max threshold with NPV ≥ alpha_minus
          2) Choose τ_plus > τ_minus to maximize gray-zone size, subject to gray-zone ≤ max_gray_pct.

        Returns
        -------
        (policy_dict | None, reason: str)
        """
        df = self.df_metrics
        scores = self.scores
        n = len(scores)

        valid_npv = df[df["NPV"] >= alpha_minus]
        if valid_npv.empty:
            return None, f"NPV constraint unmet (Max NPV={df['NPV'].max():.4f})"
        tau_minus = float(valid_npv["Threshold"].max())

        thresholds = df["Threshold"].values
        # For each candidate τ_plus > τ_minus, count gray-zone samples: τ_minus ≤ score < τ_plus
        n_gray_if_tau_plus = np.array([
            int(np.sum((scores >= tau_minus) & (scores < t))) for t in thresholds
        ])

        max_allowed = int(np.floor(n * max_gray_pct / 100.0))
        mask_valid = (thresholds > tau_minus) & (n_gray_if_tau_plus <= max_allowed)

        if not np.any(mask_valid):
            return None, "No τ_plus yields gray-zone within cap."

        idx_best = np.argmax(n_gray_if_tau_plus * mask_valid)
        tau_plus = float(thresholds[idx_best])

        if tau_minus >= tau_plus:
            return None, "Invalid: τ_minus ≥ τ_plus after selection."
        return {"tau_minus": tau_minus, "tau_plus": tau_plus}, "Success"

    def recommend_policy_max_utility(
        self,
        utility: Dict[str, float],
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        Utility-based optimal policy search over all (τ_minus, τ_plus) pairs (τ_minus < τ_plus).

        Utility model (per-case contributions):
          total_utility = TN*U_TN + FP*U_FP + TP*U_TP + FN*U_FN + N_gray*cost_gray

        At a given τ_minus row we take TN,FN; at τ_plus row we take TP,FP.
        Gray-zone cases are the remainder.

        Parameters
        ----------
        utility : dict
            Keys: 'U_TN','U_FP','U_TP','U_FN','cost_gray' (floats)

        Returns
        -------
        best_policy : dict
            {'tau_minus','tau_plus','max_utility','breakdown':{'tp','tn','fp','fn','n_gray'}}
        utility_matrix : np.ndarray of shape (n_thr, n_thr)
            Utility value for each (τ_minus index, τ_plus index); invalid pairs are -inf.
        thresholds : np.ndarray
            The descending-sorted list of candidate thresholds.
        """
        df = self.df_metrics
        thresholds = np.sort(df["Threshold"].unique())[::-1]
        n_thr = len(thresholds)
        utility_matrix = np.full((n_thr, n_thr), -np.inf)

        # Fast lookup by threshold value
        df_lookup = df.set_index("Threshold")

        best = {"max_utility": -np.inf}

        for j, tau_minus in enumerate(thresholds):   # Y-axis
            row_m = df_lookup.loc[tau_minus]
            tn, fn = int(row_m["TN"]), int(row_m["FN"])
            total_cases = int(row_m["TP"] + row_m["FP"] + row_m["TN"] + row_m["FN"])

            for i, tau_plus in enumerate(thresholds):  # X-axis
                if tau_minus >= tau_plus:
                    continue

                row_p = df_lookup.loc[tau_plus]
                tp, fp = int(row_p["TP"]), int(row_p["FP"])

                n_gray = total_cases - (tn + fn + tp + fp)

                u = (
                    tn * utility.get("U_TN", 0.0)
                    + fn * utility.get("U_FN", 0.0)
                    + tp * utility.get("U_TP", 0.0)
                    + fp * utility.get("U_FP", 0.0)
                    + n_gray * utility.get("cost_gray", 0.0)
                )

                utility_matrix[j, i] = u

                if u > best["max_utility"]:
                    best = {
                        "max_utility": float(u),
                        "tau_minus": float(tau_minus),
                        "tau_plus": float(tau_plus),
                        "breakdown": {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "n_gray": int(n_gray)},
                    }

        return best, utility_matrix, thresholds

    # ---------------------------- Policy Visuals ------------------------------

    def plot_policy_overview(
        self,
        policy_thresholds: Dict[str, float],
        title: str = "Policy-Based Safety Analysis",
        subtitle: str = "",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        2×2 panel overview for a given policy {τ_minus, τ_plus}:
          (a) Score distributions + policy zones
          (b) Jitter plot + policy zones
          (c) Predictive values (PPV/NPV) vs. threshold
          (d) Policy-colored SA-ROC segment with Γ_Area hatch & cohort stats
        """
        tau_minus = float(policy_thresholds["tau_minus"])
        tau_plus = float(policy_thresholds["tau_plus"])

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, axs = plt.subplots(2, 2, figsize=(15, 13))
        fig.suptitle(title, fontsize=16, weight="bold")
        if subtitle:
            plt.figtext(0.5, 0.95, f"({subtitle})", ha="center", fontsize=10, style="italic")

        scores_neg = self.scores[self.labels == 0]
        scores_pos = self.scores[self.labels == 1]

        # ---------- (a) Score Distributions ----------
        ax = axs[0, 0]
        sns.histplot(scores_neg, bins=np.linspace(0, 1, 50), kde=True, color="#3498db", stat="density", ax=ax, label="Negative (Y=0)")
        sns.histplot(scores_pos, bins=np.linspace(0, 1, 50), kde=True, color="#e74c3c", stat="density", ax=ax, label="Positive (Y=1)")
        ax.axvline(tau_minus, color="#3498db", ls="--", label=f"τ- = {tau_minus:.3f}")
        ax.axvline(tau_plus, color="#e74c3c", ls="--", label=f"τ+ = {tau_plus:.3f}")
        ax.axvspan(tau_minus, tau_plus, color="#bdc3c7", alpha=0.2, label="Gray Zone")
        ax.set_title("a) Score Distributions & Policy Zones", fontsize=12, loc="left", weight="bold")
        ax.legend()
        ax.grid(alpha=0.4)
        ax.set_xlim(0, 1)

        # ---------- (b) Jitter Plot ----------
        ax = axs[0, 1]
        np.random.seed(42)
        ax.scatter(scores_neg, np.random.normal(0.75, 0.05, len(scores_neg)),
                   s=15, color="#3498db", alpha=0.5, edgecolor="k", linewidth=0.5)
        ax.scatter(scores_pos, np.random.normal(0.25, 0.05, len(scores_pos)),
                   s=15, color="#e74c3c", alpha=0.5, edgecolor="k", linewidth=0.5)
        ax.axvspan(0, tau_minus, color="#3498db", alpha=0.2, zorder=0, label="Rule-out SZ")
        ax.axvspan(tau_plus, 1, color="#e74c3c", alpha=0.2, zorder=0, label="Rule-in SZ")
        ax.axvspan(tau_minus, tau_plus, color="#bdc3c7", alpha=0.2, zorder=0, label="Gray Zone")
        ax.set_title("b) Individual Scores & Policy Zones", fontsize=12, loc="left", weight="bold")
        ax.set_yticks([])
        ax.legend(loc="lower left")
        ax.set_xlim(0, 1)

        # ---------- (c) Predictive Value vs Threshold ----------
        ax = axs[1, 0]
        ax.plot(self.df_metrics["Threshold"], self.df_metrics["NPV"], color="#3498db", label="NPV")
        ax.plot(self.df_metrics["Threshold"], self.df_metrics["PPV"], color="#e74c3c", label="PPV")
        ax.axvline(tau_minus, color="#3498db", ls="--", lw=1.5)
        ax.axvline(tau_plus, color="#e74c3c", ls="--", lw=1.5)
        ax.axvspan(tau_minus, tau_plus, color="#bdc3c7", alpha=0.2)
        ax.set_title("c) Predictive Value vs. Threshold", fontsize=12, loc="left", weight="bold")
        ax.legend()
        ax.grid(True, alpha=0.4)
        ax.set_xlim(0, 1)

        # ---------- (d) Policy-Driven SA-ROC ----------
        ax = axs[1, 1]
        ax.set_aspect("equal", "box")

        # Color ROC segments by where their mid-threshold falls
        df = self.df_metrics.reset_index(drop=True)
        for i in range(len(df) - 1):
            t_mid = 0.5 * (df.loc[i, "Threshold"] + df.loc[i + 1, "Threshold"])
            color = "#95a5a6"  # default gray
            if t_mid < tau_minus:
                color = "#3498db"  # rule-out side
            elif t_mid >= tau_plus:
                color = "#e74c3c"  # rule-in side
            ax.plot(df["FPR"][i:i+2], df["TPR"][i:i+2], color=color, linewidth=5)

        # Diagonal
        ax.plot([0, 1], [0, 1], color="black", ls="--", lw=1)

        # Locate rows closest to τ_minus and τ_plus
        idx_minus = int(np.argmin(np.abs(df["Threshold"].values - tau_minus)))
        row_minus = df.iloc[idx_minus]
        idx_plus = int(np.argmin(np.abs(df["Threshold"].values - tau_plus)))
        row_plus = df.iloc[idx_plus]

        # Γ_Area rectangle & label (only if τ_minus < τ_plus)
        gamma_area = row_minus["FPR"] * (1 - row_plus["TPR"]) if tau_minus < tau_plus else 0.0

        # Cohort decomposition by zones
        n = len(self.scores)
        n_rule_out = int(np.sum(self.scores < tau_minus))
        n_rule_in = int(np.sum(self.scores >= tau_plus))
        n_gray = int(n - n_rule_out - n_rule_in)

        p_rule_out = 100.0 * n_rule_out / n
        p_rule_in = 100.0 * n_rule_in / n
        p_gray = 100.0 * n_gray / n

        ax.add_patch(
            plt.Rectangle(
                (0, row_plus["TPR"]),
                row_minus["FPR"],
                1 - row_plus["TPR"],
                facecolor="gray",
                alpha=0.25,
                zorder=0,
                hatch="///",
                label=f"Policy Γ_Area: {gamma_area:.3f} (GZ: {p_gray:.1f}%)",
            )
        )

        # Markers for τ- and τ+
        ax.plot(row_minus["FPR"], row_minus["TPR"], "D", ms=10, color="#3498db", mec="k", label=f"τ- ({tau_minus:.3f})")
        ax.plot(row_plus["FPR"], row_plus["TPR"], "*", ms=14, color="#e74c3c", mec="k", label=f"τ+ ({tau_plus:.3f})")

        # Inline analysis text
        analysis_text = (
            f"Policy Outcome:\n"
            f"-----------------\n"
            f"Rule-out Zone: {p_rule_out:.1f}% Cohort\n"
            f"  - Achieved NPV: {row_minus['NPV']:.3f}\n"
            f"Rule-in Zone: {p_rule_in:.1f}% Cohort\n"
            f"  - Achieved PPV: {row_plus['PPV']:.3f}\n"
            f"Gray Zone: {p_gray:.1f}% Cohort"
        )
        ax.text(
            0.97, 0.50, analysis_text, transform=ax.transAxes, fontsize=9,
            va="top", ha="right", bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8)
        )

        ax.set_title(f"d) Policy-Driven SA-ROC (AUC: {self.roc_auc:.3f})", fontsize=12, loc="left", weight="bold")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.4)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

        fig.tight_layout(rect=[0, 0.03, 1, 0.93])
        self._save_figure(fig, save_path, self.verbose)
        return fig

    def plot_utility_heatmap_and_breakdown(
        self,
        utility_matrix: np.ndarray,
        thresholds: np.ndarray,
        optimal_policy: Dict[str, float],
        utility: Dict[str, float],
        title: str = "Utility Analysis",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Two-panel figure:
        (a) Utility heatmap over (tau-, tau+) with optimal cell highlighted
        (b) Breakdown bars: counts (TP/TN/FP/FN/GZ) and utility contributions

        Layout note
        -----------
        - Uses a 1x3 GridSpec: [heatmap | colorbar | breakdown]
        - `constrained_layout=True` to keep both panels aligned in height
        - Axes fixed to [0,1] with 0.2 tick spacing (no distortion)
        """
        import matplotlib.patheffects as pe

        plt.style.use("seaborn-v0_8-whitegrid")
        fig = plt.figure(figsize=(19, 9), constrained_layout=True)
        gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[1.0, 0.045, 1.12])

        # ----------------------- (a) Heatmap panel -----------------------
        ax_h = fig.add_subplot(gs[0])

        # Copy and sanitize utility matrix
        U = np.array(utility_matrix, copy=True)
        valid = U[U != -np.inf]
        min_util = np.min(valid) if valid.size > 0 else 0.0
        U[U == -np.inf] = min_util

        # Reorder thresholds to ascending for conventional axes (0→1)
        thr = np.asarray(thresholds, dtype=float)
        asc_idx = np.argsort(thr)
        thr_asc = thr[asc_idx]
        U_asc = U[np.ix_(asc_idx, asc_idx)]

        # Draw heatmap with uniform data coords (0..1)
        n = len(thr_asc)
        im = ax_h.imshow(
            U_asc,
            origin="lower",
            cmap="viridis",
            interpolation="nearest",
            extent=(0.0, 1.0, 0.0, 1.0),  # x: tau+ in [0,1], y: tau- in [0,1]
            aspect="auto",                # fill the allotted subplot height (keeps panels aligned)
        )

        # Dedicated colorbar axis (keeps both panels same height)
        cax = fig.add_subplot(gs[1])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Total Expected Utility")

        # Locate optimal (tau-, tau+) → normalized cell box
        def _nearest_idx(arr: np.ndarray, target: float) -> int:
            return int(np.argmin(np.abs(arr - float(target))))

        tau_minus_opt = float(optimal_policy["tau_minus"])
        tau_plus_opt  = float(optimal_policy["tau_plus"])

        y_idx = _nearest_idx(thr_asc, tau_minus_opt)
        x_idx = _nearest_idx(thr_asc, tau_plus_opt)

        cell_w = 1.0 / n
        cell_h = 1.0 / n
        x0 = x_idx * cell_w
        y0 = y_idx * cell_h

        # Red rectangle around the optimal cell
        ax_h.add_patch(
            mpatches.Rectangle(
                (x0, y0), cell_w, cell_h,
                fill=False, edgecolor="red", lw=3, zorder=4,
            )
        )

        # Large red circle at the center of the optimal cell (with white halo)
        marker = ax_h.plot(
            x0 + 0.5 * cell_w,
            y0 + 0.5 * cell_h,
            marker="o",
            markersize=18,
            markerfacecolor="none",
            markeredgecolor="red",
            markeredgewidth=3,
            linestyle="None",
            zorder=5,
        )[0]
        marker.set_path_effects([pe.withStroke(linewidth=5, foreground="white")])

        # Axes: fixed 0..1 with 0.2-step ticks
        ticks = np.linspace(0.0, 1.0, 6)
        ax_h.set_xticks(ticks)
        ax_h.set_yticks(ticks)
        ax_h.set_xticklabels([f"{t:.1f}" for t in ticks])
        ax_h.set_yticklabels([f"{t:.1f}" for t in ticks])
        ax_h.set_xlim(0.0, 1.0)
        ax_h.set_ylim(0.0, 1.0)

        ax_h.set_xlabel("tau+ (Rule-in Threshold)")
        ax_h.set_ylabel("tau- (Rule-out Threshold)")
        ax_h.set_title("a) Utility Landscape", fontsize=14, weight="bold", loc="left")

        # Optional callout (keep/remove as you prefer)
        ax_h.text(
            0.98, 0.02,
            f"Optimal = red circle & outline\nUtility = {optimal_policy['max_utility']:.0f}",
            transform=ax_h.transAxes,
            ha="right", va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.95),
            zorder=6,
        )

        # ----------------------- (b) Breakdown panel -----------------------
        ax_b = fig.add_subplot(gs[2])

        labels = ["TP", "TN", "FP", "FN", "Gray Zone"]
        brk = optimal_policy["breakdown"]
        counts = [brk["tp"], brk["tn"], brk["fp"], brk["fn"], brk["n_gray"]]
        utils = [
            brk["tp"] * utility.get("U_TP", 0.0),
            brk["tn"] * utility.get("U_TN", 0.0),
            brk["fp"] * utility.get("U_FP", 0.0),
            brk["fn"] * utility.get("U_FN", 0.0),
            brk["n_gray"] * utility.get("cost_gray", 0.0),
        ]

        colors = ["#2ecc71", "#3498db", "#e74c3c", "#c0392b", "#95a5a6"]

        bars = ax_b.bar(labels, counts, color=colors, alpha=0.8, label="Case Count")
        ax_b.set_ylabel("Number of Cases", color="black")
        ax_b.bar_label(bars, padding=3)

        ax_t = ax_b.twinx()
        bar_w = 0.3
        positions = np.arange(len(labels)) + bar_w
        util_colors = ["green" if u >= 0 else "red" for u in utils]
        bars_u = ax_t.bar(
            positions, utils, width=bar_w, color=util_colors, alpha=0.6, label="Utility Contribution"
        )
        ax_t.set_ylabel("Utility Score", color="black")
        ax_t.axhline(0, color="gray", linestyle="--")
        ax_t.bar_label(bars_u, padding=3, fmt="%.0f")

        ax_b.set_title("b) Breakdown of Optimal Policy", fontsize=14, weight="bold", loc="left")

        fig.suptitle(title, fontsize=16, weight="bold")
        # NOTE: constrained_layout handles spacing; no tight_layout needed.

        self._save_figure(fig, save_path, self.verbose)
        return fig

# If needed, you can add CLI or tutorial in a separate script/notebook.
if __name__ == "__main__":
    pass