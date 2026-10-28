# -*- coding: utf-8 -*-
"""
detector.py — Five individual drift detection functions.

Each function is self-contained and returns a standardised result dict
with keys: test_name, drift_detected, severity, and test-specific scores.

Severity levels:
    "stable"   → no drift detected
    "warning"  → marginal drift detected
    "critical" → strong drift detected

Supported tests:
    1. ks_test()          → KS Test (numeric features)
    2. chi2_test()        → Chi-Squared Test (categorical features)
    3. psi_test()         → Population Stability Index (numeric)
    4. js_divergence()    → Jensen-Shannon Divergence (numeric & categorical)
    5. ddm_concept_drift()→ DDM error-rate concept drift (binary classification)
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp

import drift_detector.config as config
from drift_detector.config import get_logger

log = get_logger(__name__)

# Shared return-dict type alias
DriftResult = Dict[str, Any]


# ─────────────────────────────────────────────
#  Helper: Severity Classifier
# ─────────────────────────────────────────────

def _classify_severity(drift_detected: bool, strong: bool = False) -> str:
    """Map drift flags to a three-level severity string."""
    if not drift_detected:
        return "stable"
    return "critical" if strong else "warning"


# ─────────────────────────────────────────────
#  1. KS Test (Kolmogorov-Smirnov)
# ─────────────────────────────────────────────

def ks_test(ref_series: pd.Series, cur_series: pd.Series) -> DriftResult:
    """
    Kolmogorov-Smirnov two-sample test for numeric feature drift.

    Measures the maximum absolute difference between the two empirical
    cumulative distribution functions.  A small p-value indicates
    statistically significant distributional change.

    Args:
        ref_series: Reference (baseline) values — 1-D numeric Series.
        cur_series: Current (production) values — 1-D numeric Series.

    Returns:
        {
            test_name      : "KS Test",
            statistic      : float — KS distance D ∈ [0, 1],
            p_value        : float — two-sided p-value,
            threshold      : float — configured p-value threshold,
            drift_detected : bool,
            severity       : str — "stable" | "warning" | "critical",
            interpretation : str — human-readable summary,
        }
    """
    ref = ref_series.dropna().astype(float)
    cur = cur_series.dropna().astype(float)

    if len(ref) == 0 or len(cur) == 0:
        log.warning("KS Test skipped — one or both series are empty after dropping NaN.")
        return _empty_result("KS Test")

    stat, p_val = ks_2samp(ref.values, cur.values)

    drift = p_val < config.KS_P_THRESHOLD
    # Classify as critical if p-value is very low (< 0.001)
    strong = p_val < 0.001
    severity = _classify_severity(drift, strong)

    interp = (
        f"KS statistic D={stat:.4f}, p={p_val:.4f}. "
        + ("Distributions differ significantly." if drift else "No significant distributional shift.")
    )

    log.debug(f"KS Test → stat={stat:.4f}, p={p_val:.4f}, drift={drift}, severity={severity}")
    return {
        "test_name": "KS Test",
        "statistic": round(float(stat), 6),
        "p_value": round(float(p_val), 6),
        "threshold": config.KS_P_THRESHOLD,
        "drift_detected": drift,
        "severity": severity,
        "interpretation": interp,
    }


# ─────────────────────────────────────────────
#  2. Chi-Squared Test
# ─────────────────────────────────────────────

def chi2_test(ref_series: pd.Series, cur_series: pd.Series) -> DriftResult:
    """
    Chi-Squared test for categorical feature drift.

    Builds a contingency table from value counts and applies the
    chi-squared test of independence.  A small p-value indicates
    that category proportions have shifted.

    Args:
        ref_series: Reference (baseline) categorical Series.
        cur_series: Current (production) categorical Series.

    Returns:
        {
            test_name      : "Chi-Squared Test",
            statistic      : float — chi-squared statistic,
            p_value        : float,
            degrees_freedom: int,
            threshold      : float,
            drift_detected : bool,
            severity       : str,
            interpretation : str,
        }
    """
    ref = ref_series.dropna().astype(str)
    cur = cur_series.dropna().astype(str)

    if len(ref) == 0 or len(cur) == 0:
        log.warning("Chi-Squared Test skipped — one or both series are empty.")
        return _empty_result("Chi-Squared Test")

    # Build a combined contingency table (both series aligned to all categories)
    all_cats = sorted(set(ref.unique()) | set(cur.unique()))
    ref_counts = ref.value_counts().reindex(all_cats, fill_value=0)
    cur_counts = cur.value_counts().reindex(all_cats, fill_value=0)
    contingency = np.array([ref_counts.values, cur_counts.values])

    try:
        stat, p_val, dof, _ = chi2_contingency(contingency)
    except ValueError as e:
        log.warning(f"Chi-Squared Test failed: {e}")
        return _empty_result("Chi-Squared Test")

    drift = p_val < config.CHI2_P_THRESHOLD
    strong = p_val < 0.001
    severity = _classify_severity(drift, strong)

    interp = (
        f"χ²={stat:.4f}, df={dof}, p={p_val:.4f}. "
        + ("Category proportions have shifted significantly." if drift
           else "Category proportions are stable.")
    )

    log.debug(f"Chi2 Test → stat={stat:.4f}, p={p_val:.4f}, drift={drift}, severity={severity}")
    return {
        "test_name": "Chi-Squared Test",
        "statistic": round(float(stat), 6),
        "p_value": round(float(p_val), 6),
        "degrees_freedom": int(dof),
        "threshold": config.CHI2_P_THRESHOLD,
        "drift_detected": drift,
        "severity": severity,
        "interpretation": interp,
    }


# ─────────────────────────────────────────────
#  3. PSI — Population Stability Index
# ─────────────────────────────────────────────

def psi_test(ref_series: pd.Series, cur_series: pd.Series) -> DriftResult:
    """
    Population Stability Index (PSI) for numeric feature drift.

    PSI measures how much a distribution has shifted relative to a
    reference using percentile-based binning of the reference data.

    PSI Interpretation (industry standard):
        < 0.10  → Stable (no drift)
        0.10–0.20 → Minor drift (warning)
        > 0.20  → Significant drift (critical)

    Args:
        ref_series: Reference (baseline) numeric Series.
        cur_series: Current (production) numeric Series.

    Returns:
        {
            test_name        : "PSI",
            psi_score        : float — PSI value,
            warning_threshold: float,
            critical_threshold: float,
            drift_detected   : bool,
            severity         : str,
            interpretation   : str,
        }
    """
    ref = ref_series.dropna().astype(float).values
    cur = cur_series.dropna().astype(float).values

    if len(ref) == 0 or len(cur) == 0:
        log.warning("PSI Test skipped — one or both series are empty.")
        return _empty_result("PSI")

    eps = 1e-8
    bins = config.PSI_BINS

    # Create bins from reference percentiles for stable, data-driven breakpoints
    quantiles = np.linspace(0, 100, bins + 1)
    breakpoints = np.unique(np.percentile(ref, quantiles))

    if len(breakpoints) <= 1:
        log.warning("PSI Test: too few unique values to create bins — returning 0.")
        psi_score = 0.0
    else:
        ref_counts, _ = np.histogram(ref, bins=breakpoints)
        cur_counts, _ = np.histogram(cur, bins=breakpoints)

        ref_pct = ref_counts / (len(ref) + eps)
        cur_pct = cur_counts / (len(cur) + eps)

        psi_vals = (ref_pct - cur_pct) * np.log((ref_pct + eps) / (cur_pct + eps))
        psi_score = float(np.sum(psi_vals))

    drift = psi_score >= config.PSI_WARNING_THRESHOLD
    strong = psi_score >= config.PSI_CRITICAL_THRESHOLD
    severity = _classify_severity(drift, strong)

    if psi_score < config.PSI_WARNING_THRESHOLD:
        interp = f"PSI={psi_score:.4f} — Stable, no significant population shift."
    elif psi_score < config.PSI_CRITICAL_THRESHOLD:
        interp = f"PSI={psi_score:.4f} — Minor shift detected (Warning)."
    else:
        interp = f"PSI={psi_score:.4f} — Significant population shift (Critical)."

    log.debug(f"PSI → score={psi_score:.4f}, drift={drift}, severity={severity}")
    return {
        "test_name": "PSI",
        "psi_score": round(psi_score, 6),
        "warning_threshold": config.PSI_WARNING_THRESHOLD,
        "critical_threshold": config.PSI_CRITICAL_THRESHOLD,
        "drift_detected": drift,
        "severity": severity,
        "interpretation": interp,
    }


# ─────────────────────────────────────────────
#  4. Jensen-Shannon Divergence
# ─────────────────────────────────────────────

def js_divergence(ref_series: pd.Series, cur_series: pd.Series) -> DriftResult:
    """
    Jensen-Shannon Divergence between reference and current distributions.

    Works for both numeric (histogram-based) and categorical (frequency-based)
    features.  JS divergence is symmetric, bounded ∈ [0, 1], and more
    numerically stable than KL divergence.

    Args:
        ref_series: Reference Series (numeric or categorical).
        cur_series: Current Series (numeric or categorical).

    Returns:
        {
            test_name      : "JS Divergence",
            js_score       : float — JS divergence ∈ [0, 1],
            threshold      : float,
            drift_detected : bool,
            severity       : str,
            interpretation : str,
        }
    """
    ref = ref_series.dropna()
    cur = cur_series.dropna()

    if len(ref) == 0 or len(cur) == 0:
        log.warning("JS Divergence skipped — one or both series are empty.")
        return _empty_result("JS Divergence")

    # Build probability distributions
    if pd.api.types.is_numeric_dtype(ref):
        # Histogram-based for numeric: use reference breakpoints
        combined = np.concatenate([ref.astype(float).values, cur.astype(float).values])
        bins = min(config.PSI_BINS, len(np.unique(combined)))
        bin_edges = np.histogram_bin_edges(ref.astype(float).values, bins=bins)

        ref_hist, _ = np.histogram(ref.astype(float).values, bins=bin_edges, density=False)
        cur_hist, _ = np.histogram(cur.astype(float).values, bins=bin_edges, density=False)

        ref_prob = (ref_hist + 1e-10) / (ref_hist.sum() + 1e-10 * len(ref_hist))
        cur_prob = (cur_hist + 1e-10) / (cur_hist.sum() + 1e-10 * len(cur_hist))
    else:
        # Frequency-based for categorical
        all_cats = sorted(set(ref.astype(str).unique()) | set(cur.astype(str).unique()))
        ref_counts = ref.astype(str).value_counts().reindex(all_cats, fill_value=0)
        cur_counts = cur.astype(str).value_counts().reindex(all_cats, fill_value=0)

        ref_prob = (ref_counts.values + 1e-10) / (ref_counts.sum() + 1e-10 * len(all_cats))
        cur_prob = (cur_counts.values + 1e-10) / (cur_counts.sum() + 1e-10 * len(all_cats))

    js_score = float(jensenshannon(ref_prob, cur_prob))
    # jensenshannon returns sqrt of JS divergence; square for the actual divergence
    js_score = min(js_score ** 2, 1.0)

    drift = js_score >= config.JS_DIVERGENCE_THRESHOLD
    strong = js_score >= 0.20
    severity = _classify_severity(drift, strong)

    interp = (
        f"JS divergence={js_score:.4f}. "
        + ("Distributions are significantly different." if drift
           else "Distributions are similar.")
    )

    log.debug(f"JS Divergence → score={js_score:.4f}, drift={drift}, severity={severity}")
    return {
        "test_name": "JS Divergence",
        "js_score": round(js_score, 6),
        "threshold": config.JS_DIVERGENCE_THRESHOLD,
        "drift_detected": drift,
        "severity": severity,
        "interpretation": interp,
    }


# ─────────────────────────────────────────────
#  5. DDM — Concept Drift via Error Rate
# ─────────────────────────────────────────────

def ddm_concept_drift(
    y_true_ref: np.ndarray,
    y_pred_ref: np.ndarray,
    y_true_cur: np.ndarray,
    y_pred_cur: np.ndarray,
) -> DriftResult:
    """
    Drift Detection Method (DDM) for concept drift.

    DDM monitors the classification error rate.  It establishes a
    baseline mean and standard deviation on reference data, then flags
    drift when the current error rate deviates beyond configurable
    sigma thresholds.

    Concept: if P(Y|X) has changed (concept drift), a trained model's
    error rate will increase on current data.

    Args:
        y_true_ref: Ground-truth labels on reference data (0/1 array).
        y_pred_ref: Model's binary predictions on reference data.
        y_true_cur: Ground-truth labels on current data.
        y_pred_cur: Model's binary predictions on current data.

    Returns:
        {
            test_name       : "DDM Concept Drift",
            error_rate_ref  : float — baseline error rate,
            error_rate_cur  : float — current error rate,
            error_delta     : float — absolute change in error rate,
            z_score         : float — deviation in std deviations,
            warning_level   : float — configured warning sigma,
            alert_level     : float — configured alert sigma,
            drift_detected  : bool,
            severity        : str,
            interpretation  : str,
        }
    """
    if len(y_true_ref) == 0 or len(y_true_cur) == 0:
        log.warning("DDM skipped — one or both label arrays are empty.")
        return _empty_result("DDM Concept Drift")

    # Compute error rates
    err_ref = float(np.mean(np.array(y_true_ref) != np.array(y_pred_ref)))
    err_cur = float(np.mean(np.array(y_true_cur) != np.array(y_pred_cur)))

    # DDM: baseline stats from reference
    # Standard deviation of Bernoulli error rate: sqrt(p*(1-p)/n)
    n_ref = len(y_true_ref)
    std_ref = float(np.sqrt(err_ref * (1.0 - err_ref) / n_ref)) if n_ref > 0 else 1.0

    delta = err_cur - err_ref  # positive = more errors on current data
    z_score = delta / (std_ref + 1e-10)

    warning_flag = z_score >= config.DDM_WARNING_LEVEL
    alert_flag   = z_score >= config.DDM_ALERT_LEVEL

    drift = warning_flag
    severity = "critical" if alert_flag else ("warning" if warning_flag else "stable")

    if alert_flag:
        interp = (
            f"ALERT: error rate jumped from {err_ref:.3f} → {err_cur:.3f} "
            f"(z={z_score:.2f}σ). Strong concept drift detected."
        )
    elif warning_flag:
        interp = (
            f"WARNING: error rate increased from {err_ref:.3f} → {err_cur:.3f} "
            f"(z={z_score:.2f}σ). Possible concept drift."
        )
    else:
        interp = (
            f"Error rate stable: ref={err_ref:.3f}, cur={err_cur:.3f} "
            f"(z={z_score:.2f}σ). No concept drift detected."
        )

    log.debug(
        f"DDM → err_ref={err_ref:.4f}, err_cur={err_cur:.4f}, "
        f"z={z_score:.2f}, severity={severity}"
    )
    return {
        "test_name": "DDM Concept Drift",
        "error_rate_ref": round(err_ref, 6),
        "error_rate_cur": round(err_cur, 6),
        "error_delta": round(delta, 6),
        "z_score": round(z_score, 4),
        "warning_level": config.DDM_WARNING_LEVEL,
        "alert_level": config.DDM_ALERT_LEVEL,
        "drift_detected": drift,
        "severity": severity,
        "interpretation": interp,
    }


# ─────────────────────────────────────────────
#  Helper: Empty Result (when test cannot run)
# ─────────────────────────────────────────────

def _empty_result(test_name: str) -> DriftResult:
    """Return a placeholder result when a test cannot be executed."""
    return {
        "test_name": test_name,
        "drift_detected": False,
        "severity": "unknown",
        "interpretation": f"{test_name} could not run — insufficient data.",
    }


# ─────────────────────────────────────────────
#  Per-Feature Dispatcher
# ─────────────────────────────────────────────

def run_all_tests_for_feature(
    feature_name: str,
    ref_series: pd.Series,
    cur_series: pd.Series,
    is_numeric: bool,
) -> Dict[str, DriftResult]:
    """
    Run all applicable drift tests for a single feature.

    Numeric features  → KS Test, PSI, JS Divergence
    Categorical features → Chi-Squared, JS Divergence

    Args:
        feature_name: Column name (for logging).
        ref_series: Reference values for this feature.
        cur_series: Current values for this feature.
        is_numeric: True for numeric columns, False for categorical.

    Returns:
        Dict mapping test name → DriftResult dict.
    """
    log.debug(f"Testing feature '{feature_name}' (numeric={is_numeric})")
    results: Dict[str, DriftResult] = {}

    if is_numeric:
        results["ks_test"] = ks_test(ref_series, cur_series)
        results["psi_test"] = psi_test(ref_series, cur_series)
        results["js_divergence"] = js_divergence(ref_series, cur_series)
    else:
        results["chi2_test"] = chi2_test(ref_series, cur_series)
        results["js_divergence"] = js_divergence(ref_series, cur_series)

    return results


def aggregate_feature_severity(test_results: Dict[str, DriftResult]) -> str:
    """
    Aggregate individual test severities into a single feature-level severity.

    Rule: take the worst (highest) severity across all tests.

    Args:
        test_results: Dict of test_name → DriftResult.

    Returns:
        Worst severity string: "stable" | "warning" | "critical" | "unknown"
    """
    rank = {"stable": 0, "unknown": 0, "warning": 1, "critical": 2}
    worst = "stable"
    for res in test_results.values():
        sev = res.get("severity", "stable")
        if rank.get(sev, 0) > rank.get(worst, 0):
            worst = sev
    return worst
