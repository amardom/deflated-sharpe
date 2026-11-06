import numpy as np

from .clustering import cluster_sharpes
from .onc import cluster_trials as cluster_reference_trials, compute_cluster_sharpes
from .policy import assess_groups
from .statistics import (compute_deflated_sharpe_ratio, compute_expected_maximum_sharpe,
                         finite_number, compute_probabilistic_sharpe_ratio)

def validate_returns(returns, preserve_relative_scale=False):
    try:
        values = np.asarray(returns, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("returns must be a rectangular numeric matrix") from error
    if values.ndim != 2 or values.shape[0] < 4 or values.shape[1] < 2:
        raise ValueError("returns require at least 4 observations and 2 trials")
    if not np.isfinite(values).all():
        raise ValueError("returns must contain only finite values")
    scales = np.max(np.abs(values), axis=0)
    scales[scales == 0] = 1
    values = values / (scales.max() if preserve_relative_scale else scales)
    if (np.std(values, axis=0, ddof=1) == 0).any():
        raise ValueError("each trial must have positive return variance")
    return values

def estimate_independent_trials(returns, method="auto"):
    values = validate_returns(returns, preserve_relative_scale=method=="correlation_clusters")
    count = values.shape[1]
    correlation = np.corrcoef(values, rowvar=False)
    average = float((correlation.sum() - count) / (count * (count - 1)))
    average = min(1.0, max(-1.0, average))
    average = 1.0 if np.isclose(average, 1, rtol=0, atol=1e-12) else average
    details = {"supplied_trials": count, "unique_trials": None, "count_source": "estimated",
               "mean_correlation": average, "method": method,
               "method_reference": "experimental_auto" if method=="auto" else "bailey_lopez_de_prado_2014_A3"}
    if method == "auto":
        return {**details, **assess_groups(values)}
    if method == "average_correlation":
        return {**details, "independent_trials": average + (1 - average) * count}
    if method != "correlation_clusters":
        raise ValueError("method must be auto, average_correlation or correlation_clusters")
    labels, quality = cluster_reference_trials(values)
    return {**details, "method_reference": "lopez_de_prado_lewis_2018_8_A1_A2_A4",
            "independent_trials": int(len(np.unique(labels))),
            "cluster_labels": labels.tolist(), "cluster_quality": quality}

def summarize(returns, trial_names=None, independent_trials=None,
              trial_method="auto", sharpe_variance=None):
    values = validate_returns(returns, preserve_relative_scale=trial_method=="correlation_clusters")
    observations, count = values.shape
    if trial_names is None:
        trial_names = [f"trial_{i + 1}" for i in range(count)]
    elif isinstance(trial_names, (str, bytes, dict)):
        raise ValueError("trial_names must be an iterable of unique nonempty strings")
    else:
        try:
            trial_names = list(trial_names)
        except TypeError as error:
            raise ValueError("trial_names must be an iterable of unique nonempty strings") from error
    if (len(trial_names) != count or any(not isinstance(name, str) or not name for name in trial_names)
            or len(set(trial_names)) != count):
        raise ValueError("trial_names must contain one unique nonempty string per trial")
    estimate = estimate_independent_trials(values, method=trial_method)
    if independent_trials is None:
        independent_trials = estimate["independent_trials"]
        source = estimate.get("count_source", "estimated")
    else:
        source = "supplied"
    sharpes = values.mean(axis=0) / values.std(axis=0, ddof=1)
    raw_variance = float(np.var(sharpes, ddof=1))
    representatives = None
    if "cluster_labels" in estimate:
        labels = np.asarray(estimate["cluster_labels"])
        if trial_method == "correlation_clusters":
            representatives = compute_cluster_sharpes(values,labels)
        else:
            representatives = cluster_sharpes(values,labels)
        variance = float(np.var(representatives, ddof=1)) if len(representatives) > 1 else 0.0
        variance_source = estimate.get("variance_source", "cluster_representatives")
    else:
        variance = raw_variance
        variance_source = "all_trial_sharpes"
    if sharpe_variance is not None:
        variance = finite_number("sharpe_variance", sharpe_variance)
        variance_source = "supplied"
    threshold = compute_expected_maximum_sharpe(variance, independent_trials)
    centered = values - values.mean(axis=0)
    second = np.mean(centered ** 2, axis=0)
    skewness = np.mean(centered ** 3, axis=0) / second ** 1.5
    kurtosis = np.mean(centered ** 4, axis=0) / second ** 2
    rows = []
    for i, name in enumerate(trial_names):
        sharpe, skew, kurt = float(sharpes[i]), float(skewness[i]), float(kurtosis[i])
        rows.append({"trial": name, "observations": observations, "sharpe": sharpe,
                     "skewness": skew, "pearson_kurtosis": kurt, "supplied_trials": count,
                     "estimated_independent_trials": float(estimate["independent_trials"]),
                     "independent_trials": float(independent_trials), "count_source": source,
                     "estimation_method": estimate["method"],
                     "method_reference": estimate["method_reference"],
                     "count_method": estimate["method"] if source != "supplied" else "caller",
                     "mean_correlation": estimate["mean_correlation"],
                     "selection_reason": estimate.get("selection_reason", "explicit_method"),
                     "unique_trials": estimate.get("unique_trials"),
                     "structure_reason": estimate.get("structure_reason"),
                     "candidate_groups": estimate.get("candidate_groups"),
                     "within_lower_bound": estimate.get("within_lower_bound"),
                     "between_upper_bound": estimate.get("between_upper_bound"),
                     "raw_sharpe_variance": raw_variance,
                     "sharpe_variance": variance, "variance_source": variance_source,
                     "cluster": estimate.get("cluster_labels", [None] * count)[i],
                     "cluster_quality": estimate.get("cluster_quality"),
                     "representative_sharpe": representatives[estimate["cluster_labels"][i]] if representatives is not None else None,
                     "selection_threshold": threshold,
                     "psr": compute_probabilistic_sharpe_ratio(sharpe, observations, skew, kurt),
                     "dsr": compute_deflated_sharpe_ratio(sharpe, observations, skew, kurt,
                                                 variance, independent_trials)})
    return rows
