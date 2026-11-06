from statistics import NormalDist

import numpy as np

from .clustering import cluster_trials, deduplicate_trials

WITHIN_CORRELATION = 0.95
BETWEEN_CORRELATION = 0.20
FAMILY_ERROR = 0.05
MIN_SPLIT_OBSERVATIONS = 20

def assess_unique_groups(values):
    observations, count = values.shape
    result = {"independent_trials": count, "count_source": "nominal_fallback",
              "selection_reason": "insufficient_observations", "candidate_groups": None,
              "within_lower_bound": None, "between_upper_bound": None}
    if np.array_equal(values, np.repeat(values[:, :1], count, axis=1)):
        result.update(independent_trials=1, count_source="estimated",
                      selection_reason="identical_trials", candidate_groups=1,
                      cluster_labels=[0] * count, cluster_quality=None)
        return result
    split = observations // 2
    if split < MIN_SPLIT_OBSERVATIONS:
        return result
    training, holdout = values[:split], values[split:]
    if (np.ptp(training, axis=0) == 0).any() or (np.ptp(holdout, axis=0) == 0).any():
        result["selection_reason"] = "constant_split_column"
        return result
    labels, quality = cluster_trials(training)
    result["candidate_groups"] = int(len(np.unique(labels)))
    left, right = np.triu_indices(count, 1)
    within = labels[left] == labels[right]
    if not within.any():
        result["selection_reason"] = "no_redundant_groups"
        return result
    correlations = np.corrcoef(holdout, rowvar=False)[left, right]
    correlations = np.clip(correlations, -1 + 1e-15, 1 - 1e-15)
    critical = -NormalDist().inv_cdf(FAMILY_ERROR / (2 * len(left)))
    margin = critical / np.sqrt(len(holdout) - 3)
    lower = np.tanh(np.arctanh(correlations) - margin)
    upper = np.tanh(np.arctanh(correlations) + margin)
    within_lower = float(lower[within].min())
    between_upper = float(np.maximum(np.abs(lower[~within]), np.abs(upper[~within])).max()) if (~within).any() else 0.0
    result.update(within_lower_bound=within_lower, between_upper_bound=between_upper)
    if within_lower < WITHIN_CORRELATION:
        result["selection_reason"] = "within_group_redundancy_not_confirmed"
        return result
    if between_upper > BETWEEN_CORRELATION:
        result["selection_reason"] = "between_group_separation_not_confirmed"
        return result
    result.update(independent_trials=result["candidate_groups"], count_source="estimated",
                  selection_reason="heldout_groups_supported", cluster_labels=labels.tolist(),
                  cluster_quality=quality)
    return result

def assess_groups(values):
    unique_values, duplicate_labels = deduplicate_trials(values)
    unique_count = unique_values.shape[1]
    if unique_count == values.shape[1]:
        result = assess_unique_groups(values)
    else:
        result = assess_unique_groups(unique_values)
        structure_reason = result["selection_reason"]
        if "cluster_labels" in result:
            result["cluster_labels"] = np.asarray(result["cluster_labels"])[duplicate_labels].tolist()
        else:
            result.update(independent_trials=unique_count, count_source="deduplicated_fallback",
                          selection_reason="exact_duplicates_removed", cluster_labels=duplicate_labels.tolist(),
                          cluster_quality=None, variance_source="unique_trial_sharpes")
        result["structure_reason"] = structure_reason
    result["unique_trials"] = unique_count
    result.setdefault("structure_reason", result["selection_reason"])
    return result
