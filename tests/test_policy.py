import numpy as np
import pytest
from scipy.stats import pearsonr

from deflated_sharpe import estimate_independent_trials, summarize
from deflated_sharpe.policy import FAMILY_ERROR, assess_groups

def grouped(seed, observations=3000, noise=.05):
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(observations,3))
    return np.repeat(factors, 3, axis=1) + noise*rng.normal(size=(observations,9))

@pytest.mark.parametrize("seed", range(10))
def test_auto_supports_clear_redundant_groups(seed):
    result = estimate_independent_trials(grouped(seed))
    assert result["independent_trials"] == 3
    assert result["count_source"] == "estimated"
    assert result["selection_reason"] == "heldout_groups_supported"
    assert result["within_lower_bound"] >= .95
    assert result["between_upper_bound"] <= .2

@pytest.mark.parametrize("seed", range(20))
def test_auto_does_not_reduce_independent_trials(seed):
    values = np.random.default_rng(seed).normal(size=(1000,9))
    result = estimate_independent_trials(values)
    assert result["independent_trials"] == 9
    assert result["count_source"] == "nominal_fallback"
    assert result["selection_reason"] == "within_group_redundancy_not_confirmed"
    assert "cluster_labels" not in result

@pytest.mark.parametrize("seed", range(5))
def test_shared_factor_does_not_count_as_independent_groups(seed):
    rng = np.random.default_rng(seed)
    factors = np.sqrt(.6)*rng.normal(size=(3000,1)) + np.sqrt(.4)*rng.normal(size=(3000,3))
    values = np.repeat(factors,3,axis=1) + .02*rng.normal(size=(3000,9))
    result = estimate_independent_trials(values)
    assert result["independent_trials"] == 9
    assert result["selection_reason"] == "between_group_separation_not_confirmed"
    assert result["within_lower_bound"] > .95
    assert result["between_upper_bound"] > .2

def test_shift_in_dependence_is_rejected():
    values = grouped(3)
    values[1500:] = np.random.default_rng(9).normal(size=(1500,9))
    result = estimate_independent_trials(values)
    assert result["selection_reason"] == "within_group_redundancy_not_confirmed"
    assert result["independent_trials"] == 9

def test_short_history_and_two_distinct_trials():
    short = np.random.default_rng(2).normal(size=(10,3))
    result = estimate_independent_trials(short)
    assert result["selection_reason"] == "insufficient_observations"
    assert result["independent_trials"] == 3
    result = estimate_independent_trials(np.random.default_rng(2).normal(size=(100,2)))
    assert result["selection_reason"] == "no_redundant_groups"
    assert result["independent_trials"] == 2

@pytest.mark.parametrize("constant_half", [0,1])
def test_constant_split_columns_are_handled(constant_half):
    values = np.random.default_rng(9).normal(size=(100,3))
    values[constant_half*50:(constant_half+1)*50,0] = 1
    result = estimate_independent_trials(values)
    assert result["selection_reason"] == "constant_split_column"
    assert result["independent_trials"] == 3

def test_identical_returns_do_not_require_inference():
    x = np.arange(4, dtype=float)
    result = estimate_independent_trials(np.column_stack([x,x,x]))
    assert result["selection_reason"] == "identical_trials"
    assert result["independent_trials"] == 1
    assert summarize(np.column_stack([x,x,x]))[0]["variance_source"] == "cluster_representatives"

def test_perfect_correlation_without_identical_levels():
    x = np.random.default_rng(8).normal(size=200)
    result = estimate_independent_trials(np.column_stack([x, x+1]))
    assert result["independent_trials"] == 1
    assert result["selection_reason"] == "heldout_groups_supported"
    assert result["between_upper_bound"] == 0

def test_heldout_bounds_match_scipy():
    values = grouped(7)
    result = assess_groups(values)
    left, right = np.triu_indices(9,1)
    labels = np.asarray(result["cluster_labels"])
    confidence = 1-FAMILY_ERROR/len(left)
    bounds = [pearsonr(values[1500:,i],values[1500:,j]).confidence_interval(confidence)
              for i,j in zip(left,right)]
    within = labels[left] == labels[right]
    expected_lower = min(bound.low for bound,same in zip(bounds,within) if same)
    expected_upper = max(max(abs(bound.low),abs(bound.high))
                         for bound,same in zip(bounds,within) if not same)
    assert result["within_lower_bound"] == pytest.approx(expected_lower)
    assert result["between_upper_bound"] == pytest.approx(expected_upper)

def test_auto_summary_uses_matching_variance_unit():
    rows = summarize(grouped(5))
    assert rows[0]["count_method"] == "auto"
    assert rows[0]["count_source"] == "estimated"
    assert rows[0]["variance_source"] == "cluster_representatives"
    independent = np.random.default_rng(5).normal(size=(1000,9))
    row = summarize(independent)[0]
    assert row["count_source"] == "nominal_fallback"
    assert row["count_method"] == "auto"
    assert row["variance_source"] == "all_trial_sharpes"
    assert row["sharpe_variance"] == row["raw_sharpe_variance"]
    assert row["cluster"] is None

def test_explicit_override_keeps_provenance():
    row = summarize(grouped(5), independent_trials=7, sharpe_variance=.02)[0]
    assert row["independent_trials"] == 7
    assert row["count_source"] == "supplied"
    assert row["count_method"] == "caller"
    assert row["variance_source"] == "supplied"
    assert row["selection_reason"] == "heldout_groups_supported"

def test_auto_is_invariant_to_positive_units():
    values = grouped(4)
    left = summarize(values)
    right = summarize(values*np.arange(1,10))
    assert [r["dsr"] for r in left] == pytest.approx([r["dsr"] for r in right])

def test_unbalanced_duplicates_are_counted_once():
    rng = np.random.default_rng(np.random.SeedSequence([20261103,1,0]))
    factors = rng.normal(size=(3000,3))
    values = factors[:,[0,0,0,0,0,0,1,1,2]]
    result = estimate_independent_trials(values)
    assert result["candidate_groups"] == 2
    assert result["unique_trials"] == 3
    assert result["independent_trials"] == 3
    assert result["count_source"] == "deduplicated_fallback"
    assert result["selection_reason"] == "exact_duplicates_removed"
    assert result["structure_reason"] == "within_group_redundancy_not_confirmed"
