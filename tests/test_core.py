import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from deflated_sharpe import (
    compute_deflated_sharpe_ratio, estimate_independent_trials,
    compute_expected_maximum_sharpe, compute_probabilistic_sharpe_ratio, summarize,
)

@pytest.fixture
def returns():
    return np.random.default_rng(71).normal(0.001, 0.01, (250, 4))

@pytest.mark.parametrize("count,skew,kurt,expected", [
    (100, -3, 10, .9004), (46, -3, 10, .9505), (88, 0, 3, .9505),
])
def test_published_examples(count, skew, kurt, expected):
    result = compute_deflated_sharpe_ratio(2.5 / math.sqrt(250), 1250, skew, kurt, .002, count)
    assert result == pytest.approx(expected, abs=.00005)

@pytest.mark.parametrize("count", [2, 3, 10, 46, 100, 1000])
@pytest.mark.parametrize("variance", [.001, .1, 1, 4])
def test_threshold_against_numerical_integration(count, variance):
    exact = quad(lambda x: x * count * norm.pdf(x) * norm.cdf(x) ** (count - 1), -12, 12)[0]
    result = compute_expected_maximum_sharpe(variance, count)
    assert abs(result - math.sqrt(variance) * exact) < .06 * math.sqrt(variance)

@pytest.mark.parametrize("sharpe", [-2, -.1, 0, .1, 2])
@pytest.mark.parametrize("observations", [2, 10, 100, 1000])
@pytest.mark.parametrize("skew,kurt", [(0, 3), (-3, 10), (1, 4)])
def test_psr_against_normal_cdf(sharpe, observations, skew, kurt):
    z = sharpe * math.sqrt(observations - 1) / math.sqrt(1-skew*sharpe+(kurt-1)*sharpe**2/4)
    result = compute_probabilistic_sharpe_ratio(sharpe, observations, skew, kurt)
    assert result == pytest.approx(norm.cdf(z), abs=1e-15)
    assert 0 <= result <= 1

@pytest.mark.parametrize("bad", [None, True, "1", float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("position", range(6))
def test_dsr_invalid_scalar_inputs(bad, position):
    args = [.1, 100, 0, 3, .01, 10]
    args[position] = bad
    with pytest.raises(ValueError):
        compute_deflated_sharpe_ratio(*args)

@pytest.mark.parametrize("count", [-1, 0, .5, 1.1, 1.9])
def test_invalid_trial_counts(count):
    with pytest.raises(ValueError):
        compute_expected_maximum_sharpe(.1, count)

def test_variance_and_single_trial_boundaries():
    assert compute_expected_maximum_sharpe(2, 1) == 0
    assert compute_expected_maximum_sharpe(0, 20) == 0
    with pytest.raises(ValueError):
        compute_expected_maximum_sharpe(-.1, 10)
    for observations in [0, 1, 2.5]:
        with pytest.raises(ValueError):
            compute_probabilistic_sharpe_ratio(.1, observations, 0, 3)
    with pytest.raises(ValueError):
        compute_probabilistic_sharpe_ratio(.1, 10, 2, 3)
    with pytest.raises(ValueError):
        compute_probabilistic_sharpe_ratio(1, 10, 2, 5)
    with pytest.raises(ValueError):
        compute_probabilistic_sharpe_ratio(1e308, 10, 0, 3)
    assert compute_probabilistic_sharpe_ratio(0, 10, 0, 3) == .5
    assert compute_deflated_sharpe_ratio(.1, 100, 0, 3, .1, 1) == compute_probabilistic_sharpe_ratio(.1, 100, 0, 3)

@pytest.mark.parametrize("bad", [[], [1,2,3,4], [[1],[2],[3],[4]],
    [[1,2],[2,3],[3,4]], [[1,2],[2]], [["x",2]]*4,
    [[None,2],[2,3],[3,4],[4,5]], [[1,2],[2,3],[3,4],[float("inf"),5]],
    [[1,1],[1,2],[1,3],[1,4]], np.ones((4,2,2))])
def test_invalid_returns(bad):
    with pytest.raises(ValueError):
        summarize(bad)

@pytest.mark.parametrize("names", [["a"], ["a"]*4, ["a","b","c",1], ["a","b","c",""]])
def test_invalid_names(returns, names):
    with pytest.raises(ValueError):
        summarize(returns, names)

def test_correlation_limits():
    x = np.array([-2., -1., 1., 2.])
    assert estimate_independent_trials(np.column_stack([x, x]), method="average_correlation")["independent_trials"] == pytest.approx(1)
    assert estimate_independent_trials(np.column_stack([x, -x]), method="average_correlation")["independent_trials"] == pytest.approx(3)
    orthogonal = np.array([[1,1],[-1,1],[1,-1],[-1,-1]], dtype=float)
    assert estimate_independent_trials(orthogonal, method="average_correlation")["independent_trials"] == pytest.approx(2)
    with pytest.raises(ValueError, match="at least 2"):
        summarize(np.column_stack([x, x + np.array([.01, 0, 0, 0])]), trial_method="average_correlation")

def test_explicit_count_and_identical_trials():
    x = np.array([-2., -1., 1., 2.])
    rows = summarize(np.column_stack([x,x]), independent_trials=1)
    assert rows[0]["dsr"] == .5
    assert rows[0]["count_source"] == "supplied"
    assert rows[0]["count_method"] == "caller"
    assert rows[0]["sharpe_variance"] == 0

def test_summary_values(returns):
    names = ["a", "b", "c", "d"]
    rows = summarize(returns, names, trial_method="average_correlation")
    sharpes = returns.mean(axis=0) / returns.std(axis=0, ddof=1)
    assert len(rows) == 4
    for i, row in enumerate(rows):
        centered = returns[:,i]-returns[:,i].mean()
        second = np.mean(centered**2)
        assert row["trial"] == names[i]
        assert row["observations"] == 250
        assert row["supplied_trials"] == 4
        assert row["count_source"] == "estimated"
        assert row["count_method"] == "average_correlation"
        assert row["sharpe"] == pytest.approx(sharpes[i])
        assert row["skewness"] == pytest.approx(np.mean(centered**3)/second**1.5)
        assert row["pearson_kurtosis"] == pytest.approx(np.mean(centered**4)/second**2)
        assert row["sharpe_variance"] == pytest.approx(np.var(sharpes, ddof=1))
        assert row["dsr"] == pytest.approx(compute_deflated_sharpe_ratio(
            row["sharpe"], 250, row["skewness"], row["pearson_kurtosis"],
            row["sharpe_variance"], row["independent_trials"]))
    assert summarize(returns)[0]["trial"] == "trial_1"

@pytest.mark.parametrize("seed", range(20))
def test_scaling_and_permutation(seed):
    values = np.random.default_rng(seed).normal(0.001, .01, (100, 4))
    original = summarize(values)
    scaled = summarize(values * [2,3,4,5])
    permuted = summarize(values[:,::-1])
    for i in range(4):
        assert original[i]["dsr"] == pytest.approx(scaled[i]["dsr"])
        assert original[i]["dsr"] == pytest.approx(permuted[3-i]["dsr"])

def test_trial_count_and_benchmark_monotonicity():
    thresholds = [compute_expected_maximum_sharpe(.01, n) for n in [2,10,100,1000]]
    assert thresholds == sorted(thresholds)
    scores = [compute_probabilistic_sharpe_ratio(.1, 100, 0, 3, b) for b in thresholds]
    assert scores == sorted(scores, reverse=True)

@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_extreme_return_units(returns, scale):
    original = summarize(returns)
    changed = summarize(returns * scale)
    for left, right in zip(original, changed):
        assert left["dsr"] == pytest.approx(right["dsr"])

def test_zero_column_is_rejected():
    with pytest.raises(ValueError):
        summarize([[0,1],[0,2],[0,3],[0,4]])
