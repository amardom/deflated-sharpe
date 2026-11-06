import numpy as np
import pytest

from deflated_sharpe import (compute_deflated_sharpe_ratio, estimate_independent_trials,
                             compute_probabilistic_sharpe_ratio, summarize)

def returns():
    return np.random.default_rng(6).normal(size=(100,3))

def test_scalar_calculation_is_independent_of_policy(monkeypatch):
    from deflated_sharpe import core
    def fail(*args, **kwargs):
        raise AssertionError("scalar DSR must not select a trial model")
    monkeypatch.setattr(core,"assess_groups",fail)
    result = compute_deflated_sharpe_ratio(.1,100,0,3,.01,10)
    assert 0 < result < 1
    assert compute_deflated_sharpe_ratio.__module__ == "deflated_sharpe.statistics"

def test_numpy_observation_count_is_supported():
    assert compute_probabilistic_sharpe_ratio(.1,np.int64(100),0,3) == compute_probabilistic_sharpe_ratio(.1,100,0,3)

@pytest.mark.parametrize("names", ["abc",b"abc",17,True,{"a":1,"b":2,"c":3}])
def test_invalid_name_containers_are_rejected(names):
    with pytest.raises(ValueError,match="trial_names"):
        summarize(returns(),trial_names=names)

@pytest.mark.parametrize("names", [("a","b","c"), np.array(["a","b","c"]), iter(["a","b","c"])])
def test_supported_name_iterables(names):
    assert [row["trial"] for row in summarize(returns(),trial_names=names)] == ["a","b","c"]

def test_estimated_count_remains_visible_with_override():
    values=returns()
    estimated=summarize(values)[0]
    overridden=summarize(values,independent_trials=7,sharpe_variance=.02)[0]
    assert overridden["supplied_trials"] == 3
    assert overridden["unique_trials"] == 3
    assert overridden["estimated_independent_trials"] == estimated["independent_trials"]
    assert overridden["independent_trials"] == 7
    assert overridden["estimation_method"] == "auto"
    assert overridden["count_method"] == "caller"
    assert overridden["count_source"] == "supplied"
    assert overridden["variance_source"] == "supplied"

@pytest.mark.parametrize("method", ["auto","average_correlation","correlation_clusters"])
def test_estimator_exposes_count_provenance(method):
    result = estimate_independent_trials(returns(),method=method)
    assert result["supplied_trials"] == 3
    assert result["method"] == method
    assert result["count_source"] in ("estimated","nominal_fallback")
    assert "unique_trials" in result
    assert result["independent_trials"] >= 1
