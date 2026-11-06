import numpy as np
import pytest

from deflated_sharpe import estimate_independent_trials, summarize
from deflated_sharpe.clustering import cluster_sharpes, cluster_trials

def block_returns(seed=1, noise=0.01):
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(1000, 3)) + [0, .1, .2]
    return np.repeat(factors, 3, axis=1) + noise * rng.normal(size=(1000, 9))

def test_unknown_method():
    with pytest.raises(ValueError, match="method"):
        estimate_independent_trials(block_returns(), method="unknown")

@pytest.mark.parametrize("seed", range(10))
def test_recover_well_separated_blocks(seed):
    values = block_returns(seed)
    result = estimate_independent_trials(values, method="correlation_clusters")
    assert result["independent_trials"] == 3
    labels = np.asarray(result["cluster_labels"])
    assert result["cluster_quality"] > 0
    for indices in np.split(np.arange(9), 3):
        assert len(set(labels[indices])) == 1
    assert len(set(labels[[0,3,6]])) == 3

def test_cluster_variance_is_not_raw_variance():
    rng = np.random.default_rng(98)
    factors = rng.normal(size=(800, 3)) + [0,.1,.3]
    values = factors[:, [0,0,0,0,0,0,1,1,2,2]]
    rows = summarize(values, trial_method="correlation_clusters")
    reference = factors.mean(axis=0) / factors.std(axis=0, ddof=1)
    assert rows[0]["independent_trials"] == 3
    assert rows[0]["variance_source"] == "cluster_representatives"
    assert rows[0]["sharpe_variance"] == pytest.approx(np.var(reference, ddof=1))
    assert rows[0]["sharpe_variance"] != pytest.approx(rows[0]["raw_sharpe_variance"])
    assert sorted(set(row["representative_sharpe"] for row in rows)) == pytest.approx(sorted(reference))

def test_all_identical_and_two_trials():
    x = np.arange(10, dtype=float)
    labels, quality = cluster_trials(np.column_stack([x,x,x]))
    assert labels.tolist() == [0,0,0]
    assert quality is None
    rows = summarize(np.column_stack([x,x]), trial_method="correlation_clusters")
    assert rows[0]["independent_trials"] == 1
    assert rows[0]["sharpe_variance"] == 0
    assert rows[0]["dsr"] == rows[0]["psr"]
    labels, quality = cluster_trials(np.column_stack([x,x[::-1]]))
    assert labels.tolist() == [0,1]
    assert quality is None

def test_variance_override():
    rows = summarize(block_returns(), independent_trials=5,
                     trial_method="correlation_clusters", sharpe_variance=.02)
    assert rows[0]["independent_trials"] == 5
    assert rows[0]["count_source"] == "supplied"
    assert rows[0]["sharpe_variance"] == .02
    assert rows[0]["variance_source"] == "supplied"
    with pytest.raises(ValueError):
        summarize(block_returns(), sharpe_variance=-1)

def test_common_units_do_not_change_reference_cluster_results():
    values = block_returns()
    left = summarize(values, trial_method="correlation_clusters")
    right = summarize(values * 7, trial_method="correlation_clusters")
    assert [r["dsr"] for r in left] == pytest.approx([r["dsr"] for r in right])

def test_minimum_variance_representative_matches_solve():
    values = block_returns()[:,:3]
    standardized = values / values.std(axis=0, ddof=1)
    weights = np.linalg.solve(np.cov(standardized, rowvar=False), np.ones(3))
    representative = standardized @ (weights / weights.sum())
    expected = representative.mean()/representative.std(ddof=1)
    assert cluster_sharpes(values, np.zeros(3, dtype=int)) == pytest.approx([expected])

def test_opposing_trials_cannot_form_representative():
    x = np.arange(10, dtype=float)
    with pytest.raises(ValueError, match="stable"):
        cluster_sharpes(np.column_stack([x,-x]), np.zeros(2, dtype=int))

def test_zero_variance_representative_is_rejected(monkeypatch):
    x = np.arange(10, dtype=float)
    monkeypatch.setattr(np.linalg, "pinv", lambda *args, **kwargs: np.eye(2))
    with pytest.raises(ValueError, match="zero variance"):
        cluster_sharpes(np.column_stack([x,-x]), np.zeros(2, dtype=int))

def test_collapsed_kmeans_fit_is_skipped(monkeypatch):
    from deflated_sharpe import clustering
    original = clustering.KMeans.fit_predict
    calls = []
    def collapse_first(self, values):
        calls.append(1)
        if len(calls) == 1:
            return np.zeros(len(values), dtype=int)
        return original(self, values)
    monkeypatch.setattr(clustering.KMeans, "fit_predict", collapse_first)
    labels, quality = cluster_trials(block_returns())
    assert len(set(labels)) == 3
    assert quality > 0

def test_independent_columns_are_not_proof_of_detected_clusters():
    values = np.random.default_rng(31).normal(size=(1000, 8))
    estimate = estimate_independent_trials(values, method="correlation_clusters")
    assert 2 <= estimate["independent_trials"] <= 7
    average = estimate_independent_trials(values, method="average_correlation")
    assert average["independent_trials"] == pytest.approx(8, abs=.1)

def test_average_correlation_does_not_count_duplicate_groups():
    centered = np.random.default_rng(99).normal(size=(300,3))
    centered -= centered.mean(axis=0)
    orthogonal, _ = np.linalg.qr(centered)
    values = np.repeat(orthogonal, 3, axis=1)
    average = estimate_independent_trials(values, method="average_correlation")
    clustered = estimate_independent_trials(values, method="correlation_clusters")
    assert average["independent_trials"] == pytest.approx(7)
    assert clustered["independent_trials"] == 3

def test_cross_trial_spread_is_not_marginal_sampling_variance():
    rng = np.random.default_rng(5)
    marginal_variance = .002
    rho = .8
    samples = np.sqrt(marginal_variance) * (
        np.sqrt(rho)*rng.normal(size=(10000,1))
        + np.sqrt(1-rho)*rng.normal(size=(10000,20)))
    cross_trial_variance = np.var(samples, axis=1, ddof=1).mean()
    assert cross_trial_variance == pytest.approx(marginal_variance*(1-rho), rel=.02)
    assert np.var(samples[:,0], ddof=1) == pytest.approx(marginal_variance, rel=.04)

@pytest.mark.parametrize("seed", range(10))
def test_identical_columns_have_exactly_one_effective_trial(seed):
    x = np.random.default_rng(seed).normal(size=100)
    rows = summarize(np.repeat(x[:,None], 5, axis=1))
    assert rows[0]["independent_trials"] == 1
    assert rows[0]["selection_threshold"] == 0
