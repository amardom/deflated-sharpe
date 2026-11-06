import numpy as np
import pytest

from deflated_sharpe import summarize
from deflated_sharpe import onc

def test_appendix_a4_matches_constant_correlation_linear_solve():
    volatility = np.array([.01,.03,.05])
    correlation = np.array([[1,.85,.7],[.85,1,.8],[.7,.8,1]])
    covariance = correlation*np.outer(volatility,volatility)
    rho = (correlation.sum()-3)/6
    approximation = np.full((3,3),rho)
    np.fill_diagonal(approximation,1)
    approximation *= np.outer(volatility,volatility)
    expected = np.linalg.solve(approximation,np.ones(3))
    expected /= expected.sum()
    actual = onc.compute_minimum_variance_weights(covariance)
    assert actual == pytest.approx(expected,rel=1e-12,abs=1e-12)
    assert actual.sum() == pytest.approx(1)
    assert approximation@actual == pytest.approx(np.full(3,(approximation@actual)[0]))

@pytest.mark.parametrize("size", [2,3,6])
@pytest.mark.parametrize("rho", [-.1,0,.5,.95])
def test_appendix_weights_across_valid_covariances(size,rho):
    volatility = np.linspace(.1,1,size)
    correlation = np.full((size,size),rho)
    np.fill_diagonal(correlation,1)
    covariance = correlation*np.outer(volatility,volatility)
    expected = np.linalg.solve(covariance,np.ones(size))
    expected /= expected.sum()
    assert onc.compute_minimum_variance_weights(covariance) == pytest.approx(expected,rel=1e-11,abs=1e-12)

def test_weight_boundaries_and_invalid_covariances():
    assert onc.compute_minimum_variance_weights(np.array([[4.]])) == pytest.approx([1])
    assert onc.compute_minimum_variance_weights(np.diag([1.,4.])) == pytest.approx([.8,.2])
    assert onc.compute_minimum_variance_weights(np.ones((3,3))) == pytest.approx([1/3]*3)
    for covariance in [np.diag([0.,1.]),np.diag([np.inf,1.])]:
        with pytest.raises(ValueError,match="positive"):
            onc.compute_minimum_variance_weights(covariance)
    with pytest.raises(ValueError,match="singular"):
        onc.compute_minimum_variance_weights(np.array([[1.,-1.],[-1.,1.]]))
    with pytest.raises(ValueError,match="normalized"):
        onc.compute_minimum_variance_weights(np.array([[1.,2.],[2.,1.]]))

def test_zero_variance_representative_is_rejected():
    x = np.arange(10,dtype=float)
    with pytest.raises(ValueError,match="zero"):
        onc.compute_cluster_sharpes(np.column_stack([x,2*x]),np.zeros(2,dtype=int))

def test_representative_uses_original_relative_volatility():
    rng = np.random.default_rng(11)
    factor = rng.normal(size=(500,1))
    values = factor + .3*rng.normal(size=(500,3)) + [0,.1,.2]
    values *= [.5,2,5]
    covariance = np.cov(values,rowvar=False,ddof=1)
    weights = onc.compute_minimum_variance_weights(covariance)
    representative = values@weights
    expected = representative.mean()/representative.std(ddof=1)
    assert onc.compute_cluster_sharpes(values,np.zeros(3,dtype=int)) == pytest.approx([expected])
    standardized = values/values.std(axis=0,ddof=1)
    different = onc.compute_cluster_sharpes(standardized,np.zeros(3,dtype=int))[0]
    assert abs(different-expected) > .01

def test_public_reference_summary_matches_original_unit_representatives():
    rng = np.random.default_rng(3)
    factors = rng.normal(size=(700,3)) + [.01,.05,.1]
    values = np.repeat(factors,3,axis=1)+.1*rng.normal(size=(700,9))
    values *= np.arange(1,10)
    rows = summarize(values,trial_method="correlation_clusters")
    labels = np.array([row["cluster"] for row in rows])
    expected = onc.compute_cluster_sharpes(values,labels)
    assert rows[0]["sharpe_variance"] == pytest.approx(np.var(expected,ddof=1))
    assert rows[0]["method_reference"] == "lopez_de_prado_lewis_2018_8_A1_A2_A4"
    for row in rows:
        assert row["representative_sharpe"] == pytest.approx(expected[row["cluster"]])

def test_reference_is_invariant_to_common_extreme_unit_changes():
    values = np.random.default_rng(31).normal(size=(100,4))
    original = summarize(values,trial_method="correlation_clusters")
    for scale in [1e-200,1e200]:
        changed = summarize(values*scale,trial_method="correlation_clusters")
        assert [row["dsr"] for row in changed] == pytest.approx([row["dsr"] for row in original])

def test_onc_boundaries():
    x = np.arange(20,dtype=float)
    labels,quality = onc.cluster_trials(np.column_stack([x,x,x]))
    assert labels.tolist() == [0,0,0]
    assert quality is None
    labels,quality = onc.cluster_trials(np.column_stack([x,-x]))
    assert labels.tolist() == [0,1]
    assert quality is None

def test_collapsed_initialization_is_skipped(monkeypatch):
    original = onc.KMeans.fit_predict
    calls = []
    def collapse_first(self,values):
        calls.append(1)
        return np.zeros(len(values),dtype=int) if len(calls)==1 else original(self,values)
    monkeypatch.setattr(onc.KMeans,"fit_predict",collapse_first)
    labels,_ = onc.cluster_trials(np.random.default_rng(4).normal(size=(100,4)))
    assert len(set(labels))>=2

def test_no_valid_candidate_is_reported(monkeypatch):
    monkeypatch.setattr(onc.KMeans,"fit_predict",lambda self,values:np.zeros(len(values),dtype=int))
    with pytest.raises(ValueError,match="nondegenerate"):
        onc.cluster_trials(np.random.default_rng(4).normal(size=(100,4)))

@pytest.mark.parametrize("accept", [True,False])
def test_recursive_refinement_acceptance_follows_appendix_a2(monkeypatch,accept):
    rng = np.random.default_rng(18)
    factors = rng.normal(size=(500,3))
    values = factors[:,[0,0,1,1,1,2,2,2]]+.03*rng.normal(size=(500,8))
    correlation = np.corrcoef(values,rowvar=False)
    original_labels = np.repeat(np.arange(4),2)
    low = [.4,.6] if accept else [.9,.9000000000001]
    scores = np.array([1.,1.]+low*3)
    calls = []
    def controlled_base(matrix):
        calls.append(len(matrix))
        if len(matrix)==8:
            return original_labels.copy(),scores.copy()
        assert len(matrix)==6
        return np.array([0,0,0,1,1,1]),np.array([.8,.81,.82,.8,.81,.82])
    monkeypatch.setattr(onc,"cluster_base",controlled_base)
    labels,actual_scores = onc.refine_clusters(correlation)
    assert calls == [8,6]
    assert len(set(labels)) == (3 if accept else 4)
    if accept:
        from sklearn.metrics import silhouette_samples
        assert actual_scores == pytest.approx(silhouette_samples(onc.compute_distance(correlation),labels))
    else:
        assert np.array_equal(labels,original_labels)
        assert np.array_equal(actual_scores,scores)
