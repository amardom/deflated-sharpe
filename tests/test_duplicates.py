import numpy as np
import pytest

from deflated_sharpe import estimate_independent_trials, summarize
from deflated_sharpe.clustering import deduplicate_trials

def test_deduplication_preserves_first_occurrence_and_mapping():
    values = np.array([[2.,1.,2.,3.,1.], [4.,5.,4.,6.,5.]])
    unique, labels = deduplicate_trials(values)
    assert np.array_equal(unique, values[:,[0,1,3]])
    assert labels.tolist() == [0,1,0,2,1]
    assert np.array_equal(unique[:,labels], values)

@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("indices", [[0,0,0,0,0,0,1,1,2], [0,1,2,0,0,2,0], [2,2,0,1,2]])
def test_duplicate_multiplicity_does_not_change_dsr_or_variance(seed, indices):
    values = np.random.default_rng(seed).normal(size=(500,3))
    original = summarize(values)
    repeated = summarize(values[:,indices])
    assert repeated[0]["independent_trials"] == 3
    assert repeated[0]["unique_trials"] == 3
    assert repeated[0]["count_source"] == "deduplicated_fallback"
    assert repeated[0]["variance_source"] == "unique_trial_sharpes"
    assert repeated[0]["sharpe_variance"] == pytest.approx(original[0]["sharpe_variance"])
    for row, i in zip(repeated, indices):
        assert row["dsr"] == pytest.approx(original[i]["dsr"])
        assert row["selection_threshold"] == pytest.approx(original[i]["selection_threshold"])

@pytest.mark.parametrize("seed", range(5))
def test_duplicates_do_not_distort_supported_noisy_groups(seed):
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(3000,3))
    values = np.repeat(factors,3,axis=1)+.05*rng.normal(size=(3000,9))
    indices = [0,0,0,0,1,2,3,4,5,6,7,8]
    original = summarize(values)
    repeated = summarize(values[:,indices])
    assert repeated[0]["independent_trials"] == 3
    assert repeated[0]["unique_trials"] == 9
    assert repeated[0]["selection_reason"] == "heldout_groups_supported"
    assert repeated[0]["variance_source"] == "cluster_representatives"
    assert repeated[0]["sharpe_variance"] == pytest.approx(original[0]["sharpe_variance"])
    for row,i in zip(repeated,indices):
        assert row["dsr"] == pytest.approx(original[i]["dsr"])
        assert row["cluster"] == repeated[indices.index(i)]["cluster"]

def test_short_history_still_removes_exact_duplicates():
    values = np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])
    repeated = summarize(values[:,[0,0,0,1]])
    assert repeated[0]["independent_trials"] == 2
    assert repeated[0]["structure_reason"] == "insufficient_observations"
    assert repeated[0]["variance_source"] == "unique_trial_sharpes"

def test_negative_copies_and_shifts_are_not_duplicates():
    x = np.arange(100,dtype=float)
    unique, labels = deduplicate_trials(np.column_stack([x,-x,x+1]))
    assert unique.shape[1] == 3
    assert labels.tolist() == [0,1,2]
    result = estimate_independent_trials(np.column_stack([x,x,-x,-x,-x]))
    assert result["unique_trials"] == 2
    assert result["independent_trials"] == 2

def test_near_duplicates_are_not_exact_duplicates():
    x = np.random.default_rng(9).normal(size=100)
    shifted = x.copy()
    shifted[0] += 1e-12
    unique, _ = deduplicate_trials(np.column_stack([x,shifted]))
    assert unique.shape[1] == 2

def test_unique_count_does_not_assert_independence():
    rng = np.random.default_rng(3)
    common = rng.normal(size=(3000,1))
    factors = common + .2*rng.normal(size=(3000,3))
    rows = summarize(factors[:,[0,0,0,1,2]])
    assert rows[0]["unique_trials"] == 3
    assert rows[0]["independent_trials"] == 3
    assert rows[0]["count_source"] == "deduplicated_fallback"

def test_count_and_variance_override_remain_explicit():
    values = np.random.default_rng(3).normal(size=(100,3))[:,[0,0,0,1,2]]
    row = summarize(values, independent_trials=7, sharpe_variance=.03)[0]
    assert row["count_source"] == "supplied"
    assert row["independent_trials"] == 7
    assert row["variance_source"] == "supplied"
    assert row["unique_trials"] == 3

@pytest.mark.parametrize("seed", range(10))
def test_scaled_copies_keep_the_same_count_variance_and_dsr(seed):
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(500,3))
    indices = [0,0,0,0,0,0,1,1,2]
    values = factors[:,indices]
    original = summarize(values)
    scaled = summarize(values*np.array([1.1,2.3,.7,4.1,.3,1.7,3.3,1.9,.9]))
    assert scaled[0]["unique_trials"] == 3
    assert scaled[0]["independent_trials"] == 3
    assert scaled[0]["sharpe_variance"] == pytest.approx(original[0]["sharpe_variance"])
    assert [row["dsr"] for row in scaled] == pytest.approx([row["dsr"] for row in original])

def test_scaled_identical_trials_do_not_need_a_long_history():
    x = np.array([-.3,.2,.8,1.3])
    values = x[:,None]*np.array([1.1,2.3,.7,4.1])
    rows = summarize(values)
    assert rows[0]["unique_trials"] == 1
    assert rows[0]["independent_trials"] == 1
    assert rows[0]["selection_reason"] == "identical_trials"
    assert all(row["cluster"] == 0 for row in rows)
    assert rows[0]["sharpe_variance"] == 0
