import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("calibrate", ROOT/"research/calibrate.py")
calibrate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = calibrate
SPEC.loader.exec_module(calibrate)

@pytest.mark.parametrize("scenario", calibrate.SCENARIOS)
def test_null_generators_are_reproducible_and_uncentered(scenario):
    left = calibrate.simulate_returns(scenario, 100, 71)
    right = calibrate.simulate_returns(scenario, 100, 71)
    assert np.array_equal(left, right)
    assert left.shape == (100, 6 if scenario=="opposing_pairs" else 9)
    assert np.isfinite(left).all()
    assert np.any(np.abs(left.mean(axis=0)) > 1e-5)

def test_unknown_generator():
    with pytest.raises(ValueError, match="unknown scenario"):
        calibrate.simulate_returns("unknown",100,71)

def test_generator_dependence_matches_design():
    duplicate = calibrate.simulate_returns("duplicates", 100, 71)
    assert np.array_equal(duplicate[:,0], duplicate[:,5])
    assert np.array_equal(duplicate[:,6], duplicate[:,7])
    opposing = calibrate.simulate_returns("opposing_pairs", 100, 71)
    assert np.array_equal(opposing[:,:3], -opposing[:,3:])
    common = calibrate.simulate_returns("common_factor", 30000, 71)
    assert np.corrcoef(common, rowvar=False)[0,1] == pytest.approx(.8, abs=.02)
    student = calibrate.simulate_returns("student_t5", 30000, 71)
    assert student.var() == pytest.approx(1, abs=.04)
    assert np.mean(student**4)/np.mean(student**2)**2 > 4
    ar1 = calibrate.simulate_returns("ar1",30000,71)
    assert np.corrcoef(ar1[:-1,0],ar1[1:,0])[0,1] == pytest.approx(.4, abs=.02)
    assert ar1.var() == pytest.approx(1/(1-.4**2), abs=.04)

def sample_record(dsr=.5, maximum=.1, threshold=.08, error=None):
    return {"scenario":"independent", "method":"auto", "dsr":dsr,
            "maximum_sharpe":maximum, "threshold":threshold,
            "independent_trials":9, "supplied_trials":9, "count_source":"nominal_fallback",
            "selection_reason":"within_group_redundancy_not_confirmed", "error":error}

def test_rate_intervals_and_paired_errors():
    records = [sample_record(dsr=.96), sample_record(dsr=.95), sample_record(dsr=.2, threshold=.12)]
    row = calibrate.summarize_calibration(records)
    assert row["false_positives"] == 1
    assert row["false_positive_rate"] == pytest.approx(1/3)
    expected = binomtest(1,3).proportion_ci(method="exact")
    assert row["rate_ci_low"] == expected.low
    assert row["rate_ci_high"] == expected.high
    assert row["mean_threshold_error"] == pytest.approx(-.02/3)
    assert row["threshold_error_ci_low"] < row["mean_threshold_error"] < row["threshold_error_ci_high"]

def test_zero_events_still_have_uncertainty():
    row = calibrate.summarize_calibration([sample_record() for _ in range(100)])
    assert row["rate_ci_low"] == 0
    assert row["rate_ci_high"] == pytest.approx(1-.025**.01)
    assert row["rate_comparison"] == "below_5_percent"

def test_failures_are_not_counted_as_negative_results():
    records = [sample_record(dsr=.99), sample_record(error="domain failure")]
    row = calibrate.summarize_calibration(records)
    assert row["evaluated"] == 1
    assert row["failures"] == 1
    assert row["false_positive_rate"] == 1
    assert row["failure_rate_envelope_low"] == .5
    assert row["failure_rate_envelope_high"] == 1
    assert row["rate_comparison"] == "incomplete"
    assert row["threshold_error_ci_low"] is None
    assert row["failure_reasons"] == {"domain failure":1}
    missing = calibrate.summarize_calibration([sample_record(error="domain failure")])
    assert missing["false_positive_rate"] is None
    assert missing["failure_rate_envelope_low"] == 0
    assert missing["failure_rate_envelope_high"] == 1

def test_largest_sharpe_is_selected_and_all_methods_share_data():
    task = ("independent",0,80,71)
    records = calibrate.one_replication(task)
    values = calibrate.simulate_returns("independent",80,np.random.SeedSequence([71,0,0]))
    sharpes = values.mean(axis=0)/values.std(axis=0,ddof=1)
    assert all(row["selected_trial"] == int(sharpes.argmax()) for row in records)
    assert all(row["maximum_sharpe"] == sharpes.max() for row in records)
    reference = records[-1]
    assert reference["method"] == "known_iid_reference"
    assert reference["variance"] == pytest.approx(79/(77*80))

def test_method_failure_is_retained(monkeypatch):
    original = calibrate.summarize
    def fail_average(values, trial_method):
        if trial_method=="average_correlation":
            raise ValueError("invalid effective count")
        return original(values,trial_method=trial_method)
    monkeypatch.setattr(calibrate,"summarize",fail_average)
    records = calibrate.one_replication(("independent",0,80,71))
    assert records[1]["error"] == "invalid effective count"
    assert records[0]["error"] is None

@pytest.mark.parametrize("kwargs", [{"repetitions":1},{"observations":39},{"seed":-1},{"workers":0},
                                     {"scenarios":[]},{"scenarios":["unknown"]},
                                     {"scenarios":["independent","independent"]}])
def test_invalid_protocol(kwargs):
    with pytest.raises(ValueError):
        calibrate.run_calibration(**kwargs)

def test_seed_prefix_and_worker_invariance():
    small = calibrate.run_calibration(2,80,71,1,scenarios=("independent",))
    parallel = calibrate.run_calibration(2,80,71,2,scenarios=("independent",))
    assert small == parallel
    extended = calibrate.run_calibration(3,80,71,1,scenarios=("independent",))
    prefix = [row for row in extended["replications"] if row["replicate"]<2]
    assert small["replications"] == prefix
    assert len(small["source_sha256"]["research/calibrate.py"]) == 64

def test_cli_writes_auditable_output(tmp_path):
    output = tmp_path/"results.json"
    result = subprocess.run([sys.executable,str(ROOT/"research/calibrate.py"),"--repetitions","2",
                             "--observations","40","--workers","1","--output",str(output)],
                            capture_output=True,text=True,check=True)
    data = json.loads(output.read_text())
    assert len(data["summaries"]) == 23
    assert len(data["replications"]) == 46
    assert data["protocol"]["repetitions"] == 2
    assert "exact-95%-CI" in result.stdout
    assert "not a 95% quantile" in result.stdout
    assert "Completed 14/14" in result.stderr
