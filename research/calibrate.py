import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.stats import binomtest, t
import sklearn
from threadpoolctl import threadpool_limits

from deflated_sharpe import compute_expected_maximum_sharpe, compute_probabilistic_sharpe_ratio, summarize

SCENARIOS = ("independent", "duplicates", "noisy_blocks", "common_factor",
             "opposing_pairs", "student_t5", "ar1")
METHODS = ("auto", "average_correlation", "correlation_clusters")

PRESET_SEED = 20261103
DSR_CUTOFF = .95

def simulate_returns(scenario, observations, seed):
    rng = np.random.default_rng(seed)
    if scenario == "independent":
        return rng.normal(size=(observations, 9))
    if scenario == "student_t5":
        return rng.standard_t(5, size=(observations, 9)) / np.sqrt(5 / 3)
    if scenario == "ar1":
        values = rng.normal(size=(observations, 9))
        values[0] /= np.sqrt(1 - .4**2)
        for i in range(1, observations):
            values[i] += .4 * values[i-1]
        return values
    factors = rng.normal(size=(observations, 3))
    if scenario == "duplicates":
        return factors[:, [0,0,0,0,0,0,1,1,2]]
    if scenario == "noisy_blocks":
        return np.repeat(factors, 3, axis=1) + .1*rng.normal(size=(observations,9))
    if scenario == "common_factor":
        return np.sqrt(.8)*factors[:, :1] + np.sqrt(.2)*rng.normal(size=(observations,9))
    if scenario == "opposing_pairs":
        return np.column_stack([factors, -factors])
    raise ValueError(f"unknown scenario: {scenario}")

def one_replication(task):
    scenario, replicate, observations, seed = task
    sequence = np.random.SeedSequence([seed, SCENARIOS.index(scenario), replicate])
    with threadpool_limits(limits=1):
        values = simulate_returns(scenario, observations, sequence)
        sharpes = values.mean(axis=0)/values.std(axis=0, ddof=1)
        selected = int(np.argmax(sharpes))
        records = []
        selected_row = None
        for method in METHODS:
            record = {"scenario": scenario, "replicate": replicate, "method": method,
                      "selected_trial": selected, "maximum_sharpe": float(sharpes[selected])}
            try:
                row = summarize(values, trial_method=method)[selected]
                selected_row = row
                record.update(dsr=row["dsr"], psr=row["psr"],
                              threshold=row["selection_threshold"],
                              independent_trials=row["independent_trials"],
                              variance=row["sharpe_variance"], variance_source=row["variance_source"],
                              method_reference=row["method_reference"],
                              count_source=row["count_source"], selection_reason=row["selection_reason"],
                              supplied_trials=row["supplied_trials"], unique_trials=row["unique_trials"],
                              structure_reason=row["structure_reason"],
                              error=None)
            except ValueError as error:
                record.update(error=str(error))
            records.append(record)
        if scenario in ("independent", "duplicates"):
            count = 9 if scenario == "independent" else 3
            variance = (observations-1)/(observations-3)/observations
            threshold = compute_expected_maximum_sharpe(variance, count)
            dsr = compute_probabilistic_sharpe_ratio(float(sharpes[selected]), observations,
                                            selected_row["skewness"], selected_row["pearson_kurtosis"], threshold)
            records.append({"scenario": scenario, "replicate": replicate, "method": "known_iid_reference",
                            "selected_trial": selected, "maximum_sharpe": float(sharpes[selected]),
                            "dsr": dsr, "psr": selected_row["psr"], "threshold": threshold,
                            "independent_trials": count, "variance": variance,
                            "variance_source": "known_zero_normal_sharpe_sampling_variance",
                            "count_source": "known_simulation_design", "selection_reason": "reference_only",
                            "method_reference": "known_gaussian_zero_mean_design",
                            "supplied_trials": values.shape[1], "unique_trials": count,
                            "structure_reason": "reference_only",
                            "error": None})
        return records

def summarize_calibration(records):
    successful = [row for row in records if row["error"] is None]
    failures = len(records)-len(successful)
    result = {"scenario": records[0]["scenario"], "method": records[0]["method"],
              "repetitions": len(records), "evaluated": len(successful), "failures": failures,
              "failure_reasons": dict(Counter(row["error"] for row in records if row["error"] is not None))}
    if not successful:
        result.update(false_positives=0, false_positive_rate=None, rate_ci_low=None, rate_ci_high=None,
                      rate_comparison="incomplete", mean_maximum=None, mean_threshold=None,
                      mean_threshold_error=None, threshold_error_ci_low=None, threshold_error_ci_high=None,
                      mean_count=None, reduction_fraction=None, selection_reasons={})
    else:
        positives = sum(row["dsr"] > DSR_CUTOFF for row in successful)
        interval = binomtest(positives, len(successful)).proportion_ci(method="exact")
        errors = np.array([row["threshold"]-row["maximum_sharpe"] for row in successful])
        mean_error = float(errors.mean())
        margin = float(t.ppf(.975, len(errors)-1)*errors.std(ddof=1)/np.sqrt(len(errors))) if len(errors)>1 else None
        comparison = "above_5_percent" if interval.low > .05 else "below_5_percent" if interval.high < .05 else "overlaps_5_percent"
        result.update(false_positives=positives, false_positive_rate=positives/len(successful),
                      rate_ci_low=float(interval.low), rate_ci_high=float(interval.high),
                      rate_comparison="incomplete" if failures else comparison,
                      mean_maximum=float(np.mean([row["maximum_sharpe"] for row in successful])),
                      mean_threshold=float(np.mean([row["threshold"] for row in successful])),
                      mean_threshold_error=mean_error,
                      threshold_error_ci_low=mean_error-margin if margin is not None else None,
                      threshold_error_ci_high=mean_error+margin if margin is not None else None,
                      mean_count=float(np.mean([row["independent_trials"] for row in successful])),
                      reduction_fraction=sum(row["independent_trials"] < row["supplied_trials"]
                                             for row in successful)/len(successful),
                      selection_reasons=dict(Counter(row["selection_reason"] for row in successful)))
    result["failure_rate_envelope_low"] = result["false_positives"]/len(records)
    result["failure_rate_envelope_high"] = (result["false_positives"]+failures)/len(records)
    return result

def run_calibration(repetitions=1000, observations=3000, seed=PRESET_SEED, workers=1,
                    scenarios=SCENARIOS, progress=False):
    if repetitions < 2 or observations < 40 or seed < 0 or workers < 1:
        raise ValueError("require repetitions >= 2, observations >= 40, seed >= 0 and workers >= 1")
    if not scenarios or any(name not in SCENARIOS for name in scenarios) or len(set(scenarios))!=len(scenarios):
        raise ValueError("scenarios must be a nonempty unique subset of SCENARIOS")
    tasks = [(name, replicate, observations, seed) for name in scenarios for replicate in range(repetitions)]
    records = []
    if workers == 1:
        batches = map(one_replication, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        batches = executor.map(one_replication, tasks, chunksize=5)
    try:
        for i, batch in enumerate(batches, 1):
            records.extend(batch)
            if progress and (i%100==0 or i==len(tasks)):
                print(f"Completed {i}/{len(tasks)} null datasets", file=sys.stderr, flush=True)
    finally:
        if executor is not None:
            executor.shutdown()
    summaries = []
    for name in scenarios:
        methods = METHODS + (("known_iid_reference",) if name in ("independent", "duplicates") else ())
        for method in methods:
            summaries.append(summarize_calibration([row for row in records if row["scenario"]==name and row["method"]==method]))
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root/"pyproject.toml", *sorted((root/"src/deflated_sharpe").glob("*.py"))]
    return {"protocol": {"repetitions": repetitions, "observations": observations, "seed": seed,
                         "scenarios": list(scenarios), "dsr_cutoff": DSR_CUTOFF,
                         "selection": "largest observed per-period sample Sharpe",
                         "intervals": "pointwise 95%; exact binomial rates; paired Student-t mean errors",
                         "all_scenarios_have_zero_population_mean": True},
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
            "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
            "summaries": summaries, "replications": records}

def main():
    parser = argparse.ArgumentParser(description="Null calibration of the complete DSR workflow")
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--observations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=PRESET_SEED)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]/"outputs/calibration.json")
    args = parser.parse_args()
    try:
        result = run_calibration(args.repetitions, args.observations, args.seed, args.workers, progress=True)
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print("scenario        method                 evaluated  failures  DSR>0.95    exact-95%-CI       mean-error")
    for row in result["summaries"]:
        rate = f"{row['false_positive_rate']:.4f}" if row["evaluated"] else "NA"
        interval = f"[{row['rate_ci_low']:.4f}, {row['rate_ci_high']:.4f}]" if row["evaluated"] else "NA"
        error = f"{row['mean_threshold_error']:+.6f}" if row["evaluated"] else "NA"
        print(f"{row['scenario']:15s} {row['method']:22s} {row['evaluated']:9d} {row['failures']:9d} {rate:>9s} {interval:>19s} {error:>12s}")
    print(f"Full protocol, per-replication results, intervals and source hashes: {args.output}")
    print("Rates are conditional on successful evaluations; inspect failures and their envelopes.")
    print("Intervals are pointwise Monte Carlo uncertainty, not a simultaneous calibration certificate.")
    print("The expected maximum is not a 95% quantile. DSR > 0.95 need not have a 5% false-positive rate.")

if __name__ == "__main__":
    main()
