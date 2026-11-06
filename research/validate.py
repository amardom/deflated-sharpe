import argparse

import numpy as np

from deflated_sharpe import estimate_independent_trials, compute_expected_maximum_sharpe, summarize

def main():
    parser = argparse.ArgumentParser(description="Seeded trial-dependence validation")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--observations", type=int, default=3000)
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("repetitions must be at least 2")
    if args.observations < 40:
        parser.error("observations must be at least 40")
    results = {name: [] for name in ["independent", "duplicates", "noisy_blocks", "common_factor", "opposing_pairs"]}
    for seed in range(args.repetitions):
        rng = np.random.default_rng(seed)
        factors = rng.normal(size=(args.observations, 3))
        independent = rng.normal(size=(args.observations, 9))
        common = rng.normal(size=(args.observations, 1))
        scenarios = {
            "independent": independent,
            "duplicates": np.repeat(factors, 3, axis=1),
            "noisy_blocks": np.repeat(factors, 3, axis=1) + .1 * independent,
            "common_factor": np.sqrt(.8)*common + np.sqrt(.2)*independent,
            "opposing_pairs": np.column_stack([factors, -factors]),
        }
        for name, values in scenarios.items():
            average = estimate_independent_trials(values, method="average_correlation")
            clustered = summarize(values, trial_method="correlation_clusters")[0]
            automatic = summarize(values)[0]
            sharpes = values.mean(axis=0) / values.std(axis=0, ddof=1)
            average_threshold = compute_expected_maximum_sharpe(clustered["raw_sharpe_variance"], average["independent_trials"])
            results[name].append([average["independent_trials"], clustered["independent_trials"],
                                  clustered["raw_sharpe_variance"], clustered["sharpe_variance"],
                                  float(sharpes.max()), average_threshold, clustered["selection_threshold"],
                                  automatic["independent_trials"],
                                  automatic["independent_trials"] < values.shape[1]])
    print(f"Seeded Gaussian scenarios; {args.observations} observations; {args.repetitions} repetitions")
    print("scenario          avg-count  cluster-count  raw-SR-var   cluster-SR-var  observed-max  avg-threshold  cluster-threshold  auto-count  reduced-fraction")
    for name, samples in results.items():
        average_count, cluster_count, raw, clustered, maximum, avg_threshold, cluster_threshold, auto_count, reduced = np.mean(samples, axis=0)
        print(f"{name:17s} {average_count:9.3f} {cluster_count:14.3f} {raw:11.6f} {clustered:14.6f}"
              f" {maximum:13.6f} {avg_threshold:14.6f} {cluster_threshold:18.6f} {auto_count:11.3f} {reduced:17.3f}")
    print("Auto reduces only after held-out correlation screening; otherwise it uses the supplied count.")
    print("Counts are heuristic estimates. Noisy blocks have residual independent noise.")
    print("Clustering forces partitions even without groups; it can undercount independent trials.")
    print("Opposing strategies share drivers but remain distinct one-sided search alternatives.")

if __name__ == "__main__":
    main()
