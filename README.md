# Deflated Sharpe

A Python library for the Deflated Sharpe Ratio (DSR), effective trial-count
estimation and a table showing the inputs, assumptions and results.

The task is: supply the excess-return histories of all attempted strategies,
estimate the search quantities, and calculate the DSR for each original strategy.

## Run the example

Python 3.12 or newer, from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
python examples/run.py
```

The included JSON has 250 synthetic observations and nine strategies. It explicitly
selects `correlation_clusters`, the reference path described below. There is no
market dataset or claim of a profitable strategy.

```sh
python examples/run.py examples/input.json --format json
python examples/run.py examples/input.json --format csv
```

The console prints shared search quantities once and then a table of original
trials. JSON and CSV return complete rows. Console floats use six decimal places;
JSON and CSV preserve the underlying floating-point values.

## Scientific references and implementation scope

1. Bailey and Lopez de Prado (2014), *The Deflated Sharpe Ratio: Correcting for
   Selection Bias, Backtest Overfitting, and Non-Normality*.
   [Paper](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf),
   [DOI](https://doi.org/10.3905/jpm.2014.40.5.094).
2. Lopez de Prado and Lewis, *Detection of False Investment Strategies Using
   Unsupervised Learning Methods*, manuscript version November 1, 2018.
   [Manuscript](https://smallake.kr/wp-content/uploads/2020/04/SSRN-id3167017.pdf),
   [SSRN record](https://ssrn.com/abstract=3167017).

| Calculation or choice | Origin |
| --- | --- |
| Scalar DSR and expected-maximum benchmark | Reference 1, equation (2) |
| Average-correlation trial-count interpolation | Reference 1, appendix A.3 |
| ONC grouping and recursive refinement | Reference 2, section 8.1 and appendices A.1-A.2 |
| Representative returns and variance across representative Sharpes | Reference 2, section 8.2 |
| Constant-correlation approximation for minimum-variance weights | Reference 2, appendix A.4 |
| Fixed random seeds and numerical degeneracy handling | Implementation choices documented below |
| Half-sample checks and correlation thresholds in `auto` | Our experimental policy, not a published estimator |

The reference path is an independent implementation of the specified mathematical
and algorithmic procedures. It is not a bit-for-bit reproduction of a historical
Python/scikit-learn runtime. When section 8.1 and appendix A.2 differ, we follow the acceptance rule in
appendix A.2.

## Public API

All five functions are importable from `deflated_sharpe`:

```python
from deflated_sharpe import (
    summarize,
    estimate_independent_trials,
    compute_expected_maximum_sharpe,
    compute_probabilistic_sharpe_ratio,
    compute_deflated_sharpe_ratio,
)
```

### `summarize`

```python
summarize(returns, trial_names=None, independent_trials=None,
          trial_method="auto", sharpe_variance=None)
```

| Parameter | Meaning |
| --- | --- |
| `returns` | Numeric matrix, observations in rows and attempted strategies in columns; at least four observations and two columns |
| `trial_names` | Optional iterable with one unique nonempty string per column |
| `independent_trials` | Optional count overriding the selector's proposal; 1 or a finite number at least 2 |
| `trial_method` | `correlation_clusters`, `average_correlation`, or experimental `auto` |
| `sharpe_variance` | Optional nonnegative finite variance overriding the selected variance |

Returns a list of dictionaries, one per original column in input order. To use
the reference method, select it explicitly:

```python
import json
from pathlib import Path
from deflated_sharpe import summarize

inputs = json.loads(Path("examples/input.json").read_text())
rows = summarize(**inputs)
print(rows[0]["method_reference"])
```

The `auto` mode is experimental and is not a synonym for the published reference
method. The example and reference instructions select `correlation_clusters`
explicitly.

All supplied series must already be aligned, use the same observation frequency,
and represent excess returns after the costs you intend to include. Unsuccessful
experiments belong in the trial set. The library cannot recover omitted research.

A count override does not change how variance is estimated; a variance override
does not change the count. The selector still runs and its proposal remains
visible. The caller must ensure consistency of supplied overrides.

Raises `ValueError` for invalid matrices, nonfinite observations, constant columns,
invalid names, unknown methods, invalid scalar inputs or degenerate representatives.
Missing observations are rejected, not dropped. A one-column matrix is rejected;
the scalar functions support an explicitly supplied single-trial calculation.

### `estimate_independent_trials`

```python
estimate_independent_trials(returns, method="auto")
```

Accepts the same return-matrix conditions and three method choices. Returns a
dictionary with the proposed `independent_trials`, `supplied_trials`,
`unique_trials`, `count_source`, `mean_correlation`, `method` and `method_reference`.
Grouping methods include labels and quality when defined. Auto additionally
includes its screening reasons and bounds. `unique_trials` is null for the two
reference methods because they do not run auto's identity-class preprocessing.

```python
import json
from pathlib import Path
from deflated_sharpe import estimate_independent_trials

inputs = json.loads(Path("examples/input.json").read_text())
estimate = estimate_independent_trials(inputs["returns"], method="correlation_clusters")
print(estimate["independent_trials"], estimate["method_reference"])
```

Raises `ValueError` for invalid matrices, unsupported methods or a failed
nondegenerate clustering search. The average-correlation estimate can fall
strictly between 1 and 2; the estimator exposes that value, while the scalar
benchmark and `summarize` reject that approximation range. No silent rounding
or substitution is performed.

### `compute_expected_maximum_sharpe`

```python
compute_expected_maximum_sharpe(sharpe_variance, independent_trials)
```

Returns a float: the approximate expected maximum under a zero-mean Sharpe null.
Variance must be finite and nonnegative; the finite count must equal 1 or be at
least 2. Fractional counts above 2 are allowed. Count 1 or variance 0 gives a zero
benchmark as an explicit boundary convention. Booleans and invalid inputs raise
`ValueError`. The result is an expectation, not a confidence quantile.

```python
from deflated_sharpe import compute_expected_maximum_sharpe

benchmark = compute_expected_maximum_sharpe(.002, 100)
print(benchmark)
```

The output is approximately 0.113172 per observation.

### `compute_probabilistic_sharpe_ratio`

```python
compute_probabilistic_sharpe_ratio(sharpe, observations, skewness, kurtosis,
                                 benchmark=0)
```

Returns a float in [0, 1]. Sharpe and benchmark are finite per-observation values;
`observations` is an integer of at least 2, including NumPy integers. Skewness and
Pearson kurtosis are finite standardized moments. Normal reference moments are
0 and 3. No clustering or trial-count adjustment is performed.

```python
from deflated_sharpe import compute_probabilistic_sharpe_ratio

psr = compute_probabilistic_sharpe_ratio(.1, 250, 0, 3, benchmark=0)
print(psr)
```

Raises `ValueError` for invalid inputs, inconsistent moments
(`kurtosis < skewness**2 + 1`, allowing 1e-12 numerical tolerance), or a nonpositive
or nonfinite variance denominator. Booleans are rejected. The result is not a
posterior probability of investment skill and does not correct serial dependence.

### `compute_deflated_sharpe_ratio`

```python
compute_deflated_sharpe_ratio(sharpe, observations, skewness, kurtosis,
                            sharpe_variance, independent_trials)
```

Combines the scalar benchmark and PSR, returning a float in [0, 1]. Parameter
conditions and `ValueError` cases follow those two functions. It performs no
variance estimation, clustering, automatic policy or annualization.

The numerical example from Reference 1 is evaluated directly:

```python
from math import sqrt
from deflated_sharpe import compute_deflated_sharpe_ratio

dsr = compute_deflated_sharpe_ratio(
    sharpe=2.5 / sqrt(250), observations=1250,
    skewness=-3, kurtosis=10,
    sharpe_variance=1 / (2 * 250), independent_trials=100,
)
print(dsr)
```

The output is approximately 0.900397, matching the article's rounded 0.9004.
These are published summary statistics, not an original return dataset.

## Summary-table fields

| Field | Meaning |
| --- | --- |
| `trial`, `observations` | Original trial identifier and number of observations |
| `sharpe`, `skewness`, `pearson_kurtosis` | Statistics of that original trial |
| `supplied_trials` | Original number of input columns |
| `unique_trials` | Auto's numerical-identity class count; null outside auto |
| `estimated_independent_trials` | Selector proposal before an override; can be nominal |
| `independent_trials` | Count actually used in the benchmark |
| `count_source`, `count_method` | Provenance of the used count; `caller` method for an override |
| `estimation_method` | Requested selector, even if the count is overridden |
| `method_reference` | `lopez_de_prado_lewis_2018_8_A1_A2_A4`, `bailey_lopez_de_prado_2014_A3`, or `experimental_auto` |
| `raw_sharpe_variance` | Sample variance across all original trial Sharpes |
| `sharpe_variance`, `variance_source` | Variance used in the benchmark and its provenance |
| `cluster`, `cluster_quality` | Group membership and quality, when defined |
| `representative_sharpe` | Sharpe of the group's aggregate return series, when used |
| `mean_correlation` | Mean pairwise return correlation across supplied columns |
| `selection_reason`, `structure_reason` | Explicit-method marker or auto's decision reasons |
| `candidate_groups`, `within_lower_bound`, `between_upper_bound` | Auto diagnostics; null for the reference method |
| `selection_threshold`, `psr`, `dsr` | Benchmark, zero-benchmark PSR and DSR |

Shared research quantities repeat in JSON/CSV rows so an exported row remains
interpretable. `cluster_representatives` has method-specific meaning: the reference
uses Appendix A.4 representatives; auto uses its previously documented experimental
unit-volatility/pseudoinverse representatives. Read `method_reference` with the
variance source. `unique_trial_sharpes` identifies auto's deduplicated fallback;
`all_trial_sharpes` and `supplied` identify the other variance sources.

## Mathematical conventions

Calculations use per-observation Sharpe ratios and matching variance units.
For optional display, annual Sharpe is per-observation Sharpe times sqrt(A), and
annual Sharpe variance is per-observation variance times A. Do not mix units.

Sample standard deviation, return covariance and cross-trial variance use ddof=1.
For the return-derived skewness and kurtosis, empirical central moments use divisor
T without finite-sample bias correction:

$$
m_k=T^{-1}\sum_{t=1}^{T}(r_t-\bar r)^k,\qquad
g_3=m_3/m_2^{3/2},\qquad g_4=m_4/m_2^2.
$$

The 2014 zero-null benchmark and scalar DSR are

$$
b=\sqrt v\left[(1-\gamma)\Phi^{-1}(1-1/N)
+\gamma\Phi^{-1}(1-1/(Ne))\right],
$$

$$
q=1-g_3s+\frac{g_4-1}{4}s^2,\qquad
DSR=\Phi\left(\frac{(s-b)\sqrt{T-1}}{\sqrt q}\right).
$$

Phi is the standard normal CDF and gamma is Euler's constant. PSR uses a specified
benchmark in place of b. Cross-trial variance v is not the sampling variance of a
single Sharpe estimate. No observed cross-trial mean Sharpe is added to the null.

## Reference clustering and variance

### ONC: section 8.1 and appendices A.1-A.2

Compute return correlation C, then use rows of

$$
D_{ij}=\sqrt{(1-C_{ij})/2}
$$

as K-means features. Euclidean silhouette distances between those rows therefore
use distance profiles, not merely a single pairwise return correlation. The
objective is mean silhouette divided by its population standard deviation.

The base search evaluates feasible k=2,...,M-1 across ten initializations. Weak
clusters have below-average cluster-specific quality. If more than two are weak,
rerun recursively on their restricted correlation matrix, retaining the others.
Recompute full-matrix silhouettes for the combined candidate. As in appendix A.2,
accept when the mean quality across all candidate groups exceeds the old mean
quality of the groups selected for redo. This is the snippet's specific comparison,
not a claim of monotonic improvement of a different global objective.

### Representatives: section 8.2 and appendix A.4

Preserve relative input volatilities. Reference inputs are rescaled only by one
common positive factor for numerical stability. Normalizing every strategy to
unit volatility would change this estimator and is not done in this path.
Individual leverage changes can change its output; a common change of units does
not change it.

For a group with m members, volatilities sigma_i and average off-diagonal
correlation rho, Appendix A.4 uses weights proportional to

$$
w_i\ \propto\ \frac{1}{\sigma_i^2}
-\frac{\rho\sum_j1/\sigma_j}{(1+(m-1)\rho)\sigma_i}.
$$

Normalize weights to sum to one. They are unrestricted and can be negative.
This is the constant-correlation approximation, not the inverse of the complete
sample covariance. Compute aggregate returns with these weights, their sample
Sharpe ratios, then the sample variance across those representative Sharpes.

All matrix columns here share a frequency. The annualization and conversion back
to the selected strategy's frequency in section 8.2 cancel for that aligned case.
Mixed-frequency histories are outside this interface. Original strategy moments
and Sharpe are still used for each original strategy's DSR row.

### Explicit implementation conventions

| Convention | Reason and scope |
| --- | --- |
| Ten initializations, seeds 0,...,9, `n_init=1` | The appendix uses repeated single initializations; fixed seeds make our run reproducible |
| Population standard deviation for silhouette quality | Implements the stated ratio consistently; not a Student-t inference procedure |
| Quality denominator floor 1e-12 | Makes degenerate equal-score cases deterministic; a numerical convention, not a significance threshold |
| Clip correlations to [-1,1]; snap within 1e-12 of +1 | Prevent square-root roundoff errors for identical histories |
| Feasible group count bounded by distinct distance profiles, rounded to 12 decimals | Avoid requesting impossible K-means partitions; skip collapsed initializations and reject if none is valid |
| One group when maximum distance is below 1e-7 after correlation snapping; two nonidentical columns remain two groups | Explicit small/degenerate boundary handling where silhouette search is undefined |
| Positive common rescaling of all returns | Prevent overflow while preserving relative reference volatilities |
| Algebraic common rescaling of inverse-volatility weights | Avoid unnecessary overflow without changing normalized Appendix A.4 weights |
| Equal weights for the equal-volatility, perfectly correlated singular case | Symmetric choice among nonunique allocations; not a unique solution asserted by the paper |
| Reject singular/nonnormalizable weights and representative volatility <=1e-12 of the largest member volatility | Avoid assigning an arbitrary Sharpe to an undefined representative |
| Sample standard deviations and cross-trial variance with ddof=1 | Explicit finite-sample convention; empirical standardized moments use divisor T |

The quality floor can affect the selected partition for degenerate or nearly
equal silhouette scores, especially singleton groups; it is not an innocuous
claim of statistical equivalence to undefined divisions in the historical snippet.
These choices make the software defined at numerical boundaries; they must not
be advertised as additional scientific findings. The ONC search does not include
M singleton groups in its ordinary k range and can undercount unstructured trials.
Singleton groups are difficult to identify. A number of correlated groups is a
proxy for effective trials, not proof of independence or identification of all
nonlinear dependence.

## Other methods

### `average_correlation`

The original appendix A.3 interpolation is

$$
\bar\rho=\frac{\sum_{i\ne j}C_{ij}}{M(M-1)},\qquad
\widehat N=\bar\rho+(1-\bar\rho)M.
$$

Use the sample variance across original trial Sharpes. Negative mean correlation
can produce a count above M; do not silently cap it or take absolute correlations.
Nine columns made from three orthogonal histories repeated three times give count
7, not 3. This is a reference heuristic, not a duplicate counter.

### Experimental `auto`

Auto is our previous policy, not part of either publication. Its behaviour is
retained independently of the reference path:

- Normalize columns individually by maximum absolute return and coalesce histories
  differing by at most four float64 epsilons at every observation. Compare to the
  first representative, without chained approximate matching. This absorbs unit
  conversion roundoff; it is not a statistical near-duplicate threshold.
- Keep the first occurrence of each identity class and propose a base partition
  with the first chronological half. This experimental base search uses five
  fixed seeds and no recursive refinement.
- Require at least 20 observations in each half and nonconstant split columns.
- On the second half, require within-group Fisher correlation lower bounds >=0.95
  and between-group intervals wholly inside [-0.20,+0.20]. Use 0.05 nominal family
  error with Bonferroni correction across distinct-history pairs.
- If supported, use groups and unit-volatility, full-covariance/pseudoinverse
  representatives. Otherwise retain the number of distinct histories and their
  Sharpe variance. Repeated histories retain their original result rows.

The 0.95/0.20 choices and split design are our engineering criteria, not parameters
prescribed by Lopez de Prado. Fisher intervals assume an asymptotic jointly normal,
independent-in-time model. No check establishing those assumptions is claimed.
Auto's deduplicated or nominal fallback does not assert independence or universal
conservative inference. The screen's second half is subsequently reused for the
final descriptive calculation; it is not a separate untouched performance test.

The reference and auto paths intentionally have different covariance and scaling
conventions. Reference grouping is not silently wrapped in auto's acceptance rules.

## Verification and calibration

```sh
python -m pip install -e '.[dev]'
python -m pytest
python research/validate.py
python research/calibrate.py
```

Tests require 100% line and branch coverage of the installed library code. Coverage
checks execution, not scientific validity. Tests cover published scalar examples,
Appendix A.4 versus independent constant-correlation linear solves, representative
returns in original relative units, recursive acceptance/rejection, block recovery,
invalid inputs, numerical boundaries, and public JSON/table/CSV usage.
Controlled candidate fixtures exercise recursive branches; they are not empirical
evidence of cluster recovery. Real synthetic block datasets provide that separate
check. Scripts and tests remain outside the runtime package interface.

### Frozen null-calibration protocol

The experiment uses seed 20261103, 1000 repetitions per scenario and 3000
observations per history. Methods use identical datasets. The original strategy
with largest observed sample Sharpe is selected, then its DSR is evaluated through
the complete requested method. No strategy or method is selected by largest DSR.
Do not demean realized samples: random sample means are the null outcomes being
studied.

| Scenario | Zero-mean data-generating model |
| --- | --- |
| `independent` | Nine independent Gaussian histories |
| `duplicates` | Three independent Gaussian histories repeated 6, 2 and 1 times |
| `noisy_blocks` | Three factors, three variants each, independent noise standard deviation 0.1 |
| `common_factor` | Nine histories sharing 80% common Gaussian variance |
| `opposing_pairs` | Three independent Gaussian histories and their negatives |
| `student_t5` | Nine independent Student-t(5) histories scaled to unit variance |
| `ar1` | Nine Gaussian AR(1) histories with coefficient 0.4 and stationary initialization |

Every scenario/replication has a deterministic SeedSequence based on scenario
index, replication number and master seed. Worker count does not change datasets;
extending the repetition count preserves the existing prefix. Up to four workers
are used by default. Output goes to project-level `outputs/calibration.json`.

The JSON retains each evaluation, the selected original trial, method citation,
count and variance provenance, errors, source hashes and dependency versions.
`reduction_fraction` counts successful evaluations with a used count below the
number of supplied columns. Code changes invalidate any claim that historical
hashes identify the current implementation.

Report two distinct quantities:

1. Mean predicted benchmark minus mean observed maximum, with a paired Student-t
   95% interval across replications.
2. Frequency of selected-trial DSR >0.95 under the global null, with an exact
   two-sided 95% binomial interval. These are pointwise Monte Carlo intervals,
   not simultaneous coverage across all displayed methods and scenarios.

A 95% DSR cutoff is not automatically an exact 5% search-level rejection rule.
An expected maximum is not a 95th percentile. The chosen-winner event is not the
event that any returned row passes: other original rows can have different higher
moments. This null study does not estimate power, general FDR, out-of-sample
profitability or robustness to arbitrary dependence models.

The Gaussian independent and exact-duplicate cases also have a
`known_iid_reference`, using the true independent counts 9 and 3 and variance

$$
V[\widehat{SR}]=\frac{T-1}{T(T-3)}.
$$

This follows from sqrt(T) times the zero-normal sample Sharpe having a Student-t
distribution with T-1 degrees of freedom. The reference still uses the same
expected-maximum approximation and empirical moments in PSR; it is not a user
estimation method or an exact hypothesis test.

Failures are retained, not treated as non-rejections. Rate intervals and means
then describe successful evaluations conditionally, and the comparison is marked
incomplete. A separate failure envelope treats all failures as non-rejections or
rejections; that envelope is not a confidence interval. Heavy tails and AR(1) are
stress scenarios, not claims that the correlation screen or scalar DSR corrects
all violations of its assumptions.

For a smoke check rather than a calibration conclusion:

```sh
python research/calibrate.py --repetitions 10 --workers 1 --output outputs/calibration_smoke.json
```

### Recorded results

Current implementation: 1000 repetitions per scenario, T=3000 and seed 20261103.
All 23,000 evaluations completed without a calculation failure. Experiment
source SHA-256: `45317fd85323530aa31961bab10959d91ce2b4ba68cbfff09eaae1f686c180ce`.

The `correlation_clusters` rows below use the complete reference path described
above, including our explicit numerical conventions. They are not results of
the previous standardized, nonrecursive variant. Rates and binomial intervals
are percentages. Errors are per-observation Sharpe units, defined as predicted
threshold minus observed maximum.

| Scenario | Method | DSR >0.95 (%) | Exact 95% interval (%) | Mean benchmark error | Paired 95% interval |
| --- | --- | ---: | --- | ---: | --- |
| independent | auto | 0.10 | [0.003, 0.556] | -0.000317 | [-0.000849, +0.000215] |
| independent | average_correlation | 0.10 | [0.003, 0.556] | -0.000317 | [-0.000850, +0.000215] |
| independent | correlation_clusters | 12.20 | [10.235, 14.390] | -0.015468 | [-0.016222, -0.014715] |
| independent | known_iid_reference | 0.70 | [0.282, 1.437] | +0.000622 | [-0.000067, +0.001311] |
| duplicates | auto | 1.00 | [0.481, 1.831] | -0.002003 | [-0.002693, -0.001313] |
| duplicates | average_correlation | 0.60 | [0.220, 1.301] | -0.000881 | [-0.001587, -0.000175] |
| duplicates | correlation_clusters | 5.00 | [3.734, 6.539] | -0.008109 | [-0.008898, -0.007319] |
| duplicates | known_iid_reference | 2.70 | [1.787, 3.904] | -0.000268 | [-0.001146, +0.000610] |
| noisy_blocks | auto | 0.30 | [0.062, 0.874] | -0.003307 | [-0.003985, -0.002628] |
| noisy_blocks | average_correlation | 0.00 | [0.000, 0.368] | +0.002321 | [+0.001638, +0.003005] |
| noisy_blocks | correlation_clusters | 0.30 | [0.062, 0.874] | -0.003304 | [-0.003982, -0.002625] |
| common_factor | auto | 3.40 | [2.366, 4.719] | -0.000335 | [-0.001367, +0.000697] |
| common_factor | average_correlation | 8.30 | [6.665, 10.185] | -0.006512 | [-0.007549, -0.005476] |
| common_factor | correlation_clusters | 11.50 | [9.588, 13.642] | -0.009589 | [-0.010628, -0.008550] |
| opposing_pairs | auto | 0.00 | [0.000, 0.368] | -0.000336 | [-0.000528, -0.000144] |
| opposing_pairs | average_correlation | 0.00 | [0.000, 0.368] | +0.001259 | [+0.001065, +0.001454] |
| opposing_pairs | correlation_clusters | 5.20 | [3.908, 6.764] | -0.013787 | [-0.014345, -0.013230] |
| student_t5 | auto | 0.00 | [0.000, 0.368] | -0.000496 | [-0.001027, +0.000035] |
| student_t5 | average_correlation | 0.00 | [0.000, 0.368] | -0.000497 | [-0.001029, +0.000034] |
| student_t5 | correlation_clusters | 11.40 | [9.496, 13.534] | -0.015721 | [-0.016474, -0.014968] |
| ar1 | auto | 1.10 | [0.550, 1.960] | +0.000724 | [-0.000081, +0.001529] |
| ar1 | average_correlation | 1.10 | [0.550, 1.960] | +0.000724 | [-0.000081, +0.001530] |
| ar1 | correlation_clusters | 34.90 | [31.944, 37.945] | -0.022896 | [-0.024031, -0.021761] |

The reference path produces 12.20% selected-winner exceedance for independent
Gaussian trials, 11.50% with a common factor and 34.90% under AR(1). In the noisy
grouped scenario it produces 0.30%. These observations establish limitations of
this implementation under the tested designs; they do not establish a universal
property of every ONC implementation or of the original paper. In particular,
degenerate/singleton handling and stochastic initialization conventions are part
of this implementation, and no bit-for-bit replication of historical runs is claimed.

Generated results are not distributed with the repository. Run
`python research/calibrate.py` to regenerate the current experiment in
`outputs/calibration.json`, including replications, intervals, method citations,
environment versions and source hashes. The fixed seeds reproduce the datasets;
exact numerical results can depend on dependency versions. The table above records
the completed study. All intervals remain pointwise Monte Carlo uncertainty and
do not replace methodological review.

## Repository structure

```text
src/deflated_sharpe/
  __init__.py          Public imports
  statistics.py        Scalar PSR, DSR and expected maximum
  core.py              Input processing and result rows
  onc.py               Reference ONC and Appendix A.4 representatives
  clustering.py        Experimental base grouping and representatives
  policy.py            Experimental auto screening
examples/
  input.json           Synthetic example, explicitly using the reference method
  run.py               Table/JSON/CSV user example
research/
  validate.py          Quick dependence diagnostics
  calibrate.py         Full null calibration
tests/                Numerical, property and user-level tests
pyproject.toml         Build metadata and dependencies
README.md              All documentation and methodological choices
LICENSE                MIT license
.gitignore             Generated/local exclusions
```

The project is distributed under the
[MIT License](LICENSE).
