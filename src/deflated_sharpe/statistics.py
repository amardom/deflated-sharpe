from math import erfc, exp, isfinite, sqrt
from numbers import Integral, Real
from statistics import NormalDist

def finite_number(name, value):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)

def compute_expected_maximum_sharpe(sharpe_variance, independent_trials):
    variance = finite_number("sharpe_variance", sharpe_variance)
    count = finite_number("independent_trials", independent_trials)
    if variance < 0:
        raise ValueError("sharpe_variance must be nonnegative")
    if count < 1 or 1 < count < 2:
        raise ValueError("independent_trials must equal 1 or be at least 2")
    if count == 1 or variance == 0:
        return 0.0
    normal = NormalDist()
    gamma = 0.5772156649015329
    first = -normal.inv_cdf(1 / count)
    second = -normal.inv_cdf(exp(-1) / count)
    return sqrt(variance) * ((1 - gamma) * first + gamma * second)

def compute_probabilistic_sharpe_ratio(sharpe, observations, skewness, kurtosis, benchmark=0):
    sharpe = finite_number("sharpe", sharpe)
    skewness = finite_number("skewness", skewness)
    kurtosis = finite_number("kurtosis", kurtosis)
    benchmark = finite_number("benchmark", benchmark)
    if isinstance(observations, bool) or not isinstance(observations, Integral) or observations < 2:
        raise ValueError("observations must be an integer of at least 2")
    if kurtosis < skewness * skewness + 1 - 1e-12:
        raise ValueError("Pearson kurtosis must be at least skewness squared plus 1")
    denominator = 1 - skewness * sharpe + (kurtosis - 1) * sharpe * sharpe / 4
    if not isfinite(denominator) or denominator <= 0:
        raise ValueError("Sharpe variance denominator must be finite and positive")
    z = (sharpe - benchmark) * sqrt(observations - 1) / sqrt(denominator)
    return erfc(-z / sqrt(2)) / 2

def compute_deflated_sharpe_ratio(sharpe, observations, skewness, kurtosis,
                         sharpe_variance, independent_trials):
    benchmark = compute_expected_maximum_sharpe(sharpe_variance, independent_trials)
    return compute_probabilistic_sharpe_ratio(sharpe, observations, skewness, kurtosis, benchmark)

