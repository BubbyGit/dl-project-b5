import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Summary:
    n: int
    mean: float
    std: float
    ci95: float


@dataclass(frozen=True)
class Comparison:
    n: int
    diff: float
    ci_low: float
    ci_high: float
    p_paired: float
    p_welch: float
    correlation: float


def t_halfwidth(std, n, level=0.95):
    if n < 2:
        return math.nan
    return float(stats.t.ppf(0.5 + level / 2, n - 1) * std / math.sqrt(n))


def describe(values):
    x = np.asarray(list(values), dtype=float)
    n = len(x)
    std = float(x.std(ddof=1)) if n > 1 else math.nan
    return Summary(n=n, mean=float(x.mean()) if n else math.nan, std=std, ci95=t_halfwidth(std, n))


def compare_paired(a, b):
    seeds = sorted(set(a) & set(b))
    x = np.array([a[s] for s in seeds], dtype=float)
    y = np.array([b[s] for s in seeds], dtype=float)
    n = len(seeds)
    if n < 2:
        return Comparison(n, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)
    d = x - y
    half = t_halfwidth(float(d.std(ddof=1)), n)
    diff = float(d.mean())
    return Comparison(
        n=n,
        diff=diff,
        ci_low=diff - half,
        ci_high=diff + half,
        p_paired=float(stats.ttest_rel(x, y).pvalue),
        p_welch=float(stats.ttest_ind(x, y, equal_var=False).pvalue),
        correlation=float(np.corrcoef(x, y)[0, 1]),
    )


def t_test_power(df, ncp, alpha):
    crit = stats.t.ppf(1 - alpha / 2, df)
    power = stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp)
    if not math.isfinite(power):
        power = stats.norm.sf(crit - ncp) + stats.norm.cdf(-crit - ncp)
    return float(min(power, 1.0))


def power_two_sample(delta, sigma, n, alpha=0.05):
    return t_test_power(2 * n - 2, delta / (sigma * math.sqrt(2 / n)), alpha)


def power_paired(delta, sigma, n, rho=0.0, alpha=0.05):
    sd_diff = sigma * math.sqrt(2 * (1 - rho))
    return t_test_power(n - 1, delta / (sd_diff / math.sqrt(n)), alpha)


def smallest_n(power_at, target=0.8, n_min=2, n_max=1_000_000):
    hi = n_min
    while power_at(hi) < target:
        if hi >= n_max:
            return None
        hi = min(2 * hi, n_max)
    lo = max(n_min, hi // 2)
    while lo < hi:
        mid = (lo + hi) // 2
        if power_at(mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def required_n(delta, sigma, power=0.8, alpha=0.05, paired=False, rho=0.0):
    if paired:
        return smallest_n(lambda n: power_paired(delta, sigma, n, rho, alpha), power)
    return smallest_n(lambda n: power_two_sample(delta, sigma, n, alpha), power)


def minimum_detectable_effect(sigma, n, power=0.8, alpha=0.05):
    lo, hi = 0.0, 10 * sigma
    for _ in range(100):
        mid = (lo + hi) / 2
        if power_two_sample(mid, sigma, n, alpha) >= power:
            hi = mid
        else:
            lo = mid
    return hi
