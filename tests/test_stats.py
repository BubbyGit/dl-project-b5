import math

import pytest

from flipbench.stats import compare_paired, describe, minimum_detectable_effect, power_two_sample, required_n


def test_required_n_matches_textbook_values():
    assert required_n(delta=1.0, sigma=1.0) == 17
    assert required_n(delta=0.5, sigma=1.0) == 64
    assert required_n(delta=1.0, sigma=math.sqrt(0.5), paired=True, rho=0.0) == 10


def test_minimum_detectable_effect_inverts_power():
    mde = minimum_detectable_effect(sigma=0.14, n=50)
    assert power_two_sample(mde, 0.14, 50) == pytest.approx(0.8, abs=1e-3)


def test_describe():
    summary = describe([1.0, 2.0, 3.0, 4.0])
    assert summary.n == 4
    assert summary.mean == pytest.approx(2.5)
    assert summary.std == pytest.approx(1.2909944)
    assert summary.ci95 == pytest.approx(3.182446 * 1.2909944 / 2, rel=1e-5)


def test_compare_paired_uses_matching_seeds():
    a = {0: 1.0, 1: 2.0, 2: 3.0, 3: 5.0}
    b = {0: 0.5, 1: 1.0, 2: 2.0, 9: 100.0}
    comparison = compare_paired(a, b)
    assert comparison.n == 3
    assert comparison.diff == pytest.approx((0.5 + 1.0 + 1.0) / 3)
    assert comparison.ci_low < comparison.diff < comparison.ci_high
