import math

import numpy as np
import pytest

from bounded_distill._threshold import select_threshold, wilson_lower_bound

Z_95 = 1.6448536269514722  # NormalDist().inv_cdf(0.95)


# Reference values from scipy.stats.binomtest(k, n).proportion_ci(method="wilson"):
# the one-sided bound at level c equals the lower end of the two-sided interval at 2c - 1.
@pytest.mark.parametrize(
    ("k", "n", "confidence_level", "expected"),
    [
        (95, 100, 0.95, 0.9008389147209472),
        (9, 10, 0.95, 0.6522813326641417),
        (970, 1000, 0.95, 0.9597804490016412),
        (50, 100, 0.99, 0.38670784097761807),
        (1, 3, 0.8, 0.15669750760815637),
    ],
)
def test_wilson_matches_reference_values(
    k: int, n: int, confidence_level: float, expected: float
) -> None:
    assert wilson_lower_bound(k, n, confidence_level) == pytest.approx(expected, rel=1e-12)


def test_wilson_is_zero_without_data() -> None:
    assert wilson_lower_bound(0, 0, 0.95) == 0.0


def test_wilson_is_zero_without_successes() -> None:
    assert wilson_lower_bound(0, 50, 0.95) == pytest.approx(0.0, abs=1e-12)


def test_wilson_all_successes_has_closed_form() -> None:
    # With k == n the formula reduces to n / (n + z^2): never 1, even for perfect data.
    assert wilson_lower_bound(100, 100, 0.95) == pytest.approx(100 / (100 + Z_95**2))


def test_wilson_is_below_point_estimate() -> None:
    assert wilson_lower_bound(80, 100, 0.95) < 0.8


def test_wilson_decreases_with_higher_confidence_level() -> None:
    assert wilson_lower_bound(80, 100, 0.99) < wilson_lower_bound(80, 100, 0.9)


def test_wilson_increases_with_more_data_at_same_rate() -> None:
    assert wilson_lower_bound(80, 100, 0.95) < wilson_lower_bound(800, 1000, 0.95)


@pytest.mark.parametrize(("k", "n"), [(-1, 10), (11, 10)])
def test_wilson_rejects_invalid_counts(k: int, n: int) -> None:
    with pytest.raises(ValueError, match="0 <= k <= n"):
        wilson_lower_bound(k, n, 0.95)


@pytest.mark.parametrize("confidence_level", [0.0, 1.0, 1.5])
def test_wilson_rejects_invalid_confidence_level(confidence_level: float) -> None:
    with pytest.raises(ValueError, match="confidence_level"):
        wilson_lower_bound(5, 10, confidence_level)


def test_select_covers_everything_when_all_agree() -> None:
    confidences = [i / 1000 for i in range(500, 1000)]
    threshold = select_threshold(confidences, [True] * 500, 0.97, 0.95)
    assert threshold == min(confidences)


def test_select_excludes_low_confidence_disagreements() -> None:
    # 150 confident agreements, then 50 less confident disagreements.
    confidences = [0.9 + i / 10_000 for i in range(150)] + [0.5 + i / 1000 for i in range(50)]
    agrees = [True] * 150 + [False] * 50
    threshold = select_threshold(confidences, agrees, 0.98, 0.95)
    assert threshold == 0.9


def test_select_returns_inf_when_target_unreachable() -> None:
    assert select_threshold([0.9, 0.8, 0.7], [False, False, False], 0.97, 0.95) == math.inf


def test_select_returns_inf_when_too_few_items_for_the_bound() -> None:
    # 50 perfect agreements: the bound is 50 / (50 + z^2) ~ 0.949 < 0.97.
    assert select_threshold([0.99] * 50, [True] * 50, 0.97, 0.95) == math.inf


def test_select_returns_inf_for_empty_input() -> None:
    assert select_threshold([], [], 0.97, 0.95) == math.inf


def test_select_never_splits_tied_confidences() -> None:
    # 200 agreements at 0.9, then 200 items tied at 0.6, only half of them agreeing.
    # Covering only the agreeing half of the tie would pass; covering the whole tie fails.
    confidences = [0.9] * 200 + [0.6] * 200
    agrees = [True] * 200 + [True, False] * 100
    assert select_threshold(confidences, agrees, 0.97, 0.95) == 0.9


def _brute_force_threshold(
    confidences: list[float], agrees: list[bool], target: float, level: float
) -> float:
    qualifying = [
        t
        for t in set(confidences)
        if wilson_lower_bound(
            sum(a for c, a in zip(confidences, agrees, strict=True) if c >= t),
            sum(c >= t for c in confidences),
            level,
        )
        >= target
    ]
    return min(qualifying, default=math.inf)


@pytest.mark.parametrize("seed", range(20))
def test_select_matches_brute_force(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 300
    # Rounded confidences create ties; agreement is likelier at high confidence.
    confidences = [round(float(c), 2) for c in rng.uniform(0.3, 1.0, n)]
    agrees = [bool(rng.random() < c) for c in confidences]
    expected = _brute_force_threshold(confidences, agrees, 0.9, 0.95)
    assert select_threshold(confidences, agrees, 0.9, 0.95) == expected


def test_select_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        select_threshold([0.9, 0.8], [True], 0.97, 0.95)


@pytest.mark.parametrize("target", [0.0, 1.0])
def test_select_rejects_invalid_target(target: float) -> None:
    with pytest.raises(ValueError, match="target_agreement"):
        select_threshold([0.9], [True], target, 0.95)
