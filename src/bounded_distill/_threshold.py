"""Lower bounds on agreement and confidence threshold selection (pure functions).

Two ways to pick the threshold:

- ``select_threshold``: the lowest threshold whose Wilson lower bound reaches the target,
  checked at every candidate. It gives the most coverage, but the many checks are not
  corrected for.
- ``select_threshold_sgr``: Selection with Guaranteed Risk (Geifman and El-Yaniv, 2017). A
  binary search tests a few candidates, each with an exact Clopper-Pearson bound at a
  nominal error budget split between the steps. This does not give a guarantee for
  the chosen threshold on dependent out-of-fold predictions.
"""

import math
from collections.abc import Sequence
from statistics import NormalDist

from bounded_distill._types import ThresholdMethod


def wilson_lower_bound(k: int, n: int, confidence_level: float) -> float:
    """One-sided Wilson score lower bound for a binomial proportion.

    The nominal coverage for binomial trials is ``confidence_level``. Actual coverage is
    approximate. Returns 0.0 when ``n == 0``.
    """
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if not 0 < confidence_level < 1:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
    if n == 0:
        return 0.0

    z = NormalDist().inv_cdf(confidence_level)
    p = k / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / (1 + z * z / n)


def clopper_pearson_lower_bound(k: int, n: int, confidence_level: float) -> float:
    """One-sided exact (Clopper-Pearson) lower bound for a binomial proportion.

    For binomial trials and a fixed test, coverage is at least ``confidence_level``.
    Returns 0.0 when ``k == 0``,
    which includes ``n == 0``. More conservative than the Wilson bound.
    """
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if not 0 < confidence_level < 1:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
    if k == 0:
        return 0.0

    from scipy.stats import beta  # pyright: ignore[reportMissingTypeStubs]  # imported lazily

    quantile = beta.ppf(1 - confidence_level, k, n - k + 1)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return float(quantile)  # pyright: ignore[reportUnknownArgumentType]


def select_threshold(
    confidences: Sequence[float],
    agrees: Sequence[bool],
    target_agreement: float,
    confidence_level: float,
) -> float:
    """Lowest confidence threshold whose agreement lower bound reaches the target.

    A threshold ``t`` covers every item with ``confidence >= t``. Among all candidate
    thresholds (the distinct confidences), return the lowest one, i.e. maximum coverage,
    for which the Wilson lower bound of agreement on the covered items is at least
    ``target_agreement``. Return ``math.inf`` (cover nothing) if no candidate qualifies.
    """
    _check_inputs(confidences, agrees, target_agreement)

    # Walk from the most to the least confident item, keeping running counts.
    # The bound is not monotone in coverage, so every candidate must be checked.
    items = sorted(zip(confidences, agrees, strict=True), reverse=True)
    best = math.inf
    n_agree = 0
    for i, (confidence, agree) in enumerate(items):
        n_agree += agree
        is_last_of_tie = i + 1 == len(items) or items[i + 1][0] < confidence
        if not is_last_of_tie:
            continue  # equal confidences are covered together, never split
        if wilson_lower_bound(n_agree, i + 1, confidence_level) >= target_agreement:
            best = confidence
    return best


def select_threshold_sgr(
    confidences: Sequence[float],
    agrees: Sequence[bool],
    target_agreement: float,
    confidence_level: float,
) -> float:
    """Threshold from Selection with Guaranteed Risk (Geifman and El-Yaniv, 2017).

    The candidates are the distinct confidences plus "cover nothing" (``math.inf``), in
    increasing order. A binary search looks for the lowest candidate whose Clopper-Pearson
    lower bound of agreement reaches ``target_agreement``. It tests at most
    ``ceil(log2(number of candidates))`` thresholds, and each test uses the error budget
    ``1 - confidence_level`` divided by that number. Counting the realized adaptive
    path does not establish a fixed-family union bound. The search can skip
    passing candidates when agreement is not monotone in confidence.
    """
    _check_inputs(confidences, agrees, target_agreement)
    if not 0 < confidence_level < 1:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")

    candidates = sorted(set(confidences))
    level = sgr_test_level(len(candidates), confidence_level)

    def reaches_target(threshold: float) -> bool:
        covered = [a for c, a in zip(confidences, agrees, strict=True) if c >= threshold]
        bound = clopper_pearson_lower_bound(sum(covered), len(covered), level)
        return bound >= target_agreement

    # Index len(candidates) stands for math.inf, which needs no test: it covers nothing.
    low, high = 0, len(candidates)
    while low < high:
        middle = (low + high) // 2
        if reaches_target(candidates[middle]):
            high = middle
        else:
            low = middle + 1
    return candidates[low] if low < len(candidates) else math.inf


def sgr_test_level(n_candidates: int, confidence_level: float) -> float:
    """Confidence level of each SGR test: the error budget split between the tests."""
    n_tests = max(1, math.ceil(math.log2(n_candidates + 1)))
    return 1 - (1 - confidence_level) / n_tests


def choose_threshold(
    method: ThresholdMethod,
    confidences: Sequence[float],
    agrees: Sequence[bool],
    target_agreement: float,
    confidence_level: float,
) -> float:
    """The threshold picked by ``method``: ``select_threshold_sgr`` or ``select_threshold``."""
    select = select_threshold_sgr if method == "sgr" else select_threshold
    return select(confidences, agrees, target_agreement, confidence_level)


def agreement_lower_bound(
    method: ThresholdMethod, k: int, n: int, n_candidates: int, confidence_level: float
) -> float:
    """The lower bound that ``method`` checks: corrected Clopper-Pearson for SGR, else Wilson.

    ``n_candidates`` is the number of distinct confidences the threshold was chosen from;
    only SGR needs it, to split the confidence level the same way as during the search.
    """
    if method == "sgr":
        return clopper_pearson_lower_bound(k, n, sgr_test_level(n_candidates, confidence_level))
    return wilson_lower_bound(k, n, confidence_level)


def _check_inputs(
    confidences: Sequence[float], agrees: Sequence[bool], target_agreement: float
) -> None:
    if len(confidences) != len(agrees):
        raise ValueError(
            f"confidences and agrees differ in length: {len(confidences)} != {len(agrees)}"
        )
    if not 0 < target_agreement < 1:
        raise ValueError(f"target_agreement must be in (0, 1), got {target_agreement}")
