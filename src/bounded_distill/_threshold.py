"""Wilson lower bound and confidence threshold selection (pure functions)."""

import math
from collections.abc import Sequence
from statistics import NormalDist


def wilson_lower_bound(k: int, n: int, confidence_level: float) -> float:
    """One-sided Wilson score lower bound for a binomial proportion.

    With probability ``confidence_level`` the true proportion is at least the returned value,
    given ``k`` successes out of ``n`` trials. Returns 0.0 when ``n == 0``.
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
    if len(confidences) != len(agrees):
        raise ValueError(
            f"confidences and agrees differ in length: {len(confidences)} != {len(agrees)}"
        )
    if not 0 < target_agreement < 1:
        raise ValueError(f"target_agreement must be in (0, 1), got {target_agreement}")

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
