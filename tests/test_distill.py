import math
import warnings
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pytest
from synthetic import DIM, fake_embed, make_dataset

from bounded_distill._distill import distill
from bounded_distill._embed import l2_normalize
from bounded_distill._router import Router
from bounded_distill._threshold import wilson_lower_bound


def oof(router: Router) -> npt.NDArray[np.float64]:
    """Out-of-fold probabilities; always present on a router fresh from distill()."""
    assert router.oof_proba is not None
    return router.oof_proba


def training_labels(router: Router) -> list[str]:
    assert router.training is not None
    return router.training.labels


def coverage(router: Router) -> float:
    confidences = oof(router).max(axis=1)
    return float((confidences >= router.threshold).mean())


def test_clean_labels_give_a_threshold_that_meets_the_target() -> None:
    texts, labels = make_dataset(500, flip_rate=0.0)
    router = distill(texts, labels, embed=fake_embed, target_agreement=0.95)

    assert router.labels == ("billing", "other", "tech")
    assert 0 < router.threshold < 1
    assert coverage(router) > 0.5

    # The defining property: on covered out-of-fold predictions the bound reaches the target.
    confidences = oof(router).max(axis=1)
    covered = confidences >= router.threshold
    student = [router.labels[int(i)] for i in oof(router).argmax(axis=1)]
    k = sum(s == t for s, t, c in zip(student, training_labels(router), covered, strict=True) if c)
    assert wilson_lower_bound(k, int(covered.sum()), 0.95) >= 0.95


def test_noisy_labels_give_lower_coverage() -> None:
    clean = distill(*make_dataset(1000, flip_rate=0.0), embed=fake_embed, target_agreement=0.95)
    noisy = distill(*make_dataset(1000, flip_rate=0.02), embed=fake_embed, target_agreement=0.95)
    assert 0 < coverage(noisy) < coverage(clean)


def test_unreachable_target_gives_inf_threshold_and_warns_at_caller() -> None:
    # 5% random label noise caps agreement near 95%, so a 99% target cannot be met.
    texts, labels = make_dataset(200, flip_rate=0.05)
    with pytest.warns(UserWarning, match="not reachable") as record:
        router = distill(texts, labels, embed=fake_embed, target_agreement=0.99)

    assert router.threshold == math.inf
    assert coverage(router) == 0.0
    assert record[0].filename == __file__  # stacklevel points at this call, not library code


def test_invalid_label_warning_points_at_caller() -> None:
    texts, labels = make_dataset(100, flip_rate=0.0)
    llm_labels: list[str | None] = [*labels[:-1], None]
    with pytest.warns(UserWarning, match="1 missing") as record:
        distill(texts, llm_labels, embed=fake_embed, target_agreement=0.9)
    assert record[0].filename == __file__


def test_same_random_state_is_reproducible() -> None:
    texts, labels = make_dataset(200, flip_rate=0.02)
    first = distill(texts, labels, embed=fake_embed, target_agreement=0.9, random_state=7)
    second = distill(texts, labels, embed=fake_embed, target_agreement=0.9, random_state=7)

    assert first.threshold == second.threshold
    np.testing.assert_array_equal(oof(first), oof(second))


def test_binary_labels_use_sklearn_single_logit_layout() -> None:
    texts, labels = make_dataset(200, flip_rate=0.0, classes=("billing", "tech"))
    router = distill(texts, labels, embed=fake_embed, target_agreement=0.9)

    assert router.labels == ("billing", "tech")
    assert router.weights.shape == (5, DIM, 1)  # one model per fold
    assert router.intercept.shape == (5, 1)
    assert oof(router).shape == (400, 2)


def test_router_keeps_settings_and_marks_custom_embedder() -> None:
    texts, labels = make_dataset(100, flip_rate=0.0)
    router = distill(texts, labels, embed=fake_embed, target_agreement=0.9, confidence_level=0.8)

    assert router.embedder == "custom"
    assert (router.target_agreement, router.confidence_level) == (0.9, 0.8)
    assert router.weights.shape == (5, DIM, 3)
    assert oof(router).shape == (300, 3)


def test_settings_are_checked_before_embedding() -> None:
    def failing_embed(texts: Sequence[str]) -> npt.NDArray[np.float64]:
        raise AssertionError("embed must not be called")

    with pytest.raises(ValueError, match="target_agreement"):
        distill(["a"], ["x"], embed=failing_embed, target_agreement=97)


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (np.zeros((3, 4)), r"shape \(4, dim\), got \(3, 4\)"),
        (np.zeros(4), r"shape \(4, dim\), got \(4,\)"),
        (np.full((4, 2), np.nan), "NaN or infinite"),
    ],
)
def test_rejects_bad_embed_output(output: npt.NDArray[np.float64], message: str) -> None:
    def bad_embed(texts: Sequence[str]) -> npt.NDArray[np.float64]:
        return output

    with warnings.catch_warnings(), pytest.raises(ValueError, match=message):
        distill(["a", "b", "a", "b"], ["x", "y", "x", "y"], embed=bad_embed, n_folds=2)


def test_l2_normalize_gives_unit_rows_and_keeps_zero_rows() -> None:
    x = np.array([[3.0, 4.0], [0.0, 0.0], [0.0, -2.0]], dtype=np.float32)
    np.testing.assert_allclose(l2_normalize(x), [[0.6, 0.8], [0.0, 0.0], [0.0, -1.0]])


def test_default_threshold_method_is_sgr() -> None:
    router = distill(*make_dataset(300, flip_rate=0.0), embed=fake_embed, target_agreement=0.9)
    assert router.threshold_method == "sgr"


def test_wilson_answers_at_least_as_much_as_sgr() -> None:
    texts, labels = make_dataset(1000, flip_rate=0.02)
    sgr = distill(texts, labels, embed=fake_embed, target_agreement=0.95)
    wilson = distill(
        texts, labels, embed=fake_embed, target_agreement=0.95, threshold_method="wilson"
    )
    assert wilson.threshold_method == "wilson"
    assert wilson.threshold <= sgr.threshold < math.inf
    assert coverage(wilson) >= coverage(sgr) > 0


def test_unreachable_target_warning_suggests_wilson() -> None:
    texts, labels = make_dataset(200, flip_rate=0.05)
    with pytest.warns(UserWarning, match='threshold_method="wilson"'):
        distill(texts, labels, embed=fake_embed, target_agreement=0.99)
