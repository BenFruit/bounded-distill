import dataclasses
import math
import warnings

import numpy as np
import pytest
from synthetic import fake_embed, make_dataset

from bounded_distill._distill import distill
from bounded_distill._router import Router, proba_from_logits


@pytest.fixture(scope="module")
def router() -> Router:
    texts, labels = make_dataset(300, flip_rate=0.0)
    return distill(texts, labels, embed=fake_embed, target_agreement=0.95)


NEW_TEXTS = [f"{c} new {i}" for c in ("billing", "tech", "other") for i in range(50)]


def test_predict_proba_has_one_row_per_text_summing_to_one(router: Router) -> None:
    proba = router.predict_proba(NEW_TEXTS)
    assert proba.shape == (150, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_predict_batch_answers_exactly_when_confidence_reaches_threshold(router: Router) -> None:
    proba = router.predict_proba(NEW_TEXTS)
    predictions = router.predict_batch(NEW_TEXTS)

    for row, prediction in zip(proba, predictions, strict=True):
        if row.max() >= router.threshold:
            assert prediction == router.labels[int(row.argmax())]
        else:
            assert prediction is None
    assert None in predictions  # this data has both confident and uncertain texts
    assert any(p is not None for p in predictions)


def test_confident_predictions_are_mostly_right_on_new_texts(router: Router) -> None:
    answered = [
        (text.split()[0], p)
        for text, p in zip(NEW_TEXTS, router.predict_batch(NEW_TEXTS), strict=True)
        if p is not None
    ]
    accuracy = sum(true == p for true, p in answered) / len(answered)
    assert accuracy >= 0.95


def test_predict_matches_predict_batch(router: Router) -> None:
    batch = router.predict_batch(NEW_TEXTS[:10])
    assert [router.predict(text) for text in NEW_TEXTS[:10]] == batch


def test_empty_input_gives_empty_output(router: Router) -> None:
    assert router.predict_proba([]).shape == (0, 3)
    assert router.predict_batch([]) == []


def test_infinite_threshold_never_answers(router: Router) -> None:
    never = dataclasses.replace(router, threshold=math.inf)
    assert never.predict_batch(NEW_TEXTS) == [None] * len(NEW_TEXTS)


def test_binary_router_predicts_both_labels() -> None:
    texts, labels = make_dataset(200, flip_rate=0.0, classes=("billing", "tech"))
    binary = distill(texts, labels, embed=fake_embed, target_agreement=0.9)
    new = [f"billing new {i}" for i in range(20)] + [f"tech new {i}" for i in range(20)]

    proba = binary.predict_proba(new)
    assert proba.shape == (40, 2)
    assert {p for p in binary.predict_batch(new) if p is not None} == {"billing", "tech"}


def test_proba_from_logits_multiclass_is_softmax() -> None:
    logits = np.array([[0.0, 1.0, 2.0]])
    expected = np.exp(logits) / np.exp(logits).sum()
    np.testing.assert_allclose(proba_from_logits(logits), expected)


def test_proba_from_logits_binary_is_sigmoid() -> None:
    z = [-2.0, 0.0, 3.0]
    p1 = [1 / (1 + math.exp(-v)) for v in z]
    expected = [[1 - p, p] for p in p1]
    np.testing.assert_allclose(proba_from_logits(np.array([[v] for v in z])), expected)


def test_proba_from_logits_handles_extreme_logits_without_overflow() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # an overflow RuntimeWarning would fail the test
        proba = proba_from_logits(np.array([[1000.0, -1000.0, 0.0], [-1000.0, 1000.0, 999.0]]))
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert proba[0, 0] == pytest.approx(1.0)
