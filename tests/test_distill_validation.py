import warnings
from collections.abc import Sequence

import pytest

from bounded_distill._distill import (
    check_settings,
    describe_invalid,
    invalid_reason,
    prepare_training_data,
)
from bounded_distill._types import InvalidPolicy, TrainingData


def _texts(n: int) -> list[str]:
    return [f"text {i}" for i in range(n)]


def _prepare(
    llm_labels: Sequence[str | None],
    *,
    labels: Sequence[str] | None = None,
    n_folds: int = 2,
    on_invalid: InvalidPolicy = "drop",
) -> TrainingData:
    return prepare_training_data(
        _texts(len(llm_labels)), llm_labels, labels=labels, n_folds=n_folds, on_invalid=on_invalid
    )


# --- invalid_reason -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "allowed", "expected"),
    [
        (None, None, "missing"),
        (None, {"a"}, "missing"),
        ("", None, "empty"),
        ("  \n", None, "empty"),
        ("", {"a"}, "empty"),
        ("b", {"a"}, "unknown"),
        ("A", {"a"}, "unknown"),  # labels are compared exactly
        (" a", {"a"}, "unknown"),  # and never stripped
        ("a", {"a"}, None),
        ("anything", None, None),
    ],
)
def test_invalid_reason(label: str | None, allowed: set[str] | None, expected: str | None) -> None:
    assert invalid_reason(label, allowed) == expected


def test_describe_invalid_lists_reasons_in_fixed_order() -> None:
    message = describe_invalid({"unknown": 40, "missing": 12, "empty": 3}, 1000)
    assert message == (
        "55 of 1000 LLM labels are invalid: 12 missing (None), 3 empty, 40 not in labels"
    )


# --- prepare_training_data: filtering -------------------------------------------------------


def test_keeps_valid_pairs_and_infers_sorted_label_set() -> None:
    llm_labels = ["tech", "billing"] * 3
    data = _prepare(llm_labels)

    assert data.label_set == ("billing", "tech")
    assert data.texts == _texts(6)
    assert data.labels == llm_labels
    assert data.indices == list(range(6))
    assert (data.n_total, data.n_invalid, data.invalid) == (6, 0, {})


def test_drops_invalid_labels_and_keeps_original_indices() -> None:
    data = _prepare(["a", None, "b", "a", "", "b", "  "])

    assert data.labels == ["a", "b", "a", "b"]
    assert data.texts == ["text 0", "text 2", "text 3", "text 5"]
    assert data.indices == [0, 2, 3, 5]
    assert data.invalid == {"missing": 1, "empty": 2}
    assert (data.n_total, data.n_invalid) == (7, 3)


def test_drops_labels_outside_the_given_label_set() -> None:
    data = _prepare(["a", "b", "Unknown", "a", "b", "a."], labels=["b", "a"])

    assert data.labels == ["a", "b", "a", "b"]
    assert data.invalid == {"unknown": 2}


def test_given_label_set_keeps_its_order() -> None:
    assert _prepare(["a", "b"] * 2, labels=["b", "a"]).label_set == ("b", "a")


# --- prepare_training_data: on_invalid policy -----------------------------------------------


def test_drop_policy_is_silent() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail the test
        _prepare(["a", "b", None] * 2, on_invalid="drop")


def test_warn_policy_warns_with_counts() -> None:
    with pytest.warns(UserWarning, match=r"2 of 6 LLM labels are invalid: 2 missing \(None\)"):
        data = _prepare(["a", "b", None] * 2, on_invalid="warn")
    assert data.n_invalid == 2


def test_warn_policy_is_silent_when_all_labels_are_valid() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _prepare(["a", "b"] * 2, on_invalid="warn")


def test_error_policy_raises_with_counts() -> None:
    with pytest.raises(ValueError, match=r"1 of 5 LLM labels are invalid: 1 empty"):
        _prepare(["a", "b", "a", "b", ""], on_invalid="error")


# --- prepare_training_data: input errors ----------------------------------------------------


def test_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        prepare_training_data(_texts(3), ["a", "b"], labels=None, n_folds=2, on_invalid="drop")


def test_rejects_duplicate_labels() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        _prepare(["a", "b"] * 2, labels=["a", "b", "a"])


def test_rejects_empty_string_in_labels() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _prepare(["a", "b"] * 2, labels=["a", "b", " "])


def test_rejects_single_class() -> None:
    with pytest.raises(ValueError, match="at least 2 classes"):
        _prepare(["a"] * 5)


def test_rejects_when_only_one_class_survives_filtering() -> None:
    with pytest.raises(ValueError, match="at least 2 classes"):
        _prepare(["a", None, "a", ""])


def test_names_every_class_with_too_few_examples() -> None:
    llm_labels = ["big"] * 5 + ["small"] * 2 + ["tiny"]
    with pytest.raises(ValueError, match=r"too few: 'small' \(2\), 'tiny' \(1\)"):
        _prepare(llm_labels, n_folds=5)


def test_given_label_without_examples_is_too_small() -> None:
    with pytest.raises(ValueError, match=r"'c' \(0\)"):
        _prepare(["a", "b"] * 5, labels=["a", "b", "c"], n_folds=5)


# --- check_settings -------------------------------------------------------------------------

_METHOD = {"threshold_method": "sgr"}


def test_check_settings_accepts_defaults() -> None:
    check_settings(
        target_agreement=0.97, confidence_level=0.95, n_folds=5, on_invalid="warn", **_METHOD
    )


@pytest.mark.parametrize("value", [0.0, 1.0, -0.5, 97.0])
def test_check_settings_rejects_target_outside_unit_interval(value: float) -> None:
    with pytest.raises(ValueError, match="target_agreement"):
        check_settings(
            target_agreement=value, confidence_level=0.95, n_folds=5, on_invalid="warn", **_METHOD
        )


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_check_settings_rejects_confidence_level_outside_unit_interval(value: float) -> None:
    with pytest.raises(ValueError, match="confidence_level"):
        check_settings(
            target_agreement=0.97, confidence_level=value, n_folds=5, on_invalid="warn", **_METHOD
        )


def test_check_settings_rejects_fewer_than_two_folds() -> None:
    with pytest.raises(ValueError, match="n_folds"):
        check_settings(
            target_agreement=0.97, confidence_level=0.95, n_folds=1, on_invalid="warn", **_METHOD
        )


def test_check_settings_rejects_unknown_invalid_policy() -> None:
    with pytest.raises(ValueError, match=r"on_invalid must be one of \['drop', 'warn', 'error'\]"):
        check_settings(
            target_agreement=0.97, confidence_level=0.95, n_folds=5, on_invalid="wran", **_METHOD
        )


def test_check_settings_rejects_unknown_threshold_method() -> None:
    with pytest.raises(ValueError, match=r"threshold_method must be one of \['sgr', 'wilson'\]"):
        check_settings(
            target_agreement=0.97,
            confidence_level=0.95,
            n_folds=5,
            on_invalid="warn",
            threshold_method="wilsn",
        )
