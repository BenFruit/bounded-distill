"""Input validation, out-of-fold training and the distill() entry point."""

import math
import warnings
from collections import Counter
from collections.abc import Collection, Sequence
from typing import get_args

import numpy as np

from bounded_distill._embed import DEFAULT_MODEL, default_embed, embed_texts
from bounded_distill._report import format_invalid_counts
from bounded_distill._router import Router
from bounded_distill._sklearn import cross_fit
from bounded_distill._threshold import choose_threshold
from bounded_distill._types import (
    EmbedFn,
    InvalidPolicy,
    InvalidReason,
    ThresholdMethod,
    TrainingData,
)


def distill(
    texts: Sequence[str],
    llm_labels: Sequence[str | None],
    *,
    target_agreement: float = 0.97,
    confidence_level: float = 0.95,
    threshold_method: ThresholdMethod = "sgr",
    labels: Sequence[str] | None = None,
    embed: EmbedFn | None = None,
    n_folds: int = 5,
    random_state: int = 0,
    on_invalid: InvalidPolicy = "warn",
) -> Router:
    """Train a small classifier on LLM labels and pick its confidence threshold.

    Choose a threshold whose out-of-fold agreement score passes the lower-bound rule
    at the nominal ``confidence_level``:

    - ``threshold_method="sgr"`` (default): an SGR-style binary search (Geifman and
      El-Yaniv, 2017) with Clopper-Pearson bounds and a nominal per-step error budget.
    - ``threshold_method="wilson"``: the lowest threshold whose Wilson lower bound reaches the
      target, checked at every candidate. More coverage, but the many checks are not
      corrected for.

    Neither rule guarantees agreement for the deployed ensemble. Calibration uses one fold
    model per row, these rows are dependent, and thresholds are selected from the data.

    If no threshold reaches the target, the router never answers (``threshold == math.inf``)
    and a ``UserWarning`` is emitted. The router predicts with the average of the
    ``n_folds`` fold models, not the single-fold scores used to choose the threshold.
    """
    check_settings(target_agreement, confidence_level, n_folds, on_invalid, threshold_method)
    data = prepare_training_data(
        texts, llm_labels, labels=labels, n_folds=n_folds, on_invalid=on_invalid
    )
    embed_fn = default_embed if embed is None else embed
    x = embed_texts(embed_fn, data.texts)
    class_index = {label: i for i, label in enumerate(data.label_set)}
    y = np.array([class_index[label] for label in data.labels], dtype=np.intp)

    oof_proba, weights, intercept = cross_fit(x, y, len(data.label_set), n_folds, random_state)
    agrees = oof_proba.argmax(axis=1) == y
    threshold = choose_threshold(
        threshold_method,
        oof_proba.max(axis=1).tolist(),
        agrees.tolist(),
        target_agreement,
        confidence_level,
    )
    if threshold == math.inf:
        warnings.warn(
            f"target_agreement={target_agreement} is not reachable at "
            f"confidence_level={confidence_level} on this data, so the router will always "
            "return None; try a lower target, more data, cleaner labels, or "
            'threshold_method="wilson"',
            UserWarning,
            stacklevel=2,
        )

    return Router(
        labels=data.label_set,
        threshold=threshold,
        weights=weights,
        intercept=intercept,
        embed=embed_fn,
        embedder=DEFAULT_MODEL if embed is None else "custom",
        target_agreement=target_agreement,
        confidence_level=confidence_level,
        threshold_method=threshold_method,
        training=data,
        oof_proba=oof_proba,
    )


def check_settings(
    target_agreement: float,
    confidence_level: float,
    n_folds: int,
    on_invalid: str,
    threshold_method: str,
) -> None:
    """Reject invalid settings before any expensive work starts."""
    for name, value in [
        ("target_agreement", target_agreement),
        ("confidence_level", confidence_level),
    ]:
        if not 0 < value < 1:
            raise ValueError(f"{name} must be in (0, 1), got {value}")
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")
    # Literal types are checked statically; this catches callers without a type checker.
    if on_invalid not in get_args(InvalidPolicy):
        raise ValueError(
            f"on_invalid must be one of {list(get_args(InvalidPolicy))}, got {on_invalid!r}"
        )
    if threshold_method not in get_args(ThresholdMethod):
        raise ValueError(
            f"threshold_method must be one of {list(get_args(ThresholdMethod))}, "
            f"got {threshold_method!r}"
        )


def invalid_reason(label: str | None, allowed: Collection[str] | None) -> InvalidReason | None:
    """Why ``label`` is invalid, or ``None`` if it is a valid label."""
    if label is None:
        return "missing"
    if not label.strip():
        return "empty"
    if allowed is not None and label not in allowed:
        return "unknown"
    return None


def describe_invalid(invalid: dict[InvalidReason, int], n_total: int) -> str:
    """Human-readable summary, e.g. '5 of 100 LLM labels are invalid: 3 missing (None), 2 empty'."""
    n_invalid = sum(invalid.values())
    return f"{n_invalid} of {n_total} LLM labels are invalid: {format_invalid_counts(invalid)}"


def prepare_training_data(
    texts: Sequence[str],
    llm_labels: Sequence[str | None],
    *,
    labels: Sequence[str] | None,
    n_folds: int,
    on_invalid: InvalidPolicy,
) -> TrainingData:
    """Drop invalid LLM labels and check that every class has enough examples.

    An LLM label is invalid if it is ``None``, empty or whitespace-only, or not in ``labels``
    when ``labels`` is given. Invalid labels are handled according to ``on_invalid``.
    Without ``labels`` the label set is inferred from the valid data and sorted.
    """
    if len(texts) != len(llm_labels):
        raise ValueError(
            f"texts and llm_labels differ in length: {len(texts)} != {len(llm_labels)}"
        )
    if labels is not None:
        if len(set(labels)) != len(labels):
            raise ValueError(f"labels contain duplicates: {list(labels)}")
        if any(not label.strip() for label in labels):
            raise ValueError(f"labels must not be empty strings: {list(labels)}")

    allowed = None if labels is None else set(labels)
    kept: list[tuple[int, str, str]] = []
    invalid: Counter[InvalidReason] = Counter()
    for i, (text, label) in enumerate(zip(texts, llm_labels, strict=True)):
        reason = invalid_reason(label, allowed)
        if reason is not None:
            invalid[reason] += 1
            continue
        assert label is not None  # invalid_reason() reports None as "missing"
        kept.append((i, text, label))

    if invalid:
        message = describe_invalid(dict(invalid), len(texts))
        if on_invalid == "error":
            raise ValueError(message)
        if on_invalid == "warn":
            # stacklevel=3 points the warning at the user's distill() call.
            warnings.warn(f"{message}; they were excluded", UserWarning, stacklevel=3)

    kept_labels = [label for _, _, label in kept]
    label_set = tuple(labels) if labels is not None else tuple(sorted(set(kept_labels)))
    if len(label_set) < 2:
        raise ValueError(f"need at least 2 classes, got {list(label_set)}")
    counts = Counter(kept_labels)
    too_small = [f"{label!r} ({counts[label]})" for label in label_set if counts[label] < n_folds]
    if too_small:
        raise ValueError(
            f"every class needs at least n_folds={n_folds} valid examples; "
            f"too few: {', '.join(too_small)}"
        )

    return TrainingData(
        texts=[text for _, text, _ in kept],
        labels=kept_labels,
        indices=[i for i, _, _ in kept],
        label_set=label_set,
        n_total=len(texts),
        invalid=dict(invalid),
    )
