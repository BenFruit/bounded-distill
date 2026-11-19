"""Report, Suspect and GoldReport dataclasses and their builders."""

import dataclasses
import math
import statistics
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from bounded_distill._threshold import wilson_lower_bound
from bounded_distill._types import InvalidReason, TrainingData

_REASON_TEXT: dict[InvalidReason, str] = {
    "missing": "missing (None)",
    "empty": "empty",
    "unknown": "not in labels",
}
_MAX_DISAGREEMENTS_SHOWN = 10
_MAX_TEXT_SHOWN = 80  # characters of a suspect's text in its __str__


@dataclass(frozen=True)
class ClassStats:
    """Out-of-fold statistics for one LLM label."""

    label: str
    support: int  # texts the LLM gave this label
    coverage: float  # share of them the student answers
    agreement: float | None  # agreement with the LLM on the answered ones; None if none


@dataclass(frozen=True)
class Report:
    """How much traffic the student can take over, and how well it agrees with the LLM.

    All rates are out-of-fold estimates on the valid training data: each text is scored by a
    model that did not see it, as new traffic would be.
    """

    n_total: int
    invalid: dict[InvalidReason, int]  # excluded LLM labels, count per reason
    n_used: int
    labels: tuple[str, ...]
    threshold: float  # math.inf if the target is unreachable
    target_agreement: float
    confidence_level: float
    target_reached: bool
    n_covered: int  # texts the student answers
    coverage: float  # n_covered / n_used: also the share of LLM calls saved
    agreement_covered: float | None  # point estimate on covered texts; None if none covered
    agreement_covered_lower: float  # one-sided Wilson lower bound at confidence_level
    per_class: tuple[ClassStats, ...]
    disagreements: tuple[tuple[str, str, int], ...]  # (LLM label, student label, count)
    student_latency_ms: float  # median per-text latency of a single predict() call

    @property
    def n_invalid(self) -> int:
        return sum(self.invalid.values())

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-ready data: lists instead of tuples, ``None`` for an infinite threshold."""
        return {
            **dataclasses.asdict(self),
            "n_invalid": self.n_invalid,
            "labels": list(self.labels),
            "threshold": None if math.isinf(self.threshold) else self.threshold,
            "per_class": [dataclasses.asdict(stats) for stats in self.per_class],
            "disagreements": [
                {"llm_label": llm, "student_label": student, "count": count}
                for llm, student, count in self.disagreements
            ],
        }

    def __str__(self) -> str:
        invalid = f" ({format_invalid_counts(self.invalid)})" if self.invalid else ""
        target = (
            f"agreement >= {self.target_agreement:.1%} at {self.confidence_level:.0%} confidence"
        )
        lines = [
            "bounded-distill report",
            f"  texts:      {self.n_total:,} total, {self.n_used:,} used, "
            f"{self.n_invalid:,} invalid{invalid}",
        ]
        if not self.target_reached:
            lines += [
                f"  target:     {target}: NOT reached, the router always returns None",
                "              (try a lower target, more data, or cleaner labels)",
            ]
        else:
            lines += [
                f"  target:     {target}: reached",
                f"  threshold:  {self.threshold:.4f}",
                f"  coverage:   {self.coverage:.1%} "
                f"({self.n_covered:,} of {self.n_used:,} texts answered without the LLM)",
                f"  agreement:  {_pct(self.agreement_covered)} on answered texts, "
                f"lower bound {self.agreement_covered_lower:.1%}",
            ]
        lines.append(f"  latency:    {self.student_latency_ms:.2f} ms per text (median, student)")
        lines += ["", *self._per_class_table()]
        if self.disagreements:
            lines += ["", *self._disagreement_lines()]
        return "\n".join(lines)

    def _per_class_table(self) -> list[str]:
        width = max(len("per class"), *(len(stats.label) for stats in self.per_class))
        rows = [f"  {'per class':<{width}}  support  coverage  agreement"]
        rows += [
            f"  {s.label:<{width}}  {s.support:>7,}  {s.coverage:>8.1%}  {_pct(s.agreement):>9}"
            for s in self.per_class
        ]
        return rows

    def _disagreement_lines(self) -> list[str]:
        shown = self.disagreements[:_MAX_DISAGREEMENTS_SHOWN]
        width = max(len(f"{llm} -> {student}") for llm, student, _ in shown)
        lines = ["  confident disagreements (LLM label -> student label)"]
        lines += [
            f"  {f'{llm} -> {student}':<{width}}  {count:>5,}" for llm, student, count in shown
        ]
        if len(self.disagreements) > len(shown):
            lines.append(f"  ... and {len(self.disagreements) - len(shown)} more pairs")
        return lines


@dataclass(frozen=True)
class Suspect:
    """A training text the student labels differently from the LLM, with high confidence.

    Often an LLM labeling mistake or a sign of an ambiguous class definition: worth a human
    look. The library never relabels data itself.
    """

    index: int  # position in the texts given to distill(), counting invalid ones too
    text: str
    llm_label: str
    student_label: str
    confidence: float  # out-of-fold probability of student_label

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-ready data."""
        return dataclasses.asdict(self)

    def __str__(self) -> str:
        text = self.text
        if len(text) > _MAX_TEXT_SHOWN:
            text = text[: _MAX_TEXT_SHOWN - 3] + "..."
        return (
            f"#{self.index}  {self.llm_label} -> {self.student_label} "
            f"({self.confidence:.1%}): {text!r}"
        )


@dataclass(frozen=True)
class GoldReport:
    """Accuracy against human (gold) labels on texts the student was not trained on.

    Agreement with the LLM is not accuracy: this report checks what the cascade really gets
    right. The student answers only the texts it is confident about, which are usually the
    easier ones, so its accuracy is also compared with the LLM's on those same texts.
    """

    n_texts: int
    n_covered: int  # texts the student answers
    coverage: float
    llm_accuracy: float  # the LLM alone, on all texts
    cascade_accuracy: float  # student when confident, otherwise the LLM label
    student_accuracy_covered: float | None  # None if the student answers nothing
    llm_accuracy_covered: float | None  # the LLM on the texts the student answers
    agreement_covered: float | None  # student vs LLM on those texts: checks the bound

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-ready data."""
        return dataclasses.asdict(self)

    def __str__(self) -> str:
        return "\n".join(
            [
                f"bounded-distill evaluation on {self.n_texts:,} gold-labeled texts",
                f"  coverage:  {self.coverage:.1%} "
                f"({self.n_covered:,} of {self.n_texts:,} texts answered by the student)",
                "",
                "  accuracy vs gold         all texts  answered by student",
                f"  LLM alone                {self.llm_accuracy:>9.1%}  "
                f"{_pct(self.llm_accuracy_covered):>19}",
                f"  student                  {'':>9}  {_pct(self.student_accuracy_covered):>19}",
                f"  cascade (student + LLM)  {self.cascade_accuracy:>9.1%}",
                "",
                f"  agreement with the LLM on answered texts: {_pct(self.agreement_covered)}",
            ]
        )


def format_invalid_counts(invalid: dict[InvalidReason, int]) -> str:
    """Counts per reason in a fixed order, e.g. '3 missing (None), 2 empty'."""
    return ", ".join(
        f"{invalid[reason]:,} {text}" for reason, text in _REASON_TEXT.items() if reason in invalid
    )


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def build_report(
    training: TrainingData,
    oof_proba: npt.NDArray[np.float64],
    *,
    threshold: float,
    target_agreement: float,
    confidence_level: float,
    student_latency_ms: float,
) -> Report:
    """Assemble a Report from out-of-fold probabilities; pure apart from its inputs."""
    labels = training.label_set
    student = [labels[int(i)] for i in oof_proba.argmax(axis=1)]
    covered = [bool(c >= threshold) for c in oof_proba.max(axis=1)]
    llm = training.labels

    n_used = len(llm)
    n_covered = sum(covered)
    n_agree = sum(c and s == t for c, s, t in zip(covered, student, llm, strict=True))
    coverage = n_covered / n_used

    disagreement_counts = Counter(
        (t, s) for c, s, t in zip(covered, student, llm, strict=True) if c and s != t
    )
    disagreements = tuple(
        (t, s, n)
        for (t, s), n in sorted(disagreement_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    )

    return Report(
        n_total=training.n_total,
        invalid=dict(training.invalid),
        n_used=n_used,
        labels=labels,
        threshold=threshold,
        target_agreement=target_agreement,
        confidence_level=confidence_level,
        target_reached=n_covered > 0,
        n_covered=n_covered,
        coverage=coverage,
        agreement_covered=n_agree / n_covered if n_covered else None,
        agreement_covered_lower=wilson_lower_bound(n_agree, n_covered, confidence_level),
        per_class=tuple(_class_stats(label, covered, student, llm) for label in labels),
        disagreements=disagreements,
        student_latency_ms=student_latency_ms,
    )


def build_gold_report(
    student: Sequence[str | None], gold: Sequence[str], llm: Sequence[str]
) -> GoldReport:
    """Compare the student (``None`` = not confident), the LLM and the cascade with gold."""
    n_texts = len(gold)
    if n_texts == 0:
        raise ValueError("need at least one gold-labeled text to evaluate")
    rows = list(zip(student, gold, llm, strict=True))
    covered = [(s, g, t) for s, g, t in rows if s is not None]
    n_covered = len(covered)

    def share(hits: int) -> float | None:
        return hits / n_covered if n_covered else None

    return GoldReport(
        n_texts=n_texts,
        n_covered=n_covered,
        coverage=n_covered / n_texts,
        llm_accuracy=sum(t == g for _, g, t in rows) / n_texts,
        cascade_accuracy=sum((t if s is None else s) == g for s, g, t in rows) / n_texts,
        student_accuracy_covered=share(sum(s == g for s, g, _ in covered)),
        llm_accuracy_covered=share(sum(t == g for _, g, t in covered)),
        agreement_covered=share(sum(s == t for s, _, t in covered)),
    )


def build_suspects(
    training: TrainingData, oof_proba: npt.NDArray[np.float64], *, top_n: int
) -> list[Suspect]:
    """Out-of-fold disagreements with the LLM, most confident first (ties: original order)."""
    if top_n < 1:
        raise ValueError(f"top_n must be at least 1, got {top_n}")
    labels = training.label_set
    suspects = [
        Suspect(
            index=index,
            text=text,
            llm_label=llm,
            student_label=labels[int(best)],
            confidence=float(confidence),
        )
        for index, text, llm, best, confidence in zip(
            training.indices,
            training.texts,
            training.labels,
            oof_proba.argmax(axis=1),
            oof_proba.max(axis=1),
            strict=True,
        )
        if labels[int(best)] != llm
    ]
    suspects.sort(key=lambda s: -s.confidence)  # stable: ties keep the original order
    return suspects[:top_n]


def _class_stats(
    label: str, covered: Sequence[bool], student: Sequence[str], llm: Sequence[str]
) -> ClassStats:
    rows = [(c, s) for c, s, t in zip(covered, student, llm, strict=True) if t == label]
    answered = [s for c, s in rows if c]
    return ClassStats(
        label=label,
        support=len(rows),
        coverage=len(answered) / len(rows),
        agreement=sum(s == label for s in answered) / len(answered) if answered else None,
    )


def median_latency_ms(predict: Callable[[str], object], texts: Sequence[str]) -> float:
    """Median wall-clock time of single ``predict`` calls, in milliseconds.

    The first call is a warm-up and is not timed: it may load the embedding model.
    """
    if not texts:
        raise ValueError("need at least one text to measure latency")
    predict(texts[0])
    timings: list[float] = []
    for text in texts:
        start = time.perf_counter()
        predict(text)
        timings.append((time.perf_counter() - start) * 1000)
    return statistics.median(timings)
