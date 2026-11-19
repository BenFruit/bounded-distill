import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
from synthetic import fake_embed, make_dataset

from bounded_distill._distill import distill
from bounded_distill._report import ClassStats, Report, build_report, median_latency_ms
from bounded_distill._router import Router
from bounded_distill._threshold import wilson_lower_bound
from bounded_distill._types import TrainingData

# Six texts, worked out by hand. With threshold 0.6, rows 0, 1, 2, 4, 5 are covered;
# rows 0, 2, 4 agree with the LLM; row 1 is a confident a -> b, row 5 a confident c -> a.
TRAINING = TrainingData(
    texts=[f"t{i}" for i in range(6)],
    labels=["a", "a", "b", "b", "c", "c"],
    indices=[0, 1, 2, 4, 5, 6],
    label_set=("a", "b", "c"),
    n_total=7,
    invalid={"missing": 1},
)
OOF = np.array(
    [
        [0.90, 0.05, 0.05],  # a, conf 0.90, agrees
        [0.20, 0.70, 0.10],  # b, conf 0.70, LLM said a
        [0.10, 0.85, 0.05],  # b, conf 0.85, agrees
        [0.40, 0.35, 0.25],  # a, conf 0.40 (not covered), LLM said b
        [0.05, 0.15, 0.80],  # c, conf 0.80, agrees
        [0.60, 0.10, 0.30],  # a, conf 0.60 (exactly the threshold), LLM said c
    ]
)


def _build(threshold: float = 0.6) -> Report:
    return build_report(
        TRAINING,
        OOF,
        threshold=threshold,
        target_agreement=0.5,
        confidence_level=0.9,
        student_latency_ms=1.5,
    )


def test_build_report_counts() -> None:
    report = _build()

    assert (report.n_total, report.n_invalid, report.n_used) == (7, 1, 6)
    assert report.invalid == {"missing": 1}
    assert report.labels == ("a", "b", "c")
    assert (report.n_covered, report.coverage) == (5, 5 / 6)
    assert report.agreement_covered == 3 / 5
    assert report.agreement_covered_lower == wilson_lower_bound(3, 5, 0.9)
    assert report.target_reached
    assert report.student_latency_ms == 1.5


def test_build_report_per_class() -> None:
    assert _build().per_class == (
        ClassStats(label="a", support=2, coverage=1.0, agreement=0.5),
        ClassStats(label="b", support=2, coverage=0.5, agreement=1.0),
        ClassStats(label="c", support=2, coverage=1.0, agreement=0.5),
    )


def test_build_report_lists_confident_disagreements_only() -> None:
    # Row 3 also disagrees (b -> a) but is below the threshold, so it is not listed.
    assert _build().disagreements == (("a", "b", 1), ("c", "a", 1))


def test_build_report_sorts_disagreements_by_count() -> None:
    training = TrainingData(
        texts=["x"] * 4,
        labels=["a", "b", "b", "b"],
        indices=[0, 1, 2, 3],
        label_set=("a", "b"),
        n_total=4,
        invalid={},
    )
    oof = np.array([[0.1, 0.9], [0.9, 0.1], [0.9, 0.1], [0.2, 0.8]])
    report = build_report(
        training,
        oof,
        threshold=0.5,
        target_agreement=0.5,
        confidence_level=0.9,
        student_latency_ms=1.0,
    )
    assert report.disagreements == (("b", "a", 2), ("a", "b", 1))


def test_build_report_with_unreachable_target() -> None:
    report = _build(threshold=math.inf)

    assert not report.target_reached
    assert (report.n_covered, report.coverage) == (0, 0.0)
    assert report.agreement_covered is None
    assert report.agreement_covered_lower == 0.0
    assert all(c.coverage == 0.0 and c.agreement is None for c in report.per_class)
    assert report.disagreements == ()


def test_median_latency_warms_up_then_times_each_text() -> None:
    calls: list[str] = []
    latency = median_latency_ms(calls.append, ["a", "b", "c"])

    assert calls == ["a", "a", "b", "c"]  # one untimed warm-up call first
    assert latency >= 0.0


def test_median_latency_needs_texts() -> None:
    with pytest.raises(ValueError, match="at least one text"):
        median_latency_ms(lambda text: None, [])


def test_router_report_matches_its_out_of_fold_data() -> None:
    texts, labels = make_dataset(300, flip_rate=0.01)
    router = distill(texts, labels, embed=fake_embed, target_agreement=0.95)
    report = router.report()

    assert router.oof_proba is not None
    expected_coverage = float((router.oof_proba.max(axis=1) >= router.threshold).mean())
    assert report.coverage == pytest.approx(expected_coverage)
    assert report.target_reached
    assert report.agreement_covered_lower >= 0.95
    assert report.student_latency_ms > 0
    assert sum(c.support for c in report.per_class) == report.n_used == 900


def test_loaded_router_cannot_report(tmp_path: Path) -> None:
    texts, labels = make_dataset(100, flip_rate=0.0)
    distill(texts, labels, embed=fake_embed, target_agreement=0.9).save(tmp_path)
    loaded = Router.load(tmp_path, embed=fake_embed)

    with pytest.raises(ValueError, match="report\\(\\) needs the training data"):
        loaded.report()


def test_str_shows_a_readable_summary() -> None:
    assert str(_build()) == "\n".join(
        [
            "bounded-distill report",
            "  texts:      7 total, 6 used, 1 invalid (1 missing (None))",
            "  target:     agreement >= 50.0% at 90% confidence: reached",
            "  threshold:  0.6000",
            "  coverage:   83.3% (5 of 6 texts answered without the LLM)",
            "  agreement:  60.0% on answered texts, lower bound 33.0%",
            "  latency:    1.50 ms per text (median, student)",
            "",
            "  per class  support  coverage  agreement",
            "  a                2    100.0%      50.0%",
            "  b                2     50.0%     100.0%",
            "  c                2    100.0%      50.0%",
            "",
            "  confident disagreements (LLM label -> student label)",
            "  a -> b      1",
            "  c -> a      1",
        ]
    )


def test_str_explains_an_unreachable_target() -> None:
    text = str(_build(threshold=math.inf))
    assert "NOT reached, the router always returns None" in text
    assert "threshold:" not in text
    assert "  a                2      0.0%        n/a" in text
    assert "confident disagreements" not in text


def test_str_caps_the_disagreement_list() -> None:
    many = tuple((f"l{i}", "x", 20 - i) for i in range(12))
    text = str(dataclasses.replace(_build(), disagreements=many))
    assert "l9 -> x" in text
    assert "l10 -> x" not in text
    assert "... and 2 more pairs" in text


def test_to_dict_is_strict_json() -> None:
    data = _build(threshold=math.inf).to_dict()
    restored = json.loads(json.dumps(data, allow_nan=False))  # raises on inf or NaN

    assert restored["threshold"] is None
    assert restored["n_invalid"] == 1
    assert restored["per_class"][0] == {
        "label": "a",
        "support": 2,
        "coverage": 0.0,
        "agreement": None,
    }


def test_to_dict_names_disagreement_fields() -> None:
    assert _build().to_dict()["disagreements"] == [
        {"llm_label": "a", "student_label": "b", "count": 1},
        {"llm_label": "c", "student_label": "a", "count": 1},
    ]
