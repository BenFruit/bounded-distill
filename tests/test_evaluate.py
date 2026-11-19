import json
from pathlib import Path

import pytest
from synthetic import CLASSES, fake_embed, make_dataset

from bounded_distill._distill import distill
from bounded_distill._report import GoldReport, build_gold_report
from bounded_distill._router import Router

# Five texts, worked out by hand. The student answers rows 0, 2, 3 and is right on all of
# them; the LLM is wrong on rows 1 and 3, and row 3 is the one the cascade fixes.
STUDENT = ["a", None, "b", "b", None]
GOLD = ["a", "a", "b", "b", "b"]
LLM = ["a", "b", "b", "a", "b"]


def test_build_gold_report_counts() -> None:
    assert build_gold_report(STUDENT, GOLD, LLM) == GoldReport(
        n_texts=5,
        n_covered=3,
        coverage=0.6,
        llm_accuracy=3 / 5,
        cascade_accuracy=4 / 5,
        student_accuracy_covered=1.0,
        llm_accuracy_covered=2 / 3,
        agreement_covered=2 / 3,
    )


def test_build_gold_report_without_coverage() -> None:
    report = build_gold_report([None, None], ["a", "b"], ["a", "a"])

    assert (report.coverage, report.llm_accuracy, report.cascade_accuracy) == (0.0, 0.5, 0.5)
    assert report.student_accuracy_covered is None
    assert report.llm_accuracy_covered is None
    assert report.agreement_covered is None


def test_build_gold_report_needs_texts() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_gold_report([], [], [])


def test_gold_report_str() -> None:
    assert str(build_gold_report(STUDENT, GOLD, LLM)) == "\n".join(
        [
            "bounded-distill evaluation on 5 gold-labeled texts",
            "  coverage:  60.0% (3 of 5 texts answered by the student)",
            "",
            "  accuracy vs gold         all texts  answered by student",
            "  LLM alone                    60.0%                66.7%",
            "  student                                          100.0%",
            "  cascade (student + LLM)      80.0%",
            "",
            "  agreement with the LLM on answered texts: 66.7%",
        ]
    )


def test_gold_report_to_dict_is_strict_json() -> None:
    data = build_gold_report([None], ["a"], ["a"]).to_dict()
    restored = json.loads(json.dumps(data, allow_nan=False))
    assert restored["student_accuracy_covered"] is None
    assert restored["cascade_accuracy"] == 1.0


def _held_out(n_per_class: int) -> tuple[list[str], list[str]]:
    """Texts unseen in training (ids from 1000 up) with their true class as gold."""
    texts = [f"{c} {i}" for c in CLASSES for i in range(1000, 1000 + n_per_class)]
    return texts, [text.split()[0] for text in texts]


def test_evaluate_on_held_out_texts() -> None:
    train_texts, train_labels = make_dataset(300, flip_rate=0.01)
    router = distill(train_texts, train_labels, embed=fake_embed, target_agreement=0.95)
    texts, gold = _held_out(100)
    llm = list(gold)
    llm[0] = "tech"  # one LLM mistake on a 'billing' text

    report = router.evaluate(texts, gold, llm)

    assert report.n_texts == 300
    assert report.llm_accuracy == pytest.approx(299 / 300)
    assert 0 < report.n_covered < 300
    assert report.coverage == pytest.approx(report.n_covered / 300)
    assert report.agreement_covered is not None
    assert report.agreement_covered > 0.9


def test_evaluate_works_on_a_loaded_router(tmp_path: Path) -> None:
    train_texts, train_labels = make_dataset(100, flip_rate=0.0)
    router = distill(train_texts, train_labels, embed=fake_embed, target_agreement=0.9)
    router.save(tmp_path)
    loaded = Router.load(tmp_path, embed=fake_embed)
    texts, gold = _held_out(20)

    assert loaded.evaluate(texts, gold, gold) == router.evaluate(texts, gold, gold)


def test_evaluate_rejects_different_lengths() -> None:
    train_texts, train_labels = make_dataset(100, flip_rate=0.0)
    router = distill(train_texts, train_labels, embed=fake_embed, target_agreement=0.9)

    with pytest.raises(ValueError, match="differ in length: 2, 2, 1"):
        router.evaluate(["billing 1", "tech 1"], ["billing", "tech"], ["billing"])
