from pathlib import Path

import pytest
from synthetic import fake_embed, make_dataset
from test_report import OOF, TRAINING

from bounded_distill._distill import distill
from bounded_distill._report import Suspect, build_suspects
from bounded_distill._router import Router


def test_build_suspects_lists_disagreements_by_confidence() -> None:
    # Rows 1, 3, 5 of the hand-worked fixture disagree; their original indices are 1, 4, 6.
    # Unlike the report, suspects ignore the threshold: row 3 (confidence 0.40) is listed too.
    assert build_suspects(TRAINING, OOF, top_n=50) == [
        Suspect(index=1, text="t1", llm_label="a", student_label="b", confidence=0.70),
        Suspect(index=6, text="t5", llm_label="c", student_label="a", confidence=0.60),
        Suspect(index=4, text="t3", llm_label="b", student_label="a", confidence=0.40),
    ]


def test_build_suspects_keeps_top_n() -> None:
    assert [s.index for s in build_suspects(TRAINING, OOF, top_n=2)] == [1, 6]


@pytest.mark.parametrize("top_n", [0, -1])
def test_build_suspects_rejects_non_positive_top_n(top_n: int) -> None:
    with pytest.raises(ValueError, match="top_n"):
        build_suspects(TRAINING, OOF, top_n=top_n)


def test_injected_label_flips_rise_to_the_top() -> None:
    texts, labels = make_dataset(300, flip_rate=0.03)
    flipped = {i for i, text in enumerate(texts) if text.split()[0] != labels[i]}
    router = distill(texts, labels, embed=fake_embed, target_agreement=0.9)

    top = router.suspects(top_n=10)
    all_suspects = router.suspects(top_n=10_000)
    # Flips are a small share of all disagreements but most of the top ten.
    assert sum(s.index in flipped for s in all_suspects) / len(all_suspects) < 0.25
    assert sum(s.index in flipped for s in top) >= 7
    assert all(s.text == texts[s.index] and s.llm_label == labels[s.index] for s in top)


def test_suspect_index_counts_invalid_labels() -> None:
    texts, labels = make_dataset(100, flip_rate=0.03)
    noisy: list[str | None] = [None, *labels]
    router = distill(
        ["junk", *texts], noisy, embed=fake_embed, target_agreement=0.8, on_invalid="drop"
    )
    suspects = router.suspects()
    assert suspects
    for s in suspects:
        assert ["junk", *texts][s.index] == s.text


def test_loaded_router_cannot_list_suspects(tmp_path: Path) -> None:
    texts, labels = make_dataset(100, flip_rate=0.0)
    distill(texts, labels, embed=fake_embed, target_agreement=0.9).save(tmp_path)
    loaded = Router.load(tmp_path, embed=fake_embed)

    with pytest.raises(ValueError, match=r"suspects\(\) needs the training data"):
        loaded.suspects()


def test_suspect_str_and_to_dict() -> None:
    suspect = Suspect(index=7, text="x" * 100, llm_label="a", student_label="b", confidence=0.9)

    assert str(suspect) == f"#7  a -> b (90.0%): '{'x' * 77}...'"
    assert suspect.to_dict() == {
        "index": 7,
        "text": "x" * 100,
        "llm_label": "a",
        "student_label": "b",
        "confidence": 0.9,
    }
