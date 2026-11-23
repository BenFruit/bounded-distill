"""Blind audit of the errors that all teachers share: is the gold label or the LLMs' label right?

The population is the train texts where every teacher chose the same label and it differs
from gold. For a random sample, an annotator sees the text, the list of intents, and the two
labels as options A and B in random order, without knowing which is which. The answer is
``A``, ``B``, ``both`` (both fit) or ``neither``. Two annotators work independently; the
script reports each one's verdicts mapped back to gold or teacher, their agreement, and
Cohen's kappa.

    uv run python -m benchmarks.audit sample --dataset banking77   # writes the blind items
    uv run python -m benchmarks.audit score --dataset banking77    # reads the annotations

Items and annotations live in ``benchmarks/.cache/audit/`` (they contain texts). The scores,
with ids and answers but no texts, go to ``benchmarks/results/``.
"""

import argparse
import json
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

from benchmarks.data import CACHE_DIR, DATASETS
from benchmarks.label import TEACHERS
from benchmarks.run import RESULTS_DIR, llm_labels

SAMPLE_SIZE = 150
AUDIT_DIR = CACHE_DIR / "audit"
ANNOTATORS = ("annotator1", "annotator2")

Choice = Literal["A", "B", "both", "neither"]
Verdict = Literal["gold", "teacher", "both", "neither"]
VERDICTS: tuple[Verdict, ...] = ("gold", "teacher", "both", "neither")


@dataclass(frozen=True)
class Item:
    """One blind question; the key (which option is gold) is stored separately."""

    id: str
    text: str
    a: str
    b: str


@dataclass(frozen=True)
class Key:
    id: str
    gold: str
    teacher: str
    gold_is_a: bool


def shared_errors(dataset: str) -> list[tuple[str, str, str, str]]:
    """``(id, text, gold, teacher label)`` where all teachers agree on one wrong label."""
    train = DATASETS[dataset].load("train")
    labels = [llm_labels(name, dataset, train) for name in TEACHERS]
    out: list[tuple[str, str, str, str]] = []
    for i, example in enumerate(train):
        chosen = {teacher[i] for teacher in labels}
        if len(chosen) == 1:
            label = chosen.pop()
            if label is not None and label != example.gold:
                out.append((example.id, example.text, example.gold, label))
    return out


def make_items(
    errors: Sequence[tuple[str, str, str, str]], size: int, seed: int = 0
) -> tuple[list[Item], list[Key]]:
    """A random sample, each with gold and teacher label in random order."""
    rng = random.Random(seed)
    sample = rng.sample(list(errors), min(size, len(errors)))
    items: list[Item] = []
    keys: list[Key] = []
    for id_, text, gold, teacher in sample:
        gold_is_a = rng.random() < 0.5
        a, b = (gold, teacher) if gold_is_a else (teacher, gold)
        items.append(Item(id_, text, a, b))
        keys.append(Key(id_, gold, teacher, gold_is_a))
    return items, keys


def verdict(choice: Choice, gold_is_a: bool) -> Verdict:
    """Map a blind answer back: which label the annotator found right."""
    if choice in ("both", "neither"):
        return choice
    return "gold" if (choice == "A") == gold_is_a else "teacher"


def cohen_kappa(first: Sequence[str], second: Sequence[str]) -> float:
    """Agreement between two annotators beyond chance; 1 is perfect, 0 is chance level."""
    n = len(first)
    observed = sum(x == y for x, y in zip(first, second, strict=True)) / n
    counts_1, counts_2 = Counter(first), Counter(second)
    expected = sum(counts_1[c] * counts_2[c] for c in counts_1) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


@dataclass(frozen=True)
class AuditResult:
    dataset: str
    population: int  # train texts where all teachers share one wrong label
    n: int
    shares: dict[str, dict[str, float]]  # annotator -> verdict -> share
    consensus: dict[str, float]  # verdict -> share of items both annotators gave it
    agreement: float
    kappa: float
    items: list[dict[str, str]]  # id, gold, teacher and each annotator's verdict


def read_jsonl(path: Path) -> list[dict[str, str]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def score(dataset: str, population: int, keys: Sequence[Key]) -> AuditResult:
    """Combine the two annotators' blind answers with the key."""
    by_id = {key.id: key for key in keys}
    verdicts: dict[str, list[Verdict]] = {}
    for name in ANNOTATORS:
        answers = {
            row["id"]: row["choice"] for row in read_jsonl(AUDIT_DIR / f"{dataset}_{name}.jsonl")
        }
        missing = by_id.keys() - answers.keys()
        if missing:
            raise SystemExit(
                f"{name} did not answer {len(missing)} items, e.g. {next(iter(missing))}"
            )
        bad = {c for c in answers.values() if c not in ("A", "B", "both", "neither")}
        if bad:
            raise SystemExit(f"{name} gave invalid answers: {bad}")
        verdicts[name] = [verdict(cast(Choice, answers[k.id]), k.gold_is_a) for k in keys]

    first, second = (verdicts[name] for name in ANNOTATORS)
    n = len(keys)
    shares = {name: {v: verdicts[name].count(v) / n for v in VERDICTS} for name in ANNOTATORS}
    consensus = {
        v: sum(x == y == v for x, y in zip(first, second, strict=True)) / n for v in VERDICTS
    }
    items = [
        {
            "id": k.id,
            "gold": k.gold,
            "teacher": k.teacher,
            **{name: verdicts[name][i] for name in ANNOTATORS},
        }
        for i, k in enumerate(keys)
    ]
    return AuditResult(
        dataset=dataset,
        population=population,
        n=n,
        shares=shares,
        consensus=consensus,
        agreement=sum(x == y for x, y in zip(first, second, strict=True)) / n,
        kappa=cohen_kappa(first, second),
        items=items,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Blind audit of errors shared by all teachers.")
    parser.add_argument("step", choices=["sample", "score"])
    parser.add_argument("--dataset", choices=list(DATASETS), default="banking77")
    args = parser.parse_args()
    dataset: str = args.dataset

    errors = shared_errors(dataset)
    items, keys = make_items(errors, SAMPLE_SIZE)
    if args.step == "sample":
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(asdict(item), ensure_ascii=False) for item in items]
        (AUDIT_DIR / f"{dataset}_items.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        labels = "\n".join(DATASETS[dataset].labels())
        (AUDIT_DIR / f"{dataset}_labels.txt").write_text(labels + "\n", encoding="utf-8")
        print(f"{len(errors)} shared errors, {len(items)} blind items written to {AUDIT_DIR}")
        return

    result = score(dataset, len(errors), keys)
    print(f"{result.dataset}: {result.n} of {result.population} shared errors")
    print("verdict    " + "  ".join(f"{name:>10}" for name in ANNOTATORS) + "   consensus")
    for v in VERDICTS:
        row = "  ".join(f"{result.shares[name][v]:>10.1%}" for name in ANNOTATORS)
        print(f"{v:<9}  {row}   {result.consensus[v]:>9.1%}")
    print(f"agreement {result.agreement:.1%}, Cohen's kappa {result.kappa:.2f}")
    out = RESULTS_DIR / f"audit_{dataset}.json"
    out.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
