"""How many of the LLM's labeling errors on a train split do suspects catch?

An LLM error is a train text whose LLM label differs from gold. Suspects are the texts where
the out-of-fold student disagrees with the LLM. An error is *inconsistent* if the student
disagrees (a suspect: visible without gold labels) and *systematic* if the student repeats it
(invisible). Run: ``uv run python -m benchmarks.suspects [--embedder ...]``
"""

import argparse
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

from benchmarks.data import DATASETS, Dataset, Example
from benchmarks.label import DEFAULT_MODEL as DEFAULT_LLM
from benchmarks.run import embeddings, llm_labels, lookup_embed
from bounded_distill import EmbedFn, Suspect, distill
from bounded_distill._embed import DEFAULT_MODEL

RANKS = (50, 100, 500, 1000)


@dataclass(frozen=True)
class Precision:
    """Among the ``top`` most confident suspects: how often the LLM and the student are right."""

    top: int
    llm_wrong: float  # share where the LLM label differs from gold: a real labeling error
    student_right: float  # share where the student's label equals gold: a correct fix


def precision_at(suspects: Sequence[Suspect], gold: Sequence[str], top: int) -> Precision:
    """Precision of the ``top`` first suspects, judged against gold labels."""
    head = suspects[:top]
    return Precision(
        top=len(head),
        llm_wrong=sum(s.llm_label != gold[s.index] for s in head) / len(head),
        student_right=sum(s.student_label == gold[s.index] for s in head) / len(head),
    )


def all_suspects(
    train: Sequence[Example], train_llm: Sequence[str | None], embed: EmbedFn
) -> list[Suspect]:
    """Every text where the out-of-fold student disagrees with the LLM, most confident first."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        router = distill([e.text for e in train], train_llm, embed=embed)
    return router.suspects(top_n=len(train))  # all of them; the threshold is not used


def llm_errors(train: Sequence[Example], train_llm: Sequence[str | None]) -> set[int]:
    """Positions of the train texts whose LLM label differs from gold."""
    return {i for i, (e, llm) in enumerate(zip(train, train_llm, strict=True)) if llm != e.gold}


def analyse(
    dataset: Dataset, train: Sequence[Example], train_llm: Sequence[str | None], embedder: str
) -> str:
    """Train a router on the LLM labels and compare its suspects with the gold labels."""
    everything = [*train, *dataset.load("test")]
    embed = lookup_embed(everything, embeddings(dataset.name, everything, embedder))
    suspects = all_suspects(train, train_llm, embed)
    gold = [e.gold for e in train]

    errors = llm_errors(train, train_llm)
    caught = errors & {s.index for s in suspects}
    lines = [
        f"embedder: {embedder}",
        f"  LLM errors on train:      {len(errors):,} of {len(train):,} texts "
        f"({len(errors) / len(train):.1%}; the share of LLM errors among random texts)",
        f"  suspects (student != LLM): {len(suspects):,}",
        f"  inconsistent errors (caught by suspects):  {len(caught):,} "
        f"({len(caught) / len(errors):.1%} of LLM errors)",
        f"  systematic errors (student repeats them): {len(errors) - len(caught):,} "
        f"({1 - len(caught) / len(errors):.1%})",
        "  most confident suspects   LLM wrong   student right",
    ]
    for top in (*RANKS, len(suspects)):
        p = precision_at(suspects, gold, top)
        name = "all" if top == len(suspects) else f"top {top:,}"
        lines.append(f"    {name:<22}  {p.llm_wrong:>9.1%}  {p.student_right:>14.1%}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare suspects with gold labels.")
    parser.add_argument("--dataset", choices=list(DATASETS), default="banking77")
    parser.add_argument("--llm", default=DEFAULT_LLM, help="model whose cached labels to use")
    parser.add_argument("--embedder", default=DEFAULT_MODEL)
    args = parser.parse_args()

    dataset = DATASETS[args.dataset]
    train = dataset.load("train")
    print(analyse(dataset, train, llm_labels(args.llm, dataset.name, train), args.embedder))


if __name__ == "__main__":
    main()
