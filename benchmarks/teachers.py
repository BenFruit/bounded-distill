"""Compare teachers: how often each LLM is wrong, and whether the teachers share their mistakes.

For each teacher with complete labels, on the train split: its error rate against gold, and
the share of its errors that are systematic (the out-of-fold student repeats them, so no
suspect points at them). Then, for its errors, how many the other teachers make too. An error
that every teacher makes with the same wrong label is more likely an ambiguous text or a gold
label mistake than a weakness of one model.
Run: ``uv run python -m benchmarks.teachers --dataset banking77 --embedder giga-embeddings:10b``
"""

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from benchmarks.data import DATASETS, Example
from benchmarks.label import TEACHERS, cache_path, read_cache
from benchmarks.run import RESULTS_DIR, embeddings, llm_labels, lookup_embed
from benchmarks.suspects import all_suspects, llm_errors
from bounded_distill import EmbedFn
from bounded_distill._embed import DEFAULT_MODEL


@dataclass(frozen=True)
class TeacherErrors:
    """One teacher's errors on the train split, and how many of them other teachers share."""

    teacher: str
    error_rate: float
    systematic: float  # share of its errors that the student repeats
    shared_by_all: float  # share of its errors that every other teacher also makes
    same_label_by_all: float  # ... with exactly the same wrong label


def complete_teachers(dataset: str, examples: Sequence[Example]) -> list[str]:
    """The teachers whose cache has a label for every example."""
    ids = {example.id for example in examples}
    return [name for name in TEACHERS if ids <= read_cache(cache_path(name, dataset)).keys()]


def teacher_errors(
    teacher: str,
    train: Sequence[Example],
    labels: Mapping[str, Sequence[str | None]],
    embed: EmbedFn,
) -> TeacherErrors:
    """Error rate, systematic share and overlap with the other teachers for one teacher."""
    own = labels[teacher]
    errors = llm_errors(train, own)
    caught = errors & {s.index for s in all_suspects(train, own, embed)}
    others = [labels[name] for name in labels if name != teacher]
    shared = {i for i in errors if all(other[i] != train[i].gold for other in others)}
    same = {i for i in shared if all(other[i] == own[i] for other in others)}
    n = max(len(errors), 1)
    return TeacherErrors(
        teacher=teacher,
        error_rate=len(errors) / len(train),
        systematic=1 - len(caught) / n,
        shared_by_all=len(shared) / n,
        same_label_by_all=len(same) / n,
    )


def table(rows: Sequence[TeacherErrors]) -> str:
    """One line per teacher."""
    lines = ["teacher        errors  systematic  shared by all  same label by all"]
    for r in rows:
        lines.append(
            f"{r.teacher:<13}  {r.error_rate:>6.1%}  {r.systematic:>10.1%}  "
            f"{r.shared_by_all:>13.1%}  {r.same_label_by_all:>17.1%}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the teachers' labeling errors.")
    parser.add_argument("--dataset", choices=list(DATASETS), default="banking77")
    parser.add_argument("--embedder", default=DEFAULT_MODEL)
    args = parser.parse_args()

    dataset = DATASETS[args.dataset]
    train = dataset.load("train")
    teachers = complete_teachers(dataset.name, train)
    if len(teachers) < 2:
        raise SystemExit(f"need at least two fully labeled teachers, have {teachers}")
    labels = {name: llm_labels(name, dataset.name, train) for name in teachers}
    everything = [*train, *dataset.load("test")]
    embed = lookup_embed(everything, embeddings(dataset.name, everything, args.embedder))

    rows = [teacher_errors(name, train, labels, embed) for name in teachers]
    print(table(rows))
    embedder = args.embedder.replace(":", "_").replace("/", "_")
    out = RESULTS_DIR / f"teachers_{dataset.name}_{embedder}.json"
    out.write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
