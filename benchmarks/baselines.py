"""Compare ways to pick the confidence threshold, on the same student and the same data.

Every method sees the out-of-fold confidences of one distilled router and picks a threshold;
the router is then evaluated on the test split with that threshold. Methods:

- ``fixed 0.5`` and ``fixed 0.9``: a hand-picked confidence, the same for every target;
- ``point``: the lowest threshold whose out-of-fold agreement reaches the target, no bound;
- ``SGR`` and ``wilson``: the library's two methods, ``threshold_method="sgr"`` (Selection
  with Guaranteed Risk, Geifman and El-Yaniv, 2017) and ``threshold_method="wilson"``.

For each method and target it reports the mean test coverage, how often the test agreement
was below the target, and how often significantly so (see ``bound_check``).
Run: ``uv run python -m benchmarks.baselines``
"""

import json
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace

from benchmarks.bound_check import CONFIDENCE, upper_bound
from benchmarks.data import DATASETS
from benchmarks.label import TEACHERS
from benchmarks.run import RESULTS_DIR, TARGETS, embeddings, llm_labels, lookup_embed
from bounded_distill import Router, distill
from bounded_distill._embed import DEFAULT_MODEL
from bounded_distill._threshold import select_threshold, select_threshold_sgr

EMBEDDERS = (DEFAULT_MODEL, "giga-embeddings:10b")

Selector = Callable[[Sequence[float], Sequence[bool], float], float]


def point_threshold(confidences: Sequence[float], agrees: Sequence[bool], target: float) -> float:
    """Lowest threshold whose covered out-of-fold agreement is at least ``target``, no bound."""
    items = sorted(zip(confidences, agrees, strict=True), reverse=True)
    best, n_agree = math.inf, 0
    for i, (confidence, agree) in enumerate(items):
        n_agree += agree
        last_of_tie = i + 1 == len(items) or items[i + 1][0] < confidence
        if last_of_tie and n_agree / (i + 1) >= target:
            best = confidence
    return best


def sgr_threshold(confidences: Sequence[float], agrees: Sequence[bool], target: float) -> float:
    return select_threshold_sgr(confidences, agrees, target, CONFIDENCE)


def wilson_threshold(confidences: Sequence[float], agrees: Sequence[bool], target: float) -> float:
    return select_threshold(confidences, agrees, target, CONFIDENCE)


METHODS: dict[str, Selector] = {
    "fixed 0.5": lambda _c, _a, _t: 0.5,
    "fixed 0.9": lambda _c, _a, _t: 0.9,
    "point": point_threshold,
    "SGR": sgr_threshold,
    "wilson": wilson_threshold,
}


@dataclass(frozen=True)
class Outcome:
    """One method at one target on one run, measured on the test split."""

    method: str
    run: str
    target: float
    coverage: float
    agreement: float | None
    below: bool
    violated: bool
    cascade_accuracy: float


def oof_confidences(router: Router) -> tuple[list[float], list[bool]]:
    """The out-of-fold confidence of each training text and whether the student agreed."""
    assert router.training is not None
    assert router.oof_proba is not None
    best = router.oof_proba.argmax(axis=1)
    student = [router.labels[int(i)] for i in best]
    agrees = [s == llm for s, llm in zip(student, router.training.labels, strict=True)]
    return [float(c) for c in router.oof_proba.max(axis=1)], agrees


def outcomes(dataset_name: str, teacher: str, embedder: str) -> list[Outcome]:
    """Distill once, then try every method at every target."""
    dataset = DATASETS[dataset_name]
    train, test = dataset.load("train"), dataset.load("test")
    everything = [*train, *test]
    embed = lookup_embed(everything, embeddings(dataset.name, everything, embedder))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        router = distill(
            [e.text for e in train], llm_labels(teacher, dataset.name, train), embed=embed
        )
    confidences, agrees = oof_confidences(router)
    test_texts = [e.text for e in test]
    test_gold = [e.gold for e in test]
    test_llm = [label or "" for label in llm_labels(teacher, dataset.name, test)]
    run = f"{dataset.name} {teacher} {embedder}"

    out: list[Outcome] = []
    for target in TARGETS:
        for method, select in METHODS.items():
            threshold = select(confidences, agrees, target)
            report = replace(router, threshold=threshold).evaluate(test_texts, test_gold, test_llm)
            agreement = report.agreement_covered
            n = report.n_covered
            k = round((agreement or 0) * n)
            out.append(
                Outcome(
                    method=method,
                    run=run,
                    target=target,
                    coverage=report.coverage,
                    agreement=agreement,
                    below=agreement is not None and agreement < target,
                    violated=n > 0 and upper_bound(k, n, CONFIDENCE) < target,
                    cascade_accuracy=report.cascade_accuracy,
                )
            )
    return out


def table(rows: Sequence[Outcome]) -> str:
    """Per method and target: mean coverage, and how many runs missed the target."""
    n_runs = len({r.run for r in rows})
    lines = [
        f"{n_runs} runs; 'missed' = test agreement below the target, "
        "'broken' = significantly below",
        "target  method      coverage  missed  broken",
    ]
    for target in TARGETS:
        for method in METHODS:
            group = [r for r in rows if r.target == target and r.method == method]
            coverage = sum(r.coverage for r in group) / len(group)
            missed = sum(r.below for r in group)
            broken = sum(r.violated for r in group)
            lines.append(
                f"{target:.2f}    {method:<10}  {coverage:>8.1%}  {missed:>3}/{len(group)}  "
                f"{broken:>3}/{len(group)}"
            )
    return "\n".join(lines)


def main() -> None:
    rows: list[Outcome] = []
    for dataset in DATASETS:
        for teacher in TEACHERS:
            for embedder in EMBEDDERS:
                rows.extend(outcomes(dataset, teacher, embedder))
                print(f"done: {dataset} {teacher} {embedder}", flush=True)
    print(table(rows))
    out = RESULTS_DIR / "baselines.json"
    out.write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
