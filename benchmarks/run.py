"""Distill LLM labels on Banking77 train and check the routers on test against gold labels.

For each target agreement, ``distill()`` trains on the LLM's train labels; the router is then
evaluated on the test texts, which it never saw, with both the gold and the LLM's test labels.
Needs the LLM labels from ``benchmarks.label``. Run: ``uv run python -m benchmarks.run``
"""

import argparse
import json
import math
import os
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from openai import OpenAI

from benchmarks.data import CACHE_DIR, Example, load_banking77
from benchmarks.label import DEFAULT_MODEL as DEFAULT_LLM
from benchmarks.label import cache_path, read_cache
from bounded_distill import GoldReport, distill
from bounded_distill._embed import DEFAULT_MODEL, default_embed  # the library's own default

TARGETS = (0.90, 0.93, 0.95, 0.97, 0.99)
RESULTS_DIR = Path(__file__).parent / "results"  # committed: metrics only, no texts

Vectors = npt.NDArray[np.float32]
EmbedLookup = Callable[[Sequence[str]], Vectors]


@dataclass(frozen=True)
class TargetResult:
    """What distill() promised on train (out-of-fold) and what the router did on test."""

    target: float
    threshold: float | None  # None if the target was not reachable
    train_coverage: float
    train_agreement: float | None
    train_agreement_lower: float
    test: GoldReport
    seconds: float


def embeddings(examples: Sequence[Example], embedder: str) -> Vectors:
    """Vectors for ``examples``, computed once and cached on disk.

    ``embedder`` is the library's default model, or the name of an embedding model served by
    the OpenAI-compatible API configured in ``.env``.
    """
    path = CACHE_DIR / "embeddings" / _slug(embedder) / "banking77.npy"
    if path.exists():
        vectors: Vectors = np.load(path)
        if len(vectors) == len(examples):
            return vectors
    texts = [example.text for example in examples]
    vectors = default_embed(texts) if embedder == DEFAULT_MODEL else api_embed(embedder, texts)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)  # pyright: ignore[reportUnknownMemberType]
    return vectors


def api_embed(model: str, texts: Sequence[str], batch_size: int = 64) -> Vectors:
    """Embed ``texts`` as they are (no instruction or prefix) through the embeddings API."""
    client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])
    rows: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        response = client.embeddings.create(
            model=model, input=list(texts[start : start + batch_size])
        )
        rows += [item.embedding for item in response.data]
    return np.array(rows, dtype=np.float32)


def _slug(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def lookup_embed(examples: Sequence[Example], vectors: Vectors) -> EmbedLookup:
    """An embed function that returns precomputed vectors, so every distill() run reuses them."""
    row = {example.text: i for i, example in enumerate(examples)}

    def embed(texts: Sequence[str]) -> Vectors:
        return vectors[[row[text] for text in texts]]

    return embed


def llm_labels(model: str, examples: Sequence[Example]) -> list[str | None]:
    """The LLM's label for each example, in order; fails if any example is not labeled yet."""
    cached = read_cache(cache_path(model, "banking77"))
    missing = [example.id for example in examples if example.id not in cached]
    if missing:
        raise SystemExit(f"{len(missing)} texts have no LLM label yet, e.g. {missing[0]}")
    return [cached[example.id].label for example in examples]


def run_target(
    target: float,
    train: Sequence[Example],
    train_llm: Sequence[str | None],
    test: Sequence[Example],
    test_llm: Sequence[str],
    embed: EmbedLookup,
) -> TargetResult:
    """Distill at one target and collect the out-of-fold report and the gold evaluation."""
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # an unreachable target shows up as threshold None
        router = distill(
            [example.text for example in train], train_llm, target_agreement=target, embed=embed
        )
    report = router.report()
    gold = router.evaluate(
        [example.text for example in test], [example.gold for example in test], test_llm
    )
    return TargetResult(
        target=target,
        threshold=None if math.isinf(router.threshold) else router.threshold,
        train_coverage=report.coverage,
        train_agreement=report.agreement_covered,
        train_agreement_lower=report.agreement_covered_lower,
        test=gold,
        seconds=time.perf_counter() - start,
    )


def table(results: Sequence[TargetResult]) -> str:
    """One line per target: promised (train, out-of-fold) next to measured (test)."""
    lines = [
        "                train (out-of-fold)       | test (unseen texts)",
        "target  thresh  coverage  agree  lower    | coverage  agree  student  LLM same  cascade",
    ]
    for r in results:
        t = r.test
        lines.append(
            f"{r.target:.2f}    {_num(r.threshold, '.3f'):>6}  {r.train_coverage:>8.1%}  "
            f"{_num(r.train_agreement, '.1%'):>5}  {r.train_agreement_lower:>5.1%}    | "
            f"{t.coverage:>8.1%}  {_num(t.agreement_covered, '.1%'):>5}  "
            f"{_num(t.student_accuracy_covered, '.1%'):>7}  "
            f"{_num(t.llm_accuracy_covered, '.1%'):>8}  {t.cascade_accuracy:>7.1%}"
        )
    lines.append(f"LLM alone on test: accuracy {results[0].test.llm_accuracy:.1%} against gold")
    return "\n".join(lines)


def _num(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill Banking77 LLM labels and evaluate.")
    parser.add_argument("--llm", default=DEFAULT_LLM, help="model whose cached labels to use")
    parser.add_argument(
        "--embedder", default=DEFAULT_MODEL, help="the default model, or an API embedding model"
    )
    args = parser.parse_args()

    train, test = load_banking77("train"), load_banking77("test")
    train_llm = llm_labels(args.llm, train)
    test_llm = [label or "" for label in llm_labels(args.llm, test)]  # "" never matches gold
    everything = [*train, *test]
    embed = lookup_embed(everything, embeddings(everything, args.embedder))

    results: list[TargetResult] = []
    for target in TARGETS:
        results.append(run_target(target, train, train_llm, test, test_llm, embed))
        print(f"target {target:.2f} done in {results[-1].seconds:.0f} s")

    print(table(results))
    out = RESULTS_DIR / f"banking77_{_slug(args.llm)}_{_slug(args.embedder)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
