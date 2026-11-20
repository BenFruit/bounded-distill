"""Label benchmark texts with an LLM through any OpenAI-compatible API, with a resumable cache.

The LLM plays the role of a production classification pipeline: it sees the label names and
the text, and its answer is constrained to one of the labels by a JSON schema. Configure the
endpoint in a ``.env`` file (never committed):

    LLM_BASE_URL=https://your-server/v1
    LLM_API_KEY=...

Then, for example:
``uv run --env-file .env python -m benchmarks.label --dataset banking77 --split test --limit 50``

Teachers are known by a short public name, which also names their cache and result files. If
your server calls a model differently, pass its id with ``--server-model``.
"""

import argparse
import json
import os
import random
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast

from openai import OpenAI, omit
from openai.types.shared_params import ResponseFormatJSONSchema

from benchmarks.data import CACHE_DIR, DATASETS, Example, Split


@dataclass(frozen=True)
class Teacher:
    """An LLM that labels the benchmark, and the request settings it needs.

    Reasoning models spend tokens on thinking before the answer. With the default token limit
    they return an empty answer, so thinking is turned off where the model allows it
    (``thinking=False``) and kept short with a larger limit where it does not.
    """

    model: str  # the model id sent to the server
    max_tokens: int = 50
    thinking: bool | None = None  # None leaves the server's default
    reasoning_effort: Literal["low", "medium", "high"] | None = None


TEACHERS: dict[str, Teacher] = {
    "gemma-4:31b": Teacher("gemma-4:31b"),
    "qwen3-4b": Teacher("Qwen/Qwen3-4B-Instruct-2507"),
    "gpt-oss-120b": Teacher("gpt-oss:120b", max_tokens=2000, reasoning_effort="low"),
    "kimi-k2.6": Teacher("moonshotai/Kimi-K2.6", thinking=False),
}
DEFAULT_MODEL = "gemma-4:31b"


@dataclass(frozen=True)
class Labeled:
    """The LLM's answer for one text; ``label`` is None if the answer was not a valid label."""

    id: str
    label: str | None
    raw: str  # the answer as returned, kept for inspecting failures


def system_prompt(task: str, labels: Sequence[str]) -> str:
    """A plain zero-shot instruction listing the label names, as a typical pipeline would."""
    return (
        f"{task} "
        "Choose exactly one intent from this list:\n"
        + "\n".join(labels)
        + '\nAnswer in JSON: {"label": "<intent>"}.'
    )


def response_format(labels: Sequence[str]) -> ResponseFormatJSONSchema:
    """A JSON schema that only allows one of ``labels`` (constrained decoding on vLLM)."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "intent",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"label": {"type": "string", "enum": list(labels)}},
                "required": ["label"],
                "additionalProperties": False,
            },
        },
    }


def parse_label(raw: str, labels: Sequence[str]) -> str | None:
    """The label in a JSON answer, or None if the answer is not valid JSON with a known label."""
    try:
        answer: object = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(answer, dict):
        return None
    label = cast(dict[str, object], answer).get("label")
    return label if isinstance(label, str) and label in labels else None


def label_one(
    client: OpenAI, teacher: Teacher, task: str, labels: Sequence[str], example: Example
) -> Labeled:
    """Ask the LLM for one text's label, deterministically (temperature 0)."""
    thinking = None if teacher.thinking is None else {"enable_thinking": teacher.thinking}
    completion = client.chat.completions.create(
        model=teacher.model,
        messages=[
            {"role": "system", "content": system_prompt(task, labels)},
            {"role": "user", "content": example.text},
        ],
        response_format=response_format(labels),
        temperature=0,
        max_tokens=teacher.max_tokens,
        reasoning_effort=teacher.reasoning_effort or omit,
        extra_body=None if thinking is None else {"chat_template_kwargs": thinking},
    )
    raw = completion.choices[0].message.content or ""
    return Labeled(id=example.id, label=parse_label(raw, labels), raw=raw)


def cache_path(model: str, dataset: str) -> Path:
    """One JSON-lines file per teacher name and dataset, e.g. ``.cache/labels/gemma-4_31b/...``."""
    return CACHE_DIR / "labels" / model.replace(":", "_").replace("/", "_") / f"{dataset}.jsonl"


def read_cache(path: Path) -> dict[str, Labeled]:
    """Labels already obtained, by example id; a missing file means none."""
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {row["id"]: Labeled(id=row["id"], label=row["label"], raw=row["raw"]) for row in rows}


def label_all(
    client: OpenAI,
    teacher: Teacher,
    task: str,
    labels: Sequence[str],
    examples: Sequence[Example],
    cache: Path,
    workers: int,
) -> dict[str, Labeled]:
    """Label every example not yet in ``cache``, appending each answer as soon as it arrives.

    An interrupted run resumes where it stopped. Failed requests (after the client's own
    retries) are reported and not cached, so the next run tries them again. Only the requests
    run in worker threads; the cache file is written by this thread alone.
    """
    done = read_cache(cache)
    todo = [example for example in examples if example.id not in done]
    cache.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    with cache.open("a", encoding="utf-8") as file, ThreadPoolExecutor(workers) as pool:
        futures = [
            pool.submit(label_one, client, teacher, task, labels, example) for example in todo
        ]
        for i, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()
            except Exception as error:  # network or server errors: retried on the next run
                failed += 1
                print(f"request failed: {error}")
                continue
            file.write(json.dumps(result.__dict__, ensure_ascii=False) + "\n")
            file.flush()
            done[result.id] = result
            if i % 500 == 0:
                print(f"  {i:,} / {len(todo):,}")
    if failed:
        print(f"{failed} requests failed; run again to retry them")
    return done


def summary(examples: Sequence[Example], labeled: dict[str, Labeled], seconds: float) -> str:
    """Coverage of the cache, invalid answers, and the LLM's accuracy against gold."""
    rows = [(example, labeled[example.id]) for example in examples if example.id in labeled]
    invalid = sum(answer.label is None for _, answer in rows)
    correct = sum(answer.label == example.gold for example, answer in rows)
    accuracy = correct / len(rows) if rows else 0.0
    return (
        f"{len(rows):,} of {len(examples):,} texts labeled ({seconds:.0f} s this run), "
        f"{invalid} invalid answers, LLM accuracy vs gold {accuracy:.1%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Label benchmark texts with an LLM.")
    parser.add_argument("--dataset", choices=list(DATASETS), default="banking77")
    parser.add_argument("--model", choices=list(TEACHERS), default=DEFAULT_MODEL)
    parser.add_argument("--server-model", help="the model id on your server, if it differs")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--limit", type=int, help="label a fixed random sample of this size")
    parser.add_argument("--workers", type=int, default=4, help="parallel requests")
    args = parser.parse_args()
    split: Split = args.split

    dataset = DATASETS[args.dataset]
    examples = dataset.load(split)
    if args.limit is not None:
        examples = random.Random(0).sample(examples, args.limit)

    teacher = TEACHERS[args.model]
    if args.server_model is not None:
        teacher = replace(teacher, model=args.server_model)

    client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])
    start = time.perf_counter()
    labeled = label_all(
        client,
        teacher,
        dataset.task,
        dataset.labels(),
        examples,
        cache_path(args.model, dataset.name),
        args.workers,
    )
    print(summary(examples, labeled, time.perf_counter() - start))


if __name__ == "__main__":
    main()
