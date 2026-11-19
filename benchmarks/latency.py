"""Median latency of one prediction: the router with each embedder, and one LLM call.

Single texts, one at a time, after a warm-up call: the latency a request sees in production.
Run: ``uv run --env-file .env python -m benchmarks.latency``
"""

import dataclasses
import os
import statistics
import time
import warnings
from collections.abc import Callable, Sequence

from openai import OpenAI

from benchmarks.data import load_banking77
from benchmarks.label import DEFAULT_MODEL as DEFAULT_LLM
from benchmarks.label import label_one
from benchmarks.run import Vectors, api_embed, embeddings, llm_labels, lookup_embed
from bounded_distill import distill
from bounded_distill._embed import DEFAULT_MODEL, default_embed

API_EMBEDDER = "giga-embeddings:10b"
N_TEXTS = 100


def median_ms(call: Callable[[str], object], texts: Sequence[str]) -> float:
    """Median wall-clock time of ``call`` per text, in milliseconds, after one warm-up call."""
    call(texts[0])
    timings: list[float] = []
    for text in texts:
        start = time.perf_counter()
        call(text)
        timings.append((time.perf_counter() - start) * 1000)
    return statistics.median(timings)


def main() -> None:
    train, test = load_banking77("train"), load_banking77("test")
    texts = [example.text for example in test[:: len(test) // N_TEXTS][:N_TEXTS]]
    labels = sorted({example.gold for example in train})
    everything = [*train, *test]

    def api(texts: Sequence[str]) -> Vectors:
        return api_embed(API_EMBEDDER, texts)

    for embedder, live_embed in [(DEFAULT_MODEL, default_embed), (API_EMBEDDER, api)]:
        cached = lookup_embed(everything, embeddings(everything, embedder))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            router = distill([e.text for e in train], llm_labels(DEFAULT_LLM, train), embed=cached)
        live = dataclasses.replace(router, embed=live_embed)  # embeds each new text for real
        print(f"router predict, {embedder}: {median_ms(live.predict, texts):.1f} ms")

    client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])
    by_text = {example.text: example for example in test}
    llm_ms = median_ms(lambda text: label_one(client, DEFAULT_LLM, labels, by_text[text]), texts)
    print(f"LLM call, {DEFAULT_LLM}: {llm_ms:.1f} ms")


if __name__ == "__main__":
    main()
