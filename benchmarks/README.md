# Benchmarks

Scripts behind the benchmark in the main README. They are not part of the published package.

## Setup

```bash
uv sync            # installs the dev and bench dependency groups
```

Any OpenAI-compatible server works for the LLM and for API embedders (vLLM, Ollama, a cloud
provider). Put its address in a `.env` file at the repository root; it is git-ignored:

```
LLM_BASE_URL=https://your-server/v1
LLM_API_KEY=...
```

## Steps

```bash
# 1. Download the datasets (pinned revisions, SHA-256 checked) and print a summary
uv run python -m benchmarks.data

# 2. Label train and test with the LLM (resumable cache; --limit 50 for a quick probe)
uv run --env-file .env python -m benchmarks.label --dataset banking77 --split test
uv run --env-file .env python -m benchmarks.label --dataset banking77 --split train

# 3. Distill at targets 0.90-0.99 and evaluate on test against the human labels
uv run python -m benchmarks.run --dataset banking77
uv run --env-file .env python -m benchmarks.run --dataset banking77 --embedder giga-embeddings:10b

# 4. Compare suspects with the human labels
uv run python -m benchmarks.suspects --dataset banking77

# 5. Latency of one prediction, and the chart
uv run --env-file .env python -m benchmarks.latency
uv run python -m benchmarks.chart
```

Datasets: `banking77` (Banking77, English, 77 intents) and `massive_ru` (MASSIVE 1.0,
Russian, 59 intents; the one intent with only 4 train texts is left out). Each script takes
`--dataset`; the commands above use Banking77.

Downloads, LLM labels and embeddings are cached in `benchmarks/.cache/` (git-ignored).
Results (metrics only, no texts) are written to `benchmarks/results/`.

The default teacher is `gemma-4:31b` (Gemma 4 31B); pass `--model` to `label` and `--llm` to
`run` for another one. Model names are those your server uses.
