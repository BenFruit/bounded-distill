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

# 5. Compare the LLMs' mistakes (needs at least two labeled LLMs), and test the bound on
#    every saved run
uv run --env-file .env python -m benchmarks.teachers --dataset banking77 --embedder giga-embeddings:10b
uv run python -m benchmarks.bound_check

# 6. Compare five ways to pick the threshold (fixed, no bound, Wilson, SGR) on every LLM,
#    dataset and embedder
uv run python -m benchmarks.baselines

# 7. Latency of one prediction, and the chart
uv run --env-file .env python -m benchmarks.latency
uv run python -m benchmarks.chart
```

`run` uses `threshold_method="wilson"`, the only method in 0.1, so the numbers in the main
README stay reproducible. `baselines` compares it with the default, `"sgr"`.

Datasets: `banking77` (Banking77, English, 77 intents) and `massive_ru` (MASSIVE 1.0,
Russian, 59 intents; the one intent with only 4 train texts is left out). Each script takes
`--dataset`; the commands above use Banking77.

Downloads, LLM labels and embeddings are cached in `benchmarks/.cache/` (git-ignored).
Results (metrics only, no texts) are written to `benchmarks/results/`.

The LLMs that label the data are listed in `TEACHERS` in `label.py`, by short names:
`gemma-4:31b` (the default), `qwen3-4b`, `gpt-oss-120b` and `kimi-k2.6`. Pass the name with
`--model` to `label` and with `--llm` to `run`, `suspects` and the other scripts. The name
also names the cache and the result files. If your server calls the model differently, add
`--server-model <id>` when labeling.

Reasoning models need their own request settings. With a small token limit they spend it all
on thinking and return an empty answer, so `label.py` turns thinking off where the model
allows it (Qwen, Kimi) and asks for low effort with a larger limit where it doesn't
(gpt-oss).
