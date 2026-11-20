# Changelog

## Unreleased

No changes to the library.

- The benchmark now has four LLMs: Gemma 4 31B, Qwen3-4B-Instruct, gpt-oss-120b and
  Kimi-K2.6. The results for all of them are in the README.
- New benchmark scripts: `teachers.py` compares the LLMs' mistakes, and `bound_check.py` tests
  the promised agreement on the test texts of every run.
- `label.py` takes per-model request settings, needed for reasoning models, and a
  `--server-model` option.

## 0.1.1 (2026-11-19)

No code changes.

- The README and the design doc were rewritten in plainer language, and the roadmap is up to
  date.
- The design doc moved from `HLD/offline_core.md` to `docs/design.md`.
- Added `CONTRIBUTING.md`, a pull request template and a changelog.
- Development: `uv sync` now reinstalls the package when the version changes.

## 0.1.0 (2026-11-19)

First public release: `distill()`, `Router` with `predict`, `report`, `suspects`, `evaluate`,
`save` and `load`, a multilingual default embedder, and a benchmark on Banking77 and Russian
MASSIVE with real LLM labels.
