# Changelog

## 0.2.1 (2026-11-23)

No code changes, so routers and results behave exactly as in 0.2.0.

Corrected:

- 0.2.0 called `threshold_method="sgr"` a strict guarantee. That was wrong. SGR's
  guarantee assumes independent agreement indicators from one fixed model chosen on a
  separate calibration set. Here the indicators are out-of-fold scores from five fold
  models that share training data, the search over thresholds is adaptive, and the router
  that answers new texts is the average of the fold models. So the confidence level is
  nominal for both methods. The README, the design doc, the changelog wording and the
  docstrings now say so.

Documentation:

- The README notes that 208 of the 2,974 MASSIVE-ru test texts also occur in its train
  split, which makes the MASSIVE-ru numbers slightly optimistic.
- The benchmark adds `benchmarks/audit.py`, the blind audit of errors shared by all four
  LLMs, with its results. The annotators are two LLMs, not people.

## 0.2.0 (2026-11-20)

Changed:

- The threshold is now picked with a strict guarantee by default:
  `threshold_method="sgr"`, Selection with Guaranteed Risk (Geifman and El-Yaniv, 2017).
  The router may answer a few percent less traffic than in 0.1. To get the 0.1 behavior,
  pass `threshold_method="wilson"`.
- With SGR, `Report.agreement_covered_lower` is the Clopper-Pearson bound that SGR checked.
- `import bounded_distill` takes about 0.3 s instead of 2.4 s. scikit-learn is loaded on
  the first `distill()` call, so a router loaded from disk to predict never imports it.
- `scipy` is now a listed dependency. scikit-learn already needed it, so nothing new is
  installed.

Added:

- `distill(threshold_method=...)`, `Router.threshold_method`, `Report.threshold_method` and
  the `ThresholdMethod` type. The method is saved with the router. Routers saved by 0.1 load
  as `"wilson"`.

Benchmark:

- `baselines.py` compares five ways to pick the threshold on 16 runs. The results are in the
  README.
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
