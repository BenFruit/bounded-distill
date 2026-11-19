# bounded-distill — design of the offline core

This document describes version **0.1.0** and the reasons behind its design decisions.
Scope: the offline mode only (labels already exist). An online mode is out of scope.

## 1. Goal

Given texts and the labels an LLM pipeline already produced for them, build a small, fast
classifier and a confidence threshold such that, on the texts the classifier answers,
agreement with the LLM is at least `target_agreement` **with a stated statistical confidence**.
Everything below the threshold is returned as "not confident" (`None`) so the caller can use
their own LLM pipeline. The library never calls an LLM.

Secondary outputs: a report (coverage, agreement with bound, per-class stats, savings) and a
list of "suspects" — confident disagreements that often reveal LLM labeling errors.

## 2. Non-goals (v0.1)

Multi-label, hierarchical labels, generation tasks, calling LLM providers, online mode,
fine-tuning transformers, multiple model backends, per-class thresholds, drift monitoring,
plugin systems. Each can be added later only if users ask.

## 3. Public API

```python
EmbedFn = Callable[[Sequence[str]], np.ndarray]   # list of texts -> (n, d) float array

def distill(
    texts: Sequence[str],
    llm_labels: Sequence[str | None],
    *,
    target_agreement: float = 0.97,
    confidence_level: float = 0.95,
    labels: Sequence[str] | None = None,   # closed label set; inferred from data if None
    embed: EmbedFn | None = None,          # default: multilingual sentence embedder
    n_folds: int = 5,
    random_state: int = 0,
    on_invalid: Literal["drop", "warn", "error"] = "warn",   # policy for invalid LLM labels
) -> Router: ...

class Router:
    labels: tuple[str, ...]
    threshold: float                       # math.inf if the target is unreachable

    def predict(self, text: str) -> str | None: ...
    def predict_batch(self, texts: Sequence[str]) -> list[str | None]: ...
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray: ...   # (n, n_labels)
    def report(self) -> Report: ...
    def suspects(self, top_n: int = 50) -> list[Suspect]: ...
    def evaluate(self, texts: Sequence[str], gold_labels: Sequence[str],
                 llm_labels: Sequence[str]) -> GoldReport: ...
    def save(self, path: str | Path) -> None: ...
    @classmethod
    def load(cls, path: str | Path, embed: EmbedFn | None = None) -> "Router": ...
```

`Report`, `Suspect`, `GoldReport` are frozen dataclasses with `to_dict()` and a readable `__str__`.

Usage contract (README): `if (label := router.predict(text)) is None: label = my_pipeline(item)`.

### Input rules
- `texts` and `llm_labels` must have equal length (else `ValueError`).
- An LLM label is **invalid** if it is `None` (reason `missing`), an empty or whitespace-only
  string (`empty`, a typical parsing failure), or not in `labels` when `labels` is given
  (`unknown`). Labels are compared exactly; the library never strips or lowercases them.
- Invalid pairs are excluded from training and counted per reason in the report (signal about
  prompt and parsing quality). `on_invalid` decides what else happens:
  `"drop"` — nothing; `"warn"` (default) — a `UserWarning` with the counts per reason;
  `"error"` — a `ValueError` with the same counts. A literal type rather than an enum: users
  pass a plain string, the type checker still catches typos, and a runtime check catches the rest.
- `labels`, when given, must be unique and non-empty.
- After filtering: at least 2 classes, and every class needs at least `n_folds` examples
  (else `ValueError` naming every such class).
- `0 < target_agreement < 1`, `0 < confidence_level < 1`, `n_folds >= 2`. Settings are checked
  before embedding, so a typo fails fast.

## 4. Algorithm

1. **Validate and filter** inputs (§3).
2. **Embed**: `X = embed(texts)`, L2-normalized, float32.
3. **Out-of-fold probabilities**: `StratifiedKFold(n_folds, shuffle=True, random_state)`;
   for each fold fit `LogisticRegression(max_iter=1000)` on the other folds and predict
   probabilities on the held-out fold. Result: every training text gets a probability vector
   from a model that did not see it.
4. **Confidence** = max predicted probability; **student label** = its argmax.
   Probability calibration (Platt/isotonic) is not needed: the threshold is chosen
   empirically, only the ranking of confidences matters.
5. **Threshold selection** (pure function `select_threshold`):
   - candidates = unique out-of-fold confidences;
   - for candidate `t`: covered = texts with confidence ≥ t, `n` = covered count,
     `k` = covered texts where student label == LLM label;
   - `lb = wilson_lower_bound(k, n, confidence_level)` (one-sided);
   - choose the **lowest** `t` (maximum coverage) with `lb ≥ target_agreement`;
   - if none qualifies: `threshold = math.inf` (router always returns `None`), the report says
     the target is unreachable, and a `UserWarning` is emitted. No exception.
6. **Prediction model**: the `n_folds` fold models from step 3; probabilities on a new text are
   their average, and the chosen threshold applies to that average's maximum. No refit on all
   data: a refit model is more confident than the fold models the threshold was chosen on, so
   the threshold would mean something else for it. In the benchmark (Banking77, MASSIVE ru,
   two embedders) a refit model answered 0.5–10 points more test traffic than the report
   estimated, and fell below the target agreement in 6 of 18 cases; the fold-model average
   matched the estimated coverage within 1.7 points and missed the target in 4 of 18, all
   within sampling error. Cost: `n_folds` small weight matrices instead of one.
7. **Keep for reporting**: out-of-fold confidences/labels (for report and suspects).

### Wilson one-sided lower bound
`z = NormalDist().inv_cdf(confidence_level)` (stdlib, no scipy), `p = k / n`:

    lb = (p + z²/(2n) − z·sqrt(p(1−p)/n + z²/(4n²))) / (1 + z²/n)

`n = 0` → `lb = 0`.

### Known limitations (documented, not solved in v0.1)
- The threshold is selected and evaluated on the same out-of-fold data, so the bound is
  slightly optimistic (selection over many candidates). A separate calibration split may come
  in v0.2 if needed.
- The fold-model average is slightly smoother than a single fold model, so its confidences
  are close to, not identical with, the out-of-fold ones (see step 6 for measured effects).
- Agreement with the LLM is not accuracy; systematic LLM errors are learned by the student.

## 5. Report

`Report` fields:
- `n_total`, `n_invalid` (excluded LLM labels) with counts per reason, `n_used`, `labels`, `threshold`, `target_reached`.
- `coverage` — share of texts the student answers (out-of-fold estimate).
- `agreement_covered` — point estimate, and `agreement_covered_lower` — Wilson bound.
- `per_class` — for each LLM label: support, coverage, agreement on covered.
- `disagreements` — counts of (LLM label → student label) pairs among covered disagreements
  (points to ambiguous class definitions in the prompt).
- `n_covered` and `coverage`; coverage is also the share of LLM calls saved (no separate field).
- `student_latency_ms` — median latency of single `predict()` calls (embed + LR) on up to 20
  training texts, after one untimed warm-up call (which may load the model).

`report()` and `suspects()` need the in-memory training data, so they work on the router
returned by `distill()`; on a loaded router they raise `ValueError` with that explanation.

`Suspect`: `index` (position in the texts given to `distill()`, invalid ones included),
`text`, `llm_label`, `student_label`, `confidence` — from out-of-fold predictions where
student ≠ LLM, sorted by confidence descending (ties keep the original order), at most
`top_n`. The threshold is not applied: it is a review queue, not routing. The library never
relabels data automatically.

`GoldReport` (from `evaluate` on held-out texts with gold labels and the LLM's labels; needs
no training data, so it works on a loaded router too): `n_texts`, `n_covered`, `coverage`;
accuracy vs gold of the LLM on all texts, of the cascade (student when confident, otherwise
the given LLM label), of the student on the texts it answers, and of the LLM on those same
texts (the student answers the easier texts, so this is the fair comparison); and agreement
between student and LLM on the answered texts (an independent check of the bound).

## 6. Default embedder

- `sentence-transformers` (core dependency), model `intfloat/multilingual-e5-small`
  (multilingual, small, CPU-friendly). e5 models expect a `"query: "` prefix; the default
  embedder adds it. Batched encoding, normalized output.
- The model is loaded lazily on first use and cached in the process.
- A custom `embed` function replaces it completely (e.g. API embeddings, vectors already in a DB).

## 7. Persistence

`save(path)` writes a directory, no pickle:
- `router.json`: format version, library version, labels, threshold, settings, and the
  embedder model name (or `"custom"`). No report: it depends on the training data.
- The threshold `math.inf` (target unreachable) is stored as JSON `null`: strict JSON has no
  Infinity. Training texts and out-of-fold data are never written to disk (they may hold
  private user data); a loaded router predicts, while data-dependent reports are computed
  at `distill()` time.
- `weights.npz`: the fold models' `weights` (each scikit-learn's `coef_` transposed, shape
  `(n_folds, dim, n_logits)`, so model `k` has `logits = x @ weights[k] + intercept[k]`) and
  `intercept` `(n_folds, n_logits)`. Format version 2; version 1 (a single refit model) is
  not read.

Inference uses numpy only (softmax for multiclass, sigmoid for binary), so a loaded router does
not need scikit-learn objects. `load(path, embed=None)` rebuilds the default embedder by name;
if the router was saved with a custom `embed`, it must be passed again (else `ValueError`).

## 8. Module layout

```
src/bounded_distill/
  __init__.py      # public exports: distill, Router, Report, ClassStats, Suspect, GoldReport,
                   # EmbedFn, InvalidPolicy
  _types.py        # EmbedFn, InvalidPolicy, TrainingData and other small shared types
  _sklearn.py      # the only module importing scikit-learn: typed boundary (cross_fit: OOF probas + fold models)
  _embed.py        # default embedder (lazy sentence-transformers)
  _threshold.py    # wilson_lower_bound, select_threshold — pure functions
  _distill.py      # input validation, out-of-fold training, distill()
  _router.py       # Router: predict, report, suspects, evaluate, save/load
  _report.py       # Report, Suspect, GoldReport dataclasses + builders
```

Small pure functions, no class hierarchies, no extension points.

## 9. Dependencies and tooling

- Runtime: `numpy`, `scikit-learn`, `sentence-transformers` (README: install CPU-only torch).
- Dev: `ruff` (lint + format), `pyright` in strict mode, `pytest`; CI on the oldest and newest
  supported Python. Benchmark scripts (`benchmarks/`, not in the package) use a separate
  `bench` dependency group: `openai`, `pyarrow`, `matplotlib`.
- Python ≥ 3.10.

## 10. Testing

- `wilson_lower_bound`: known values, edge cases (`n = 0`, `k = n`).
- `select_threshold`: synthetic confidences/agreements, unreachable target → `inf`.
- `distill` end-to-end with a **fake deterministic `embed`** (no model download): separable
  synthetic data gives high coverage; noisy labels give lower coverage; input validation errors.
- Suspects: injected label flips appear at the top.
- Every fold model's numpy inference matches scikit-learn's `predict_proba`; the router
  averages the fold models.
- `save`/`load` round trip gives identical predictions.
- One slow test with the real default embedder, marked and skipped by default.

## 11. Decisions on earlier open questions

- **Defaults `target_agreement=0.97`, `confidence_level=0.95`**: kept. They are conservative
  and the benchmark shows a cascade as accurate as the LLM alone at 0.97.
- **No separate minimum covered count.** The Wilson bound already demands one: even with
  every covered text agreeing, the bound reaches 0.90 / 0.95 / 0.97 / 0.99 only from 25 / 52 /
  88 / 268 covered texts on, at 95% confidence. A second rule would duplicate it.
- **`n_folds` and `random_state` are keyword arguments.** `n_folds` also sets the number of
  models the router averages; `random_state` makes runs reproducible.

Open for later versions: per-class thresholds, a separate calibration split against the
selection optimism (§4), an online mode.
