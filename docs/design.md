# bounded-distill: design of the offline core

This document describes the 0.1 versions and explains why it is built the way it is. It only
covers the offline mode, where the LLM labels already exist. An online mode is out of scope.

## 1. Goal

Take texts and the labels an LLM pipeline already gave them, and build a small, fast
classifier plus a confidence threshold. On the texts the classifier answers, agreement with
the LLM should be at least `target_agreement`, with a stated statistical confidence. For
everything below the threshold the classifier returns `None`, and the caller uses their own
LLM pipeline. The library never calls an LLM.

It also produces a report (coverage, agreement with its lower bound, per-class numbers,
latency) and a list of suspects: confident disagreements, which often turn out to be LLM
labeling mistakes.

## 2. Non-goals for 0.1

Multi-label or hierarchical labels, generation tasks, calling LLM providers, an online mode,
fine-tuning transformers, several model backends, per-class thresholds, drift monitoring,
plugins. Any of these can come later if users ask for them.

## 3. Public API

```python
EmbedFn = Callable[[Sequence[str]], np.ndarray]   # list of texts -> (n, d) float array

def distill(
    texts: Sequence[str],
    llm_labels: Sequence[str | None],
    *,
    target_agreement: float = 0.97,
    confidence_level: float = 0.95,
    labels: Sequence[str] | None = None,   # fixed label set; inferred from the data if None
    embed: EmbedFn | None = None,          # default: multilingual sentence embedder
    n_folds: int = 5,
    random_state: int = 0,
    on_invalid: Literal["drop", "warn", "error"] = "warn",   # what to do with invalid labels
) -> Router: ...

class Router:
    labels: tuple[str, ...]
    threshold: float                       # math.inf if the target can't be reached

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

`Report`, `Suspect` and `GoldReport` are frozen dataclasses with `to_dict()` and a readable
`__str__`.

Intended use: `if (label := router.predict(text)) is None: label = my_pipeline(text)`.

### Input rules

- `texts` and `llm_labels` must have the same length, otherwise `ValueError`.
- An LLM label is invalid if it is `None` (reason `missing`), an empty or whitespace-only
  string (`empty`, a typical parsing failure), or not in `labels` when `labels` is given
  (`unknown`). Labels are compared exactly; the library never strips or lowercases them.
- Invalid pairs are left out of training and counted per reason in the report, since they
  say something about the prompt and the parsing. `on_invalid` decides what else happens:
  `"drop"` does nothing more, `"warn"` (the default) emits a `UserWarning` with the counts,
  `"error"` raises a `ValueError` with the same counts. It is a string literal rather than an
  enum: users pass a plain string, the type checker still catches typos, and a runtime check
  catches the rest.
- `labels`, if given, must be unique and non-empty.
- After filtering there must be at least 2 classes, and every class needs at least `n_folds`
  examples. Otherwise `ValueError`, naming every class that is too small.
- `0 < target_agreement < 1`, `0 < confidence_level < 1`, `n_folds >= 2`. Settings are checked
  before embedding, so a typo fails right away.

## 4. Algorithm

1. Validate and filter the inputs (section 3).
2. Embed: `X = embed(texts)`, L2-normalized, float32.
3. Out-of-fold probabilities: `StratifiedKFold(n_folds, shuffle=True, random_state)`. For each
   fold, fit `LogisticRegression(max_iter=1000)` on the other folds and predict the held-out
   fold. Every training text gets class probabilities from a model that didn't see it.
4. Confidence is the highest predicted probability, and the student label is its class. No
   probability calibration (Platt, isotonic) is needed: the threshold is chosen empirically,
   so only the ranking of confidences matters.
5. Threshold selection, in the pure function `select_threshold`:
   - the candidates are the unique out-of-fold confidences;
   - for a candidate `t`, the covered texts are those with confidence ≥ t, `n` is their count
     and `k` is how many of them have student label == LLM label;
   - `lb = wilson_lower_bound(k, n, confidence_level)`, one-sided;
   - pick the lowest `t` (the most coverage) with `lb ≥ target_agreement`;
   - if no candidate qualifies, `threshold = math.inf`: the router always returns `None`, the
     report says the target can't be reached, and a `UserWarning` is emitted. No exception.
6. Prediction uses the `n_folds` models from step 3: the probabilities for a new text are their
   average, and the threshold applies to the highest averaged probability. There is no refit
   on all the data. A refit model is more confident than the fold models the threshold was
   chosen on, so the threshold would mean something different for it. In the benchmark
   (Banking77 and Russian MASSIVE, two embedders) a refit model answered 0.5 to 10 points more
   test traffic than the report estimated and fell below the target in 6 of 18 runs. The
   average of the fold models matched the estimated coverage within 1.7 points and missed the
   target in 4 of 18 runs, all within sampling error. The cost is `n_folds` small weight
   matrices instead of one.
7. The out-of-fold confidences and labels are kept in memory for the report and suspects.

### Wilson one-sided lower bound

With `z = NormalDist().inv_cdf(confidence_level)` (standard library, no scipy) and `p = k / n`:

    lb = (p + z²/(2n) − z·sqrt(p(1−p)/n + z²/(4n²))) / (1 + z²/n)

For `n = 0`, `lb = 0`.

### Known limitations

- The threshold is chosen and evaluated on the same out-of-fold data, and choosing the best of
  many candidates makes the bound a little optimistic. A separate calibration split may come
  later if it turns out to matter.
- The average of the fold models is a bit smoother than any single fold model, so its
  confidences are close to the out-of-fold ones but not identical. Step 6 has the measured
  effect.
- Agreement with the LLM is not accuracy. The student learns the LLM's consistent mistakes.

## 5. Report

`Report` fields:

- `n_total`, `n_invalid` with counts per reason, `n_used`, `labels`, `threshold`,
  `target_reached`;
- `n_covered` and `coverage`, the share of texts the student answers (out-of-fold). Coverage
  is also the share of LLM calls saved, so there is no separate field for that;
- `agreement_covered`, the point estimate, and `agreement_covered_lower`, the Wilson bound;
- `per_class`: for each LLM label, its support, coverage and agreement on the covered texts;
- `disagreements`: counts of (LLM label, student label) pairs among confident disagreements,
  which point at classes the prompt doesn't separate well;
- `student_latency_ms`: the median time of a single `predict()` call (embedding plus logistic
  regression) on up to 20 training texts, after one untimed warm-up call that may load the
  model.

`report()` and `suspects()` need the training data, which is only kept in memory. They work on
the router returned by `distill()`; on a loaded router they raise a `ValueError` that says so.

`Suspect` has `index` (the position in the texts given to `distill()`, invalid ones
included), `text`, `llm_label`, `student_label` and `confidence`. Suspects come from the
out-of-fold predictions where the student disagrees with the LLM, most confident first (ties
keep the original order), at most `top_n` of them. The threshold doesn't apply here: this is
a list for a human to review, not routing. The library never relabels data by itself.

`GoldReport` comes from `evaluate()`, which takes held-out texts with human labels and the
LLM's labels. It needs no training data, so it also works on a loaded router. It has
`n_texts`, `n_covered` and `coverage`; the accuracy against human labels of the LLM on all
texts, of the combination (student when confident, the given LLM label otherwise), of the
student on the texts it answers, and of the LLM on those same texts; and the agreement
between student and LLM on the answered texts, which checks the bound on new data. The LLM's
accuracy on the answered texts is there because the student answers the easier texts, so
comparing it with the LLM on all texts would be unfair.

## 6. Default embedder

- `sentence-transformers` (a core dependency) with `intfloat/multilingual-e5-small`: it is
  multilingual, small and fine on a CPU. e5 models expect a `"query: "` prefix, which the
  default embedder adds. Texts are encoded in batches and the output is normalized.
- The model loads lazily on first use and is cached in the process.
- A custom `embed` function replaces it completely, for example API embeddings or vectors you
  already store in a database.

## 7. Persistence

`save(path)` writes a directory, without pickle:

- `router.json` with the format version, the library version, the labels, the threshold, the
  settings, and the embedder's model name (or `"custom"`). No report, since that depends on
  the training data.
- A threshold of `math.inf` is stored as JSON `null`, because strict JSON has no Infinity.
  Training texts and out-of-fold data are never written to disk, since they may contain
  private user data. A loaded router can predict; reports that need the data are made at
  `distill()` time.
- `weights.npz` with the fold models' `weights`, each one scikit-learn's `coef_` transposed,
  shape `(n_folds, dim, n_logits)`, so model `k` computes `logits = x @ weights[k] +
  intercept[k]`; and `intercept`, shape `(n_folds, n_logits)`. This is format version 2.
  Version 1, which held a single refit model, can't be read.

Prediction uses only numpy (softmax for several classes, sigmoid for two), so a loaded router
needs no scikit-learn objects. `load(path, embed=None)` rebuilds the default embedder by name.
If the router was saved with a custom `embed`, it has to be passed again, otherwise
`ValueError`.

## 8. Module layout

```
src/bounded_distill/
  __init__.py      # public exports: distill, Router, Report, ClassStats, Suspect, GoldReport,
                   # EmbedFn, InvalidPolicy
  _types.py        # EmbedFn, InvalidPolicy, TrainingData and other small shared types
  _sklearn.py      # the only module that imports scikit-learn: cross_fit returns the
                   # out-of-fold probabilities and the fold models
  _embed.py        # default embedder (sentence-transformers, loaded lazily)
  _threshold.py    # wilson_lower_bound and select_threshold, both pure functions
  _distill.py      # input validation, cross-validation, distill()
  _router.py       # Router: predict, report, suspects, evaluate, save and load
  _report.py       # Report, Suspect, GoldReport and the functions that build them
```

Small pure functions, no class hierarchies, no extension points.

## 9. Dependencies and tooling

- Runtime: `numpy`, `scikit-learn`, `sentence-transformers`. The README explains how to
  install the CPU build of PyTorch.
- Development: `ruff` for linting and formatting, `pyright` in strict mode, `pytest`. CI runs
  on the oldest and the newest supported Python. The benchmark scripts in `benchmarks/` (not
  part of the package) use a separate `bench` dependency group: `openai`, `pyarrow`,
  `matplotlib`.
- Python 3.10 or newer.

## 10. Testing

- `wilson_lower_bound`: known values and edge cases (`n = 0`, `k = n`).
- `select_threshold`: synthetic confidences and agreements, and an unreachable target giving
  `inf`.
- `distill` end to end with a fake, deterministic `embed`, so no model is downloaded:
  well-separated synthetic data gives high coverage, noisy labels give lower coverage, and
  bad inputs raise the expected errors.
- Suspects: deliberately flipped labels show up at the top.
- Each fold model's numpy prediction matches scikit-learn's `predict_proba`, and the router
  averages the fold models.
- A `save` and `load` round trip gives identical predictions.
- One slow test with the real default embedder, marked and skipped by default.

## 11. Decisions on earlier open questions

- Defaults `target_agreement=0.97` and `confidence_level=0.95`: kept. They are on the safe
  side, and in the benchmark the combination is as accurate as the LLM alone at 0.97.
- No separate minimum number of covered texts. The Wilson bound already requires one: even
  when every covered text agrees, the bound only reaches 0.90, 0.95, 0.97 or 0.99 from 25, 52,
  88 or 268 covered texts on, at 95% confidence. A second rule would repeat it.
- `n_folds` and `random_state` are keyword arguments. `n_folds` is also the number of models
  the router averages, and `random_state` makes runs reproducible.

Still open for later versions: per-class thresholds, a separate calibration split to remove
the selection optimism (section 4), an online mode.
