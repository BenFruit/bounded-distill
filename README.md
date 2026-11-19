<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/BenFruit/bounded-distill/v0.1.0/assets/logo-dark.svg">
    <img alt="bounded-distill" src="https://raw.githubusercontent.com/BenFruit/bounded-distill/v0.1.0/assets/logo-light.svg" width="350">
  </picture>
</p>

<p align="center">
  <a href="https://pypi.org/project/bounded-distill/"><img alt="PyPI" src="https://img.shields.io/pypi/v/bounded-distill?color=blue"></a>
  <a href="https://pypi.org/project/bounded-distill/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/bounded-distill"></a>
  <a href="https://github.com/BenFruit/bounded-distill/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/BenFruit/bounded-distill/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/BenFruit/bounded-distill/blob/main/LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-green"></a>
  <a href="https://microsoft.github.io/pyright/"><img alt="Checked with pyright" src="https://microsoft.github.io/pyright/img/pyright_badge.svg"></a>
  <a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
</p>

Train a small, fast classifier on the labels your LLM already produced, and let it answer
only when it agrees with the LLM often enough, with a statistical guarantee. Everything else
goes back to the LLM.

Status: alpha (0.1.0). It works and is tested and benchmarked, but the API may change before 1.0.

## Why

A lot of text classification now runs through an LLM: a prompt, an API call, some output
parsing. It works, but each call is slow and costs money. The usual fix is to train a small
model on the LLM's own answers and fall back to the LLM when the small model is unsure.

Training the small model is the easy part. The hard part is deciding when to trust it. "It
looked fine on a validation set" is not a great reason to replace an LLM in production.

## What it does

You pass in texts with the labels your LLM gave them (from your logs, for example) and a
target like "agree with the LLM at least 97% of the time". The library then:

1. trains a small classifier: frozen sentence embeddings plus logistic regression;
2. picks a confidence threshold so that the lower confidence bound of agreement on the texts
   it answers meets your target, not just the average;
3. reports how much traffic the small model takes over, its agreement per class, and its
   latency;
4. lists suspects: texts where the small model is confident but disagrees with the LLM.
   These are often the LLM's own mistakes, or a sign that two classes in your prompt overlap;
5. checks accuracy against human labels, if you have some.

Texts below the threshold go to your LLM as before. The library itself never calls an LLM.

## Install

```bash
pip install bounded-distill
```

The default embedder runs on `sentence-transformers`, which needs PyTorch. Without a GPU,
install the CPU build of PyTorch first, it is much smaller:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Quickstart

```python
from bounded_distill import distill

# texts and the labels your LLM gave them
router = distill(texts, llm_labels, target_agreement=0.97)

if (label := router.predict(text)) is None:  # not confident enough
    label = my_llm_pipeline(text)

print(router.report())
for suspect in router.suspects(top_n=20):
    print(suspect)

router.save("router/")  # later: Router.load("router/")
```

`llm_labels` can contain `None` or empty strings, which is what failed calls and parsing
errors usually look like. Such texts are left out of training, counted in the report, and
you get a warning. Use `on_invalid="drop"` to silence it or `on_invalid="error"` to fail
instead. If you pass `labels=[...]`, any label outside that list is treated as invalid too.

The default embedder is `intfloat/multilingual-e5-small`: it handles many languages and runs
fine on a CPU. To use your own embeddings, pass `embed=`, a function that takes a list of
texts and returns an `(n, dim)` array.

## Reading the report

Here is a report on synthetic data with 3% wrong LLM labels and a 90% target. The data was
embedded with a fake embedder, which is why the latency is so low.

```
bounded-distill report
  texts:      900 total, 898 used, 2 invalid (1 missing (None), 1 empty)
  target:     agreement >= 90.0% at 95% confidence: reached
  threshold:  0.8218
  coverage:   53.6% (481 of 898 texts answered without the LLM)
  agreement:  92.3% on answered texts, lower bound 90.1%
  latency:    0.02 ms per text (median, student)

  per class  support  coverage  agreement
  billing        299     55.5%      95.8%
  other          297     52.2%      89.0%
  tech           302     53.0%      91.9%

  confident disagreements (LLM label -> student label)
  other -> tech         9
  other -> billing      8
  ...
```

Coverage is the share of texts the small model answers, which is also the share of LLM calls
you save. The lower bound is what the threshold is based on: with 95% confidence, agreement
on the answered texts is at least this high. If no threshold can reach the target, the
router doesn't answer anything and the report tells you so.

The per-class table shows what the total hides. Here the class `other` is below 90% even
though the overall number is above it. The disagreement list points at classes the LLM finds
hard to tell apart, and a catch-all class like `other` is the usual suspect.

All these numbers are out-of-fold: each text is scored by a model that didn't see it in
training. `print(report)` is meant for reading, `report.to_dict()` gives you plain JSON.

Suspects are single texts, most confident first. In this example 8 of the top 10 are labels
that were broken on purpose, even though broken labels make up only 18% of all
disagreements:

```
#90  tech -> billing (96.8%): 'billing 90'
#359  other -> tech (96.5%): 'tech 59'
```

The number in front is the text's position in your input, so you can find it in your data.

## Checking against human labels

Agreeing with the LLM is not the same as being right. If you have a few hundred texts with
human labels that weren't used for training, you can compare the LLM, the small model and
the two combined:

```python
print(router.evaluate(texts, gold_labels, llm_labels))
```

The small model only answers when it's confident, and those tend to be the easier texts. So
the report also shows the LLM's accuracy on exactly the texts the small model answered,
which is the fair comparison.

## Benchmark

I tested the library on two intent classification datasets that have human labels:

- Banking77: English banking questions, 77 intents, 10,003 train and 3,080 test texts;
- MASSIVE, Russian part: voice assistant commands, 59 intents, 11,510 train and 2,974 test
  texts.

The LLM was Gemma 4 31B, self-hosted on vLLM. It got the list of label names in the
prompt and had to answer with one of them, no examples. Like any LLM in production, it makes
mistakes: its accuracy against the human labels is 78.5% on Banking77 and 83.6% on MASSIVE.

The router was trained only on the LLM's labels for the train texts. The human labels were
used only at the end, to check the router on test texts it had never seen.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/BenFruit/bounded-distill/v0.1.0/benchmarks/results/banking77_dark.png">
  <img alt="Banking77: share of test traffic answered without the LLM against cascade accuracy, for two embedders and five target agreements" src="https://raw.githubusercontent.com/BenFruit/bounded-distill/v0.1.0/benchmarks/results/banking77_light.png">
</picture>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/BenFruit/bounded-distill/v0.1.0/benchmarks/results/massive_ru_dark.png">
  <img alt="MASSIVE Russian: share of test traffic answered without the LLM against cascade accuracy, for two embedders and five target agreements" src="https://raw.githubusercontent.com/BenFruit/bounded-distill/v0.1.0/benchmarks/results/massive_ru_light.png">
</picture>

Note that the y-axes are zoomed in. On these test sets, 0.3 points is about ten texts.

With `target_agreement=0.97`:

| Dataset | Embedder | Traffic without the LLM | Agreement with the LLM there | Accuracy of the combination | LLM alone |
|---|---|---|---|---|---|
| Banking77 | multilingual-e5-small (default) | 24.9% | 97.3% | 78.4% | 78.5% |
| Banking77 | Giga-Embeddings 10B (via `embed=`) | 76.1% | 97.7% | 79.3% | 78.5% |
| MASSIVE ru | multilingual-e5-small (default) | 26.2% | 96.9% | 83.7% | 83.6% |
| MASSIVE ru | Giga-Embeddings 10B (via `embed=`) | 66.6% | 97.3% | 84.2% | 83.6% |

One prediction takes about 13 ms with e5-small on a laptop CPU and 55 ms with
Giga-Embeddings on a GPU server, network included. One LLM call took 117 ms.

What I took away from it:

- The report can be trusted on new data. The coverage measured on test was within 1.7
  points of what the report predicted, in all 18 runs. Agreement reached the target in 14 of
  them. The other four fell short by up to 1.1 points, which is within the measurement noise
  for test sets of this size.
- The embedder matters most. With a stronger one the small model answers 2.5 to 3 times as
  much traffic at the same target. At 0.97 the combination is as accurate as the LLM alone
  with either embedder, and with Giga-Embeddings it is a bit more accurate at every target:
  the small model averages over similar texts and smooths out some of the LLM's random
  mistakes.
- Suspects find real LLM mistakes. With Giga-Embeddings, 75 of the top 100 suspects on
  Banking77 and 87 on MASSIVE were texts the LLM got wrong, while only 24% and 16% of all
  texts were. Usually the small model also suggested the correct label. With e5-small the
  numbers were lower, 39 and 55 out of 100. Some of these "LLM mistakes" are actually mistakes
  in the human labels.
- Between 40% and 70% of the LLM's mistakes are consistent ones. The small model learns them
  and agrees with the LLM, so no agreement metric can catch them. You need human labels for
  that, which is what `evaluate()` is for.

Some caveats. So far there is only one LLM in the benchmark. Both datasets are part of the
MTEB benchmark, so embedding models may have been trained on them; Giga-Embeddings doesn't
publish its training data, so its numbers could be a bit optimistic. Latency depends on
hardware, and the LLM server used here is fast, so the main saving is GPU time rather than wall
time. Texts were embedded as they are, without the instruction prefix that Giga-Embeddings
recommends for classification.

Scripts to reproduce all of this are in
[benchmarks/](https://github.com/BenFruit/bounded-distill/blob/main/benchmarks/README.md).

## How it works

1. Embed the texts with a frozen sentence embedding model.
2. Train k logistic regressions with k-fold cross-validation, so every training text gets
   class probabilities from a model that didn't see it.
3. For each possible threshold, count the texts above it and how many of them agree with the
   LLM, and compute the one-sided Wilson lower bound for that agreement.
4. Take the lowest threshold whose bound reaches the target. Lowest means the most coverage.
5. To predict, average the probabilities of the k models. These are the same models the
   threshold was picked for, so it means the same thing on new texts. A prediction is one
   embedding and a few small matrix multiplications in numpy.

A saved router is a JSON file plus an `.npz` file. There is no pickle and no training text on
disk. The design and the reasoning behind it are in
[docs/design.md](https://github.com/BenFruit/bounded-distill/blob/main/docs/design.md).

## How it differs from similar tools

There are several projects that distill an LLM classifier into a small model with an LLM
fallback. This one is different in a few ways:

- it works offline, on labels you already have, and never runs or wraps the LLM;
- the threshold comes from a statistical bound instead of being picked by hand;
- it helps you audit the LLM: confident disagreements point at its labeling mistakes, in the
  spirit of Confident Learning (Northcutt et al., 2021);
- it is small: one function call, strict typing, few dependencies.

## Limitations

- Agreement with the LLM is not accuracy. If the LLM keeps making the same mistake, the
  small model learns it too. Check with human labels.
- The threshold is chosen and evaluated on the same out-of-fold data, so the bound is a bit
  optimistic. In the benchmark the misses on new texts stayed within the noise.
- Only single-label classification with a fixed set of labels.
- When the small model answers, you get just the label, not the other fields your pipeline's
  structured output may have.

## Roadmap

- [x] Core: `distill`, `Router.predict`, save and load
- [x] Report: coverage, agreement with a lower bound, per-class numbers, latency
- [x] Suspects
- [x] Evaluation against human labels
- [x] Benchmark on Banking77 and Russian MASSIVE with real LLM labels
- [x] Release 0.1.0 on PyPI
- [ ] More LLMs in the benchmark
- [ ] Optional online mode that collects examples from a running pipeline

## Contributing

Issues and pull requests are welcome, see [CONTRIBUTING.md](https://github.com/BenFruit/bounded-distill/blob/main/CONTRIBUTING.md).

## License

Apache License 2.0, see [LICENSE](https://github.com/BenFruit/bounded-distill/blob/main/LICENSE)
and [NOTICE](https://github.com/BenFruit/bounded-distill/blob/main/NOTICE).
