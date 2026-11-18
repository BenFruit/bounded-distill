# bounded-distill

**Distill your LLM classifier into a small, fast model — with a statistical bound on agreement.**

> **Status: pre-alpha (planning).** Version 0.0.1 contains no working API yet.
> This README describes the planned design. Feedback is welcome.

## The problem

Many teams classify text with an LLM pipeline: a prompt, a provider, output parsing.
It works, but every call is slow and costs money. A common fix is to train a small
classifier on the LLM's own labels and fall back to the LLM only when the small model
is unsure.

Training that classifier is easy. Deciding **when it is safe to trust it** is the hard
part. Nobody wants to replace an LLM in production on the basis of "it looked fine on
a validation set".

## What bounded-distill does

You give it texts and the labels your LLM already produced (for example, from your
logs). You state a target, such as *"agree with the LLM at least 97% of the time"*.
bounded-distill:

1. trains a small classifier (frozen sentence embeddings + logistic regression);
2. picks a confidence threshold so that the **lower confidence bound** of agreement on
   the texts it answers meets your target — not just the point estimate;
3. reports how much traffic the small model can take over, and the estimated cost and
   latency savings;
4. lists **suspects**: texts where the small model is confident but disagrees with the
   LLM. These are often LLM labeling mistakes and point to ambiguous class definitions
   in your prompt.

Texts below the threshold go back to your own LLM pipeline. The library never calls an
LLM itself.

## Planned API

```python
from bounded_distill import distill

router = distill(texts, llm_labels, target_agreement=0.97)

label = router.predict(text)
if label is None:              # not confident enough
    label = my_llm_pipeline(text)

print(router.report())         # coverage, agreement (with bound), per-class stats, savings
router.suspects(top_n=50)      # confident disagreements worth a human look
```

Optional: pass a small gold-labeled set to see accuracy next to agreement with the LLM.

## How it differs from similar tools

Several projects distill an LLM classifier into a small model with an LLM fallback.
bounded-distill focuses on:

- **Offline first.** Works on labels you already have; it does not run or wrap the LLM.
- **A statistical bound**, not a hand-picked threshold.
- **Auditing the teacher.** Confident disagreements surface LLM labeling errors
  (in the spirit of Confident Learning, Northcutt et al., 2021).
- **Small and simple.** One function call, strict typing, few dependencies.

## Honest limitations

- Agreement with the LLM is not accuracy. If the LLM makes the same mistake
  consistently, the small model learns it too. Use a gold set to check.
- Single-label classification with a fixed, known set of labels only.
- When the small model answers, you get only the label, not other fields of your
  pipeline's structured output.

## Roadmap

- [ ] Offline core: `distill`, `Router.predict`, save/load
- [ ] Report: coverage, agreement with lower bound, per-class metrics, savings
- [ ] Suspects (confident disagreements)
- [ ] Benchmarks on public datasets with real LLM labels
- [ ] Optional online mode (collect pairs from a running pipeline)

## License

MIT
