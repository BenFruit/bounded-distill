"""Router: prediction, report, suspects, evaluation, save/load."""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from bounded_distill._embed import DEFAULT_MODEL, default_embed, embed_texts
from bounded_distill._report import (
    GoldReport,
    Report,
    Suspect,
    build_gold_report,
    build_report,
    build_suspects,
    median_latency_ms,
)
from bounded_distill._types import EmbedFn, TrainingData

FORMAT_VERSION = 1
_LATENCY_SAMPLE = 20  # texts timed one by one for Report.student_latency_ms
_META_FILE = "router.json"
_WEIGHTS_FILE = "weights.npz"


@dataclass(frozen=True, eq=False)  # eq=False: numpy arrays do not support == as a bool
class Router:
    """A distilled classifier with a calibrated confidence threshold. Built by ``distill()``.

    Inference uses numpy only, so a router does not need scikit-learn objects.
    """

    labels: tuple[str, ...]
    threshold: float  # math.inf if the target agreement is unreachable
    weights: npt.NDArray[np.float64]  # (dim, n_logits): logits = x @ weights + intercept
    intercept: npt.NDArray[np.float64]  # (n_logits,); n_logits is 1 for two classes
    embed: EmbedFn
    embedder: str  # default model name, or "custom"
    target_agreement: float
    confidence_level: float
    # Training texts and out-of-fold probabilities for report() and suspects(). In memory
    # only: save() never writes the user's texts to disk, so a loaded router has None here.
    training: TrainingData | None = None
    oof_proba: npt.NDArray[np.float64] | None = None  # (n_used, n_labels)

    def report(self) -> Report:
        """Coverage, agreement with its lower bound, per-class stats and latency.

        Needs the training data, so it works on a router returned by ``distill()``, not on
        one from ``load()``.
        """
        training, oof_proba = self._training_data("report()")
        return build_report(
            training,
            oof_proba,
            threshold=self.threshold,
            target_agreement=self.target_agreement,
            confidence_level=self.confidence_level,
            student_latency_ms=median_latency_ms(self.predict, training.texts[:_LATENCY_SAMPLE]),
        )

    def suspects(self, top_n: int = 50) -> list[Suspect]:
        """Training texts the student confidently labels differently from the LLM.

        A review queue, most confident disagreements first: these are often LLM labeling
        mistakes. Scored out-of-fold, so the student never saw the text it judges. Works on a
        router returned by ``distill()``, not on one from ``load()``.
        """
        training, oof_proba = self._training_data("suspects()")
        return build_suspects(training, oof_proba, top_n=top_n)

    def evaluate(
        self, texts: Sequence[str], gold_labels: Sequence[str], llm_labels: Sequence[str]
    ) -> GoldReport:
        """Accuracy of the LLM, the student and the cascade against human (gold) labels.

        Use texts the router was not trained on, with the labels your LLM gave them. Needs
        no training data, so it also works on a router from ``load()``.
        """
        if not len(texts) == len(gold_labels) == len(llm_labels):
            raise ValueError(
                "texts, gold_labels and llm_labels differ in length: "
                f"{len(texts)}, {len(gold_labels)}, {len(llm_labels)}"
            )
        return build_gold_report(self.predict_batch(texts), gold_labels, llm_labels)

    def _training_data(self, method: str) -> tuple[TrainingData, npt.NDArray[np.float64]]:
        if self.training is None or self.oof_proba is None:
            raise ValueError(
                f"{method} needs the training data, which is kept in memory only: call it on "
                "the router returned by distill(), not on one from Router.load()"
            )
        return self.training, self.oof_proba

    def save(self, path: str | Path) -> None:
        """Write the router to directory ``path``: ``router.json`` plus ``weights.npz``.

        No pickle: the JSON holds labels, threshold and settings, the npz holds two float
        arrays. A custom ``embed`` function is not saved and must be passed to ``load()``.
        """
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        meta = {
            "format_version": FORMAT_VERSION,
            "library_version": version("bounded-distill"),
            "labels": list(self.labels),
            # Strict JSON has no Infinity; null means "target unreachable, never answer".
            "threshold": None if math.isinf(self.threshold) else self.threshold,
            "target_agreement": self.target_agreement,
            "confidence_level": self.confidence_level,
            "embedder": self.embedder,
        }
        (directory / _META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        # numpy's stub types the *args of savez as partially unknown; the call itself is plain.
        np.savez(  # pyright: ignore[reportUnknownMemberType]
            directory / _WEIGHTS_FILE,
            allow_pickle=False,
            weights=self.weights,
            intercept=self.intercept,
        )

    @classmethod
    def load(cls, path: str | Path, embed: EmbedFn | None = None) -> "Router":
        """Read a router written by ``save()``.

        ``embed`` is required if the router was trained with a custom embed function; for the
        default model it is rebuilt by name (and downloaded lazily on first prediction).
        """
        directory = Path(path)
        meta: dict[str, Any] = json.loads((directory / _META_FILE).read_text(encoding="utf-8"))
        if meta["format_version"] != FORMAT_VERSION:
            raise ValueError(
                f"unsupported router format_version {meta['format_version']}, "
                f"this library reads {FORMAT_VERSION}"
            )
        embedder = str(meta["embedder"])
        if embed is None:
            if embedder != DEFAULT_MODEL:
                raise ValueError(
                    f"this router was trained with a custom embed function ({embedder!r}); "
                    "pass the same function: Router.load(path, embed=...)"
                )
            embed = default_embed

        with np.load(directory / _WEIGHTS_FILE, allow_pickle=False) as arrays:
            weights = np.asarray(arrays["weights"], dtype=np.float64)
            intercept = np.asarray(arrays["intercept"], dtype=np.float64)
        labels = tuple(str(label) for label in meta["labels"])
        n_logits = 1 if len(labels) == 2 else len(labels)
        if weights.ndim != 2 or weights.shape[1] != n_logits or intercept.shape != (n_logits,):
            raise ValueError(
                f"weights {weights.shape} and intercept {intercept.shape} do not match "
                f"{len(labels)} labels"
            )

        threshold = meta["threshold"]
        return cls(
            labels=labels,
            threshold=math.inf if threshold is None else float(threshold),
            weights=weights,
            intercept=intercept,
            embed=embed,
            embedder=embedder,
            target_agreement=float(meta["target_agreement"]),
            confidence_level=float(meta["confidence_level"]),
        )

    def predict(self, text: str) -> str | None:
        """The student's label if its confidence reaches the threshold, else ``None``."""
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: Sequence[str]) -> list[str | None]:
        """``predict`` for many texts at once; embeds them in one batch."""
        proba = self.predict_proba(texts)
        return [
            self.labels[int(best)] if confidence >= self.threshold else None
            for best, confidence in zip(proba.argmax(axis=1), proba.max(axis=1), strict=True)
        ]

    def predict_proba(self, texts: Sequence[str]) -> npt.NDArray[np.float64]:
        """Class probabilities, shape ``(len(texts), len(labels))``, columns in ``labels`` order."""
        if not texts:
            return np.zeros((0, len(self.labels)), dtype=np.float64)
        x = embed_texts(self.embed, texts)
        return proba_from_logits(x @ self.weights + self.intercept)


def proba_from_logits(logits: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Logistic regression probabilities from logits in scikit-learn layout.

    Multiclass: one logit per class, probabilities = softmax. Binary: scikit-learn keeps a
    single logit ``z`` for class 1, and ``sigmoid(z)`` equals ``softmax([0, z])[1]``, so the
    binary case is turned into two columns and shares the same code path.
    """
    if logits.shape[1] == 1:
        two_columns = np.zeros((len(logits), 2), dtype=np.float64)
        two_columns[:, 1] = logits[:, 0]
        logits = two_columns
    # Subtracting the row maximum does not change softmax but prevents exp() overflow.
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)
