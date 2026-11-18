"""Synthetic data shared by tests: a deterministic fake embedder, no model download."""

import zlib
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

CLASSES = ("billing", "tech", "other")
DIM = 8


def fake_embed(texts: Sequence[str]) -> npt.NDArray[np.float64]:
    """Deterministic stand-in for a sentence embedder; no model download.

    A text "<class> <i>" maps to Gaussian noise (seeded by the text itself) plus a unit
    signal in the dimension of its true class, so classes overlap only partly.
    """
    rows: list[npt.NDArray[np.float64]] = []
    for text in texts:
        rng = np.random.default_rng(zlib.crc32(text.encode()))
        row = rng.normal(0.0, 0.5, DIM)
        row[CLASSES.index(text.split()[0])] += 1.0
        rows.append(row)
    return np.array(rows)


def make_dataset(
    n_per_class: int, flip_rate: float, seed: int = 0, classes: Sequence[str] = CLASSES
) -> tuple[list[str], list[str]]:
    """Texts with their true class as the LLM label, except a share of random wrong labels."""
    rng = np.random.default_rng(seed)
    texts = [f"{c} {i}" for c in classes for i in range(n_per_class)]
    labels: list[str] = []
    for text in texts:
        label = text.split()[0]
        if rng.random() < flip_rate:
            others = [c for c in classes if c != label]
            label = others[int(rng.integers(len(others)))]
        labels.append(label)
    return texts, labels
