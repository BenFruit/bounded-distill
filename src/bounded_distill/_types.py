"""Shared type aliases."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating[Any]]

EmbedFn = Callable[[Sequence[str]], FloatArray]
"""Maps a batch of texts to an ``(n_texts, dim)`` float array, one row per text."""

InvalidPolicy = Literal["drop", "warn", "error"]
"""What to do with invalid LLM labels: drop silently, drop with a warning, or raise."""

ThresholdMethod = Literal["sgr", "wilson"]
"""How the threshold is chosen: "sgr" with a strict guarantee, "wilson" for more coverage."""

InvalidReason = Literal["missing", "empty", "unknown"]
"""Why an LLM label is invalid: None, empty/whitespace-only, or not in the label set."""


@dataclass(frozen=True)
class TrainingData:
    """Valid (text, LLM label) pairs left after filtering, plus what was filtered out."""

    texts: list[str]
    labels: list[str]
    indices: list[int]  # position of each kept pair in the caller's original input
    label_set: tuple[str, ...]  # class order used for probability columns
    n_total: int
    invalid: dict[InvalidReason, int]  # count per reason; only reasons that occurred

    @property
    def n_invalid(self) -> int:
        return sum(self.invalid.values())
