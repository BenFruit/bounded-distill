"""Default embedder: a lazily loaded multilingual sentence-transformers model."""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from bounded_distill._types import EmbedFn

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
_E5_PREFIX = "query: "  # e5 models expect this prefix for classification-style inputs


def default_embed(texts: Sequence[str]) -> npt.NDArray[np.float32]:
    """Embed texts with the default model; the model is downloaded and loaded on first call."""
    model = _load_model(DEFAULT_MODEL)
    # sentence-transformers annotates `encode` with untyped **kwargs and a bare ndarray return;
    # with older numpy stubs (numpy < 2.3, Python 3.10) that ndarray is partially unknown.
    # The result is converted to a precisely typed array right away.
    vectors = model.encode(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        with_e5_prefix(texts), batch_size=64, convert_to_numpy=True
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_texts(embed: EmbedFn, texts: Sequence[str]) -> npt.NDArray[np.float32]:
    """Run ``embed``, check its output shape and values, and L2-normalize the rows."""
    x = np.asarray(embed(texts), dtype=np.float32)
    if x.ndim != 2 or x.shape[0] != len(texts):
        raise ValueError(f"embed must return an array of shape ({len(texts)}, dim), got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("embed returned NaN or infinite values")
    return l2_normalize(x)


def l2_normalize(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Scale each row to unit length; all-zero rows stay zero."""
    norms = np.sqrt((x * x).sum(axis=1, keepdims=True))
    # A tiny floor keeps zero rows at zero instead of dividing by zero. numpy's stubs widen
    # this arithmetic to float64, so the dtype is restated (a no-op cast at runtime).
    return (x / np.maximum(norms, 1e-12)).astype(np.float32, copy=False)


def with_e5_prefix(texts: Sequence[str]) -> list[str]:
    return [_E5_PREFIX + text for text in texts]


@functools.cache
def _load_model(name: str) -> SentenceTransformer:
    # Imported here, not at module level: importing torch takes seconds, and users
    # with their own embed function should not pay for it.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)
