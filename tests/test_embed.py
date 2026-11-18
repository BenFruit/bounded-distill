import math
import subprocess
import sys

import numpy as np
import numpy.typing as npt
import pytest

from bounded_distill._embed import default_embed, with_e5_prefix


def test_with_e5_prefix_adds_query_prefix() -> None:
    assert with_e5_prefix(["hello", ""]) == ["query: hello", "query: "]


def test_importing_the_package_does_not_import_sentence_transformers() -> None:
    code = (
        "import sys, bounded_distill, bounded_distill._embed; "
        "print('sentence_transformers' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


@pytest.mark.slow
def test_default_embed_returns_float32_rows_with_meaningful_similarity() -> None:
    texts = [
        "Мне не пришёл счёт за прошлый месяц",
        "I did not receive my invoice for last month",
        "Сегодня солнечная погода",
    ]
    vectors = default_embed(texts)

    assert vectors.shape == (3, 384)
    assert vectors.dtype == np.float32
    assert np.isfinite(vectors).all()

    paraphrase = _cosine(vectors[0], vectors[1])  # same meaning, different language
    unrelated = _cosine(vectors[0], vectors[2])  # same language, different meaning
    assert paraphrase > unrelated


def _cosine(a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]) -> float:
    return float(a @ b) / math.sqrt(float(a @ a) * float(b @ b))
