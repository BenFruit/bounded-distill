"""Public datasets with human (gold) labels for the benchmark, cached locally.

Each file is pinned to a dataset revision and checked by SHA-256, so every run of the
benchmark uses exactly the same data. Run ``python -m benchmarks.data`` for a summary.
"""

# pyarrow ships no type stubs; its calls are confined to _read_parquet().
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

import hashlib
import json
import urllib.request
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq

CACHE_DIR = Path(__file__).parent / ".cache"

Split = Literal["train", "test"]

# Banking77 (Casanueva et al., 2020; CC BY 4.0): online banking queries, 77 intents.
_BANKING77_URL = (
    "https://huggingface.co/datasets/PolyAI/banking77/resolve/"
    "1846984189f2bc0ac51c6fc3a600fa9e1714e0c6/data/{split}-00000-of-00001.parquet"
)
_BANKING77_SHA256: dict[Split, str] = {
    "train": "8dc89b87a188509c42cce5a086e8f7c1ff4a225bdd9ae3f23e6cdfa5806e363a",
    "test": "1f40f777c33f0edfa6c530ff84fe2878db6490838c12d22d036f626948905341",
}

# MASSIVE 1.0 (FitzGerald et al., 2022; CC BY 4.0), Russian: voice assistant commands, 60 intents.
_MASSIVE_URL = (
    "https://huggingface.co/datasets/AmazonScience/massive/resolve/"
    "ed58ac423a2f4121720918bf5301577edce4ffd3/ru-RU/{split}/0000.parquet"
)
_MASSIVE_SHA256: dict[Split, str] = {
    "train": "89ecdc8ddfd00b38d05624c8061dbd10ea7b63f4587ee82750e0e48ebdfb91eb",
    "test": "d53f1c58d753e64b7cf7f10face6c4a57ac93e7f9db82826b43058eacaf2e46a",
}
# Too rare to train on (4 train texts, fewer than the 5 folds) and absent from test.
_MASSIVE_DROPPED = frozenset({"cooking_query"})


@dataclass(frozen=True)
class Example:
    """One text with its human label."""

    id: str  # stable across runs, e.g. "banking77/train/42"; keys the LLM label cache
    text: str
    gold: str


def load_banking77(split: Split) -> list[Example]:
    """Banking77 train (10,003 texts) or test (3,080 texts) with gold intent labels."""
    path = download(
        _BANKING77_URL.format(split=split),
        CACHE_DIR / "banking77" / f"{split}.parquet",
        _BANKING77_SHA256[split],
    )
    texts, label_ids, names = _read_parquet(path, text_column="text", label_column="label")
    return [
        Example(id=f"banking77/{split}/{i}", text=text, gold=names[label_id])
        for i, (text, label_id) in enumerate(zip(texts, label_ids, strict=True))
    ]


def load_massive_ru(split: Split) -> list[Example]:
    """MASSIVE Russian train (11,510 texts) or test (2,974 texts), 59 intents.

    The one intent too rare to train on is left out (see ``_MASSIVE_DROPPED``); ids keep the
    row numbers of the original file.
    """
    path = download(
        _MASSIVE_URL.format(split=split),
        CACHE_DIR / "massive_ru" / f"{split}.parquet",
        _MASSIVE_SHA256[split],
    )
    texts, label_ids, names = _read_parquet(path, text_column="utt", label_column="intent")
    return [
        Example(id=f"massive_ru/{split}/{i}", text=text, gold=names[label_id])
        for i, (text, label_id) in enumerate(zip(texts, label_ids, strict=True))
        if names[label_id] not in _MASSIVE_DROPPED
    ]


def download(url: str, path: Path, sha256: str) -> Path:
    """Fetch ``url`` into ``path`` once; the file must match ``sha256`` (cached ones too)."""
    data = path.read_bytes() if path.exists() else _fetch(url)
    actual = hashlib.sha256(data).hexdigest()
    if actual != sha256:
        raise ValueError(f"{path.name}: SHA-256 {actual} does not match the pinned {sha256}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return path


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def _read_parquet(
    path: Path, *, text_column: str, label_column: str
) -> tuple[list[str], list[int], list[str]]:
    """Texts, integer label ids, and label names from a Hugging Face parquet file."""
    table = pq.read_table(path, columns=[text_column, label_column])
    # Hugging Face keeps the class names of an integer label column in the schema metadata.
    features = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]
    texts = [str(text) for text in table.column(text_column).to_pylist()]
    label_ids = [int(label_id) for label_id in table.column(label_column).to_pylist()]
    names = [str(name) for name in features[label_column]["names"]]
    return texts, label_ids, names


@dataclass(frozen=True)
class Dataset:
    """A benchmark dataset and the task description its LLM prompt starts with."""

    name: str  # used in ids, cache files and result files
    title: str  # for charts and tables
    load: Callable[[Split], list[Example]]
    task: str  # first sentence of the LLM's system prompt

    def labels(self) -> list[str]:
        """The label set, defined by the train split."""
        return sorted({example.gold for example in self.load("train")})


DATASETS = {
    dataset.name: dataset
    for dataset in (
        Dataset(
            "banking77",
            "Banking77",
            load_banking77,
            "You classify customer messages to an online bank by intent.",
        ),
        Dataset(
            "massive_ru",
            "MASSIVE (Russian)",
            load_massive_ru,
            "You classify Russian commands to a voice assistant by intent.",
        ),
    )
}


def _summary(name: str, examples: list[Example]) -> str:
    counts = Counter(example.gold for example in examples)
    return (
        f"{name}: {len(examples):,} texts, {len(counts)} classes, "
        f"{min(counts.values())}-{max(counts.values())} texts per class; "
        f"e.g. {examples[0].text!r} -> {examples[0].gold}"
    )


if __name__ == "__main__":
    for dataset in DATASETS.values():
        for split in ("train", "test"):
            print(_summary(f"{dataset.name}/{split}", dataset.load(split)))
