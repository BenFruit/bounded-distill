import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
from synthetic import fake_embed, make_dataset

from bounded_distill._distill import distill
from bounded_distill._embed import DEFAULT_MODEL, default_embed
from bounded_distill._router import Router

NEW_TEXTS = [f"{c} new {i}" for c in ("billing", "tech", "other") for i in range(30)]


@pytest.fixture(scope="module")
def router() -> Router:
    texts, labels = make_dataset(200, flip_rate=0.01)
    return distill(texts, labels, embed=fake_embed, target_agreement=0.9)


def _reject_non_standard_json(value: str) -> float:
    raise AssertionError(f"router.json must be strict JSON, found {value}")


def test_round_trip_gives_identical_predictions(router: Router, tmp_path: Path) -> None:
    router.save(tmp_path)
    loaded = Router.load(tmp_path, embed=fake_embed)

    np.testing.assert_array_equal(loaded.predict_proba(NEW_TEXTS), router.predict_proba(NEW_TEXTS))
    assert loaded.predict_batch(NEW_TEXTS) == router.predict_batch(NEW_TEXTS)
    assert loaded.labels == router.labels
    assert loaded.threshold == router.threshold
    assert (loaded.target_agreement, loaded.confidence_level) == (0.9, 0.95)
    assert loaded.embedder == "custom"


def test_saves_two_files_and_no_training_texts(router: Router, tmp_path: Path) -> None:
    router.save(tmp_path / "nested" / "dir")  # missing parent directories are created
    directory = tmp_path / "nested" / "dir"

    assert sorted(p.name for p in directory.iterdir()) == ["router.json", "weights.npz"]
    meta = json.loads((directory / "router.json").read_text(encoding="utf-8"))
    assert meta["format_version"] == 2
    assert "billing 0" not in (directory / "router.json").read_text(encoding="utf-8")


def test_loaded_router_has_no_training_data(router: Router, tmp_path: Path) -> None:
    router.save(tmp_path)
    loaded = Router.load(tmp_path, embed=fake_embed)
    assert loaded.training is None
    assert loaded.oof_proba is None


def test_infinite_threshold_is_stored_as_json_null(router: Router, tmp_path: Path) -> None:
    dataclasses.replace(router, threshold=math.inf).save(tmp_path)

    text = (tmp_path / "router.json").read_text(encoding="utf-8")
    meta = json.loads(text, parse_constant=_reject_non_standard_json)
    assert meta["threshold"] is None
    assert Router.load(tmp_path, embed=fake_embed).threshold == math.inf


def test_binary_router_round_trip(tmp_path: Path) -> None:
    texts, labels = make_dataset(100, flip_rate=0.0, classes=("billing", "tech"))
    binary = distill(texts, labels, embed=fake_embed, target_agreement=0.9)
    new = [f"billing new {i}" for i in range(10)] + [f"tech new {i}" for i in range(10)]

    binary.save(tmp_path)
    loaded = Router.load(tmp_path, embed=fake_embed)
    np.testing.assert_array_equal(loaded.predict_proba(new), binary.predict_proba(new))


def test_custom_embedder_must_be_passed_again(router: Router, tmp_path: Path) -> None:
    router.save(tmp_path)
    with pytest.raises(ValueError, match="custom embed function"):
        Router.load(tmp_path)


def test_default_embedder_is_rebuilt_by_name(router: Router, tmp_path: Path) -> None:
    dataclasses.replace(router, embedder=DEFAULT_MODEL).save(tmp_path)
    loaded = Router.load(tmp_path)  # no model download: the model loads on first prediction
    assert loaded.embed is default_embed
    assert loaded.embedder == DEFAULT_MODEL


def test_rejects_unknown_format_version(router: Router, tmp_path: Path) -> None:
    router.save(tmp_path)
    _edit_meta(tmp_path, format_version=99)
    with pytest.raises(ValueError, match="format_version 99"):
        Router.load(tmp_path, embed=fake_embed)


def test_rejects_weights_that_do_not_match_labels(router: Router, tmp_path: Path) -> None:
    router.save(tmp_path)
    _edit_meta(tmp_path, labels=["billing", "other", "tech", "extra"])
    with pytest.raises(ValueError, match="do not match 4 labels"):
        Router.load(tmp_path, embed=fake_embed)


def _edit_meta(directory: Path, **changes: object) -> None:
    path = directory / "router.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta.update(changes)
    path.write_text(json.dumps(meta), encoding="utf-8")
