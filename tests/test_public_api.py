"""End-to-end tests through the public import only, the way users call the library."""

from pathlib import Path

import pytest
from synthetic import fake_embed, make_dataset

import bounded_distill
from bounded_distill import (
    ClassStats,
    EmbedFn,
    GoldReport,
    InvalidPolicy,
    Report,
    Router,
    Suspect,
    distill,
)


def test_public_names() -> None:
    assert sorted(bounded_distill.__all__) == [
        "ClassStats",
        "EmbedFn",
        "GoldReport",
        "InvalidPolicy",
        "Report",
        "Router",
        "Suspect",
        "__version__",
        "distill",
    ]


def test_distill_predict_save_load_workflow(tmp_path: Path) -> None:
    embed: EmbedFn = fake_embed
    policy: InvalidPolicy = "drop"
    texts, labels = make_dataset(300, flip_rate=0.01)
    new_texts = ["billing new 1", "tech new 2", "other new 3"]

    router = distill(texts, labels, target_agreement=0.95, embed=embed, on_invalid=policy)
    predictions = router.predict_batch(new_texts)
    router.save(tmp_path / "router")
    loaded = Router.load(tmp_path / "router", embed=embed)

    assert loaded.predict_batch(new_texts) == predictions
    for text, prediction in zip(new_texts, predictions, strict=True):
        if (label := router.predict(text)) is None:
            label = text.split()[0]  # stands in for the user's LLM pipeline
        assert label == (prediction or text.split()[0])


def test_report_suspects_evaluate_workflow() -> None:
    texts, labels = make_dataset(300, flip_rate=0.03)
    router = distill(texts, labels, target_agreement=0.9, embed=fake_embed)
    held_out = [f"{c} {i}" for c in ("billing", "tech", "other") for i in range(1000, 1050)]
    gold = [text.split()[0] for text in held_out]

    report: Report = router.report()
    stats: tuple[ClassStats, ...] = report.per_class
    suspects: list[Suspect] = router.suspects(top_n=5)
    gold_report: GoldReport = router.evaluate(held_out, gold, gold)

    assert report.target_reached
    assert {s.label for s in stats} == set(router.labels)
    assert len(suspects) == 5
    assert gold_report.llm_accuracy == 1.0
    assert str(report).startswith("bounded-distill report")
    assert str(gold_report).startswith("bounded-distill evaluation on 150")


@pytest.mark.slow
def test_end_to_end_with_the_default_embedder(tmp_path: Path) -> None:
    billing = [
        "Мне не пришёл счёт",
        "Где мой счёт за этот месяц?",
        "С карты списали деньги дважды",
        "Верните деньги за подписку",
        "Не могу оплатить заказ картой",
        "Почему сумма в счёте больше?",
        "I was charged twice",
        "Please refund my payment",
    ]
    tech = [
        "Приложение вылетает при запуске",
        "Не могу войти в аккаунт, ошибка 500",
        "Сайт не открывается",
        "Не приходит код подтверждения",
        "Кнопка сохранить не работает",
        "Страница грузится очень долго",
        "The app crashes on start",
        "I cannot log in to my account",
    ]
    texts = billing + tech
    labels = ["billing"] * len(billing) + ["tech"] * len(tech)

    # 16 examples are far too few for a 97% bound, so the router must refuse honestly.
    with pytest.warns(UserWarning, match="not reachable"):
        router = distill(texts, labels)
    assert router.predict("Мне дважды списали оплату") is None

    proba = router.predict_proba(["Мне дважды списали оплату", "Приложение не запускается"])
    assert [router.labels[int(i)] for i in proba.argmax(axis=1)] == ["billing", "tech"]

    router.save(tmp_path)
    assert Router.load(tmp_path).embedder == "intfloat/multilingual-e5-small"
