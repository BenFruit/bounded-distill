"""Distill an LLM classifier into a small model with a statistical bound on agreement.

Typical use::

    router = distill(texts, llm_labels, target_agreement=0.97)
    if (label := router.predict(text)) is None:
        label = my_llm_pipeline(text)  # not confident enough: fall back to the LLM
"""

from bounded_distill._distill import distill
from bounded_distill._report import ClassStats, GoldReport, Report, Suspect
from bounded_distill._router import Router
from bounded_distill._types import EmbedFn, InvalidPolicy

__version__ = "0.1.0"

__all__ = [
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
