"""Chart for the README: share of traffic answered without the LLM vs cascade accuracy.

Reads the results written by ``benchmarks.run`` and draws one light and one dark PNG.
Run: ``uv run python -m benchmarks.chart``
"""

# matplotlib types most styling options as untyped **kwargs.
# pyright: reportUnknownMemberType=false

import json
from dataclasses import dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")  # files only, no window

import matplotlib.pyplot as plt

from benchmarks.run import RESULTS_DIR

# (results file suffix, legend name) in a fixed order: the order picks the colors.
SERIES = (
    ("intfloat_multilingual-e5-small", "multilingual-e5-small (local CPU)"),
    ("giga-embeddings_10b", "Giga-Embeddings 10B (GPU server)"),
)


@dataclass(frozen=True)
class Theme:
    """Colors of one mode; categorical slots validated for both surfaces (CVD and contrast)."""

    name: str
    surface: str
    text: str
    muted: str
    grid: str
    series: tuple[str, str]


THEMES = (
    Theme("light", "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", ("#2a78d6", "#eb6834")),
    Theme("dark", "#1a1a19", "#f0efec", "#898781", "#2c2c2a", ("#3987e5", "#d95926")),
)


def load(suffix: str) -> list[dict[str, Any]]:
    path = RESULTS_DIR / f"banking77_gemma-4_31b_{suffix}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def draw(theme: Theme) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    fig.patch.set_facecolor(theme.surface)
    ax.set_facecolor(theme.surface)

    llm_alone = load(SERIES[0][0])[0]["test"]["llm_accuracy"] * 100
    ax.axhline(llm_alone, color=theme.muted, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(1, llm_alone + 0.15, f"LLM alone: {llm_alone:.1f}%", color=theme.muted, fontsize=9)

    for (suffix, name), color in zip(SERIES, theme.series, strict=True):
        rows = [row for row in load(suffix) if row["threshold"] is not None]
        x = [row["test"]["coverage"] * 100 for row in rows]
        y = [row["test"]["cascade_accuracy"] * 100 for row in rows]
        ax.plot(
            x,
            y,
            color=color,
            linewidth=2,
            marker="o",
            markersize=7,
            label=name,
            markeredgecolor=theme.surface,
            markeredgewidth=2,
        )
        for row, xi, yi in zip(rows, x, y, strict=True):
            ax.annotate(
                f"{row['target']:.2f}",
                (xi, yi),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=8,
                color=theme.muted,
            )

    ax.set_xlim(0, 103)
    ax.set_xlabel("Test traffic answered without the LLM (%)", color=theme.text)
    ax.set_ylabel("Cascade accuracy vs gold (%)", color=theme.text)
    ax.set_title(
        "Banking77, teacher Gemma 4 31B (points: target agreement)",
        color=theme.text,
        fontsize=11,
        loc="left",
    )
    ax.grid(True, color=theme.grid, linewidth=0.8)
    ax.tick_params(colors=theme.muted, labelcolor=theme.text)
    for spine in ax.spines.values():
        spine.set_visible(False)
    legend = ax.legend(loc="lower left", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(theme.text)

    fig.tight_layout()
    out = RESULTS_DIR / f"banking77_{theme.name}.png"
    fig.savefig(out, facecolor=theme.surface)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    for theme in THEMES:
        draw(theme)
