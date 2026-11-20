"""Does the promised agreement hold on unseen texts? A test over every saved benchmark run.

The router promises that the true agreement with the LLM on the texts it answers is at least
the target, at 95% confidence. On the test split each run gives ``k`` agreements out of ``n``
covered texts. If even the one-sided 95% upper bound of that agreement is below the target,
the test data contradict the promise. With an honest bound this happens in at most about 5%
of runs. A point estimate a little below the target is expected and is not a violation.
Run: ``uv run python -m benchmarks.bound_check``
"""

import json
from dataclasses import dataclass
from pathlib import Path

from benchmarks.run import RESULTS_DIR
from bounded_distill._threshold import wilson_lower_bound

CONFIDENCE = 0.95


@dataclass(frozen=True)
class Check:
    """One run at one target, measured on the test split."""

    run: str
    target: float
    n: int
    agreement: float
    upper: float  # one-sided upper confidence bound of the test agreement

    @property
    def below_target(self) -> bool:
        return self.agreement < self.target

    @property
    def violated(self) -> bool:
        return self.upper < self.target


def upper_bound(k: int, n: int, confidence_level: float) -> float:
    """One-sided Wilson upper bound: one minus the lower bound of the disagreement rate."""
    return 1 - wilson_lower_bound(n - k, n, confidence_level)


def checks(path: Path) -> list[Check]:
    """The checks for one saved run; targets the router could not reach are skipped."""
    out: list[Check] = []
    for row in json.loads(path.read_text(encoding="utf-8")):
        test = row["test"]
        n, agreement = test["n_covered"], test["agreement_covered"]
        if row["threshold"] is None or agreement is None or n == 0:
            continue
        k = round(agreement * n)
        out.append(Check(path.stem, row["target"], n, agreement, upper_bound(k, n, CONFIDENCE)))
    return out


def main() -> None:
    paths = sorted(p for p in RESULTS_DIR.glob("*.json") if not p.name.startswith("teachers_"))
    all_checks = [c for path in paths for c in checks(path)]
    below = [c for c in all_checks if c.below_target]
    violated = [c for c in all_checks if c.violated]
    gap = sum(c.agreement - c.target for c in all_checks) / len(all_checks)
    print(f"{len(paths)} runs, {len(all_checks)} reachable targets")
    share = len(all_checks)
    print(f"  test agreement below the target: {len(below)} ({len(below) / share:.0%})")
    print(f"  significantly below (promise broken): {len(violated)} ({len(violated) / share:.0%})")
    print(f"  mean test agreement minus target: {gap * 100:+.2f} points")
    for c in sorted(below, key=lambda c: c.agreement - c.target)[:5]:
        print(
            f"    {c.run}  target {c.target:.2f}: {c.agreement:.1%} of {c.n:,}, "
            f"upper bound {c.upper:.1%}"
        )


if __name__ == "__main__":
    main()
