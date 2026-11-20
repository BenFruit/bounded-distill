import subprocess
import sys

HEAVY = ("sklearn", "scipy", "torch", "sentence_transformers")


def test_import_does_not_load_heavy_libraries() -> None:
    # A fresh interpreter: in this process other tests have already imported them.
    code = f"import sys, bounded_distill; print(','.join(m for m in {HEAVY!r} if m in sys.modules))"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == ""
