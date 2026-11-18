from importlib.metadata import version

import bounded_distill


def test_version_matches_package_metadata() -> None:
    assert bounded_distill.__version__ == version("bounded-distill")
