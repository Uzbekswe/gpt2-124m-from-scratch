"""Foundation checks for the package."""

import gpt2_124m


def test_package_imports() -> None:
    """The installable package exposes its initial version."""
    assert gpt2_124m.__version__ == "0.1.0"
