"""Docstring."""

# This is some code to fail both ruff and mypy


def test(a, b: int) -> None:
    """Make bad function."""
    c = a + b
    return c
