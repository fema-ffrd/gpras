# This is some code to fail both ruff and mypy


def test(a, b: int) -> None:
    c = a + b
    return c
