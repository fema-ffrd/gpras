"""Convenient functions for working with files."""

from pathlib import PurePath, PurePosixPath
from urllib.parse import unquote, urlparse

import chardet


def get_filename(path_str: str) -> str:
    """Get the trailing name of a file."""
    parsed = urlparse(path_str)
    if parsed.scheme == "s3":
        return PurePosixPath(unquote(parsed.path)).name
    return PurePath(path_str).name


def detect_file_properties(path: str, sample_size: int = 4096) -> tuple[str, str]:
    """Checks encoding and newline for files."""
    with open(path, "rb") as f:
        raw = f.read(sample_size)
    detected = chardet.detect(raw)
    encoding = detected["encoding"] or "utf-8"
    newline = None
    if b"\r\n" in raw:
        newline = "\r\n"
    elif b"\r" in raw:
        newline = "\r"
    else:
        newline = "\n"
    return encoding, newline
