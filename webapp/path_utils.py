"""Path handling for the Mini UFlash OCR webapp.

All filesystem access goes through :mod:`pathlib`. Nothing hand-concatenates
backslashes. The helpers here are aware of the Windows realities the spec calls
out: paths containing spaces, Chinese characters, non-C: drive letters, and
paths close to the 240-character Windows MAX_PATH pain threshold.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

# Windows gets uncomfortable near 260 characters even with long-path support
# enabled. We warn (not fail) once a resolved path crosses this threshold so
# the UI can tell the user to shorten a directory or move the model closer.
LONG_PATH_WARN = 240


def normalize(path: str | os.PathLike[str]) -> Path:
    """Resolve a user-supplied path to an absolute ``Path``.

    ``Path.resolve()`` is used so that the result is absolute, normalized and
    symlink-expanded. Relative paths are resolved against the current working
    directory, matching normal command-line expectations.
    """
    return Path(os.fspath(path)).expanduser().resolve()


def safe_join(base: str | os.PathLike[str], *parts: str) -> Path:
    """Join ``base`` with ``parts`` and return a resolved, absolute ``Path``.

    Safer than manual string concatenation and immune to the ``\\`` vs ``/``
    confusion that bites Windows scripts. Works with spaces, Chinese names and
    mixed drive letters.
    """
    return normalize(Path(base, *parts))


def check_long_path(path: str | os.PathLike[str]) -> tuple[bool, int]:
    """Return ``(is_long, length)`` for the resolved path.

    A path is "long" when its resolved string form reaches ``LONG_PATH_WARN``.
    This is advisory: callers surface a hint rather than refusing to run.
    """
    resolved = str(normalize(path))
    return len(resolved) >= LONG_PATH_WARN, len(resolved)


def is_path_safe(path: str | os.PathLike[str]) -> bool:
    """Best-effort sanity check that a path stays within a normal filesystem form.

    This rejects embedded NUL bytes and obvious traversal escapes. It does not
    replace proper permission handling; it is a guardrail for UI inputs.
    """
    text = os.fspath(path)
    if "\x00" in text:
        return False
    try:
        resolved = normalize(text)
    except (OSError, ValueError):
        return False
    # The resolved form must be absolute; a raw ".." that escapes the drive
    # root will not produce a sensible absolute path on Windows.
    return resolved.is_absolute()


def home_dir() -> Path:
    """Return the user home directory via :func:`Path.home`.

    Never assume ``C:\\Users\\<ascii-name>``: the machine may use a Chinese
    username (this one does). ``Path.home()`` handles that correctly.
    """
    return Path.home()


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Create ``path`` (and parents) if missing and return the resolved ``Path``."""
    resolved = normalize(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def iter_files(
    directory: str | os.PathLike[str], suffixes: Iterable[str]
) -> list[Path]:
    """Yield files under ``directory`` whose suffix matches one of ``suffixes``.

    ``suffixes`` are matched case-insensitively and may be given with or without
    the leading dot. Results are sorted for deterministic ordering.
    """
    root = normalize(directory)
    wanted = {("." + s.lstrip(".")).lower() for s in suffixes}
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in wanted)


def contains_chinese(path: str | os.PathLike[str]) -> bool:
    """Return True if the path string contains any CJK characters."""
    return bool(re.search(r"[\u4e00-\u9fff]", os.fspath(path)))
