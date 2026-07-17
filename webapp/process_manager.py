"""PID file management for the Mini UFlash webapp.

``launch_webapp.ps1`` writes the foreground process id to ``webapp.pid`` at the
project root; ``stop_webapp.ps1`` reads it back and terminates *only* that one
process. The functions here are the Python-side helpers used by the app and by
the tests; the PowerShell scripts read/write the same file directly.

Safety rules enforced here, mirroring the spec:

* Never ``taskkill /IM python.exe`` — that would kill unrelated Python jobs.
* Validate that a stored PID is still alive *and* that its command line
  belongs to this project before considering it a duplicate/running instance.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except Exception:  # pragma: no cover - psutil is a declared dependency
    _HAS_PSUTIL = False


# A command line is considered "ours" when it launches the webapp module.
# This is intentionally loose (matches ``-m webapp.app`` and a direct script
# path) so the check survives being launched either way.
_OWN_MARKERS = ("webapp.app", "webapp\\app.py", "webapp/app.py")


def pid_file_path(project_root: str | os.PathLike[str]) -> Path:
    """Return the canonical ``webapp.pid`` path for ``project_root``."""
    return Path(os.fspath(project_root)) / "webapp.pid"


def write_pid(project_root: str | os.PathLike[str], pid: int) -> Path:
    """Atomically write ``pid`` to the project's PID file and return its path."""
    path = pid_file_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(str(int(pid)), encoding="utf-8")
    tmp.replace(path)
    return path


def read_pid(project_root: str | os.PathLike[str]) -> Optional[int]:
    """Return the stored PID, or ``None`` if the file is missing/invalid."""
    path = pid_file_path(project_root)
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def remove_pid(project_root: str | os.PathLike[str]) -> None:
    """Delete the PID file if present (no error if it is already gone)."""
    path = pid_file_path(project_root)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _cmdline_belongs_to_project(cmdline: list[str]) -> bool:
    joined = " ".join(cmdline)
    return any(marker in joined for marker in _OWN_MARKERS)


def pid_is_running(pid: int) -> bool:
    """Return True if ``pid`` is a live process. Uses psutil when available."""
    if not _HAS_PSUTIL:
        return False
    try:
        return psutil.pid_exists(int(pid))
    except Exception:  # pragma: no cover - defensive
        return False


def pid_is_ours(pid: int) -> bool:
    """Return True if ``pid`` is live *and* its command line belongs to this app.

    This is the guard that prevents ``stop_webapp.ps1`` from killing an
    unrelated Python process that happened to reuse the PID.
    """
    if not pid_is_running(pid) or not _HAS_PSUTIL:
        return False
    try:
        proc = psutil.Process(int(pid))
        cmdline = proc.cmdline()
    except Exception:  # pragma: no cover - defensive
        return False
    return _cmdline_belongs_to_project(cmdline)


def another_instance_is_running(project_root: str | os.PathLike[str]) -> bool:
    """Return True when a PID file points at a still-running webapp instance."""
    pid = read_pid(project_root)
    if pid is None:
        return False
    return pid_is_ours(pid)


if __name__ == "__main__":  # pragma: no cover - manual helper
    root = Path(__file__).resolve().parent.parent
    pid = read_pid(root)
    if pid is None:
        print("No PID file found.")
        sys.exit(0)
    print(f"Stored PID: {pid}")
    print(f"Running: {pid_is_running(pid)}")
    print(f"Belongs to this project: {pid_is_ours(pid)}")
