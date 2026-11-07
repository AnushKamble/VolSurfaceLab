"""Provenance tracking, environment capture, and cryptographic hashing."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def compute_file_hash(path: str | Path) -> str:
    """Compute SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_content_hash(content: str | bytes) -> str:
    """Compute SHA-256 hash of in-memory string or bytes."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def get_git_commit(cwd: str | Path | None = None) -> str:
    """Retrieve current git commit hash if in a git repository."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown_or_uncommitted"


def capture_environment_metadata() -> Dict[str, str]:
    """Capture environment metadata for reproducibility logs."""
    import numpy
    import scipy
    import pandas
    import pydantic

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "os_platform": platform.platform(),
        "architecture": platform.machine(),
        "numpy_version": getattr(numpy, "__version__", "unknown"),
        "scipy_version": getattr(scipy, "__version__", "unknown"),
        "pandas_version": getattr(pandas, "__version__", "unknown"),
        "pydantic_version": getattr(pydantic, "__version__", "unknown"),
        "git_commit": get_git_commit(),
    }
