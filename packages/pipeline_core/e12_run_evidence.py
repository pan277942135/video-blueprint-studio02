from __future__ import annotations

import hashlib
import pathlib
from collections.abc import Mapping
from typing import Any


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_surviving_outputs(paths: Mapping[str, pathlib.Path]) -> dict[str, dict[str, Any]]:
    """Return byte-exact evidence for outputs that already exist at failure time."""
    surviving: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        if not path.is_file():
            continue
        surviving[name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    return surviving


__all__ = ["collect_surviving_outputs"]
