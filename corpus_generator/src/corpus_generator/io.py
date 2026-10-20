"""Shared file plumbing: atomic writes, JSON/JSONL readers, hashing, and
flock'd read-modify-write for the few cross-thread/cross-process JSON files."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


def atomic_write(path: Path, text: str) -> None:
    """Write-fsync-rename so readers never see a partial file."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_jsonl_atomic(path: Path, lines: Iterable[str]) -> None:
    """Stream pre-serialized JSONL lines (no trailing newline) to a tmp file,
    fsync, atomically rename — never builds the whole file in memory."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
            f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_jsonl(path: Path, missing_ok: bool = False) -> Iterator[Any]:
    """Yield one parsed object per non-blank line."""
    path = Path(path)
    if missing_ok and not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def locked_json_update(
    path: Path,
    mutate: Callable[[Any], None],
    default_factory: Callable[[], Any],
    tolerate_corrupt: bool = False,
) -> None:
    """Read-modify-write a small JSON file under an exclusive flock (safe
    across threads AND processes). ``mutate`` edits the loaded value in place;
    ``default_factory`` supplies the value when the file is missing.
    ``tolerate_corrupt`` resets unreadable JSON to the default instead of
    raising (heartbeat-style files only — registries must fail loudly)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a+") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            data = default_factory()
            if path.exists():
                try:
                    data = read_json(path)
                except json.JSONDecodeError:
                    if not tolerate_corrupt:
                        raise
                    data = default_factory()
            mutate(data)
            atomic_write(path, json.dumps(data, indent=1))
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
