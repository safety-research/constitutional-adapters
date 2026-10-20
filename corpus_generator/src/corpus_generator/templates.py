"""Jinja2 template rendering + the template-hash registry.

One file per stage under prompts/; the sha256 of each template FILE is the
template hash recorded in request sets, stamps, and manifests, so any prompt
drift within a run fails the request_set_hash check loudly.

Variant runs (the register-matched control corpus) pass an ``overlay_dir``:
a template rel-path present there wins over the production file, and every
rel-path absent falls through to prompts/ unchanged — so the overlay
directory listing IS the exact set of intentionally-divergent templates,
and the hashes recorded for a run are the hashes of the files it actually
rendered.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from jinja2 import Environment, StrictUndefined


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


class TemplateStore:
    def __init__(self, prompts_dir: Path | None = None, overlay_dir: Path | None = None):
        self.prompts_dir = Path(prompts_dir) if prompts_dir else repo_root() / "prompts"
        self.overlay_dir = Path(overlay_dir) if overlay_dir else None
        self._env = Environment(
            undefined=StrictUndefined,
            keep_trailing_newline=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._cache: dict[str, tuple[Any, str]] = {}  # rel_path -> (compiled, sha256)

    def _resolve(self, rel_path: str) -> Path:
        if self.overlay_dir is not None and (self.overlay_dir / rel_path).exists():
            return self.overlay_dir / rel_path
        return self.prompts_dir / rel_path

    def _load(self, rel_path: str):
        if rel_path not in self._cache:
            src = self._resolve(rel_path).read_text(encoding="utf-8")
            self._cache[rel_path] = (
                self._env.from_string(src),  # compiled ONCE per store
                hashlib.sha256(src.encode("utf-8")).hexdigest(),
            )
        return self._cache[rel_path]

    def render(self, rel_path: str, **context) -> str:
        return self._load(rel_path)[0].render(**context)

    def hash(self, rel_path: str) -> str:
        return self._load(rel_path)[1]

    def hashes(self, rel_paths: list[str]) -> dict[str, str]:
        return {p: self.hash(p) for p in sorted(set(rel_paths))}

    def registry(self) -> dict[str, str]:
        """sha256 of every template file under prompts/ (plus any overlay-only
        files), each hashed through the store's own resolution — overlay wins."""
        rels = {
            str(p.relative_to(self.prompts_dir))
            for p in self.prompts_dir.rglob("*.md")
        }
        if self.overlay_dir is not None:
            rels |= {
                str(p.relative_to(self.overlay_dir))
                for p in self.overlay_dir.rglob("*.md")
            }
        return {rel: self.hash(rel) for rel in sorted(rels)}
