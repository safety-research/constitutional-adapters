"""Reference-tokenizer staging and token accounting.

Official corpus token counts are computed with a set of three open reference
tokenizers (Llama-3, Gemma, Qwen families) staged from the local HF cache into
``data/tokenizers/``, and the enforced/reported number is the MINIMUM across
the three — targets are met under the most compressive tokenizer.

Counting is fully offline and reproducible from the staged files: the
``tokenizers`` library over the staged ``tokenizer.json``,
``add_special_tokens=False``. Never the API count_tokens endpoint (wrong
tokenizer set, and it would break "reproducible without re-calling the API").
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer

from .io import sha256_file

# (name, [candidate repos in preference order]) — first repo with a complete
# cached snapshot wins; candidates verified in the local cache 2026-08-31.
REFERENCE_TOKENIZERS: list[tuple[str, list[str]]] = [
    ("llama3", ["meta-llama/Llama-3.1-8B-Instruct", "NousResearch/Meta-Llama-3.1-8B-Instruct"]),
    ("gemma", ["google/gemma-3-27b-it", "unsloth/gemma-3-27b-it"]),
    ("qwen", ["Qwen/Qwen3-8B"]),
]

STAGE_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
REQUIRED_FILES = ["tokenizer.json"]


class StagingError(RuntimeError):
    pass


def _snapshots_dir(hf_hub: Path, repo: str) -> Path:
    return hf_hub / f"models--{repo.replace('/', '--')}" / "snapshots"


def _complete(d: Path) -> bool:
    return d.is_dir() and all((d / f).exists() for f in REQUIRED_FILES)


def _cached_snapshot(hf_hub: Path, repo: str) -> Path | None:
    """Newest complete snapshot dir for a repo in the local HF cache (the
    UNPINNED path — kept for the generation-time workflow; a reproduction
    pins the snapshot with ``revisions``, see :func:`stage_tokenizers`)."""
    repo_dir = _snapshots_dir(hf_hub, repo)
    if not repo_dir.is_dir():
        return None
    candidates = sorted(
        (d for d in repo_dir.iterdir() if _complete(d)),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _pinned_snapshot(hf_hub: Path, repo: str, revision: str) -> Path | None:
    """The snapshot dir for EXACTLY ``revision`` (a commit sha) — a cached
    snapshot under another revision never substitutes for it."""
    d = _snapshots_dir(hf_hub, repo) / revision
    return d if _complete(d) else None


def resolve_hf_hub(hf_home: Path | None = None) -> Path:
    """The ``hub/`` directory of an HF cache: ``hf_home`` (or ``$HF_HOME``)
    + ``hub``, else ``~/.cache/huggingface/hub``. A caller may also pass the
    ``hub/`` directory itself."""
    import os

    if hf_home is None and os.environ.get("HF_HOME"):
        hf_home = Path(os.environ["HF_HOME"])
    if hf_home is None:
        return Path.home() / ".cache/huggingface/hub"
    hf_home = Path(hf_home)
    return hf_home if hf_home.name == "hub" else hf_home / "hub"


def load_pins(path: Path) -> dict:
    """A pins file (the harness's ``corpora/TOKENIZERS.json``): ``{"tokenizers":
    {name: {"repo", "revision", "files": [{"name", "sha256"}, ...]}}}``, checked
    for shape and returned with its own sha256 under ``_sha256`` (recorded in the
    staging manifest as what the staged files were verified against)."""
    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise StagingError(f"pins file {path}: unreadable ({e})") from e
    fams = doc.get("tokenizers") if isinstance(doc, dict) else None
    if not isinstance(fams, dict) or not fams:
        raise StagingError(f"pins file {path}: no 'tokenizers' map")
    known = {name for name, _ in REFERENCE_TOKENIZERS}
    for name, t in fams.items():
        ok = (name in known and isinstance(t, dict) and isinstance(t.get("repo"), str) and "/" in t["repo"]
              and isinstance(t.get("revision"), str) and len(t["revision"]) == 40
              and isinstance(t.get("files"), list) and t["files"]
              and all(isinstance(f, dict) and isinstance(f.get("name"), str) and isinstance(f.get("sha256"), str)
                      and len(f["sha256"]) == 64 for f in t["files"]))
        if not ok:
            raise StagingError(
                f"pins file {path}: family {name!r} is not {{repo, revision (40 hex), files: [{{name, sha256}}]}} "
                f"for a known reference tokenizer {sorted(known)}"
            )
    return {"tokenizers": fams, "_sha256": sha256_file(path), "_name": path.name}


def _check_against_pins(name: str, repo: str, snap: Path, files_src: dict[str, Path], pin: dict) -> None:
    """The staged family must be the pinned repo at the pinned revision with EXACTLY the pinned files, each hashing to its
    pinned sha256 — a stale or tampered file under the correct snapshot path in a shared, multi-user HF cache is otherwise
    self-attested by the manifest this tool writes and then used to count and build pools with another tokenization.
    Checked against the SOURCE files before anything is copied."""
    if repo != pin["repo"] or snap.name != pin["revision"]:
        raise StagingError(
            f"reference tokenizer {name!r}: staged {repo}@{snap.name[:12]} but the pins file names "
            f"{pin['repo']}@{pin['revision'][:12]} — refusing to attest another snapshot"
        )
    expected = {f["name"]: f["sha256"] for f in pin["files"]}
    if set(files_src) != set(expected):
        raise StagingError(
            f"reference tokenizer {name!r}: snapshot {snap.name[:12]} holds {sorted(files_src)} but the pins file lists "
            f"{sorted(expected)} — refusing to stage a file set the pins do not attest"
        )
    for fname, want in expected.items():
        got = sha256_file(files_src[fname])
        if got != want:
            raise StagingError(
                f"reference tokenizer {name!r}: {fname} under the pinned revision {snap.name[:12]} hashes "
                f"{got[:12]}… but the pins file records {want[:12]}… — the cache holds other content; re-fetch "
                f"(hf download {repo} --revision {snap.name} --include '{fname}') and restage"
            )


def stage_tokenizers(
    dest: Path,
    hf_hub: Path | None = None,
    revisions: dict[str, str] | None = None,
    pins: dict | None = None,
) -> dict:
    """Copy the three reference tokenizers from the local HF cache into dest.

    Errors loudly if any of the three families cannot be staged. Records full
    provenance (source repo, snapshot sha, per-file sha256) in
    ``dest/MANIFEST.json`` and returns it.

    ``revisions`` maps a family name (``llama3`` / ``gemma`` / ``qwen``) to the
    HF commit sha whose snapshot MUST be staged (a reproduction pins them from
    the corpus manifest); a family without a pin falls back to the newest
    cached snapshot, and a pinned revision that is not cached refuses — it is
    never replaced by whatever else is in the cache.

    ``pins`` (:func:`load_pins`) pins every family it names — repo, revision
    AND every file's sha256 — and the staging refuses a family that does not
    reproduce them exactly (checked on the source files, before any copy, so a
    refusal leaves ``dest`` untouched and no MANIFEST.json is written); a
    ``revisions`` entry that disagrees with the pins file refuses too. The
    manifest records the pins file's name and sha256 under ``verified_against``.
    """
    hf_hub = Path(hf_hub) if hf_hub else resolve_hf_hub()
    revisions = dict(revisions or {})
    unknown = sorted(set(revisions) - {name for name, _ in REFERENCE_TOKENIZERS})
    if unknown:
        raise StagingError(
            f"--revision names unknown reference tokenizer(s) {unknown}; known: "
            f"{[name for name, _ in REFERENCE_TOKENIZERS]}"
        )
    pinned_families = (pins or {}).get("tokenizers", {})
    for name, pin in pinned_families.items():
        if name in revisions and revisions[name] != pin["revision"]:
            raise StagingError(
                f"reference tokenizer {name!r}: --revision {revisions[name][:12]} disagrees with the pins file's "
                f"{pin['revision'][:12]} — one source of truth, refusing"
            )
        revisions[name] = pin["revision"]
    dest = Path(dest)
    manifest: dict = {"tokenizers": {}}
    if pins:
        manifest["verified_against"] = {"pins_file": pins.get("_name"), "sha256": pins.get("_sha256"),
                                        "families": sorted(pinned_families)}
    plan: list[tuple[str, str, Path, dict[str, Path], bool]] = []
    for name, repos in REFERENCE_TOKENIZERS:
        staged = None
        pin = revisions.get(name)
        for repo in repos:
            snap = (_pinned_snapshot(hf_hub, repo, pin) if pin
                    else _cached_snapshot(hf_hub, repo))
            if snap is None:
                continue
            files_src = {fname: snap / fname for fname in STAGE_FILES if (snap / fname).exists()}
            if name in pinned_families:
                _check_against_pins(name, repo, snap, files_src, pinned_families[name])
            staged = (name, repo, snap, files_src, bool(pin))
            break
        if staged is None:
            if pin:
                raise StagingError(
                    f"reference tokenizer {name!r}: pinned revision {pin} is not a "
                    f"complete cached snapshot of any of {repos} under {hf_hub} — "
                    f"fetch it (hf download <repo> --revision {pin} --include "
                    f"'tokenizer*' --include special_tokens_map.json) and restage; "
                    f"another cached revision is never substituted"
                )
            raise StagingError(
                f"reference tokenizer {name!r}: none of {repos} has a complete "
                f"local HF-cache snapshot under {hf_hub} — refusing to continue "
                f"with fewer than 3 reference tokenizers"
            )
        plan.append(staged)
    # every family resolved and (where pinned) verified: only now touch dest
    for name, repo, snap, files_src, is_pinned in plan:
        tok_dir = dest / name
        tok_dir.mkdir(parents=True, exist_ok=True)
        files = {}
        for fname, src in files_src.items():
            shutil.copy2(src, tok_dir / fname)
            files[fname] = sha256_file(tok_dir / fname)
        manifest["tokenizers"][name] = {
            "source_repo": repo,
            "snapshot": snap.name,
            "pinned": is_pinned,
            "files": files,
        }
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


@dataclass
class TokenCounter:
    """Counts text under all staged reference tokenizers; official = min."""

    tokenizers: dict[str, Tokenizer]
    provenance: dict

    @classmethod
    def load(cls, staged_dir: Path) -> "TokenCounter":
        staged_dir = Path(staged_dir)
        manifest_path = staged_dir / "MANIFEST.json"
        if not manifest_path.exists():
            raise StagingError(
                f"{manifest_path} missing — run `corpusgen stage-tokenizers` first"
            )
        provenance = json.loads(manifest_path.read_text(encoding="utf-8"))
        toks = {}
        for name in provenance["tokenizers"]:
            tok_file = staged_dir / name / "tokenizer.json"
            actual = sha256_file(tok_file)
            recorded = provenance["tokenizers"][name]["files"]["tokenizer.json"]
            if actual != recorded:
                raise StagingError(
                    f"staged tokenizer {name} sha256 mismatch vs MANIFEST.json — "
                    f"restage before counting"
                )
            toks[name] = Tokenizer.from_file(str(tok_file))
        return cls(tokenizers=toks, provenance=provenance)

    def counts(self, text: str) -> dict[str, int]:
        return {
            name: len(tok.encode(text, add_special_tokens=False).ids)
            for name, tok in self.tokenizers.items()
        }


def official_tokens(records: Iterable[dict]) -> int:
    """THE official corpus size: min across the per-tokenizer corpus TOTALS
    (sum each tokenizer's counts over all records, then take the minimum).
    Every consumer — filter report, top-up deficits — uses this definition."""
    totals: dict[str, int] = {}
    for r in records:
        for name, n in (r.get("token_counts") or {}).items():
            totals[name] = totals.get(name, 0) + n
    return min(totals.values()) if totals else 0


def record_min_tokens(record: dict) -> int | None:
    """Per-record min-across-tokenizers count (None when uncounted) — the
    per-record ESTIMATOR for top-up sizing, not the official corpus number."""
    counts = record.get("token_counts")
    return min(counts.values()) if counts else None


def render_advice_text(record: dict) -> str:
    """The documented trainable-text renderer for advice records.

    system + "\\n\\n" + user + "\\n\\n" + assistant; a record with no system
    prompt contributes user + assistant only.
    """
    parts = [record.get("system") or "", record["user"], record["assistant"]]
    return "\n\n".join(p for p in parts if p)


def render_story_text(record: dict) -> str:
    return record["text"]
