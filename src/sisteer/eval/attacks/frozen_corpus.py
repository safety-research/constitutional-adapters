"""The generic FROZEN-CORPUS adapter — the ingested-kind engine, shared by the rewriter kind.

This is the single highest-leverage piece of held-out scaffolding: ONE adapter
that takes a pinned artifact (a path, or an HF pin + sha) plus a small mapping
spec (which column is the prompt, which the behaviour/goal id) and exposes it
through the standard replay contract (``static_common.run_static``). One adapter
unlocks the ENTIRE class of "expensive/adaptive to generate, but static once
produced" attacks at once — they differ only in artifact PROVENANCE, not in how
they are evaluated.

Rewriter build kind (LLM-rewriter) families reuse this same engine: their build step writes
a frozen corpus in the canonical shape (via :func:`write_frozen_corpus`), and
their replay is a :func:`make_frozen_corpus_family` binding. So build-vs-replay
is separated for B and C identically — the only difference is whether the corpus
was produced by an LLM (B) or ingested from elsewhere (C).

CANONICAL CORPUS SHAPE (on disk, a private-dir JSONL — one row per prompt):
    ``{"goal_id": str, "sample_idx": int, "variant_name": str, "prompt": str}``
    Multiple rows per ``goal_id`` = multiple variants for that behaviour,
    numbered by ``sample_idx`` (the key ``static_common`` persists). This is
    exactly the shape :func:`heldout_common.materialize_corpus` produces, so a
    frozen corpus IS a materialised corpus and hashes identically.

CONTAINMENT. A frozen corpus holds attack PROMPTS (harmful text), so it lives
ONLY in a mode-0700 private dir — never committed. The MANIFEST written beside
it (:func:`heldout_common.build_manifest`) is the sanitised, servable artifact:
content hash, provenance, row count, leakage result — no prompt
text. The ONE exception is a SYNTHETIC, non-harmful demonstrator corpus used to
prove the wiring (``tests/fixtures/heldout/frozen_demo.jsonl``); benign
placeholder behaviours are safe to commit.

INGESTIBLE FAMILIES (drop in a pinned artifact, register via
:func:`make_frozen_corpus_family` — do NOT generate the artifacts here):
    Rainbow Teaming, Ruby Teaming, Ferret, DiveR-CT, Curiosity-driven Red
    Teaming, GPTFuzzer archives, I-GCG, COLD-Attack, GBDA, AdvPrompter — each
    needs its published adversarial-prompt table (a CSV/JSONL/parquet of
    prompts, ideally keyed by HarmBench BehaviorID) pinned by path or HF pin +
    sha, plus a :class:`MappingSpec` naming the prompt and behaviour columns.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..jailbreak_artifacts import append_jsonl, private_umask
from .heldout_common import HeldOutFamily, build_manifest
from .static_common import run_static


# --------------------------------------------------------------------------- #
# The mapping spec + loader                                                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MappingSpec:
    """How to read an external artifact into the canonical corpus shape.

    ``prompt_col`` names the column holding the attack prompt. ``behavior_col``
    names the column whose value maps to a ``Goal.goal_id`` (so replay presents
    the artifact prompt to the RIGHT eval goal) — when None, rows are keyed by
    position (``row0``, ``row1`` …), i.e. the corpus defines its own behaviours.
    ``variant_col`` optionally labels a sub-variant; when None the label is
    ``"frozen"``. ``fmt`` is ``"jsonl"``, ``"csv"``, or ``"parquet"``.
    ``goal_id_prefix`` is prepended to each ``behavior_col`` value (e.g. to turn
    a HarmBench BehaviorID into the eval harness's goal-id namespace)."""

    prompt_col: str
    behavior_col: Optional[str] = None
    variant_col: Optional[str] = None
    fmt: str = "jsonl"
    goal_id_prefix: str = ""


def _read_records(path: Path, fmt: str) -> list[dict]:
    """Read an artifact into a list of dict records. Supports jsonl / csv /
    parquet (parquet needs pandas+pyarrow, both repo deps)."""
    if fmt == "jsonl":
        out = []
        for line in path.read_text().splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out
    if fmt == "csv":
        with path.open(newline="") as f:
            return list(csv.DictReader(f))
    if fmt == "parquet":
        import pandas as pd  # lazy: only the parquet path pays for pandas

        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(f"unknown artifact format {fmt!r} (jsonl|csv|parquet)")


def artifact_to_rows(path, spec: MappingSpec) -> list[dict]:
    """Map an external artifact into canonical corpus rows ``{goal_id,
    sample_idx, variant_name, prompt}``. Rows sharing a ``goal_id`` are numbered
    by their order of appearance. Raises loudly if the named columns are absent
    (never a silent drop / degrade)."""
    path = Path(path)
    records = _read_records(path, spec.fmt)
    if not records:
        raise ValueError(f"artifact {path} is empty")
    if spec.prompt_col not in records[0]:
        raise ValueError(
            f"artifact {path} has no prompt column {spec.prompt_col!r}; "
            f"columns present: {sorted(records[0])}")
    if spec.behavior_col is not None and spec.behavior_col not in records[0]:
        raise ValueError(
            f"artifact {path} has no behaviour column {spec.behavior_col!r}; "
            f"columns present: {sorted(records[0])}")

    counts: dict[str, int] = {}
    rows: list[dict] = []
    for pos, rec in enumerate(records):
        if spec.behavior_col is not None:
            gid = f"{spec.goal_id_prefix}{rec[spec.behavior_col]}"
        else:
            gid = f"{spec.goal_id_prefix}row{pos}"
        idx = counts.get(gid, 0)
        counts[gid] = idx + 1
        name = (str(rec[spec.variant_col]) if spec.variant_col
                and spec.variant_col in rec else "frozen")
        rows.append({
            "goal_id": str(gid),
            "sample_idx": idx,
            "variant_name": name,
            "prompt": str(rec[spec.prompt_col]),
        })
    return rows



def corpus_content_hash(rows: Iterable[dict]) -> str:
    """A STABLE sha256 over a corpus's ``(goal_id, sample_idx, variant_name,
    prompt)`` rows, independent of row order or dict-key order. Two builds that
    produce the same prompts hash identically — the pin the manifest records and
    replay verifies."""
    canon = sorted(
        (str(r["goal_id"]), int(r["sample_idx"]), str(r["variant_name"]),
         str(r["prompt"]))
        for r in rows
    )
    h = hashlib.sha256()
    for gid, idx, name, prompt in canon:
        for part in (gid, str(idx), name, prompt):
            b = part.encode("utf-8")
            h.update(str(len(b)).encode("ascii"))
            h.update(b"\0")
            h.update(b)
        h.update(b"\n")
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# The leakage gate (hard requirement 1)                                        #
# --------------------------------------------------------------------------- #

class FrozenCorpus:
    """An in-memory, content-hash-verified frozen corpus: ``{goal_id:
    [(variant_name, prompt), ... in sample_idx order]}``. Loaded once, replayed
    forever."""

    def __init__(self, by_goal: dict[str, list[tuple[str, str]]],
                 *, content_hash: str):
        self.by_goal = by_goal
        self.content_hash = content_hash

    @classmethod
    def from_rows(cls, rows: list[dict], *, expected_sha: Optional[str] = None):
        chash = corpus_content_hash(rows)
        if expected_sha is not None and chash != expected_sha:
            raise ValueError(
                f"frozen corpus content-hash mismatch: got {chash}, pinned "
                f"{expected_sha}. The corpus is not the one that was checked for "
                "leakage and manifested — refusing to replay it.")
        by_goal: dict[str, list[tuple[int, str, str]]] = {}
        for r in rows:
            by_goal.setdefault(str(r["goal_id"]), []).append(
                (int(r["sample_idx"]), str(r["variant_name"]), str(r["prompt"])))
        ordered = {
            gid: [(name, prompt) for _, name, prompt in sorted(items)]
            for gid, items in by_goal.items()
        }
        return cls(ordered, content_hash=chash)

    @classmethod
    def load(cls, corpus_path, spec: Optional[MappingSpec] = None,
             *, expected_sha: Optional[str] = None):
        """Load a corpus from disk. When ``spec`` is None the file is already in
        the canonical shape (a frozen corpus we wrote); otherwise it is an
        external artifact mapped through ``spec``."""
        corpus_path = Path(corpus_path)
        if spec is None:
            rows = [json.loads(line) for line in
                    corpus_path.read_text().splitlines() if line.strip()]
        else:
            rows = artifact_to_rows(corpus_path, spec)
        return cls.from_rows(rows, expected_sha=expected_sha)

    def variants_for(self, goal, seed: int = 0) -> list[tuple[str, str]]:
        """The ordered variants for ``goal`` — empty if the corpus does not
        cover this behaviour (a ingested-kind corpus legitimately covers only a
        subset of the eval goals). ``seed`` is accepted for signature parity."""
        return list(self.by_goal.get(goal.goal_id, []))


# --------------------------------------------------------------------------- #
# Writing a frozen corpus (build side, the rewriter and ingested kinds)                          #
# --------------------------------------------------------------------------- #


def write_frozen_corpus(rows: list[dict], out_path) -> str:
    """Write canonical corpus ``rows`` to ``out_path`` as JSONL in a mode-0700
    private dir (harmful prompt text — private only), returning the content hash.
    Sorted for a byte-stable file. The caller writes the sanitised manifest
    beside it (:func:`build_manifest`)."""
    out_path = Path(out_path)
    with private_umask():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_path.parent.chmod(0o700)
    except OSError:
        pass
    canon = sorted(rows, key=lambda r: (str(r["goal_id"]), int(r["sample_idx"])))
    if out_path.exists():
        out_path.unlink()
    for r in canon:
        append_jsonl(out_path, {
            "goal_id": str(r["goal_id"]),
            "sample_idx": int(r["sample_idx"]),
            "variant_name": str(r["variant_name"]),
            "prompt": str(r["prompt"]),
        })
    return corpus_content_hash(canon)


class _ReportOnlyLeak:
    """Duck-typed stand-in for heldout_common's leakage report: carries the
    REPORT-ONLY record into ``build_manifest`` (which calls ``.as_manifest()``)
    without any refusing gate — the 2026-08-25 goal-level-holdout directive
    retired refusal on train/eval family overlap."""

    def __init__(self, record: dict) -> None:
        self._record = dict(record)

    def as_manifest(self) -> dict:
        return dict(self._record)


def ingest_artifact(
    artifact_path,
    spec: MappingSpec,
    out_path,
    *,
    fingerprint,
    provenance: dict,
    family_name: str = "frozen_corpus",
    seed: int = 0,
) -> dict:
    """INGESTED-KIND BUILD STEP: read a pinned artifact through ``spec``, record
    the training-``fingerprint`` overlap in the manifest (REPORT-ONLY -- the
    refusing leakage gate is retired under the 2026-08-25 goal-level-holdout
    directive), write the frozen corpus to ``out_path`` (private), and write the
    sanitised manifest beside it. Returns the manifest. This is the whole of ingested-kind provenance —
    everything downstream replays ``out_path`` exactly like a static family."""
    rows = artifact_to_rows(artifact_path, spec)
    # LEAKAGE GATE DEMOTED to report-only: families are allowed --
    # encouraged -- to appear on both sides, and a corpus built over
    # TRAINING goals overlaps the training fingerprint BY CONSTRUCTION,
    # so refuse=True would refuse every legitimate build.
    # ``fingerprint`` is still accepted and recorded below, so the manifest
    # states what the corpus overlaps -- reporting it, no longer refusing.
    leak = _ReportOnlyLeak({
        'gate': 'removed_by_directive_2026_08_25',
        'n_checked': len(rows), 'fingerprint_n': len(fingerprint or ())})
    chash = write_frozen_corpus(rows, out_path)
    fam = HeldOutFamily(
        name=family_name, build_kind="ingested", provenance=provenance,
        variants_fn=lambda g, s=0: [], run_fn=lambda *a, **k: {},  # manifest-only
    )
    manifest = build_manifest(fam, rows, seed=seed, leakage=leak)
    manifest["corpus_content_hash"] = chash  # identical, made explicit
    manifest_path = Path(str(out_path) + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


# --------------------------------------------------------------------------- #
# The registered generic family + the factory (replay side)                    #
# --------------------------------------------------------------------------- #


def make_frozen_corpus_family(
    name: str,
    corpus_path,
    spec: Optional[MappingSpec] = None,
    *,
    expected_sha: Optional[str] = None,
    provenance: dict,
    build_kind: str = "ingested",
) -> HeldOutFamily:
    """Bind a specific pinned corpus into a :class:`HeldOutFamily`. THIS is what
    turns "a pinned artifact + a mapping spec" into a first-class held-out
    family that the sweep dispatches like any static one. Rewriter build kind families call
    this with ``build_kind="rewriter"`` after their LLM build step has written the corpus."""
    corpus_holder: dict[str, FrozenCorpus] = {}

    def _corpus() -> FrozenCorpus:
        if "c" not in corpus_holder:
            corpus_holder["c"] = FrozenCorpus.load(
                corpus_path, spec, expected_sha=expected_sha)
        return corpus_holder["c"]

    def variants_fn(goal, seed: int = 0):
        return _corpus().variants_for(goal, seed)

    def run_fn(defense, goals, *, out_dir, trial: int = 0, seed: int = 0,
               max_new_tokens: int = 512, resume: bool = True):
        return run_static(defense, goals, out_dir=out_dir, family=name,
                          variants_fn=variants_fn, trial=trial, seed=seed,
                          max_new_tokens=max_new_tokens, resume=resume)

    return HeldOutFamily(name=name, build_kind=build_kind,                         provenance=provenance, variants_fn=variants_fn,
                         run_fn=run_fn)


# -- the generic, env-configured entry (the ingested-kind adapter as a registry family)


_ENV_PATH = "SISTEER_FROZEN_CORPUS_PATH"
_ENV_SHA = "SISTEER_FROZEN_CORPUS_SHA"
_ENV_PROMPT_COL = "SISTEER_FROZEN_CORPUS_PROMPT_COL"
_ENV_BEHAVIOR_COL = "SISTEER_FROZEN_CORPUS_BEHAVIOR_COL"
_ENV_FMT = "SISTEER_FROZEN_CORPUS_FMT"


_ENV_CACHE: dict[tuple, FrozenCorpus] = {}


def _env_corpus() -> Optional[FrozenCorpus]:
    """Load the corpus named by the environment, or None if unconfigured. A
    ``prompt_col`` env var means the path is a raw artifact to map; otherwise the
    path is a canonical frozen corpus. Cached by its env-key so replay (one
    ``variants_fn`` call per goal) does not re-read the corpus file each goal."""
    path = os.environ.get(_ENV_PATH)
    if not path:
        return None
    key = (path, os.environ.get(_ENV_SHA), os.environ.get(_ENV_PROMPT_COL),
           os.environ.get(_ENV_BEHAVIOR_COL), os.environ.get(_ENV_FMT))
    if key not in _ENV_CACHE:
        spec = None
        if os.environ.get(_ENV_PROMPT_COL):
            spec = MappingSpec(
                prompt_col=os.environ[_ENV_PROMPT_COL],
                behavior_col=os.environ.get(_ENV_BEHAVIOR_COL) or None,
                fmt=os.environ.get(_ENV_FMT, "jsonl"),
            )
        _ENV_CACHE[key] = FrozenCorpus.load(
            path, spec, expected_sha=os.environ.get(_ENV_SHA) or None)
    return _ENV_CACHE[key]


def frozen_corpus_variants(goal, seed: int = 0):
    """Generic replay seam for the env-configured corpus. Returns [] for a goal
    the corpus does not cover; when NO corpus is configured it returns [] too,
    which the driver's zero-variant path records as a no-op (``run_frozen_corpus``
    is the loud one)."""
    corpus = _env_corpus()
    return corpus.variants_for(goal, seed) if corpus is not None else []


#: ``(path, content hash)`` pairs training has already cleared, so the per-goal lookup checks once
_TRAINING_CLEARED: set[tuple[str, str]] = set()


def training_variants(goal, seed: int = 0):
    """Training-pool sampling's lookup for the env-configured corpus (``sisteer.sampling.sources``):
    :func:`frozen_corpus_variants`, after refusing an evaluation-only corpus
    (:mod:`sisteer.eval.attacks.eval_only`: the shipped ReNeLLM paper corpus, by path, file hash or
    content hash), so this generic adapter cannot carry it into a training pool."""
    corpus = _env_corpus()
    if corpus is None:
        return []
    path = os.environ.get(_ENV_PATH)
    key = (str(path), corpus.content_hash)
    if key not in _TRAINING_CLEARED:
        from . import eval_only

        eval_only.refuse_corpus(consumer="training-pool sampling (frozen_corpus)", path=path,
                                content_hash=corpus.content_hash)
        _TRAINING_CLEARED.add(key)
    return corpus.variants_for(goal, seed)


def run_frozen_corpus(defense, goals, *, out_dir, trial: int = 0, seed: int = 0,
                      max_new_tokens: int = 512, resume: bool = True) -> dict:
    """Generic ingested-kind replay over the env-configured corpus. LOUD-fails when no
    corpus is configured — an unconfigured generic family being silently skipped
    would hide a wiring error (fail loudly, never degrade silently)."""
    if _env_corpus() is None:
        raise RuntimeError(
            f"frozen_corpus family requires ${_ENV_PATH} (a pinned corpus path). "
            f"Set it (and ${_ENV_SHA} to pin the content hash) or register a "
            "concrete corpus with make_frozen_corpus_family instead of the "
            "generic entry. Nothing was generated.")
    return run_static(defense, goals, out_dir=out_dir, family="frozen_corpus",
                      variants_fn=frozen_corpus_variants, trial=trial, seed=seed,
                      max_new_tokens=max_new_tokens, resume=resume)


# The registry SPEC for the generic, env-configured ingested-kind entry.
# an ingested artifact was optimised against another model, so replaying it
# measures TRANSFER (see heldout_common's scientific caveat).
GENERIC_SPEC = HeldOutFamily(
    name="frozen_corpus",
    build_kind="ingested",
    variants_fn=frozen_corpus_variants,
    run_fn=run_frozen_corpus,
    provenance={"method": "env_configured_ingestion"},
)
