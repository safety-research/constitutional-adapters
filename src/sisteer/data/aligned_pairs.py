"""Judge-in-the-loop rejection sampling for ALIGNED training pairs.

Why this module exists: ``pairs.generate_pairs_rejection`` deliberately
RAISES on ``checker == "external"`` — the realign suite's safety SIs
(si_constitution, si_caa) have no programmatic checker, so
acceptance cannot be scored in-process. Here the accept/reject decision
is an LLM-judge verdict (rubric ``align.v1``, filed on the grading host by
``scripts/realign_filter.py``), which splits the sampler into three
phases that run on different machines at different times:

1. ``generate_candidates`` / ``generate_negs`` — GPU side. K sampled
   completions per prompt under ``system=instruction.text`` (and one
   greedy no-SI neg per prompt), appended to JSONL with item-level
   resume and an atomic per-round ``candidates.done.r{n}`` sentinel
   (record count + sha256 — the F16 producer manifest). Before the
   FIRST row is appended, an immutable run manifest
   (``candidates.manifest.json`` / ``negs.manifest.json``) freezes the
   full prompt-idx→text map, instruction id/text, teacher model
   revision, k, seed, temperature and token limit; resume requires an
   EXACT manifest match and validates every existing row against it,
   so a partial round can never be silently continued with a different
   model, prompt list, or sampling configuration.
2. The judge files a verdict for EVERY candidate (elsewhere; this
   module never talks to an API).
3. ``select_pairs`` (pure) + ``publish_canonical_pairs`` — keep the
   lowest-(round, sample_idx) accepted candidate per prompt and publish
   the canonical ``pairs.jsonl`` behind the F9 two-round gate.

Retry policy is FIXED and pre-registered: MAX_ROUNDS=2 rounds of
DEFAULT_K=8 samples; a prompt with no accepted candidate after round 2
yields no pair (it is reported, never silently padded).

The publication gate, spelled out: during sampling only round-specific
PROVISIONAL selections may be written (``save_provisional`` — never the
canonical name). The canonical ``pairs.jsonl`` (via ``pairs.save_pairs``,
atomic) is published ONLY after ``verify_rounds_complete`` has checked,
against the frozen run manifest, that (a) both rounds' sentinels exist
and match the candidate rows' count+sha256, (b) every candidate row
matches the manifest with UNIQUE candidate keys and round 0 covers
EXACTLY every expected prompt x sample 0..K-1, (c) every candidate has
a filed verdict (blocked counts as filed; MISSING does not), (d) round
1 resampled EXACTLY the eligible prompts left with no accepted round-0
verdict (all K samples — no round-2 bypass, no silent disappearance),
and (e) the final unresolved prompt ids — ranging over the FULL
expected train prompt set — are recorded. Canonical publication further
REQUIRES the split manifest (calibration excluded, held-out/unknown
hard errors), exactly one validated negative per selectable prompt
(prompt text verified against the run manifest, rows verified against
``negs.done``), and per_prompt=1. ``filter_stats.json`` is published
(atomically) BEFORE ``pairs.jsonl``, so the existence of pairs implies
the stats that describe them exist too; on REpublication the old
``pairs.jsonl`` is removed (and the directory fsync'd) before any new
metadata is written, so a crash can never leave stale pairs beside a
newer manifest/stats.

Pair schema is UNCHANGED (``pairs.Pair``), so ``pairs_to_distill_examples``
and ``pairs_to_caa_inputs`` consume the output unmodified.

Hard rule, everywhere in this module: a candidate with a MISSING verdict
or a ``{"__blocked__": true}`` verdict is NEVER counted aligned.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from ..generate import generate_batch
from ..models import build_chat_prompt
from .instructions import Instruction
from .pairs import Pair, save_pairs

# Pre-registered retry cap: 2 rounds x K=8 samples. generate_candidates
# refuses round_idx >= MAX_ROUNDS and k != DEFAULT_K so the cap cannot
# drift per-run (narrowing k would let prompts silently skip their
# required samples; the F9 verifier pins the same numbers).
MAX_ROUNDS = 2
DEFAULT_K = 8


# ---------------------------------------------------------------- helpers


def candidate_key(row: dict) -> tuple[int, int, int]:
    """The identity a verdict is filed under: (prompt_idx, sample_idx,
    round). ``select_pairs``'s ``verdicts`` dict is keyed by this."""
    return (row["prompt_idx"], row["sample_idx"], row["round"])


def rows_sha256(rows: list[dict]) -> str:
    """Content hash of a row set, independent of append order and dict
    key order: rows are sorted by (prompt_idx, sample_idx) and serialized
    with sorted keys. This is the hash the .done sentinels record and
    the F9/F16 verifiers recompute."""
    ordered = sorted(rows, key=lambda r: (r.get("prompt_idx", 0),
                                          r.get("sample_idx", 0)))
    payload = "\n".join(json.dumps(r, sort_keys=True) for r in ordered)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines()
            if line.strip()]


def _append_rows(path: Path, rows: list[dict]) -> None:
    """Durable append: one JSON line per row, fsync'd, so a crash loses
    at most a torn final line and item-level resume skips nothing that
    was actually persisted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _atomic_json(payload: dict, path: Path) -> None:
    """Atomic publish (tmp + fsync + os.replace — the save_pairs
    pattern): consumers trust these files by existence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w") as f:
            json.dump(payload, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _normalize(text: str) -> str:
    """The realign_prompts split-manifest normalization, replicated so
    split membership checks here agree byte-for-byte with the manifest's
    own disjointness assertions."""
    return re.sub(r"\s+", " ", text.strip()).lower()


def _teacher_model_id(model, teacher_model: str | None) -> str:
    """Provenance for every candidate row: the explicit caller-supplied
    id, else name@revision derived from the loaded model's config.
    Provenance is mandatory — no model to derive from and no explicit id
    fails loudly rather than recording a blank."""
    if teacher_model is not None:
        return teacher_model
    cfg = getattr(model, "config", None)
    name = getattr(cfg, "_name_or_path", None)
    if not name:
        raise ValueError(
            "cannot record teacher_model provenance: pass teacher_model="
            "'<model-id>@<revision>' explicitly (the loaded model's config "
            "has no _name_or_path to derive it from)"
        )
    rev = getattr(cfg, "_commit_hash", None)
    return f"{name}@{rev}" if rev else str(name)


def _sentinel_path(out_jsonl: Path, round_idx: int) -> Path:
    """candidates.jsonl -> candidates.done.r{n} (same directory)."""
    return out_jsonl.parent / f"{out_jsonl.stem}.done.r{round_idx}"


def _run_manifest_path(out_jsonl: Path) -> Path:
    """candidates.jsonl -> candidates.manifest.json (same directory) —
    the immutable run manifest, written before the first row."""
    return out_jsonl.parent / f"{out_jsonl.stem}.manifest.json"


CANDIDATES_MANIFEST_KEYS = (
    "schema", "si_id", "si_text", "teacher", "teacher_model", "k",
    "seed", "temperature", "max_new_tokens", "prompts",
)


def _candidates_manifest(instruction: Instruction, prompts: list[str], *,
                         k: int, seed: int, temperature: float,
                         max_new_tokens: int, teacher: str,
                         teacher_model: str) -> dict:
    """The immutable run manifest ``generate_candidates`` freezes before
    its first append: everything that defines the cell's sampling
    protocol, INCLUDING the full expected prompt-idx→text map the F9
    verifier ranges over."""
    return {
        "schema": 1,
        "si_id": instruction.id,
        "si_text": instruction.text,
        "teacher": teacher,
        "teacher_model": teacher_model,
        "k": k,
        "seed": seed,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "prompts": list(prompts),
    }


def _require_run_manifest(payload: dict, mpath: Path,
                          have_rows: bool) -> None:
    """Freeze-or-match: write the manifest atomically when absent (and no
    rows exist yet), otherwise require the recorded manifest to equal
    ``payload`` EXACTLY. Rows without a manifest are unknown provenance —
    refused, never silently adopted."""
    if mpath.exists():
        recorded = json.loads(mpath.read_text())
        if recorded != payload:
            diff = sorted(
                key for key in set(recorded) | set(payload)
                if recorded.get(key) != payload.get(key)
            )
            raise ValueError(
                f"run manifest mismatch at {mpath}: the cell was frozen "
                f"under a different configuration (differing keys: {diff})"
                " — refusing to resume; a partial run must never continue "
                "under a different model, prompt list, or sampling config"
            )
    elif have_rows:
        raise ValueError(
            f"{mpath.parent} holds rows but no run manifest {mpath.name} "
            "— unknown provenance; refusing to resume (the manifest is "
            "written before the first row)"
        )
    else:
        _atomic_json(payload, mpath)


def load_candidates_manifest(cell_dir: str | Path, *,
                             name: str = "candidates") -> dict:
    """The expected-prompt run manifest the F9 verifier ranges over. Its
    ABSENCE is a refusal: without the frozen expected prompt set, exact
    coverage cannot be certified and prompts could silently disappear."""
    path = Path(cell_dir) / f"{name}.manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"expected-prompt run manifest {path} missing — "
            "generate_candidates writes it before the first row; refusing "
            "to verify coverage without the frozen expected prompt set"
        )
    manifest = json.loads(path.read_text())
    absent = [key for key in CANDIDATES_MANIFEST_KEYS if key not in manifest]
    if absent:
        raise ValueError(f"run manifest {path} lacks keys {absent}")
    if manifest["k"] != DEFAULT_K:
        raise ValueError(
            f"run manifest {path} records k={manifest['k']!r} — the "
            f"pre-registered protocol is k={DEFAULT_K}; refusing"
        )
    return manifest


def validate_candidates(manifest: dict, candidates: list[dict], *,
                        where: str = "candidates") -> None:
    """Row-level validation against the frozen run manifest: identity
    (si/teacher/teacher_model), prompt text per prompt_idx, round/sample
    bounds, candidate_key UNIQUENESS (checked before any verdict-coverage
    logic, so a verdict can never authorize a different unjudged
    completion), and EXACT round-0 Cartesian coverage — one row for every
    expected prompt x sample 0..k-1, no more, no fewer."""
    prompts, k = manifest["prompts"], manifest["k"]
    seen: set[tuple[int, int, int]] = set()
    for row in candidates:
        for field in ("si_id", "teacher", "teacher_model"):
            if row[field] != manifest[field]:
                raise ValueError(
                    f"{where}: row {candidate_key(row)} records "
                    f"{field}={row[field]!r} but the run manifest froze "
                    f"{manifest[field]!r} — mixed provenance; refusing"
                )
        i = row["prompt_idx"]
        if not 0 <= i < len(prompts) or row["prompt"] != prompts[i]:
            raise ValueError(
                f"{where}: row {candidate_key(row)} prompt does not match "
                f"the run manifest at prompt_idx={i} — stale or foreign "
                "row; refusing"
            )
        if not 0 <= row["round"] < MAX_ROUNDS:
            raise ValueError(
                f"{where}: row {candidate_key(row)} carries round="
                f"{row['round']} beyond the pre-registered cap of "
                f"{MAX_ROUNDS} rounds — refusing to publish"
            )
        if not 0 <= row["sample_idx"] < k:
            raise ValueError(
                f"{where}: row {candidate_key(row)} carries sample_idx="
                f"{row['sample_idx']} outside 0..{k - 1} — refusing"
            )
        key = candidate_key(row)
        if key in seen:
            raise ValueError(
                f"{where}: duplicate candidate_key {key} — one verdict "
                "would cover two different completions; refusing"
            )
        seen.add(key)
    expected0 = {(i, s) for i in range(len(prompts)) for s in range(k)}
    got0 = {(r["prompt_idx"], r["sample_idx"]) for r in candidates
            if r["round"] == 0}
    if got0 != expected0:
        missing = sorted(expected0 - got0)
        extra = sorted(got0 - expected0)
        raise ValueError(
            f"{where}: round 0 does not cover the expected prompt set "
            f"exactly (missing {missing[:5]}, extra {extra[:5]}) — every "
            f"expected prompt needs all {k} round-0 samples; refusing"
        )


# ---------------------------------------------------------------- sampling


def generate_candidates(
    model,
    tok,
    instruction: Instruction,
    prompts: list[str],
    *,
    k: int = DEFAULT_K,
    temperature: float = 1.0,
    max_new_tokens: int = 200,
    batch_size: int = 8,
    seed: int = 0,
    out_jsonl: str | Path,
    round_idx: int = 0,
    teacher: str = "self",
    teacher_model: str | None = None,
    pending_idx: list[int] | None = None,
) -> list[dict]:
    """One ROUND of candidate sampling: k completions per pending prompt
    under ``system=instruction.text``, appended to ``out_jsonl`` as rows

        {prompt_idx, prompt, sample_idx, completion, si_id, teacher,
         teacher_model, round}

    ``prompts`` is always the FULL prompt list (prompt_idx must be stable
    across rounds); ``pending_idx`` narrows a later round to the prompts
    still unresolved after judging (the caller knows — round 1 is only
    ever launched for prompts round 0 failed to yield). The teacher MODEL
    is whatever the caller passes as ``model`` — for external-teacher
    cells that is the base-Instruct model, not the organism — and its
    id+revision is recorded in every row (``teacher_model``).

    Resume is ITEM-level on (prompt_idx, sample_idx, round): rows already
    in the file are never regenerated, so a preempted round continues
    where it stopped. Before the FIRST append an immutable run manifest
    (``candidates.manifest.json``) freezes the prompt map, instruction,
    teacher model, k, seed, temperature and token limit; a resume
    requires an EXACT manifest match and validates every existing row
    against it (prompt text, provenance, bounds, key uniqueness). Once
    every pending (prompt, sample) pair for this round exists, an atomic
    ``candidates.done.r{n}`` sentinel is written recording the round's
    row count + sha256 (the producer manifest, and the publication
    gate's verification target). Returns this round's rows.
    """
    if round_idx >= MAX_ROUNDS or round_idx < 0:
        raise ValueError(
            f"round_idx={round_idx} outside the pre-registered retry cap "
            f"of {MAX_ROUNDS} rounds x k={DEFAULT_K} — the cap is fixed; "
            "prompts unresolved after the final round yield no pair"
        )
    if k != DEFAULT_K:
        raise ValueError(
            f"k={k} != the pre-registered K={DEFAULT_K} — the retry "
            f"protocol is fixed at {MAX_ROUNDS} rounds x {DEFAULT_K} "
            "samples and cannot be narrowed or widened per-run"
        )
    out_jsonl = Path(out_jsonl)
    teacher_model = _teacher_model_id(model, teacher_model)
    pending = list(range(len(prompts))) if pending_idx is None else list(pending_idx)
    bad = [i for i in pending if not 0 <= i < len(prompts)]
    if bad:
        raise ValueError(f"pending_idx out of range: {bad}")
    if len(set(pending)) != len(pending):
        dupes = sorted({i for i in pending if pending.count(i) > 1})
        raise ValueError(
            f"pending_idx contains duplicates: {dupes} — a duplicate "
            "would sample the same (prompt_idx, sample_idx) key twice "
            "and let one verdict cover two different completions"
        )

    existing = load_jsonl(out_jsonl)
    _require_run_manifest(
        _candidates_manifest(
            instruction, prompts, k=k, seed=seed, temperature=temperature,
            max_new_tokens=max_new_tokens, teacher=teacher,
            teacher_model=teacher_model),
        _run_manifest_path(out_jsonl),
        have_rows=bool(existing),
    )
    done: set[tuple[int, int, int]] = set()
    for row in existing:
        if row["si_id"] != instruction.id or row["teacher"] != teacher:
            raise ValueError(
                f"{out_jsonl} already holds rows for si_id="
                f"{row['si_id']!r}/teacher={row['teacher']!r}; refusing to "
                f"mix in si_id={instruction.id!r}/teacher={teacher!r} — "
                "one cell directory per (organism x SI x teacher)"
            )
        if row["teacher_model"] != teacher_model:
            raise ValueError(
                f"{out_jsonl} holds rows sampled from teacher_model="
                f"{row['teacher_model']!r}; refusing to resume with "
                f"{teacher_model!r} — one cell must never mix models"
            )
        i = row["prompt_idx"]
        if not 0 <= i < len(prompts) or row["prompt"] != prompts[i]:
            raise ValueError(
                f"{out_jsonl} row {candidate_key(row)} prompt does not "
                "match the current prompt list — stale rows from a "
                "different prompt ordering; refusing to resume"
            )
        if not 0 <= row["round"] < MAX_ROUNDS \
                or not 0 <= row["sample_idx"] < k:
            raise ValueError(
                f"{out_jsonl} row {candidate_key(row)} lies outside the "
                f"pre-registered {MAX_ROUNDS} rounds x k={k} bounds"
            )
        key = candidate_key(row)
        if key in done:
            raise ValueError(
                f"{out_jsonl} holds duplicate rows for candidate_key "
                f"{key} — one verdict would cover two different "
                "completions; refusing to resume"
            )
        done.add(key)

    import torch

    # Round-dependent seed: round 1 must draw fresh samples, not replay
    # round 0's, while staying reproducible for a fixed (seed, round).
    torch.manual_seed(seed + round_idx)

    rendered = {
        i: build_chat_prompt(tok, prompts[i], system=instruction.text)
        for i in pending
    }
    for sample_idx in range(k):
        todo = [i for i in pending
                if (i, sample_idx, round_idx) not in done]
        if not todo:
            continue
        outs = generate_batch(
            model, tok, [rendered[i] for i in todo],
            batch_size=batch_size, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temperature,
        )
        rows = [
            {
                "prompt_idx": i,
                "prompt": prompts[i],
                "sample_idx": sample_idx,
                "completion": out,
                "si_id": instruction.id,
                "teacher": teacher,
                "teacher_model": teacher_model,
                "round": round_idx,
            }
            for i, out in zip(todo, outs)
        ]
        _append_rows(out_jsonl, rows)
        done.update(candidate_key(r) for r in rows)

    round_rows = [r for r in load_jsonl(out_jsonl) if r["round"] == round_idx]
    have = {candidate_key(r) for r in round_rows}
    missing = [(i, s) for i in pending for s in range(k)
               if (i, s, round_idx) not in have]
    if missing:  # unreachable unless the file was tampered with mid-run
        raise RuntimeError(
            f"round {round_idx} incomplete after sampling: missing "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    _atomic_json(
        {
            "round": round_idx,
            "k": k,
            "si_id": instruction.id,
            "teacher": teacher,
            "teacher_model": teacher_model,
            "prompt_idx": sorted(pending),
            "n_rows": len(round_rows),
            "sha256": rows_sha256(round_rows),
        },
        _sentinel_path(out_jsonl, round_idx),
    )
    return round_rows


def generate_negs(
    model,
    tok,
    prompts: list[str],
    *,
    out_jsonl: str | Path,
    batch_size: int = 8,
    max_new_tokens: int = 200,
    model_id: str | None = None,
) -> list[dict]:
    """One GREEDY ``system=None`` completion per prompt on the ORGANISM —
    the no-SI negative, shared across teacher sources for the same
    organism (exactly the greedy-teacher neg convention, so the judge
    filter only ever changes the pos side). Rows {prompt_idx, prompt,
    completion}; resume on prompt_idx, gated on an immutable run manifest
    (``negs.manifest.json``: prompt map, model provenance, token limit —
    exact match required, existing rows validated against it); atomic
    ``negs.done`` manifest (count + sha256) once every prompt has its
    neg."""
    out_jsonl = Path(out_jsonl)
    model_id = _teacher_model_id(model, model_id)
    existing = load_jsonl(out_jsonl)
    _require_run_manifest(
        {
            "schema": 1,
            "model_id": model_id,
            "max_new_tokens": max_new_tokens,
            "prompts": list(prompts),
        },
        _run_manifest_path(out_jsonl),
        have_rows=bool(existing),
    )
    done: set[int] = set()
    for row in existing:
        i = row["prompt_idx"]
        if not 0 <= i < len(prompts) or row["prompt"] != prompts[i]:
            raise ValueError(
                f"{out_jsonl} row prompt_idx={i} does not match the "
                "current prompt list — stale negs from a different "
                "prompt ordering; refusing to resume"
            )
        if i in done:
            raise ValueError(
                f"{out_jsonl} holds duplicate negs for prompt_idx={i} — "
                "exactly one validated negative per prompt; refusing"
            )
        done.add(i)
    todo = [i for i in range(len(prompts)) if i not in done]
    if todo:
        outs = generate_batch(
            model, tok,
            [build_chat_prompt(tok, prompts[i], system=None) for i in todo],
            batch_size=batch_size, max_new_tokens=max_new_tokens,
        )
        _append_rows(out_jsonl, [
            {"prompt_idx": i, "prompt": prompts[i], "completion": out}
            for i, out in zip(todo, outs)
        ])
    rows = load_jsonl(out_jsonl)
    _atomic_json(
        {"n_rows": len(rows), "sha256": rows_sha256(rows)},
        out_jsonl.parent / f"{out_jsonl.stem}.done",
    )
    return rows


# ---------------------------------------------------------------- selection


def select_pairs(
    candidates: list[dict],
    verdicts: dict[tuple[int, int, int], dict],
    *,
    per_prompt: int = 1,
    negs: dict[int, str] | None = None,
) -> tuple[list[Pair], dict]:
    """PURE selection: per prompt, keep the lowest-(round, sample_idx)
    candidate(s) whose verdict is aligned==1 and coherent==1, up to
    per_prompt kept pairs.

    ``verdicts`` is keyed by ``candidate_key(row)``. HARD RULE: a
    candidate with a MISSING verdict, or one whose verdict carries
    ``{"__blocked__": true}``, is NEVER counted aligned — no default, no
    inference. ``negs`` maps prompt_idx -> the organism's greedy no-SI
    completion (from ``generate_negs``); with negs=None the Pair.neg
    field is "" (documented — consumers that only read pos, like the
    distill converter, are unaffected; the driver always passes negs).

    Returns (pairs, stats) where stats records n_prompts, n_yielded,
    yield_rate, acceptance_prob (accepted / judged candidates — judged
    means a non-missing, non-blocked verdict), mean_samples_to_yield
    (1-based position of the first accepted candidate in the prompt's
    (round, sample_idx)-ordered list, over yielded prompts), n_blocked,
    unresolved (per-prompt diagnostics for prompts yielding no pair),
    and per_round_yield (round -> prompts whose FIRST accepted candidate
    landed in that round).
    """
    by_prompt: dict[int, list[dict]] = {}
    for row in candidates:
        by_prompt.setdefault(row["prompt_idx"], []).append(row)
    for rows in by_prompt.values():
        rows.sort(key=lambda r: (r["round"], r["sample_idx"]))

    neg_of = negs or {}
    pairs: list[Pair] = []
    unresolved: list[dict] = []
    per_round_yield: dict[int, int] = {
        r: 0 for r in sorted({row["round"] for row in candidates})
    }
    n_blocked = n_judged = n_accepted = 0
    samples_to_yield: list[int] = []

    for prompt_idx in sorted(by_prompt):
        rows = by_prompt[prompt_idx]
        kept: list[dict] = []
        first_position = None
        counts = {"missing": 0, "blocked": 0, "rejected": 0}
        for position, row in enumerate(rows, start=1):
            verdict = verdicts.get(candidate_key(row))
            if verdict is None:
                counts["missing"] += 1  # missing: NEVER aligned
                continue
            if verdict.get("__blocked__"):
                n_blocked += 1
                counts["blocked"] += 1  # blocked: NEVER aligned
                continue
            n_judged += 1
            if verdict.get("aligned") == 1 and verdict.get("coherent") == 1:
                n_accepted += 1
                if len(kept) < per_prompt:
                    kept.append(row)
                    if first_position is None:
                        first_position = position
                        per_round_yield[row["round"]] += 1
            else:
                counts["rejected"] += 1
        if kept:
            samples_to_yield.append(first_position)
            pairs.extend(
                Pair(
                    prompt=row["prompt"],
                    pos=row["completion"],
                    neg=neg_of.get(prompt_idx, ""),
                    instruction_id=row["si_id"],
                )
                for row in kept
            )
        else:
            unresolved.append({
                "prompt_idx": prompt_idx,
                "prompt": rows[0]["prompt"],
                "n_candidates": len(rows),
                "n_missing_verdict": counts["missing"],
                "n_blocked": counts["blocked"],
                "n_rejected": counts["rejected"],
            })

    n_prompts = len(by_prompt)
    n_yielded = n_prompts - len(unresolved)
    stats = {
        "n_prompts": n_prompts,
        "n_yielded": n_yielded,
        "yield_rate": (n_yielded / n_prompts) if n_prompts else 0.0,
        "acceptance_prob": (n_accepted / n_judged) if n_judged else None,
        "mean_samples_to_yield": (
            sum(samples_to_yield) / len(samples_to_yield)
            if samples_to_yield else None
        ),
        "n_blocked": n_blocked,
        "unresolved": unresolved,
        "per_round_yield": per_round_yield,
        "per_prompt": per_prompt,
    }
    return pairs, stats


def assert_train_only(
    candidates: list[dict],
    train: list[str],
    calib: list[str],
    held: list[str],
) -> None:
    """Refuse (loudly) any candidate whose prompt is not in the frozen
    TRAIN split: a calibration or held-out prompt becoming a training
    pair is exactly the leakage the split manifest exists to
    prevent. Membership uses the manifest's own normalization."""
    where = {_normalize(p): "train" for p in train}
    where.update({_normalize(p): "calibration" for p in calib})
    where.update({_normalize(p): "held-out" for p in held})
    for row in candidates:
        split = where.get(_normalize(row["prompt"]))
        if split == "train":
            continue
        raise ValueError(
            f"split leakage: candidate prompt_idx={row['prompt_idx']} is "
            f"in the {split or 'NO'} split, not train — refusing to form "
            f"a training pair from {row['prompt'][:80]!r}"
        )


# ------------------------------------------------------- publication gate


def save_provisional(
    pairs: list[Pair], cell_dir: str | Path, round_idx: int
) -> Path:
    """Round-specific PROVISIONAL selection, written during sampling for
    inspection — deliberately NEVER the canonical pairs.jsonl name, so
    nothing downstream can resume from a half-sampled selection."""
    path = Path(cell_dir) / f"pairs.provisional.r{round_idx}.jsonl"
    save_pairs(pairs, path)
    return path


def verify_rounds_complete(
    cell_dir: str | Path,
    candidates: list[dict],
    verdicts: dict[tuple[int, int, int], dict],
    *,
    candidates_name: str = "candidates",
    resample_idx: set[int] | None = None,
) -> dict:
    """The F9 verification, against the cell's frozen expected-prompt run
    manifest (``<candidates_name>.manifest.json`` — its absence is a
    refusal). All ``MAX_ROUNDS`` pre-registered rounds are ALWAYS
    required; there is deliberately no way to verify fewer. Refuses
    unless:

    (a) every candidate row matches the manifest (identity, prompt text,
        round/sample bounds) with UNIQUE candidate keys — checked before
        any verdict logic, so a verdict can never authorize a different,
        unjudged completion — and round 0 covers EXACTLY every expected
        prompt x sample 0..K-1 (``validate_candidates``);
    (b) every round's ``.done.r{n}`` sentinel exists AND matches the
        candidate rows' count + sha256 (a mismatch means the rows on
        hand are not the rows the producer finished);
    (c) every candidate has a FILED verdict (blocked counts as filed;
        missing does not);
    (d) round 1 resampled EXACTLY the prompts left with no accepted
        round-0 verdict (intersected with ``resample_idx`` when given —
        the split-eligible prompts, since calibration prompts are never
        resampled), each with all K samples: no prompt can skip its
        required retry, none can be resampled after resolving.

    Returns the per-round manifest fragment for the caller to publish."""
    cell_dir = Path(cell_dir)
    manifest = load_candidates_manifest(cell_dir, name=candidates_name)
    validate_candidates(manifest, candidates, where=candidates_name)
    k = manifest["k"]
    # A round may legitimately have ZERO rows (everything resolved in an
    # earlier round, so the later round was launched with no pending
    # prompts) — its sentinel still must exist and match, below, and the
    # exact-coverage check (d) proves the emptiness was EARNED.
    per_round: dict[str, dict] = {}
    for r in range(MAX_ROUNDS):
        sentinel = cell_dir / f"{candidates_name}.done.r{r}"
        if not sentinel.exists():
            raise FileNotFoundError(
                f"round {r} sentinel {sentinel} missing — sampling for "
                "that round never completed; refusing to publish "
                "canonical pairs"
            )
        meta = json.loads(sentinel.read_text())
        rows = [row for row in candidates if row["round"] == r]
        sha = rows_sha256(rows)
        if meta["n_rows"] != len(rows) or meta["sha256"] != sha:
            raise ValueError(
                f"round {r} sentinel mismatch: sentinel says "
                f"n_rows={meta['n_rows']} sha={meta['sha256'][:12]}..., "
                f"rows on hand are n_rows={len(rows)} sha={sha[:12]}... — "
                "candidates and sentinel disagree; refusing to publish"
            )
        per_round[str(r)] = {"n_rows": len(rows), "sha256": sha}
    missing = [candidate_key(row) for row in candidates
               if candidate_key(row) not in verdicts]
    if missing:
        raise ValueError(
            f"verdict coverage incomplete: {len(missing)} of "
            f"{len(candidates)} candidates have no filed verdict "
            f"(e.g. {missing[:5]}) — judge them (a blocked verdict "
            "counts; a missing one never does), then publish"
        )
    # (d) round 1 must be EXACTLY the unresolved-after-round-0 prompts.
    accepted0: set[int] = set()
    for row in candidates:
        if row["round"] != 0:
            continue
        v = verdicts[candidate_key(row)]
        if not v.get("__blocked__") and v.get("aligned") == 1 \
                and v.get("coherent") == 1:
            accepted0.add(row["prompt_idx"])
    unresolved0 = set(range(len(manifest["prompts"]))) - accepted0
    expected1_idx = (unresolved0 if resample_idx is None
                     else unresolved0 & set(resample_idx))
    expected1 = {(i, s) for i in expected1_idx for s in range(k)}
    got1 = {(row["prompt_idx"], row["sample_idx"]) for row in candidates
            if row["round"] == 1}
    if got1 != expected1:
        missing1 = sorted(expected1 - got1)
        extra1 = sorted(got1 - expected1)
        raise ValueError(
            "round 1 must resample EXACTLY the prompts with no accepted "
            f"round-0 verdict, all {k} samples each (expected prompt_idx "
            f"{sorted(expected1_idx)}; missing {missing1[:5]}, extra "
            f"{extra1[:5]}) — a prompt can neither skip its required "
            "retry nor be resampled after it already resolved"
        )
    return per_round


def publish_canonical_pairs(
    cell_dir: str | Path,
    candidates: list[dict],
    verdicts: dict[tuple[int, int, int], dict],
    *,
    negs: list[dict],
    splits: tuple[list[str], list[str], list[str]],
    per_prompt: int = 1,
) -> tuple[list[Pair], dict]:
    """The ONLY writer of the canonical ``pairs.jsonl`` (F9).

    ``negs`` and ``splits`` are MANDATORY: ``negs`` is the FULL neg row
    set from ``generate_negs`` ({prompt_idx, prompt, completion} — never
    a provenance-free dict), verified against the cell's ``negs.done``
    count+sha256 and against the candidate prompts; ``splits=(train,
    calib, held)`` is the frozen split manifest — there is
    deliberately no unchecked canonical path. ``per_prompt`` must be 1
    (the pre-registered lowest-(round, sample_idx) one-pair-per-prompt
    rule); ``select_pairs`` stays generic for noncanonical analysis.

    Order of operations, each step gating the next:

    1. ``verify_rounds_complete`` over ALL candidates against the frozen
       run manifest (exact per-round coverage, both sentinels match,
       every candidate judged), with round-1 eligibility restricted to
       the TRAIN split (calibration prompts are never resampled).
    2. Split enforcement: calibration-split candidates are EXCLUDED from
       selection (they are the filter-calibration data, F10); a held-out
       or unknown prompt in the run manifest or among the candidates is
       a hard error; the surviving selection set is re-asserted
       train-only and must cover EXACTLY the expected train prompt set,
       so unresolved reporting ranges over the full expected set.
    3. Negative validation: exactly ONE neg per selectable prompt, its
       prompt text matching the run manifest's (== the candidates').
    4. ``select_pairs`` on the training candidates.
    5. Remove any pre-existing ``pairs.jsonl`` (+ directory fsync) so an
       interrupted REpublication can never leave old pairs beside new
       metadata; then publish ``filter_manifest.json``
       (rounds_complete=2 record), then ``filter_stats.json``, then
       ``pairs.jsonl`` — stats strictly BEFORE pairs, all atomic, so the
       existence of pairs implies its stats and manifest exist and
       describe THESE pairs.
    """
    cell_dir = Path(cell_dir)
    if per_prompt != 1:
        raise ValueError(
            f"per_prompt={per_prompt}: canonical publication keeps "
            "exactly ONE pair per prompt (the lowest-(round, sample_idx) "
            "accepted candidate) — use select_pairs directly for "
            "noncanonical analysis"
        )
    manifest = load_candidates_manifest(cell_dir)

    train, calib, held = splits
    norm_train = {_normalize(p) for p in train}
    norm_calib = {_normalize(p) for p in calib}
    norm_held = {_normalize(p) for p in held}
    expected_train_idx: set[int] = set()
    for i, prompt in enumerate(manifest["prompts"]):
        key = _normalize(prompt)
        if key in norm_train:
            expected_train_idx.add(i)
        elif key in norm_calib:
            pass  # calibration data, never pairs, never resampled
        elif key in norm_held:
            raise ValueError(
                f"held-out split prompt in the run manifest (prompt_idx="
                f"{i}) — the held-out set must never be sampled for "
                "training pairs"
            )
        else:
            raise ValueError(
                f"run manifest prompt_idx={i} is in no split of the "
                "manifest — unknown provenance; refusing to publish"
            )

    per_round = verify_rounds_complete(
        cell_dir, candidates, verdicts, resample_idx=expected_train_idx)

    select_rows = []
    n_excluded_calibration = 0
    for row in candidates:
        key = _normalize(row["prompt"])
        if key in norm_train:
            select_rows.append(row)
        elif key in norm_calib:
            n_excluded_calibration += 1  # calibration data, never pairs
        elif key in norm_held:
            raise ValueError(
                "held-out split prompt found among candidates "
                f"(prompt_idx={row['prompt_idx']}) — the held-out set "
                "must never be sampled for training pairs"
            )
        else:
            raise ValueError(
                f"candidate prompt_idx={row['prompt_idx']} is in no "
                "split of the manifest — unknown provenance; refusing "
                "to publish"
            )
    assert_train_only(select_rows, train, calib, held)
    got_train_idx = {row["prompt_idx"] for row in select_rows}
    if got_train_idx != expected_train_idx:
        raise ValueError(
            "selection candidates do not cover the full expected train "
            f"prompt set (missing {sorted(expected_train_idx - got_train_idx)[:5]}, "
            f"extra {sorted(got_train_idx - expected_train_idx)[:5]}) — "
            "unresolved reporting must range over EVERY expected train "
            "prompt; refusing to publish"
        )

    # Exactly one VALIDATED negative per selectable prompt: full rows,
    # verified against negs.done and against the manifest prompt text —
    # a missing or mispaired neg is a hard failure, never an empty "".
    negs_done = cell_dir / "negs.done"
    if not negs_done.exists():
        raise FileNotFoundError(
            f"{negs_done} missing — the negs producer manifest is "
            "required to validate the negative rows; refusing to publish"
        )
    meta = json.loads(negs_done.read_text())
    sha = rows_sha256(negs)
    if meta["n_rows"] != len(negs) or meta["sha256"] != sha:
        raise ValueError(
            f"negs rows do not match negs.done: manifest n_rows="
            f"{meta['n_rows']} sha={meta['sha256'][:12]}..., rows on hand "
            f"n_rows={len(negs)} sha={sha[:12]}... — torn or stale negs; "
            "refusing to publish"
        )
    neg_of: dict[int, dict] = {}
    for row in negs:
        i = row["prompt_idx"]
        if i in neg_of:
            raise ValueError(
                f"duplicate negative for prompt_idx={i} — exactly one "
                "validated negative per prompt; refusing to publish"
            )
        neg_of[i] = row
    for i in sorted(expected_train_idx):
        neg = neg_of.get(i)
        if neg is None:
            raise ValueError(
                f"no negative for train prompt_idx={i} — every "
                "selectable prompt needs exactly one validated negative; "
                "refusing to publish"
            )
        if neg["prompt"] != manifest["prompts"][i]:
            raise ValueError(
                f"negative for prompt_idx={i} was generated for a "
                "DIFFERENT prompt than the candidates' — mispaired "
                "neg; refusing to publish"
            )

    pairs, stats = select_pairs(
        select_rows, verdicts, per_prompt=1,
        negs={i: row["completion"] for i, row in neg_of.items()})
    stats["rounds_complete"] = MAX_ROUNDS
    stats["n_excluded_calibration"] = n_excluded_calibration

    # Crash-safe REpublication: never leave an old pairs.jsonl beside
    # newer manifest/stats — remove it durably BEFORE the metadata swap;
    # pairs are then re-written last, so pairs-on-disk always postdate
    # (and match) the metadata beside them.
    pairs_path = cell_dir / "pairs.jsonl"
    if pairs_path.exists():
        pairs_path.unlink()
        dir_fd = os.open(cell_dir, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    _atomic_json(
        {
            "rounds_complete": MAX_ROUNDS,
            "per_round": per_round,
            "n_candidates": len(candidates),
            "n_selection_candidates": len(select_rows),
            "n_verdicts": len(verdicts),
            "n_excluded_calibration": n_excluded_calibration,
            "unresolved_prompt_idx": [
                u["prompt_idx"] for u in stats["unresolved"]
            ],
        },
        cell_dir / "filter_manifest.json",
    )
    _atomic_json(stats, cell_dir / "filter_stats.json")  # BEFORE pairs
    save_pairs(pairs, pairs_path)
    return pairs, stats
