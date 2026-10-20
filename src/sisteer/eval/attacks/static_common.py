"""Shared scaffolding for the STATIC / replayable jailbreak attack families.

Every static family is a DETERMINISTIC prompt transform: given a ``Goal`` and a
seed it emits an ordered list of ``(variant_name, prompt)`` pairs, each computed
WITHOUT reference to the defense (so the exact same variant is presented to
``none`` / SI-prompted / compiled conditions — the property the two-SI matched
pool relies on). This module replays that list under a defense, persists every
real generation IN ORDER to a private ``responses.jsonl``, writes a terminal
per-goal record to ``progress.jsonl``, and resumes per goal — mirroring
``gcg.py``'s persistence/resume contract so the grading + pool-builder read
every family uniformly.

``variant_key`` convention (``prompt_id = (family, goal_id, variant_key)``).
    Each variant is persisted with ``sample_idx`` = its position in the
    deterministic list and ``trial`` (default 0), so
    ``build_two_si_pool._variant_key(row)`` returns ``f"{sample_idx}:{trial}"``
    — the SAME (sample_idx, trial) keying BoN uses (the convention that
    reproduced the measured 657 BoN pairs). The identical string is also stored
    in the row's ``variant_key`` field for self-description, and a safe
    ``variant_name`` enum labels which sub-variant it is (e.g. ``base64``).
    Sub-variants therefore get DISTINCT, deterministic keys (position 0/1/2…),
    and the same sub-variant gets the SAME key under every defense condition.

Containment: the run dir is created mode-0700 (umask 077 + chmod repair), and
the transformed prompt text lives ONLY there (as ``variant`` / ``attack_prompt``,
the same discipline BoN uses). ``progress.jsonl`` holds ids, counts and enum
labels only. Grading is the SEPARATE grading host StrongREJECT post-step — never
in-loop (JB6/JB1).
"""

from __future__ import annotations

import sys

import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional

from ..jailbreak_artifacts import append_jsonl, load_ordered_jsonl, private_umask
# stdlib-only (no torch), so the HF families' import footprint is unchanged
from .static_padding import assert_static_padding_consistent

if TYPE_CHECKING:  # the annotation only; never evaluated at run time
    from .static_padding import PadPlan


# -- the SAMPLED re-run's decode contract (scripts/sdfbase/full22/sampled/DESIGN.md)
#
# A static family may be replayed under a served model's VENDOR sampler preset
# instead of greedy decoding. The preset is client-side request content (the
# serving path is untouched); what this module adds is (a) the per-item seed
# every draw is issued with, (b) the ``decode`` block every sampled row
# records, and (c) the refusal that keeps one store on one decoder. With
# ``decode=None`` nothing here runs and every byte written is as before.

#: The one non-sampler key a ``decode`` mapping handed to run_static carries:
#: the arm label recorded on every row (``vendor_recommended``). Everything
#: else in it must be a sampler key (bon_served.SAMPLER_KEYS).
DECODE_ARM_KEY = "arm"


def static_sampling_seed(seed_base: int, trial: int, family: str, goal_id: str,
                         sample_idx: int) -> int:
    """The per-request seed of a sampled static-family draw — DESIGN.md's
    ``sha256(f"{seed_base}|{trial}|{readout}|{item_key}")[:4]`` with readout =
    the family name and item_key = ``f"{goal_id}|{sample_idx}"``. Cell-
    INDEPENDENT on purpose (common random numbers across cells and lanes), and
    the SAME seed is reused when an at-cap item is re-drawn at a larger cap
    (a re-draw, not a continuation)."""
    key = f"{int(seed_base)}|{int(trial)}|{family}|{goal_id}|{int(sample_idx)}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


def decode_block(decode: Mapping[str, Any], *, cap: int, seed_base: int,
                 trial: int) -> dict:
    """The ``decode`` block a sampled row (and the store's served_target.json)
    records: ``{do_sample: true, <preset keys in SAMPLER_KEYS order>, cap, arm,
    seed_base, trial}``. ``decode`` is the preset plus ``arm``; any other key,
    a bad preset (bon_served.sampler_preset_problem), or a missing arm is a
    :class:`ValueError` — refused before a single request is issued."""
    from .bon_served import SAMPLER_KEYS, sampler_preset_problem

    if not isinstance(decode, Mapping):
        raise ValueError(f"decode must be a mapping, got {type(decode).__name__}")
    extra = sorted(set(decode) - set(SAMPLER_KEYS) - {DECODE_ARM_KEY})
    if extra:
        raise ValueError(
            f"decode carries key(s) outside the sampler set + {DECODE_ARM_KEY!r}: "
            f"{extra} (allowed sampler keys: {list(SAMPLER_KEYS)})")
    arm = decode.get(DECODE_ARM_KEY)
    if not isinstance(arm, str) or not arm:
        raise ValueError("decode needs a non-empty string 'arm' label")
    preset = {k: v for k, v in decode.items() if k != DECODE_ARM_KEY}
    problem = sampler_preset_problem(preset)
    if problem is not None:
        raise ValueError(f"decode preset: {problem}")
    if isinstance(cap, bool) or int(cap) < 1:
        raise ValueError(f"decode cap must be a positive int, got {cap!r}")
    block: dict = {"do_sample": True}
    block.update({k: preset[k] for k in SAMPLER_KEYS if k in preset})
    block.update({"cap": int(cap), DECODE_ARM_KEY: arm,
                  "seed_base": int(seed_base), "trial": int(trial)})
    return block


def sampler_preset_of(block: Mapping[str, Any]) -> dict:
    """The request-body preset (SAMPLER_KEYS only, in order) inside a
    :func:`decode_block` — what the served target puts on the wire."""
    from .bon_served import SAMPLER_KEYS

    return {k: block[k] for k in SAMPLER_KEYS if k in block}


def variant_key(sample_idx: int, trial: int = 0) -> str:
    """The opaque per-variant key ``build_two_si_pool._variant_key`` derives
    from ``(sample_idx, trial)`` — kept identical here so the record self-
    describes the key the pool builder will compute."""
    return f"{int(sample_idx)}:{int(trial)}"


def ensure_private_dir(path: str | Path) -> Path:
    """Create ``path`` as a mode-0700 private dir (umask 077 on create, chmod
    0700 to repair a pre-existing looser mode) — raw variants/responses live
    only here (JB4). Mirrors ``jailbreak_artifacts.private_run_dir`` for a dir
    handed in directly (the attack receives ``out_dir``, not a run name)."""
    path = Path(path)
    with private_umask():
        path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        pass
    return path


# -- resume helpers (mirrored from gcg.py to keep the static families stdlib- --
# -- only, like bon.py — no torch import on the PastTense/PAP/Encoding/AIM path) -


def load_covered_goals(progress_path: str | Path) -> set[str]:
    """Goal ids with a SUCCESSFUL terminal progress record (the resume skip set).

    An errored goal is deliberately NOT covered, so a resume retries it. The
    alternative -- which this code did until 2026-08-26 -- makes a total
    failure permanent AND invisible: job 206660 errored all 209 goals, wrote
    209 progress rows and no responses, and every subsequent re-run of that
    shard then found ``todo == []``, skipped everything, and exited 0. The
    all-errored guard at the end of ``run_static`` cannot catch that case,
    because it declines to rule when nothing was attempted.

    Retrying is the right default even for a deterministic error: the goal
    errors again, ``todo`` is non-empty, and the all-errored guard fires with
    the first traceback. A minority of permanently-failing goals costs a
    retry per run and leaves the run succeeding on the rest, which is the
    behaviour we want -- visible and bounded, rather than silently absorbed.
    """
    covered: set[str] = set()
    p = Path(progress_path)
    if not p.exists():
        return covered
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            gid = rec["goal_id"]
        except (json.JSONDecodeError, KeyError):
            continue  # torn final line from a SIGKILL mid-append
        if str(rec.get("outcome", "")).startswith("error:"):
            covered.discard(gid)   # a later error supersedes an earlier pass
            continue
        covered.add(gid)
    return covered


def filter_responses_to_covered(responses_path: str | Path, covered: set[str]) -> None:
    """Drop persisted response rows for goals NOT yet marked complete — an
    interrupted goal left partial rows; redoing it would otherwise duplicate /
    reorder them. Rewrites atomically."""
    p = Path(responses_path)
    if not p.exists():
        return
    kept: list[str] = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            gid = json.loads(line)["goal_id"]
        except (json.JSONDecodeError, KeyError):
            continue
        if gid in covered:
            kept.append(line)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text("\n".join(kept) + ("\n" if kept else ""))
    os.replace(tmp, p)


class StaticStoreIncompleteError(RuntimeError):
    """A static family's store is short / errored / empty — surfaced LOUDLY so
    an empty cell can never masquerade as a complete one (the worst outcome
    available on this path). Raised by :func:`assert_static_store_complete`."""


def _store_manifest_engine(out_dir: str | Path):
    """What a static store's RUN MANIFEST says its rows' engine is — for the
    SILENT-row rule of :func:`_assert_static_engine_consistent`. ``None`` when
    the store has no manifest (``served_target.json`` absent); else
    ``(engine_tag, serving_engine, target_engine)``: the row ``engine`` tag the
    manifest's ``target_engine`` stands for (``served`` ->
    ``bon_served.SERVED_ENGINE``; ``hf`` or anything else -> ``None``, no tag,
    as an HF row carries none; a manifest naming no ``target_engine`` -> both
    ``None``), and the ``serving_engine`` it records (``None`` when none). Read
    through ``served_target.read_target_manifest`` — THE SAME LOOKUP the
    serving-engine compare uses (one file name, one parse, one set of
    refusals) — so the two resume guards cannot disagree about which manifest
    a store has; an unreadable manifest is re-raised as this store's error."""
    from .served_target import (SERVING_ENGINE_FIELD, TARGET_ENGINE_SERVED, ServedTargetError,
                                read_target_manifest)
    try:
        manifest = read_target_manifest(out_dir)
    except ServedTargetError as e:
        raise StaticStoreIncompleteError(str(e)) from e
    if manifest is None:
        return None
    target_engine = manifest.get("target_engine")
    if not isinstance(target_engine, str) or not target_engine:
        target_engine = None
    tag = None
    if target_engine == TARGET_ENGINE_SERVED:
        from .bon_served import SERVED_ENGINE
        tag = SERVED_ENGINE
    recorded = manifest.get(SERVING_ENGINE_FIELD)
    return tag, (recorded if isinstance(recorded, str) and recorded else None), target_engine


def _assert_static_engine_consistent(responses_path: str | Path, engine_tag: str, *,
                                     serving_engine: Optional[str] = None) -> None:
    """Refuse to append rows of one engine onto a responses file that holds any
    row of another (an HF row carries no ``engine`` field; a served row carries
    ``engine=SERVED_ENGINE``). The engines are not bit-identical, so mixing them
    in one file — hence one figure — is invalid; a torn tail line is tolerated
    (skipped), matching resume. Mirrors ``bon_served._assert_engine_consistent``
    for the static-family store.

    THE ROW RULE (2026-09-17; one rule with the read-side compare):

    * a row that STATES ``engine`` is compared LITERALLY to ``engine_tag``; a
      torn store (any row stating another engine) refuses whatever the
      manifest says;
    * a SILENT row (no ``engine`` key) is described by the store's RUN
      MANIFEST (``served_target.json``, :func:`_store_manifest_engine`) when
      that manifest names an engine: ``target_engine: served`` stands for
      ``SERVED_ENGINE``, and a recorded ``serving_engine`` must match this
      run's ``serving_engine`` (``None`` = not compared: the bare call).
      Every served static store written while ``ServedChatTarget`` carried no
      ``engine_tag`` (2026-09-10 -> 2026-09-17) is of this shape — silent rows
      beside a manifest naming their engine — and the manifest, the owning
      run's own declaration, is what admits the same-engine resume; the rows
      are never relabelled;
    * the ERA INFERENCE (silent = HF) survives only when there is no manifest
      or the manifest names no engine; a manifest naming ``hf`` (or anything
      but ``served``) refuses a served resume outright.
    """
    p = Path(responses_path)
    if not p.exists():
        return
    from sisteer.capeval.serving.run_identity import geometry_stamp_problem

    from .served_target import TARGET_MANIFEST_FILE
    manifest_path = p.parent / TARGET_MANIFEST_FILE
    described = None       # the manifest's description of a silent row: read once, on the first one
    described_read = False
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn tail from a SIGKILL mid-append
        if "engine" in rec:
            got = rec["engine"]
            if got != engine_tag:
                raise StaticStoreIncompleteError(
                    f"{p} holds a row with engine={got!r} (not {engine_tag!r}) — "
                    "refusing to mix engines in one responses file (HF and "
                    "vLLM-served rows are not bit-identical and belong to different "
                    "figures); rerun this (family, cell) into a fresh dir on one "
                    "engine")
        else:
            # a SILENT row: described by the run manifest, else HF by era
            if not described_read:
                described, described_read = _store_manifest_engine(p.parent), True
            if described is None:
                raise StaticStoreIncompleteError(
                    f"{p} holds a row with no engine field and the store has no run "
                    f"manifest ({manifest_path} absent) — an HF-era row by inference "
                    f"(not {engine_tag!r}); refusing to mix engines in one responses "
                    "file (HF and vLLM-served rows are not bit-identical and belong to "
                    "different figures); rerun this (family, cell) into a fresh dir on "
                    "one engine")
            m_tag, m_serving, m_target = described
            if m_target is None:
                raise StaticStoreIncompleteError(
                    f"{p} holds a row with no engine field and the store's run manifest "
                    f"({manifest_path}) names no engine (no target_engine) — an HF-era "
                    f"row by inference (not {engine_tag!r}); refusing to mix engines in "
                    "one responses file (HF and vLLM-served rows are not bit-identical "
                    "and belong to different figures); rerun this (family, cell) into a "
                    "fresh dir on one engine")
            if m_tag != engine_tag:
                raise StaticStoreIncompleteError(
                    f"{p} holds a row with no engine field that the store's run manifest "
                    f"({manifest_path}) records as target_engine={m_target!r} — not a "
                    f"{engine_tag!r} store; refusing to mix engines in one responses file "
                    "(HF and vLLM-served rows are not bit-identical and belong to "
                    "different figures); rerun this (family, cell) into a fresh dir on "
                    "one engine")
            if m_serving is not None and serving_engine is not None and m_serving != serving_engine:
                raise StaticStoreIncompleteError(
                    f"{p} holds a row with no engine field that the store's run manifest "
                    f"({manifest_path}) describes as served on the {m_serving!r} serving "
                    f"engine, while this run serves on {serving_engine!r} — refusing to "
                    "append rows of another engine to this store (a lens row and a native "
                    "row are two measurements); run this cell into a fresh dir")
        # the GEOMETRY stamp (run_identity): one serving geometry per store —
        # bf16 greedy output is not batch-composition invariant, so rows of
        # two geometries are two measurements (review R1)
        problem = geometry_stamp_problem(rec)
        if problem is not None:
            raise StaticStoreIncompleteError(
                f"{p}: {problem} — refusing to append rows of this run's serving "
                "geometry onto a store of another; rerun into a fresh dir")


def _assert_static_serving_engine_consistent(responses_path: str | Path,
                                             serving_engine: Optional[str]) -> None:
    """ONE SERVING ENGINE PER STORE: refuse
    to append rows served on ``serving_engine`` (``"lens"`` | ``"lora_native"``,
    ``served_target.serving_engine_of(defense)``) onto a responses file — or a
    store whose ``served_target.json`` sidecar — that names another. A row (or
    sidecar) WITHOUT ``serving_engine`` predates the native engine and was lens
    BY CONSTRUCTION, so a lens run resumes it (the in-flight-campaign case) and
    a native run refuses it. ``None`` (the HF path, which stamps nothing) checks
    nothing. Runs BEFORE :func:`filter_responses_to_covered` in ``run_static``,
    so a refusal leaves ``responses.jsonl`` and ``progress.jsonl`` byte-identical.
    The compare itself is ``served_target.assert_store_serving_engine_compatible``
    (one rule for the static, multi-turn and PyRIT writers), imported lazily —
    this module stays import-light for the HF families — and re-raised as this
    store's :class:`StaticStoreIncompleteError`, like the engine-tag guard."""
    if serving_engine is None:
        return
    from .served_target import ServedTargetError, assert_store_serving_engine_compatible

    p = Path(responses_path)
    try:
        assert_store_serving_engine_compatible(p.parent, serving_engine, responses_name=p.name)
    except ServedTargetError as e:
        raise StaticStoreIncompleteError(str(e)) from e


def _assert_static_decode_consistent(responses_path: str | Path,
                                     expected: Optional[Mapping[str, Any]]) -> None:
    """ONE DECODER PER STORE (DESIGN.md invariant): refuse to append rows of
    this run's decoder onto a responses file holding any row of another —
    a greedy row (no ``decode`` block) while this run samples, a sampled row
    while this run is greedy, or a sampled row under a DIFFERENT block (another
    preset, cap, arm, seed_base or trial). Greedy-vs-sampled is the contrast
    the sampled re-run measures, so one store must never hold both; a torn
    tail line is tolerated (skipped), matching resume. The same shape as
    :func:`_assert_static_engine_consistent`, run on every resume regardless of
    engine (the wave's greedy served rows carry no engine tag)."""
    p = Path(responses_path)
    if not p.exists():
        return
    want = dict(expected) if expected is not None else None
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn tail from a SIGKILL mid-append
        got = rec.get("decode")
        if got == want:
            continue
        if want is None:
            raise StaticStoreIncompleteError(
                f"{p} holds a SAMPLED row (decode={got!r}) while this run decodes "
                "GREEDY — refusing to mix decoders in one responses file (greedy "
                "vs sampled is the contrast being measured); run into a fresh "
                "private_run")
        if got is None:
            raise StaticStoreIncompleteError(
                f"{p} holds a GREEDY row (no decode block) while this run samples "
                f"under {want!r} — refusing to mix decoders in one responses file; "
                "a sampled run needs its own private_run, never a greedy store")
        raise StaticStoreIncompleteError(
            f"{p} holds a row decoded under {got!r}, not this run's {want!r} — "
            "refusing to mix decoders (preset / cap / arm / seed_base / trial) "
            "in one responses file; run into a fresh private_run")


def assert_static_store_complete(summary, out_dir: str | Path, *, family: str,
                                 n_errors: int = 0) -> None:
    """FAIL LOUDLY if a static family's store is short, errored, or empty.

    ``run_static`` swallows a per-batch generation exception (finalizing the
    batch's goals as ``error:<type>`` and continuing) — so a family that failed
    every batch returns a normal-looking summary while its ``responses.jsonl``
    is empty or short. Called after a SERVED family, this reconciles the store
    against its own progress records so that landmine is a loud abort, not a
    silent "complete" cell:

    - ``n_errors > 0`` (served requests that failed even under the record-mode
      policy) → refuse.
    - every terminal progress record must be outcome ``done`` (no ``error:*``).
    - the response-row count must equal the summed ``n_variants`` of the covered
      goals (== the rows the unbatched loop would have written), and be > 0.

    A loud, recorded skip (``summary["skipped"]`` — e.g. renellm's corpus is
    absent) is NOT an incompleteness and passes through. Raises
    :class:`StaticStoreIncompleteError` on any shortfall."""
    if isinstance(summary, dict) and summary.get("skipped"):
        return
    if n_errors:
        raise StaticStoreIncompleteError(
            f"{family}: {n_errors} served request(s) failed — refusing to treat "
            "the store as complete (an empty/short cell that looks complete is "
            "the worst outcome available here)")
    out_dir = Path(out_dir)
    progress_path = out_dir / "progress.jsonl"
    responses_path = out_dir / "responses.jsonl"
    recs: list[dict] = []
    if progress_path.exists():
        for line in progress_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not recs:
        raise StaticStoreIncompleteError(
            f"{family}: no terminal progress records under {out_dir} — the "
            "family generated nothing (an empty cell that looks complete is the "
            "worst outcome available here); refusing")
    errored = [r.get("goal_id") for r in recs
               if str(r.get("outcome", "")).startswith("error")]
    if errored:
        raise StaticStoreIncompleteError(
            f"{family}: {len(errored)} goal(s) finalized with an error outcome "
            f"(first: {errored[0]!r}) — a swallowed batch failure; refusing to "
            "treat the store as complete")
    expected = sum(int(r.get("n_variants", 0)) for r in recs)
    actual = 0
    if responses_path.exists():
        actual = sum(1 for line in responses_path.read_text().splitlines()
                     if line.strip())
    if expected == 0:
        raise StaticStoreIncompleteError(
            f"{family}: every one of {len(recs)} covered goal(s) had ZERO "
            "variants (expected row count 0) — the variant builder produced "
            "nothing for this goal set; refusing (empty cell)")
    if actual != expected:
        raise StaticStoreIncompleteError(
            f"{family}: {actual} response rows != {expected} expected (sum of "
            f"n_variants over {len(recs)} covered goals) — the store is short; "
            "refusing (an empty/short cell that looks complete is the worst "
            "outcome available here)")


def load_persisted_responses(responses_path: str | Path) -> dict[str, list[str]]:
    """``{goal_id: [response, ... in sample_idx order]}`` from a static family's
    private ``responses.jsonl`` (the input to StrongREJECT grading). Reassembles
    by ``sample_idx`` — the static families' variant index."""
    return load_ordered_jsonl(responses_path, order_key="sample_idx",
                              value_key="response")


# -- the shared run loop -------------------------------------------------------


# One-shot latch so the FIRST per-goal error is announced with its message
# and traceback, and the rest are silent. A list rather than a bool so the
# closure can set it without a global statement.
_ERR_ANNOUNCED: list = []


def goal_label(g) -> str:
    return str(getattr(g, "goal_id", g))


def run_static(
    defense,
    goals,
    *,
    out_dir,
    family: str,
    variants_fn,
    trial: int = 0,
    seed: int = 0,
    max_new_tokens: int = 512,
    resume: bool = True,
    progress_extra_fn=None,
    batch_size: int | None = None,
    decode: Mapping[str, Any] | None = None,
    seed_base: int | None = None,
    padding: "PadPlan | None" = None,
) -> dict:
    """Replay one static family over ``goals`` against ``defense``, BATCHED.

    PADDING (``padding``; :mod:`static_padding`). A :class:`~static_padding.PadPlan`
    inserts a MEASURED number of neutral filler tokens into every prompt before
    it is sent, before or after the family's transform output. The row keeps
    ``variant`` / ``attack_prompt`` as the UNPADDED transform output (one meaning
    for those fields across padded and unpadded stores) and ADDITIONALLY records
    ``padding`` = the plan's store-identity block plus the per-item record
    (``pad_tokens_target``, ``pad_tokens_actual``, ``filler_offset_tokens``,
    ``padded_prompt_sha256`` — what was actually sent, pinned); the terminal
    progress record carries ``padding_tokens`` and the summary ``padding``. Every
    padded prompt of a chunk is built BEFORE its request is issued, so a padding
    failure raises with nothing written and the chunk's goals uncovered for a
    resume (the loud served policy). The resume guard refuses a store holding
    padded rows when this run pads nothing, unpadded rows when it pads, or rows
    under another padding block — padded vs unpadded is a contrast measured
    ACROSS stores. With ``padding=None`` every row, file and call is
    BYTE-IDENTICAL to before.

    SAMPLED DECODING (``decode`` + ``seed_base``; DESIGN.md of the full22
    sampled re-run). ``decode`` = the vendor sampler preset (SAMPLER_KEYS) plus
    ``arm``; when given, every variant is drawn with ``do_sample=True`` under
    that preset through the defense's DETAILED surface
    (``defense.generate_detailed(prompts, max_new_tokens=, decode=, seeds=, labels=)``,
    the served target's — there is no HF leg, a defense without it is
    refused), seeded per item with :func:`static_sampling_seed`
    ``(seed_base, trial, family, goal_id, sample_idx)``, and every row
    ADDITIONALLY records ``decode`` (:func:`decode_block`), ``sampling_seed``,
    ``finish_reason``, ``completion_tokens``, ``prompt_tokens``, ``max_tokens``
    (actually sent), ``at_cap`` (finish_reason == "length") and the clamp
    fields when the window clamp fired. A sampled chunk that fails RAISES
    (nothing finalized, the goals stay uncovered for a resume) — the loud
    served policy, since only a served target can sample here. The resume
    guard refuses a store holding rows of another decoder (or greedy rows).
    With ``decode=None`` every row and file is BYTE-IDENTICAL to before.

    ``variants_fn(goal, seed) -> list[(variant_name, prompt)]`` MUST be
    deterministic and defense-independent (same seed -> identical prompts).
    Prompts are generated in DETERMINISTIC fixed-size batches (the earlier
    one-prompt-at-a-time loop made every fixed-prompt leg batch-1 and was the
    sweep's dominant cost): the goal-major, variant-order work list is chunked
    into consecutive slices of ``batch_size`` (default: the defense's own
    ``batch_size``), each served through ONE ``defense.generate(chunk)`` call.
    Fixed order + fixed size means a re-run reproduces the same batches.
    Response rows are appended IN THE SAME ORDER as the unbatched loop (the
    flat order is preserved because chunks are consecutive slices), and a
    goal's terminal ``progress.jsonl`` record is written ONLY once all its
    variants have generated — so per-goal resume is unchanged (skip goals with
    a terminal record; drop an interrupted goal's partial rows first; a crash
    mid-batch leaves its goals uncovered and their partial rows are dropped on
    resume).

    BATCH COMPOSITION IS SERVING-RELEVANT: bf16 batched generation is not
    byte-identical to unbatched (known from the n=400 work), so the batch size
    used is recorded in the returned summary (and the sweep records it on the
    engine stamp) — not a pairing-refusal axis, but a recorded identity field so
    a per-item divergence is explicable. Per-row RESPONSE-mask correctness under
    the left padding this batching forces is pinned by
    tests/test_phase_probe_gpu.py (mixed-length batch: every row's assistant
    header is its last H columns, so the scalar boundary is exact per row).

    A failed batch marks every goal with a variant in that batch as
    ``error:<type>`` (terminal, like the old per-goal error record) and skips
    those goals' remaining variants; other goals continue. NO forward cap is
    applied (these families are replayable/exempt like GCG), and grading is the
    separate grading host post-step. ``progress_extra_fn(goal) -> dict`` may add
    family-specific fields to the per-goal record."""
    out_dir = ensure_private_dir(out_dir)
    responses_path = out_dir / "responses.jsonl"
    progress_path = out_dir / "progress.jsonl"

    # A served defense self-describes its engine (SERVED_ENGINE); an HF defense
    # exposes none. When present it is (a) stamped on every row so a served
    # store never silently mixes with HF rows, and (b) the switch onto the LOUD
    # error path below. Absent (HF) leaves the row schema and error semantics
    # BYTE-IDENTICAL to before.
    engine_tag = getattr(defense, "engine_tag", None)
    if engine_tag is None and getattr(defense, "serving_engine", None) is not None:
        # A defense that STATES a serving engine is a served target; one that carries no
        # engine_tag would fall through every ``if engine_tag is not None`` below and write
        # HF-shaped rows (no ``engine``, no ``serving_engine``, no guards, the swallow-error
        # path) that nothing downstream can tell from an HF store. That is exactly how every
        # served sampled-static store lost its provenance (an untagged ServedChatTarget,
        # 2026-09-17); refuse the caller's hole loudly, before a byte is written. The HF
        # target has neither attribute and never reaches this branch.
        raise TypeError(
            f"[{family}] {type(defense).__name__} states serving_engine="
            f"{getattr(defense, 'serving_engine')!r} but carries no engine_tag — a served "
            "defense handed to run_static must declare engine_tag (bon_served.SERVED_ENGINE, "
            "as ServedChatTarget and ServedStaticDefense do) or its rows would be written "
            "as HF rows with no engine provenance; refusing")
    # THE SERVING ENGINE: a served defense states it (lens /
    # lora_native, read once from the proxy identity by served_target /
    # bon_served); stamped beside ``engine`` on every served row and held
    # constant per store by the guard below. None on the HF path (no import,
    # no field, no check — byte-identical).
    serving_engine: Optional[str] = None
    if engine_tag is not None:
        from .served_target import serving_engine_of
        serving_engine = serving_engine_of(defense)

    # SAMPLED decoding: validate everything BEFORE any row is written or any
    # request issued — the block recorded on every row, the request preset,
    # and the detailed surface (served only; never an HF leg).
    block: Optional[dict] = None
    preset: Optional[dict] = None
    if decode is not None:
        if seed_base is None:
            raise ValueError(
                f"[{family}] run_static(decode=...) needs seed_base — every "
                "sampled draw is seeded from (seed_base, trial, family, goal_id, "
                "sample_idx); refusing an unseeded sampled run")
        block = decode_block(decode, cap=max_new_tokens, seed_base=seed_base,
                             trial=trial)
        preset = sampler_preset_of(block)
        if not callable(getattr(defense, "generate_detailed", None)):
            raise TypeError(
                f"[{family}] sampled decoding needs the served target's detailed "
                "surface (defense.generate_detailed: finish_reason per row is what "
                f"the at-cap ladder keys on); {type(defense).__name__} has none — "
                "the HF path is not a sampled leg; refusing")

    # PADDING: validate the plan BEFORE any row is written or any request
    # issued — the store-identity block recorded on every row, the renderer,
    # and the token count the progress record carries. Duck-typed (a PadPlan
    # exposes all three); a bare int, a PadSpec, or anything else is refused
    # naming what was passed, like the decode refusals above.
    plan = None
    pad_block: Optional[dict] = None
    if padding is not None:
        got_block = getattr(padding, "block", None)
        got_tokens = getattr(getattr(padding, "spec", None), "tokens", None)
        if (not isinstance(got_block, Mapping)
                or not callable(getattr(padding, "apply", None))
                or isinstance(got_tokens, bool) or not isinstance(got_tokens, int)):
            raise TypeError(
                f"[{family}] run_static(padding=...) needs a static_padding.PadPlan "
                "(a .block mapping, a callable .apply and .spec.tokens), got "
                f"{type(padding).__name__} — build one with PadPlan(PadSpec(...), tok, "
                "filler_text); None means the knob is off; refusing")
        plan = padding
        pad_block = dict(got_block)

    covered = load_covered_goals(progress_path) if resume else set()
    if resume:
        if engine_tag is not None:
            # never append served rows onto HF rows (or vice versa) — the
            # engines are not bit-identical and belong to different figures —
            # and never rows of one SERVING ENGINE onto a store of another.
            # A row that states its engine is compared literally; a SILENT row
            # (the served static stores written before ServedChatTarget carried
            # engine_tag) is described by the store's run manifest
            # (served_target.json: target_engine + serving_engine) and read as
            # HF by era only when no manifest names an engine.
            # BOTH run BEFORE filter_responses_to_covered (the
            # earlier order rewrote the file first), so a refused cross-engine
            # attempt leaves responses.jsonl and progress.jsonl byte-identical.
            _assert_static_engine_consistent(responses_path, engine_tag, serving_engine=serving_engine)
            _assert_static_serving_engine_consistent(responses_path, serving_engine)
        # never append PADDED rows onto an unpadded store (or vice versa), nor
        # rows under another padding block — the contrast is measured across
        # stores. This sits WITH the engine guards, BEFORE
        # filter_responses_to_covered, and NOT beside the decode guard below:
        # the filter REWRITES responses.jsonl (dropping an interrupted goal's
        # partial rows), so a guard placed after it alters a store it then
        # refuses to write to. A refused padding mismatch must leave
        # responses.jsonl and progress.jsonl byte-identical; the decode guard's
        # inherited post-filter position is left where it is so the decode
        # path stays byte-identical to before (moving it is its own change).
        assert_static_padding_consistent(responses_path, pad_block)
        filter_responses_to_covered(responses_path, covered)
        # never append rows of one DECODER onto a store of another (greedy vs
        # sampled, or two presets) — the contrast being measured lives across
        # stores, never inside one.
        _assert_static_decode_consistent(responses_path, block)
    todo = [g for g in goals if g.goal_id not in covered]

    bs = int(batch_size if batch_size is not None
             else (getattr(defense, "batch_size", 8) or 8))
    bs = max(1, bs)

    # The deterministic goal-major work list; per-goal state for the
    # write-progress-only-when-complete contract.
    work: list[tuple] = []          # (goal_id, sample_idx, name, prompt)
    state: dict[str, dict] = {}     # goal_id -> {goal, n, done, outcome}
    for g in todo:
        variants = list(variants_fn(g, seed))
        state[g.goal_id] = {"goal": g, "n": len(variants), "done": 0,
                            "outcome": "done"}
        for i, (name, prompt) in enumerate(variants):
            work.append((g.goal_id, i, name, prompt))

    per_goal: list[dict] = []
    finalized: set[str] = set()

    def _finalize(gid: str) -> None:
        st = state[gid]
        g = st["goal"]
        rec = {
            "goal_id": gid, "category": g.category, "family": family,
            "n_variants": st["n"],
            # one target forward per generated prompt — identical accounting to
            # the unbatched loop (defense.generate counts len(prompts), JB7).
            "n_target_forwards": st["done"],
            "outcome": st["outcome"],
            "capped": False,  # static/replayable families are forward_cap-exempt
        }
        if plan is not None:
            # the padding width this store was generated under (ids and counts
            # only — progress.jsonl never carries prompt text)
            rec["padding_tokens"] = int(plan.spec.tokens)
        if progress_extra_fn is not None:
            rec.update(progress_extra_fn(g))
        append_jsonl(progress_path, rec)
        per_goal.append(rec)
        finalized.add(gid)

    n_batches = 0
    pos = 0
    while pos < len(work):
        # consecutive fixed-size slices, skipping items of already-failed goals
        # (deterministic given the same inputs and the same failure pattern).
        chunk: list[tuple] = []
        while pos < len(work) and len(chunk) < bs:
            item = work[pos]
            pos += 1
            if item[0] in finalized:  # a goal failed in an earlier batch
                continue
            chunk.append(item)
        if not chunk:
            continue
        n_batches += 1
        # WHAT IS SENT: the transform output as-is, or — under a plan — every
        # item's padded rendering, built for the WHOLE chunk before a request
        # is issued, so a padding failure raises here with nothing written and
        # the chunk's goals uncovered for a resume (the loud served policy;
        # never an error placeholder row). With padding=None this is the same
        # list the loop always sent.
        pad_recs: Optional[list] = None
        if plan is not None:
            sent_texts: list = []
            pad_recs = []
            for (gid, i, _name, prompt) in chunk:
                sent, prec = plan.apply(prompt, goal_id=gid)
                sent_texts.append(sent)
                pad_recs.append(prec)
        else:
            sent_texts = [it[3] for it in chunk]
        details: list = []
        if block is not None:
            # SAMPLED: the detailed surface, one seed per item. A failure
            # RAISES like the served path below (nothing finalized, the chunk's
            # goals stay uncovered for a resume): only a served target can
            # sample here, and an error placeholder must never become a row.
            seeds = [static_sampling_seed(seed_base, trial, family, gid, i)
                     for (gid, i, _name, _prompt) in chunk]
            # request-evidence: ONE label per
            # request — the store's canonical row task (``static/<family>``, the string
            # the audit's ``filed_pairs_static`` / ``read_static_store`` key by) and the
            # row's own ``(goal_id, sample_idx)``. The driver is the ONE caller that
            # knows the item identity, so it labels the send; with the hook OFF the
            # served target ignores the labels (byte-identical request), with it ON an
            # unlabelled task request is refused by name (every static child died in
            # ``require_label`` when this call carried no ``labels=``).
            from ..request_evidence_client import (BINDING_FIELDS, LEG_STATIC,
                                                   store_task_name)
            task_label = store_task_name(LEG_STATIC, family)
            labels = [(task_label, str(gid), i) for (gid, i, _name, _prompt) in chunk]
            details = defense.generate_detailed(sent_texts,
                                                max_new_tokens=max_new_tokens,
                                                decode=preset, seeds=seeds,
                                                labels=labels)
            if len(details) != len(chunk):
                raise RuntimeError(
                    f"[{family}] generate_detailed returned {len(details)} records "
                    f"for {len(chunk)} prompts — refusing a misaligned chunk")
            for d, seed in zip(details, seeds):
                if d.get("error") is not None:
                    raise RuntimeError(
                        f"[{family}] a sampled request failed: {d['error']} — "
                        "refusing to write an error placeholder as a row")
                if d.get("sampling_seed") != seed:
                    raise RuntimeError(
                        f"[{family}] the detailed surface echoed sampling_seed "
                        f"{d.get('sampling_seed')!r} for a draw issued with {seed} "
                        "— refusing a row whose seed provenance disagrees")
                if d.get("finish_reason") is None:
                    raise RuntimeError(
                        f"[{family}] the server returned no finish_reason for a "
                        "sampled draw — the at-cap ladder keys on it; refusing to "
                        "write a row that cannot say whether it terminated")
            responses = [d["text"] for d in details]
        else:
            try:
                responses = defense.generate(sent_texts,
                                             max_new_tokens=max_new_tokens)
            except Exception as exc:  # noqa: BLE001
                if engine_tag is not None:
                    # SERVED path: do NOT finalize the chunk's goals (leave
                    # them UNCOVERED so a resume regenerates them, exactly
                    # like a SIGKILL) and FAIL LOUDLY — writing zero rows and
                    # no progress record here, so nothing is torn and the
                    # store never looks complete-but-empty. A served
                    # giant-chunk fails all-or-nothing
                    # (ServedStaticDefense.generate raises AFTER the gather,
                    # before any row is appended), so there is nothing
                    # partial to clean.
                    raise
                # HF path (BYTE-IDENTICAL to before): record every affected
                # goal as terminal-error and continue with the other goals.
                for gid in dict.fromkeys(it[0] for it in chunk):
                    if gid not in finalized:
                        state[gid]["outcome"] = f"error:{type(exc).__name__}"
                        _finalize(gid)
                continue
        for k, ((gid, i, name, prompt), resp) in enumerate(zip(chunk, responses)):
            row = {
                "goal_id": gid,
                "sample_idx": i,
                "trial": trial,
                "variant_key": variant_key(i, trial),
                "variant_name": name,
                "family": family,
                # live-only prompt text (private 0700 dir), both names so the
                # downstream labelling step reads it whether it expects the
                # BoN field (``variant``) or the store field (``attack_prompt``).
                "variant": prompt,
                "attack_prompt": prompt,
                "response": resp,
            }
            if engine_tag is not None:
                # provenance on every served row (HF rows carry no such field —
                # byte-identical); the resume guard above keeps the two apart.
                row["engine"] = engine_tag
                if serving_engine is not None:
                    # the SERVING ENGINE beside the engine tag:
                    # lens / lora_native, so a passthrough row of either era
                    # is self-describing; the resume guard holds it per store
                    row["serving_engine"] = serving_engine
                # the run's geometry stamp (run_identity.stamp_env), when a
                # launcher exported one — a mixed store is detectable per row
                from sisteer.capeval.serving.run_identity import geometry_stamp
                row.update(geometry_stamp())
            if block is not None:
                # the sampled row's decode provenance (DESIGN.md row schema):
                # the block, the seed, the server's termination facts, and
                # at_cap — what the cap-escalation ladder selects on.
                d = details[k]
                row["decode"] = dict(block)
                row["sampling_seed"] = int(d["sampling_seed"])
                row["finish_reason"] = d["finish_reason"]
                row["completion_tokens"] = d.get("completion_tokens")
                row["prompt_tokens"] = d.get("prompt_tokens")
                row["max_tokens"] = int(d["max_tokens"])
                row["at_cap"] = (d["finish_reason"] == "length")
                if d.get("max_tokens_clamped_from") is not None:
                    row["max_tokens_clamped_from"] = int(d["max_tokens_clamped_from"])
                    row["prompt_tokens_source"] = d.get("prompt_tokens_source")
                if d.get("window_mismatch") is not None:
                    row["window_mismatch"] = d["window_mismatch"]
                # the attempt binding the served target stamped on the record
                # (request evidence ON: ``{request_id, completion_id}`` — the
                # audit's ``bind_rows`` reads them off the row; a static row
                # without them "binds to no attempt"); absent (hook off) the
                # row's key set is unchanged
                for key in BINDING_FIELDS:
                    if d.get(key) is not None:
                        row[key] = d[key]
            if pad_recs is not None:
                # the padding provenance: the store-identity block (what every
                # row of this store shares; the resume guard compares it) plus
                # this item's record (target / actual / offset / sha256 of the
                # text actually sent). ``variant`` / ``attack_prompt`` above
                # stay the UNPADDED transform output.
                row["padding"] = {**pad_block, **pad_recs[k]}
            append_jsonl(responses_path, row)
            st = state[gid]
            st["done"] += 1
            if st["done"] == st["n"] and gid not in finalized:
                _finalize(gid)
    # zero-variant goals never enter a batch — finalize them terminally.
    for gid, st in state.items():
        if gid not in finalized and st["n"] == 0:
            _finalize(gid)

    # Per-family COVERAGE, stated rather than inferable (2026-08-28): a
    # frozen-corpus family can cover fewer goals than it is run on
    # (renellm's corpus covers only the original 259 test+JBB goal ids, so
    # held-out VAL goals yield zero variants — ~92/112 goals where every
    # other family runs all 112). Zero-variant goals finalize with
    # n_variants=0 above; this count makes the family's REAL n visible in
    # the summary so per-family tables print their own denominator and a
    # composite never averages rates over unequal n as if they matched.
    n_with_variants = sum(1 for st in state.values() if st["n"] > 0)

    # A pass that ran goals and produced NO responses is a failed run, not a
    # complete one. Without this, an exception common to every goal exits 0
    # with "goals_run=N/N" and an absent responses file -- measured on job
    # 206660. The errored count comes from the records just written, so this
    # cannot pass vacuously on a resume that legitimately had nothing to do
    # (todo empty -> no claim made).
    n_err = sum(1 for r in per_goal if str(r.get("outcome", "")).startswith("error:"))
    if todo and n_err == len(todo):
        raise RuntimeError(
            f"[{family}] EVERY goal errored ({n_err}/{len(todo)}) and no "
            f"responses were written to {responses_path}. Reporting this as "
            f"a completed run would make a total failure indistinguishable "
            f"from success -- see the first-error traceback above for the "
            f"cause.")
    summary = {
        "responses_path": str(responses_path),
        "progress_path": str(progress_path),
        "per_goal": per_goal,
        "n_goals_run": len(todo),
        "n_goals_total": len(goals),
        "n_goals_with_variants": n_with_variants,
        "n_goals_zero_variants": len(todo) - n_with_variants,
        "n_target_forwards_total": defense.n_target_forwards,
        "batch_size": bs,
        "n_batches": n_batches,
        "n_goals_errored": n_err,
    }
    if block is not None:
        # the sampled run's decode provenance, for the driver's manifest (a
        # greedy run's summary is unchanged — no key added)
        summary["decode"] = dict(block)
    if pad_block is not None:
        # the padded run's store-identity block, for the driver's manifest (an
        # unpadded run's summary is unchanged — no key added)
        summary["padding"] = dict(pad_block)
    return summary
