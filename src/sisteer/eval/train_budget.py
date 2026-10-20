"""Per-(SI) sequence budget and per-(SI, arm) batch size — resolution plus a
CPU-only measurement harness (efficiency optimisations 1, 2 and 5).

Two knobs, deliberately kept SEPARATE because they mean different things:

- ``max_len`` is a MEMORY knob (opt. 1). One shared ``max_len`` sizes both SI
  arms for the worst case: the safety arm's teacher prompt is ~129 tokens but a
  shared ``max_len: 8192`` reserves activation memory for ~12x that. Per-SI
  ``max_len`` (``max_len_by_si``, falling back to the scalar ``max_len``) sizes
  each SI to its own measured prompt + the SHARED completion cap + EOS +
  headroom. This does NOT touch what either arm learns: as long as each SI's
  ``max_len`` holds prompt + FULL completion (no truncation), the completion
  tokens entering the loss are IDENTICAL across SIs — the comparability
  invariant (asserted by tests/test_train_budget.py). Only the SHARED knob
  that must stay shared is ``completion_tokens`` (equal supervision volume),
  NOT ``max_len``.

- batch size is a THROUGHPUT knob (opt. 2). One global batch size lets the
  constitution's constraint cap the safety arm too. Per-(SI, arm) resolution
  (``batch_size_by_si`` + per-arm ``batch_size_by_arm``, with a fallback) lets
  each cell run as wide as it safely can. The one HARD constraint encoded
  here: the ``alphakl_fisher`` rung forces math-backend SDPA inside its
  forward-AD passes (flash has no forward-mode AD), so attention memory is
  O(T^2) and it must run at batch 1-2; plain distillation is O(T) and can go
  much wider. Concrete widths are NOT invented here — the measurement harness
  reports the shapes/token counts that drive memory, and a GPU pass fills in
  peak memory and tokens/s.

The measurement harness (``measure_si_budget`` / ``format_budget_table``) is
CPU-only: it tokenizes the rendered prompts and completions, counts trainable
parameters (building the object on the loaded model — no forward), and reports
the O(T)-vs-O(T^2) attention cost class and a FLOP-proportional per-step work
proxy. Peak memory and tokens/s are explicitly marked NEEDS_GPU; a ``--time``
estimate is derived only once a measured tokens/s is supplied.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

# The arm whose forward-AD passes force O(T^2) attention memory (math-backend
# SDPA — flash has no forward-mode AD). Its batch size is HARD-capped
# regardless of config, because the cap is a memory fact, not a preference.
FISHER_ARMS: frozenset[str] = frozenset({"alphakl_fisher"})
FISHER_MAX_BATCH = 2

# The default batch size when neither config nor a per-arm override applies.
DEFAULT_BATCH_SIZE = 2

_NEEDS_GPU = "NEEDS_GPU"


# --------------------------------------------------------------------------- #
# Training-budget SCHEMA (sdf-base-steering §5.3): document-token fits          #
# --------------------------------------------------------------------------- #
# The legacy schema (every SI-pair fit to date) has NO ``training_budget_kind``
# key: the budget is the shared per-example completion cap
# (``completion_tokens``, in-code fallback 512), and every consumer keeps that
# behavior byte-for-byte. The document-CE fits carry
# ``training_budget_kind: "document_tokens"``: the cell's budget is document
# tokens under the sampler's official counting rule (the sample manifest,
# pinned per fit via ``meta_extra.sample_manifest_sha``), there is NO
# completion cap (``completion_tokens: null``, explicit), and the sequence
# bound is ``sequence_max_len``. Writing the legacy 512 fallback into a
# document fit's sidecar would be FALSE provenance the serving-side budget
# check then reads as a real training cap — hence the explicit kind and the
# branching helpers below (consumed by the sidecar writer in
# scripts/train_two_si_objects.py and the sidecar budget checks).

DOCUMENT_TOKENS_KIND = "document_tokens"


def resolve_budget_kind(cfg_or_meta: dict) -> str | None:
    """The training-budget schema of a fit config or budget sidecar:
    :data:`DOCUMENT_TOKENS_KIND` for the document-mode schema, ``None`` for
    the legacy SI-pair schema (key absent — every existing fit). Any OTHER
    value refuses loudly: an unrecognized kind silently treated as legacy
    would re-manufacture exactly the false-512 provenance this schema
    exists to prevent."""
    kind = (cfg_or_meta or {}).get("training_budget_kind")
    if kind is None or kind == DOCUMENT_TOKENS_KIND:
        return kind
    raise ValueError(
        f"unknown training_budget_kind {kind!r} — known kinds: "
        f"{DOCUMENT_TOKENS_KIND!r} (document-CE fits) or the key absent "
        "(legacy SI-pair fits). Refusing to guess a budget schema.")


def document_budget_problems(meta: dict, *,
                             expected_sample_manifest_sha: str | None = None,
                             require_sample_manifest_sha: bool = False,
                             ) -> list[str]:
    """The budget check's ``document_tokens`` branch, as ONE shared
    verdict for every sidecar consumer (the transfer/serving checks in
    two_si_run/defense_cell/cell_controls branch here on
    ``training_budget_kind``; SI-pair sidecars must never reach this
    function — their legacy checks are byte-identical elsewhere).

    A document sidecar must carry the kind, a positive integer
    ``sequence_max_len``, an explicitly-null ``completion_tokens`` (a
    number here means something other than the schema wrote the sidecar),
    and a ``sample_manifest_sha`` field (null = a stated absence — a
    synthetic pool with no staged manifest; a MISSING key predates the
    schema). ``expected_sample_manifest_sha``, when supplied by a consumer
    that staged the manifest, must match the recorded one.
    ``require_sample_manifest_sha=True`` is the SERVING-side strictness
    (two_si_run's budget gate): there a null sha — however honestly stated —
    is itself a problem, because generating from an object whose sample
    provenance cannot be verified is exactly what that gate exists to
    refuse; resume/retrain consumers keep the default and accept the
    stated absence. Returns problem strings (empty = valid)."""
    problems: list[str] = []
    if meta.get("training_budget_kind") != DOCUMENT_TOKENS_KIND:
        problems.append(
            f"training_budget_kind={meta.get('training_budget_kind')!r} — "
            f"not a {DOCUMENT_TOKENS_KIND} sidecar")
        return problems
    sml = meta.get("sequence_max_len")
    if not isinstance(sml, int) or isinstance(sml, bool) or sml < 1:
        problems.append(
            f"sequence_max_len={sml!r} — a document fit's sequence bound "
            "must be a positive integer")
    if meta.get("completion_tokens") is not None:
        problems.append(
            f"completion_tokens={meta.get('completion_tokens')!r} — a "
            "document fit has no completion cap; the schema records null")
    if "sample_manifest_sha" not in meta:
        problems.append(
            "sample_manifest_sha key missing — the sidecar predates the "
            "document schema (null is a stated absence; absence of the "
            "key is not)")
    elif require_sample_manifest_sha and meta.get(
            "sample_manifest_sha") is None:
        problems.append(
            "sample_manifest_sha is null — a stated absence (no staged "
            "sample manifest); this consumer requires verified sample "
            "provenance")
    elif (expected_sample_manifest_sha is not None
          and meta.get("sample_manifest_sha")
          != expected_sample_manifest_sha):
        problems.append(
            f"sample_manifest_sha={meta.get('sample_manifest_sha')!r} does "
            f"not match the staged manifest "
            f"{expected_sample_manifest_sha!r} — this object was trained "
            "on a different sample than the one staged")
    return problems


# --------------------------------------------------------------------------- #
# Consumer-level staged-cell identity validation (sdf-base-steering)           #
# --------------------------------------------------------------------------- #
# ONE verdict for "is this staged cell dir the artifact it claims to be",
# run by every consumer BEFORE any model load: the explicit-cell registry
# build (defense_cell.registry_from_explicit_cells), cell_controls_run's
# generate phase, and the campaign measurement scripts
# (scripts/sdfbase/base_mmlu.py, scripts/sdfbase/probe_score.py).
# two_si_run._check_budget_sidecar predates this function and carries the
# same identity checks plus its own config-cap comparison; the kind
# resolution and document-schema verdict are shared through
# resolve_budget_kind / document_budget_problems in BOTH, so there is one
# schema truth even where the wrappers differ.


def staged_cell_problems(object_path, cell_id: str, *,
                         require_sidecar: bool = True,
                         require_sample_manifest_sha: bool = False,
                         expected_sample_manifest_sha: str | None = None,
                         ) -> list[str]:
    """Problem strings (empty = valid) for ONE staged cell artifact,
    CPU-cheap and model-free, so a wrong-cell or stale artifact refuses in
    milliseconds instead of after a GPU load. Checks:

    - the ``train_budget_meta.json`` sidecar exists (unless
      ``require_sidecar=False`` — the Class B exemption: those fits write
      no sidecar; a PRESENT one still validates) and parses;
    - the sidecar's ``cell_id`` is the REQUESTED cell (the one accepted
      mismatch is the cell's declared legacy spelling —
      ``defense_cell.legacy_cell_id`` / methods.LEGACY_ARM_ALIASES);
    - the budget kind is valid (``resolve_budget_kind`` — an unknown kind
      refuses; ``document_tokens`` sidecars additionally pass the ONE
      document-schema verdict :func:`document_budget_problems`, with the
      caller's ``require_sample_manifest_sha`` strictness knob: serving
      consumers refuse a stated-absence null, resume-grade consumers
      accept it);
    - the sidecar's ``object_sha256`` matches the on-disk artifact bytes
      (a stale object/sidecar pair is never trusted);
    - the COMPLETE marker, WHEN PRESENT, is coherent: it parses, names
      this cell (legacy spelling accepted), and every file its manifest
      promises exists with the recorded sha256 — a torn or tampered
      publication must not read as a valid cell. An ABSENT marker is not
      a problem: staged handovers may ship object+sidecar only, and the
      sidecar checks above already bind the identity.
    """
    from .defense_cell import legacy_cell_id, sha256_file

    obj_path = Path(object_path)
    problems: list[str] = []
    if not obj_path.is_file():
        return [f"no staged artifact at {obj_path}"]
    legacy = legacy_cell_id(cell_id)
    accepted_ids = {cell_id} | ({legacy} if legacy else set())

    meta = None
    meta_path = obj_path.parent / BUDGET_META_NAME
    if not meta_path.is_file():
        if require_sidecar:
            problems.append(
                f"budget sidecar {meta_path} missing — the sidecar is the "
                "authoritative identity/budget record; a legacy artifact "
                "must be migrated deliberately, never trusted bare")
    else:
        try:
            meta = json.loads(meta_path.read_text())
        except ValueError as e:
            problems.append(f"unreadable budget sidecar {meta_path}: {e}")
            meta = None
    if meta is not None:
        try:
            kind = resolve_budget_kind(meta)
        except ValueError as e:
            problems.append(str(e))
            kind = None
        recorded = meta.get("cell_id")
        if recorded is None:
            problems.append(
                f"sidecar {meta_path} records no cell_id — a pre-stamp "
                "legacy sidecar; migrate it deliberately")
        elif recorded not in accepted_ids:
            problems.append(
                f"sidecar {meta_path} records cell_id={recorded!r} but "
                f"this cell is {cell_id!r} — a wrong-cell artifact "
                "(store corruption or a mis-staged copy)")
        if kind == DOCUMENT_TOKENS_KIND:
            problems.extend(document_budget_problems(
                meta,
                require_sample_manifest_sha=require_sample_manifest_sha,
                expected_sample_manifest_sha=expected_sample_manifest_sha))
        elif kind is None and "training_budget_kind" not in (meta or {}):
            # Legacy SI-pair schema completeness (identity fields only —
            # the completion-cap-vs-config comparison is the serving
            # gate's own budget check, not an identity fact).
            if meta.get("completion_tokens") is None:
                problems.append(
                    f"sidecar {meta_path} (legacy SI-pair schema) records "
                    "no completion_tokens — an incomplete budget record")
        if meta.get("object_sha256") is None:
            problems.append(
                f"sidecar {meta_path} records no object_sha256 — a "
                "pre-stamp legacy sidecar; migrate it deliberately (the "
                f"actual artifact hashes to {sha256_file(obj_path)})")
        elif meta["object_sha256"] != sha256_file(obj_path):
            problems.append(
                f"object_sha256 mismatch: sidecar "
                f"{str(meta['object_sha256'])[:12]}… != on-disk "
                f"{sha256_file(obj_path)[:12]}… — a stale object/sidecar "
                "pair (crashed save or out-of-band overwrite)")

    problems.extend(_complete_marker_problems(obj_path, cell_id, accepted_ids,
                                              allow_plain_text=False))
    return problems


def _complete_marker_problems(obj_path: Path, cell_id: str,
                              accepted_ids: set, *,
                              allow_plain_text: bool) -> list[str]:
    """The COMPLETE-marker coherence check shared by the fitted-cell
    verdicts: when the marker is PRESENT it must name this cell and every
    file its manifest promises must exist with the recorded sha — a torn
    or tampered publication must not read as a valid cell. An ABSENT marker
    is never a problem (staged handovers may ship object + sidecar only).

    ``allow_plain_text`` is the one lineage difference: the training path's
    publisher writes a JSON manifest, while the HPO-promotion lineage writes
    a one-line NOTE (``promoted from <trial dir> on <date>``). A note makes
    no verifiable claim — there is no manifest to check against — so under
    the promotion lineage's identity (``fit_provenance.json``) a non-JSON
    marker is accepted as what it is; under the training path's identity it
    stays a refusal, exactly as before."""
    from .defense_cell import sha256_file

    problems: list[str] = []
    marker_path = obj_path.parent / "COMPLETE"
    if not marker_path.is_file():
        return problems
    try:
        marker = json.loads(marker_path.read_text())
    except ValueError as e:
        if allow_plain_text:
            return problems
        problems.append(f"COMPLETE marker {marker_path} does not "
                        f"parse: {e}")
        marker = None
    if isinstance(marker, dict):
        m_cell = marker.get("cell_id")
        if m_cell not in accepted_ids:
            problems.append(
                f"COMPLETE marker names cell_id={m_cell!r}, not "
                f"{cell_id!r} — the publication record belongs to a "
                "different cell")
        files = marker.get("files")
        if not isinstance(files, dict) or not files:
            problems.append(
                "COMPLETE marker carries no publication manifest "
                "(files) — a pre-manifest marker; the publication "
                "cannot be verified")
        else:
            for name, sha in files.items():
                p = obj_path.parent / str(name)
                if not p.is_file() or sha256_file(p) != sha:
                    problems.append(
                        f"COMPLETE-manifested file {name} is missing "
                        "or does not match its published sha256 — "
                        "the publication was torn or modified")
    elif marker is not None:
        problems.append("COMPLETE marker is not a JSON object")
    return problems


def assert_staged_cell(object_path, cell_id: str, *, consumer: str,
                       require_sidecar: bool = True,
                       require_sample_manifest_sha: bool = False,
                       expected_sample_manifest_sha: str | None = None,
                       ) -> None:
    """:func:`staged_cell_problems`, raising ``SystemExit`` naming the
    consumer — the script-side wrapper (defense_cell folds the problem
    list into its own registry-validation aggregation instead)."""
    problems = staged_cell_problems(
        object_path, cell_id, require_sidecar=require_sidecar,
        require_sample_manifest_sha=require_sample_manifest_sha,
        expected_sample_manifest_sha=expected_sample_manifest_sha)
    if problems:
        raise SystemExit(
            f"[{consumer}] cell {cell_id!r}: staged artifact at "
            f"{object_path} failed identity validation:\n  - "
            + "\n  - ".join(problems)
            + "\nRefusing to load/serve an artifact that cannot be proven "
              "to be this cell's work (retrain or re-stage it "
              "deliberately).")


# --------------------------------------------------------------------------- #
# Derived-object identity (sdf-base-steering round 2)                         #
# --------------------------------------------------------------------------- #
# A DERIVED object is never trained: it is composed from fitted parents
# (today: scripts/sdfbase/lora_subtract.py's ``lora_subtraction`` = ΔW(mix) −
# ΔW(ctl); the round-2 PSR residual kinds ride the same contract). It carries
# no ``train_budget_meta.json`` — a budget sidecar would let a bank walker
# mistake it for a fit — but it must be JUST AS provable: its sidecar
# (``derived_meta.json``) binds the object bytes (object_sha256) to the
# PARENT identities (cell ids + parent object shas + parent budget-sidecar
# shas + model id/revision), and the same parent identities ride INSIDE the
# artifact (blob["provenance"]) so the bytes and the sidecar can be checked
# against each other. A consumer that loaded a derived artifact through the
# staged-cell gate's "no sidecar → bare ad-hoc object" leniency would score a
# blob whose parents nobody verified — the gap this section closes. ONE
# verdict, run by every derived-object consumer before any load.

BUDGET_META_NAME = "train_budget_meta.json"
DERIVED_META_NAME = "derived_meta.json"

#: Per-parent fields a derived sidecar must record NON-NULL (the identity of
#: the object this derivation actually consumed — a refit parent under the
#: same cell id is a DIFFERENT object, which is exactly why the sha is
#: required). ``model_revision`` is required here even though a parent's
#: budget sidecar may record it null (the qwen35_4b_base bank does): the
#: BUILDER resolves it from the bank's ``_store_identity.json`` — the
#: store-level identity every writer had to match — and records the source
#: (lora_subtract.load_parent), so a derived object never ships without the
#: revision its parents were fit against.
DERIVED_PARENT_FIELDS = ("cell_id", "model_key", "object_sha256",
                         "budget_meta_sha256", "model_id", "model_revision")

#: Parent fields that must AGREE across the parents of one derived object:
#: a task-vector difference across models, revisions, layer sets, fit
#: legs or chat-template treatments is not a defined object.
DERIVED_PARENT_MATCH_FIELDS = ("model_id", "model_revision", "target_layers",
                               "positions", "response_header_len",
                               "template_hash")


def _parent_blocks(meta: dict) -> dict[str, dict]:
    """The ``parent_<role>`` mappings of a derived sidecar/provenance."""
    return {k: v for k, v in (meta or {}).items()
            if k.startswith("parent_") and isinstance(v, dict)}


def derived_object_problems(object_path, *, derived_id: str | None = None,
                            parents_root=None) -> list[str]:
    """Problem strings (empty = valid) for ONE derived-object artifact,
    CPU-cheap (the blob read is a weights-only ``torch.load`` of a
    megabyte-scale artifact, no model). Checks:

    - ``derived_meta.json`` sits beside the object and parses, and NO
      ``train_budget_meta.json`` does (a dir claiming both identities is
      ambiguous — refused rather than guessed)
    - the sidecar's ``object_sha256`` matches the on-disk bytes; the sidecar
      RECORDS a non-empty ``derived_id`` (an unnamed derived object is not
      bound to any cell), and when the caller names the expected id the two
      must be EXACTLY equal — an absent record never satisfies an
      expectation: a coherent artifact staged or requested under the wrong
      cell must refuse
    - the sidecar records ≥ 2 ``parent_*`` blocks, each with every
      :data:`DERIVED_PARENT_FIELDS` entry non-null, and the parents agree on
      :data:`DERIVED_PARENT_MATCH_FIELDS` (fields present on any parent
      must be present and equal on all)
    - the artifact's embedded ``provenance`` names the SAME parents
      (cell_id + object_sha256 per block) and the same ``kind`` as the
      sidecar — the identity is bound from inside the bytes as well as
      through the sha
    - with ``parents_root``: each parent cell at
      ``<parents_root>/<model_key>/cells_all/<cell_id>/`` passes the
      staged-cell gate (resume-grade strictness) and its object hashes to
      the recorded ``object_sha256`` (the bank holds the object THIS
      derivation consumed, not a later refit under the same id). The check
      is SHA-BOUND and reads nothing else in the parent dir — a
      ``superseded_by.json`` beside a parent (a continuation's snapshot
      record) neither passes nor fails it; ``eval.derived_preflight``
      records and cross-checks that file separately.
    """
    from .defense_cell import sha256_file

    obj_path = Path(object_path)
    problems: list[str] = []
    if not obj_path.is_file():
        return [f"no derived artifact at {obj_path}"]
    meta_path = obj_path.parent / DERIVED_META_NAME
    if (obj_path.parent / BUDGET_META_NAME).is_file():
        problems.append(
            f"{obj_path.parent} carries BOTH {BUDGET_META_NAME} and "
            f"{DERIVED_META_NAME} — a dir cannot be a fitted cell and a "
            "derived object at once; refusing to pick an identity")
    if not meta_path.is_file():
        problems.append(
            f"derived sidecar {meta_path} missing — a derived object is "
            "only provable through its sidecar (parent identities + object "
            "sha); an unsidecared artifact is not trusted")
        return problems
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except ValueError as e:
        return problems + [f"unreadable derived sidecar {meta_path}: {e}"]
    if not isinstance(meta, dict):
        return problems + [f"derived sidecar {meta_path} is not a JSON object"]

    kind = meta.get("kind")
    if not kind:
        problems.append(f"derived sidecar {meta_path} records no kind")
    if meta.get("object_sha256") is None:
        problems.append(
            f"derived sidecar {meta_path} records no object_sha256 (the "
            f"artifact hashes to {sha256_file(obj_path)})")
    elif meta["object_sha256"] != sha256_file(obj_path):
        problems.append(
            f"object_sha256 mismatch: sidecar "
            f"{str(meta['object_sha256'])[:12]}… != on-disk "
            f"{sha256_file(obj_path)[:12]}… — a stale object/sidecar pair")
    recorded_id = meta.get("derived_id")
    if not isinstance(recorded_id, str) or not recorded_id:
        problems.append(
            f"derived sidecar {meta_path} records no derived_id — a derived "
            "object is bound to the cell it was built AS; an unnamed sidecar "
            "cannot prove which cell these bytes belong to"
            + (f" (this object is requested as {derived_id!r})"
               if derived_id is not None else ""))
    elif derived_id is not None and recorded_id != derived_id:
        problems.append(
            f"derived sidecar records derived_id={recorded_id!r} but this "
            f"object is {derived_id!r} — a wrong-identity artifact")

    parents = _parent_blocks(meta)
    if len(parents) < 2:
        problems.append(
            f"derived sidecar records {len(parents)} parent block(s) "
            "(parent_<role> mappings) — a derived object has ≥ 2 parents")
    for role, block in parents.items():
        missing = [f for f in DERIVED_PARENT_FIELDS if block.get(f) is None]
        if missing:
            problems.append(
                f"{role} records no {missing} — the parent identity is "
                "incomplete (a null model identity was the R2 review's "
                "exact finding)")
    for field_name in DERIVED_PARENT_MATCH_FIELDS:
        present = {role: block.get(field_name)
                   for role, block in parents.items() if field_name in block}
        if not present:
            continue
        if len(present) != len(parents) or len(
                {json.dumps(v, sort_keys=True, default=str)
                 for v in present.values()}) != 1:
            problems.append(
                f"parents disagree on {field_name}: "
                + ", ".join(f"{r}={block.get(field_name)!r}"
                            for r, block in parents.items())
                + " — a derivation across that difference is not a "
                  "defined object")

    # The bytes say who their parents are, too.
    blob = None
    try:
        import torch

        blob = torch.load(obj_path, map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001 — every failure is a problem here
        problems.append(f"{obj_path} does not load as a weights-only torch "
                        f"blob: {type(e).__name__}: {e}")
    if isinstance(blob, dict):
        if kind and blob.get("kind") != kind:
            problems.append(
                f"artifact kind {blob.get('kind')!r} != sidecar kind "
                f"{kind!r}")
        embedded = _parent_blocks(blob.get("provenance") or {})
        if set(embedded) != set(parents):
            problems.append(
                f"artifact provenance names parents {sorted(embedded)} but "
                f"the sidecar names {sorted(parents)}")
        for role in set(embedded) & set(parents):
            for f in ("cell_id", "object_sha256"):
                if embedded[role].get(f) != parents[role].get(f):
                    problems.append(
                        f"{role}.{f} differs between the artifact's embedded "
                        f"provenance ({embedded[role].get(f)!r}) and the "
                        f"sidecar ({parents[role].get(f)!r}) — the bytes "
                        "and the sidecar describe different derivations")
    elif blob is not None:
        problems.append(f"{obj_path} is not a dict blob")

    if parents_root is not None:
        for role, block in parents.items():
            cell_id = block.get("cell_id")
            key = block.get("model_key")
            if not cell_id or not key:
                continue  # already reported above
            cell_dir = parent_cell_dir(parents_root, block)
            p_obj = cell_dir / "object.pt"
            for p in staged_cell_problems(p_obj, str(cell_id)):
                problems.append(f"{role} ({cell_dir}): {p}")
            if (p_obj.is_file() and block.get("object_sha256") is not None
                    and sha256_file(p_obj) != block["object_sha256"]):
                problems.append(
                    f"{role}: the bank's {p_obj} hashes to "
                    f"{sha256_file(p_obj)[:12]}…, not the recorded "
                    f"{str(block['object_sha256'])[:12]}… — the parent this "
                    "derivation consumed is not the object staged under "
                    "that id now (a refit? then the derived object must be "
                    "rebuilt, or its parents read from the snapshot bank)")
            p_meta = cell_dir / BUDGET_META_NAME
            if (p_meta.is_file() and block.get("budget_meta_sha256") is not None
                    and sha256_file(p_meta) != block["budget_meta_sha256"]):
                problems.append(
                    f"{role}: {p_meta} does not hash to the recorded "
                    "budget_meta_sha256 — the parent's identity record "
                    "changed under the derivation")
    return problems


def parent_cell_dir(parents_root, parent_block: dict) -> Path:
    """The bank dir of ONE recorded parent under ``parents_root``:
    ``<parents_root>/<model_key>/cells_all/<cell_id>/`` — the layout the
    sdf-base fitting lane publishes to (the volume's ``out/`` tree), and
    the ONE spelling both :func:`derived_object_problems` and the derived
    preflight resolve a parent through."""
    return (Path(parents_root) / str(parent_block.get("model_key"))
            / "cells_all" / str(parent_block.get("cell_id")))


def assert_derived_object(object_path, *, consumer: str,
                          derived_id: str | None = None,
                          parents_root=None) -> None:
    """:func:`derived_object_problems`, raising ``SystemExit`` naming the
    consumer."""
    problems = derived_object_problems(object_path, derived_id=derived_id,
                                       parents_root=parents_root)
    if problems:
        raise SystemExit(
            f"[{consumer}] derived object at {object_path}"
            + (f" ({derived_id!r})" if derived_id else "")
            + " failed identity validation:\n  - "
            + "\n  - ".join(problems)
            + "\nRefusing to load/serve a derived artifact whose parents "
              "cannot be proven (rebuild it with "
              "scripts/sdfbase/lora_subtract.py).")


# --------------------------------------------------------------------------- #
# The HPO-promotion lineage's fit record (``fit_provenance.json``)             #
# --------------------------------------------------------------------------- #
# Two fitting lineages publish the SAME fit-time facts under different
# spellings. The training lane writes ``train_budget_meta.json``; the
# HPO-promotion lineage (tune_arm.py trials promoted into a bank — the
# scale-12 safety-SI cells on qwen35_4b) writes ``fit_provenance.json``, the
# stage-3 mirror gate's NATIVE record (``sweep.STAGE3_FIT_PROVENANCE_FILE``:
# resolved model id + revision + the MEASURED chat-template hash + the
# fit-time writer). Neither carries the other's file, and nothing upstream
# exists to re-stage from — the staged copies faithfully reflect their
# sources. So a fitted cell is provable under EITHER spelling, and the ONE
# normaliser both are read through is ``sweep.read_stage3_fit_provenance``
# (never a translated second file on disk, which could drift from the real
# one). The registry, the bank view, the shell presence check and the
# drivers all dispatch on which file the dir declares; a dir declaring more
# than one identity is ambiguous and refuses.
#
# WHAT THIS RECORD DOES NOT CARRY, stated so nobody reads its acceptance as
# the budget gate: no ``cell_id`` and no ``object_sha256``. The object's
# bytes are bound by the bank view's PIN (the staging manifest's sha,
# verified on both clusters) and the cell by the roster's dir mapping; the
# promotion lineage's companion ``provenance.json`` (si + arm) is
# cross-checked WHEN PRESENT, never required — declared-when-present, the
# ``registry_adapter._check_meta_json`` rule. The blob's own recorded fit
# leg is cross-checked against the record's ``positions_at_fit`` when both
# exist.
FIT_PROVENANCE_NAME = "fit_provenance.json"
#: the promotion lineage's companion record (si, arm, positions,
#: response_header_len, fit_fingerprint, …) — optional, checked when present
PROMOTION_PROVENANCE_NAME = "provenance.json"
#: every identity sidecar a staged dir may declare; exactly ONE is legal
IDENTITY_SIDECARS = (BUDGET_META_NAME, DERIVED_META_NAME, FIT_PROVENANCE_NAME)
_IDENTITY_OF_SIDECAR = {BUDGET_META_NAME: "staged",
                        DERIVED_META_NAME: "derived",
                        FIT_PROVENANCE_NAME: "fit_provenance"}


def fit_provenance_problems(object_path, cell_id: str) -> list[str]:
    """Problem strings (empty = valid) for ONE fitted cell whose identity
    record is the HPO-promotion lineage's ``fit_provenance.json``,
    CPU-cheap (a weights-only torch load of a megabyte-scale artifact).
    Checks:

    - the record normalises through the ONE shared reader
      (``sweep.read_stage3_fit_provenance``) to a VERIFIED fit-time record
      whose source IS the native spelling — every
      ``STAGE3_FIT_REQUIRED_KEYS`` field non-null (a record missing any,
      or carrying it null, refuses exactly as the mirror gate refuses it);
    - NO other identity sidecar sits beside it;
    - ``provenance.json``, WHEN PRESENT and naming ``si`` + ``arm``, names
      THIS cell (its declared legacy spelling accepted);
    - the artifact's own recorded fit leg (``positions``) agrees with the
      record's ``positions_at_fit`` when both exist — the served region is
      read from the blob, so a disagreement means the identity record and
      the bytes describe different fits;
    - a PRESENT ``COMPLETE`` marker: a JSON manifest is verified like the
      training lane's; a plain-text promotion note is accepted as a note
      (it makes no verifiable claim).
    """
    from .defense_cell import legacy_cell_id
    from .sweep import (STAGE3_FIT_REQUIRED_KEYS, STAGE3_FIT_PROVENANCE_FILE,
                        read_stage3_fit_provenance)

    obj_path = Path(object_path)
    problems: list[str] = []
    if not obj_path.is_file():
        return [f"no staged artifact at {obj_path}"]
    d = obj_path.parent
    others = [s for s in IDENTITY_SIDECARS
              if s != FIT_PROVENANCE_NAME and (d / s).is_file()]
    if others:
        problems.append(
            f"{d} carries {FIT_PROVENANCE_NAME} AND {others} — a dir "
            "declares exactly one identity; refusing to pick one")
    rec_path = d / FIT_PROVENANCE_NAME
    if not rec_path.is_file():
        return problems + [f"fit record {rec_path} missing"]
    try:
        raw = json.loads(rec_path.read_text(encoding="utf-8"))
    except ValueError as e:
        return problems + [f"unreadable fit record {rec_path}: {e}"]
    if not isinstance(raw, dict):
        return problems + [f"fit record {rec_path} is not a JSON object"]
    missing = [k for k in STAGE3_FIT_REQUIRED_KEYS if not raw.get(k)]
    if missing:
        problems.append(
            f"fit record {rec_path} lacks (or carries null) "
            f"{missing} — the stage-3 gate's required fit-time facts "
            f"({list(STAGE3_FIT_REQUIRED_KEYS)}); a record that never "
            "loaded the tokenizer cannot be verified")
    rec = read_stage3_fit_provenance(d)
    if rec is None or not str(rec.get("source", "")).startswith(
            f"{STAGE3_FIT_PROVENANCE_FILE.split('.')[0]}:"):
        if not missing:  # the reader disagrees with the raw check: say so
            problems.append(
                f"fit record {rec_path} did not normalise as the native "
                f"spelling (reader returned {rec!r})")
        rec = None
    legacy = legacy_cell_id(cell_id)
    accepted_ids = {cell_id} | ({legacy} if legacy else set())

    prov_path = d / PROMOTION_PROVENANCE_NAME
    if prov_path.is_file():
        try:
            prov = json.loads(prov_path.read_text(encoding="utf-8"))
        except ValueError as e:
            problems.append(f"unreadable {prov_path}: {e}")
            prov = None
        if isinstance(prov, dict) and prov.get("si") and prov.get("arm"):
            named = f"{prov['si']}__{prov['arm']}"
            if named not in accepted_ids:
                problems.append(
                    f"{prov_path} names cell {named!r}, not {cell_id!r} "
                    "(or its legacy spelling) — a wrong-cell artifact "
                    "(a mis-staged copy)")

    if rec is not None and rec.get("positions_at_fit") is not None:
        blob = None
        try:
            import torch

            blob = torch.load(obj_path, map_location="cpu",
                              weights_only=True)
        except Exception as e:  # noqa: BLE001 — every failure is a problem
            problems.append(f"{obj_path} does not load as a weights-only "
                            f"torch blob: {type(e).__name__}: {e}")
        if isinstance(blob, dict) and "positions" in blob \
                and blob["positions"] != rec["positions_at_fit"]:
            problems.append(
                f"the artifact records positions={blob['positions']!r} but "
                f"{rec_path} records positions_at_fit="
                f"{rec['positions_at_fit']!r} — the identity record and "
                "the bytes describe different fits")

    problems.extend(_complete_marker_problems(obj_path, cell_id, accepted_ids,
                                              allow_plain_text=True))
    return problems


def assert_fit_provenance_cell(object_path, cell_id: str, *,
                               consumer: str) -> None:
    """:func:`fit_provenance_problems`, raising ``SystemExit`` naming the
    consumer."""
    problems = fit_provenance_problems(object_path, cell_id)
    if problems:
        raise SystemExit(
            f"[{consumer}] cell {cell_id!r}: staged artifact at "
            f"{object_path} failed fit-record identity validation:\n  - "
            + "\n  - ".join(problems)
            + "\nRefusing to load/serve an artifact whose fit-time record "
              "cannot be verified.")


def object_identity_kind(object_path) -> str | None:
    """Which identity contract the artifact's dir declares: ``"staged"``
    (a ``train_budget_meta.json`` fit sidecar), ``"derived"``
    (``derived_meta.json``), ``"fit_provenance"`` (the HPO-promotion
    lineage's ``fit_provenance.json`` — the SAME fit facts as the budget
    sidecar under the stage-3 gate's native spelling), ``None`` (none —
    bare). More than one at once is reported as ``"ambiguous"``: the two
    fit spellings are one contract, but no in-repo writer emits both, so a
    dir carrying two records is a dir whose records could disagree, and
    which one is authoritative is not a guess this function makes."""
    d = Path(object_path).parent
    present = [s for s in IDENTITY_SIDECARS if (d / s).is_file()]
    if len(present) > 1:
        return "ambiguous"
    if present:
        return _IDENTITY_OF_SIDECAR[present[0]]
    return None


def assert_object_identity(object_path, cell_id: str, *, consumer: str,
                           parents_root=None, **staged_kw) -> str:
    """The ONE consumer entry point for "may I load this artifact as
    ``cell_id``": dispatches on the sidecar the dir carries —
    :func:`assert_staged_cell` for a training-lane fit,
    :func:`assert_fit_provenance_cell` for an HPO-promotion fit,
    :func:`assert_derived_object` (``derived_id=cell_id``) for a derived
    object — and refuses a dir claiming more than one. A dir with NO
    sidecar goes down the staged path, whose own "sidecar missing" verdict
    names the fix; a consumer that wants to tolerate bare ad-hoc artifacts
    decides that BEFORE calling here (the probe scorer's scratch-object
    leniency). Returns the kind validated (``"staged"`` |
    ``"fit_provenance"`` | ``"derived"``). ``staged_kw`` are the staged
    gate's strictness knobs (``require_sidecar`` & co.), read only on the
    staged path."""
    kind = object_identity_kind(object_path)
    if kind == "ambiguous":
        d = Path(object_path).parent
        raise SystemExit(
            f"[{consumer}] {d} carries more than one identity sidecar "
            f"({[s for s in IDENTITY_SIDECARS if (d / s).is_file()]}) — a "
            "dir declares exactly one; refusing to guess which to validate.")
    if kind == "derived":
        assert_derived_object(object_path, consumer=consumer,
                              derived_id=cell_id, parents_root=parents_root)
        return "derived"
    if kind == "fit_provenance":
        assert_fit_provenance_cell(object_path, cell_id, consumer=consumer)
        return "fit_provenance"
    assert_staged_cell(object_path, cell_id, consumer=consumer, **staged_kw)
    return "staged"


# --------------------------------------------------------------------------- #
# Opt. 1 — per-SI max_len (a memory knob; the completion cap stays shared)     #
# --------------------------------------------------------------------------- #


def derive_max_len(
    rendered_prompt_tokens: int,
    completion_cap: int,
    *,
    attack_allowance: int = 0,
    eos: int = 1,
    round_up_pow2: bool = True,
) -> int:
    """The minimum ``max_len`` that holds one SI uncut: the measured rendered
    teacher prompt + an attack-text allowance + the SHARED completion cap +
    EOS. This is the exact arithmetic ``configs/two_si.yaml`` documents for the
    scalar ``max_len``, applied PER SI so the safety arm is not sized for the
    constitution. Rounds up to the next power of two by default (a little
    headroom), which is what the config comment does.

    Passing the SAME ``completion_cap`` for every SI is what keeps supervision
    volume equal — this function never changes it, only how much room is
    left AROUND it.
    """
    required = rendered_prompt_tokens + attack_allowance + completion_cap + eos
    if not round_up_pow2:
        return required
    p = 1
    while p < required:
        p <<= 1
    return p


def resolve_max_len(cfg: dict, si_id: str) -> int:
    """The ``max_len`` for one SI: ``max_len_by_si[si_id]`` if present, else the
    scalar ``max_len`` fallback (additive — an absent ``max_len_by_si`` leaves
    behaviour identical to the single shared budget). Fails loudly if neither
    is set."""
    by_si = cfg.get("max_len_by_si") or {}
    if si_id in by_si:
        return int(by_si[si_id])
    if "max_len" in cfg:
        return int(cfg["max_len"])
    raise KeyError(
        f"no max_len for SI {si_id!r}: set max_len_by_si[{si_id!r}] or the "
        "scalar max_len in the config")


# --------------------------------------------------------------------------- #
# Opt. 2 — per-(SI, arm) batch size (a throughput knob)                        #
# --------------------------------------------------------------------------- #


def resolve_batch_size(cfg: dict, si_id: str, arm: str) -> int:
    """The batch size for one (SI, arm) cell, most-specific-wins:

    1. ``batch_size_by_arm[arm]``  (per-arm override — e.g. a rung's own limit)
    2. ``batch_size_by_si[si_id]`` (per-SI default — e.g. narrower for the
       constitution's long context)
    3. ``train_kwargs.default.batch_size`` (the inherited global)
    4. :data:`DEFAULT_BATCH_SIZE`

    then HARD-capped by any arm-intrinsic memory constraint (the fisher rung's
    O(T^2) attention forces batch <= :data:`FISHER_MAX_BATCH`, regardless of
    what the config asks for — a wider request is clamped, never honoured into
    an OOM).
    """
    by_arm = cfg.get("batch_size_by_arm") or {}
    by_si = cfg.get("batch_size_by_si") or {}
    tk = (cfg.get("train_kwargs") or {}).get("default") or {}
    if arm in by_arm:
        size = int(by_arm[arm])
    elif si_id in by_si:
        size = int(by_si[si_id])
    elif "batch_size" in tk:
        size = int(tk["batch_size"])
    else:
        size = DEFAULT_BATCH_SIZE
    if arm in FISHER_ARMS:
        size = min(size, FISHER_MAX_BATCH)
    return max(1, size)


def attention_cost_class(arm: str) -> str:
    """``"O(T^2)"`` for arms whose attention memory scales with the square of
    the sequence length (the fisher rung's math-backend forward-AD), else
    ``"O(T)"`` (flash/sdpa distillation)."""
    return "O(T^2)" if arm in FISHER_ARMS else "O(T)"


# --------------------------------------------------------------------------- #
# Opt. 5 — the CPU-only measurement harness                                    #
# --------------------------------------------------------------------------- #


@dataclass
class CellBudget:
    """The CPU-measurable shape/token facts for one (SI, arm, batch_size)
    cell, plus the GPU-only fields left as ``NEEDS_GPU`` until a real pass."""

    si_id: str
    arm: str
    batch_size: int
    max_len: int
    teacher_prompt_tokens: int   # max over examples of the rendered teacher len
    student_prompt_tokens: int   # max over examples of the (injected) student len
    completion_tokens: int       # the shared cap actually entering the loss
    total_seq_tokens: int        # longest prompt + completion (the padded T)
    n_examples: int
    n_batches: int
    supervision_tokens_per_epoch: int  # completion tokens summed over examples
    attention_cost: str          # "O(T)" | "O(T^2)"
    per_step_work_proxy: float   # batch * (T^2 if O(T^2) else T) — relative
    teacher_forward: str         # "shared" (lockstep) | "solo"
    n_trainable_params: int | None
    # GPU-only, filled by a later measured pass:
    peak_memory_gb: str = _NEEDS_GPU
    tokens_per_s: str = _NEEDS_GPU
    est_seconds: str = _NEEDS_GPU

    def as_dict(self) -> dict:
        return asdict(self)


def _prompt_len(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=True)["input_ids"])


def _completion_len(tok, text: str) -> int:
    n = len(tok(text, add_special_tokens=False)["input_ids"])
    return n + (1 if tok.eos_token_id is not None else 0)


def measure_si_budget(
    cfg: dict,
    tok,
    examples,
    *,
    si_id: str,
    arms: list[str],
    completion_cap: int | None = None,
    candidate_batch_sizes: list[int] | None = None,
    object_provider: Callable[[str], object] | None = None,
    compatible: Callable[[str], bool] | None = None,
    tokens_per_s: float | None = None,
    epochs: int | None = None,
    measured_path=None,
) -> list[CellBudget]:
    """Per-(arm[, candidate batch size]) budget rows for one SI, CPU only.

    ``examples`` are the SI's :class:`~sisteer.extraction.distill.DistillExample`
    list (teacher = ``prompt_with_si``, student = ``prompt_without``); the
    harness measures rendered token lengths from them. ``object_provider(arm)``
    (optional) returns the arm's intervention object so trainable params can be
    counted and the student-side inject reflected (a soft-token arm adds its
    sentinel); when absent, param counts are ``None`` and the student prompt is
    measured un-injected. ``compatible(arm)`` marks lockstep teacher-sharing so
    the table can say whether the teacher forward is shared. ``tokens_per_s``,
    if supplied from a GPU pass, turns the ``NEEDS_GPU`` time field into a real
    ``--time`` estimate.

    Measured-table consumption (capeval-controls §5.2, additive): after the
    rows are built, the measured GPU-pass table
    (``scripts/measure_train_budget.py``) is consumed when available —
    ``measured_path`` argument, else the config's ``measured_path`` key, else
    :data:`DEFAULT_MEASURED_PATH` — gated on :func:`measurement_inputs_sha`
    compatibility ONLY (r2 #1: never the per-row run fingerprint, which is the
    GPU pass's own resume key). See :func:`apply_measured` for the fill
    semantics (r3 #2: the exact resolved-bs trial row, ``<= safe_batch_size``
    enforced loudly). With no measured table in play, behaviour is byte-for-
    byte unchanged (``NEEDS_GPU`` marks stay).
    """
    if resolve_budget_kind(cfg) == DOCUMENT_TOKENS_KIND:
        # SI-pair machinery: the whole table is shaped around a completion
        # cap and a teacher prompt, neither of which a document fit has —
        # int(None) on the explicit completion_tokens: null would crash
        # three lines down anyway; refuse with the reason instead.
        raise ValueError(
            "measure_si_budget is SI-pair machinery (completion cap + "
            "teacher prompt lengths); a training_budget_kind: "
            "document_tokens config has neither — its budget is the sample "
            "manifest's document tokens (see document_budget_problems).")
    cap = completion_cap if completion_cap is not None else int(
        cfg.get("completion_tokens", 512))
    ep = epochs if epochs is not None else int(
        (cfg.get("train_kwargs") or {}).get("default", {}).get("epochs", 6))
    max_len = resolve_max_len(cfg, si_id)

    teacher_lens = [_prompt_len(tok, ex.prompt_with_si) for ex in examples]
    # Completion tokens actually entering the loss: the shared cap (each
    # example's completion, capped — but truncation to fit max_len is what we
    # forbid, so the cap is the supervision volume). Measure the real per
    # example completion length, capped, for the supervision total.
    comp_lens = [min(_completion_len(tok, ex.completion), cap) for ex in examples]
    max_teacher = max(teacher_lens) if teacher_lens else 0

    rows: list[CellBudget] = []
    for arm in arms:
        obj = object_provider(arm) if object_provider is not None else None
        # Student prompt length: reflect the object's inject (soft-token arms
        # add one sentinel token) when we have the object; else un-injected.
        if obj is not None and hasattr(obj, "inject"):
            student_lens = [_prompt_len(tok, obj.inject(ex.prompt_without))
                            for ex in examples]
        else:
            student_lens = [_prompt_len(tok, ex.prompt_without) for ex in examples]
        max_student = max(student_lens) if student_lens else 0
        n_params = None
        if obj is not None and hasattr(obj, "n_params"):
            try:
                n_params = int(obj.n_params())
            except Exception:  # noqa: BLE001 — a param count is best-effort
                n_params = None

        total_seq = max(max_teacher, max_student) + cap
        supervision = sum(comp_lens)
        cost = attention_cost_class(arm)
        is_o2 = cost == "O(T^2)"
        shared = (compatible(arm) if compatible is not None else False)

        bss = (candidate_batch_sizes if candidate_batch_sizes is not None
               else [resolve_batch_size(cfg, si_id, arm)])
        for bs in bss:
            work = bs * (total_seq ** 2 if is_o2 else total_seq)
            n_batches = (len(examples) + bs - 1) // bs if examples else 0
            row = CellBudget(
                si_id=si_id, arm=arm, batch_size=bs, max_len=max_len,
                teacher_prompt_tokens=max_teacher,
                student_prompt_tokens=max_student,
                completion_tokens=cap, total_seq_tokens=total_seq,
                n_examples=len(examples), n_batches=n_batches,
                supervision_tokens_per_epoch=supervision,
                attention_cost=cost, per_step_work_proxy=float(work),
                teacher_forward=("shared" if shared else "solo"),
                n_trainable_params=n_params,
            )
            if tokens_per_s:
                # forward tokens over training: teacher + student forward per
                # example per epoch (~2 x seq); shared teacher amortises the
                # teacher half across the group, but per-cell we cost the full
                # pass (an upper bound for a single cell's own --time).
                fwd_tokens = ep * len(examples) * total_seq * 2
                row.est_seconds = f"{fwd_tokens / tokens_per_s:.3g}"
            rows.append(row)

    # capeval-controls §5.2: consume the measured GPU-pass table when one is
    # available and measurement_inputs_sha-compatible (see the docstring).
    measured = _maybe_load_measured(cfg, measured_path)
    if measured is not None:
        apply_measured(rows, measured, epochs=ep)
    return rows


_TABLE_COLS = [
    ("si_id", "SI", 16),
    ("arm", "arm", 15),
    ("batch_size", "bs", 4),
    ("max_len", "max_len", 8),
    ("teacher_prompt_tokens", "teach_tok", 10),
    ("student_prompt_tokens", "stud_tok", 9),
    ("completion_tokens", "comp_tok", 9),
    ("total_seq_tokens", "T", 7),
    ("n_batches", "nbat", 5),
    ("attention_cost", "attn", 7),
    ("per_step_work_proxy", "work", 14),
    ("teacher_forward", "teacher", 8),
    ("n_trainable_params", "params", 12),
    ("peak_memory_gb", "peak_gb", 9),
    ("tokens_per_s", "tok/s", 9),
    ("est_seconds", "est_s", 10),
]


def format_budget_table(rows: list[CellBudget]) -> str:
    """A small fixed-width table of the budget rows — the harness's emitted
    view. GPU-only columns read ``NEEDS_GPU`` until a measured pass fills them.
    """
    header = "  ".join(f"{label:>{w}}" for _, label, w in _TABLE_COLS)
    lines = [header, "  ".join("-" * w for _, _, w in _TABLE_COLS)]
    for r in rows:
        d = r.as_dict()
        cells = []
        for key, _, w in _TABLE_COLS:
            v = d[key]
            if isinstance(v, float):
                s = f"{v:.3g}"
            elif v is None:
                s = "-"
            else:
                s = str(v)
            cells.append(f"{s:>{w}}")
        lines.append("  ".join(cells))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Measured GPU-pass consumption (ADDITIVE)                                     #
# --------------------------------------------------------------------------- #
# The GPU pass (scripts/measure_train_budget.py) commits its trial
# rows to configs/two_si_budget_measured.json. The functions below let the
# EXISTING CPU harness consume that table with zero changes to any caller:
# measure_si_budget picks it up through the config's `measured_path` key (or
# this default path) whenever the table's `measurement_inputs_sha` matches the
# current config + code.
#
# TWO DISTINCT identity checks, deliberately separate (r2 #1):
#   * the GPU pass's OWN resume uses an exact-match per-row run fingerprint
#     (model/config sha/code sha/GPU/torch/shapes) — that lives in
#     measure_train_budget.py and is NEVER checked here;
#   * CONSUMPTION is gated on `measurement_inputs_sha` ONLY — a hash over the
#     normalized measurement-relevant inputs (below), so the §5.3 config
#     commit (derived batch_size_by_* widths) or an unrelated rebase can never
#     self-invalidate the table, while a material trainer/arm-spec edit
#     rejects it (r3 #1).

#: Where the committed measured table lives (repo-relative).
DEFAULT_MEASURED_PATH = "configs/two_si_budget_measured.json"

#: Version tag of the synthetic worst-case sequence construction.
#: Bump when measure_train_budget.py changes HOW trial sequences are built —
#: it is hashed into measurement_inputs_sha, so a bump invalidates the table.
SEQ_CONSTRUCTION_VERSION = "two_si_budget_seq_v1"

_REPO_ROOT = Path(__file__).resolve().parents[3]

# The measurement HARNESS's own files. Hashed into the CONSUMPTION gate
# (measurement_inputs_sha, r3 #1) but deliberately NOT into the per-row
# resume identity (arm_scoped_inputs_sha): a harness edit that changes
# measurement SEMANTICS must bump SEQ_CONSTRUCTION_VERSION (which IS in the
# row identity); a mechanical harness edit must not invalidate finished
# trial rows (2026-08-19: a one-key per-arm config fix would otherwise have
# forced re-measuring every finished ladder).
_HARNESS_SOURCES: tuple[str, ...] = (
    "src/sisteer/eval/train_budget.py",
    "scripts/measure_train_budget.py",
)

# Measurement-relevant sources hashed by CONTENT into measurement_inputs_sha
# (r3 #1: file contents, never the git SHA — an unrelated rebase must not
# perturb the hash). The renderer (chat template + SI texts + pair->example
# conversion), the encoder/collator, the registry wiring, and the measurement
# logic itself (both halves).
_MEASUREMENT_SOURCES: tuple[str, ...] = (
    # renderer: build_chat_prompt / pairs_to_distill_examples / the SI texts
    "src/sisteer/models.py",
    "src/sisteer/data/pairs.py",
    "src/sisteer/data/instructions.py",
    "src/sisteer/data/safety_texts.py",
    # encoder + collator (the tensors every gradient trainer builds)
    "src/sisteer/extraction/traindata.py",
    "src/sisteer/extraction/distill.py",
    # arm registry wiring (trainer dispatch + arm-intrinsic defaults)
    "src/sisteer/methods.py",
    # the measurement logic, both halves
    *_HARNESS_SOURCES,
)

# Per-arm trainer source modules (r3 #1: the EFFECTIVE per-arm training spec
# includes a content hash of each arm's own trainer code). Covers every
# registered arm so any config-selected arm resolves; an arm absent here is a
# LOUD refusal, never a silently unhashed trainer.
_ARM_TRAINER_SOURCES: dict[str, tuple[str, ...]] = {
    "vector": ("src/sisteer/extraction/distill.py",),
    "vector_rkl": ("src/sisteer/extraction/distill.py",),
    "vector_js": ("src/sisteer/extraction/distill.py",),
    "vector_all": ("src/sisteer/extraction/objects.py",
                   "src/sisteer/extraction/multilayer.py"),
    "lora_r1": ("src/sisteer/extraction/objects.py",),
    "lora_r1_all": ("src/sisteer/extraction/objects.py",
                    "src/sisteer/extraction/multilayer.py"),
    # lora_r4 = the canonical spelling of lora_r1_all (the old name is
    # kept in methods.LEGACY_ARM_ALIASES) — same trainer sources by
    # construction, so both names hash the same measurement inputs.
    "lora_r4": ("src/sisteer/extraction/objects.py",
                "src/sisteer/extraction/multilayer.py"),
    "token_bos": ("src/sisteer/extraction/objects.py",),
    "token_faithful": ("src/sisteer/extraction/objects.py",
                       "src/sisteer/extraction/tokens.py"),
    "psr": ("src/sisteer/extraction/psr.py",),
    "psr_kl": ("src/sisteer/extraction/psr.py",),
    "psr_kl_all": ("src/sisteer/extraction/psr.py",),
    "psr_mse_all": ("src/sisteer/extraction/psr.py",),
    "psr_kl_hard": ("src/sisteer/extraction/psr.py",),
    "psr_kl_all_hard": ("src/sisteer/extraction/psr.py",),
    # psr_xfer_* (2026-08-28): cross-model TRANSFERRED PSR units. Never
    # trained in-repo (methods._make_train_transferred refuses), so these
    # entries exist only so any budget-spec resolution over the registered
    # arm set stays total; the artifact class is extraction/psr.py's.
    "psr_xfer_from_e2b": ("src/sisteer/extraction/psr.py",),
    "psr_xfer_from_e2b_rw1": ("src/sisteer/extraction/psr.py",),
    "psr_xfer_from_e4b": ("src/sisteer/extraction/psr.py",),
    "psr_xfer_from_e4b_rw1": ("src/sisteer/extraction/psr.py",),
    "loreft": ("src/sisteer/extraction/reft.py",),
    "reft_r1": ("src/sisteer/extraction/reft.py",),
    "bidpo": ("src/sisteer/extraction/bidpo.py",),
    "lofit": ("src/sisteer/extraction/lofit.py",),
    # cast / circuit_breaker left methods.ARMS entirely (2026-08-20): no
    # entry here — an arm absent from this table is a LOUD refusal, which is
    # exactly right for an arm that can no longer be trained or enumerated.
    "antidote": ("src/sisteer/extraction/antidote.py",),
    "alphasteer": ("src/sisteer/extraction/alphasteer.py",),
    "alphakl_null": ("src/sisteer/extraction/alphakl.py",),
    "alphakl_soft": ("src/sisteer/extraction/alphakl.py",),
    "alphakl_fisher": ("src/sisteer/extraction/alphakl.py",),
    "diffmean": ("src/sisteer/extraction/diffmean.py",),
    # bidpo's objective (transcribed into bidpo_lora.py — the equivalence
    # test in tests/test_bidpo_lora.py ties the two) on lora_r1's substrate
    # (objects.py): a material edit to either is a material edit to this
    # arm's training, so both hash.
    "bidpo_lora": ("src/sisteer/extraction/bidpo_lora.py",
                   "src/sisteer/extraction/objects.py"),
    # the CAA / diff-in-means Class B floor (2026-08-21): its fit bottoms
    # out in alphasteer.refusal_direction, so both files hash.
    "caa_static_r": ("src/sisteer/extraction/caa_static_r.py",
                     "src/sisteer/extraction/alphasteer.py"),
}


def _canonical_sha(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"),
                   default=str).encode("utf-8")).hexdigest()


def _source_shas(rel_paths, repo_root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in sorted(set(rel_paths)):
        p = Path(repo_root) / rel
        if not p.is_file():
            raise FileNotFoundError(
                f"measurement source missing: {p} — measurement_inputs_sha "
                "refuses to hash an absent source (r3 #1); fix the path table "
                "rather than skipping the file")
        out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _effective_arm_specs(cfg: dict, arms: list[str]) -> dict[str, dict]:
    """The normalized EFFECTIVE per-arm training spec (r3 #1): the config-
    resolved train_kwargs (default block + any per-arm block) minus
    ``batch_size`` — batch WIDTH is the derived knob the consumption gate
    handles separately (``<= safe_batch_size``), so hashing it here would let
    the §5.3 config commit self-invalidate the table (the r2 #1 failure) —
    plus each arm's own trainer sources and the arm-intrinsic hard caps."""
    tk_all = cfg.get("train_kwargs") or {}
    default_tk = dict(tk_all.get("default") or {})
    default_tk.pop("batch_size", None)
    specs: dict[str, dict] = {}
    for arm in arms:
        if arm not in _ARM_TRAINER_SOURCES:
            raise ValueError(
                f"arm {arm!r} has no entry in _ARM_TRAINER_SOURCES — refusing "
                "to compute measurement_inputs_sha with an unhashed trainer "
                "(r3 #1); add the arm's trainer module(s) to the table")
        arm_tk = dict(tk_all.get(arm) or {})
        arm_tk.pop("batch_size", None)
        specs[arm] = {
            "train_kwargs": {**default_tk, **arm_tk},
            "trainer_sources": sorted(set(_ARM_TRAINER_SOURCES[arm])),
            "fisher_max_batch": FISHER_MAX_BATCH if arm in FISHER_ARMS else None,
        }
    return specs


def measurement_inputs_sha(cfg: dict, *, repo_root=None) -> str:
    """The consumption-compatibility hash (r2 #1, widened r3 #1): sha256 over
    ONLY the normalized measurement-relevant inputs —

    - model + revision, dtype, attn implementation, layer;
    - the sequence budget: ``max_len`` / ``max_len_by_si``, the shared
      completion cap, ``si_ids``, and :data:`SEQ_CONSTRUCTION_VERSION`;
    - the REGISTERED arm set (config ``class_a_arms`` filtered to
      ``methods.ARMS``) with each arm's normalized effective training spec
      (hyperparams as resolved from the config; optimizer/objective via the
      trainer-source content hashes);
    - file-content sha256 of every measurement-relevant source
      (:data:`_MEASUREMENT_SOURCES` + the arms' trainer modules).

    EXPLICITLY EXCLUDED: the derived ``batch_size_by_si``/``batch_size_by_arm``
    overrides, ``train_kwargs.*.batch_size``, and all output paths — the §5.3
    config commit or an unrelated rebase can never self-invalidate the table;
    a material trainer/arm-spec edit rejects it.
    """
    from ..methods import ARMS
    from .defense_cell import CLASS_A_ARMS

    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    arms = [a for a in (cfg.get("class_a_arms") or CLASS_A_ARMS) if a in ARMS]
    arm_specs = _effective_arm_specs(cfg, arms)
    sources = list(_MEASUREMENT_SOURCES)
    for spec in arm_specs.values():
        sources.extend(spec["trainer_sources"])
    payload = {
        "seq_construction_version": SEQ_CONSTRUCTION_VERSION,
        "model": cfg.get("model"),
        "model_revision": cfg.get("model_revision"),
        "dtype": cfg.get("dtype"),
        "attn_implementation": cfg.get("attn_implementation"),
        "layer": cfg.get("layer"),
        "si_ids": list(cfg.get("si_ids") or []),
        "max_len": cfg.get("max_len"),
        "max_len_by_si": dict(cfg.get("max_len_by_si") or {}),
        "completion_tokens": cfg.get("completion_tokens"),
        "arms": arms,
        "arm_specs": arm_specs,
        "source_shas": _source_shas(sources, root),
    }
    return _canonical_sha(payload)


def arm_scoped_inputs_sha(cfg: dict, si_id: str, arm: str, *,
                          repo_root=None) -> str:
    """The PER-(si, arm) resume identity of one GPU-pass trial row: sha256
    over every input that determines THAT row's measured numbers —

    - the model pin, dtype, attn implementation, layer;
    - THIS SI's resolved ``max_len`` + the shared completion cap;
    - THIS arm's normalized effective training spec (config train_kwargs
      default merged under the per-arm block, minus batch width) and its
      trainer-source content hashes;
    - the shared renderer/collator/registry sources;
    - :data:`SEQ_CONSTRUCTION_VERSION` — the MANUAL version of the
      harness's sequence-construction/timing semantics.

    Deliberately EXCLUDED (vs :func:`measurement_inputs_sha`, which gates
    CONSUMPTION and stays r3 #1-complete): the harness's own files
    (:data:`_HARNESS_SOURCES`) — bump SEQ_CONSTRUCTION_VERSION when a
    harness edit changes what a trial MEANS; a mechanical harness edit must
    not invalidate finished rows — and every other arm's/SI's inputs, so a
    one-arm config fix (e.g. train_kwargs.alphakl_soft.beta, 2026-08-19)
    re-runs ONLY that arm's trials on resume.
    """
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    spec = _effective_arm_specs(cfg, [arm])[arm]
    sources = [p for p in _MEASUREMENT_SOURCES if p not in _HARNESS_SOURCES]
    sources.extend(spec["trainer_sources"])
    payload = {
        "seq_construction_version": SEQ_CONSTRUCTION_VERSION,
        "model": cfg.get("model"),
        "model_revision": cfg.get("model_revision"),
        "dtype": cfg.get("dtype"),
        "attn_implementation": cfg.get("attn_implementation"),
        "layer": cfg.get("layer"),
        "si_id": si_id,
        "max_len": resolve_max_len(cfg, si_id),
        "completion_tokens": cfg.get("completion_tokens"),
        "arm": arm,
        "arm_spec": spec,
        "source_shas": _source_shas(sources, root),
    }
    return _canonical_sha(payload)


def load_measured(path) -> dict:
    """Load + structurally validate the measured GPU-pass table. Loud on a
    missing file or a table without the required keys — never a guess."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"measured budget table not found at {p} — run "
            "scripts/measure_train_budget.py (plan §5.1) and commit its "
            "output, or drop the measured_path override")
    measured = json.loads(p.read_text())
    for key in ("measurement_inputs_sha", "safe_batch_size", "trials"):
        if key not in measured:
            raise ValueError(
                f"measured budget table {p} is missing {key!r} — not a "
                "measure_train_budget.py output; refusing to consume it")
    return measured


def _successful_trials(measured: dict) -> dict[tuple[str, str, int], dict]:
    out: dict[tuple[str, str, int], dict] = {}
    for t in measured["trials"]:
        if t.get("oom") or not t.get("ok"):
            continue
        out[(t["si_id"], t["arm"], int(t["batch_size"]))] = t
    return out


def apply_measured(rows: list[CellBudget], measured: dict, *,
                   epochs: int = 6) -> list[CellBudget]:
    """Fill the NEEDS_GPU fields of ``rows`` from the measured table, in
    place (r3 #2 semantics, all three loud):

    a. the row's batch size IS the currently-resolved width (rows are built
       through :func:`resolve_batch_size`);
    b. it must be ``<= safe_batch_size(si, arm)`` — a later config override
       above the measured-safe width REFUSES here, never passes silently;
    c. peak memory / tokens-per-s come from the EXACT successful
       ``(si, arm, batch_size=resolved)`` trial row — absent is a loud
       failure, never the max-safe trial's throughput.

    ``peak_memory_gb`` is filled with the trial's RESERVED peak (what the
    allocator actually held — the 0.85-of-device guard reads the same number);
    the alloc peak stays available in the measured JSON. ``est_seconds`` uses
    the same forward-token convention as :func:`measure_si_budget`
    (``ep * n_examples * total_seq * 2 / tokens_per_s``).
    """
    trials = _successful_trials(measured)
    safe_map = measured["safe_batch_size"]
    for row in rows:
        try:
            safe = int(safe_map[row.si_id][row.arm])
        except (KeyError, TypeError):
            raise ValueError(
                f"measured table has no safe_batch_size for "
                f"({row.si_id!r}, {row.arm!r}) — the table does not cover "
                "this cell; re-run the GPU pass rather than guessing")
        if row.batch_size > safe:
            raise ValueError(
                f"resolved batch size {row.batch_size} for ({row.si_id!r}, "
                f"{row.arm!r}) exceeds the measured-safe width {safe} — "
                "refusing to fill (r3 #2). Lower the config override or "
                "re-measure at the wider batch")
        trial = trials.get((row.si_id, row.arm, row.batch_size))
        if trial is None:
            raise ValueError(
                f"no successful measured trial at the exact resolved batch "
                f"size ({row.si_id!r}, {row.arm!r}, bs={row.batch_size}) — "
                "refusing to fill from another trial's throughput (r3 #2); "
                "re-run the GPU pass at this width")
        row.peak_memory_gb = float(trial["peak_gb_reserved"])
        row.tokens_per_s = float(trial["tokens_per_s_median"])
        fwd_tokens = epochs * row.n_examples * row.total_seq_tokens * 2
        row.est_seconds = f"{fwd_tokens / row.tokens_per_s:.3g}"
    return rows


def _maybe_load_measured(cfg: dict, measured_path) -> dict | None:
    """Resolve + gate the measured table for :func:`measure_si_budget`.

    EXPLICIT request (a ``measured_path`` argument or config key): every
    problem — missing file, malformed table, sha mismatch — raises loudly.

    IMPLICIT default (no key, the committed :data:`DEFAULT_MEASURED_PATH`
    merely exists): an absent file leaves behaviour unchanged, and a
    sha-incompatible table is REJECTED with a stderr warning while the rows
    keep their explicit ``NEEDS_GPU`` marks — the pre-measurement rendering,
    visibly unfilled, never silently-wrong numbers. (The implicit path cannot
    raise: unrelated configs — tests, other lanes — must not be taken hostage
    by a committed table that is not about them.)
    """
    explicit = measured_path is not None or "measured_path" in cfg
    path = measured_path or cfg.get("measured_path") or DEFAULT_MEASURED_PATH
    path = Path(path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    if not path.is_file():
        if explicit:
            raise FileNotFoundError(
                f"measured_path explicitly set but no table at {path} — run "
                "scripts/measure_train_budget.py (plan §5.1) first")
        return None
    measured = load_measured(path)
    want = measurement_inputs_sha(cfg)
    have = measured["measurement_inputs_sha"]
    if have != want:
        msg = (f"measured budget table {path} is measurement_inputs_sha-"
               f"incompatible with this config/code (table {have[:12]}… != "
               f"current {want[:12]}…) — a material model/sequence/trainer "
               "input changed since the GPU pass; re-run "
               "scripts/measure_train_budget.py")
        if explicit:
            raise ValueError(msg)
        print(f"[train-budget] WARNING: {msg}; leaving NEEDS_GPU marks.",
              file=sys.stderr)
        return None
    return measured
