"""Parameterized store validators for the GSM8K paired-200 thinking-ON experiment
(``inspect/gsm8k_paired200``).

The MATH-500 analysis (``scripts/tinycap_math500_thinking.py``) binds its
validators to module globals — ONE task, ONE cap, three cell kinds, one pin per
model, no declared cell. Every validator here takes its facts from the MATRIX
ROW instead (task, cap per rung, view, the (view, cell) pin, the declaration
class, the population), so one function reads the 8,192 leg, the 16,384 rung
leg, a declared header-located cell and the thinking-OFF overlap leg alike.
What IS imported from the MATH-500 module is its genuinely task-agnostic
machinery (plan F10): the pin / producer-tree checks, the sampling-row rule, the
stamp triple, the render / lens certification records, the probe inventory, the
loop rule, the store readers of ``tinycap_oos_transfer.py``. Nothing is copied
and neither module is edited.

THE READER CONTRACT WITH LANES H AND B is the constants block below, taken from
the PRODUCERS by import wherever a producer spells it (the integration rule:
the reader adopts the producer's spellings, never the reverse): the ladder
record's rule / threshold / report line / exits / field names from
``inspect_rung`` (the record's AUTHOR; ``ladder_plan.py`` is its one WRITER),
the subset file's schema / fields / validator / digest and the rung's
``cap_rung`` identity from ``contracts`` (``ITEM_SUBSET_*``, ``CAP_RUNG_FIELDS``,
``item_subset_sha16``, ``cap_rung_identity``), the standard row task
(``GSM8K_STANDARD_TASK``), the stamp triples from the contract's own derivation.
What remains a literal here is what no producer exports as a name (the matrix
row fields ``make_gsm8k_matrix.py`` writes; the attempt-scoped fields; the
analysis' relative layout), each pinned by test to the producer's constant
whenever this tree exports one.

THE NO-SWITCH ARM: a matrix row's ARM (``thinking_mode`` — ``on`` on every
published row, ``not_available`` on a family whose chat template carries no
thinking switch, ``contracts.NO_THINKING_CHANNEL_FAMILIES``) selects every
validator's class. Under ``not_available``: the row decodes under
the model's shipped ``card`` sampling row and carries NO out-of-support tuple
(:func:`row_problem`); the hashed identity runs ``not_available`` on both
thinking keys with NO ``out_of_support_thinking`` and NO ``thinking_header_ids``,
the manifest's ``thinking.enable_thinking`` is NULL (no switch was sent) with
the ``native`` prefill render label, the fact evidence says ``split_mode none``
/ ``thinking_switch false`` (:func:`identity_problem`); every accepted ledger
row reads ``thinking`` null (:func:`ledger_check`); every wire body carries no
``enable_thinking`` (:func:`request_evidence_check`); the certification is the
NO-SWITCH CLASS — (A″) the render measured once live, (B′) verbatim under the
``unspecified`` triple, (C″) the EXPLICIT ``lens_on_bite`` inapplicability
record REQUIRED on every fitted cell, the bites control under the single render
(:func:`certification_check`); the probe inventory is the class's exact one.
Every switch-arm reading is byte-identical to before.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from sisteer.capeval import manifest as CM
from sisteer.capeval.tiers import cap_ladder as CL
from sisteer.capeval.tiers import inspect_rung as IR
from sisteer.capeval.tinycap import completion
from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.serving import proxy as PX     # the resumed-prefill policy vocabulary, by reference (the served-steer class)

_HERE = Path(__file__).resolve().parent


def _load_module(name: str, path: Path) -> Any:
    """Import a script module by path ONCE per process: the module already in
    ``sys.modules`` is returned (so this module and the analysis share one
    MATH-500 module object), else it is registered before executing (a
    dataclass defined in it resolves its module through ``sys.modules``)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load_module("tinycap_math500_thinking", _HERE / "tinycap_math500_thinking.py")
T = M.T
AnalysisError = M.AnalysisError
_mapping = T._mapping
_iter_jsonl = T._iter_jsonl
_read_json = T._read_json

# --------------------------------------------------------------------------- #
# the experiment's vocabulary                                                 #
# --------------------------------------------------------------------------- #

EXPERIMENT = "gsm8k-paired200-paper-objects"
#: the standard paired row task — the subset file's / the adapter's task (contracts.GSM8K_STANDARD_TASK == GSM8K_PAIRED_TASK)
TASK: str = C.GSM8K_STANDARD_TASK                     # inspect/gsm8k_paired200
TASK_NAME: str = C.GSM8K_PAIRED_TASK_NAME             # gsm8k_paired200 (the runner's task name)
N_ITEMS: int = C.GSM8K_PAIRED_N                        # 200
ITEM_SET: str = C.GSM8K_ITEM_SET_PAIRED                # paired200
CAP_BASE = 8192
CAP_RUNG = CL.CAP_CEILING                              # 16,384 — the ceiling, no rung above
MODE_PRIMARY: str = C.SAMPLING_THINKING_CARD
#: llama-paper-rows (P6 / P7): the ARMS a paper row runs under — the thinking-ON arm every published row runs (a
#: schema-1 row reads as `on`) and the NO-SWITCH arm `not_available` of a family with no thinking channel — and the
#: sampling MODE each arm's rows decode under: the four thinking families' `thinking_card`, the no-switch family's
#: shipped `card` row (a "thinking card" for a model with no thinking mode would be a misnomer). Both from the contract;
#: the row's model must have a row in that mode's table (contracts.SAMPLING_TABLE_BY_MODE), never a typed number.
THINKING_ARM_ON: str = C.THINKING_ON
THINKING_ARM_NOT_AVAILABLE: str = C.THINKING_NOT_AVAILABLE
THINKING_ARMS: tuple[str, ...] = (THINKING_ARM_ON, THINKING_ARM_NOT_AVAILABLE)
SAMPLING_MODE_BY_ARM: Mapping[str, str] = {THINKING_ARM_ON: MODE_PRIMARY, THINKING_ARM_NOT_AVAILABLE: C.SAMPLING_CARD}
#: the analysis' label of the no-switch arm on every figure / table row of such a leg (the comparability rule:
#: "thinking: not available" — never "OFF", which would name a switch that is not there)
THINKING_LABEL_BY_ARM: Mapping[str, str] = {THINKING_ARM_ON: "thinking: on", THINKING_ARM_NOT_AVAILABLE: "thinking: not available"}
#: the matrix document schemas (lane B's `g8k_cells.MATRIX_SCHEMA` / `MATRIX_SCHEMA_2`, pinned equal by test): a schema-1
#: document (the published 42-row matrix) carries NO `thinking_mode` on any row and no `wave_root_rel`; a schema-2 one
#: carries `thinking_mode` on EVERY row and names its wave root
MATRIX_SCHEMA_1 = "g8k_matrix/1"
MATRIX_SCHEMA_2 = "g8k_matrix/2"
MATRIX_SCHEMAS: tuple[str, ...] = (MATRIX_SCHEMA_1, MATRIX_SCHEMA_2)
VIEW_MAIN, VIEW_IT = "main", "it"
CELL_NONE: str = C.ROUTE_NONE_CELL
#: the row STATUS vocabulary is the MATH-500 module's — one spelling across both instruments: ``launch`` /
#: ``complete`` are EVIDENCE (required whenever their wave is analysed; ``complete`` a tolerated synonym no producer
#: emits); ``held`` / ``deferred`` (the steering stubs) are DESIGN statuses — reported under
#: ``design.status_by_row``, never collected. STATUSES keeps THIS instrument's order (it is the refusal message's text).
STATUS_LAUNCH, STATUS_COMPLETE, STATUS_HELD, STATUS_DEFERRED = M.STATUS_LAUNCH, M.STATUS_COMPLETE, M.STATUS_HELD, M.STATUS_DEFERRED
STATUSES: tuple[str, ...] = (STATUS_LAUNCH, STATUS_COMPLETE, STATUS_HELD, STATUS_DEFERRED)
EVIDENCE_STATUSES: tuple[str, ...] = M.EVIDENCE_STATUSES
DESIGN_STATUSES: tuple[str, ...] = (STATUS_HELD, STATUS_DEFERRED)
#: the wave a row without a ``wave`` field belongs to (the published matrix carries none), and the out-root prefixes: the
#: published ``controls/`` tail, or a WAVE-ISOLATED ``controls_<wave>/`` (plan P2 / F11a: a pull of one wave's subtree
#: never carries another wave's leg) — the MATH-500 module's spellings
WAVE_PUBLISHED, OUT_ROOT_PREFIX = M.WAVE_PUBLISHED, M.OUT_ROOT_PREFIX
#: the external-object arms name their bank by ``object_source`` instead of a full22 pin (design-only rows here)
OBJECT_SOURCE_EXTERNAL = "external"
#: E3 (math-eight-arms-extension; decisions-e3): the INSTRUCT-DIRECT ("it-doc") batch-64 RE-FIT rows name their bank by
#: ``object_source: itdoc`` — the pin is the it-doc ADMISSION RECORD's (``configs/sdfbase/itdoc_admission.json``, keyed by (lane,
#: model key, view, cell id) -> object sha256) and rides the ROW as ``object_sha256``; never the full22 pin file's. The same spelling
#: as the generator's (``g8k_cells.OBJECT_SOURCE_ITDOC`` / ``full22.itdoc_bind.OBJECT_SOURCE_ITDOC``; pinned equal by test).
OBJECT_SOURCE_ITDOC = "itdoc"
OBJECT_SOURCES: tuple[str, ...] = (OBJECT_SOURCE_EXTERNAL, OBJECT_SOURCE_ITDOC)
#: the it-doc row fields this reader carries beside every it-doc reading (the lane the pin is keyed under; the pair's QUALIFICATION —
#: a verbatim sentence read off the ROW, never retyped here — and the cross-model qualification PATTERN sentence, when the row carries
#: one): each ``None`` on a row of any other source
MATRIX_ROW_FIELD_LANE = "lane"
MATRIX_ROW_FIELD_QUALIFICATION = "qualification"
MATRIX_ROW_FIELD_QUALIFICATION_PATTERN = "qualification_pattern"
ITDOC_ROW_FIELDS: tuple[str, ...] = (MATRIX_ROW_FIELD_LANE, MATRIX_ROW_FIELD_QUALIFICATION, MATRIX_ROW_FIELD_QUALIFICATION_PATTERN)
#: a wave mixing the served batch-8 constitution with the batch-64 control + difference records the mix on every row:
#: the object GENERATION of a fitted row — ``batch8`` | ``batch64`` — is READ from the object's FIT PROVENANCE by its sha256,
#: never inferred from a staged id's spelling. Two records bind a sha to an effective batch: the it-doc admission record
#: (``configs/sdfbase/itdoc_admission.json``: a fitted entry's ``fit_record.effective_batch``; a derived entry's TWO PARENTS' — they must
#: agree) and the declared batch-8 fit record (:data:`OBJECT_FIT_BATCHES_REL`: the direct-fit objects' ``train_budget_meta.json`` read on
#: the bank, each one named by an admission entry's ``superseded`` block — the record cannot bind an object the admission record does not
#: name). The word per effective batch is :data:`OBJECT_GENERATION_BY_EFFECTIVE_BATCH`; any other batch has no word and refuses by name.
#: A row that DECLARES ``object_generation`` (the it-doc matrix's rows do, from the launcher) is held to the binding: a declared word its
#: provenance does not bind, or binds differently, refuses by name; an undeclared, unbound fitted row stays unstamped (the export's ``n/a``).
OBJECT_FIT_BATCHES_REL = Path("configs/sdfbase/gsm8k/object_fit_batches.json")
OBJECT_FIT_BATCHES_SCHEMA = "object_fit_batches/1"
ITDOC_ADMISSION_REL = Path("configs/sdfbase/itdoc_admission.json")
OBJECT_GENERATION_BATCH8, OBJECT_GENERATION_BATCH64 = "batch8", "batch64"
OBJECT_GENERATION_BY_EFFECTIVE_BATCH: Mapping[int, str] = {8: OBJECT_GENERATION_BATCH8, 64: OBJECT_GENERATION_BATCH64}
MATRIX_ROW_FIELD_OBJECT_GENERATION = "object_generation"
_FIT_BATCH_ENTRY_REQUIRED = ("model_key", "view", "cell", "staged_id", "object_kind", "effective_batch", "model_id", "source", "named_by")
_FIT_BATCH_SOURCE_REQUIRED = ("file", "bank_rel", "cluster_role", "object_pt_sha256_on_disk")
_HEX_CHARS = frozenset("0123456789abcdef")


def _is_sha256(s: Any) -> bool:
    return isinstance(s, str) and len(s) == 64 and set(s) <= _HEX_CHARS


def _generation_of_effective_batch(n: Any, *, where: str) -> tuple[str | None, str | None]:
    """``(word, problem)``: the generation word of an effective batch, or the problem when the batch is not one the vocabulary names."""
    if not isinstance(n, int) or isinstance(n, bool):
        return None, f"{where}: effective_batch {n!r} is not an integer"
    word = OBJECT_GENERATION_BY_EFFECTIVE_BATCH.get(n)
    if word is None:
        return None, f"{where}: effective_batch {n} names no generation word (the vocabulary: {dict(OBJECT_GENERATION_BY_EFFECTIVE_BATCH)})"
    return word, None


def load_object_fit_batches(source: Path | str | Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The declared batch-8 fit record (:data:`OBJECT_FIT_BATCHES_REL`) -> ``{sha256: entry}``; every entry validated by name at load:
    the schema word, a 64-hex key equal to the entry's ``source.object_pt_sha256_on_disk``, the required fields, the source's
    read-on-disk fields, an effective batch the vocabulary names, the generation table equal to the reader's."""
    doc = read_json_if_file(Path(source).expanduser()) if isinstance(source, (str, Path)) else dict(source)
    if not isinstance(doc, Mapping):
        raise AnalysisError(f"object fit batches: no record at {source}")
    if doc.get("schema") != OBJECT_FIT_BATCHES_SCHEMA:
        raise AnalysisError(f"object fit batches: schema {doc.get('schema')!r} != {OBJECT_FIT_BATCHES_SCHEMA!r}")
    table = doc.get("generation_by_effective_batch")
    try:
        table_ok = isinstance(table, Mapping) and {int(k): v for k, v in table.items()} == dict(OBJECT_GENERATION_BY_EFFECTIVE_BATCH)
    except (TypeError, ValueError):
        table_ok = False
    if not table_ok:
        raise AnalysisError(f"object fit batches: generation_by_effective_batch {table!r} != the reader's {dict(OBJECT_GENERATION_BY_EFFECTIVE_BATCH)}")
    entries = doc.get("entries")
    if not isinstance(entries, Mapping) or not entries:
        raise AnalysisError("object fit batches: no entries")
    out: dict[str, dict[str, Any]] = {}
    for sha, e in entries.items():
        where = f"object fit batches: entry {str(sha)[:16]}"
        if not _is_sha256(sha):
            raise AnalysisError(f"{where}: the key is not a 64-hex sha256")
        if not isinstance(e, Mapping):
            raise AnalysisError(f"{where}: not a mapping")
        missing = [f for f in _FIT_BATCH_ENTRY_REQUIRED if f not in e]
        if missing:
            raise AnalysisError(f"{where}: missing {missing}")
        src = e.get("source")
        lacking = [f for f in _FIT_BATCH_SOURCE_REQUIRED if not isinstance(src, Mapping) or not src.get(f)]
        if lacking:
            raise AnalysisError(f"{where}: source lacks {lacking} (the read-on-disk provenance is the whole point of the entry)")
        if src.get("object_pt_sha256_on_disk") != sha:
            raise AnalysisError(f"{where}: source.object_pt_sha256_on_disk {str(src.get('object_pt_sha256_on_disk'))[:16]} != the key")
        _word, problem = _generation_of_effective_batch(e.get("effective_batch"), where=where)
        if problem:
            raise AnalysisError(problem)
        out[sha] = dict(e)
    return out


def object_generation_bindings(itdoc_admission: Mapping[str, Any] | None, fit_batches: Mapping[str, Mapping[str, Any]] | None
                               ) -> dict[str, dict[str, Any]]:
    """``{object sha256: {generation, effective_batch, source}}`` from the two provenance records, refusing a defect by name: a
    fitted admission entry binds through ``fit_record.effective_batch``; a derived entry through its two parents' fit records (the
    ``parents`` block of its model, each parent resolved to the fitted entry carrying that sha; the two must agree); a declared
    batch-8 entry binds only when the admission record names its sha as a ``superseded`` object. The same sha bound twice must agree."""
    out: dict[str, dict[str, Any]] = {}

    def bind(sha: Any, generation: str, batch: int, source: str, *, where: str) -> None:
        if not _is_sha256(sha):
            raise AnalysisError(f"{where}: object sha256 {sha!r} is not 64-hex")
        prev = out.get(sha)
        if prev is not None and prev["generation"] != generation:
            raise AnalysisError(f"{where}: object {sha[:16]} bound to {generation} here and to {prev['generation']} by {prev['source']}")
        out[sha] = {"generation": generation, "effective_batch": batch, "source": source}

    superseded: set[str] = set()
    if itdoc_admission is not None:
        entries = itdoc_admission.get("entries") if isinstance(itdoc_admission, Mapping) else None
        if not isinstance(entries, Mapping):
            raise AnalysisError("it-doc admission record: no entries")
        fitted_by_sha: dict[str, tuple[str, Mapping[str, Any]]] = {}
        flat: list[tuple[str, Mapping[str, Any]]] = []
        for key, by_view in entries.items():
            for view, by_cell in (by_view or {}).items():
                for cell, e in (by_cell or {}).items():
                    where = f"it-doc admission entries.{key}.{view}.{cell}"
                    if not isinstance(e, Mapping):
                        continue
                    flat.append((where, e))
                    if e.get("object_kind") == "fitted" and _is_sha256(e.get("object_sha256")):
                        fitted_by_sha[e["object_sha256"]] = (where, e)
                    sup = e.get("superseded")
                    if isinstance(sup, Mapping) and _is_sha256(sup.get("object_sha256")):
                        superseded.add(sup["object_sha256"])
        parents_all = itdoc_admission.get("parents") if isinstance(itdoc_admission.get("parents"), Mapping) else {}
        for where, e in flat:
            if not _is_sha256(e.get("object_sha256")):
                continue
            kind = e.get("object_kind")
            if kind == "fitted":
                rec = e.get("fit_record")
                if not isinstance(rec, Mapping) or "effective_batch" not in rec:
                    raise AnalysisError(f"{where}: a fitted entry with no fit_record.effective_batch")
                word, problem = _generation_of_effective_batch(rec["effective_batch"], where=f"{where}.fit_record")
                if problem:
                    raise AnalysisError(problem)
                bind(e["object_sha256"], word, int(rec["effective_batch"]), f"{where}.fit_record.effective_batch", where=where)
            elif kind == "derived":
                parents = parents_all.get(e.get("model_key"))
                if not isinstance(parents, Mapping) or not parents:
                    raise AnalysisError(f"{where}: a derived entry whose model has no parents block")
                words: dict[str, int] = {}
                for role, p in parents.items():
                    psha = p.get("object_sha256") if isinstance(p, Mapping) else None
                    hit = fitted_by_sha.get(psha) if isinstance(psha, str) else None
                    if hit is None:
                        raise AnalysisError(f"{where}: parent {role!r} ({str(psha)[:16]}) is not a fitted entry of the record")
                    prec = hit[1].get("fit_record") or {}
                    pw, problem = _generation_of_effective_batch(prec.get("effective_batch"), where=f"{hit[0]}.fit_record")
                    if problem:
                        raise AnalysisError(problem)
                    words[pw] = int(prec["effective_batch"])
                if len(words) != 1:
                    raise AnalysisError(f"{where}: the parents' effective batches disagree ({sorted(words)})")
                ((word, batch),) = words.items()
                bind(e["object_sha256"], word, batch, f"{where}: the two parents' fit_record.effective_batch", where=where)
            else:
                raise AnalysisError(f"{where}: object_kind {kind!r} is neither fitted nor derived")
    if fit_batches:
        for sha, e in fit_batches.items():
            where = f"object fit batches entry {str(sha)[:16]}"
            if itdoc_admission is not None and sha not in superseded:
                raise AnalysisError(f"{where}: not a `superseded` object of the it-doc admission record — the declared record may bind only "
                                    "objects the admission record names")
            word, problem = _generation_of_effective_batch(e.get("effective_batch"), where=where)
            if problem:
                raise AnalysisError(problem)
            src = e.get("source") if isinstance(e.get("source"), Mapping) else {}
            bind(sha, word, int(e["effective_batch"]),
                 f"{OBJECT_FIT_BATCHES_REL}: {src.get('bank_rel')}/{src.get('file')} effective_batch (read on the {src.get('cluster_role')} bank)",
                 where=where)
    return out


def object_generation_of_row(row: Mapping[str, Any], bindings: Mapping[str, Mapping[str, Any]] | None) -> tuple[str | None, str | None]:
    """``(stamp, problem)`` for one matrix row: the generation word its ``object_sha256`` is bound to (None on a non-fitted row, and on
    a fitted row no record binds — unless the row DECLARES one, which is then a problem by name); a declared word that disagrees with
    the binding is a problem by name. The word is never read off ``staged_id``."""
    declared = row.get(MATRIX_ROW_FIELD_OBJECT_GENERATION)
    tag = f"row {row.get('index')} ({row.get('model_key')}/{row.get('view')}/{row.get('cell')})"
    if declared is not None and declared not in OBJECT_GENERATION_BY_EFFECTIVE_BATCH.values():
        return None, f"{tag}: declared object_generation {declared!r} not in {sorted(OBJECT_GENERATION_BY_EFFECTIVE_BATCH.values())}"
    if not row.get("fitted"):
        return (None, f"{tag}: declared object_generation {declared!r} on a non-fitted row") if declared is not None else (None, None)
    sha = row.get("object_sha256")
    b = (bindings or {}).get(sha) if isinstance(sha, str) else None
    if b is None:
        if declared is not None:
            return None, (f"{tag}: declares object_generation {declared!r} but no fit provenance binds object {str(sha)[:16]} "
                          f"(the it-doc admission record's fit records / {OBJECT_FIT_BATCHES_REL})")
        return None, None
    if declared is not None and declared != b["generation"]:
        return None, f"{tag}: declares object_generation {declared!r} but its fit provenance binds {b['generation']} ({b['source']})"
    return str(b["generation"]), None


POSITIONS_ALL, POSITIONS_RESPONSE = "all", "response"
SERVED_POSITIONS: tuple[str, ...] = (POSITIONS_ALL, POSITIONS_RESPONSE)
#: the overlap check's thinking-OFF greedy leg
OVERLAP_CAP = 512
OVERLAP_SAMPLING: str = C.SAMPLING_GREEDY
RETIRED_TASK = "s2parity/gsm8k_0shot"
RETIRED_DOMAIN = "math_s2parity"

# -- the matrix row: the fields this reader needs; a row may carry more --
MATRIX_ROW_FIELDS: tuple[str, ...] = (
    "index", "job_id", "model_key", "view", "tag", "cell", "cell_slug", "harness_cell", "staged_id", "object_sha256",
    "fitted", "route_kind", "served_positions_expected", "declared", "tuple_name", "sampling_mode", "out_root",
    "rung_out_root", "family", "status", "held_reason")
#: llama-paper-rows (P16): the OPTIONAL row field a schema-2 matrix adds — the row's arm (`on` | `not_available`);
#: absent (every published row) reads as `on`
MATRIX_ROW_FIELD_THINKING_MODE = "thinking_mode"
#: llama-paper-rows: the OPTIONAL row field a schema-2 matrix adds — the row's ladder TRIGGER (the
#: harmonised 1 %); absent (every published row) reads as the adapter's 3 % under which those records were produced
MATRIX_ROW_FIELD_TRIGGER_THRESHOLD = "trigger_threshold"


def row_trigger_threshold(row: Mapping[str, Any]) -> float:
    return float(row.get(MATRIX_ROW_FIELD_TRIGGER_THRESHOLD, GSM8K_TRIGGER_THRESHOLD))

RUNG_OUT_ROOT_SUFFIX = "__rung16384"

# -- the attempt-scoped producer contract: every ledger row and request-evidence record carries the
#    attempt it belongs to; the manifest names the ACCEPTED generation attempt; every recorded attempt has a kind --
LEDGER_ATTEMPT_ID_FIELD = "attempt_id"
REQUEST_EVIDENCE_ATTEMPT_ID_FIELD = "attempt_id"
MANIFEST_KEY_ACCEPTED_ATTEMPT = "accepted_generation_attempt"
RESUME_ATTEMPT_ID_FIELD = "attempt_id"
RESUME_ATTEMPT_KIND_FIELD = "kind"
ATTEMPT_KIND_GENERATION = "generation"
ATTEMPT_KIND_FINALISATION = "finalisation"
ATTEMPT_KINDS: tuple[str, ...] = (ATTEMPT_KIND_GENERATION, ATTEMPT_KIND_FINALISATION)
#: the MATH driver's four per-run fields, still required on every entry (a superset is admitted here)
RESUME_ATTEMPT_REQUIRED_FIELDS: tuple[str, ...] = tuple(M.RESUME_ATTEMPT_FIELDS)
#: the attempt's start / end stamps (two of the contract's attempt-scoped fields, ``contracts.RESUME_ATTEMPT_SCOPED_FIELDS``,
#: pinned by test): the interval the accepted attempt generated in — what a retained ``.eval`` log is bound to
RESUME_ATTEMPT_STARTED_FIELD = "started"
RESUME_ATTEMPT_ENDED_FIELD = "ended"

# -- the rung identity and the subset: the CONTRACT's spellings, by import --
IDENTITY_KEY_CAP_RUNG: str = C.IDENTITY_KEY_CAP_RUNG
CAP_RUNG_FIELDS: tuple[str, ...] = tuple(C.CAP_RUNG_FIELDS)
#: the manifest field a rung's ``source_identity_hash`` binds to (the tinycap identity hash of the source store; the
#: adapter's ``StoreFacts.identity_hash`` reads the same field)
SOURCE_IDENTITY_HASH_FIELD = "harness_config_hash"
#: the rung manifest's UNHASHED subset block beside ``cap_rung`` (the driver writes it from the validated subset file)
MANIFEST_KEY_ITEM_SUBSET: str = C.MANIFEST_KEY_ITEM_SUBSET
ITEM_SUBSET_BLOCK_FIELDS: tuple[str, ...] = tuple(C.ITEM_SUBSET_BLOCK_FIELDS)
#: the subset digest: the producer's own function (sorted unique ids, canonical JSON, 16 hex)
subset_sha16 = C.item_subset_sha16
RUNG_SUBSET_NAME = "rung_subset.json"                  # lane B's g8k_cells.RUNG_SUBSET_FILENAME (pinned by test when importable)
RUNG_SUBSET_REL = "rung16384/{tag}/{cell_slug}/" + RUNG_SUBSET_NAME   # $WAVE_ROOT/<g8k_cells.RUNG_DIR_REL>/<TAG>/<slug>/
RUNG_SUBSET_SCHEMA: str = C.ITEM_SUBSET_SCHEMA
RUNG_SUBSET_FIELDS: tuple[str, ...] = tuple(C.ITEM_SUBSET_FIELDS)
validate_item_subset = C.validate_item_subset

# -- the ladder record THIS instrument writes: inspect_rung.classify_store is its AUTHOR (every spelling below is the
#    adapter's, by import), ladder_plan.py its one WRITER (<out_root>/cap_ladder.json; cap_ladder.write_cell_record's
#    convention, <cell_dir>/cap_ladder.json, is read too — see :func:`read_ladder_record`) --
LADDER_RECORD_NAME: str = CL.RECORD_NAME
LADDER_RECORD_SCHEMA: str = CL.RECORD_SCHEMA
GSM8K_TRIGGER_THRESHOLD: float = IR.GSM8K_TRIGGER_THRESHOLD
GSM8K_REPORT_LINE: float = IR.GSM8K_REPORT_LINE
GSM8K_TRIGGER_RULE: str = IR.GSM8K_TRIGGER_RULE
#: the rule -> threshold pairs a RUNG IDENTITY or SUBSET FILE may carry (the current pair and the retired 3 % pair the first
#: publication planned under; inspect_rung.GSM8K_TRIGGER_RULES_ADMITTED): a rung planned under the retired pair is exactly the rung
#: the harmonised rule plans (the eligible set does not depend on the threshold; 3 % implies 1 %), and its manifest hashes the rule
#: it was planned under into the store identity — immutable evidence, admitted and REPORTED, never re-written. A cap_ladder.json
#: RECORD is re-filed by the planner and is held to the current pair alone.
GSM8K_TRIGGER_RULES_ADMITTED: Mapping[str, float] = IR.GSM8K_TRIGGER_RULES_ADMITTED
GSM8K_TRIGGER_RULE_3PCT_RETIRED: str = IR.GSM8K_TRIGGER_RULE_3PCT_RETIRED
GSM8K_TRIGGER_PROVENANCE: str = IR.GSM8K_TRIGGER_PROVENANCE
#: the label a rung's planning rule is reported under
RUNG_PLANNED_UNDER_CURRENT, RUNG_PLANNED_UNDER_RETIRED_3PCT = "1pct", "3pct_retired_2026-09-10"


def trigger_pct_label(threshold: float = GSM8K_TRIGGER_THRESHOLD) -> str:
    """``0.01 -> '1 %'`` — the one spelling of the trigger every report line, legend and caption derives from the constant."""
    v = 100.0 * float(threshold)
    return f"{v:g} %"


def rung_planned_under(rule: Any) -> str | None:
    """Which admitted pair a rung identity / subset file was planned under (``None`` for a rule string this instrument never used)."""
    if rule == GSM8K_TRIGGER_RULE:
        return RUNG_PLANNED_UNDER_CURRENT
    if rule == GSM8K_TRIGGER_RULE_3PCT_RETIRED:
        return RUNG_PLANNED_UNDER_RETIRED_3PCT
    return None


def admitted_trigger_rules(trigger_threshold: float = GSM8K_TRIGGER_THRESHOLD) -> Mapping[str, float]:
    """The rule -> threshold pairs a rung identity / subset file may carry under a ROW's trigger (g8k_cells.row_trigger_threshold):
    at the instrument's current trigger, :data:`GSM8K_TRIGGER_RULES_ADMITTED` (the current pair and the retired 3 % one — a 3 %
    trigger implies a 1 % one, so a rung planned under it is the rung the current rule plans); at any other declared trigger,
    that trigger's own rule alone. A cap_ladder.json RECORD is never read through this: it is held to its row's pair alone."""
    if M._same_number(trigger_threshold, GSM8K_TRIGGER_THRESHOLD):
        return GSM8K_TRIGGER_RULES_ADMITTED
    return {IR.gsm8k_trigger_rule(trigger_threshold): float(trigger_threshold)}


EXIT_TRIGGERED_NO_ELIGIBLE: str = IR.EXIT_TRIGGERED_NO_ELIGIBLE
EXITS_NOT_TRIGGERED: tuple[str, ...] = tuple(IR.NOT_TRIGGERED_EXITS)
EXITS_TRIGGERED_TERMINAL: tuple[str, ...] = tuple(IR.TRIGGERED_TERMINAL_EXITS)
#: the adapter's ``decide`` names the rung (``next_cap``) on BOTH exits that reached the transport check
EXITS_NAMING_A_RUNG: tuple[str, ...] = (IR.EXIT_ESCALATE, CL.EXIT_TRANSPORT_REFUSED)
LADDER_EXITS: tuple[str, ...] = tuple(IR.EXITS)
#: every instrument field the adapter asserts onto its record (``inspect_rung.RECORD_INSTRUMENT_FIELDS``)
LADDER_RECORD_FIELDS: tuple[str, ...] = tuple(IR.RECORD_INSTRUMENT_FIELDS)
#: the three id lists the reader reads off the record — the adapter's own names
LADDER_ELIGIBLE_IDS: str = IR.RECORD_FIELD_ELIGIBLE_IDS
LADDER_NON_TERMINATING_IDS: str = IR.RECORD_FIELD_NON_TERMINATING_IDS
LADDER_BOUNDARY_IDS: str = IR.RECORD_FIELD_BOUNDARY_IDS
#: inspect_ai's at-cap stop reason, the producer's one definition (the adapter re-exports inspect_served's)
C_STOP_AT_CAP: str = IR.STOP_REASON_AT_CAP
#: the redraw outcomes, Evals' vocabulary, RE-DERIVED here from the rung outcome's stop reason and token count
REDRAW_STILL_AT_CAP, REDRAW_EXTENDED, REDRAW_BOUNDARY_LANDING = (CL.REDRAW_STILL_AT_CAP, CL.REDRAW_EXTENDED,
                                                                 CL.REDRAW_BOUNDARY_LANDING)
#: the outcome-defining per-draw facts the retained ``.eval`` log must agree with the outcome record on, draw by draw
#: (beside the parent's score availability / value): ``inspect_served.ItemResult``'s producer facts as
#: ``outcome_record`` files them
OUTCOME_FIELDS_FROM_LOG: tuple[str, ...] = ("stop_reason", "at_cap", "completion_tokens", "prompt_tokens", "think_block_present",
                                            "think_block_unterminated", "stray_open_tag", "content_empty_with_reasoning")

# -- the stamp triple per cell class: the CONTRACT's own exports (derived there by its rule, never spelt here): the base /
#    prompted route, an undeclared all-position object attested `always`, a DECLARED header-located object --
STAMPS_BASE: tuple[str, ...] = tuple(C.G8K_STAMPS_BASE)
STAMPS_FITTED: tuple[str, ...] = tuple(C.G8K_STAMPS_FITTED)
STAMPS_DECLARED: tuple[str, ...] = tuple(C.G8K_STAMPS_DECLARED)
#: the NO-SWITCH arm's two classes (llama-paper-rows P7): the same pairs under the `unspecified` render the proxy stamps
#: when no switch rides — the contract's own derivation
STAMPS_BASE_NOT_AVAILABLE: tuple[str, ...] = tuple(C.G8K_STAMPS_BASE_NOT_AVAILABLE)
STAMPS_FITTED_NOT_AVAILABLE: tuple[str, ...] = tuple(C.G8K_STAMPS_FITTED_NOT_AVAILABLE)
#: the render regime every stamp of a no-switch leg reads (no switch is sent, so the proxy stamps the absent key) — the
#: contract's table row for the arm (contracts.RENDER_REGIME_BY_ARM), never a re-typed literal
RENDER_REGIME_NOT_AVAILABLE: str = C.RENDER_REGIME_BY_ARM[C.THINKING_NOT_AVAILABLE]

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX16 = re.compile(r"^[0-9a-f]{16}$")


# --------------------------------------------------------------------------- #
# the matrix                                                                   #
# --------------------------------------------------------------------------- #


def leg_key(model_key: str, view: str, cell: str, cap: int) -> str:
    """``<model_key>/<view>/<cell>@<cap>`` — every per-leg number is keyed on the full leg."""
    return f"{model_key}/{view}/{cell}@{int(cap)}"


def cell_key(model_key: str, view: str, cell: str) -> str:
    return f"{model_key}/{view}/{cell}"


def tag_of(model_key: str, view: str) -> str:
    """``f22_resolve_config``'s spelling: the bare key for ``main``, ``<key>_<view>`` otherwise."""
    return model_key if view == VIEW_MAIN else f"{model_key}_{view}"


TUPLE_NAME_SUFFIX_BASE = f"_{CAP_BASE}"


def tuple_name_for_cap(tuple_name: str, cap: int) -> str:
    """The authorised tuple a leg at ``cap`` is recorded under (twelve names
    ``gsm8k_paired200_<model>[_declared]_<cap>``): the matrix row carries
    the 8,192 name; the rung's is the same name at 16,384 — the CONTRACT's own
    spelling (``contracts.g8k_tuple_name``), and a name the registry authorises."""
    if not tuple_name.endswith(TUPLE_NAME_SUFFIX_BASE):
        raise AnalysisError(f"tuple name {tuple_name!r} does not end with {TUPLE_NAME_SUFFIX_BASE!r} — the matrix carries the 8,192 tuple")
    stem = tuple_name[: -len(TUPLE_NAME_SUFFIX_BASE)]
    declared = stem.endswith("_declared")
    model_key = stem.removeprefix(f"{C.G8K_TUPLE_PREFIX}_").removesuffix("_declared")
    name = C.g8k_tuple_name(model_key, cap, declared=declared)
    if name not in C.G8K_EXPERIMENT_NAMES:
        raise AnalysisError(f"tuple name {name!r} (from {tuple_name!r} at cap {cap}) is not one of the registry's authorised GSM8K tuples")
    return name


def expected_tuple_name(row: Mapping[str, Any], cap: int) -> str:
    """The contract's tuple name for a fitted row's leg at ``cap``: the all-position / declared tuple of the row's model
    (``contracts.g8k_tuple_name``); for an EXTERNAL-object row (``object_source: external`` — the AlphaSteer / Arditi
    rung, math-eight-arms-extension P8) the rung's own ``_steer_<cap>`` tuple (``contracts.g8k_steer_tuple_name``). An
    IT-DOC row (E3, ``object_source: itdoc``) is a fitted, undeclared, all-position LoRA row of the `it` view and runs the
    model's ALL-POSITION tuple like every other fitted row — never the ``_steer`` tuple.

    An external row's tuple is PER CELL (arditi-allpos-cap, 2026-09-17): the ALL-POSITIONS Arditi arm
    (``M.CELL_ARDITI_ALLPOS``) runs its OWN ``_steer_allpos_<cap>`` tuple, never the rung's — the ``_steer`` tuples
    describe a prefill-only, residual-norm-fraction-dosed arm picked inside its over-refusal budget, which this one is
    not, and their ``cells`` are recorded by name by finished legs. The name comes from ``contracts`` itself
    (``g8k_allpos_tuple_name``) — never a literal kept in two places."""
    if row.get("object_source") == OBJECT_SOURCE_EXTERNAL:
        key = str(row["model_key"])
        if str(row.get("cell")) == M.CELL_ARDITI_ALLPOS:
            return C.g8k_allpos_tuple_name(key, cap)
        return C.g8k_steer_tuple_name(key, cap)
    return C.g8k_tuple_name(str(row["model_key"]), cap, declared=bool(row["declared"]))


def expected_stamps_for(*, fitted: bool, declared: bool, arm: str = THINKING_ARM_ON) -> tuple[str, ...]:
    """The class's stamp triple — the contract's own rule (``contracts.g8k_expected_stamps``);
    ``arm`` (llama-paper-rows P7) defaults to the thinking-ON arm every published row runs."""
    return tuple(C.g8k_expected_stamps(fitted=fitted, declared=declared, arm=arm))


#: the stamp classes a leg may be held to: the matrix row's class (``contracts.g8k_expected_stamps``: base / fitted / declared)
#: and the SERVED-STEER class of the served-steering route
STAMP_CLASS_ROW, STAMP_CLASS_SERVED_STEER, STAMP_CLASS_NATIVE = "matrix_row", "served_steer", "native_engine"
#: native-serving-default: a natively served leg's triple is the ENGINE's own (contracts.expected_stamp_regimes, lens=False —
#: the cell-stable KV namespace, the proxy's `not-consulted` policy, the arm's render), selected off the store's route
STAMP_CLASS_NATIVE_SELECTOR = f"route {C.SERVED_ENGINE_FIELD} / serving_identity -> certification_path_of == {M.STEERED_PATH_LORA!r}"
STAMP_CLASS_SERVED_STEER_SELECTOR = f"evidence.{C.OOS_EVIDENCE_BITES_BLOCK}.{C.OOS_EVIDENCE_BITES_LIVENESS_GATE} == {C.LENS_LIVENESS_LOGPROB!r}"


def served_steer_stamps(arm: str) -> tuple[str, ...]:
    """The served-steer class's stamp triple, DERIVED through the contract's own rule (never spelled): a lens leg whose
    resumed-prefill policy is the proxy's ``never`` — the KV namespace ``request-unique`` under that policy, the render of
    ``arm`` (``contracts.expected_stamp_regimes``)."""
    return tuple(C.expected_stamp_regimes(lens=True, arm=arm, policy=PX.RESUMED_PREFILL_NEVER))


def leg_expected_stamps(row: Mapping[str, Any], evidence: Mapping[str, Any] | None, *, arm: str,
                        route: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The stamp class ONE LEG is held to, and its triple. The matrix row's class (``row["expected_stamps"]``) unless the
    leg's own steering-bites record says the LOGPROB liveness statistic ruled it (:data:`STAMP_CLASS_SERVED_STEER_SELECTOR`):
    such a leg was served by the request-unique / never-resume route of the served-steer launcher, whose stamps
    are :func:`served_steer_stamps` — a class selected by the LEG's record, never by the cell's name. A declared (header-
    located) row or a non-fitted row is never served-steer: the selector is read only on an undeclared fitted row; any other
    row carrying it is a refusal by name (``problem``). With ``route`` (the store's own route block) the ENGINE selects
    first: a NATIVE leg — the native engine's certification path (``certification_path_of``) — is held to the engine's OWN
    triple (:data:`STAMP_CLASS_NATIVE`; ``contracts.expected_stamp_regimes(lens=False, …)``: the cell-stable KV namespace,
    the proxy's ``not-consulted`` policy, the arm's render), never the matrix row's LENS triple (the loader files the lens
    class on every row, engine-unaware) and never the served-steer class, whatever its liveness statistic — the producer
    writes ``liveness_gate == logprob`` on EITHER engine (the run's liveness mode × the object kind), while the served-steer
    stamps are the LENS launcher's request-unique / never-resume triple (native-serving-default). Returns
    ``{class, stamps, selected_by, problem}``."""
    fitted, declared = bool(row.get("fitted")), bool(row.get("declared"))
    bites = _mapping(_mapping(evidence).get(C.OOS_EVIDENCE_BITES_BLOCK)) if isinstance(evidence, Mapping) else {}
    gate = bites.get(C.OOS_EVIDENCE_BITES_LIVENESS_GATE)
    row_stamps = tuple(str(x) for x in (row.get("expected_stamps") or ()))
    if route is not None:
        path, _path_problems = M.certification_path_of(route)        # a torn route is named by certification_check itself
        if path == M.STEERED_PATH_LORA:
            # the engine's own triple, before any liveness statistic: the row's triple is the LENS class the loader files on
            # every row, and the served-steer class is the lens launcher's — neither is a natively served leg's
            stamps = list(C.expected_stamp_regimes(lens=False, arm=arm, policy=C.POLICY_NOT_CONSULTED))
            return {"class": STAMP_CLASS_NATIVE, "stamps": stamps, "selected_by": STAMP_CLASS_NATIVE_SELECTOR, "problem": None}
    if gate != C.LENS_LIVENESS_LOGPROB:
        return {"class": STAMP_CLASS_ROW, "stamps": list(row_stamps), "selected_by": "the matrix row's class (contracts.g8k_expected_stamps)",
                "problem": None}
    if not fitted or declared:
        return {"class": STAMP_CLASS_ROW, "stamps": list(row_stamps), "selected_by": STAMP_CLASS_SERVED_STEER_SELECTOR,
                "problem": (f"steering bites: liveness_gate {C.LENS_LIVENESS_LOGPROB!r} on a {'declared' if declared else 'non-fitted'} row — "
                            "the served-steer class is an UNDECLARED fitted route's; another producer's record")}
    return {"class": STAMP_CLASS_SERVED_STEER, "stamps": list(served_steer_stamps(arm)), "selected_by": STAMP_CLASS_SERVED_STEER_SELECTOR,
            "problem": None}


def served_steer_problems(evidence: Mapping[str, Any], *, cert_class: str = C.CERT_CLASS_SWITCH) -> list[str]:
    """The served-steer class's OWN requirements beyond its stamps, each refused by name: the logprob liveness statistic
    ruled with at least the record's minimum of CONCLUSIVE probes (``steering_bites.logprob_liveness.n_conclusive >=
    min_conclusive_probes``), and — on the SWITCH certification class — the lens-ON bite is EITHER the ordinary token-bite
    record (``differs_from_base`` True, no ``liveness_evidence`` key — the producer writes none there) OR the class-E record
    admitted by that same statistic (``liveness_evidence == contracts.LENS_ON_BITE_EVIDENCE_LOGPROB``). Under the NO-SWITCH
    class (a family with no thinking switch, P18) the lens-ON bite is by contract the EXPLICIT inapplicability record (C″),
    which the class's own check requires; the liveness requirement there is the conclusive logprob verdict alone. Read from
    the leg's evidence only."""
    out: list[str] = []
    bites = _mapping(evidence.get(C.OOS_EVIDENCE_BITES_BLOCK))
    lp = _mapping(bites.get(C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS))
    n_conc, n_min = lp.get("n_conclusive"), lp.get("min_conclusive_probes")
    ok_int = lambda v: isinstance(v, int) and not isinstance(v, bool) and v > 0     # noqa: E731
    if not lp:
        out.append(f"served-steer class: no {C.OOS_EVIDENCE_BITES_BLOCK}.{C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS} block — the liveness statistic that "
                   "selects this class is not on the record")
    elif not (ok_int(n_conc) and ok_int(n_min) and n_conc >= n_min):
        out.append(f"served-steer class: {C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS} n_conclusive {n_conc!r} < min_conclusive_probes {n_min!r} — "
                   "the class requires a CONCLUSIVE logprob liveness verdict")
    bite = evidence.get(M.EVIDENCE_LENS_ON_BITE)
    if cert_class == C.CERT_CLASS_NO_SWITCH:
        return out            # (C″): the explicit inapplicability record is the no-switch class's own requirement
    if not isinstance(bite, Mapping):
        out.append(f"served-steer class: no {M.EVIDENCE_LENS_ON_BITE} record")
    else:
        admitted_by = bite.get(C.LENS_ON_BITE_LIVENESS_EVIDENCE_KEY)
        if admitted_by is None:
            if bite.get("differs_from_base") is not True:
                out.append(f"served-steer class: {M.EVIDENCE_LENS_ON_BITE}.differs_from_base is {bite.get('differs_from_base')!r} and the record "
                           f"carries no {C.LENS_ON_BITE_LIVENESS_EVIDENCE_KEY} — neither a token bite nor the logprob-admitted record")
        elif admitted_by != C.LENS_ON_BITE_EVIDENCE_LOGPROB:
            out.append(f"served-steer class: {M.EVIDENCE_LENS_ON_BITE}.{C.LENS_ON_BITE_LIVENESS_EVIDENCE_KEY} {admitted_by!r} is not "
                       f"{C.LENS_ON_BITE_EVIDENCE_LOGPROB!r} (the contract's one logprob-admitted value)")
    return out


def row_thinking(row: Mapping[str, Any]) -> str:
    """The row's ARM: its ``thinking_mode`` (a schema-2 row) or ``on`` (a schema-1
    row of the published matrix) — lane B's ``g8k_cells.row_thinking`` rule."""
    return str(row.get(MATRIX_ROW_FIELD_THINKING_MODE) or THINKING_ARM_ON)


def thinking_label(arm: str) -> str:
    """The figure / table label of an arm (``THINKING_LABEL_BY_ARM``); an arm outside the two refuses."""
    if arm not in THINKING_LABEL_BY_ARM:
        raise AnalysisError(f"thinking arm {arm!r} is not one of {THINKING_ARMS}")
    return THINKING_LABEL_BY_ARM[arm]


def is_itdoc_row(row: Mapping[str, Any]) -> bool:
    """An it-doc re-fit row (E3): ``object_source: itdoc`` — its pin is the admission record's, its identity the row's."""
    return row.get("object_source") == OBJECT_SOURCE_ITDOC


def itdoc_fields(row: Mapping[str, Any] | None) -> dict[str, Any]:
    """The it-doc carriage of a row (E3): ``{lane, qualification, qualification_pattern}`` read VERBATIM off the row when it is an
    it-doc row (``None`` for a field the row does not carry), every field ``None`` on a row of any other source (or no row) — so
    every reading a consumer files carries the same three keys and the qualification is never retyped by this reader."""
    if row is None or not is_itdoc_row(row):
        return {f: None for f in ITDOC_ROW_FIELDS}
    return {f: (str(row[f]) if isinstance(row.get(f), str) and str(row[f]).strip() else None) for f in ITDOC_ROW_FIELDS}


def qualification_text(row: Mapping[str, Any] | None) -> str | None:
    """The sentence(s) a caption / finding carries for an it-doc reading: the row's ``qualification`` followed by its
    ``qualification_pattern`` (one space between), whichever the row carries; ``None`` when neither (a non-it-doc row)."""
    f = itdoc_fields(row)
    parts = [f[MATRIX_ROW_FIELD_QUALIFICATION], f[MATRIX_ROW_FIELD_QUALIFICATION_PATTERN]]
    text = " ".join(x for x in parts if x)
    return text or None


def row_problem(row: Mapping[str, Any], i: int) -> str | None:
    """Why one matrix row is not a row this reader can hold a store to. The
    row's ARM (:func:`row_thinking`) selects the class: under ``on`` every rule
    is the published matrix's; under ``not_available`` (llama-paper-rows P6 /
    P7) a fitted row carries NO tuple (there is no out-of-support tuple for a
    family with no thinking switch), is never declared, decodes under the
    model's ``card`` row, and its stamps are the ``unspecified`` render's."""
    missing = [f for f in MATRIX_ROW_FIELDS if f not in row]
    if missing:
        return f"matrix row {i}: missing field(s) {missing}"
    if MATRIX_ROW_FIELD_TRIGGER_THRESHOLD in row:
        t = row[MATRIX_ROW_FIELD_TRIGGER_THRESHOLD]
        if not (isinstance(t, (int, float)) and not isinstance(t, bool) and 0.0 < float(t) < 1.0):
            return f"matrix row {i}: {MATRIX_ROW_FIELD_TRIGGER_THRESHOLD} {t!r} is not a share in (0, 1)"
    key, view, cell = str(row["model_key"]), str(row["view"]), str(row["cell"])
    arm = row_thinking(row)
    if arm not in THINKING_ARMS:
        return f"matrix row {i}: {MATRIX_ROW_FIELD_THINKING_MODE} {row.get(MATRIX_ROW_FIELD_THINKING_MODE)!r} is not one of {THINKING_ARMS}"
    no_switch = arm == THINKING_ARM_NOT_AVAILABLE
    status = str(row["status"])
    if status not in STATUSES:
        return f"matrix row {i}: status {row['status']!r} is not one of {STATUSES}"
    if status == STATUS_HELD and not row.get("held_reason"):
        return f"matrix row {i}: a held row carries no held_reason"
    if status in EVIDENCE_STATUSES and row.get("held_reason"):
        return f"matrix row {i}: a {status} row carries a held_reason {row['held_reason']!r}"
    wave = row.get("wave")
    if wave is not None and (not isinstance(wave, str) or not wave.strip()):
        return f"matrix row {i}: wave {wave!r} is not a non-empty name"
    if str(row["tag"]) != tag_of(key, view):
        return f"matrix row {i}: tag {row['tag']!r} != the established spelling {tag_of(key, view)!r}"
    # the arm is a FAMILY fact (P7): the no-switch arm iff the family has no thinking switch — before any per-arm rule
    if not isinstance(row.get("family"), str) or not row["family"]:
        return f"matrix row {i}: no family (the render fact and the split mode are keyed on it)"
    if no_switch != (str(row["family"]) in C.NO_THINKING_CHANNEL_FAMILIES):
        return (f"matrix row {i}: {MATRIX_ROW_FIELD_THINKING_MODE} {arm!r} on family {row['family']!r} — the no-switch arm is for a "
                f"family in {list(C.NO_THINKING_CHANNEL_FAMILIES)} and only there (P7)")
    want_mode = SAMPLING_MODE_BY_ARM[arm]
    if str(row["sampling_mode"]) != want_mode:
        return (f"matrix row {i}: sampling_mode {row['sampling_mode']!r} != {want_mode!r} "
                + ("(the thinking-card row)" if not no_switch else f"(the shipped card row of a {arm} family; P6)"))
    if key not in C.SAMPLING_TABLE_BY_MODE.get(want_mode, {}):
        return f"matrix row {i}: model {key!r} has no {want_mode!r} sampling row in the contract's table (contracts.SAMPLING_TABLE_BY_MODE)"
    if str(row["cell_slug"]) != M.cell_slug(cell):
        return f"matrix row {i}: cell_slug {row['cell_slug']!r} != {M.cell_slug(cell)!r}"
    fitted = bool(row["fitted"])
    declared = bool(row["declared"])
    if no_switch and declared:
        return f"matrix row {i}: a {arm} row cannot be declared — a no-switch family has no thinking-ON render to declare (P7)"
    if status not in EVIDENCE_STATUSES:
        # a DESIGN-ONLY row (held / deferred) is never collected, so its store-side contract is not owed yet (the
        # provisional `arditi` spelling is not a cell the grammar admits, and its stub carries no pin / tuple)
        problem = out_root_problem(row, i)
        if problem:
            return problem
        if not isinstance(row.get("family"), str) or not row["family"]:
            return f"matrix row {i}: no family (the render fact and the split mode are keyed on it)"
        return None
    try:
        is_lens = C.is_lens_cell(cell)
        C.cell_spellings(cell, declared=declared)
    except C.CellNameError as e:
        # an EVIDENCE row needs a cell the grammar admits (`alphasteer` is; the provisional `arditi` spelling is a
        # design-only arm — a held / deferred row — until the arditi cell id lands)
        return f"matrix row {i}: {cell!r} is not a cell this tree's grammar admits ({e}); it may be a held / deferred row, never {status!r}"
    if fitted != is_lens:
        return f"matrix row {i}: fitted={fitted} disagrees with the cell grammar for {cell!r}"
    if fitted:
        pin = row.get("object_sha256")
        if not isinstance(pin, str) or not _HEX64.match(pin):
            return f"matrix row {i}: a fitted row needs a 64-hex object_sha256, got {pin!r}"
        if not row.get("staged_id") or (not no_switch and not row.get("tuple_name")):
            return f"matrix row {i}: a fitted row needs staged_id and tuple_name"
        if is_itdoc_row(row):
            # an it-doc EVIDENCE row — the batch-64 instruct-direct re-fit, identity BY THE ROW'S PIN (the admission record's,
            # held to 64-hex above; the full22 pin file stages none of these): fitted (above), undeclared, all-position, keyed to
            # the record's LANE (a cell id is not unique across lanes: the base lane, the superseded direct-fit lane and this one
            # share it), and the model's ALL-POSITION tuple (checked below) — never the steering `_steer` tuple
            if declared or row.get("served_positions_expected") != POSITIONS_ALL:
                return (f"matrix row {i}: an it-doc row ({OBJECT_SOURCE_ITDOC}) is a fitted, undeclared, all-position row — got declared="
                        f"{declared}, served_positions_expected={row.get('served_positions_expected')!r}")
            lane = row.get(MATRIX_ROW_FIELD_LANE)
            if not isinstance(lane, str) or not lane.strip():
                return f"matrix row {i}: an it-doc row names its {MATRIX_ROW_FIELD_LANE} (the admission record's), got {lane!r}"
        if no_switch:
            if row.get("tuple_name") is not None or row.get("tuple_names"):
                return (f"matrix row {i}: a fitted {arm} row carries tuple_name {row.get('tuple_name')!r} / tuple_names "
                        f"{row.get('tuple_names')!r} — there is no out-of-support tuple for a family with no thinking switch (P7)")
        else:
            if not str(row["tuple_name"]).endswith(TUPLE_NAME_SUFFIX_BASE):
                return f"matrix row {i}: tuple_name {row['tuple_name']!r} does not end with {TUPLE_NAME_SUFFIX_BASE!r} (the 8,192 tuple; plan §3.2)"
            # an EXTERNAL-object row's tuple is PER CELL: the rung's `_steer_<cap>` one for the rung's two arms, the
            # all-positions arm's own `_steer_allpos_<cap>` one for that arm (expected_tuple_name dispatches); every other
            # fitted row's the model's all-position / declared tuple (contracts.g8k_tuple_name). THE `cell` MUST BE PASSED —
            # this call once built a dict without it, so the per-cell branch could never fire and the all-positions row was
            # held to the rung's tuple; a projection that drops the field the callee dispatches on always reads as the default.
            probe = {"model_key": key, "declared": declared, "object_source": row.get("object_source"), "cell": row.get("cell")}
            want_tuple = expected_tuple_name(probe, CAP_BASE)
            if str(row["tuple_name"]) != want_tuple:
                source = ("g8k_tuple_name" if row.get("object_source") != OBJECT_SOURCE_EXTERNAL
                          else "g8k_allpos_tuple_name" if str(row.get("cell")) == M.CELL_ARDITI_ALLPOS else "g8k_steer_tuple_name")
                return (f"matrix row {i}: tuple_name {row['tuple_name']!r} != the contract's {want_tuple!r} (contracts.{source})"
                        + (" — an it-doc row runs the model's ALL-POSITION tuple, never the W4 `_steer` tuple" if is_itdoc_row(row) else ""))
        if row.get("served_positions_expected") not in SERVED_POSITIONS:
            return (f"matrix row {i}: served_positions_expected {row.get('served_positions_expected')!r} is not one of "
                    f"{SERVED_POSITIONS}")
        if declared != (row["served_positions_expected"] == POSITIONS_RESPONSE):
            return (f"matrix row {i}: declared={declared} but served_positions_expected={row['served_positions_expected']!r} — "
                    "a header-located (response) object is declared, an all-position one is not (plan §3.3)")
        if str(row["harness_cell"]) != C.cell_spellings(cell, declared=False).harness:
            return f"matrix row {i}: harness_cell {row['harness_cell']!r} != {C.cell_spellings(cell, declared=False).harness!r}"
    else:
        if declared:
            return f"matrix row {i}: a non-fitted cell cannot be declared"
        if row.get("object_sha256") or row.get("staged_id") or row.get("tuple_name") or row.get("served_positions_expected"):
            return f"matrix row {i}: a non-fitted row carries an object pin / staged id / tuple / served positions"
    problem = out_root_problem(row, i)
    if problem:
        return problem
    if not isinstance(row.get("family"), str) or not row["family"]:
        return f"matrix row {i}: no family (the render fact and the split mode are keyed on it)"
    stamps = row.get("expected_stamps")
    if stamps is not None and tuple(str(s) for s in stamps) != expected_stamps_for(fitted=fitted, declared=declared, arm=arm):
        return (f"matrix row {i}: expected_stamps {list(stamps)!r} != the contract's triple for this class "
                f"{list(expected_stamps_for(fitted=fitted, declared=declared, arm=arm))!r}")
    return None


def matrix_schema_problem(doc: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str | None:
    """Why a matrix DOCUMENT is not consistent with its own schema (llama-paper-rows
    P16): a schema-1 document (the published matrix; a document naming no
    schema reads as schema 1) carries NO ``thinking_mode`` on any row and no
    ``wave_root_rel``; a schema-2 document names its ``wave_root_rel`` and every
    row carries a ``thinking_mode`` in :data:`THINKING_ARMS`; any other schema
    refuses. A bare LIST of rows (an in-memory matrix) has no schema to be held
    to and is never passed here."""
    schema = doc.get("schema", MATRIX_SCHEMA_1)
    if schema not in MATRIX_SCHEMAS:
        return f"matrix schema {schema!r} is not one of {MATRIX_SCHEMAS}"
    if schema == MATRIX_SCHEMA_1:
        if "wave_root_rel" in doc:
            return f"a {MATRIX_SCHEMA_1} matrix carries `wave_root_rel` ({doc.get('wave_root_rel')!r}) — a schema-2 field"
        with_mode = [i for i, r in enumerate(rows) if MATRIX_ROW_FIELD_THINKING_MODE in r]
        if with_mode:
            return (f"a {MATRIX_SCHEMA_1} matrix carries `{MATRIX_ROW_FIELD_THINKING_MODE}` on {len(with_mode)} row(s) (first row "
                    f"{with_mode[0]}: {rows[with_mode[0]].get(MATRIX_ROW_FIELD_THINKING_MODE)!r}) — a schema-2 field")
        return None
    if not isinstance(doc.get("wave_root_rel"), str) or not doc.get("wave_root_rel"):
        return f"a {MATRIX_SCHEMA_2} matrix names its wave root (`wave_root_rel`); got {doc.get('wave_root_rel')!r}"
    without = [i for i, r in enumerate(rows) if r.get(MATRIX_ROW_FIELD_THINKING_MODE) not in THINKING_ARMS]
    if without:
        return (f"a {MATRIX_SCHEMA_2} matrix carries `{MATRIX_ROW_FIELD_THINKING_MODE}` in {THINKING_ARMS} on every row; row "
                f"{without[0]} carries {rows[without[0]].get(MATRIX_ROW_FIELD_THINKING_MODE)!r}")
    return None


def out_root_prefix(out_root: str, row: Mapping[str, Any]) -> str | None:
    """The prefix segment of a row's out root — ``controls`` (the published layout) or a wave-isolated
    ``controls_<wave>`` — when the root ends with ``<prefix>/<TAG>/<cell slug>/<mode>``; ``None`` for any other layout."""
    parts = str(out_root).rstrip("/").split("/")
    want = [str(row["tag"]), str(row["cell_slug"]), SAMPLING_MODE_BY_ARM[row_thinking(row)]]   # the row's ARM picks the mode segment (`card` under not_available)
    if len(parts) < 4 or parts[-3:] != want:
        return None
    prefix = parts[-4]
    if prefix != OUT_ROOT_PREFIX and not prefix.startswith(OUT_ROOT_PREFIX + "_"):
        return None
    return prefix


def out_root_tail(row: Mapping[str, Any]) -> str:
    """The reader's relative out root of a row: its own prefix (``controls`` or ``controls_<wave>``) + the analysis' layout."""
    prefix = out_root_prefix(str(row["out_root"]), row) or OUT_ROOT_PREFIX
    return f"{prefix}/{row['tag']}/{row['cell_slug']}/{SAMPLING_MODE_BY_ARM[row_thinking(row)]}"


def out_root_problem(row: Mapping[str, Any], i: int) -> str | None:
    """Why a row's ``out_root`` / ``rung_out_root`` are not the analysis' layout (a wave-isolated prefix admitted)."""
    tail = f"{OUT_ROOT_PREFIX}/{row['tag']}/{row['cell_slug']}/{SAMPLING_MODE_BY_ARM[row_thinking(row)]}"
    out_root = str(row["out_root"]).rstrip("/")
    if out_root_prefix(out_root, row) is None:
        return (f"matrix row {i}: out_root {out_root!r} does not end with the analysis' layout {tail!r} (or a wave-isolated "
                f"{OUT_ROOT_PREFIX}_<wave>/... of it)")
    rung_root = str(row["rung_out_root"]).rstrip("/")
    if not rung_root.endswith(out_root_tail(row) + RUNG_OUT_ROOT_SUFFIX):
        return f"matrix row {i}: rung_out_root {rung_root!r} is not out_root + {RUNG_OUT_ROOT_SUFFIX!r}"
    return None


#: a matrix row's wave — its `wave` field, else `published` (the frozen matrix carries none): the MATH-500 module's reader
row_wave = M.row_wave


def required_rows(rows: Sequence[Mapping[str, Any]], waves: Sequence[str] | None) -> list[dict[str, Any]]:
    """The REQUIRED-EVIDENCE set (plan P5): the EVIDENCE rows (``launch`` / ``complete``) whose wave is among the
    analysed ``waves`` (``None`` = every evidence row, the published behaviour). Every row of it is mandatory — a missing
    leg fails the completeness gate; ``held`` / ``deferred`` rows and evidence rows of other waves are design statuses."""
    return [dict(r) for r in M.required_rows(rows, waves)]       # every GSM8K row carries `status`; copies, never the matrix's dicts


def load_matrix(source: Path | str | Sequence[Any] | Mapping[str, Any]) -> dict[str, Any]:
    """Lane B's frozen matrix (``configs/sdfbase/gsm8k/matrix.json``, the
    schema-2 ``matrix_llama.json``, or an in-memory list of rows): every row
    validated (:func:`row_problem`), a document held to its schema
    (:func:`matrix_schema_problem`), a (model, view, cell) named twice refused,
    ``out_root`` / ``rung_out_root`` reduced to the analysis' relative layout,
    every row's ARM made explicit (``thinking_mode``: a schema-1 row's ``on``).
    Returns ``{rows, launch, held, sha256, counts, schema, wave_root_rel}`` —
    the DESIGN SET is ``launch``; ``held`` rows are filed by
    name and are never legs; ``wave_root_rel`` is None on a schema-1 document
    and on a bare list (the published root, by default)."""
    if isinstance(source, (str, Path)):
        p = Path(source).expanduser()
        if not p.is_file():
            raise AnalysisError(f"matrix {p}: no such file")
        blob = p.read_bytes()
        doc = json.loads(blob.decode("utf-8"))
        sha = hashlib.sha256(blob).hexdigest()
        path = str(p)
    else:
        # an in-memory matrix: a bare list of rows, or a DOCUMENT ({schema, wave_root_rel, rows}) held to its schema like a file
        doc = dict(source) if isinstance(source, Mapping) else list(source)
        sha = hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        path = None
    rows_in = doc.get("rows") if isinstance(doc, Mapping) else doc
    if not isinstance(rows_in, Sequence) or isinstance(rows_in, (str, bytes)):
        raise AnalysisError(f"matrix: expected a list of rows (or {{rows: [...]}}), got {type(rows_in).__name__}")
    rows_d = [M._row_dict(r) for r in rows_in]
    schema: str | None = None
    wave_root_rel: str | None = None
    if isinstance(doc, Mapping):
        problem = matrix_schema_problem(doc, rows_d)
        if problem:
            raise AnalysisError(f"matrix: {problem}")
        schema = str(doc.get("schema", MATRIX_SCHEMA_1))
        wave_root_rel = doc.get("wave_root_rel") if schema == MATRIX_SCHEMA_2 else None
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for i, row in enumerate(rows_d):
        problem = row_problem(row, i)
        if problem:
            raise AnalysisError(problem)
        k = (str(row["model_key"]), str(row["view"]), str(row["cell"]))
        if k in seen:
            raise AnalysisError(f"matrix names {cell_key(*k)} twice")
        seen.add(k)
        row["fitted"], row["declared"] = bool(row["fitted"]), bool(row["declared"])
        arm = row_thinking(row)
        row[MATRIX_ROW_FIELD_THINKING_MODE] = arm
        # the row's OWN prefix is kept (`controls/` for the published rows — byte for byte the frozen layout — or a
        # wave-isolated `controls_<wave>/`, plan P2 / F11a); the rung root is that tail + the rung suffix
        tail = out_root_tail(row)
        row["out_root"], row["rung_out_root"] = tail, tail + RUNG_OUT_ROOT_SUFFIX
        row["expected_stamps"] = list(expected_stamps_for(fitted=row["fitted"], declared=row["declared"], arm=arm))
        row.setdefault("rung_subset", RUNG_SUBSET_REL.format(tag=row["tag"], cell_slug=row["cell_slug"]))
        rows.append(row)
    # `launch` is the EVIDENCE set of every wave (launch + complete rows) — the analysis narrows it to the analysed waves
    # (:func:`required_rows`); `held` / `deferred` rows are filed by name and are never legs
    launch = [r for r in rows if r["status"] in EVIDENCE_STATUSES]
    held = [r for r in rows if r["status"] == STATUS_HELD]
    deferred = [r for r in rows if r["status"] == STATUS_DEFERRED]
    return {"path": path, "sha256": sha, "rows": rows, "launch": launch, "held": held, "deferred": deferred, "counts": matrix_counts(rows),
            "schema": schema, "wave_root_rel": wave_root_rel}


def thinking_mode_by_model(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """``{model_key: arm}`` over the LAUNCH rows — one arm per model; a model
    whose rows run under two arms refuses by name (one instrument per model)."""
    out: dict[str, set[str]] = {}
    for r in rows:
        if r["status"] != STATUS_LAUNCH:
            continue
        out.setdefault(str(r["model_key"]), set()).add(row_thinking(r))
    mixed = {k: sorted(v) for k, v in out.items() if len(v) > 1}
    if mixed:
        raise AnalysisError(f"the matrix runs {sorted(mixed)} under two thinking arms {mixed} — one arm per model")
    return {k: next(iter(v)) for k, v in out.items()}


def matrix_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """The acceptance numbers, DERIVED from the matrix."""
    launch = [r for r in rows if r["status"] in EVIDENCE_STATUSES]
    views = {(r["model_key"], r["view"]) for r in launch}
    return {"rows": len(rows), "held": sum(1 for r in rows if r["status"] == STATUS_HELD), "launched": len(launch),
            "fitted_launched": sum(1 for r in launch if r["fitted"]),
            "in_support": sum(1 for r in launch if not r["fitted"]),
            "declared": sum(1 for r in launch if r["declared"]),
            "views": len(views),
            "contrasts_designed": sum(1 for r in launch if r["cell"] != CELL_NONE),
            "expected_refusals": sum(1 for r in launch if r.get("expected_refusal"))}


def matrix_pins_problem(rows: Sequence[Mapping[str, Any]], pins: Mapping[str, Any]) -> str | None:
    """Every fitted LAUNCH row's ``object_sha256`` equals the pin file's entry for
    its ``(model_key, staged_id)`` — one pin per (view, cell); a row
    whose staged id the pin file does not carry refuses (a served object is
    always pinned before its bank view builds)."""
    for r in rows:
        if r["status"] not in EVIDENCE_STATUSES or not r["fitted"]:
            continue                # a held / deferred design row owes no pin yet (the stubs carry `object_sha256: null`)
        if r.get("object_source") == OBJECT_SOURCE_EXTERNAL:
            # an EXTERNAL-object row: the pin is the ADMISSION RECORD's object sha256 and rides the row
            # (`row_problem` holds it to 64-hex); the full22 pin file stages neither bank, by construction — the
            # launcher holds the served bytes to the row's pin at three doors (the bank check, the standup, the driver)
            continue
        if is_itdoc_row(r):
            # an IT-DOC row — the pin is the ROW's (the it-doc ADMISSION RECORD's entry for (lane, model key, view, cell id),
            # `row_problem` holds it to 64-hex), NEVER the full22 pin file's: that file's entry for the cell's staged id names
            # ANOTHER lane's object (the superseded direct-fit `itdoc__…` parent, or a base-lane cell of the same id), so a
            # lookup there would either miss the `itdoc64__` id or compare against the wrong lane's bytes; the launcher
            # holds the served bytes to the row's pin at four doors (the bank binding, the bank view's byte hash, the standup,
            # the driver + provenance mirror)
            continue
        entry = _mapping(pins.get(str(r["model_key"]))).get(str(r["staged_id"]))
        sha = entry.get("object_sha256") if isinstance(entry, Mapping) else entry
        if sha is None:
            return f"{cell_key(r['model_key'], r['view'], r['cell'])}: staged id {r['staged_id']!r} has no pin in the pin file"
        if sha != r["object_sha256"]:
            return (f"{cell_key(r['model_key'], r['view'], r['cell'])}: matrix object_sha256 {r['object_sha256']} != the pin "
                    f"file's {sha} for {r['staged_id']!r}")
    return None


# --------------------------------------------------------------------------- #
# the producer's runs and the accepted attempt                                #
# --------------------------------------------------------------------------- #


def producer_problem(manifest: Mapping[str, Any], *, pin: str, also: Sequence[str] = ()) -> str | None:
    """Why the store's producer is not the pinned tree: the ``producer`` block
    (a 40-hex tree sha carrying the pin — or, plan P5, one of the DECLARED
    trees ``also`` — the served driver) and the
    ``resume_attempts`` record — one entry per run, numbered from 1, EVERY run
    from the same tree, the last run completed, each entry carrying the MATH
    driver's four fields (a superset admitted: lane H's attempt-scoped entries
    add ``attempt_id`` / ``kind`` / ...). When the manifest names an
    ``accepted_generation_attempt``, that id must be a recorded GENERATION
    attempt and every entry must carry an id and a kind."""
    block = _mapping(manifest.get(C.MANIFEST_KEY_PRODUCER))
    if not block:
        return f"the manifest carries no {C.MANIFEST_KEY_PRODUCER} block — the tree that produced the store is unknown"
    sha = block.get("tree_sha")
    if not isinstance(sha, str) or not M._HEX40.match(sha):
        return f"manifest {C.MANIFEST_KEY_PRODUCER}.tree_sha {sha!r} is not a 40-hex commit"
    if block.get("driver") != C.PRODUCER_DRIVER_SERVED:
        return f"manifest {C.MANIFEST_KEY_PRODUCER}.driver {block.get('driver')!r} != {C.PRODUCER_DRIVER_SERVED!r}"
    if not sha.startswith(pin) and sha not in also:
        declared = f" and is not among the declared trees {list(also)}" if also else ""
        return f"manifest {C.MANIFEST_KEY_PRODUCER}.tree_sha {sha} does not carry the wave pin {pin}{declared} — a store from another tree"
    attempts = manifest.get(C.MANIFEST_KEY_RESUME_ATTEMPTS)
    if not isinstance(attempts, list) or not attempts:
        return f"the manifest carries no {C.MANIFEST_KEY_RESUME_ATTEMPTS} record — the runs that produced it cannot be attributed"
    accepted = manifest.get(MANIFEST_KEY_ACCEPTED_ATTEMPT)
    for i, a in enumerate(attempts):
        if not isinstance(a, Mapping) or not set(RESUME_ATTEMPT_REQUIRED_FIELDS) <= set(a):
            return f"{C.MANIFEST_KEY_RESUME_ATTEMPTS}[{i}] {a!r} lacks the driver's {RESUME_ATTEMPT_REQUIRED_FIELDS} fields"
        if a.get("attempt") != i + 1:
            return f"{C.MANIFEST_KEY_RESUME_ATTEMPTS}[{i}].attempt {a.get('attempt')!r} != {i + 1} — the runs are not numbered from 1"
        if a.get("tree_sha") != sha:
            return (f"{C.MANIFEST_KEY_RESUME_ATTEMPTS}[{i}].tree_sha {a.get('tree_sha')!r} != the manifest's producer {sha} — a run of "
                    "this store from another tree (a mixed-producer leg)")
        if not isinstance(a.get("completed"), bool):
            return f"{C.MANIFEST_KEY_RESUME_ATTEMPTS}[{i}].completed {a.get('completed')!r} is not the driver's boolean"
        if accepted is not None:
            if not isinstance(a.get(RESUME_ATTEMPT_ID_FIELD), str) or not a.get(RESUME_ATTEMPT_ID_FIELD):
                return f"{C.MANIFEST_KEY_RESUME_ATTEMPTS}[{i}] carries no {RESUME_ATTEMPT_ID_FIELD} (plan §4.5)"
            if a.get(RESUME_ATTEMPT_KIND_FIELD) not in ATTEMPT_KINDS:
                return f"{C.MANIFEST_KEY_RESUME_ATTEMPTS}[{i}].{RESUME_ATTEMPT_KIND_FIELD} {a.get(RESUME_ATTEMPT_KIND_FIELD)!r} not in {ATTEMPT_KINDS}"
    if attempts[-1].get("completed") is not True:
        return (f"{C.MANIFEST_KEY_RESUME_ATTEMPTS}[{len(attempts) - 1}] did not complete — the store's last run never passed the "
                "producer's gates")
    if accepted is None:
        return f"the manifest names no {MANIFEST_KEY_ACCEPTED_ATTEMPT} — the attempt whose calls count cannot be read (plan §4.5)"
    gen = {a[RESUME_ATTEMPT_ID_FIELD] for a in attempts if a.get(RESUME_ATTEMPT_KIND_FIELD) == ATTEMPT_KIND_GENERATION}
    if accepted not in gen:
        return f"{MANIFEST_KEY_ACCEPTED_ATTEMPT} {accepted!r} is not a recorded generation attempt {sorted(gen)}"
    return None


def recorded_attempts(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """``{attempt_id: entry}`` over the manifest's ``resume_attempts``."""
    out: dict[str, dict[str, Any]] = {}
    for a in manifest.get(C.MANIFEST_KEY_RESUME_ATTEMPTS) or []:
        if isinstance(a, Mapping) and isinstance(a.get(RESUME_ATTEMPT_ID_FIELD), str):
            out[a[RESUME_ATTEMPT_ID_FIELD]] = dict(a)
    return out


def accepted_attempt(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    """The entry of the attempt the manifest names as its accepted generation
    attempt, or ``None`` when it names none or names one it did not record
    (:func:`producer_problem` refuses both; a reader that reached this far
    holds a manifest it already admitted)."""
    accepted = manifest.get(MANIFEST_KEY_ACCEPTED_ATTEMPT)
    return recorded_attempts(manifest).get(accepted) if isinstance(accepted, str) else None


def _utc(stamp: Any) -> datetime | None:
    """An ISO-8601 stamp (``Z`` or an offset) as an aware UTC datetime — ``None``
    for anything else, a NAIVE stamp included: a time with no zone cannot be
    placed against one that has, and guessing a zone is how a window check
    passes something it should not."""
    if not isinstance(stamp, str) or not stamp:
        return None
    try:
        dt = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.astimezone(timezone.utc) if dt.tzinfo is not None else None


def _utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def attempt_window(entry: Mapping[str, Any]) -> tuple[datetime, datetime] | None:
    """A recorded attempt's ``[started, ended]`` as aware UTC datetimes — ``None``
    when either stamp is absent or unplaceable, or the two are out of order."""
    start, end = _utc(entry.get(RESUME_ATTEMPT_STARTED_FIELD)), _utc(entry.get(RESUME_ATTEMPT_ENDED_FIELD))
    return (start, end) if start is not None and end is not None and start <= end else None


def log_created(log: Any) -> datetime | None:
    """A retained ``.eval`` log's ``eval.created`` as an aware UTC datetime (Inspect
    stamps it when the eval starts, inside the run that generated it), or ``None``."""
    return _utc(getattr(getattr(log, "eval", None), "created", None))


def log_outside_attempt(log: Any, entry: Mapping[str, Any]) -> str | None:
    """Why ``log`` is not the accepted generation attempt's OWN log: the attempt
    records no complete window (nothing to bind to — a refusal, never a pass),
    the log records no placeable creation time, or its ``eval.created`` lies
    outside ``[started, ended]``. The record's rows were generated inside that
    window (the producer stamps the attempt around the run and files the rows
    from its log), so a qualifying log created outside it is another run's
    evidence — the other tree's log that a merged pull put beside this record
    on 2026-09-11, which the newest-by-mtime rule then read against it."""
    aid = entry.get(RESUME_ATTEMPT_ID_FIELD)
    window = attempt_window(entry)
    if window is None:
        return (f"the accepted generation attempt {aid!r} records no complete [{RESUME_ATTEMPT_STARTED_FIELD}, {RESUME_ATTEMPT_ENDED_FIELD}] "
                f"window ({entry.get(RESUME_ATTEMPT_STARTED_FIELD)!r}, {entry.get(RESUME_ATTEMPT_ENDED_FIELD)!r}) — the retained log cannot be "
                "bound to the run that filed the record")
    name = Path(str(getattr(log, "location", "") or "?")).name
    created = log_created(log)
    if created is None:
        return (f"{name}: records no placeable creation time (eval.created {getattr(getattr(log, 'eval', None), 'created', None)!r}) — it "
                f"cannot be bound to the accepted generation attempt {aid!r}")
    start, end = window
    if not start <= created <= end:
        return (f"{name}: created {_utc_iso(created)} lies outside the accepted generation attempt {aid!r} window "
                f"[{_utc_iso(start)}, {_utc_iso(end)}] — the record's rows were generated in that window, so this log is another run's evidence")
    return None


def qualifying_logs(log_dir: Path, *, expected_n: int, expected_ids: Sequence[str], expected_max_tokens: int) -> list[Path]:
    """Every retained ``.eval`` under ``log_dir`` that the PRODUCER's own rule
    (``inspect_served.reusable_log``: status success, the attempted count and
    ids, no errored sample, generated at this cap) accepts as the completed log
    of this (task, sample) — each candidate put to the rule ALONE, in name
    order. The producer's selector answers "the newest qualifying log" and
    stops: the right answer for a driver resuming its own store, the wrong
    question for a reader, which needs the COUNT — a dir holding two qualifying
    logs is two generations of one leg, and nothing in the record names the one
    its rows were filed from. Judging each file in a scratch dir holding a link
    to it alone keeps the reader on the producer's predicate without a second
    copy of it (the tree that wrote the stores is pinned; a copy here would
    drift from it silently). The caller refuses on any count but one."""
    from sisteer.capeval.tiers import inspect_served as isv

    out: list[Path] = []
    for p in M._eval_log_entries(log_dir):
        if not p.is_file():
            continue                        # a dangling link or a non-file: the producer's rule passes it over the same way
        with tempfile.TemporaryDirectory(prefix="gsm8k-eval-candidate-") as scratch:
            (Path(scratch) / p.name).symlink_to(p.resolve())
            if isv.reusable_log(Path(scratch), expected_n=expected_n, expected_ids=expected_ids, expected_max_tokens=expected_max_tokens) is not None:
                out.append(p)
    return out


outcomes_producer_problem = M.outcomes_producer_problem


# ---------------------------------------------------------------------------- #
# a CERTIFIED refusal (F12): the driver refused the cell's                      #
# thinking-ON render before generating — a manifest, evidence.json, no summary  #
# ---------------------------------------------------------------------------- #

#: the wrapper's per-store attempt ledger beside the store (``served_thinking.sh`` ``st_attempt_begin`` /
#: ``st_attempt_end``: ``{schema: g8k_ladder_state/1, attempts: [{attempt_id, started, ended, rc, result, sha, ...}],
#: accepted_generation_attempt}``) and its RESULT word for a driver that exited non-zero (``gsm8k_thinking.sbatch``)
LADDER_STATE_FILENAME = "ladder_state.json"
LADDER_RESULT_DRIVER_FAILED = "DRIVER_FAILED"
#: the render-probe draw classes that make a refusal BEHAVIOURAL — the MODEL's per-draw choice under a render whose
#: prompt side certified — rather than a misconfigured serving path: today only the draw that finished inside the
#: prompt-opened block without its close tag (``served_capability_run.PROBE_CLASS_UNCLOSED_BLOCK``, pinned equal by
#: test). A refusal whose draws include ANY other class is not admitted; a further class is added here deliberately,
#: after it is observed, never inferred.
BEHAVIOURAL_REFUSAL_CLASSES: tuple[str, ...] = ("unclosed_block",)
CERTIFYING_DRAW_CLASS_FIELD = "certifying_draw_class"
#: the certification problems a behavioural refusal may name — the think-shape reading and the driver's class gate
#: (``contracts.render_certified`` / ``served_capability_run._render_certification``); a header or regime problem is
#: the serving path, not the model, and blocks
THINK_SHAPE_PROBLEM_PREFIXES: tuple[str, ...] = ("the ON probe completion does not carry the family's think shape",
                                                 "the certifying ON draw's class is")
RENDER_SOURCE_SERVED = "served_tokenize"


def _parse_utc(value: Any):
    """``datetime`` of an ISO-8601 stamp (``Z`` or an offset; a naive stamp reads as UTC), or ``None``."""
    from datetime import UTC, datetime
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def refused_producer_problem(manifest: Mapping[str, Any], *, pin: str, also: Sequence[str] = ()) -> tuple[str | None, dict[str, Any] | None]:
    """The producer / attempt checks of :func:`producer_problem` for a store the
    driver REFUSED before generating: the same pin, driver and per-attempt
    consistency (every run from the producer tree, numbered from 1, ids carried),
    but the LAST recorded run must be INCOMPLETE and no generation attempt may be
    accepted — a summary-less store whose last run completed, or which names an
    accepted attempt, is torn. Returns ``(problem, last_attempt)``."""
    block = _mapping(manifest.get(C.MANIFEST_KEY_PRODUCER))
    if not block:
        return f"the manifest carries no {C.MANIFEST_KEY_PRODUCER} block — the tree that produced the store is unknown", None
    sha = block.get("tree_sha")
    if not isinstance(sha, str) or not M._HEX40.match(sha):
        return f"manifest {C.MANIFEST_KEY_PRODUCER}.tree_sha {sha!r} is not a 40-hex commit", None
    if block.get("driver") != C.PRODUCER_DRIVER_SERVED:
        return f"manifest {C.MANIFEST_KEY_PRODUCER}.driver {block.get('driver')!r} != {C.PRODUCER_DRIVER_SERVED!r}", None
    if not sha.startswith(pin) and sha not in also:
        declared = f" and is not among the declared trees {list(also)}" if also else ""
        return f"manifest {C.MANIFEST_KEY_PRODUCER}.tree_sha {sha} does not carry the wave pin {pin}{declared} — a store from another tree", None
    attempts = manifest.get(C.MANIFEST_KEY_RESUME_ATTEMPTS)
    if not isinstance(attempts, list) or not attempts:
        return f"the manifest carries no {C.MANIFEST_KEY_RESUME_ATTEMPTS} record — the run that refused cannot be attributed", None
    for i, a in enumerate(attempts):
        if not isinstance(a, Mapping) or not set(RESUME_ATTEMPT_REQUIRED_FIELDS) <= set(a):
            return f"{C.MANIFEST_KEY_RESUME_ATTEMPTS}[{i}] {a!r} lacks the driver's {RESUME_ATTEMPT_REQUIRED_FIELDS} fields", None
        if a.get("attempt") != i + 1:
            return f"{C.MANIFEST_KEY_RESUME_ATTEMPTS}[{i}].attempt {a.get('attempt')!r} != {i + 1} — the runs are not numbered from 1", None
        if a.get("tree_sha") != sha:
            return (f"{C.MANIFEST_KEY_RESUME_ATTEMPTS}[{i}].tree_sha {a.get('tree_sha')!r} != the manifest's producer {sha} — a run of "
                    "this store from another tree (a mixed-producer leg)"), None
        if not isinstance(a.get("completed"), bool):
            return f"{C.MANIFEST_KEY_RESUME_ATTEMPTS}[{i}].completed {a.get('completed')!r} is not the driver's boolean", None
        if not isinstance(a.get(RESUME_ATTEMPT_ID_FIELD), str) or not a.get(RESUME_ATTEMPT_ID_FIELD):
            return f"{C.MANIFEST_KEY_RESUME_ATTEMPTS}[{i}] carries no {RESUME_ATTEMPT_ID_FIELD} (plan §4.5)", None
        kind = a.get(RESUME_ATTEMPT_KIND_FIELD)
        if kind is not None and kind not in ATTEMPT_KINDS:
            return f"{C.MANIFEST_KEY_RESUME_ATTEMPTS}[{i}].{RESUME_ATTEMPT_KIND_FIELD} {kind!r} not in {ATTEMPT_KINDS}", None
    last = dict(attempts[-1])
    if last.get("completed") is not False:
        return (f"{C.MANIFEST_KEY_RESUME_ATTEMPTS}[{len(attempts) - 1}] ({last.get(RESUME_ATTEMPT_ID_FIELD)}) COMPLETED — a completed run "
                "writes a summary; a summary-less store whose last run completed is torn, not a refusal"), None
    if manifest.get(MANIFEST_KEY_ACCEPTED_ATTEMPT) is not None:
        return (f"the manifest names {MANIFEST_KEY_ACCEPTED_ATTEMPT} {manifest.get(MANIFEST_KEY_ACCEPTED_ATTEMPT)!r} yet filed no summary — "
                "torn, not a refusal"), None
    return None, last


def refusal_binding_problem(evidence: Mapping[str, Any], *, last_attempt: Mapping[str, Any], ladder_state: Mapping[str, Any] | None,
                            pin: str) -> str | None:
    """Why ``evidence.json`` is NOT the record of the store's LAST attempt — the one
    that refused: the driver writes the manifest's attempt entry
    (``started`` / ``ts``) BEFORE its probes and the evidence after them, so
    the evidence's ``written_at`` must not predate the last attempt's start (an
    older refusal's evidence beside a newer run that died before rewriting it is
    unattributable); and when the wrapper's ``ladder_state.json`` is beside the
    store, its latest attempt must be the manifest's, must have ended
    ``DRIVER_FAILED`` at the wave pin, with the evidence written before that end
    and no generation attempt accepted."""
    written = _parse_utc(evidence.get("written_at"))
    if written is None:
        return "evidence.json carries no readable written_at — it cannot be bound to a run"
    aid = last_attempt.get(RESUME_ATTEMPT_ID_FIELD)
    started = _parse_utc(last_attempt.get("started") or last_attempt.get("ts"))
    if started is None:
        return f"the last recorded attempt {aid!r} carries no readable started / ts — the evidence cannot be bound to it"
    if written < started:
        return (f"evidence.json written_at {evidence.get('written_at')!r} predates the last recorded attempt {aid!r} (started "
                f"{last_attempt.get('started') or last_attempt.get('ts')!r}) — a newer run beside an older refusal's evidence; the refusal "
                "cannot be attributed to the run that ended the store")
    if ladder_state is None:
        return None
    ls = _mapping(ladder_state)
    runs = [a for a in (ls.get("attempts") or []) if isinstance(a, Mapping)]
    if not runs:
        return f"{LADDER_STATE_FILENAME} records no attempt beside the store"
    cur = runs[-1]
    if cur.get(RESUME_ATTEMPT_ID_FIELD) != aid:
        return (f"{LADDER_STATE_FILENAME}'s latest attempt {cur.get(RESUME_ATTEMPT_ID_FIELD)!r} != the manifest's latest {aid!r} — a "
                "newer run the driver never recorded; unresolved")
    if cur.get("result") != LADDER_RESULT_DRIVER_FAILED:
        return f"{LADDER_STATE_FILENAME}'s latest attempt {aid!r} ended {cur.get('result')!r}, not {LADDER_RESULT_DRIVER_FAILED!r}"
    sha = cur.get("sha")
    if isinstance(sha, str) and sha and not sha.startswith(pin):
        return f"{LADDER_STATE_FILENAME}'s latest attempt {aid!r} ran at {sha}, not the wave pin {pin}"
    ended = _parse_utc(cur.get("ended"))
    if ended is not None and written > ended:
        return (f"evidence.json written_at {evidence.get('written_at')!r} is after the wrapper recorded attempt {aid!r} ending "
                f"({cur.get('ended')!r}) — written by another run")
    if ls.get(MANIFEST_KEY_ACCEPTED_ATTEMPT) is not None:
        return f"{LADDER_STATE_FILENAME} names an accepted generation attempt {ls.get(MANIFEST_KEY_ACCEPTED_ATTEMPT)!r} — not a refusal"
    return None


def certified_refusal_problem(evidence: Mapping[str, Any], *, manifest: Mapping[str, Any], row: Mapping[str, Any],
                              fact: Mapping[str, Any]) -> str | None:
    """Why a ``render_certified: false`` evidence record is NOT a BEHAVIOURAL
    refusal the analysis may admit: the non-behavioural predicates
    must hold independently — the served header ids measured live EQUAL the
    family's tuples under both arms (and the record's expected tuples are the
    fact's), both probe regime stamps are their arms', the only problems named
    are the think-shape reading / the class gate, every recorded ON draw is in
    :data:`BEHAVIOURAL_REFUSAL_CLASSES` (none of another class), the certifying
    draw's class is absent or behavioural, and the evidence names THIS cell's
    route (cell, harness id, virtual model = the manifest's) with a steering
    selector iff the row is fitted. Anything else is the serving path, a
    different cell, or an unread shape — blocking, never admitted."""
    rc = _mapping(evidence.get("render_certified"))
    if rc.get("render_certified") is not False:
        return f"render_certified.render_certified is {rc.get('render_certified')!r}, not false"
    if M._header_ids_differ(rc.get("expected_header_ids_by_arm"), fact):
        return f"render_certified.expected_header_ids_by_arm {rc.get('expected_header_ids_by_arm')!r} != the {row['family']!r} fact's tuples"
    if M._header_ids_differ(rc.get("measured_header_ids_by_arm"), fact):
        return (f"the live served header ids {rc.get('measured_header_ids_by_arm')!r} != the {row['family']!r} fact's tuples — the SERVING "
                "PATH failed certification, not the model's draws")
    if rc.get("render_source") != RENDER_SOURCE_SERVED:
        return f"render_certified.render_source {rc.get('render_source')!r} != {RENDER_SOURCE_SERVED!r} — the header ids were not measured live"
    if rc.get("split_mode") != fact.get("split_mode"):
        return f"render_certified.split_mode {rc.get('split_mode')!r} != the fact's {fact.get('split_mode')!r}"
    probe = _mapping(evidence.get("render_probe"))
    regimes = {arm: _mapping(probe.get(arm)).get("render_regime") for arm in (C.THINKING_OFF, C.THINKING_ON)}
    want = {C.THINKING_OFF: C.RENDER_REGIME_THINKING_OFF, C.THINKING_ON: C.RENDER_REGIME_THINKING_ON}
    if regimes != want:
        return f"the probe completions' render_regime stamps {regimes!r} != {want!r} — the render, not the model's draws"
    problems = [str(x) for x in (rc.get("certification_problems") or [])]
    if not problems:
        return "render_certified names no certification problem"
    other = [x for x in problems if not x.startswith(THINK_SHAPE_PROBLEM_PREFIXES)]
    if other:
        return f"certification problem(s) outside the think-shape reading: {other}"
    attempts = [a for a in (probe.get("on_attempts") or []) if isinstance(a, Mapping)]
    if not attempts:
        return "render_probe records no ON draw (on_attempts) — the refusal's draws cannot be read"
    classes = [str(a.get("class")) for a in attempts]
    outside = sorted({c for c in classes if c not in BEHAVIOURAL_REFUSAL_CLASSES})
    if outside:
        return f"ON draw classes {classes} include {outside}, outside the behavioural set {list(BEHAVIOURAL_REFUSAL_CLASSES)}"
    cdc = rc.get(CERTIFYING_DRAW_CLASS_FIELD)
    if cdc is not None and cdc not in BEHAVIOURAL_REFUSAL_CLASSES:
        return f"{CERTIFYING_DRAW_CLASS_FIELD} {cdc!r} is outside the behavioural set {list(BEHAVIOURAL_REFUSAL_CLASSES)}"
    cell, declared, fitted = str(row["cell"]), bool(row["declared"]), bool(row["fitted"])
    spell = C.cell_spellings(cell, declared=declared)
    if evidence.get("cell_id") != cell:
        return f"evidence cell_id {evidence.get('cell_id')!r} != the leg's cell {cell!r}"
    if evidence.get("harness_cell_id") != spell.harness:
        return f"evidence harness_cell_id {evidence.get('harness_cell_id')!r} != {spell.harness!r}"
    route = _mapping(manifest.get(C.MANIFEST_KEY_ROUTE))
    if evidence.get("virtual_model") != spell.vmodel or route.get("virtual_model") != spell.vmodel:
        return (f"evidence virtual_model {evidence.get('virtual_model')!r} / manifest route {route.get('virtual_model')!r} != the cell's "
                f"{'declared' if declared else 'plain'} spelling {spell.vmodel!r}")
    bites = _mapping(evidence.get(C.OOS_EVIDENCE_BITES_BLOCK))
    if fitted and (bites.get(C.OOS_EVIDENCE_BITES_APPLICABLE) is not True or evidence.get("mask") != "generated"):
        return "a fitted row's evidence carries no steering selector (steering_bites not applicable / mask not 'generated')"
    if not fitted and (bites.get(C.OOS_EVIDENCE_BITES_APPLICABLE) is True or evidence.get("mask") is not None):
        return "a bare row's evidence carries a steering selector — not this cell's route"
    if fitted:
        # the steering-bites control runs BEFORE the render probe and is recorded whatever the render did: a fitted run
        # whose control FAILED (an inert route, a probe leg's stamp not its expected row) is an independent serving
        # failure the behavioural refusal must not hide. The later probes ((B'), (C) / (B)) the failed
        # render prevented are NOT required. On a DECLARED row the control is the ON-render, declared proof
        # (`_declared_bites_problems`: the probe rendered thinking-ON with the declaration, every steered reply wearing
        # the declared triple) — the same requirement `certification_check` makes of a produced leg; an OFF-rendered
        # or unstamped record there is another producer's, never a behavioural outcome
        problems = steering_bites_problems(
            bites, serving_identity=M._mapping(manifest.get(C.MANIFEST_KEY_ROUTE)).get("serving_identity"))
        if declared:
            problems += _declared_bites_problems(bites, expected_stamps=row["expected_stamps"])
        if problems:
            return "the steering-bites control the producer ran on this fitted route FAILED — " + "; ".join(problems)
    return None


# --------------------------------------------------------------------------- #
# the standup record of a DECLARED cell's served positions (the record gap)    #
# --------------------------------------------------------------------------- #

#: the standup's per-job record beside the served proxy, as the manifest's own ``routing_table`` path names it on the
#: cluster: ``<wave root>/serving/<jobdir>/gsm8k/routing_table.json``. A pull relocates the wave root; the trailing
#: layout is what identifies the record (the same relocation-aware rule ``same_store_tail`` applies to out roots).
STANDUP_DIR_PARENT = "serving"
STANDUP_TASK_SUBDIR = "gsm8k"
STANDUP_ROUTING_TABLE = "routing_table.json"              # the exporter's stamp of the resolved specs — what endpoint.json copied
STANDUP_JOBS = "standup_jobs.jsonl"                       # job id, node, tree sha, geometry
STANDUP_ATTESTATION = "steering_attestation.json"         # the workers' attestation, bound to the Slurm job id
STANDUP_REGISTRY = "lens_registry/registry.json"          # the resolved spec the workers bound their ops to
STANDUP_RECORD_FILES: tuple[str, ...] = (STANDUP_ROUTING_TABLE, STANDUP_JOBS, STANDUP_ATTESTATION, STANDUP_REGISTRY)
#: the ENGINE-RESOLVED SAMPLER attestation beside the store: the launcher runs
#: scripts/sdfbase/record_resolved_sampler.py after the driver into `<out root>/resolved_sampler.json`
#: (g8k_cells.RESOLVED_SAMPLER_FILENAME, pinned equal); its verdicts are the recorder's own words. A FINDING, never a gate.
RESOLVED_SAMPLER_FILENAME = "resolved_sampler.json"
RESOLVED_SAMPLER_AGREES = "AGREES"
RESOLVED_SAMPLER_VERDICTS: tuple[str, ...] = ("AGREES", "DIVERGED", "UNREADABLE", "UNCOMPARED", "UNCLASSIFIED_REQUESTS",
                                              "FLAG_SET_NO_REQUESTS_YET", "REFUSED")
RESOLVED_SAMPLER_KEYS: tuple[str, ...] = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")
#: the refusals, by name: no record to read / a record that does not bind to this manifest
STANDUP_RECORD_MISSING = "STANDUP_RECORD_MISSING"
STANDUP_RECORD_MISMATCH = "STANDUP_RECORD_MISMATCH"
#: the manifest fields the driver writes the served-positions fact into (served_capability_run.py 4517-4519, 4560-4564)
RECORD_GAP_FIELD_TRANSFER = f"{C.MANIFEST_KEY_TRANSFER_PROVENANCE}.served_positions"
RECORD_GAP_FIELD_ROUTE = f"{C.MANIFEST_KEY_ROUTE}.{C.LENS_SERVED_POSITIONS_KEY}"
RECORD_GAP_FIELD_TOP = C.LENS_SERVED_POSITIONS_KEY
RECORD_GAP_FIELDS: tuple[str, ...] = (RECORD_GAP_FIELD_TRANSFER, RECORD_GAP_FIELD_ROUTE, RECORD_GAP_FIELD_TOP)
#: the producer defect that left the fields null on every DECLARED cell: the driver looked the standup's map up under
#: the cell's BARE harness key while a declared standup keys it by the variant spelling (``<harness>@thinking``); the
#: miss returned None, which ``contracts.header_located(None)`` reads fail-closed as header-located, so the serve
#: proceeded as designed and the None was recorded three times (served-positions-null-diagnosis.md §1)
RECORD_GAP_CAUSE_BARE_KEY_LOOKUP = "PRODUCER_SERVED_POSITIONS_BARE_KEY_LOOKUP served_capability_run.py:4007-4008 @8f5f1495"
#: the NATIVE standup's record beside its job (the same ``serving/<jobdir>/gsm8k/`` layout): the exporter's routing table,
#: ``standup_jobs.jsonl`` and the SERVING attestation (``serving_attestation.SERVING_ATTESTATION_KIND``) — a native standup
#: writes no lens registry and no steering attestation, so the declared record's file set is not its shape
STANDUP_SERVING_ATTESTATION = "serving_attestation.json"
NATIVE_STANDUP_RECORD_FILES: tuple[str, ...] = (STANDUP_ROUTING_TABLE, STANDUP_JOBS, STANDUP_SERVING_ATTESTATION)
#: the ENGINE-NEUTRAL served-positions map a standup of EITHER engine stamps into its routing table (``{harness cell id:
#: <positions>}``; the native-route producer fix — new records), read FIRST; and the routes[] attestation of a native
#: route's position mode (``proxy.RouteEntry.petri_position_mode``, off the adapter's residual-slot digest), the ONE
#: served-positions record the existing native cards' tables carry (their lens map is ``{}``), read when the neutral map
#: does not name the cell. Neither is ever defaulted: a route attesting no mode under either key refuses by name.
SERVED_POSITIONS_KEY = C.SERVED_POSITIONS_KEY
ROUTE_POSITION_MODE_KEY = "petri_position_mode"
NATIVE_RESOLVED_FROM_NEUTRAL_MAP, NATIVE_RESOLVED_FROM_ROUTES = "served_positions map", "routes[].petri_position_mode"
#: the refusal, by name, of a native route whose record attests NO position mode under either key
NATIVE_POSITION_MODE_UNATTESTED = "NATIVE_POSITION_MODE_UNATTESTED"
#: ... and of a RESOLVED mode the contract cannot classify (a native-vocabulary mode outside the lens one, which
#: ``contracts.header_located`` would read as header-located, fail closed — a wrong positive): the contract's own name,
#: shared with the served driver's read site (``contracts.unclassified_native_modes``)
NATIVE_POSITION_MODE_UNCLASSIFIED = C.NATIVE_POSITION_MODE_UNCLASSIFIED
#: the two transfer-provenance fields the producer DERIVED from the null: ``contracts.transfer_provenance_block`` computes
#: ``header_located`` / ``header_check_class`` through ``header_located(served_positions)``, and a None reads header-located,
#: fail closed — so the null propagated into two more fields, each filed as a gap when the record re-derives it
RECORD_GAP_FIELD_TRANSFER_HEADER_LOCATED = f"{C.MANIFEST_KEY_TRANSFER_PROVENANCE}.header_located"
RECORD_GAP_FIELD_TRANSFER_HEADER_CHECK_CLASS = f"{C.MANIFEST_KEY_TRANSFER_PROVENANCE}.header_check_class"
NATIVE_RECORD_GAP_FIELDS: tuple[str, ...] = (RECORD_GAP_FIELD_TRANSFER, RECORD_GAP_FIELD_TRANSFER_HEADER_LOCATED,
                                             RECORD_GAP_FIELD_TRANSFER_HEADER_CHECK_CLASS)
#: the producer defect on every NATIVELY served fitted cell: the driver reads the served positions ONLY off the standup's
#: LENS map (``_standup_served_positions`` over ``lens_served_positions``), which a native standup writes EMPTY (its
#: routes[] attest the mode instead), so the None stands (5505, 5591); the driver then sets the top-level ``header_located``
#: to the engine's False (5601) but ``transfer_provenance_block`` (6323) re-derives ``header_located`` / ``header_check_class``
#: from the None, fail closed — a MISSING FACT read fail-closed, never a render certification (the real cards' tree
#: b367a35d; the lines are identical at 9c3d853b)
RECORD_GAP_CAUSE_NATIVE_LENS_MAP_ONLY = "PRODUCER_SERVED_POSITIONS_NATIVE_LENS_MAP_ONLY served_capability_run.py:5505,5591-5601,6323 @b367a35d"
#: the leg / report key the gaps are filed under
RECORD_GAPS_KEY = "record_gaps"
#: the attestation's content identity of the registry the workers installed: ``sha256(registry.json bytes)[:16]`` —
#: the PRODUCER's own function (``vllm_server._attestation_bindings``: ``cm.sha256_hex16(index.read_bytes())``), by import
ATTESTATION_KEY_REGISTRY_INDEX_SHA16 = "registry_index_sha16"
registry_index_sha16 = CM.sha256_hex16
#: predicate (B)'s record on a DECLARED header-located cell (the driver's ``thinking_header_by_arm`` block once
#: ``_certify_header`` merged the certification into it) and the certification flags it must carry
EVIDENCE_THINKING_HEADER_BY_ARM = "thinking_header_by_arm"
THINKING_HEADER_CERTIFIED_FLAGS: tuple[str, ...] = (C.OOS_EVIDENCE_HEADER_CERTIFIED, "header_attested_agrees",
                                                     C.OOS_EVIDENCE_PROMPT_OPENED_THINK_CERTIFIED)
_JOB_ID_SUFFIX = re.compile(r"_(\d+)$")


def standup_record_dir(manifest: Mapping[str, Any], *, gen_root: Path,
                       files: Sequence[str] = STANDUP_RECORD_FILES) -> tuple[Path | None, str | None, str | None]:
    """``(<gen_root>/serving/<jobdir>/gsm8k, <jobdir>, problem)`` from the
    manifest's own ``routing_table`` path (the ABSOLUTE cluster path; only its
    trailing ``serving/<jobdir>/gsm8k/routing_table.json`` is read, so a pulled
    copy resolves). The problem names :data:`STANDUP_RECORD_MISSING` when the
    path is absent or not that layout, the directory is not there, or any of
    ``files`` (:data:`STANDUP_RECORD_FILES`, the declared lens record's set;
    :data:`NATIVE_STANDUP_RECORD_FILES` for a native job's) is missing — never
    a fall-through."""
    path = manifest.get("routing_table")
    if not isinstance(path, str) or not path:
        return None, None, f"{STANDUP_RECORD_MISSING}: the manifest names no routing_table path — the standup record cannot be located"
    parts = PurePosixPath(path).parts
    if len(parts) < 4 or parts[-1] != STANDUP_ROUTING_TABLE or parts[-2] != STANDUP_TASK_SUBDIR or parts[-4] != STANDUP_DIR_PARENT:
        return None, None, (f"{STANDUP_RECORD_MISSING}: manifest routing_table {path!r} does not end in "
                            f"{STANDUP_DIR_PARENT}/<jobdir>/{STANDUP_TASK_SUBDIR}/{STANDUP_ROUTING_TABLE} — not a standup record path")
    jobdir = parts[-3]
    d = Path(gen_root) / STANDUP_DIR_PARENT / jobdir / STANDUP_TASK_SUBDIR
    if not d.is_dir():
        return None, jobdir, (f"{STANDUP_RECORD_MISSING}: no standup record at {d} (the cluster's serving/{jobdir}/{STANDUP_TASK_SUBDIR}/ "
                              f"{{{', '.join(files)}}} is not pulled beside the stores)")
    missing = [f for f in files if not (d / f).is_file()]
    if missing:
        return d, jobdir, f"{STANDUP_RECORD_MISSING}: standup record {d} lacks {missing}"
    return d, jobdir, None


def resolved_sampler_record(out_root: Path, *, cap: int, sampling: Mapping[str, Any]) -> dict[str, Any]:
    """The engine-resolved sampler attestation beside one leg's store — the
    recorder's verdict (``AGREES`` / ``DIVERGED`` / ...), its ``why``, the
    client reference it compared against and the caps it classified by — plus
    THIS reader's ``problem`` when the record does not bind to the row: a
    verdict outside the recorder's vocabulary, a client-cap set that does not
    name this leg's cap, or a client reference that is not this row's sampling
    row (the launcher passes ``--expect`` per key of the row; a recorder run
    against another reference attests nothing about this leg). ``present``
    False (a store produced before the recorder rode the launcher) is legal and
    is reported as NOT RECORDED. A finding, never a gate: the recorder's own
    exit policy is "recorded, not propagated"."""
    path = Path(out_root) / RESOLVED_SAMPLER_FILENAME
    out: dict[str, Any] = {"path": str(path), "present": path.is_file(), "verdict": None, "why": None, "problem": None,
                           "n_client_requests": None, "n_probe_requests": None, "n_unclassified_requests": None,
                           "reference": None, "client_caps": None, "divergences": None}
    if not path.is_file():
        return out
    try:
        doc = _mapping(_read_json(path))
    except AnalysisError as e:
        out["problem"] = f"{RESOLVED_SAMPLER_FILENAME} unreadable: {e}"
        return out
    verdict = doc.get("verdict")
    out.update(verdict=verdict, why=doc.get("why"), reference=doc.get("reference"), client_caps=doc.get("client_caps"),
               divergences=doc.get("divergences"),
               **{k: doc.get(k) for k in ("n_client_requests", "n_probe_requests", "n_unclassified_requests")})
    problems: list[str] = []
    if verdict not in RESOLVED_SAMPLER_VERDICTS:
        problems.append(f"verdict {verdict!r} is not one of the recorder's {RESOLVED_SAMPLER_VERDICTS}")
    caps = [str(c) for c in (doc.get("client_caps") or [])]
    if str(int(cap)) not in caps:
        problems.append(f"client_caps {caps} do not name this leg's cap {int(cap)} (the launcher passes --client-cap <rung>)")
    want = {k: float(sampling[k]) for k in RESOLVED_SAMPLER_KEYS if k in sampling}
    ref = doc.get("reference")
    got = ({k: float(v) for k, v in ref.items() if k in RESOLVED_SAMPLER_KEYS} if isinstance(ref, Mapping) else None)
    if verdict != "REFUSED" and got != want:
        problems.append(f"the recorder's client reference {ref!r} is not this row's sampling row {want!r} (the launcher passes --expect per key)")
    out["problem"] = "; ".join(problems) or None
    return out


def declared_served_positions_record(manifest: Mapping[str, Any], *, row: Mapping[str, Any], gen_root: Path,
                                     fact: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The AUTHORITATIVE served-positions fact of a DECLARED header-located cell,
    read off the standup record beside its job (served-positions-null-diagnosis
    §2 / §4.2) — never accepted from a null, never guessed. Resolves
    ``routing_table.json[lens_served_positions][<declared harness key>]``
    (the very map ``endpoint.json`` and the proxy received) and holds it to:

    * the map naming the DECLARED key only — a map that also names the bare
      key is not a declared standup's shape — with exactly ``[<expected>]``;
    * the ``routes[]`` entry for that key: ``virtual_model`` == the manifest
      route's, ``intervention_hash`` == the manifest's;
    * the lens registry READ AS BYTES ONCE and BOUND to the attestation before
      it is parsed: ``sha256(bytes)[:16]`` == ``steering_attestation.json
      .registry_index_sha16`` (the producer's content identity of the registry
      the workers installed — :func:`registry_index_sha16` is its own function;
      review of 337efb7a, HIGH: copied ``expect_hash`` / ``artifact_sha16``
      fields do not bind the file to the attestation), the attestation's
      ``registry_dir`` naming this job's ``lens_registry``; then the entry for
      that key: ``resolved_spec.positions`` == the expected mode,
      ``serve_out_of_support`` == the declaration, ``artifact_sha16`` == the
      (view, cell) pin's first 16 hex, ``expect_hash`` == the manifest's
      ``intervention_hash``, and — when the family ``fact`` is given — the
      registry's ``assistant_header_ids`` / ``assistant_header_ids_thinking_on``
      equal to the fact's off / on tuples (the header the workers located the
      response mask against);
    * ``steering_attestation.json.slurm_job_id`` and ``standup_jobs.jsonl[0]``'s
      ``job_id`` == the job in the directory name, its ``sha`` == the manifest
      producer's ``tree_sha``;
    * the manifest's own declared-class facts: ``header_located`` and
      ``transfer_provenance.header_located`` true, the serve declaration under
      the declared key, ``served_header_ids`` for both arms;
    * every manifest served-positions field that is NOT null agrees with it.

    Returns ``{"resolved", "resolved_from", "problem", "standup_dir", "jobdir",
    "job_id", "gaps"}`` — ``problem`` names :data:`STANDUP_RECORD_MISSING` /
    :data:`STANDUP_RECORD_MISMATCH` (every disagreement, joined) and leaves
    ``resolved`` None; ``gaps`` lists one entry per manifest field the producer
    wrote null (:data:`RECORD_GAP_FIELDS`), each naming the field, the
    manifest's value, the resolved value, its source and the producer cause
    (:data:`RECORD_GAP_CAUSE_BARE_KEY_LOOKUP`). Call it on a declared fitted row
    only — anything else is a caller bug."""
    cell, declared, fitted = str(row["cell"]), bool(row["declared"]), bool(row["fitted"])
    if not (declared and fitted):
        raise AnalysisError(f"declared_served_positions_record: {cell!r} is not a declared fitted row (declared={declared}, fitted={fitted})")
    spell = C.cell_spellings(cell, declared=True)
    bare = C.cell_spellings(cell, declared=False).harness
    expect = str(row["served_positions_expected"])
    out: dict[str, Any] = {"resolved": None, "resolved_from": None, "problem": None, "standup_dir": None, "jobdir": None, "job_id": None,
                           "registry_index_sha16": None, "gaps": []}
    d, jobdir, problem = standup_record_dir(manifest, gen_root=gen_root)
    out.update(standup_dir=(str(d) if d is not None else None), jobdir=jobdir)
    if problem:
        out["problem"] = problem
        return out
    assert d is not None and jobdir is not None
    problems: list[str] = []
    m = _JOB_ID_SUFFIX.search(jobdir)
    job_id = m.group(1) if m else None
    if job_id is None:
        problems.append(f"{STANDUP_RECORD_MISMATCH}: the standup dir {jobdir!r} does not end in _<Slurm job id>")
    try:
        table = _mapping(_read_json(d / STANDUP_ROUTING_TABLE))
        attestation = _mapping(_read_json(d / STANDUP_ATTESTATION))
        jobs = [j for j in _iter_jsonl(d / STANDUP_JOBS) if isinstance(j, Mapping)]
        registry_bytes = (d / STANDUP_REGISTRY).read_bytes()
    except (AnalysisError, OSError) as e:
        out["problem"] = f"{STANDUP_RECORD_MISSING}: standup record {d} is unreadable — {e}"
        return out
    # the registry is bound to the attestation BEFORE it is read: the same bytes the producer hashed into
    # registry_index_sha16 at launch, the same function — a registry substituted beside a stale attestation is refused
    # here whatever identifiers its entries copy
    digest = registry_index_sha16(registry_bytes)
    attested = attestation.get(ATTESTATION_KEY_REGISTRY_INDEX_SHA16)
    if not isinstance(attested, str) or not _HEX16.match(attested):
        out["problem"] = (f"{STANDUP_RECORD_MISMATCH}: {STANDUP_ATTESTATION} records no {ATTESTATION_KEY_REGISTRY_INDEX_SHA16} ({attested!r}) — the "
                          f"registry cannot be bound to the workers' install; refusing")
        return out
    if attested != digest:
        out["problem"] = (f"{STANDUP_RECORD_MISMATCH}: {STANDUP_REGISTRY} bytes digest {digest} != the attestation's "
                          f"{ATTESTATION_KEY_REGISTRY_INDEX_SHA16} {attested} — not the registry the workers installed; refusing")
        return out
    out["registry_index_sha16"] = digest
    try:
        registry = _mapping(json.loads(registry_bytes.decode("utf-8")))
    except (ValueError, UnicodeDecodeError) as e:
        out["problem"] = f"{STANDUP_RECORD_MISSING}: {STANDUP_REGISTRY} is not a JSON object — {e}"
        return out
    reg_dir = attestation.get("registry_dir")
    if not same_store_tail(reg_dir, f"{STANDUP_DIR_PARENT}/{jobdir}/{STANDUP_TASK_SUBDIR}/{PurePosixPath(STANDUP_REGISTRY).parent}"):
        problems.append(f"{STANDUP_RECORD_MISMATCH}: {STANDUP_ATTESTATION} registry_dir {reg_dir!r} does not name this job's "
                        f"{STANDUP_DIR_PARENT}/{jobdir}/{STANDUP_TASK_SUBDIR}/{PurePosixPath(STANDUP_REGISTRY).parent}")
    if fact is not None:
        for field, arm in (("assistant_header_ids", "off_ids"), ("assistant_header_ids_thinking_on", "on_ids")):
            have = registry.get(field)
            if not isinstance(have, list) or [int(i) for i in have] != [int(i) for i in fact[arm]]:
                problems.append(f"{STANDUP_RECORD_MISMATCH}: {STANDUP_REGISTRY} {field} {have!r} != the {row['family']!r} fact's {arm} "
                                f"{list(fact[arm])} — the workers located the mask against another header")
    key = C.LENS_SERVED_POSITIONS_KEY
    served_map = table.get(key)
    modes: list[str] | None = None
    if not isinstance(served_map, Mapping):
        problems.append(f"{STANDUP_RECORD_MISMATCH}: {STANDUP_ROUTING_TABLE} carries no {key} map")
    else:
        if bare in served_map:
            problems.append(f"{STANDUP_RECORD_MISMATCH}: {STANDUP_ROUTING_TABLE}[{key}] names the BARE key {bare!r} "
                            f"({sorted(served_map)}) — a map naming the bare spelling is not a declared standup's shape")
        modes = C.served_positions_of_cell(table, spell.harness)
        if modes is None:
            problems.append(f"{STANDUP_RECORD_MISMATCH}: {STANDUP_ROUTING_TABLE}[{key}] names {sorted(served_map)}, not the declared key {spell.harness!r}")
        elif not _served_positions_match(modes, expect):
            problems.append(f"{STANDUP_RECORD_MISMATCH}: {STANDUP_ROUTING_TABLE}[{key}][{spell.harness!r}] = {modes!r} != [{expect!r}]")
    route = _mapping(manifest.get(C.MANIFEST_KEY_ROUTE))
    entries = [r for r in (table.get("routes") or []) if isinstance(r, Mapping) and r.get("cell_id") == spell.harness]
    if len(entries) != 1:
        problems.append(f"{STANDUP_RECORD_MISMATCH}: {STANDUP_ROUTING_TABLE} routes name {spell.harness!r} {len(entries)} time(s), not once")
    else:
        if entries[0].get("virtual_model") != route.get("virtual_model"):
            problems.append(f"{STANDUP_RECORD_MISMATCH}: routes[{spell.harness!r}].virtual_model {entries[0].get('virtual_model')!r} != the manifest route's "
                            f"{route.get('virtual_model')!r}")
        if entries[0].get("intervention_hash") != manifest.get("intervention_hash"):
            problems.append(f"{STANDUP_RECORD_MISMATCH}: routes[{spell.harness!r}].intervention_hash {entries[0].get('intervention_hash')!r} != the manifest's "
                            f"{manifest.get('intervention_hash')!r}")
    pin = str(row["object_sha256"])
    want_decl = list(C.expected_serve_out_of_support(lens=True, arm=C.THINKING_ON, out_of_support=True, header_located=True))
    regs = [e for e in (registry.get("entries") or []) if isinstance(e, Mapping) and e.get("cell_id") == spell.harness]
    if len(regs) != 1:
        problems.append(f"{STANDUP_RECORD_MISMATCH}: {STANDUP_REGISTRY} names {spell.harness!r} {len(regs)} time(s), not once")
    else:
        spec = _mapping(regs[0].get("resolved_spec"))
        if spec.get("positions") != expect:
            problems.append(f"{STANDUP_RECORD_MISMATCH}: {STANDUP_REGISTRY} resolved_spec.positions {spec.get('positions')!r} != {expect!r}")
        if list(spec.get(C.SERVE_OUT_OF_SUPPORT_KEY) or []) != want_decl:
            problems.append(f"{STANDUP_RECORD_MISMATCH}: {STANDUP_REGISTRY} resolved_spec.{C.SERVE_OUT_OF_SUPPORT_KEY} "
                            f"{spec.get(C.SERVE_OUT_OF_SUPPORT_KEY)!r} != {want_decl!r}")
        if regs[0].get("artifact_sha16") != pin[:16]:
            problems.append(f"{STANDUP_RECORD_MISMATCH}: {STANDUP_REGISTRY} artifact_sha16 {regs[0].get('artifact_sha16')!r} != the (view, cell) pin's {pin[:16]!r}")
        if regs[0].get("expect_hash") != manifest.get("intervention_hash"):
            problems.append(f"{STANDUP_RECORD_MISMATCH}: {STANDUP_REGISTRY} expect_hash {regs[0].get('expect_hash')!r} != the manifest's intervention_hash "
                            f"{manifest.get('intervention_hash')!r}")
    if job_id is not None:
        if str(attestation.get("slurm_job_id")) != job_id:
            problems.append(f"{STANDUP_RECORD_MISMATCH}: {STANDUP_ATTESTATION} slurm_job_id {attestation.get('slurm_job_id')!r} != the dir's job {job_id}")
        if not jobs:
            problems.append(f"{STANDUP_RECORD_MISMATCH}: {STANDUP_JOBS} records no job")
        else:
            if str(jobs[0].get("job_id")) != job_id:
                problems.append(f"{STANDUP_RECORD_MISMATCH}: {STANDUP_JOBS}[0].job_id {jobs[0].get('job_id')!r} != the dir's job {job_id}")
            tree = _mapping(manifest.get(C.MANIFEST_KEY_PRODUCER)).get("tree_sha")
            if jobs[0].get("sha") != tree:
                problems.append(f"{STANDUP_RECORD_MISMATCH}: {STANDUP_JOBS}[0].sha {jobs[0].get('sha')!r} != the manifest producer's tree_sha {tree!r}")
    if manifest.get("header_located") is not True:
        problems.append(f"{STANDUP_RECORD_MISMATCH}: manifest header_located {manifest.get('header_located')!r} is not true on a DECLARED cell")
    tp = _mapping(manifest.get(C.MANIFEST_KEY_TRANSFER_PROVENANCE))
    if tp.get("header_located") is not True:
        problems.append(f"{STANDUP_RECORD_MISMATCH}: {C.MANIFEST_KEY_TRANSFER_PROVENANCE}.header_located {tp.get('header_located')!r} is not true on a DECLARED cell")
    decl = _mapping(_mapping(route.get(C.ROUTE_KEY_SERVE_DECLARATIONS)).get(spell.harness))
    if dict(decl) != {C.SERVE_OUT_OF_SUPPORT_KEY: want_decl}:
        problems.append(f"{STANDUP_RECORD_MISMATCH}: route.{C.ROUTE_KEY_SERVE_DECLARATIONS}[{spell.harness!r}] {dict(decl)!r} != "
                        f"{{{C.SERVE_OUT_OF_SUPPORT_KEY!r}: {want_decl!r}}}")
    ids = _mapping(tp.get("served_header_ids"))
    if not ids.get(C.THINKING_OFF) or not ids.get(C.THINKING_ON):
        problems.append(f"{STANDUP_RECORD_MISMATCH}: {C.MANIFEST_KEY_TRANSFER_PROVENANCE}.served_header_ids lacks an arm ({sorted(ids)}) — both arms were measured "
                        "on every declared serve")
    if problems:
        out["problem"] = "; ".join(problems)
        return out
    assert modes is not None
    resolved_from = f"{jobdir}/{STANDUP_TASK_SUBDIR}/{STANDUP_ROUTING_TABLE}[{key}][{spell.harness!r}]"
    values = {RECORD_GAP_FIELD_TRANSFER: tp.get("served_positions", M._ABSENT), RECORD_GAP_FIELD_ROUTE: route.get(key, M._ABSENT),
              RECORD_GAP_FIELD_TOP: manifest.get(key, M._ABSENT)}
    gaps: list[dict[str, Any]] = []
    for field, v in values.items():
        if v is None or v is M._ABSENT:
            gaps.append({"field": field, "manifest": None, "resolved": list(modes), "resolved_from": resolved_from,
                         "cause": RECORD_GAP_CAUSE_BARE_KEY_LOOKUP})
        elif M._served_positions(v) != list(modes):
            problems.append(f"{STANDUP_RECORD_MISMATCH}: manifest {field} {v!r} disagrees with the standup record's {modes!r}")
    if problems:
        out["problem"] = "; ".join(problems)
        return out
    out.update(resolved=list(modes), resolved_from=resolved_from, job_id=job_id, gaps=gaps)
    return out


def native_served_positions_gap(manifest: Mapping[str, Any]) -> bool:
    """Whether a manifest carries the NATIVE record gap: a route the
    certification predicates read on the native path (a ``+lora+`` identity
    beside ``serving_engine: lora_native``, not torn — the MATH-500 reader's
    one rule ``certification_path_of``) with a transfer-provenance block whose
    ``served_positions`` is null. THE ONE predicate for the caller that resolves
    the standup record and for :func:`identity_problem`, which consumes it. A
    LENS route with the same null is not this gap (it refuses exactly as before),
    and neither is a torn native route (the lens predicates name the tear)."""
    route = _mapping(manifest.get(C.MANIFEST_KEY_ROUTE))
    path, problems = M.certification_path_of(route)
    if path != M.STEERED_PATH_LORA or problems:
        return False
    tp = manifest.get(C.MANIFEST_KEY_TRANSFER_PROVENANCE)
    return isinstance(tp, Mapping) and tp.get("served_positions") is None


def native_served_positions_stamped(manifest: Mapping[str, Any]) -> bool:
    """Whether a manifest on the native certification path STATES a served-positions
    map in the LENS-named field — ``route.lens_served_positions`` or its top-level
    copy non-null. The FIXED producer's record: the driver's standup lookup reads
    the routing table's engine-neutral ``served_positions`` map first
    (``served_capability_run._standup_served_positions``) and writes the value into
    the lens-named field on every engine (``served_capability_run.py`` 6315-6316).
    The caller resolves the native standup record for such a leg too, so the
    certification can hold the stamp to the record's engine-neutral map instead of
    refusing it as a lens fact; a torn native route is not this case."""
    route = _mapping(manifest.get(C.MANIFEST_KEY_ROUTE))
    path, problems = M.certification_path_of(route)
    if path != M.STEERED_PATH_LORA or problems:
        return False
    return any(v is not None for v in (route.get(C.LENS_SERVED_POSITIONS_KEY), manifest.get(C.LENS_SERVED_POSITIONS_KEY)))


def native_stamp_problem(stamp: Any, *, neutral: Sequence[str] | None) -> str | None:
    """Why a served-positions value in a LENS-named field of a NATIVE record is NOT the routing table's engine-neutral
    fact — ``None`` when it is exactly ``neutral``, the record's ``served_positions[<harness>]`` as a list of modes
    (``None``: the table names no entry for the cell). The neutral map is the ONLY record a native stamp is held to: the
    routes[] ``petri_position_mode`` attestation resolves the positions when the map is absent but admits no stamp, and the
    engine's name admits nothing."""
    modes = M._served_positions(stamp)
    if modes is None:
        return f"{stamp!r} is not a served-positions value (a mode, or a list of modes)"
    if neutral is None:
        return (f"{STANDUP_ROUTING_TABLE} carries no {SERVED_POSITIONS_KEY} entry for the cell — the engine-neutral map is the only "
                f"record a native stamp is held to (a routes[].{ROUTE_POSITION_MODE_KEY} attestation admits no stamp)")
    if modes != list(neutral):
        return f"the stamp {modes!r} != {STANDUP_ROUTING_TABLE}[{SERVED_POSITIONS_KEY}] {list(neutral)!r}"
    return None


def native_stamp_record_problem(stamp: Any, *, standup_record: Mapping[str, Any] | None) -> str | None:
    """:func:`native_stamp_problem` through the native standup record the caller resolved
    (:func:`native_served_positions_record`): the record must have been read, resolve
    without a problem, and resolve FROM THE ENGINE-NEUTRAL MAP (:data:`NATIVE_RESOLVED_FROM_NEUTRAL_MAP`
    — a record resolved from the routes[] attestation means the table carries no neutral entry)."""
    if not isinstance(standup_record, Mapping):
        return (f"no native standup record was read ({STANDUP_DIR_PARENT}/<jobdir>/{STANDUP_TASK_SUBDIR}/{STANDUP_ROUTING_TABLE}), "
                "so the stamp cannot be held to one")
    if standup_record.get("problem"):
        return f"the standup record does not resolve it — {standup_record['problem']}"
    if standup_record.get("source") != NATIVE_RESOLVED_FROM_NEUTRAL_MAP:
        return (f"the standup record resolved the positions from {standup_record.get('source')!r}, not the {SERVED_POSITIONS_KEY} map — "
                f"{STANDUP_ROUTING_TABLE} carries no engine-neutral entry for the cell, and only that map admits a stamp")
    return native_stamp_problem(stamp, neutral=standup_record.get("resolved"))


def native_served_positions_record(manifest: Mapping[str, Any], *, row: Mapping[str, Any], gen_root: Path) -> dict[str, Any]:
    """The AUTHORITATIVE served-positions fact of a NATIVELY served UNDECLARED
    fitted cell, read off the standup record beside its job — never accepted
    from the null, never defaulted from the engine's name. The producer wrote
    ``transfer_provenance.served_positions: null`` on every native fitted cell
    (:data:`RECORD_GAP_CAUSE_NATIVE_LENS_MAP_ONLY`: it reads the LENS map only,
    which a native standup writes empty) and derived ``header_located`` /
    ``header_check_class`` from that null, fail closed. The record's fact:

    * ``routing_table.json[served_positions][<harness key>]`` — the ENGINE-
      NEUTRAL map new standups stamp — read FIRST; else the routes[] entry's
      ``petri_position_mode`` (the native attestation the existing native
      cards' tables carry); both present must agree; NEITHER present refuses
      :data:`NATIVE_POSITION_MODE_UNATTESTED`;
    * held to the ROW: exactly ``[<served_positions_expected>]``; and to the
      contract: ``contracts.header_located(resolved)`` must equal the row's
      declaration (declared iff header-located — a header-located or
      unclassifiable mode on an undeclared row refuses);
    * bound to the manifest: the routes[] entry for the harness key exactly
      once, its ``virtual_model`` / ``intervention_hash`` the manifest's, its
      ``serving_identity`` the manifest route's AND a native (``+lora+``) one;
      the manifest route on the native certification path; the lens map naming
      the cell under NEITHER spelling (not a native standup's shape); the
      manifest's own native facts (top-level ``header_located`` False — the
      engine's fact the driver writes; no served-positions map in the route
      block or at the top level — a lens fact on a native record);
    * bound to the job: ``serving_attestation.json.kind`` the native engine,
      its ``slurm_job_id`` and ``standup_jobs.jsonl[0].job_id`` the job in the
      directory name, the standup's ``sha`` the manifest producer's ``tree_sha``,
      its ``engine`` the native engine;
    * the transfer-provenance block: a STATED ``served_positions`` must equal
      the resolved fact with ``header_located`` / ``header_check_class`` equal
      to the contract's derivation from it (the fixed producer — no gap); a
      NULL is accepted ONLY as the named gap, i.e. with ``header_located`` /
      ``header_check_class`` EXACTLY the contract's fail-closed derivation from
      None — any other pairing is another producer's record and refuses.

    Returns ``{"resolved", "resolved_from", "source", "problem", "standup_dir",
    "jobdir", "job_id", "engine", "header_located", "header_check_class",
    "gaps"}`` — ``problem`` names :data:`STANDUP_RECORD_MISSING` /
    :data:`STANDUP_RECORD_MISMATCH` / :data:`NATIVE_POSITION_MODE_UNATTESTED`
    (every disagreement, joined) and leaves ``resolved`` None; ``gaps`` lists
    one entry per transfer-provenance field the null reached
    (:data:`NATIVE_RECORD_GAP_FIELDS`), each naming the field, the manifest's
    value, the re-derived value, its source and the producer cause. Call it on
    an undeclared fitted row only — anything else is a caller bug."""
    cell, declared, fitted = str(row["cell"]), bool(row["declared"]), bool(row["fitted"])
    if declared or not fitted:
        raise AnalysisError(f"native_served_positions_record: {cell!r} is not an undeclared fitted row (declared={declared}, fitted={fitted})")
    spell = C.cell_spellings(cell, declared=False)
    declared_key = C.cell_spellings(cell, declared=True).harness
    expect = str(row["served_positions_expected"])
    out: dict[str, Any] = {"resolved": None, "resolved_from": None, "source": None, "problem": None, "standup_dir": None, "jobdir": None,
                           "job_id": None, "engine": C.SERVING_ENGINE_LORA_NATIVE, "header_located": None, "header_check_class": None, "gaps": []}
    route = _mapping(manifest.get(C.MANIFEST_KEY_ROUTE))
    path, path_problems = M.certification_path_of(route)
    if path != M.STEERED_PATH_LORA or path_problems:
        out["problem"] = (f"{STANDUP_RECORD_MISMATCH}: the manifest's route is not a natively served one ("
                          + ("; ".join(path_problems) if path_problems else f"certification path {path!r}")
                          + ") — the native standup record resolves a native route's positions only")
        return out
    tp = manifest.get(C.MANIFEST_KEY_TRANSFER_PROVENANCE)
    if not isinstance(tp, Mapping):
        out["problem"] = f"{STANDUP_RECORD_MISMATCH}: no {C.MANIFEST_KEY_TRANSFER_PROVENANCE} block on a fitted job's manifest"
        return out
    d, jobdir, problem = standup_record_dir(manifest, gen_root=gen_root, files=NATIVE_STANDUP_RECORD_FILES)
    out.update(standup_dir=(str(d) if d is not None else None), jobdir=jobdir)
    if problem:
        out["problem"] = problem
        return out
    assert d is not None and jobdir is not None
    problems: list[str] = []
    m = _JOB_ID_SUFFIX.search(jobdir)
    job_id = m.group(1) if m else None
    if job_id is None:
        problems.append(f"{STANDUP_RECORD_MISMATCH}: the standup dir {jobdir!r} does not end in _<Slurm job id>")
    try:
        table = _mapping(_read_json(d / STANDUP_ROUTING_TABLE))
        attestation = _mapping(_read_json(d / STANDUP_SERVING_ATTESTATION))
        jobs = [j for j in _iter_jsonl(d / STANDUP_JOBS) if isinstance(j, Mapping)]
    except (AnalysisError, OSError) as e:
        out["problem"] = f"{STANDUP_RECORD_MISSING}: standup record {d} is unreadable — {e}"
        return out
    # the record is the NATIVE engine's by its own words — the attestation's kind and the standup's engine — never by
    # the engine name on the manifest alone
    if attestation.get("kind") != C.SERVING_ENGINE_LORA_NATIVE:
        problems.append(f"{STANDUP_RECORD_MISMATCH}: {STANDUP_SERVING_ATTESTATION} kind {attestation.get('kind')!r} != {C.SERVING_ENGINE_LORA_NATIVE!r} — "
                        "not a native standup's attestation")
    if job_id is not None:
        if str(attestation.get("slurm_job_id")) != job_id:
            problems.append(f"{STANDUP_RECORD_MISMATCH}: {STANDUP_SERVING_ATTESTATION} slurm_job_id {attestation.get('slurm_job_id')!r} != the dir's job {job_id}")
        if not jobs:
            problems.append(f"{STANDUP_RECORD_MISMATCH}: {STANDUP_JOBS} records no job")
        else:
            if str(jobs[0].get("job_id")) != job_id:
                problems.append(f"{STANDUP_RECORD_MISMATCH}: {STANDUP_JOBS}[0].job_id {jobs[0].get('job_id')!r} != the dir's job {job_id}")
            tree = _mapping(manifest.get(C.MANIFEST_KEY_PRODUCER)).get("tree_sha")
            # THE SERVING STACK'S COMMIT, from its authoritative source. `standup_jobs.jsonl[0].sha` is the tree the
            # STANDUP ran from — the checkout that built and served the adapter — where `producer.tree_sha` is the
            # DRIVER's. For these launchers the two are one eval-repo checkout, so they MUST agree, and the disagreement
            # below is the refusal: two stacks under one row is exactly what a serving-commit field exists to catch.
            # Exposed here (not re-read downstream) so the export reads the same bytes this check validated.
            out["standup_sha"], out["producer_tree_sha"] = jobs[0].get("sha"), tree
            if jobs[0].get("sha") != tree:
                problems.append(f"{STANDUP_RECORD_MISMATCH}: {STANDUP_JOBS}[0].sha {jobs[0].get('sha')!r} != the manifest producer's tree_sha {tree!r}")
            if jobs[0].get("engine") != C.SERVING_ENGINE_LORA_NATIVE:
                problems.append(f"{STANDUP_RECORD_MISMATCH}: {STANDUP_JOBS}[0].engine {jobs[0].get('engine')!r} != {C.SERVING_ENGINE_LORA_NATIVE!r}")
    entries = [r for r in (table.get("routes") or []) if isinstance(r, Mapping) and r.get("cell_id") == spell.harness]
    if len(entries) != 1:
        problems.append(f"{STANDUP_RECORD_MISMATCH}: {STANDUP_ROUTING_TABLE} routes name {spell.harness!r} {len(entries)} time(s), not once")
    else:
        entry = entries[0]
        if entry.get("virtual_model") != route.get("virtual_model"):
            problems.append(f"{STANDUP_RECORD_MISMATCH}: routes[{spell.harness!r}].virtual_model {entry.get('virtual_model')!r} != the manifest route's "
                            f"{route.get('virtual_model')!r}")
        if entry.get("intervention_hash") != manifest.get("intervention_hash"):
            problems.append(f"{STANDUP_RECORD_MISMATCH}: routes[{spell.harness!r}].intervention_hash {entry.get('intervention_hash')!r} != the manifest's "
                            f"{manifest.get('intervention_hash')!r}")
        identity = entry.get("serving_identity")
        if identity != route.get("serving_identity"):
            problems.append(f"{STANDUP_RECORD_MISMATCH}: routes[{spell.harness!r}].serving_identity {identity!r} != the manifest route's "
                            f"{route.get('serving_identity')!r}")
        try:
            entry_path = M.steered_path_of(identity)
        except ValueError:
            entry_path = None
        if entry_path != M.STEERED_PATH_LORA:
            problems.append(f"{STANDUP_RECORD_MISMATCH}: routes[{spell.harness!r}].serving_identity {identity!r} is not a native (+lora+) identity")
    lens_map = table.get(C.LENS_SERVED_POSITIONS_KEY)
    if isinstance(lens_map, Mapping) and (spell.harness in lens_map or declared_key in lens_map):
        problems.append(f"{STANDUP_RECORD_MISMATCH}: {STANDUP_ROUTING_TABLE}[{C.LENS_SERVED_POSITIONS_KEY}] names the cell "
                        f"({sorted(k for k in lens_map if k in (spell.harness, declared_key))}) — a lens map naming a natively served cell is not a "
                        "native standup's shape")
    # the fact: the engine-neutral map first, the routes[] attestation second; both present must agree; neither refuses
    neutral: list[str] | None = None
    neutral_map = table.get(SERVED_POSITIONS_KEY)
    if isinstance(neutral_map, Mapping) and spell.harness in neutral_map:
        neutral = M._served_positions(neutral_map[spell.harness])
        if neutral is None:
            problems.append(f"{STANDUP_RECORD_MISMATCH}: {STANDUP_ROUTING_TABLE}[{SERVED_POSITIONS_KEY}][{spell.harness!r}] = "
                            f"{neutral_map[spell.harness]!r} is not a served-positions value (a mode, or a list of modes)")
    attested: list[str] | None = None
    if len(entries) == 1 and entries[0].get(ROUTE_POSITION_MODE_KEY) is not None:
        mode = entries[0].get(ROUTE_POSITION_MODE_KEY)
        if isinstance(mode, str) and mode:
            attested = [mode]
        else:
            problems.append(f"{STANDUP_RECORD_MISMATCH}: routes[{spell.harness!r}].{ROUTE_POSITION_MODE_KEY} {mode!r} is not a position mode")
    if neutral is not None and attested is not None and neutral != attested:
        problems.append(f"{STANDUP_RECORD_MISMATCH}: {STANDUP_ROUTING_TABLE}[{SERVED_POSITIONS_KEY}][{spell.harness!r}] {neutral!r} != the routes[] "
                        f"attestation {ROUTE_POSITION_MODE_KEY} {attested!r} — one standup's two records disagree")
    resolved: list[str] | None = None
    source = resolved_from = None
    if neutral is not None:
        resolved, source = neutral, NATIVE_RESOLVED_FROM_NEUTRAL_MAP
        resolved_from = f"{jobdir}/{STANDUP_TASK_SUBDIR}/{STANDUP_ROUTING_TABLE}[{SERVED_POSITIONS_KEY}][{spell.harness!r}]"
    elif attested is not None:
        resolved, source = attested, NATIVE_RESOLVED_FROM_ROUTES
        resolved_from = f"{jobdir}/{STANDUP_TASK_SUBDIR}/{STANDUP_ROUTING_TABLE}.routes[{spell.harness!r}].{ROUTE_POSITION_MODE_KEY}"
    else:
        problems.append(f"{NATIVE_POSITION_MODE_UNATTESTED}: native route {spell.harness!r} attests no position mode — {STANDUP_ROUTING_TABLE} "
                        f"carries no {SERVED_POSITIONS_KEY}[{spell.harness!r}] and routes[{spell.harness!r}] has no {ROUTE_POSITION_MODE_KEY}; "
                        "the served positions cannot be resolved (never defaulted from the engine's name)")
    located: bool | None = None
    klass: str | None = None
    if resolved is not None:
        unclassified = C.unclassified_native_modes(resolved)
        if unclassified:
            # a native-vocabulary mode the contract cannot classify: header_located would read it header-located, fail closed —
            # the null bug one vocabulary over. Refused by name; whether such a mode locates a header is not this reader's call
            problems.append(f"{NATIVE_POSITION_MODE_UNCLASSIFIED}: {unclassified[0]!r} — the record resolves {resolved!r}, a native-vocabulary "
                            f"mode outside the served vocabulary {list(PX.POSITION_MODES)} that contracts.header_located would read as "
                            "header-located (fail closed); the leg refuses rather than derive a header fact from it")
        if not _served_positions_match(resolved, expect):
            problems.append(f"{STANDUP_RECORD_MISMATCH}: the standup record resolves {resolved!r} != [{expect!r}] (the row's served_positions_expected)")
        # declared iff header-located, applied to the RESOLVED fact through the contract's own predicate (a mode outside the
        # served vocabulary reads header-located there, fail closed — refused here by name, never admitted as `all`)
        located = C.header_located(resolved)
        if located is not declared:
            problems.append(f"{STANDUP_RECORD_MISMATCH}: the resolved positions {resolved!r} read header_located={located} on an UNDECLARED row "
                            f"(declared iff header-located; contracts.header_located reads a mode outside {list(PX.POSITION_MODES)} as located)")
        klass = C.header_check_class(fit_response_header_len=tp.get("fit_response_header_len"), served_positions=resolved)
        out.update(header_located=located, header_check_class=klass)
    # the manifest's own native facts
    if manifest.get("header_located") is not False:
        problems.append(f"{STANDUP_RECORD_MISMATCH}: manifest header_located {manifest.get('header_located')!r} is not false on a natively served "
                        "cell — the engine's fact the driver writes on every native lens-grammar cell")
    for field, v in ((RECORD_GAP_FIELD_ROUTE, route.get(C.LENS_SERVED_POSITIONS_KEY, M._ABSENT)),
                     (RECORD_GAP_FIELD_TOP, manifest.get(C.LENS_SERVED_POSITIONS_KEY, M._ABSENT))):
        if v is not None and v is not M._ABSENT:
            # the FIXED producer (the driver's neutral-map lookup, served_capability_run._standup_served_positions -> the
            # manifest write at 6315-6316) puts the ENGINE-NEUTRAL value into the lens-named field too; it is admitted ONLY
            # as the neutral map's own value for this cell (`neutral`) — never against the routes[] attestation, never the
            # engine's name — and refuses, naming what differs, otherwise
            why = native_stamp_problem(v, neutral=neutral)
            if why is not None:
                problems.append(f"{STANDUP_RECORD_MISMATCH}: manifest {field} = {v!r} on a natively served cell — a served-positions map is a lens fact; "
                                f"on a native record it is admitted only as the standup record's engine-neutral value, and {why}")
    gaps: list[dict[str, Any]] = []
    if resolved is not None:
        sp = tp.get("served_positions")
        if sp is None:
            # the NAMED gap: the block's two derived fields must be EXACTLY the contract's fail-closed derivation from the null
            # (what the producer computed); any other pairing is another producer's record, not this gap
            null_located = C.header_located(None)
            null_class = C.header_check_class(fit_response_header_len=tp.get("fit_response_header_len"), served_positions=None)
            if tp.get("header_located") is not null_located:
                problems.append(f"{STANDUP_RECORD_MISMATCH}: {RECORD_GAP_FIELD_TRANSFER_HEADER_LOCATED} {tp.get('header_located')!r} beside a null "
                                f"served_positions is not the producer's fail-closed derivation from the null ({null_located}) — another producer's "
                                "record, not the named gap")
            if tp.get("header_check_class") != null_class:
                problems.append(f"{STANDUP_RECORD_MISMATCH}: {RECORD_GAP_FIELD_TRANSFER_HEADER_CHECK_CLASS} {tp.get('header_check_class')!r} beside a "
                                f"null served_positions is not the producer's fail-closed derivation from the null ({null_class!r}) — another "
                                "producer's record, not the named gap")
            gaps.append({"field": RECORD_GAP_FIELD_TRANSFER, "manifest": None, "resolved": list(resolved), "resolved_from": resolved_from,
                         "source": source, "cause": RECORD_GAP_CAUSE_NATIVE_LENS_MAP_ONLY})
            for field, have, want, fn in ((RECORD_GAP_FIELD_TRANSFER_HEADER_LOCATED, tp.get("header_located"), located, "contracts.header_located"),
                                          (RECORD_GAP_FIELD_TRANSFER_HEADER_CHECK_CLASS, tp.get("header_check_class"), klass, "contracts.header_check_class")):
                if have != want:
                    gaps.append({"field": field, "manifest": have, "resolved": want, "resolved_from": resolved_from, "source": source,
                                 "derivation": f"{fn}(<the resolved positions>)", "cause": RECORD_GAP_CAUSE_NATIVE_LENS_MAP_ONLY})
        else:
            # the fixed producer: the stated value agrees, and so do its two derivations — no gap
            if M._served_positions(sp) != resolved:
                problems.append(f"{STANDUP_RECORD_MISMATCH}: manifest {RECORD_GAP_FIELD_TRANSFER} {sp!r} disagrees with the standup record's {resolved!r}")
            if tp.get("header_located") is not located:
                problems.append(f"{STANDUP_RECORD_MISMATCH}: {RECORD_GAP_FIELD_TRANSFER_HEADER_LOCATED} {tp.get('header_located')!r} != "
                                f"contracts.header_located({resolved!r}) = {located}")
            if tp.get("header_check_class") != klass:
                problems.append(f"{STANDUP_RECORD_MISMATCH}: {RECORD_GAP_FIELD_TRANSFER_HEADER_CHECK_CLASS} {tp.get('header_check_class')!r} != "
                                f"contracts.header_check_class(...) = {klass!r}")
    if problems:
        out["problem"] = "; ".join(problems)
        return out
    assert resolved is not None
    out.update(resolved=list(resolved), resolved_from=resolved_from, source=source, job_id=job_id, gaps=gaps)
    return out


# --------------------------------------------------------------------------- #
# identity (identity_problem_g8k)                                             #
# --------------------------------------------------------------------------- #


def _served_positions_match(v: Any, expect: str) -> bool:
    """A served-positions value is exactly ``[expect]`` — ``all`` on an
    undeclared object, ``response`` on a declared one (the header-located
    mode the MATH reader refused is THIS experiment's declared class)."""
    pos = M._served_positions(v)
    return pos is not None and set(pos) == {expect}


def _arm_of_flags(thinking_on: bool, arm: str | None) -> str:
    """The leg's whole arm from the two spellings the validators take: an
    explicit ``arm`` (llama-paper-rows: ``on`` | ``off`` | ``not_available``)
    wins; else the legacy ``thinking_on`` boolean (ON / the overlap check's OFF)."""
    if arm is not None:
        if arm not in C.THINKING_MODES:
            raise AnalysisError(f"arm {arm!r} is not one of {C.THINKING_MODES}")
        return str(arm)
    return C.THINKING_ON if thinking_on else C.THINKING_OFF


def identity_problem(manifest: Mapping[str, Any], summary: Mapping[str, Any] | None, *, row: Mapping[str, Any], cap: int,
                     model_pin: tuple[str, str], sampling_row: Mapping[str, Any], fact: Mapping[str, Any],
                     rung_source_manifest: Mapping[str, Any] | None = None, thinking_on: bool = True,
                     sampling_mode: str | None = None, rung_subset: Mapping[str, Any] | None = None,
                     standup_record: Mapping[str, Any] | None = None, arm: str | None = None) -> str | None:
    """Why the store under a leg's out root is NOT this leg's production run —
    ``None`` when it is. ``summary=None`` is the CERTIFIED-REFUSAL path (review
    P1: a store the driver refused before generating has a manifest and no
    summary): every manifest-side check runs unchanged; the two summary-bound
    ones (``item_set``, the summary's sampling numbers) have nothing to read and
    step aside — the manifest's own sampling block is checked against the row
    exactly as before. Every fact comes from the matrix row and the pinned
    inputs: the TAG's served model pin, the cell, the declaration class
    (``row.declared`` — the ``@thinking`` spellings, ``serve_out_of_support``
    present, ``response`` positions), the fitted class (the OOS override, the
    object pin, the transfer provenance), ``cap`` = this leg's rung (8,192 or
    16,384 — ``max_tokens_by_task``), the sampling row, the family's header
    tuples, and on a rung leg the ``cap_rung`` key bound to its SOURCE store
    (``rung_source_manifest``) and — when the planner's subset file is in hand
    (``rung_subset``) — EQUAL to the contract's own derivation from it
    (``contracts.cap_rung_identity``); a base leg carries no ``cap_rung``.
    ``thinking_on=False`` reads the overlap check's OFF greedy leg.
    ``standup_record`` (:func:`declared_served_positions_record`) is consulted
    ONLY where a DECLARED row's ``transfer_provenance.served_positions`` is null
    — the producer record gap of served-positions-null-diagnosis.md — and must
    then resolve the expected mode; a null on an UNDECLARED LENS row, or on a
    declared row with no record or a record that does not bind, refuses by
    name exactly as before; a stated value is held to the row's expected mode
    as before and never replaced. On a NATIVELY served undeclared row with the
    null (:func:`native_served_positions_gap`) the record is
    :func:`native_served_positions_record`'s: it must resolve the row's expected
    mode, the resolved mode must read NOT header-located through the contract
    (declared iff header-located, on the resolved fact), and the block's
    ``header_located`` is admitted only as the contract's fail-closed
    derivation from the null it sits beside (the named gap); no record, a
    record that does not bind, or any other pairing refuses by name.

    ``arm`` (llama-paper-rows P7 / P17 / I11) names the leg's whole arm
    explicitly; ``not_available`` reads the NO-SWITCH class: the hashed
    ``thinking.mode`` / ``thinking_by_task`` at that literal, NO
    ``out_of_support_thinking`` (the fitted cells are in support), NO
    ``thinking_header_ids`` (present iff any task is ON), the manifest's
    unhashed ``thinking`` block with ``enable_thinking`` NULL (no switch was
    sent — never ``false``), ``render`` ``native`` (the prefill render label,
    I11), no reasoning parser; the ``render_fact_evidence`` block with
    ``split_mode none`` and ``thinking_switch false``; a fitted cell's
    ``transfer_provenance`` served under the ``unspecified`` render with the
    family's ONE header under the one arm and NO authorised tuple. Every
    other arm reads exactly as before."""
    key, cell, fitted, declared = str(row["model_key"]), str(row["cell"]), bool(row["fitted"]), bool(row["declared"])
    arm = _arm_of_flags(thinking_on, arm)
    thinking_on = arm == C.THINKING_ON
    no_switch = arm == C.THINKING_NOT_AVAILABLE
    mode = sampling_mode or str(row["sampling_mode"])
    hf, rev = model_pin
    if manifest.get("model_id") != hf:
        return f"manifest model_id {manifest.get('model_id')!r} != the battery pin {hf!r} for {row['tag']!r}"
    if manifest.get("model_revision") != rev:
        return f"manifest model_revision {manifest.get('model_revision')!r} != the battery pin {rev!r} for {row['tag']!r}"
    if manifest.get("cell_id") != cell:
        return f"manifest cell_id {manifest.get('cell_id')!r} != the leg's cell {cell!r}"
    spell = C.cell_spellings(cell, declared=declared)
    route = _mapping(manifest.get(C.MANIFEST_KEY_ROUTE))
    if route.get("virtual_model") != spell.vmodel:
        return f"manifest route virtual_model {route.get('virtual_model')!r} != the {'declared' if declared else 'plain'} spelling {spell.vmodel!r}"
    tinycap = _mapping(manifest.get(C.MANIFEST_KEY_TINYCAP))
    if tinycap.get(C.IDENTITY_KEY_DIAGNOSTIC):
        return f"the store's identity carries {C.IDENTITY_KEY_DIAGNOSTIC}: {tinycap.get(C.IDENTITY_KEY_DIAGNOSTIC)!r} — a diagnostic store"
    if tinycap.get(C.IDENTITY_KEY_IDENTITY_VERSION) != C.TINYCAP_IDENTITY_VERSION_2:
        return f"{C.IDENTITY_KEY_IDENTITY_VERSION} {tinycap.get(C.IDENTITY_KEY_IDENTITY_VERSION)!r} != {C.TINYCAP_IDENTITY_VERSION_2}"
    if tinycap.get(C.IDENTITY_KEY_MAX_TOKENS_BY_TASK) != {TASK: int(cap)}:
        return f"max_tokens_by_task {tinycap.get(C.IDENTITY_KEY_MAX_TOKENS_BY_TASK)!r} != {{{TASK!r}: {int(cap)}}}"
    if _mapping(tinycap.get(C.IDENTITY_KEY_THINKING_BY_TASK)).get(TASK) != arm:
        return f"thinking_by_task {tinycap.get(C.IDENTITY_KEY_THINKING_BY_TASK)!r} does not run {TASK!r} {arm}"
    if _mapping(tinycap.get("thinking")).get("mode") != arm:
        return f"identity thinking.mode {_mapping(tinycap.get('thinking')).get('mode')!r} != {arm!r}"
    if no_switch:
        problem = _no_switch_manifest_problem(manifest, fact=fact, family=str(row["family"]))
        if problem:
            return problem
    if summary is not None and summary.get("item_set") != ITEM_SET:
        return f"summary item_set {summary.get('item_set')!r} != {ITEM_SET!r}"
    if _mapping(tinycap.get(C.IDENTITY_KEY_SAMPLES_BY_TASK)).get(TASK) != 1:
        return f"samples_by_task {tinycap.get(C.IDENTITY_KEY_SAMPLES_BY_TASK)!r} is not k = 1 on {TASK!r}"
    want_decl = C.expected_serve_out_of_support(lens=True, arm=C.THINKING_ON, out_of_support=True, header_located=True)
    decl = tinycap.get(C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT, M._ABSENT)
    if declared and decl != want_decl:
        return (f"identity {C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT} {None if decl is M._ABSENT else decl!r} != {want_decl!r} on a "
                "DECLARED cell")
    if not declared and decl is not M._ABSENT:
        return f"identity carries {C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT} = {decl!r} on an undeclared cell"
    oos = tinycap.get(C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING, M._ABSENT)
    if fitted and thinking_on and oos is not True:
        return f"a fitted job's identity must carry {C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING}: true (got {None if oos is M._ABSENT else oos!r})"
    if (not fitted or not thinking_on) and oos is not M._ABSENT:
        return f"identity carries {C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING} = {oos!r} on an in-support run"
    rung = tinycap.get(IDENTITY_KEY_CAP_RUNG, M._ABSENT)
    if int(cap) == CAP_RUNG and (thinking_on or no_switch):
        problem = cap_rung_problem(rung, source_manifest=rung_source_manifest, subset=rung_subset, trigger_threshold=row_trigger_threshold(row))
        if problem:
            return problem
    elif rung is not M._ABSENT:
        return f"identity carries {IDENTITY_KEY_CAP_RUNG} = {rung!r} on a non-rung leg"
    # no summary (a certified refusal): the manifest's sampling block stands in for the summary's numbers, so the
    # block-vs-row check is made once and the summary-vs-row check has nothing to disagree with
    problem = M.sampling_row_problem(manifest, summary if summary is not None else _mapping(manifest.get(C.MANIFEST_KEY_SAMPLING)),
                                     model_key=key, mode=mode, row=sampling_row)
    if problem:
        return problem
    if no_switch:
        # the "present iff any task is ON" rule: a no-switch store hashes NO header ids (the one render is certified
        # live, predicate (A″)); a present key is a switched family's identity under a mislabel
        if tinycap.get(C.IDENTITY_KEY_THINKING_HEADER_IDS, M._ABSENT) is not M._ABSENT:
            return (f"identity carries {C.IDENTITY_KEY_THINKING_HEADER_IDS} = {tinycap.get(C.IDENTITY_KEY_THINKING_HEADER_IDS)!r} on a "
                    f"{arm} store — the key is present iff a task runs thinking-ON (a family with no switch hashes none)")
    else:
        header_ids = _mapping(tinycap.get(C.IDENTITY_KEY_THINKING_HEADER_IDS))
        if M._header_ids_differ(header_ids, fact):
            return f"identity thinking_header_ids {dict(header_ids)!r} != the {row['family']!r} render fact's tuples"
    if fitted:
        pin = str(row["object_sha256"])
        digest = _mapping(manifest.get(C.MANIFEST_KEY_PROVENANCE)).get("artifact_sha256")
        if digest != pin:
            return f"provenance.artifact_sha256 {digest!r} != the (view, cell) pin {pin!r} — another object was served"
        by_cell = _mapping(route.get(C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL))
        if by_cell.get(spell.harness) != pin:
            return f"route.{C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL}[{spell.harness!r}] = {by_cell.get(spell.harness)!r} != the pin"
        tp = manifest.get(C.MANIFEST_KEY_TRANSFER_PROVENANCE)
        if not isinstance(tp, Mapping):
            return f"no {C.MANIFEST_KEY_TRANSFER_PROVENANCE} block on a fitted job's manifest"
        expect = str(row["served_positions_expected"])
        sp = tp.get("served_positions")
        if not declared and native_served_positions_gap(manifest):
            # the NATIVE record gap (RECORD_GAP_CAUSE_NATIVE_LENS_MAP_ONLY): the producer wrote null on every natively served
            # fitted cell and derived the block's `header_located` / `header_check_class` from that null, fail closed. The
            # fact is recovered from the native standup record beside the job (the engine-neutral map, else the routes[]
            # attestation), held to the row and re-read through the contract — or the leg refuses; never accepted as-is,
            # never defaulted from the engine's name. The manifest's located header is admitted ONLY as the named gap:
            # exactly the contract's fail-closed derivation from the null it sits beside.
            where = f"{C.MANIFEST_KEY_TRANSFER_PROVENANCE}.served_positions None != [{expect!r}] on a natively served UNDECLARED cell"
            if standup_record is None:
                return (f"{where} — resolvable only from the native standup record ({STANDUP_DIR_PARENT}/<jobdir>/{STANDUP_TASK_SUBDIR}/"
                        f"{STANDUP_ROUTING_TABLE}: the {SERVED_POSITIONS_KEY} map, else routes[].{ROUTE_POSITION_MODE_KEY}), and none was read")
            if standup_record.get("problem"):
                return f"{where}, and the standup record does not resolve it — {standup_record['problem']}"
            resolved = standup_record.get("resolved")
            unclassified = C.unclassified_native_modes(resolved)
            if unclassified:
                return (f"{where}; {NATIVE_POSITION_MODE_UNCLASSIFIED}: the standup record resolves {resolved!r} — {unclassified[0]!r} is a "
                        "native-vocabulary mode contracts.header_located cannot classify (it would read it header-located, fail closed)")
            if not _served_positions_match(resolved, expect):
                return f"{where}; the standup record resolves {resolved!r} != [{expect!r}]"
            if C.header_located(resolved) is not declared:
                return (f"{where}; the resolved positions {resolved!r} read header-located on an undeclared cell (declared iff header-located, "
                        "on the RESOLVED fact)")
            if tp.get("header_located") is not C.header_located(sp):
                return (f"{C.MANIFEST_KEY_TRANSFER_PROVENANCE}.header_located {tp.get('header_located')!r} beside the null served_positions is not the "
                        f"contract's fail-closed derivation from it ({C.header_located(sp)}) — not the named record gap; another producer's record")
        else:
            if tp.get("header_located") is not declared:
                return f"{C.MANIFEST_KEY_TRANSFER_PROVENANCE}.header_located {tp.get('header_located')!r} != {declared} (declared iff header-located)"
            if sp is None and declared:
                # the record gap (served-positions-null-diagnosis.md §1): the producer wrote null on every declared cell;
                # the fact is recovered from the standup record beside the job, or the leg refuses — never accepted as-is
                if standup_record is None:
                    return (f"{C.MANIFEST_KEY_TRANSFER_PROVENANCE}.served_positions None != [{expect!r}] on a DECLARED cell — resolvable only from "
                            f"the standup record ({STANDUP_DIR_PARENT}/<jobdir>/{STANDUP_TASK_SUBDIR}/{STANDUP_ROUTING_TABLE}), and none was read")
                if standup_record.get("problem"):
                    return (f"{C.MANIFEST_KEY_TRANSFER_PROVENANCE}.served_positions None != [{expect!r}] on a DECLARED cell, and the standup record "
                            f"does not resolve it — {standup_record['problem']}")
                if not _served_positions_match(standup_record.get("resolved"), expect):
                    return (f"{C.MANIFEST_KEY_TRANSFER_PROVENANCE}.served_positions None on a DECLARED cell; the standup record resolves "
                            f"{standup_record.get('resolved')!r} != [{expect!r}]")
            elif not _served_positions_match(sp, expect):
                return f"{C.MANIFEST_KEY_TRANSFER_PROVENANCE}.served_positions {sp!r} != [{expect!r}]"
        want_render = C.RENDER_REGIME_BY_ARM[arm] if no_switch else C.RENDER_REGIME_THINKING_ON
        if tp.get("served_render") != want_render:
            return f"{C.MANIFEST_KEY_TRANSFER_PROVENANCE}.served_render {tp.get('served_render')!r} != {want_render}"
        if tp.get("family") != row["family"]:
            return f"{C.MANIFEST_KEY_TRANSFER_PROVENANCE}.family {tp.get('family')!r} != the leg's family {row['family']!r}"
        if no_switch:
            # the ONE render, under the one arm (measured live through /v1/tokenize with no kwargs; the fact records it
            # under both id fields) — never the switch class's {off, on} pair
            got = M._header_ids_by_arm(tp.get("served_header_ids"))
            if got != M._single_render_ids(fact, arm=arm):
                return (f"{C.MANIFEST_KEY_TRANSFER_PROVENANCE}.served_header_ids {got!r} != the {row['family']!r} fact's single render under "
                        f"{arm!r} ({list(fact['off_ids'])})")
        elif M._header_ids_differ(_mapping(tp.get("served_header_ids")), fact):
            return f"{C.MANIFEST_KEY_TRANSFER_PROVENANCE}.served_header_ids != the {row['family']!r} render fact's tuples"
        name = manifest.get(C.MANIFEST_KEY_AUTHORISED_EXPERIMENT)
        if name != row["tuple_name"]:
            return f"manifest {C.MANIFEST_KEY_AUTHORISED_EXPERIMENT} {name!r} != the matrix row's tuple {row['tuple_name']!r}"
    elif manifest.get(C.MANIFEST_KEY_TRANSFER_PROVENANCE) is not None:
        return f"a {C.MANIFEST_KEY_TRANSFER_PROVENANCE} block on a non-fitted cell"
    return None


#: the per-row ARM CONTRACT (plan P5 (i′) / lane B B5) and the route class by the cell grammar are the MATH-500 module's —
#: ONE rule for both instruments (its grammar reading covers the appendix cells this matrix carries too)
#: the per-row ARM CONTRACT, the route class and the steered PATH (lens | lora | None) are the MATH-500
#: module's — ONE rule for both instruments
route_class_of, arm_contract_problem = M.route_class_of, M.arm_contract_problem
steered_path_of = M.steered_path_of


def _no_switch_manifest_problem(manifest: Mapping[str, Any], *, fact: Mapping[str, Any], family: str, task: str = TASK) -> str | None:
    """The NO-SWITCH arm's UNHASHED manifest facts (llama-paper-rows P7 / P17 /
    I11), each at the path the driver writes it — ``served_capability_run``'s
    manifest block, pinned by ``tests/test_served_capability_not_available.py``:
    ``thinking.enable_thinking`` NULL (no switch was sent; ``false`` would record
    a switch that was never there), ``thinking.mode`` the arm, ``thinking.render``
    ``native`` (the prefill render label; I11), ``thinking.reasoning_parser``
    None, ``thinking.by_task`` the arm on the row task; the family fact the
    driver copied (``render_fact_evidence``) with ``split_mode none`` and
    ``thinking_switch`` False; ``support`` in support; NO authorised tuple.
    ``task``: the row task ``thinking.by_task`` is read at — this instrument's by
    default; the MATH-500 reader passes its own (math500-steer: ONE rule for both)."""
    arm = C.THINKING_NOT_AVAILABLE
    block = _mapping(manifest.get(C.MANIFEST_KEY_THINKING))
    if not block:
        return f"the manifest carries no {C.MANIFEST_KEY_THINKING} block on a {arm} store"
    if block.get("enable_thinking", M._ABSENT) is not None:
        return (f"manifest {C.MANIFEST_KEY_THINKING}.enable_thinking {block.get('enable_thinking')!r} on a {arm} store — no switch is sent, "
                "so the block records null, never a boolean (P17)")
    if block.get("mode") != arm:
        return f"manifest {C.MANIFEST_KEY_THINKING}.mode {block.get('mode')!r} != {arm!r}"
    if block.get("render") != C.THINKING_RENDER_NATIVE:
        return f"manifest {C.MANIFEST_KEY_THINKING}.render {block.get('render')!r} != {C.THINKING_RENDER_NATIVE!r} (the prefill render label; I11)"
    if block.get("reasoning_parser") is not None:
        return f"manifest {C.MANIFEST_KEY_THINKING}.reasoning_parser {block.get('reasoning_parser')!r} on a family with no reasoning channel"
    if _mapping(block.get("by_task")).get(task) != arm:
        return f"manifest {C.MANIFEST_KEY_THINKING}.by_task {block.get('by_task')!r} does not run {task!r} {arm}"
    ev = _mapping(manifest.get(C.MANIFEST_KEY_RENDER_FACT_EVIDENCE))
    if not ev:
        return f"the manifest carries no {C.MANIFEST_KEY_RENDER_FACT_EVIDENCE} block on a {arm} store"
    if ev.get("family") != family:
        return f"manifest {C.MANIFEST_KEY_RENDER_FACT_EVIDENCE}.family {ev.get('family')!r} != the leg's family {family!r}"
    if ev.get("split_mode") != C.THINK_SPLIT_NONE or fact.get("split_mode") != C.THINK_SPLIT_NONE:
        return (f"manifest {C.MANIFEST_KEY_RENDER_FACT_EVIDENCE}.split_mode {ev.get('split_mode')!r} / the fact's {fact.get('split_mode')!r} != "
                f"{C.THINK_SPLIT_NONE!r} on a {arm} store")
    if ev.get("thinking_switch") is not False or fact.get("thinking_switch") is not False:
        return (f"manifest {C.MANIFEST_KEY_RENDER_FACT_EVIDENCE}.thinking_switch {ev.get('thinking_switch')!r} / the fact's "
                f"{fact.get('thinking_switch')!r} — the {arm} arm is for a family whose fact says the template has NO switch")
    if manifest.get("support", C.SUPPORT_IN_SUPPORT) != C.SUPPORT_IN_SUPPORT:
        return f"manifest support {manifest.get('support')!r} on a {arm} store — there is no ON render to be out of support of (P7)"
    if manifest.get(C.MANIFEST_KEY_AUTHORISED_EXPERIMENT) is not None:
        return (f"manifest {C.MANIFEST_KEY_AUTHORISED_EXPERIMENT} {manifest.get(C.MANIFEST_KEY_AUTHORISED_EXPERIMENT)!r} on a {arm} store — "
                "no out-of-support tuple exists for a family with no thinking switch")
    return None


def cap_rung_problem(rung: Any, *, source_manifest: Mapping[str, Any] | None,
                     subset: Mapping[str, Any] | None = None, trigger_threshold: float = GSM8K_TRIGGER_THRESHOLD) -> str | None:
    """The rung store's hashed ``cap_rung`` key: exactly
    :data:`CAP_RUNG_FIELDS`, ``source_cap`` 8,192, one of this instrument's ADMITTED
    rule strings (the current one, or the retired 3 % one a rung planned before the
    harmonisation of 2026-09-12 was hashed under), a 16-hex subset digest, and — when the source store is in hand — the
    ``source_identity_hash`` equal to the source manifest's identity hash; when
    the planner's VALIDATED subset file is in hand too, the whole key equal to
    the contract's own derivation from it (``contracts.cap_rung_identity``)."""
    if rung is M._ABSENT or not isinstance(rung, Mapping):
        return f"a rung leg's identity carries no {IDENTITY_KEY_CAP_RUNG} mapping (plan §4.4)"
    if set(rung) != set(CAP_RUNG_FIELDS):
        return f"{IDENTITY_KEY_CAP_RUNG} keys {sorted(rung)} != {sorted(CAP_RUNG_FIELDS)}"
    if rung.get("source_cap") != CAP_BASE:
        return f"{IDENTITY_KEY_CAP_RUNG}.source_cap {rung.get('source_cap')!r} != {CAP_BASE}"
    admitted = admitted_trigger_rules(trigger_threshold)
    if rung.get("rule") not in admitted:
        return (f"{IDENTITY_KEY_CAP_RUNG}.rule is not this instrument's rule string at the row's trigger ({trigger_pct_label(trigger_threshold)}) — "
                f"the current one, or the retired 3 % one a rung planned before 2026-09-12 carries")
    if not isinstance(rung.get("subset_sha16"), str) or not _HEX16.match(rung["subset_sha16"]):
        return f"{IDENTITY_KEY_CAP_RUNG}.subset_sha16 {rung.get('subset_sha16')!r} is not 16 hex"
    if source_manifest is not None:
        want = source_manifest.get(SOURCE_IDENTITY_HASH_FIELD)
        if rung.get("source_identity_hash") != want:
            return (f"{IDENTITY_KEY_CAP_RUNG}.source_identity_hash {rung.get('source_identity_hash')!r} != the source store's "
                    f"{SOURCE_IDENTITY_HASH_FIELD} {want!r}")
    if subset is not None:
        want_key = C.cap_rung_identity(subset)
        if dict(rung) != want_key:
            return (f"{IDENTITY_KEY_CAP_RUNG} {dict(rung)!r} != the contract's derivation from the planner's subset file "
                    f"{want_key!r} (contracts.cap_rung_identity)")
    return None


def same_store_tail(path: Any, tail: str) -> bool:
    """A relocation-aware root match (a pull relocates every root): the
    ABSOLUTE out root a producer recorded on the cluster names the same store
    as the reader's relative layout iff its trailing components ARE that layout
    (``controls/<TAG>/<cell slug>/<mode>``); the source's identity hash is held
    beside it, so a same-named store from another wave is still refused."""
    want = PurePosixPath(str(tail).strip("/")).parts
    have = PurePosixPath(str(path).rstrip("/")).parts
    return bool(want) and len(have) >= len(want) and tuple(have[-len(want):]) == tuple(want)


def item_subset_block_problem(manifest: Mapping[str, Any], *, subset: Mapping[str, Any], source_manifest: Mapping[str, Any],
                              source_out_root: str) -> str | None:
    """The rung manifest's UNHASHED ``item_subset`` block (the driver writes
    ``contracts.item_subset_block`` of the validated subset file it ran): EXACTLY
    the contract's derivation from the planner's subset — the same ids, digest,
    caps, task, source binding — with ``source_out_root`` naming the source leg
    (relocation-aware) and ``source_identity_hash`` the source store's."""
    block = manifest.get(MANIFEST_KEY_ITEM_SUBSET)
    if not isinstance(block, Mapping):
        return f"a rung leg's manifest carries no {MANIFEST_KEY_ITEM_SUBSET} block (contracts §4.4: the unhashed subset beside cap_rung)"
    if set(block) != set(ITEM_SUBSET_BLOCK_FIELDS):
        return f"{MANIFEST_KEY_ITEM_SUBSET} keys {sorted(block)} != {sorted(ITEM_SUBSET_BLOCK_FIELDS)}"
    want = C.item_subset_block(subset, path=str(block.get("path")))
    diff = sorted(k for k in ITEM_SUBSET_BLOCK_FIELDS if k != "path" and block.get(k) != want[k])
    if diff:
        return f"{MANIFEST_KEY_ITEM_SUBSET}.{diff[0]} {block.get(diff[0])!r} != the contract's derivation from the subset file {want[diff[0]]!r}"
    if block.get("source_identity_hash") != source_manifest.get(SOURCE_IDENTITY_HASH_FIELD):
        return f"{MANIFEST_KEY_ITEM_SUBSET}.source_identity_hash {block.get('source_identity_hash')!r} != the source store's"
    if not same_store_tail(block.get("source_out_root"), source_out_root):
        return f"{MANIFEST_KEY_ITEM_SUBSET}.source_out_root {block.get('source_out_root')!r} does not name the source leg {source_out_root!r}"
    return None


# --------------------------------------------------------------------------- #
# certification (certification_problem_g8k)                                   #
# --------------------------------------------------------------------------- #


def steering_bites_problems(bites: Mapping[str, Any], *, serving_identity: Any = None) -> list[str]:
    """Why the producer's steering-bites control on a LENS leg does not vouch for
    the served object (the block ``_served_evidence`` writes): not applicable or
    not verified, no differing probe, a bad probe count, a twin that is not the
    registry's zero cell for the engine (``none__lens0`` on a lens route,
    ``none__lora0`` on a natively served one — ``serving_identity`` absent keeps
    the lens twin, so every pre-engine caller reads byte-identically), a
    recorded ``error`` / ``stamp_error`` (the
    fidelity refusal, or a probe leg whose stamp triple was not its expected
    row), or counts that do not support each other (``n_differing`` above
    ``n_probes``). Shared by the ordinary certification (``certification_check``)
    and the certified-refusal admission (``certified_refusal_problem``, review
    round 2): a fitted run whose bites control FAILED is a serving failure the
    render refusal must not hide."""
    failures: list[str] = []
    if bites.get(C.OOS_EVIDENCE_BITES_APPLICABLE) is not True or bites.get(C.OOS_EVIDENCE_BITES_VERIFIED) is not True:
        failures.append(f"steering bites not verified on a steered leg (applicable={bites.get(C.OOS_EVIDENCE_BITES_APPLICABLE)!r}, "
                        f"verified={bites.get(C.OOS_EVIDENCE_BITES_VERIFIED)!r})")
    n_diff = bites.get(C.OOS_EVIDENCE_BITES_N_DIFFERING)
    gate = bites.get(C.OOS_EVIDENCE_BITES_LIVENESS_GATE, C.LENS_LIVENESS_TEXT)
    if gate == C.LENS_LIVENESS_LOGPROB:
        # a logprob-gated leg, read as the MATH-500 reader reads it: the LOGPROB statistic ruled this leg — the text
        # count is REPORTED, not required; the ruling block must be verified, carry the MEASURED floor, the divergence
        # clearing the bar and the not-a-dose-axis sentence (``M._logprob_liveness_problems``). Before this the GSM8K
        # reader knew only the text gate and would have refused a logprob-gated leg on `n_differing 0 < 1`.
        failures += M._logprob_liveness_problems(bites)
    elif gate == C.LENS_LIVENESS_TEXT:
        if not (isinstance(n_diff, int) and n_diff >= 1):
            failures.append(f"steering bites: n_differing {n_diff!r} < 1")
    else:
        failures.append(f"steering bites: liveness_gate {gate!r} is not one of {list(C.LENS_LIVENESS_MODES)}")
    n_probes = bites.get("n_probes")
    if not M._positive_int(n_probes):
        failures.append(f"steering bites: n_probes {n_probes!r} is not a positive count")
    elif isinstance(n_diff, int) and n_diff > n_probes:
        failures.append(f"steering bites: n_differing {n_diff} > n_probes {n_probes} — the counts do not support each other")
    try:
        want_twin = (C.twin_cell_id_for(serving_identity) if serving_identity is not None
                     else C.TWIN_CELL_ID)
    except ValueError as e:                          # a passthrough identity on a fitted route
        want_twin = None
        failures.append(f"steering bites twin cannot be resolved: {e}")
    if want_twin is not None and bites.get(C.OOS_EVIDENCE_BITES_TWIN) != want_twin:
        failures.append(f"steering bites twin {bites.get(C.OOS_EVIDENCE_BITES_TWIN)!r} != {want_twin!r} "
                        "(the twin is the ENGINE's zero cell)")
    for k in (C.OOS_EVIDENCE_BITES_ERROR, "stamp_error"):
        if k in bites:
            failures.append(f"steering bites carries {k}: {bites.get(k)!r}")
    return failures


def _declared_header_problems(block: Any, *, fact: Mapping[str, Any], family: str) -> list[str]:
    """Predicate (B) as the driver recorded it on a DECLARED header-located cell
    (``thinking_header_by_arm`` after ``_certify_header`` merged
    ``contracts.certify_thinking_header``'s verdict into it): a lens measurement
    through the live ``/v1/tokenize`` (no fallback), both arms measured, the
    measured / expected / attested header tuples equal to the family fact's,
    every certification flag true and no problem named. The block's
    ``verified`` is the header-INVARIANCE verdict (``served_capability_run
    .measure_thinking_header_by_arm``: "the honest answer to 'does the
    invariance hold?'"), so it is held to the FAMILY FACT, never read as a
    pass: an invariant header (gemma) reads ``verified: true``; a MOVING header
    (qwen) reads ``verified: false`` with the lifted mixed-arm refusal recorded
    beside it (``overridden_refusal``) — the out-of-support override's
    documented shape, which the header certification then judges."""
    name = EVIDENCE_THINKING_HEADER_BY_ARM
    if not isinstance(block, Mapping):
        return [f"no {name} block on a DECLARED cell — predicate (B) was not recorded"]
    out: list[str] = []
    for flag in ("lens", C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING):
        if block.get(flag) is not True:
            out.append(f"{name}.{flag} is {block.get(flag)!r}, not true — not a declared out-of-support lens serve's record")
    for flag in THINKING_HEADER_CERTIFIED_FLAGS:
        if block.get(flag) is not True:
            out.append(f"{name}.{flag} is {block.get(flag)!r}" + (" (null: the certification never ran)" if block.get(flag) is None else ""))
    if block.get("certification_problems") != []:
        out.append(f"{name}.certification_problems {block.get('certification_problems')!r} is not empty")
    if block.get(C.OOS_EVIDENCE_HEADER_SOURCE) != M.RENDER_SOURCE_SERVED:
        out.append(f"{name}.{C.OOS_EVIDENCE_HEADER_SOURCE} {block.get(C.OOS_EVIDENCE_HEADER_SOURCE)!r} != {M.RENDER_SOURCE_SERVED!r} — the header was not "
                   "measured through the served /v1/tokenize")
    if block.get("fallback_reason") is not None:
        out.append(f"{name}.fallback_reason {block.get('fallback_reason')!r} — a fallback measurement certifies nothing about the served route")
    if block.get("thinking_header_error") is not None:
        out.append(f"{name}.thinking_header_error {block.get('thinking_header_error')!r} — the proxy attested a partial header")
    measured = {str(a) for a in (block.get("measured_arms") or [])}
    if not {C.THINKING_OFF, C.THINKING_ON} <= measured:
        out.append(f"{name}.measured_arms {sorted(measured)} do not cover both arms")
    for field in (C.OOS_EVIDENCE_HEADER_IDS_BY_ARM, "expected_header_ids_by_arm"):
        if M._header_ids_differ(block.get(field), fact):
            out.append(f"{name}.{field} {dict(_mapping(block.get(field)))!r} != the {family!r} fact's tuples off {fact['off_ids']} / on {fact['on_ids']}")
    att = _mapping(block.get("attested_thinking_header"))
    if [list(att.get("off_ids") or []), list(att.get("on_ids") or [])] != [list(fact["off_ids"]), list(fact["on_ids"])]:
        out.append(f"{name}.attested_thinking_header {dict(att)!r} != the {family!r} fact's tuples — the proxy attested another header")
    moves = list(fact["off_ids"]) != list(fact["on_ids"])
    if block.get(C.OOS_EVIDENCE_HEADER_MOVES) is not moves:
        out.append(f"{name}.{C.OOS_EVIDENCE_HEADER_MOVES} {block.get(C.OOS_EVIDENCE_HEADER_MOVES)!r} != the {family!r} fact's {moves}")
    if block.get("invariant") is not (not moves):
        out.append(f"{name}.invariant {block.get('invariant')!r} != the {family!r} fact's {not moves}")
    overridden = block.get("overridden_refusal")
    if moves:
        if block.get("verified") is not False:
            out.append(f"{name}.verified is {block.get('verified')!r} on a MOVING-header family — the invariance verdict cannot hold there; "
                       "another producer's record")
        if not (isinstance(overridden, str) and overridden):
            out.append(f"{name}.overridden_refusal is {overridden!r} on a MOVING-header family — the lifted mixed-arm refusal was not recorded")
    else:
        if block.get("verified") is not True:
            out.append(f"{name}.verified is {block.get('verified')!r} on an INVARIANT-header family")
        if overridden is not None:
            out.append(f"{name}.overridden_refusal {overridden!r} present on an INVARIANT-header family")
    return out


def _declared_bites_problems(bites: Mapping[str, Any], *, expected_stamps: Sequence[str]) -> list[str]:
    """The steering-bites control on a DECLARED cell beyond
    :func:`steering_bites_problems`: the probe rendered thinking-ON (the
    declared class's probe), every steered probe reply stamped with
    the row's declared triple, the probe body the declared fragment."""
    out: list[str] = []
    if bites.get(C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM) != C.THINKING_ON:
        out.append(f"steering bites: {C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM} {bites.get(C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM)!r} != {C.THINKING_ON!r} on a "
                   "DECLARED cell")
    want = [str(s) for s in expected_stamps]
    steered = _mapping(bites.get("stamps_by_leg")).get("steered")
    if not isinstance(steered, list) or not steered:
        out.append(f"steering bites: no stamps_by_leg.steered on a DECLARED cell ({steered!r})")
    else:
        off = [i for i, s in enumerate(steered) if [_mapping(s).get(f) for f in C.EXPECTED_STAMP_TRIPLE_FIELDS] != want]
        if off:
            out.append(f"steering bites: {len(off)} steered probe stamp triple(s) != the declared {want} (first #{off[0]}: {steered[off[0]]!r})")
        n_probes = bites.get("n_probes")
        if isinstance(n_probes, int) and len(steered) != n_probes:
            out.append(f"steering bites: {len(steered)} steered stamps != n_probes {n_probes}")
    want_body = C.render_fragment(lens=True, arm=C.THINKING_ON, header_located=True)
    if dict(_mapping(bites.get("probe_extra_body"))) != dict(want_body):
        out.append(f"steering bites: probe_extra_body {bites.get('probe_extra_body')!r} != the declared fragment {want_body!r}")
    return out


def _no_switch_bites_problems(bites: Mapping[str, Any]) -> list[str]:
    """The steering-bites control on a fitted NO-SWITCH cell beyond
    :func:`steering_bites_problems` (llama-paper-rows P18, (C″)): the probe
    rendered under the family's ONE render — ``probe_render_arm not_available``,
    an EMPTY probe body fragment (no kwargs, no declaration), every probe reply
    on both legs stamped ``render_regime unspecified``."""
    arm = C.THINKING_NOT_AVAILABLE
    out: list[str] = []
    if bites.get(C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM) != arm:
        out.append(f"steering bites: {C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM} {bites.get(C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM)!r} != {arm!r} on a "
                   "no-switch cell (the single render)")
    want_body = C.render_fragment(lens=True, arm=arm, header_located=False)
    if dict(_mapping(bites.get("probe_extra_body"))) != dict(want_body):
        out.append(f"steering bites: probe_extra_body {bites.get('probe_extra_body')!r} != the no-switch fragment {want_body!r} (no kwargs are sent)")
    by_leg = _mapping(bites.get("stamps_by_leg"))
    for leg in ("steered", "scale0"):
        stamps = by_leg.get(leg)
        if not isinstance(stamps, list) or not stamps:
            out.append(f"steering bites: no stamps_by_leg.{leg} on a no-switch cell ({stamps!r})")
            continue
        off = [i for i, st in enumerate(stamps) if _mapping(st).get(C.STAMP_KEY_RENDER_REGIME) != RENDER_REGIME_NOT_AVAILABLE]
        if off:
            out.append(f"steering bites: {len(off)} {leg} probe stamp(s) whose render_regime != {RENDER_REGIME_NOT_AVAILABLE!r} "
                       f"(first #{off[0]}: {stamps[off[0]]!r}) — a switch rode a no-switch probe")
    return out


def _no_switch_render_probe_problems(evidence: Mapping[str, Any]) -> list[str]:
    """Predicate (A″)'s probe record beside ``render_certified`` (the driver's
    ``render_probe`` block under the no-switch class): exactly ONE render probe,
    keyed by the one arm, stamped ``unspecified``, its completion free of any
    think-tag literal and of a reasoning channel; NO OFF / ON probe."""
    from sisteer.capeval.tiers.think_split import THINK_CLOSE, THINK_OPEN

    arm = C.THINKING_NOT_AVAILABLE
    probe = evidence.get(M.EVIDENCE_RENDER_PROBE)
    if not isinstance(probe, Mapping):
        return [f"no {M.EVIDENCE_RENDER_PROBE} block on a no-switch cell — predicate (A″)'s probe was not recorded"]
    out: list[str] = []
    arms = {k for k in probe if k != "on_attempts"}
    if arms != {arm}:
        out.append(f"{M.EVIDENCE_RENDER_PROBE} carries probes under {sorted(arms)} — the no-switch class sends ONE probe under {arm!r}")
    rec = _mapping(probe.get(arm))
    if rec.get("render_regime") != RENDER_REGIME_NOT_AVAILABLE:
        out.append(f"{M.EVIDENCE_RENDER_PROBE}[{arm!r}].render_regime {rec.get('render_regime')!r} != {RENDER_REGIME_NOT_AVAILABLE!r}")
    content = str(rec.get("content") or "")
    stray = [t for t in (THINK_OPEN, THINK_CLOSE) if t in content]
    if stray or rec.get("reasoning"):
        out.append(f"{M.EVIDENCE_RENDER_PROBE}[{arm!r}] carries {stray or 'a reasoning channel'} on a family with no thinking channel — a stray")
    if probe.get("on_attempts"):
        out.append(f"{M.EVIDENCE_RENDER_PROBE}.on_attempts {probe.get('on_attempts')!r} on a no-switch cell — there is no ON draw")
    return out


def _no_switch_header_problems(block: Any, *, fact: Mapping[str, Any], family: str) -> list[str]:
    """The no-switch header measurement the driver records
    (``measure_no_switch_header`` -> ``thinking_header_by_arm``): the header
    measured ONCE through the live ``/v1/tokenize`` (no fallback), under the
    one arm, equal to the fact's single render, ``cert_class no_switch``,
    ``header_moves`` False, ``invariant`` and ``verified`` True."""
    name = EVIDENCE_THINKING_HEADER_BY_ARM
    arm = C.THINKING_NOT_AVAILABLE
    if not isinstance(block, Mapping):
        return [f"no {name} block on a no-switch cell — the header measurement was not recorded"]
    out: list[str] = []
    if block.get("cert_class") != C.CERT_CLASS_NO_SWITCH:
        out.append(f"{name}.cert_class {block.get('cert_class')!r} != {C.CERT_CLASS_NO_SWITCH!r}")
    if block.get(C.OOS_EVIDENCE_HEADER_SOURCE) != M.RENDER_SOURCE_SERVED:
        out.append(f"{name}.{C.OOS_EVIDENCE_HEADER_SOURCE} {block.get(C.OOS_EVIDENCE_HEADER_SOURCE)!r} != {M.RENDER_SOURCE_SERVED!r} — the header was not "
                   "measured through the served /v1/tokenize")
    if block.get("fallback_reason") is not None:
        out.append(f"{name}.fallback_reason {block.get('fallback_reason')!r} — a fallback measurement certifies nothing about the served route")
    if list(block.get("measured_arms") or []) != [arm]:
        out.append(f"{name}.measured_arms {block.get('measured_arms')!r} != [{arm!r}] — the no-switch class measures the header ONCE")
    got = M._header_ids_by_arm(block.get(C.OOS_EVIDENCE_HEADER_IDS_BY_ARM))
    if got != M._single_render_ids(fact, arm=arm):
        out.append(f"{name}.{C.OOS_EVIDENCE_HEADER_IDS_BY_ARM} {got!r} != the {family!r} fact's single render {list(fact['off_ids'])}")
    if block.get(C.OOS_EVIDENCE_HEADER_MOVES) is not False or block.get("invariant") is not True:
        out.append(f"{name}: header_moves {block.get(C.OOS_EVIDENCE_HEADER_MOVES)!r} / invariant {block.get('invariant')!r} — one render never moves")
    if block.get("verified") is not True or block.get("error") is not None:
        out.append(f"{name}.verified {block.get('verified')!r} / error {block.get('error')!r} — the measurement did not verify")
    return out


def certification_check(evidence: Mapping[str, Any], manifest: Mapping[str, Any], *, row: Mapping[str, Any],
                        fact: Mapping[str, Any], standup_record: Mapping[str, Any] | None = None,
                        arm: str = C.THINKING_ON, expected_stamps: Sequence[str] | None = None,
                        stamp_class: str = STAMP_CLASS_ROW) -> dict[str, Any]:
    """The certification predicates as the driver recorded them, PER CLASS:
    ``render_certified`` (A) on EVERY cell; on an UNDECLARED fitted cell the
    lens-position record (B′) (``served_positions`` == ``all`` and the object
    digest == the (view, cell) pin) and the lens-ON bite (C); on a DECLARED
    fitted cell predicate (B)'s record (:func:`_declared_header_problems`)
    instead — the driver never writes (B′)/(C) there (``served_capability_run
    .py`` 4726: ``object_on_run and not header_located``), so their PRESENCE
    refuses; the steering bites on every fitted cell, plus the declared
    probe's ON render and stamps (:func:`_declared_bites_problems`);
    ``route.lens_serve_declarations`` names the cell iff declared;
    ``route.lens_served_positions`` == [expected] on a fitted cell and null
    otherwise — on a DECLARED cell a null there is resolved through the same
    standup record ``identity_problem`` used (``standup_record``), with the
    source recorded; the family fact certified.

    ``arm`` (llama-paper-rows P18) selects the CERTIFICATION CLASS — the class
    is a function of the arm and the family fact (``contracts.cert_class_for``:
    ``not_available`` on a fact whose ``thinking_switch`` is False is the
    NO-SWITCH class; a switch arm on such a fact, or the no-switch arm on a
    switched fact, refuses by name). Under the no-switch class: (A″) the
    render measured ONCE through the live ``/v1/tokenize`` with no kwargs
    (``contracts.no_switch_render_certified``'s record; the header block; one
    render probe stamped ``unspecified``); on a FITTED cell (B′) verbatim (the
    class's ``unspecified`` triple), the EXPLICIT ``lens_on_bite`` inapplicability
    record — REQUIRED, never "not applicable by default" — and the ordinary
    steering-bites control under the single render; a declared no-switch cell
    is a matrix defect and refuses. Every switch-arm leg reads as before.

    ``expected_stamps`` / ``stamp_class`` (the E3 follow-up brief's item 6): the
    triple predicate (B′)'s record is held to — the caller's per-LEG class
    (:func:`leg_expected_stamps`), the row's by default; under the served-steer
    class the class's own requirements (:func:`served_steer_problems`) apply."""
    failures: list[str] = []
    want_stamps_leg = [str(s) for s in (expected_stamps if expected_stamps is not None else row["expected_stamps"])]
    fitted, declared, family = bool(row["fitted"]), bool(row["declared"]), str(row["family"])
    no_switch_fact = fact.get("thinking_switch") is False
    if arm == C.THINKING_NOT_AVAILABLE and not no_switch_fact:
        failures.append(f"arm {arm!r} on family {family!r}, whose render fact carries a thinking switch — the no-switch class is for a "
                        "switch-less family only (contracts.cert_class_for; P7)")
    if arm != C.THINKING_NOT_AVAILABLE and no_switch_fact:
        failures.append(f"arm {arm!r} on the switch-less family {family!r} — there is no thinking-ON class for it; serve it under "
                        f"{C.THINKING_NOT_AVAILABLE!r} (contracts.cert_class_for)")
    cert_class = C.CERT_CLASS_NO_SWITCH if (arm == C.THINKING_NOT_AVAILABLE and no_switch_fact) else C.CERT_CLASS_SWITCH
    if stamp_class == STAMP_CLASS_SERVED_STEER:
        failures += served_steer_problems(evidence, cert_class=cert_class)
    if fact.get("evidence_status") != C.RENDER_EVIDENCE_CERTIFIED:
        failures.append(f"family {family!r}: render fact evidence_status {fact.get('evidence_status')!r} — not certified")
    rc = evidence.get(M.EVIDENCE_RENDER_CERTIFIED)
    failures += M._render_certified_problems(rc, fact=fact, family=family)
    lp, bite = evidence.get(M.EVIDENCE_LENS_POSITION), evidence.get(M.EVIDENCE_LENS_ON_BITE)
    header = evidence.get(EVIDENCE_THINKING_HEADER_BY_ARM)
    bites = _mapping(evidence.get(M.EVIDENCE_BITES))
    route = _mapping(manifest.get(C.MANIFEST_KEY_ROUTE))
    spell = C.cell_spellings(str(row["cell"]), declared=declared)
    expect = str(row["served_positions_expected"]) if fitted else None
    # native-serving-default: the predicates are read under the ENGINE the store's own route states
    # (the MATH-500 reader's one rule, `certification_path_of`): a `+lora+` identity with the native engine stamped beside
    # it reads predicate (B′)'s EXPLICIT inapplicability record and no served-positions map; a `+lens+` identity, or a
    # route naming no identity (every lens store on record), reads the lens predicates byte for byte as before
    path, path_problems = M.certification_path_of(route)
    failures += path_problems
    native = path == M.STEERED_PATH_LORA
    if cert_class == C.CERT_CLASS_NO_SWITCH:
        failures += _no_switch_render_probe_problems(evidence)
        failures += _no_switch_header_problems(header, fact=fact, family=family)
        if declared:
            failures.append(f"a DECLARED cell under the no-switch arm — a family with no thinking switch has no declared render (P7); a matrix defect")
    if fitted and declared:
        # the DECLARED header-located class: predicate (B) + the ON bites; the undeclared cell's records must be ABSENT
        if native:
            failures.append(f"a DECLARED (header-located) row on a natively served store — the native engine serves undeclared "
                            f"all-position cells only (a declared configuration refuses at the standup); the store is not this row's")
        for name, block, pred in ((M.EVIDENCE_LENS_POSITION, lp, "(B′)"), (M.EVIDENCE_LENS_ON_BITE, bite, "(C)")):
            if block is not None:
                failures.append(f"{name} block on a DECLARED header-located cell — the driver never records predicate {pred} there; another producer's record")
        failures += _declared_header_problems(header, fact=fact, family=family)
        failures += steering_bites_problems(bites, serving_identity=route.get("serving_identity"))
        failures += _declared_bites_problems(bites, expected_stamps=row["expected_stamps"])
    elif fitted and native:
        # (B′) on the native engine: the driver's explicit inapplicability record, the MATH-500 reader's rule
        failures += M._native_lens_position_problems(lp)
        # class E on the native engine too: (C) is engine-neutral (the driver runs the ON bite against the base route on either engine), so a record admitted by the logprob verdict is read against the SAME leg's steering_bites here as well
        failures += M._lens_on_bite_problems(bite, route_vmodel=str(route.get("virtual_model")), cert_class=cert_class, bites=bites)
        failures += steering_bites_problems(bites, serving_identity=route.get("serving_identity"))
        if cert_class == C.CERT_CLASS_NO_SWITCH:
            failures += _no_switch_bites_problems(bites)
    elif fitted:
        pin = str(row["object_sha256"])
        if not isinstance(lp, Mapping):
            failures.append(f"no {M.EVIDENCE_LENS_POSITION} block on a fitted job — predicate (B′) was not recorded")
        elif lp.get("status") == C.NOT_APPLICABLE:
            # a key no lens record carries: the native engine's inapplicability record on a lens-served job
            failures.append(f"{M.EVIDENCE_LENS_POSITION} is the native inapplicability record (status {C.NOT_APPLICABLE!r}) on a "
                            "lens-served job — another engine's record; predicate (B′) was not attested")
        elif lp.get("verified") is False or "error" in lp:
            failures.append(f"{M.EVIDENCE_LENS_POSITION}: predicate (B′) FAILED as recorded — {lp.get('error')!r}")
        else:
            missing = [f for f in M.LENS_POSITION_ATTESTED_FIELDS if f not in lp]
            if missing:
                failures.append(f"{M.EVIDENCE_LENS_POSITION} lacks {missing}")
            if not _served_positions_match(lp.get("served_positions"), expect):
                failures.append(f"{M.EVIDENCE_LENS_POSITION}.served_positions {lp.get('served_positions')!r} != [{expect!r}]")
            if lp.get("header_located") is not declared:
                failures.append(f"{M.EVIDENCE_LENS_POSITION}.header_located {lp.get('header_located')!r} != {declared}")
            digest = _mapping(lp.get("object_digest"))
            if digest.get("scalar") != pin:
                failures.append(f"{M.EVIDENCE_LENS_POSITION}.object_digest.scalar {digest.get('scalar')!r} != the pin")
            if dict(_mapping(digest.get("by_cell"))) != {spell.harness: pin}:
                failures.append(f"{M.EVIDENCE_LENS_POSITION}.object_digest.by_cell {digest.get('by_cell')!r} != {{{spell.harness!r}: the pin}}")
            want_stamps = want_stamps_leg
            if lp.get("attested_policy") != want_stamps[1]:
                failures.append(f"{M.EVIDENCE_LENS_POSITION}.attested_policy {lp.get('attested_policy')!r} != the class's {want_stamps[1]!r}")
            if list(lp.get("expected_stamps") or []) != want_stamps:
                failures.append(f"{M.EVIDENCE_LENS_POSITION}.expected_stamps {lp.get('expected_stamps')!r} != {want_stamps}")
            if lp.get("own_key") != spell.harness:
                failures.append(f"{M.EVIDENCE_LENS_POSITION}.own_key {lp.get('own_key')!r} != {spell.harness!r}")
        # (C) on the switch class; (C″) — the EXPLICIT inapplicability record, REQUIRED — on the no-switch class
        # class E: a record admitted by the logprob verdict is read against the SAME leg's steering_bites
        failures += M._lens_on_bite_problems(bite, route_vmodel=str(route.get("virtual_model")), cert_class=cert_class, bites=bites)
        failures += steering_bites_problems(bites, serving_identity=route.get("serving_identity"))
        if cert_class == C.CERT_CLASS_NO_SWITCH:
            failures += _no_switch_bites_problems(bites)
    else:
        for name, block in ((M.EVIDENCE_LENS_POSITION, lp), (M.EVIDENCE_LENS_ON_BITE, bite)):
            if block is not None:
                failures.append(f"{name} block on a non-lens cell")
        if bites.get(C.OOS_EVIDENCE_BITES_APPLICABLE) is not False:
            failures.append(f"steering bites applicable={bites.get(C.OOS_EVIDENCE_BITES_APPLICABLE)!r} on a non-lens cell")
    decl = _mapping(route.get(C.ROUTE_KEY_SERVE_DECLARATIONS))
    if declared:
        want = C.expected_serve_out_of_support(lens=True, arm=C.THINKING_ON, out_of_support=True, header_located=True)
        if decl.get(spell.harness) != want and _mapping(decl.get(spell.harness)).get(C.SERVE_OUT_OF_SUPPORT_KEY) != want:
            failures.append(f"route.{C.ROUTE_KEY_SERVE_DECLARATIONS} does not name {spell.harness!r} with {want!r} on a DECLARED cell: {dict(decl)!r}")
    elif decl:
        failures.append(f"route.{C.ROUTE_KEY_SERVE_DECLARATIONS} names {sorted(decl)} on an undeclared cell")
    served = route.get(C.LENS_SERVED_POSITIONS_KEY, M._ABSENT)
    top = manifest.get(C.LENS_SERVED_POSITIONS_KEY, M._ABSENT)
    # the record gap on a DECLARED cell (served-positions-null-diagnosis.md §1.3): the driver wrote the null into the
    # route block and the manifest top level too; both are resolved from the standup record identity_problem read, never
    # from each other and never accepted as null
    resolved = (list(standup_record["resolved"]) if fitted and declared and isinstance(standup_record, Mapping)
                and not standup_record.get("problem") and standup_record.get("resolved") is not None else None)
    served_from_record = top_from_record = False
    if resolved is not None and served in (M._ABSENT, None):
        served, served_from_record = list(resolved), True
    if resolved is not None and top in (M._ABSENT, None):
        top, top_from_record = list(resolved), True
    if fitted and native:
        # a native adapter applies at EVERY position and its standup records NO served-positions map: the driver writes
        # null (the ENGINE's fact — `header_located = False` — never a missing lens record); the row must expect `all`
        # and a stated map is a lens fact on a native record (the MATH-500 reader's rule)
        if expect != M.NATIVE_SERVED_POSITIONS_EXPECTED:
            failures.append(f"the matrix row expects served positions {expect!r} on a natively served cell — a native adapter "
                            f"applies at every position ({M.NATIVE_SERVED_POSITIONS_EXPECTED!r}); a header-located row has no native standup")
        if served not in (M._ABSENT, None):
            # the driver's neutral-map lookup writes the ENGINE-NEUTRAL value into the lens-named field on a native leg too
            # (served_capability_run._standup_served_positions -> the manifest write at 6315-6316): admitted ONLY as the native
            # standup record's `served_positions[<harness>]` (the record the caller resolved, `native_served_positions_record`,
            # which already holds the manifest's stamp to the same map) — never the engine's name, never the stamp alone
            why = native_stamp_record_problem(served, standup_record=standup_record)
            if why is not None:
                failures.append(f"route.{C.LENS_SERVED_POSITIONS_KEY} = {served!r} on a natively served cell — a served-positions map is a "
                                f"lens fact; on a native record it is admitted only as the standup record's engine-neutral value, and {why}")
    elif fitted:
        if served is M._ABSENT or not _served_positions_match(served, expect):
            why = ""
            if declared and served in (M._ABSENT, None):
                why = (" — a DECLARED cell's null is resolvable only from the standup record, "
                       + (str(standup_record.get("problem")) if isinstance(standup_record, Mapping) and standup_record.get("problem") else "and none was read"))
            failures.append(f"route.{C.LENS_SERVED_POSITIONS_KEY} = {None if served is M._ABSENT else served!r} != [{expect!r}]{why}")
    elif served not in (M._ABSENT, None):
        failures.append(f"route.{C.LENS_SERVED_POSITIONS_KEY} = {served!r} on a non-lens cell")
    if top is not M._ABSENT and served is not M._ABSENT and top != served:
        failures.append(f"manifest {C.LENS_SERVED_POSITIONS_KEY} {top!r} != route.{C.LENS_SERVED_POSITIONS_KEY} {served!r}")
    header_summary = None
    if isinstance(header, Mapping):
        header_summary = {k: header.get(k) for k in ("lens", C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING, "verified", "invariant", C.OOS_EVIDENCE_HEADER_MOVES,
                                                      *THINKING_HEADER_CERTIFIED_FLAGS, "certification_problems", "measured_arms",
                                                      C.OOS_EVIDENCE_HEADER_IDS_BY_ARM, C.OOS_EVIDENCE_HEADER_SOURCE)}
        header_summary["overridden_refusal_present"] = bool(header.get("overridden_refusal"))
    return {"passed": not failures, "failures": failures, "pending_stamps": [], "cert_class": cert_class, "arm": arm,
            "render_fact_status": fact.get("evidence_status"),
            M.EVIDENCE_RENDER_CERTIFIED: dict(rc) if isinstance(rc, Mapping) else None,
            M.EVIDENCE_LENS_POSITION: dict(lp) if isinstance(lp, Mapping) else None,
            M.EVIDENCE_LENS_ON_BITE: dict(bite) if isinstance(bite, Mapping) else None,
            EVIDENCE_THINKING_HEADER_BY_ARM: header_summary,
            M.EVIDENCE_BITES: {k: bites.get(k) for k in (C.OOS_EVIDENCE_BITES_APPLICABLE, C.OOS_EVIDENCE_BITES_VERIFIED,
                                                         C.OOS_EVIDENCE_BITES_N_DIFFERING, "n_probes", C.OOS_EVIDENCE_BITES_TWIN,
                                                         C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM)},
            "route_lens_served_positions": (None if served is M._ABSENT else served),
            "route_lens_served_positions_source": ("standup_record" if served_from_record else "manifest"),
            "manifest_lens_served_positions_source": ("standup_record" if top_from_record else "manifest"),
            "standup_record": ({k: standup_record.get(k) for k in ("resolved", "resolved_from", "source", "problem", "jobdir", "job_id")}
                               if isinstance(standup_record, Mapping) else None),
            "route_lens_serve_declarations": dict(decl), "declared": declared, "served_positions_expected": expect}


# --------------------------------------------------------------------------- #
# the ledger by ACCEPTED attempt                                              #
# --------------------------------------------------------------------------- #


def _attempt_of(row: Mapping[str, Any], field: str) -> str | None:
    v = row.get(field)
    return v if isinstance(v, str) and v else None


def ledger_check(path: Path, *, rows: Sequence[Mapping[str, Any]] | None, header: Mapping[str, Any],
                 summary: Mapping[str, Any], attempted_ids: Sequence[str], vmodel: str, max_model_len: Any,
                 manifest: Mapping[str, Any], cap: int, thinking_on: bool = True, arm: str | None = None) -> dict[str, Any]:
    """The out root's usage ledger against the leg, by ATTEMPT:
    the ACCEPTED generation attempt's rows of :data:`TASK` are exactly one
    forwarded call per attempted draw (``expected_calls`` = the population —
    200, or the rung's subset size), 0 rejected, every ``error_class`` none,
    NO ``error_code`` (a ``thinking_on_*`` code is a named refusal), every row
    ``thinking`` == the arm at ``max_tokens_requested`` == ``cap``, the
    ``finish_reason == length`` count == the summary's ``n_at_cap``, the window
    holds, the binding names the standup; EVERY OTHER row belongs to a recorded
    attempt (attributed by id: forwarded / rejected / refusals per attempt) —
    a row carrying no ``attempt_id``, or one the manifest does not record, is a
    STRAY row and fails. Probe rows are held to "none rejected or errored".
    Also the decode rate and the completion-token quantiles of the accepted
    attempt (the ledger's own record).

    ``arm`` (llama-paper-rows P17): under ``not_available`` every accepted task
    row must read ``thinking`` NULL — the ledger producer (``usage_ledger
    .body_facts``) writes null when the body carries NO switch and a boolean
    only when one was supplied, so a ``false`` (or ``true``) refuses: a switch
    was sent to a family that has none. The two switch arms read the boolean
    as before."""
    failures: list[str] = []
    arm = _arm_of_flags(thinking_on, arm)
    want_thinking: Any = None if arm == C.THINKING_NOT_AVAILABLE else (arm == C.THINKING_ON)
    if rows is None:
        return {"available": False, "path": str(path), "passed": False,
                "failures": [f"no {path.name} at the out-root {path.parent}"], "n_calls": None, "error_code": {}}
    accepted = manifest.get(MANIFEST_KEY_ACCEPTED_ATTEMPT)
    recorded = recorded_attempts(manifest)
    if not isinstance(accepted, str) or accepted not in recorded:
        failures.append(f"the manifest names no recorded {MANIFEST_KEY_ACCEPTED_ATTEMPT} (got {accepted!r}; recorded {sorted(recorded)})")
    probe_rows = [r for r in rows if C.is_probe_task(r.get("task"))]
    other = [r for r in rows if r.get("task") != TASK and not C.is_probe_task(r.get("task"))]
    task_rows = [r for r in rows if r.get("task") == TASK]
    bad_probe = [r for r in probe_rows if r.get("rejected") is True or r.get("error_class") != C.LEDGER_ERROR_CLASS_NONE
                 or r.get("error_code") not in (None, "")]
    if bad_probe:
        failures.append(f"{len(bad_probe)} probe row(s) rejected or errored (first task {bad_probe[0].get('task')!r} "
                        f"error_code {bad_probe[0].get('error_code')!r})")
    stray = [r for r in rows if _attempt_of(r, LEDGER_ATTEMPT_ID_FIELD) not in recorded]
    if stray:
        failures.append(f"{len(stray)} ledger row(s) belong to no recorded attempt (first {LEDGER_ATTEMPT_ID_FIELD}="
                        f"{stray[0].get(LEDGER_ATTEMPT_ID_FIELD)!r}, task {stray[0].get('task')!r}) — a stray row (plan §4.5)")
    attributed: dict[str, dict[str, Any]] = {}
    for aid in recorded:
        mine = [r for r in task_rows if _attempt_of(r, LEDGER_ATTEMPT_ID_FIELD) == aid]
        probes = [r for r in probe_rows if _attempt_of(r, LEDGER_ATTEMPT_ID_FIELD) == aid]
        attributed[aid] = {"kind": recorded[aid].get(RESUME_ATTEMPT_KIND_FIELD), "completed": recorded[aid].get("completed"),
                           "n_task_rows": len(mine), "n_forwarded": sum(1 for r in mine if r.get("rejected") is not True),
                           "n_rejected": sum(1 for r in mine if r.get("rejected") is True),
                           "n_unique_samples": len({str(r.get("sample")) for r in mine if r.get("rejected") is not True}),
                           "error_code": dict(Counter(str(r.get("error_code")) for r in mine if r.get("error_code") not in (None, ""))),
                           "n_probe_rows": len(probes), "accepted": aid == accepted}
    acc_rows = [r for r in task_rows if _attempt_of(r, LEDGER_ATTEMPT_ID_FIELD) == accepted] if accepted in recorded else []
    expected = int(header["n_attempted_samples"])
    attempted = {str(i) for i in attempted_ids}
    forwarded = [r for r in acc_rows if r.get("rejected") is not True]
    by_sample = Counter(str(r.get("sample")) for r in forwarded)
    if len(by_sample) != expected:
        failures.append(f"the accepted attempt {accepted!r} forwarded {len(by_sample)} unique calls of {TASK} ({len(forwarded)} rows), expected {expected}")
    dup = sorted(s for s, c in by_sample.items() if c > 1)
    if dup:
        failures.append(f"the accepted attempt forwarded {len(dup)} sample(s) more than once (first {dup[:3]}) — one draw per item")
    unknown = sorted({M._sample_item(s) for s in by_sample} - attempted)
    if unknown:
        failures.append(f"{len(unknown)} forwarded sample(s) name items the record never attempted (first {unknown[:3]})")
    n_rejected = sum(1 for r in acc_rows if r.get("rejected") is True)
    if n_rejected:
        failures.append(f"censored: {n_rejected} rejected call(s) in the accepted attempt (a rejected draw is re-sampled by the client)")
    summary_rejected = summary.get(T.SUMMARY_REJECTED_FIELD)
    if summary_rejected is None:
        failures.append(f"the summary carries no {T.SUMMARY_REJECTED_FIELD}")
    elif summary_rejected != 0:
        failures.append(f"summary {T.SUMMARY_REJECTED_FIELD} {summary_rejected!r} != 0 (one pass, no retry)")
    classes = Counter(str(r.get("error_class")) for r in acc_rows)
    if acc_rows and set(classes) != {C.LEDGER_ERROR_CLASS_NONE}:
        failures.append(f"error_class counts {dict(classes)} — every accepted call must be {C.LEDGER_ERROR_CLASS_NONE!r}")
    codes = Counter(str(r.get("error_code")) for r in acc_rows if r.get("error_code") not in (None, ""))
    if codes:
        named = {k: v for k, v in codes.items() if k in C.THINKING_REFUSAL_CODES}
        failures.append(f"refused by name: error_code counts {dict(codes)}" + (f" — the proxy's thinking refusals {named}" if named else ""))
    thinks = sorted({r.get("thinking") for r in acc_rows}, key=lambda v: (v is None, bool(v)))
    if acc_rows and thinks != [want_thinking]:
        failures.append(f"thinking {thinks} — every row must be thinking: {want_thinking!r}"
                        + (" (no switch is sent under the no-switch arm; a boolean records a switch that was never there — P17)"
                           if want_thinking is None else ""))
    caps = sorted({r.get("max_tokens_requested") for r in acc_rows}, key=lambda v: (v is None, v))
    if acc_rows and caps != [int(cap)]:
        failures.append(f"max_tokens_requested {caps} != [{int(cap)}]")
    n_length = sum(1 for r in forwarded if r.get("finish_reason") == "length")
    n_at_cap = summary.get("n_at_cap")
    if n_at_cap is None:
        failures.append(f"the summary's n_at_cap is null while the ledger counts {n_length} finish_reason == length")
    elif n_length != n_at_cap:
        failures.append(f"ledger finish_reason == length {n_length} != summary n_at_cap {n_at_cap}")
    pre = [r.get("prompt_tokens_pre") for r in forwarded]
    pre_ints = [p for p in pre if isinstance(p, int) and not isinstance(p, bool)]
    ledger_max = max(pre_ints) if pre_ints and len(pre_ints) == len(forwarded) else None
    window: dict[str, Any] = {"prompt_max": ledger_max, "cap": int(cap), "max_model_len": max_model_len, "fits": None}
    if forwarded and ledger_max is None:
        failures.append("a ledger row carries no integer prompt_tokens_pre")
    if ledger_max is not None and isinstance(max_model_len, int):
        window["fits"] = ledger_max + int(cap) <= max_model_len
        if not window["fits"]:
            failures.append(f"max(prompt_tokens_pre) {ledger_max} + cap {int(cap)} > served max_model_len {max_model_len}")
    elif forwarded:
        failures.append("the window check needs the ledger max and the manifest's served max_model_len")
    binding = _mapping(manifest.get(M.MANIFEST_KEY_BINDING))
    if not binding:
        failures.append(f"the manifest carries no {M.MANIFEST_KEY_BINDING} block (the driver's verified endpoints)")
    elif binding.get("ledger_upstream") != binding.get("proxy_url"):
        failures.append(f"binding.ledger_upstream {binding.get('ledger_upstream')!r} != binding.proxy_url {binding.get('proxy_url')!r}")
    tok = [(r.get("usage_completion_tokens"), r.get("latency_s")) for r in forwarded]
    rates = [t / lat for t, lat in tok if isinstance(t, (int, float)) and isinstance(lat, (int, float)) and lat > 0]
    total_tok = sum(t for t, _lat in tok if isinstance(t, (int, float)))
    total_lat = sum(lat for _t, lat in tok if isinstance(lat, (int, float)))
    toks = sorted(t for t, _lat in tok if isinstance(t, (int, float)))
    quantiles = ({"p50": C.nearest_rank_percentile(toks, 0.5), "p90": C.nearest_rank_percentile(toks, 0.9),
                  "p99": C.nearest_rank_percentile(toks, 0.99), "max": toks[-1]} if toks else None)
    other_vmodels = Counter(str(r.get("vmodel")) for r in task_rows if r.get("vmodel") != vmodel)
    return {"available": True, "path": str(path), "accepted_attempt": accepted, "attempts": attributed,
            "n_calls": len(forwarded), "n_unique_samples": len(by_sample), "expected_calls": expected,
            "n_rejected": n_rejected, "summary_n_http_rejected_retried": summary_rejected,
            "n_task_rows_all_attempts": len(task_rows), "n_stray_rows": len(stray),
            "error_class": dict(classes), "error_code": dict(codes), "thinking": thinks, "max_tokens_requested": caps,
            "n_length": n_length, "summary_n_at_cap": n_at_cap, "window": window,
            "decode_tok_s_mean": (sum(rates) / len(rates)) if rates else None,
            "decode_tok_s_aggregate": (total_tok / total_lat) if total_lat else None,
            "completion_tokens_quantiles": quantiles, "n_other_task_rows": len(other), "n_probe_rows": len(probe_rows),
            "probe_rows_by_kind": dict(Counter(str(r.get("task")) for r in probe_rows)),
            "task_rows_on_other_vmodel": dict(other_vmodels), "passed": not failures, "failures": failures}


# --------------------------------------------------------------------------- #
# request evidence per class and per attempt (request_evidence_problem_g8k)   #
# --------------------------------------------------------------------------- #


def _reconcile_by_request_id(recs: Sequence[Mapping[str, Any]], ledger_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """The identity reconciliation of the records and the ledger's attempts
    (steps 2–5 of the producer's rule, restated here so it applies to a
    DECLARED leg and an OFF leg too — the producer's own rule refuses any
    declaration and any OFF task call by construction): every ledger row
    carries a request id and a wire digest; ids unique on each side; the two
    inventories are one set; the two sides of one id agree on the digest and
    the task / sample labels."""
    out: list[str] = []
    bare = [r for r in ledger_rows if not r.get(C.LEDGER_REQUEST_ID_FIELD) or not r.get(C.LEDGER_BODY_SHA256_FIELD)]
    if bare:
        out.append(f"{len(bare)} ledger row(s) carry no request id / body digest — a ledger that cannot be reconciled")
        return out
    ev = Counter(str(r.get("request_id")) for r in recs)
    led = Counter(str(r.get(C.LEDGER_REQUEST_ID_FIELD)) for r in ledger_rows)
    if any(n > 1 for n in ev.values()):
        out.append("a request_id appears on two request-evidence records")
    if any(n > 1 for n in led.values()):
        out.append("a request_id appears on two ledger rows")
    missing = sorted(set(led) - set(ev))
    extra = sorted(set(ev) - set(led))
    if missing:
        out.append(f"{len(missing)} ledger attempt(s) have no request-evidence record (first {missing[:2]})")
    if extra:
        out.append(f"{len(extra)} request-evidence record(s) have no ledger attempt (first {extra[:2]})")
    by_led = {str(r.get(C.LEDGER_REQUEST_ID_FIELD)): r for r in ledger_rows}
    disagree = 0
    for r in recs:
        row = by_led.get(str(r.get("request_id")))
        if row is None:
            continue
        if (r.get("body_sha256") != row.get(C.LEDGER_BODY_SHA256_FIELD) or str(r.get("task")) != str(row.get("task"))
                or str(r.get("sample")) != str(row.get("sample"))):
            disagree += 1
    if disagree:
        out.append(f"{disagree} attempt(s) whose record and ledger row disagree on the wire digest or the task / sample labels")
    return out


def _probe_arm(record: Mapping[str, Any], *, arm: str) -> str:
    """The render arm ONE probe record was sent under, as the inventory keys it:
    the switch arms read ``enable_thinking_value`` (``M._arm_of``, byte-identical);
    under ``not_available`` a record whose body carried NO switch
    (``enable_thinking_present`` False) is the no-switch arm's, and a record that
    DID carry one keeps its boolean's arm — a stray the inventory then refuses."""
    if arm == C.THINKING_NOT_AVAILABLE and record.get("enable_thinking_present") is False:
        return C.THINKING_NOT_AVAILABLE
    return M._arm_of(record.get("enable_thinking_value"))


def probe_inventory_per_run(evidence: Mapping[str, Any], *, route_vmodel: str, fitted: bool,
                           serving_identity: Any = None,
                            declared: bool, cert_class: str = C.CERT_CLASS_SWITCH) -> tuple[dict[tuple[str, str, str], int], str | None]:
    """The probe attempts ONE run of the driver sends, per ``(vmodel, render
    arm, probe kind)``, read off what the driver recorded — PER CLASS. An
    undeclared cell's inventory is the MATH module's (``M.probe_inventory_per_run``:
    two render probes, the bites probe thinking-OFF on the route and the twin,
    predicate (C)'s lens-ON bite on the lens and base routes). A DECLARED
    header-located cell sends two render probes and the bites probe under the
    arm ``steering_bites.probe_render_arm`` records (thinking-ON — the
    certification holds it to ON; the inventory reads the record) on
    the route and the twin, and NO lens-ON bite (predicate (C) is the
    undeclared cell's; the driver never sends it there — the four gsm8k-paired200
    declared cells at 8f5f1495 record exactly this inventory). Returns
    ``(inventory, problem)``; a problem means the inventory could not be
    derived and the caller refuses. ``cert_class`` :data:`contracts.CERT_CLASS_NO_SWITCH`
    (llama-paper-rows P18) is the no-switch class's EXACT inventory
    (``contracts.no_switch_probe_inventory`` through the MATH module): one
    render probe under ``not_available``, the bites probe on the route and the
    twin under it on a fitted cell, nothing else; a declared no-switch cell
    refuses (no such class)."""
    if cert_class == C.CERT_CLASS_NO_SWITCH:
        if declared:
            return ({(route_vmodel, C.THINKING_NOT_AVAILABLE, C.PROBE_KIND_RENDER): 1},
                    "a declared cell under the no-switch class — a family with no thinking switch has no declared render (P7)")
        return M.probe_inventory_per_run(evidence, route_vmodel=route_vmodel, fitted=fitted, cert_class=cert_class,
                                         serving_identity=serving_identity)
    if not declared:
        return M.probe_inventory_per_run(evidence, route_vmodel=route_vmodel, fitted=fitted,
                                         serving_identity=serving_identity)
    inv: dict[tuple[str, str, str], int] = {(route_vmodel, C.THINKING_OFF, C.PROBE_KIND_RENDER): 1,
                                            (route_vmodel, C.THINKING_ON, C.PROBE_KIND_RENDER): 1}
    if not fitted:
        return inv, "a declared cell is a fitted cell — an unfitted declared row is a matrix defect"
    bites = _mapping(evidence.get(M.EVIDENCE_BITES))
    n_probes, arm = bites.get("n_probes"), bites.get(C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM)
    if not M._positive_int(n_probes):
        return inv, f"{M.EVIDENCE_BITES}.n_probes {n_probes!r} is not a positive count — the bites probe's attempts cannot be counted"
    if arm not in (C.THINKING_OFF, C.THINKING_ON):
        return inv, f"{M.EVIDENCE_BITES}.{C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM} {arm!r} is not an arm — the bites probe's render cannot be placed"
    # the twin is the ENGINE's zero cell; absent identity keeps the lens twin
    twin_vm = C.twin_vmodel_for(serving_identity) if serving_identity is not None else C.twin_vmodel()
    for vm in (route_vmodel, twin_vm):
        inv[(vm, str(arm), C.PROBE_KIND_BITES)] = inv.get((vm, str(arm), C.PROBE_KIND_BITES), 0) + int(n_probes)
    return inv, None


def request_evidence_check(out_root: Path, *, manifest: Mapping[str, Any], evidence: Mapping[str, Any],
                           attempted_samples: Sequence[str], ledger_rows: Sequence[Mapping[str, Any]] | None,
                           route_vmodel: str, fitted: bool, declared: bool, cap: int, thinking_on: bool = True,
                           arm: str | None = None) -> dict[str, Any]:
    """The driver's versioned record of EVERY HTTP attempt, per class and per
    attempt: the producer's shape (a superset admitted for lane H's
    ``attempt_id``); the declaration PRESENT with value true on every task call
    of a DECLARED cell and on NO record of an undeclared one; every task call
    this task, this route, the arm's render, the cap; the ACCEPTED attempt's
    task records exactly the attempted samples (one each); every other record
    attributed to a recorded attempt (a stray record fails); the identity
    reconciliation with the ledger by request id — and on an undeclared ON leg
    the producer's own rule ``contracts.request_evidence_reconciliation`` as a
    cross-check; the probes labelled and the accepted attempt's probe
    inventory complete (the driver's per-run inventory).

    ``arm`` (llama-paper-rows P17 / P18): under ``not_available`` every TASK
    record reads ``enable_thinking_present`` False (no switch on the wire — the
    producer's whole-record rule ``contracts.request_evidence_no_switch_absent``
    is the cross-check, and its reconciliation runs under the no-switch arm),
    the probe inventory is the no-switch class's, and a probe record is placed
    under the arm its body shows (:func:`_probe_arm`). The switch arms read as
    before."""
    path = out_root / C.REQUEST_EVIDENCE_FILENAME
    failures: list[str] = []
    arm = _arm_of_flags(thinking_on, arm)
    no_switch = arm == C.THINKING_NOT_AVAILABLE
    cert_class = C.CERT_CLASS_NO_SWITCH if no_switch else C.CERT_CLASS_SWITCH
    if not path.is_file():
        return {"available": False, "path": str(path), "passed": False, "n_records": 0,
                "failures": [f"no {C.REQUEST_EVIDENCE_FILENAME} at the out-root {out_root} — declaration presence cannot be demonstrated"]}
    named = manifest.get(C.MANIFEST_KEY_REQUEST_EVIDENCE)
    if Path(str(named or "")).name != C.REQUEST_EVIDENCE_FILENAME:
        failures.append(f"manifest {C.MANIFEST_KEY_REQUEST_EVIDENCE} {named!r} does not name {C.REQUEST_EVIDENCE_FILENAME}")
    recs = list(_iter_jsonl(path))
    if not recs:
        failures.append("no request-evidence records — absence of evidence is never evidence of absence")
    # ifeval-canonical-battery (P29): a record is held to the field set of the
    # version it was WRITTEN under (2 = the GSM8K stores; 3 adds the wire's
    # sampling terms) — C.REQUEST_EVIDENCE_ACCEPTED_VERSIONS
    def _want_fields(r):
        return set(C.REQUEST_EVIDENCE_FIELDS_V2 if r.get("request_evidence_version") == 2 else C.REQUEST_EVIDENCE_FIELDS)
    off_shape = [i for i, r in enumerate(recs) if not _want_fields(r) <= set(r)]
    if off_shape:
        failures.append(f"{len(off_shape)} record(s) lack the producer's fields (first #{off_shape[0]}; "
                        f"v2: {sorted(C.REQUEST_EVIDENCE_FIELDS_V2)}, v3: {sorted(C.REQUEST_EVIDENCE_FIELDS)})")
    versions = Counter(r.get("request_evidence_version") for r in recs)
    if recs and not set(versions) <= set(C.REQUEST_EVIDENCE_ACCEPTED_VERSIONS):
        failures.append(f"request_evidence_version values {dict(versions)} not within "
                        f"{list(C.REQUEST_EVIDENCE_ACCEPTED_VERSIONS)}")
    kinds = Counter(r.get("kind") for r in recs)
    unknown_kinds = sorted(str(k) for k in kinds if k not in C.REQUEST_EVIDENCE_KINDS)
    if unknown_kinds:
        failures.append(f"record kind(s) {unknown_kinds} outside {list(C.REQUEST_EVIDENCE_KINDS)}")
    bad_digest = [r for r in recs if not C.is_sha256_hex(r.get("body_sha256"))]
    if bad_digest:
        failures.append(f"{len(bad_digest)} record(s) whose body_sha256 is not a 64-hex digest")
    task_recs = [r for r in recs if r.get("kind") == C.REQUEST_EVIDENCE_KIND_TASK]
    probe_recs = [r for r in recs if r.get("kind") == C.REQUEST_EVIDENCE_KIND_PROBE]
    if declared:
        undeclared = [r for r in task_recs if r.get("petri_thinking_on_present") is not True or r.get("petri_thinking_on_value") is not True]
        if undeclared:
            failures.append(f"{len(undeclared)} task call(s) of a DECLARED cell carried no {C.OOS_THINKING_DECLARATION_KEY}: true "
                            f"(first {[r.get('sample') for r in undeclared[:3]]})")
    else:
        with_decl = [r for r in recs if r.get("petri_thinking_on_present") is not False or r.get("petri_thinking_on_value") is not None]
        if with_decl:
            failures.append(f"{len(with_decl)} record(s) carry {C.OOS_THINKING_DECLARATION_KEY} on an UNDECLARED cell "
                            f"(first {[(r.get('kind'), r.get('task'), r.get('sample')) for r in with_decl[:3]]})")
    attempted = {str(s) for s in attempted_samples}
    other_tasks = Counter(str(r.get("task")) for r in task_recs if r.get("task") != TASK)
    if other_tasks:
        failures.append(f"task record(s) naming other tasks {dict(other_tasks)} — this run files {TASK!r} only")
    other_vm = Counter(str(r.get("vmodel")) for r in task_recs if r.get("vmodel") != route_vmodel)
    if other_vm:
        failures.append(f"task record(s) on another vmodel {dict(other_vm)} != the leg's route {route_vmodel!r}")
    if no_switch:
        wrong_arm = [r for r in task_recs if r.get("enable_thinking_present") is not False or r.get("enable_thinking_value") is not None]
        if wrong_arm:
            failures.append(f"{len(wrong_arm)} task call(s) CARRIED {C.CHAT_TEMPLATE_KWARGS_KEY}.{C.ENABLE_THINKING_KEY} on a {arm} leg (first "
                            f"{[(r.get('sample'), r.get('enable_thinking_value')) for r in wrong_arm[:3]]}) — the no-switch arm sends none (P7 / P17)")
        absent_rule = C.request_evidence_no_switch_absent(recs) if recs else None
        if absent_rule is not None:
            failures.append(f"the producer's no-switch rule refuses: {absent_rule}")
    else:
        wrong_arm = [r for r in task_recs if r.get("enable_thinking_present") is not True or r.get("enable_thinking_value") is not bool(thinking_on)]
        if wrong_arm:
            failures.append(f"{len(wrong_arm)} task call(s) without enable_thinking present and {bool(thinking_on)} (first {[r.get('sample') for r in wrong_arm[:3]]})")
    off_cap = [r for r in task_recs if r.get("max_tokens") != int(cap)]
    if off_cap:
        failures.append(f"{len(off_cap)} task call(s) whose wire body requests max_tokens {sorted({r.get('max_tokens') for r in off_cap}, key=str)} != {int(cap)}")
    # -- attempt scoping: the accepted attempt's task records are exactly the attempted samples; others attributed --
    accepted = manifest.get(MANIFEST_KEY_ACCEPTED_ATTEMPT)
    recorded = recorded_attempts(manifest)
    stray = [r for r in recs if _attempt_of(r, REQUEST_EVIDENCE_ATTEMPT_ID_FIELD) not in recorded]
    if stray:
        failures.append(f"{len(stray)} request-evidence record(s) belong to no recorded attempt (first {REQUEST_EVIDENCE_ATTEMPT_ID_FIELD}="
                        f"{stray[0].get(REQUEST_EVIDENCE_ATTEMPT_ID_FIELD)!r}) — a stray record (plan §4.5)")
    acc_task = [r for r in task_recs if _attempt_of(r, REQUEST_EVIDENCE_ATTEMPT_ID_FIELD) == accepted]
    acc_by_sample = Counter(str(r.get("sample")) for r in acc_task if r.get("task") == TASK)
    unknown = sorted(set(acc_by_sample) - attempted)
    if unknown:
        failures.append(f"{len(unknown)} accepted-attempt task record(s) name samples the record never attempted (first {unknown[:3]})")
    absent = sorted(attempted - set(acc_by_sample))
    if absent:
        failures.append(f"no accepted-attempt request-evidence record for {len(absent)} attempted call(s) (first {absent[:3]})")
    twice = sorted(s for s, n in acc_by_sample.items() if n > 1)
    if twice:
        failures.append(f"{len(twice)} attempted sample(s) recorded twice in the accepted attempt (first {twice[:3]})")
    per_attempt = {aid: {"n_task_records": sum(1 for r in task_recs if _attempt_of(r, REQUEST_EVIDENCE_ATTEMPT_ID_FIELD) == aid),
                         "n_probe_records": sum(1 for r in probe_recs if _attempt_of(r, REQUEST_EVIDENCE_ATTEMPT_ID_FIELD) == aid),
                         "accepted": aid == accepted} for aid in recorded}
    # -- the identity reconciliation with the ledger, by request id --
    n_ledger_rows = None
    producer_rule: str | None = None
    if ledger_rows is None:
        failures.append("the ledger's attempts are unavailable — the records cannot be reconciled per attempted call")
    else:
        n_ledger_rows = len(ledger_rows)
        failures += _reconcile_by_request_id(recs, ledger_rows)
        if not declared and (thinking_on or no_switch):
            producer_rule = C.request_evidence_reconciliation(recs, ledger_rows, filed=[(TASK, s.split("/", 1)[0]) for s in attempted],
                                                              expected_enable_thinking=(None if arm == THINKING_ARM_NOT_AVAILABLE else True))
            if producer_rule is not None:
                failures.append(f"the producer's reconciliation rule refuses: {producer_rule}")
    # -- the probes: labelled, on their own vmodel; the accepted attempt's inventory complete; strays refused --
    bad_label = [r for r in probe_recs if M.probe_kind_of(r.get("task")) not in C.PROBE_KINDS or not M._probe_sample_ok(r.get("sample"), r.get("vmodel"))]
    if bad_label:
        failures.append(f"{len(bad_label)} probe record(s) not labelled as the driver's wrapper labels them (first "
                        f"{(bad_label[0].get('task'), bad_label[0].get('sample'), bad_label[0].get('vmodel'))})")
    inventory, inv_problem = probe_inventory_per_run(
        evidence, route_vmodel=route_vmodel, fitted=fitted, declared=declared, cert_class=cert_class,
        serving_identity=M._mapping(manifest.get(C.MANIFEST_KEY_ROUTE)).get("serving_identity"))
    if inv_problem:
        failures.append(f"the probe inventory cannot be derived: {inv_problem}")
    acc_probe = [r for r in probe_recs if _attempt_of(r, REQUEST_EVIDENCE_ATTEMPT_ID_FIELD) == accepted]
    got = Counter((str(r.get("vmodel")), _probe_arm(r, arm=arm), str(M.probe_kind_of(r.get("task")))) for r in acc_probe)
    got_all = Counter((str(r.get("vmodel")), _probe_arm(r, arm=arm), str(M.probe_kind_of(r.get("task")))) for r in probe_recs)
    stray_probe = {f"{vm}/{arm}/{kind}": n for (vm, arm, kind), n in got_all.items() if (vm, arm, kind) not in inventory}
    if stray_probe:
        failures.append(f"probe record(s) to a route / render / kind the driver never sends: {stray_probe}")
    table: dict[str, dict[str, int]] = {}
    for (vm, arm, kind), per_run in inventory.items():
        n = got.get((vm, arm, kind), 0)
        table[f"{vm}/{arm}/{kind}"] = {"expected_per_run": per_run, "records_accepted_attempt": n, "records_all": got_all.get((vm, arm, kind), 0)}
        if n < per_run:
            failures.append(f"probe shortfall in the accepted attempt on {vm}/{arm}/{kind}: {n} record(s) < {per_run}")
    return {"available": True, "path": str(path), "n_records": len(recs), "n_task_records": len(task_recs),
            "n_probe_records": len(probe_recs), "n_ledger_rows": n_ledger_rows, "accepted_attempt": accepted,
            "per_attempt": per_attempt, "n_stray_records": len(stray), "producer_rule": producer_rule,
            "probe_inventory": table, "declared": declared, "cert_class": cert_class, "passed": not failures, "failures": failures}


# --------------------------------------------------------------------------- #
# the .eval re-derivation (RETAINED in kind)                                  #
# --------------------------------------------------------------------------- #


#: the `none` split's extra outcome-defining fact (``contracts.TASK_OUTCOME_ITEM_FIELDS_NO_SWITCH``; the writer files it on a
#: no-switch store ONLY), compared draw by draw when the record carries it — every other store's comparison is byte-identical
OUTCOME_FIELD_STRAY_CLOSE_TAG = "stray_close_tag"


def _log_fact(result: Any, field: str) -> Any:
    v = getattr(result, field)
    return bool(v) if field in ("at_cap", "think_block_present", "think_block_unterminated", "stray_open_tag", "content_empty_with_reasoning",
                                OUTCOME_FIELD_STRAY_CLOSE_TAG) else v


def per_item_outcome_check(results: Sequence[Any], items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The retained log against the outcome record DRAW BY DRAW, keyed by
    ``(item_id, sample_idx)``: the parent's score availability / value
    (``tinycap_oos_transfer.per_item_log_check``, unchanged) AND every
    outcome-defining producer fact (:data:`OUTCOME_FIELDS_FROM_LOG` — the stop
    reason and the at-cap bit the residual mask reads, the completion and prompt
    token counts the ladder's redraw outcomes and the truncation block read,
    the think-block facts the primary policy reads). A count-preserving swap of
    ``at_cap`` / ``stop_reason`` / ``completion_tokens`` between two items
    survives every marginal and the parent's score comparison; it fails here
    on both items by name. The log side is the writer's own reading
    (``inspect_served.ItemResult`` -> ``outcome_record``), so the comparison
    is exact."""
    per = dict(T.per_item_log_check(results, items))
    by_key = {(str(r.item_id), int(getattr(r, "sample_idx", 0))): r for r in results}
    record = {(str(it["item_id"]), int(it.get("sample_idx", 0))): it for it in items}
    field_mismatches: list[dict[str, Any]] = []
    for key in sorted(k for k in record if k in by_key):
        r, it = by_key[key], record[key]
        fields = OUTCOME_FIELDS_FROM_LOG + ((OUTCOME_FIELD_STRAY_CLOSE_TAG,) if OUTCOME_FIELD_STRAY_CLOSE_TAG in it else ())
        for f in fields:
            log_side, rec_side = _log_fact(r, f), (bool(it.get(f)) if isinstance(it.get(f), bool) or f == "at_cap" else it.get(f))
            if log_side != rec_side:
                field_mismatches.append({"item_id": key[0], "sample_idx": key[1], "field": f, "log": log_side, "record": rec_side})
    per.update({"fields_compared": list(OUTCOME_FIELDS_FROM_LOG), "n_field_mismatched": len(field_mismatches),
                "field_mismatches": field_mismatches, "agrees": bool(per["agrees"]) and not field_mismatches})
    return per


def per_item_outcome_detail(per: Mapping[str, Any], *, show: int = 5) -> str | None:
    if per["agrees"]:
        return None
    parts = [T._per_item_detail({**per, "agrees": not (per["mismatches"] or per["missing_in_log"] or per["extra_in_log"])})]
    fm = per.get("field_mismatches") or []
    if fm:
        shown = "; ".join(f"{m['item_id']}/{m['sample_idx']}.{m['field']}: log {m['log']!r} != record {m['record']!r}" for m in fm[:show])
        more = len(fm) - min(show, len(fm))
        parts.append(f"{len(fm)} outcome-defining field(s) differ between the log and the record ({shown}" + (f"; and {more} more" if more else "") + ")")
    return "; ".join(p for p in parts if p) or None


def eval_cross_check(cell_dir: Path, header: Mapping[str, Any], items: Sequence[Mapping[str, Any]], *,
                     max_tokens: int, task: str = TASK, task_name: str = TASK_NAME,
                     accepted_attempt: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The authoritative per-item SCORE RE-DERIVATION from the retained ``.eval``
    log (the outcomes file is the producer's own record, not an independent
    reader): the ONE log the PRODUCER's own rule qualifies
    (``inspect_served.reusable_log`` — status success, the attempted count and
    ids, generated at ``max_tokens``, the cap this leg's identity was validated
    to) re-derived through ``results_from_log`` and held equal to the record
    item by item (score availability and value per ``(item_id, sample_idx)``)
    and in its counts. Also every draw's two text channels (``channels``) for
    the at-cap classification — the unscored case-(b) draws have no row — and
    the samples' input texts (``inputs``) for the overlap pairing's byte check.
    Logs present but none qualifying: NOT RUN with the reason (an unwaivable
    pending on the scoring path); no log: the record stands alone.

    Two refusals the reader adds over the producer's rule (2026-09-11: a
    local pull merged two clusters' trees, so each store held two complete
    logs of one leg — one tree's record beside another tree's newer log — and
    the newest-by-mtime rule read the wrong one as a per-item disagreement):
    MORE THAN ONE qualifying log in a log dir is NOT RUN, every candidate named
    with its mtime, never the newest taken (:func:`qualifying_logs`); and, when
    the caller passes the manifest's ``accepted_attempt`` entry, the one log
    must have been created inside that attempt's ``[started, ended]`` window
    (:func:`log_outside_attempt`) — the binding the record itself does not
    carry (nothing in it names its log). ``None`` = no binding asked for."""
    k = int(header["samples_per_item"])
    log_dirs = [C.inspect_log_dir(cell_dir, task, idx) for idx in range(k)]
    entries = {d: M._eval_log_entries(d) for d in log_dirs}
    present = any(entries.values())
    if not present:
        return {**M._eval_check_not_run(f"no Inspect .eval log under {cell_dir / C.INSPECT_LOGS_SUBDIR} — the outcome record stands alone",
                                        present=False, selected=[]), "channels": None, "inputs": None}
    try:
        from inspect_ai.log import (
            read_eval_log,  # noqa: F401 — a venv probe: inspect_served reads the .eval log through inspect_ai lazily
        )

        from sisteer.capeval.tiers import inspect_served as isv
    except ImportError as e:            # pragma: no cover - venv-dependent
        return {**M._eval_check_not_run(f"inspect_ai / inspect_served not importable here ({e})", present=True, selected=[]),
                "channels": None, "inputs": None}
    results: list[Any] = []
    texts: list[str] = []
    channels: dict[str, tuple[str | None, str | None]] = {}
    inputs: dict[str, str] = {}
    selected: list[str] = []
    bound: list[dict[str, Any]] = []
    for idx, d in enumerate(log_dirs):
        ids = [str(it["item_id"]) for it in items if int(it.get("sample_idx", 0)) == idx]
        if not entries[d]:
            return {**M._eval_check_not_run(f"no Inspect .eval log under {d} for sample {idx} while another sample's log is present",
                                            present=True, selected=selected), "channels": None, "inputs": None}
        candidates = qualifying_logs(d, expected_n=len(ids), expected_ids=ids, expected_max_tokens=max_tokens)
        if len(candidates) > 1:
            listing = "; ".join(f"{p} (mtime {_utc_iso(datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc))})" for p in candidates)
            return {**M._eval_check_not_run(
                f"{len(candidates)} retained .eval logs under {d} each qualify as the completed log of this task's population under the "
                f"producer's own rule — {len(candidates)} generations of one leg, and nothing in the record names the one its rows were "
                f"filed from; refusing to take the newest by mtime: {listing}", present=True, selected=selected), "channels": None, "inputs": None}
        log = isv.reusable_log(d, expected_n=len(ids), expected_ids=ids, expected_max_tokens=max_tokens)
        if log is None:
            why = isv.retained_log_reason(d, expected_n=len(ids), expected_ids=ids, expected_max_tokens=max_tokens)
            return {**M._eval_check_not_run(
                f"no retained .eval log under {d} is the completed log of this task's population — the producer's own selection "
                f"rule (inspect_served.reusable_log: status success, {len(ids)} samples, no errored sample, the attempted ids, "
                f"generated at completion cap {max_tokens}): {why}", present=True, selected=selected), "channels": None, "inputs": None}
        if accepted_attempt is not None:
            why = log_outside_attempt(log, accepted_attempt)
            if why is not None:
                return {**M._eval_check_not_run(why, present=True, selected=selected), "channels": None, "inputs": None}
            created = log_created(log)
            bound.append({"log": Path(str(log.location)).name, "created": _utc_iso(created) if created is not None else None})
        try:
            results.extend(isv.results_from_log(log, sample_idx=idx, primary_scorer=isv.PRIMARY_SCORER.get(task_name)))
        except Exception as e:                                   # noqa: BLE001 — the reader's refusal is the reason
            return {**M._eval_check_not_run(f"{log.location}: {type(e).__name__}: {str(e)[:200]}", present=True, selected=selected),
                    "channels": None, "inputs": None}
        selected.append(Path(str(log.location)).name)
        for smp in (log.samples or ()):
            out = getattr(smp, "output", None)
            comp = getattr(out, "completion", None) if out is not None else None
            if comp is not None:
                texts.append(str(comp))
            channels[str(smp.id)] = isv.output_channels(out)
            inp = getattr(smp, "input", None)
            if isinstance(inp, str):
                inputs[str(smp.id)] = inp
            elif isinstance(inp, list):
                # a chat-shaped input: the LAST user message's text is the question
                for msg in reversed(inp):
                    if getattr(msg, "role", None) == "user":
                        content = getattr(msg, "content", None)
                        inputs[str(smp.id)] = content if isinstance(content, str) else getattr(msg, "text", "")
                        break
    derived = T.counts_from_results(results)
    diffs = {f: (v, header.get(f)) for f, v in derived.items() if header.get(f) is not None and header.get(f) != v}
    count_detail = "; ".join(f"{f}: log {a} != record {b}" for f, (a, b) in diffs.items()) or None
    per = per_item_outcome_check(results, items)
    per_detail = per_item_outcome_detail(per)
    window = attempt_window(accepted_attempt) if accepted_attempt is not None else None
    binding = None if accepted_attempt is None else {
        "attempt_id": accepted_attempt.get(RESUME_ATTEMPT_ID_FIELD), "started": _utc_iso(window[0]) if window else None,
        "ended": _utc_iso(window[1]) if window else None, "logs": bound}
    return {"available": True, "reason": None, "agrees": not diffs and per["agrees"], "log_present": True,
            "selected_logs": selected, "attempt_binding": binding, "derived": derived, "count_detail": count_detail, "per_item": per,
            "detail": "; ".join(x for x in (count_detail, per_detail) if x) or None, "text": T.text_signature(texts, None),
            "at_cap_tails": M.eval_at_cap_tails(items, channels), "channels": channels, "inputs": inputs}


# --------------------------------------------------------------------------- #
# the ladder record and the rung subset                                       #
# --------------------------------------------------------------------------- #


def read_ladder_record(out_root: Path, cell_dir: Path) -> tuple[Path, Any, str | None]:
    """``(path, record | None, problem)`` — the cell's ``cap_ladder.json`` where
    its WRITERS put it: lane B's ``ladder_plan.py`` files the adapter's record
    at ``<out_root>/cap_ladder.json`` (the layout of THIS instrument's wave);
    Evals' ``cap_ladder.write_cell_record`` files a record beside
    ``manifest.json`` (``<cell_dir>/cap_ladder.json``). Both are read; two
    records that disagree refuse (two writers of one fact); none -> ``None``
    at the planner's path."""
    planner, evals = Path(out_root) / LADDER_RECORD_NAME, Path(cell_dir) / LADDER_RECORD_NAME
    docs = {p: read_json_if_file(p) for p in (planner, evals)}
    present = [p for p, d in docs.items() if d is not None]
    if not present:
        return planner, None, None
    if len(present) == 2 and docs[planner] != docs[evals]:
        return planner, docs[planner], f"two {LADDER_RECORD_NAME} records disagree: {planner} (the planner's) vs {evals} (beside the manifest)"
    return present[0], docs[present[0]], None


def _id_list(record: Mapping[str, Any], field: str) -> set[str] | None:
    v = record.get(field)
    return {str(i) for i in v} if isinstance(v, list) else None


def ladder_record_problem(record: Any, *, n: int, at_cap_ids: Sequence[str],
                          source_manifest: Mapping[str, Any] | None = None,
                          trigger_threshold: float = GSM8K_TRIGGER_THRESHOLD) -> str | None:
    """Why the cell's ``cap_ladder.json`` is not THIS instrument's record over
    THIS store — the ADAPTER's record (``inspect_rung.classify_store``), read
    in the adapter's own spellings: its schema and every instrument field
    (:data:`LADDER_RECORD_FIELDS`), the serialized rule / threshold / report
    line (:data:`GSM8K_TRIGGER_RULE` — never Evals' 1 % truncated-share
    string), ``n`` and ``n_at_cap`` equal to the store's, the exit in the
    adapter's vocabulary and consistent with the trigger (share > the threshold iff the
    exit is a triggered one), every at-cap draw classified (``eligible_ids`` +
    ``non_terminating_ids`` + ``boundary_ids`` == the at-cap draws, each count
    its list's length, the boundary class empty at k = 1), ``escalate`` the
    exit's bit, ``next_cap`` 16,384 exactly on the two exits that reached the
    transport check (``escalate`` / ``transport_refused`` — the adapter's
    ``decide`` names the rung on both) and null otherwise, the transport
    verdict present and agreeing with the exit on those two, and the
    ``source`` block naming the source store (task, identity hash, not a rung)."""
    if not isinstance(record, Mapping):
        return f"no {LADDER_RECORD_NAME} record (the ladder planner did not run on this cell)"
    if record.get("schema") != LADDER_RECORD_SCHEMA:
        return f"{LADDER_RECORD_NAME}.schema {record.get('schema')!r} != {LADDER_RECORD_SCHEMA!r}"
    missing = [f for f in LADDER_RECORD_FIELDS if f not in record]
    if missing:
        return f"{LADDER_RECORD_NAME} lacks the adapter's field(s) {missing}"
    rule = IR.gsm8k_trigger_rule(trigger_threshold)
    if record.get("rule") != rule:
        return (f"{LADDER_RECORD_NAME}.rule is not this instrument's rule string at the row's trigger ({trigger_pct_label(trigger_threshold)}) "
                f"(got {str(record.get('rule'))[:60]!r}...)")
    if record.get("threshold") != trigger_threshold or record.get("report_line") != GSM8K_REPORT_LINE:
        return f"{LADDER_RECORD_NAME} threshold / report_line {record.get('threshold')!r} / {record.get('report_line')!r} != {trigger_threshold} / {GSM8K_REPORT_LINE}"
    if record.get("n") != int(n):
        return f"{LADDER_RECORD_NAME}.n {record.get('n')!r} != the store's {int(n)}"
    at_cap = {str(i) for i in at_cap_ids}
    if record.get("n_at_cap") != len(at_cap):
        return f"{LADDER_RECORD_NAME}.n_at_cap {record.get('n_at_cap')!r} != the store's {len(at_cap)} at-cap draws"
    exit_ = record.get("exit")
    if exit_ not in LADDER_EXITS:
        return f"{LADDER_RECORD_NAME}.exit {exit_!r} not in {LADDER_EXITS}"
    share = len(at_cap) / int(n) if n else None
    if not M._same_number(record.get("share_at_cap"), None if share is None else round(share, 6)):
        return f"{LADDER_RECORD_NAME}.share_at_cap {record.get('share_at_cap')!r} != {share!r}"
    if record.get("share_at_cap_over_1pct") is not (share is not None and share > GSM8K_REPORT_LINE):
        return f"{LADDER_RECORD_NAME}.share_at_cap_over_1pct {record.get('share_at_cap_over_1pct')!r} disagrees with the share {share!r}"
    triggered = share is not None and share > trigger_threshold
    if triggered and exit_ in EXITS_NOT_TRIGGERED:
        return f"{LADDER_RECORD_NAME}.exit {exit_!r} on a TRIGGERED cell ({len(at_cap)} of {n} at cap > {trigger_pct_label(trigger_threshold)})"
    if not triggered and exit_ not in EXITS_NOT_TRIGGERED:
        return f"{LADDER_RECORD_NAME}.exit {exit_!r} on an un-triggered cell ({len(at_cap)} of {n} at cap <= {trigger_pct_label(trigger_threshold)})"
    if exit_ == CL.EXIT_CLEAN and at_cap:
        return f"{LADDER_RECORD_NAME}.exit clean with {len(at_cap)} at-cap draws"
    elig, nonterm, boundary = (_id_list(record, f) for f in (LADDER_ELIGIBLE_IDS, LADDER_NON_TERMINATING_IDS, LADDER_BOUNDARY_IDS))
    if elig is None or nonterm is None or boundary is None:
        return f"{LADDER_RECORD_NAME}: {LADDER_ELIGIBLE_IDS} / {LADDER_NON_TERMINATING_IDS} / {LADDER_BOUNDARY_IDS} are not all lists"
    if not elig <= at_cap:
        return f"{LADDER_RECORD_NAME}.{LADDER_ELIGIBLE_IDS} names {len(elig - at_cap)} id(s) that are not at cap"
    if not nonterm <= at_cap:
        return f"{LADDER_RECORD_NAME}.{LADDER_NON_TERMINATING_IDS} names {len(nonterm - at_cap)} id(s) that are not at cap"
    if elig & nonterm:
        return f"{LADDER_RECORD_NAME}: {len(elig & nonterm)} id(s) both eligible and non-terminating"
    counts = {"n_eligible": len(elig), "n_non_terminating": len(nonterm), "n_boundary": len(boundary)}
    for f, want in counts.items():
        v = record.get(f)
        if isinstance(v, bool) or not isinstance(v, int) or v != want:
            return f"{LADDER_RECORD_NAME}.{f} {v!r} != its id list's length {want}"
    if boundary:
        return f"{LADDER_RECORD_NAME}.{LADDER_BOUNDARY_IDS} names {len(boundary)} id(s) — the class is empty at k = 1 (the adapter's BOUNDARY_RULE_K1)"
    if len(elig) + len(nonterm) + len(boundary) != len(at_cap):
        return (f"{LADDER_RECORD_NAME}: eligible {len(elig)} + non_terminating {len(nonterm)} + boundary {len(boundary)} != at-cap {len(at_cap)} — "
                "not every at-cap draw is classified")
    if record.get("escalate") is not (exit_ == CL.ESCALATE):
        return f"{LADDER_RECORD_NAME}.escalate {record.get('escalate')!r} disagrees with exit {exit_!r}"
    transport = record.get("transport")
    if exit_ in EXITS_NAMING_A_RUNG:
        if record.get("next_cap") != CAP_RUNG:
            return f"{LADDER_RECORD_NAME}.next_cap {record.get('next_cap')!r} != {CAP_RUNG} on exit {exit_!r} (the rung the transport check was run for)"
        if not isinstance(transport, Mapping) or transport.get("ok") is not (exit_ == CL.ESCALATE):
            return f"{LADDER_RECORD_NAME}.exit {exit_!r} without a transport verdict agreeing with it (transport={transport!r})"
        if exit_ == CL.ESCALATE and not elig:
            return f"{LADDER_RECORD_NAME}.exit escalate with no eligible item"
    else:
        if record.get("next_cap") is not None:
            return f"{LADDER_RECORD_NAME}.next_cap {record.get('next_cap')!r} on exit {exit_!r} (no rung is named)"
        if transport is not None:
            return f"{LADDER_RECORD_NAME}.transport {transport!r} on exit {exit_!r} (the transport check was never reached)"
        if exit_ == EXIT_TRIGGERED_NO_ELIGIBLE and elig:
            return f"{LADDER_RECORD_NAME}.exit triggered_no_eligible with {len(elig)} eligible item(s)"
    source = record.get("source")
    if not isinstance(source, Mapping):
        return f"{LADDER_RECORD_NAME}.source is not the adapter's source block"
    if source.get("task") != TASK or source.get("is_rung") is not False:
        return f"{LADDER_RECORD_NAME}.source {dict(source)!r} is not a base store's record of {TASK!r}"
    if source_manifest is not None and source.get("identity_hash") != source_manifest.get(SOURCE_IDENTITY_HASH_FIELD):
        return (f"{LADDER_RECORD_NAME}.source.identity_hash {source.get('identity_hash')!r} != this store's "
                f"{SOURCE_IDENTITY_HASH_FIELD} {source_manifest.get(SOURCE_IDENTITY_HASH_FIELD)!r} — a record over another store")
    return None


def rung_subset_problem(subset: Any, *, record: Mapping[str, Any], source_manifest: Mapping[str, Any],
                        source_out_root: str | None = None, trigger_threshold: float = GSM8K_TRIGGER_THRESHOLD) -> str | None:
    """The planner's ``rung_subset.json`` against the ladder record and the
    SOURCE store: the CONTRACT's own validator first (``contracts.
    validate_item_subset`` — exactly its fields, its schema, a row task,
    ``source_cap < cap <= ceiling``, ids == the eligible difference, a PASSED
    transport verdict), then this instrument's facts: the standard row task,
    source cap 8,192 / cap 16,384, one of this instrument's ADMITTED rule / threshold
    pairs (the current, or the retired 3 % pair a subset planned before 2026-09-12
    carries — its threshold must be the one its own rule states),
    ``ids`` == the record's ``eligible_ids``, the counts and the transport
    verdict equal to the record's, ``source_identity_hash`` == the source
    manifest's identity hash, and ``source_out_root`` naming the source leg
    RELOCATION-AWARE (the planner records the cluster's absolute root; the
    reader holds a local pull — :func:`same_store_tail`)."""
    if not isinstance(subset, Mapping):
        return f"no {RUNG_SUBSET_NAME} for an escalated cell"
    try:
        validate_item_subset(subset, where=RUNG_SUBSET_NAME)
    except C.ItemSubsetError as e:
        return str(e)
    if subset.get("task") != TASK:
        return f"{RUNG_SUBSET_NAME}.task {subset.get('task')!r} != the standard row task {TASK!r}"
    if subset.get("source_cap") != CAP_BASE or subset.get("cap") != CAP_RUNG:
        return f"{RUNG_SUBSET_NAME} source_cap / cap {subset.get('source_cap')!r} / {subset.get('cap')!r} != {CAP_BASE} / {CAP_RUNG}"
    admitted = admitted_trigger_rules(trigger_threshold)
    if subset.get("rule") not in admitted:
        return (f"{RUNG_SUBSET_NAME}.rule is not this instrument's at the row's trigger ({trigger_pct_label(trigger_threshold)}) — "
                f"the current one, or the retired 3 % one a subset planned before 2026-09-12 carries")
    if not M._same_number(subset.get("threshold"), admitted[subset["rule"]]):
        return (f"{RUNG_SUBSET_NAME}.threshold {subset.get('threshold')!r} is not the threshold its own rule states "
                f"({admitted[subset['rule']]})")
    if {str(i) for i in subset["ids"]} != (_id_list(record, LADDER_ELIGIBLE_IDS) or set()):
        return f"{RUNG_SUBSET_NAME}.ids != the ladder record's {LADDER_ELIGIBLE_IDS}"
    if subset.get("source_identity_hash") != source_manifest.get(SOURCE_IDENTITY_HASH_FIELD):
        return f"{RUNG_SUBSET_NAME}.source_identity_hash {subset.get('source_identity_hash')!r} != the source store's {SOURCE_IDENTITY_HASH_FIELD}"
    for f in ("n_at_cap", "n_non_terminating", "n_boundary"):
        if subset.get(f) != record.get(f):
            return f"{RUNG_SUBSET_NAME}.{f} {subset.get(f)!r} != the ladder record's {record.get(f)!r}"
    if subset.get("transport_check") != record.get("transport"):
        return f"{RUNG_SUBSET_NAME}.transport_check != the ladder record's transport verdict"
    if source_out_root is not None and not same_store_tail(subset.get("source_out_root"), source_out_root):
        return (f"{RUNG_SUBSET_NAME}.source_out_root {subset.get('source_out_root')!r} does not name the source leg "
                f"{source_out_root!r} (its trailing components are another store's)")
    return None


def redraw_outcome(source: Mapping[str, Any], new: Mapping[str, Any], *, source_cap: int = CAP_BASE) -> dict[str, Any]:
    """One redrawn item's outcome, from the two outcome records' ``stop_reason``
    and ``completion_tokens`` — NEVER by re-tokenizing:
    ``still_at_cap`` when the 16,384 draw stopped at its cap; ``boundary_landing``
    when it terminated with fewer tokens than the source cap (the k = 1
    retrospective boundary evidence — this draw would have fit at 8,192);
    ``extended`` when it terminated at or beyond the source cap."""
    new_at_cap = bool(new.get("at_cap")) or new.get("stop_reason") == C_STOP_AT_CAP
    n_new = new.get("completion_tokens")
    if new_at_cap:
        outcome = REDRAW_STILL_AT_CAP
    elif isinstance(n_new, int) and n_new < int(source_cap):
        outcome = REDRAW_BOUNDARY_LANDING
    else:
        outcome = REDRAW_EXTENDED
    return {"source_cap": int(source_cap), "new_cap": CAP_RUNG, "source_completion_tokens": source.get("completion_tokens"),
            "new_completion_tokens": n_new, "source_stop_reason": source.get("stop_reason"), "new_stop_reason": new.get("stop_reason"),
            "still_at_cap": outcome == REDRAW_STILL_AT_CAP, "extended": outcome == REDRAW_EXTENDED,
            "boundary_landing": outcome == REDRAW_BOUNDARY_LANDING, "outcome": outcome,
            "source_scored": bool(source.get("scored")), "new_scored": bool(new.get("scored")),
            "source_value": source.get("value"), "new_value": new.get("value")}


def eligible_from_text(at_cap_ids: Sequence[str], texts: Mapping[str, str | None]) -> dict[str, Any]:
    """This reader's INDEPENDENT derivation of the eligible set from the draws'
    text under Evals' degeneracy rule (``cap_ladder.degeneracy``: whitespace
    4-gram repetition over the whole text or its last 200 words) — the
    non-terminating class the planner excluded — with the boundary class empty
    at k = 1. A draw whose text is not in hand is counted apart (``unavailable``)
    and the derivation is then PARTIAL (``complete`` false): it compares to the
    planner's set only when complete."""
    non_terminating, eligible, unavailable = [], [], []
    per: dict[str, Any] = {}
    for i in sorted(str(x) for x in at_cap_ids):
        text = texts.get(i)
        if text is None:
            unavailable.append(i)
            per[i] = None
            continue
        deg = CL.degeneracy(text)
        per[i] = deg
        (non_terminating if deg["degenerate"] else eligible).append(i)
    return {"rule": CL.DEGENERATE_RULE, "n_at_cap": len(at_cap_ids), "eligible": eligible, "non_terminating": non_terminating,
            "boundary": [], "unavailable": unavailable, "complete": not unavailable, "per_draw": per}


# --------------------------------------------------------------------------- #
# the population (RETAINED in kind)                                           #
# --------------------------------------------------------------------------- #


def population_problem(attempted_ids: Sequence[str], canonical_ids: Sequence[str], *, subset_ids: Sequence[str] | None = None) -> str | None:
    """A base leg's attempted ids == the canonical 200 (derived ONCE, independently
    of any store); a rung leg's == its subset file's ids, which are ⊆ canonical."""
    have = {str(i) for i in attempted_ids}
    canon = {str(i) for i in canonical_ids}
    if len(canon) != N_ITEMS:
        return f"the supplied canonical id set has {len(canon)} ids, not {N_ITEMS}"
    if subset_ids is None:
        if have != canon:
            return (f"the attempted ids are not the canonical GSM8K paired-200 set — {len(have - canon)} attempted but not canonical "
                    f"(first {sorted(have - canon)[:3]}), {len(canon - have)} canonical but not attempted (first {sorted(canon - have)[:3]})")
        return None
    want = {str(i) for i in subset_ids}
    if not want <= canon:
        return f"the rung subset names {len(want - canon)} id(s) outside the canonical set"
    if have != want:
        return (f"the rung's attempted ids != its subset — {len(have - want)} attempted but not in the subset, "
                f"{len(want - have)} in the subset but not attempted")
    return None


def read_json_if_file(p: Path) -> Any:
    return _read_json(p) if Path(p).is_file() else None


def read_store(out_root: Path) -> tuple[Path | None, dict[str, Any] | None, str | None]:
    """``(cell_dir, manifest, reason)`` of the single store under an out root."""
    if not Path(out_root).is_dir():
        return None, None, f"no out-root at {out_root}"
    cd, why = T._cell_dir(Path(out_root))
    if cd is None:
        return None, None, why
    manifest = completion.read_store_manifest(cd)
    if manifest is None:
        return cd, None, f"{cd / 'manifest.json'} is not a JSON object"
    return cd, manifest, None

