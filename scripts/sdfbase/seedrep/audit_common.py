#!/usr/bin/env python3
"""The Qwen seed-replicate battery — the ONE set of store readers every post-wave audit table shares
(`request_evidence_audit.py`, `determinism_check.py`, `engine_identity_table.py`,
`utilisation_table.py`, `coverage_table.py`, `at_cap_table.py`).

Three rules hold everywhere in this module and in the six tables built on it:

* **Absence refuses, never blanks.** A missing input — an attempt root without its `ATTEMPT.json`, a store
  whose rows carry no `request_id`, a seed rule that has not landed — is an :class:`AuditRefusal` naming the
  file and the field, and the table exits non-zero without writing a partial table. No column is ever
  quietly `null` (memory: "absence is the default failure mode", "null flag renders as clean").
* **The store's own writer names the task.** Every leg's canonical `X-Tinycap-Task` is derived HERE
  from the keys the store's rows carry — never from a static table — and the header side
  (`request_evidence_client.store_task_name`) must agree with :func:`canonical_task` by construction
  (:func:`check_client_task_names` refuses drift when that module is importable).
* **Symbols, never line numbers.** Everything read off the tree is imported by name from
  `sisteer.capeval.tinycap.contracts` and the store writers (`static_common`, `bon`, `pyrit_adapters`,
  `sampled_controls`, `misalign.runner`, `petri_runner`, `judged`, `parity`, `native_standup`).

Attempt-root layout read (`RS/stores/<leg>/<view>/<shard>/` = the pulled ACCEPTED attempt root):
`ATTEMPT.json` (first), `COMPLETE.json` (last), `RESULT`, the serve dir (`endpoint.json`, `cells_served.json`,
`serving_attestation.json`, `routing_table.json`, `standup_jobs.jsonl`, `vllm_server.log[.gz]`),
`usage_ledger.jsonl`, `request_evidence.jsonl` (with the additive `body_b64` wire bytes) + `response_evidence.jsonl` (with the additive `text_sha256`),
`gpu_dmon.log`, `vllm_metrics.jsonl`, `driver_inflight.jsonl`, `determinism.json`, `modal_evidence.json`,
and the drivers' stores EXACTLY as they write them. Files are located BY NAME under the root (bounded walk);
zero or several matches refuse — the layout's sub-directory names belong to the writer and are never assumed here.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, NoReturn, Optional, Sequence, Set, Tuple

REPO = Path(__file__).resolve().parents[3]
_SRC = REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sisteer.capeval.tinycap import contracts as C  # noqa: E402

# --------------------------------------------------------------------------- #
# Refusals and exit codes                                                       #
# --------------------------------------------------------------------------- #

#: exit code of a table that could not be built (a missing / ambiguous / malformed input)
EXIT_REFUSED = 2
#: exit code of a table that WAS built and records at least one failed verdict / bar / identity assertion
EXIT_FAILED = 1
EXIT_OK = 0


class AuditRefusal(RuntimeError):
    """A table cannot be built: the reason names the input that is missing or malformed."""


def refuse(msg: str) -> NoReturn:
    raise AuditRefusal(msg)


def run_main(fn: Callable[[Sequence[str]], int], argv: Optional[Sequence[str]] = None) -> int:
    """Run a table's ``main``; an :class:`AuditRefusal` prints ``REFUSED: <reason>`` and exits
    :data:`EXIT_REFUSED`. Every other exception propagates (a traceback is a bug, not a refusal)."""
    try:
        return int(fn(list(sys.argv[1:] if argv is None else argv)))
    except AuditRefusal as e:
        print(f"REFUSED: {e}", file=sys.stderr, flush=True)
        return EXIT_REFUSED


# --------------------------------------------------------------------------- #
# Legs, arms, views                                                             #
# --------------------------------------------------------------------------- #

STATIC_FAMILIES: Tuple[str, ...] = ("pasttense", "codeattack", "renellm", "grok420")
MULTITURN_ATTACKS: Tuple[str, ...] = ("pair", "crescendo", "actorattack")
CONTROLS: Tuple[str, ...] = ("xstest", "or_bench_hard_1k")
MISALIGN_BEHAVIORS: Tuple[str, ...] = ("leaking", "murder")
#: the 15 TABLE legs (15 legs × 27 arm-runs = 405 rows): the four static families, BoN, the
#: three multi-turn attacks, the two over-refusal controls, MMLU, GSM8K, Petri, the two misalign behaviours
LEGS: Tuple[str, ...] = (*STATIC_FAMILIES, "bon", *MULTITURN_ATTACKS, *CONTROLS, "mmlu", "gsm8k", "petri",
                         "misalign_leaking", "misalign_murder")
assert len(LEGS) == 15
#: the CONTAINER leg a table leg's store lives in (one controls/MMLU standup, one misalign
#: standup); every other table leg is its own container leg. `ATTEMPT.json.launch_plan_row.leg` and the
#: `RS/stores/<leg>/` directory name carry the CONTAINER leg.
CONTAINER_LEG_OF: Dict[str, str] = {**{l: l for l in LEGS},
                                    "xstest": "controls", "or_bench_hard_1k": "controls", "mmlu": "controls",
                                    "misalign_leaking": "misalign", "misalign_murder": "misalign"}
CONTAINER_LEGS: Tuple[str, ...] = tuple(dict.fromkeys(CONTAINER_LEG_OF[l] for l in LEGS))
#: the seed views and the Arditi engine's view label
SEED_VIEWS: Tuple[str, ...] = ("s1", "s2", "s3")
ARDITI_VIEW = "arditi"
#: the Arditi arm as the standup serves it (`--cell arditi_allpos`) and its staged id
ARDITI_CELL = "arditi_allpos"
ARDITI_STAGED_ID = "ar_c0p3__arditi_allpos"
PROMPTED_ARMS: Tuple[str, ...] = ("prompted__si_constitution", "prompted__safety_refusal")
_VIEW_RE = re.compile(r"^s([1-9][0-9]*)$")


def seed_of_view(view: str) -> Optional[int]:
    """``s1`` → 1 … ; ``arditi`` (or anything else) → None."""
    m = _VIEW_RE.match(str(view))
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# Canonical task names — derived from each store's OWN writer                 #
# --------------------------------------------------------------------------- #

from sisteer.capeval.tiers import judged as _judged  # noqa: E402
from sisteer.capeval.tiers import parity as _parity  # noqa: E402
from sisteer.capeval.tiers import petri_runner as _petri  # noqa: E402

STATIC_TASK_PREFIX = "static/"
MULTITURN_TASK_PREFIX = "multi_turn/"
MISALIGN_TASK_PREFIX = "agentic/"
BON_TASK = "strongreject_bon"
#: the sampled controls' PRODUCTION row tasks (`cell_controls_run.JUDGED_TASK_OF` → `judged.TASK_*`; the
#: spool header's `task` and every judged row's) — NOT `contracts.inspect_task("xstest")`, the Inspect
#: variant, which the sampled stores never file under (rev 6)
CONTROLS_TASK_OF: Dict[str, str] = {"xstest": _judged.TASK_XSTEST, "or_bench_hard_1k": _judged.TASK_ORBENCH}
MMLU_TASK = _parity.PARITY_TASK_MMLU
GSM8K_TASK = C.inspect_task(C.GSM8K_PAIRED_TASK_NAME)
PETRI_TASK_PREFIX = _petri.PETRI_TASK_PREFIX


def static_task(family: str) -> str:
    """``static/<family>`` — the `attack` the sampled `private_run` rows (`family`) and `progress.jsonl` are keyed by."""
    return f"{STATIC_TASK_PREFIX}{family}"


def multiturn_task(attack: str) -> str:
    """``multi_turn/<attack>`` — the attack the run tree `<private_run>/<attack>/<defense>/shard*` is keyed by."""
    return f"{MULTITURN_TASK_PREFIX}{attack}"


def misalign_task(behavior: str) -> str:
    """``agentic/<behavior>`` — the runner's `family/behavior` its `agentic/<cell>/responses.jsonl` rows carry."""
    return f"{MISALIGN_TASK_PREFIX}{behavior}"


def petri_task(behaviour: str) -> str:
    """``petri/<behaviour>`` — the Petri store's own row task (`petri_runner.PETRI_TASK_PREFIX + behavior`)."""
    return f"{PETRI_TASK_PREFIX}{behaviour}"


def canonical_task(leg: str, sub: Optional[str] = None) -> str:
    """THE task string a leg's target requests carry in `X-Tinycap-Task` = the string its store files rows
    under. ``sub`` = the behaviour for `petri`; unused elsewhere (the leg names the family / attack / control)."""
    if leg in STATIC_FAMILIES:
        return static_task(leg)
    if leg == "bon":
        return BON_TASK
    if leg in MULTITURN_ATTACKS:
        return multiturn_task(leg)
    if leg in CONTROLS:
        return CONTROLS_TASK_OF[leg]
    if leg == "mmlu":
        return MMLU_TASK
    if leg == "gsm8k":
        return GSM8K_TASK
    if leg == "petri":
        if not sub:
            refuse("canonical_task('petri') needs the behaviour")
        return petri_task(sub)
    if leg.startswith("misalign_"):
        return misalign_task(leg[len("misalign_"):])
    refuse(f"unknown leg {leg!r} (one of {LEGS})")


def table_legs_of(container_leg: str) -> List[str]:
    """The table legs served on one container leg (`CONTAINER_LEG_OF` inverted, in `LEGS` order)."""
    return [l for l in LEGS if CONTAINER_LEG_OF[l] == container_leg]


def tasks_of_leg(leg: str, petri_behaviours: Sequence[str] = ()) -> Tuple[str, ...]:
    if leg == "petri":
        return tuple(petri_task(b) for b in petri_behaviours)
    return (canonical_task(leg),)


def check_client_task_names(client_module: Any) -> List[str]:
    """Drift check against the header builder: for every leg, `client_module.store_task_name(leg[, sub])`
    must equal :func:`canonical_task`. Returns the list of disagreements (empty = agree). The audit refuses
    when the module is importable and disagrees; when it is not importable the header side is simply not
    present on this tree and the check is recorded as `client_module: absent`."""
    fn = getattr(client_module, "store_task_name", None)
    if fn is None:
        return [f"{client_module!r} has no store_task_name"]
    out: List[str] = []
    for leg in LEGS:
        subs: Tuple[Optional[str], ...] = ("leaking",) if leg == "petri" else (None,)
        for sub in subs:
            want = canonical_task(leg, sub)
            try:
                got = fn(leg, sub) if sub is not None else fn(leg)
            except Exception as e:  # noqa: BLE001
                out.append(f"{leg}: store_task_name raised {type(e).__name__}: {e}")
                continue
            if got != want:
                out.append(f"{leg}: store_task_name → {got!r}, canonical_task → {want!r}")
    return out


# --------------------------------------------------------------------------- #
# File helpers                                                                  #
# --------------------------------------------------------------------------- #

_MAX_WALK_DEPTH = 10


def read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        refuse(f"{path}: missing")
    except json.JSONDecodeError as e:
        refuse(f"{path}: not JSON ({e})")


def read_jsonl(path: Path) -> List[dict]:
    """Every non-blank line parsed; a torn FINAL line is refused by name (never silently dropped)."""
    p = Path(path)
    if not p.is_file():
        refuse(f"{p}: missing")
    opener = gzip.open if p.suffix == ".gz" else open
    rows: List[dict] = []
    with opener(p, "rt", encoding="utf-8") as f:  # type: ignore[call-arg]
        lines = f.read().splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            refuse(f"{p}: line {i + 1} is not JSON (a torn or foreign line; refusing to read around it)")
    return rows


def walk_files(root: Path, name: str, *, max_depth: int = _MAX_WALK_DEPTH) -> List[Path]:
    """Every file called ``name`` under ``root`` within ``max_depth`` levels (a BOUNDED walk — never an
    unbounded rglob over a store tree; memory `bounded-walks-over-store-trees`)."""
    root = Path(root)
    out: List[Path] = []
    if not root.is_dir():
        return out
    base_depth = len(root.parts)
    stack = [root]
    while stack:
        d = stack.pop()
        if len(d.parts) - base_depth > max_depth:
            continue
        try:
            entries = sorted(d.iterdir())
        except OSError:
            continue
        for e in entries:
            if e.is_dir() and not e.is_symlink():
                stack.append(e)
            elif e.name == name and e.is_file():
                out.append(e)
    return sorted(out)


def find_one(root: Path, name: str, *, required: bool = True, alternatives: Sequence[str] = ()) -> Optional[Path]:
    """EXACTLY one file called ``name`` (or one of ``alternatives``, tried in order when ``name`` is absent)
    under ``root``; none refuses when ``required``, several always refuse (an ambiguous root is not read)."""
    for nm in (name, *alternatives):
        hits = walk_files(root, nm)
        if len(hits) > 1:
            refuse(f"{root}: {len(hits)} files called {nm!r} (first: {[str(h) for h in hits[:3]]}) — an ambiguous "
                   "attempt root is not read")
        if hits:
            return hits[0]
    if required:
        refuse(f"{root}: no {name!r}" + (f" (nor {list(alternatives)})" if alternatives else "") + " under the attempt root")
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> Path:
    """Atomic CSV write with a FIXED column roster (a row missing a column refuses — no blank cell is ever
    written by accident); values are rendered `json.dumps` for lists / dicts, `str` otherwise, `''` for None
    ONLY where the roster marks the column optional by a trailing `?`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c.rstrip("?") for c in columns]
    optional = {c.rstrip("?") for c in columns if c.endswith("?")}
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            missing = [c for c in cols if c not in r]
            if missing:
                refuse(f"{path}: a row lacks column(s) {missing} — refusing to write a table with blank cells")
            out = {}
            for c in cols:
                v = r[c]
                if v is None and c not in optional:
                    refuse(f"{path}: column {c!r} is None on row {json.dumps({k: r[k] for k in list(r)[:4]}, default=str)} — "
                           "a null in a required column would render as clean")
                out[c] = "" if v is None else (json.dumps(v, sort_keys=True, default=str) if isinstance(v, (list, dict)) else str(v))
            w.writerow(out)
    tmp.replace(path)
    return path


def write_json(path: Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


# --------------------------------------------------------------------------- #
# Attempt roots                                                                 #
# --------------------------------------------------------------------------- #

ATTEMPT_FILE = "ATTEMPT.json"
COMPLETE_FILE = "COMPLETE.json"
RESULT_FILE = "RESULT"
ENDPOINT_FILE = "endpoint.json"
ATTESTATION_FILE = "serving_attestation.json"
CELLS_SERVED_FILE = "cells_served.json"
ROUTING_TABLE_FILE = "routing_table.json"
STANDUP_JOBS_FILE = "standup_jobs.jsonl"
#: HOW a store's arm matched the standup's records (`VmodelBinding.spelling`). The drivers name their stores by the train/test
#: CELL id (`jailbreak_run` / `bon_served_run` `--defense constv2-mix-full__lora_r4_ce`; the multi-turn attacks likewise) while
#: `endpoint.json.vmodels` is keyed by the served HARNESS id (`si_constv2_mix_full__lora_r4_ce`) — the identity seam, in the
#: audit's arm -> vmodel binding. The standup RECORDS the seam itself (`cells` <-> `harness_cells`, index-aligned, on the endpoint and
#: on `cells_served.json`), so the binding is read off those records — never a string transform.
ARM_SPELLING_HARNESS = "harness"                # the arm IS a `vmodels` key (the served harness cell id — misalign / CellStore readers)
ARM_SPELLING_TRAINTEST = "traintest"            # the arm is a TrainTest cell id the records bind to a served harness id (static / BoN / multi-turn)
ARM_SPELLING_BOTH = "harness==traintest"        # the two spellings coincide (`none`, `arditi_allpos`)
LEDGER_FILE = "usage_ledger.jsonl"
EVIDENCE_FILE = C.REQUEST_EVIDENCE_FILENAME
RESPONSE_EVIDENCE_FILE = "response_evidence.jsonl"
#: the response record at its shipped version 2 (`request_evidence_client.RESPONSE_EVIDENCE_FIELDS`, one row per attempt that
#: returned or died: the version first, then `text_sha256` + `text_sha256_of` after `finish_reason` — the sha256 of the first choice's
#: content, the response side of the replay check; both None on a failed or dead attempt) — restated here for the fixture's pin, held
#: equal to the client's constant by tests/test_seedrep_request_evidence_audit.py; `extra` rides beside them. The readers key on
#: `request_id` / `text_sha256` only.
RESPONSE_EVIDENCE_VERSION = 2
RESPONSE_EVIDENCE_FIELDS: Tuple[str, ...] = ("response_evidence_version", "ts", "request_id", "completion_id", "status", "vmodel",
                                             "finish_reason", "text_sha256", "text_sha256_of", "task", "sample", "error")
RESPONSE_TEXT_SHA_FIELD = "text_sha256"
RESPONSE_TEXT_SHA_OF_FIELD = "text_sha256_of"
#: the ADDITIVE request-record field replay needs beside the recorder's `body_sha256` / `body_bytes` (a digest and a
#: LENGTH — the recorder keeps no bytes): the exact wire bytes, base64. A record without it REFUSES the replay of its
#: draw — a body is never reconstructed, and no side file (`request_bodies.jsonl`) is read: the record is the producer.
REQUEST_BODY_B64_FIELD = "body_b64"
ENGINE_LOG_FILES: Tuple[str, ...] = ("vllm_server.log.gz", "vllm_server.log")
DMON_FILE = "gpu_dmon.log"
METRICS_FILE = "vllm_metrics.jsonl"
INFLIGHT_FILE = "driver_inflight.jsonl"
DETERMINISM_FILE = "determinism.json"
STAGE_BASE = "base"
#: the attempt root's STANDUP dir — `seedrep_launch_plan.SERVE_DIR_REL`, where `seedrep_served.ATTEMPT_ROOT_LAYOUT` declares
#: `endpoint.json` / `cells_served.json` / `serving_attestation.json` / `routing_table.json` / `standup_jobs.jsonl` live (held equal to
#: the plan's constant by tests/test_seedrep_fz_standup_record.py). A root with this dir is read THERE for its standup record and
#: nowhere else (the GSM8K assembly writes a per-cell copy of `endpoint.json` beside it and `find_one` refused every audit
#: on a gsm8k row as ambiguous — a copy elsewhere in the root is a CELL record, never the standup's).
SERVE_DIR_REL = "exp/serve"
#: the GSM8K assembly's PER-CELL copies of the standup record: `seedrep_served.gsm8k_child_serve_dir` writes
#: `<root>/exp/serve_gsm8k/<cell>__s<s>/{routing_table.json, endpoint.json}` (`seedrep_launch_plan.GSM8K_LEDGER_SERVE_REL`) with
#: `routing_table` rewritten to the copy and `ledger_serve_copy_of` naming the standup's record. :func:`gsm8k_cell_files` exposes them
#: for a consumer that WANTS the per-cell record; no table reads one today — the at-cap and engine-identity tables read the STANDUP's
#: (`vmodels` and the attested identity are the same bytes on every copy; the copies differ only in `routing_table`).
GSM8K_CELL_SERVE_DIR_REL = "exp/serve_gsm8k"
CELL_ENDPOINT_COPY_OF_FIELD = "ledger_serve_copy_of"
#: `mt_serving_up.py` unlinks `endpoint.json` at teardown (the pollers' down signal), so an audit that runs after teardown
#: (`request_evidence_audit`, in-container) used to find no record. The Modal app now RESTORES the READY record's bytes at
#: the SAME path once the standup process has exited (`seedrep_served.restore_endpoint_record`, evidence
#: `endpoint_json_restored_after_teardown`), so the in-container audit reads the same file the volume holds — ONE filename, no
#: alternate; an absent record refuses by name.
#: the keys `read_standup` REQUIRES of `standup_jobs.jsonl`'s first line — the standup's identity as the cluster's `f22_standup`
#: writes it (`common.sh`: job_id, node, key, view, mode, eval, sha, engine, serving_path, …, 27 keys) and as the Modal app writes
#: it (one line). `sha` is CONSUMED (the tree the standup served from — `engine_identity_table.engine_row` holds it to
#: ATTEMPT.json's W); the other five are the record's identity axes. Geometry (`max_num_seqs` / `max_model_len` / `attested_*`) and
#: `thinking_mode` are NOT required here: every reader takes them from `endpoint.json`'s attested fields, the source the cluster's
#: writer itself copied them from.
STANDUP_JOBS_REQUIRED_KEYS: Tuple[str, ...] = ("job_id", "key", "view", "mode", "eval", "sha")
#: where a base CHILD writes its store — and, with it, its evidence files. `seedrep_launch_plan.STORE_ROOT_FLAG_OF` / `child_store_root`:
#: a child's `SISTEER_REQUEST_EVIDENCE_DIR` IS its store root (`--out-root` for controls / GSM8K, `--out` for misalign, `--private-root`
#: for static / BoN / multi-turn), every one of them under `<root>/exp/` (`GSM8K_OUT_REL` = exp/sampled/gsm8k_thinking_on_cap{cap}__s{s}/
#: {cell}, `MISALIGN_OUT_REL` = exp/misalign/agentic__s{s}__{behavior}, …); a GSM8K child's usage ledger is started PER CHILD STORE at
#: `<out root>/usage_ledger.jsonl` (`gsm8k_ledger_argv`, `GSM8K_LEDGER_FILENAME`). So a multi-child root holds one `request_evidence.jsonl`
#: / `response_evidence.jsonl` / `usage_ledger.jsonl` PER CHILD, and a rung consumer's evidence lands at the LITERAL attempt root
#: (`rung_child_env`, `REQUEST_EVIDENCE_FILES`). The owned readers below gather BOTH owners and refuse a copy anywhere else.
CHILD_STORE_TREE_REL = "exp"
#: the owner of an evidence file, as :func:`owned_evidence_files` classifies it
OWNER_ATTEMPT_ROOT = "attempt_root"
OWNER_CHILD_STORE = "child_store"
#: reader-side provenance stamped on every row an owned reader returns: the ROOT-RELATIVE path of the file it came from. A reader's
#: field, never a producer's (the ledger's `ledger_row` schema knows nothing of it), so a table that carries a row carries its source.
ROW_SOURCE_FIELD = "_source"


@dataclass
class AttemptRoot:
    """One pulled ACCEPTED attempt root and the identity `ATTEMPT.json` records for it."""

    path: Path
    attempt_id: str
    W: str
    leg: str          # the CONTAINER leg (`CONTAINER_LEGS`)
    view: str         # `s1` | `s2` | `s3` | `arditi`
    shard: str
    stage: str        # `base` | `2048` | `8192` | `16384` | `petri-rerun`
    started_utc: Optional[str]
    attempt: Dict[str, Any]
    #: the wave this attempt belongs to (the record's `wave_uuid`, held equal to the `control_key`'s when both are present;
    #: the control key's alone otherwise) — REQUIRED: an attempt that cannot say which wave it ran in binds to no plan
    wave_uuid: str = ""
    #: the `control_key` that `attempt_record` writes (`W<n>/wave_uuid/leg/view/stage/shard`); None when absent
    control_key: Optional[str] = None
    #: the SMOKE this attempt ran under (`smoke-<S>__…` on its `launch_plan_row`, `seedrep_launch_plan.smoke_row`); None for a
    #: production row. F-U: a smoke row's record used to REFUSE at the id parse, so no smoke attempt ever got a determinism / audit record
    smoke_id: Optional[str] = None
    #: when this attempt was CARRIED from a prior tree / wave, the VALIDATED carry record's provenance
    #: (`{W, wave_uuid, standup_id (historical), source, attempt_root, accepted_key}`) -- set by :func:`bind_roots_to_plan` from the
    #: carry mapping. The identity fields above stay the HISTORICAL ones (never rewritten); this names the CURRENT planned standup the
    #: attempt stands for. None for an attempt of this wave.
    carried_from: Optional[Dict[str, Any]] = None

    @property
    def s(self) -> Optional[int]:
        return seed_of_view(self.view)

    @property
    def standup_id(self) -> str:
        """The launch-plan row this attempt executed, spelled the way the PLAN spells it: `<leg>__<view>__<stage>__shard<NN>` for a
        production row, `smoke-<S>__<leg>__<view>__<stage>__shard<NN>` for a smoke row (:func:`smoke_standup_id_of`). Every source the
        record names (`launch_plan_row`, `control_key`, the top level) was held to these axes by :func:`read_attempt_root`, so the id is
        spelled from them — never a second, possibly disagreeing, copy."""
        if self.smoke_id is not None:
            return smoke_standup_id_of(self.smoke_id, self.leg, self.view, self.stage, self.shard)
        return standup_id_of(self.leg, self.view, self.stage, self.shard)

    @property
    def production_standup_id(self) -> str:
        """The axes-only id (`<leg>__<view>__<stage>__shard<NN>`): for a smoke root, the production row the smoke re-keyed."""
        return standup_id_of(self.leg, self.view, self.stage, self.shard)

    @property
    def label(self) -> str:
        return f"{self.leg}/{self.view}/{self.stage}/{self.shard}"


STANDUP_ID_SEP = "__"
CONTROL_KEY_AXES: Tuple[str, ...] = ("W7", "wave_uuid", "leg", "view", "stage", "shard")


def _shard_name(shard: Any) -> str:
    """`shard00`-style spelling: an int (or an all-digit string, the control key's spelling) → `shard<NN>`."""
    if isinstance(shard, bool):
        refuse(f"shard {shard!r} is a bool")
    if isinstance(shard, int):
        return f"shard{shard:02d}"
    s = str(shard)
    return f"shard{int(s):02d}" if s.isdigit() else s


def standup_id_of(leg: str, view: str, stage: Any, shard: Any) -> str:
    """The launch plan's `standup_id` (`seedrep_launch_plan.standup_id`): `<leg>__<view>__<stage>__shard<NN>`."""
    return STANDUP_ID_SEP.join((str(leg), str(view), str(stage), _shard_name(shard)))


#: a SMOKE row's standup id is the production row's under ONE prefix (`seedrep_launch_plan.smoke_row`: `smoke-<smoke_id>__<standup_id>`,
#: the smoke id held to the plan's `SMOKE_ID_RE`, mirrored here because this module imports no plan). THE parser of both forms is
#: :func:`split_standup_id`; every reader that keys on a standup id goes through it (`read_attempt_root`, hence the determinism check, the
#: request-evidence audit, the utilisation / engine-identity tables). F-U: a second, production-only regex refused every smoke row.
SMOKE_STANDUP_PREFIX = "smoke-"
SMOKE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
STANDUP_ID_FORMS = "<leg>__<view>__<stage>__shard<NN> (production) or smoke-<S>__<leg>__<view>__<stage>__shard<NN> (a smoke row)"


def smoke_standup_id_of(smoke_id: str, leg: str, view: str, stage: Any, shard: Any) -> str:
    """The plan's SMOKE standup id (`seedrep_launch_plan.smoke_row`): `smoke-<smoke_id>__<leg>__<view>__<stage>__shard<NN>`."""
    if not SMOKE_ID_RE.match(str(smoke_id or "")):
        refuse(f"smoke id {smoke_id!r} does not match {SMOKE_ID_RE.pattern}")
    return f"{SMOKE_STANDUP_PREFIX}{smoke_id}{STANDUP_ID_SEP}{standup_id_of(leg, view, stage, shard)}"


def split_standup_id(sid: str) -> Tuple[Optional[str], str, str, str, str]:
    """BOTH id forms through one parser → `(smoke_id | None, leg, view, stage, shard)`. A production id parses with smoke_id None; a
    smoke id (`smoke-<S>__` + the production id, `S` matching the plan's smoke-id shape) parses to its smoke id AND the production
    axes; anything else — a wrong part count, a shard not spelled `shard<NN>`, a `smoke-` prefix with an id outside the plan's shape —
    REFUSES by name. (A smoke id containing `__` would split into extra parts and refuse too: the plan's own smokes are `S1`…`S5`.)"""
    parts = str(sid).split(STANDUP_ID_SEP)
    smoke: Optional[str] = None
    if len(parts) == 5 and parts[0].startswith(SMOKE_STANDUP_PREFIX):
        cand = parts[0][len(SMOKE_STANDUP_PREFIX):]
        if SMOKE_ID_RE.match(cand):
            smoke, parts = cand, parts[1:]
    if len(parts) != 4 or not parts[3].startswith("shard") or not parts[3][5:].isdigit():
        refuse(f"standup_id {sid!r} is neither {STANDUP_ID_FORMS}")
    return smoke, parts[0], parts[1], parts[2], parts[3]


def parse_standup_id(sid: str) -> Tuple[str, str, str, str]:
    """The four axes of EITHER id form (:func:`split_standup_id` minus the smoke id)."""
    return split_standup_id(sid)[1:]


def parse_control_key(key: str) -> Dict[str, str]:
    """`W<n>/wave_uuid/leg/view/stage/shard[/extra…]` (the `control_key`) → the six axes (shard spelled `shard<NN>`)."""
    parts = str(key).split("/")
    if len(parts) < len(CONTROL_KEY_AXES):
        refuse(f"control_key {key!r} lacks the six axes {CONTROL_KEY_AXES}")
    out = dict(zip(CONTROL_KEY_AXES, parts[: len(CONTROL_KEY_AXES)]))
    out["shard"] = _shard_name(out["shard"])
    return out


def _row_field(row: Mapping[str, Any], *names: str) -> Optional[Any]:
    for n in names:
        if n in row and row[n] is not None and row[n] != "":
            return row[n]
    return None


def read_attempt_root(path: Path, *, path_hint: Optional[Tuple[str, str, str]] = None) -> AttemptRoot:
    """`ATTEMPT.json` → :class:`AttemptRoot`. The axes leg / view / stage / shard come from EVERY source the record
    carries and all must agree: `launch_plan_row` as the plan writes it (the plan row's `standup_id` string) or as a
    dict (the `read_attempt` shape: keys `leg`, `view`, `shard`, `stage`|`cap`), `control_key` (`W<n>/wave_uuid/leg/
    view/stage/shard`, whose leading W and wave_uuid must match the record's `W` / `wave_uuid`), the record's top
    level, then the pulled path's `<leg>/<view>/<shard>` (`path_hint`) for a record naming none. `W`, `attempt_id`
    and the WAVE (the record's `wave_uuid`, or the control key's) are required: an attempt of no known wave binds to no plan."""
    path = Path(path)
    ap = path / ATTEMPT_FILE
    if not ap.is_file():
        refuse(f"{path}: no {ATTEMPT_FILE} — not an attempt root")
    att = read_json(ap)
    if not isinstance(att, dict):
        refuse(f"{ap}: not a JSON object")
    aid = att.get("attempt_id")
    W = att.get("W")
    if not isinstance(aid, str) or not aid:
        refuse(f"{ap}: attempt_id missing")
    if not isinstance(W, str) or not W:
        refuse(f"{ap}: W missing (the tree the attempt ran from)")
    wave_uuid = att.get("wave_uuid") if isinstance(att.get("wave_uuid"), str) else None
    sources: List[Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]] = []
    lpr = att.get("launch_plan_row")
    row: Dict[str, Any] = {}
    # the smoke id, from every source that spells one (the id's `smoke-<S>__` prefix, a smoke row dict's `smoke.smoke_id`); all must agree
    smoke_sources: List[Tuple[str, str]] = []
    if isinstance(lpr, str) and lpr:
        smoke, *axes = split_standup_id(lpr)
        sources.append(("launch_plan_row", *axes))
        if smoke is not None:
            smoke_sources.append(("launch_plan_row", smoke))
    elif isinstance(lpr, dict):
        row = lpr
        if isinstance(lpr.get("standup_id"), str) and lpr["standup_id"]:
            smoke, *axes = split_standup_id(lpr["standup_id"])
            sources.append(("launch_plan_row.standup_id", *axes))
            if smoke is not None:
                smoke_sources.append(("launch_plan_row.standup_id", smoke))
        sm = lpr.get("smoke")
        if isinstance(sm, dict) and isinstance(sm.get("smoke_id"), str) and sm["smoke_id"]:
            smoke_sources.append(("launch_plan_row.smoke.smoke_id", sm["smoke_id"]))
    smoke_id: Optional[str] = None
    for name, v in smoke_sources:
        if smoke_id is None:
            smoke_id = v
        elif smoke_id != v:
            refuse(f"{ap}: smoke id {v!r} ({name}) disagrees with {smoke_id!r} named earlier in the record")
    control_key = att.get("control_key") if isinstance(att.get("control_key"), str) and att.get("control_key") else None
    if control_key:
        ck = parse_control_key(control_key)
        if ck["W7"] != W[:7]:
            refuse(f"{ap}: control_key W7 {ck['W7']!r} != W {W[:7]!r}")
        if wave_uuid is not None and ck["wave_uuid"] != wave_uuid:
            refuse(f"{ap}: control_key wave_uuid {ck['wave_uuid']!r} != wave_uuid {wave_uuid!r}")
        sources.append(("control_key", ck["leg"], ck["view"], ck["stage"], ck["shard"]))
    for name, rec in (("launch_plan_row", row), ("top level", att)):
        leg_f = _row_field(rec, "leg", "container_leg")
        view_f = _row_field(rec, "view")
        shard_f = _row_field(rec, "shard")
        stage_f = _row_field(rec, "stage", "cap", "stage_or_cap")
        if any(v is not None for v in (leg_f, view_f, shard_f, stage_f)):
            sources.append((name, None if leg_f is None else str(leg_f), None if view_f is None else str(view_f),
                            None if stage_f is None else str(stage_f), None if shard_f is None else _shard_name(shard_f)))
    resolved: Dict[str, Optional[str]] = {"leg": None, "view": None, "stage": None, "shard": None}
    for name, *vals in sources:
        for axis, v in zip(("leg", "view", "stage", "shard"), vals):
            if v is None:
                continue
            if resolved[axis] is None:
                resolved[axis] = v
            elif resolved[axis] != v:
                refuse(f"{ap}: {axis} {v!r} ({name}) disagrees with {resolved[axis]!r} named earlier in the record — the attempt "
                       "record does not know which planned standup it executed")
    if path_hint:  # the pulled path's <leg>/<view>/<shard>: the LAST source, filling only what the record left unnamed
        for axis, v in zip(("leg", "view", "shard"), path_hint):
            if resolved[axis] is None and v:
                resolved[axis] = str(v)
    if wave_uuid is None and control_key:
        wave_uuid = parse_control_key(control_key)["wave_uuid"]
    if not wave_uuid:
        refuse(f"{ap}: the attempt names no wave — neither wave_uuid nor a control_key carrying one; an attempt of no known wave "
               "binds to no launch plan")
    leg, view, stage, shard = resolved["leg"], resolved["view"], resolved["stage"] or STAGE_BASE, resolved["shard"]
    if leg is None or view is None or shard is None:
        refuse(f"{ap}: neither launch_plan_row nor control_key nor the top level names leg/view/shard and the path {path} does not "
               "spell <leg>/<view>/<shard>")
    if leg not in CONTAINER_LEGS:
        refuse(f"{ap}: leg {leg!r} is not a container leg ({CONTAINER_LEGS})")
    if view not in SEED_VIEWS and view != ARDITI_VIEW:
        refuse(f"{ap}: view {view!r} is not one of {SEED_VIEWS + (ARDITI_VIEW,)}")
    return AttemptRoot(path=path, attempt_id=aid, W=W, leg=leg, view=view, shard=shard, stage=stage,
                       started_utc=att.get("started_utc"), attempt=att, wave_uuid=wave_uuid, control_key=control_key, smoke_id=smoke_id)


def discover_attempt_roots(stores_root: Path) -> List[AttemptRoot]:
    """Every attempt root under `RS/stores/` (a dir holding `ATTEMPT.json`, ≤ 4 levels down). An empty result
    refuses — a stores root with nothing pulled is not a wave to tabulate."""
    stores_root = Path(stores_root)
    if not stores_root.is_dir():
        refuse(f"{stores_root}: not a directory")
    hits = walk_files(stores_root, ATTEMPT_FILE, max_depth=5)
    roots: List[AttemptRoot] = []
    for ap in hits:
        rel = ap.parent.relative_to(stores_root).parts
        hint = (rel[0], rel[1], rel[2]) if len(rel) >= 3 else None
        roots.append(read_attempt_root(ap.parent, path_hint=hint))
    if not roots:
        refuse(f"{stores_root}: no attempt root ({ATTEMPT_FILE}) under it — nothing pulled")
    dup = duplicates(r.attempt_id for r in roots)
    if dup:
        refuse(f"{stores_root}: attempt id(s) {dup[:3]} appear under two roots — the pull mixed attempts")
    return roots


def complete_record(root: AttemptRoot) -> Optional[Dict[str, Any]]:
    p = root.path / COMPLETE_FILE
    if not p.is_file():
        return None
    rec = read_json(p)
    return rec if isinstance(rec, dict) else {"raw": rec}


# --------------------------------------------------------------------------- #
# Standup records                                                               #
# --------------------------------------------------------------------------- #

def standup_dir(root: Path) -> Optional[Path]:
    """`<root>/exp/serve` when the root has the declared layout (`SERVE_DIR_REL`), else None (a pre-layout root)."""
    d = Path(root) / SERVE_DIR_REL
    return d if d.is_dir() else None


def gsm8k_cell_files(root: Path, name: str) -> Dict[str, Path]:
    """The GSM8K assembly's per-cell copies of `name` — `{<cell>__s<s>: <root>/exp/serve_gsm8k/<cell>__s<s>/<name>}`, empty when the
    root has none. CELL records: a consumer that wants the per-cell record asks here by name; :func:`standup_file` never returns one."""
    d = Path(root) / GSM8K_CELL_SERVE_DIR_REL
    if not d.is_dir():
        return {}
    out: Dict[str, Path] = {}
    for sub in sorted(d.iterdir()):
        if sub.is_dir() and not sub.is_symlink() and (sub / name).is_file():
            out[sub.name] = sub / name
    return out


def standup_file(root: Path, name: str, *, required: bool = True, alternatives: Sequence[str] = ()) -> Optional[Path]:
    """The STANDUP's copy of `name` (or the first present of `alternatives`, tried in order). A root with the declared layout
    (:func:`standup_dir`) is read THERE and nowhere else: exactly one hit under `exp/serve/` (a second refuses — an ambiguous standup
    dir is not read), and a copy elsewhere in the root — the GSM8K assembly's `exp/serve_gsm8k/<cell>__s<s>/endpoint.json` — is a
    CELL record that is neither returned nor counted (F-AE). None under `exp/serve/` refuses BY NAME when `required`, naming the cell
    copies as what they are. A root WITHOUT the layout falls to :func:`find_one` over the whole root (the pre-layout shape,
    unchanged: several anywhere refuse)."""
    root = Path(root)
    sd = standup_dir(root)
    if sd is None:
        return find_one(root, name, required=required, alternatives=alternatives)
    for nm in (name, *alternatives):
        hits = walk_files(sd, nm)
        if len(hits) > 1:
            refuse(f"{root}: {len(hits)} files called {nm!r} under {SERVE_DIR_REL}/ ({[str(h) for h in hits[:3]]}) — an ambiguous "
                   "standup dir is not read")
        if hits:
            return hits[0]
    if required:
        cells = gsm8k_cell_files(root, name)
        refuse(f"{root}: no {name!r}" + (f" (nor {list(alternatives)})" if alternatives else "") + f" under {SERVE_DIR_REL}/ (the "
               "standup record)" + (f"; the {len(cells)} under {GSM8K_CELL_SERVE_DIR_REL}/ ({sorted(cells)[:3]}) are the GSM8K "
                                   "assembly's per-cell records, not the standup's" if cells else ""))
    return None


def standup_endpoint_path(root: Path) -> Path:
    """The standup's endpoint record, `exp/serve/endpoint.json` (the READY bytes, restored at the same path after teardown by the
    Modal app); absent refuses by name."""
    return standup_file(root, ENDPOINT_FILE)  # type: ignore[return-value]


STANDUP_RECORD_FROM_JOBS = STANDUP_JOBS_FILE
STANDUP_RECORD_DERIVED = f"derived({ENDPOINT_FILE}+{ATTESTATION_FILE}; W from {ATTEMPT_FILE})"


@dataclass(frozen=True)
class VmodelBinding:
    """One arm bound to the virtual model the standup served it as, and HOW. `arm` as the caller (a store reader) handed it;
    `harness_id` the served cell id it resolved to (the `vmodels` key); `traintest_id` the TrainTest spelling the records bind to that
    cell (None when the records spell none); `spelling` one of `ARM_SPELLING_*`; `records` which standup files were read and whether
    each was present — carried on every verdict that binds an arm, so a reader sees the seam it crossed."""

    arm: str
    harness_id: str
    vmodel: str
    spelling: str
    traintest_id: Optional[str]
    records: Dict[str, str]

    def as_dict(self) -> Dict[str, Any]:
        return {"arm": self.arm, "harness_id": self.harness_id, "vmodel": self.vmodel, "spelling": self.spelling,
                "traintest_id": self.traintest_id, "records": dict(self.records)}


@dataclass
class StandupIdentity:
    """The cell identity ONE standup recorded: `vmodels` (harness id -> virtual model, `endpoint.json`), and the TrainTest <-> harness
    map its records spell (`endpoint.json` `cells` <-> `harness_cells`, held equal to `cells_served.json` `cells_requested` <->
    `harness_cells` when that record is present). `cells_served.json`'s canonical / controls rows and `routing_table.json`'s routes are
    the same binding written twice more; both are held to `vmodels` when present (:func:`standup_identity`), so a resolution here is
    a resolution against every record the standup wrote of itself. `records` names each file and `present` | `absent`."""

    where: str
    vmodels: Dict[str, str]
    harness_of_traintest: Dict[str, str]
    traintest_of_harness: Dict[str, str]
    records: Dict[str, str]

    def resolve(self, arm: str) -> VmodelBinding:
        """`arm` in EITHER spelling -> its binding; refuses by name when neither spelling resolves, when the arm is a TrainTest spelling of a
        cell the standup lists no vmodel for, or when the two spellings resolve to DIFFERENT virtual models (never picked by preference)."""
        vm = self.vmodels
        direct = vm.get(arm)
        h = self.harness_of_traintest.get(arm)
        via_tt = vm.get(h) if h is not None else None
        if direct is not None and via_tt is not None:
            if direct != via_tt or h != arm:
                refuse(f"{self.where}: arm {arm!r} resolves to vmodel {direct!r} as a served cell id and to {via_tt!r} through its TrainTest "
                       f"spelling (harness cell {h!r}) — the standup's records bind one arm two ways; neither is picked")
            return VmodelBinding(arm=arm, harness_id=arm, vmodel=direct, spelling=ARM_SPELLING_BOTH, traintest_id=arm, records=dict(self.records))
        if direct is not None:
            return VmodelBinding(arm=arm, harness_id=arm, vmodel=direct, spelling=ARM_SPELLING_HARNESS,
                                 traintest_id=self.traintest_of_harness.get(arm), records=dict(self.records))
        if via_tt is not None:
            return VmodelBinding(arm=arm, harness_id=h, vmodel=via_tt, spelling=ARM_SPELLING_TRAINTEST, traintest_id=arm,  # type: ignore[arg-type]
                                 records=dict(self.records))
        if h is not None:
            refuse(f"{self.where}: arm {arm!r} is the TrainTest spelling of harness cell {h!r}, which the standup lists no vmodel for "
                   f"(vmodels: {sorted(vm)}) — a cell the standup skipped or never served")
        tail = (f"; TrainTest spellings: {sorted(self.harness_of_traintest)}" if self.harness_of_traintest
                else " and the standup's records spell no TrainTest ids (no `cells` <-> `harness_cells`)")
        refuse(f"{self.where}: arm {arm!r} is not a served cell (vmodels: {sorted(vm)}{tail})")


def _aligned_spellings(tt: Any, hs: Any, *, where: str, tt_key: str, hs_key: str) -> Dict[str, str]:
    """`tt[i]` <-> `hs[i]` as the writer aligns them (`native_standup`: `traintest_cells` = "the TrainTest spellings of served_cells",
    filtered together); different lengths or a repeated id refuse — that is not a map."""
    if not isinstance(tt, list) or not isinstance(hs, list):
        refuse(f"{where}: {tt_key} / {hs_key} are not lists ({type(tt).__name__} / {type(hs).__name__})")
    if len(tt) != len(hs):
        refuse(f"{where}: {tt_key} ({len(tt)}) and {hs_key} ({len(hs)}) are not index-aligned — the TrainTest <-> harness map cannot be read")
    tt_s, hs_s = [str(x) for x in tt], [str(x) for x in hs]
    for name, lst in ((tt_key, tt_s), (hs_key, hs_s)):
        d = duplicates(lst)
        if d:
            refuse(f"{where}: {name} repeats {sorted(d)[:3]} — not a map")
    return dict(zip(tt_s, hs_s))


def standup_identity(endpoint: Mapping[str, Any], *, endpoint_path: Path, cells_served: Optional[Mapping[str, Any]] = None,
                     cells_served_path: Optional[Path] = None, routing_table: Optional[Mapping[str, Any]] = None,
                     routing_table_path: Optional[Path] = None) -> StandupIdentity:
    """The standup's identity from its records, cross-checked. REQUIRED: the endpoint's `vmodels`. The train/test <-> harness map is
    the endpoint's `cells` <-> `harness_cells` when it spells them, else `cells_served.json`'s `cells_requested` <-> `harness_cells`;
    when both spell it they must agree. `cells_served.json`'s canonical / controls rows (`cell_id` -> `virtual_model_name`) and
    `routing_table.json`'s routes (`cell_id` -> `virtual_model`) must agree with `vmodels` on every cell they share — two records of one
    standup that disagree refuse by name. An ABSENT `cells_served.json` / `routing_table.json` is recorded (`records[<file>] = absent`),
    never invented and never read as a passed cross-check."""
    where = str(endpoint_path)
    vm_raw = endpoint.get("vmodels")
    if not isinstance(vm_raw, dict) or not vm_raw:
        refuse(f"{where}: no `vmodels` map (cell → virtual model) — the arm cannot be bound to a vmodel")
    vmodels = {str(k): str(v) for k, v in vm_raw.items()}
    records: Dict[str, str] = {ENDPOINT_FILE: "present", CELLS_SERVED_FILE: "present" if cells_served is not None else "absent",
                               ROUTING_TABLE_FILE: "present" if routing_table is not None else "absent"}
    ep_map: Optional[Dict[str, str]] = None
    if endpoint.get("cells") is not None or endpoint.get("harness_cells") is not None:
        ep_map = _aligned_spellings(endpoint.get("cells"), endpoint.get("harness_cells"), where=where, tt_key="cells", hs_key="harness_cells")
    cs_map: Optional[Dict[str, str]] = None
    if cells_served is not None:
        cs_where = str(cells_served_path or CELLS_SERVED_FILE)
        if cells_served.get("cells_requested") is not None or cells_served.get("harness_cells") is not None:
            cs_map = _aligned_spellings(cells_served.get("cells_requested"), cells_served.get("harness_cells"), where=cs_where,
                                        tt_key="cells_requested", hs_key="harness_cells")
            if ep_map is not None and cs_map != ep_map:
                refuse(f"{cs_where}: cells_requested <-> harness_cells {sorted(cs_map.items())} disagree with {ENDPOINT_FILE}'s cells <-> "
                       f"harness_cells {sorted(ep_map.items())} — two records of one standup spell its cells differently")
        for row in list(cells_served.get("canonical") or []) + list(cells_served.get("controls") or []):
            if not isinstance(row, dict) or row.get("cell_id") is None or row.get("virtual_model_name") is None:
                continue
            cid, v = str(row["cell_id"]), str(row["virtual_model_name"])
            if cid in vmodels and vmodels[cid] != v:
                refuse(f"{cs_where}: canonical/controls row for cell {cid!r} names virtual_model_name {v!r} but {ENDPOINT_FILE}.vmodels names "
                       f"{vmodels[cid]!r} — two records of one standup disagree")
    if routing_table is not None:
        rt_where = str(routing_table_path or ROUTING_TABLE_FILE)
        routes = routing_table.get("routes")
        if not isinstance(routes, list):
            refuse(f"{rt_where}: `routes` is not a list ({type(routes).__name__}) — not a routing table this audit can hold to the endpoint")
        for r in routes:
            if not isinstance(r, dict) or r.get("cell_id") is None or r.get("virtual_model") is None:
                continue
            cid, v = str(r["cell_id"]), str(r["virtual_model"])
            if cid in vmodels and vmodels[cid] != v:
                refuse(f"{rt_where}: route for cell {cid!r} names virtual model {v!r} but {ENDPOINT_FILE}.vmodels names {vmodels[cid]!r} — "
                       "two records of one standup disagree")
    tt_map = ep_map if ep_map is not None else (cs_map or {})
    records["traintest_spellings"] = (f"{ENDPOINT_FILE} cells<->harness_cells" if ep_map is not None
                                      else (f"{CELLS_SERVED_FILE} cells_requested<->harness_cells" if cs_map else "absent"))
    return StandupIdentity(where=where, vmodels=vmodels, harness_of_traintest=dict(tt_map),
                           traintest_of_harness={h: t for t, h in tt_map.items()}, records=records)


def read_standup_identity(root_dir: Path, *, endpoint: Optional[Mapping[str, Any]] = None) -> StandupIdentity:
    """The standup's identity off the attempt root's `exp/serve/` records (:func:`standup_file`: the STANDUP's copy, never a GSM8K
    per-cell one — F-AE): `endpoint.json` (required; `endpoint` when the caller already read it), `cells_served.json` and
    `routing_table.json` when present."""
    ep_p = standup_endpoint_path(root_dir)
    ep = endpoint if endpoint is not None else read_json(ep_p)
    if not isinstance(ep, dict):
        refuse(f"{ep_p}: not a JSON object")
    cs_p = standup_file(root_dir, CELLS_SERVED_FILE, required=False)
    rt_p = standup_file(root_dir, ROUTING_TABLE_FILE, required=False)
    cs = read_json(cs_p) if cs_p is not None else None
    rt = read_json(rt_p) if rt_p is not None else None
    if cs is not None and not isinstance(cs, dict):
        refuse(f"{cs_p}: not a JSON object")
    if rt is not None and not isinstance(rt, dict):
        refuse(f"{rt_p}: not a JSON object")
    return standup_identity(ep, endpoint_path=ep_p, cells_served=cs, cells_served_path=cs_p, routing_table=rt, routing_table_path=rt_p)


def standup_vmodel_of(root_dir: Path, arm: str) -> VmodelBinding:
    """The ONE resolver every reader that hands a store's arm goes through (F-AN): `arm` in either spelling -> the virtual model the
    attempt root's standup served it as, through the standup's own records (:func:`read_standup_identity`), recording which spelling
    matched. An arm the standup did not serve under either spelling still refuses by name."""
    return read_standup_identity(root_dir).resolve(arm)


@dataclass
class Standup:
    """One attempt's standup as its records spell it. `provenance` says which files were read and how (`standup_jobs_jsonl`:
    present | absent; `endpoint_path`; `W_source`); every table that consumes the record carries :attr:`record_source` so a derived
    standup is never read as a recorded one."""

    root: AttemptRoot
    endpoint: Dict[str, Any]
    attestation: Dict[str, Any]
    standup_jobs: List[dict]
    endpoint_path: Path
    attestation_path: Path
    standup_jobs_path: Optional[Path]
    provenance: Dict[str, Any] = field(default_factory=dict)
    #: F-AN: the cell identity the standup's records spell (`read_standup` fills it from `exp/serve/`); a Standup built without one
    #: resolves off its endpoint record alone (`standup_identity(endpoint)`), which binds the harness spelling only
    identity: Optional[StandupIdentity] = None

    @property
    def record_source(self) -> str:
        return STANDUP_RECORD_FROM_JOBS if self.standup_jobs else STANDUP_RECORD_DERIVED

    @property
    def W(self) -> str:
        """The tree the standup served from: `standup_jobs.jsonl[0].sha` when the record exists (a line without one refuses); else
        ATTEMPT.json's W — the app stands the engine up from the tree at W, and `provenance["W_source"]` records the derivation."""
        if self.standup_jobs:
            W = self.standup_jobs[0].get("sha")
            if not isinstance(W, str) or not W:
                refuse(f"{self.standup_jobs_path}: line 0 carries no sha (the tree the standup served from)")
            return W
        return self.root.W

    @property
    def vmodels(self) -> Dict[str, str]:
        vm = self.endpoint.get("vmodels")
        if not isinstance(vm, dict) or not vm:
            refuse(f"{self.endpoint_path}: no `vmodels` map (cell → virtual model) — the arm cannot be bound to a vmodel")
        return {str(k): str(v) for k, v in vm.items()}

    def binding_of(self, arm: str) -> VmodelBinding:
        """`arm` in EITHER spelling (the TrainTest cell id a static / BoN / multi-turn store dir is named by, or the served harness id) ->
        its :class:`VmodelBinding`, through the standup's records; an unserved arm refuses by name."""
        ident = self.identity
        if ident is None:
            ident = standup_identity(self.endpoint, endpoint_path=self.endpoint_path)
        return ident.resolve(arm)

    def vmodel_of(self, arm: str) -> str:
        return self.binding_of(arm).vmodel

    def serving_identity_of(self, harness_id: str) -> Optional[str]:
        """How the standup's OWN records say cell `harness_id` is served: `"lora"` | `"passthrough"` (the base weights) | None (no record names
        the cell). Read from `cells_served.json`'s canonical / controls rows and `routing_table.json`'s routes (`serving_identity`:
        `proxy_vllm+lora+...` / `proxy_vllm+passthrough+...`; a row's `kind` when it states no identity); two records that disagree refuse."""
        found: Dict[str, str] = {}
        for name, rows_of in ((CELLS_SERVED_FILE, lambda d: list(d.get("canonical") or []) + list(d.get("controls") or [])),
                              (ROUTING_TABLE_FILE, lambda d: list(d.get("routes") or []))):
            p = standup_file(self.root.path, name, required=False)
            if p is None:
                continue
            d = read_json(p)
            for row in (rows_of(d) if isinstance(d, dict) else []):
                if not isinstance(row, dict) or str(row.get("cell_id")) != str(harness_id):
                    continue
                si = str(row.get("serving_identity") or "")
                v = "lora" if "+lora" in si else ("passthrough" if "passthrough" in si else
                                                  ("passthrough" if row.get("kind") == "none" else ("lora" if row.get("kind") in ("lora", "vector_all") else None)))
                if v is not None:
                    found[name] = v
        if len(set(found.values())) > 1:
            refuse(f"{self.endpoint_path.parent}: the standup's records disagree on how cell {harness_id!r} is served ({found}) -- neither is picked")
        return next(iter(found.values()), None)

    def engine_lora_expectation(self, arm: str) -> Tuple[Optional[str], str]:
        """fc6: the LoRA name the ENGINE must state (`lora_request: LoRARequest(lora_name=...)`) on every request of `arm`,
        derived from the standup's AUTHORITATIVE records -- `serving_attestation.json.adapters` (the adapters the engine registered, keyed by
        the LoRA name it serves them under == the harness cell id; `id_name_map` the engine's int id -> that name) cross-checked with the
        cell's `serving_identity` in `cells_served.json` / `routing_table.json` -- and WHERE that came from. `(name, source)` for a fitted
        arm; `(None, source)` ONLY for an arm the records say is served on the base weights (`none`, a prompted cell: no adapter attested AND
        routed passthrough). Never inferred from what the engine log happens to contain; records that cannot establish it refuse by name."""
        b = self.binding_of(arm)
        h = b.harness_id
        adapters = self.attestation.get("adapters")
        if not isinstance(adapters, dict):
            refuse(f"{self.attestation_path}: no `adapters` map -- a lora_native standup attests the adapters it registered; whether arm {arm!r} "
                   f"(cell {h!r}) is LoRA-served cannot be established")
        route = self.serving_identity_of(h)
        id_names = {str(v) for v in (self.attestation.get("id_name_map") or {}).values()} if isinstance(self.attestation.get("id_name_map"), dict) else set()
        if h in adapters:
            if route == "passthrough":
                refuse(f"{self.attestation_path}: adapters attests cell {h!r} as a registered LoRA but the standup routes it on the base weights "
                       f"(passthrough) -- the records disagree; neither is picked")
            if id_names and h not in id_names:
                refuse(f"{self.attestation_path}: adapters attests {h!r} but id_name_map {sorted(id_names)} names no such LoRA -- the engine's own "
                       "registration does not know this adapter")
            return h, f"{ATTESTATION_FILE}.adapters[{h!r}] (lora_int_id {adapters[h].get('lora_int_id') if isinstance(adapters[h], dict) else '?'}); route {route or 'unrecorded'}"
        if route == "lora":
            refuse(f"{self.endpoint_path.parent}: cell {h!r} is routed as a LoRA cell (serving_identity lora) but {ATTESTATION_FILE}.adapters attests no "
                   f"adapter for it -- the records disagree; neither is picked")
        if route != "passthrough":
            # fc8: a base-weight identity is READ from a recorded passthrough route, never inferred from the cell's name -- `none` and a
            # `prompted__*` cell included; a standup record that establishes no identity for the cell (the identity files absent, or silent on
            # it: a partially pulled record) gets a refusal, never a positive identity verdict
            refuse(f"{self.endpoint_path.parent}: the standup's records establish no serving identity for cell {h!r} (arm {arm!r}) -- no attested adapter in "
                   f"{ATTESTATION_FILE} and no recorded passthrough route in {CELLS_SERVED_FILE} / {ROUTING_TABLE_FILE} (absent, or silent on the cell); a base-weight "
                   "identity is never inferred from a cell's name")
        return None, f"{ATTESTATION_FILE}.adapters names no adapter for {h!r}; served on the base weights per the recorded passthrough identity ({CELLS_SERVED_FILE} / {ROUTING_TABLE_FILE})"

    @property
    def S(self) -> int:
        s = self.endpoint.get("attested_max_num_seqs")
        if s is None:
            refuse(f"{self.endpoint_path}: attested_max_num_seqs missing")
        return int(s)

    @property
    def L(self) -> int:
        l = self.endpoint.get("attested_max_model_len")
        if l is None:
            refuse(f"{self.endpoint_path}: attested_max_model_len missing")
        return int(l)


def attestation_geometry_problems(endpoint: Mapping[str, Any], attestation: Mapping[str, Any]) -> List[str]:
    """The endpoint's attested S / L held to the workers' OWN reports (`serving_attestation.scheduler_geometry.worker_reports[*]
    .max_num_seqs / .max_model_len` — runtime evidence read inside every worker). What grounds a standup derived without
    `standup_jobs.jsonl`: the geometry the tables read is the engine's, not one file's copy of it."""
    problems: List[str] = []
    sg = attestation.get("scheduler_geometry")
    reports = sg.get("worker_reports") if isinstance(sg, dict) else None
    if not isinstance(reports, list) or not reports:
        return [f"{ATTESTATION_FILE}: scheduler_geometry.worker_reports absent or empty — no worker attested the geometry"]
    for k_ep, k_wr in (("attested_max_num_seqs", "max_num_seqs"), ("attested_max_model_len", "max_model_len")):
        ep_v = endpoint.get(k_ep)
        if ep_v is None:
            problems.append(f"{ENDPOINT_FILE}: {k_ep} missing")
            continue
        wr_vals = sorted({r.get(k_wr) for r in reports if isinstance(r, dict)}, key=str)
        if wr_vals != [ep_v]:
            problems.append(f"{k_ep} {ep_v!r} != the workers' {k_wr} {wr_vals!r}")
    return problems


def read_standup(root: AttemptRoot) -> Standup:
    """The attempt's standup from its records under `exp/serve/` (:func:`standup_file`): the endpoint record (`endpoint.json` — the
    READY bytes, which the Modal app restores at the same path after teardown), the attestation, and `standup_jobs.jsonl` when the
    standup wrote one (its first line must carry
    `STANDUP_JOBS_REQUIRED_KEYS`). ABSENT `standup_jobs.jsonl` (the Modal app wrote none) does not refuse: the identity
    the tables read — attested L / S, thinking, geometry, vmodels — lives in the endpoint record and is held to the workers' reports
    in the attestation (:func:`attestation_geometry_problems`); the tree sha comes from ATTEMPT.json. The derivation is RECORDED in
    `provenance` (`standup_jobs_jsonl: absent`, `derived_from`, `W_source`) and surfaces on every consuming table as
    `Standup.record_source`. What the derivation cannot supply is the INDEPENDENT sha cross-check (`engine_identity_table`):
    it is recorded as not applicable, never as passed."""
    ep_p = standup_endpoint_path(root.path)
    at_p = standup_file(root.path, ATTESTATION_FILE)  # type: ignore[assignment]
    sj_p = standup_file(root.path, STANDUP_JOBS_FILE, required=False)
    ep = read_json(ep_p)
    att = read_json(at_p)
    if not isinstance(ep, dict):
        refuse(f"{ep_p}: not a JSON object")
    if not isinstance(att, dict):
        refuse(f"{at_p}: not a JSON object")
    prov: Dict[str, Any] = {"endpoint_file": ep_p.name, "endpoint_path": str(ep_p), "attestation_path": str(at_p),
                            "standup_dir": SERVE_DIR_REL if standup_dir(root.path) is not None else "(pre-layout root: whole-root walk)"}
    sj: List[dict] = []
    if sj_p is not None:
        sj = read_jsonl(sj_p)
        if not sj:
            refuse(f"{sj_p}: empty — the standup wrote no job line")
        if not isinstance(sj[0], dict):
            refuse(f"{sj_p}: line 0 is not a JSON object")
        missing = [k for k in STANDUP_JOBS_REQUIRED_KEYS if k not in sj[0]]
        if missing:
            refuse(f"{sj_p}: line 0 lacks {missing} — the standup record's required keys are {list(STANDUP_JOBS_REQUIRED_KEYS)} "
                   f"(present: {sorted(sj[0])[:12]})")
        prov.update(standup_jobs_jsonl="present", standup_jobs_path=str(sj_p), W_source=f"{STANDUP_JOBS_FILE}[0].sha")
    else:
        problems = attestation_geometry_problems(ep, att)
        if problems:
            refuse(f"{root.path}: no {STANDUP_JOBS_FILE} under {SERVE_DIR_REL}/ and the standup cannot be derived from the attested "
                   f"records: " + "; ".join(problems))
        prov.update(standup_jobs_jsonl="absent", derived_from=[ep_p.name, ATTESTATION_FILE, ATTEMPT_FILE],
                    W_source=f"{ATTEMPT_FILE} W ({STANDUP_JOBS_FILE} absent; the app stands up from the tree at W — the sha "
                             "cross-check is not applicable)",
                    geometry_check=f"{ENDPOINT_FILE} attested S/L == {ATTESTATION_FILE} scheduler_geometry.worker_reports")
    identity = read_standup_identity(root.path, endpoint=ep)
    prov["cell_identity_records"] = dict(identity.records)
    return Standup(root=root, endpoint=ep, attestation=att, standup_jobs=sj, endpoint_path=ep_p, attestation_path=at_p, identity=identity,
                   standup_jobs_path=sj_p, provenance=prov)


def kernel_ceiling_adjustment(endpoint: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The `lora_kernel_index_ceiling` step of `endpoint.geometry.adjustments` (`geometry.KERNEL_ADJUSTMENT_
    FIELDS` = kind, from_S, to_S, D, ceiling), or None when the geometry recorded no such step."""
    geom = endpoint.get("geometry") if isinstance(endpoint.get("geometry"), dict) else {}
    for adj in geom.get("adjustments") or []:
        if isinstance(adj, dict) and adj.get("kind") == "lora_kernel_index_ceiling":
            return dict(adj)
    return None


# --------------------------------------------------------------------------- #
# Engine log                                                                    #
# --------------------------------------------------------------------------- #

#: vLLM 0.25.1 `request_logger.py` INFO line under `--enable-log-requests` — the id only (the params are
#: `record_resolved_sampler.py`'s business; the audit binds ids)
_RECEIVED_RE = re.compile(r"Received request ([^\s:]+):")


def engine_request_ids(log_path: Path) -> List[str]:
    """Every `Received request <id>` id in the engine log, in order, duplicates kept (a duplicate id is a
    binding ambiguity the caller reports)."""
    p = Path(log_path)
    opener = gzip.open if p.suffix == ".gz" else open
    ids: List[str] = []
    with opener(p, "rt", encoding="utf-8", errors="replace") as f:  # type: ignore[call-arg]
        for line in f:
            m = _RECEIVED_RE.search(line)
            if m:
                ids.append(m.group(1))
    return ids


_PARAMS_LINE_RE = re.compile(r"Received request ([^\s:]+): params: SamplingParams\((.*?)\)(?:, lora_request: (None|LoRARequest\(lora_name='([^']*)'))?")
_PARAM_KEYS: Tuple[str, ...] = ("seed", "max_tokens", "temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty", "logprobs")


def _param_value(body: str, key: str) -> Any:
    m = re.search(r"(?:^|, )" + re.escape(key) + r"=([^,)]+)", body)
    if not m:
        return None
    raw = m.group(1).strip()
    if raw == "None":
        return None
    try:
        return int(raw) if re.fullmatch(r"-?[0-9]+", raw) else float(raw)
    except ValueError:
        return raw


def engine_request_params(log_path: Path) -> Dict[str, dict]:
    """Every `Received request <id>: params: SamplingParams(...)` line of the engine log parsed to what the controls-events binding
    holds a draw to -- `seed`, `max_tokens`, the six sampler keys, `logprobs`, and `lora_name` (the LoRA the engine served the request under;
    None = the base weights) -- keyed by the engine's id, FIRST occurrence kept (a duplicate id is the audit's own finding, `dup_engine_ids`).
    vLLM 0.25.1 `request_logger.py:63`; a line the pattern does not match is simply not indexed (a draw bound to it then refuses by name)."""
    p = Path(log_path)
    opener = gzip.open if p.suffix == ".gz" else open
    out: Dict[str, dict] = {}
    with opener(p, "rt", encoding="utf-8", errors="replace") as f:  # type: ignore[call-arg]
        for line in f:
            m = _PARAMS_LINE_RE.search(line)
            if not m:
                continue
            eid, body, lora, lora_name = m.group(1), m.group(2), m.group(3), m.group(4)
            if eid in out:
                continue
            rec = {k: _param_value(body, k) for k in _PARAM_KEYS}
            rec["lora_name"] = lora_name if (lora is not None and lora != "None") else None
            rec["lora_request_stated"] = lora is not None
            out[eid] = rec
    return out


def find_engine_log(root: AttemptRoot) -> Path:
    """The standup's engine log under `exp/serve/` (`GSM8K_VLLM_LOG_REL`; the per-cell serve dirs copy no log), else the pre-layout walk."""
    return standup_file(root.path, ENGINE_LOG_FILES[0], alternatives=ENGINE_LOG_FILES[1:])  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Seed rules — imported by name from each leg's writer; unavailable = refuse, never guess                    #
# --------------------------------------------------------------------------- #

from sisteer.eval.attacks.static_common import (StaticStoreIncompleteError, assert_static_store_complete,  # noqa: E402
                                                static_sampling_seed)
from sisteer.eval.attacks.bon import bon_decode_seed_v1  # noqa: E402
from sisteer.eval.sampled_controls import sampling_seed as controls_sampling_seed  # noqa: E402

try:  # the multi-turn rule is static_sampling_seed with the turn as the sample index (pyrit_adapters docstring)
    from sisteer.eval.attacks.pyrit_adapters import multiturn_seed as _multiturn_seed
    MULTITURN_SEED_NOTE = "sisteer.eval.attacks.pyrit_adapters.multiturn_seed"
except Exception as _e:  # noqa: BLE001 — pyrit is an optional dependency on some trees
    def _multiturn_seed(seed_base, trial, attack, goal_id, turn):  # type: ignore[misc]
        return static_sampling_seed(seed_base, trial, attack, goal_id, turn)
    MULTITURN_SEED_NOTE = (f"static_common.static_sampling_seed(seed_base, trial, attack, goal_id, turn) — the rule "
                           f"pyrit_adapters.multiturn_seed delegates to (module unimportable here: {type(_e).__name__})")

SeedRule = Callable[[str, str, int], int]


def _misalign_seed_local(seed_base: int, behavior: str, item_id: str, sample_idx: int) -> int:
    """The local spelling of the seed rule: sha256(f"{s}|misalign|{behavior}|{item_id}|{sample_idx}")[:4] big-endian."""
    key = f"{int(seed_base)}|misalign|{behavior}|{item_id}|{int(sample_idx)}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big")


def misalign_seed_rule() -> Tuple[Callable[[int, str, str, int], int], str]:
    """`misalign.runner.misalign_seed` when landed (cross-checked against the local spelling on one
    probe — drift refuses), else the local spelling with the note saying so."""
    from sisteer.capeval.misalign import runner as _runner
    fn = getattr(_runner, "misalign_seed", None)
    if fn is None:
        return _misalign_seed_local, "plan §6 spelling (misalign.runner.misalign_seed not on this tree)"
    probe = (7, "leaking", "scenario-1", 3)
    if int(fn(*probe)) != _misalign_seed_local(*probe):
        refuse("misalign.runner.misalign_seed disagrees with plan §6's spelling sha256(f'{s}|misalign|{behavior}|{item_id}|"
               f"{{sample_idx}}')[:4] on probe {probe} — the audit will not guess which is the store's rule")
    return fn, "sisteer.capeval.misalign.runner.misalign_seed"


def petri_seed_rule() -> Tuple[Optional[Callable[[int, str, str, int], int]], str]:
    """`petri_runner.petri_target_seed(seed_base, behaviour, scenario_id, turn)` when landed; else
    (None, why) — the Petri seed gate then REFUSES by name (the audit never invents a seed rule)."""
    fn = getattr(_petri, "petri_target_seed", None)
    if fn is None:
        return None, "petri_runner.petri_target_seed is not on this tree (WS1-B's rule has not landed)"
    return fn, "sisteer.capeval.tiers.petri_runner.petri_target_seed"


#: test / lane injection point: a callable `(seed_base, behaviour, scenario_id, turn) -> int` set here overrides
#: the import (the tests inject a synthetic Petri rule; production leaves it None)
PETRI_SEED_RULE_OVERRIDE: Optional[Callable[[int, str, str, int], int]] = None


# --------------------------------------------------------------------------- #
# Expected sampling rows per leg                                                #
# --------------------------------------------------------------------------- #

MODEL_KEY = "qwen35_4b"
#: the vendor rows' EXACT registry refs (the three new rows are `is_default 0` / absent on some trees, so
#: they are resolved by `PolicyFile.by_ref(ref + '@v1')` with `superseded_by_policy_id is None` asserted when
#: a policy file is given, never by default lookup)
VENDOR_REFS: Dict[str, str] = {"bon": "qwen35_4b_it|bon|off|vendor_preset",
                               "multi_turn": "qwen35_4b_it|multi_turn|off|vendor_preset",
                               "petri": "qwen35_4b_it|petri|off|vendor_preset"}
SAMPLING_MODE_OF_LEG: Dict[str, str] = {**{l: C.SAMPLING_VENDOR_PRESET for l in LEGS},
                                        "gsm8k": "thinking_card", "mmlu": C.SAMPLING_GREEDY}
#: the wire's thinking switch every TASK record must show, per leg: GSM8K renders
#: thinking ON through `chat_template_kwargs.enable_thinking true`; every other leg's body carries NO switch
#: (the standup's `--default-chat-template-kwargs` renders OFF server-side) → the NO-SWITCH reading
#: (`request_evidence_no_switch_absent`, `expected_enable_thinking=None`). Overridable per leg on the CLI.
THINKING_EXPECT_DEFAULT: Dict[str, Optional[bool]] = {**{l: None for l in LEGS}, "gsm8k": True}


def _row_from_policy(pol: Any, source: str) -> "C.SamplingRow":
    n = pol.numbers()
    return C.SamplingRow(temperature=float(n["temperature"]),
                         top_p=None if n.get("top_p") is None else float(n["top_p"]),
                         top_k=None if n.get("top_k") is None else int(n["top_k"]),
                         source=source,
                         min_p=None if n.get("min_p") is None else float(n["min_p"]),
                         presence_penalty=None if n.get("presence_penalty") is None else float(n["presence_penalty"]),
                         repetition_penalty=None if n.get("repetition_penalty") is None else float(n["repetition_penalty"]))


def expected_sampling(leg: str, *, policy_file: Optional[Path] = None, model_key: str = MODEL_KEY
                      ) -> Tuple["C.SamplingRow", str]:
    """The frozen row a leg's TARGET requests must show on the wire, and where it came from. BoN / multi-turn /
    Petri resolve their EXACT registry refs when `policy_file` is given (refusing a superseded or absent row);
    without a policy file every vendor leg reads `contracts.VENDOR_PRESET_SAMPLING_BY_MODEL_KEY` (the same six
    numbers, registry-backed through the contracts' own reader); GSM8K the `thinking_card` row; MMLU greedy."""
    mode = SAMPLING_MODE_OF_LEG[leg]
    if mode == C.SAMPLING_GREEDY:
        return C.sampling_for(model_key, C.SAMPLING_GREEDY), "contracts.sampling_for(greedy)"
    if mode == "thinking_card":
        return C.sampling_for(model_key, "thinking_card"), "contracts.sampling_for(thinking_card)"
    ref_key = "bon" if leg == "bon" else ("multi_turn" if leg in MULTITURN_ATTACKS else ("petri" if leg == "petri" else None))
    if ref_key is not None and policy_file is not None:
        from sisteer.registry.decode_policy import PolicyFile
        pf = PolicyFile.load(str(policy_file))
        ref = VENDOR_REFS[ref_key] + "@v1"
        try:
            pol = pf.by_ref(ref)
        except Exception as e:  # noqa: BLE001
            refuse(f"{policy_file}: registry row {ref!r} not resolvable ({type(e).__name__}: {e}) — the {leg} leg's "
                   "decode contract has no registered row (D9: exact ref, never a default lookup)")
        if getattr(pol, "superseded_by_policy_id", None) is not None:
            refuse(f"{policy_file}: registry row {ref!r} is superseded by {pol.superseded_by_policy_id!r}")
        return _row_from_policy(pol, f"PolicyFile.by_ref({ref!r})"), f"registry {ref}"
    row = C.VENDOR_PRESET_SAMPLING_BY_MODEL_KEY.get(model_key)
    if row is None:
        refuse(f"no vendor_preset sampling row for model key {model_key!r}")
    return row, f"contracts.VENDOR_PRESET_SAMPLING_BY_MODEL_KEY[{model_key!r}]"


# --------------------------------------------------------------------------- #
# Store reads — one per leg, each from the writer's OWN row schema                                          #
# --------------------------------------------------------------------------- #

@dataclass
class Draw:
    """One stored draw = one filed (task, item, sample) and the attempt it claims to bind to."""

    task: str
    item_id: str
    sample_idx: int
    request_id: Optional[str]
    completion_id: Optional[str]
    text: Optional[str] = None
    text_sha256: Optional[str] = None
    finish_reason: Optional[str] = None
    at_cap: Optional[bool] = None
    #: static ladder rows (escalate_static's copy-forward provenance): the rung that PRODUCED this text (a base row: the
    #: store's cap; a copied-forward rung row: the lower rung it was drawn at — its request_id / completion_id bind on
    #: THAT rung's accepted attempt root, never this one) and the writer's `regenerated` flag (None on a base row)
    source_cap: Optional[int] = None
    regenerated: Optional[bool] = None


@dataclass
class StoreRead:
    leg: str                       # table leg
    arm: str                       # the served cell id (`none`, `<cell>`, `prompted__…`, `arditi_allpos`)
    s: Optional[int]               # the seed the STORE records (None = the store records no base: MMLU)
    trial: Optional[int]
    seed_base: Optional[int]
    tasks: Tuple[str, ...]
    filed: Set[Tuple[str, str]]
    draws: List[Draw]
    vmodel: Optional[str]          # the store's OWN recorded virtual model (None = the store records none)
    expected_seed: Optional[SeedRule]
    seed_rule_note: str
    store_dir: Path
    complete: bool
    completeness_note: str
    n_rows: int
    notes: List[str] = field(default_factory=list)
    #: fc11k: the EXACT candidate store files (`responses.jsonl` / `rows.jsonl` / `bon_responses.jsonl`, absolute) this reader consumed -- set at discovery,
    #: what `inventory_stores` compares the root's walk against (a candidate no reader consumed is unclassified, whatever its ancestors)
    consumed_files: List[str] = field(default_factory=list)
    #: the store's own cap (static: the base cap, or a rung root's cap_to) — None where the notion does not apply. Setting it ACTIVATES the
    #: static rung / copy-forward semantics of :meth:`filed_here` (only draws made at this cap count), so a store without that notion never sets it
    cap: Optional[int] = None
    #: METADATA only: the per-reply cap the driver's final manifest records for a multi-turn store (identity), with
    #: NO effect on :meth:`filed_here` -- a multi-turn store keeps EVERY declared goal filed here, zero-draw goals included
    reply_cap: Optional[int] = None
    #: True for the Inspect-family writers (sampled controls, tinycap) whose rows CANNOT carry request ids: their binding is the
    #: `(request_id, completion_id)` the response hook recorded in `response_evidence.jsonl` per (task, sample, vmodel), and an
    #: id-less draw resolves exactly one such attempt on the evidence chosen for it; the aiohttp drivers (static, BoN,
    #: multi-turn, misalign) record the ids on the row, so an id-less row of theirs is a defect, never resolved around
    binds_by_response_evidence: bool = False

    def copied_cap(self, d: Draw) -> Optional[int]:
        """The cap a COPIED-FORWARD draw was produced at (escalate_static's `source_cap` below this store's own cap — its
        request is an attempt on THAT rung's accepted root, never this one); None for a draw made at this store's cap or on
        a store without the notion."""
        return d.source_cap if (self.cap is not None and d.source_cap is not None and d.source_cap != self.cap) else None

    @property
    def copied_draws(self) -> List[Draw]:
        return [d for d in self.draws if self.copied_cap(d) is not None]

    @property
    def filed_here(self) -> Set[Tuple[str, str]]:
        """The filed items with at least one draw made at THIS store's cap = the items whose requests are on THIS attempt
        root's evidence (a rung store's copied-forward items are filed but were drawn on an ancestor root); the whole filed
        set on a store without the notion."""
        if self.cap is None:
            return self.filed
        return {(d.task, d.item_id) for d in self.draws if d.source_cap == self.cap}


_SEEDREP_NS_RE = re.compile(r"_seedrep_s([1-9][0-9]*)_")
#: the app's controls child namespace (`seedrep_launch_plan.CONTROLS_OUT_REL` = `exp/sampled/controls_cap<cap>__s<s>`) -- the REAL
#: shape of a controls attempt root (verified on a controls-arditi root: `exp/sampled/controls_cap512__s1/cells/<cell>`); the Arditi
#: MMLU store records no seed base, so this is the ONLY place its seed is spelled
_CONTROLS_NS_RE = re.compile(r"controls_cap[0-9]+__s([1-9][0-9]*)(?:/|$)")


def _s_from_namespace(path: Path) -> Optional[int]:
    """The seed spelled by a lane namespace in the path (`…_seedrep_s<s>_sampled_t0…`; or the app's `controls_cap<cap>__s<s>`
    controls child dir), else None; the two spellings disagreeing refuses."""
    text = str(path)
    m1, m2 = _SEEDREP_NS_RE.search(text), _CONTROLS_NS_RE.search(text)
    if m1 and m2 and m1.group(1) != m2.group(1):
        refuse(f"{path}: the path spells two seeds (`_seedrep_s{m1.group(1)}_` and `controls_cap*__s{m2.group(1)}`) -- its seed cannot be established")
    m = m1 or m2
    return int(m.group(1)) if m else None


def _binding(row: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """The (request_id, completion_id) a stored row carries — top-level (the writer's additive fields) or inside a
    JSON `extra_json` / `extra` block; absent → (None, None), which the audit refuses by name."""
    rid = row.get("request_id")
    cid = row.get("completion_id")
    if rid is None or cid is None:
        extra = row.get("extra_json")
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except json.JSONDecodeError:
                extra = None
        if not isinstance(extra, dict):
            extra = row.get("extra") if isinstance(row.get("extra"), dict) else {}
        rid = rid if rid is not None else extra.get("request_id")
        cid = cid if cid is not None else extra.get("completion_id")
    return (str(rid) if rid is not None else None, str(cid) if cid is not None else None)


#: the seed bases an Arditi-engine record may name in `extra.s` — the literal seeds of the three views
VALID_SEED_BASES: Tuple[int, ...] = tuple(int(seed_of_view(v)) for v in SEED_VIEWS)  # type: ignore[arg-type]


def record_extra_s(rec: Mapping[str, Any]) -> Optional[int]:
    """The seed base the client stamped on a request-evidence record's EXTRA block (`extra: {leg, arm, s, attempt_id}`,
    fields the reconciliation ignores); None when absent or not an int."""
    extra = rec.get("extra")
    v = extra.get("s") if isinstance(extra, Mapping) else None
    return int(v) if isinstance(v, int) and not isinstance(v, bool) else None


def extra_s_by_request_id(records: Iterable[Mapping[str, Any]]) -> Dict[str, Optional[int]]:
    """request_id → `extra.s` over the TASK records (None = the record carries no valid seed attribution)."""
    return {str(r["request_id"]): record_extra_s(r) for r in records
            if r.get("kind") == C.REQUEST_EVIDENCE_KIND_TASK and r.get("request_id")}


ATTEMPT_SOURCES: Tuple[str, ...] = ("ledger", "response_evidence")


def partition_identity_problems(records: Iterable[Mapping[str, Any]], attempt_rows: Iterable[Mapping[str, Any]], *,
                                task: str, vmodel: str, attempt_source: str = "ledger") -> List[str]:
    """The identity clause of `contracts.request_evidence_reconciliation` restated for ONE (task, vmodel) partition, shared
    by every table that reads an attempt partition: every TASK record of the partition has an attempt and every
    attempt its record — an orphan on either side is named — and a matched pair agrees on its task / sample labels. Digests /
    sampling stay the canonical reconciliation's business (the request-evidence audit); this is the prerequisite a share read
    cannot skip. `attempt_rows` are the usage ledger's rows (`attempt_source="ledger"`) or, on a `not_on_path` row, the
    response hook's rows (`"response_evidence"`, one per attempt that returned or died); an attempt row joins the partition the
    way the store verdict admits it (`request_evidence_audit.store_partition`, fz3/fz4): by its OWN labels when it carries a
    vmodel (a ledger row always does), through its REQUEST RECORD when it carries none (the response hook's record of a FAILED
    attempt has no body to read `model` off), and a row naming NO record and no vmodel joins every partition so it refuses by
    name rather than hiding in none."""
    if attempt_source not in ATTEMPT_SOURCES:
        refuse(f"attempt_source {attempt_source!r} is not one of {list(ATTEMPT_SOURCES)} — the attempt inventory is unnamed")
    side = "ledger" if attempt_source == "ledger" else "response-evidence"
    side_long = "ledger" if attempt_source == "ledger" else "response evidence"
    recs_by_id: Dict[str, Mapping[str, Any]] = {}
    all_rec_ids: Set[str] = set()
    for r in records:
        if r.get("kind") != C.REQUEST_EVIDENCE_KIND_TASK or not r.get("request_id"):
            continue
        rid = str(r["request_id"])
        all_rec_ids.add(rid)
        if r.get("task") == task and r.get("vmodel") == vmodel:
            recs_by_id[rid] = r
    rows_by_id: Dict[str, Mapping[str, Any]] = {}
    for row in attempt_rows:
        rid = row.get(C.LEDGER_REQUEST_ID_FIELD)
        if not rid:
            continue
        rid = str(rid)
        if row.get("vmodel") is not None:
            admitted = row.get("task") == task and row.get("vmodel") == vmodel
        else:
            admitted = rid in recs_by_id or rid not in all_rec_ids
        if admitted:
            rows_by_id[rid] = row
    out: List[str] = []
    no_attempt = sorted(set(recs_by_id) - set(rows_by_id))
    if no_attempt:
        out.append(f"{len(no_attempt)} request-evidence record(s) of ({task!r}, {vmodel!r}) have NO {side} attempt (first: {no_attempt[:3]})")
    no_record = sorted(set(rows_by_id) - set(recs_by_id))
    if no_record:
        out.append(f"{len(no_record)} {side} attempt(s) of ({task!r}, {vmodel!r}) have NO request-evidence record (first: {no_record[:3]})")
    for rid in sorted(set(recs_by_id) & set(rows_by_id)):
        rec, row = recs_by_id[rid], rows_by_id[rid]
        if rec.get("task") != row.get("task") or rec.get("sample") != row.get("sample"):
            out.append(f"request_id {rid!r}: the {side_long} labels the attempt task {row.get('task')!r} sample {row.get('sample')!r} but "
                       f"the record says task {rec.get('task')!r} sample {rec.get('sample')!r}")
    return out


def owned_evidence_files(root: Path, name: str) -> List[Tuple[Path, str]]:
    """Every file called `name` under the attempt root, each with its OWNER: `OWNER_ATTEMPT_ROOT` for `<root>/<name>` (a rung consumer's
    evidence, or the single-ledger shape), `OWNER_CHILD_STORE` for a copy under `<root>/exp/…` (a base child's store root). A copy
    anywhere else (under `children/`, `audits/`, the standup dir…) has no writer in the layout and REFUSES by name — never selected,
    never skipped."""
    root = Path(root)
    out: List[Tuple[Path, str]] = []
    for p in walk_files(root, name):
        rel = p.relative_to(root)
        if rel.parts == (name,):
            out.append((p, OWNER_ATTEMPT_ROOT))
        elif rel.parts[0] == CHILD_STORE_TREE_REL and len(rel.parts) > 2:
            out.append((p, OWNER_CHILD_STORE))
        else:
            refuse(f"{p}: a {name!r} that is neither the attempt root's nor under a child store root ({CHILD_STORE_TREE_REL}/…) — an "
                   "unowned evidence file is not read")
    return out


def _read_owned_rows(root: Path, files: Sequence[Tuple[Path, str]], *, name: str, id_field: str) -> List[dict]:
    """The rows of every owned file, each stamped `ROW_SOURCE_FIELD` (root-relative path); a request id that appears in TWO files
    refuses by name (one request is served — and recorded — exactly once; a silent merge would let one file's row shadow another's)."""
    root = Path(root)
    rows: List[dict] = []
    seen: Dict[str, str] = {}
    for p, _owner in sorted(files):
        src = str(p.relative_to(root))
        for r in read_jsonl(p):
            if not isinstance(r, dict):
                refuse(f"{p}: a non-object row")
            rid = r.get(id_field)
            if rid:
                prior = seen.get(str(rid))
                if prior is not None and prior != src:
                    refuse(f"{root}: {id_field} {rid!r} appears in two {name!r} files ({prior} and {src}) — one request is recorded once; "
                           "a collision across child stores is not merged")
                seen[str(rid)] = src
            rows.append({**r, ROW_SOURCE_FIELD: src})
    return rows


def read_owned_jsonl(root_dir: Path, name: str, *, id_field: str, required: bool = True) -> List[dict]:
    """A root's `name` rows from EVERY owner (:func:`owned_evidence_files`) on a root with the declared layout (:func:`standup_dir`);
    a pre-layout root reads exactly one file by :func:`find_one`, as before. None refuses when `required`."""
    root_dir = Path(root_dir)
    if standup_dir(root_dir) is None:
        p = find_one(root_dir, name, required=required)
        return _read_owned_rows(root_dir, [(p, OWNER_ATTEMPT_ROOT)], name=name, id_field=id_field) if p is not None else []
    files = owned_evidence_files(root_dir, name)
    if not files:
        if required:
            refuse(f"{root_dir}: no {name!r} at the attempt root nor under any child store root ({CHILD_STORE_TREE_REL}/…)")
        return []
    return _read_owned_rows(root_dir, files, name=name, id_field=id_field)


def read_request_evidence(root_dir: Path, *, required: bool = True) -> List[dict]:
    """Every request-evidence record of the root — the attempt root's file and every child store's — each stamped `_source`. None refuses
    when ``required`` (every leg but controls); ``required=False`` returns [] for a root with none (the controls driver writes none, and
    `request_evidence_audit.load_evidence` then decides the controls-events contract by name -- never here, never silently)."""
    return read_owned_jsonl(root_dir, EVIDENCE_FILE, id_field="request_id", required=required)


GSM8K_LEG = "gsm8k"
#: the usage ledger's place in a row's request path, as the audit records it. `declared`: the row's plan children declare
#: `ledger.ledger_path_rel` (today: `seedrep_launch_plan._gsm8k_children`, one per GSM8K child store) — the ledger is REQUIRED and
#: every declared file must exist (missing refuses, never `not_on_path`). `not_on_path`: the plan declares none and
#: none exists — `RoutingProxy` writes no ledger, so the reconciliation runs over request evidence + response evidence + the engine
#: log under the REVISED, QUALIFIED contract (`RECONCILIATION_CONTRACT`). `present_undeclared`: the plan declares none but a
#: `usage_ledger.jsonl` exists (the single-ledger rule reads it) — reconciled ledger-backed as before, recorded as such.
LEDGER_DECLARED = "declared"
LEDGER_NOT_ON_PATH = "not_on_path"
LEDGER_PRESENT_UNDECLARED = "present_undeclared"
LEDGER_STATUSES: Tuple[str, ...] = (LEDGER_DECLARED, LEDGER_NOT_ON_PATH, LEDGER_PRESENT_UNDECLARED)
#: the contract version marker every request-evidence verdict carries: `…/1` = the ledger-backed reconciliation as written;
#: `…/2-D106` = the same gate with `attempt_source="response_evidence"` on a `not_on_path` row — a revised, qualified certification
#: contract, NOT equivalent evidence (the ledger's independent body digest and pre-engine attempt record have no counterpart)
RECONCILIATION_CONTRACT_LEDGER = "seedrep_request_evidence_reconciliation/1"
RECONCILIATION_CONTRACT_LEDGERLESS = "seedrep_request_evidence_reconciliation/2-D106"
#: The controls driver (`served_capability_run.py generate`, the judged over-refusal controls + the MMLU letter readout) wires NO
#: request-evidence client -- a controls attempt root carries neither `request_evidence.jsonl` nor `response_evidence.jsonl` anywhere
#: (verified on a controls-arditi root), so the ledgerless reconciliation (`RECONCILIATION_CONTRACT_LEDGERLESS`) has nothing to run over
#: and the audit refused every controls row by name. Two further REVISED, QUALIFIED contracts, one per evidence shape the controls
#: driver actually writes, never equivalent evidence:
#: `RECONCILIATION_CONTRACT_CONTROLS_EVENTS` (a SAMPLED controls store): every filed draw binds through the store's own per-draw event
#: stream (`controls_events.jsonl` -- the driver's `--controls-events`: cap, sampling_seed AS SENT, finish_reason, text_sha256 of the
#: spooled text, the engine's completion_id) to EXACTLY ONE engine `Received request` line whose seed, max_tokens, sampler and LoRA name
#: are RE-READ from the engine and held to the draw, the store's seed rule and the frozen row; failed attempts are known from the client's
#: own event stream alone; no body digest exists. `RECONCILIATION_CONTRACT_MMLU_COUNT` (the MMLU letter readout): the parity emitter
#: records no request id per item and fires no controls event, so there is NO per-draw binding at all -- identity is held at the COUNT
#: level only (at least one MMLU-shaped engine request -- max_tokens 1, temperature 0.0, seed 0 -- per filed item under this arm's
#: weights, the LoRA name re-read from the engine); the readout is a teacher-forced greedy argmax the sampler cannot move
#: (LADDER_RULES.md). Every verdict says which contract it was issued under.
RECONCILIATION_CONTRACT_CONTROLS_EVENTS = "seedrep_request_evidence_reconciliation/3-D133-controls-events"
RECONCILIATION_CONTRACT_MMLU_COUNT = "seedrep_request_evidence_reconciliation/3b-D133-mmlu-count"
RECONCILIATION_CONTRACTS: Tuple[str, ...] = (RECONCILIATION_CONTRACT_LEDGER, RECONCILIATION_CONTRACT_LEDGERLESS,
                                             RECONCILIATION_CONTRACT_CONTROLS_EVENTS, RECONCILIATION_CONTRACT_MMLU_COUNT)
ATTEMPT_SOURCE_LEDGER = "ledger"
ATTEMPT_SOURCE_RESPONSE_EVIDENCE = "response_evidence"
ATTEMPT_SOURCE_CONTROLS_EVENTS = "controls_events"
ATTEMPT_SOURCE_ENGINE_COUNT = "engine_log_count"
CONTRACT_OF_ATTEMPT_SOURCE: Dict[str, str] = {ATTEMPT_SOURCE_LEDGER: RECONCILIATION_CONTRACT_LEDGER, ATTEMPT_SOURCE_RESPONSE_EVIDENCE: RECONCILIATION_CONTRACT_LEDGERLESS,
                                              ATTEMPT_SOURCE_CONTROLS_EVENTS: RECONCILIATION_CONTRACT_CONTROLS_EVENTS,
                                              ATTEMPT_SOURCE_ENGINE_COUNT: RECONCILIATION_CONTRACT_MMLU_COUNT}
#: the container leg whose driver writes the controls evidence shape (the ONLY leg the controls-events / mmlu-count contracts apply to)
CONTROLS_CONTAINER_LEG = "controls"
MMLU_LEG = "mmlu"                                                        # the fixed-cap controls table leg (the letter readout)
from sisteer.eval.controls_events import CONTROLS_EVENTS_FILENAME  # noqa: E402  -- the driver's stream file name (`controls_events.jsonl`), the writer's own constant
#: the plan-builder fact the leg rule stands on when no plan is bound: only the GSM8K children declare a ledger path
LEDGER_DECLARING_LEGS: Tuple[str, ...] = (GSM8K_LEG,)


def ledger_declarations(root: AttemptRoot, plan: Optional["LaunchPlan"] = None) -> Tuple[List[str], str]:
    """The ledger paths (root-relative) the row's PLAN children declare, and the source of that fact. With a bound plan whose row
    is this attempt's, every child's `ledger.ledger_path_rel` (the topology comes from the plan); without one, the plan
    builders' contract by leg (`LEDGER_DECLARING_LEGS`: `_gsm8k_children` is the only builder that declares a ledger) with the
    declared paths unknown (an empty list under a declaring leg means "one per child store, paths unlisted"). NEVER from which
    files happen to exist."""
    if plan is not None:
        row = plan.rows.get(root.production_standup_id) or plan.rows.get(root.standup_id)
        if row is None:
            refuse(f"{plan.path}: no row {root.production_standup_id!r} — the attempt's topology cannot be read off the bound plan")
        kids = row.get("children")
        if not isinstance(kids, list):
            refuse(f"{plan.path}: row {root.production_standup_id!r} carries no children list — the ledger topology is undeclared")
        paths = [str(k["ledger"]["ledger_path_rel"]) for k in kids
                 if isinstance(k, Mapping) and isinstance(k.get("ledger"), Mapping) and k["ledger"].get("ledger_path_rel")]
        return paths, f"plan {plan.path.name} row {root.production_standup_id}: {len(paths)} of {len(kids)} children declare ledger.ledger_path_rel"
    if root.leg in LEDGER_DECLARING_LEGS:
        return [], f"leg rule (no plan bound): {root.leg} children declare ledger.ledger_path_rel (seedrep_launch_plan._gsm8k_children)"
    return [], f"leg rule (no plan bound): no {root.leg} child declares a ledger path (only {list(LEDGER_DECLARING_LEGS)} do)"


def ledger_status(root: AttemptRoot, plan: Optional["LaunchPlan"] = None) -> Tuple[str, str]:
    """`(status, source)` — one of `LEDGER_STATUSES` and the fact it rests on (:func:`ledger_declarations`). A DECLARED ledger that is
    missing refuses here by name, per declared child path when the plan lists them — a declared ledger never becomes `not_on_path`."""
    paths, source = ledger_declarations(root, plan)
    declared = bool(paths) or (plan is None and root.leg in LEDGER_DECLARING_LEGS)
    if declared:
        missing = [rel for rel in paths if not (root.path / rel).is_file()]
        if missing:
            refuse(f"{root.path}: {len(missing)} declared child ledger(s) missing ({missing[:3]}) — the plan declares a usage ledger per "
                   "child store; a declared ledger that is absent refuses, it does not make the row ledger-less")
        return LEDGER_DECLARED, source
    present = walk_files(root.path, LEDGER_FILE)
    if present:
        return LEDGER_PRESENT_UNDECLARED, source + f"; {len(present)} usage_ledger.jsonl present ({[str(p.relative_to(root.path)) for p in present[:3]]})"
    return LEDGER_NOT_ON_PATH, source


def read_usage_ledger(root_dir: Path, *, leg: Optional[str] = None) -> List[dict]:
    """The root's usage-ledger rows under the leg's OWNERSHIP rule. A GSM8K row starts ONE ledger PER CHILD STORE (`gsm8k_ledger_argv`:
    `<out root>/usage_ledger.jsonl`, created when the writer starts), so on `leg == "gsm8k"` with the declared layout EVERY child
    ledger is loaded (rows stamped `_source`, a request id in two ledgers refuses), an attempt-level `<root>/usage_ledger.jsonl` refuses
    by name (nothing in the app writes one on a GSM8K row — `ATTEMPT_ROOT_LAYOUT` declares the name, no writer exists — so its
    provenance is unknown: neither merged nor ignored), and none at all refuses. Every other leg keeps the single-ledger rule: exactly
    one `usage_ledger.jsonl` anywhere under the root (`find_one`), several refuse as ambiguous."""
    root_dir = Path(root_dir)
    if leg == GSM8K_LEG and standup_dir(root_dir) is not None:
        files = owned_evidence_files(root_dir, LEDGER_FILE)
        at_root = [p for p, owner in files if owner == OWNER_ATTEMPT_ROOT]
        if at_root:
            refuse(f"{at_root[0]}: an attempt-level {LEDGER_FILE!r} on a GSM8K row — the app starts one ledger per child store "
                   f"(`gsm8k_ledger_argv`) and nothing writes one at the root; a ledger of unknown provenance is not read")
        if not files:
            refuse(f"{root_dir}: no child {LEDGER_FILE!r} under {CHILD_STORE_TREE_REL}/ — a GSM8K row starts one per child store")
        return _read_owned_rows(root_dir, files, name=LEDGER_FILE, id_field=C.LEDGER_REQUEST_ID_FIELD)
    p = find_one(root_dir, LEDGER_FILE)
    return _read_owned_rows(root_dir, [(p, OWNER_ATTEMPT_ROOT)], name=LEDGER_FILE, id_field=C.LEDGER_REQUEST_ID_FIELD)  # type: ignore[list-item]


def response_text_shas_by_rid(root_dir: Path) -> Dict[str, str]:
    """`request_id → text_sha256` off `response_evidence.jsonl` (every owner; a rejected attempt records None and is skipped)."""
    out: Dict[str, str] = {}
    for r in read_owned_jsonl(root_dir, RESPONSE_EVIDENCE_FILE, id_field="request_id", required=False):
        if r.get("request_id") and r.get(RESPONSE_TEXT_SHA_FIELD):
            out[str(r["request_id"])] = str(r[RESPONSE_TEXT_SHA_FIELD])
    return out


def response_evidence_rows(root_dir: Path) -> List[dict]:
    """EVERY row of `response_evidence.jsonl` from every owner (absent → []), each stamped `_source` — the response hook's
    ATTEMPT inventory: one row per attempt that returned or died (a failed attempt has `completion_id` None and a non-2xx
    status). Attempt evidence, never completion evidence: a consumer that wants completions reads
    :func:`completed_response_rows` (a retry's failed attempt is an attempt, not an extra completion)."""
    return read_owned_jsonl(root_dir, RESPONSE_EVIDENCE_FILE, id_field="request_id", required=False)


def _is_2xx(status: Any) -> bool:
    return isinstance(status, int) and not isinstance(status, bool) and 200 <= status < 300


def response_row_is_completion(r: Mapping[str, Any]) -> bool:
    """The ONE rule that makes a `response_evidence.jsonl` row a COMPLETION (:func:`completed_response_rows`): a `request_id`, a
    REAL non-empty `completion_id`, and a 2xx status when one is recorded — a failed attempt (503, `completion_id` None) is an
    attempt, never a completion."""
    rid, cid = r.get("request_id"), r.get("completion_id")
    if not rid or cid is None or str(cid) in ("", "None"):
        return False
    if "status" in r and r.get("status") is not None and not _is_2xx(r.get("status")):
        return False
    return True


def completed_response_rows(root_dir: Path) -> List[dict]:
    """The COMPLETION-facing view of `response_evidence.jsonl`: rows with BOTH a `request_id` and a REAL, non-empty
    `completion_id` (the pre-fz2 filter, restored) AND a 2xx status when one is recorded — a failed attempt (status 503,
    `completion_id` None) is never represented as a completion, so a 503-then-retry binds ONCE (the Inspect hooks mint a
    fresh request id per attempt and record both responses). Two rows naming ONE request refuse: the same completion twice
    is a duplicate record, two different completions are contradictory — neither is a binding to guess from."""
    out: List[dict] = []
    seen: Dict[str, str] = {}
    for r in response_evidence_rows(root_dir):
        if not response_row_is_completion(r):
            continue
        rid, cid = r.get("request_id"), r.get("completion_id")
        prior = seen.get(str(rid))
        if prior is not None:
            refuse(f"{root_dir}: request {rid!r} has two completed response records ({RESPONSE_EVIDENCE_FILE}: completion_id {prior!r} "
                   f"and {cid!r}) — " + ("a duplicate completion record" if prior == str(cid) else "two records of one attempt name two "
                                                                                                  "completions") + "; never resolved by guessing")
        seen[str(rid)] = str(cid)
        out.append(r)
    return out


def _uniform(values: Iterable[Any], what: str, where: Path) -> Any:
    vals = {json.dumps(v, sort_keys=True, default=str) for v in values}
    if len(vals) != 1:
        refuse(f"{where}: {what} is not uniform across the store's rows ({len(vals)} values: {sorted(vals)[:3]})")
    return json.loads(next(iter(vals)))


def import_by_path(name: str, rel: str, *, what: str) -> Any:
    """A harness script imported by path as a private module object (`scripts/sdfbase/full22/sampled/*` are scripts,
    not a package) — the ONE importer the tables share. A missing file refuses, naming the rule that cannot be read."""
    import importlib.util
    p = REPO / rel
    if not p.is_file():
        refuse(f"{p}: missing — {what} cannot be imported")
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    # registered BEFORE exec_module: a module whose classes are `@dataclass` resolves its annotations through
    # sys.modules[<module name>] at class creation (Python 3.12 dataclasses._is_type) — unregistered, that lookup
    # is None and the import dies with AttributeError ('NoneType' object has no attribute '__dict__')
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_ESCALATE_STATIC: Any = None


def escalate_static_module() -> Any:
    """`scripts/sdfbase/full22/sampled/escalate_static.py` imported ONCE by path as a private module object: its
    `row_source_cap(row, store_cap=…, where=…)` and `PROVENANCE_KEYS` ARE the static rung writer's copy-forward provenance
    rule (a copied-forward row keeps the decode block of the rung that produced its text — `decode.cap == source_cap` —
    so a rung store is a MIXED-CAP store by design; a base row carries no provenance keys and was drawn at the store's
    cap). Its module-level `refuse` (a `sys.exit(2)`) is rebound to :func:`refuse` so a provenance refusal is the audit's
    refusal, named — never a bare exit."""
    global _ESCALATE_STATIC
    if _ESCALATE_STATIC is None:
        mod = import_by_path("_seedrep_audit_escalate_static", "scripts/sdfbase/full22/sampled/escalate_static.py",
                             what="the static rung writer's provenance rule")
        mod.refuse = refuse
        _ESCALATE_STATIC = mod
    return _ESCALATE_STATIC


def duplicates(keys: Iterable[Any]) -> List[Any]:
    """The keys that occur more than once, sorted — the one duplicate-detection idiom every reader and inventory uses."""
    return sorted({k for k, n in Counter(keys).items() if n > 1})


def declared_vs_stored(declared: Set[Tuple[str, int]], stored: Sequence[Tuple[str, int]], *, what: str) -> List[str]:
    """The completeness rule of every store reader whose writer declares sample IDENTITIES (static — the
    progress records; misalign — meta item_ids × n_samples; tinycap — the outcomes header): the identities the writer
    DECLARES must equal the identities it STORED, each stored exactly once, and the declaration must be non-empty. (BoN,
    the sampled controls and a misalign RERUN declare COUNTS, not identities, and hold their rows to the count in their
    own readers.) Returns the named shortfalls (empty = exact correspondence); every filed sample therefore has exactly
    one draw for the audit to bind."""
    out: List[str] = []
    dup = duplicates(stored)
    if dup:
        out.append(f"{len(dup)} {what}(s) stored twice (first: {dup[:3]})")
    missing = sorted(declared - set(stored))
    if missing:
        out.append(f"{len(missing)} declared {what}(s) without a stored response (first: {missing[:3]})")
    extra = sorted(set(stored) - declared)
    if extra:
        out.append(f"{len(extra)} stored {what}(s) the writer declares nowhere (first: {extra[:3]})")
    if not declared:
        out.append(f"the writer declares no {what} — an empty store")
    return out


def read_static_store(shard_dirs: Sequence[Path], *, family: str, arm: str) -> StoreRead:
    """`<private_run>/<family>/<arm>/shard*/{responses,progress}.jsonl` (static_common.run_static rows:
    goal_id, sample_idx, trial, family, response, decode{seed_base, trial, …}, sampling_seed, finish_reason,
    at_cap [+ request_id, completion_id]; a RUNG store's rows add escalate_static's `regenerated` / `source_cap` /
    `escalated_from_cap`). The sampler / seed identity (`decode` minus `cap`) must be ONE across the rows; the cap is
    PER ROW — the writer's provenance rule (`escalate_static.row_source_cap`) names the rung that produced each text, and
    a copied-forward row keeps that rung's block. The store's own cap is `escalation.json.cap_to` on a rung store, else
    the rows' uniform `max_tokens`."""
    rows: List[dict] = []
    progress: List[dict] = []
    for sd in shard_dirs:
        rows.extend(read_jsonl(sd / "responses.jsonl"))
        pp = sd / "progress.jsonl"
        if not pp.is_file():
            refuse(f"{sd}: responses.jsonl without progress.jsonl — a static store is the pair")
        progress.extend(read_jsonl(pp))
    where = shard_dirs[0].parent
    if not rows:
        refuse(f"{where}: no static rows")
    for r in rows:
        if r.get("family") != family:
            refuse(f"{where}: a row carries family {r.get('family')!r}, not {family!r} — the store is not this leg's")
        if "sample_idx" not in r or "goal_id" not in r:
            refuse(f"{where}: a row lacks goal_id / sample_idx — not a static_common row")
    esc = escalate_static_module()
    esc_files = [sd / "escalation.json" for sd in shard_dirs if (sd / "escalation.json").is_file()]
    if esc_files:
        store_cap = int(_uniform((read_json(p).get("cap_to") for p in esc_files), "escalation.json cap_to", where))
    else:
        caps = {int(r["max_tokens"]) for r in rows if r.get("max_tokens") is not None}
        if len(caps) != 1:
            refuse(f"{where}: a base static store whose rows' max_tokens is not one value ({sorted(caps)[:3]}) — its cap "
                   "cannot be established (a rung store carries escalation.json)")
        store_cap = next(iter(caps))
    identities: List[Dict[str, Any]] = []
    source_caps: List[int] = []
    for r in rows:
        dec = r.get("decode")
        if not isinstance(dec, dict) or dec.get("seed_base") is None or dec.get("trial") is None:
            refuse(f"{where}: a row's decode block records no seed_base / trial ({dec!r}) — a greedy or pre-sampled store")
        # the writer's rule: the rung that produced THIS row's text, verified against the row's own decode.cap
        source_caps.append(int(esc.row_source_cap(r, store_cap=store_cap, where=str(where))))
        identities.append({k: v for k, v in dec.items() if k != "cap"})
    ident = _uniform(identities, "sampler / seed identity (the decode block minus its cap)", where)
    seed_base, trial = int(ident["seed_base"]), int(ident["trial"])
    task = static_task(family)
    draws = []
    for r, src_cap in zip(rows, source_caps):
        rid, cid = _binding(r)
        text = r.get("response")
        regen = r.get("regenerated")
        draws.append(Draw(task=task, item_id=str(r["goal_id"]), sample_idx=int(r["sample_idx"]), request_id=rid,
                          completion_id=cid, text=text if isinstance(text, str) else None,
                          text_sha256=text_sha256(text) if isinstance(text, str) else None,
                          finish_reason=r.get("finish_reason"), at_cap=r.get("at_cap"), source_cap=src_cap,
                          regenerated=regen if isinstance(regen, bool) else None))
    # completeness = the WRITER's own rule (`assert_static_store_complete`: no error outcome, rows == Σ n_variants, > 0) per
    # shard, plus exact correspondence between the (goal_id, sample_idx) the progress records DECLARE and the rows stored
    incomplete: List[str] = []
    for sd in shard_dirs:
        try:
            assert_static_store_complete({}, sd, family=family)
        except StaticStoreIncompleteError as e:
            incomplete.append(f"run_static's completeness rule: {e}")
    declared = {(str(p["goal_id"]), i) for p in progress for i in range(int(p.get("n_variants") or 0))}
    incomplete += declared_vs_stored(declared, [(str(r["goal_id"]), int(r["sample_idx"])) for r in rows], what="(goal_id, sample_idx)")
    goals = {g for g, _ in declared}
    return StoreRead(leg=family, arm=arm, s=seed_base, trial=trial, seed_base=seed_base, tasks=(task,),
                     filed={(task, g) for g in goals}, draws=draws, vmodel=None,
                     expected_seed=lambda t, i, k, _b=seed_base, _t=trial, _f=family: static_sampling_seed(_b, _t, _f, i, k),
                     seed_rule_note=f"static_common.static_sampling_seed(seed_base={seed_base}, trial={trial}, family={family!r}, goal_id, sample_idx)",
                     store_dir=where, complete=not incomplete,
                     completeness_note=("every declared (goal_id, sample_idx) has exactly one stored response and run_static's own rule holds"
                                        if not incomplete else " | ".join(incomplete)),
                     n_rows=len(rows), cap=store_cap,
                     notes=[f"rows by source cap: {dict(sorted(Counter(source_caps).items()))}"])


def read_bon_store(cell_dir: Path, *, arm: str) -> StoreRead:
    """`<private_run>/<arm>/bon_responses.jsonl` + `bon_served_manifest.json` (bon_served rows: goal_id,
    sample_idx, trial, response, finish_reason, decode, sampling_seed, decode_seed_policy, decode_seed_base)."""
    rows = read_jsonl(cell_dir / "bon_responses.jsonl")
    mp = cell_dir / "bon_served_manifest.json"
    manifest = read_json(mp) if mp.is_file() else None
    if not rows:
        refuse(f"{cell_dir}: no BoN rows")
    policy = _uniform((r.get("decode_seed_policy") for r in rows), "decode_seed_policy", cell_dir)
    if policy != "replicate_v1":
        refuse(f"{cell_dir}: decode_seed_policy is {policy!r}, not 'replicate_v1' — the rows were not drawn under the "
               "replicate decode-seed rule (bon_decode_seed_v1)")
    seed_base = int(_uniform((r.get("decode_seed_base") for r in rows), "decode_seed_base", cell_dir))
    trial = int(_uniform((r.get("trial") for r in rows), "trial", cell_dir))
    draws = []
    for r in rows:
        rid, cid = _binding(r)
        text = r.get("response")
        draws.append(Draw(task=BON_TASK, item_id=str(r["goal_id"]), sample_idx=int(r["sample_idx"]), request_id=rid,
                          completion_id=cid, text=text if isinstance(text, str) else None,
                          text_sha256=text_sha256(text) if isinstance(text, str) else None,
                          finish_reason=r.get("finish_reason"),
                          at_cap=(r.get("finish_reason") == "length") if r.get("finish_reason") is not None else None))
    vmodel = manifest.get("virtual_model") if isinstance(manifest, dict) else None
    units = manifest.get("units") if isinstance(manifest, dict) and isinstance(manifest.get("units"), dict) else None
    incomplete: List[str] = []
    # completeness (F-S2): the manifest states the PLANNED workload (units.n_planned = goals x n_max) and the planned
    # slots a durable row covers after the run (units.n_done); a `PromptOverWindow` slot writes no row and stays uncovered,
    # so n_done < n_planned and the store is INCOMPLETE (bon_curves refuses the goal; a resume at a wider window fills it).
    # `n_done == len(rows)` alone is NOT completion — a surviving-row count equals the rows by construction.
    n_planned = units.get("n_planned") if units is not None else None
    n_done = units.get("n_done") if units is not None else None
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in (n_planned, n_done)):
        incomplete.append("bon_served_manifest.json missing or without units.n_planned / units.n_done — the run did not finish")
    else:
        if int(n_done) != int(n_planned):
            incomplete.append(f"{int(n_planned) - int(n_done)} of {n_planned} planned (goal_id, sample_idx) slot(s) uncovered "
                              f"(units.n_done {n_done}; the last run filed {units.get('n_prompt_over_window')} prompt_over_window) "
                              "— the workload is not complete; a resume at a wider window fills the slots")
        if int(n_done) != len(rows):
            incomplete.append(f"{len(rows)} stored BoN row(s) != units.n_done {n_done} the manifest declares")
    dup = duplicates((str(r["goal_id"]), int(r["sample_idx"])) for r in rows)
    if dup:
        incomplete.append(f"{len(dup)} (goal_id, sample_idx) stored twice (first: {dup[:3]})")
    return StoreRead(leg="bon", arm=arm, s=seed_base, trial=trial, seed_base=seed_base, tasks=(BON_TASK,),
                     filed={(BON_TASK, str(r["goal_id"])) for r in rows}, draws=draws,
                     vmodel=str(vmodel) if vmodel else None,
                     expected_seed=lambda t, i, k, _b=seed_base, _t=trial: bon_decode_seed_v1(i, k, _t, _b),
                     seed_rule_note=f"bon.bon_decode_seed_v1(goal_id, sample_idx, trial={trial}, decode_seed_base={seed_base})",
                     store_dir=cell_dir, complete=not incomplete,
                     completeness_note=("bon_served_manifest.json units.n_done == units.n_planned == stored rows, each (goal_id, sample_idx) once"
                                        if not incomplete else " | ".join(incomplete)),
                     n_rows=len(rows))


MULTITURN_MANIFEST_FILE = "served_target.json"


def read_multiturn_store(shard_dirs: Sequence[Path], *, attack: str, arm: str) -> StoreRead:
    """`<private_run>/<attack>/<arm>/shard*/{responses,progress}.jsonl` (multiturn.run rows: goal_id, order,
    response, decode [+ request_id, completion_id]; progress rows: goal_id, outcome, target_decode{sampler keys}).
    The `order` of a reply is the TURN = the sample index of the seed rule.

    THE STORE'S IDENTITY: the REAL runners (`multiturn.run_actorattack`, the PyRIT crescendo adapter)
    persist only the SAMPLER keys under progress `target_decode` -- the seed base, trial, cap and decode MODE are the
    driver's FINAL manifest `decode` block (`served_target.json` beside each shard, `jailbreak_run`'s mt_decode_block),
    so that block is read first and BOUND: the progress rows' sampler keys are held equal to it on every key both name;
    a non-final manifest is INCOMPLETE (the driver did not finish). A store with no manifest `decode` block falls to the
    progress rows' own `seed_base` / `trial` when every row carries them (the pre-manifest writer's shape); a store that
    states its seed base nowhere refuses -- never a guess."""
    rows: List[dict] = []
    progress: List[dict] = []
    manifests: List[dict] = []
    for sd in shard_dirs:
        rows.extend(read_jsonl(sd / "responses.jsonl"))
        pp = sd / "progress.jsonl"
        if not pp.is_file():
            refuse(f"{sd}: responses.jsonl without progress.jsonl — a multi-turn store is the pair")
        progress.extend(read_jsonl(pp))
        mp = sd / MULTITURN_MANIFEST_FILE
        if mp.is_file():
            m = read_json(mp)
            if not isinstance(m, dict):
                refuse(f"{mp}: not an object")
            manifests.append(m)
    where = shard_dirs[0].parent
    if not rows:
        refuse(f"{where}: no multi-turn rows")
    for r in rows:
        if "order" not in r or "goal_id" not in r:
            refuse(f"{where}: a row lacks goal_id / order — not a multiturn.run row")
    tds = [p.get("target_decode") for p in progress]
    if not tds or any(not isinstance(t, dict) for t in tds):
        refuse(f"{where}: progress rows carry no target_decode block — the target decode (seed_base, trial) is unrecorded")
    incomplete: List[str] = []
    decs = [m["decode"] for m in manifests if isinstance(m.get("decode"), dict)]
    store_cap: Optional[int] = None
    if decs:
        dec = _uniform(decs, f"{MULTITURN_MANIFEST_FILE} decode", where)
        if dec.get("seed_base") is None:
            refuse(f"{where}: {MULTITURN_MANIFEST_FILE} decode block carries no seed_base — the seed rule cannot be evaluated (an unseeded draw is not this battery's)")
        seed_base = int(dec["seed_base"])
        trial = int(dec.get("trial", 0) or 0)
        for t in tds:
            dis = {k: (t[k], dec.get(k)) for k in t if k in dec and t[k] != dec[k]}
            if dis:
                refuse(f"{where}: a progress row's target_decode disagrees with the manifest's decode block on {dis}")
        if isinstance(dec.get("cap"), int) and not isinstance(dec.get("cap"), bool):
            store_cap = int(dec["cap"])
        if any(m.get("manifest_stage") != "final" for m in manifests if isinstance(m.get("decode"), dict)):
            incomplete.append(f"{MULTITURN_MANIFEST_FILE} manifest_stage is not 'final' — the driver did not finish")
    else:
        if any(t.get("seed_base") is None for t in tds):
            refuse(f"{where}: no {MULTITURN_MANIFEST_FILE} decode block and the progress rows carry no seed_base — the store does not say which "
                   "target seed base it was drawn under (the real runners persist sampler keys only; the driver's final manifest carries the identity)")
        seed_base = int(_uniform((t.get("seed_base") for t in tds), "target_decode.seed_base", where))
        trial = int(_uniform((t.get("trial", 0) for t in tds), "target_decode.trial", where))
    task = multiturn_task(attack)
    draws = []
    for r in rows:
        rid, cid = _binding(r)
        text = r.get("response")
        draws.append(Draw(task=task, item_id=str(r["goal_id"]), sample_idx=int(r["order"]), request_id=rid,
                          completion_id=cid, text=text if isinstance(text, str) else None,
                          text_sha256=text_sha256(text) if isinstance(text, str) else None,
                          finish_reason=r.get("finish_reason"), at_cap=None))
    # completeness = the writer's per-goal declaration: one terminal progress record per goal, outcome not an error, and
    # its n_target_forwards (one target forward per reply) == the reply rows stored for the goal, each (goal, order) once
    n_declared: Dict[str, int] = {}
    for p in progress:
        g = str(p["goal_id"])
        if str(p.get("outcome", "")).startswith("error"):
            incomplete.append(f"goal {g!r} finalized with outcome {p.get('outcome')!r}")
        n = p.get("n_target_forwards")
        if isinstance(n, bool) or not isinstance(n, int):
            incomplete.append(f"goal {g!r}: the progress record carries no integer n_target_forwards ({n!r})")
            continue
        if g in n_declared:
            incomplete.append(f"goal {g!r} has two terminal progress records")
        n_declared[g] = n
    stored = Counter(str(r["goal_id"]) for r in rows)
    for g, n in sorted(n_declared.items()):
        if stored.get(g, 0) != n:
            incomplete.append(f"goal {g!r}: {stored.get(g, 0)} stored reply row(s) != n_target_forwards {n} the progress record declares")
    extra = sorted(set(stored) - set(n_declared))
    if extra:
        incomplete.append(f"{len(extra)} goal(s) with reply rows but no progress record (first: {extra[:3]})")
    dup = duplicates((str(r["goal_id"]), int(r["order"])) for r in rows)
    if dup:
        incomplete.append(f"{len(dup)} (goal_id, order) stored twice (first: {dup[:3]})")
    if not n_declared:
        incomplete.append("no progress record declares a goal — an empty store")
    goals = set(n_declared)
    return StoreRead(leg=attack, arm=arm, s=seed_base, trial=trial, seed_base=seed_base, tasks=(task,),
                     filed={(task, g) for g in goals}, draws=draws, vmodel=None,
                     expected_seed=lambda t, i, k, _b=seed_base, _t=trial, _a=attack: int(_multiturn_seed(_b, _t, _a, i, k)),
                     seed_rule_note=f"{MULTITURN_SEED_NOTE} (seed_base={seed_base}, trial={trial}, attack={attack!r}, goal_id, turn)",
                     store_dir=where, complete=not incomplete,
                     completeness_note=("every goal's reply rows equal its progress record's n_target_forwards, each (goal_id, order) once"
                                        if not incomplete else " | ".join(incomplete)),
                     n_rows=len(rows), reply_cap=store_cap,          # NEVER `cap=`: that is the static rung semantics of filed_here
                     notes=([f"identity from {MULTITURN_MANIFEST_FILE} decode block (seed_base {seed_base}, trial {trial}, reply cap {store_cap})"] if decs
                            else ["identity from the progress rows' target_decode (no manifest decode block)"]))


def read_misalign_store(cell_dir: Path, *, behavior: str, arm: str) -> StoreRead:
    """`agentic/<arm>/responses.jsonl` + `meta.json` (misalign.runner RESPONSE_FIELDS: family, cell_id,
    virtual_model, item_id, sample_idx, behavior, response, finish_reason [+ request_id, completion_id];
    meta.json identity gains `seed_policy` / `seed_base`)."""
    all_rows = read_jsonl(cell_dir / "responses.jsonl")
    rows = [r for r in all_rows if r.get("behavior") == behavior]
    meta = read_json(cell_dir / "meta.json")
    if not rows:
        refuse(f"{cell_dir}: no responses for behavior {behavior!r}")
    if not isinstance(meta, dict):
        refuse(f"{cell_dir}/meta.json: not a JSON object")
    if meta.get("seed_policy") != "per_response":
        refuse(f"{cell_dir}/meta.json: seed_policy is {meta.get('seed_policy')!r}, not 'per_response' — the rows carry no "
               "per-response seed rule")
    if meta.get("seed_base") is None:
        refuse(f"{cell_dir}/meta.json: seed_base missing")
    seed_base = int(meta["seed_base"])
    if meta.get("sampler") != "vendor":
        refuse(f"{cell_dir}/meta.json: sampler is {meta.get('sampler')!r}, not 'vendor'")
    vmodel = _uniform((r.get("virtual_model") for r in rows), "virtual_model", cell_dir)
    rule, note = misalign_seed_rule()
    task = misalign_task(behavior)
    draws = []
    for r in rows:
        rid, cid = _binding(r)
        text = r.get("response")
        draws.append(Draw(task=task, item_id=str(r["item_id"]), sample_idx=int(r["sample_idx"]), request_id=rid,
                          completion_id=cid, text=text if isinstance(text, str) else None,
                          text_sha256=text_sha256(text) if isinstance(text, str) else None,
                          finish_reason=r.get("finish_reason"),
                          at_cap=(r.get("finish_reason") == "length") if r.get("finish_reason") is not None else None))
    items = {str(r["item_id"]) for r in rows}
    # completeness is STORE-level (both behaviours share one responses.jsonl + meta.json): the runner declares
    # `item_ids` × `n_samples`; a ladder rung (`meta.run.rerun_of`) re-drew exactly the SELECTED pairs and declares
    # their count (`rerun_selection[cell].selected`, minus the pairs a behaviour filter left) — the rows must match it
    n_samples = meta.get("n_samples")
    item_ids = meta.get("item_ids")
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1 or not isinstance(item_ids, list) or not item_ids:
        refuse(f"{cell_dir}/meta.json: n_samples ({n_samples!r}) / item_ids declare no sample set — the store's expected rows are unknown")
    stored_all = [(str(r["item_id"]), int(r["sample_idx"])) for r in all_rows]
    run = meta.get("run") if isinstance(meta.get("run"), dict) else {}
    incomplete: List[str] = []
    if run.get("rerun_of"):
        sel = (run.get("rerun_selection") or {}).get(meta.get("cell_id"))
        if not isinstance(sel, dict) or not isinstance(sel.get("selected"), int):
            refuse(f"{cell_dir}/meta.json: a rerun store (run.rerun_of) without run.rerun_selection[{meta.get('cell_id')!r}].selected "
                   "— the pairs it re-drew are undeclared")
        left = (run.get("rerun_selection_left_by_behavior_filter") or {}).get(meta.get("cell_id"), 0)
        n_expected = int(sel["selected"]) - int(left or 0)
        if len(stored_all) != n_expected:
            incomplete.append(f"{len(stored_all)} stored row(s) != the {n_expected} selected pair(s) the rerun declares "
                              f"(selected {sel['selected']}, left by the behaviour filter {left})")
        dup = duplicates(stored_all)
        if dup:
            incomplete.append(f"{len(dup)} (item_id, sample_idx) stored twice (first: {dup[:3]})")
        known = {str(i) for i in item_ids}
        outside = sorted({k for k in stored_all if k[0] not in known or not 0 <= k[1] < n_samples})
        if outside:
            incomplete.append(f"{len(outside)} stored pair(s) outside item_ids × range(n_samples) (first: {outside[:3]})")
        declared_note = f"rerun: {n_expected} selected pair(s)"
    else:
        declared = {(str(i), k) for i in item_ids for k in range(n_samples)}
        incomplete += declared_vs_stored(declared, stored_all, what="(item_id, sample_idx)")
        declared_note = f"item_ids ({len(item_ids)}) × n_samples ({n_samples})"
    return StoreRead(leg=f"misalign_{behavior}", arm=arm, s=seed_base, trial=None, seed_base=seed_base, tasks=(task,),
                     filed={(task, i) for i in items}, draws=draws, vmodel=str(vmodel) if vmodel else None,
                     expected_seed=lambda t, i, k, _b=seed_base, _beh=behavior, _r=rule: int(_r(_b, _beh, i, k)),
                     seed_rule_note=f"{note} (seed_base={seed_base}, behavior={behavior!r}, item_id, sample_idx)",
                     store_dir=cell_dir, complete=not incomplete,
                     completeness_note=(f"every declared pair stored once ({declared_note})" if not incomplete else " | ".join(incomplete)),
                     n_rows=len(rows), notes=[f"declared: {declared_note}"])


def _tinycap_rows(cell_dir: Path) -> List[dict]:
    rp = cell_dir / "rows.jsonl"
    if not rp.is_file():
        refuse(f"{cell_dir}: no rows.jsonl — not a CellStore cell dir")
    return read_jsonl(rp)


def _cell_manifest(cell_dir: Path) -> Dict[str, Any]:
    m = read_json(cell_dir / "manifest.json")
    if not isinstance(m, dict) or not m.get("cell_id"):
        refuse(f"{cell_dir}/manifest.json: no cell_id")
    return m


def _completed_keys(rows: Sequence[dict]) -> Set[Tuple[str, str, str]]:
    """`CellStore.completed_keys()` over parsed rows — (task, item_id, metric)."""
    out = set()
    for r in rows:
        try:
            out.add((str(r["task"]), str(r["item_id"]), str(r["metric"])))
        except KeyError as e:
            refuse(f"a rows.jsonl row lacks {e} — not a CellStore row")
    return out


#: the TARGET turns' attempt bindings ride every row of a Petri scenario under `extra_json.target_turns`
#: — a list, in wire order, of `{turn (int ≥ 1), request_id (str), completion_id (str | null), status (int | null), task}`
#: identical on every dimension row of the scenario; retries are a second entry at the same turn, rollbacks a later entry at
#: an earlier turn. The audit's draws are the SUCCESSFUL entries (2xx with a completion), one per (task, scenario, turn,
#: request_id), `sample_idx` = the turn.
PETRI_TARGET_TURNS_KEY = "target_turns"
#: exactly the keys `petri_sample_turn_sink` files per attempt — an entry with any other key set is not the writer's
PETRI_TARGET_TURN_ENTRY_KEYS: frozenset = frozenset({"turn", "request_id", "completion_id", "status", "task"})


def _row_extra(r: Mapping[str, Any]) -> Mapping[str, Any]:
    extra: Any = r.get("extra_json")
    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
        except json.JSONDecodeError:
            extra = None
    if not isinstance(extra, Mapping):
        extra = r.get("extra") if isinstance(r.get("extra"), Mapping) else {}
    return extra


def petri_target_turns(rows: Sequence[Mapping[str, Any]], *, where: Path) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """`(task, scenario) → the validated ordered target_turns list` over a Petri store's rows, read FAIL-CLOSED: the lane
    client's `petri_target_turn_draws` skipped what it could not read, so a scenario with no list, a malformed list or a
    boolean turn stayed filed with ZERO draws and the audit read AGREES. Here every scenario row must
    carry a list; every entry must be a mapping with a positive-int `turn` (bools refused), a non-empty str `request_id`, a
    `completion_id` that is a str or None, a `status` that is an int or None (bools refused) and the ROW's task, and EXACTLY the
    sink's five keys (:data:`PETRI_TARGET_TURN_ENTRY_KEYS`); a request id filed twice in one list, DIFFERENT lists across a
    scenario's dimension rows, or no scenario row at all refuse — one scenario, one ordered list."""
    per_scenario: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        if C.is_aggregate_item(r.get("item_id")):
            continue
        task, sid = str(r.get("task")), str(r.get("item_id"))
        turns = _row_extra(r).get(PETRI_TARGET_TURNS_KEY)
        if not isinstance(turns, list):
            refuse(f"{where}: {task} {sid} (metric {r.get('metric')!r}) carries no extra_json.{PETRI_TARGET_TURNS_KEY} list "
                   f"({type(turns).__name__}) — the target turns' bindings were never filed on this scenario; a Petri row without them "
                   "binds to no attempt and is refused, never read as zero draws")
        clean: List[Dict[str, Any]] = []
        seen_rid: Set[str] = set()
        for i, e in enumerate(turns):
            w = f"{where}: {task} {sid} {PETRI_TARGET_TURNS_KEY}[{i}]"
            if not isinstance(e, Mapping):
                refuse(f"{w}: not a mapping ({type(e).__name__})")
            if set(e) != PETRI_TARGET_TURN_ENTRY_KEYS:
                refuse(f"{w}: keys {sorted(map(str, e))} are not the sink's {sorted(PETRI_TARGET_TURN_ENTRY_KEYS)} — not the writer's entry")
            turn, rid, cid, st, et = e.get("turn"), e.get("request_id"), e.get("completion_id"), e.get("status"), e.get("task")
            if isinstance(turn, bool) or not isinstance(turn, int) or turn < 1:
                refuse(f"{w}: turn {turn!r} is not a positive int")
            if not isinstance(rid, str) or not rid:
                refuse(f"{w}: request_id {rid!r} is not a non-empty string")
            if cid is not None and (not isinstance(cid, str) or not cid):
                refuse(f"{w}: completion_id {cid!r} is neither a non-empty string nor null")
            if st is not None and (isinstance(st, bool) or not isinstance(st, int)):
                refuse(f"{w}: status {st!r} is neither an int nor null")
            if et != task:
                refuse(f"{w}: filed under task {et!r} but the row's task is {task!r} — another behaviour's binding")
            if rid in seen_rid:
                refuse(f"{w}: request_id {rid!r} is filed twice in one scenario's list")
            seen_rid.add(rid)
            ok = isinstance(st, int) and 200 <= st < 300
            if ok != bool(cid):
                refuse(f"{w}: status {st!r} with completion_id {cid!r} — a 2xx attempt has a completion and a failed one none; a "
                       "half-filed attempt is malformed")
            clean.append({"turn": turn, "request_id": rid, "completion_id": cid, "status": st, "task": et})
        prior = per_scenario.get((task, sid))
        if prior is not None and prior != clean:
            refuse(f"{where}: {task} {sid}: the scenario's dimension rows carry DIFFERENT {PETRI_TARGET_TURNS_KEY} lists — one scenario, "
                   "one ordered list on every row")
        per_scenario[(task, sid)] = clean
    if not per_scenario:
        refuse(f"{where}: no Petri scenario row — an empty row set files nothing and binds nothing")
    return per_scenario


def petri_target_turn_draws(rows: Sequence[Mapping[str, Any]], *, where: Path) -> List[Draw]:
    """One :class:`Draw` per SUCCESSFUL target attempt (`sample_idx` = the turn) of every scenario — every validated entry
    with a 2xx status and a completion, nothing else (a failed attempt is no draw: the ledger and the reconciliation account
    for it), so the emitted draws ARE the valid successful target attempts. A FILED scenario none of whose attempts succeeded
    is refused: zero draws would leave it filed, reconciled and AGREED — a scenario that never
    got a target turn is not a materialised scenario."""
    out: List[Draw] = []
    for (task, sid), turns in petri_target_turns(rows, where=where).items():
        ok = [e for e in turns if isinstance(e["status"], int) and 200 <= e["status"] < 300 and e["completion_id"]]
        if not ok:
            refuse(f"{where}: {task} {sid}: none of its {len(turns)} target attempt(s) succeeded — a filed scenario with no successful "
                   "target turn is not a materialised scenario (never read as zero draws)")
        out.extend(Draw(task=task, item_id=sid, sample_idx=int(e["turn"]), request_id=e["request_id"], completion_id=e["completion_id"]) for e in ok)
    return out


def read_petri_store(cell_dir: Path, *, arm: str) -> StoreRead:
    """A Petri CellStore cell dir (`cells/<run_key>/rows.jsonl`, task `petri/<behaviour>`, item = scenario id at
    the single epoch): filed = `filed_item_pairs(completed_keys())`; the rows record `seed_base` (on the
    row or its extra_json); the draws are the scenario's SUCCESSFUL target turns from `extra_json.target_turns`
    (:func:`petri_target_turn_draws`, fail-closed — a row without the list refuses), `sample_idx` = the turn."""
    rows = _tinycap_rows(cell_dir)
    manifest = _cell_manifest(cell_dir)
    keys = _completed_keys(rows)
    filed = C.filed_item_pairs(keys)
    if not filed:
        refuse(f"{cell_dir}: no filed Petri items")
    tasks = tuple(sorted({t for t, _ in filed}))
    bad = [t for t in tasks if not t.startswith(PETRI_TASK_PREFIX)]
    if bad:
        refuse(f"{cell_dir}: row task(s) {bad[:3]} are not petri/<behaviour>")
    bases = set()
    for r in rows:
        if C.is_aggregate_item(r.get("item_id")):
            continue
        b = r.get("seed_base")
        if b is None:
            extra = r.get("extra_json")
            if isinstance(extra, str):
                try:
                    extra = json.loads(extra)
                except json.JSONDecodeError:
                    extra = None
            b = (extra or {}).get("seed_base") if isinstance(extra, dict) else None
        if b is not None:
            bases.add(int(b))
    if len(bases) != 1:
        refuse(f"{cell_dir}: the Petri rows record {len(bases)} seed_base values ({sorted(bases)[:3]}) — exactly one is required")
    seed_base = next(iter(bases))
    rule = PETRI_SEED_RULE_OVERRIDE
    note = "PETRI_SEED_RULE_OVERRIDE (injected)"
    if rule is None:
        rule, note = petri_seed_rule()
    draws = petri_target_turn_draws(rows, where=cell_dir)
    results_present = (cell_dir / "results.json").is_file()
    expected = (lambda t, i, k, _b=seed_base, _r=rule: int(_r(_b, t[len(PETRI_TASK_PREFIX):], i, k))) if rule else None
    return StoreRead(leg="petri", arm=arm, s=seed_base, trial=None, seed_base=seed_base, tasks=tasks, filed=filed,
                     draws=draws, vmodel=None, expected_seed=expected,
                     seed_rule_note=f"{note} (seed_base={seed_base}, behaviour, scenario_id, turn)", store_dir=cell_dir,
                     complete=results_present,
                     completeness_note=("results.json present (the ONLY Petri cell-done marker)" if results_present
                                        else "results.json absent — the Petri cell is not done"),
                     n_rows=len(rows), notes=[f"cell_id {manifest['cell_id']}", f"target turns: {len(draws)} successful draw(s) over "
                                                f"{len(filed)} scenario(s)"])


ResponseIndex = Dict[Tuple[str, str, str], List[Tuple[str, str]]]


def response_evidence_index(root_dir: Path) -> ResponseIndex:
    """`response_evidence.jsonl` rows `{request_id, completion_id[, task, sample, text_sha256]}` joined to their REQUEST
    record's (task, sample, vmodel) → the completed attempts per (task, sample, vmodel). The vmodel is the axis that keeps
    the arms served on one root apart (three arms complete `item0/0` each, one per vmodel); a response row whose request has
    no record is unattributable and is left for the identity clause to refuse."""
    resp = completed_response_rows(root_dir)  # completions only: a failed retry attempt is not a candidate binding (fz2 HOLD A)
    if not resp:
        return {}
    by_rid: Dict[str, Tuple[str, str, str]] = {}
    for rec in read_owned_jsonl(root_dir, EVIDENCE_FILE, id_field="request_id", required=False):
        if rec.get("request_id") and rec.get("task") and rec.get("sample") and rec.get("vmodel"):
            by_rid[str(rec["request_id"])] = (str(rec["task"]), str(rec["sample"]), str(rec["vmodel"]))
    out: ResponseIndex = {}
    for r in resp:
        key = by_rid.get(str(r["request_id"]))
        if key is None:
            continue
        out.setdefault(key, []).append((str(r["request_id"]), str(r["completion_id"])))
    return out


def vmodel_of_arm(root_dir: Path, arm: str) -> str:
    """The virtual model the attempt root's standup served `arm` as — the arm axis of the response-evidence index — resolved through the
    standup's records in either spelling (:func:`standup_vmodel_of`, F-AN; the STANDUP's record, F-AE: never a per-cell copy; the
    durable copy after teardown); an arm the standup served under neither spelling refuses."""
    return standup_vmodel_of(root_dir, arm).vmodel


def _bind_from_response_evidence(draw: Draw, index: ResponseIndex, where: Path, *, vmodel: str) -> Draw:
    """An Inspect-family draw (its row carries no binding) takes the ONE completed attempt `response_evidence.jsonl`
    recorded for its (task, sample, vmodel) on THIS root; none leaves it for the audit (named there), several refuse
    (ambiguous binding is never resolved by guessing)."""
    if draw.request_id and draw.completion_id:
        return draw
    key = (draw.task, C.sample_header_value(draw.item_id, draw.sample_idx), vmodel)
    cands = index.get(key, [])
    if len(cands) == 1:
        draw.request_id, draw.completion_id = cands[0]
        return draw
    if not cands:
        return draw  # the audit refuses it by name (no binding)
    refuse(f"{where}: {len(cands)} completed attempts recorded for {key} in {RESPONSE_EVIDENCE_FILE} and the filed row "
           "carries no request_id / completion_id — the binding is ambiguous")


def _tinycap_filed_vs_declared(rows: Sequence[dict], *, task: str, recs: Sequence[Mapping[str, Any]], keys: Sequence[Tuple[str, int]], k_per: int,
                               done_p: Optional[Path], out: Mapping[str, Any], notes: List[str]) -> List[str]:
    """F-AU: the completeness rule of an Inspect-family (tinycap) store whose writer declares EVERY attempted draw in
    ``outcomes/<task>.json`` (`inspect_served.outcome_record`: one record per (item, sample) FINAL attempt, ``scored`` / ``at_cap`` /
    ``unparseable`` typed) and files rows for them: an ``acc`` row per scored draw (``acc@<i>`` at k > 1, plus the per-item ``acc`` mean
    when any sample scored) and -- since F-AU -- a typed ``unscored`` row per unscored draw (``unscored@<i>`` at k > 1, the bare row when
    every sample was unscored). The record is the declaration; the rows are held to it:

    * every SCORED outcome has its filed acc row (a missing one is a score the writer dropped -- a genuine gap).
    * every filed acc row is a SCORED outcome (an acc row the record does not declare, or declares unscored, is a torn store).
    * every filed ``unscored`` row is an UNSCORED outcome; an unscored outcome WITHOUT a row is the pre-F-AU writer's documented shape
      (§PW-nanfix-1), typed here off the record (`contracts.outcome_type`) and recorded in the notes, never refused.
    * no (item, sample) is stored twice, on either side.
    * the outcome keys are EXACTLY the declared items x ``range(samples_per_item)`` -- a missing OUTCOME at any sample
      index refuses by (item, sample), whatever the header / marker say about themselves; only a missing ROW behind an unscored outcome
      is exempt.
    * the DONE marker's ``n_items`` is the record's ``n_attempted_items`` (the writer's own identity: `finish_task(n_items=n)` where `n`
      is the attempted item count), its ``n_rows`` the number of distinct filed non-aggregate row keys, and its ``n_planned`` (a post-F-AU
      marker) equal to ``n_items`` (a done marker is written only on the full planned set); a pre-F-AU marker without ``n_planned`` is
      recorded as such.
    Returns the named shortfalls (empty = complete)."""
    problems: List[str] = []
    dup = duplicates(list(keys))
    if dup:
        problems.append(f"{len(dup)} (item_id, sample_idx) outcome record(s) stored twice (first: {dup[:3]})")
    declared_scored = {(str(r["item_id"]), int(r.get("sample_idx", 0))) for r in recs if r.get("scored") is True}
    declared_unscored = {(str(r["item_id"]), int(r.get("sample_idx", 0))) for r in recs if r.get("scored") is not True}
    types = Counter(C.outcome_type(scored=bool(r.get("scored") is True), at_cap=r.get("at_cap"), unparseable=r.get("unparseable")) for r in recs)
    scored_filed: List[Tuple[str, int]] = []
    unscored_filed: List[Tuple[str, int]] = []
    mean_rows: Set[str] = set()
    unscored_item_rows: Set[str] = set()
    unknown: Counter = Counter()
    row_keys: Set[Tuple[str, str, str]] = set()
    for r in rows:
        if r.get("task") != task or C.is_aggregate_item(r.get("item_id")):
            continue
        item, metric = str(r["item_id"]), str(r["metric"])
        row_keys.add((task, item, metric))
        base, idx = C.split_sample_metric(metric)
        if base == C.METRIC_ACC:
            if idx is not None:
                scored_filed.append((item, idx))
            elif k_per == 1:
                scored_filed.append((item, 0))
            else:
                mean_rows.add(item)
        elif base == C.METRIC_UNSCORED:
            if idx is not None:
                unscored_filed.append((item, idx))
            elif k_per == 1:
                unscored_filed.append((item, 0))
            else:
                unscored_item_rows.add(item)
        else:
            unknown[metric] += 1
    if unknown:
        problems.append(f"{sum(unknown.values())} filed row(s) under metric(s) this reader does not know for {task!r}: {dict(unknown)}")
    for what, filed_l in (("acc", scored_filed), (C.METRIC_UNSCORED, unscored_filed)):
        d = duplicates(filed_l)
        if d:
            problems.append(f"{len(d)} filed {what} row key(s) stored twice (first: {d[:3]})")
    scored_set, unscored_set = set(scored_filed), set(unscored_filed)
    # The SAMPLE AXIS is held whole, independently of scored / unscored filing -- the outcome keys must be EXACTLY
    # declared_item_ids x range(samples_per_item), the declared items being every item the record or the rows name (a store whose
    # writer attempted only sample 0 of a k = 2 plan reports one attempted sample in its own header, files one row, and its marker
    # agrees with itself: nothing above notices the missing sample). A missing OUTCOME is a hole, named per (item, sample); the
    # historical exemption below covers missing ROWS backed by unscored outcomes ONLY, never a missing outcome.
    declared_items = {i for i, _ in keys} | {i for i, _ in scored_filed} | {i for i, _ in unscored_filed} | mean_rows | unscored_item_rows
    want_axis = {(i, s_) for i in declared_items for s_ in range(k_per)}
    holes = sorted(want_axis - set(keys))
    if holes:
        problems.append(f"{len(holes)} declared (item_id, sample_idx) outcome record(s) absent from the outcomes record -- the sample axis "
                        f"({len(declared_items)} item(s) x samples_per_item {k_per}) is not whole (first: {holes[:3]})")
    missing = sorted(declared_scored - scored_set)
    if missing:
        problems.append(f"{len(missing)} scored outcome record(s) without a filed acc row -- a score the writer dropped (first: {missing[:3]})")
    extra = sorted(scored_set - declared_scored)
    if extra:
        problems.append(f"{len(extra)} filed acc row(s) the outcomes record does not declare scored (first: {extra[:3]})")
    stray = sorted(unscored_set - declared_unscored)
    if stray:
        problems.append(f"{len(stray)} filed {C.METRIC_UNSCORED} row(s) the outcomes record does not declare unscored (first: {stray[:3]})")
    unfiled = sorted(declared_unscored - unscored_set)
    if k_per > 1:
        items_scored = {i for i, _ in declared_scored}
        no_mean = sorted(items_scored - mean_rows)
        if no_mean:
            problems.append(f"{len(no_mean)} item(s) with a scored sample but no per-item acc mean row (first: {no_mean[:3]})")
        stray_mean = sorted(mean_rows - items_scored)
        if stray_mean:
            problems.append(f"{len(stray_mean)} per-item acc mean row(s) for item(s) with no scored sample (first: {stray_mean[:3]})")
        all_unscored = {i for i, _ in declared_unscored} - items_scored
        stray_u = sorted(unscored_item_rows - all_unscored)
        if stray_u:
            problems.append(f"{len(stray_u)} per-item {C.METRIC_UNSCORED} row(s) for item(s) with a scored sample (first: {stray_u[:3]})")
    if not recs:
        problems.append("the writer declares no (item_id, sample_idx) outcome record — an empty store")
    n_items_rec = len({i for i, _ in keys})
    if done_p is not None and done_p.is_file():
        dm = read_json(done_p)
        if dm.get("n_items") != out["n_attempted_items"] or dm.get("n_items") != n_items_rec:
            problems.append(f"done marker n_items={dm.get('n_items')!r} != the outcomes record's n_attempted_items={out['n_attempted_items']!r} "
                            f"({n_items_rec} distinct item(s) recorded)")
        if dm.get("n_rows") != len(row_keys):
            problems.append(f"done marker n_rows={dm.get('n_rows')!r} != {len(row_keys)} filed row key(s) under {task!r}")
        n_planned = dm.get(C.DONE_MARKER_N_PLANNED_FIELD)
        if C.DONE_MARKER_N_PLANNED_FIELD in dm and n_planned != dm.get("n_items"):
            problems.append(f"done marker {C.DONE_MARKER_N_PLANNED_FIELD}={n_planned!r} != n_items={dm.get('n_items')!r} -- a done marker is written only on the full planned set")
        notes.append(f"done marker: n_items {dm.get('n_items')!r}, n_rows {dm.get('n_rows')!r}, {C.DONE_MARKER_N_PLANNED_FIELD} "
                     f"{n_planned!r}" + ("" if C.DONE_MARKER_N_PLANNED_FIELD in dm else " (absent: a pre-F-AU marker)"))
    notes.append(f"outcomes: {len(recs)} attempted draw(s) over {n_items_rec} item(s) -- {dict(sorted(types.items()))}; "
                 f"{len(scored_set)} scored row(s) filed, {len(unscored_set)} typed unscored row(s) filed, {len(unfiled)} unscored outcome(s) "
                 f"without a row" + (" (the pre-F-AU writer's shape: §PW-nanfix-1 files no acc row for an unscored draw)" if unfiled else ""))
    return problems


def read_tinycap_store(cell_dir: Path, *, task: str, leg: str, arm: str, root_dir: Optional[Path] = None) -> StoreRead:
    """A tinycap CellStore cell dir for ONE row task (`inspect/gsm8k_paired200`, `s2parity/mmlu_letter`):
    filed = `filed_item_pairs(completed_keys())` under `task`; draws from `outcomes/<task>.json` items (GSM8K;
    per (item, sample), `request_id` / `completion_id` additive) or the rows (MMLU); GSM8K's seed base from
    `manifest.seeds.inspect.replicate.seed_base`; MMLU's seed is frozen 0 (`instrument_frozen`)."""
    rows = _tinycap_rows(cell_dir)
    manifest = _cell_manifest(cell_dir)
    keys = _completed_keys(rows)
    filed = {(t, i) for t, i in C.filed_item_pairs(keys) if t == task}
    if not filed:
        refuse(f"{cell_dir}: no filed items under task {task!r} (tasks present: {sorted({t for t, _, _ in keys})[:5]})")
    from sisteer.capeval.tinycap import completion as _completion
    is_tinycap_task = task in C.TINYCAP_TASKS
    outcomes_p = cell_dir / C.TASK_OUTCOMES_DIR / _completion.marker_filename(task) if is_tinycap_task else None
    done_p = cell_dir / C.DONE_DIR / _completion.marker_filename(task) if is_tinycap_task else None
    draws: List[Draw] = []
    incomplete: List[str] = []
    notes_c: List[str] = []
    filed_ids = {i for _, i in filed}
    if outcomes_p is not None and outcomes_p.is_file():
        # the outcomes record is the writer's ALL-ATTEMPTED declaration: its OWN header counts (`TASK_OUTCOMES_OWN_FIELDS`)
        # must agree with its item records, and the records must be exactly the filed items × samples_per_item. Read raw,
        # not through `completion.read_task_outcomes`: the additive request_id / completion_id on the items are
        # outside the contract's exact item shapes and would refuse there
        out = read_json(outcomes_p)
        for k in (*C.TASK_OUTCOMES_OWN_FIELDS, "samples_per_item"):
            if k not in out:
                refuse(f"{outcomes_p}: header lacks {k!r} — not a task outcomes record")
        recs = out["items"] if isinstance(out["items"], list) else refuse(f"{outcomes_p}: items is not a list")
        k_per = out["samples_per_item"]
        if isinstance(k_per, bool) or not isinstance(k_per, int) or k_per < 1:
            refuse(f"{outcomes_p}: samples_per_item {k_per!r} is not a positive int")
        keys: List[Tuple[str, int]] = []
        for it in recs:
            rid, cid = _binding(it)
            key = (str(it["item_id"]), int(it.get("sample_idx", 0)))
            keys.append(key)
            draws.append(Draw(task=task, item_id=key[0], sample_idx=key[1], request_id=rid, completion_id=cid,
                              finish_reason=it.get("stop_reason"), at_cap=it.get("at_cap")))
        if len(recs) != out["n_attempted_samples"]:
            incomplete.append(f"outcomes record lists {len(recs)} item record(s) but its header declares n_attempted_samples="
                              f"{out['n_attempted_samples']!r}")
        if len({i for i, _ in keys}) != out["n_attempted_items"]:
            incomplete.append(f"outcomes record names {len({i for i, _ in keys})} distinct item(s) but its header declares "
                              f"n_attempted_items={out['n_attempted_items']!r}")
        bad_idx = sorted({k for k in keys if not 0 <= k[1] < k_per})
        if bad_idx:
            incomplete.append(f"{len(bad_idx)} outcome record(s) with sample_idx outside [0, {k_per}) (first: {bad_idx[:3]})")
        # qwen-seedrep-native-battery F-AU: the outcomes record IS the writer's ALL-ATTEMPTED declaration (its docstring
        # above), so the filed ROWS are held to IT -- never the other way round. Before F-AU this reader took the `acc` rows as the
        # declaration and read every UNSCORED draw (which §PW-nanfix-1 files no acc row for, by design) as "a stored outcome record the
        # writer declares nowhere", and the done marker's `n_items` (the items ATTEMPTED) against the scored-row count: the gsm8k `S3`
        # smoke (200 attempted, 4 at-cap unscored, 196 acc rows) refused on both. A GENUINE gap is: a scored outcome without its row (a
        # score the writer dropped), a row the record does not declare (or declares unscored), a typed unscored row the record calls
        # scored, a duplicate, or a marker whose counts are not the record's / the filed rows'. An unscored outcome WITHOUT a row is the
        # pre-F-AU writer's documented shape (typed here off the record's own fields, `contracts.outcome_type`) and is not a gap.
        incomplete += _tinycap_filed_vs_declared(rows, task=task, recs=recs, keys=keys, k_per=k_per, done_p=done_p, out=out, notes=notes_c)
    else:
        if is_tinycap_task:
            incomplete.append(f"no {C.TASK_OUTCOMES_DIR}/{outcomes_p.name if outcomes_p else task} record — the attempted draws are undeclared")  # type: ignore[union-attr]
        keys = []
        for r in rows:
            if r.get("task") != task or C.is_aggregate_item(r.get("item_id")):
                continue
            rid, cid = _binding(r)
            idx = int(r.get("sample_idx", 0) or 0)
            keys.append((str(r["item_id"]), idx))
            draws.append(Draw(task=task, item_id=keys[-1][0], sample_idx=idx, request_id=rid, completion_id=cid))
        incomplete += declared_vs_stored({(i, 0) for i in filed_ids}, keys, what="(item_id, sample_idx) row")
    if root_dir is not None:
        idx_map, vmodel = response_evidence_index(root_dir), vmodel_of_arm(root_dir, arm)
        draws = [_bind_from_response_evidence(d, idx_map, cell_dir, vmodel=vmodel) for d in draws]
    if leg == "gsm8k":
        rep = ((manifest.get("seeds") or {}).get("inspect") or {}).get("replicate") if isinstance(manifest.get("seeds"), dict) else None
        if not isinstance(rep, dict) or rep.get("policy") != C.INSPECT_SEED_POLICY_REPLICATE or rep.get("seed_base") is None:
            refuse(f"{cell_dir}/manifest.json: seeds.inspect.replicate is {rep!r} — the GSM8K store did not run under the "
                   "replicate seed policy with a recorded base")
        seed_base = int(rep["seed_base"])
        expected = lambda t, i, k, _b=seed_base: int(C.inspect_replicate_seed(i, k, _b))  # noqa: E731
        note = f"contracts.inspect_replicate_seed(item_id, sample_idx, seed_base={seed_base})"
        s: Optional[int] = seed_base
    elif leg == "mmlu":
        seed_base = None
        expected = lambda t, i, k: 0  # noqa: E731
        note = "MMLU: seed frozen 0 (instrument_frozen, sampler-moot)"
        s = None
    else:
        refuse(f"read_tinycap_store: leg {leg!r} is not a tinycap leg")
    if is_tinycap_task:
        if not (done_p and done_p.is_file()):
            incomplete.append(f"done marker {done_p.name if done_p else task} absent")
    elif not any(r.get("task") == task and C.is_aggregate_item(r.get("item_id")) for r in rows):
        # `s2parity/mmlu_letter` is the parity emitter's task, not a tinycap task: its ONLY completion artefact is the
        # `__aggregate__` row it files after the last item (no done marker, no outcomes record)
        incomplete.append(f"{task} aggregate row absent — the parity emitter did not finish")
    complete = not incomplete
    note_c = (("done marker + outcomes record present, header counts == records; every scored outcome has its filed row, every filed row its "
               "outcome; marker n_items == the record's attempted items, n_rows == the filed row keys" if is_tinycap_task
               else f"{task} aggregate row present, every filed item stored once") if complete else " | ".join(incomplete))
    return StoreRead(leg=leg, arm=arm, s=s, trial=0, seed_base=seed_base, tasks=(task,), filed=filed, draws=draws,
                     vmodel=None, expected_seed=expected, seed_rule_note=note, store_dir=cell_dir, complete=complete,
                     completeness_note=note_c, n_rows=len(rows), notes=[f"cell_id {manifest['cell_id']}", *notes_c],
                     binds_by_response_evidence=True)


def read_controls_store(cell_dir: Path, *, control: str, arm: str, root_dir: Optional[Path] = None, is_rung: bool = False) -> StoreRead:
    """A sampled-controls cell dir: the `judged.GenSpool` `private/gens_*.jsonl` whose header `task` is the
    control's production task (`judged/xstest_v2` / `judged/or_bench_hard_1k`) + its `GenMeta` sidecar
    (`META_ROW_KEYS` rows — ids and numbers only, never request ids: `GenMeta.append` refuses any other key). Filed =
    (header task, item_id) per spooled item; seed_base / trial and the store's CAP from the header's `gen_cfg`; the seed
    rule `sampled_controls.sampling_seed`. A rung store is MIXED-CAP by the writer's design (the static reader's
    sibling): a copied-forward row keeps `regenerated: false` and the `source_cap` of the rung that produced its
    text, a regenerated row carries this store's cap — the draws carry that provenance, and a copied draw's attempt is
    bound on the ANCESTOR root's response evidence by the audit (never on this root's, which never saw it). The Inspect
    family records `(request_id, completion_id)` in `response_evidence.jsonl` per (task, sample); a draw drawn HERE takes
    its own root's; a copied draw is left for :func:`request_evidence_audit.bind_rows` to resolve on its ancestor."""
    task = CONTROLS_TASK_OF[control]
    priv = cell_dir / "private"
    if not priv.is_dir():
        refuse(f"{cell_dir}: no private/ spool dir")
    spool: Optional[Path] = None
    header: Optional[dict] = None
    for p in sorted(priv.glob("gens_*.jsonl")):
        if p.name.endswith(".meta.jsonl"):
            continue
        lines = read_jsonl(p)
        if not lines:
            continue
        h = lines[0].get("__spool_header__") if isinstance(lines[0], dict) else None
        if isinstance(h, dict) and h.get("task") == task:
            if spool is not None:
                refuse(f"{cell_dir}: two spools carry the task {task!r} ({spool.name}, {p.name})")
            spool, header = p, h
    if spool is None or header is None:
        refuse(f"{cell_dir}: no spool under private/ whose header task is {task!r}")
    entries = read_jsonl(spool)[1:]
    gen = header.get("gen_cfg") if isinstance(header.get("gen_cfg"), dict) else {}
    if gen.get("seed_base") is None or gen.get("trial") is None:
        refuse(f"{spool}: header gen_cfg records no seed_base / trial ({gen!r}) — a greedy or pre-sampled controls store")
    seed_base, trial = int(gen["seed_base"]), int(gen["trial"])
    store_cap = gen.get("max_new_tokens")
    if isinstance(store_cap, bool) or not isinstance(store_cap, int) or store_cap < 1:
        refuse(f"{spool}: header gen_cfg.max_new_tokens {store_cap!r} is not a positive int — the store's cap is unreadable")
    from sisteer.eval.sampled_controls import GenMeta
    meta = GenMeta(spool, repair_torn=False)  # a SOURCE store is read, never repaired
    if not meta.exists():
        refuse(f"{spool}: no GenMeta sidecar {meta.path.name} — a sampled store carries one")
    texts: Dict[str, str] = {}
    for e in entries:
        iid = str(e["item_id"])
        if iid in texts:
            refuse(f"{spool}: item {iid!r} is spooled twice — one draw per item")
        texts[iid] = e["text"]
    try:
        # the writer's OWN binding rule: per spooled item the last meta row whose text_sha256 matches THAT item's text
        # (item identity + hash — a hash-only index hands identical responses another item's metadata)
        bound, _n_superseded = meta.bound(texts)
    except RuntimeError as e:
        refuse(str(e))
    draws = []
    source_caps: List[int] = []
    for iid, text in texts.items():
        m = bound[iid]
        src_cap, regen, mt = m.get("source_cap"), m.get("regenerated"), m.get("max_tokens")
        if isinstance(src_cap, bool) or not isinstance(src_cap, int) or not isinstance(regen, bool):
            refuse(f"{meta.path}: item {iid!r}: source_cap {src_cap!r} / regenerated {regen!r} are not the writer's (int, bool) — "
                   "the row's provenance cannot be read")
        # the writer's provenance rule (sampled_controls: `max_tokens` = the cap of the REQUEST that produced the text, `source_cap`
        # the rung it was drawn at — one number written twice; a regenerated or fresh row carries THIS store's cap, a copied-forward
        # row the lower rung's). A relabelled `source_cap` with the request's `max_tokens` left intact used to read as
        # drawn here and bind an unrelated completed attempt on this root — the two must agree, and a rung's un-regenerated rows
        # must come from a LOWER cap while a base store's rows are all its own cap's
        if isinstance(mt, bool) or not isinstance(mt, int) or mt < 1 or mt != src_cap:
            refuse(f"{meta.path}: item {iid!r}: max_tokens {mt!r} is not the row's source_cap {src_cap} — the request's cap and the rung "
                   "that produced the text disagree (a relabelled provenance)")
        if regen and src_cap != store_cap:
            refuse(f"{meta.path}: item {iid!r} is marked regenerated but records source_cap {src_cap}, not this store's cap {store_cap}")
        if not regen and src_cap > store_cap:
            refuse(f"{meta.path}: item {iid!r} records source_cap {src_cap} above this store's cap {store_cap} — no rung below produced it")
        if is_rung and not regen and src_cap >= store_cap:
            refuse(f"{meta.path}: item {iid!r} is an un-regenerated row on a RUNG store recording source_cap {src_cap}, not a rung below "
                   f"{store_cap} — a rung's un-regenerated rows are copied forward from a lower cap")
        if not is_rung and (regen or src_cap != store_cap):
            refuse(f"{meta.path}: item {iid!r} on a BASE store records regenerated {regen} / source_cap {src_cap} — a base store's rows are "
                   f"fresh draws at its own cap {store_cap}")
        source_caps.append(src_cap)
        draws.append(Draw(task=task, item_id=iid, sample_idx=trial, request_id=None, completion_id=None, text=text,
                          text_sha256=text_sha256(text), finish_reason=m.get("finish_reason"), at_cap=m.get("at_cap"),
                          source_cap=src_cap, regenerated=regen))
    items: Set[str] = set(texts)
    if root_dir is not None:
        # only the draws made HERE bind on this root's response evidence; a copied-forward draw's attempt lives on its
        # ancestor root and is resolved there by the audit (an own-root lookup would find nothing, or the wrong attempt)
        idx_map, vmodel = response_evidence_index(root_dir), vmodel_of_arm(root_dir, arm)
        draws = [_bind_from_response_evidence(d, idx_map, cell_dir, vmodel=vmodel) if d.source_cap == store_cap else d for d in draws]
    gs_p = cell_dir / "gen_stats.json"
    blk = (read_json(gs_p) or {}).get(control) if gs_p.is_file() else None
    incomplete: List[str] = []
    if isinstance(blk, dict):
        # the writer's `_meta_stats` stamps the cap it drew at; a block without it is not the writer's, and NO default is read
        #; the two declarations of the cap — spool header and gen_stats — must agree
        gcap = blk.get("cap")
        if isinstance(gcap, bool) or not isinstance(gcap, int):
            refuse(f"{gs_p}: the {control} block carries no cap ({gcap!r}) — the writer's _meta_stats stamps it; a block without it is "
                   "not the writer's, and no default is read")
        if gcap != store_cap:
            refuse(f"{gs_p}: {control} cap {gcap} != the spool header's gen_cfg.max_new_tokens {store_cap} — two declarations of one store's cap disagree")
    if not isinstance(blk, dict) or not isinstance(blk.get("n_total"), int):
        incomplete.append(f"gen_stats.json missing or without a {control} block carrying n_total — the control did not finish")
    elif int(blk["n_total"]) != len(items):
        incomplete.append(f"{len(items)} spooled item(s) != gen_stats.json[{control}].n_total {blk['n_total']} the writer declares")
    return StoreRead(leg=control, arm=arm, s=seed_base, trial=trial, seed_base=seed_base, tasks=(task,),
                     filed={(task, i) for i in items}, draws=draws, vmodel=None,
                     expected_seed=lambda t, i, k, _b=seed_base, _t=trial, _c=control: controls_sampling_seed(_b, _t, _c, i),
                     seed_rule_note=f"sampled_controls.sampling_seed(seed_base={seed_base}, trial={trial}, readout={control!r}, item_id)",
                     store_dir=cell_dir, complete=not incomplete,
                     completeness_note=(f"gen_stats.json[{control}].n_total == spooled items, each bound to its own meta row"
                                        if not incomplete else " | ".join(incomplete)),
                     n_rows=len(entries), cap=store_cap, binds_by_response_evidence=True,
                     notes=[f"rows by source cap: {dict(sorted(Counter(source_caps).items()))}"])


# --------------------------------------------------------------------------- #
# Store discovery inside an attempt root                                        #
# --------------------------------------------------------------------------- #

def _shard_dirs(defense_dir: Path) -> List[Path]:
    shards = sorted(p for p in defense_dir.glob("shard*") if p.is_dir() and (p / "responses.jsonl").is_file())
    if not shards and (defense_dir / "responses.jsonl").is_file():
        shards = [defense_dir]
    return shards


MISALIGN_LAYOUT_FLAT = "agentic/<arm>"
MISALIGN_LAYOUT_PER_BEHAVIOUR = "agentic__s<N>__<behaviour>/<arm>"
MISALIGN_PER_BEHAVIOUR_DIR_RE = re.compile(r"agentic__s(?P<s>[1-9][0-9]*)__(?P<behaviour>[A-Za-z0-9_]+)")


def _consumed(store: StoreRead, files: Sequence[Path]) -> None:
    """fc11k: record on the store the exact candidate files its reader consumed (absolute, resolved)."""
    object.__setattr__(store, "consumed_files", sorted({str(Path(f).resolve()) for f in files}))


def discover_stores(root: AttemptRoot, table_leg: str, *, petri_behaviours: Sequence[str] = (), required: bool = True) -> List[StoreRead]:
    """Every store of ``table_leg`` under the attempt root, one :class:`StoreRead` per served arm (and per seed
    namespace on the Arditi root). Refuses when the root holds none (``required=False``: an empty list instead — a root
    that serves other legs; every malformed store still refuses)."""
    if CONTAINER_LEG_OF.get(table_leg) != root.leg:
        refuse(f"{root.label}: table leg {table_leg!r} does not live in container leg {root.leg!r}")
    out: List[StoreRead] = []
    if table_leg in STATIC_FAMILIES or table_leg in MULTITURN_ATTACKS:
        for rp in walk_files(root.path, "responses.jsonl"):
            # <private_run>/<attack>/<defense>/shard*/responses.jsonl — the attack dir is two or three levels up
            parts = rp.parts
            if len(parts) < 4:
                continue
            defense_dir = rp.parent if not rp.parent.name.startswith("shard") else rp.parent.parent
            attack_dir = defense_dir.parent
            if attack_dir.name != table_leg:
                continue
            arm = defense_dir.name
            shards = _shard_dirs(defense_dir)
            if any(s.store_dir == defense_dir for s in out):
                continue
            if table_leg in STATIC_FAMILIES:
                out.append(read_static_store(shards, family=table_leg, arm=arm))
            else:
                out.append(read_multiturn_store(shards, attack=table_leg, arm=arm))
            _consumed(out[-1], [sd / "responses.jsonl" for sd in shards])
    elif table_leg == "bon":
        for rp in walk_files(root.path, "bon_responses.jsonl"):
            out.append(read_bon_store(rp.parent, arm=rp.parent.name))
            _consumed(out[-1], [rp])
    elif table_leg.startswith("misalign_"):
        behavior = table_leg[len("misalign_"):]
        # fc11c: TWO layouts hold a misalign cell store, both `<out>/<arm>/{responses.jsonl, meta.json}` (the runner's documented shape):
        #   MISALIGN_LAYOUT_FLAT           <...>/agentic/<arm>/            -- one out dir for both behaviours, the rows filtered by `behavior`
        #   MISALIGN_LAYOUT_PER_BEHAVIOUR  <...>/agentic__s<N>__<behaviour>/<arm>/  -- the plan's `MISALIGN_OUT_REL` (one child per
        #                                  behaviour: every root the plan's `--out` writes), keyed by the behaviour in the dir name
        # The reader recognised the flat shape only, so a per-behaviour root read `no misalign_leaking store` and the re-audit exited 2 (fc11c). A root
        # holding BOTH shapes for one behaviour is refused as ambiguous, never merged; the matched layout is recorded on the store's notes.
        found: Dict[str, List[Path]] = {MISALIGN_LAYOUT_FLAT: [], MISALIGN_LAYOUT_PER_BEHAVIOUR: []}
        for rp in walk_files(root.path, "responses.jsonl"):
            if not (rp.parent / "meta.json").is_file():
                continue
            out_name = rp.parent.parent.name
            m = MISALIGN_PER_BEHAVIOUR_DIR_RE.fullmatch(out_name)
            if out_name == "agentic":
                layout = MISALIGN_LAYOUT_FLAT
            elif m is not None and m.group("behaviour") == behavior:
                layout = MISALIGN_LAYOUT_PER_BEHAVIOUR
            else:
                continue
            if not any(r.get("behavior") == behavior for r in read_jsonl(rp)):
                continue  # a cell with no rows of this behaviour (a rung root re-draws the other behaviour's pairs only)
            found[layout].append(rp.parent)
        present = [k for k in (MISALIGN_LAYOUT_FLAT, MISALIGN_LAYOUT_PER_BEHAVIOUR) if found[k]]
        if len(present) > 1:
            refuse(f"{root.label}: {table_leg} stores under BOTH layouts {present} ({[str(d.relative_to(root.path)) for k in present for d in found[k]][:4]}) -- "
                   "ambiguous: one root writes one layout; nothing is merged")
        for cell_dir in (found[present[0]] if present else []):
            store = read_misalign_store(cell_dir, behavior=behavior, arm=cell_dir.name)
            store.notes.append(f"store_layout={present[0]} ({cell_dir.parent.name}/{cell_dir.name})")
            _consumed(store, [cell_dir / "responses.jsonl"])
            out.append(store)
    elif table_leg == "petri":
        for rp in walk_files(root.path, "rows.jsonl"):
            cell_dir = rp.parent
            if not (cell_dir / "manifest.json").is_file():
                continue
            rows = read_jsonl(rp)
            if not rows or not any(str(r.get("task", "")).startswith(PETRI_TASK_PREFIX) for r in rows):
                continue
            out.append(read_petri_store(cell_dir, arm=_arm_of_cell(cell_dir)))
            _consumed(out[-1], [rp])
    elif table_leg in CONTROLS or table_leg in ("mmlu", "gsm8k"):
        for rp in walk_files(root.path, "rows.jsonl"):
            cell_dir = rp.parent
            if not (cell_dir / "manifest.json").is_file():
                continue
            arm = _arm_of_cell(cell_dir)
            if table_leg in CONTROLS:
                if not (cell_dir / "private").is_dir():
                    continue
                try:
                    out.append(read_controls_store(cell_dir, control=table_leg, arm=arm, root_dir=root.path, is_rung=root.stage != STAGE_BASE))
                    _consumed(out[-1], [rp])
                except AuditRefusal as e:
                    if "no spool under private/" in str(e):
                        continue
                    raise
            else:
                task = canonical_task(table_leg)
                rows = read_jsonl(rp)
                if not any(r.get("task") == task for r in rows):
                    continue
                out.append(read_tinycap_store(cell_dir, task=task, leg=table_leg, arm=arm, root_dir=root.path))
                _consumed(out[-1], [rp])
    else:
        refuse(f"unknown table leg {table_leg!r}")
    if not out and required:
        refuse(f"{root.label}: no {table_leg} store under {root.path}")
    return out


STORE_CANDIDATE_FILES: Tuple[str, ...] = ("responses.jsonl", "rows.jsonl", "bon_responses.jsonl")


def store_identity(leg: str, s: Any, arm: str, store_rel: str) -> str:
    """fc11i: ONE store's canonical identity -- `<table leg>|s<seed>|<arm>|<store dir, root-relative>` (a seed the store records none of, MMLU, is read
    off its namespace; `s?` when neither states one)."""
    return f"{leg}|s{s if s is not None else '?'}|{arm}|{store_rel}"


def inventory_stores(root_path: Path, legs: Sequence[str], *, petri_behaviours: Sequence[str] = (), root: Optional[AttemptRoot] = None) -> Dict[str, Any]:
    """fc11i: EVERY store a pulled attempt root holds, by IDENTITY -- table leg, seed, arm, store path -- never an arm-name set (the
    controls / Arditi root holds nine stores, three table legs x seeds 1-3, one arm: an arm set per leg collapsed them to three). Per table leg
    `discover_stores` (the audit's own reader) enumerates the stores; a leg whose stores refuse to read is a NAMED problem. Then every CANDIDATE store
    file under the root (`responses.jsonl` / `rows.jsonl` / `bon_responses.jsonl`, the serve dir excluded) that NO leg claimed -- an unknown layout, a
    cell without its metadata, a sibling of another shape -- is a NAMED problem under `unclassified` (never a silent skip), so a scope derived from
    this inventory refuses when the root holds something the reader could not classify. Returns
    ``{"stores": [{leg, s, arm, store_dir, identity}], "problems": {leg: why}, "unclassified": [root-relative paths]}``."""
    root_path = Path(root_path)
    out: Dict[str, Any] = {"stores": [], "problems": {}, "unclassified": []}
    try:
        ar = root or read_attempt_root(root_path)
    except Exception as e:  # noqa: BLE001 -- an unreadable root enumerates nothing; the caller refuses by name
        out["problems"]["(root)"] = f"{type(e).__name__}: {e}"
        return out
    consumed: Set[str] = set()
    for leg in legs:
        try:
            stores = discover_stores(ar, str(leg), petri_behaviours=[str(b) for b in petri_behaviours], required=False)
        except Exception as e:  # noqa: BLE001
            out["problems"][str(leg)] = f"{type(e).__name__}: {e}"
            continue
        for st in stores:
            sd = Path(st.store_dir)
            try:
                rel = sd.relative_to(root_path).as_posix()
            except ValueError:
                rel = sd.as_posix()
            s_ = st.s if st.s is not None else _s_from_namespace(sd)
            files_rel = []
            for f in st.consumed_files:
                consumed.add(f)
                try:
                    files_rel.append(Path(f).relative_to(root_path.resolve()).as_posix())
                except ValueError:
                    files_rel.append(f)
            if not st.consumed_files:
                out["problems"][f"{leg}:{rel}"] = "the reader recorded no consumed candidate file for this store -- its files cannot be told from strays"
            out["stores"].append({"leg": str(leg), "s": s_, "arm": str(st.arm), "store_dir": rel, "identity": store_identity(str(leg), s_, str(st.arm), rel),
                                  "consumed_files": sorted(files_rel)})
    # fc11k: a candidate is classified ONLY when a reader CONSUMED that exact file -- a stray nested beneath a recognized cell dir is as
    # unclassified as one outside it; an ancestor test established nothing
    serve = (root_path / CHILD_STORE_TREE_REL / "serve").resolve()
    for name in STORE_CANDIDATE_FILES:
        for f in walk_files(root_path, name):
            fr = f.resolve()
            if serve in fr.parents:
                continue
            if str(fr) in consumed:
                continue
            out["unclassified"].append(f.relative_to(root_path).as_posix())
    out["unclassified"] = sorted(set(out["unclassified"]))
    return out


def _arm_of_cell(cell_dir: Path) -> str:
    """The served cell id of a CellStore cell dir: `manifest.json.cell_id` (the authoritative spelling; the dir
    name mangles `__` → `--` and hashes the gen block)."""
    m = _cell_manifest(cell_dir)
    return str(m["cell_id"])


@dataclass
class AttemptInventory:
    """The ATTEMPT inventory on ONE attempt root's request path — what the identity clause and every share read run over:
    the usage ledger's rows where the bound plan declares a ledger (`ledger` = `declared`) or one exists undeclared
    (`present_undeclared`), the response hook's rows — EVERY attempt that returned or died, failed ones included — where none
    is on the path (`not_on_path`). The same topology `request_evidence_audit.load_evidence` records on every verdict
    (`ledger` / `attempt_source` / `reconciliation_contract`), so a table row can carry the qualification the verdict carries."""

    ledger: str
    ledger_source: str
    attempt_source: str
    rows: List[dict]

    @property
    def contract(self) -> str:
        return RECONCILIATION_CONTRACT_LEDGERLESS if self.attempt_source == "response_evidence" else RECONCILIATION_CONTRACT_LEDGER

    @property
    def by_id(self) -> Dict[str, dict]:
        return {str(r.get(C.LEDGER_REQUEST_ID_FIELD)): r for r in self.rows if r.get(C.LEDGER_REQUEST_ID_FIELD)}

    @property
    def qualification(self) -> Dict[str, str]:
        """The three keys every request-evidence verdict carries (`request_evidence_audit.SUMMARY_COLUMNS`)."""
        return {"ledger": self.ledger, "attempt_source": self.attempt_source, "reconciliation_contract": self.contract}


def attempt_inventory(root: AttemptRoot, plan: Optional["LaunchPlan"] = None, *, ledger_rows: Optional[List[dict]] = None) -> AttemptInventory:
    """:class:`AttemptInventory` of `root` from :func:`ledger_status` (the BOUND plan's children, else the plan builders' leg rule —
    never file absence): `declared` / `present_undeclared` read the ledger (a declared one missing refuses in :func:`ledger_status` /
    :func:`read_usage_ledger` by name), `not_on_path` reads the response hook's rows and refuses when there are none — no attempt
    inventory means nothing to reconcile a record against, never an empty inventory read as "no attempts". `ledger_rows` = the
    ledger already read for this root (a :class:`RootReads` memo), else it is read here."""
    ledger, source = ledger_status(root, plan)
    if ledger == LEDGER_NOT_ON_PATH:
        rows = list(response_evidence_rows(root.path))
        if not rows:
            refuse(f"{root.path}: the plan declares no usage ledger for this row and {RESPONSE_EVIDENCE_FILE} holds no rows — no attempt "
                   "inventory to reconcile the request records against (D106 needs the response hook's rows)")
        return AttemptInventory(ledger=ledger, ledger_source=source, attempt_source="response_evidence", rows=rows)
    rows = list(ledger_rows) if ledger_rows is not None else read_usage_ledger(root.path, leg=root.leg)
    return AttemptInventory(ledger=ledger, ledger_source=source, attempt_source="ledger", rows=rows)


class RootReads:
    """The per-BUILD memo of the reads every table repeats per arm-run — a root's stores of a leg, its usage ledger, its
    request-evidence records, its attempt inventory. One instance per `build()` / `main()`, never module-level (a test that
    rewrites a store and re-builds in one process must read the new bytes), so ~405 arm-runs × ~130 roots read each file ONCE.
    The readers are pure over the on-disk tree and no caller mutates a :class:`StoreRead` after it is returned; a refusal is
    not memoised."""

    def __init__(self) -> None:
        self._stores: Dict[Tuple[str, str, Tuple[str, ...]], List[StoreRead]] = {}
        self._ledger: Dict[str, List[dict]] = {}
        self._records: Dict[str, List[dict]] = {}
        self._inventory: Dict[Tuple[str, Optional[str]], AttemptInventory] = {}

    def stores(self, root: AttemptRoot, table_leg: str, petri_behaviours: Sequence[str] = ()) -> List[StoreRead]:
        """:func:`discover_stores` with ``required=False`` — an empty list for a root that holds no store of the leg."""
        key = (str(root.path), table_leg, tuple(petri_behaviours))
        if key not in self._stores:
            self._stores[key] = discover_stores(root, table_leg, petri_behaviours=petri_behaviours, required=False)
        return self._stores[key]

    def ledger_rows(self, root: AttemptRoot) -> List[dict]:
        key = str(root.path)
        if key not in self._ledger:
            self._ledger[key] = read_usage_ledger(root.path, leg=root.leg)
        return self._ledger[key]

    def records(self, root: AttemptRoot, *, required: bool = True) -> List[dict]:
        key = f"{root.path}|required={required}"
        if key not in self._records:
            self._records[key] = read_request_evidence(root.path, required=required)
        return self._records[key]

    def attempt_inventory(self, root: AttemptRoot, plan: Optional["LaunchPlan"] = None) -> AttemptInventory:
        """:func:`attempt_inventory` memoised per (root, bound plan); its ledger rows come through :meth:`ledger_rows`."""
        key = (str(root.path), str(plan.path) if plan is not None else None)
        if key not in self._inventory:
            status, _ = ledger_status(root, plan)
            self._inventory[key] = attempt_inventory(root, plan, ledger_rows=(self.ledger_rows(root) if status != LEDGER_NOT_ON_PATH else None))
        return self._inventory[key]


# --------------------------------------------------------------------------- #
# The expected arm-runs file (`coverage_table.py expected`), read by the coverage and at-cap tables                #
# --------------------------------------------------------------------------- #

def read_expected(path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    exp = read_json(path)
    legs = exp.get("legs") if isinstance(exp, dict) else None
    runs = exp.get("arm_runs") if isinstance(exp, dict) else None
    if not isinstance(legs, list) or not legs or not isinstance(runs, list) or not runs:
        refuse(f"{path}: expected file needs non-empty `legs` and `arm_runs` lists")
    bad = [l for l in legs if l not in LEGS]
    if bad:
        refuse(f"{path}: unknown leg(s) {bad}")
    for r in runs:
        if not isinstance(r, dict) or not r.get("arm") or not r.get("view") or r.get("s") is None:
            refuse(f"{path}: arm_run {r!r} needs arm, view, s")
    keys = [(r["arm"], r["view"], int(r["s"])) for r in runs]
    if len(set(keys)) != len(keys):
        refuse(f"{path}: duplicate arm_runs")
    return [str(l) for l in legs], runs


# --------------------------------------------------------------------------- #
# The launch plan (`seedrep_launch_plan.py`, schema seedrep_launch_plan/1): the DECLARED standup / shard inventory  #
# --------------------------------------------------------------------------- #

LAUNCH_PLAN_SCHEMA = "seedrep_launch_plan/1"
#: the keys every plan row (`standups[]` / `rungs[]`) must carry — the audit reads standup_id, control_key, kind, leg, view,
#: stage, shard and arm_runs; eval_legs, nshards and seed_bases are held PRESENT as the schema pin (the generator writes many more)
LAUNCH_PLAN_ROW_KEYS: Tuple[str, ...] = ("standup_id", "control_key", "kind", "leg", "eval_legs", "view", "stage", "shard",
                                         "nshards", "seed_bases", "arm_runs")


@dataclass
class LaunchPlan:
    """`RS/launch_plan.json` as the coverage table reads it: every planned standup (base and rung rows) by `standup_id`,
    with the arm-runs each serves (`arm_runs[].cell` = the served cell id, `.s` = the seed base — the Arditi row labels
    its arm `ar_c0p3__arditi_allpos` while its cell is `arditi_allpos`; the stores are keyed by cell)."""

    path: Path
    W: str
    wave_uuid: str
    rows: Dict[str, Dict[str, Any]]

    @property
    def W7(self) -> str:
        return self.W[:7]

    def rows_at(self, leg: str, view: str, stage: str) -> List[Dict[str, Any]]:
        return sorted((r for r in self.rows.values() if (r["leg"], r["view"], str(r["stage"])) == (leg, view, str(stage))),
                      key=lambda r: r["standup_id"])

    @staticmethod
    def serves(row: Mapping[str, Any], cell: str, s: int) -> bool:
        return any(isinstance(ar, Mapping) and ar.get("cell") == cell and ar.get("s") is not None and int(ar["s"]) == int(s)
                   for ar in row.get("arm_runs") or [])

    def serving(self, leg: str, view: str, stage: str, cell: str, s: int) -> List[Dict[str, Any]]:
        """The planned standups of (leg, view, stage) whose `arm_runs` serve (cell, s) — the inventory one stage of an
        arm-run's ancestry is verified against."""
        return [r for r in self.rows_at(leg, view, stage) if self.serves(r, cell, s)]


def read_launch_plan(path: Path) -> LaunchPlan:
    """The plan file, shape-checked: schema, a 40-hex `W`, `wave_uuid`, `standups` + `rungs` lists whose rows carry
    :data:`LAUNCH_PLAN_ROW_KEYS`, a `standup_id` that spells its own axes, a `control_key` that agrees with them and with
    the plan's `W` / `wave_uuid`, unique ids, and — per (leg, view, stage) — ONE positive integer `nshards` identical on
    every row that EQUALS the rows declared, whose shards are exactly `shard00 .. shard<n-1>` (a plan saying `nshards: 2`
    with one shard row is not an inventory). Anything else refuses — an inventory that cannot be read is no inventory."""
    d = read_json(path)
    if not isinstance(d, dict) or d.get("schema") != LAUNCH_PLAN_SCHEMA:
        refuse(f"{path}: schema {d.get('schema') if isinstance(d, dict) else None!r} != {LAUNCH_PLAN_SCHEMA!r}")
    W, wave = d.get("W"), d.get("wave_uuid")
    if not isinstance(W, str) or len(W) != 40:
        refuse(f"{path}: W {W!r} is not the 40-hex tree sha")
    if not isinstance(wave, str) or len(wave) < 8:
        refuse(f"{path}: wave_uuid {wave!r} is not the uuid minted at launch")
    rows: Dict[str, Dict[str, Any]] = {}
    for section in ("standups", "rungs"):
        lst = d.get(section)
        if not isinstance(lst, list):
            refuse(f"{path}: {section} is not a list")
        for r in lst:
            if not isinstance(r, dict):
                refuse(f"{path}: a {section} row is not an object")
            missing = [k for k in LAUNCH_PLAN_ROW_KEYS if k not in r]
            if missing:
                refuse(f"{path}: {section} row {r.get('standup_id')!r} lacks {missing}")
            sid = str(r["standup_id"])
            leg, view, stage, shard = str(r["leg"]), str(r["view"]), str(r["stage"]), _shard_name(r["shard"])
            if sid != standup_id_of(leg, view, stage, shard):
                refuse(f"{path}: standup_id {sid!r} does not spell its row's axes {standup_id_of(leg, view, stage, shard)!r}")
            if (r["kind"] == "base") != (stage == STAGE_BASE):
                refuse(f"{path}: {sid}: kind {r['kind']!r} does not match stage {stage!r}")
            if leg not in CONTAINER_LEGS or (view not in SEED_VIEWS and view != ARDITI_VIEW):
                refuse(f"{path}: {sid}: leg {leg!r} / view {view!r} unknown")
            ck = parse_control_key(str(r["control_key"]))
            if (ck["W7"], ck["wave_uuid"], ck["leg"], ck["view"], ck["stage"], ck["shard"]) != (W[:7], wave, leg, view, stage, shard):
                refuse(f"{path}: {sid}: control_key {r['control_key']!r} does not spell W7 / wave_uuid / the row's axes")
            if not isinstance(r["arm_runs"], list) or not all(isinstance(a, dict) and a.get("cell") and a.get("s") is not None for a in r["arm_runs"]):
                refuse(f"{path}: {sid}: arm_runs must be a list of {{arm, cell, s}} objects")
            if sid in rows:
                refuse(f"{path}: standup_id {sid!r} declared twice")
            rows[sid] = {**r, "leg": leg, "view": view, "stage": stage, "shard": shard}
    if not rows:
        refuse(f"{path}: the plan declares no standup")
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for r in rows.values():
        groups.setdefault((r["leg"], r["view"], r["stage"]), []).append(r)
    for (leg, view, stage), grp in sorted(groups.items()):
        ns = {r["nshards"] for r in grp}
        if len(ns) != 1:
            refuse(f"{path}: ({leg}, {view}, {stage}): nshards is not one value across its rows ({sorted(map(str, ns))})")
        n = next(iter(ns))
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            refuse(f"{path}: ({leg}, {view}, {stage}): nshards {n!r} is not a positive int")
        if n != len(grp):
            refuse(f"{path}: ({leg}, {view}, {stage}): nshards {n} but {len(grp)} row(s) declared — the shard inventory is not the plan's")
        want = [_shard_name(k) for k in range(n)]
        have = sorted(r["shard"] for r in grp)
        if have != want:
            refuse(f"{path}: ({leg}, {view}, {stage}): shards {have} are not the contiguous inventory {want}")
    return LaunchPlan(path=Path(path), W=W, wave_uuid=wave, rows=rows)


# --------------------------------------------------------------------------- the carry-over record (fj5)
#: `RS/carry_over.json` as `seedrep_carry_over` writes it (schema pinned equal by test): every CARRIED attempt with its provenance and
#: the five clause verdicts. The coverage table reads it to bind a carried attempt's HISTORICAL identity to its CURRENT planned standup.
CARRY_OVER_SCHEMA = "seedrep_carry_over/1"
CARRY_CLAUSES: Tuple[str, ...] = ("row_identity", "producer_set", "complete", "engine_identity", "ladder_queue")
CARRY_PROVENANCE_KEYS: Tuple[str, ...] = ("W", "W7", "wave_uuid", "attempt_id", "tree_sha", "standup_id", "source", "accepted_key")
CARRY_SOURCES: Tuple[str, ...] = ("production", "s5_smoke", "s3_smoke", "current_s5_full")
S5_SMOKE_ID = "S5"
#: fj12 / fj17: the smoke id each smoke source's historical standup id spells (`smoke-<id>__<sid>`); a production source spells none; a
#: `current_s5_full` source is THIS tree's own S5 FULL row -- the one source whose W equals the plan's
SMOKE_ID_OF_CARRY_SOURCE: Dict[str, Optional[str]] = {"production": None, "s5_smoke": "S5", "s3_smoke": "S3", "current_s5_full": "S5"}
CURRENT_TREE_SOURCES: Tuple[str, ...] = ("current_s5_full",)


@dataclass(frozen=True)
class CarryBinding:
    """ONE carried attempt as the coverage table binds it: the attempt id, THIS plan's standup it stands for, and the HISTORICAL
    identity the pulled root must spell exactly (its tree W, its wave -- the smoke wave for an `s5_smoke` source -- and its own
    standup id, `smoke-S5__<sid>` for a smoke-wave FULL row)."""

    attempt_id: str
    standup_id: str
    W: str
    wave_uuid: str
    historical_standup_id: str
    source: str
    attempt_root: str
    accepted_key: str

    def provenance(self) -> Dict[str, Any]:
        return {"W": self.W, "wave_uuid": self.wave_uuid, "standup_id": self.historical_standup_id, "source": self.source,
                "attempt_root": self.attempt_root, "accepted_key": self.accepted_key, "bound_to": self.standup_id}


def carry_mapping(record: Any, plan: LaunchPlan) -> Dict[str, CarryBinding]:
    """The VALIDATED carry mapping `{attempt_id: CarryBinding}` of a carry-over record against THIS plan: the schema, the record's
    W / wave the plan's, every carried entry a plan row (its `accepted_key` the row's when the row carries one), a full provenance
    (:data:`CARRY_PROVENANCE_KEYS`) from a PRIOR tree, all five clauses `ok`, a source in :data:`CARRY_SOURCES`, and a historical
    standup id that spells the plan row's axes (`smoke-S5__<sid>` for an S5 source, `<sid>` for a production one). Anything else
    refuses -- a carry the coverage table cannot verify binds nothing."""
    if not isinstance(record, dict) or record.get("schema") != CARRY_OVER_SCHEMA:
        refuse(f"carry_over: schema {record.get('schema') if isinstance(record, dict) else None!r} != {CARRY_OVER_SCHEMA!r}")
    if record.get("W") != plan.W or record.get("wave_uuid") != plan.wave_uuid:
        refuse(f"carry_over: evaluated for W {str(record.get('W'))[:12]} / wave {record.get('wave_uuid')}, the launch plan is W {plan.W7} / wave {plan.wave_uuid}")
    carried = record.get("carried")
    if not isinstance(carried, dict):
        refuse("carry_over: `carried` is not a mapping")
    out: Dict[str, CarryBinding] = {}
    for sid, e in carried.items():
        row = plan.rows.get(str(sid))
        if row is None:
            refuse(f"carry_over: {sid}: not a standup the launch plan declares")
        if not isinstance(e, dict) or not isinstance(e.get("attempt_id"), str) or not e["attempt_id"]:
            refuse(f"carry_over: {sid}: entry lacks an attempt id")
        aid = e["attempt_id"]
        if "accepted_key" in row and e.get("accepted_key") != row["accepted_key"]:
            refuse(f"carry_over: {sid}: accepted_key {e.get('accepted_key')!r} is not the plan row's {row['accepted_key']!r}")
        prov = e.get("carried_from")
        if not isinstance(prov, dict) or [k for k in CARRY_PROVENANCE_KEYS if k not in prov]:
            refuse(f"carry_over: {sid}: provenance missing or incomplete (carried_from needs {list(CARRY_PROVENANCE_KEYS)})")
        if prov["attempt_id"] != aid:
            refuse(f"carry_over: {sid}: provenance names attempt {prov['attempt_id']!r}, the entry {aid!r}")
        if prov["source"] not in CARRY_SOURCES:
            refuse(f"carry_over: {sid}: source {prov['source']!r} is not one of {list(CARRY_SOURCES)}")
        if not isinstance(prov["W"], str) or len(prov["W"]) != 40:
            refuse(f"carry_over: {sid}: provenance W {str(prov['W'])[:12]!r} is not a 40-hex sha")
        if (prov["W"] == plan.W) != (prov["source"] in CURRENT_TREE_SOURCES):
            refuse(f"carry_over: {sid}: source {prov['source']!r} with provenance W {prov['W'][:12]} -- a source from a prior tree must not name this tree, and "
                   f"this tree's S5 FULL source must (fj17 / D147)")
        clauses = e.get("clauses")
        if not isinstance(clauses, dict) or any(not isinstance(clauses.get(c), dict) or clauses[c].get("ok") is not True for c in CARRY_CLAUSES):
            refuse(f"carry_over: {sid}: not every clause {list(CARRY_CLAUSES)} is recorded ok")
        hist = str(prov["standup_id"])
        smoke, *axes = split_standup_id(hist)
        if tuple(axes) != (row["leg"], row["view"], row["stage"], row["shard"]):
            refuse(f"carry_over: {sid}: the historical standup {hist!r} does not spell the plan row's axes")
        if smoke != SMOKE_ID_OF_CARRY_SOURCE[prov["source"]]:
            refuse(f"carry_over: {sid}: source {prov['source']!r} does not agree with the historical standup {hist!r} (an S5 / S3 FULL source spells smoke-S5__ / smoke-S3__)")
        if not isinstance(e.get("attempt_root"), str) or not e["attempt_root"]:
            refuse(f"carry_over: {sid}: no attempt_root")
        if aid in out:
            refuse(f"carry_over: attempt {aid} carried under two standups ({out[aid].standup_id}, {sid})")
        out[aid] = CarryBinding(attempt_id=aid, standup_id=str(sid), W=prov["W"], wave_uuid=str(prov["wave_uuid"]), historical_standup_id=hist,
                                source=prov["source"], attempt_root=e["attempt_root"], accepted_key=str(e.get("accepted_key")))
    return out


def carry_source_ws(record: Any) -> Dict[str, str]:
    """`{attempt_id: source W}` of a carry-over record, validated WITHOUT a plan (the identity table has none): the schema, every carried entry
    with an attempt id, a full provenance (:data:`CARRY_PROVENANCE_KEYS`) whose W is a 40-hex sha other than the record's own W (or the record's
    own for a `current_s5_full` source), and all five clauses ok. The identity table holds a carried root to ITS source
    tree's W, never the wave's."""
    if not isinstance(record, dict) or record.get("schema") != CARRY_OVER_SCHEMA:
        refuse(f"carry_over: schema {record.get('schema') if isinstance(record, dict) else None!r} != {CARRY_OVER_SCHEMA!r}")
    carried = record.get("carried")
    if not isinstance(carried, dict):
        refuse("carry_over: `carried` is not a mapping")
    out: Dict[str, str] = {}
    for sid, e in carried.items():
        if not isinstance(e, dict) or not isinstance(e.get("attempt_id"), str) or not e["attempt_id"]:
            refuse(f"carry_over: {sid}: entry lacks an attempt id")
        prov = e.get("carried_from")
        if not isinstance(prov, dict) or [k for k in CARRY_PROVENANCE_KEYS if k not in prov]:
            refuse(f"carry_over: {sid}: provenance missing or incomplete (carried_from needs {list(CARRY_PROVENANCE_KEYS)})")
        if prov["attempt_id"] != e["attempt_id"]:
            refuse(f"carry_over: {sid}: provenance names attempt {prov['attempt_id']!r}, the entry {e['attempt_id']!r}")
        if not isinstance(prov["W"], str) or len(prov["W"]) != 40 or (prov["W"] == record.get("W") and prov.get("source") not in CURRENT_TREE_SOURCES):
            refuse(f"carry_over: {sid}: provenance W {str(prov['W'])[:12]!r} is not a prior tree's 40-hex sha (nor this tree's S5 FULL source)")
        clauses = e.get("clauses")
        if not isinstance(clauses, dict) or any(not isinstance(clauses.get(c), dict) or clauses[c].get("ok") is not True for c in CARRY_CLAUSES):
            refuse(f"carry_over: {sid}: not every clause {list(CARRY_CLAUSES)} is recorded ok")
        if e["attempt_id"] in out:
            refuse(f"carry_over: attempt {e['attempt_id']} carried under two standups")
        out[e["attempt_id"]] = prov["W"]
    return out


def read_carry_mapping(path: Optional[Path], plan: LaunchPlan) -> Optional[Dict[str, CarryBinding]]:
    """`RS/carry_over.json` -> :func:`carry_mapping`; None when no path is given (a wave that carried nothing)."""
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        refuse(f"{p}: the carry-over record is not a file")
    return carry_mapping(read_json(p), plan)


def bind_roots_to_plan(roots: Sequence[AttemptRoot], plan: LaunchPlan, *, carried: Optional[Mapping[str, CarryBinding]] = None) -> Dict[str, AttemptRoot]:
    """Every pulled attempt root bound to the ONE planned standup it executed: its `standup_id` (spelled from the axes
    every source of its record agreed on) must be a plan row whose axes agree with the root's; the root's W AND wave
    (always recorded — :func:`read_attempt_root` requires one) must be the plan's. An attempt the plan does not declare, an
    attempt from another tree or wave, or two accepted attempts for one planned standup REFUSE — the evidence would not be
    bound to the stores it certifies.

    ``carried`` (fj5): the VALIDATED carry mapping (:func:`carry_mapping`). A root whose attempt id it names is a CARRIED
    attempt: its recorded W, wave and standup id must be EXACTLY the historical identity the record names (never rewritten,
    never the plan's), and it binds to the CURRENT planned standup the record names (its axes must agree); an S5 FULL source
    (`smoke-S5__<sid>`) is admitted here and nowhere else. The binding is stamped on the root (`carried_from`)."""
    out: Dict[str, AttemptRoot] = {}
    for root in roots:
        c = (carried or {}).get(root.attempt_id)
        if c is not None:
            if root.W != c.W:
                refuse(f"{root.label}: carried attempt {root.attempt_id} ran from W {root.W} but the carry record names W {c.W} -- the pulled root is not the carried one")
            if root.wave_uuid != c.wave_uuid:
                refuse(f"{root.label}: carried attempt {root.attempt_id} belongs to wave {root.wave_uuid} but the carry record names wave {c.wave_uuid}")
            if root.standup_id != c.historical_standup_id:
                refuse(f"{root.label}: carried attempt {root.attempt_id} executed standup {root.standup_id!r} but the carry record names {c.historical_standup_id!r}")
            row = plan.rows.get(c.standup_id)
            if row is None:
                refuse(f"{root.label}: carried attempt {root.attempt_id} is bound to standup {c.standup_id!r}, which the launch plan does not declare")
            if (row["leg"], row["view"], row["stage"], row["shard"]) != (root.leg, root.view, root.stage, root.shard):
                refuse(f"{root.label}: carried attempt {root.attempt_id} is bound to standup {c.standup_id!r} but its axes disagree with the plan row's "
                       f"({row['leg']}/{row['view']}/{row['stage']}/{row['shard']})")
            if c.standup_id in out:
                refuse(f"{root.label}: two accepted attempts pulled for planned standup {c.standup_id!r} ({out[c.standup_id].attempt_id}, {root.attempt_id}) "
                       "— exactly one attempt is accepted per standup")
            root.carried_from = c.provenance()
            out[c.standup_id] = root
            continue
        if root.W != plan.W:
            refuse(f"{root.label}: attempt {root.attempt_id} ran from W {root.W} but the launch plan is W {plan.W}")
        if root.wave_uuid != plan.wave_uuid:
            refuse(f"{root.label}: attempt {root.attempt_id} belongs to wave {root.wave_uuid} but the launch plan is wave {plan.wave_uuid}")
        sid = root.standup_id
        if root.smoke_id is not None:
            refuse(f"{root.label}: attempt {root.attempt_id} executed SMOKE standup {sid!r} (smoke {root.smoke_id}); a smoke attempt "
                   "never binds to a production row — it lives under the smoke's own results root")
        row = plan.rows.get(sid)
        if row is None:
            refuse(f"{root.label}: attempt {root.attempt_id} executed standup {sid!r}, which the launch plan does not declare "
                   f"({len(plan.rows)} planned standups)")
        if (row["leg"], row["view"], row["stage"], row["shard"]) != (root.leg, root.view, root.stage, root.shard):
            refuse(f"{root.label}: attempt {root.attempt_id} names standup {sid!r} but its axes disagree with the plan row's "
                   f"({row['leg']}/{row['view']}/{row['stage']}/{row['shard']})")
        if sid in out:
            refuse(f"{root.label}: two accepted attempts pulled for planned standup {sid!r} ({out[sid].attempt_id}, {root.attempt_id}) "
                   "— exactly one attempt is accepted per standup")
        out[sid] = root
    return out


def seed_of_store(root: AttemptRoot, store: StoreRead) -> int:
    """THE seed s of a store: on a view root the view's s (the store's recorded base must EQUAL it — the literal
    seed rule); on the Arditi root the store's recorded base, or the lane namespace's
    `_seedrep_s<s>_` for a store that records none (MMLU)."""
    view_s = root.s
    if view_s is not None:
        if store.s is not None and int(store.s) != view_s:
            refuse(f"{store.store_dir}: the store records seed_base {store.s} but sits on view {root.view} (s={view_s}) — the "
                   "literal-seed rule (seed base = s) is broken or the store is mis-filed")
        return view_s
    if store.s is not None:
        return int(store.s)
    ns = _s_from_namespace(store.store_dir)
    if ns is None:
        refuse(f"{store.store_dir}: an Arditi-root store with no recorded seed_base and no `_seedrep_s<s>_` namespace in its "
               "path — its seed cannot be established")
    return ns
