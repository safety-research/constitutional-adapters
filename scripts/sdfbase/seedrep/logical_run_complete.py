#!/usr/bin/env python3
"""Logical-run completion across shards for the Qwen seed-replicate native battery.

A run that spans containers (a multi-turn goal sub-shard fan-out, a static leg split by S3) has its shards emit
``run_state running`` and item rows only; the ``run_state generated`` + the completion ``run_patch``
(``n_items_generated`` = the union) are emitted ONCE, here, on the grading host, after the pull, and only after the shard union
is verified DISJOINT and COMPLETE against the planned item set — a shard never emits ``generated`` itself, and a run
whose shards overlap or fall short is REFUSED (verdict written, nothing emitted).

Inputs: ONE planned-run file (every route of a per-task map is one logical run), the wave's accepted map, the pulled
attempt roots (``registry_publish.discover_attempts``), the grading host spool to emit into and a verdict directory. The
shards of a run are the ACCEPTED, non-superseded, COMPLETE base-stage attempts of the run's CONTAINER and view — the
container is the launch plan's standup leg, which every planned identity carries under ``extra.container`` (the eval leg
for a single-leg container, ``controls`` for the combined xstest + or_bench + mmlu container), so one accepted controls
attempt completes all three logical runs; with ``--launch-plan`` the launch plan's own rows (``standups[]`` / ``rungs[]``,
each a ``control_key`` and ``children[].registry_identity_rel``) are joined to the plan file and must agree. A shard's
item ids come from ``<attempt root>/registry/items/<run_dir>.json`` (``{"run_id", "item_ids": [...]}``, written by the
container at store COMPLETE) or, when absent, from the ``item_result`` rows the shard's
own spool holds for the run; a shard that shows neither is refused, never assumed empty.

RESUMABLE AND IDEMPOTENT. The verdict file ``<verdict-dir>/<run_dir_name>.json`` is the durable admission:
``{run_id, verdict: ADMITTED|REFUSED, reasons, shards, n_union, expected_n, admission, completion, emitted}``. On ADMITTED the
``admission`` block binds (run id, W7, wave, container, view, the complete accepted-attempt set — each attempt with its six
axes and the logical sha of every record file it holds for the run) and carries a digest; the completion records are
DERIVED from it deterministically (``record_id`` = uuid5 over the payload, ``written_utc`` = the admission stamp), written
once each (an id already on the spool is never minted again), and only after both are re-read from disk does
``completion.state`` flip from ``pending`` to ``committed``. ``registry_publish.py`` publishes a run only against a
COMMITTED admission, so a crash after the verdict write and before the records leaves publication forbidden; a re-run
after any crash finishes the partial emission without a second ``generated``; a re-run after commit is a no-op; a
committed admission is never overwritten (a changed shard set refuses loudly).

  logical_run_complete.py --plan RS/registry/plans/<leg>/<arm>__s<s>.json --accepted RS/accepted.json \\
                          --attempts RS/stores --spool ~/registry/records --verdict-dir RS/registry/union \\
                          [--launch-plan RS/launch_plan.json] [--expected-shards N] [--expected-items items.json] \\
                          [--items-source auto|items_json|spool] [--dry-run]
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path and not any(p.endswith("/src") and "sisteer" in os.listdir(p) for p in sys.path if os.path.isdir(p)):
    sys.path.insert(0, str(REPO / "src"))
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from sisteer.registry import db as DB  # noqa: E402
from sisteer.registry import emit, identity  # noqa: E402
from sisteer.registry import records as R  # noqa: E402

import registry_plans as RP  # noqa: E402
import registry_publish as PUB  # noqa: E402

SCRIPT = "scripts/sdfbase/seedrep/logical_run_complete.py"
SESSION = "grading_host:logical_run_complete"          # a fixed producer session, so a retry mints byte-identical records
VERDICT_ADMITTED = PUB.VERDICT_ADMITTED
VERDICT_REFUSED = "REFUSED"
BASE_STAGE = PUB.BASE_STAGE


class CompletionRefused(ValueError):
    """The shard union is not disjoint-and-complete, a shard already declared completion, or the admission on disk cannot be
    honoured (a committed admission re-derived differently): nothing is emitted."""


def _write_json(path: Path, obj) -> None:
    R.write_pointer(str(path), json.dumps(obj, indent=1, sort_keys=True))     # .tmp + fsync + rename


CLAIM_TIMEOUT_S = 600.0      # a completion pass takes well under a second; the wait covers a peer's pass — an flock dies with its holder, never stale


@contextlib.contextmanager
def _completion_claim(verdict_dir: Path, run_dir: str):
    """The per-logical-run claim: an exclusive ``flock`` on ``<verdict_dir>/<run_dir>.lock`` — the registry's own
    ``db.write_lock`` — taken BEFORE the verdict on disk or the record ids on the spool are read, and held through the pending
    write, the record writes, the disk verification and the committed transition, so two overlapping passes serialise and the
    second reads the first's COMMITTED verdict (a no-op) instead of minting a second ``generated``."""
    try:
        with DB.write_lock(str(verdict_dir), timeout_s=CLAIM_TIMEOUT_S, name=f"{run_dir}.lock"):
            yield
    except DB.LockHeld as exc:
        raise CompletionRefused(f"{Path(verdict_dir) / (run_dir + '.lock')}: {exc} after {CLAIM_TIMEOUT_S:.0f}s; another completion pass holds "
                                "this run — nothing written") from None


# --------------------------------------------------------------------------------------------
# which attempts are a run's shards
# --------------------------------------------------------------------------------------------


def shards_of(attempts: Sequence[PUB.AttemptRoot], accepted: PUB.Accepted, container: str, view: str) -> List[PUB.AttemptRoot]:
    """The accepted, non-superseded, COMPLETE attempts of (container, view) whose stage IS ``base`` (a stated axis, never a None read
    as base), by shard index. ``is_accepted`` already demands all six axes exactly."""
    out = [a for a in attempts
           if a.container == container and a.view == view and accepted.is_accepted(a)[0] and not a.superseded_by and a.complete
           and a.stage == BASE_STAGE]
    out.sort(key=lambda a: (a.shard if a.shard is not None else -1, a.attempt_id))
    return out


def load_launch_plan(path) -> List[Dict[str, object]]:
    """The launch plan's container rows: ``standups[]`` + ``rungs[]`` (``seedrep_launch_plan/1``: each a ``control_key`` and
    ``children[]`` carrying ``registry_identity_rel``), or a bare list of such rows."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    shape = f"{path}: a launch plan is {{'standups': [...], 'rungs': [...]}} (seedrep_launch_plan/1) or a list of its rows"
    if isinstance(data, list):
        rows = list(data)
    elif isinstance(data, dict) and ("standups" in data or "rungs" in data):
        # a dict naming neither key is NOT an empty plan: it is another schema. Each key PRESENT must be a list — `{}`, 0, false, a
        # string or an integer there is refused by shape, never read as "no rows"
        rows = []
        for k in ("standups", "rungs"):
            if k in data:
                if not isinstance(data[k], list):
                    raise CompletionRefused(f"{shape}; {k!r} is a {type(data[k]).__name__}, not a list")
                rows.extend(data[k])
    else:
        raise CompletionRefused(shape)
    bad = [i for i, r in enumerate(rows) if not isinstance(r, dict)]
    if bad:
        raise CompletionRefused(f"{shape}; row(s) {bad[:5]} are not mappings")
    return rows


def launch_plan_join(rows: Sequence[Mapping[str, object]], pf: RP.PlanFile) -> Tuple[str, int]:
    """``(container, n_base_shards)`` for a plan file from the launch plan's rows whose children name it (``registry_identity_rel``
    ends in the file's ``<leg>/<name>``): exactly one container across them; the base-stage rows for the file's view count as
    the declared shard count."""
    tail = str(Path(*pf.path.resolve().parts[-2:]))
    def names_file(c) -> bool:
        rel = c.get("registry_identity_rel")                  # read in its stated type: a non-string never names the file
        return isinstance(rel, str) and rel.endswith(tail)
    mine = [r for r in rows if any(names_file(c) for c in (r.get("children") or []) if isinstance(c, Mapping))]
    if not mine:
        raise CompletionRefused(f"launch plan: no row's children name plan file {tail}")
    axes = []
    for r in mine:
        try:
            axes.append(PUB.parse_axes(r.get("control_key")))       # a row's key parses whole or the join refuses
        except PUB.PublishRefused as exc:
            raise CompletionRefused(f"launch plan: row {r.get('standup_id')!r} naming plan file {tail}: {exc}") from None
    containers = {ax["container"] for ax in axes}             # parse_axes guarantees six typed axes
    if len(containers) != 1:
        raise CompletionRefused(f"launch plan: plan file {tail} is named by rows of {len(containers)} containers {sorted(containers)}; a run has one")
    base = [ax for ax in axes if ax["view"] == pf.view and ax["stage"] == BASE_STAGE]
    return containers.pop(), len(base)


# --------------------------------------------------------------------------------------------
# the shard union
# --------------------------------------------------------------------------------------------


@dataclass
class ShardItems:
    attempt: PUB.AttemptRoot
    item_ids: Set[str]
    source: str                      # items_json | spool
    states: List[str] = field(default_factory=list)   # the run_state states this shard emitted for the run
    files: Dict[str, str] = field(default_factory=dict)   # {record file: logical sha} of the shard's run dir (the admission binds them)


def scan_run_dir(rd: Path, run_id: str) -> Tuple[Dict[str, str], List[str], Set[str]]:
    """ONE pass over a run directory: ``({file: logical sha}, [run_state states], {item ids})`` for ``run_id`` — the hasher is
    fed the same decompressed bytes the parser reads, so the sha costs no second read. EVERY record of every file must bind to
    ``run_id`` through its type's own field (``PUB.record_run_id``: ``payload.run_id``; ``artifact``: ``extra.run_id``): a mixed-run, run-less
    or run-unbound file REFUSES the shard, it is never hashed in."""
    files: Dict[str, str] = {}
    states: List[str] = []
    items: Set[str] = set()
    for p in R.iter_record_files(str(rd)):
        h = hashlib.sha256()
        for i, rec, _sha in R.read_record_file(p, hasher=h):
            payload = rec.get("payload") if isinstance(rec.get("payload"), dict) else {}
            rid = PUB.record_run_id(rec)                      # by the record type's OWN binding field (payload.run_id; artifact: extra.run_id)
            if rid != run_id:
                raise CompletionRefused(f"{p}: record {i} ({rec.get('record_type')}) names run {rid!r}, the run directory is {run_id}'s; a record "
                                        "file with a missing, foreign or unbound run is never admitted")
            if rec.get("record_type") == "run_state":
                states.append(payload.get("state"))           # stated type; the completion check is `"generated" in states`
            elif rec.get("record_type") == "item_result":
                item = payload.get("item_id")
                if not isinstance(item, str) or not item:     # item ids are compared across shards and to the sidecar: never coerced
                    raise CompletionRefused(f"{p}: record {i} (item_result) has item_id {item!r} ({type(item).__name__}), not a non-empty string; "
                                            "a shard's items are read in their stated type, never coerced")
                items.add(item)
        files[os.path.relpath(p, str(rd))] = h.hexdigest()
    return files, states, items


def shard_items(a: PUB.AttemptRoot, campaign: str, run_id: str, source: str = "auto") -> ShardItems:
    """The shard's item set for ``run_id`` from its NATIVE ``item_result`` rows, which must exist as a run directory in the attempt's
    spool — a sidecar alone certifies nothing: ``registry/items/<run_dir>.json`` is the shard-item INTERFACE, not a replacement
    for the rows. With a sidecar (``auto`` / ``items_json``) its item set must EQUAL the native set; a shard with
    no ``item_result`` rows counts as empty only when its run dir holds run-state evidence for the run."""
    run_dir = identity.run_dir_name(run_id)
    rd = a.spool / campaign / run_dir
    items_json = a.items_dir / f"{run_dir}.json"
    if not rd.is_dir():
        raise CompletionRefused(f"{a.root}: no native run directory {rd.relative_to(a.root)} for run {run_id}"
                                + (f" (the sidecar {items_json.relative_to(a.root)} alone certifies nothing)" if items_json.is_file() else "")
                                + "; a shard that recorded nothing for the run cannot be counted as empty")
    files, states, spool_ids = scan_run_dir(rd, run_id)
    if not spool_ids and not states:
        raise CompletionRefused(f"{rd}: no item_result rows and no run_state for run {run_id}; an empty shard needs run-state evidence, "
                                "a run directory that holds neither recorded nothing")
    if source in ("auto", "items_json") and items_json.is_file():
        with open(items_json, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or data.get("run_id") != run_id or not isinstance(data.get("item_ids"), list):
            raise CompletionRefused(f"{items_json}: not {{run_id: {run_id!r}, item_ids: [...]}}")
        ids = list(data["item_ids"])                          # in their stated types: a numeric id is not the string id of a native row
        untyped = [x for x in ids if not isinstance(x, str) or not x]
        if untyped:
            raise CompletionRefused(f"{items_json}: item_ids carry {len(untyped)} member(s) that are not non-empty strings, e.g. {untyped[:5]!r}; "
                                    "the sidecar is read in its stated types, never coerced")
        if len(set(ids)) != len(ids):
            raise CompletionRefused(f"{items_json}: item_ids repeat within the shard ({len(ids) - len(set(ids))} duplicates)")
        if set(ids) != spool_ids:
            only_side, only_native = sorted(set(ids) - spool_ids), sorted(spool_ids - set(ids))
            raise CompletionRefused(f"{items_json}: the sidecar's item set differs from the native item_result rows in {rd} (sidecar-only "
                                    f"{len(only_side)}, e.g. {only_side[:5]}; native-only {len(only_native)}, e.g. {only_native[:5]}); the sidecar is "
                                    "an interface to the rows, not a substitute for them")
        return ShardItems(a, set(ids), "items_json", states, files)
    if source == "items_json":
        raise CompletionRefused(f"{a.root}: no {items_json.relative_to(a.root)} for run {run_id}")
    return ShardItems(a, spool_ids, "spool", states, files)

def check_union(shards: Sequence[ShardItems], *, expected_items: Optional[Set[str]], expected_n: Optional[int],
                allow_empty_shard: bool = False) -> Dict[str, object]:
    """Disjoint AND complete. Returns the verdict block (``verdict``, ``reasons``, ``n_union``, ``expected_n``)."""
    reasons: List[str] = []
    seen: Dict[str, str] = {}
    overlaps: Dict[str, List[str]] = {}
    for sh in shards:
        if not sh.item_ids and not allow_empty_shard:
            reasons.append(f"shard {sh.attempt.shard} ({sh.attempt.attempt_id}) recorded no items")
        for it in sh.item_ids:
            if it in seen:
                overlaps.setdefault(it, [seen[it]]).append(sh.attempt.attempt_id)
            else:
                seen[it] = sh.attempt.attempt_id
    if overlaps:
        sample = sorted(overlaps)[:5]
        reasons.append(f"{len(overlaps)} item(s) in more than one shard (not disjoint), e.g. {sample}")
    union = set(seen)
    if expected_items is not None:
        missing = sorted(expected_items - union)
        extra = sorted(union - expected_items)
        if missing:
            reasons.append(f"{len(missing)} planned item(s) missing from the union, e.g. {missing[:5]}")
        if extra:
            reasons.append(f"{len(extra)} item(s) outside the planned set, e.g. {extra[:5]}")
        exp_n: Optional[int] = len(expected_items)
    elif expected_n is not None:
        exp_n = int(expected_n)
        if len(union) != exp_n:
            reasons.append(f"union holds {len(union)} items, the plan expects {exp_n}")
    else:
        exp_n = None
        reasons.append("no expected item set: the plan carries no n_items_planned and --expected-items was not given")
    return {"verdict": VERDICT_ADMITTED if not reasons else VERDICT_REFUSED, "reasons": reasons, "n_union": len(union), "expected_n": exp_n}


# --------------------------------------------------------------------------------------------
# the admission and its deterministic completion records
# --------------------------------------------------------------------------------------------


def admission_block(ident: Mapping[str, object], container: str, view: str, per_shard: Sequence[ShardItems]) -> Dict[str, object]:
    """What publication is bound to: the run, the wave (every shard's W7 / wave_uuid, which must agree), the container, the
    view, and the COMPLETE accepted-attempt set — each attempt with its six axes and the logical sha of every record file it
    holds for the run."""
    ids = [sh.attempt.attempt_id for sh in per_shard]
    for sh in per_shard:                       # independently of the selection: every admitted attempt PROVES all six axes. The
        bad = PUB.absent_or_invalid_axes(sh.attempt.axes)   # base-stage rule is the completion's (shards_of): a publication admission may
        if bad:                                              # legitimately bind a base attempt and a rung attempt of one run (P1-2)
            raise CompletionRefused(f"attempt {sh.attempt.attempt_id} cannot be admitted: ATTEMPT.json states no {bad}; an admission binds six "
                                    "proven axes per attempt")
    for ax, what in (("wave_uuid", "wave_uuid"), ("w7", "W")):
        vals = {sh.attempt.axes.get(ax) for sh in per_shard}
        if None in vals or len(vals) != 1:
            raise CompletionRefused(f"shards {ids} do not name one {what} ({sorted(map(str, vals))}); ATTEMPT.json must record it (plan §4.6) — "
                                    "the wave axis is what binds the admission")
    plan_w7 = str(ident.get("code_sha") or "")[:7] or None
    w7 = per_shard[0].attempt.w7
    if plan_w7 and w7 != plan_w7:
        raise CompletionRefused(f"shards ran at W {w7}, the plan file was cut at W {plan_w7}; a run's records are bound to the plan's W")
    adm: Dict[str, object] = {
        "run_id": str(ident["run_id"]), "campaign": str(ident["campaign"]), "w7": w7, "wave_uuid": per_shard[0].attempt.wave_uuid,
        "container": container, "view": view,
        "attempts": {sh.attempt.attempt_id: {**sh.attempt.axes, "root": str(sh.attempt.root), "n_items": len(sh.item_ids), "files": sh.files}
                     for sh in per_shard},
    }
    adm["digest"] = R.record_sha(adm)
    return adm


def completion_payloads(adm: Mapping[str, object], n_union: int, admission_utc: str, verdict_path: Path) -> Dict[str, Tuple[str, Dict[str, object]]]:
    """``{record_type: (record_id, payload)}`` — pure functions of the admission, so a retry derives the SAME ids."""
    rid = str(adm["run_id"])
    state = {"run_id": rid, "state": "generated", "event_utc": admission_utc, "evidence_kind": "completion_artifact", "evidence_uri": str(verdict_path)}
    patch = {"run_id": rid, "fields": {"n_items_generated": int(n_union),
                                      "extra_json": {"logical_run_shards": sorted(adm["attempts"]), "union_verdict_uri": str(verdict_path),   # type: ignore[call-overload]
                                                     "completed_by": SCRIPT, "admission_digest": adm["digest"]}}}
    return {"run_state": (identity.adapter_record_id(state, "run_state"), state), "run_patch": (identity.adapter_record_id(patch, "run_patch"), patch)}


def record_ids_on_disk(run_dir: Path) -> Dict[str, str]:
    """``{record_id: file path}`` of every record a run directory holds."""
    return {rec.get("record_id"): p for p in R.iter_record_files(str(run_dir)) for _i, rec, _sha in R.read_record_file(p)}


def write_record_once(run_dir: Path, record_type: str, rec: Mapping[str, object]) -> str:
    """One record into its own immutable file under ``run_dir``: the next free sequence number, created exclusively (the
    ``RecordWriter`` discipline: never append to a file the transport may have copied). ``RecordWriter.write_many`` mints
    its own uuid4 ids, so a prebuilt deterministic record needs this writer."""
    run_dir.mkdir(parents=True, exist_ok=True)
    heads = [fn.split("-", 1)[0] for fn in os.listdir(run_dir)]
    seq = max((int(h) for h in heads if re.fullmatch(r"[0-9]+", h)), default=0)      # ASCII digits only (str.isdigit admits others)
    for _ in range(1000):
        seq += 1
        try:
            return R.write_records(str(run_dir / R.record_file_name(seq, record_type)), [rec], exclusive=True)
        except FileExistsError:
            continue
    raise CompletionRefused(f"{run_dir}: no free sequence number for the completion record")


def _completion_state(v: Optional[Mapping[str, object]]) -> Optional[str]:
    comp = (v or {}).get("completion")
    return comp.get("state") if isinstance(comp, dict) and comp.get("state") else None


def _digest(v: Optional[Mapping[str, object]]) -> Optional[str]:
    adm = (v or {}).get("admission")
    return adm.get("digest") if isinstance(adm, dict) and adm.get("digest") else None


# --------------------------------------------------------------------------------------------
# completion
# --------------------------------------------------------------------------------------------


def evaluate_run(pf: RP.PlanFile, route: str, attempts: Sequence[PUB.AttemptRoot], accepted: PUB.Accepted, *, container: str,
                 expected_items: Optional[Set[str]], expected_shards: Optional[int], items_source: str, allow_empty_shard: bool,
                 now: str) -> Dict[str, object]:
    """The verdict of one logical run — pure: reads the pulled tree, writes nothing. ADMITTED carries the admission block."""
    ident = pf.identities[route]
    run_id, campaign, view = str(ident["run_id"]), str(ident["campaign"]), pf.view
    ex = ident.get("extra") or {}
    reasons: List[str] = []
    # FAIL CLOSED before anything is decided: an attempt the accepted map places in THIS run's (container, view) but whose
    # ATTEMPT.json does not prove all six axes is a malformed pulled tree — refused here as an exception, so no verdict and no record
    # is written; it is not a scientific refusal of the union (a legacy spelled-out root would otherwise wildcard its missing axes)
    for a in attempts:
        key = accepted.by_id.get(a.attempt_id)
        if key is None or key["container"] != container or key["view"] != view:      # parse_axes guarantees six typed axes
            continue
        bad = PUB.absent_or_invalid_axes(a.axes)
        if bad:
            raw = next((k for k, v in accepted.raw.items() if v == a.attempt_id), "?")
            raise CompletionRefused(f"{a.root}: accepted under {raw!r} for container {container!r} view {view!r}, but its ATTEMPT.json states no "
                                    f"{bad}; an attempt that cannot prove its axes never completes a run — refusing before any verdict is written")
    shards = shards_of(attempts, accepted, container, view)
    if not shards:
        reasons.append(f"no accepted, complete, base-stage attempt for container {container!r} (leg {ex.get('leg')}), view {view}")
    if expected_shards is not None and len(shards) != int(expected_shards):
        reasons.append(f"{len(shards)} accepted shard(s), the launch plan declares {expected_shards}")
    per_shard: List[ShardItems] = []
    if not reasons:
        try:
            per_shard = [shard_items(a, campaign, run_id, items_source) for a in shards]
        except CompletionRefused as exc:
            reasons.append(str(exc))
    union: Dict[str, object] = {"n_union": 0, "expected_n": None}
    already_generated = False
    admission: Optional[Dict[str, object]] = None
    if not reasons:
        n_planned = ident.get("n_items_planned")
        union = check_union(per_shard, expected_items=expected_items, expected_n=int(n_planned) if n_planned is not None else None,
                            allow_empty_shard=allow_empty_shard)
        reasons.extend(union["reasons"])   # type: ignore[arg-type]
        gen_shards = [sh.attempt.attempt_id for sh in per_shard if "generated" in sh.states]
        if gen_shards and len(per_shard) > 1:
            reasons.append(f"shard(s) {gen_shards} emitted run_state generated themselves; a shard of a multi-shard run never declares completion")
        elif gen_shards:
            already_generated = True   # a single-container store completed itself through its own hook: nothing to add
        if not reasons:
            try:
                admission = admission_block(ident, container, view, per_shard)
            except CompletionRefused as exc:
                reasons.append(str(exc))
    return {
        "run_id": run_id, "campaign": campaign, "leg": ex.get("leg"), "container": container, "view": view, "arm": ex.get("arm"), "s": ex.get("s"),
        "route": route, "verdict": VERDICT_ADMITTED if not reasons else VERDICT_REFUSED, "reasons": reasons,
        "shards": [{"attempt_id": sh.attempt.attempt_id, "shard": sh.attempt.shard, "n_items": len(sh.item_ids), "items_source": sh.source,
                    "states": sh.states, "root": str(sh.attempt.root)} for sh in per_shard],
        "n_union": union["n_union"], "expected_n": union["expected_n"], "already_generated": already_generated, "emitted": [],
        "admission": admission, "completion": None, "utc": now, "script": SCRIPT,
    }


def _admission_stamp(existing: Optional[Mapping[str, object]], admission: Mapping[str, object], run_spool_dir: Path, verdict_path: Path,
                     now: str) -> str:
    """The stamp the completion records derive from: a PENDING admission of the same digest resumes with ITS stamp; a pending one
    of another digest may be superseded only while none of its records exist; a committed one is handled by the caller."""
    if _completion_state(existing) != PUB.COMPLETION_PENDING or existing is None:
        return now
    comp = existing["completion"]   # type: ignore[index]
    if _digest(existing) == admission["digest"]:
        return comp.get("admission_utc") or now
    on_disk = record_ids_on_disk(run_spool_dir)
    stale = [rid for rid in (comp.get("record_ids") or {}).values() if rid in on_disk]
    if stale:
        raise CompletionRefused(f"{verdict_path}: a PENDING admission (digest {_digest(existing)}) already put records {stale} on the spool and "
                                f"this pass derives a different admission ({admission['digest']}); refusing to mint a second completion")
    return now


def complete_run(pf: RP.PlanFile, route: str, attempts: Sequence[PUB.AttemptRoot], accepted: PUB.Accepted, *, spool: Path, verdict_dir: Path,
                 expected_items: Optional[Set[str]] = None, expected_shards: Optional[int] = None, items_source: str = "auto",
                 dry_run: bool = False, allow_empty_shard: bool = False, now: Optional[str] = None,
                 launch_plan: Optional[Sequence[Mapping[str, object]]] = None) -> Dict[str, object]:
    ident = pf.identities[route]
    run_id = str(ident["run_id"])
    run_dir = identity.run_dir_name(run_id)
    verdict_path = Path(verdict_dir) / f"{run_dir}.json"
    container = pf.container
    if launch_plan is not None:
        # the launch plan's own run -> container join, cross-checked against the plan file: they must agree
        lp_container, lp_shards = launch_plan_join(launch_plan, pf)
        if lp_container != container:
            raise CompletionRefused(f"{pf.path}: the launch plan places run {run_id} in container {lp_container!r}, the plan file says {container!r}")
        if expected_shards is None:
            expected_shards = lp_shards
    now = now or R.utc_now()
    verdict = evaluate_run(pf, route, attempts, accepted, container=container, expected_items=expected_items, expected_shards=expected_shards,
                           items_source=items_source, allow_empty_shard=allow_empty_shard, now=now)
    run_spool_dir = Path(spool) / str(ident["campaign"]) / run_dir
    if dry_run:                                   # writes nothing, so it takes no claim: the verdict as this pass would decide it
        return _decide_and_commit(verdict, PUB.read_verdict(verdict_dir, run_dir), run_id, run_spool_dir, verdict_path, now,
                                  code_sha=str(ident.get("code_sha") or "") or None, dry_run=True)
    with _completion_claim(Path(verdict_dir), run_dir):
        # the verdict on disk and the record ids on the spool are read ONLY under the claim, which is held through the pending
        # write, the record writes, the disk verification and the committed transition
        return _decide_and_commit(verdict, PUB.read_verdict(verdict_dir, run_dir), run_id, run_spool_dir, verdict_path, now,
                                  code_sha=str(ident.get("code_sha") or "") or None, dry_run=False)


def _decide_and_commit(verdict: Dict[str, object], existing: Optional[Mapping[str, object]], run_id: str, run_spool_dir: Path,
                       verdict_path: Path, now: str, *, code_sha: Optional[str], dry_run: bool) -> Dict[str, object]:
    """Reconcile this pass's verdict with the one on disk and, when admitted, write the completion ONCE (module docstring); called
    under the run's claim unless ``dry_run``."""
    committed = _completion_state(existing) == PUB.COMPLETION_COMMITTED
    if verdict["verdict"] != VERDICT_ADMITTED:
        if committed:
            raise CompletionRefused(f"{verdict_path}: a COMMITTED admission exists for {run_id} but the pulled tree no longer supports it "
                                    f"({verdict['reasons']}); the committed verdict is kept, nothing is rewritten — investigate the tree")
        if not dry_run:
            _write_json(verdict_path, verdict)
        return verdict
    admission = verdict["admission"]
    assert isinstance(admission, dict)
    if committed:
        if _digest(existing) == admission["digest"]:
            return {**existing, "noop": True}      # type: ignore[dict-item]  # idempotent: the committed verdict, nothing minted
        raise CompletionRefused(f"{verdict_path}: a COMMITTED admission exists for {run_id} with digest {_digest(existing)}, this pass derives "
                                f"{admission['digest']} (shards {sorted(admission['attempts'])}); a committed admission is never re-derived")
    admission_utc = _admission_stamp(existing, admission, run_spool_dir, verdict_path, now)
    payloads = {} if verdict["already_generated"] else completion_payloads(admission, int(verdict["n_union"]), admission_utc, verdict_path)
    verdict["completion"] = {"state": PUB.COMPLETION_PENDING, "admission_utc": admission_utc, "record_ids": {rt: rid for rt, (rid, _p) in payloads.items()}}
    if dry_run:
        return verdict
    _write_json(verdict_path, verdict)           # the admission is durable BEFORE any record: a crash here leaves publication forbidden
    if payloads:
        producer = emit.producer_block(SCRIPT, code_sha=code_sha, session=SESSION)
        on_disk = record_ids_on_disk(run_spool_dir)
        for rt, (rid, payload) in payloads.items():
            if rid not in on_disk:                    # written by an earlier pass: never minted again
                write_record_once(run_spool_dir, rt, R.make_record(rt, payload, producer, rid, admission_utc))
        on_disk = record_ids_on_disk(run_spool_dir)   # verify from disk, never from what this pass believes it wrote
        missing = [rt for rt, (rid, _p) in payloads.items() if rid not in on_disk]
        if missing:
            raise CompletionRefused(f"{run_spool_dir}: completion record(s) {missing} not on the spool after writing; the admission stays pending")
        verdict["emitted"] = [on_disk[rid] for _rt, (rid, _p) in payloads.items()]
    verdict["completion"]["state"] = PUB.COMPLETION_COMMITTED   # type: ignore[index]
    _write_json(verdict_path, verdict)
    return verdict

def complete_plan_file(plan_path, *, accepted_path, attempt_roots: Sequence, spool, verdict_dir, routes: Optional[Sequence[str]] = None,
                       expected_items: Optional[Set[str]] = None, expected_shards: Optional[int] = None, items_source: str = "auto",
                       dry_run: bool = False, allow_empty_shard: bool = False, launch_plan_path=None) -> List[Dict[str, object]]:
    pf = RP.load_plan_file(plan_path)
    accepted = PUB.load_accepted(accepted_path)
    attempts = PUB.discover_attempts(attempt_roots)
    launch_plan = load_launch_plan(launch_plan_path) if launch_plan_path else None
    out = []
    for route in (routes or list(pf.identities)):
        if route not in pf.identities:
            raise CompletionRefused(f"{plan_path}: no route {route!r}; routes: {sorted(pf.identities)}")
        out.append(complete_run(pf, route, attempts, accepted, spool=Path(spool), verdict_dir=Path(verdict_dir), expected_items=expected_items,
                                expected_shards=expected_shards, items_source=items_source, dry_run=dry_run, allow_empty_shard=allow_empty_shard,
                                launch_plan=launch_plan))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plan", required=True, help="one planned-run file (every route of a per-task map is completed)")
    ap.add_argument("--route", nargs="*", default=None, help="only these routes of a per-task map")
    ap.add_argument("--accepted", required=True)
    ap.add_argument("--attempts", nargs="+", required=True)
    ap.add_argument("--spool", required=True, help="the grading host registry spool to emit into (~/registry/records)")
    ap.add_argument("--verdict-dir", required=True, help="RS/registry/union")
    ap.add_argument("--launch-plan", default=None, help="RS/launch_plan.json (seedrep_launch_plan/1): the run -> container join and the shard count")
    ap.add_argument("--expected-items", default=None, help="JSON list of the planned item ids (else n_items_planned counts)")
    ap.add_argument("--expected-shards", type=int, default=None, help="the launch plan's declared shard count for (container, view)")
    ap.add_argument("--items-source", choices=("auto", "items_json", "spool"), default="auto")
    ap.add_argument("--allow-empty-shard", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    expected_items = None
    if a.expected_items:
        with open(a.expected_items, "r", encoding="utf-8") as fh:
            items = json.load(fh)
        if not isinstance(items, list):
            print("logical_run_complete: --expected-items must be a JSON list", file=sys.stderr)
            return 2
        if not all(isinstance(x, str) and x for x in items):
            print("logical_run_complete: --expected-items must be a JSON list of non-empty strings (item ids are never coerced)", file=sys.stderr)
            return 2
        expected_items = set(items)
    try:
        verdicts = complete_plan_file(a.plan, accepted_path=a.accepted, attempt_roots=a.attempts, spool=a.spool, verdict_dir=a.verdict_dir,
                                      routes=a.route, expected_items=expected_items, expected_shards=a.expected_shards, items_source=a.items_source,
                                      dry_run=a.dry_run, allow_empty_shard=a.allow_empty_shard, launch_plan_path=a.launch_plan)
    except (CompletionRefused, PUB.PublishRefused, RP.RegistryPlanRefused) as exc:
        print(f"logical_run_complete: REFUSED: {exc}", file=sys.stderr)
        return 2
    for v in verdicts:
        print(json.dumps({k: v.get(k) for k in ("run_id", "verdict", "reasons", "n_union", "expected_n", "already_generated", "emitted", "noop")}
                         | {"completion": _completion_state(v)}, sort_keys=True))
    return 0 if all(v["verdict"] == VERDICT_ADMITTED for v in verdicts) else 1


if __name__ == "__main__":
    sys.exit(main())
