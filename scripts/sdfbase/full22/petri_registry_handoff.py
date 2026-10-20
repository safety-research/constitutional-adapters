#!/usr/bin/env python
"""petri_registry_handoff.py — the WRITE side of the native Petri registry hand-off: every step is registry-native.

The Petri proxy job registers its runs on the CLUSTER at job start (petri_proxy.sbatch -> common.sh f22_registry_register ->
emit_cli register --from-config: one run per declared eval x object, the pending file $SHARED/registry/pending/<jobid>.json with
each run's ROUTING facts — `route.object` = the wave config's cell entry, `route.tasks` = the eval's producer spellings). The
audit itself runs on the GRADING host: petri_drive_full22.sh drives scripts/petri_run.py, whose CellStore
(`~/.local/state/petri_native/petri_full22_<tag>/cells/<petri_run_key>`) records through the default store recorder
(sisteer.registry.hooks) ONLY when the job's identity reaches it — so before this module nothing patched those runs'
`store_uri`, and 06 --gate petri found null stores. Three steps, all run by the drive under its venv:

  import    --pending <the proxy's pending file, fetched from the cluster> --job <jobid> --out-dir <drive state dir> [--sh]
            Verifies the document (the job id, a non-empty `runs` list) and writes the DRIVE's identity file
            <out-dir>/<jobid>.json: the same runs, each `route.object` re-spelled from the wave config's cell entry to the
            Petri harness cell id — sisteer.capeval.petri.registry_adapter.harness_cell_id, the ONE translation the Petri
            manifest's `cells` are generated from (tests/test_sdfbase_full22_wave.py pins the generator to it) — because the
            Petri store names its object in `arm_name` / `cell_id` in the HARNESS spelling (`si_constv2_mix_full__psr_ce_all`,
            `prompted__constitution`) and the recorder routes rows by `"<object>|<task>"`. `route.object_wave` keeps the
            config entry. An entry with no harness translation refuses by name (never a silently unrouted object). `--sh`
            prints the two exports the drive evals: SISTEER_REGISTRY_RUN_IDENTITY=<file> and SISTEER_REGISTRY_SPOOL=<the grading
            host's spool: $SISTEER_REGISTRY_SPOOL, else emit.spool_root() — ~/registry/records there, the directory the
            harvester folds>; no spool resolving is a refusal (the drive is the producer; its runs would never get a store).
  close-leg --identity <file> --cell-dir <cells/<run key>>
            After a leg ENDS. Reads the cell's manifest (`cell_id`) and requires results.json (the runner's cell-done marker,
            written only at zero unresolved required dimensions — H9); a cell without it is INCOMPLETE: nothing filed, its runs
            stay `running`. For every run whose `route.object` is this cell: the six behaviour runs were opened and closed by
            the store recorder itself (rows route by `petri/<behaviour>`; on_results filed `run_patch store_uri` +
            `run_state generated`) — a run whose spool directory already holds `generated` is left alone; the POOLED run
            (`petri_pooled`) has no row spelling to route by and is filed HERE: `run_patch {store_uri, store_completion_artifact_uri,
            store_completion_sha256, n_items_generated}` + `run_state generated` (evidence: results.json), then a push. A cell
            no registered run names (the `none__lora0` control leg) files nothing and says so.
  finish    --identity <file>
            A final push of every run's record directory (a no-op without $SISTEER_REGISTRY_PUSH: on the grading host the spool IS
            the harvester's records directory).
  verify    --identity <file> [--summary <path>] [--skipped <objects,…>] [--objects <objects,…>]
            Every planned run of the identity is FILED on this host's spool — a `run_state generated` and a `store_uri` in its record
            directory (what the store recorder / close-leg write) — one line per run; `--summary` writes {n_planned, n_filed, runs:
            [{run_id, eval_id, object, state, store_uri, filed}]} for the drive's verdict; `--skipped` names objects the standup
            skipped by name (their runs can never be filed: reported `skipped`, not unfiled); `--objects` restricts the check to those
            objects (a drive of selected legs). Exit 2 (REFUSED) when any run in scope is unfiled — the drive's exit code reads this,
            never its log (a drive whose legs failed otherwise exits 0 and is recorded complete).

Every refusal is `REFUSED: <why>` on stderr, exit 2; everything else is one line per run. Nothing here names a host or a user.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import socket
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
from sisteer.registry import emit, identity  # noqa: E402
from sisteer.registry import records as R  # noqa: E402
from sisteer.registry.records import utc_now  # noqa: E402

SCRIPT = "sdfbase.full22.petri_registry_handoff"
IDENTITY_ENV = "SISTEER_REGISTRY_RUN_IDENTITY"
SPOOL_ENV = emit.SPOOL_ENV
POOLED_EVAL = "petri_pooled"
PETRI_TASK_PREFIX = "petri/"


class Refused(SystemExit):
    def __init__(self, msg: str):
        print(f"REFUSED: {msg}", file=sys.stderr, flush=True)
        super().__init__(2)


def _load(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as exc:
        raise Refused(f"cannot read {path}: {exc}")
    if not isinstance(doc, dict):
        raise Refused(f"{path}: not a JSON object")
    return doc


def _runs(doc: dict, path: str) -> List[dict]:
    runs = doc.get("runs")
    if not isinstance(runs, list) or not runs:
        raise Refused(f"{path}: the pending document names no `runs` (the proxy job registered nothing — f22_registry_register's "
                      "[registry] lines in its log say why); the drive cannot hand its stores to unregistered runs")
    out = []
    for r in runs:
        if not isinstance(r, dict) or not r.get("run_id") or not isinstance(r.get("route"), dict) or not isinstance(r.get("plan"), dict):
            raise Refused(f"{path}: a `runs` entry lacks run_id / route / plan: {json.dumps(r)[:160]}")
        out.append(r)
    return out


# ------------------------------------------------------------------------------------------------------------ import
def harness_object(entry: str) -> str:
    """The Petri harness cell id of a wave config cell entry — registry_adapter.harness_cell_id (imported lazily: the adapter
    pulls the train/test defense-cell vocabulary). Refuses by name on an entry the harness cannot spell."""
    from sisteer.capeval.petri import registry_adapter as ra
    try:
        return ra.harness_cell_id(entry)
    except Exception as exc:  # noqa: BLE001 — RegistryAdapterError and the grammar's refusals alike: named, never dropped
        raise Refused(f"wave config cell {entry!r} has no Petri harness cell id ({type(exc).__name__}: {exc}) — the proxy registered a run "
                      "for an object the Petri store could never name; fix the registration / the roster, never route around it")


def resolve_spool(explicit: Optional[str]) -> str:
    spool = explicit or emit.spool_root()
    if not spool:
        raise Refused(f"no registry spool resolves on this host ({SPOOL_ENV} unset, ~/registry absent) — the drive is the producer of these "
                      "runs' stores and must write their records somewhere the harvester folds; provision the grading host's registry "
                      "(scripts/registry/install_grading_host.sh) or set the variable")
    return spool


def cmd_import(a) -> int:
    doc = _load(a.pending)
    if str(doc.get("job_id") or "") != str(a.job):
        raise Refused(f"{a.pending}: pending document is job {doc.get('job_id')!r}, the drive's proxy is job {a.job!r} — not this proxy's identities")
    runs = _runs(doc, a.pending)
    translated = []
    seen_objects: Dict[str, str] = {}
    for r in runs:
        entry = str(r["route"].get("object") or r["plan"].get("producer_object_key_legacy") or "")
        if not entry:
            raise Refused(f"{a.pending}: run {r['run_id']} names no route.object")
        if entry not in seen_objects:
            seen_objects[entry] = harness_object(entry)
        route = dict(r["route"], object=seen_objects[entry], object_wave=entry)
        translated.append({**r, "route": route})
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{a.job}.json"
    ident = {**doc, "runs": translated, "imported_from": os.path.abspath(a.pending), "imported_utc": utc_now(),
             "imported_by": SCRIPT, "drive_host": socket.gethostname(),
             "object_translation": {k: v for k, v in seen_objects.items()}}
    tmp = str(out) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(ident, fh, indent=1, sort_keys=True)
    os.replace(tmp, out)
    spool = resolve_spool(a.spool or os.environ.get(SPOOL_ENV))
    evals = sorted({str(r["route"].get("eval_id") or r["plan"].get("eval_id")) for r in translated})
    if a.sh:
        print(f"export {IDENTITY_ENV}={_shq(str(out))}")
        print(f"export {SPOOL_ENV}={_shq(spool)}")
    else:
        print(f"identity {out}")
        print(f"spool {spool}")
    print(f"# petri registry hand-off: job {a.job}: {len(translated)} run(s) over {len(seen_objects)} object(s) x {len(evals)} eval(s); objects "
          + ", ".join(f"{k}->{v}" for k, v in seen_objects.items()), file=sys.stderr)
    return 0


def _shq(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


# --------------------------------------------------------------------------------------------------------- close-leg
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _generated_already(spool: str, campaign: str, run_id: str) -> bool:
    """Does the run's record directory under the spool already hold a `run_state generated` (the store recorder's on_results)?"""
    d = os.path.join(spool, campaign, identity.run_dir_name(run_id))
    if not os.path.isdir(d):
        return False
    for fn in sorted(os.listdir(d)):
        m = R.RECORD_FILE_RE.match(fn)
        if not m or m.group("type") != "run_state":
            continue
        try:
            for _i, rec, _sha in R.read_record_file(os.path.join(d, fn), compute_sha=False):
                p = rec.get("payload") if isinstance(rec, dict) else None
                if isinstance(p, dict) and p.get("run_id") == run_id and p.get("state") == "generated":
                    return True
        except Exception:  # noqa: BLE001 — an unreadable record is not evidence
            continue
    return False


def _recorded_generated_total(spool: str, campaign: str, run_id: str) -> Optional[int]:
    """The LAST `n_items_generated` a run_patch in the run's record directory under the spool carries (the store recorder's
    completion patch, or an earlier close-leg's), by record-file order then record order; None when none states one."""
    d = os.path.join(spool, campaign, identity.run_dir_name(run_id))
    if not os.path.isdir(d):
        return None
    last: Optional[int] = None
    for fn in sorted(os.listdir(d)):
        m = R.RECORD_FILE_RE.match(fn)
        if not m or m.group("type") != "run_patch":
            continue
        try:
            for _i, rec, _sha in R.read_record_file(os.path.join(d, fn), compute_sha=False):
                p = rec.get("payload") if isinstance(rec, dict) else None
                f = (p or {}).get("fields") if isinstance(p, dict) else None
                if isinstance(f, dict) and p.get("run_id") == run_id and f.get("n_items_generated") is not None:
                    last = int(f["n_items_generated"])
        except Exception:  # noqa: BLE001 — an unreadable record is not evidence
            continue
    return last


def _rows_snapshot(rows_path: str) -> Tuple[bytes, str]:
    """rows.jsonl read ONCE: ``(bytes, sha256)`` of the same snapshot, so a count and a hash taken from it describe the same bytes
    (a second read under a concurrent writer does not prove the counted bytes). ``(b"", sha256(b""))`` when absent."""
    if not os.path.isfile(rows_path):
        data = b""
    else:
        with open(rows_path, "rb") as fh:
            data = fh.read()
    return data, hashlib.sha256(data).hexdigest()


def _epochs_of(results: Mapping[str, Any]) -> Optional[int]:
    """The manifest-owned epoch count the cell was audited under -- results.json's top-level ``epochs`` (written by the runner); for
    records written before that field existed, derived from the behaviours block (``n_items`` = scenarios x epochs, one value across
    behaviours); None when neither derives (the caller refuses by name: the item-id format cannot be known)."""
    if not isinstance(results, Mapping):
        return None
    e = results.get("epochs")
    if isinstance(e, int) and e >= 1:
        return e
    bs = results.get("behaviors")
    derived: set = set()
    if isinstance(bs, Mapping):
        for b in bs.values():
            if (isinstance(b, Mapping) and isinstance(b.get("n_items"), int) and isinstance(b.get("n_scenarios"), int)
                    and b["n_scenarios"] > 0 and b["n_items"] % b["n_scenarios"] == 0):
                derived.add(b["n_items"] // b["n_scenarios"])
    return derived.pop() if len(derived) == 1 else None


def _refusal_item_id(scenario_id: str, epoch: int, epochs: int) -> str:
    """The runner's row item id for a (scenario, epoch): the bare scenario at one epoch, ``scenario#eN`` otherwise
    (petri_runner._item_id)."""
    return scenario_id if epochs == 1 else f"{scenario_id}#e{int(epoch)}"


def _terminal_judge_refused_items(cell_dir: str, epochs: int) -> set:
    """(behavior, item_id) pairs petri_judge_refused.json files as TERMINAL judge refusals, each refusal RECORD normalised to the runner's
    item id for its epoch: transcripts the audit GENERATED and the judge declined to score. They are
    generated items that carry no rows until a judge-only re-score scores them -- counting rows alone undercounted n_items_generated
    by exactly these (9/7/59/57 against 10/10/60/60 on the u0 prompted cells) and the re-score then failed the coverage check. Two
    refused epochs of one scenario are two generated items. An AUDITOR refusal produced no transcript and is not generated."""
    out: set = set()
    p = os.path.join(cell_dir, "petri_judge_refused.json")
    if not os.path.isfile(p):
        return out
    try:
        doc = _load(p)
    except Refused:
        return out
    for rec in doc.get("judge_refused") or []:
        if isinstance(rec, dict) and rec.get("behavior") and rec.get("scenario_id"):
            out.add((str(rec["behavior"]), _refusal_item_id(str(rec["scenario_id"]), int(rec.get("epoch") or 1), epochs)))
    return out


def _items_generated_in(data: bytes, behavior: Optional[str], judge_refused: Iterable[Tuple[str, str]] = ()) -> int:
    """Distinct Petri item ids GENERATED in a rows.jsonl snapshot — of one behaviour's task, or of every `petri/*` task (the pooled run):
    the items with rows PLUS the cell's terminal judge-refused ITEMS of that behaviour (``judge_refused``: (behavior, item_id) pairs in
    the runner's item-id format, from :func:`_terminal_judge_refused_items`), so a judge refusal never shrinks
    the count and a later re-score of that item never exceeds it."""
    items = set()
    for b, item in judge_refused:
        if behavior and b != behavior:
            continue
        items.add((PETRI_TASK_PREFIX + str(b), str(item)))
    for line in data.decode("utf-8", errors="replace").split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        task = str(rec.get("task", ""))
        if not task.startswith(PETRI_TASK_PREFIX) or rec.get("item_id") in (None, "", "__aggregate__"):
            continue
        if behavior and task != PETRI_TASK_PREFIX + behavior:
            continue
        items.add((task, str(rec["item_id"])))
    return len(items)


def _items_generated(rows_path: str, behavior: Optional[str], cell_dir: Optional[str] = None, epochs: int = 1) -> int:
    """Distinct Petri item ids generated — rows.jsonl's items of one behaviour's task (or of every `petri/*` task, the pooled run) plus
    the cell's terminal judge-refused items when ``cell_dir`` is given (``epochs`` keys the item-id format)."""
    return _items_generated_in(_rows_snapshot(rows_path)[0], behavior, _terminal_judge_refused_items(cell_dir, epochs) if cell_dir else ())


#: the close-leg's standing refusals, beside the drive's identity file: ONE immutable file per run under
#: close_leg_refusals/<run dir name>.json ({run_id, cell_id, eval_id, recorded, store_total, why, rows_sha256, utc}) — written by the
#: refusing leg through a unique temp name + os.replace, CLEARED by removing that run's file only (a later close-leg that recounts or
#: agrees), READ by `verify` as a directory (a run with a standing refusal is UNFILED, whatever its spool says). Per-run files because
#: the drive runs distinct legs CONCURRENTLY under one identity (a shared read-modify-replace with one .tmp name lost a concurrent
#: leg's refusal); a run belongs to one cell and so to one leg, so no two legs ever write the same run's file.
REFUSALS_DIR = "close_leg_refusals"
#: the (E3) shape: ONE json map {run_id: refusal} beside the identity — migrated into per-run files on read,
#: then renamed to a `.migrated` sibling; never read as the standing set, never deleted
LEGACY_REFUSALS_NAME = "close_leg_refusals.json"
REQUIRED_DIMS = ("eval_awareness", "scenario_realism")


def _refusals_path(identity_path: str) -> str:
    """The refusals DIRECTORY beside the identity file."""
    return os.path.join(os.path.dirname(os.path.abspath(identity_path)), REFUSALS_DIR)


def _refusal_file(identity_path: str, run_id: str) -> str:
    return os.path.join(_refusals_path(identity_path), identity.run_dir_name(run_id) + ".json")


#: a clear's durable record: absence is never the signal — migration never re-imports a legacy entry over a
#: tombstone at least as new as the entry; a FRESH refusal by close-leg removes the tombstone
TOMBSTONE_SUFFIX = ".cleared.json"
LOCK_NAME = "close_leg_refusals.lock"


def _tombstone_file(identity_path: str, run_id: str) -> str:
    return os.path.join(_refusals_path(identity_path), identity.run_dir_name(run_id) + TOMBSTONE_SUFFIX)


@contextlib.contextmanager
def _refusals_lock(identity_path: str):
    """ONE short per-identity flock (a file beside the identity) around every migration, per-run write and clear: the drive's legs and
    the verifier run concurrently on the grading host, and a migration that read the legacy map before a concurrent clear must not write its
    stale snapshot after it. Not reentrant: callers never nest it."""
    path = os.path.join(os.path.dirname(os.path.abspath(identity_path)), LOCK_NAME)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _write_json_unique(final: str, doc: Mapping[str, object]) -> None:
    """mkstemp in the target directory + os.replace: never a shared temp name."""
    d = os.path.dirname(final)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(final) + ".", suffix=".tmp", dir=d)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(dict(doc), fh, indent=1, sort_keys=True); fh.write("\n")
    os.replace(tmp, final)


def _load_optional(path: str) -> Optional[dict]:
    return _load(path) if os.path.isfile(path) else None


def _import_legacy_map(identity_path: str, doc: Mapping[str, object], legacy: str) -> int:
    """The write half of a legacy migration, for a map already LOADED (the lock must be held): one per-run file per entry, skipping a
    run whose per-run file stands and a run whose tombstone is at least as new as the entry (a clear after the map was written —
    the stale-snapshot case). Returns the number carried."""
    carried = 0
    for rid, rec in doc.items():
        if not isinstance(rec, dict) or not str(rid):
            continue
        rid = str(rid)
        if os.path.isfile(_refusal_file(identity_path, rid)):
            continue
        tomb = _load_optional(_tombstone_file(identity_path, rid))
        if tomb is not None and str(tomb.get("cleared_utc") or "") >= str(rec.get("utc") or ""):
            continue                                     # cleared since: the tombstone, not absence, says so
        _write_json_unique(_refusal_file(identity_path, rid), {**rec, "run_id": rid, "migrated_from": legacy, "migrated_utc": utc_now()})
        carried += 1
    return carried


def _legacy_watermark(identity_path: str, legacy_sha: str) -> str:
    return os.path.join(_refusals_path(identity_path), f".legacy-{legacy_sha}.migrated")


def _migrate_legacy_refusals(identity_path: str) -> int:
    """The (E3) single-file map beside the identity, carried into per-run files: one mkstemp+replace per
    refusal — an existing per-run file WINS, so a refusal already cleared or re-filed by a later close-leg is not resurrected and a
    standing one is preserved until an explicit recount/agree clears it — then the legacy file is renamed to a `.migrated` sibling
    (a stamped name when one already exists); never deleted, never read as the standing set. Idempotent; a lost rename race between
    concurrent legs is ignored (the winner's rename is the migration). Returns the number of refusals carried over."""
    legacy = os.path.join(os.path.dirname(os.path.abspath(identity_path)), LEGACY_REFUSALS_NAME)
    if not os.path.isfile(legacy):
        return 0
    with _refusals_lock(identity_path):
        if not os.path.isfile(legacy):
            return 0                                 # a concurrent leg migrated it while we waited for the lock
        with open(legacy, "rb") as fh:
            raw = fh.read()
        legacy_sha = hashlib.sha256(raw).hexdigest()
        try:
            doc = json.loads(raw.decode("utf-8"))
        except ValueError as exc:
            raise Refused(f"cannot read {legacy}: {exc}")
        if not isinstance(doc, dict):
            raise Refused(f"{legacy}: not a JSON object")
        mark = _legacy_watermark(identity_path, legacy_sha)
        if os.path.isfile(mark):
            carried = 0                              # this exact map was migrated before: a copy re-appearing imports nothing
            why = "already migrated (watermark)"
        else:
            carried = _import_legacy_map(identity_path, doc, legacy)
            _write_json_unique(mark, {"legacy_sha256": legacy_sha, "migrated_utc": utc_now(), "entries": len(doc), "carried": carried})
            why = "carried into per-run files"
        target = legacy + ".migrated"
        if os.path.exists(target):
            target = f"{legacy}.migrated-{utc_now().replace(':', '').replace('-', '')}"
        try:
            os.replace(legacy, target)
        except FileNotFoundError:
            pass
    print(f"close-leg refusals: migrated {carried} standing refusal(s) from the legacy {legacy} into {_refusals_path(identity_path)}/ "
          f"({why}; legacy renamed to {os.path.basename(target)}; never deleted)", file=sys.stderr, flush=True)
    return carried


def _read_refusals(identity_path: str) -> Dict[str, dict]:
    """{run_id: refusal} over every per-run file in the directory ({} without one), after carrying a legacy single-file map into
    it (:func:`_migrate_legacy_refusals`); an unreadable file is a refusal, never a silent gap."""
    _migrate_legacy_refusals(identity_path)
    d = _refusals_path(identity_path)
    if not os.path.isdir(d):
        return {}
    out: Dict[str, dict] = {}
    with _refusals_lock(identity_path):
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json") or fn.endswith(TOMBSTONE_SUFFIX) or fn.startswith("."):
                continue
            rec = _load(os.path.join(d, fn))
            rid = str(rec.get("run_id") or "")
            if not rid:
                raise Refused(f"{os.path.join(d, fn)}: a close-leg refusal file names no run_id")
            out[rid] = dict(rec)
    return out


def _write_refusal(identity_path: str, run_id: str, rec: Mapping[str, object]) -> str:
    """Write ONE run's refusal file atomically under the identity lock (a unique temp name, then os.replace); a FRESH refusal
    supersedes an earlier clear, so the run's tombstone goes."""
    final = _refusal_file(identity_path, run_id)
    with _refusals_lock(identity_path):
        _write_json_unique(final, {**dict(rec), "run_id": run_id})
        try:
            os.remove(_tombstone_file(identity_path, run_id))
        except FileNotFoundError:
            pass
    return final


def _clear_refusal(identity_path: str, run_id: str, *, rows_sha256: str, cleared_by: str) -> bool:
    """Clear ONE run's refusal under the identity lock: write its durable tombstone (the refusal it cleared, the agreeing
    rows_sha256, who cleared it, when) and remove the refusal file. The tombstone is written whether or not a refusal stood, so a
    later legacy copy can never re-import one for this run over the clear. True when a refusal stood."""
    with _refusals_lock(identity_path):
        stood = _load_optional(_refusal_file(identity_path, run_id))
        _write_json_unique(_tombstone_file(identity_path, run_id),
                           {"run_id": run_id, "cleared_utc": utc_now(), "agreeing_rows_sha256": rows_sha256, "cleared_by": cleared_by,
                            "cleared_refusal": stood})
        try:
            os.remove(_refusal_file(identity_path, run_id))
            return True
        except FileNotFoundError:
            return stood is not None


def _terminal_refused_scenarios(cell_dir: str) -> set:
    """(behavior, scenario_id, epoch) triples the runner filed as TERMINAL — petri_auditor_refused.json (`auditor_refused`) and
    petri_judge_refused.json (`judge_refused`) — the items results.json counts outside `audited.items`. One triple per refusal
    RECORD: two refused epochs of one scenario are two items."""
    out: set = set()
    for name, key in (("petri_auditor_refused.json", "auditor_refused"), ("petri_judge_refused.json", "judge_refused")):
        p = os.path.join(cell_dir, name)
        if not os.path.isfile(p):
            continue
        try:
            doc = _load(p)
        except Refused:
            continue
        for rec in doc.get(key) or []:
            if isinstance(rec, dict) and rec.get("behavior") and rec.get("scenario_id"):
                out.add((str(rec["behavior"]), str(rec["scenario_id"]), int(rec.get("epoch") or 1)))
    return out


def _store_complete_now(cell_dir: str, rows_data: bytes, results: Mapping[str, Any]) -> Tuple[bool, str]:
    """A completion proof BOUND to the current rows.jsonl snapshot: results.json's audited footing,
    re-verified against the snapshot — the scenarios scored with EVERY required dimension equal `audited.items`, and scored plus the
    sidecars' terminal refusals equal `audited.expected`. A results.json without the footing, or a snapshot that no longer bears it
    out (rows lost after completion), is no proof: ``(False, why)``."""
    audited = results.get("audited") if isinstance(results, Mapping) else None
    if not isinstance(audited, Mapping) or not isinstance(audited.get("items"), int) or not isinstance(audited.get("expected"), int):
        return False, "results.json carries no `audited` footing (items / expected) to bind a completion proof to the rows"
    metrics: Dict[Tuple[str, str], set] = {}
    for line in rows_data.decode("utf-8", errors="replace").split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        task = str(rec.get("task", ""))
        if not task.startswith(PETRI_TASK_PREFIX) or rec.get("item_id") in (None, "", "__aggregate__"):
            continue
        metrics.setdefault((task, str(rec["item_id"])), set()).add(str(rec.get("metric", "")))
    scored = sum(1 for (task, _i), ms in metrics.items() if {task[len(PETRI_TASK_PREFIX):], *REQUIRED_DIMS} <= ms)
    refused = len(_terminal_refused_scenarios(cell_dir))
    ok = scored == int(audited["items"]) and scored + refused == int(audited["expected"])
    why = (f"snapshot: {scored} scenario(s) scored with every required dimension + {refused} terminal refusal(s) vs results.json audited "
           f"items {audited['items']} / expected {audited['expected']}")
    return ok, why


def _ledger_settled(cell_dir: str) -> Optional[bool]:
    """Is the cell's failure ledger (petri_incomplete.json, the runner's H9 sidecar) SETTLED — present with `unresolved: []`? None when
    absent or unreadable (no evidence either way)."""
    p = os.path.join(cell_dir, "petri_incomplete.json")
    if not os.path.isfile(p):
        return None
    try:
        led = _load(p)
    except Exception:  # noqa: BLE001 — unreadable is no evidence
        return None
    unresolved = led.get("unresolved") if isinstance(led, dict) else None
    return isinstance(unresolved, list) and len(unresolved) == 0


def cmd_close_leg(a) -> int:
    ident = _load(a.identity)
    runs = _runs(ident, a.identity)
    cell_dir = os.path.abspath(a.cell_dir)
    man_p = os.path.join(cell_dir, "manifest.json"); res_p = os.path.join(cell_dir, "results.json"); rows_p = os.path.join(cell_dir, "rows.jsonl")
    if not os.path.isfile(man_p):
        raise Refused(f"{cell_dir}: no manifest.json — not a Petri cell store")
    man = _load(man_p)
    cell_id = str(man.get("cell_id") or man.get("arm_name") or "")
    if not cell_id:
        raise Refused(f"{man_p}: names no cell_id")
    mine = [r for r in runs if str(r["route"].get("object")) == cell_id]
    if not mine:
        print(f"close-leg {cell_id}: no registered run names this cell (a control leg such as none__lora0 is not a registered object) — nothing filed")
        return 0
    if not os.path.isfile(res_p):
        print(f"close-leg {cell_id}: INCOMPLETE — no results.json (the runner's cell-done marker); {len(mine)} run(s) stay running; nothing filed")
        return 0
    spool = resolve_spool(a.spool or os.environ.get(SPOOL_ENV))
    producer = emit.producer_block(SCRIPT, code_sha=a.code_sha)
    res_sha = _sha256(res_p)
    now = utc_now()
    filed = skipped = 0
    recounted = refused = 0
    rows_data, rows_sha = _rows_snapshot(rows_p)          # ONE read: the counts and the hash below describe these bytes
    settled = _ledger_settled(cell_dir)
    results = _load(res_p)
    proof_ok, proof_why = _store_complete_now(cell_dir, rows_data, results)
    epochs = _epochs_of(results)
    if epochs is None:
        # the item-id format depends on the manifest's epoch count; a record that states none (and whose behaviours block
        # does not derive one value) cannot be counted -- refused by name, nothing filed
        print(f"close-leg {cell_id}: NOT filed -- results.json carries no `epochs` and its behaviours block derives no single epoch count "
              "(n_items / n_scenarios); the generated-item count depends on the runner's item-id format (bare scenario at one epoch, "
              "scenario#eN otherwise); re-run the leg on a runner that stamps `epochs`, then close again; exit 3", file=sys.stderr, flush=True)
        return 3
    judge_refused = _terminal_judge_refused_items(cell_dir, epochs)   # generated transcripts the judge declined: counted as generated, per epoch
    refused_now: List[str] = []
    refusal_of: Dict[str, dict] = {}
    _migrate_legacy_refusals(a.identity)             # a legacy (E3) map is carried into per-run files BEFORE any clear below can miss it
    for r in mine:
        rid = str(r["run_id"]); plan = r["plan"]
        campaign = str(plan.get("campaign") or identity.parse_run_id(rid)["campaign"])
        eval_id = str(r["route"].get("eval_id") or plan.get("eval_id") or "")
        behavior = None if eval_id == POOLED_EVAL else (eval_id[len("petri_"):] if eval_id.startswith("petri_") else eval_id)
        if _generated_already(spool, campaign, rid):
            # a `generated` run is left alone ONLY while its recorded total agrees with the store. Corrected here,
            # with provenance (the field is the run's store total at the hand-off): a MISSING total (the recorder writes
            # `run_state generated` before its n_items_generated patch, in separate files — a crash or partial push between them
            # leaves the state without the number) and a stale INCREMENT (an older recorder) alike.
            # A DECREASE (the store holds fewer items than recorded: truncated, partially restored) is never an ordinary fix: it
            # is filed under its own direction and cause ONLY when the store is complete by its own evidence — results.json
            # present (this branch) AND the failure ledger settled — and refused by name otherwise.
            total = _items_generated_in(rows_data, behavior, judge_refused)
            n_refused_counted = total - _items_generated_in(rows_data, behavior)      # the terminal judge refusals without rows, counted in
            recorded = _recorded_generated_total(spool, campaign, rid)
            if recorded is None or recorded != total:
                if recorded is not None and total < recorded and (settled is not True or not proof_ok):
                    # results.json and a settled ledger are markers of a PAST completion, not of these
                    # rows — a decrease needs the completion proof re-verified against the snapshot, else lost rows would be filed
                    # as the store total; refused by name, persisted, and the exit says so
                    why = (f"the store holds fewer items than recorded ({total} < {recorded}) and no completion proof binds to the current "
                           f"rows: failure ledger {'settled' if settled else ('absent/unreadable' if settled is None else 'NOT settled')}; "
                           f"{proof_why} ({'proof holds' if proof_ok else 'PROOF FAILS'}) — a truncated or partially restored store is not "
                           "a recount; restore the rows or re-run the leg, then close again (D784 E2/E3)")
                    refused += 1; refused_now.append(rid)
                    refusal_of[rid] = {"cell_id": cell_id, "eval_id": eval_id, "recorded": recorded, "store_total": total, "why": why,
                                       "rows_sha256": rows_sha, "utc": now}
                    _write_refusal(a.identity, rid, refusal_of[rid])        # this run's own file, atomically; no other leg writes it
                    print(f"close-leg {cell_id}: {eval_id} {rid[:40]}… REFUSED to lower n_items_generated {recorded} -> {total}: {why}")
                    continue
                if recorded is None:
                    direction, cause = "missing", ("the run reached `generated` with no n_items_generated recorded (the recorder's state "
                                                   "and total ride separate files; a crash or partial push between them)")
                elif total > recorded and n_refused_counted and recorded + n_refused_counted == total:
                    direction, cause = "increase", (f"{n_refused_counted} terminal judge refusal(s) are generated transcripts the judge declined and "
                                                    "count as generated (D1212/D1219); the recorded total counted rows alone")
                elif total > recorded:
                    direction, cause = "increase", "the store recorder wrote a resumed pass's increment (hooks before D784)"
                else:
                    direction, cause = "decrease", ("the store holds fewer items than recorded and the completion proof binds to these rows "
                                                    f"({proof_why}; failure ledger settled), so the store total stands")
                w = emit.RecordWriter(spool, campaign, rid, producer, push_script=a.push_script or os.environ.get(emit.PUSH_SCRIPT_ENV))
                w.emit_patch(fields={
                    "n_items_generated": total, "store_uri": cell_dir, "store_completion_artifact_uri": res_p,
                    "store_completion_sha256": res_sha,
                    "extra_json": {"n_items_generated_recount": {
                        "previous": recorded, "store_total": total, "direction": direction, "cause": cause,
                        "ledger_settled": settled, "completion_proof": {"holds": proof_ok, "evidence": proof_why},
                        "rows_sha256": rows_sha, "recounted_utc": now,
                        "by": f"{SCRIPT} close-leg", "why": "D784 (E/E2): the field is the run's store total at the hand-off (D769)"},
                        "arm_name_legacy": cell_id, "engine_legacy": str(man.get("engine") or "")}})
                w.push(final=True)
                recounted += 1
                _clear_refusal(a.identity, rid, rows_sha256=rows_sha, cleared_by=f"{SCRIPT} close-leg recount {direction}")   # tombstoned
                print(f"close-leg {cell_id}: {eval_id} {rid[:40]}… n_items_generated {recorded} -> {total} ({direction}: {cause}; recounted, D784)")
                continue
            skipped += 1
            _clear_refusal(a.identity, rid, rows_sha256=rows_sha, cleared_by=f"{SCRIPT} close-leg agree")   # agreement clears it, tombstoned
            print(f"close-leg {cell_id}: {eval_id} {rid[:40]}… already generated (the store recorder filed it; total {recorded} agrees with the store) — left alone")
            continue
        w = emit.RecordWriter(spool, campaign, rid, producer, push_script=a.push_script or os.environ.get(emit.PUSH_SCRIPT_ENV))
        fields = {"store_uri": cell_dir, "store_completion_artifact_uri": res_p, "store_completion_sha256": res_sha,
                  "n_items_generated": _items_generated_in(rows_data, behavior, judge_refused), "generated_utc": now}
        w.emit_patch(fields=fields)
        w.emit_state("generated", "completion_artifact", res_p, event_utc=now,
                     note=f"filed by the drive's registry hand-off ({SCRIPT} close-leg): the Petri per-cell store on the grading host")
        w.push(final=True)
        filed += 1
        print(f"close-leg {cell_id}: {eval_id} {rid[:40]}… store_uri={cell_dir} generated")
    print(f"close-leg {cell_id}: {filed} run(s) filed, {skipped} already generated, {recounted} recounted, {refused} refused, of {len(mine)} registered for this cell")
    if refused:
        # a refusal is the cell's terminal outcome, not a stdout line — exit 3, the runs named, persisted for verify
        named = "; ".join(f"{refusal_of[r]['eval_id']} {r[:40]}…: {refusal_of[r]['why'][:160]}" for r in refused_now)
        print(f"close-leg {cell_id}: NOT filed complete — {refused} refusal(s) persisted in {_refusals_path(a.identity)} ({named}); exit 3",
              file=sys.stderr, flush=True)
        return 3
    return 0


# ------------------------------------------------------------------------------------------------------------ finish
def cmd_finish(a) -> int:
    ident = _load(a.identity)
    runs = _runs(ident, a.identity)
    spool = resolve_spool(a.spool or os.environ.get(SPOOL_ENV))
    push_script = a.push_script or os.environ.get(emit.PUSH_SCRIPT_ENV)
    producer = emit.producer_block(SCRIPT, code_sha=a.code_sha)
    n = 0
    for r in runs:
        rid = str(r["run_id"]); campaign = str(r["plan"].get("campaign") or identity.parse_run_id(rid)["campaign"])
        w = emit.RecordWriter(spool, campaign, rid, producer, push_script=push_script)
        if w.dir and os.path.isdir(w.dir):
            w.push(final=True); n += 1
    print(f"finish: {n} run record dir(s) under {spool} pushed final" + ("" if push_script else " (no push script: the spool is the harvester's records dir)"))
    return 0


def _filed_state(spool: str, campaign: str, run_id: str) -> Tuple[Optional[str], Optional[str]]:
    """(the run's LATEST `run_state` state, its `store_uri` from the run_patch records) in its record directory under the spool; (None,
    None) with no directory. The same files `_generated_already` reads; record order = file order (the writer's sequence)."""
    d = os.path.join(spool, campaign, identity.run_dir_name(run_id))
    if not os.path.isdir(d):
        return None, None
    state = store = None
    for fn in sorted(os.listdir(d)):
        m = R.RECORD_FILE_RE.match(fn)
        if not m or m.group("type") not in ("run_state", "run_patch"):
            continue
        try:
            for _i, rec, _sha in R.read_record_file(os.path.join(d, fn), compute_sha=False):
                p = rec.get("payload") if isinstance(rec, dict) else None
                if not isinstance(p, dict) or p.get("run_id") != run_id:
                    continue
                if m.group("type") == "run_state" and p.get("state"):
                    state = str(p["state"])
                fields = p.get("fields") if m.group("type") == "run_patch" else None
                if isinstance(fields, dict) and fields.get("store_uri"):
                    store = str(fields["store_uri"])
        except Exception:  # noqa: BLE001 — an unreadable record is not evidence
            continue
    return state, store


def cmd_verify(a) -> int:
    ident = _load(a.identity)
    runs = _runs(ident, a.identity)
    spool = resolve_spool(a.spool or os.environ.get(SPOOL_ENV))
    skipped = {x.strip() for x in (a.skipped or "").split(",") if x.strip()}
    only = {x.strip() for x in (a.objects or "").split(",") if x.strip()}
    rows: List[dict] = []
    n_filed = n_unfiled = n_skipped = n_out = 0
    standing_refusals = _read_refusals(a.identity)        # a run close-leg refused stays UNFILED until a later close-leg clears it
    for r in runs:
        rid = str(r["run_id"]); plan = r["plan"]; route = r["route"]
        campaign = str(plan.get("campaign") or identity.parse_run_id(rid)["campaign"])
        eval_id = str(route.get("eval_id") or plan.get("eval_id") or "")
        objs = {str(route.get("object") or ""), str(route.get("object_wave") or "")} - {""}
        row = {"run_id": rid, "eval_id": eval_id, "object": str(route.get("object") or ""), "object_wave": str(route.get("object_wave") or route.get("object") or ""),
               "state": None, "store_uri": None, "filed": False}
        if only and not (objs & only):
            row["scope"] = "out"; n_out += 1; rows.append(row); continue
        if objs & skipped:
            row["scope"] = "skipped"; n_skipped += 1; rows.append(row)
            print(f"verify {eval_id} {row['object']} {rid[:40]}… skipped by the standup (never filed, by design)")
            continue
        state, store = _filed_state(spool, campaign, rid)
        refusal = standing_refusals.get(rid)
        row.update({"scope": "in", "state": state, "store_uri": store, "filed": state == "generated" and bool(store) and refusal is None,
                    "close_leg_refusal": refusal})
        rows.append(row)
        if row["filed"]:
            n_filed += 1
        else:
            n_unfiled += 1
        tail = f"REFUSED by close-leg ({refusal['why'][:140]})" if refusal else ("filed" if row["filed"] else "UNFILED")
        print(f"verify {eval_id} {row['object']} {rid[:40]}… state={state} store_uri={store} {tail}")
    n_planned = n_filed + n_unfiled
    summary = {"identity": a.identity, "spool": spool, "n_registered": len(runs), "n_planned": n_planned, "n_filed": n_filed, "n_unfiled": n_unfiled,
               "n_skipped": n_skipped, "n_out_of_scope": n_out, "runs": rows, "utc": utc_now()}
    if a.summary:
        os.makedirs(os.path.dirname(os.path.abspath(a.summary)), exist_ok=True)
        tmp = a.summary + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True); fh.write("\n")
        os.replace(tmp, a.summary)
    print(f"verify: {n_filed} of {n_planned} planned run(s) filed under {spool}" + (f", {n_skipped} skipped by the standup" if n_skipped else "")
          + (f", {n_out} outside the selected objects" if n_out else "") + (f"; summary {a.summary}" if a.summary else ""))
    if n_unfiled:
        n_ref = sum(1 for x in rows if x.get("close_leg_refusal"))
        raise Refused(f"{n_unfiled} of {n_planned} planned run(s) are NOT filed (no `generated` state or no store_uri on this host's spool"
                      + (f"; {n_ref} with a standing close-leg refusal in {_refusals_path(a.identity)}" if n_ref else "") + ") — the audit is not complete")
    if n_planned == 0:
        raise Refused("no planned run is in scope (every run skipped or outside --objects) — nothing to verify is not a complete audit")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spool", default=None, help=f"the spool to write (default ${SPOOL_ENV}, else the host's resolved spool)")
    ap.add_argument("--push-script", default=None); ap.add_argument("--code-sha", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("import"); p.add_argument("--pending", required=True); p.add_argument("--job", required=True)
    p.add_argument("--out-dir", required=True); p.add_argument("--sh", action="store_true")
    p = sub.add_parser("close-leg"); p.add_argument("--identity", required=True); p.add_argument("--cell-dir", required=True)
    p = sub.add_parser("finish"); p.add_argument("--identity", required=True)
    p = sub.add_parser("verify"); p.add_argument("--identity", required=True); p.add_argument("--summary", default=None)
    p.add_argument("--skipped", default=None, help="comma list of objects the standup skipped by name (their runs are never filed)")
    p.add_argument("--objects", default=None, help="comma list of objects to verify (default every planned run)")
    a = ap.parse_args(argv)
    return {"import": cmd_import, "close-leg": cmd_close_leg, "finish": cmd_finish, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Refused as exc:
        sys.exit(exc.code)
