"""Registry filing for a judge re-run: a NEW ``judge`` row, NEW ``judgment`` rows and ``item_result`` rows, written as RECORD FILES
through the registry's own writer and validator and delivered to the harvester's INBOX -- never a database write.

What is filed (the Sonnet regrade twin's precedent, ``adapters/regrade_twin.py``: a second judge of a run is a second ``judgment`` row
and its own ``item_result`` rows, which ``v_judge_side_by_side`` pairs with the first judge's per (run, item, trial, sample, turn)):
  judge         one record per (judge model, rubric, protocol): ``judge_id = identity.judge_id(judge_model, rubric_sha256,
                protocol_sha256, threshold_rule)``, so it can never equal an existing judge's id
  judgment      one per run, PLAIN form ``judgment_id = identity.judgment_id(run_id, judge_id)`` (the form every adapter mints), status
                ``complete`` (every unit of the run has a terminal outcome) or ``partial``; counts, the OpenRouter batch ids, the code sha
  item_result   the track's rows, each carrying the new ``judgment_id`` in its key, ``source_revision_id = ''`` (natively emitted)
What is NOT filed, on purpose: no ``run_state``, no ``run_patch`` (``hooks.JudgingScope.complete`` would patch ``run.judge_id`` -- the
run's PRIMARY judge, which a re-judge never changes), no ``run_metric`` / ``run_contrast`` (the paper frame reads metrics; the Sonnet
twin filed none either), no ``superseded`` / ``withdrawal`` record of anything. A judgment of a judge that is not the run's required
judge is not what any paper view selects; the scratch proof (``README.md``) diffs every pre-existing row and every paper view to show it.

Layout. Records are written by :class:`sisteer.registry.emit.RecordWriter` into a STAGING spool, in their own namespace so no producer's
sequence numbers can ever collide with them::

    <staging>/rejudge/<track>__<judge_id[:12]>/judge/000001-judge.jsonl
    <staging>/rejudge/<track>__<judge_id[:12]>/<run_dir>/000001-judgment.jsonl
    <staging>/rejudge/<track>__<judge_id[:12]>/<run_dir>/000002-item_result.jsonl.gz

FOLLOW-UP FILINGS. A (run, judge) can be filed again -- a re-ask finished after a partial filing, a correction -- because the registry
folds a later record of the same key over the earlier one: ``judgment`` is keyed by ``judgment_id`` and its columns take the latest
record's values (``reducer.fold``, ordered by written_utc), and ``item_result`` keeps, per natural key (run, item, metric, trial, sample,
turn, judgment, source revision), the row of the greatest order key (``harvest``'s whole-row fold). Neither touches any OTHER judge's
rows. What a follow-up cannot do is remove a row filed earlier under a key it no longer restates (that needs an ``item_withdrawal``),
which is why the filing does not file infrastructure outcomes as item rows by default. For a follow-up to be deliverable its files must
not reuse a name: given the registry root, :class:`Filer` allocates each run directory's sequence numbers ABOVE every number that run
directory already holds in ``records/`` or the inbox (:class:`FloorRecordWriter`), and skips the judge record when the registry already
has it (the same judge id, the same payload apart from its stamps).

:meth:`Filer.validate` runs ``sisteer.registry.validate.validate_record`` over every record of every staged file (a refusal fails the
filing by name before anything is delivered). :meth:`Filer.deliver` then moves the files into ``<registry root>/inbox/<host label>/``
(each copied under a dot-name and renamed, so the inbox merge -- ``harvest.scan_inbox``, blind to dot-files -- sees whole files only).
Delivery is RESUMABLE: a destination whose logical name already stands (inbox or ``records/``, plain or ``.gz``) with the SAME logical
bytes (``records.file_logical_sha``) is skipped, so a delivery interrupted part-way is finished by running it again; a name that stands
with OTHER bytes refuses the whole delivery before anything moves. The harvester's next cycle merges the files into
``records/rejudge/...`` and folds them. Deliver the judge first (``only="judge"``) and the runs after it has folded, so no judgment
waits in the dependency queue (it would fold from there too, one cycle later).

DRY RUN: :meth:`Filer.dry_run_report` builds and validates in a temporary staging directory and reports counts; nothing is delivered.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from sisteer.registry import emit, identity
from sisteer.registry import records as RR
from sisteer.registry import validate as RV

REJUDGE_NAMESPACE = "rejudge"
DEFAULT_HOST_LABEL = "rejudge"
#: the records' ``producer.session`` when neither the caller nor the environment (``emit.SESSION_ENV_VARS``) names one
DEFAULT_SESSION = "rejudge"


def utc_ms(ts: Optional[str] = None) -> str:
    """A ``YYYY-MM-DDTHH:MM:SS.mmmZ`` stamp (the registry's UTC_MS); ``ts`` may be ``...Z`` without milliseconds."""
    if ts is None:
        return RR.utc_now()
    if len(ts) == 20 and ts.endswith("Z"):
        return ts[:-1] + ".000Z"
    return ts


def canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def judge_payload(*, judge_model: str, rubric_name: str, rubric_version: str, rubric_sha256: str, protocol_sha256: str,
                  score_scale: str, threshold_rule: Optional[str], judge_backend: str, label_vocabulary: Mapping[str, Any],
                  notes: str, extra: Mapping[str, Any], judge_role: str = "judge") -> Dict[str, Any]:
    p: Dict[str, Any] = {"judge_model": judge_model, "judge_role": judge_role, "rubric_name": rubric_name, "rubric_version": rubric_version,
                         "rubric_sha256": rubric_sha256, "protocol_sha256": protocol_sha256, "score_scale": score_scale,
                         "threshold_rule": threshold_rule, "label_vocabulary_json": json.dumps(dict(label_vocabulary), sort_keys=True),
                         "judge_backend": judge_backend, "notes": notes, "extra": dict(extra)}
    p["judge_id"] = identity.judge_id(judge_model, rubric_sha256, protocol_sha256, threshold_rule)
    return p


def judgment_payload(*, run_id: str, judge_id: str, status: str, n_items_judged: int, n_judge_refusals: int, n_errors: int,
                     started_utc: Optional[str], finished_utc: Optional[str], batch_ids: Sequence[str], grading_code_sha: str,
                     rubric_sha256: Optional[str], judge_cache_kind: Optional[str], store_uri: Optional[str],
                     producer_session: Optional[str], extra: Mapping[str, Any]) -> Dict[str, Any]:
    p: Dict[str, Any] = {"judgment_id": identity.judgment_id(run_id, judge_id), "run_id": run_id, "judge_id": judge_id, "status": status,
                         "n_items_judged": int(n_items_judged), "n_judge_refusals": int(n_judge_refusals), "n_errors": int(n_errors),
                         "batch_ids_json": json.dumps(sorted(set(batch_ids))), "grading_code_sha": grading_code_sha,
                         "judge_cache_kind": judge_cache_kind, "store_uri": store_uri, "producer_session": producer_session,
                         "extra": dict(extra)}
    if started_utc:
        p["started_utc"] = utc_ms(started_utc)
    if finished_utc:
        p["finished_utc"] = utc_ms(finished_utc)
    if rubric_sha256:
        p["rubric_sha256"] = rubric_sha256
    return {k: v for k, v in p.items() if v is not None}


def item_row(*, run_id: str, judgment_id: str, item_id: str, metric_id: str, trial_index: int, sample_index: int, turn_index: int,
             value: Optional[float], label: Optional[str], judged_utc: Optional[str], created_utc: str, extra: Mapping[str, Any],
             api_pool: Optional[str] = None) -> Dict[str, Any]:
    """A WHOLE ``item_result`` row (every column the validator requires; ``api_pool`` NULL: the vocabulary's pools are Anthropic's)."""
    row: Dict[str, Any] = {c: None for c in RR.ITEM_WHOLE_ROW_COLUMNS["item_result"]}
    row.update({"run_id": run_id, "item_id": item_id, "metric_id": metric_id, "trial_index": int(trial_index), "sample_index": int(sample_index),
                "turn_index": int(turn_index), "judgment_id": judgment_id, "value": value, "label": label, "api_pool": api_pool,
                "created_utc": created_utc, "judged_utc": utc_ms(judged_utc) if judged_utc else None, "source_revision_id": "",
                "extra": dict(extra)})
    return row


class FilingRefused(RuntimeError):
    pass


def seq_high(path: str) -> int:
    """The highest record sequence number any entry of a directory carries (0 for none): every ``<seq>-...`` name, a ``.gz`` twin and a
    ``.conflict-`` copy included -- the rule ``emit.RecordWriter._dir_seq_high`` applies to its own directory."""
    high = 0
    if os.path.isdir(path):
        for fn in os.listdir(path):
            head = fn.lstrip(".").split("-", 1)[0]
            if head.isdigit():
                high = max(high, int(head))
    return high


class FloorRecordWriter(emit.RecordWriter):
    """The registry's record writer with a sequence FLOOR: every number it allocates is above ``seq_floor`` as well as above its own
    directory's listing, so a follow-up filing staged into a fresh directory never reuses a name the registry already holds."""

    def __init__(self, *a: Any, seq_floor: int = 0, **kw: Any):
        super().__init__(*a, **kw)
        self.seq_floor = int(seq_floor)

    def _dir_seq_high(self) -> int:
        return max(super()._dir_seq_high(), self.seq_floor)


class Filer:
    """Stage, validate and deliver one track's records. ``code_sha`` is the producing checkout's commit (the records' ``producer.code_sha``
    and every judgment's ``grading_code_sha``); ``script`` names the driver. ``registry_root`` (read only; optional) makes the filing a
    possible FOLLOW-UP: sequence numbers start above what the registry holds for each run directory, and the judge record is not staged
    again when the registry has it (module docstring)."""

    def __init__(self, staging: str, track: str, judge: Mapping[str, Any], script: str, code_sha: str, session: Optional[str] = None,
                 substrate: str = "grading_host", registry_root: Optional[str] = None, host_label: str = DEFAULT_HOST_LABEL):
        self.staging = staging
        self.track = track
        self.judge = dict(judge)
        self.judge_id = str(judge["judge_id"])
        self.namespace = f"{REJUDGE_NAMESPACE}/{track}__{self.judge_id[:12]}"
        self.code_sha = code_sha
        self.session = session or emit.session_from_env() or DEFAULT_SESSION
        self.producer = emit.producer_block(script, code_sha=code_sha, session=self.session, substrate=substrate)
        self.registry_root = os.path.abspath(os.path.expanduser(registry_root)) if registry_root else None
        self.host_label = host_label
        self.files: List[str] = []
        self.seq_floors: Dict[str, int] = {}
        self._last_utc = ""

    def registry_dirs(self, dir_id: str) -> List[str]:
        """Where the registry root holds (or will hold) this namespace's record directory ``dir_id``: ``records/`` and the inbox."""
        if not self.registry_root:
            return []
        rel = os.path.join(self.namespace, identity.run_dir_name(dir_id))
        return [os.path.join(self.registry_root, "records", rel), os.path.join(self.registry_root, "inbox", self.host_label, rel)]

    def seq_floor(self, dir_id: str) -> int:
        """The highest sequence number the registry root holds for this record directory (0 when none: a first filing); a positive
        floor marks the filing a FOLLOW-UP (recorded in :attr:`seq_floors`)."""
        floor = max([seq_high(d) for d in self.registry_dirs(dir_id)] or [0])
        if floor:
            self.seq_floors[dir_id] = floor
        return floor

    def _writer(self, dir_id: str) -> emit.RecordWriter:
        floor = self.seq_floor(dir_id)
        w = FloorRecordWriter(self.staging, self.namespace, dir_id, self.producer, disabled=False, payload_extra={}, seq_floor=floor)
        w.not_before(self._last_utc)
        return w

    def judge_on_record(self) -> Optional[str]:
        """The path of a judge record file the registry root already holds (``records/`` or the inbox) for this judge id, else None.
        The id is a hash of the judge model, rubric, protocol and threshold, so the same id IS the same judge."""
        for d in self.registry_dirs("judge"):
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                if fn.startswith(".") or "-judge.jsonl" not in fn or ".conflict-" in fn:
                    continue
                path = os.path.join(d, fn)
                for _line, rec, _sha in RR.read_record_file(path):
                    if rec.get("record_type") == "judge" and (rec.get("payload") or {}).get("judge_id") == self.judge_id:
                        return path
        return None

    def _done(self, w: emit.RecordWriter) -> None:
        self._last_utc = max(self._last_utc, w.last_written_utc)
        self.files.extend(w.written)

    def stage_judge(self) -> Optional[str]:
        """Stage the judge record, unless the registry root already holds it (returns None then)."""
        if self.judge_on_record():
            return None
        w = self._writer("judge")
        path = w.write("judge", self.judge)
        self._done(w)
        return path

    def stage_run(self, run_id: str, judgment: Mapping[str, Any], items: Sequence[Mapping[str, Any]]) -> List[str]:
        """One run's judgment record file, then its item_result file (gzip), stamped after it."""
        if judgment["judge_id"] != self.judge_id or judgment["run_id"] != run_id:
            raise FilingRefused(f"judgment for {run_id} names judge {judgment['judge_id']} / run {judgment['run_id']}")
        w = self._writer(run_id)
        out = [w.write("judgment", dict(judgment))]
        if items:
            for r in items:
                if r["judgment_id"] != judgment["judgment_id"] or r["run_id"] != run_id:
                    raise FilingRefused(f"item row of {run_id} names another run or judgment")
            out.append(w.write_many("item_result", [dict(r) for r in items], gz=True))
        self._done(w)
        return out

    # -- validation ----------------------------------------------------------------------------
    def validate(self) -> Dict[str, Any]:
        """Every record of every staged file through the registry's validator; returns counts and refusals (empty = clean)."""
        counts: Counter = Counter()
        refusals: List[str] = []
        for path in self.files:
            for line_no, rec, sha in RR.read_record_file(path):
                v = RV.validate_record(rec, f"{path}#{line_no}", sha)
                if isinstance(v, RV.Malformed):
                    refusals.append(str(v))
                else:
                    counts[v.record_type] += 1
        return {"records": dict(counts), "files": len(self.files), "refusals": refusals}

    # -- delivery ------------------------------------------------------------------------------
    def staged_files(self, only: Optional[str] = None, run_dirs: Optional[Iterable[str]] = None) -> List[Tuple[str, str]]:
        """``[(absolute path, path relative to the staging root)]``; ``only="judge"`` the judge file, ``only="runs"`` the rest;
        ``run_dirs`` keeps only those run directory names (``emit.run_dir_name(run_id)``)."""
        return _select(((p, os.path.relpath(p, self.staging)) for p in self.files), only, run_dirs)

    def deliver(self, registry_root: str, host_label: str = DEFAULT_HOST_LABEL, only: Optional[str] = None,
                run_dirs: Optional[Iterable[str]] = None) -> List[str]:
        """Copy staged files into ``<root>/inbox/<host_label>/<rel>`` (dot-name, fsync, rename). A destination whose logical name
        already stands (inbox or ``records/``, plain or ``.gz``) with the same logical bytes is skipped (a resumed delivery); one that
        stands with OTHER bytes refuses the whole delivery before anything moves. Returns the destinations written now."""
        root = os.path.abspath(os.path.expanduser(registry_root))
        plan = []
        skipped = 0
        for src, rel in self.staged_files(only, run_dirs):
            dest = os.path.join(root, "inbox", host_label, rel)
            logical_rel = rel[:-3] if rel.endswith(".gz") else rel
            standing = [c for c in (os.path.join(root, "inbox", host_label, logical_rel), os.path.join(root, "inbox", host_label, logical_rel + ".gz"),
                                    os.path.join(root, "records", logical_rel), os.path.join(root, "records", logical_rel + ".gz")) if os.path.exists(c)]
            if standing:
                mine = RR.file_logical_sha(src)
                other = [c for c in standing if RR.file_logical_sha(c) != mine]
                if other:
                    raise FilingRefused(f"destination name already stands with other bytes: {other[0]} (nothing delivered)")
                skipped += 1
                continue
            plan.append((src, dest))
        done = []
        for src, dest in plan:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            tmp = os.path.join(os.path.dirname(dest), "." + os.path.basename(dest) + ".part")
            shutil.copyfile(src, tmp)
            with open(tmp, "rb") as fh:
                os.fsync(fh.fileno())
            os.replace(tmp, dest)
            done.append(dest)
        self.last_delivery = {"written": len(done), "already_standing_identical": skipped}
        return done


def _select(pairs: Iterable[Tuple[str, str]], only: Optional[str], run_dirs: Optional[Iterable[str]]) -> List[Tuple[str, str]]:
    keep = set(run_dirs) if run_dirs is not None else None
    out = []
    for p, rel in pairs:
        parts = rel.split(os.sep)
        if len(parts) < 4:
            continue
        is_judge = parts[2] == "judge"
        if (only == "judge" and not is_judge) or (only == "runs" and is_judge):
            continue
        if keep is not None and not is_judge and parts[2] not in keep:
            continue
        out.append((p, rel))
    return sorted(out, key=lambda x: x[1])


def staged_files(staging: str, only: Optional[str] = None, run_dirs: Optional[Iterable[str]] = None) -> List[Tuple[str, str]]:
    """Every record file already staged under ``<staging>/rejudge/`` (``only="judge"`` the judge files, ``"runs"`` the rest; ``run_dirs``
    keeps those run directory names), sorted."""
    pairs = []
    base = os.path.join(staging, REJUDGE_NAMESPACE)
    for dirpath, _dirs, files in os.walk(base):
        for fn in files:
            if fn.startswith(".") or not (fn.endswith(".jsonl") or fn.endswith(".jsonl.gz")):
                continue
            p = os.path.join(dirpath, fn)
            pairs.append((p, os.path.relpath(p, staging)))
    return _select(pairs, only, run_dirs)


def deliver_staged(staging: str, registry_root: str, host_label: str = DEFAULT_HOST_LABEL, only: Optional[str] = None,
                   run_dirs: Optional[Iterable[str]] = None) -> List[str]:
    """Deliver files staged by an EARLIER filing (the second half of judge-first delivery: the same bytes the dry run validated);
    resumable like :meth:`Filer.deliver`."""
    f = Filer.__new__(Filer)
    f.staging = staging
    f.files = [p for p, _ in staged_files(staging)]
    return f.deliver(registry_root, host_label, only, run_dirs)


# --------------------------------------------------------------------------------------------
# a filing under judge rows already on record: what the staging may hold, checked on disk
# --------------------------------------------------------------------------------------------
#: the MiniMax-M3 judge rows already on record in the registry (``judge_backend`` batch_or_live: the batch track's protocol plus its live
#: route), per records namespace track and rubric. A filing that reuses them stages judgments under exactly these ids and no judge record
#: (``--expect-pinned-judges`` of both filers); ``tests/test_rejudge_paper_judges.py`` recomputes the same ids from the code's protocol.
MINIMAX_M3_JUDGE_IDS: Dict[str, Dict[str, str]] = {
    "strongreject_minimax_m3": {"sr": "192df392a676e6b046e86c08ebb96484d31cad7573736d10e7c5351a5128605c",
                                "sr2": "d6c07d238e6ae370846ed272524201b423bc9885f3d1b66933af47bfc7eb6be7"},
    "cpi_minimaxm3": {"or_bench_cpi": "0a4bfa498117f48293bb1636c37b74dc4fca2595032b2b32941cb83d71689907",
                      "xstest_cpi": "5e79a37817fef85e8b4d0d8b0043a762ecb8b53870354737ed98bee6e675a2c0"},
}


def logical_uri(rel: str) -> str:
    """The registry's ``file_uri`` of a staged file (``records/<rel>``, the ``.gz`` dropped: ``record_delivery`` / ``record_ingest`` key
    the decompressed logical name)."""
    return "records/" + RR.logical_uri(rel.replace(os.sep, "/"))


def pin_refusal(name: str, filer: Filer, pinned_judge_id: Optional[str]) -> Optional[str]:
    """Why a filing under judge rows already on record may not stage under ``filer`` (None = it may): its judge must be the pinned row
    ``pinned_judge_id`` and already on record in the filer's registry root, so no judge record is staged. ``name`` names the judge
    (its rubric) in the refusal."""
    if filer.judge_id != pinned_judge_id:
        return f"the {name} judge this filing would stage under is {filer.judge_id}, not the pinned {pinned_judge_id}"
    if not filer.judge_on_record():
        return f"the {name} judge {filer.judge_id[:12]} is not on record in {filer.registry_root}: its judge record would be staged"
    return None


def check_staging(staging: str, allowed_judge_ids: Iterable[str], run_ids: Optional[Iterable[str]] = None,
                  allow_judge_record: bool = False) -> Dict[str, Any]:
    """Read EVERY record staged under ``<staging>/rejudge/`` from disk (independently of the filer that wrote them) and report what the
    filing would deliver, with every problem by name:

      * a ``judge`` record (unless ``allow_judge_record``): a filing under judge rows already on record stages none
      * a record type other than judge / judgment / item_result
      * a judgment whose ``judge_id`` is not one of ``allowed_judge_ids``, whose ``run_id`` is not in ``run_ids`` (when given), whose
        ``judgment_id`` is not the plain-form id over (run, judge) recomputed here, or which sits in another run's record directory or
        another judge's namespace
      * an item_result row naming another run or a judgment that is not its directory's staged judgment
      * a run of ``run_ids`` (when given: the runs the filing was asked to file, i.e. its scope's runs narrowed by its run filter)
        with nothing staged -- the filer skips a run with no unit in the unit map, or with fewer terminal units than its
        ``--min-terminal``, without a word, and this is where that is said

    Returns ``{"problems", "judge_ids", "runs": {run_id: {judgment_id, judge_id, status, n_items_judged, item_rows: {metric: n},
    files: [{rel, uri, logical_sha256, record_types}]}}, "record_files": [uri...], "n_record_files", "expected_runs"}``: the filing's
    judgment ids, item rows and record files, the input of an attribution after delivery, and the runs it was held to (``run_ids``
    sorted, or None)."""
    allowed = set(allowed_judge_ids)
    want_runs = set(run_ids) if run_ids is not None else None
    problems: List[str] = []
    runs: Dict[str, Dict[str, Any]] = {}
    files_by_dir: Dict[str, List[Tuple[str, str]]] = {}
    for path, rel in staged_files(staging):
        parts = rel.replace(os.sep, "/").split("/")
        files_by_dir.setdefault("/".join(parts[:3]), []).append((path, rel))
    record_files: List[str] = []
    for d, files in sorted(files_by_dir.items()):
        _ns, track_judge, dir_name = d.split("/")
        judge_prefix = track_judge.rsplit("__", 1)[-1]
        judgments: Dict[str, Dict[str, Any]] = {}
        items: List[Tuple[str, Dict[str, Any]]] = []
        dir_files: List[Dict[str, Any]] = []
        for path, rel in files:
            types: Counter = Counter()
            logical = hashlib.sha256()       # fed every logical byte read: the file's logical sha256 without a second pass
            for _line, rec, _sha in RR.read_record_file(path, hasher=logical, compute_sha=False):
                rt = rec.get("record_type")
                types[str(rt)] += 1
                p = rec.get("payload") or {}
                if rt == "judge":
                    if not allow_judge_record:
                        problems.append(f"{rel}: a judge record is staged (judge {str(p.get('judge_id'))[:12]}); this filing's judges must "
                                        f"already be on record")
                    if p.get("judge_id") not in allowed:
                        problems.append(f"{rel}: judge {p.get('judge_id')} is not one of the expected judges")
                elif rt == "judgment":
                    rid, jg, jid = p.get("run_id"), p.get("judge_id"), p.get("judgment_id")
                    if jg not in allowed:
                        problems.append(f"{rel}: judgment {jid} names judge {jg}, not one of the expected judges")
                    if want_runs is not None and rid not in want_runs:
                        problems.append(f"{rel}: judgment {jid} is on run {rid}, which is not in the filing's run list")
                    if not isinstance(rid, str) or not isinstance(jg, str) or jid != identity.judgment_id(rid, jg):
                        problems.append(f"{rel}: judgment id {jid} is not the plain-form id over ({rid}, {jg})")
                    if isinstance(rid, str) and identity.run_dir_name(rid) != dir_name:
                        problems.append(f"{rel}: judgment of run {rid} sits in record directory {dir_name}")
                    if isinstance(jg, str) and not jg.startswith(judge_prefix):
                        problems.append(f"{rel}: judgment of judge {jg[:12]} sits in namespace {track_judge}")
                    if jid in judgments:
                        problems.append(f"{rel}: judgment {jid} staged twice")
                    judgments[jid] = p
                elif rt == "item_result":
                    items.append((rel, p))
                else:
                    problems.append(f"{rel}: a {rt} record is staged (a re-judge files judge, judgment and item_result records only)")
            uri = logical_uri(rel)
            record_files.append(uri)
            dir_files.append({"rel": rel, "uri": uri, "logical_sha256": logical.hexdigest(), "record_types": dict(types)})
        if dir_name == "judge":
            continue
        if len(judgments) != 1:
            problems.append(f"{d}: {len(judgments)} judgment records (one per run directory)")
            continue
        (jid, jp), = judgments.items()
        rid = jp.get("run_id")
        metrics: Counter = Counter()
        for rel, p in items:
            if p.get("run_id") != rid or p.get("judgment_id") != jid:
                problems.append(f"{rel}: an item row names run {p.get('run_id')} / judgment {p.get('judgment_id')}, not {rid} / {jid}")
                continue
            metrics[str(p.get("metric_id"))] += 1
        r = runs.setdefault(rid, {"files": []})
        if "judgment_id" in r:
            problems.append(f"run {rid}: staged in two record directories")
        r["files"].extend(dir_files)
        r.update({"judgment_id": jid, "judge_id": jp.get("judge_id"), "status": jp.get("status"), "n_items_judged": jp.get("n_items_judged"),
                  "item_rows": dict(metrics)})
    if want_runs is not None:
        missing = sorted(want_runs - set(runs))
        if missing:
            problems.append(f"{len(missing)} runs of the list have nothing staged (e.g. {missing[:3]}): the filer skipped them (no unit "
                            f"in the unit map, or fewer terminal units than --min-terminal)")
    return {"problems": problems, "judge_ids": sorted({r.get("judge_id") for r in runs.values() if r.get("judge_id")}),
            "runs": runs, "record_files": sorted(record_files), "n_record_files": len(record_files),
            "expected_runs": sorted(want_runs) if want_runs is not None else None}


def staging_check_summary(check: Mapping[str, Any]) -> str:
    """One line of a :func:`check_staging` report: its runs, judges, record files and problems counted."""
    return (f"staging check: {len(check['runs'])} runs, judges {[j[:12] for j in check['judge_ids']]}, {check['n_record_files']} record "
            f"files, {len(check['problems'])} problems")


def write_filing_ids(path: str, staging: str, code_sha: str, check: Mapping[str, Any]) -> None:
    """The ``--ids-out`` file of a filing (the input of an attribution after delivery): the staging, the code sha, and from its
    :func:`check_staging` report the judge ids, the judgment id per run, each run's item rows and record files, and every record file."""
    with open(path, "w") as fh:
        json.dump({"staging": os.path.abspath(staging), "code_sha": code_sha, "judge_ids": check["judge_ids"],
                   "judgments": {r: v["judgment_id"] for r, v in sorted(check["runs"].items())}, "runs": check["runs"],
                   "record_files": check["record_files"]}, fh, indent=1, sort_keys=True)


def staged_check_refusal(staging: str) -> Optional[str]:
    """Why a staging may not be delivered on the word of the check made when it was staged (None = it may): the ``staging_check`` its
    ``filing_report.json`` recorded -- against the runs that filing was asked to file -- found problems. A staging with no report or no
    recorded check says nothing here (the caller re-reads the staged files itself)."""
    try:
        with open(os.path.join(staging, "filing_report.json")) as fh:
            problems = (json.load(fh).get("staging_check") or {}).get("problems") or []
    except (OSError, ValueError):
        return None
    return "; ".join(str(p) for p in problems[:10]) if problems else None


def only_run_dirs(staging: str, run_ids: Sequence[str], staged_runs: Iterable[str]) -> List[str]:
    """The record directory names of ``run_ids`` for a delivery of those runs alone; refuses (:class:`FilingRefused`) a run the staging's
    report does not hold or that has no staged record file."""
    staged_runs = set(staged_runs)
    names = []
    dirs = {rel.replace(os.sep, "/").split("/")[2] for _p, rel in staged_files(staging, only="runs")}
    for rid in run_ids:
        if rid not in staged_runs:
            raise FilingRefused(f"--only-run {rid}: not a run of this staging's filing_report.json")
        name = identity.run_dir_name(rid)
        if name not in dirs:
            raise FilingRefused(f"--only-run {rid}: no record file staged under {name}")
        names.append(name)
    return names


def file_sha256(path: str) -> str:
    return RR.file_sha256(path)
