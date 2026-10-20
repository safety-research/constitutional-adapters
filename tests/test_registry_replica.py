"""The cluster replica: ``sisteer.registry.replica`` and the two chain files
``replica.sbatch`` / ``live_db_slurm.sh``, rehearsed on synthetic worlds.

* ``frame_projection/1``: its column list is pinned to the code's paper frame object (minus exactly ``ingest_seq``; a moved column fails
  here until the version is bumped); identical content folded in another file order projects to one sha (order independence); the
  three set-valued ``*_ids_json`` arrays sort, JSON keys sort, floats are ``repr``; a per-column negative case names the column.
* the head: one read transaction at a cursor-free boundary listing EVERY delivered file (every status); a REAL partial-file fold -- a
  plain file streamed in windows of two records, cut after its first window -- leaves a ``record_cursor`` row and the head REFUSES by
  name, then waits and returns once the last window lands; the first window's records had changed projected rows, so the two heads
  differ in ``lifecycle_state`` on more than one run; the replica built from the first head ingests the second head's increment and is
  equal to it.
* the allowlist: an unlisted plain file on disk changes nothing; an unlisted batch file inside a listed revision directory is ignored
  and counted (``unlisted_ignored``); an unlisted file inside a listed dimension directory is not folded; a replica holding a delivered
  file the head lacks refuses by name (``ReplicaAhead``: the out-of-order landing) and ``check`` is unequal.
* generations: publish is a compare-and-swap on CURRENT (a stale parent refuses; a non-superset allowlist refuses), the last three are
  kept with CURRENT never pruned, rollback moves CURRENT to the parent; migrate republishes a copy and refuses a shape no migration
  expresses; refold-compare with one tree on both sides finds no difference and moves no projection.
* the C1 certification: a kept database built under the NEW order whose log carries one same-stamp ``judged`` +
  ``superseded`` pair the OLD order drew ``judged`` -- against a pre-tie-break tree (the repo's package with the order's one spelling
  patched back), the OLD-order fold of the copy hashes to the head taken from the live (old-order) registry, the one mover is in the
  declared-superseded class and the verdict is PASS, with the same verdict on a copy built under the OLD order; a ``judged`` +
  ``withdrawn`` mover FAILS the class check; a wrong expected hash FAILS; a hash prefix, a head whose projection is not the expected
  hash and a database with a non-empty WAL are refused before any work; a declared mover set is compared as an exact set; the
  sbatch's ``--db`` form copies the input node-local, hashes it, never reads CURRENT, writes nothing under the replica root and exits
  1 on FAIL, while the optionless form still measures CURRENT's generation. A certification is BOUND to what it declares: the input's
  independently computed sha256, the order the copy was built under (a live-order copy declared as the kept rebuild FAILS), and both
  trees' commits (a wrong HEAD, a dirty checkout, a directory that is no checkout, a snapshot nested under a checkout's ignored directory
  and an ignored module beside the committed code all refuse), the NEW-order hash and the exact mover set; a mover without a live row
  refuses.
* the kept-database modes, run as ``python -m sisteer.registry.replica`` under a committed tree the way the sbatch runs them, every
  declaration REQUIRED (the input's sha, the head's ``head_sha256``, the running tree's commit -- bound by git and by the module's own
  location -- the migrating tree's commit, the target schema): ``update`` copies a kept database node-local (hashed as read, the input
  never written; the declared sha must match), requires the head to strictly supersede the copy's delivered ledger (an older head refuses
  by name with no output and no mirror hash; the copy's own head -- nothing to fold -- refuses as HEAD_NOT_NEWER), folds the increment to
  the newer head, checks it with the full projection written beside the verdict and writes the result out hashed, with nothing under the
  replica root written and no generation; an unequal head leaves the output standing with exit 7; an output directory that is the input's own directory, the work directory, a
  path under the generations or a directory holding a generation's ``meta.json`` is refused before anything is cleared. ``migrate --db``
  carries a kept copy to the MIGRATING tree's schema by that tree's own ``registry migrate --db`` (a committed tree with one more migration
  file advances the version by exactly one and its plan reads clean), refuses a migration whose shape that tree's DDL does not declare
  (exit 8, no output), refuses a tree whose migrations stop short of the declared schema before the copy and a copy that stops short
  after it (``SCHEMA_NOT_REACHED``), binds both trees by commit and the input by sha; ``replica.sbatch update`` / ``migrate --db`` run the
  real module with ``srun`` shimmed and refuse an option set that leaves a declaration out.
* the landing job: ``land`` runs pytest against the node-local copy with ``REGISTRY_LIVE_DB`` and ``REGISTRY_PIN_L1`` set, accounts the
  skips (one more than expected refuses), publishes the equal generation, stamps every result with the job's identity (job id, tree,
  head sha), removes the outputs it owns from a reused directory first, and turns a REFUSED publication (CURRENT moved between the copy
  and the publication) into a failing status whatever pytest said; ``replica.sbatch land`` runs the real module with ``srun`` shimmed and
  the venv's python replaced by a shim that writes the junit; ``live_db_slurm.sh`` takes a real head of a synthetic registry, calls the
  mirror sync, ships the head to a per-submission path, submits (recorded argv) with a per-submission output directory, watches, reads
  the job's state from the accounting, collects into a fresh directory, writes ``state/replica_check.json`` and refuses an unequal check,
  a red pytest, unaccounted skips, a job that did not COMPLETE (the earlier submission's results cannot stand in: its directory is its
  own), results whose identity is not this job's / tip's / head's, a job that published nothing, and a CURRENT that no longer names the
  published generation; the head's exit 6 defers once to the harvester's next completed cycle. In a RECOVERY (``LIVE_DB_REPLICA_TREE``)
  the sbatch script submitted is the chain's own ``replica.sbatch`` -- probed with ``replica.sbatch contract`` (``land/2 … REPLICA_TREE``)
  before the head is taken, shipped beside the head into the submission's own directory, sha256-verified there -- never a tree's: the
  ACTUAL earlier script's code (vendored, sha256-pinned) is run under nine positionals and shown to run the tip's code on the replica side.
Nothing here reaches a cluster or the live registry.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Sequence
from unittest import mock

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry import replica as RP
from sisteer.registry import site_aliases as SA
from sisteer.registry import synth
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import runner as RUN

from _env import have_exe

#: live_db_slurm.sh ships the head and collects the results with host-prefixed rsync operands over its ssh transport: with the real binary
#: the world's ssh shim runs the remote ``rsync --server`` locally, so the wire protocol is spoken end to end and the step under test is the
#: transport itself. The test fallback (tests/_rsync_fallback.py) runs no remote shell and refuses a remote operand, so these tests need
#: the binary and skip where it is absent, rather than passing on a local copy that exercised no transport.
NEEDS_RSYNC = pytest.mark.skipif(not have_exe("rsync"), reason="live_db_slurm.sh ships and collects over rsync's ssh transport (host-prefixed operands "
                                                                "through the ssh shim); the test fallback runs no remote shell")

REPO = Path(__file__).resolve().parent.parent
LANDING = REPO / "scripts" / "registry" / "landing"
PY = sys.executable
# The script a RECOVERY must never submit: replica.sbatch as it stood before its land mode took the 9th positional REPLICA_TREE
# (bdd22205c; the same script the bootstrap generation's tree 87d9a84e carries). VENDORED with its code byte for byte but for two log
# messages' plain-text tails, its plain comment lines reworded (tests/fixtures/registry/README.md), and sha256-pinned as vendored.
# ``_pre_c9_identity`` keeps every line but the plain comments and masks those two tails (``PRE_C9_INERT_LOGS``), and its sha256 is
# taken from the commit's object, so the battery's shallow exact-sha export (where ``git show <sha>:<path>`` exits 128) still checks
# that the copy runs the commit's code; ``test_pre_c9_dispatcher_fixture_matches_the_git_object`` also reads the object itself
# wherever the commit is reachable.
PRE_C9_SHA = "bdd22205c21ab60c80ad01851b1f949d98d0c64d"
PRE_C9_REL = "scripts/registry/landing/replica.sbatch"
PRE_C9_FIXTURE = REPO / "tests" / "fixtures" / "registry" / "replica_sbatch_pre_c9_bdd22205c.sh"
PRE_C9_SHA256 = "b227d8f1ce11522fb2040d9840f1519dcdacc975e3ce3c27b2b5d8f7eaa42274"          # the vendored copy
PRE_C9_OBJECT_SHA256 = "7d41d5b648072e288b242733ab969a7ec2e1eeeb92596dae9c6a9227265d10fe"   # the commit's object, byte for byte
PRE_C9_CODE_SHA256 = "c98408976fafd04c693d3496e6dc6c816668e102f17a9298cba5a2df50e56746"     # _pre_c9_identity of either
#: The only executable text the vendored copy rewords: the plain-text tails of two log messages (the unequal build's and the missing
#: head file's), which bash only prints. Each pattern pins its whole line but the tail, and a tail admits no `$`, backtick, quote or
#: backslash, so the mask can hide no expansion or command; each must match exactly once, so it can neither go dead nor widen.
PRE_C9_INERT_LOGS = (
    re.compile(rb'^(      log "UNEQUAL build: keeping the work database as \$OUT/registry\.sqlite\.unequal for )[^"$`\\\n]*(")$', re.M),
    re.compile(rb'^(need_head\(\) \{ \[ -s "\$1" \] \|\| \{ log "no head file at \$1 \(registry replica head --out, shipped by )'
               rb'[^"$`\\\n]*(\)"; exit 2; \}; \})$', re.M),
)
LAND_CONTRACT = "land/3 TREE HEAD VENV BRANCH REPLICA_ROOT OUT_DIR EXPECT_SKIPPED HASH_MODE REPLICA_TREE ACCEPT_C1_CLASS"
#: The cutover declaration as the operator passes it: free text, spaces and commas included, a `~` that must NOT expand.
C1_DECLARATION = "refold-compare job 1000001, ~/refold/refold_compare.json"


def _pre_c9_dispatcher_bytes() -> bytes:
    data = PRE_C9_FIXTURE.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    assert digest == PRE_C9_SHA256, f"{PRE_C9_FIXTURE.name} drifted from its vendored bytes: sha256 {digest}"
    return data


def _pre_c9_code(data: bytes) -> bytes:
    """The dispatcher's code: every line but a plain comment line. A line whose first non-blank character is `#` is dropped unless it
    is a directive (the shebang, an `#SBATCH` line, which sbatch reads, or a shellcheck directive), and a comment after code on its
    line stays, so every line bash, sbatch or the linter acts on is compared."""
    keep = []
    for line in data.splitlines(keepends=True):
        s = line.lstrip()
        if s.startswith(b"#") and not s.startswith((b"#!", b"#SBATCH", b"# shellcheck")):
            continue
        keep.append(line)
    return b"".join(keep)


def _pre_c9_identity(data: bytes) -> bytes:
    """What the vendored copy and the commit's object must share: ``_pre_c9_code`` with each ``PRE_C9_INERT_LOGS`` tail masked, every
    pattern matching exactly one line."""
    code = _pre_c9_code(data)
    for pat in PRE_C9_INERT_LOGS:
        code, n = pat.subn(rb"\1<log text>\2", code)
        assert n == 1, f"an inert-log pattern matches {n} lines, not one: {pat.pattern[:72]!r}"
    return code


# ---- worlds ----------------------------------------------------------------------------------------------------------------------
def _ingest(root, **kw):
    kw.setdefault("workers", 0)
    kw.setdefault("min_free", 0)
    return H.ingest_once(root, **kw)


def _q(db, sql, *params):
    with DB.reading(db) as conn:
        return [tuple(r) for r in conn.execute(sql, params).fetchall()]


def _world(tmp_path, n_runs=3, n_items=6):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=n_runs, n_items=n_items)
    _ingest(root)
    return root, ids


def _head(root, **kw):
    kw.setdefault("wait_min", 0)
    return RP.take_head(DB.db_path(root), **kw)


def _bootstrap(tmp_path, root, head, name="replica"):
    lay = RP.init_layout(str(tmp_path / name), root)
    work = tmp_path / f"work-{name}"
    work.mkdir()
    res = RP.build(lay, head, str(work / "registry.sqlite"), workers=0)
    assert res["check"]["equal"], RP.format_check(res["check"])
    RP.publish(lay, str(work / "registry.sqlite"), head, mode="build", parent=None, built_by_tree="a" * 40)
    return lay


def _copy_current(tmp_path, lay, name):
    d = tmp_path / name
    d.mkdir()
    db = str(d / "registry.sqlite")
    RP._copy_db(lay.db_of(lay.current()), db)
    return db


def _now_rec(record_type, payload, i):
    """A record stamped now (+ i ms): folds AFTER the harvester's own stamps, so a lifecycle event here moves the projected row."""
    dt = datetime.fromtimestamp((R.utc_to_epoch_ms(R.utc_now()) + i) / 1000.0, tz=timezone.utc)
    return synth.rec(record_type, payload, R._fmt_utc(dt))


class _StoreAdapter:
    """`<root>/*/store.json` -> one judged run per store, one comparison group per store (the core-fixes adapter, trimmed)."""
    covers_sources = ("synthetic",)
    rng_scope = "none"
    allow_symlinks = False

    def __init__(self, store_root: str, name: str = "synthetic_store"):
        self.name = name
        self.roots = (store_root,)

    def discover(self) -> List[AB.Input]:
        out = []
        for d in sorted(os.listdir(self.roots[0])):
            p = os.path.join(self.roots[0], d, "store.json")
            if os.path.lexists(p):
                out.append(AB.check_input(p, role="report"))
        return out

    def comparison_group(self, inp: AB.Input) -> str:
        return os.path.basename(os.path.dirname(inp.uri))

    def read(self, ctx, group, inputs: Sequence[AB.Input]) -> Iterator[AB.RecordBatch]:
        for inp in inputs:
            with open(inp.path) as fh:
                store = json.load(fh)
            batch = AB.RecordBatch(RUN.source_id_for(self.name, group), group)
            key = store["key"]
            rid = I.run_id(synth.CAMPAIGN, key, store["object"], synth.EVAL_ID, synth.INSTRUMENT_ID)
            obj_id = "none" if store["object"] == "none" else synth.OBJECT_SHA
            p = synth.run_payload(key, store["object"], obj_id, i=1000)
            p.update(run_id=rid, lifecycle_state="generated", started_utc=synth.t(1001), generated_utc=synth.t(1002),
                     n_items_generated=len(store["items"]), n_items_judged=len(store["items"]), judge_id=synth.JUDGE_ID,
                     store_completion_artifact_uri=inp.uri, store_completion_sha256=inp.sha256, store_uri=inp.uri)
            jid = I.judgment_id(rid, synth.JUDGE_ID)
            batch.add("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID, "n_items_judged": len(store["items"]),
                                   "status": "complete"}, ctx.producer, ctx.now)
            batch.add("run", p, ctx.producer, ctx.now)
            batch.add("run_state", {"run_id": rid, "state": "judged", "event_utc": synth.t(1003), "evidence_kind": "completion_artifact",
                                    "evidence_uri": inp.uri}, ctx.producer, ctx.now, observe=False)
            vals = [it["score"] for it in store["items"]]
            batch.add("run_metric", {"run_id": rid, "judgment_id": jid, "metric_id": "asr", "threshold": 0.5,
                                     "value": sum(vals) / len(vals), "k": sum(1 for v in vals if v > 0.5), "n": len(vals),
                                     "computed_by": "registry", "source_uri": inp.uri}, ctx.producer, ctx.now)
            for it in store["items"]:
                batch.add("item_result", {"run_id": rid, "item_id": it["id"], "metric_id": "asr", "trial_index": 0, "sample_index": 0,
                                          "turn_index": -1, "judgment_id": jid, "value": it["score"], "label": None, "api_pool": None,
                                          "created_utc": ctx.now, "judged_utc": None, "withdrawn_by_revision_id": None}, ctx.producer, ctx.now)
            yield batch

    def source_states(self, ctx, inputs):
        return [AB.SourceStateRow(RUN.source_id_for(self.name, "all"), i.uri, "$.object", 1, 1, [json.load(open(i.path))["object"]], i.sha256)
                for i in inputs]


def _revision_world(tmp_path):
    """A registry with one adapter revision directory folded live (a judged run with items), the replica bootstrapped from its head."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    _ingest(root)
    store_root = str(tmp_path / "stores")
    os.makedirs(os.path.join(store_root, "s1"))
    with open(os.path.join(store_root, "s1", "store.json"), "w") as fh:
        json.dump({"key": "store_1", "object": "constitution_lora@aaaa", "items": [{"id": f"q{i}", "score": s} for i, s in enumerate([0.2, 0.8, 0.4])]}, fh)
    ad = _StoreAdapter(store_root)
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=synth.t(2000))
    assert rep.results and rep.results[0].revision_id
    ing = _ingest(root)
    assert ing.refused == 0 and ing.blocked == 0, (ing.refused, ing.blocked, ing.refusals)
    rev_dirs = [d for d in Path(root, "records", "adapters", ad.name).iterdir() if d.is_dir() and not d.name.startswith("_")]
    assert len(rev_dirs) == 1, rev_dirs
    return root, str(rev_dirs[0])


# ---- the refusal counts beside the verdict -----------------------------------------------------------------------------------------------
def test_check_names_the_record_refusal_counts_and_the_increment_folds_the_harvester_facts_after_the_revisions(tmp_path):
    """``check`` reports ``record_refusal`` as named counts -- ``run_ingest`` is the certification line of a bootstrap (0 when the replay
    folded every stamp live folded) -- printed beside the verdict; and ``ingest_head`` folds an increment in ``rebuild``'s order: the
    harvester's facts after the revision directories and the blocked-queue pass, so a stamp whose run rides a revision of the same
    increment applies on its first attempt."""
    from test_registry_rebuild import _late_carrier_world, _delivered
    root, rid, stamp_uri = _late_carrier_world(tmp_path)
    head = _head(root)
    lay = RP.init_layout(str(tmp_path / "replica"), root)
    work = tmp_path / "work"
    work.mkdir()
    res = RP.build(lay, head, str(work / "registry.sqlite"), workers=0)
    assert res["check"]["equal"], RP.format_check(res["check"])
    ref = res["check"]["record_refusal"]
    assert ref["run_ingest"] == 0 and ref["harvester_facts"] == 0 and ref["total"] == 0 and ref["by_type"] == {}
    assert "record_refusal: run_ingest 0 (must be 0), harvester facts 0, total 0" in RP.format_check(res["check"])
    # the increment path: a replica of the dimensions alone takes the whole world as its delta
    dims_only = {u for u in RP.head_uris(head) if u.startswith("records/dims/")}
    seed = tmp_path / "seed"
    seed.mkdir()
    db = str(seed / "registry.sqlite")
    rep = H.rebuild(root, db, workers=0, files=sorted(dims_only))
    assert rep.refused == 0
    inc = RP.ingest_head(db, lay, head, workers=0)
    assert inc["report"]["refused"] == 0 and inc["revision_dirs"] == 3, inc
    chk = RP.check(db, head)
    assert chk["equal"], RP.format_check(chk)
    assert chk["record_refusal"]["run_ingest"] == 0
    assert _q(db, "SELECT status, n_ok, n_blocked FROM record_delivery WHERE file_uri = ?", stamp_uri) == [("ok", 1, 0)]
    assert _q(db, "SELECT lifecycle_state FROM run WHERE run_id = ?", rid) == [("ingested",)]


def test_a_refused_harvester_stamp_makes_the_check_unequal_even_when_the_projection_and_the_ledger_match(tmp_path):
    """A replay that refused a `run_ingest` the live fold applied could still read ``equal`` -- the ledger compares file uri and sha, and
    a REDUNDANT stamp (127 of the bootstrap's 128) leaves the projected lifecycle to another stamp of the run -- so ``replica land``
    would have published a generation with a permanently refused live fact. ``equal`` now requires ``record_refusal.run_ingest == 0``
    beside the projection and the ledger: an equal build whose database then carries one refused stamp, projection and ledger untouched,
    is UNEQUAL naming the component, `build` / `land` publish nothing on it, and the blocked harvester facts are counted beside it
    (reporting only: a blocked record is the queue's to retry)."""
    root, ids = _world(tmp_path)
    head = _head(root)
    lay = RP.init_layout(str(tmp_path / "replica"), root)
    work = tmp_path / "work"
    work.mkdir()
    db = str(work / "registry.sqlite")
    res = RP.build(lay, head, db, workers=0)
    chk0 = res["check"]
    assert chk0["equal"] and chk0["components"] == {"projection": True, "ledger": True, "record_refusal.run_ingest": True, "cursor_free": True}
    assert chk0["failing"] == [] and chk0["record_refusal"]["run_ingest"] == 0 and chk0["blocked_records"] == {"harvester_facts": 0, "by_type": {}, "total": 0}
    assert "replica check: equal --" in RP.format_check(chk0) and "blocked_records: harvester facts 0, total 0 (reporting only)" in RP.format_check(chk0)
    # one refused stamp of a run that reads `ingested` from another stamp: the row the trigger refused (judged_with the code's stamp)
    stamp_uri = f"records/harvester/2026-01-01/run_ingest-{'a' * 12}.jsonl"
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO record_refusal(record_sha256, record_uri, record_type, field, reason, refused_utc, judged_with) VALUES (?,?,?,?,?,?,?)",
                 ("b" * 64, stamp_uri + "#1", "run_ingest", None, "IntegrityError: run: every *_utc of a reached lifecycle state must be non-null (trg_run_state_utc)",
                  synth.t(900), H.validator_stamp()))
    # and one blocked harvester fact waiting on evidence that never lands (the fold-order rule's quiet side): counted, not a verdict
    conn.execute("INSERT INTO blocked_records(record_sha256, record_uri, record_type, missing_parent_table, missing_parent_id, first_seen_utc, "
                 "last_checked_utc, attempts, stale, extra_json) VALUES (?,?,?,?,?,?,?,1,0,'{}')",
                 ("c" * 64, f"records/harvester/2026-01-01/run_ingest-{'c' * 12}.jsonl#1", "run_ingest", "run_state_event", f"{ids['run_ids'][0]}:judged",
                  synth.t(901), synth.t(901)))
    conn.commit()
    conn.close()
    chk = RP.check(db, head)
    assert not chk["equal"] and chk["failing"] == ["record_refusal.run_ingest"]
    assert chk["components"] == {"projection": True, "ledger": True, "record_refusal.run_ingest": False, "cursor_free": True}
    assert chk["ledger"]["equal"] and chk["replica"]["sha256"] == head["projection"]["sha256"] and "diff" not in chk
    assert chk["record_refusal"] == {"run_ingest": 1, "harvester_facts": 1, "by_type": {"run_ingest": 1}, "total": 1}
    assert chk["blocked_records"] == {"harvester_facts": 1, "by_type": {"run_ingest": 1}, "total": 1}
    text = RP.format_check(chk)
    assert text.startswith("replica check: UNEQUAL (record_refusal.run_ingest) --")
    assert "record_refusal: run_ingest 1 (must be 0), harvester facts 1, total 1" in text and "blocked_records: harvester facts 1, total 1" in text
    # the bootstrap's publication gate is the same verdict: `build` returns it unequal and the CLI's check exits 1
    import io, contextlib
    hp = str(tmp_path / "head.json.gz")
    RP.write_head(head, hp)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = RP.main(["check", "--db", db, "--expect", hp])
    assert rc == 1 and "UNEQUAL (record_refusal.run_ingest)" in buf.getvalue()


# ---- the projection ---------------------------------------------------------------------------------------------------------------
def test_projection_columns_are_pinned_to_the_frame_object_minus_ingest_seq(tmp_path):
    conn = DB.open_fresh(str(tmp_path / "fresh.sqlite"))
    try:
        cols = tuple(r[1] for r in conn.execute(f"PRAGMA table_info({RP.PROJECTION_OBJECT})"))
    finally:
        conn.close()
    assert "ingest_seq" in cols and RP.PROJECTION_EXCLUDED == ("ingest_seq",)
    assert tuple(c for c in cols if c not in RP.PROJECTION_EXCLUDED) == RP.PROJECTION_COLUMNS, (
        "the paper frame object's columns moved: bump RP.PROJECTION_VERSION and PROJECTION_COLUMNS together", cols)
    assert RP.PROJECTION_VERSION == "frame_projection/1" and len(RP.PROJECTION_COLUMNS) == 30
    assert {"lifecycle_state", "is_complete", "selection_state", "in_current_paper"} <= set(RP.PROJECTION_COLUMNS), "every content column stays"
    assert set(RP.SET_VALUED_JSON_COLUMNS) == {"comparison_ids_json", "roster_entry_ids_json", "ruling_ids_json"}


def test_projection_encoding_sorts_set_valued_arrays_and_keys_and_spells_floats_by_repr():
    cols = ("run_id", "strength_nominal", "ruling_ids_json", "coverage_detail_json", "is_complete")
    a = RP.encode_row(("r1", 0.1 + 0.2, '["b", "a"]', '{"z": 1, "a": [2, 1]}', 1), cols)
    b = RP.encode_row(("r1", 0.30000000000000004, '["a", "b"]', '{"a": [2, 1], "z": 1}', True), cols)
    assert a == b == ["r1", ["float", "0.30000000000000004"], '["a","b"]', '{"a":[2,1],"z":1}', 1]
    assert RP.encode_value("coverage_detail_json", "not json") == "not json" and RP.encode_value("x", None) is None
    assert RP.encode_value("x", b"\x01") == ["bytes", "01"]


def test_identical_content_in_another_file_order_projects_to_one_sha_and_the_diff_is_empty(tmp_path):
    root, _ = _world(tmp_path)
    copy = str(tmp_path / "copy")
    files = list(R.iter_record_files(os.path.join(root, "records")))
    random.Random(3).shuffle(files)
    for f in files:
        dest = os.path.join(copy, "records", os.path.relpath(f, os.path.join(root, "records")))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(f, dest)
    out = str(tmp_path / "rebuilt.sqlite")
    H.rebuild(copy, out, workers=0)
    with DB.reading(DB.db_path(root)) as a, DB.reading(out) as b:
        pa, pb = RP.frame_projection(a), RP.frame_projection(b)
    assert pa["sha256"] == pb["sha256"] and pa["counts"] == pb["counts"] and pa["n_rows"] == 3
    d = RP.projection_diff(pa, pb)
    assert d["sha_equal"] and not d["columns"] and d["n_only_in_a"] == d["n_only_in_b"] == 0


def test_check_reports_a_per_column_difference_and_the_cli_exits_1(tmp_path, capsys):
    root, _ = _world(tmp_path)
    head = _head(root)
    out = str(tmp_path / "tampered.sqlite")
    RP._copy_db(DB.db_path(root), out)
    c = DB.connect(out)                                   # this code's connection: a bare sqlite3.connect has no registry_fold_order() and the fence refuses the write
    c.execute("UPDATE run SET campaign = 'tampered' WHERE run_id = (SELECT MIN(run_id) FROM run)")
    c.commit()
    c.close()
    chk = RP.check(out, head)
    assert not chk["equal"] and chk["ledger"]["equal"] and chk["columns_equal"]
    assert set(chk["diff"]["columns"]) == {"campaign"} and chk["diff"]["columns"]["campaign"]["rows"] == 1
    assert chk["diff"]["columns"]["campaign"]["examples"][0][1:] == ["synthcampaign", "tampered"]
    text = RP.format_check(chk)
    assert text.startswith("replica check: UNEQUAL") and "column campaign: 1 row(s) differ" in text
    hp = str(tmp_path / "head.json.gz")
    RP.write_head(head, hp)
    from sisteer.registry import cli
    assert cli.main(["replica", "check", "--db", out, "--expect", hp]) == 1
    assert "column campaign" in capsys.readouterr().out
    assert cli.main(["replica", "check", "--db", DB.db_path(root), "--expect", hp]) == 0
    assert capsys.readouterr().out.startswith("replica check: equal")


# ---- the head ---------------------------------------------------------------------------------------------------------------------
def test_head_lists_every_delivered_file_with_its_status_and_the_projection_counts(tmp_path):
    root, _ = _world(tmp_path)
    head = _head(root)
    db = DB.db_path(root)
    delivered = _q(db, "SELECT file_uri, file_sha256, status FROM record_delivery ORDER BY file_uri")
    assert head["files"] == delivered and head["n_files"] == len(delivered) == sum(head["by_status"].values())
    assert head["files"] == sorted(head["files"]) and head["cursor_free"] is True and head["polls"] == 0
    assert head["allowlist_sha256"] == RP.allowlist_sha256(delivered)
    assert head["ledger"]["n_ingest"] == _q(db, "SELECT COUNT(*) FROM record_ingest")[0][0]
    p = head["projection"]
    assert p["version"] == "frame_projection/1" and p["columns"] == list(RP.PROJECTION_COLUMNS) and p["n_rows"] == 3 and len(p["rows"]) == 3
    assert p["counts"] == {"rows": 3, "in_current_paper": _q(db, "SELECT COUNT(*) FROM v_paper_current WHERE in_current_paper = 1")[0][0],
                           "selected": _q(db, "SELECT COUNT(*) FROM v_paper_current WHERE selection_state = 'selected'")[0][0]}
    assert head["schema_version"] == head["code_version"]
    hp = str(tmp_path / "h.json.gz")
    RP.write_head(head, hp)
    back = RP.read_head(hp)
    assert back["files"] == head["files"] and back["projection"]["sha256"] == p["sha256"] and RP.head_sha256(back) == RP.head_sha256(head)
    small = _head(root, with_rows=False)
    assert "rows" not in small["projection"] and small["projection"]["sha256"] == p["sha256"]


def test_a_real_partial_file_fold_blocks_the_head_until_the_last_window_lands_and_the_replica_follows(tmp_path, monkeypatch):
    """A plain file of six now-stamped run-scale records streamed in windows of two; the cycle is cut after its FIRST window commits:
    the window's records (a `failed` and a `cancelled` run_state) are on the live rows, `record_cursor` holds the file, and the head
    refuses by name. With a fake clock the head waits, the sleep finishes the fold, and the next poll returns a head whose projection
    differs from the first head's in lifecycle_state on two runs. The replica built from head 1 ingests head 2's increment and equals it."""
    root, ids = _world(tmp_path)
    head1 = _head(root)
    lay = _bootstrap(tmp_path, root, head1)
    recs = [_now_rec("run_state", {"run_id": ids["run_ids"][0], "state": "failed", "event_utc": R.utc_now(), "evidence_kind": "job_log"}, 1),
            _now_rec("run_state", {"run_id": ids["run_ids"][2], "state": "cancelled", "event_utc": R.utc_now(), "evidence_kind": "job_log"}, 2),
            _now_rec("run_patch", {"run_id": ids["run_ids"][1], "fields": {"job_name": "w2"}}, 3),
            _now_rec("run_patch", {"run_id": ids["run_ids"][0], "fields": {"gpu_count": 2}}, 4),
            _now_rec("run_patch", {"run_id": ids["run_ids"][2], "fields": {"job_name": "w3"}}, 5),
            _now_rec("run_patch", {"run_id": ids["run_ids"][1], "fields": {"node_hostname": "TESTHOST"}}, 6)]
    path = R.write_records(os.path.join(root, "records", "late", "000001-late.jsonl"), recs)
    uri = H.logical_uri(root, path)
    monkeypatch.setattr(H, "LARGE_FILE_BYTES", 0)                 # streamed in-process, every file
    monkeypatch.setattr(H, "ITEM_ROWS_PER_TRANSACTION", 2)        # windows of two records, one transaction each
    real = H.Ingestor.ingest_chunk
    commits = {"n": 0}

    def cut(self, chunk, now):
        if commits["n"] == 1:
            raise RuntimeError("cut at a window boundary")
        real(self, chunk, now)
        commits["n"] += 1
    monkeypatch.setattr(H.Ingestor, "ingest_chunk", cut)
    with pytest.raises(RuntimeError):
        _ingest(root)
    db = DB.db_path(root)
    assert _q(db, "SELECT file_uri, records_done FROM record_cursor") == [(uri, 2)]
    assert _q(db, "SELECT COUNT(*) FROM record_delivery WHERE file_uri = ?", uri) == [(0,)]
    assert _q(db, "SELECT lifecycle_state FROM run WHERE run_id = ?", ids["run_ids"][0]) == [("failed",)], "the first window IS on the live rows"
    with pytest.raises(RP.NoCursorFreeBoundary, match=rf"1 file\(s\) mid-fold \({uri}\)"):
        RP.take_head(db, wait_min=0)
    # the bounded wait: a fake clock advancing 10 s per read, the sleep finishing the fold (the harvester's next cycle)
    clock = {"t": 0.0}

    def tick():
        clock["t"] += 10.0
        return clock["t"]
    waits = []

    def sleep(s):
        waits.append(s)
        monkeypatch.setattr(H.Ingestor, "ingest_chunk", real)
        _ingest(root)
    head2 = RP.take_head(db, wait_min=5, poll_s=15, clock=tick, sleep=sleep)
    assert waits == [15.0] and head2["polls"] == 1 and head2["cursor_free"] and head2["waited_s"] > 0
    assert _q(db, "SELECT COUNT(*) FROM record_cursor") == [(0,)]
    assert any(f[0] == uri and f[2] == "ok" for f in head2["files"]) and not any(f[0] == uri for f in head1["files"])
    diff = RP.projection_diff(head1["projection"], head2["projection"])
    assert diff["columns"]["lifecycle_state"]["rows"] == 2 and diff["columns"]["is_complete"]["rows"] == 2, diff["columns"]
    assert head2["projection"]["sha256"] != head1["projection"]["sha256"]
    # the replica: head 1's generation, head 2's increment, equal
    work = _copy_current(tmp_path, lay, "landcopy")
    ing = RP.ingest_head(work, lay, head2, workers=0)
    assert ing["delta_files"] >= 1 and ing["unlisted_ignored"] == 0
    chk = RP.check(work, head2)
    assert chk["equal"], RP.format_check(chk)
    # the deferred bound: still mid-fold past the bound -> the refusal names the file and how long it waited
    monkeypatch.setattr(H.Ingestor, "ingest_chunk", cut)
    commits["n"] = 0
    R.write_records(os.path.join(root, "records", "late", "000002-late.jsonl"), [_now_rec("run_patch", {"run_id": ids["run_ids"][1], "fields": {"job_name": f"x{i}"}}, i) for i in range(4)])
    with pytest.raises(RuntimeError):
        _ingest(root)
    clock["t"] = 0.0
    with pytest.raises(RP.NoCursorFreeBoundary, match=r"no cursor-free boundary within 0.5 min: 1 file\(s\) mid-fold"):
        RP.take_head(db, wait_min=0.5, poll_s=1, clock=tick, sleep=lambda s: None)


def test_every_delivery_status_enters_the_allowlist_and_the_replay_reproduces_it(tmp_path):
    """A file with one malformed record beside a sound one is delivered `partial`; a run_state for an unknown run is `blocked`; both are
    allowlisted and the bootstrap reproduces their deliveries and their rows."""
    root, ids = _world(tmp_path)
    good = _now_rec("run_patch", {"run_id": ids["run_ids"][1], "fields": {"job_name": "mixed"}}, 1)
    bad = _now_rec("run_patch", {"run_id": ids["run_ids"][1], "fields": {"not_a_field": 1}}, 2)
    R.write_records(os.path.join(root, "records", "mixed", "000001-mixed.jsonl"), [good, bad])
    ghost = I.run_id(synth.CAMPAIGN, "ghost", "none", synth.EVAL_ID, synth.INSTRUMENT_ID)      # a well-formed id with no run row: blocked on it
    R.write_records(os.path.join(root, "records", "orphan", "000001-orphan.jsonl"),
                    [_now_rec("run_state", {"run_id": ghost, "state": "running", "event_utc": R.utc_now(), "evidence_kind": "job_log"}, 3)])
    rep = _ingest(root)
    assert rep.refused >= 1 and rep.blocked >= 1
    db = DB.db_path(root)
    statuses = dict(_q(db, "SELECT file_uri, status FROM record_delivery WHERE file_uri LIKE 'records/mixed/%' OR file_uri LIKE 'records/orphan/%'"))
    assert set(statuses.values()) == {"partial", "blocked"}, statuses
    head = _head(root)
    assert head["by_status"].get("partial") == 1 and head["by_status"].get("blocked", 0) >= 1 and sum(head["by_status"].values()) == head["n_files"]
    assert {f[2] for f in head["files"] if f[0] in statuses} == {"partial", "blocked"}
    lay = _bootstrap(tmp_path, root, head)
    assert dict(_q(lay.db_of(lay.current()), "SELECT file_uri, status FROM record_delivery WHERE file_uri LIKE 'records/mixed/%' OR file_uri LIKE 'records/orphan/%'")) == statuses
    assert _q(lay.db_of(lay.current()), "SELECT job_name FROM run WHERE run_id = ?", ids["run_ids"][1]) == [("mixed",)]


# ---- the allowlist ----------------------------------------------------------------------------------------------------------------
def test_an_unlisted_plain_file_and_an_unlisted_file_inside_a_listed_dimension_directory_change_nothing(tmp_path):
    root, ids = _world(tmp_path)
    head = _head(root)
    # written AFTER the head: a plain file and a second file inside the (listed) dimension revision directory
    R.write_records(os.path.join(root, "records", "late", "000001-late.jsonl"),
                    [_now_rec("run_state", {"run_id": ids["run_ids"][0], "state": "failed", "event_utc": R.utc_now(), "evidence_kind": "job_log"}, 1)])
    extra_model = synth.rec("model", {"model_id": "unlisted_model", "family": "qwen35", "size_label": "4b", "checkpoint_kind": "instruct",
                                      "hf_repo": "synthetic/unlisted-4b", "hf_revision": "e" * 40, "chat_template_sha256": synth.SHA256_A,
                                      "generation_config_present": 1, "generation_config_json": {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
                                      "thinking_channel": "available", "paper_status": "in_paper"}, synth.t(999))
    R.write_records(os.path.join(root, "records", "dimensions", "seed", "000002-extra.jsonl"), [extra_model])
    out = str(tmp_path / "replica.sqlite")
    rep = H.rebuild(root, out, workers=0, files=[f[0] for f in head["files"]])
    assert rep.files <= head["n_files"] and rep.unlisted_ignored == 0
    assert _q(out, "SELECT COUNT(*) FROM model WHERE model_id = 'unlisted_model'") == [(0,)]
    assert _q(out, "SELECT COUNT(*) FROM record_delivery WHERE file_uri LIKE 'records/late/%' OR file_uri LIKE '%000002-extra%'") == [(0,)]
    assert _q(out, "SELECT lifecycle_state FROM run WHERE run_id = ?", ids["run_ids"][0]) == [("ingested",)]
    chk = RP.check(out, head)
    assert chk["equal"], RP.format_check(chk)
    # the whole-disk replay DOES take them: the allowlist is what made the difference
    out2 = str(tmp_path / "whole.sqlite")
    H.rebuild(root, out2, workers=0)
    assert _q(out2, "SELECT COUNT(*) FROM model WHERE model_id = 'unlisted_model'") == [(1,)]
    # a listed file not on disk refuses the replay by name before anything is written
    with pytest.raises(FileNotFoundError, match="1 allowlisted file"):
        H.rebuild(root, str(tmp_path / "x.sqlite"), workers=0, files=[f[0] for f in head["files"]] + ["records/ghost/000001-ghost.jsonl"])
    assert not os.path.exists(str(tmp_path / "x.sqlite")) and not os.path.exists(str(tmp_path / "x.sqlite") + DB.REBUILD_TMP_SUFFIX)


def test_an_unlisted_batch_file_inside_a_listed_revision_directory_is_ignored_and_counted(tmp_path):
    root, rev_dir = _revision_world(tmp_path)
    head = _head(root)
    listed = sorted(os.path.basename(f) for f in os.listdir(rev_dir))
    assert any("source_revision" in f for f in listed) and any("item_result" in f for f in listed)
    # a batch file dropped into the published directory after the head: a different metric value
    rid = _q(DB.db_path(root), "SELECT run_id FROM run_metric")[0][0]
    jid = _q(DB.db_path(root), "SELECT judgment_id FROM run_metric")[0][0]
    extra = synth.rec("run_metric", {"run_id": rid, "judgment_id": jid, "metric_id": "asr", "threshold": 0.5, "value": 0.99, "k": 3, "n": 3,
                                     "computed_by": "registry"}, R.utc_now())
    extra_path = R.write_records(os.path.join(rev_dir, "000099-run_metric.jsonl"), [extra])
    skipped: list = []
    files = H.revision_dir_files(root, rev_dir, only={f[0] for f in head["files"]}, unlisted=skipped)
    assert skipped == [H.logical_uri(root, extra_path)] and all(p != extra_path for p, _ in files) and len(files) == len(listed)
    assert len(H.revision_dir_files(root, rev_dir)) == len(listed) + 1, "without a list the whole directory"
    out = str(tmp_path / "replica.sqlite")
    rep = H.rebuild(root, out, workers=0, files=[f[0] for f in head["files"]])
    assert rep.unlisted_ignored == 1 and rep.unlisted_sample == [H.logical_uri(root, extra_path)]
    assert _q(out, "SELECT value FROM run_metric WHERE run_id = ?", rid) == _q(DB.db_path(root), "SELECT value FROM run_metric WHERE run_id = ?", rid)
    chk = RP.check(out, head)
    assert chk["equal"], RP.format_check(chk)
    # the increment path counts it the same way (the directory is re-listed when its marker is in the delta)
    lay = RP.init_layout(str(tmp_path / "replica"), root)
    RP.publish(lay, out, head, mode="build", parent=None)
    _ingest(root)                                  # the live registry folds the extra file now
    head2 = _head(root)
    assert any(f[0] == H.logical_uri(root, extra_path) for f in head2["files"])
    work = _copy_current(tmp_path, lay, "inc")
    ing = RP.ingest_head(work, lay, head2, workers=0)
    assert ing["delta_files"] >= 1 and RP.check(work, head2)["equal"]


def test_a_replica_ahead_of_the_fold_refuses_by_name_and_an_older_head_is_unequal(tmp_path):
    """The out-of-order landing: a head older than the replica's generation.

    The fixture GUARANTEES ``arrival_utc == first_written_utc`` for the late file: the cycle that folds it reads one clock
    (``ingest_once(now=...)``), pinned here to the late record's own stamp, so the file arrives at the very millisecond of its
    stamp -- the designed no-fact case (:class:`harvest.Ranker`) -- ``derive_file_arrival_records`` writes no ``file_arrival``
    fact, and the second head names exactly ONE file the first does not. Left to the wall clock the cycle's reading fell a
    millisecond past the stamp often enough to derive a fact: a second unnamed file, and a refusal counting two."""
    root, ids = _world(tmp_path)
    head1 = _head(root)
    late = _now_rec("run_patch", {"run_id": ids["run_ids"][1], "fields": {"job_name": "later"}}, 1)
    R.write_records(os.path.join(root, "records", "late", "000001-late.jsonl"), [late])
    _ingest(root, now=late["written_utc"])
    assert _q(DB.db_path(root), "SELECT arrival_utc = first_written_utc, (SELECT COUNT(*) FROM file_arrival f WHERE f.file_uri = i.file_uri) "
              "FROM record_ingest i WHERE i.file_uri = 'records/late/000001-late.jsonl'") == [(1, 0)], "the fixture's guarantee: arrival at the stamp, no fact"
    head2 = _head(root)
    lay = _bootstrap(tmp_path, root, head2)
    work = _copy_current(tmp_path, lay, "w")
    with pytest.raises(RP.ReplicaAhead, match=r"replica ahead of the fold: 1 delivered file\(s\) the head does not name \(records/late/000001-late.jsonl\)"):
        RP.ingest_head(work, lay, head1, workers=0)
    chk = RP.check(work, head1)
    assert not chk["equal"] and not chk["ledger"]["equal"] and chk["ledger"]["extra_in_replica"] == ["records/late/000001-late.jsonl"]
    assert "1 file(s) only in the replica" in RP.format_check(chk)
    # the same head again is idempotent: nothing to fold, still equal
    ing = RP.ingest_head(work, lay, head2, workers=0)
    assert ing["delta_files"] == 0 and RP.check(work, head2)["equal"]
    # a listed file the mirror lacks refuses before any fold
    lay2 = RP.init_layout(str(tmp_path / "replica2"), str(tmp_path / "emptylog"))
    v = RP.verify_mirror(lay2, head2)
    assert not v["ok"] and v["n_missing"] == head2["n_files"]
    with pytest.raises(RP.MirrorBehind, match="mirror behind the head"):
        RP.require_mirror(lay2, head2)
    fresh = str(tmp_path / "fresh.sqlite")
    DB.open_fresh(fresh).close()                 # everything is delta for an empty replica, and nothing is on the empty log
    with pytest.raises(RP.MirrorBehind, match=rf"mirror behind the head by {head2['n_files']} file\(s\)"):
        RP.ingest_head(fresh, lay2, head2, workers=0)


def test_verify_mirror_hashes_every_listed_file_or_trusts_the_verified_manifest(tmp_path):
    root, _ = _world(tmp_path)
    head = _head(root)
    lay = RP.init_layout(str(tmp_path / "replica"), root)
    v = RP.verify_mirror(lay, head, "all", threads=2)
    assert v["ok"] and v["hashed"] == head["n_files"] and v["from_manifest"] == 0
    # a manifest naming every file with its sha: nothing hashed; a differing sha in the manifest falls back to hashing
    mdir = Path(root, "manifests")
    mdir.mkdir(exist_ok=True)
    lines = [f"{sha}  {uri}\n" for uri, sha, _ in head["files"]]
    lines[0] = f"{'0' * 64}  {head['files'][0][0]}\n"
    (mdir / "CHECKSUMS-000000000001.sha256").write_text("".join(lines))
    R.write_pointer(str(mdir / "CHECKSUMS.CURRENT"), "CHECKSUMS-000000000001.sha256")
    v = RP.verify_mirror(lay, head, "manifest", threads=2)
    assert v["ok"] and v["from_manifest"] == head["n_files"] - 1 and v["hashed"] == 1
    # a corrupted file on the mirror is a differing sha
    uri = head["files"][1][0]
    p = Path(root, uri)
    p.write_text(p.read_text() + "\n")
    v = RP.verify_mirror(lay, head, "all", threads=2)
    assert not v["ok"] and v["n_differing"] == 1 and v["differing"][0]["file_uri"] == uri


# ---- generations -----------------------------------------------------------------------------------------------------------------
def test_publish_is_a_compare_and_swap_keeps_three_generations_and_rollback_moves_current_to_the_parent(tmp_path):
    root, ids = _world(tmp_path)
    head = _head(root)
    lay = _bootstrap(tmp_path, root, head)
    g1 = lay.current()
    assert lay.meta(g1)["parent_gen_id"] is None and lay.meta(g1)["mode"] == "build" and lay.meta(g1)["built_by_tree"] == "a" * 40
    db = _copy_current(tmp_path, lay, "c")
    m2 = RP.publish(lay, db, head, mode="ingest", parent="current")
    g2 = m2["gen_id"]
    assert lay.current() == g2 and lay.meta(g2)["parent_gen_id"] == g1 and lay.generations() == [g1, g2]
    # a stale parent: the writer built from g1 while g2 was published
    with pytest.raises(RP.PublishRefused, match=f"CURRENT names '{g2}', not the parent '{g1}'"):
        RP.publish(lay, db, head, mode="ingest", parent=g1)
    assert lay.generations() == [g1, g2] and len(RP.status(lay)["unpublished"]) == 1
    # a head whose allowlist is not a superset of the parent's
    smaller = dict(head, files=head["files"][1:], n_files=head["n_files"] - 1, allowlist_sha256=RP.allowlist_sha256(head["files"][1:]))
    with pytest.raises(RP.PublishRefused, match="lacks 1 file"):
        RP.publish(lay, db, smaller, mode="ingest", parent="current")
    # the last three kept, CURRENT never pruned
    m3 = RP.publish(lay, db, head, mode="ingest", parent="current")
    m4 = RP.publish(lay, db, head, mode="ingest", parent="current")
    assert m4["pruned"] == [g1] and lay.generations() == [g2, m3["gen_id"], m4["gen_id"]] and not os.path.exists(lay.gen(g1))
    assert lay.current() == m4["gen_id"]
    # rollback: CURRENT -> the parent (m3), then m3's parent (g2); g2's parent g1 is gone -> refused
    assert RP.rollback(lay)["to"] == m3["gen_id"] and lay.current() == m3["gen_id"]
    assert RP.rollback(lay)["to"] == g2 and lay.current() == g2
    with pytest.raises(RP.PublishRefused, match="not a kept generation"):
        RP.rollback(lay)
    st = RP.status(lay)
    assert st["current"] == g2 and [g["gen_id"] for g in st["generations"]] == [g2, m3["gen_id"], m4["gen_id"]]
    # the layout: records / rulings link onto the log, config.toml names it, init is idempotent
    assert os.path.islink(os.path.join(lay.root, "records")) and os.path.realpath(os.path.join(lay.root, "records")) == os.path.realpath(os.path.join(root, "records"))
    assert lay.log_root() == os.path.abspath(root)
    assert RP.init_layout(lay.root, root).root == lay.root


def test_migrate_republishes_a_copy_under_the_parents_head_and_refuses_a_shape_no_migration_expresses(tmp_path, monkeypatch):
    root, _ = _world(tmp_path)
    head = _head(root)
    lay = _bootstrap(tmp_path, root, head)
    g1 = lay.current()
    res = RP.migrate(lay, str(tmp_path / "mw"), built_by_tree="b" * 40)
    assert res["applied"] == [] and res["from"] == res["to"] == head["schema_version"] and res["parent"] == g1
    g2 = lay.current()
    assert g2 == res["generation"] and lay.meta(g2)["mode"] == "migrate" and lay.meta(g2)["parent_gen_id"] == g1
    assert lay.meta(g2)["head"]["allowlist_sha256"] == head["allowlist_sha256"]
    monkeypatch.setattr(DB, "schema_drift", lambda conn: ["table run: a column the code declares is missing"])
    with pytest.raises(RP.MigrationNeedsRebuild, match="fall back to a full rebuild"):
        RP.migrate(lay, str(tmp_path / "mw2"))
    assert lay.current() == g2, "a refused migration publishes nothing"


def test_refold_compare_with_one_tree_on_both_sides_finds_no_difference_and_moves_no_projection(tmp_path):
    root, _ = _world(tmp_path)
    head = _head(root)
    lay = _bootstrap(tmp_path, root, head)
    db = _copy_current(tmp_path, lay, "rc")
    with DB.reading(db) as conn:
        dump = RP.refold_dump(conn)
        rows = {str(r[0]): dict(r) for r in conn.execute("SELECT * FROM run")}
    assert len(dump) == 3 and all(d["n_events"] >= 5 for d in dump)
    for d in dump:
        rid = json.loads(d["key"])["run_id"]
        assert d["state"]["lifecycle_state"] == rows[rid]["lifecycle_state"] and d["state"]["campaign"] == rows[rid]["campaign"]
    assert RP.compare_dumps(dump, dump)["n_differ"] == 0
    rec = RP.refold_compare(db, str(REPO), str(REPO), str(tmp_path / "out"), python=PY)
    assert rec["rows"]["n_differ"] == 0 and rec["n_applied"] == 0 and rec["projection"]["differs"] is False
    assert rec["projection"]["before"]["sha256"] == rec["projection"]["after"]["sha256"] == head["projection"]["sha256"]
    rec_file = json.load(open(tmp_path / "out" / "refold_compare.json"))
    assert rec_file["schema"] == "refold_compare/2" and rec["verdict"]["pass"] is None and "no --expect-head-hash" in rec["verdict"]["reason"]
    assert rec["projection"]["as_built_order"] == "new" and rec["verdict"]["components"]["old_tree_projection_agrees"] is True
    # refold-apply rewrites the live row from the fold (here: a no-op by content)
    conn = DB.connect(db)
    try:
        n = RP.refold_apply(conn, [d["key"] for d in dump])
    finally:
        conn.close()
    # the tail re-stamps every rewritten row: the rewrite drops the derived decode stamp from field_versions_json and stamp_run_rows restores it
    assert n == 3 and RP.LAST_RESTAMPED == 3 and RP.check(db, head)["equal"]



# ---- the C1 certification: refold-compare over a kept database ------- --------------------------------------------------------
def _tree(tmp_path: Path, name: str, pre_tie_break: bool, patch=None) -> tuple:
    """A committed, clean git checkout standing in for an exported tree (tree_export.sh): every file of the repo's package linked under
    ``src/sisteer``; with ``pre_tie_break`` the ``registry/reducer.py`` is a patched copy ordering run-scale events by the PRE-C1 key
    ``(written_utc, record_sha256)`` -- the order's one spelling rewritten; ``patch(tree)``, when given, edits the tree before the commit.
    Returns ``(path, HEAD sha)``."""
    tree = tmp_path / name
    pkg = tree / "src" / "sisteer"
    shutil.copytree(REPO / "src" / "sisteer", pkg, copy_function=os.symlink, ignore=shutil.ignore_patterns("__pycache__"))
    if patch is not None:
        patch(tree)
    if pre_tie_break:
        red = pkg / "registry" / "reducer.py"
        text = (REPO / "src" / "sisteer" / "registry" / "reducer.py").read_text()
        patched = text.replace('ORDER_SQL = f"written_utc, {TIE_RANK_SQL}, record_sha256"', 'ORDER_SQL = "written_utc, record_sha256"')
        patched = patched.replace('ORDER_DESC_SQL = f"written_utc DESC, {TIE_RANK_SQL} DESC, record_sha256 DESC"', 'ORDER_DESC_SQL = "written_utc DESC, record_sha256 DESC"')
        assert patched != text and 'ORDER_SQL = "written_utc, record_sha256"' in patched and 'ORDER_DESC_SQL = "written_utc DESC, record_sha256 DESC"' in patched
        red.unlink()
        red.write_text(patched)
        rep = pkg / "registry" / "replica.py"
        rtext = (REPO / "src" / "sisteer" / "registry" / "replica.py").read_text()
        # the tail line lives in `_refold_rows` (the one fold writer the restamp-keys plan and the apply share) at four spaces
        rpatched = rtext.replace("    LAST_RESTAMPED = _restamp_rewritten(conn, rewritten) if table == \"run\" else 0\n", "    LAST_RESTAMPED = 0\n")
        assert rpatched != rtext, "the pre-tail refold-apply of the exported pre-C1 tree (87d9a84e1): rewrites without the stamp tail"
        rep.unlink()
        rep.write_text(rpatched)
    (tree / ".gitignore").write_text("__pycache__/\nout/\n*.local.py\n")      # the repo ignores out/ too: a snapshot there borrows nothing
    git = ["git", "-C", str(tree), "-c", "user.name=tree", "-c", "user.email=tree@example.invalid", "-c", "commit.gpgsign=false"]
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "tree"], check=True)
    sha = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert len(sha) == 40 and subprocess.run(git + ["status", "--porcelain"], capture_output=True, text=True).stdout == ""
    return tree, sha


def _pre_tie_break_tree(tmp_path: Path) -> Path:
    return _tree(tmp_path, "old-tree", True)[0]


def _same_stamp_draw(rid: str, stamp: str, lateral: str):
    """A ``judged`` and a ``<lateral>`` run_state at ONE written_utc whose shas put ``judged`` LAST under ``(written_utc, record_sha256)``,
    so the pre-C1 order drew ``judged`` and C1's order folds the lateral state."""
    judged = synth.rec("run_state", {"run_id": rid, "state": "judged", "event_utc": stamp, "evidence_kind": "completion_artifact"}, stamp)
    for n in range(512):
        lat = synth.rec("run_state", {"run_id": rid, "state": lateral, "event_utc": stamp, "evidence_kind": "ruling",
                                      "note": f"a later run of the same key ({n})"}, stamp)
        if R.record_sha(lat) < R.record_sha(judged):
            return judged, lat
    raise AssertionError("no adversarial note in 512 tries")


def _pair_world(tmp_path: Path, lateral: str = "superseded", split_files: bool = False):
    """The starting shape: a registry whose one run carries a same-stamp ``judged`` + ``<lateral>`` pair folded LIVE under the pre-C1
    order (the sha draw: ``judged``), the head taken there (the OLD-order projection), and the NEW-order rebuild of its log -- a kept
    database that is UNEQUAL to the head by exactly the pair's draw. ``split_files`` puts the two events of the pair in DISTINCT record
    files (``000002-judged.jsonl``, ``000003-lateral.jsonl``), so the final event's CARRIER moves with the order too.
    Returns ``(root, run_id, head, kept_db)``."""
    from sisteer.registry import reducer as RD
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    # the producer's run record carries the placeholder decode_policy_match = no_policy (decode_match.py); the fold writes it and the
    # post-fold stamp derives the real value (here `unrecorded`: no decode_policy dimension), so a refold without the tail is visible
    rid, recs = synth.run_records("k", 100, through="generated", decode_policy_match="no_policy")
    R.write_records(os.path.join(root, "records", "c", "000001-run.jsonl"), recs)
    _ingest(root, derive_facts=False)
    assert _q(DB.db_path(root), "SELECT decode_policy_match FROM run WHERE run_id = ?", rid) == [("unrecorded",)], "the derived stamp"
    stamp = synth.t(300)
    judged, lat = _same_stamp_draw(rid, stamp, lateral)
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    judgment = synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID, "started_utc": stamp, "finished_utc": stamp,
                                      "n_items_judged": 200, "status": "complete", "api_pool": "lowprio"}, stamp)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(RD, "ORDER_SQL", "written_utc, record_sha256")
        mp.setattr(RD, "ORDER_DESC_SQL", "written_utc DESC, record_sha256 DESC")
        if split_files:
            R.write_records(os.path.join(root, "records", "c", "000002-judged.jsonl"), [judgment, judged])
            R.write_records(os.path.join(root, "records", "c", "000003-lateral.jsonl"), [lat])
        else:
            R.write_records(os.path.join(root, "records", "c", "000002-pair.jsonl"), [judgment, judged, lat])
        rep = _ingest(root, derive_facts=False)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    assert _q(DB.db_path(root), "SELECT lifecycle_state FROM run WHERE run_id = ?", rid) == [("judged",)], "the old order's draw"
    head = _head(root)
    kept = str(tmp_path / "kept.sqlite")
    H.rebuild(root, kept, workers=0)
    assert _q(kept, "SELECT lifecycle_state FROM run WHERE run_id = ?", rid) == [(lateral,)], "the new order's fold"
    return root, rid, head, kept


def _sha_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _snapshot(root: str) -> dict:
    """Every path under ``root`` with what identifies its content: regular files by size + sha256, symlinks by target."""
    out = {}
    for dp, dns, fns in os.walk(root):
        for name in dns + fns:
            p = os.path.join(dp, name)
            rel = os.path.relpath(p, root)
            if os.path.islink(p):
                out[rel] = ("link", os.readlink(p))
            elif os.path.isfile(p):
                out[rel] = ("file", os.path.getsize(p), _sha_file(p))
            else:
                out[rel] = ("dir",)
    return out


def _certification(head, kept_sha, old_sha, new_sha, new_hash, keys, **over) -> dict:
    """The full expectation set a certification must declare; ``over`` replaces entries."""
    out = {"expect_head_hash": head["projection"]["sha256"], "expect_input_sha256": kept_sha, "expect_as_built": "new", "expect_old_tree": old_sha,
           "expect_new_tree": new_sha, "expect_new_hash": new_hash, "expect_keys": list(keys), "head": head}
    out.update(over)
    return out


def test_refold_compare_certifies_a_kept_database_against_the_head_folded_under_the_old_order(tmp_path):
    root, rid, head, kept = _pair_world(tmp_path)
    old_tree, old_sha = _tree(tmp_path, "old-tree", True)
    new_tree, new_sha = _tree(tmp_path, "new-tree", False)
    key = I.canonical_json({"run_id": rid})
    exp = head["projection"]["sha256"]
    # the kept database is UNEQUAL to the head as built -- the C1 delta, the starting point
    chk = RP.check(kept, head)
    assert not chk["equal"] and chk["failing"] == ["projection"] and set(chk["diff"]["columns"]) >= {"lifecycle_state", "is_complete"}, RP.format_check(chk)
    kept_sha = _sha_file(kept)
    staged = RP.stage_input(kept, str(tmp_path / "work"))
    assert staged["sha256"] == kept_sha and staged["bytes"] == os.path.getsize(kept) and staged["wal_bytes"] == 0 and staged["copy"] == str(tmp_path / "work" / "registry.sqlite")
    cert = _certification(head, kept_sha, old_sha, new_sha, chk["replica"]["sha256"], [rid])
    rec = RP.refold_compare(staged["copy"], str(old_tree), str(new_tree), str(tmp_path / "out"), python=PY, input_record=staged, **cert)
    assert rec["schema"] == "refold_compare/2" and rec["input"]["sha256"] == kept_sha and rec["input"]["path"] == kept
    assert rec["rows"]["n_differ"] == 1 and rec["rows"]["columns"] == {"lifecycle_state": 1} and rec["rows"]["run_ids"] == [rid]
    assert rec["rows"]["keys"] == [{"key": key, "columns": {"lifecycle_state": ["judged", "superseded"]}}] and rec["applied"] == {"old": 1, "new": 1}
    # the fold says `no_policy`; the OLD tree (pre-tail) rewrote it so, and the NEW tree's restamp restored the derived `unrecorded` (1 row);
    # the NEW tree's own refold-apply ran the tail itself, so the restamp after it changed nothing (0) -- and both projections round-trip
    old_states = {json.loads(l)["key"]: json.loads(l)["state"] for l in open(tmp_path / "out" / "old.jsonl")}
    assert old_states[key]["decode_policy_match"] == "no_policy" and rec["restamped"] == {"old": 1, "new": 0}
    assert rec["verdict"]["components"]["stamp_code_identical"] is True and rec["trees"]["old"]["decode_match_sha256"] == rec["trees"]["new"]["decode_match_sha256"]
    assert _q(staged["copy"], "SELECT decode_policy_match FROM run WHERE run_id = ?", rid) == [("unrecorded",)]
    facts = rec["rows"]["class"][key]
    assert facts["in_class"] and facts["pair_at_latest"] and facts["states_at_latest"] == ["judged", "superseded"] and facts["n_pair_stamps"] == 1
    assert facts["latest_run_state_utc"] == synth.t(300) and rec["rows"]["in_class"] == {key: True}
    p = rec["projection"]
    assert p["old"]["sha256"] == exp == p["old_by_old_tree"]["sha256"], "the OLD-order fold of the copy is the head"
    assert p["new"]["sha256"] == chk["replica"]["sha256"] == p["as_built"]["sha256"] and p["as_built_order"] == "new"
    assert p["differs"] and p["old"]["counts"] == p["new"]["counts"] == head["projection"]["counts"]
    # the projection moves on the mover alone, in lifecycle_state and the CASE columns that read it (is_complete, is_current -- this run has
    # no superseded_by_run_id, so a superseded lifecycle is what retires it -- and why), each counted separately
    assert p["moved_run_ids"] == [rid] and set(p["moved_columns"][rid]) == {"lifecycle_state", "is_complete", "is_current", "why"} and p["rows_on_one_side"] == []
    assert p["moved_column_counts"] == {"is_complete": 1, "is_current": 1, "lifecycle_state": 1, "why": 1}, "the derived CASE columns are counted beside lifecycle_state"
    assert set(p["moved_columns"][rid]) <= {"lifecycle_state"} | set(p["lifecycle_derived_columns"]) == {"lifecycle_state", *RP.LIFECYCLE_DERIVED_COLUMNS}
    assert p["diff"]["columns"]["lifecycle_state"]["examples"] == [[rid, "judged", "superseded"]]
    v = rec["verdict"]
    assert v["pass"] is True and v["reason"] == "PASS" and v["failing"] == [] and v["movers_out_of_class"] == [], v
    assert set(v["components"]) == {"old_order_hash_equals_expected", "movers_in_class", "projection_moves_only_movers", "counts_equal", "as_built_order_known",
                                    "old_tree_projection_agrees", "new_order_hash_equals_expected", "keys_equal_expected", "as_built_order_expected",
                                    "input_sha256_expected", "trees_bound", "stamp_code_identical"} and all(v["components"].values())
    assert v["keys"] == {"n_expected": 1, "n_movers": 1, "equal": True, "symmetric_difference": [], "only_expected": [], "only_movers": []}
    assert v["expected"] == {"head_hash": exp, "new_hash": chk["replica"]["sha256"], "n_keys": 1, "input_sha256": kept_sha, "as_built": "new", "old_tree": old_sha, "new_tree": new_sha}
    assert rec["head"]["projection_sha256"] == exp and rec["head"]["taken_utc"] == head["taken_utc"] and rec["head"]["head_sha256"] == RP.head_sha256(head)
    assert rec["head"]["allowlist_sha256"] == head["allowlist_sha256"] and rec["head"]["n_files"] == head["n_files"]
    assert rec["trees"]["old"] == dict(rec["trees"]["old"], sha=old_sha, git_head=old_sha, clean=True, status_sample=[], bound_to=old_sha, path=str(old_tree),
                                       is_toplevel=True, git_toplevel=os.path.realpath(str(old_tree)), untracked_source=[])
    # refold_apply alone is a faithful rewrite: the rewritten row carries the derived stamp again, not the fold's placeholder
    conn = DB.connect(staged["copy"])
    try:
        with DB.transaction(conn):
            conn.execute("UPDATE run SET decode_policy_match = 'no_match' WHERE run_id = ?", (rid,))      # a legal value the stamp would not derive
        assert RP.refold_apply(conn, [key]) == 1 and RP.LAST_RESTAMPED == 1
        assert tuple(conn.execute("SELECT decode_policy_match, lifecycle_state FROM run WHERE run_id = ?", (rid,)).fetchone()) == ("unrecorded", "superseded")
        assert RP.restamp_runs(conn, [key]) == 0, "idempotent once the stamp stands"
    finally:
        conn.close()
    assert rec["trees"]["new"]["bound_to"] == new_sha == rec["trees"]["new"]["sha"] and rec["trees"]["new"]["reducer_tie_rank_mentions"] > 0
    assert rec["trees"]["old"]["reducer_sha256"] != rec["trees"]["new"]["reducer_sha256"]
    assert {"dump_old_s", "dump_new_s", "projection_as_built_s", "projection_old_s", "projection_old_by_old_tree_s", "projection_new_s", "apply_old_s", "apply_new_s", "total_s"} <= set(rec["timings_s"])
    for name in ("old.jsonl", "new.jsonl", "keys.json", "projection_as_built.json", "projection_old.json", "projection_old_by_old_tree.json", "projection_new.json", "refold_compare.json"):
        assert (tmp_path / "out" / name).exists(), name
    assert json.load(open(tmp_path / "out" / "keys.json")) == [key]
    assert json.load(open(tmp_path / "out" / "refold_compare.json"))["verdict"]["pass"] is True
    assert _sha_file(kept) == kept_sha, "the input was never written"
    # the copy ends under the NEW order (old applied, then new)
    assert _q(staged["copy"], "SELECT lifecycle_state FROM run WHERE run_id = ?", rid) == [("superseded",)]
    # a copy built under the OLD order (the live file) certifies only when DECLARED as such: as_built "old" PASSES, the kept rebuild's
    # declaration ("new") FAILS on it -- the wrong database cannot stand in for the kept one
    RP._checkpoint(DB.db_path(root))
    live_sha = _sha_file(DB.db_path(root))
    staged2 = RP.stage_input(DB.db_path(root), str(tmp_path / "work2"))
    rec2 = RP.refold_compare(staged2["copy"], str(old_tree), str(new_tree), str(tmp_path / "out2"), python=PY, input_record=staged2,
                             **_certification(head, live_sha, old_sha, new_sha, chk["replica"]["sha256"], [rid], expect_as_built="old"))
    assert rec2["projection"]["as_built_order"] == "old" and rec2["projection"]["as_built"]["sha256"] == exp and rec2["rows"]["n_differ"] == 1
    assert rec2["verdict"]["pass"] is True and rec2["verdict"]["components"]["new_order_hash_equals_expected"] is True and rec2["verdict"]["keys"]["equal"] is True
    staged3 = RP.stage_input(DB.db_path(root), str(tmp_path / "work3"))
    rec3 = RP.refold_compare(staged3["copy"], str(old_tree), str(new_tree), str(tmp_path / "out3"), python=PY, input_record=staged3,
                             **_certification(head, live_sha, old_sha, new_sha, chk["replica"]["sha256"], [rid]))
    assert rec3["verdict"]["pass"] is False and rec3["verdict"]["failing"] == ["as_built_order_expected"] and rec3["projection"]["as_built_order"] == "old"


def test_refold_compare_fails_a_mover_outside_the_declared_superseded_class(tmp_path):
    """A same-stamp ``judged`` + ``withdrawn`` pair moves the fold the same way (the old order drew ``judged``) but is NOT the class C1
    is certified on: the hash equality holds and the verdict still FAILS on the class check, naming the key."""
    root, rid, head, kept = _pair_world(tmp_path, lateral="withdrawn")
    old_tree, old_sha = _tree(tmp_path, "old-tree", True)
    new_tree, new_sha = _tree(tmp_path, "new-tree", False)
    key = I.canonical_json({"run_id": rid})
    exp = head["projection"]["sha256"]
    staged = RP.stage_input(kept, str(tmp_path / "work"))
    rec = RP.refold_compare(staged["copy"], str(old_tree), str(new_tree), str(tmp_path / "out"), python=PY, input_record=staged,
                            **_certification(head, staged["sha256"], old_sha, new_sha, RP.check(kept, head)["replica"]["sha256"], [rid]))
    assert rec["rows"]["n_differ"] == 1 and rec["rows"]["keys"][0]["columns"] == {"lifecycle_state": ["judged", "withdrawn"]}
    facts = rec["rows"]["class"][key]
    assert facts["states_at_latest"] == ["judged", "withdrawn"] and facts["pair_at_latest"] is False and facts["n_pair_stamps"] == 0 and facts["in_class"] is False
    v = rec["verdict"]
    assert v["pass"] is False and v["components"]["old_order_hash_equals_expected"] is True and v["components"]["movers_in_class"] is False
    assert "movers_in_class" in v["failing"] and v["movers_out_of_class"] == [key] and v["reason"].startswith("FAIL: ")
    assert rec["projection"]["old"]["sha256"] == exp, "the equality is not what failed"
    # the class check itself, from the record facts: a pair at an EARLIER stamp than the latest run_state does not qualify either
    with DB.reading(staged["copy"]) as conn:
        assert RP.declared_superseded_class(conn, [key])[key]["in_class"] is False
    assert RP.mover_in_class({"columns": {"lifecycle_state": ["judged", "superseded"]}}, {"pair_at_latest": False}) is False
    assert RP.mover_in_class({"columns": {"lifecycle_state": ["judged", "superseded"], "notes": ["a", "b"]}}, {"pair_at_latest": True}) is False
    assert RP.mover_in_class({"columns": {"lifecycle_state": ["generated", "superseded"]}}, {"pair_at_latest": True}) is False
    assert RP.mover_in_class({"columns": {"lifecycle_state": ["judged", "superseded"]}}, {"pair_at_latest": True}) is True


def _pair_landing_world(tmp_path: Path, lateral: str = "superseded"):
    """The cutover landing's shape: a registry of TWO runs, the replica bootstrapped at a head taken BEFORE the pair, then the
    same-stamp ``judged`` + ``<lateral>`` pair of run ``k`` folded LIVE under the pre-C1 order (the draw: ``judged``) and a second head
    taken there. A landing of the second head folds the pair under C1's order (``<lateral>``) and differs from the head on run ``k``
    alone; run ``m`` is untouched by the increment. Returns ``(root, run k, run m, head1, layout, head2)``."""
    from sisteer.registry import reducer as RD
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    rid, recs = synth.run_records("k", 100, through="generated", decode_policy_match="no_policy")
    rid2, recs2 = synth.run_records("m", 200, through="generated")
    R.write_records(os.path.join(root, "records", "c", "000001-run.jsonl"), recs)
    R.write_records(os.path.join(root, "records", "c", "000002-run.jsonl"), recs2)
    _ingest(root, derive_facts=False)
    head1 = _head(root)
    lay = _bootstrap(tmp_path, root, head1)
    stamp = synth.t(300)
    judged, lat = _same_stamp_draw(rid, stamp, lateral)
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    judgment = synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID, "started_utc": stamp, "finished_utc": stamp,
                                      "n_items_judged": 200, "status": "complete", "api_pool": "lowprio"}, stamp)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(RD, "ORDER_SQL", "written_utc, record_sha256")
        mp.setattr(RD, "ORDER_DESC_SQL", "written_utc DESC, record_sha256 DESC")
        R.write_records(os.path.join(root, "records", "c", "000003-pair.jsonl"), [judgment, judged, lat])
        rep = _ingest(root, derive_facts=False)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    assert _q(DB.db_path(root), "SELECT lifecycle_state FROM run WHERE run_id = ?", rid) == [("judged",)], "the old order's draw"
    head2 = _head(root)
    return root, rid, rid2, head1, lay, head2


def test_check_classifies_the_c1_same_stamp_class_and_reads_it_false_on_any_other_difference(tmp_path):
    """Beside the strict ``equal``, ``check`` classifies its diff. A database whose ONLY difference from the head is the certified
    same-stamp ``judged`` → ``superseded`` draw reads ``c1_class`` true with the movers named; the same diff plus ``decode_policy_match``
    on one row (the cutover landing's shape), a row on one side only, a ``judged`` + ``withdrawn`` pair (outside the declared pair) and a
    head without rows all read false, each naming the failing rule; an equal check reads false too (nothing to classify). ``equal``
    itself never moves: the CLI still exits 1 on the class alone."""
    root, rid, rid2, head1, lay, head2 = _pair_landing_world(tmp_path)
    kept = str(tmp_path / "kept.sqlite")
    H.rebuild(root, kept, workers=0)
    assert _q(kept, "SELECT lifecycle_state FROM run WHERE run_id = ?", rid) == [("superseded",)], "the new order's fold"
    chk = RP.check(kept, head2)
    assert not chk["equal"] and chk["failing"] == ["projection"] and chk["c1_class"] is True, RP.format_check(chk)
    assert chk["movers"] == [rid] and chk["n_movers"] == 1 and chk["n_only_in_head"] == 0 and chk["n_only_in_replica"] == 0
    assert set(chk["columns_differing"]) == {"lifecycle_state", "is_complete", "is_current", "why"} == set(RP.C1_CLASS_COLUMNS), chk["columns_differing"]
    assert all(chk["c1_components"].values()) and chk["n_movers_in_class"] == 1 and chk["movers_out_of_class"] == [] and chk["n_movers_out_of_class"] == 0
    assert chk["diff"]["columns"]["lifecycle_state"]["examples"] == [[rid, "judged", "superseded"]] and chk["c1_class_pair"] == ["judged", "superseded"]
    text = RP.format_check(chk)
    assert text.startswith("replica check: UNEQUAL (projection)") and "c1_class: yes -- 1 mover(s), 0 outside the same-stamp pair" in text
    assert RP._check_summary(chk)["c1_class"] is True and RP._check_summary(chk)["n_movers"] == 1
    # the CLI: exit 1 still (equal is strict); --out writes the classification, the movers inline
    hp = str(tmp_path / "head2.json.gz")
    RP.write_head(head2, hp)
    from sisteer.registry import cli
    assert cli.main(["replica", "check", "--db", kept, "--expect", hp, "--out", str(tmp_path / "chk.json")]) == 1
    written = json.load(open(tmp_path / "chk.json"))
    assert written["c1_class"] is True and written["movers"] == [rid] and written["movers_truncated"] is False and written["movers_file"] is None
    # the class PLUS decode_policy_match on a row: false, the column rule fails, both columns counted, the class still present
    c = DB.connect(kept)                                  # this code's connection (the fold-order fence on run)
    c.execute("UPDATE run SET decode_policy_match = 'no_match' WHERE run_id = ?", (rid2,))
    c.commit()
    c.close()
    chk = RP.check(kept, head2)
    assert chk["c1_class"] is False and chk["c1_components"]["columns_in_class"] is False and sorted(chk["movers"]) == sorted([rid, rid2])
    assert chk["columns_differing"]["decode_policy_match"] == 1 and chk["columns_differing"]["lifecycle_state"] == 1 and chk["n_movers_in_class"] == 0
    assert "c1_class: no -- 2 mover(s)" in RP.format_check(chk) and "failing rules: columns_in_class, movers_in_class" in RP.format_check(chk)
    c = DB.connect(kept)
    c.execute("UPDATE run SET decode_policy_match = 'unrecorded' WHERE run_id = ?", (rid2,))
    c.commit()
    c.close()
    assert RP.check(kept, head2)["c1_class"] is True, "restored"
    # a row on one side only beside the class mover: false
    c = sqlite3.connect(kept)
    c.execute("PRAGMA foreign_keys = OFF")
    c.execute("DELETE FROM run WHERE run_id = ?", (rid2,))
    c.commit()
    c.close()
    chk = RP.check(kept, head2)
    assert chk["c1_class"] is False and chk["c1_components"]["no_one_sided_rows"] is False and chk["c1_components"]["counts_equal"] is False
    assert chk["n_only_in_head"] == 1 and chk["n_only_in_replica"] == 0 and chk["movers"] == [rid] and chk["n_movers_in_class"] == 0
    # a head without rows cannot be classified
    small = _head(root, with_rows=False)
    chk = RP.check(DB.db_path(root), small)
    assert chk["equal"] and chk["c1_class"] is False and chk["c1_components"]["rows_available"] is False and chk["movers"] == []
    # equal: nothing to classify
    chk = RP.check(DB.db_path(root), head2)
    assert chk["equal"] and chk["c1_class"] is False and chk["c1_components"]["movers_present"] is False and chk["movers"] == [] and chk["columns_differing"] == {}
    assert "c1_class" not in RP.format_check(chk)
    # a judged + withdrawn pair is NOT the class: false, the mover named, nothing counted in class
    root_w, rid_w, head_w, kept_w = _pair_world(tmp_path / "withdrawn", lateral="withdrawn")
    chk = RP.check(kept_w, head_w)
    assert chk["c1_class"] is False and chk["c1_components"]["movers_in_class"] is False and chk["movers"] == [rid_w] and chk["n_movers_in_class"] == 0
    # the same-stamp-pair predicate itself, over the fold's own records: the SAME lifecycle move (judged -> superseded, nothing else) is in
    # the class for run k, whose latest run_state stamp carries both states, and OUT of it for run m, whose history carries no pair --
    # the classification calls declared_superseded_class and mover_in_class, never a re-spelling of the rule
    cols = list(RP.PROJECTION_COLUMNS)
    i_lc = cols.index("lifecycle_state")

    def projection(rows):
        return {"version": RP.PROJECTION_VERSION, "columns": cols, "counts": {"rows": len(rows), "in_current_paper": 0, "selected": 0}, "rows": rows}
    with DB.reading(kept) as conn:
        base = {rid: [None] * len(cols) for rid in (rid, rid2)}
        for r in base.values():
            r[i_lc] = "judged"
        for rid_ in base:
            base[rid_][0] = rid_
        moved = {k: list(v) for k, v in base.items()}
        for r in moved.values():
            r[i_lc] = "superseded"
        comps = {"projection": False, "ledger": True, "record_refusal.run_ingest": True, "cursor_free": True}
        only_k = RP.classify_projection_diff(conn, projection([base[rid], base[rid2]]), projection([moved[rid], base[rid2]]), comps)
        assert only_k["c1_class"] is True and only_k["movers"] == [rid] and only_k["n_movers_in_class"] == 1 and only_k["movers_out_of_class"] == []
        both = RP.classify_projection_diff(conn, projection([base[rid], base[rid2]]), projection([moved[rid], moved[rid2]]), comps)
        assert both["c1_class"] is False and both["c1_components"]["columns_in_class"] is True and both["c1_components"]["movers_in_class"] is False
        assert sorted(both["movers"]) == sorted([rid, rid2]) and both["movers_out_of_class"] == [rid2] and both["n_movers_in_class"] == 1
        # the move's direction is the predicate's too: superseded -> judged (the head ahead of the replica) is out of the class
        back = RP.classify_projection_diff(conn, projection([moved[rid], base[rid2]]), projection([base[rid], base[rid2]]), comps)
        assert back["c1_class"] is False and back["movers_out_of_class"] == [rid]
        # another failing component (the ledger) refuses the class whatever the rows say
        ledger = RP.classify_projection_diff(conn, projection([base[rid], base[rid2]]), projection([moved[rid], base[rid2]]), dict(comps, ledger=False))
        assert ledger["c1_class"] is False and ledger["c1_components"]["projection_is_the_only_failing_component"] is False and ledger["n_movers_in_class"] == 1


def test_write_check_json_caps_the_inline_movers_and_writes_the_whole_list_beside(tmp_path):
    movers = [f"run{i:04d}" for i in range(RP.MOVERS_INLINE_CAP + 7)]
    out = tmp_path / "o" / "check.json"
    doc = RP.write_check_json({"equal": False, "movers": movers, "n_movers": len(movers)}, str(out))
    assert doc["movers"] == movers[:RP.MOVERS_INLINE_CAP] and doc["movers_truncated"] is True and doc["movers_file"] == str(tmp_path / "o" / "movers.json")
    assert json.load(open(out))["movers_truncated"] is True and json.load(open(tmp_path / "o" / "movers.json")) == movers
    # a list that fits is inline, and a stale movers.json beside it is removed
    doc = RP.write_check_json({"equal": False, "movers": movers[:3], "n_movers": 3}, str(out))
    assert doc["movers"] == movers[:3] and doc["movers_truncated"] is False and doc["movers_file"] is None and not (tmp_path / "o" / "movers.json").exists()


def test_land_accepts_the_c1_class_only_under_the_declaration_and_refuses_everything_else(tmp_path):
    """A landing whose check differs from the head on the certified C1 same-stamp class alone is UNEQUAL without the operator's
    declaration (no pytest, nothing published, exit 1, check.json carrying c1_class true); with it, pytest runs, the generation is
    published and the summary records ``check_accepted_as = c1_class`` with the mover count and the declaration verbatim (``check_equal``
    stays false); with it and a diff on any other column too, the landing refuses exactly as before."""
    root, rid, rid2, head1, lay, head2 = _pair_landing_world(tmp_path)
    g1 = lay.current()
    # 1. the class alone, no declaration: UNEQUAL as before
    runner = _fake_runner()
    rc = RP.land(lay, head2, str(REPO), str(tmp_path / "w1"), str(tmp_path / "o1"), PY, "probe", workers=0, runner=runner, built_by_tree="c" * 40, job_id="1",
                 log=lambda m: None)
    s = json.load(open(tmp_path / "o1" / "summary.json"))
    assert rc == 1 and s["status"] == "UNEQUAL" and s["check_equal"] is False and s["check_c1_class"] is True and s["check_accepted_as"] is None
    assert s["accept_c1_class_declared"] is False and "not declared" in s["reason"] and "LAND_ACCEPT_C1_CLASS" in s["reason"] and "published" not in s
    assert not runner.calls and lay.current() == g1
    chk = json.load(open(tmp_path / "o1" / "check.json"))
    assert chk["equal"] is False and chk["c1_class"] is True and chk["movers"] == [rid] and chk["n_movers"] == 1 and chk["movers_truncated"] is False
    assert set(chk["columns_differing"]) <= set(RP.C1_CLASS_COLUMNS) and "lifecycle_state" in chk["columns_differing"] and chk["identity"]["job_id"] == "1"
    # 2. the class alone, declared: accepted, pytest runs, published, recorded
    logs = []
    runner = _fake_runner()
    rc = RP.land(lay, head2, str(REPO), str(tmp_path / "w2"), str(tmp_path / "o2"), PY, "probe", workers=0, runner=runner, built_by_tree="c" * 40, job_id="2",
                 log=logs.append, accept_c1_class=C1_DECLARATION)
    s = json.load(open(tmp_path / "o2" / "summary.json"))
    assert rc == 0 and s["status"] == "PASS" and s["check_accepted_as"] == "c1_class" and s["check_equal"] is False and s["check_c1_class"] is True
    assert s["c1_class_movers"] == 1 and s["c1_class_declaration"] == C1_DECLARATION and s["accept_c1_class_declared"] is True
    assert runner.calls and s["published"] == lay.current() != g1 and lay.meta(lay.current())["job_id"] == "2"
    assert json.load(open(tmp_path / "o2" / "check.json"))["c1_class"] is True and json.load(open(tmp_path / "o2" / "check.json"))["equal"] is False
    assert any(f"check ACCEPTED as equal modulo the C1 same-stamp class (1 movers) by declaration: {C1_DECLARATION}" in m for m in logs), logs
    assert _q(lay.db_of(lay.current()), "SELECT lifecycle_state FROM run WHERE run_id = ?", rid) == [("superseded",)], "the published generation folds C1's order"
    # 3. declared, but another column differs too (run m's campaign tampered in the generation the landing copies): refused, nothing published
    lay3 = _bootstrap(tmp_path, root, head1, name="replica3")
    g3 = lay3.current()
    c = DB.connect(lay3.db_of(g3))                        # this code's connection (the fold-order fence on run refuses a bare sqlite3.connect)
    c.execute("UPDATE run SET campaign = 'tampered' WHERE run_id = ?", (rid2,))
    c.commit()
    c.close()
    runner = _fake_runner()
    rc = RP.land(lay3, head2, str(REPO), str(tmp_path / "w3"), str(tmp_path / "o3"), PY, "probe", workers=0, runner=runner, built_by_tree="c" * 40, job_id="3",
                 log=lambda m: None, accept_c1_class=C1_DECLARATION)
    s = json.load(open(tmp_path / "o3" / "summary.json"))
    assert rc == 1 and s["status"] == "UNEQUAL" and s["check_accepted_as"] is None and s["check_c1_class"] is False and s["accept_c1_class_declared"] is True
    assert "does not cover" in s["reason"] and not runner.calls and lay3.current() == g3 and "published" not in s
    chk = json.load(open(tmp_path / "o3" / "check.json"))
    assert chk["c1_class"] is False and sorted(chk["movers"]) == sorted([rid, rid2]) and chk["columns_differing"]["campaign"] == 1 and chk["columns_differing"]["lifecycle_state"] == 1
    assert chk["c1_components"]["columns_in_class"] is False
    # 4. an equal check under the declaration is recorded as equal, not as the class (a plain world: no pair anywhere)
    root4, ids4 = _world(tmp_path / "plain")
    h4a = _head(root4)
    lay4 = _bootstrap(tmp_path / "plain", root4, h4a)
    R.write_records(os.path.join(root4, "records", "late", "000001-late.jsonl"),
                    [_now_rec("run_state", {"run_id": ids4["run_ids"][0], "state": "failed", "event_utc": R.utc_now(), "evidence_kind": "job_log"}, 1)])
    _ingest(root4)
    h4b = _head(root4)
    runner = _fake_runner()
    rc = RP.land(lay4, h4b, str(REPO), str(tmp_path / "w4"), str(tmp_path / "o4"), PY, "probe", workers=0, runner=runner, built_by_tree="c" * 40, job_id="4",
                 log=lambda m: None, accept_c1_class=C1_DECLARATION)
    s = json.load(open(tmp_path / "o4" / "summary.json"))
    assert rc == 0 and s["status"] == "PASS" and s["check_accepted_as"] == "equal" and s["check_equal"] is True and s["check_c1_class"] is False
    assert s["accept_c1_class_declared"] is True and "c1_class_movers" not in s and "c1_class_declaration" not in s and s["published"] == lay4.current()
    assert json.load(open(tmp_path / "o4" / "check.json"))["c1_class"] is False


def test_refold_compare_fails_a_wrong_expected_hash_and_refuses_malformed_inputs_before_any_work(tmp_path):
    root, rid, head, kept = _pair_world(tmp_path)
    old_tree, old_sha = _tree(tmp_path, "old-tree", True)
    new_tree, new_sha = _tree(tmp_path, "new-tree", False)
    exp = head["projection"]["sha256"]
    staged = RP.stage_input(kept, str(tmp_path / "work"))
    kept_sha = staged["sha256"]
    new_hash_true = RP.check(kept, head)["replica"]["sha256"]
    cert = _certification(head, kept_sha, old_sha, new_sha, new_hash_true, [rid])
    run = lambda out, **over: RP.refold_compare(staged["copy"], str(old_tree), str(new_tree), str(tmp_path / out), python=PY, input_record=staged, **{**cert, **over})
    # the wrong database: a head whose projection is NOT the OLD-order fold of this copy (another database's fold; or the NEW-order hash
    # offered as the head's) FAILS on the equality alone, with the class and every other gate green
    wrong = hashlib.sha256(b"another fold").hexdigest()
    other_head = lambda h: dict(head, projection=dict(head["projection"], sha256=h))
    rec = run("out", expect_head_hash=wrong, head=other_head(wrong))
    assert rec["verdict"]["pass"] is False and rec["verdict"]["failing"] == ["old_order_hash_equals_expected"]
    assert rec["verdict"]["components"]["movers_in_class"] is True and rec["projection"]["old"]["sha256"] == exp
    new_hash = rec["projection"]["new"]["sha256"]
    rec = run("out-b", expect_head_hash=new_hash, head=other_head(new_hash))
    assert rec["verdict"]["pass"] is False and rec["verdict"]["failing"] == ["old_order_hash_equals_expected"], "the NEW-order hash is not the head's"
    # a declared mover set is compared as an exact set, never a subset or superset
    rec = run("out-c", expect_keys=[rid, "sdfbase_full22:other:0:0"])
    assert rec["verdict"]["pass"] is False and rec["verdict"]["failing"] == ["keys_equal_expected"]
    assert rec["verdict"]["keys"]["symmetric_difference"] == [I.canonical_json({"run_id": "sdfbase_full22:other:0:0"})] and rec["verdict"]["keys"]["only_movers"] == []
    rec = run("out-d", expect_keys=[])
    assert rec["verdict"]["pass"] is False and rec["verdict"]["keys"] == {"n_expected": 0, "n_movers": 1, "equal": False, "symmetric_difference": [I.canonical_json({"run_id": rid})],
                                                                          "only_expected": [], "only_movers": [I.canonical_json({"run_id": rid})]}
    rec = run("out-e", expect_new_hash=exp)
    assert rec["verdict"]["pass"] is False and rec["verdict"]["failing"] == ["new_order_hash_equals_expected"]
    # the declared build order: the kept rebuild is "new"; declaring "old" over it FAILS
    rec = run("out-e2", expect_as_built="old")
    assert rec["verdict"]["pass"] is False and rec["verdict"]["failing"] == ["as_built_order_expected"] and rec["projection"]["as_built_order"] == "new"
    # refused before any work: a hash prefix, a head whose projection is not the expected hash, a malformed new hash, a certification that
    # leaves its head / input / build order / trees undeclared, a wrong input sha, a wrong tree sha, a dirty tree, a tree that is no checkout
    with pytest.raises(RP.RefoldInputRefused, match="full 64 hex"):
        run("out-f", expect_head_hash=exp[:12])
    with pytest.raises(RP.RefoldInputRefused, match="two different folds"):
        run("out-f", expect_head_hash=wrong)
    with pytest.raises(RP.RefoldInputRefused, match="full 64 hex"):
        run("out-f", expect_new_hash="abc")
    with pytest.raises(RP.RefoldInputRefused, match="missing --head, --expect-input-sha256, --expect-as-built, --expect-old-tree, --expect-new-tree, --expect-new-hash, --expect-keys$"):
        RP.refold_compare(staged["copy"], str(old_tree), str(new_tree), str(tmp_path / "out-f"), python=PY, expect_head_hash=exp)
    with pytest.raises(RP.RefoldInputRefused, match="missing --expect-as-built$"):
        run("out-f", expect_as_built=None)
    with pytest.raises(RP.RefoldInputRefused, match="missing --expect-new-hash$"):
        run("out-f", expect_new_hash=None)
    with pytest.raises(RP.RefoldInputRefused, match="missing --expect-keys$"):
        run("out-f", expect_keys=None)
    with pytest.raises(RP.RefoldInputRefused, match="must be one of"):
        run("out-f", expect_as_built="live")
    with pytest.raises(RP.RefoldInputRefused, match="not the database this certification names"):
        run("out-f", expect_input_sha256=wrong)
    with pytest.raises(RP.RefoldInputRefused, match="full 40 hex"):
        run("out-f", expect_old_tree=old_sha[:8])
    with pytest.raises(RP.RefoldInputRefused, match=f"HEAD {old_sha} is not the expected {new_sha}"):
        run("out-f", expect_old_tree=new_sha)
    (old_tree / "src" / "sisteer" / "registry" / "extra_module.py").write_text("ORDER = 'anything'\n")
    with pytest.raises(RP.RefoldInputRefused, match="not clean"):
        run("out-f")
    (old_tree / "src" / "sisteer" / "registry" / "extra_module.py").unlink()
    # an IGNORED module beside the committed code reads clean to git status and is still refused: every *.py under src/ must be tracked
    (old_tree / "src" / "sisteer" / "registry" / "extra.local.py").write_text("ORDER = 'anything'\n")
    assert RP.tree_identity(str(old_tree))["clean"] is True and RP.untracked_source(str(old_tree)) == ["src/sisteer/registry/extra.local.py"]
    with pytest.raises(RP.RefoldInputRefused, match="does not track .*extra.local.py"):
        run("out-f")
    (old_tree / "src" / "sisteer" / "registry" / "extra.local.py").unlink()
    assert RP.untracked_source(str(old_tree)) == []
    # a snapshot nested under the checkout's ignored out/ borrows the parent's HEAD and clean status from `git -C`; the top-level check refuses it
    nest = old_tree / "out" / "snap"
    shutil.copytree(old_tree / "src", nest / "src", symlinks=True)
    ident = RP.tree_identity(str(nest))
    assert ident["git_head"] == old_sha and ident["clean"] is True and ident["is_toplevel"] is False and ident["git_toplevel"] == os.path.realpath(str(old_tree))
    with pytest.raises(RP.RefoldInputRefused, match="not the checkout's own top level"):
        RP.require_tree(str(nest), old_sha)
    with pytest.raises(RP.RefoldInputRefused, match="not the checkout's own top level"):
        RP.refold_compare(staged["copy"], str(nest), str(new_tree), str(tmp_path / "out-f"), python=PY, input_record=staged, **cert)
    shutil.rmtree(nest.parent)
    assert RP.require_tree(str(old_tree), old_sha)["bound_to"] == old_sha
    plain = tmp_path / "plain-tree"
    shutil.copytree(REPO / "src" / "sisteer", plain / "src" / "sisteer", copy_function=os.symlink, ignore=shutil.ignore_patterns("__pycache__"))
    with pytest.raises(RP.RefoldInputRefused, match="not a git checkout"):
        RP.refold_compare(staged["copy"], str(old_tree), str(plain), str(tmp_path / "out-f"), python=PY, input_record=staged, **{**cert, "expect_new_tree": new_sha})
    assert not (tmp_path / "out-f").exists(), "a refused expectation writes nothing"
    assert RP.RefoldInputRefused.exit_code == 13
    # a measurement with no expectation declared still runs over trees of any identity
    rec = RP.refold_compare(staged["copy"], str(old_tree), str(plain), str(tmp_path / "out-g"), python=PY)
    assert rec["verdict"]["pass"] is None and rec["trees"]["new"]["git_head"] is None and rec["trees"]["new"]["clean"] is None and rec["rows"]["n_differ"] == 1
    # a mover without a live row: refold-apply would skip it, so the certification refuses rather than projecting a copy it never rewrote
    conn = DB.connect(staged["copy"])
    try:
        conn.execute("PRAGMA foreign_keys = OFF")          # the inconsistency itself: a key in row_history with no run row (children left behind)
        with DB.transaction(conn):
            conn.execute("DELETE FROM run WHERE run_id = ?", (rid,))
    finally:
        conn.close()
    with pytest.raises(RP.RefoldInputRefused, match="rewrote 0 of 1 mover"):
        run("out-h")
    # the input: a missing file, a WAL twin holding bytes (the file alone is not the database), too little free space
    with pytest.raises(RP.RefoldInputRefused, match="no database"):
        RP.stage_input(str(tmp_path / "absent.sqlite"), str(tmp_path / "w"))
    Path(kept + "-wal").write_bytes(b"\0" * 4096)
    with pytest.raises(RP.RefoldInputRefused, match="holds 4096 bytes"):
        RP.stage_input(kept, str(tmp_path / "w"))
    os.remove(kept + "-wal")
    with mock.patch.object(RP.shutil, "disk_usage", return_value=types.SimpleNamespace(free=10)):
        with pytest.raises(RP.RefoldInputRefused, match="bytes free"):
            RP.stage_input(kept, str(tmp_path / "w"))
    assert not (tmp_path / "w" / "registry.sqlite").exists()


def test_replica_sbatch_refold_compare_over_a_kept_database_reads_no_current_and_writes_nothing_under_the_root(tmp_path):
    root, rid, head, kept = _pair_world(tmp_path)
    old_tree, old_sha = _tree(tmp_path, "old-tree", True)
    new_tree, new_sha = _tree(tmp_path, "new-tree", False)
    exp = head["projection"]["sha256"]
    hp = str(tmp_path / "head.json.gz")
    RP.write_head(head, hp)
    keys_file = tmp_path / "expected_keys.json"
    keys_file.write_text(json.dumps([rid]))
    kept_sha = _sha_file(kept)
    new_hash = RP.check(kept, head)["replica"]["sha256"]
    DECL = ["--expect-input-sha256", kept_sha, "--expect-as-built", "new", "--expect-old-tree", old_sha, "--expect-new-tree", new_sha, "--expect-new-hash", new_hash]
    # a replica root WITH a published generation, so a CURRENT exists to be left alone
    root2, _ = _world(tmp_path / "second")
    lay = _bootstrap(tmp_path / "second", root2, _head(root2))
    assert lay.current()
    before = _snapshot(lay.root)
    venv = _fake_venv(tmp_path)
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    _exe(bin_ / "srun", '#!/bin/bash\nwhile [ $# -gt 0 ]; do case "$1" in --ntasks=*|-n) shift ;; *) break ;; esac; done\nexec "$@"\n')
    env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", REPLICA_WORK_ROOT=str(tmp_path / "nodelocal"), SLURM_JOB_ID="778", SLURM_CPUS_PER_TASK="2",
               USER=os.environ.get("USER", "tester"))
    out = tmp_path / "r3b"
    argv = ["bash", str(LANDING / "replica.sbatch"), "refold-compare", "--db", kept, "--head", hp, "--expect-head-hash", exp, "--expect-keys", str(keys_file), *DECL,
            str(old_tree), str(new_tree), str(venv), str(out), lay.root]
    r = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "replica[778] mode=refold-compare" in r.stdout and "RESULT mode=refold-compare rc=0" in r.stdout
    assert f"db={kept}" in r.stdout and "generation=" not in r.stdout and f"expect_head_hash={exp}" in r.stdout and f"head={hp}" in r.stdout
    assert f"expect_input_sha256={kept_sha}" in r.stdout and "expect_as_built=new" in r.stdout and f"expect_old_tree={old_sha}" in r.stdout and f"expect_new_tree={new_sha}" in r.stdout
    assert f"expect_new_hash={new_hash}" in r.stdout and f"expect_keys={keys_file}" in r.stdout
    assert '"verdict": "PASS"' in r.stdout and f'"input_sha256": "{kept_sha}"' in r.stdout and '"as_built_order": "new"' in r.stdout
    rec = json.load(open(out / "refold_compare.json"))
    assert rec["verdict"]["pass"] is True and rec["input"]["sha256"] == kept_sha and rec["input"]["bytes"] == os.path.getsize(kept) and rec["input"]["path"] == kept
    assert rec["input"]["copy"] == f"{tmp_path}/nodelocal/778/registry.sqlite" and rec["db"] == rec["input"]["copy"]
    assert rec["rows"]["n_differ"] == 1 and rec["rows"]["run_ids"] == [rid] and rec["verdict"]["components"]["keys_equal_expected"] is True
    assert rec["projection"]["old"]["sha256"] == exp and rec["head"]["head_sha256"] == RP.head_sha256(head)
    assert rec["trees"]["old"]["bound_to"] == old_sha and rec["trees"]["new"]["bound_to"] == new_sha and rec["verdict"]["components"]["trees_bound"] is True
    assert _snapshot(lay.root) == before, "nothing under the replica root moved: CURRENT, its generation, builds, pins"
    assert _sha_file(kept) == kept_sha, "the kept database is read, never written"
    assert not (tmp_path / "nodelocal" / "778").exists(), "the node-local copy is removed by the trap"
    # FAIL is exit 1 with the record written; the guards refuse with exit 2 before any work
    r = subprocess.run(["bash", str(LANDING / "replica.sbatch"), "refold-compare", "--db", kept, "--head", hp, "--expect-head-hash", "0" * 64, *DECL,
                        str(old_tree), str(new_tree), str(venv), str(tmp_path / "r3b-fail"), lay.root], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 13 and "two different folds" in r.stderr + r.stdout, r.stdout + r.stderr    # the head file contradicts the hash: refused
    r = subprocess.run(["bash", str(LANDING / "replica.sbatch"), "refold-compare", "--db", kept, "--head", hp, "--expect-head-hash", exp, "--expect-keys", str(keys_file),
                        *DECL[:8], "--expect-new-hash", exp, str(old_tree), str(new_tree), str(venv), str(tmp_path / "r3b-fail2"), lay.root], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 1 and "RESULT mode=refold-compare rc=1" in r.stdout and '"verdict": "FAIL: new_order_hash_equals_expected"' in r.stdout, r.stdout + r.stderr
    assert json.load(open(tmp_path / "r3b-fail2" / "refold_compare.json"))["verdict"]["pass"] is False
    # a certification missing its input sha / build order / tree commits is refused (13) before the copy; nothing is written
    r = subprocess.run(["bash", str(LANDING / "replica.sbatch"), "refold-compare", "--db", kept, "--head", hp, "--expect-head-hash", exp,
                        str(old_tree), str(new_tree), str(venv), str(tmp_path / "x")], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 13 and "missing --expect-input-sha256, --expect-as-built, --expect-old-tree, --expect-new-tree, --expect-new-hash, --expect-keys" in r.stderr, r.stdout + r.stderr
    assert not (tmp_path / "nodelocal" / "778" / "registry.sqlite").exists() and not (tmp_path / "x" / "refold_compare.json").exists()
    r = subprocess.run(["bash", str(LANDING / "replica.sbatch"), "refold-compare", "--db", kept, "--head", hp, "--expect-head-hash", exp, "--expect-keys", str(keys_file),
                        *DECL[:6], "--expect-new-tree", old_sha, *DECL[8:], str(old_tree), str(new_tree), str(venv), str(tmp_path / "x")], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 13 and f"HEAD {new_sha} is not the expected {old_sha}" in r.stderr, r.stdout + r.stderr
    r = subprocess.run(["bash", str(LANDING / "replica.sbatch"), "refold-compare", "--db", kept, str(old_tree), str(new_tree), str(venv), str(tmp_path / "x")], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "--db needs --head HEAD and --expect-head-hash HASH" in r.stdout
    r = subprocess.run(["bash", str(LANDING / "replica.sbatch"), "refold-compare", "--db", str(tmp_path / "absent.sqlite"), "--head", hp, "--expect-head-hash", exp,
                        str(old_tree), str(new_tree), str(venv), str(tmp_path / "x")], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "no database at" in r.stdout
    r = subprocess.run(["bash", str(LANDING / "replica.sbatch"), "refold-compare", "--bogus", "1", str(old_tree), str(new_tree), str(venv), str(tmp_path / "x")], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "unknown refold-compare option --bogus" in r.stdout
    assert not (tmp_path / "x" / "refold_compare.json").exists()
    assert _snapshot(lay.root) == before
    # the optionless form still measures CURRENT's generation (one tree on both sides: no mover, no verdict declared)
    r = subprocess.run(["bash", str(LANDING / "replica.sbatch"), "refold-compare", str(REPO), str(REPO), str(venv), str(tmp_path / "cur"), lay.root],
                       capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"generation={lay.current()}" in r.stdout and "db=" not in r.stdout.split("refold-compare: ", 1)[1].split("\n", 1)[0]
    rec = json.load(open(tmp_path / "cur" / "refold_compare.json"))
    assert rec["rows"]["n_differ"] == 0 and rec["verdict"]["pass"] is None and rec["input"]["path"] == lay.db_of(lay.current())
    assert _snapshot(lay.root) == before
    text = (LANDING / "replica.sbatch").read_text()
    assert "--export" not in [l.strip().split("=")[0] for l in text.splitlines() if l.startswith("#SBATCH")] and "--mem=0" not in text


# ---- the landing job ---------------------------------------------------------------------------------------------------------------
def _fake_runner(skipped=0, rc=0):
    calls = []

    def runner(argv, env, cwd, log_path, timeout):
        calls.append({"argv": list(argv), "env": {k: env.get(k) for k in ("REGISTRY_LIVE_DB", "REGISTRY_PIN_L1", "PYTHONPATH")}, "cwd": cwd})
        junit = [a for a in argv if a.startswith("--junitxml=")][0].split("=", 1)[1]
        with open(junit, "w") as fh:
            fh.write(f'<testsuites><testsuite name="pytest" tests="4" failures="{1 if rc else 0}" errors="0" skipped="{skipped}"/></testsuites>')
        with open(log_path, "w") as fh:
            fh.write("fake pytest\n")
        return rc
    runner.calls = calls   # type: ignore[attr-defined]
    return runner


def test_land_runs_pytest_against_the_copy_accounts_skips_and_publishes_the_equal_generation(tmp_path):
    root, ids = _world(tmp_path)
    head1 = _head(root)
    lay = _bootstrap(tmp_path, root, head1)
    g1 = lay.current()
    R.write_records(os.path.join(root, "records", "late", "000001-late.jsonl"),
                    [_now_rec("run_state", {"run_id": ids["run_ids"][0], "state": "failed", "event_utc": R.utc_now(), "evidence_kind": "job_log"}, 1)])
    _ingest(root)
    head2 = _head(root)
    runner = _fake_runner(skipped=0)
    rc = RP.land(lay, head2, str(REPO), str(tmp_path / "w1"), str(tmp_path / "o1"), PY, "probe", expected_skipped=0, workers=0, runner=runner,
                 built_by_tree="c" * 40, job_id="4242", log=lambda m: None)
    assert rc == 0
    s = json.load(open(tmp_path / "o1" / "summary.json"))
    assert s["status"] == "PASS" and s["check_equal"] and s["pytest_rc"] == 0 and s["counts"]["skipped"] == 0 and s["parent_generation"] == g1
    assert s["published"] == lay.current() != g1 and lay.meta(lay.current())["mode"] == "ingest" and lay.meta(lay.current())["job_id"] == "4242"
    assert set(s["modules"]) == {"tests/test_registry_bon_decode_precedence.py", "tests/test_registry_instrument_padding_identity.py",
                                 "tests/test_registry_paper_spelling.py", "tests/test_registry_paper_views_contrast_current.py", "tests/test_registry_twins_live.py"}
    call = runner.calls[0]
    assert call["env"]["REGISTRY_LIVE_DB"] == str(tmp_path / "w1" / "registry.sqlite") and call["env"]["REGISTRY_PIN_L1"] == lay.pin_l1
    assert call["env"]["PYTHONPATH"].startswith(str(REPO / "src")) and call["cwd"] == str(REPO)
    assert call["argv"][:3] == [PY, "-m", "pytest"] and "-m" in call["argv"][3:] and "live_db" in call["argv"] and call["argv"][-len(s["modules"]):] == sorted(s["modules"])
    chk = json.load(open(tmp_path / "o1" / "check.json"))
    assert chk["equal"] and chk["head"]["sha12"] == head2["projection"]["sha256"][:12]
    assert json.load(open(tmp_path / "o1" / "mirror.json"))["ok"] and json.load(open(tmp_path / "o1" / "ingest.json"))["delta_files"] >= 1
    # every result names the job, the tree and the head it was produced for (the grading host accepts nothing without that identity)
    ident = {"job_id": "4242", "built_by_tree": "c" * 40, "head_sha256": RP.head_sha256(head2), "allowlist_sha256": head2["allowlist_sha256"],
             "projection_sha256": head2["projection"]["sha256"], "out_dir": str(tmp_path / "o1")}
    assert s["identity"] == ident and chk["identity"] == ident and s["published_utc"] and s["current_after_publish"] == s["published"]
    # one more skip than expected refuses (the generation is still published: it equals the fold)
    cur = lay.current()
    rc = RP.land(lay, head2, str(REPO), str(tmp_path / "w2"), str(tmp_path / "o2"), PY, "probe", expected_skipped=0, workers=0, runner=_fake_runner(skipped=1), log=lambda m: None)
    s2 = json.load(open(tmp_path / "o2" / "summary.json"))
    assert rc == 1 and s2["status"] == "SKIPS" and "1 skipped, expected 0" in s2["reason"] and s2["published"] == lay.current() != cur
    # red pytest -> FAIL; the expected skip count honored
    rc = RP.land(lay, head2, str(REPO), str(tmp_path / "w3"), str(tmp_path / "o3"), PY, "probe", expected_skipped=1, workers=0, runner=_fake_runner(skipped=1, rc=1), log=lambda m: None)
    assert rc == 1 and json.load(open(tmp_path / "o3" / "summary.json"))["status"] == "FAIL"
    # an older head: refused before any pytest, nothing published, check.json says why -- and a REUSED output directory holds only this
    # job's results afterwards (the earlier occupant's junit/pytest.log/ingest.json are removed before the job starts, so a death before
    # check.json cannot leave an older verdict in place)
    cur = lay.current()
    runner = _fake_runner()
    (tmp_path / "o4").mkdir()
    for stale in ("check.json", "summary.json", "junit.xml", "pytest.log", "ingest.json", "mirror.json"):
        (tmp_path / "o4" / stale).write_text('{"equal": true, "status": "PASS", "stale": true}')
    rc = RP.land(lay, head1, str(REPO), str(tmp_path / "w4"), str(tmp_path / "o4"), PY, "probe", workers=0, runner=runner, log=lambda m: None)
    assert rc == RP.ReplicaAhead.exit_code and not runner.calls and lay.current() == cur
    assert json.load(open(tmp_path / "o4" / "check.json"))["equal"] is False and "replica ahead" in json.load(open(tmp_path / "o4" / "summary.json"))["reason"]
    assert not any((tmp_path / "o4" / stale).exists() for stale in ("junit.xml", "pytest.log", "ingest.json"))
    assert "stale" not in json.load(open(tmp_path / "o4" / "summary.json")) and json.load(open(tmp_path / "o4" / "check.json"))["identity"]["head_sha256"] == RP.head_sha256(head1)
    # CURRENT moves between the copy and the publication (an operator's rollback while pytest ran): the CAS refuses, and the landing is
    # a FAILING status (exit 10) even though pytest passed -- the next landing would otherwise start from a generation this one never made
    before = lay.current()
    target = [g for g in lay.generations() if g != before][0]   # an older kept generation
    inner = _fake_runner(skipped=0)

    def moving(argv, env, cwd, log_path, timeout):
        R.write_pointer(lay.current_path, target)      # the operator points CURRENT back at an older generation mid-job
        return inner(argv, env, cwd, log_path, timeout)
    rc = RP.land(lay, head2, str(REPO), str(tmp_path / "w7"), str(tmp_path / "o7"), PY, "probe", expected_skipped=0, workers=0, runner=moving, job_id="4343", log=lambda m: None)
    s7 = json.load(open(tmp_path / "o7" / "summary.json"))
    assert rc == RP.PublishRefused.exit_code == 10 and s7["status"] == "PUBLISH_REFUSED" and s7["published"] is None and s7["pytest_rc"] == 0
    assert "not published" in s7["reason"] and f"CURRENT names {target!r}, not the parent {before!r}" in s7["publish_refused"] and s7["current_after_publish"] == target == lay.current()
    assert json.load(open(tmp_path / "o7" / "check.json"))["equal"] and inner.calls, "the check was equal and pytest ran; only the publication failed"
    assert glob.glob(os.path.join(lay.gen_root, "*.unpublished")), "the refused generation is left for the operator"
    R.write_pointer(lay.current_path, before)
    # a generation whose schema is not the code's refuses by name before any fold (an increment never migrates on the side)
    import sisteer.registry.db as DBM

    def drift(path):
        raise DBM.SchemaDrift(path, 21, 22, [])
    with mock.patch.object(DBM, "open_or_create", drift):
        rc = RP.land(lay, head2, str(REPO), str(tmp_path / "w6"), str(tmp_path / "o6"), PY, "probe", workers=0, runner=runner, log=lambda m: None)
    assert rc == RP.SchemaBehind.exit_code == 12 and "replica migrate" in json.load(open(tmp_path / "o6" / "summary.json"))["reason"] and lay.current() == cur
    # no generation at all
    lay2 = RP.init_layout(str(tmp_path / "replica2"), root)
    rc = RP.land(lay2, head2, str(REPO), str(tmp_path / "w5"), str(tmp_path / "o5"), PY, "probe", workers=0, runner=runner, log=lambda m: None)
    assert rc == RP.NoGeneration.exit_code and json.load(open(tmp_path / "o5" / "summary.json"))["status"] == "NO_GENERATION"


def _exe(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _fake_venv(tmp_path: Path, skipped=0, rc=0) -> Path:
    """A `.gate-ready.json` venv whose python runs the real interpreter, except `-m pytest`, which writes a junit and exits rc."""
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / ".gate-ready.json").write_text("{}")
    _exe(venv / "bin" / "python", f'''#!/bin/bash
if [ "$1" != "-m" ] || [ "$2" != "pytest" ]; then exec {PY} "$@"; fi
echo "fake pytest $*" >> "{tmp_path}/pytest.calls"
junit=""; for a in "$@"; do case "$a" in --junitxml=*) junit=${{a#--junitxml=}} ;; esac; done
printf '<testsuites><testsuite name="pytest" tests="4" failures="{1 if rc else 0}" errors="0" skipped="{skipped}"/></testsuites>' > "$junit"
echo "env REGISTRY_LIVE_DB=$REGISTRY_LIVE_DB REGISTRY_PIN_L1=$REGISTRY_PIN_L1" >> "{tmp_path}/pytest.calls"
exit {rc}
''')
    return venv


def test_replica_sbatch_takes_the_shared_root_from_the_environment_else_from_the_gate_venv(tmp_path):
    """The shared root (the default replica root, the default output directory, the cluster's hf_env.sh) is the environment's
    SISTEER_SHARED_ROOT, else its CLUSTER_SHARED_ROOT, else the root the gate venv argument sits under (<shared>/envs/<venv>,
    the layout gate_venv.sbatch writes; the venv is the argument holding .gate-ready.json). The script names no site root."""
    shared = tmp_path / "shared"
    venv = _fake_venv(shared / "envs")
    head = tmp_path / "head.json.gz"
    head.write_text("x")
    base = {k: v for k, v in os.environ.items() if k not in ("SISTEER_SHARED_ROOT", "CLUSTER_SHARED_ROOT")}
    env = dict(base, REPLICA_WORK_ROOT=str(tmp_path / "nodelocal"), SLURM_JOB_ID="778", USER=os.environ.get("USER", "tester"))
    argv = ["bash", str(LANDING / "replica.sbatch"), "land", str(REPO), str(head), str(venv), "probe"]
    r = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 2 and f"no replica root at {shared}/registry/replica " in r.stdout, r.stdout + r.stderr
    for var in ("CLUSTER_SHARED_ROOT", "SISTEER_SHARED_ROOT"):
        r = subprocess.run(argv, capture_output=True, text=True, env=dict(env, **{var: str(tmp_path / var.lower())}), timeout=120)
        assert r.returncode == 2 and f"no replica root at {tmp_path / var.lower()}/registry/replica " in r.stdout, r.stdout + r.stderr
    r = subprocess.run(argv, capture_output=True, text=True, timeout=120,
                       env=dict(env, SISTEER_SHARED_ROOT=str(tmp_path / "first"), CLUSTER_SHARED_ROOT=str(tmp_path / "second")))
    assert f"no replica root at {tmp_path / 'first'}/registry/replica " in r.stdout, r.stdout + r.stderr
    text = (LANDING / "replica.sbatch").read_text()
    code = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    assert not [ln for ln in code if "/workspace" in ln], "the shared root comes from the environment or the venv, never a literal"


def test_replica_sbatch_land_runs_the_real_module_node_local_with_srun_shimmed(tmp_path):
    root, ids = _world(tmp_path)
    head1 = _head(root)
    lay = _bootstrap(tmp_path, root, head1)
    R.write_records(os.path.join(root, "records", "late", "000001-late.jsonl"),
                    [_now_rec("run_patch", {"run_id": ids["run_ids"][1], "fields": {"job_name": "sbatch"}}, 1)])
    _ingest(root)
    head2 = _head(root)
    hp = str(tmp_path / "head.json.gz")
    RP.write_head(head2, hp)
    venv = _fake_venv(tmp_path)
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    _exe(bin_ / "srun", '#!/bin/bash\nwhile [ $# -gt 0 ]; do case "$1" in --ntasks=*|-n) shift ;; *) break ;; esac; done\nexec "$@"\n')
    _exe(bin_ / "git", '#!/bin/bash\necho ' + "d" * 40 + '\n')
    out = tmp_path / "out"
    env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", REPLICA_WORK_ROOT=str(tmp_path / "nodelocal"), SLURM_JOB_ID="777", SLURM_CPUS_PER_TASK="2", USER=os.environ.get("USER", "tester"))
    r = subprocess.run(["bash", str(LANDING / "replica.sbatch"), "land", str(REPO), hp, str(venv), "probe", lay.root, str(out), "0", "all"],
                       capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "replica[777] mode=land" in r.stdout and "RESULT mode=land rc=0" in r.stdout and "replica check: equal -- frame_projection/1" in r.stdout
    assert "[replica land] PASS rc=0" in r.stdout
    s = json.load(open(out / "summary.json"))
    assert s["status"] == "PASS" and s["published"] == lay.current() and lay.meta(lay.current())["job_id"] == "777" and lay.meta(lay.current())["built_by_tree"] == "d" * 40
    assert s["identity"]["job_id"] == "777" and s["identity"]["built_by_tree"] == "d" * 40 and s["identity"]["head_sha256"] == RP.head_sha256(head2)
    assert json.load(open(out / "check.json"))["identity"] == s["identity"]
    calls = (tmp_path / "pytest.calls").read_text()
    assert "-m live_db" in calls and f"REGISTRY_LIVE_DB={tmp_path}/nodelocal/777/registry.sqlite" in calls and f"REGISTRY_PIN_L1={lay.pin_l1}" in calls
    assert not (tmp_path / "nodelocal" / "777").exists(), "the node-local work directory is removed by the trap"
    # the guards: a missing head / tree / venv / root refuse with exit 2 before any work
    r = subprocess.run(["bash", str(LANDING / "replica.sbatch"), "land", str(REPO), str(tmp_path / "nohead"), str(venv), "probe", lay.root], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "no head file" in r.stdout
    r = subprocess.run(["bash", str(LANDING / "replica.sbatch"), "land", str(REPO), hp, str(venv), "probe", str(tmp_path / "noroot")], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "no replica root" in r.stdout
    r = subprocess.run(["bash", str(LANDING / "replica.sbatch"), "bogus"], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "unknown mode" in r.stdout
    # build mode on the same head: equal, published under the parent CURRENT
    cur = lay.current()
    r = subprocess.run(["bash", str(LANDING / "replica.sbatch"), "build", str(REPO), str(venv), hp, lay.root], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "replica build: UNEQUAL" not in r.stdout and "[replica build] published generation" in r.stdout and lay.current() != cur
    assert lay.meta(lay.current())["mode"] == "build" and lay.meta(lay.current())["parent_gen_id"] == cur
    bj = sorted(Path(lay.builds).glob("build-*/build.json"))
    assert bj and json.load(open(bj[-1]))["records_per_s"] is not None and json.load(open(bj[-1]))["peak_rss_bytes"] > 0
    text = (LANDING / "replica.sbatch").read_text()
    assert "--dependency=singleton" in text and "#SBATCH --job-name=registry-replica" in text and "--export" not in [l.strip().split("=")[0] for l in text.splitlines() if l.startswith("#SBATCH")]
    assert "--mem=0" not in text and "/home/$USER/replica-work" in text
    assert "live_db/job-${SLURM_JOB_ID:-local-$$}" in text, "a hand submission's default output directory is its own job's, never a shared one"
    # an UNEQUAL bootstrap (a head whose projection is not what the log folds to) is the STOP: exit 7, nothing published, the
    # database kept beside check.json for inspection
    bad = dict(head2)
    bad["projection"] = dict(head2["projection"], sha256="0" * 64, counts={"rows": 99, "in_current_paper": 0, "selected": 0})
    bp = str(tmp_path / "bad.json.gz")
    RP.write_head(bad, bp)
    cur = lay.current()
    r = subprocess.run(["bash", str(LANDING / "replica.sbatch"), "build", str(REPO), str(venv), bp, lay.root], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 7 and "replica build: UNEQUAL -- STOP; the work database stays at" in r.stderr + r.stdout and lay.current() == cur, r.stdout + r.stderr
    kept = sorted(Path(lay.builds).glob("build-*/registry.sqlite.unequal"))
    assert kept and "keeping the work database" in r.stdout and not json.load(open(kept[-1].parent / "check.json"))["equal"]


# ---- the kept-database modes: update and migrate --db (cutover prerequisites) ------------------------------------------------
def _kept_copy(tmp_path: Path, root: str, head: dict, name: str = "kept.sqlite") -> str:
    """A rebuild of the log at ``head`` that no generation names -- the shape of a kept rebuild."""
    kept = str(tmp_path / name)
    H.rebuild(root, kept, workers=0)
    assert RP.check(kept, head)["equal"]
    return kept


def _migrating_tree(tmp_path: Path, name: str, body: str, slug: str = "replica_test_step") -> tuple:
    """A committed tree standing in for a NEWER export: the repo's package plus ONE more migration file (the next version, ``slug``)
    whose statements are ``body``."""
    def patch(tree: Path) -> None:
        mig = tree / "src" / "sisteer" / "registry" / "migrations"
        nxt = max(int(f.name[:4]) for f in mig.iterdir() if re.match(r"^\d{4}_", f.name)) + 1
        (mig / f"{nxt:04d}_{slug}.sql").write_text(body)
    return _tree(tmp_path, name, False, patch=patch)


def _module(tree: Path, *args: str, timeout: int = 600) -> subprocess.CompletedProcess:
    """``python -m sisteer.registry.replica …`` under ``tree``'s code -- the way replica.sbatch runs it (PYTHONPATH=<tree>/src), so the
    running module lives under the tree a record binds."""
    env = dict(os.environ, PYTHONPATH=str(tree / "src"), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([PY, "-m", "sisteer.registry.replica", *args], capture_output=True, text=True, env=env, timeout=timeout)


def test_update_brings_a_kept_copy_to_a_newer_head_checks_it_and_writes_the_output_with_its_identity(tmp_path):
    root, ids = _world(tmp_path)
    head1 = _head(root)
    kept = _kept_copy(tmp_path, root, head1)
    kept_sha, kept_stat = _sha_file(kept), os.stat(kept)
    R.write_records(os.path.join(root, "records", "late", "000001-late.jsonl"),
                    [_now_rec("run_state", {"run_id": ids["run_ids"][0], "state": "failed", "event_utc": R.utc_now(), "evidence_kind": "job_log"}, 1)])
    _ingest(root)
    head2 = _head(root)
    # the late file, plus whatever harvester fact the cycle filed beside it (a file_arrival fact dated the run day joins on some days)
    assert head2["n_files"] > head1["n_files"] and head2["projection"]["sha256"] != head1["projection"]["sha256"]
    n_delta = head2["n_files"] - head1["n_files"]
    hp1, hp2 = str(tmp_path / "head1.json.gz"), str(tmp_path / "head2.json.gz")
    RP.write_head(head1, hp1)
    RP.write_head(head2, hp2)
    lay = RP.init_layout(str(tmp_path / "replica"), root)
    before = _snapshot(lay.root)
    run_tree, run_sha = _tree(tmp_path, "run-tree", False)
    h1, h2 = RP.head_sha256(head1), RP.head_sha256(head2)
    common = ["--replica-root", lay.root, "--tree", str(run_tree), "--expect-tree", run_sha, "--workers", "0"]
    out1 = tmp_path / "o1"
    r = _module(run_tree, "update", "--db", kept, "--head", hp2, "--expect-head-sha256", h2, "--work", str(tmp_path / "w1"), "--out", str(out1),
                "--expect-input-sha256", kept_sha, "--job-id", "4242", *common)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "replica check: equal -- frame_projection/1" in r.stdout and "[replica update] EQUAL rc=0" in r.stdout
    rec = json.load(open(out1 / "update.json"))
    assert rec["schema"] == "replica_update/1" and rec["status"] == "EQUAL" and rec["rc"] == 0 and rec["check_equal"] is True
    assert rec["input"]["sha256"] == kept_sha and rec["input"]["path"] == kept and rec["expected_input_sha256"] == kept_sha
    assert rec["output"]["path"] == str(out1 / "registry.sqlite") and rec["output"]["sha256"] == _sha_file(out1 / "registry.sqlite")
    assert rec["output"]["bytes"] == (out1 / "registry.sqlite").stat().st_size and rec["schema_version"] == head2["schema_version"]
    assert rec["ingest"]["delta_files"] == n_delta >= 1 and rec["ingest"]["report"]["records"] >= 1 and rec["mirror"]["ok"] and rec["mirror"]["n_missing"] == 0
    assert rec["ledger"] == {"n_delivered_in_copy": head1["n_files"], "n_in_head": head2["n_files"], "n_ahead": 0, "ahead_sample": [], "n_new_in_head": n_delta,
                             "new_sample": rec["ledger"]["new_sample"]} and len(rec["ledger"]["new_sample"]) == min(5, n_delta)
    assert rec["expected_head_sha256"] == h2 and rec["identity"]["head_sha256"] == h2
    assert rec["check"]["equal"] and rec["check"]["columns_differing"] == {} and rec["check"]["replica"]["sha256"] == head2["projection"]["sha256"]
    assert rec["head"] == RP.head_summary(head2) and rec["head"]["head_sha256"] == RP.head_sha256(head2)
    assert rec["tree"]["bound_to"] == run_sha and rec["tree"]["is_toplevel"] and rec["tree"]["clean"] and rec["tree"]["untracked_source"] == []
    assert rec["identity"] == {"job_id": "4242", "built_by_tree": run_sha, "head_sha256": RP.head_sha256(head2), "allowlist_sha256": head2["allowlist_sha256"],
                               "projection_sha256": head2["projection"]["sha256"], "out_dir": str(out1), "input_sha256": kept_sha}
    assert json.load(open(out1 / "check.json"))["identity"] == rec["identity"] and json.load(open(out1 / "check.json"))["equal"]
    proj = json.load(open(out1 / "projection.json"))
    assert proj["sha256"] == head2["projection"]["sha256"] and len(proj["rows"]) == proj["counts"]["rows"] == head2["projection"]["counts"]["rows"]
    assert (out1 / "mirror.json").is_file() and json.load(open(out1 / "ingest.json"))["delta_files"] == n_delta
    assert _sha_file(kept) == kept_sha and os.stat(kept).st_mtime_ns == kept_stat.st_mtime_ns, "the input is never written"
    assert _snapshot(lay.root) == before, "nothing under the replica root: no generation, no CURRENT"
    assert RP.check(str(out1 / "registry.sqlite"), head2)["equal"]
    out_sha = rec["output"]["sha256"]
    # the brought-forward copy against the OLDER head: refused by name before the mirror hash and any fold, exit 7, no output
    out2 = tmp_path / "o2"
    r = _module(run_tree, "update", "--db", str(out1 / "registry.sqlite"), "--head", hp1, "--expect-head-sha256", h1, "--work", str(tmp_path / "w2"), "--out", str(out2),
                "--expect-input-sha256", out_sha, *common)
    rec2 = json.load(open(out2 / "update.json"))
    assert r.returncode == 7 and rec2["status"] == "REFUSED" and "replica ahead of the fold" in rec2["reason"] and not (out2 / "registry.sqlite").exists()
    assert not (out2 / "check.json").exists() and not (out2 / "mirror.json").exists() and rec2["ledger"]["n_ahead"] == n_delta and rec2["ledger"]["n_new_in_head"] == 0
    # the kept copy against ITS OWN head (an older head offered in place of today's): nothing to fold -> HEAD_NOT_NEWER, exit 13, recorded, no output
    out2b = tmp_path / "o2b"
    r = _module(run_tree, "update", "--db", kept, "--head", hp1, "--expect-head-sha256", h1, "--work", str(tmp_path / "w2b"), "--out", str(out2b),
                "--expect-input-sha256", kept_sha, *common)
    rec2b = json.load(open(out2b / "update.json"))
    assert r.returncode == 13 and "names no file the copy has not folded" in r.stderr and rec2b["status"] == "HEAD_NOT_NEWER" and rec2b["rc"] == 13
    assert rec2b["ledger"]["n_new_in_head"] == 0 and rec2b["ledger"]["n_ahead"] == 0 and not (out2b / "registry.sqlite").exists() and not (out2b / "mirror.json").exists()
    # the head's identity is declared: a head whose head_sha256 is not the declared one refuses before any copy; the declaration is required
    r = _module(run_tree, "update", "--db", kept, "--head", hp2, "--expect-head-sha256", h1, "--work", str(tmp_path / "w2c"), "--out", str(tmp_path / "o2c"),
                "--expect-input-sha256", kept_sha, *common)
    assert r.returncode == 13 and "the head's identity is" in r.stderr and "not the expected" in r.stderr and not (tmp_path / "w2c").exists() and not (tmp_path / "o2c").exists()
    r = _module(run_tree, "update", "--db", kept, "--head", hp2, "--work", str(tmp_path / "w2c"), "--out", str(tmp_path / "o2c"), "--expect-input-sha256", kept_sha, *common)
    assert r.returncode == 1 and "--expect-head-sha256 is required" in r.stderr and not (tmp_path / "o2c").exists()
    # a head whose projection is not what the log folds to: the output stands, the verdict is UNEQUAL, exit 7
    bad = dict(head2)
    bad["projection"] = dict(head2["projection"], sha256="0" * 64)
    bp = str(tmp_path / "bad.json.gz")
    RP.write_head(bad, bp)
    out3 = tmp_path / "o3"
    r = _module(run_tree, "update", "--db", kept, "--head", bp, "--expect-head-sha256", RP.head_sha256(bad), "--work", str(tmp_path / "w3"), "--out", str(out3),
                "--expect-input-sha256", kept_sha, *common)
    rec3 = json.load(open(out3 / "update.json"))
    assert r.returncode == 7 and rec3["status"] == "UNEQUAL" and rec3["check_equal"] is False and (out3 / "registry.sqlite").is_file()
    assert json.load(open(out3 / "check.json"))["equal"] is False and rec3["output"]["sha256"] == _sha_file(out3 / "registry.sqlite")
    # the declarations: a sha the staged copy does not hash to refuses (recorded) and writes no output; a prefix, a missing one, a tree at
    # another commit and a path under which this module does not live refuse before any copy; a dirty tree refuses
    out4 = tmp_path / "o4"
    r = _module(run_tree, "update", "--db", kept, "--head", hp2, "--expect-head-sha256", h2, "--work", str(tmp_path / "w4"), "--out", str(out4), "--expect-input-sha256", "0" * 64, *common)
    rec4 = json.load(open(out4 / "update.json"))
    assert r.returncode == 13 and "not the expected" in r.stderr and rec4["status"] == "INPUT_REFUSED" and rec4["rc"] == 13
    assert not (out4 / "registry.sqlite").exists() and not (out4 / "mirror.json").exists()
    r = _module(run_tree, "update", "--db", kept, "--head", hp2, "--expect-head-sha256", h2, "--work", str(tmp_path / "w5"), "--out", str(tmp_path / "o5"), "--expect-input-sha256", "abc", *common)
    assert r.returncode == 13 and "64 hex" in r.stderr and not (tmp_path / "w5").exists() and not (tmp_path / "o5").exists()
    r = _module(run_tree, "update", "--db", kept, "--head", hp2, "--expect-head-sha256", h2, "--work", str(tmp_path / "w5"), "--out", str(tmp_path / "o5"), "--replica-root", lay.root,
                "--tree", str(run_tree), "--expect-tree", run_sha, "--workers", "0")
    assert r.returncode == 1 and "--expect-input-sha256 is required" in r.stderr and not (tmp_path / "o5").exists()
    other_tree, other_sha = _tree(tmp_path, "other-tree", False)
    r = _module(run_tree, "update", "--db", kept, "--head", hp2, "--expect-head-sha256", h2, "--work", str(tmp_path / "w5"), "--out", str(tmp_path / "o5"), "--expect-input-sha256", kept_sha,
                "--replica-root", lay.root, "--tree", str(run_tree), "--expect-tree", other_sha, "--workers", "0")
    assert r.returncode == 13 and "is not the expected" in r.stderr
    r = _module(run_tree, "update", "--db", kept, "--head", hp2, "--expect-head-sha256", h2, "--work", str(tmp_path / "w5"), "--out", str(tmp_path / "o5"), "--expect-input-sha256", kept_sha,
                "--replica-root", lay.root, "--tree", str(other_tree), "--expect-tree", other_sha, "--workers", "0")
    assert r.returncode == 13 and "the running code is" in r.stderr and "--tree must name the tree whose code runs" in r.stderr
    (run_tree / "src" / "sisteer" / "registry" / "stray.py").write_text("x = 1\n")
    r = _module(run_tree, "update", "--db", kept, "--head", hp2, "--expect-head-sha256", h2, "--work", str(tmp_path / "w5"), "--out", str(tmp_path / "o5"), "--expect-input-sha256", kept_sha, *common)
    assert r.returncode == 13 and ("not clean" in r.stderr or "does not track" in r.stderr)
    (run_tree / "src" / "sisteer" / "registry" / "stray.py").unlink()
    assert not (tmp_path / "w5").exists() and not (tmp_path / "o5").exists()
    # the output directory may alias nothing: the input's own directory (a kept build's record files live beside it), the work directory,
    # a path under the replica root's generations, a directory holding a generation's meta.json -- refused BEFORE anything is cleared
    kept_dir = tmp_path / "kept-build"
    kept_dir.mkdir()
    shutil.copy(kept, kept_dir / "registry.sqlite.unequal")
    (kept_dir / "check.json").write_text("{}")
    r = _module(run_tree, "update", "--db", str(kept_dir / "registry.sqlite.unequal"), "--head", hp2, "--expect-head-sha256", h2, "--work", str(tmp_path / "w6"), "--out", str(kept_dir),
                "--expect-input-sha256", kept_sha, *common)
    assert r.returncode == 13 and "the input's own directory" in r.stderr and (kept_dir / "check.json").is_file() and (kept_dir / "registry.sqlite.unequal").is_file()
    r = _module(run_tree, "update", "--db", kept, "--head", hp2, "--expect-head-sha256", h2, "--work", str(tmp_path / "w6"), "--out", str(tmp_path / "w6" / "o"), "--expect-input-sha256", kept_sha, *common)
    assert r.returncode == 13 and "nest or coincide" in r.stderr
    gen_out = Path(lay.gen_root) / "some-generation"
    r = _module(run_tree, "update", "--db", kept, "--head", hp2, "--expect-head-sha256", h2, "--work", str(tmp_path / "w6"), "--out", str(gen_out), "--expect-input-sha256", kept_sha, *common)
    assert r.returncode == 13 and "replica root's generations" in r.stderr and not gen_out.exists()
    meta_dir = tmp_path / "gen-like"
    meta_dir.mkdir()
    (meta_dir / "meta.json").write_text("{}")
    (meta_dir / "registry.sqlite").write_text("not a database")
    r = _module(run_tree, "update", "--db", kept, "--head", hp2, "--expect-head-sha256", h2, "--work", str(tmp_path / "w6"), "--out", str(meta_dir), "--expect-input-sha256", kept_sha, *common)
    assert r.returncode == 13 and "meta.json" in r.stderr and (meta_dir / "registry.sqlite").read_text() == "not a database"
    assert not (tmp_path / "w6").exists()
    # a reused output directory holds only this run's files
    (out1 / "stale.txt").write_text("x")
    for name in RP.UPDATE_OUTPUTS:
        assert (out1 / name).exists()
    r = _module(run_tree, "update", "--db", kept, "--head", hp2, "--expect-head-sha256", h2, "--work", str(tmp_path / "w7"), "--out", str(out1), "--expect-input-sha256", kept_sha, *common)
    assert r.returncode == 0 and (out1 / "stale.txt").exists() and json.load(open(out1 / "update.json"))["identity"]["job_id"] is None


def test_migrate_db_carries_a_kept_copy_to_the_migrating_trees_schema_and_records_the_shape(tmp_path):
    root, _ = _world(tmp_path)
    head = _head(root)
    hp = str(tmp_path / "head.json.gz")
    RP.write_head(head, hp)
    kept = _kept_copy(tmp_path, root, head)
    kept_sha, kept_stat = _sha_file(kept), os.stat(kept)
    v = head["schema_version"]
    assert v == head["code_version"]
    run_tree, run_sha = _tree(tmp_path, "run-tree", False)
    # a NEWER tree: the repo's package plus one more migration -- a version row and nothing else, so the shape stays that tree's DDL
    step_tree, step_sha = _migrating_tree(tmp_path, "step-tree", "SELECT 1;\n")
    # the site-alias table every --db migration declares: here the empty one (a file with no pairs, and its digest)
    aliases = str(tmp_path / "site_aliases.toml")
    Path(aliases).write_text("[site_aliases]\n")
    al = ["--site-aliases", aliases, "--expect-site-aliases-sha256", SA.digest({})]
    common = ["--tree", str(run_tree), "--expect-tree", run_sha, "--migrate-tree", str(step_tree), "--expect-migrate-tree", step_sha, "--python", PY, *al]
    hs = RP.head_sha256(head)
    out1 = tmp_path / "o1"
    r = _module(run_tree, "migrate", "--db", kept, "--work", str(tmp_path / "w1"), "--out", str(out1), "--head", hp, "--expect-head-sha256", hs, "--expect-input-sha256", kept_sha,
                "--expect-schema", str(v + 1), "--job-id", "4243", *common)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"registry migrate under {step_sha}: rc 0, {v} -> {v + 1}, applied [{v + 1}]" in r.stdout and "[replica migrate] EQUAL rc=0" in r.stdout
    rec = json.load(open(out1 / "migrate.json"))
    assert rec["schema"] == "replica_migrate/1" and rec["status"] == "EQUAL" and rec["rc"] == 0 and rec["check_equal"] is True
    assert rec["before"] == v and rec["after"] == v + 1 and rec["expected_schema"] == v + 1 and [a["version"] for a in rec["applied"]] == [v + 1]
    assert rec["applied"][0]["description"] == "replica_test_step" and rec["applied"][0]["applied_utc"]
    assert rec["migrate_tree"]["bound_to"] == step_sha and rec["migrate_tree"]["sha"] == step_sha and rec["migrate_tree"]["clean"] is True
    assert rec["migrate_tree"]["highest_migration"] == v + 1 and f"{v + 1:04d}_replica_test_step.sql" in rec["migrate_tree"]["migration_files"]
    assert rec["tree"]["bound_to"] == run_sha and rec["running_code_version"] == v and rec["migrate_rc"] == 0 and rec["plan_rc"] == 0
    assert rec["expected_head_sha256"] == hs and rec["identity"]["head_sha256"] == hs
    assert any("pending: none" in line for line in rec["plan"]) and any("matches the code's schema" in line for line in rec["plan"])
    out_db = out1 / "registry.sqlite"
    assert rec["output"]["sha256"] == _sha_file(out_db) and rec["output"]["bytes"] == out_db.stat().st_size and rec["input"]["sha256"] == kept_sha
    assert RP.schema_migration_rows(str(out_db))[-1]["version"] == v + 1 and RP.schema_migration_rows(kept)[-1]["version"] == v
    assert _sha_file(kept) == kept_sha and os.stat(kept).st_mtime_ns == kept_stat.st_mtime_ns, "the input is never written"
    assert rec["identity"] == {"job_id": "4243", "built_by_tree": run_sha, "migrate_tree": step_sha, "out_dir": str(out1), "input_sha256": kept_sha,
                               "expect_schema": v + 1, "head_sha256": RP.head_sha256(head), "allowlist_sha256": head["allowlist_sha256"],
                               "projection_sha256": head["projection"]["sha256"],
                               "site_aliases": {"path": aliases, "digest": SA.digest({}), "pairs": []}}
    chk = json.load(open(out1 / "check.json"))
    assert chk["equal"] and chk["identity"] == rec["identity"] and rec["check"]["equal"] and rec["check"]["columns_differing"] == {}
    proj = json.load(open(out1 / "projection.json"))
    assert proj["sha256"] == head["projection"]["sha256"] and len(proj["rows"]) == proj["counts"]["rows"]
    assert (out1 / "migrate.log").read_text().startswith("$ ") and "[rc 0]" in (out1 / "migrate.log").read_text() and "[rc 0]" in (out1 / "plan.txt").read_text()
    # the running tree's code is now BEHIND the copy: the increment refuses by name (an increment never migrates on the side)
    lay = RP.init_layout(str(tmp_path / "replica"), root)
    with pytest.raises(RP.SchemaBehind, match="replica migrate"):
        RP.ingest_head(str(out_db), lay, head)
    # without a head: no check, status MIGRATED, exit 0; the running tree as the migrating tree: nothing pending, before == after == the declared schema
    out2 = tmp_path / "o2"
    r = _module(run_tree, "migrate", "--db", kept, "--work", str(tmp_path / "w2"), "--out", str(out2), "--expect-input-sha256", kept_sha, "--expect-schema", str(v),
                "--tree", str(run_tree), "--expect-tree", run_sha, "--migrate-tree", str(run_tree), "--expect-migrate-tree", run_sha, "--python", PY, *al)
    assert r.returncode == 0, r.stdout + r.stderr
    rec2 = json.load(open(out2 / "migrate.json"))
    assert rec2["status"] == "MIGRATED" and rec2["check_equal"] is None and rec2["before"] == rec2["after"] == v and rec2["applied"] == []
    assert "check" not in rec2 and not (out2 / "check.json").exists() and rec2["output"]["sha256"] == _sha_file(out2 / "registry.sqlite")
    # the declared target schema: a tree whose migration stops short of it refuses before any copy; a copy that stops short refuses after
    # (exit 8, no output, recorded)
    r = _module(run_tree, "migrate", "--db", kept, "--work", str(tmp_path / "w3"), "--out", str(tmp_path / "o3"), "--expect-input-sha256", kept_sha,
                "--expect-schema", str(v + 2), *common)
    assert r.returncode == 13 and f"carries migrations up to {v + 1:04d}, below the declared target schema {v + 2}" in r.stderr and not (tmp_path / "w3").exists()
    r = _module(run_tree, "migrate", "--db", kept, "--work", str(tmp_path / "w3"), "--out", str(tmp_path / "o3"), "--expect-input-sha256", kept_sha,
                "--expect-schema", str(v), *common)
    rec3 = json.load(open(tmp_path / "o3" / "migrate.json"))
    assert r.returncode == 8 and f"stands at schema {v + 1}" in r.stderr and f"not the declared target {v}" in r.stderr
    assert rec3["status"] == "SCHEMA_NOT_REACHED" and rec3["rc"] == 8 and not (tmp_path / "o3" / "registry.sqlite").exists()
    # a migration that leaves an object the tree's DDL does not declare is refused by that tree's own plan: exit 8, no output, the record names it
    bad_tree, bad_sha = _migrating_tree(tmp_path, "bad-tree", "CREATE VIEW v_zz_undeclared AS SELECT 1 AS one;\n", slug="replica_test_undeclared")
    out4 = tmp_path / "o4"
    r = _module(run_tree, "migrate", "--db", kept, "--work", str(tmp_path / "w4"), "--out", str(out4), "--head", hp, "--expect-head-sha256", hs, "--expect-input-sha256", kept_sha,
                "--expect-schema", str(v + 1), "--tree", str(run_tree), "--expect-tree", run_sha, "--migrate-tree", str(bad_tree), "--expect-migrate-tree", bad_sha, "--python", PY, *al)
    rec4 = json.load(open(out4 / "migrate.json"))
    assert r.returncode == 8 and "shape" in r.stderr and rec4["status"] == "MIGRATION_REFUSED" and rec4["rc"] == 8
    assert not (out4 / "registry.sqlite").exists() and not (out4 / "check.json").exists()
    assert rec4["after"] == v + 1 and (out4 / "migrate.log").is_file() and "v_zz_undeclared" in (out4 / "migrate.log").read_text() + rec4["reason"]
    # the gates before any copy: a tree at another commit, a hash prefix, a missing declaration, a non-integer schema, a tree with no
    # migrations; a wrong input sha after the copy (recorded, no output)
    r = _module(run_tree, "migrate", "--db", kept, "--work", str(tmp_path / "w5"), "--out", str(tmp_path / "o5"), "--expect-input-sha256", kept_sha, "--expect-schema", str(v + 1),
                "--tree", str(run_tree), "--expect-tree", run_sha, "--migrate-tree", str(step_tree), "--expect-migrate-tree", bad_sha, "--python", PY, *al)
    assert r.returncode == 13 and "is not the expected" in r.stderr
    r = _module(run_tree, "migrate", "--db", kept, "--work", str(tmp_path / "w5"), "--out", str(tmp_path / "o5"), "--expect-input-sha256", "abc", "--expect-schema", str(v + 1), *common)
    assert r.returncode == 13 and "64 hex" in r.stderr
    r = _module(run_tree, "migrate", "--db", kept, "--work", str(tmp_path / "w5"), "--out", str(tmp_path / "o5"), "--expect-input-sha256", kept_sha, *common)
    assert r.returncode == 1 and "--expect-schema is required" in r.stderr
    r = _module(run_tree, "migrate", "--db", kept, "--work", str(tmp_path / "w5"), "--out", str(tmp_path / "o5"), "--expect-input-sha256", kept_sha, "--expect-schema", "x", *common)
    assert r.returncode == 2 and "invalid int value" in r.stderr
    # a head without its declaration, and a head whose identity is not the declared one, refuse before any copy
    r = _module(run_tree, "migrate", "--db", kept, "--work", str(tmp_path / "w5"), "--out", str(tmp_path / "o5"), "--head", hp, "--expect-input-sha256", kept_sha,
                "--expect-schema", str(v + 1), *common)
    assert r.returncode == 1 and "--expect-head-sha256 is required" in r.stderr
    r = _module(run_tree, "migrate", "--db", kept, "--work", str(tmp_path / "w5"), "--out", str(tmp_path / "o5"), "--head", hp, "--expect-head-sha256", "1" * 64,
                "--expect-input-sha256", kept_sha, "--expect-schema", str(v + 1), *common)
    assert r.returncode == 13 and "the head's identity is" in r.stderr
    (tmp_path / "notree").mkdir()
    r = _module(run_tree, "migrate", "--db", kept, "--work", str(tmp_path / "w5"), "--out", str(tmp_path / "o5"), "--expect-input-sha256", kept_sha, "--expect-schema", str(v + 1),
                "--tree", str(run_tree), "--expect-tree", run_sha, "--migrate-tree", str(tmp_path / "notree"), "--expect-migrate-tree", step_sha, "--python", PY, *al)
    assert r.returncode == 13 and "not a git checkout" in r.stderr
    assert not (tmp_path / "w5").exists() and not (tmp_path / "o5").exists()
    out6 = tmp_path / "o6"
    r = _module(run_tree, "migrate", "--db", kept, "--work", str(tmp_path / "w6"), "--out", str(out6), "--expect-input-sha256", "0" * 64, "--expect-schema", str(v + 1), *common)
    rec6 = json.load(open(out6 / "migrate.json"))
    assert r.returncode == 13 and rec6["status"] == "INPUT_REFUSED" and rec6["rc"] == 13 and not (out6 / "registry.sqlite").exists() and not (out6 / "migrate.log").exists()
    # the output directory may alias nothing: the input's own directory is refused before anything is cleared
    kept_dir = tmp_path / "kept-build"
    kept_dir.mkdir()
    shutil.copy(kept, kept_dir / "registry.sqlite.unequal")
    (kept_dir / "check.json").write_text("{}")
    r = _module(run_tree, "migrate", "--db", str(kept_dir / "registry.sqlite.unequal"), "--work", str(tmp_path / "w7"), "--out", str(kept_dir), "--expect-input-sha256", kept_sha,
                "--expect-schema", str(v + 1), *common)
    assert r.returncode == 13 and "the input's own directory" in r.stderr and (kept_dir / "check.json").is_file() and not (tmp_path / "w7").exists()


def test_replica_sbatch_update_and_migrate_db_run_the_real_module_node_local_with_srun_shimmed(tmp_path):
    root, ids = _world(tmp_path)
    head1 = _head(root)
    kept = _kept_copy(tmp_path, root, head1)
    kept_sha = _sha_file(kept)
    R.write_records(os.path.join(root, "records", "late", "000001-late.jsonl"),
                    [_now_rec("run_state", {"run_id": ids["run_ids"][0], "state": "failed", "event_utc": R.utc_now(), "evidence_kind": "job_log"}, 1)])
    _ingest(root)
    head2 = _head(root)
    hp = str(tmp_path / "head.json.gz")
    RP.write_head(head2, hp)
    lay = RP.init_layout(str(tmp_path / "replica"), root)
    run_tree, run_sha = _tree(tmp_path, "run-tree", False)       # a clean committed tree: the sbatch's TREE, the record's bound tree
    venv = _fake_venv(tmp_path)
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    _exe(bin_ / "srun", '#!/bin/bash\nwhile [ $# -gt 0 ]; do case "$1" in --ntasks=*|-n) shift ;; *) break ;; esac; done\nexec "$@"\n')
    env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", REPLICA_WORK_ROOT=str(tmp_path / "nodelocal"), SLURM_JOB_ID="778", SLURM_CPUS_PER_TASK="2",
               USER=os.environ.get("USER", "tester"))
    sb = str(LANDING / "replica.sbatch")
    out1 = tmp_path / "out1"
    h2 = RP.head_sha256(head2)
    r = subprocess.run(["bash", sb, "update", "--db", kept, "--head", hp, "--expect-head-sha256", h2, "--out", str(out1), "--expect-input-sha256", kept_sha, "--expect-tree", run_sha,
                        str(run_tree), str(venv), lay.root], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "replica[778] mode=update" in r.stdout and "RESULT mode=update rc=0" in r.stdout and "replica check: equal -- frame_projection/1" in r.stdout
    assert "[replica update] EQUAL rc=0" in r.stdout and f"expect_input_sha256={kept_sha}" in r.stdout and f"tree={run_sha} expect_tree={run_sha}" in r.stdout
    assert f"expect_head_sha256={h2}" in r.stdout and "the head supersedes the copy's ledger" in r.stdout
    rec = json.load(open(out1 / "update.json"))
    assert rec["status"] == "EQUAL" and rec["identity"]["job_id"] == "778" and rec["identity"]["built_by_tree"] == run_sha and rec["tree"]["bound_to"] == run_sha
    assert rec["input"]["copy"] == f"{tmp_path}/nodelocal/778/registry.sqlite" and rec["output"]["sha256"] == _sha_file(out1 / "registry.sqlite")
    assert not (tmp_path / "nodelocal" / "778").exists(), "the node-local work directory is removed by the trap"
    assert lay.current() is None and not os.listdir(lay.gen_root)
    # migrate --db over the update's output, the running tree as the migrating tree (nothing pending): before == after == the declared schema, the check equal
    out2 = tmp_path / "out2"
    aliases = str(tmp_path / "site_aliases.toml")
    Path(aliases).write_text("[site_aliases]\n")
    al = ["--site-aliases", aliases, "--expect-site-aliases-sha256", SA.digest({})]
    r = subprocess.run(["bash", sb, "migrate", "--db", str(out1 / "registry.sqlite"), "--out", str(out2), "--head", hp, "--expect-head-sha256", h2, "--expect-input-sha256", rec["output"]["sha256"],
                        "--expect-tree", run_sha, "--migrate-tree", str(run_tree), "--expect-migrate-tree", run_sha, "--expect-schema", str(head2["schema_version"]), *al,
                        str(run_tree), str(venv)], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "replica[778] mode=migrate" in r.stdout and "RESULT mode=migrate rc=0" in r.stdout and f"migrate_tree={run_sha} expect_migrate_tree={run_sha}" in r.stdout
    assert f"expect_schema={head2['schema_version']}" in r.stdout and "replica check: equal -- frame_projection/1" in r.stdout and "[replica migrate] EQUAL rc=0" in r.stdout
    rec2 = json.load(open(out2 / "migrate.json"))
    assert rec2["status"] == "EQUAL" and rec2["before"] == rec2["after"] == head2["schema_version"] and rec2["applied"] == []
    assert rec2["migrate_tree"]["bound_to"] == run_sha and rec2["tree"]["bound_to"] == run_sha and rec2["input"]["sha256"] == rec["output"]["sha256"] and rec2["identity"]["job_id"] == "778"
    assert RP.check(str(out2 / "registry.sqlite"), head2)["equal"] and lay.current() is None
    # the guards, before any work: the option sets a mode cannot read; every declaration required
    r = subprocess.run(["bash", sb, "update", str(run_tree), str(venv), lay.root], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "update needs --db PATH --head HEAD --expect-head-sha256 H --out OUT_DIR --expect-input-sha256 H --expect-tree SHA" in r.stdout
    r = subprocess.run(["bash", sb, "update", "--db", kept, "--head", hp, "--expect-head-sha256", h2, "--out", str(tmp_path / "x"), "--expect-input-sha256", kept_sha, str(run_tree), str(venv), lay.root],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "update needs" in r.stdout, "no --expect-tree: refused"
    r = subprocess.run(["bash", sb, "update", "--db", kept, "--head", hp, "--out", str(tmp_path / "x"), "--expect-input-sha256", kept_sha, "--expect-tree", run_sha, str(run_tree), str(venv), lay.root],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "update needs" in r.stdout, "no --expect-head-sha256: refused"
    r = subprocess.run(["bash", sb, "update", "--db", kept, "--head", hp, "--expect-head-sha256", h2, "--out", str(tmp_path / "x"), "--expect-input-sha256", kept_sha, "--expect-tree", run_sha,
                        str(run_tree), str(venv), str(tmp_path / "noroot")], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "no replica root" in r.stdout
    r = subprocess.run(["bash", sb, "migrate", "--db", kept, "--out", str(tmp_path / "y"), "--head", hp, "--expect-input-sha256", kept_sha, "--expect-tree", run_sha,
                        "--migrate-tree", str(run_tree), "--expect-migrate-tree", run_sha, "--expect-schema", "1", *al, str(run_tree), str(venv)], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "--head needs --expect-head-sha256" in r.stdout
    r = subprocess.run(["bash", sb, "migrate", "--db", kept, "--out", str(tmp_path / "y"), str(run_tree), str(venv)], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "--db needs --out OUT_DIR --expect-input-sha256 H --expect-tree SHA --migrate-tree MTREE --expect-migrate-tree SHA --expect-schema N" in r.stdout
    r = subprocess.run(["bash", sb, "migrate", "--out", str(tmp_path / "y"), str(run_tree), str(venv), lay.root], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "need --db PATH" in r.stdout
    r = subprocess.run(["bash", sb, "migrate", "--db", str(tmp_path / "nodb"), "--out", str(tmp_path / "y"), "--expect-input-sha256", kept_sha, "--expect-tree", run_sha,
                        "--migrate-tree", str(run_tree), "--expect-migrate-tree", run_sha, "--expect-schema", "1", *al, str(run_tree), str(venv)], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "no database at" in r.stdout
    r = subprocess.run(["bash", sb, "update", "--bogus", "1", str(run_tree), str(venv), lay.root], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "unknown update option" in r.stdout
    assert not (tmp_path / "x").exists() and not (tmp_path / "y").exists()
    text = (LANDING / "replica.sbatch").read_text()
    assert "MODE (build|land|migrate|update|refold-compare|contract)" in text


SSH_SHIM = r'''#!/bin/bash
# the login node: everything runs here (the cluster paths are local directories); tree_export over stdin answers TREE existing;
# sacct answers $MOCK/job_state (COMPLETED by default); sbatch records its argv, runs the fake replica job synchronously (writes the
# out dir from $MOCK knobs, every result stamped with the job's identity the way `land` does, CURRENT written) and prints a job id
while [ $# -gt 0 ]; do case "$1" in -o) shift 2 ;; -*) shift ;; *) break ;; esac; done
host=$1; shift; cmd="$*"
echo "ssh $host $cmd" >> "$MOCK/ssh.calls"
case "$cmd" in
  "bash -s --"*) cat > /dev/null; set -- $cmd; echo "TREE existing $6"; exit 0 ;;
  sacct*) cat "$MOCK/job_state" 2>/dev/null || echo COMPLETED; exit 0 ;;
  sbatch*)
    echo "$cmd" >> "$MOCK/sbatch.calls"
    # the remote shell's word splitting and un-escaping (the 10th positional rides `printf %q`-escaped): eval, as the login node's bash would
    eval "set -- $cmd"; while [ "$1" != "land" ]; do shift; done; shift
    TREE=$1 HEAD=$2 ROOT=$5 OUT=$6 RTREE=${9:-$1} ACCEPT=${10:-}; mkdir -p "$OUT" "$ROOT"
    printf '%s\n' "$ACCEPT" > "$MOCK/accept.received"      # what the job received as ACCEPT_C1_CLASS, verbatim
    [ -f "$MOCK/job_dies" ] && { echo "4242;cluster"; exit 0; }       # the job died before writing anything
    HS=$("$LAND_PY" -c 'import sys; from sisteer.registry import replica as RP; print(RP.head_sha256(RP.read_head(sys.argv[1])))' "$HEAD")
    JID=4242; [ -f "$MOCK/job_wrong_identity" ] && JID=4141
    # built_by_tree is the REPLICA side's tree (the 9th positional in a recovery, else TREE); a job that ignored the pin stamps TREE
    BUILT=$(basename "$RTREE"); [ -f "$MOCK/job_ignores_rtree" ] && BUILT=$(basename "$TREE")
    TESTS_TREE=$TREE; [ -f "$MOCK/job_tests_from_rtree" ] && TESTS_TREE=$RTREE
    ID="\"identity\":{\"job_id\":\"$JID\",\"built_by_tree\":\"$BUILT\",\"head_sha256\":\"$HS\"},\"tree\":\"$TESTS_TREE\""
    GEN=g1; [ -f "$MOCK/job_current_moved" ] && GEN=g2
    echo "$GEN" > "$ROOT/CURRENT"
    if [ -f "$MOCK/job_unequal" ]; then
      # the cutover landing's shape: the C1 class AND decode_policy_match -- not the class alone, so c1_class is false
      echo "{\"schema\":\"replica_check/1\",\"equal\":false,\"c1_class\":false,\"n_movers\":1262,\"columns_differing\":{\"lifecycle_state\":2,\"is_complete\":2,\"decode_policy_match\":1260},\"diff\":{\"columns\":{\"lifecycle_state\":{\"rows\":2},\"is_complete\":{\"rows\":2},\"decode_policy_match\":{\"rows\":1260}}},\"replica\":{\"sha12\":\"aaaaaaaaaaaa\",\"counts\":{}},\"projection_version\":\"frame_projection/1\",$ID}" > "$OUT/check.json"
      echo "{\"status\":\"UNEQUAL\",\"check_equal\":false,\"check_accepted_as\":null,\"parent_generation\":\"g0\",$ID}" > "$OUT/summary.json"
    elif [ -f "$MOCK/job_c1_class" ] || [ -f "$MOCK/job_claims_c1" ]; then
      # the check differs on the certified C1 same-stamp class alone (two movers); the job accepts it under its declaration (the 10th
      # positional) -- or, job_claims_c1, WITHOUT one, which the grading host must refuse by name
      echo "{\"schema\":\"replica_check/1\",\"equal\":false,\"c1_class\":true,\"n_movers\":2,\"movers\":[\"r1\",\"r2\"],\"columns_differing\":{\"lifecycle_state\":2,\"is_complete\":2},\"n_only_in_head\":0,\"n_only_in_replica\":0,\"diff\":{\"columns\":{\"lifecycle_state\":{\"rows\":2},\"is_complete\":{\"rows\":2}}},\"replica\":{\"sha12\":\"cccccccccccc\",\"counts\":{\"rows\":3}},\"projection_version\":\"frame_projection/1\",$ID}" > "$OUT/check.json"
      if [ -n "$ACCEPT" ] || [ -f "$MOCK/job_claims_c1" ]; then
        DECL=$("$LAND_PY" -c 'import json, sys; print(json.dumps(sys.argv[1] or "undeclared"))' "$ACCEPT")
        echo "{\"status\":\"PASS\",\"check_equal\":false,\"check_c1_class\":true,\"check_accepted_as\":\"c1_class\",\"c1_class_movers\":2,\"c1_class_declaration\":$DECL,\"pytest_rc\":0,\"counts\":{\"tests\":4,\"failures\":0,\"errors\":0,\"skipped\":0},\"published\":\"g1\",\"parent_generation\":\"g0\",$ID}" > "$OUT/summary.json"
        printf '<testsuites><testsuite name="pytest" tests="4" failures="0" errors="0" skipped="0"><testcase classname="tests.test_registry_bon_decode_precedence" name="t" time="1.0"/></testsuite></testsuites>' > "$OUT/junit.xml"
        echo "fake pytest log" > "$OUT/pytest.log"
      else
        echo "{\"status\":\"UNEQUAL\",\"check_equal\":false,\"check_c1_class\":true,\"check_accepted_as\":null,\"reason\":\"replica check UNEQUAL (the C1 same-stamp class, not declared: no allowance without LAND_ACCEPT_C1_CLASS)\",\"parent_generation\":\"g0\",$ID}" > "$OUT/summary.json"
      fi
    else
      echo "{\"schema\":\"replica_check/1\",\"equal\":true,\"replica\":{\"sha12\":\"bbbbbbbbbbbb\",\"counts\":{\"rows\":3}},\"projection_version\":\"frame_projection/1\",$ID}" > "$OUT/check.json"
      sk=0; [ -f "$MOCK/job_skips" ] && sk=1; rc=0; [ -f "$MOCK/job_red" ] && rc=1
      PUB='"published":"g1"'; ST=PASS
      [ -f "$MOCK/job_unpublished" ] && { PUB='"published":null,"publish_refused":"CURRENT names g7, not the parent g0 this generation was built from"'; ST=PUBLISH_REFUSED; }
      echo "{\"status\":\"$ST\",\"check_equal\":true,\"pytest_rc\":$rc,\"counts\":{\"tests\":4,\"failures\":$rc,\"errors\":0,\"skipped\":$sk},$PUB,\"parent_generation\":\"g0\",$ID}" > "$OUT/summary.json"
      printf '<testsuites><testsuite name="pytest" tests="4" failures="%s" errors="0" skipped="%s"><testcase classname="tests.test_registry_bon_decode_precedence" name="t" time="1.0"/></testsuite></testsuites>' "$rc" "$sk" > "$OUT/junit.xml"
      echo "fake pytest log" > "$OUT/pytest.log"
    fi
    echo "4242;cluster"; exit 0 ;;
  *) exec bash -c "$cmd" ;;
esac
'''


#: Fake systemd for the record-log mirror unit (live_db_slurm.sh step 2). `show` answers Key=Value from $MOCK/unit.{state,result,
#: start,exit} (defaults inactive / success / empty / empty); a `$MOCK/unit.ticks` countdown ends a pre-seeded running pass after that
#: many `show` reads (the stale pass). `start` on an idle unit runs ONE pass: a fresh start stamp, the synthetic root's manifest
#: written by the module's own writer and switched, state/mirror_state.json with the cluster_a and cluster_b rows -- `$MOCK/pass_fails`
#: fails it (rc 1, Result=failed); `$MOCK/pass_budget_stopped` (consumed) leaves the rows unswitched with stopped=budget, as a
#: budget-stopped `mirrors sync` does while exiting 0; `$MOCK/manifest_drop` names a file the census misses (the missed harvester fact
#: record); `$MOCK/pass_race` (consumed) makes the pass miss that file, then has the timer start the NEXT pass at once, which rewrites
#: the same-named manifest with the file before the chain reads it. Manifests are named as production names them,
#: `CHECKSUMS-<max ingest_seq>`, so passes without an ingest REPLACE one file; a switched row carries the pass's `mirror_ack` record with
#: the manifest's digest, as `mirrors.sync_all` writes it. `start` on a running unit merges into that pass, as systemd's job merge does:
#: it ends with the EXISTING start stamp. Every call is appended to $MOCK/systemctl.calls.
SYSTEMCTL_SHIM = r'''#!/bin/bash
echo "systemctl $*" >> "$MOCK/systemctl.calls"
[ "$1" = "--user" ] && shift
now() { date -u +"%a %Y-%m-%d %H:%M:%S.%6N UTC"; }
rd() { cat "$MOCK/unit.$1" 2>/dev/null || echo "$2"; }
census() {   # $1 mode: switched | budget | manifest_only; $2 a file to leave out of the manifest
  "$LAND_PY" - "$REGISTRY_ROOT" "$1" "$2" <<'PY' || exit 1
import os, sys
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import mirrors as M
from sisteer.registry import records as R
root, mode, drop = sys.argv[1:4]
with DB.reading(DB.db_path(root)) as conn:   # production's name: the ledger's max ingest_seq, the same between ingests
    seq = conn.execute("SELECT COALESCE(MAX(ingest_seq), 0) FROM record_ingest").fetchone()[0]
mw = M.write_manifest_report(root, int(seq))
p = os.path.join(root, M.MANIFEST_DIR, mw.name)
if drop:
    lines = [l for l in open(p, encoding="utf-8").read().splitlines() if not l.endswith("  " + drop)]
    open(p, "w", encoding="utf-8").write("\n".join(lines) + "\n")
if mode == "manifest_only":
    sys.exit(0)
switched = mode == "switched"
if switched:
    M.switch_current(root, mw.name)
now = R.utc_now()
rows = []
for d in ("cluster_a", "cluster_b"):
    row = {"destination": d, "kind": "ssh", "manifest": mw.name, "copied": 0, "ok": mw.entries if switched else 0, "missing": 0, "corrupt": 0,
           "switched": switched, "error": None, "stopped": None if switched else "budget", "mode": "incremental", "utc": now}
    if switched:
        row["ack_record"] = H.write_harvester_record(root, "mirror_ack", {"destination": d, "acked_through_seq": int(seq), "manifest_name": mw.name,
                                                                          "file_count": mw.entries, "bytes": mw.bytes, "checksums_sha256": M.manifest_sha(p),
                                                                          "acked_utc": now, "last_error": None}, now)
    rows.append(row)
M.write_state(root, rows, now)
PY
}
pass() {   # one pass's side effects on the synthetic registry root, then the unit idle
  if [ -f "$MOCK/pass_fails" ]; then echo failed > "$MOCK/unit.result"; now > "$MOCK/unit.exit"; echo inactive > "$MOCK/unit.state"; exit 1; fi
  MODE=switched; [ -f "$MOCK/pass_budget_stopped" ] && { MODE=budget; rm -f "$MOCK/pass_budget_stopped"; }
  DROP=$(cat "$MOCK/manifest_drop" 2>/dev/null)
  census "$MODE" "$DROP"
  echo success > "$MOCK/unit.result"; now > "$MOCK/unit.exit"; echo inactive > "$MOCK/unit.state"
  if [ -f "$MOCK/pass_race" ]; then
    # the timer fires the next pass at once; its census (the file now present) replaces the same-named manifest before the chain reads it
    rm -f "$MOCK/pass_race" "$MOCK/manifest_drop"; now > "$MOCK/unit.start"; : > "$MOCK/unit.exit"; echo activating > "$MOCK/unit.state"
    census manifest_only ""
  fi
}
case "$1" in
  show)
    if [ -f "$MOCK/unit.ticks" ]; then
      t=$(( $(cat "$MOCK/unit.ticks") - 1 ))
      if [ "$t" -le 0 ]; then rm -f "$MOCK/unit.ticks"; echo inactive > "$MOCK/unit.state"; echo success > "$MOCK/unit.result"; now > "$MOCK/unit.exit"
      else echo "$t" > "$MOCK/unit.ticks"; fi
    fi
    echo "Result=$(rd result success)"; echo "ExecMainStartTimestamp=$(rd start '')"; echo "ExecMainExitTimestamp=$(rd exit '')"; echo "ActiveState=$(rd state inactive)" ;;
  start)
    if [ "$(rd state inactive)" = activating ]; then rm -f "$MOCK/unit.ticks"; pass
    else now > "$MOCK/unit.start"; : > "$MOCK/unit.exit"; echo activating > "$MOCK/unit.state"; sleep 0.2; pass; fi ;;
  *) echo "fake systemctl: unknown verb $1" >&2; exit 2 ;;
esac
exit 0
'''


def _live_db_world(tmp_path):
    root, ids = _world(tmp_path)
    mock = tmp_path / "mock"
    bin_ = tmp_path / "bin"
    shared = tmp_path / "shared"
    state = tmp_path / "state"
    for d in (mock, bin_, shared / "envs", shared / "gates" / "trees", state):
        d.mkdir(parents=True)
    _exe(bin_ / "ssh", SSH_SHIM)
    _exe(bin_ / "systemctl", SYSTEMCTL_SHIM)
    _exe(mock / "watch-job.sh", '#!/bin/bash\necho "watch $*" >> "$MOCK/watch.calls"\necho "RUNNING $2 node=TESTNODE"\necho "DONE $2 state=COMPLETED"\nexit 0\n')
    _exe(mock / "mirror-repos.sh", '#!/bin/bash\necho "mirror $*" >> "$MOCK/mirror.calls"\n')
    _exe(mock / "mirror_history.sh", '#!/bin/bash\necho "history $*" >> "$MOCK/history.calls"\necho "HISTORY current 0 refs on $2 (shim)"\nexit 0\n')
    sha = "e" * 40
    # the worktree the step reads pyproject/uv.lock from (git show at the tip): a tiny repo with one commit
    wt = tmp_path / "repo" / "worktrees" / "probe"
    wt.mkdir(parents=True)
    (wt / "pyproject.toml").write_text("[project]\nname='x'\n")
    (wt / "uv.lock").write_text("lock\n")
    env_git = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@TESTHOST", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@TESTHOST")
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "seed"]):
        subprocess.run(["git", *args], cwd=str(wt), check=True, env=env_git, capture_output=True)
    tip = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(wt), capture_output=True, text=True, check=True).stdout.strip()
    ph = subprocess.run(["sha256sum"], input=(wt / "pyproject.toml").read_bytes(), capture_output=True).stdout.split()[0].decode()
    lh = subprocess.run(["sha256sum"], input=(wt / "uv.lock").read_bytes(), capture_output=True).stdout.split()[0].decode()
    h12 = subprocess.run(["sha256sum"], input=f"{ph}\n{lh}\n".encode(), capture_output=True).stdout.split()[0].decode()[:12]
    venv = shared / "envs" / f"repo-gate-{h12}"          # <the checkout's directory name>-gate-<hash12>: the checkout is tmp_path / "repo"
    venv.mkdir()
    (venv / ".gate-ready.json").write_text("{}")
    (shared / "gates" / "trees" / tip).mkdir()
    # the script ships the head and collects the results with rsync (`host:path` operands through the ssh shim, which runs the remote
    # `rsync --server` locally): the real binary, which every test of this world requires (NEEDS_RSYNC) -- the test fallback runs no
    # remote shell and would refuse these operands, and opting it in would pass the world without the transport it exercises
    env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", MOCK=str(mock), REGISTRY_LANDING_STATE=str(state), REGISTRY_ROOT=root,
               SISTEER_REPO=str(tmp_path / "repo"), LAND_PY=PY, LIVE_DB_PYTHONPATH=str(REPO / "src"), GATE_HOST="TESTHOST", GATE_RUSER="tester",
               GATE_SHARED=str(shared), CLUSTER_DEST="cluster_a", MIRROR_REPOS_SH=str(mock / "mirror-repos.sh"),
               MIRROR_HISTORY_SH=str(mock / "mirror_history.sh"),
               WATCH_JOB_SH=str(mock / "watch-job.sh"), LIVE_DB_MIRROR_POLL_S="1", LIVE_DB_PIN_L1=str(tmp_path / "nopin"), LIVE_DB_WAIT_MIN="0", LIVE_DB_DEFER_MIN="1",
               LIVE_DB_DEFER_POLL_S="1")
    return types.SimpleNamespace(root=root, ids=ids, mock=mock, state=state, shared=shared, env=env, tip=tip, venv=venv)


def _live_db(w, *extra):
    return subprocess.run([str(LANDING / "live_db_slurm.sh"), w.tip, "--branch", "probe", "--main", "f" * 40, *extra], capture_output=True, text=True, env=w.env, timeout=300)


@NEEDS_RSYNC
def test_live_db_slurm_takes_the_head_syncs_ships_submits_watches_collects_and_writes_the_verdict(tmp_path):
    w = _live_db_world(tmp_path)
    r = _live_db(w)
    out = r.stdout
    assert r.returncode == 0, out + r.stderr
    stamps = re.findall(r"^\[land\] @ live-db-([a-z-]+):(start|end) ", out, re.M)
    assert [s for s, e in stamps if e == "start"] == ["head", "mirror-sync", "tree", "ship", "job", "collect"] and all((s, "end") in stamps for s, _ in stamps)
    head = RP.read_head(str(w.state / "land" / "probe.head.json.gz"))
    assert head["n_files"] > 0 and "[land] live-db head: n_files=" in out and f"n_files={head['n_files']}" in out
    assert out.index("@ live-db-head:end") < out.index("@ live-db-mirror-sync:start"), "the mirror is synced AFTER the head"
    # the unit idle at entry: ONE pass, started after the head and verified from its own record; nothing waited out
    calls = (w.mock / "systemctl.calls").read_text().splitlines()
    assert [c for c in calls if " start " in c] == ["systemctl --user start registry-mirrors.service"] and "stale pass ended" not in out
    assert "[land] live-db: fresh pass started " in out and f"[land] live-db: covers head: yes (started after taken_utc {head['taken_utc']}; manifest CHECKSUMS-" in out, out
    assert out.index("fresh pass started") < out.index("@ live-db-mirror-sync:end") < out.index("@ live-db-tree:start")
    assert (w.mock / "mirror.calls").read_text().startswith("mirror repo\n"), "the repo-mirror tool gets the checkout's directory name"
    # then the archived branches and tags of the worktree's repository, to the mirror's URL, before the export (a problem would be a note)
    assert (w.mock / "history.calls").read_text() == f"history {tmp_path / 'repo' / 'worktrees' / 'probe'} ssh://TESTHOST{w.shared}/repos/repo\n"
    assert out.index("@ live-db-tree:start") < out.index("TREE existing") and "mirror_history.sh reported a problem" not in out
    assert "TREE existing" in out
    sb = (w.mock / "sbatch.calls").read_text().splitlines()
    assert len(sb) == 1 and sb[0].startswith("sbatch --parsable -J registry-replica --dependency=singleton --time=01:30:00 ") and "/replica.sbatch land " in sb[0], sb
    parts = sb[0].split(" land ", 1)[1].split()
    tree, head_remote, venv, branch, root, out_dir, expect, hashm = parts
    assert tree == str(w.shared / "gates" / "trees" / w.tip) and venv == str(w.venv) and branch == "probe" and expect == "0" and hashm == "all"
    # every submission owns its paths: the head under builds/land-<B>-<tip8>-<token>/, the results under gates/<tip>/live_db/<token>/
    token = os.path.basename(out_dir)
    assert re.fullmatch(r"\d{8}T\d{6}Z-\d+-[0-9a-f]{4}", token), token
    assert root == str(w.shared / "registry" / "replica") and head_remote == f"{root}/builds/land-probe-{w.tip[:8]}-{token}/head.json.gz"
    assert out_dir == str(w.shared / "gates" / w.tip / "live_db" / token) and os.path.isfile(head_remote), "the head was shipped before the submission"
    assert RP.read_head(head_remote)["allowlist_sha256"] == head["allowlist_sha256"]
    assert "[land] live-db job 4242 (registry-replica, singleton, 01:30:00" in out and (w.mock / "watch.calls").read_text().startswith("watch --slurm 4242 --host TESTHOST --poll 30 --stall-min 40 --max-hours 3")
    assert "[land] live-db watch: DONE 4242" in out and "[land] live-db: job 4242 state COMPLETED" in out
    assert (w.state / "land" / "probe.junit.xml").is_file() and (w.state / "land" / "probe.pytest.log").is_file() and (w.state / "land" / "probe.live_db" / "check.json").is_file()
    rc_state = json.load(open(Path(w.root) / "state" / "replica_check.json"))
    assert rc_state["equal"] and rc_state["ok"] and rc_state["job"] == "4242" and rc_state["generation"] == "g1" and rc_state["head"]["n_files"] == head["n_files"]
    assert rc_state["projection_version"] == "frame_projection/1" and rc_state["pytest"]["counts"]["skipped"] == 0 and rc_state["tip"] == w.tip
    assert rc_state["job_state"] == "COMPLETED" and rc_state["current_generation"] == "g1" and rc_state["problems"] == [] and rc_state["headline"] == "pass"
    assert rc_state["submission"] == token and rc_state["out_remote"] == out_dir and rc_state["head"]["head_sha12"] == RP.head_sha256(head)[:12]
    assert "replica check: equal -- frame_projection/1" in out and "[land] live-db on Slurm PASS for merged tip" in out and "generation g1" in out
    assert f"JOB=4242 TOKEN={token} OUT={out_dir}" in (w.state / "jobs.log").read_text()
    # an unequal check refuses (exit 5), names the cause, records it
    (w.mock / "job_unequal").write_text("")
    r = _live_db(w)
    assert r.returncode == 5 and "replica check: UNEQUAL" in r.stdout and "LIVE-DB REPLICA UNEQUAL (job 4242)" in r.stdout and "REASON: projection differs" in r.stdout, r.stdout
    assert json.load(open(Path(w.root) / "state" / "replica_check.json"))["equal"] is False
    (w.mock / "job_unequal").unlink()
    # a skip beyond the expected count refuses; the expected count honored passes
    (w.mock / "job_skips").write_text("")
    r = _live_db(w)
    assert r.returncode == 5 and "skips unaccounted" in r.stdout and "expected_skipped=0" in r.stdout
    r = _live_db(w, "--expect-skipped", "1")
    assert r.returncode == 0 and "expected_skipped=1" in r.stdout
    (w.mock / "job_skips").unlink()
    # red pytest refuses -- the job exits non-zero for it, so the accounting says FAILED too: results that ARE this job's explain the
    # failure, and the headline is the tests, not the job state
    (w.mock / "job_red").write_text("")
    (w.mock / "job_state").write_text("FAILED\n")
    r = _live_db(w)
    assert r.returncode == 5 and "LIVE-DB TESTS RED" in r.stdout and "REFUSED: pytest rc=1" in r.stdout and "REFUSED: job 4242 state 'FAILED'" in r.stdout, r.stdout + r.stderr
    rc_state = json.load(open(Path(w.root) / "state" / "replica_check.json"))
    assert rc_state["pytest"]["rc"] == 1 and rc_state["job_state"] == "FAILED" and rc_state["headline"] == "tests" and not rc_state["ok"]
    (w.mock / "job_red").unlink()
    (w.mock / "job_state").unlink()
    # A STALE PASS ON DISK: the job dies before writing check.json (a node failure). The earlier PASS of this tip is on disk under ITS
    # submission's directory; this submission's directory is its own and empty, the local collection is fresh, the accounting says
    # NODE_FAIL -- refused by the job's state, nothing of the earlier run read as this one's
    n_before = len((w.mock / "sbatch.calls").read_text().splitlines())
    (w.mock / "job_dies").write_text("")
    (w.mock / "job_state").write_text("NODE_FAIL\n")
    r = _live_db(w)
    assert r.returncode == 5 and "LIVE-DB JOB 4242 DID NOT COMPLETE (state NODE_FAIL)" in r.stdout and "REFUSED: no check.json under" in r.stdout, r.stdout + r.stderr
    assert not (w.state / "land" / "probe.live_db" / "check.json").exists() and not (w.state / "land" / "probe.live_db" / "summary.json").exists()
    sb = (w.mock / "sbatch.calls").read_text().splitlines()
    assert len(sb) == n_before + 1
    outs = [l.split(" land ", 1)[1].split()[5] for l in sb]
    heads = [l.split(" land ", 1)[1].split()[1] for l in sb]
    assert len(set(outs)) == len(outs) and len(set(heads)) == len(heads), "every submission minted its own output directory and head path"
    assert os.path.isfile(os.path.join(outs[0], "check.json")) and not os.listdir(outs[-1]), "the first submission's results are untouched; the dead job's directory is empty"
    rc_state = json.load(open(Path(w.root) / "state" / "replica_check.json"))
    assert not rc_state["ok"] and rc_state["job_state"] == "NODE_FAIL" and rc_state["headline"] == "job" and rc_state["generation"] is None and rc_state["submission"] == os.path.basename(outs[-1])
    (w.mock / "job_dies").unlink()
    (w.mock / "job_state").unlink()
    # results that are not this job's (another job id in the identity) refuse by name even from a COMPLETED job
    (w.mock / "job_wrong_identity").write_text("")
    r = _live_db(w)
    assert r.returncode == 5 and "LIVE-DB RESULT IDENTITY MISMATCH (job 4242)" in r.stdout and "check.json names job '4141', not the submitted job 4242" in r.stdout, r.stdout
    assert "summary.json names job '4141'" in r.stdout and json.load(open(Path(w.root) / "state" / "replica_check.json"))["headline"] == "identity"
    (w.mock / "job_wrong_identity").unlink()
    # the job published nothing (its CAS lost): a failing status on the job (exit 10, FAILED) and a named refusal here
    (w.mock / "job_unpublished").write_text("")
    (w.mock / "job_state").write_text("FAILED\n")
    r = _live_db(w)
    assert r.returncode == 5 and "LIVE-DB GENERATION NOT PUBLISHED or CURRENT moved (job 4242)" in r.stdout, r.stdout
    assert "REFUSED: no generation published: CURRENT names g7" in r.stdout and "the job's own verdict is 'PUBLISH_REFUSED'" in r.stdout
    assert json.load(open(Path(w.root) / "state" / "replica_check.json"))["headline"] == "publish"
    (w.mock / "job_unpublished").unlink()
    (w.mock / "job_state").unlink()
    # the job published g1 but CURRENT names g2 by the time the grading host looks (a rollback, another writer): refused -- the next landing
    # would not start from this one's generation
    (w.mock / "job_current_moved").write_text("")
    r = _live_db(w)
    assert r.returncode == 5 and "CURRENT moved (job 4242)" in r.stdout and "REFUSED: CURRENT names 'g2', not the generation 'g1' this job published" in r.stdout, r.stdout
    rc_state = json.load(open(Path(w.root) / "state" / "replica_check.json"))
    assert rc_state["generation"] == "g1" and rc_state["current_generation"] == "g2" and not rc_state["ok"] and rc_state["headline"] == "publish"
    (w.mock / "job_current_moved").unlink()
    # a failed mirror pass refuses before any submission
    n = len((w.mock / "sbatch.calls").read_text().splitlines())
    (w.mock / "pass_fails").write_text("")
    r = _live_db(w)
    assert r.returncode == 5 and "mirror pass failed" in r.stdout and len((w.mock / "sbatch.calls").read_text().splitlines()) == n
    (w.mock / "pass_fails").unlink()
    # the venv not ready refuses, naming gate_venv.sbatch, before any submission
    (w.venv / ".gate-ready.json").unlink()
    r = _live_db(w)
    assert r.returncode == 5 and "gate venv" in r.stdout and "gate_venv.sbatch" in r.stdout and len((w.mock / "sbatch.calls").read_text().splitlines()) == n
    (w.venv / ".gate-ready.json").write_text("{}")
    # usage
    r = subprocess.run([str(LANDING / "live_db_slurm.sh"), "nothex", "--branch", "b"], capture_output=True, text=True, env=w.env)
    assert r.returncode == 5 and "usage" in r.stdout


@NEEDS_RSYNC
def test_live_db_slurm_accepts_the_c1_class_only_under_the_declaration_and_records_it(tmp_path):
    """The job's check differs on the certified C1 same-stamp class alone. Without LAND_ACCEPT_C1_CLASS the step refuses as before
    (exit 5, the hint names the variable, eight positionals, nothing accepted). With it the declaration rides as the 10th positional --
    the 9th then TREE itself -- `printf %q`-escaped so the text arrives verbatim (spaces, commas, an unexpanded `~`); the verdict prints
    the ACCEPTED line, passes, and records c1_class / accepted_as / n_movers / the declaration in state/replica_check.json. The
    declaration covers nothing else: a check that also differs on decode_policy_match refuses with it; an equal check under it is
    recorded as equal; a job that published under an allowance the grading host did not declare is refused by name; in a recovery the
    9th positional stays the pin."""
    w = _live_db_world(tmp_path)
    state_path = Path(w.root) / "state" / "replica_check.json"
    (w.mock / "job_c1_class").write_text("")
    r = _live_db(w)
    assert r.returncode == 5 and "LIVE-DB REPLICA UNEQUAL (job 4242)" in r.stdout and "ACCEPTED:" not in r.stdout, r.stdout + r.stderr
    assert "REFUSED: replica check UNEQUAL: projection differs: " in r.stdout and "the C1 same-stamp class, 2 movers -- NOT declared: no allowance without LAND_ACCEPT_C1_CLASS" in r.stdout
    assert "REASON: projection differs" in r.stdout and "ALLOWANCE DECLARED" not in r.stdout
    assert len((w.mock / "sbatch.calls").read_text().splitlines()[-1].split(" land ", 1)[1].split()) == 8 and (w.mock / "accept.received").read_text() == "\n"
    st = json.load(open(state_path))
    assert st["equal"] is False and st["c1_class"] is True and st["accepted_as"] is None and st["n_movers"] == 2 and st["declaration"] is None
    assert not st["ok"] and st["headline"] == "unequal" and st["columns_differing"] == {"lifecycle_state": 2, "is_complete": 2} and st["job_check_accepted_as"] is None
    # the declaration
    w.env["LAND_ACCEPT_C1_CLASS"] = C1_DECLARATION
    r = _live_db(w)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"[land] live-db: ALLOWANCE DECLARED (LAND_ACCEPT_C1_CLASS) -- a check equal modulo the certified C1 same-stamp class alone will be accepted: {C1_DECLARATION}" in r.stdout
    assert "replica check: UNEQUAL -- frame_projection/1" in r.stdout, "the strict verdict line stays truthful"
    assert f"ACCEPTED: equal modulo the C1 same-stamp class (2 movers) -- {C1_DECLARATION}" in r.stdout and "REASON:" not in r.stdout and "REFUSED:" not in r.stdout
    assert "HEADLINE: pass" in r.stdout and ", C1 allowance declared)" in r.stdout
    assert re.search(r"^\[land\] live-db on Slurm PASS for merged tip [0-9a-f]{8} \(job 4242, generation g1; accepted as equal modulo the C1 same-stamp class \(2 movers\) -- ", r.stdout, re.M), r.stdout
    sb = (w.mock / "sbatch.calls").read_text().splitlines()[-1]
    parts = sb.split(" land ", 1)[1].split()
    assert len(parts) > 9 and parts[8] == str(w.shared / "gates" / "trees" / w.tip), "the 9th positional is TREE itself when no recovery pin is set"
    escaped = subprocess.run(["bash", "-c", 'printf "%q" "$1"', "_", C1_DECLARATION], capture_output=True, text=True, check=True).stdout
    assert sb.endswith(" " + escaped) and " " in escaped and "\\" in escaped, (sb, escaped)
    assert (w.mock / "accept.received").read_text() == C1_DECLARATION + "\n", "the job received the declaration verbatim through the escape"
    st = json.load(open(state_path))
    assert st["equal"] is False and st["c1_class"] is True and st["accepted_as"] == "c1_class" and st["n_movers"] == 2 and st["declaration"] == C1_DECLARATION
    assert st["ok"] and st["headline"] == "pass" and st["problems"] == [] and st["generation"] == "g1" and st["job_check_accepted_as"] == "c1_class"
    # declared, but the diff is not the class alone (the cutover landing's decode_policy_match rows): refused as before
    (w.mock / "job_c1_class").unlink()
    (w.mock / "job_unequal").write_text("")
    r = _live_db(w)
    assert r.returncode == 5 and "LIVE-DB REPLICA UNEQUAL (job 4242)" in r.stdout and "ACCEPTED:" not in r.stdout, r.stdout
    assert "REFUSED: replica check UNEQUAL: projection differs: " in r.stdout and '"decode_policy_match": 1260' in r.stdout and "not the C1 class alone: the declaration does not cover it" in r.stdout
    st = json.load(open(state_path))
    assert st["accepted_as"] is None and st["c1_class"] is False and not st["ok"] and st["declaration"] == C1_DECLARATION and st["columns_differing"]["decode_policy_match"] == 1260
    (w.mock / "job_unequal").unlink()
    # declared, equal: accepted as equal, no clause anywhere
    r = _live_db(w)
    assert r.returncode == 0 and "ACCEPTED:" not in r.stdout and "accepted as" not in r.stdout and "replica check: equal -- frame_projection/1" in r.stdout, r.stdout
    st = json.load(open(state_path))
    assert st["accepted_as"] == "equal" and st["equal"] and st["c1_class"] is False and st["ok"]
    # a job that accepted the class WITHOUT a declaration here (its summary says c1_class) is refused by name, whatever it published
    del w.env["LAND_ACCEPT_C1_CLASS"]
    (w.mock / "job_claims_c1").write_text("")
    r = _live_db(w)
    assert r.returncode == 5 and "REFUSED: the job accepted its check as 'c1_class', not as None here (declaration here: False)" in r.stdout, r.stdout
    assert "NOT declared: no allowance without LAND_ACCEPT_C1_CLASS" in r.stdout and not json.load(open(state_path))["ok"]
    (w.mock / "job_claims_c1").unlink()
    # a recovery with the declaration: the module that folds and checks is the PINNED tree's, so a pin whose replica.py knows no
    # --accept-c1-class is refused by name before any submission; a pin that carries the allowance is accepted with the
    # 9th positional the pin and the 10th the declaration
    rt = "a" * 40
    (w.shared / "gates" / "trees" / rt / "src" / "sisteer" / "registry").mkdir(parents=True)
    (w.shared / "gates" / "trees" / rt / "src" / "sisteer" / "registry" / "replica.py").write_text("# the pre-allowance module\n")
    w.env["LIVE_DB_REPLICA_TREE"] = rt
    w.env["LAND_ACCEPT_C1_CLASS"] = C1_DECLARATION
    (w.mock / "job_c1_class").write_text("")
    n_sb = len((w.mock / "sbatch.calls").read_text().splitlines())
    r = _live_db(w)
    assert r.returncode == 5 and f"REFUSED: the replica tree {rt[:12]} (LIVE_DB_REPLICA_TREE) predates the C1 allowance" in r.stdout and "nothing submitted" in r.stdout, r.stdout
    assert len((w.mock / "sbatch.calls").read_text().splitlines()) == n_sb and "ACCEPTED:" not in r.stdout
    (w.shared / "gates" / "trees" / rt / "src" / "sisteer" / "registry" / "replica.py").write_text("# a module that knows --accept-c1-class\n")
    r = _live_db(w)
    assert r.returncode == 0 and "ACCEPTED: equal modulo the C1 same-stamp class (2 movers)" in r.stdout, r.stdout + r.stderr
    parts = (w.mock / "sbatch.calls").read_text().splitlines()[-1].split(" land ", 1)[1].split()
    assert parts[8] == str(w.shared / "gates" / "trees" / rt) and (w.mock / "accept.received").read_text() == C1_DECLARATION + "\n"
    st = json.load(open(state_path))
    assert st["ok"] and st["accepted_as"] == "c1_class" and st["replica_tree"] == rt


@NEEDS_RSYNC
def test_live_db_slurm_pins_the_replica_side_to_another_tree_in_a_recovery_and_refuses_a_job_that_ignored_the_pin(tmp_path):
    """The replica side runs the pinned tree's code (recover.sh / LIVE_DB_REPLICA_TREE): the tree rides as the land mode's 9th
    positional, the land/2 contract, while the tests run from the tip; the verdict requires identity.built_by_tree == the pinned tree
    and summary.tree == the tip. The sbatch script submitted is the CHAIN's own replica.sbatch: probed for the contract first, shipped
    beside the head into the submission's own directory, sha256-verified there -- never an older script a tree may carry, which ignores
    the positional (the next test), and never the tip's. A pin that is not exported refuses before any submission; a job whose identity
    names the tip (it ignored the pin) or whose tests ran from the pinned tree is refused by name."""
    w = _live_db_world(tmp_path)
    rt = "a" * 40
    w.env["LIVE_DB_REPLICA_TREE"] = rt
    n0 = len((w.mock / "sbatch.calls").read_text().splitlines()) if (w.mock / "sbatch.calls").exists() else 0
    r = _live_db(w)
    assert r.returncode == 5 and "LIVE_DB_REPLICA_TREE" in r.stdout and "is not exported" in r.stdout and "tree_export.sh" in r.stdout, r.stdout
    assert not (w.mock / "sbatch.calls").exists() or len((w.mock / "sbatch.calls").read_text().splitlines()) == n0
    (w.shared / "gates" / "trees" / rt / "src" / "sisteer" / "registry").mkdir(parents=True)
    r = _live_db(w)
    out = r.stdout
    assert r.returncode == 0, out + r.stderr
    dsha = hashlib.sha256((LANDING / "replica.sbatch").read_bytes()).hexdigest()
    assert f"[land] live-db: RECOVERY -- the replica side runs the code of tree {rt[:12]}" in out and f"replica code from tree {rt[:8]}" in out
    assert "the dispatcher is the chain's own " in out and f"/replica.sbatch (sha256 {dsha[:12]}; {LAND_CONTRACT})" in out, out
    sb = (w.mock / "sbatch.calls").read_text().splitlines()[-1]
    script = sb.split(" land ", 1)[0].split()[-1]
    parts = sb.split(" land ", 1)[1].split()
    assert len(parts) == 9 and parts[0] == str(w.shared / "gates" / "trees" / w.tip) and parts[8] == str(w.shared / "gates" / "trees" / rt), parts
    assert script == os.path.join(os.path.dirname(parts[1]), "replica.sbatch"), (script, "the dispatcher rides beside the head, in the submission's own directory")
    assert Path(script).read_bytes() == (LANDING / "replica.sbatch").read_bytes(), "the shipped dispatcher is the chain's own file, byte for byte"
    assert f"/gates/trees/{rt}/scripts" not in sb and f"/gates/trees/{w.tip}/scripts" not in sb, "neither the pinned tree's nor the tip's script is submitted in a recovery"
    assert f"[land] live-db: shipped the chain's dispatcher (sha256 {dsha[:12]}) beside the head: {script}" in out and f", the chain dispatcher {dsha[:12]})" in out
    rc_state = json.load(open(Path(w.root) / "state" / "replica_check.json"))
    assert rc_state["ok"] and rc_state["replica_tree"] == rt and rc_state["tip"] == w.tip and rc_state["headline"] == "pass"
    assert rc_state["dispatcher"] == {"source": "chain", "sha256": dsha, "path": script}
    # the job ignored the pin (its identity names the tip): refused by name
    (w.mock / "job_ignores_rtree").write_text("")
    r = _live_db(w)
    assert r.returncode == 5 and f"not the pinned replica tree {rt[:12]} (LIVE_DB_REPLICA_TREE)" in r.stdout and "LIVE-DB RESULT IDENTITY MISMATCH" in r.stdout, r.stdout
    (w.mock / "job_ignores_rtree").unlink()
    # the tests ran from the pinned tree instead of the tip: refused by name
    (w.mock / "job_tests_from_rtree").write_text("")
    r = _live_db(w)
    assert r.returncode == 5 and f"summary.json's tests tree is '{rt[:12]}', not the tip {w.tip[:12]} (a recovery runs the tests from the tip)" in r.stdout, r.stdout
    (w.mock / "job_tests_from_rtree").unlink()
    # a malformed pin refuses before anything
    w.env["LIVE_DB_REPLICA_TREE"] = "nothex"
    n = len((w.mock / "sbatch.calls").read_text().splitlines())
    r = _live_db(w)
    assert r.returncode == 5 and "LIVE_DB_REPLICA_TREE must be a 40-hex sha" in r.stdout and len((w.mock / "sbatch.calls").read_text().splitlines()) == n
    del w.env["LIVE_DB_REPLICA_TREE"]
    # without the pin the plain rule stands: built_by_tree must be the tip; the sbatch has eight positionals and the tip's script
    r = _live_db(w)
    rc_state = json.load(open(Path(w.root) / "state" / "replica_check.json"))
    assert r.returncode == 0 and rc_state["replica_tree"] is None and rc_state["dispatcher"] == {"source": "tip", "sha256": None, "path": None}
    sb = (w.mock / "sbatch.calls").read_text().splitlines()[-1]
    assert len(sb.split(" land ", 1)[1].split()) == 8 and f" {w.shared}/gates/trees/{w.tip}/scripts/registry/landing/replica.sbatch land " in sb


def _dispatch_world(tmp_path):
    """Two exported trees (the tip and a pin: bare directories named by their shas, each with src/sisteer/registry/), a replica root, a
    head file, srun shimmed to exec its command, git shimmed to FAIL (so replica.sbatch's tree_sha falls back to the directory's name)
    and a venv whose python RECORDS the module call -- PYTHONPATH and argv -- instead of running it."""
    tip, pin, old = "e" * 40, "a" * 40, "b" * 40
    trees = tmp_path / "trees"
    for t in (tip, pin, old):
        (trees / t / "src" / "sisteer" / "registry").mkdir(parents=True)
    # the tip and the pin carry a module that knows the allowance (replica.sbatch probes the text before any work); `old` predates it
    for t in (tip, pin):
        (trees / t / "src" / "sisteer" / "registry" / "replica.py").write_text("# stands in for the module; it knows --accept-c1-class\n")
    (trees / old / "src" / "sisteer" / "registry" / "replica.py").write_text("# a module from before the allowance\n")
    root = tmp_path / "root"
    root.mkdir()
    (root / "config.toml").write_text("log_root = 'x'\n")
    (root / "log").mkdir()
    (root / "records").symlink_to(root / "log")
    head = tmp_path / "head.json.gz"
    head.write_bytes(b"never read: the venv's python is a shim")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / ".gate-ready.json").write_text("{}")
    calls = tmp_path / "module.calls"
    _exe(venv / "bin" / "python", f'#!/bin/bash\necho "PYTHONPATH=$PYTHONPATH argv=$*" >> "{calls}"\nexit 0\n')
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    _exe(bin_ / "srun", '#!/bin/bash\nwhile [ $# -gt 0 ]; do case "$1" in --ntasks=*|-n) shift ;; *) break ;; esac; done\nexec "$@"\n')
    _exe(bin_ / "git", '#!/bin/bash\nexit 1\n')
    env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", REPLICA_WORK_ROOT=str(tmp_path / "nodelocal"), SLURM_JOB_ID="778", SLURM_CPUS_PER_TASK="2",
               USER=os.environ.get("USER", "tester"))
    nine = [str(trees / tip), str(head), str(venv), "probe", str(root), str(tmp_path / "out"), "0", "all", str(trees / pin)]
    return types.SimpleNamespace(tip=tip, pin=pin, old=old, trees=trees, calls=calls, env=env, nine=nine)


@NEEDS_RSYNC
def test_the_pre_c9_dispatcher_runs_the_tips_code_under_nine_positionals_so_a_recovery_ships_the_chains_own(tmp_path):
    """THE BLOCKING CASE: the first recovery pins the bootstrap generation, whose tree carries the earlier
    replica.sbatch -- a land mode of EIGHT positionals. Submitted from that tree with nine, it ignores REPLICA_TREE and runs the TIP's
    module (PYTHONPATH and --tree-sha from TREE): the replica side runs the code the recovery exists to escape, and `replica land`
    publishes its generation before the grading host identity check can refuse it. Shown here on the ACTUAL script's code (vendored, pinned)
    beside the chain's dispatcher under the same nine arguments (PYTHONPATH and --tree-sha from the pin; the tests' --tree the tip).
    `replica.sbatch contract` tells the two apart before any work; live_db_slurm.sh refuses a dispatcher without the contract before
    taking a head, syncing or submitting, and otherwise ships ITS OWN replica.sbatch beside the head, sha256-verified, and submits that
    path; without a pin the knob is not consulted and the tip's own script is submitted as before."""
    d = _dispatch_world(tmp_path / "disp")
    old = tmp_path / "pre_c9.sbatch"
    old.write_bytes(_pre_c9_dispatcher_bytes())
    # the actual earlier script, nine positionals: the TIP's code and sha on the replica side; the 9th positional is never read
    r = subprocess.run(["bash", str(old), "land", *d.nine], capture_output=True, text=True, env=d.env, timeout=120)
    assert r.returncode == 0 and "RESULT mode=land rc=0" in r.stdout, r.stdout + r.stderr
    call = d.calls.read_text().splitlines()[-1]
    assert call.startswith(f"PYTHONPATH={d.trees / d.tip}/src argv=-m sisteer.registry.replica land ") and f"--tree-sha {d.tip} " in call, call
    assert str(d.trees / d.pin) not in call and "replica_tree=" not in r.stdout, "the older land mode has eight positionals: the pin is ignored"
    # the chain's dispatcher, the same nine: the PIN's code and sha on the replica side, the tip as the tests' tree
    r = subprocess.run(["bash", str(LANDING / "replica.sbatch"), "land", *d.nine], capture_output=True, text=True, env=d.env, timeout=120)
    assert r.returncode == 0 and f"land: tree={d.tip} replica_tree={d.pin} " in r.stdout, r.stdout + r.stderr
    call = d.calls.read_text().splitlines()[-1]
    assert call.startswith(f"PYTHONPATH={d.trees / d.pin}/src argv=-m sisteer.registry.replica land ") and f"--tree-sha {d.pin} " in call, call
    assert f"--tree {d.trees / d.tip} " in call, call
    # the contract probe tells them apart before any work: no work directory, nothing on disk
    probe_env = dict(d.env, REPLICA_WORK_ROOT=str(tmp_path / "probe-work"))
    r = subprocess.run(["bash", str(LANDING / "replica.sbatch"), "contract"], capture_output=True, text=True, env=probe_env)
    assert r.returncode == 0 and r.stdout.strip() == LAND_CONTRACT and not (tmp_path / "probe-work").exists(), r.stdout + r.stderr
    r = subprocess.run(["bash", str(old), "contract"], capture_output=True, text=True, env=probe_env)
    assert r.returncode == 2 and "unknown mode contract" in r.stdout and "land/" not in r.stdout, r.stdout + r.stderr
    # the land/3 contract's 10th positional: the declaration reaches the module as --accept-c1-class, verbatim; nine positionals
    # (a recovery without the allowance) and eight (a plain landing) pass none
    r = subprocess.run(["bash", str(LANDING / "replica.sbatch"), "land", *d.nine, C1_DECLARATION], capture_output=True, text=True, env=d.env, timeout=120)
    assert r.returncode == 0 and f"land: tree={d.tip} replica_tree={d.pin} " in r.stdout and f"accept_c1_class={C1_DECLARATION}" in r.stdout, r.stdout + r.stderr
    call = d.calls.read_text().splitlines()[-1]
    assert call.endswith(f" --accept-c1-class {C1_DECLARATION}") and f"--tree-sha {d.pin} " in call, call
    r = subprocess.run(["bash", str(LANDING / "replica.sbatch"), "land", *d.nine], capture_output=True, text=True, env=d.env, timeout=120)
    assert r.returncode == 0 and "accept_c1_class=none" in r.stdout and "--accept-c1-class" not in d.calls.read_text().splitlines()[-1], r.stdout + r.stderr
    r = subprocess.run(["bash", str(LANDING / "replica.sbatch"), "land", *d.nine[:8]], capture_output=True, text=True, env=d.env, timeout=120)
    assert r.returncode == 0 and "accept_c1_class=none" in r.stdout and "--accept-c1-class" not in d.calls.read_text().splitlines()[-1], r.stdout + r.stderr
    # a pinned tree that predates the allowance (its module knows no --accept-c1-class) with the declaration: refused before any work, exit 2,
    # the module never called (it would otherwise die on the argument after a full submission); without the declaration the same pin
    # runs as before
    n_calls = len(d.calls.read_text().splitlines())
    out_old = tmp_path / "out-old"
    r = subprocess.run(["bash", str(LANDING / "replica.sbatch"), "land", *d.nine[:5], str(out_old), "0", "all", str(d.trees / d.old), C1_DECLARATION],
                       capture_output=True, text=True, env=d.env, timeout=120)
    assert r.returncode == 2 and f"the replica tree {d.trees / d.old} predates the C1 allowance" in r.stdout and "nothing run" in r.stdout, r.stdout + r.stderr
    assert len(d.calls.read_text().splitlines()) == n_calls and not out_old.exists()
    r = subprocess.run(["bash", str(LANDING / "replica.sbatch"), "land", *d.nine[:8], str(d.trees / d.old)], capture_output=True, text=True, env=d.env, timeout=120)
    assert r.returncode == 0 and f"replica_tree={d.old} " in r.stdout and d.calls.read_text().splitlines()[-1].startswith(f"PYTHONPATH={d.trees / d.old}/src "), r.stdout + r.stderr
    # live_db_slurm.sh pointed at the earlier script in a recovery: refused by name BEFORE the head, the mirror sync or any submission
    w = _live_db_world(tmp_path)
    rt = "a" * 40
    (w.shared / "gates" / "trees" / rt / "src" / "sisteer" / "registry").mkdir(parents=True)
    w.env["LIVE_DB_REPLICA_TREE"] = rt
    w.env["LIVE_DB_DISPATCHER"] = str(old)
    r = _live_db(w)
    assert r.returncode == 5 and f"REFUSED: the dispatcher {old} does not carry the land/3 contract" in r.stdout and "unknown mode contract" in r.stdout, r.stdout
    assert "runs the tip's code on the replica side" in r.stdout
    assert not (w.state / "land" / "probe.head.json.gz").exists() and not (w.mock / "systemctl.calls").exists() and not (w.mock / "sbatch.calls").exists()
    # a dispatcher path that does not exist refuses the same way
    w.env["LIVE_DB_DISPATCHER"] = str(tmp_path / "absent.sbatch")
    r = _live_db(w)
    assert r.returncode == 5 and "REFUSED: no dispatcher at" in r.stdout and not (w.mock / "sbatch.calls").exists(), r.stdout
    # the default: the chain's own file is probed, shipped beside the head and submitted from there
    del w.env["LIVE_DB_DISPATCHER"]
    r = _live_db(w)
    assert r.returncode == 0, r.stdout + r.stderr
    sb = (w.mock / "sbatch.calls").read_text().splitlines()[-1]
    script = sb.split(" land ", 1)[0].split()[-1]
    assert os.path.basename(os.path.dirname(script)).startswith(f"land-probe-{w.tip[:8]}-") and Path(script).read_bytes() == (LANDING / "replica.sbatch").read_bytes(), sb
    assert "/scripts/registry/landing/replica.sbatch" not in sb, "no tree's script is submitted in a recovery"
    # no pin: the knob is not consulted; the tip's own script is submitted, as in every plain landing
    del w.env["LIVE_DB_REPLICA_TREE"]
    w.env["LIVE_DB_DISPATCHER"] = str(old)
    r = _live_db(w)
    assert r.returncode == 0, r.stdout + r.stderr
    sb = (w.mock / "sbatch.calls").read_text().splitlines()[-1]
    assert f" {w.shared}/gates/trees/{w.tip}/scripts/registry/landing/replica.sbatch land " in sb and str(old) not in sb


def test_pre_c9_dispatcher_fixture_runs_the_commits_code():
    """The vendored copy's code (``_pre_c9_identity``: every line but the plain comments, which are reworded, with two log messages'
    plain-text tails masked) hashes to the digest taken from commit bdd22205c's object, with or without history, so reworded plain
    comments and those two printed tails are all that sets the copy apart. The old script names no REPLICA_TREE; the chain's does."""
    data = _pre_c9_dispatcher_bytes()
    assert b"REPLICA_TREE" not in data and b"REPLICA_TREE" in (LANDING / "replica.sbatch").read_bytes()
    digest = hashlib.sha256(_pre_c9_identity(data)).hexdigest()
    assert digest == PRE_C9_CODE_SHA256, f"{PRE_C9_FIXTURE.name}'s code is not {PRE_C9_REL}@{PRE_C9_SHA[:8]}'s: sha256 {digest}"
    # the normaliser drops plain comments only: a code line, a directive and a trailing comment all survive it
    assert _pre_c9_code(b"#!/bin/bash\n#SBATCH --qos=high\n# a note\n  # an indented note\n# shellcheck disable=SC1091\nx=1  # why\n") \
        == b"#!/bin/bash\n#SBATCH --qos=high\n# shellcheck disable=SC1091\nx=1  # why\n"
    # the mask hides a plain-text tail and nothing else: two wordings mask alike, and a tail with an expansion or a substitution matches
    # no pattern, so the identity refuses it
    unequal = b'      log "UNEQUAL build: keeping the work database as $OUT/registry.sqlite.unequal for %s"\n'
    head = b'need_head() { [ -s "$1" ] || { log "no head file at $1 (registry replica head --out, shipped by %s)"; exit 2; }; }\n'
    assert _pre_c9_identity(unequal % b"one reader" + head % b"a host") == _pre_c9_identity(unequal % b"another" + head % b"the host")
    for tail in (b"$WORK", b"`id`", b'x" ; rm -rf "$OUT'):
        with pytest.raises(AssertionError, match="matches 0 lines"):
            _pre_c9_identity(unequal % tail + head % b"a host")
    with pytest.raises(AssertionError, match="matches 2 lines"):
        _pre_c9_identity(unequal % b"a" + unequal % b"b" + head % b"a host")


def test_pre_c9_dispatcher_fixture_matches_the_git_object():
    """The second level of the pin (the precedent: tests/test_registry_worklist_drivers.py), wherever the commit is reachable: ``git show
    <sha>:<path>`` is the recorded object byte for byte, and its code equals the vendored copy's line for line, the two inert log tails
    masked. In a shallow exact-sha
    export (the battery's and the gate's trees) this SKIPS naming that, and the code digest in
    ``test_pre_c9_dispatcher_fixture_runs_the_commits_code`` is the guard there."""
    try:
        probe = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{PRE_C9_SHA}^{{commit}}"], capture_output=True, text=True)
    except FileNotFoundError:
        pytest.skip("git is not installed here; the code digest covers the fixture")
    if probe.returncode != 0:
        pytest.skip(f"commit {PRE_C9_SHA[:8]} is not reachable from {REPO} (a shallow exact-sha export carries no history): "
                    f"{probe.stderr.strip() or f'rc {probe.returncode}'}; the code digest covers the fixture")
    shown = subprocess.run(["git", "-C", str(REPO), "show", f"{PRE_C9_SHA}:{PRE_C9_REL}"], capture_output=True, check=True).stdout
    assert hashlib.sha256(shown).hexdigest() == PRE_C9_OBJECT_SHA256, f"git show {PRE_C9_SHA[:8]}:{PRE_C9_REL} is not the recorded object"
    assert _pre_c9_identity(shown) == _pre_c9_identity(_pre_c9_dispatcher_bytes()), f"{PRE_C9_FIXTURE.name}'s code != git show {PRE_C9_SHA[:8]}:{PRE_C9_REL}'s"


@NEEDS_RSYNC
def test_live_db_slurm_defers_once_on_a_mid_fold_head_and_refuses_when_no_cycle_completes(tmp_path, monkeypatch):
    """The head exits 6 (a cursor present): the step waits for state/harvest_cycle.json to move (the harvester's next completed cycle)
    and retries once; with no cycle inside the bound it refuses by name without submitting anything."""
    w = _live_db_world(tmp_path)
    db = DB.db_path(w.root)
    # a real cursor: a plain file cut after its first window (the harvester died mid-file)
    recs = [_now_rec("run_patch", {"run_id": w.ids["run_ids"][1], "fields": {"job_name": f"c{i}"}}, i) for i in range(4)]
    R.write_records(os.path.join(w.root, "records", "cut", "000001-cut.jsonl"), recs)
    monkeypatch.setattr(H, "LARGE_FILE_BYTES", 0)
    monkeypatch.setattr(H, "ITEM_ROWS_PER_TRANSACTION", 2)
    real = H.Ingestor.ingest_chunk
    commits = {"n": 0}

    def cut(self, chunk, now):
        if commits["n"] == 1:
            raise RuntimeError("cut")
        real(self, chunk, now)
        commits["n"] += 1
    monkeypatch.setattr(H.Ingestor, "ingest_chunk", cut)
    with pytest.raises(RuntimeError):
        _ingest(w.root)
    assert _q(db, "SELECT COUNT(*) FROM record_cursor") == [(1,)]
    cycle = Path(w.root) / "state" / "harvest_cycle.json"
    before = cycle.read_text()
    # no cycle completes within the bound (1 min; the poll every second): refused, nothing submitted
    w.env["LIVE_DB_DEFER_MIN"] = "0"
    r = _live_db(w)
    assert r.returncode == 5 and "no cursor-free boundary within 0 min" in r.stdout and "deferring until the harvester's next completed cycle" in r.stdout
    assert "completed no cycle within 0 min" in r.stdout and not (w.mock / "sbatch.calls").exists() and not (w.mock / "systemctl.calls").exists()
    # the harvester finishes the file during the deferral: the retry takes the head and the step goes on
    monkeypatch.setattr(H.Ingestor, "ingest_chunk", real)
    _ingest(w.root)
    assert cycle.read_text() != before and _q(db, "SELECT COUNT(*) FROM record_cursor") == [(0,)]
    r = _live_db(w)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[land] live-db job 4242" in r.stdout and RP.read_head(str(w.state / "land" / "probe.head.json.gz"))["cursor_free"]


def _iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


@NEEDS_RSYNC
def test_live_db_slurm_mirror_sync_waits_out_a_running_pass_and_ships_only_after_a_fresh_pass_that_started_after_the_head(tmp_path):
    """The record-log mirror unit runs back to back, so a pass running when the step enters took its census BEFORE the head (land
    job 925158: MIRROR_BEHIND by one harvester fact file written seconds before the head, after the chain had waited on exactly that
    pass). The step waits that pass out, starts a fresh one and ships only when the fresh pass's own record covers the head: a start
    stamp after taken_utc, the destination's row switched, the manifest it switched to naming every file the head names with its sha.
    A budget-stopped pass (exit 0, nothing switched) is followed by another; a census that misses a head file refuses by name before
    anything ships; a pass that never ends refuses by name within the bound (exit 4); the retired override knob refuses before the head;
    a world with no mirror skips the step."""
    w = _live_db_world(tmp_path)
    builds = w.shared / "registry" / "replica" / "builds"
    # a stale pass running at entry (started long before any head), ending after three `show` reads
    (w.mock / "unit.state").write_text("activating\n")
    (w.mock / "unit.start").write_text("Mon 2000-01-03 00:00:00.000000 UTC\n")
    (w.mock / "unit.ticks").write_text("3\n")
    r = _live_db(w)
    out = r.stdout
    assert r.returncode == 0, out + r.stderr
    head = RP.read_head(str(w.state / "land" / "probe.head.json.gz"))
    marks = ("a mirror pass is running (started 2000-01-03T00:00:00.000000Z; the head was taken " + head["taken_utc"], "stale pass ended ",
             "starting a fresh mirror pass (systemctl --user start registry-mirrors.service)", "fresh pass started ",
             "covers head: yes (started after taken_utc " + head["taken_utc"], "@ live-db-mirror-sync:end", "@ live-db-tree:start")
    idx = [out.index(m) for m in marks]
    assert idx == sorted(idx) and out.index("@ live-db-head:end") < idx[0], out
    fresh = re.search(r"fresh pass started (\S+) ended (\S+) \(success; ExecMainStartTimestamp\)", out)
    assert fresh and _iso(fresh.group(1)) > _iso(head["taken_utc"]) and _iso(fresh.group(2)) >= _iso(fresh.group(1)), out
    stale_end = re.search(r"stale pass ended (\S+)", out).group(1)
    assert _iso(head["taken_utc"]) < _iso(stale_end) <= _iso(fresh.group(1)), out
    manifest = re.search(r"covers head: yes \(started after taken_utc \S+; manifest (CHECKSUMS-\d{12}\.sha256) \(sha12 ([0-9a-f]{12}), acked\) holds all (\d+) files the head names, (\d+) with the head's sha", out)
    assert manifest and int(manifest.group(3)) == head["n_files"] == int(manifest.group(4)), out
    mpath = Path(w.root) / "manifests" / manifest.group(1)
    assert mpath.is_file() and hashlib.sha256(mpath.read_bytes()).hexdigest()[:12] == manifest.group(2)
    calls = (w.mock / "systemctl.calls").read_text().splitlines()
    starts = [i for i, c in enumerate(calls) if c == "systemctl --user start registry-mirrors.service"]
    assert len(starts) == 1 and len(calls[:starts[0]]) >= 3 and all(c.startswith("systemctl --user show --timestamp=us+utc -p ") for c in calls[:starts[0]]), calls
    assert len((w.mock / "sbatch.calls").read_text().splitlines()) == 1
    # a budget-stopped first pass (mirrors sync exits 0 with nothing switched): the step reads the row, runs another pass, then ships
    (w.mock / "pass_budget_stopped").write_text("")
    r = _live_db(w)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "cluster_a did not switch to CHECKSUMS-" in r.stdout and "stopped=budget" in r.stdout and "it resumes on the next pass; starting one" in r.stdout
    assert r.stdout.count("fresh pass started ") == 2 and r.stdout.count("covers head: yes") == 1 and r.stdout.index("stopped=budget") < r.stdout.index("covers head: yes")
    assert (w.mock / "systemctl.calls").read_text().count("systemctl --user start registry-mirrors.service") == 3
    assert len((w.mock / "sbatch.calls").read_text().splitlines()) == 2 and not (w.mock / "pass_budget_stopped").exists()
    # the census of a pass that started after the head misses a file the head names: refused by name,
    # the manifest and the file named, nothing shipped, nothing submitted
    n_builds = len(os.listdir(builds))
    uri = sorted(u for u, s, _ in head["files"] if s != RP.EMPTY_SHA)[0]
    (w.mock / "manifest_drop").write_text(uri)
    r = _live_db(w)
    assert r.returncode == 5 and "does not cover the head" in r.stdout and "BEHIND the manifest CHECKSUMS-" in r.stdout, r.stdout
    assert f"lacks 1 and differs on 0 of the {head['n_files']} files the head names: {uri}" in r.stdout and "@ live-db-tree:start" not in r.stdout
    assert len(os.listdir(builds)) == n_builds and len((w.mock / "sbatch.calls").read_text().splitlines()) == 2
    (w.mock / "manifest_drop").unlink()
    # an EMPTY_SHA entry (a delivery the harvester could not read) must still be PRESENT on the mirror -- the cluster's require_mirror
    # demands presence of every entry and compares the sha only where there is one: absent, refused by name here
    ghost = "records/harvester/2026-10-06/unreadable-000000000000.jsonl"
    with sqlite3.connect(DB.db_path(w.root)) as conn:
        conn.execute("INSERT INTO record_delivery (file_uri, file_sha256, arrived_utc, status, refusal_reason) VALUES (?, ?, ?, 'refused', 'unreadable')",
                     (ghost, RP.EMPTY_SHA, "2026-10-06T00:00:00.000Z"))
    r = _live_db(w)
    assert r.returncode == 5 and f"lacks 1 and differs on 0 of the {head['n_files'] + 1} files the head names: {ghost}" in r.stdout, r.stdout
    assert "@ live-db-tree:start" not in r.stdout and len((w.mock / "sbatch.calls").read_text().splitlines()) == 2
    with sqlite3.connect(DB.db_path(w.root)) as conn:
        conn.execute("DELETE FROM record_delivery WHERE file_uri = ?", (ghost,))
    # the fresh pass misses a head file and the timer's next pass rewrites the SAME-NAMED manifest with it before the chain reads it
    # (production names manifests CHECKSUMS-<max ingest_seq>; a pass replaces the file): the row's switched state is not trusted against a
    # manifest whose digest is not the one that pass acked -- the chain waits for the rewriting pass and ships on ITS record
    (w.mock / "manifest_drop").write_text(uri)
    (w.mock / "pass_race").write_text("")
    r = _live_db(w)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "is not the one the pass acked" in r.stdout and "a newer pass has rewritten it; verifying the pass that rewrote it" in r.stdout, r.stdout
    assert "the timer started a mirror pass at " in r.stdout and r.stdout.count("covers head: yes") == 1 and r.stdout.index("rewritten it") < r.stdout.index("covers head: yes")
    assert (w.mock / "systemctl.calls").read_text().count("systemctl --user start registry-mirrors.service") == 7 and not (w.mock / "pass_race").exists()
    assert len((w.mock / "sbatch.calls").read_text().splitlines()) == 3
    names = sorted(os.listdir(Path(w.root) / "manifests"))
    assert len([n for n in names if n.startswith("CHECKSUMS-")]) == 1, names   # one name, replaced by every pass, as in production
    # a pass running at entry that never ends: refused by name within the bound (exit 4), no pass started, nothing shipped or submitted
    n_starts = (w.mock / "systemctl.calls").read_text().count(" start ")
    n_builds = len(os.listdir(builds))
    (w.mock / "unit.state").write_text("activating\n")
    (w.mock / "unit.start").write_text("Mon 2000-01-03 00:00:00.000000 UTC\n")
    w.env["LIVE_DB_MIRROR_WAIT_S"] = "2"
    r = _live_db(w)
    assert r.returncode == 4 and "REFUSED: no record-log mirror pass that started after the head (taken " in r.stdout, r.stdout
    assert "completed within 2s -- the pass started 2000-01-03T00:00:00.000000Z is still running; nothing shipped, nothing submitted" in r.stdout
    assert "@ live-db-tree:start" not in r.stdout and (w.mock / "systemctl.calls").read_text().count(" start ") == n_starts
    assert len(os.listdir(builds)) == n_builds and len((w.mock / "sbatch.calls").read_text().splitlines()) == 3
    del w.env["LIVE_DB_MIRROR_WAIT_S"]
    (w.mock / "unit.state").write_text("inactive\n")
    # the retired override command: refused by name before the head is taken
    (w.state / "land" / "probe.head.json.gz").unlink()
    w.env["LIVE_DB_MIRROR_SYNC"] = "true"
    r = _live_db(w)
    assert r.returncode == 5 and "LIVE_DB_MIRROR_SYNC is retired" in r.stdout and "LIVE_DB_MIRROR_UNIT=none skips the step" in r.stdout
    assert not (w.state / "land" / "probe.head.json.gz").exists() and (w.mock / "systemctl.calls").read_text().count(" start ") == n_starts
    del w.env["LIVE_DB_MIRROR_SYNC"]
    # a world with no mirror: the step is skipped, nothing asked of systemd, the stamps still printed
    w.env["LIVE_DB_MIRROR_UNIT"] = "none"
    (w.mock / "systemctl.calls").unlink()
    r = _live_db(w)
    assert r.returncode == 0 and not (w.mock / "systemctl.calls").exists() and "@ live-db-mirror-sync:end" in r.stdout, r.stdout + r.stderr
    del w.env["LIVE_DB_MIRROR_UNIT"]


def test_the_live_db_modules_read_their_database_and_pin_from_the_environment():
    # the env-keyed spelling is ASSEMBLED here: the landing-chain census of live-database openers (test_live_db_marker_is_registered_…)
    # reads every tests/test_registry_*.py for this very string, and this module opens no live database and carries no marker
    live_env = 'os.environ.get("REGISTRY_LIVE_DB' + '", "~/registry/registry.sqlite")'
    for name in ("test_registry_bon_decode_precedence.py", "test_registry_instrument_padding_identity.py"):
        text = (REPO / "tests" / name).read_text()
        assert live_env in text, name
    assert 'os.environ.get("REGISTRY_PIN_L1", ' in (REPO / "tests" / "test_registry_paper_spelling.py").read_text()
    assert live_env in (REPO / "tests" / "registry_live_fixtures.py").read_text()
    assert set(RP.live_db_modules(str(REPO))) == {"tests/test_registry_bon_decode_precedence.py", "tests/test_registry_instrument_padding_identity.py",
                                                  "tests/test_registry_paper_spelling.py", "tests/test_registry_paper_views_contrast_current.py", "tests/test_registry_twins_live.py"}


def test_cli_replica_round_trip_init_head_build_status_rollback(tmp_path, capsys):
    from sisteer.registry import cli
    root, _ = _world(tmp_path)
    rep_root = str(tmp_path / "rep")
    assert cli.main(["replica", "init", "--replica-root", rep_root, "--log-root", root]) == 0
    assert json.loads(capsys.readouterr().out)["current"] is None
    hp = str(tmp_path / "h.json.gz")
    assert cli.main(["--root", root, "replica", "head", "--out", hp, "--wait-min", "0"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["out"] == hp and "files" not in summary and summary["projection"]["version"] == "frame_projection/1"
    assert cli.main(["replica", "build", "--replica-root", rep_root, "--head", hp, "--work", str(tmp_path / "bw"), "--workers", "0", "--tree-sha", "f" * 40]) == 0
    out = capsys.readouterr().out
    assert "[replica build] wall" in out and "replica check: equal" in out and "published generation" in out
    assert cli.main(["replica", "status", "--replica-root", rep_root, "--json"]) == 0
    st = json.loads(capsys.readouterr().out)
    assert st["current"] and st["generations"][0]["mode"] == "build"
    assert cli.main(["replica", "verify-mirror", "--replica-root", rep_root, "--head", hp]) == 0
    assert cli.main(["replica", "rollback", "--replica-root", rep_root]) == RP.PublishRefused.exit_code
    assert "not a kept generation" in capsys.readouterr().err
    assert cli.main(["replica", "land", "--replica-root", rep_root]) == 1 and "--head is required" in capsys.readouterr().err
    assert cli.main(["replica", "projection", "--db", DB.db_path(root), "--no-rows"]) == 0
    assert json.loads(capsys.readouterr().out)["counts"]["rows"] == 3


# ---- the lifecycle restamp of named runs on the LIVE database (window.sh WINDOW_RESTAMP_KEYS) --------------------------------------------
def test_restamp_keys_plans_the_movers_lifecycle_move_applies_it_in_one_transaction_and_verifies_by_re_reading(tmp_path):
    """The certified C1 movers adopt the C1 order on the live database inside a window: `restamp_keys_plan` is a DRY RUN of the registry's own
    refold writer (`_refold_rows`, the body of `refold_apply`, under a savepoint rolled back) listing every (run, column, old, new) the apply
    would write -- here the one same-stamp mover, `lifecycle_state` judged -> superseded plus its provenance stamp -- in the lifecycle class;
    it refuses by name a key with no live row and ANY change outside the class (a column edited by hand that the refold would restore);
    `restamp_keys_apply` is bound to the recorded plan (re-computed, identical change set), rewrites in one transaction with the post-fold
    decode stamp kept, and verifies every planned (run, column) by re-reading; a second plan has nothing to change."""
    root, rid, head, kept = _pair_world(tmp_path)
    live = DB.db_path(root)
    key = I.canonical_json({"run_id": rid})
    (orig_campaign,) = _q(live, "SELECT campaign FROM run WHERE run_id = ?", rid)[0]
    assert _q(live, "SELECT lifecycle_state FROM run WHERE run_id = ?", rid) == [("judged",)]
    history_sql = "SELECT table_name, natural_key_json, written_utc, record_sha256, folded_state_json FROM row_history ORDER BY 1, 2, 3, 4"
    history_before = _q(live, history_sql)
    # the live database is UNEQUAL to its new-order rebuild before the restamp, in the live row AND in the stored history prefixes
    with DB.reading(live) as a, DB.reading(kept) as b:
        diff_before = H.verify_equal(a, b)
    assert diff_before["run"] > 0 and diff_before["row_history"] > 0, diff_before
    conn = DB.connect(live)
    try:
        plan = RP.restamp_keys_plan(conn, [rid], now=synth.t(900))
        assert _q(live, history_sql) == history_before, "a plan is a dry run: the history prefixes are rolled back with the live row"
        assert RP.history_mismatches(conn, [key]), "before the apply the stored prefixes are the old order's"
        assert plan["ok"] and plan["schema"] == RP.RESTAMP_SCHEMA and plan["keys"] == [key] and plan["n_keys"] == 1 and plan["unknown"] == [], plan
        in_class = [c for c in plan["changes"] if c["column"] in RP.RESTAMP_CLASS_COLUMNS]
        assert in_class == [{"run_id": rid, "column": "lifecycle_state", "old": "judged", "new": "superseded"}], plan["changes"]
        assert {c["column"] for c in plan["changes"]} <= set(RP.RESTAMP_CLASS_COLUMNS) | set(RP.RESTAMP_PROVENANCE_COLUMNS)
        assert plan["n_runs_lifecycle_moved"] == 1 and plan["runs_lifecycle_moved"] == [rid] and plan["out_of_class"] == [] and plan["n_would_rewrite"] == 1
        assert plan["columns"]["lifecycle_state"] == 1 and plan["class_columns"] == list(RP.RESTAMP_CLASS_COLUMNS) and "lifecycle_state" in RP.RESTAMP_CLASS_COLUMNS
        assert _q(live, "SELECT lifecycle_state FROM run WHERE run_id = ?", rid) == [("judged",)], "a plan writes nothing"
        assert RP.restamp_keys_plan(conn, [rid])["changes_sha256"] == plan["changes_sha256"], "the plan is a pure function of the database"
        # a key the database does not hold: refused by name, no change computed
        bad = RP.restamp_keys_plan(conn, [rid, "no_such_run:x"])
        assert not bad["ok"] and bad["unknown"] == [{"run_id": "no_such_run:x", "why": "no live run row"}] and bad["changes"] == [] and bad["n_keys"] == 2
        with pytest.raises(RP.RefoldInputRefused, match="not ok"):
            RP.restamp_keys_apply(conn, bad)
        # a natural key spelled as a mapping is the same key
        assert RP.restamp_keys_plan(conn, [{"run_id": rid}])["keys"] == [key]
        # the apply: one transaction, the planned values on the rows, every change verified by re-read
        rec = RP.restamp_keys_apply(conn, plan, now=synth.t(901))
        assert rec["verified"] and rec["n_rows_rewritten"] == 1 and rec["n_runs_changed"] == 1 and rec["n_verified"] == rec["n_changes"] == len(plan["changes"])
        assert rec["mismatches"] == [] and rec["plan_changes_sha256"] == plan["changes_sha256"] and rec["plan_utc"] == synth.t(900) and rec["n_keys"] == 1
    finally:
        conn.close()
    assert _q(live, "SELECT lifecycle_state, decode_policy_match FROM run WHERE run_id = ?", rid) == [("superseded", "unrecorded")]   # the tail kept the derived stamp
    assert _q(kept, "SELECT lifecycle_state FROM run WHERE run_id = ?", rid) == [("superseded",)], "the live row now reads as the new-order rebuild's"
    # the restamped live database EQUALS its new-order rebuild in every table verify_equal reads -- row_history's prefixes included
    with DB.reading(live) as a, DB.reading(kept) as b:
        diff_after = H.verify_equal(a, b)
    assert all(v == 0 for v in diff_after.values()), diff_after
    assert _q(live, history_sql) != history_before and rec["n_history_snapshots_verified"] >= 5 and rec["history_mismatches"] == 0
    conn = DB.connect(live)
    try:
        assert RP.history_mismatches(conn, [key]) == []
        # a history snapshot corrupted by hand is what the verification catches: the apply's re-read refuses and ROLLS BACK
        with DB.transaction(conn):
            conn.execute("UPDATE row_history SET folded_state_json = ? WHERE table_name = 'run' AND natural_key_json = ? AND written_utc = ?",
                         (I.canonical_json({"run_id": rid, "lifecycle_state": "queued"}), key, synth.t(300)))
        bad_hist = RP.history_mismatches(conn, [key])
        assert len(bad_hist) >= 1 and bad_hist[0]["key"] == key and json.loads(bad_hist[0]["stored"]) == {"run_id": rid, "lifecycle_state": "queued"}
        plan_h = RP.restamp_keys_plan(conn, [rid])
        assert plan_h["ok"] and plan_h["changes"] == []          # the live row is already the fold's; only the history is wrong
        rec_h = RP.restamp_keys_apply(conn, plan_h)                 # the refold rewrites the prefixes and the re-read agrees
        assert rec_h["verified"] and RP.history_mismatches(conn, [key]) == []
        again = RP.restamp_keys_plan(conn, [rid])
        assert again["ok"] and again["changes"] == [] and again["n_runs_lifecycle_moved"] == 0 and again["n_would_rewrite"] == 1
        # a hand-edited column OUTSIDE the class: the refold would restore it -> out of class, refused by name, nothing applied
        with DB.transaction(conn):
            conn.execute("UPDATE run SET campaign = 'edited-by-hand' WHERE run_id = ?", (rid,))
        ooc = RP.restamp_keys_plan(conn, [rid])
        assert not ooc["ok"] and ooc["out_of_class"] == [{"run_id": rid, "column": "campaign", "old": "edited-by-hand", "new": orig_campaign}] and ooc["n_out_of_class"] == 1
        with pytest.raises(RP.RefoldInputRefused, match="not ok"):
            RP.restamp_keys_apply(conn, ooc)
        assert _q(live, "SELECT campaign FROM run WHERE run_id = ?", rid) == [("edited-by-hand",)], "refused: nothing written"
        with DB.transaction(conn):
            conn.execute("UPDATE run SET campaign = ? WHERE run_id = ?", (orig_campaign, rid))
        # an ok plan whose change set moved since it was read: refused by name, nothing written
        moved = dict(RP.restamp_keys_plan(conn, [rid]), changes_sha256="0" * 64)
        with pytest.raises(RP.RefoldInputRefused, match="change set moved"):
            RP.restamp_keys_apply(conn, moved)
    finally:
        conn.close()


def test_restamp_keys_cli_writes_the_plan_refuses_an_unknown_key_by_name_and_applies_only_a_recorded_identical_plan(tmp_path, capsys):
    """`registry replica restamp-keys --db DB --keys FILE --out DIR` (what window.sh runs in its restamp slot): without --apply the plan,
    every lifecycle-class change on its own line and the verdict line LAST (the chain log tails the output), written to DIR/restamp_plan.json
    -- exit 13 and nothing written on an unknown key; `--apply` before a plan is refused; `--apply` re-computes the plan, requires the same
    change set, rewrites and verifies (DIR/restamp_apply.json), exit 0; a --keys that is not a list is refused."""
    from sisteer.registry import cli as CLI
    root, rid, head, kept = _pair_world(tmp_path)
    live = DB.db_path(root)
    out = tmp_path / "restamp"
    keys = tmp_path / "keys.json"
    keys.write_text(json.dumps([rid]))
    assert CLI.main(["--root", root, "replica", "restamp-keys", "--db", live, "--keys", str(keys), "--out", str(out), "--apply"]) == 13
    assert "no plan at" in capsys.readouterr().err and not (out / "restamp_apply.json").exists()
    flag = H.maintenance_path(root)
    assert CLI.main(["--root", root, "replica", "restamp-keys", "--db", live, "--keys", str(keys), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert f"  {rid} lifecycle_state: 'judged' -> 'superseded'" in lines and "OUT OF CLASS" not in text
    assert lines[-1].startswith("restamp keys: plan over 1 key(s): 1 run(s) move in the lifecycle class (") and "0 unknown key(s), 0 change(s) out of class -> OK (plan " in lines[-1]
    plan = json.load(open(out / "restamp_plan.json"))
    assert plan["ok"] and plan["schema"] == RP.RESTAMP_SCHEMA and _q(live, "SELECT lifecycle_state FROM run WHERE run_id = ?", rid) == [("judged",)]
    # an unknown key: refused by name, exit 13, the plan file says why, nothing applied
    keys2 = tmp_path / "keys2.json"
    keys2.write_text(json.dumps([rid, "ghost:run"]))
    assert CLI.main(["--root", root, "replica", "restamp-keys", "--db", live, "--keys", str(keys2), "--out", str(tmp_path / "r2")]) == 13
    text2 = capsys.readouterr().out
    assert "  UNKNOWN KEY ghost:run: no live run row" in text2 and text2.strip().splitlines()[-1].endswith("-> REFUSED (plan " + str(tmp_path / "r2" / "restamp_plan.json") + ")")
    assert json.load(open(tmp_path / "r2" / "restamp_plan.json"))["ok"] is False
    # a --keys file that is not a list: refused before anything is read
    (tmp_path / "bad.json").write_text("{}")
    assert CLI.main(["--root", root, "replica", "restamp-keys", "--db", live, "--keys", str(tmp_path / "bad.json"), "--out", str(out)]) == 13
    assert "a JSON list of run ids" in capsys.readouterr().err
    # WINDOW-ONLY: without the maintenance flag the window holds, the apply is refused by name and writes nothing
    assert not os.path.exists(flag)
    assert CLI.main(["--root", root, "replica", "restamp-keys", "--db", live, "--keys", str(keys), "--out", str(out), "--apply"]) == 13
    assert "no maintenance flag at" in capsys.readouterr().err and _q(live, "SELECT lifecycle_state FROM run WHERE run_id = ?", rid) == [("judged",)]
    os.makedirs(os.path.dirname(flag), exist_ok=True)
    with open(flag, "w") as fh:
        fh.write('{"reason": "test window"}\n')
    # the apply over the recorded plan, the flag held: exit 0, the record, the row moved, the kept rebuild agrees in every table
    assert CLI.main(["--root", root, "replica", "restamp-keys", "--db", live, "--keys", str(keys), "--out", str(out), "--apply"]) == 0
    text3 = capsys.readouterr().out.strip().splitlines()
    assert text3[-1].startswith("restamp keys: applied over 1 key(s): 1 row(s) rewritten, ") and "1 run(s) changed" in text3[-1] and "history snapshot(s) verified by re-read -> OK (record " in text3[-1]
    rec = json.load(open(out / "restamp_apply.json"))
    assert rec["verified"] and rec["plan_changes_sha256"] == plan["changes_sha256"] and rec["n_verified"] == rec["n_changes"] == plan["n_changes"] and rec["history_mismatches"] == 0
    assert _q(live, "SELECT lifecycle_state FROM run WHERE run_id = ?", rid) == _q(kept, "SELECT lifecycle_state FROM run WHERE run_id = ?", rid) == [("superseded",)]
    with DB.reading(live) as a, DB.reading(kept) as b:
        assert all(v == 0 for v in H.verify_equal(a, b).values()), "the restamped live database equals the new-order rebuild, row_history included"
    # the help names the action without the banned nouns (tests/test_registry_vocabulary.py reads the CLI help; here the action is reachable)
    assert "restamp-keys" in RP.ACTIONS


def test_restamp_keys_moves_the_row_provenance_to_the_new_orders_final_event_when_the_pair_sits_in_two_files(tmp_path, capsys):
    """A same-stamp `judged` + `superseded` pair whose events sit in DISTINCT record files draws a
    different FINAL event under the new order, and a rebuild derives the live row's `record_uri` (the final event's carrier) and
    `ingest_seq` (that file's ledger rank) from it -- a refold that kept the old row's carrier left a "verified" restamp unequal to the
    rebuild on exactly those two columns, which the plan never listed and the verification never read. Now `_refold_rows` writes the final
    event's carrier and rank (`refold_provenance`, the live fold's own rule), the plan lists and ADMITS the two provenance moves beside
    the stamps, and the apply verifies the WHOLE row against the refold (`live_row_mismatches`), so `harvest.verify_equal(live, rebuild)`
    reads zero in every table -- `run.record_uri` and `run.ingest_seq` included."""
    from sisteer.registry import cli as CLI
    root, rid, head, kept = _pair_world(tmp_path, split_files=True)
    live = DB.db_path(root)
    key = I.canonical_json({"run_id": rid})
    ((live_uri, live_seq),) = _q(live, "SELECT record_uri, ingest_seq FROM run WHERE run_id = ?", rid)
    ((kept_uri, kept_seq),) = _q(kept, "SELECT record_uri, ingest_seq FROM run WHERE run_id = ?", rid)
    assert live_uri.endswith("000002-judged.jsonl") and kept_uri.endswith("000003-lateral.jsonl") and live_seq != kept_seq, (live_uri, kept_uri)
    with DB.reading(live) as a, DB.reading(kept) as b:
        diff_before = H.verify_equal(a, b)
    assert diff_before["run"] > 0 and diff_before["row_history"] > 0, diff_before
    conn = DB.connect(live)
    try:
        before = RP.live_row_mismatches(conn, [key])
        assert {m["column"] for m in before} >= {"record_uri", "ingest_seq", "lifecycle_state"}, before
        assert next(m for m in before if m["column"] == "record_uri") == {"key": key, "column": "record_uri", "expected": kept_uri, "read": live_uri}
        plan = RP.restamp_keys_plan(conn, [rid], now=synth.t(900))
        assert plan["ok"] and plan["out_of_class"] == [], plan
        by_col = {c["column"]: c for c in plan["changes"] if c["run_id"] == rid}
        assert by_col["lifecycle_state"] == {"run_id": rid, "column": "lifecycle_state", "old": "judged", "new": "superseded"}
        assert by_col["record_uri"] == {"run_id": rid, "column": "record_uri", "old": live_uri, "new": kept_uri}
        assert by_col["ingest_seq"] == {"run_id": rid, "column": "ingest_seq", "old": live_seq, "new": kept_seq}
        assert set(by_col) <= set(RP.RESTAMP_CLASS_COLUMNS) | set(RP.RESTAMP_PROVENANCE_COLUMNS) and {"record_uri", "ingest_seq"} <= set(RP.RESTAMP_PROVENANCE_COLUMNS)
        assert plan["n_runs_lifecycle_moved"] == 1, "the provenance moves are admitted beside the lifecycle move, never a class move of their own"
        assert _q(live, "SELECT record_uri, ingest_seq FROM run WHERE run_id = ?", rid) == [(live_uri, live_seq)], "a plan writes nothing"
        rec = RP.restamp_keys_apply(conn, plan, now=synth.t(901))
        assert rec["verified"] and rec["row_mismatches"] == 0 and rec["unplanned_moves"] == [] and rec["history_mismatches"] == 0
        assert rec["n_rows_verified"] == 1 and rec["n_columns_verified"] > rec["n_changes"] == len(plan["changes"]) == rec["n_verified"], rec
        assert RP.live_row_mismatches(conn, [key]) == [] and RP.history_mismatches(conn, [key]) == []
        again = RP.restamp_keys_plan(conn, [rid])
        assert again["ok"] and again["changes"] == []
    finally:
        conn.close()
    assert (_q(live, "SELECT record_uri, ingest_seq, lifecycle_state FROM run WHERE run_id = ?", rid)
            == _q(kept, "SELECT record_uri, ingest_seq, lifecycle_state FROM run WHERE run_id = ?", rid) == [(kept_uri, kept_seq, "superseded")])
    with DB.reading(live) as a, DB.reading(kept) as b:
        diff_after = H.verify_equal(a, b)
    assert all(v == 0 for v in diff_after.values()), diff_after
    # the plan's chain-log rendering counts the provenance moves (the stamps, the carrier, the rank) on one line
    root2, rid2, _head2, _kept2 = _pair_world(tmp_path / "two", split_files=True)
    keys = tmp_path / "keys.json"
    keys.write_text(json.dumps([rid2]))
    assert CLI.main(["--root", root2, "replica", "restamp-keys", "--db", DB.db_path(root2), "--keys", str(keys), "--out", str(tmp_path / "plan2")]) == 0
    text = capsys.readouterr().out
    assert "  (+ 3 provenance change(s) in field_versions_json, record_uri, ingest_seq: the stamps of the moved columns, the final event's carrier and rank)" in text, text
    assert f"  {rid2} lifecycle_state: 'judged' -> 'superseded'" in text
    assert not [ln for ln in text.splitlines() if ln.startswith(f"  {rid2} record_uri") or ln.startswith(f"  {rid2} ingest_seq")], "provenance is counted, never listed per line"


def test_restamp_apply_verifies_the_whole_row_against_the_refold_and_rolls_back_a_rewrite_that_is_not_it(tmp_path):
    """The whole-row verification, exercised against the two ways a rewrite can be wrong without a planned value being
    wrong: (a) the pre-fix writer -- the live row rewritten with the OLD carrier and rank (nothing the plan listed differs, yet the row is
    not the refold's: `live_row_mismatches` names `record_uri` and `ingest_seq`); (b) a column the rewrite moved that the plan never listed
    (`unplanned_moves`). Both are `RestampVerifyFailed` and the WHOLE transaction rolls back: the live row, its history prefixes and the
    database's inequality to the rebuild stand exactly as before."""
    from sisteer.registry import reducer as RD
    root, rid, head, kept = _pair_world(tmp_path, split_files=True)
    live = DB.db_path(root)
    key = I.canonical_json({"run_id": rid})
    ((old_uri, old_seq),) = _q(live, "SELECT record_uri, ingest_seq FROM run WHERE run_id = ?", rid)
    history_sql = "SELECT table_name, natural_key_json, written_utc, record_sha256, folded_state_json FROM row_history ORDER BY 1, 2, 3, 4"
    history_before = _q(live, history_sql)
    conn = DB.connect(live)
    try:
        plan = RP.restamp_keys_plan(conn, [rid])
        assert plan["ok"] and {"record_uri", "ingest_seq"} <= {c["column"] for c in plan["changes"]}
        # (a) the pre-fix writer: every live row written with the old row's carrier and rank, whatever the order's final event
        real_write = RD.Reducer._write_live_row

        def keep_old_carrier(self, table, kd, state, fv, record_uri, ingest_seq):
            return real_write(self, table, kd, state, fv, old_uri, old_seq)

        with mock.patch.object(RD.Reducer, "_write_live_row", keep_old_carrier):
            plan_old = RP.restamp_keys_plan(conn, [rid])
            assert plan_old["ok"] and not {"record_uri", "ingest_seq"} & {c["column"] for c in plan_old["changes"]}, "that writer's plan lists no provenance move"
            with pytest.raises(RP.RestampVerifyFailed, match=r"2 row column\(s\) off the refold") as ei:
                RP.restamp_keys_apply(conn, plan_old)
        assert f"refold {key} record_uri" in str(ei.value) and f"refold {key} ingest_seq" in str(ei.value), str(ei.value)
        assert _q(live, "SELECT lifecycle_state, record_uri, ingest_seq FROM run WHERE run_id = ?", rid) == [("judged", old_uri, old_seq)], "rolled back"
        assert _q(live, history_sql) == history_before and RP.history_mismatches(conn, [key]), "the prefixes are still the old order's"
        # (b) a column moved that the plan never listed: the apply's own rewrite (the second refold inside the apply; the first is its
        # fresh plan's savepoint run) drifts a column outside the change set
        real_refold = RP._refold_rows
        calls: List[int] = []

        def drifting(conn_, keys_, table="run"):
            out = real_refold(conn_, keys_, table)
            calls.append(1)
            if len(calls) == 2:
                conn_.execute("UPDATE run SET campaign = 'drifted-by-the-rewrite' WHERE run_id = ?", (rid,))
            return out

        with mock.patch.object(RP, "_refold_rows", drifting):
            with pytest.raises(RP.RestampVerifyFailed, match=r"1 unplanned move\(s\)") as ei2:
                RP.restamp_keys_apply(conn, plan)
        assert f"unplanned {rid} campaign" in str(ei2.value) and f"refold {key} campaign" in str(ei2.value), str(ei2.value)
        assert _q(live, "SELECT lifecycle_state, record_uri FROM run WHERE run_id = ?", rid) == [("judged", old_uri)] and _q(live, history_sql) == history_before
        assert _q(live, "SELECT campaign FROM run WHERE run_id = ?", rid) != [("drifted-by-the-rewrite",)]
        with DB.reading(live) as a, DB.reading(kept) as b:
            assert H.verify_equal(a, b)["run"] > 0, "nothing of either refused apply reached the database"
        # the honest writer: the same plan applies, verifies whole, and the database equals the rebuild
        rec = RP.restamp_keys_apply(conn, plan)
        assert rec["verified"] and rec["row_mismatches"] == 0 and rec["unplanned_moves"] == []
    finally:
        conn.close()
    with DB.reading(live) as a, DB.reading(kept) as b:
        assert all(v == 0 for v in H.verify_equal(a, b).values())
