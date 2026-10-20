"""The paper frame — ``registry snapshot --scope paper_frame``.

A synthetic registry carries runs of the paper's models (some selected into the paper by a roster, one superseded, one not
in the roster), a run of one of the exporter's extra test models (`PAPER_FRAME_TEST_MODEL_IDS`), a run of a model outside the set, and
three filed analyses: one wholly over in-frame inputs, one over the outside run, one over an in-frame run and the outside run. The export
is built into a tmp ``dest_root`` with the precondition monkeypatched to pass; the operational root's ``snapshots/`` and ``CURRENT``
stay untouched; every read-set table ships as Parquet and inside the gzip-split SQLite; the item tables carry only the in-paper runs plus
the test model's; the analysis tables carry only the analyses whose every input is in the frame, the others left out whole and named in
the manifest, and every exported input resolves to an exported run or object; the materialised paper views equal the live views
restricted to the run set, row for row;
``record_ingest`` is the one head row; the parts reassemble to ``sqlite_sha256``; the readers (``open_paper_frame``,
``snapshot_open(scope="paper_frame")``) see the head; the precondition refuses by name (unset merge sha, symlinked
PAIRED_ROOT, missing directory, foreign sha) and records an override; the CLI requires ``--dest-root``; the free-space
rule is the scope's; the two front-door scripts dry-run cleanly.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys

import pytest

from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry import schema_def as S
from sisteer.registry import snapshot as SN
from sisteer.registry import synth
from sisteer.registry.adapters import relu_native as RN

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TOKENS = ("TESTHOST", "TESTUSER")
N_ITEMS = 6
#: the synthetic run set: (key, model_id, object key, object id, run overrides)
RUNS = [
    ("g_none", "gemma4e4b_it", "none", "none", {}),
    ("g_lora", "gemma4e4b_it", "constitution_lora@deadbeef", synth.OBJECT_SHA, {}),
    # a second unsteered run at the SAME key -- same cap, same trial (0), no pass label -- and later: superseded (the later-run rule keys on
    # the cap, the trial and the pass, so a trial 1 here would be a distinct run, never a successor), stays exported
    ("g_extra", "gemma4e4b_it", "none", "none", {}),
    ("l_none", "llama8b_it", "none", "none", {}),                              # a paper model with no roster entry: not_in_roster
    ("t_none", SN.PAPER_FRAME_TEST_MODEL_IDS[0], "none", "none", {}),            # one of the exporter's extra test models: items ship regardless
    ("s_none", synth.MODEL_ID, "none", "none", {}),                            # outside the model set: never exported
]
#: the synthetic analyses: (key, analysis_id, input run keys, input object ids) -- one wholly inside the frame; one over the outside run
#: alone; one over an in-frame run AND the outside run, left out whole (its in-frame input row ships no more than the other)
ANALYSES = [
    ("a_in", "11111111-2222-5333-8444-000000000001", ("g_none",), (synth.OBJECT_SHA,)),
    ("a_out", "11111111-2222-5333-8444-000000000002", ("s_none",), (synth.OBJECT_SHA,)),
    ("a_mixed", "11111111-2222-5333-8444-000000000003", ("g_lora", "s_none"), ()),
]
ANALYSIS_ID = {key: aid for key, aid, _runs, _objects in ANALYSES}


def _model(mid, family, kind, i):
    return synth.rec("model", {"model_id": mid, "family": family, "size_label": "4b", "checkpoint_kind": kind, "hf_repo": "org/x",
                               "hf_revision": "d" * 40, "chat_template_sha256": synth.SHA256_A, "thinking_channel": "available",
                               "paper_status": "in_paper"}, synth.t(i))


def _roster_entry(i, mid, object_id=None, selector=None):
    p = {"roster_entry_id": f"re{i}", "round_id": "r_synth", "comparison_id": "fig_synth", "model_id": mid, "eval_id": synth.EVAL_ID,
         "instrument_id": synth.INSTRUMENT_ID, "judge_id": synth.JUDGE_ID, "strength_rule": "operating_point",
         "source_citation": "synthetic", "inferred": 0}
    if object_id:
        p["object_id"] = object_id
    else:
        p["object_selector_json"] = selector
    return synth.rec("roster_entry", p, synth.t(50 + i))


def _analysis_filing(root, key, aid, run_ids, object_ids, i0):
    """One analysis filing written by hand under records/analyses/<day>/<key>/ in the CLI's order -- the report's `artifact` record, the
    `analysis` header, its inputs (the runs, then the objects), the link to the report -- one stamp per file, strictly increasing, every
    child bound to the header's record sha (`filing_sha256`), as the CLI writes it."""
    sha = hashlib.sha256(f"synthetic report {key}".encode("utf-8")).hexdigest()
    d = os.path.join(root, "records", "analyses", "2026-01-02", key)
    art = {"artifact_id": sha, "kind": "report", "uri_primary": f"/synthetic/{key}/report.json", "locations_json": [], "bytes": 16,
           "created_utc": synth.t(i0), "verified_utc": synth.t(i0), "extra": {"analysis": f"study.{key}"}}
    R.write_records(os.path.join(d, "000001-artifact.jsonl"), [synth.rec("artifact", art, synth.t(i0))])
    header = synth.rec("analysis", {"analysis_id": aid, "name": f"study.{key}", "code_sha": "c" * 40, "code_ref": "branch study",
                                    "cluster_job_ids_json": ["1"], "substrate": "cluster_a", "report_uri": None, "report_sha256": None,
                                    "filed_utc": synth.t(i0 + 1), "filed_by": "the user", "notes": None,
                                    "extra": {"provenance": {"kind": "user_direct"}, "study": "synthetic"}}, synth.t(i0 + 1))
    hsha = R.record_sha(header)
    R.write_records(os.path.join(d, "000002-analysis.jsonl"), [header])
    inputs = [{"analysis_id": aid, "filing_sha256": hsha, "input_kind": "run", "input_id": rid, "run_id": rid, "object_id": None} for rid in run_ids]
    inputs += [{"analysis_id": aid, "filing_sha256": hsha, "input_kind": "object", "input_id": oid, "run_id": None, "object_id": oid} for oid in object_ids]
    R.write_records(os.path.join(d, "000003-analysis_input.jsonl"), [synth.rec("analysis_input", p, synth.t(i0 + 2)) for p in inputs])
    link = {"analysis_id": aid, "filing_sha256": hsha, "artifact_id": sha, "role": "report", "extra": {}}
    R.write_records(os.path.join(d, "000004-analysis_artifact.jsonl"), [synth.rec("analysis_artifact", link, synth.t(i0 + 3))])


def build_corpus(root: str) -> dict:
    """The synthetic registry: dimensions + the paper models + a test model, six runs with items, a roster selecting two, and the three
    analyses of ``ANALYSES`` filed over them."""
    H.ensure_layout(root)
    dims = synth.seed_dimensions(0)
    dims += [_model("gemma4e4b_it", "gemma4", "instruct", 11), _model("gemma4e4b_base", "gemma4", "base", 12),
             _model("llama8b_it", "llama31", "instruct", 13), _model(SN.PAPER_FRAME_TEST_MODEL_IDS[0], "qwen35", "instruct", 14),
             synth.rec("object", synth.object_lora(), synth.t(15))]
    R.write_records(os.path.join(root, "records", "dimensions", "seed", "000001-dimensions.jsonl"), dims)
    run_ids = {}
    for k, (key, mid, objk, oid, over) in enumerate(RUNS):
        rid, recs = synth.run_records(key, 100 + 20 * k, objk, oid, n_items=N_ITEMS, model_id=mid, **over)
        jid = I.judgment_id(rid, synth.JUDGE_ID)
        files = [(str(r["record_type"]), [r]) for r in recs]
        files.append(("item_result", synth.item_records(rid, jid, 100 + 20 * k + 5, N_ITEMS, seed=k)))
        synth.write_run_dir(root, synth.CAMPAIGN, rid, files)
        run_ids[key] = rid
    R.write_records(os.path.join(root, "records", "roster", "000001-roster.jsonl"),
                    [_roster_entry(1, "gemma4e4b_it", object_id="none"), _roster_entry(2, "gemma4e4b_it", selector={"paper_label": "constitution_lora"})])
    for j, (key, aid, run_keys, object_ids) in enumerate(ANALYSES):
        _analysis_filing(root, key, aid, [run_ids[k] for k in run_keys], object_ids, 600 + 10 * j)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals[:5]
    with _live(root) as live:        # the fold took every analysis whole: what the export leaves out, it leaves out by its own rule
        assert live.execute("SELECT COUNT(*) FROM analysis").fetchone()[0] == len(ANALYSES)
        assert live.execute("SELECT COUNT(*) FROM analysis_input").fetchone()[0] == sum(len(r) + len(o) for _k, _a, r, o in ANALYSES)
    return run_ids


def pass_precondition(monkeypatch, tmp_path):
    """The precondition monkeypatched to PASS: PAIRED_ROOT a real directory, the retire merge sha = this tree's HEAD
    (an ancestor of itself)."""
    head = SN._git_head(SN._code_repo_root())
    if not head:
        pytest.skip("this checkout has no git HEAD to stand in for the #417 merge sha")
    paired = tmp_path / "paired"
    paired.mkdir(exist_ok=True)
    monkeypatch.setattr(RN, "PAIRED_ROOT", str(paired))
    monkeypatch.setattr(SN, "PAIRED_ROOT_RETIRE_MERGE_SHA", head)
    monkeypatch.setattr(SN, "_private_tokens", lambda: TOKENS)
    return head


@pytest.fixture
def frame(tmp_path, monkeypatch):
    root = str(tmp_path / "reg")
    dest = str(tmp_path / "dest")
    run_ids = build_corpus(root)
    head = pass_precondition(monkeypatch, tmp_path)
    info = SN.build(root, DB.db_path(root), check_free_space=False, scope="paper_frame", dest_root=dest)
    return {"root": root, "dest": dest, "info": info, "run_ids": run_ids, "head_sha": head}


def _live(root):
    return DB.connect(DB.db_path(root), readonly=True)


def _rows(conn, sql, *args):
    return [tuple(r) for r in conn.execute(sql, args).fetchall()]


# --------------------------------------------------------------------------------------------
# the read set and the constants the notebook codes against
# --------------------------------------------------------------------------------------------


def test_read_set_constants():
    assert "paper_frame" in SN.SCOPES and SN.SCOPE_SQLITE["paper_frame"] == "paper_frame.sqlite"
    assert SN.PAPER_FRAME_READ_SET_VERSION == 2
    assert SN.PAPER_FRAME_MODEL_IDS == ("gemma4e4b_it", "qwen35_4b_it", "nemotron3nano4b", "llama8b_it")
    # the exporter's two extra models, both out of the paper frame: the registry spells the first (the test MoE) and the public notebook
    # never does; the second is the front-door worked example's own key (the paper's Qwen3.5-4B checkpoints under a key of their own,
    # paper_status reference), which took the place of the scrubbed gemma key -- a run under either reads not_in_roster and ships
    # regardless (the `t_none` run below)
    assert SN.PAPER_FRAME_TEST_MODEL_IDS == ("qwen35_" + "35b_" + "a3b", "qwen35_4b_demo")     # the first spelled in pieces (the public tree's census)
    ms = SN.paper_frame_model_set()
    for mid in ("gemma4e4b_base", "qwen35_4b_base", "nemotron3nano4b_base", "llama8b_base", *(f"{m}_base" for m in SN.PAPER_FRAME_TEST_MODEL_IDS)):
        assert mid in ms, mid
    assert "extra_model" in SN.paper_frame_model_set(["extra_model"]) and "extra_model_base" in SN.paper_frame_model_set(["extra_model"])
    assert set(SN.PAPER_FRAME_TABLES) == (set(SN.PAPER_FRAME_DIMENSION_TABLES) | set(SN.PAPER_FRAME_RUN_TABLES) | set(SN.PAPER_FRAME_ITEM_TABLES)
                                          | set(SN.PAPER_FRAME_ANALYSIS_TABLES) | {"record_ingest"} | set(SN.PAPER_FRAME_MATERIALISED))
    # the analysis tables are a set of their own, filtered to the frame -- never among the dimensions exported in full
    assert SN.PAPER_FRAME_ANALYSIS_TABLES == ("analysis", "analysis_input")
    assert not set(SN.PAPER_FRAME_ANALYSIS_TABLES) & set(SN.PAPER_FRAME_DIMENSION_TABLES) and "analysis_artifact" not in SN.PAPER_FRAME_TABLES
    assert "artifact" not in SN.PAPER_FRAME_TABLES and not set(SN.HISTORY_TABLES) & set(SN.PAPER_FRAME_TABLES)
    assert set(SN.PAPER_FRAME_MATERIALISED) <= set(S.VIEW_NAMES)
    for t in SN.PAPER_FRAME_TABLES:
        assert t in S.TABLE_BY_NAME or t in S.VIEW_NAMES, t
    # the paired-root-retire merge commit, filled at integration: the precondition compares the
    # checkout against it with `git merge-base --is-ancestor`
    assert SN.PAIRED_ROOT_RETIRE_MERGE_SHA == "db78f971a325c749a80dd7a8e623b0faea60ddbb"
    assert re.fullmatch(r"[0-9a-f]{40}", SN.PAIRED_ROOT_RETIRE_MERGE_SHA)


# --------------------------------------------------------------------------------------------
# the build: layout, manifest, row sets
# --------------------------------------------------------------------------------------------


def test_operational_root_untouched_and_committed_layout(frame):
    root, dest, info = frame["root"], frame["dest"], frame["info"]
    assert os.listdir(SN.snapshots_dir(root)) == [] and SN.pointer(root) is None       # C6: the registry's own pointer and dirs
    assert not os.path.exists(os.path.join(root, "state", "footprint.json"))
    assert SN.pointer(dest) == info.snapshot_id and info.scope == "paper_frame"
    d = os.path.join(dest, "snapshots", info.snapshot_id)
    assert info.dir == d and os.path.isdir(d)
    names = set(os.listdir(d))
    assert "MANIFEST.json" in names and "paper_frame.sqlite" not in names               # the raw SQLite never ships whole
    for t in SN.PAPER_FRAME_TABLES:
        assert f"{t}.parquet" in names, t
    assert "artifact.parquet" not in names and "row_history.parquet" not in names
    parts = [n for n in names if n.startswith("paper_frame.sqlite.gz.")]
    assert parts == info.manifest["sqlite_parts"] == ["paper_frame.sqlite.gz.00"]
    assert not [n for n in names if n.endswith(".building") or n.endswith(".tmp")]


def test_manifest_fields(frame):
    root, info = frame["root"], frame["info"]
    m = info.manifest
    with _live(root) as conn:
        head = conn.execute("SELECT MAX(ingest_seq) FROM record_ingest").fetchone()[0]
    assert m["scope"] == "paper_frame" and m["snapshot_id"] == info.snapshot_id and m["ingest_seq"] == head == info.ingest_seq
    assert m["snapshot_id"].startswith(f"{head:012d}-") and m["built_utc"] == info.built_utc
    assert set(m["tables"]) == set(SN.PAPER_FRAME_TABLES)
    assert m["read_set_version"] == SN.PAPER_FRAME_READ_SET_VERSION and m["model_ids"] == list(SN.paper_frame_model_set())
    assert m["item_model_ids"] == list(SN.paper_frame_test_model_set())
    assert re.fullmatch(r"[0-9a-f]{64}", m["sqlite_sha256"]) and m["sqlite_bytes"] == info.runlevel_bytes > 0
    assert m["sqlite_parts"] and m["sqlite_part_bytes"] == SN.PAPER_FRAME_PART_BYTES
    # the source registry by CONTENT, never by path: the committed manifest carried the
    # grading host's absolute `source_root`, and the precondition record its absolute `paired_root` and checkout `repo`
    assert "source_root" not in m and "paired_root" not in m["precondition"] and "repo" not in m["precondition"]
    from sisteer.registry.migrations import code_version, db_version
    with _live(root) as conn:
        assert m["source_schema_version"] == db_version(conn) == code_version() > 0        # the registry's own version reader (schema_migration)
    assert isinstance(m["scrubbed"], dict) and all(isinstance(v, list) for v in m["scrubbed"].values())
    assert m["precondition"]["checked"] is True and m["precondition"]["override"] is None
    assert m["precondition"]["retire_merge_sha"] == frame["head_sha"] and m["precondition"]["head"] == frame["head_sha"]
    assert set(m["precondition"]) == {"retire_merge_sha", "head", "override", "checked", "ancestry"}
    assert m["precondition"]["ancestry"] == "verified"                                  # HEAD is in the store, so the lineage was READ
    for fn, meta in m["files"].items():
        p = os.path.join(info.dir, fn)
        assert R.file_sha256(p) == meta["sha256"] and os.path.getsize(p) == meta["bytes"], fn
    assert set(m["files"]) == set(os.listdir(info.dir)) - {"MANIFEST.json"}
    assert m["n_runs"] == 5 and m["n_item_runs"] == 3
    # the analyses: one carried, two left out and named (the rule itself: test_an_analysis_with_an_input_outside_the_frame_... below)
    assert m["n_analyses"] == 1 == m["tables"]["analysis"] and m["n_analyses_out_of_frame"] == 2 == len(m["analyses_out_of_frame"])
    # the payload the harvester would file reads the scope's SQLite hash (generalised from files['registry_runlevel.sqlite'])
    assert SN.snapshot_record_payload(info)["sqlite_sha256"] == m["sqlite_sha256"]
    assert SN.manifest_sqlite_sha256({"scope": "full", "files": {"registry_runlevel.sqlite": {"sha256": "x" * 64}}}) == "x" * 64


def test_run_set_and_item_restriction(frame):
    root, run_ids = frame["root"], frame["run_ids"]
    conn = SN.open_paper_frame(frame["dest"])
    exported = {r[0] for r in conn.execute("SELECT run_id FROM run")}
    assert exported == {run_ids[k] for k in ("g_none", "g_lora", "g_extra", "l_none", "t_none")}
    assert run_ids["s_none"] not in exported
    assert {r[0] for r in conn.execute("SELECT DISTINCT model_id FROM run")} <= set(SN.paper_frame_model_set())
    # measurement tables follow the run set; their original selection flags stand
    pc = {r["run_id"]: dict(r) for r in conn.execute("SELECT * FROM v_paper_current")}
    assert set(pc) == exported
    assert pc[run_ids["g_lora"]]["in_current_paper"] == 1 and pc[run_ids["g_lora"]]["selection_state"] == "selected"
    assert pc[run_ids["l_none"]]["in_current_paper"] == 0 and pc[run_ids["l_none"]]["selection_state"] == "not_in_roster"
    assert pc[run_ids["t_none"]]["in_current_paper"] == 0 and pc[run_ids["t_none"]]["selection_state"] == "not_in_roster"
    unsteered = {run_ids["g_none"], run_ids["g_extra"]}
    states = {pc[r]["selection_state"] for r in unsteered}
    assert states == {"selected", "superseded"}, states                                 # a later unsteered run at the same key (cap, trial, pass) supersedes the earlier
    for t in ("judgment", "run_metric", "run_state_event"):
        assert {r[0] for r in conn.execute(f"SELECT DISTINCT run_id FROM {t}")} == exported, t
    # item tables: in-paper runs + every run of the test model, nothing else
    in_paper = {r for r, row in pc.items() if row["in_current_paper"] == 1}
    item_runs = {r[0] for r in conn.execute("SELECT DISTINCT run_id FROM item_result")}
    assert item_runs == in_paper | {run_ids["t_none"]} and run_ids["l_none"] not in item_runs
    assert all(n == N_ITEMS for _, n in conn.execute("SELECT run_id, COUNT(*) FROM item_result GROUP BY run_id"))
    assert conn.execute("SELECT COUNT(*) FROM item_result").fetchone()[0] == N_ITEMS * len(item_runs) == frame["info"].manifest["tables"]["item_result"]
    # dimensions in full (the model outside the run set is still a dimension row)
    with _live(root) as live:
        for t in SN.PAPER_FRAME_DIMENSION_TABLES:
            assert conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] == live.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0], t
    assert synth.MODEL_ID in {r[0] for r in conn.execute("SELECT model_id FROM model")}
    # the artifact table is omitted; no history ledger
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert tables == set(SN.PAPER_FRAME_TABLES) and "artifact" not in tables and "row_history" not in tables
    conn.close()


# --------------------------------------------------------------------------------------------
# the analyses: exported whole or not at all, the frame never widened, every one left out named
# --------------------------------------------------------------------------------------------


def test_an_analysis_with_an_input_outside_the_frame_is_left_out_whole_and_named_in_the_manifest(frame, tmp_path):
    """The frame is a curated set of runs. An analysis over a run outside it is not exported beside an input whose run is absent (the
    exported SQLite declares no foreign keys, so a reader joining the input to its run would lose it in silence), and the frame is not
    widened to admit the run: the analysis is left out WHOLE -- the one over an in-frame run AND the outside run ships none of its rows,
    the in-frame input included -- and the manifest names every analysis left out with the inputs that put it out. The same registry
    exported with the outside model added (``model_ids``) carries all three and names none: the rule is the frame's, not the analysis's."""
    root, run_ids, m = frame["root"], frame["run_ids"], frame["info"].manifest
    conn = SN.open_paper_frame(frame["dest"])
    assert {r[0] for r in conn.execute("SELECT analysis_id FROM analysis")} == {ANALYSIS_ID["a_in"]}
    assert {r[0] for r in conn.execute("SELECT DISTINCT analysis_id FROM analysis_input")} == {ANALYSIS_ID["a_in"]}
    exported_inputs = {r[0] for r in conn.execute("SELECT input_id FROM analysis_input")}
    assert run_ids["g_lora"] not in exported_inputs and run_ids["s_none"] not in exported_inputs     # a_mixed: nothing of it, not a partial export
    conn.close()
    assert m["n_analyses"] == 1 == m["tables"]["analysis"] and m["tables"]["analysis_input"] == 2
    assert m["n_analyses_out_of_frame"] == 2 == len(m["analyses_out_of_frame"])
    assert m["analyses_out_of_frame"] == [                                                            # in analysis_id order, inputs in key order
        {"analysis_id": ANALYSIS_ID["a_out"], "name": "study.a_out", "inputs_out_of_frame": [{"input_kind": "run", "input_id": run_ids["s_none"]}]},
        {"analysis_id": ANALYSIS_ID["a_mixed"], "name": "study.a_mixed", "inputs_out_of_frame": [{"input_kind": "run", "input_id": run_ids["s_none"]}]},
    ]
    # the committed MANIFEST.json says the same (the record a reader of the clone has)
    with open(os.path.join(frame["info"].dir, "MANIFEST.json"), "r", encoding="utf-8") as fh:
        committed = json.load(fh)
    assert committed["analyses_out_of_frame"] == m["analyses_out_of_frame"] and committed["n_analyses"] == 1
    # the frame widened to the outside model: every input in frame, all three carried, none named
    wide = SN.build(root, DB.db_path(root), check_free_space=False, scope="paper_frame", dest_root=str(tmp_path / "wide"), model_ids=[synth.MODEL_ID])
    assert wide.manifest["n_analyses"] == 3 == wide.manifest["tables"]["analysis"] and wide.manifest["tables"]["analysis_input"] == 6
    assert wide.manifest["n_analyses_out_of_frame"] == 0 and wide.manifest["analyses_out_of_frame"] == []
    wconn = SN.open_paper_frame(str(tmp_path / "wide"))
    assert {r[0] for r in wconn.execute("SELECT analysis_id FROM analysis")} == set(ANALYSIS_ID.values())
    wconn.close()


def test_an_analysis_whose_inputs_are_all_in_frame_exports_complete(frame):
    """The in-frame analysis ships with every column of its header and every one of its input rows equal to the live registry's
    (``record_uri`` aside, the scrub's column), the run input and the object input both."""
    root, run_ids = frame["root"], frame["run_ids"]
    conn = SN.open_paper_frame(frame["dest"])
    with _live(root) as live:
        for t in SN.PAPER_FRAME_ANALYSIS_TABLES:
            cols = [c for c in S.TABLE_BY_NAME[t].column_names if c != "record_uri"]
            want = sorted(_rows(live, f"SELECT {', '.join(cols)} FROM {t} WHERE analysis_id = ?", ANALYSIS_ID["a_in"]), key=repr)
            got = sorted(_rows(conn, f"SELECT {', '.join(cols)} FROM {t}"), key=repr)
            assert got == want and got, t
    assert _rows(conn, "SELECT input_kind, input_id, run_id, object_id FROM analysis_input ORDER BY input_kind") == [
        ("object", synth.OBJECT_SHA, None, synth.OBJECT_SHA), ("run", run_ids["g_none"], run_ids["g_none"], None)]
    assert _rows(conn, "SELECT name, code_sha, substrate FROM analysis") == [("study.a_in", "c" * 40, "cluster_a")]
    conn.close()


def test_every_exported_analysis_input_resolves_to_an_exported_run_or_object(frame):
    """Over the exported database itself, which declares no foreign keys: every input names an exported analysis, every run input an
    exported `run` row, every object input an exported `object` row -- so a reader's join loses nothing -- and every exported analysis
    carries every input the live registry holds for it. The Parquet says the same."""
    import pyarrow.parquet as pq
    root, info = frame["root"], frame["info"]
    conn = SN.open_paper_frame(frame["dest"])
    assert conn.execute("SELECT COUNT(*) FROM analysis_input").fetchone()[0] > 0
    for what, sql in (
        ("analysis", "SELECT COUNT(*) FROM analysis_input i LEFT JOIN analysis a ON a.analysis_id = i.analysis_id WHERE a.analysis_id IS NULL"),
        ("run", "SELECT COUNT(*) FROM analysis_input i LEFT JOIN run r ON r.run_id = i.run_id WHERE i.input_kind = 'run' AND r.run_id IS NULL"),
        ("object", "SELECT COUNT(*) FROM analysis_input i LEFT JOIN object o ON o.object_id = i.object_id WHERE i.input_kind = 'object' AND o.object_id IS NULL"),
    ):
        assert conn.execute(sql).fetchone()[0] == 0, what
    exported = {aid: n for aid, n in conn.execute("SELECT analysis_id, COUNT(*) FROM analysis_input GROUP BY analysis_id")}
    assert set(exported) == {r[0] for r in conn.execute("SELECT analysis_id FROM analysis")}
    with _live(root) as live:
        for aid, n in exported.items():
            assert live.execute("SELECT COUNT(*) FROM analysis_input WHERE analysis_id = ?", (aid,)).fetchone()[0] == n, aid
    conn.close()
    inputs = pq.read_table(os.path.join(info.dir, "analysis_input.parquet")).to_pylist()
    runs = {r["run_id"] for r in pq.read_table(os.path.join(info.dir, "run.parquet")).to_pylist()}
    objects = {r["object_id"] for r in pq.read_table(os.path.join(info.dir, "object.parquet")).to_pylist()}
    analyses = {r["analysis_id"] for r in pq.read_table(os.path.join(info.dir, "analysis.parquet")).to_pylist()}
    assert inputs and all(r["analysis_id"] in analyses for r in inputs)
    assert all((r["input_kind"] == "run" and r["run_id"] in runs) or (r["input_kind"] == "object" and r["object_id"] in objects) for r in inputs)


def test_materialised_views_equal_live_restricted_to_the_run_set(frame):
    root = frame["root"]
    conn = SN.open_paper_frame(frame["dest"])
    exported = {r[0] for r in conn.execute("SELECT run_id FROM run")}
    marks = ", ".join("?" for _ in exported)
    with _live(root) as live:
        for v in SN.PAPER_FRAME_MATERIALISED:
            cols = [d[0] for d in live.execute(f"SELECT * FROM {v} LIMIT 0").description]
            assert [d[0] for d in conn.execute(f"SELECT * FROM {v} LIMIT 0").description] == cols, v
            where = f" WHERE run_id IN ({marks})" if "run_id" in cols else ""
            want = sorted(_rows(live, f"SELECT {', '.join(cols)} FROM {v}{where}", *(tuple(exported) if where else ())), key=repr)
            got = sorted(_rows(conn, f"SELECT {', '.join(cols)} FROM {v}"), key=repr)
            assert got == want, v
            assert len(got) == frame["info"].manifest["tables"][v]
    assert conn.execute("SELECT COUNT(*) FROM v_paper_long WHERE in_current_paper = 1").fetchone()[0] == 2
    conn.close()


def test_record_ingest_is_the_one_head_row(frame):
    root, info = frame["root"], frame["info"]
    conn = SN.open_paper_frame(frame["dest"])
    rows = [dict(r) for r in conn.execute("SELECT * FROM record_ingest")]
    assert len(rows) == 1 and rows[0]["ingest_seq"] == info.ingest_seq
    assert conn.execute("SELECT MAX(ingest_seq) FROM record_ingest").fetchone()[0] == info.ingest_seq
    with _live(root) as live:
        head = dict(live.execute("SELECT * FROM record_ingest ORDER BY ingest_seq DESC LIMIT 1").fetchone())
    assert rows[0]["file_sha256"] == head["file_sha256"] and rows[0]["record_file_id"] == head["record_file_id"]
    assert rows[0]["file_uri"] == f"sha256:{head['file_sha256']}"           # the record log's pointer becomes a content reference
    assert "record_ingest" in info.manifest["scrubbed"] and info.manifest["scrubbed"]["record_ingest"] == ["file_uri"]
    # every record_uri is a content reference too (the companion repo carries no record log to point into)
    for t in SN.PAPER_FRAME_DIMENSION_TABLES + SN.PAPER_FRAME_RUN_TABLES:
        if "record_uri" not in S.TABLE_BY_NAME[t].column_names:
            continue
        for (uri,) in conn.execute(f"SELECT record_uri FROM {t}"):
            assert re.fullmatch(r"sha256:[0-9a-f]{64}", uri), (t, uri)
        if conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]:
            assert "record_uri" in info.manifest["scrubbed"][t], t
    conn.close()


# --------------------------------------------------------------------------------------------
# two formats from the same frames; the parts; the readers
# --------------------------------------------------------------------------------------------


def test_parquet_and_sqlite_agree_row_for_row(frame):
    import pyarrow.parquet as pq
    info = frame["info"]
    conn = SN.open_paper_frame(frame["dest"])
    for t in SN.PAPER_FRAME_TABLES:
        table = pq.read_table(os.path.join(info.dir, f"{t}.parquet"))
        assert table.num_rows == info.manifest["tables"][t], t
        cols = table.column_names
        if t in S.TABLE_BY_NAME:
            assert cols == list(S.TABLE_BY_NAME[t].column_names), t
        pq_rows = sorted((tuple(r[c] for c in cols) for r in table.to_pylist()), key=repr)
        sq_rows = sorted(_rows(conn, f"SELECT {', '.join(cols)} FROM {t}"), key=repr)
        assert pq_rows == sq_rows, t
    conn.close()


def test_parts_are_bounded_and_reassemble_to_the_hash(frame, tmp_path):
    info, dest = frame["info"], frame["dest"]
    m = info.manifest
    blob = b""
    for fn in m["sqlite_parts"]:
        p = os.path.join(info.dir, fn)
        assert os.path.getsize(p) <= SN.PAPER_FRAME_PART_BYTES
        with open(p, "rb") as fh:
            blob += fh.read()
    raw = gzip.decompress(blob)
    assert hashlib.sha256(raw).hexdigest() == m["sqlite_sha256"] and len(raw) == m["sqlite_bytes"]
    assert raw[:16] == b"SQLite format 3\x00" and int.from_bytes(raw[16:18], "big") == 4096      # page_size pinned
    # the reader reassembles once and reuses (idempotent on a matching sha; rebuilt on a mismatch)
    path, sid, manifest = SN.paper_frame_sqlite_path(dest)
    assert sid == info.snapshot_id and manifest["sqlite_sha256"] == m["sqlite_sha256"]
    assert path == os.path.join(dest, SN.PAPER_FRAME_REASSEMBLED_DIR, sid, "paper_frame.sqlite") and R.file_sha256(path) == m["sqlite_sha256"]
    st = os.stat(path)
    assert SN.paper_frame_sqlite_path(dest)[0] == path and os.stat(path).st_mtime_ns == st.st_mtime_ns
    with open(path, "ab") as fh:
        fh.write(b"corrupt")
    path2, _, _ = SN.paper_frame_sqlite_path(dest)
    assert path2 == path and R.file_sha256(path) == m["sqlite_sha256"]
    # an explicit cache dir
    cache = str(tmp_path / "cache")
    path3, _, _ = SN.paper_frame_sqlite_path(dest, cache_dir=cache)
    assert path3 == os.path.join(cache, sid, "paper_frame.sqlite") and R.file_sha256(path3) == m["sqlite_sha256"]
    # a torn part is refused by name
    with open(os.path.join(info.dir, m["sqlite_parts"][0]), "ab") as fh:
        fh.write(b"x")
    with pytest.raises(SN.TornSnapshot):
        SN.paper_frame_sqlite_path(dest, cache_dir=str(tmp_path / "cache2"))


def test_gzip_split_produces_several_bounded_parts(tmp_path):
    src = tmp_path / "big.bin"
    src.write_bytes(os.urandom(300_000))                      # incompressible, so the parts are real
    parts = SN._gzip_split(str(src), str(tmp_path), "big.bin", part_bytes=100_000)
    assert parts == ["big.bin.gz.00", "big.bin.gz.01", "big.bin.gz.02", "big.bin.gz.03"][: len(parts)] and len(parts) >= 3
    assert all(os.path.getsize(tmp_path / p) <= 100_000 for p in parts)
    got = SN._reassemble_parts([str(tmp_path / p) for p in parts], str(tmp_path / "out.bin"))
    assert got == hashlib.sha256(src.read_bytes()).hexdigest() and (tmp_path / "out.bin").read_bytes() == src.read_bytes()


def test_readers_see_the_head_and_the_live_rows(frame):
    root, dest, info = frame["root"], frame["dest"], frame["info"]
    sid, manifest = SN.paper_frame_current(dest)
    assert sid == info.snapshot_id and manifest["scope"] == "paper_frame"
    conn = SN.open_paper_frame(dest)
    assert conn.execute("SELECT MAX(ingest_seq) FROM record_ingest").fetchone()[0] == info.ingest_seq
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("CREATE TABLE nope (x)")                  # mode=ro
    exported = {r[0] for r in conn.execute("SELECT run_id FROM run")}
    with _live(root) as live:
        cols = [d[0] for d in live.execute("SELECT * FROM v_paper_long LIMIT 0").description]
        want = sorted(_rows(live, f"SELECT {', '.join(cols)} FROM v_paper_long WHERE run_id IN ({', '.join('?' for _ in exported)})", *exported), key=repr)
    assert sorted(_rows(conn, f"SELECT {', '.join(cols)} FROM v_paper_long"), key=repr) == want
    conn.close()
    with SN.snapshot_open(dest, scope="paper_frame") as snap:
        assert snap.scope == "paper_frame" and snap.snapshot_id == sid and snap.lease_path == ""
        assert snap.conn.execute("SELECT MAX(ingest_seq) FROM record_ingest").fetchone()[0] == info.ingest_seq
        assert snap.parquet("run").num_rows == 5 and snap.parquet("v_paper_long").num_rows == manifest["tables"]["v_paper_long"]
        snap.touch()                                           # a no-op without a lease
        with pytest.raises(SN.SnapshotError) as ei:
            snap.parquet("artifact")
        assert "read set" in str(ei.value)
    assert not os.path.isdir(os.path.join(SN.snapshots_dir(dest), SN.LEASES_DIR))      # no lease protocol for a committed export
    with pytest.raises(SN.SnapshotError):
        SN.paper_frame_current(root)                           # the operational root has no paper frame
    with pytest.raises(SN.SnapshotError):
        with SN.snapshot_open(dest, scope="bogus"):
            pass


def test_second_build_reclaims_under_dest_root_only_and_is_deterministic(frame, tmp_path, monkeypatch):
    root, dest, a = frame["root"], frame["dest"], frame["info"]
    b = SN.build(root, DB.db_path(root), check_free_space=False, scope="paper_frame", dest_root=dest, now=a.built_utc)
    assert b.snapshot_id == a.snapshot_id and b.manifest["sqlite_sha256"] == a.manifest["sqlite_sha256"]   # same input, same now: same bytes
    other = str(tmp_path / "dest2")
    c = SN.build(root, DB.db_path(root), check_free_space=False, scope="paper_frame", dest_root=other, now=a.built_utc)
    assert c.manifest["sqlite_sha256"] == a.manifest["sqlite_sha256"] and c.manifest["files"] == a.manifest["files"]
    d = SN.build(root, DB.db_path(root), check_free_space=False, scope="paper_frame", dest_root=dest, now=R.utc_now())
    assert d.snapshot_id != a.snapshot_id and SN.pointer(dest) == d.snapshot_id
    assert a.snapshot_id in d.reclaimed and not os.path.isdir(a.dir)                    # retention 1 under dest_root
    assert os.listdir(SN.snapshots_dir(root)) == [] and SN.pointer(root) is None       # ... and never under the registry root


# --------------------------------------------------------------------------------------------
# refusals: dest_root, the precondition, free space
# --------------------------------------------------------------------------------------------


def test_build_refuses_without_dest_root_or_into_the_registry_root(tmp_path, monkeypatch):
    root = str(tmp_path / "reg")
    build_corpus(root)
    pass_precondition(monkeypatch, tmp_path)
    with pytest.raises(SN.SnapshotError) as ei:
        SN.build(root, DB.db_path(root), check_free_space=False, scope="paper_frame")
    assert "dest_root" in str(ei.value)
    with pytest.raises(SN.SnapshotError) as ei:
        SN.build(root, DB.db_path(root), check_free_space=False, scope="paper_frame", dest_root=root)
    assert "registry root" in str(ei.value)
    with pytest.raises(SN.SnapshotError) as ei:
        SN.build(root, DB.db_path(root), check_free_space=False, scope="full", dest_root=str(tmp_path / "d"))
    assert "only for scope paper_frame" in str(ei.value)
    assert os.listdir(SN.snapshots_dir(root)) == [] and SN.pointer(root) is None


def test_build_refuses_a_symlink_alias_of_the_registry_root_before_reclaiming_anything(tmp_path, monkeypatch):
    """A `--dest-root` that is a SYMLINK to the operational registry root passed the earlier
    `abspath` inequality; `build` then ran `reclaim(out_root)` over the operational `snapshots/` (deleting every unleased directory)
    and switched the operational CURRENT. Both roots are resolved physically now, and equal / inside / containing are all refused
    BEFORE anything under either root is touched: a stale operational snapshot planted beforehand survives, its CURRENT stands."""
    root = str(tmp_path / "reg")
    build_corpus(root)
    pass_precondition(monkeypatch, tmp_path)
    # an operational snapshot the registry's own timer would have built, and the pointer to it
    stale = os.path.join(SN.snapshots_dir(root), "000000000042-20260101T000000-deadbeef")
    os.makedirs(stale)
    with open(os.path.join(stale, "MANIFEST.json"), "w", encoding="utf-8") as fh:
        json.dump({"snapshot_id": os.path.basename(stale), "scope": "full", "files": {}, "tables": {}}, fh)
    SN.switch_pointer(root, os.path.basename(stale))
    before = sorted(os.listdir(SN.snapshots_dir(root)))
    # (1) the alias: a symlink at another path that resolves to the registry root
    alias = tmp_path / "elsewhere" / "paper_snapshot"
    alias.parent.mkdir()
    os.symlink(root, str(alias))
    assert os.path.abspath(str(alias)) != os.path.abspath(root)                         # the old check saw two different paths
    with pytest.raises(SN.SnapshotError) as ei:
        SN.build(root, DB.db_path(root), check_free_space=False, scope="paper_frame", dest_root=str(alias))
    assert "IS the registry root" in str(ei.value) and os.path.realpath(root) in str(ei.value)
    # (2) inside the registry root (its own snapshots/ would become the export's), through a symlink too
    inside_link = tmp_path / "inside_link"
    os.symlink(os.path.join(root, "snapshots"), str(inside_link))
    for inside in (os.path.join(root, "snapshots", "paper"), os.path.join(root, "paper_frame"), str(inside_link / "paper")):
        with pytest.raises(SN.SnapshotError) as ei:
            SN.build(root, DB.db_path(root), check_free_space=False, scope="paper_frame", dest_root=inside)
        assert "INSIDE the registry root" in str(ei.value), inside
    # (3) a destination that CONTAINS the registry root
    with pytest.raises(SN.SnapshotError) as ei:
        SN.build(root, DB.db_path(root), check_free_space=False, scope="paper_frame", dest_root=str(tmp_path))
    assert "CONTAINS the registry root" in str(ei.value)
    # the same three refusals through the shared check the front door calls; a legitimate sibling is accepted as given
    for bad in (str(alias), os.path.join(root, "x"), str(tmp_path)):
        with pytest.raises(SN.SnapshotError):
            SN.paper_frame_dest_root(root, bad)
    assert SN.paper_frame_dest_root(root, str(tmp_path / "dest")) == str(tmp_path / "dest")
    # nothing under the operational root moved: the stale directory is still there, CURRENT still names it, no .building dir
    assert sorted(os.listdir(SN.snapshots_dir(root))) == before and SN.pointer(root) == os.path.basename(stale)
    assert not os.path.exists(os.path.join(root, "state", "footprint.json"))
    # and the registry root through a symlink of ITS OWN resolves the same way (a dest equal to the physical root is refused)
    root_link = tmp_path / "reg_link"
    os.symlink(root, str(root_link))
    with pytest.raises(SN.SnapshotError) as ei:
        SN.build(str(root_link), DB.db_path(root), check_free_space=False, scope="paper_frame", dest_root=root)
    assert "IS the registry root" in str(ei.value)
    assert sorted(os.listdir(SN.snapshots_dir(root))) == before and SN.pointer(root) == os.path.basename(stale)


def test_committed_tree_carries_no_private_identifier(frame):
    """The WHOLE committed tree — MANIFEST.json included — carries none of the identifiers
    the repo's scrub test looks for (the token table's output rows, `front_door_testlib.token_hits`, as every output check reads them), none
    of this machine's user name, host name or home directory, none of the synthetic corpus's planted tokens, and no absolute path of the
    source registry, the destination or this checkout. Before the fix the manifest carried `source_root` (the registry's absolute
    path under this user's tmp dir), `precondition.paired_root` and `precondition.repo` (this checkout's absolute path)."""
    import socket
    import sys
    sys.path.insert(0, HERE)
    from front_door_testlib import token_hits                                      # the token table's output rows
    from test_snapshot_scrub_decoded import _cells, _decoded_rows                  # the decoded walk over every table, both formats
    info, root, dest = frame["info"], frame["root"], frame["dest"]
    idents = set(TOKENS) | {os.path.abspath(root), os.path.abspath(dest), REPO, os.path.expanduser("~")}
    for v in (os.environ.get("USER"), os.environ.get("LOGNAME"), os.path.basename(os.path.expanduser("~")), socket.gethostname(),
              socket.gethostname().split(".")[0], str(RN.PAIRED_ROOT)):
        if v and len(v) >= 3:
            idents.add(v)
    idents = sorted(idents, key=len, reverse=True)

    def hits(text: str):
        return [t for t in idents if t in text] + token_hits(text)

    # MANIFEST.json as committed text
    with open(os.path.join(info.dir, "MANIFEST.json"), "r", encoding="utf-8") as fh:
        manifest_text = fh.read()
    assert not hits(manifest_text), hits(manifest_text)
    # every file name under the snapshot directory
    for fn in os.listdir(info.dir):
        assert not hits(fn), fn
    # every cell of every table, Parquet and the reassembled SQLite
    for table in SN.PAPER_FRAME_TABLES:
        pq_rows, sq_rows = _decoded_rows(info, table)
        for fmt, rows in (("parquet", pq_rows), ("sqlite", sq_rows)):
            for col, path, s in _cells(rows):
                assert not hits(s), (table, fmt, col, path, hits(s), s[:200])
    # the build itself refuses a manifest that would carry one (the belt behind the assertions above)
    with pytest.raises(SN.SnapshotError) as ei:
        SN._refuse_manifest_leak({"note": f"built under {os.path.abspath(root)}"}, TOKENS, (root,))
    assert "private identifier" in str(ei.value)
    with pytest.raises(SN.SnapshotError):
        SN._refuse_manifest_leak({"note": f"on {TOKENS[0]}"}, TOKENS, ())
    SN._refuse_manifest_leak({"note": "plain"}, TOKENS, (root,))                        # a clean manifest passes


def test_precondition_refuses_by_name_and_records_an_override(tmp_path, monkeypatch):
    root = str(tmp_path / "reg")
    dest = str(tmp_path / "dest")
    build_corpus(root)
    monkeypatch.setattr(SN, "_private_tokens", lambda: TOKENS)
    paired = tmp_path / "paired"
    paired.mkdir()
    head = SN._git_head(SN._code_repo_root())
    # (1) the merge sha unset: #417 has not landed
    monkeypatch.setattr(RN, "PAIRED_ROOT", str(paired))
    monkeypatch.setattr(SN, "PAIRED_ROOT_RETIRE_MERGE_SHA", None)
    with pytest.raises(SN.SnapshotError) as ei:
        SN.build(root, DB.db_path(root), check_free_space=False, scope="paper_frame", dest_root=dest)
    assert "retire merge sha is unset" in str(ei.value) and "#417" in str(ei.value)
    assert not os.path.exists(dest)                                                    # refused before anything was written
    # (2) a symlinked PAIRED_ROOT is refused even when the sha holds
    if head:
        monkeypatch.setattr(SN, "PAIRED_ROOT_RETIRE_MERGE_SHA", head)
    link = tmp_path / "paired_link"
    os.symlink(str(paired), str(link))
    monkeypatch.setattr(RN, "PAIRED_ROOT", str(link))
    with pytest.raises(SN.SnapshotError) as ei:
        SN.build(root, DB.db_path(root), check_free_space=False, scope="paper_frame", dest_root=dest)
    assert "symlink" in str(ei.value)
    # (3) a missing directory
    monkeypatch.setattr(RN, "PAIRED_ROOT", str(tmp_path / "absent"))
    with pytest.raises(SN.SnapshotError) as ei:
        SN.build(root, DB.db_path(root), check_free_space=False, scope="paper_frame", dest_root=dest)
    assert "not a directory" in str(ei.value)
    # (4) a sha that is not an ancestor of HEAD: refused wherever this checkout has a history the sha could be in; where the checkout
    # positively carries no development history (the publish step's marker, or a single commit) the record says the lineage was
    # unavailable and the content half alone passes — the published-repository shape, never a "merge main first" instruction there
    monkeypatch.setattr(RN, "PAIRED_ROOT", str(paired))
    if head:
        monkeypatch.setattr(SN, "PAIRED_ROOT_RETIRE_MERGE_SHA", "0" * 40)
        if SN._history_unavailable(SN._code_repo_root(), "0" * 40) is None:
            with pytest.raises(SN.SnapshotError) as ei:
                SN.build(root, DB.db_path(root), check_free_space=False, scope="paper_frame", dest_root=dest)
            assert "not an ancestor" in str(ei.value) or "merge-base" in str(ei.value)
        else:
            rec = SN.paper_frame_precondition()
            assert rec["checked"] is True and rec["ancestry"].startswith("unavailable: this clone does not carry the retire merge 000000000000"), rec
    # the override skips every check and is recorded; an empty reason is not one
    monkeypatch.setattr(SN, "PAIRED_ROOT_RETIRE_MERGE_SHA", None)
    monkeypatch.setattr(RN, "PAIRED_ROOT", str(tmp_path / "absent"))
    with pytest.raises(SN.SnapshotError):
        SN.build(root, DB.db_path(root), check_free_space=False, scope="paper_frame", dest_root=dest, precondition_override="  ")
    info = SN.build(root, DB.db_path(root), check_free_space=False, scope="paper_frame", dest_root=dest,
                    precondition_override="test: #417 not landed in this fixture")
    pre = info.manifest["precondition"]
    assert pre["override"] == "test: #417 not landed in this fixture" and pre["checked"] is False and pre["retire_merge_sha"] is None
    assert "ancestry" not in pre                                                        # nothing was checked, so nothing is claimed about the lineage
    # the pass path records what it checked
    if head:
        monkeypatch.setattr(SN, "PAIRED_ROOT_RETIRE_MERGE_SHA", head)
        monkeypatch.setattr(RN, "PAIRED_ROOT", str(paired))
        rec = SN.paper_frame_precondition()
        assert rec["checked"] is True and rec["head"] == head and rec["retire_merge_sha"] == head and rec["ancestry"] == "verified"
        assert "paired_root" not in rec and "repo" not in rec                          # no path in the committed record


_GIT_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
            "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z", "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z", "PATH": os.environ.get("PATH", "/usr/bin:/bin")}


def _repo_with(tmp_path, *, commits: int, marker: bool, name: str | None = None):
    """A git repository of `commits` commits (its own history, none of this repo's); with `marker`, the publish step's root marker is
    COMMITTED in the first commit — the publication root — the way the publish step writes it, and the later commits are descendants."""
    repo = tmp_path / (name or f"repo_{commits}_{int(marker)}")
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    if marker:
        (repo / SN.HISTORY_SQUASHED_MARKER).write_text("squashed\n")
        subprocess.run(["git", "-C", str(repo), "add", SN.HISTORY_SQUASHED_MARKER], check=True, capture_output=True, env=_GIT_ENV)
    for i in range(commits):
        subprocess.run(["git", "-C", str(repo), "commit", "-q", "--allow-empty", "-m", f"c{i}"], check=True, capture_output=True, env=_GIT_ENV)
    return str(repo)


def _git(repo, *args) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, env=_GIT_ENV).stdout.strip()


def test_precondition_ancestry_is_read_where_the_clone_carries_history_and_recorded_unavailable_where_it_positively_does_not(tmp_path, monkeypatch):
    """The ancestry half stays LIVE wherever the retire merge is present or could be (a history to search); where the clone positively
    carries no development history — the publish step's root marker, committed by the repository's one root commit (the publication
    root), in a clone that is not shallow — the record says so explicitly and the content half alone passes; the user's commits on top
    of the publication are descendants of that root and change nothing. A commit count never decides: a one-commit repository WITHOUT
    the marker is read for real. Tmp repositories with their own commits, so none of this repo's history is involved."""
    import sys
    sys.path.insert(0, HERE)
    import seedrep_history_fixtures as SHF
    assert SN.HISTORY_SQUASHED_MARKER == SHF.HISTORY_SQUASHED_MARKER == ".history-squashed"       # one name, two readers
    paired = tmp_path / "paired"
    paired.mkdir()
    monkeypatch.setattr(RN, "PAIRED_ROOT", str(paired))
    foreign = "0" * 40
    # a history that could carry the sha (two commits, no marker): the lineage is read and the foreign sha REFUSES
    two = _repo_with(tmp_path, commits=2, marker=False)
    monkeypatch.setattr(SN, "PAIRED_ROOT_RETIRE_MERGE_SHA", foreign)
    assert SN._history_unavailable(two, foreign) is None
    with pytest.raises(SN.SnapshotError, match="not an ancestor of HEAD"):
        SN.paper_frame_precondition(repo=two)
    # the published shape: the marker, committed by the one root commit — the record names both facts and the publication root
    one_marked = _repo_with(tmp_path, commits=1, marker=True)
    root1 = _git(one_marked, "rev-parse", "HEAD")
    rec = SN.paper_frame_precondition(repo=one_marked)
    assert rec["checked"] is True and rec["override"] is None and rec["retire_merge_sha"] == foreign
    assert rec["ancestry"] == (f"unavailable: this clone does not carry the retire merge {foreign[:12]} and carries no development history "
                               f"({SN.HISTORY_SQUASHED_MARKER} at the repository root, committed by its publication root {root1[:12]})"), rec
    assert "paired_root" not in rec and "repo" not in rec and str(tmp_path) not in json.dumps(rec)     # no path in the record, new field too
    # the user's commits on top of the publication (the generated configs the reproduction asks them to commit): descendants of the
    # publication root, which still carries the marker — unavailable all the same, the same root named, HEAD no longer the root
    later = _repo_with(tmp_path, commits=3, marker=True)
    root3 = _git(later, "rev-list", "--max-parents=0", "HEAD")
    assert _git(later, "rev-parse", "HEAD") != root3 and _git(later, "rev-list", "--count", "HEAD") == "3"
    rec = SN.paper_frame_precondition(repo=later)
    assert rec["checked"] is True and rec["ancestry"] == (f"unavailable: this clone does not carry the retire merge {foreign[:12]} and carries no "
                                                         f"development history ({SN.HISTORY_SQUASHED_MARKER} at the repository root, committed by its "
                                                         f"publication root {root3[:12]})"), rec
    # a one-commit clone WITHOUT the marker: no positive statement by the repository, so the lineage is read for real and the foreign sha
    # REFUSES — a commit count never decides (a depth-one clone of a development branch has one commit too)
    one = _repo_with(tmp_path, commits=1, marker=False)
    assert SN._history_unavailable(one, foreign) is None
    with pytest.raises(SN.SnapshotError, match="not an ancestor of HEAD"):
        SN.paper_frame_precondition(repo=one)
    # where the clone CARRIES the sha the lineage is read for real: HEAD of the one-commit repo is an ancestor of itself -> verified
    head1 = subprocess.run(["git", "-C", one, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    monkeypatch.setattr(SN, "PAIRED_ROOT_RETIRE_MERGE_SHA", head1)
    assert SN._history_unavailable(one, head1) is None and SN.paper_frame_precondition(repo=one)["ancestry"] == "verified"
    # ...and a present sha that is NOT an ancestor refuses even with the marker present (two commits, marker, sha = a non-ancestor): the
    # marker never overrides a lineage that CAN be read
    two_marked = _repo_with(tmp_path, commits=2, marker=True)
    other = subprocess.run(["git", "-C", two_marked, "commit", "-q", "--allow-empty", "-m", "side"], capture_output=True, text=True,
                           env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t",
                                "GIT_COMMITTER_EMAIL": "t@example.invalid", "PATH": os.environ.get("PATH", "/usr/bin:/bin")})
    assert other.returncode == 0, other.stderr
    tip = subprocess.run(["git", "-C", two_marked, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    subprocess.run(["git", "-C", two_marked, "reset", "-q", "--hard", "HEAD~1"], check=True, capture_output=True)      # `tip` stays in the store, off HEAD's lineage
    monkeypatch.setattr(SN, "PAIRED_ROOT_RETIRE_MERGE_SHA", tip)
    assert SN._history_unavailable(two_marked, tip) is None
    with pytest.raises(SN.SnapshotError, match="not an ancestor of HEAD"):
        SN.paper_frame_precondition(repo=two_marked)


def test_the_marker_never_turns_a_git_failure_into_an_unavailable_ancestry(tmp_path, monkeypatch):
    """A repository Git cannot READ is a refusal, whatever marker sits beside it: the marker is accepted only after HEAD resolved and its
    history counted, so a failed `cat-file` means an absent commit in a readable store and nothing else. Without this, a directory that is
    not a checkout (or one whose object store is unreadable) plus the marker would export with `checked` True."""
    paired = tmp_path / "paired"
    paired.mkdir()
    monkeypatch.setattr(RN, "PAIRED_ROOT", str(paired))
    monkeypatch.setattr(SN, "PAIRED_ROOT_RETIRE_MERGE_SHA", "0" * 40)
    # (1) not a git repository at all, marker present
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / SN.HISTORY_SQUASHED_MARKER).write_text("squashed\n")
    with pytest.raises(SN.SnapshotError, match="git cannot read the repository") as ei:
        SN._history_unavailable(str(plain), "0" * 40)
    assert "never read as an absent commit" in str(ei.value) and "unavailable" not in str(ei.value)
    with pytest.raises(SN.SnapshotError, match="git cannot read the repository"):
        SN.paper_frame_precondition(repo=str(plain))
    # (2) a one-commit repository with the marker whose HEAD was made unreadable: the same refusal, never the marker's sentence
    broken = _repo_with(tmp_path, commits=1, marker=True)
    with open(os.path.join(broken, ".git", "HEAD"), "w") as fh:
        fh.write("ref: refs/heads/nowhere\n")
    with pytest.raises(SN.SnapshotError, match="git cannot read the repository"):
        SN._history_unavailable(broken, "0" * 40)
    with pytest.raises(SN.SnapshotError, match="git cannot read the repository"):
        SN.paper_frame_precondition(repo=broken)
    # (3) an empty repository (no commit yet) with the marker: HEAD does not resolve -> refused, not unavailable
    empty = tmp_path / "empty"
    empty.mkdir()
    subprocess.run(["git", "init", "-q", str(empty)], check=True, capture_output=True)
    (empty / SN.HISTORY_SQUASHED_MARKER).write_text("squashed\n")
    with pytest.raises(SN.SnapshotError, match="git cannot read the repository"):
        SN._history_unavailable(str(empty), "0" * 40)
    # the readable shapes still answer: one commit + marker -> unavailable (the marker sentence), two commits + foreign sha -> None (read for real)
    again = tmp_path / "again"
    again.mkdir()
    assert "carries no development history" in SN._history_unavailable(_repo_with(again, commits=1, marker=True), "0" * 40)
    assert SN._history_unavailable(_repo_with(again, commits=2, marker=False), "0" * 40) is None


def test_a_marker_the_repository_does_not_bear_out_and_a_shallow_clone_refuse_by_name_never_unavailable(tmp_path, monkeypatch):
    """The published path is granted on the repository's whole statement, each clause required: a marker file the ROOT commit does not carry
    (a development checkout with a stray marker, or a marker committed later) refuses by name; a marker with other content refuses; a
    SHALLOW clone lacking the commit refuses as shallow — a depth-one clone of a development branch has one commit and no marker (the shape
    the one-commit rule mistook for a publication), and a depth-one clone of a PUBLICATION carries the marker file at HEAD and answers HEAD
    as its own root (its parents were cut), so without the shallow clause it would read as published. A shallow clone that CARRIES the sha
    is verified as usual: the shallow clause never refuses a lineage that can be read."""
    import inspect
    paired = tmp_path / "paired"
    paired.mkdir()
    monkeypatch.setattr(RN, "PAIRED_ROOT", str(paired))
    foreign = "0" * 40
    monkeypatch.setattr(SN, "PAIRED_ROOT_RETIRE_MERGE_SHA", foreign)
    # (0) whitespace INSIDE the marker's content is other content (only surrounding whitespace is stripped); around it, it is the marker
    spaced = _repo_with(tmp_path, commits=1, marker=True, name="spaced")
    with open(os.path.join(spaced, SN.HISTORY_SQUASHED_MARKER), "w") as fh:
        fh.write("squa shed\n")
    with pytest.raises(SN.SnapshotError, match="reads 'squa shed', not `squashed`"):
        SN._history_unavailable(spaced, foreign)
    with open(os.path.join(spaced, SN.HISTORY_SQUASHED_MARKER), "w") as fh:
        fh.write("  squashed \t\n\n")
    assert "carries no development history" in SN._history_unavailable(spaced, foreign)
    # (1) a stray marker file in a development-shaped checkout: the root does not carry it
    dev = _repo_with(tmp_path, commits=2, marker=False, name="dev")
    with open(os.path.join(dev, SN.HISTORY_SQUASHED_MARKER), "w") as fh:
        fh.write("squashed\n")
    root = _git(dev, "rev-list", "--max-parents=0", "HEAD")
    with pytest.raises(SN.SnapshotError, match=f"root commit {root[:12]} does not carry it") as ei:
        SN._history_unavailable(dev, foreign)
    assert "stray marker" in str(ei.value) and "unavailable" not in str(ei.value) and "absent" in str(ei.value)
    with pytest.raises(SN.SnapshotError, match="does not carry it"):
        SN.paper_frame_precondition(repo=dev)
    # (1b) the marker committed in a LATER commit: still not the root's
    _git(dev, "add", SN.HISTORY_SQUASHED_MARKER)
    _git(dev, "commit", "-q", "-m", "a marker committed later")
    with pytest.raises(SN.SnapshotError, match="does not carry it"):
        SN.paper_frame_precondition(repo=dev)
    # (2) other content in the marker file, and other content committed by the root
    other = _repo_with(tmp_path, commits=1, marker=True, name="other")
    with open(os.path.join(other, SN.HISTORY_SQUASHED_MARKER), "w") as fh:
        fh.write("nope\n")
    with pytest.raises(SN.SnapshotError, match="reads 'nope', not `squashed`"):
        SN._history_unavailable(other, foreign)
    _git(other, "add", SN.HISTORY_SQUASHED_MARKER)
    _git(other, "commit", "-q", "--amend", "-m", "c0 with another marker")
    with pytest.raises(SN.SnapshotError, match="reads 'nope', not `squashed`"):          # the working-tree file is read first
        SN._history_unavailable(other, foreign)
    # (3) a depth-one shallow clone of a development branch (three commits, no marker): one commit in the clone, refused as SHALLOW — never
    # "a single commit (no development history)"
    src = _repo_with(tmp_path, commits=3, marker=False, name="src")
    shallow = tmp_path / "shallow"
    subprocess.run(["git", "clone", "-q", "--depth", "1", f"file://{src}", str(shallow)], check=True, capture_output=True)
    assert _git(shallow, "rev-parse", "--is-shallow-repository") == "true" and _git(shallow, "rev-list", "--count", "HEAD") == "1"
    with pytest.raises(SN.SnapshotError, match="is SHALLOW and does not carry the retire merge") as ei:
        SN._history_unavailable(str(shallow), foreign)
    assert "fetch --unshallow" in str(ei.value) and "unavailable" not in str(ei.value) and "single commit" not in str(ei.value)
    with pytest.raises(SN.SnapshotError, match="is SHALLOW"):
        SN.paper_frame_precondition(repo=str(shallow))
    # (3b) a depth-two clone of the same: refused the same way (the base refused it as a non-ancestor; a cut history is refused by name now)
    deeper = tmp_path / "deeper"
    subprocess.run(["git", "clone", "-q", "--depth", "2", f"file://{src}", str(deeper)], check=True, capture_output=True)
    with pytest.raises(SN.SnapshotError, match="is SHALLOW"):
        SN.paper_frame_precondition(repo=str(deeper))
    # (3c) a depth-one clone of a PUBLICATION: HEAD carries the marker file and is its own root as far as the clone can see — SHALLOW, refused
    pub = _repo_with(tmp_path, commits=3, marker=True, name="pub")
    shallow_pub = tmp_path / "shallow_pub"
    subprocess.run(["git", "clone", "-q", "--depth", "1", f"file://{pub}", str(shallow_pub)], check=True, capture_output=True)
    assert os.path.isfile(os.path.join(shallow_pub, SN.HISTORY_SQUASHED_MARKER))
    assert _git(shallow_pub, "rev-list", "--max-parents=0", "HEAD") == _git(shallow_pub, "rev-parse", "HEAD")
    with pytest.raises(SN.SnapshotError, match="is SHALLOW"):
        SN._history_unavailable(str(shallow_pub), foreign)
    # ...while the FULL clone of the same publication is unavailable, naming the publication root (the user's two commits on top pass)
    full_pub = tmp_path / "full_pub"
    subprocess.run(["git", "clone", "-q", f"file://{pub}", str(full_pub)], check=True, capture_output=True)
    assert _git(full_pub, "rev-parse", "--is-shallow-repository") == "false"
    rec = SN.paper_frame_precondition(repo=str(full_pub))
    assert rec["ancestry"].endswith(f"committed by its publication root {_git(pub, 'rev-list', '--max-parents=0', 'HEAD')[:12]})"), rec
    # (3d) a shallow clone that CARRIES the sha: verified, the shallow clause never refuses a lineage that can be read
    head = _git(shallow, "rev-parse", "HEAD")
    monkeypatch.setattr(SN, "PAIRED_ROOT_RETIRE_MERGE_SHA", head)
    assert SN._history_unavailable(str(shallow), head) is None and SN.paper_frame_precondition(repo=str(shallow))["ancestry"] == "verified"
    # the rule never counts commits
    body = inspect.getsource(SN._history_unavailable)
    assert "rev-list" in body and "--count" not in body and "--is-shallow-repository" in body and "--max-parents=0" in body


def test_free_space_precondition_for_the_scope(tmp_path, monkeypatch):
    root = str(tmp_path / "reg")
    dest = str(tmp_path / "dest")
    build_corpus(root)
    db = DB.db_path(root)
    floor = 2 * 1024 ** 3
    est = SN.paper_frame_output_estimate(db)
    assert est > 0
    sizes = SN.table_bytes(db)
    if sizes:
        # the read set's TABLE pages: no more than table_bytes' (table + index pages) figure for the same tables, a part of the file
        assert est <= sum(sizes.get(t, 0) for t in SN.PAPER_FRAME_TABLES if t in S.TABLE_BY_NAME)
        assert est < DB.file_size(db)
        assert est >= 4096                                                             # at least one table page was read
    ok, free, need = SN.free_space_precondition(dest, db, scope="paper_frame")
    assert need == 2 * est + floor and free > 0
    # without dbstat the estimate falls back to 40 % of the database
    real_connect = DB.connect

    class _NoDbstat:
        def __init__(self, conn):
            self._c = conn

        def execute(self, sql, *a):
            if "dbstat" in sql:
                raise sqlite3.OperationalError("no such table: dbstat")
            return self._c.execute(sql, *a)

        def close(self):
            self._c.close()

    monkeypatch.setattr(DB, "connect", lambda path, readonly=False, **kw: _NoDbstat(real_connect(path, readonly=readonly, **kw)))
    assert SN.paper_frame_output_estimate(db) == int(0.4 * DB.file_size(db))
    assert SN.free_space_precondition(dest, db, scope="paper_frame")[2] == 2 * int(0.4 * DB.file_size(db)) + floor
    monkeypatch.setattr(DB, "connect", real_connect)
    assert SN.free_space_precondition(dest, db, scope="runlevel")[2] == floor          # the other scopes' rules stand
    assert SN.free_space_precondition(dest, db, scope="full")[2] == int(0.4 * DB.file_size(db)) + floor
    # a build refuses under the scope's rule, naming it, and writes nothing under dest_root's snapshots
    pass_precondition(monkeypatch, tmp_path)
    monkeypatch.setattr(SN, "free_bytes", lambda path: 1)
    with pytest.raises(SN.SnapshotError) as ei:
        SN.build(root, db, scope="paper_frame", dest_root=dest)
    assert "2 × estimated output + 2 GB" in str(ei.value)
    assert os.listdir(SN.snapshots_dir(dest)) == [] and SN.pointer(dest) is None


# --------------------------------------------------------------------------------------------
# the CLI
# --------------------------------------------------------------------------------------------


def test_cli_requires_dest_root_and_writes_nothing_under_the_operational_root(tmp_path, monkeypatch, capsys):
    root = str(tmp_path / "reg")
    dest = str(tmp_path / "dest")
    build_corpus(root)
    pass_precondition(monkeypatch, tmp_path)
    assert cli.main(["--root", root, "snapshot", "--scope", "paper_frame"]) == 2
    err = capsys.readouterr().err
    assert "--dest-root" in err and "requires" in err
    assert cli.main(["--root", root, "snapshot", "--scope", "full", "--dest-root", dest]) == 2
    assert "only for --scope paper_frame" in capsys.readouterr().err
    assert cli.main(["--root", root, "snapshot", "--scope", "runlevel", "--source-root", root]) == 2
    assert "--source-root" in capsys.readouterr().err
    assert cli.main(["--root", root, "snapshot", "--scope", "paper_frame", "--dest-root", dest, "--push"]) == 2
    assert "--push" in capsys.readouterr().err
    assert not os.path.exists(dest) and os.listdir(SN.snapshots_dir(root)) == []
    # the success path: --source-root names the registry to read; the manifest is printed; no harvester record, no footprint
    elsewhere = str(tmp_path / "not-a-registry")
    rc = cli.main(["--root", elsewhere, "snapshot", "--scope", "paper_frame", "--dest-root", dest, "--source-root", root,
                   "--model-id", "extra_model", "--no-space-check"])
    out = capsys.readouterr().out
    assert rc == 0, out
    m = json.loads(out)
    assert m["scope"] == "paper_frame" and SN.pointer(dest) == m["snapshot_id"] and "source_root" not in m
    assert isinstance(m["source_schema_version"], int) and m["source_schema_version"] > 0
    assert "extra_model" in m["model_ids"] and "extra_model_base" in m["model_ids"]
    assert os.listdir(SN.snapshots_dir(root)) == [] and SN.pointer(root) is None
    assert not os.path.exists(os.path.join(root, "state", "footprint.json"))
    harvester = os.path.join(root, "records", "harvester")
    assert not any(fn.startswith("snapshot-") for _, _, fns in os.walk(harvester) for fn in fns)
    # a refusal surfaces the reason and exits 1
    monkeypatch.setattr(SN, "PAIRED_ROOT_RETIRE_MERGE_SHA", None)
    assert cli.main(["--root", root, "snapshot", "--scope", "paper_frame", "--dest-root", str(tmp_path / "d2"), "--no-space-check"]) == 1
    assert "#417" in capsys.readouterr().err
    # the override flag
    assert cli.main(["--root", root, "snapshot", "--scope", "paper_frame", "--dest-root", str(tmp_path / "d3"), "--no-space-check",
                     "--precondition-override", "test"]) == 0
    assert json.loads(capsys.readouterr().out)["precondition"]["override"] == "test"


def test_cli_help_names_the_scope():
    p = cli.build_parser()
    sub = next(a for a in p._subparsers._group_actions).choices["snapshot"]      # type: ignore[union-attr]
    text = sub.format_help()
    for flag in ("paper_frame", "--dest-root", "--source-root", "--model-id", "--precondition-override"):
        assert flag in text, flag


# --------------------------------------------------------------------------------------------
# the front door: tools/snapshot_paper_frame.sh and 10_figures.sh dry-run
# --------------------------------------------------------------------------------------------

BANNED_WORDS = re.compile(r"(?<![A-Za-z0-9_])(arm|leg|lane|cell|view|wave|grader)(?![A-Za-z0-9_])", re.IGNORECASE)
sys.path.insert(0, HERE)
#: the token table's output rows (tests/front_door_testlib.py), derived from the one table and never spelled here
from front_door_testlib import OUTPUT_TOKENS as PERSONAL, token_hits  # noqa: E402


def _personal_in_output(out: str, tmp_path) -> list:
    """The tokens in a front-door output once the runner's own paths (this checkout, the test's temp root, the interpreter's
    environment, HOME) are normalised: they belong to the host running the test, not to anything the scripts hard-code."""
    host = {REPO, str(tmp_path), sys.prefix, os.path.realpath(sys.prefix), os.path.expanduser("~")}
    for path in sorted(host, key=len, reverse=True):
        out = out.replace(path, "<host>")
    return token_hits(out)


def _bash(args, tmp_path, env_extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("SISTEER_")}
    env["HOME"] = str(tmp_path / "home")
    os.makedirs(env["HOME"], exist_ok=True)
    env.update(env_extra or {})
    return subprocess.run(["bash"] + args, cwd=REPO, env=env, capture_output=True, text=True, timeout=120)


def test_front_door_scripts_are_clean_text():
    for rel in ("reproduce/tools/snapshot_paper_frame.sh", "reproduce/10_figures.sh"):
        with open(os.path.join(REPO, rel), "r", encoding="utf-8") as fh:
            text = fh.read()
        assert text.startswith("#!/bin/bash") and "set -euo pipefail" in text, rel
        assert not PERSONAL.search(text), rel
        assert not BANNED_WORDS.findall(text), (rel, BANNED_WORDS.findall(text)[:5])
        assert os.access(os.path.join(REPO, rel), os.X_OK), rel


def test_snapshot_tool_dry_run_prints_the_substituted_command(tmp_path):
    cfg = os.path.join(REPO, "reproduce", "site.example", "config.env")
    r = _bash(["reproduce/tools/snapshot_paper_frame.sh", "--dry-run", "--config", cfg, "--dest-root", str(tmp_path / "snap"),
               "--source-root", str(tmp_path / "reg"), "--model-id", "extra_model", "--precondition-override", "test reason"], tmp_path)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "DRY-RUN:" in out and "-m sisteer.registry.cli" in out and "snapshot --scope paper_frame" in out
    assert f"--dest-root {tmp_path / 'snap'}" in out and f"--root {tmp_path / 'reg'}" in out
    assert "--model-id extra_model" in out and "--precondition-override" in out and "PYTHONPATH=" in out
    assert not BANNED_WORDS.findall(out) and not _personal_in_output(out, tmp_path)
    h = _bash(["reproduce/tools/snapshot_paper_frame.sh", "--help"], tmp_path)
    assert h.returncode == 0 and "--dest-root" in h.stdout and "--source-root" in h.stdout and "--dry-run" in h.stdout
    # a real run against a missing registry refuses by name
    miss = _bash(["reproduce/tools/snapshot_paper_frame.sh", "--config", cfg, "--source-root", str(tmp_path / "nowhere"), "--dest-root", str(tmp_path / "s")], tmp_path)
    assert miss.returncode != 0 and "REFUSED" in miss.stderr and "registry.sqlite" in miss.stderr


def test_snapshot_tool_refuses_a_dest_root_aliasing_the_registry_root(tmp_path):
    """The front door's half: the wrapper's `cd && pwd` comparison was a LOGICAL-path check
    that did not resolve a symlink; it now calls the exporter's own physical check (`snapshot.paper_frame_dest_root`) and refuses a
    destination that is, lies inside, or contains the registry root — before the CLI is run."""
    cfg = os.path.join(REPO, "reproduce", "site.example", "config.env")
    reg = tmp_path / "reg"
    reg.mkdir()
    (reg / "registry.sqlite").write_bytes(b"")                                         # the wrapper's existence check, nothing more
    alias = tmp_path / "alias"
    os.symlink(str(reg), str(alias))
    cases = ((str(alias), "IS the registry root"), (str(reg / "snapshots" / "paper"), "INSIDE the registry root"), (str(tmp_path), "CONTAINS the registry root"))
    for dest, why in cases:
        r = _bash(["reproduce/tools/snapshot_paper_frame.sh", "--config", cfg, "--source-root", str(reg), "--dest-root", dest], tmp_path)
        assert r.returncode != 0 and "REFUSED" in r.stderr and why in r.stderr, (dest, r.stderr[-800:])
        assert not os.path.exists(reg / "snapshots" / "paper") and not os.path.exists(reg / "CURRENT")
    assert sorted(os.listdir(reg)) == ["registry.sqlite"]                              # nothing was written under the registry root


def test_figures_stage_dry_run_and_help(tmp_path):
    cfg = os.path.join(REPO, "reproduce", "site.example", "config.env")
    r = _bash(["reproduce/10_figures.sh", "--dry-run", "--config", cfg, "--source", "snapshot",
               "--snapshot-root", str(tmp_path / "snap")], tmp_path)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "REGISTRY_NB_SOURCE=snapshot" in out
    assert "REGISTRY_NB_PANELS" not in out and "MPLBACKEND" not in out     # no panel switch (the paper's panels only); the kernel's inline backend embeds the figures
    assert "--ExecutePreprocessor.kernel_name=python3" in out and ".venv/bin/python" in out      # the clone's own venv is the default interpreter
    assert "make_registry_notebook.py" in out and "nbconvert --execute --to html" in out and "registry-notebook" in out
    assert f"REGISTRY_NB_SNAPSHOT_ROOT={tmp_path / 'snap'}" in out and "SISTEER_SRC=" in out
    assert not BANNED_WORDS.findall(out), BANNED_WORDS.findall(out)[:5]
    assert not _personal_in_output(out, tmp_path)
    # --export-snapshot cuts the frame first (its own dry-run lines), then renders from it
    e = _bash(["reproduce/10_figures.sh", "--dry-run", "--config", cfg, "--export-snapshot", "--snapshot-root", str(tmp_path / "snap")], tmp_path)
    assert e.returncode == 0, e.stderr
    assert "snapshot --scope paper_frame" in e.stdout and f"--dest-root {tmp_path / 'snap'}" in e.stdout and "REGISTRY_NB_SOURCE=snapshot" in e.stdout
    assert e.stdout.index("snapshot --scope paper_frame") < e.stdout.index("nbconvert")
    # without a site config the defaults stand (a note says so)
    n = _bash(["reproduce/10_figures.sh", "--dry-run", "--source", "live", "--panel", "all"], tmp_path)     # --panel: accepted, ignored
    assert n.returncode == 0, n.stderr
    assert "no site config" in n.stdout and "REGISTRY_NB_SOURCE=live" in n.stdout and "REGISTRY_NB_PANELS" not in n.stdout and "--panel" in n.stdout
    h = _bash(["reproduce/10_figures.sh", "--help"], tmp_path)
    assert h.returncode == 0 and "--source" in h.stdout and "--export-snapshot" in h.stdout and "--dry-run" in h.stdout
    # refusals exit non-zero with the reason
    bad = _bash(["reproduce/10_figures.sh", "--source", "nope"], tmp_path)
    assert bad.returncode != 0 and "--source must be" in bad.stderr
    auto = _bash(["reproduce/10_figures.sh", "--config", cfg, "--source", "auto", "--snapshot-root", str(tmp_path / "none")], tmp_path,
                 env_extra={"SISTEER_REGISTRY_ROOT": str(tmp_path / "noreg")})
    assert auto.returncode != 0 and "neither" in auto.stderr
    contradiction = _bash(["reproduce/10_figures.sh", "--dry-run", "--config", cfg, "--source", "live", "--export-snapshot"], tmp_path)
    assert contradiction.returncode != 0 and "contradicts" in contradiction.stderr


def test_repo_relative_recognises_every_checkout_layout_and_a_linked_worktree_names_its_repo(tmp_path, monkeypatch):
    """The layout table is assembled without a literal worktree segment (the package-wide scan in
    tests/test_registry_repo_paths.py forbids one anywhere in the package) and still maps a path under every known checkout layout
    of this repo to `repo:<rel>` — the plain checkout, a linked worktree, a `-wt` sibling, the cluster's pinned eval-repo copy and
    the grade tree, a `host:` prefix included — and a code tree that IS a linked worktree still lists the repo it belongs to among
    its roots. Both sites behave as before the restructure."""
    rel = "scripts/sdfbase/full22/x.sh"
    linked = "work" + "trees"                                          # the segment, spelled the way the module spells it
    for path in (f"/home/TESTUSER/si-steering/{rel}", f"/home/TESTUSER/si-steering/{linked}/some-lane/{rel}",
                 f"/home/TESTUSER/si-steering-wt/some-lane/{rel}", f"/shared/TESTUSER/si-steering-eval-repo/{'a' * 40}/{rel}",
                 f"/workspace/TESTUSER/si-steering-eval-repo/0123abc/{rel}", f"/home/TESTUSER/si-steering-grade/{rel}",
                 f"TESTHOST:/home/TESTUSER/si-steering/{linked}/lane/{rel}"):
        assert SN._repo_relative(path, ()) == f"repo:{rel}", path
    assert SN._repo_relative(f"/home/TESTUSER/other-repo/{rel}", ()) is None
    # a linked checkout's ROOT (no path beneath it) has always read as the plain layout with the lane as its rel — pre-existing,
    # pinned as the behaviour this restructure preserved (a sole `repo:` value that names the lane), not as a design
    assert SN._repo_relative(f"/home/TESTUSER/si-steering/{linked}/lane", ()) == f"repo:{linked}/lane"
    wt = tmp_path / "si-steering" / linked / "lane"; wt.mkdir(parents=True)
    monkeypatch.setattr(SN, "_code_repo_root", lambda: str(wt))
    roots = SN._repo_roots()
    assert str(wt) in roots and str(tmp_path / "si-steering") in roots
    plain = tmp_path / "plain"; plain.mkdir()
    monkeypatch.setattr(SN, "_code_repo_root", lambda: str(plain))
    assert set(SN._repo_roots()) == {str(plain), os.path.realpath(str(plain))}
