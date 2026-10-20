"""`scripts/registry/audit_new_validator_rules.py` (plan `recordlog-d635-d636-d639` section 5, commit C7): the read-only whole-log
audit of the validator rules the record-log branch added, against a synthetic registry -- a clean corpus passes with every expectation
held and the live figures the window reads (schema version, refusals and facts, the two currency rules, the C1c buckets); one record
per rule planted in the log (a judgment about a claim-backed judgment stated without the claim's identity inputs, a placeholder
`code_sha`, a derived-pair `run_patch`, a token-cap ladder disagreement, a `named_role` outside the vocabulary, a legacy top-level
`extra`) is found, named and refused by the verdict until grandfathered; the grandfather accounting both ways (an unlisted hit, a listed
sha that is no hit); a file the live harvester gzips between the listing and the read is still read, once; the CLI's exit codes; and
nothing written under the root.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import os
import subprocess
import sys

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import migrations as MIG
from sisteer.registry import pending as P
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry import validate as VAL

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts", "registry", "audit_new_validator_rules.py")
PY = sys.executable


def _load():
    """The script as a module (registered in ``sys.modules`` first: its dataclasses look their module up there)."""
    spec = importlib.util.spec_from_file_location("audit_new_validator_rules_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _root(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    return root


def _write(root, sub, name, recs):
    return R.write_records(os.path.join(root, "records", sub, name), recs)


def _ingest(root):
    return H.ingest_once(root, workers=0, min_free=0)


def _seed(root):
    """Dimensions plus one generated run in its own directory, folded."""
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid, recs = synth.run_records("k1", 100, through="generated")
    sub = os.path.join(synth.CAMPAIGN, I.run_dir_name(rid))
    _write(root, sub, "000001-run.jsonl", recs)
    rep = _ingest(root)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    return rid, sub


def _files_under(root):
    """Every file under the root but SQLite's side files of a read-only open (the documented allowance)."""
    out = set()
    for dp, _dn, fns in os.walk(root):
        for fn in fns:
            if fn.endswith(("-wal", "-shm")):
                continue
            out.add(os.path.relpath(os.path.join(dp, fn), root))
    return out


def _run_dir_sub(rid):
    return os.path.join(synth.CAMPAIGN, I.run_dir_name(rid))


# --------------------------------------------------------------------------------------------
# a clean corpus: every expectation holds and the window's live figures are read
# --------------------------------------------------------------------------------------------


def test_a_clean_corpus_passes_and_reports_the_windows_live_figures_without_writing(tmp_path, monkeypatch):
    """The live grandfather list names records of the LIVE log, none of which a synthetic log holds (the next test pins what the
    audit says about that); a clean verdict here is judged against an empty list."""
    A = _load()
    root = _root(tmp_path)
    rid, _sub = _seed(root)
    monkeypatch.setattr(VAL, "GRANDFATHERED_RECORD_SHAS", frozenset())
    before = _files_under(root)
    rep = A.run_audit(root)
    assert _files_under(root) == before                                # nothing written: no state/last_run, no file of its own
    assert rep["ok"] is True and rep["failures"] == []
    log = rep["log"]
    assert log["file_kinds"] == list(A.AUDITED_RECORD_TYPES) and log["files_read"] == 1 and log["misfiled"] == {}
    assert log["run"]["records"] == 1 and log["run"]["ladder_empty_or_absent"] == 1 and log["rung_violations"]["count"] == 0
    assert log["placeholder_run_code_sha"]["count"] == 0 and log["run"]["code_sha_unverified"] == 0
    assert log["run_patch"]["records"] == 2 and log["run_patch"]["derived_pair_patches"]["count"] == 0
    assert log["run_patch"]["top_level_extra_records"] == 0 and log["judgment"]["records"] == 0 and log["object"]["records"] == 0
    gf = rep["grandfather"]
    assert gf["listed"] == 0 and gf["grandfathered_unaccounted"] == [] and gf["judgment_id_form_misses"]["hits"] == 0
    db = rep["database"]
    assert db["schema_version"] == MIG.code_version() == db["code_version"]
    rf = db["refusals"]
    assert rf["current_refusals"] == 0 and rf["judged_with_column_present"] and rf["fact_table_present"] and rf["facts"] == 0
    assert rf["current_without_fact"] == 0 and rf["refused_holds"] == 0 and rf["fact_rows_ceiling_after_first_cycle"] == 0
    cr = rep["database"]["currency_rules"]
    assert cr["d744_partial_displaced_by_complete"] == {"partial_judgments": 0, "run_judge_pairs": 0, "witness_identity_current": True}
    assert cr["d757_identity_bound_at_claim"]["binding_columns_present"] and cr["d757_identity_bound_at_claim"]["stale_bound"] == 0
    assert db["object_lineage"]["columns_present"] and db["object_lineage"]["column_named_role"] == 0
    assert rep["c1c"]["runs"] == 0 and rep["c1c"]["runs_by_bucket"]["missing"] == 0
    text = A.format_report(rep)
    assert "every pre-landing expectation holds" in text or "grandfathered_unaccounted" in text


def test_the_live_grandfather_list_is_unaccounted_in_a_synthetic_log_and_hash_all_locates_nothing(tmp_path, monkeypatch):
    """The real list names records of the LIVE log; a synthetic log holds none of them, so the accounting reports every entry as
    unaccounted (a failure: the list must equal the audit's hits), and `--hash-all` says they are not in this log at all."""
    A = _load()
    root = _root(tmp_path)
    _seed(root)
    monkeypatch.setattr(VAL, "GRANDFATHERED_RECORD_SHAS", frozenset({"f" * 64}))
    rep = A.run_audit(root, hash_all=True)
    gf = rep["grandfather"]
    assert gf["grandfathered_unaccounted"] == ["f" * 64] and gf["grandfathered_unaccounted_not_in_log"] == ["f" * 64]
    assert gf["grandfathered_unaccounted_in_log"] == {} and rep["log"]["hashed_all"] is True
    assert rep["ok"] is False and any(f.startswith("grandfathered_unaccounted") for f in rep["failures"])


# --------------------------------------------------------------------------------------------
# one planted record per rule: found, named, a failure until grandfathered
# --------------------------------------------------------------------------------------------


def _plant(root, rid, sub):
    """One record per rule, written into the log (the harvester refuses what the rules refuse; the files stay, as every file does).
    Returns the planted records by rule."""
    planted = {}
    # a record ABOUT a claim-backed judgment that omits the claim's identity inputs (the shape the live log holds 12 of on
    # 2026-10-04: supersedes and corrections re-stating a claimed id with run_id / judge_id alone)
    jid = P.claim(root, rid, synth.JUDGE_ID, "session-A", now=synth.t(500))
    assert I.judgment_id_form({"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID}) is None
    bare = synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID, "status": "complete", "n_items_judged": 200},
                     synth.t(600))
    _write(root, sub, "000002-judgment.jsonl", [bare])
    planted["judgment_id_form_misses"] = bare
    # a placeholder code_sha on a run record
    rid2, recs2 = synth.run_records("k2", 200, through="queued", code_sha="b" * 40)
    _write(root, _run_dir_sub(rid2), "000001-run.jsonl", recs2)
    planted["placeholder_run_code_sha"] = recs2[0]
    # an index outside its ladder
    rid3, recs3 = synth.run_records("k3", 300, through="queued", token_cap_tokens=512, token_cap_rung_index=5, token_cap_ladder_json=[512, 1024])
    _write(root, _run_dir_sub(rid3), "000001-run.jsonl", recs3)
    planted["rung_violations"] = recs3[0]
    # a patch naming a derived column
    derived = synth.rec("run_patch", {"run_id": rid, "fields": {"decode_policy_id": "p1"}}, synth.t(610))
    _write(root, sub, "000003-run_patch.jsonl", [derived])
    planted["run_patch_derived_pair"] = derived
    # a named_role outside the vocabulary, under extra as the old producers wrote it
    obj = synth.rec("object", dict(synth.object_lora(sha="e" * 64), extra={"named_role": "bogus_role"}), synth.t(620))
    _write(root, "objects", "000001-object.jsonl", [obj])
    planted["object_lineage_violations"] = obj
    # the legacy stamps C1c reconciles: a top-level extra the fold drops (folds; the key is MISSING on the live row)
    legacy = synth.rec("run_patch", {"run_id": rid, "fields": {"job_name": "legacy"}, "extra": {"cluster_label": "TESTSITE"}}, synth.t(630))
    _write(root, sub, "000004-run_patch.jsonl", [legacy])
    planted["legacy_extra"] = legacy
    rep = _ingest(root)
    assert rep.refused == 5, [str(m) for m in rep.refusals]      # the five rule hits; the legacy patch folds
    return planted


def test_one_planted_record_per_rule_is_found_named_and_a_failure_until_grandfathered(tmp_path, monkeypatch):
    A = _load()
    root = _root(tmp_path)
    rid, sub = _seed(root)
    monkeypatch.setattr(VAL, "GRANDFATHERED_RECORD_SHAS", frozenset())
    planted = _plant(root, rid, sub)
    rep = A.run_audit(root)
    log, gf = rep["log"], rep["grandfather"]
    assert rep["ok"] is False
    for rule in ("judgment_id_form_misses", "placeholder_run_code_sha", "rung_violations", "object_lineage_violations"):
        assert log[rule]["count"] == 1 and log[rule]["records"][0]["sha"] == R.record_sha(planted[rule]), rule
        assert gf[rule]["not_grandfathered"] == [R.record_sha(planted[rule])] and gf[rule]["grandfathered"] == 0, rule
        assert any(f.startswith(rule) for f in rep["failures"]), (rule, rep["failures"])
    miss = log["judgment_id_form_misses"]["records"][0]
    assert miss["status"] == "complete" and miss["claimed_by_session"] is None and miss["script"] == "tests.synthetic"
    assert miss["uri"] == f"records/{sub}/000002-judgment.jsonl#1" and miss["run_id"] == rid
    assert log["judgment"]["by_form"] == {"claimed_by_session": 1, "none": 1}       # the claim record itself conforms
    assert log["judgment"]["distinct_ids"] == 1 and log["judgment"]["multi_record_ids"] == 1 and log["judgment"]["run_id_changes"]["count"] == 0
    assert log["placeholder_run_code_sha"]["records"][0]["code_sha"] == "b" * 40
    assert log["rung_violations"]["records"][0]["field"] == "token_cap_rung_index" and log["run"]["index_stated"] == 1
    assert log["run_patch"]["derived_pair_patches"]["records"][0]["fields"] == ["decode_policy_id"]
    assert log["run_patch"]["top_level_extra_records"] == 1 and log["run_patch"]["top_level_extra_runs"] == 1
    assert log["object_lineage_violations"]["records"][0]["field"] == "named_role" and log["object"]["named_role_stated"] == 1
    assert gf["run_patch_derived_pair"]["not_grandfathered"] == [R.record_sha(planted["run_patch_derived_pair"])]
    # the database side saw the five refusals as facts, and the C1c dry run the one missing key
    rf = rep["database"]["refusals"]
    assert rf["current_refusals"] == 5 and rf["facts"] == 5 and rf["current_without_fact"] == 0 and rf["fact_rows_ceiling_after_first_cycle"] == 5
    assert rep["c1c"]["runs"] == 1 and rep["c1c"]["runs_by_bucket"]["missing"] == 1 and rep["c1c"]["keys_by_bucket"]["missing"] == 1
    assert rep["c1c"]["records_by_status"] == {"accepted": 1, "refused": 0, "blocked": 0, "not_ingested": 0}
    text = A.format_report(rep)
    assert "NOT GRANDFATHERED" in text and "FAILED expectations:" in text and "c1c buckets" in text
    # grandfather the two the plan's mechanism covers: those two rules clear, the others still fail, nothing unaccounted
    monkeypatch.setattr(VAL, "GRANDFATHERED_RECORD_SHAS",
                        frozenset({R.record_sha(planted["judgment_id_form_misses"]), R.record_sha(planted["placeholder_run_code_sha"])}))
    rep2 = A.run_audit(root)
    gf2 = rep2["grandfather"]
    assert gf2["judgment_id_form_misses"] == {"hits": 1, "grandfathered": 1, "not_grandfathered": []}
    assert gf2["placeholder_run_code_sha"] == {"hits": 1, "grandfathered": 1, "not_grandfathered": []}
    assert gf2["grandfathered_unaccounted"] == [] and rep2["ok"] is False
    assert not any(f.startswith(("judgment_id_form_misses", "placeholder_run_code_sha")) for f in rep2["failures"])
    assert any(f.startswith("rung_violations") for f in rep2["failures"]) and any(f.startswith("object lineage") for f in rep2["failures"])
    assert "grandfathered " in A.format_report(rep2)


def test_the_d744_live_count_sees_a_partial_beside_a_complete_of_the_same_run_and_judge(tmp_path):
    """A claim-backed partial judgment and an adapter-form complete judgment of the same (run, judge) are two rows; the live count
    names the pair, with an identity-current witness (the binding columns exist on the branch's schema)."""
    A = _load()
    root = _root(tmp_path)
    rid, sub = _seed(root)
    jid = P.claim(root, rid, synth.JUDGE_ID, "session-A", now=synth.t(500))
    partial = synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID, "status": "partial", "n_items_judged": 100,
                                     "claimed_by_session": "session-A", "claimed_utc": synth.t(500)}, synth.t(700))
    complete = synth.rec("judgment", {"judgment_id": I.judgment_id(rid, synth.JUDGE_ID), "run_id": rid, "judge_id": synth.JUDGE_ID,
                                      "status": "complete", "n_items_judged": 200}, synth.t(710))
    _write(root, sub, "000002-judgment.jsonl", [partial, complete])
    rep = _ingest(root)
    assert rep.refused == 0 and rep.blocked == 0, [str(m) for m in rep.refusals]
    out = A.run_audit(root, reconcile=False)
    d744 = out["database"]["currency_rules"]["d744_partial_displaced_by_complete"]
    assert d744 == {"partial_judgments": 1, "run_judge_pairs": 1, "witness_identity_current": True}
    jd = out["database"]["judgment"]
    assert jd["by_status"] == {"complete": 1, "partial": 1}
    assert jd["live_rows_by_identity_form"] == {"claimed_by_session": {"partial": 1}, "run_judge": {"complete": 1}}
    assert out["c1c"] == {"skipped": "--skip-reconcile"}
    assert out["log"]["judgment"]["by_form"] == {"claimed_by_session": 2, "run_judge": 1}


# --------------------------------------------------------------------------------------------
# the listing against a live harvester: a file gzipped between the listing and the read
# --------------------------------------------------------------------------------------------


def test_a_file_gzipped_after_the_listing_is_read_once_through_its_new_spelling(tmp_path):
    A = _load()
    root = _root(tmp_path)
    rid, sub = _seed(root)
    records_root = os.path.join(root, "records")
    plain = os.path.join(records_root, sub, "000001-run.jsonl")
    listed = A.audited_record_files(records_root)
    assert listed == [plain]
    # the harvester's gzip pass: the .gz written, then the plain file removed
    with open(plain, "rb") as src, gzip.GzipFile(plain + ".gz", "wb", mtime=0) as dst:
        dst.write(src.read())
    assert A.existing_spelling(plain) == plain                 # both spellings: the plain one still exists
    os.remove(plain)
    assert A.existing_spelling(plain) == plain + ".gz"          # the plain one gone: its twin
    audit = A.walk_log(records_root)
    assert audit.files_read == 1 and audit.run_records == 1 and audit.records_read == len(synth.run_records("k1", 100, through="generated")[1])
    # a listing that sees both spellings names the file once
    with gzip.open(plain + ".gz", "rb") as src, open(plain, "wb") as dst:
        dst.write(src.read())
    assert A.audited_record_files(records_root) == [plain]
    os.remove(plain + ".gz")
    os.remove(plain)
    try:
        A.existing_spelling(plain)
    except FileNotFoundError as exc:
        assert "vanished" in str(exc)
    else:
        raise AssertionError("a vanished file is a listing error, never a skip")


def test_every_file_widens_the_walk_and_reports_a_misfiled_record(tmp_path):
    """The synthetic corpus files a run's `run_state` / `run_patch` records in its `-run` file (walked by default, the file being a
    `run` file); a `judgment` written into a `-run_state` file is outside the default walk and `--every-file` names it as misfiled."""
    A = _load()
    root = _root(tmp_path)
    rid, sub = _seed(root)
    stray = synth.rec("judgment", {"judgment_id": I.judgment_id(rid, synth.JUDGE_ID), "run_id": rid, "judge_id": synth.JUDGE_ID,
                                   "status": "complete", "n_items_judged": 200}, synth.t(700))
    _write(root, sub, "000002-run_state.jsonl", [stray])
    narrow = A.run_audit(root, with_db=False)
    assert narrow["log"]["judgment"]["records"] == 0 and narrow["log"]["misfiled"] == {} and narrow["database"] == {"skipped": "--no-db"}
    wide = A.run_audit(root, with_db=False, every_file=True)
    assert wide["log"]["file_kinds"] == "every non-item-scale kind" and wide["log"]["judgment"]["records"] == 1
    # the stray judgment, and the synthetic corpus's two objects in its `-dims` file (a synth-only name: the live dimension files are
    # typed, `000007-object.jsonl.gz`, so the default walk reads every live object record)
    assert wide["log"]["misfiled"] == {"judgment": 1, "object": 2} and wide["log"]["files_read"] == 3        # dims, the run file, the stray
    assert wide["log"]["object"]["records"] == 2 and any(f.startswith("misfiled") for f in wide["failures"])


# --------------------------------------------------------------------------------------------
# the CLI: exit codes, --json, --out refused under the root, nothing written
# --------------------------------------------------------------------------------------------


def _cli(root, *args):
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"))
    return subprocess.run([PY, SCRIPT, "--root", root, *args], capture_output=True, text=True, env=env, timeout=600)


def test_the_cli_exits_by_the_verdict_prints_json_and_writes_nothing_under_the_root(tmp_path):
    root = _root(tmp_path)
    rid, sub = _seed(root)
    before = _files_under(root)
    out_path = str(tmp_path / "audit.json")
    r = _cli(root, "--json", "--out", out_path, "--skip-reconcile")
    assert r.returncode in (0, 1), r.stderr                     # 1 iff the real grandfather list is non-empty (unaccounted in a synthetic log)
    rep = json.loads(r.stdout)
    assert rep["root"] == root and rep["log"]["run"]["records"] == 1 and rep["c1c"] == {"skipped": "--skip-reconcile"}
    assert (r.returncode == 0) == rep["ok"] == (not VAL.GRANDFATHERED_RECORD_SHAS)
    if r.returncode == 1:
        assert "grandfathered_unaccounted" in r.stderr
    with open(out_path, encoding="utf-8") as fh:
        assert json.load(fh)["root"] == root
    assert _files_under(root) == before and not os.path.exists(os.path.join(root, "state", "last_run"))
    # --out under the root is refused before anything runs
    r2 = _cli(root, "--out", os.path.join(root, "audit.json"))
    assert r2.returncode == 2 and "writes nothing there" in r2.stderr and _files_under(root) == before
    # a root with no log is refused by name
    r3 = _cli(str(tmp_path / "nowhere"))
    assert r3.returncode == 2 and "no record log" in r3.stderr
    # the text rendering names the sections
    r4 = _cli(root, "--no-db")
    assert r4.returncode in (0, 1) and "database: skipped (--no-db)" in r4.stdout and "c1c buckets: skipped" in r4.stdout
