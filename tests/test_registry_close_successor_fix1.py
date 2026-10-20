"""The successor relabel and the grading carrier, pinned.

(a) a FAILED attempt's store-less registration seeds no successor; (b) every retired run's store is validated BEFORE the first
successor record is emitted (one bad store, nothing written for any); (c) --objects is always explicit; (d) the successor states
the PRODUCING code_sha read from the retired run's record, never the relabel checkout's; (e) the carrier verifies every expected
record type on the successor, replays only the missing ones and gates "carried" on the push rc; (f) only the COMPLETE judgment's
rows travel; (h) the closer's dry-mode summary says what it WOULD file. Synthetic ids and TESTHOST only; the real seeds for the shape.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
import test_registry_carry_grading as cg  # noqa: E402
from test_registry_close_stores import CLOSER, OBJECTS, TAG, _close, _generated, _records, _register, _runs, _seeded_registry  # noqa: E402
from test_registry_close_successor_identity import _clean_env, _full_stores, _run_record, _sampled_cfg, _states  # noqa: E402
from sisteer.registry import emit, identity as I, synth  # noqa: E402

PROD_SHA = hashlib.sha1(b"the producing tree (U0STATICSUCC-fix-1 tests)").hexdigest()   # a non-placeholder synthetic sha
JOB = "808182"; JN = f"f22s-{TAG}-static_base_t0"
SUCC = ("--nshards", "9", "--successor-token-cap", "4096", "--successor-pass-label", "base_cap")


def _setup(tmp_path, monkeypatch, *, code_sha=None):
    _clean_env(monkeypatch)
    root = tmp_path / "reg"; _seeded_registry(root); spool = root / "records"; wave = tmp_path / "wave"
    pending = _register(tmp_path, root, job=JOB, job_name=JN, evals="pasttense", code_sha=code_sha)
    old = _runs(pending); assert len(old) == 2
    _full_stores(wave, old)
    return root, spool, wave, pending, old


def _files(spool: Path) -> list:
    return sorted(str(p.relative_to(spool)) for p in spool.rglob("*") if p.is_file())


def test_a_failed_attempts_store_less_registration_seeds_no_successor(tmp_path, monkeypatch):
    """(a): the FAILED attempt registered the same cells and never closed — no store_uri, no completion artifact — so a relabel run
    against ITS pending file refuses by name and writes nothing (no successor run, no pairs file, no successor pending file)."""
    root, spool, wave, pending, old = _setup(tmp_path, monkeypatch)
    orphan = _register(tmp_path, root, job="805197", job_name=JN, evals="pasttense")
    assert _close(pending, spool, wave, "--nshards", "9", job_name=JN).returncode == 0          # the producing attempt files its store
    cfg = _sampled_cfg(tmp_path, "805197", 0); pairs = tmp_path / "pairs.tsv"; before = _files(spool)
    r = _close(orphan, spool, wave, *SUCC, "--successor-config", str(cfg), "--objects", ",".join(OBJECTS), "--pairs-out", str(pairs), job_name=JN)
    assert r.returncode == 2 and "filed no store_uri / completion artifact" in r.stdout and "D2084 (a)" in r.stdout and "nothing registered" in r.stdout, r.stdout + r.stderr
    assert _files(spool) == before and not pairs.exists() and not (orphan.parent / "succ" / orphan.name).exists()
    for o in _runs(orphan).values():
        assert _states(spool, o["run_id"], "superseded") == [] and _generated(spool, o["run_id"]) == (None, None)


def test_one_incomplete_store_refuses_before_any_successor_is_emitted(tmp_path, monkeypatch):
    """(b): the retired runs FILED their stores, but one store is no longer complete when the relabel runs — every retired run is checked
    first, so NOTHING is emitted for either cell (no successor run record, no superseded state, no successor pending file)."""
    root, spool, wave, pending, old = _setup(tmp_path, monkeypatch)
    r0 = _close(pending, spool, wave, "--nshards", "9", job_name=JN); assert r0.returncode == 0 and "2 filed" in r0.stdout, r0.stdout
    ev, obj = sorted(old)[1]
    shutil.rmtree(wave / "transcripts_results" / f"sdfbase_full22_{TAG}_sampled_t0" / ev / obj / "shard03")
    cfg = _sampled_cfg(tmp_path, JOB, 0); pairs = tmp_path / "pairs.tsv"; before = _files(spool)
    r = _close(pending, spool, wave, *SUCC, "--successor-config", str(cfg), "--objects", ",".join(OBJECTS), "--pairs-out", str(pairs), job_name=JN)
    assert r.returncode == 2 and "is not complete" in r.stdout and "D2084 (b)" in r.stdout and "nothing registered" in r.stdout, r.stdout + r.stderr
    assert "successor registered" not in r.stdout and "superseded now" not in r.stdout
    assert _files(spool) == before and not pairs.exists() and not (pending.parent / "succ" / pending.name).exists()
    for o in old.values():
        assert _states(spool, o["run_id"], "superseded") == []


def test_the_relabel_names_its_objects_and_its_wave_root_or_refuses(tmp_path, monkeypatch):
    """(c): the source set is stated, never the pending file's whole cell set (a steered view's legacy `none` rows are not relabelled);
    and (a)/(b) need the wave root to verify the stores against — both refusals name the rule and write nothing."""
    root, spool, wave, pending, old = _setup(tmp_path, monkeypatch)
    assert _close(pending, spool, wave, "--nshards", "9", job_name=JN).returncode == 0
    cfg = _sampled_cfg(tmp_path, JOB, 0); before = _files(spool)
    r = _close(pending, spool, wave, *SUCC, "--successor-config", str(cfg), job_name=JN)
    assert r.returncode == 2 and "name --objects explicitly" in r.stdout and "D2084 (c)" in r.stdout, r.stdout + r.stderr
    env = dict(os.environ, PYTHONPATH=str(REPO / "src")); env.pop("SISTEER_REGISTRY_DISABLED", None)
    r2 = subprocess.run([sys.executable, str(CLOSER), "--pending", str(pending), "--tag", TAG, "--spool", str(spool), "--code-sha", "b" * 40, *SUCC,
                         "--successor-config", str(cfg), "--objects", ",".join(OBJECTS), "--job-name", JN], capture_output=True, text=True, env=env, timeout=300)
    assert r2.returncode == 2 and "needs --wave-root" in r2.stdout, r2.stdout + r2.stderr
    assert _files(spool) == before and not (pending.parent / "succ" / pending.name).exists()


def test_the_successor_states_the_producing_code_sha_never_the_relabel_checkouts(tmp_path, monkeypatch):
    """(d): the retired run was registered by the PRODUCING tree (its `run` record's code_sha); the relabel runs from another checkout
    (`--code-sha` synth.code_sha("b") here) — the successor's plan, its spooled run record and its lineage all name the producing sha."""
    root, spool, wave, pending, old = _setup(tmp_path, monkeypatch, code_sha=PROD_SHA)
    for o in old.values():
        assert _run_record(spool, o["run_id"])["code_sha"] == PROD_SHA
    assert _close(pending, spool, wave, "--nshards", "9", job_name=JN).returncode == 0
    cfg = _sampled_cfg(tmp_path, JOB, 0); pairs = tmp_path / "pairs.tsv"
    r = _close(pending, spool, wave, *SUCC, "--successor-config", str(cfg), "--objects", ",".join(OBJECTS), "--pairs-out", str(pairs), job_name=JN)
    assert r.returncode == 0 and "2 pair(s), 2 superseded now, 2 successor(s) closed now" in r.stdout and f"code_sha {PROD_SHA[:12]}" in r.stdout, r.stdout + r.stderr
    new = _runs(pending.parent / "succ" / pending.name); assert set(new) == set(old)
    for (ev, obj), n in new.items():
        assert n["plan"]["code_sha"] == PROD_SHA and n["plan"]["extra"]["producing_code_sha"] == PROD_SHA and n["plan"]["extra"]["successor_of"] == old[(ev, obj)]["run_id"]
        rr = _run_record(spool, n["run_id"])
        assert rr["code_sha"] == PROD_SHA and rr["code_sha"] != synth.code_sha("b") and rr["cluster_job_id"] == JOB and rr["lifecycle_state"] == "running"
        (sup,) = _states(spool, old[(ev, obj)]["run_id"], "superseded"); assert sup["extra"]["superseded_by_run_id"] == n["run_id"]


def test_the_dry_close_says_what_it_would_file_and_writes_nothing(tmp_path, monkeypatch):
    """(h): a --dry-run close reports `would file` counts, never a `filed` summary; the spool is untouched."""
    root, spool, wave, pending, old = _setup(tmp_path, monkeypatch)
    before = _files(spool)
    r = _close(pending, spool, wave, "--nshards", "9", "--dry-run", job_name=JN)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "close (DRY RUN — nothing written): would file 2, skip 0, refuse 0 of 2 registered run(s)" in r.stdout, r.stdout
    assert "close: 2 filed" not in r.stdout and _files(spool) == before
    for o in old.values():
        assert _generated(spool, o["run_id"]) == (None, None)
    r = _close(pending, spool, wave, "--nshards", "9", job_name=JN)
    assert r.returncode == 0 and "close: 2 filed, 0 skipped, 0 refused of 2 registered run(s)" in r.stdout and "DRY RUN" not in r.stdout, r.stdout


# --------------------------------------------------------------------------------------------------------- the carrier (e) / (f)
ABANDONED_JID = "e7ae0d59-c95e-5679-8868-000000000009"


def _judgment_payload(root: Path) -> dict:
    (jp,) = [p for t, p in cg._records(root / cg.CAMPAIGN / I.run_dir_name(cg.OLD)) if t == "judgment"]
    return jp


def _run_with_push(pairs: Path, root: Path, spool: Path, push: Path, *extra: str) -> subprocess.CompletedProcess:
    env = {"PATH": os.environ["PATH"], "HOME": str(root.parent), "PYTHONPATH": str(REPO / "src"), emit.PUSH_SCRIPT_ENV: str(push)}
    return subprocess.run([sys.executable, str(cg.CARRIER), "--pairs", str(pairs), "--records-root", str(root), "--spool", str(spool), "--campaign", cg.CAMPAIGN,
                           "--code-sha", "d" * 40, *extra], capture_output=True, text=True, env=env, timeout=120)


def test_only_the_complete_judgments_rows_travel_a_partial_successor_is_replayed_and_a_failed_push_is_not_carried(tmp_path):
    root = tmp_path / "records"; spool = tmp_path / "spool"
    d = cg._old_run_dir(root)
    # (f): an ABANDONED earlier attempt on the same retired run — its own judgment id and three item rows bound to it — stays behind
    jp = _judgment_payload(root)
    cg._write(d, 3, "judgment", [dict(jp, judgment_id=ABANDONED_JID, status="abandoned", n_items_judged=0)])
    cg._write(d, 4, "item_result", [{"run_id": cg.OLD, "judgment_id": ABANDONED_JID, "item_id": f"g{i}", "metric_id": "strongreject_score", "trial_index": 0, "sample_index": 0,
                                     "turn_index": 0, "value": 0.1, "label": "x", "judged_utc": "2026-09-30T19:00:00Z", "created_utc": "2026-09-30T19:00:00Z", "api_pool": "lowprio",
                                     "extra": {"run_overrides": {"id": "synthetic_v1"}}} for i in range(3)], gz=True)
    cg._new_run_dir(spool)
    pairs = tmp_path / "pairs.tsv"; pairs.write_text("# header\n" + f"{cg.OLD}\t{cg.NEW}\tpasttense\tnone\tt\t0\t808182\tkey\n")
    r = cg._run(pairs, root, spool, "--write")
    assert r.returncode == 0 and "carried" in r.stderr and "REPLAY" not in r.stderr, r.stderr
    new_jid = I.judgment_id(cg.NEW, cg.JUDGE)
    recs = cg._records(spool / cg.CAMPAIGN / I.run_dir_name(cg.NEW))
    items = [p for t, p in recs if t == "item_result"]; judgments = [p for t, p in recs if t == "judgment"]
    assert len(items) == 3 and all(p["judgment_id"] == new_jid and p["value"] != 0.1 for p in items), items
    assert len(judgments) == 1 and judgments[0]["judgment_id"] == new_jid and judgments[0]["status"] == "complete" and judgments[0]["extra"]["carried_from_judgment_id"] == cg.OLD_JID
    assert sum(1 for t, _p in recs if t == "run_metric") == 2 and sum(1 for t, _p in recs if t == "run_degeneration") == 1
    # (e): a successor left with the judgment and NO rows (a write interrupted after the judgment file) is not "idempotent" — the missing
    # types are replayed, only them, and the carried lifecycle records name their origin
    spool2 = tmp_path / "spool2"; d2 = cg._new_run_dir(spool2)
    cg._write(d2, 2, "judgment", [judgments[0]])
    r2 = cg._run(pairs, root, spool2, "--write")
    assert r2.returncode == 0 and "replayed" in r2.stderr and "REPLAY of the missing" not in r2.stderr, r2.stderr
    recs2 = cg._records(d2); by = {}
    for t, p in recs2:
        by.setdefault(t, []).append(p)
    assert len(by["judgment"]) == 1 and len(by["item_result"]) == 3 and len(by["run_metric"]) == 2 and len(by["run_degeneration"]) == 1
    assert [p["state"] for p in by["run_state"]] == ["judged"] and by["run_state"][0]["extra"]["carried_from_run_id"] == cg.OLD
    assert by["run_patch"][0]["extra"]["carried_from_run_id"] == cg.OLD
    dry = cg._run(pairs, root, spool2)
    assert dry.returncode == 0 and "idempotent; every expected record type verified" in dry.stderr, dry.stderr
    # a partially carried successor is described as such in a dry run
    spool3 = tmp_path / "spool3"; d3 = cg._new_run_dir(spool3); cg._write(d3, 2, "judgment", [judgments[0]])
    dry3 = cg._run(pairs, root, spool3)
    assert dry3.returncode == 0 and "REPLAY of the missing" in dry3.stderr and "item_result" in dry3.stderr, dry3.stderr
    # the push gate: a push script that fails leaves the successor NOT carried, by name, with the records spooled for a re-run
    spool4 = tmp_path / "spool4"; cg._new_run_dir(spool4)
    push = tmp_path / "push_fails.sh"; push.write_text("#!/bin/bash\nexit 3\n"); push.chmod(0o755)
    r4 = _run_with_push(pairs, root, spool4, push, "--write")
    assert "REFUSED" in r4.stderr and "push returned rc=3" in r4.stderr and "NOT carried" in r4.stderr, r4.stderr
    assert r4.returncode != 0
