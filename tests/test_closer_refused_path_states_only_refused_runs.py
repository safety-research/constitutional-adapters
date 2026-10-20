"""A refused close states `running` for the REFUSED runs alone.

The closer writes the FULL ids of the runs it refused to a sidecar beside the pending file (<pending>.refused_run_ids.json; removed when a
close refuses nothing) and the job's exit path (common.sh f22_registry_closer_refused) states `running` with `--run-id` for exactly those —
never `--pending-dir … --job`, which re-stated `running` over the twins the same closer had just superseded (888525's two g5678 twins
folded running/current again). Synthetic ids and TESTHOST only; the real seeds for the shape.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
from test_registry_close_stores import TAG, _bon_plan_config, _bon_store, _close, _generated, _nofit_row, _records, _register, _runs, _seeded_registry, _static_store  # noqa: E402
from test_registry_close_successor_identity import _clean_env  # noqa: E402
from sisteer.registry import db as DB, emit, harvest as H  # noqa: E402

JOB = "810071"; JN = f"f22s-{TAG}-static_base_t0"
SHA = hashlib.sha1(b"refused-path tests synthetic producer sha").hexdigest()


def _goals_variants(run: dict):
    goals, variants = (int(x) for x in run["plan"]["item_set_id"].rsplit("_", 1)[1].split("x"))
    return goals, variants


def test_the_closer_lists_the_full_ids_of_exactly_the_refused_runs_and_clears_the_list_on_a_clean_close(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    root = tmp_path / "reg"; _seeded_registry(root); spool = root / "records"; wave = tmp_path / "wave"
    pending = _register(tmp_path, root, job=JOB, job_name=JN, evals="pasttense"); old = _runs(pending); assert len(old) == 2
    (ev, obj_ok), (_ev2, obj_missing) = sorted(old)
    goals, variants = _goals_variants(old[(ev, obj_ok)])
    _static_store(wave, 0, ev, obj_ok, nshards=9, goals=goals, variants=variants)          # one complete store, the other cell has none
    r = _close(pending, spool, wave, "--nshards", "9", job_name=JN)
    assert r.returncode == 2 and "1 filed, 0 skipped, 1 refused of 2 registered run(s)" in r.stdout, r.stdout + r.stderr
    side = Path(str(pending) + ".refused_run_ids.json")
    assert side.exists() and "refused-run list: 1 run id(s)" in r.stdout and "D2100 (b)" in r.stdout, r.stdout
    doc = json.loads(side.read_text())
    assert doc["refused_run_ids"] == [old[(ev, obj_missing)]["run_id"]] and doc["job_id"] == JOB and doc["decision"] == "D2100"
    assert len(doc["refused_run_ids"][0]) > 40                      # the FULL id, never the log line's 40-character prefix
    # a dry close changes nothing about the list
    r = _close(pending, spool, wave, "--nshards", "9", "--dry-run", job_name=JN)
    assert side.exists() and json.loads(side.read_text()) == doc, r.stdout
    # the missing store appears: the next close refuses nothing and the list goes
    _static_store(wave, 0, ev, obj_missing, nshards=9, goals=goals, variants=variants)
    r = _close(pending, spool, wave, "--nshards", "9", job_name=JN)
    assert r.returncode == 0 and "0 refused" in r.stdout and "removed (this close refused nothing)" in r.stdout, r.stdout + r.stderr
    assert not side.exists()


def _shell_env(tmp_path: Path, shared: Path, spool: Path, wave: Path | None = None) -> dict:
    """The RESULT hook's environment: the job's shell variables as f22_init and the sbatch leave them — the wave root the store lock is named
    from, and F22_PY = this interpreter (f22_driver_venv hands the driver venv to the trap; the refusal-note mode runs under it)."""
    env = {k: v for k, v in os.environ.items() if k not in ("F22_PY", "PYTHONPATH", "SISTEER_REGISTRY_DISABLED")}
    (tmp_path / "figs").mkdir(exist_ok=True)
    env.update({"SHARED": str(shared), "SLURM_JOB_ID": JOB, "SLURM_JOB_NAME": f"f22-att-{TAG}-g012-crescendo", "SLURMD_NODENAME": "TESTHOST",
                "REPO_DIR": str(REPO), "SISTEER_REGISTRY_SPOOL": str(spool), "FIGS_DIR": str(tmp_path / "figs"), "WAVE_SHA": SHA, "TAG": TAG, "F22_PY": sys.executable})
    if wave is not None:
        env["WAVE_ROOT"] = str(wave)
    return env


def test_the_exit_path_states_running_for_the_refused_run_alone_and_leaves_two_superseded_siblings_untouched(tmp_path, monkeypatch):
    """Two of the job's four registered runs carry a `superseded` statement (the twin retirement its own closer filed), one was
    refused, one is untouched: the exit path restates `running` on exactly the refused one; without a list it states nothing, loudly."""
    _clean_env(monkeypatch)
    root = tmp_path / "reg"; _seeded_registry(root); spool = root / "records"; shared = tmp_path / "shared"
    pending = _register(tmp_path, root, job=JOB, job_name=JN, evals="pasttense,grok420")
    ids = sorted(r["run_id"] for r in _runs(pending).values()); assert len(ids) == 4
    pdir = shared / "registry" / "pending"; pdir.mkdir(parents=True); (pdir / f"{JOB}.json").write_text(pending.read_text())
    refused, sup_a, sup_b, other = ids
    producer = emit.producer_block("test", code_sha=SHA)
    for rid in (sup_a, sup_b):
        emit.RecordWriter(str(spool), "sdfbase_full22", rid, producer).write("run_state", {
            "run_id": rid, "state": "superseded", "event_utc": "2026-10-04T17:35:09Z", "evidence_kind": "completion_artifact",
            "evidence_uri": "/s/shard00/progress.jsonl", "extra": {"superseded_by_run_id": other, "reason": "duplicate_registration", "decision": "D1816"}})
    (pdir / f"{JOB}.json.refused_run_ids.json").write_text(json.dumps({"job_id": JOB, "refused_run_ids": [refused], "decision": "D2100"}))
    before = {rid: len(_records(spool, rid)) for rid in ids}
    cmd = ["bash", "-c", f"source {REPO}/scripts/sdfbase/full22/common.sh && f22_registry_closer_refused 2 python3 'REFUSED pasttense none x: short'"]
    env = _shell_env(tmp_path, shared, spool, tmp_path / "wave")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO), timeout=300); out = r.stdout + r.stderr
    assert r.returncode == 0 and "state running: stated for 1 refused run(s) of job 810071 (1 listed; the refused-run list; D2100 (b))" in out and "applied to" not in out, out[-3000:]
    # round-3 Major on cb9ad5267: the statement goes through the closer's refusal-note mode (per run, never over a close), not the bare emitter by --run-id
    (ev_r, obj_r), = [k for k, r_ in _runs(pending).items() if r_["run_id"] == refused]
    assert f"[registry] running {ev_r} {obj_r} {refused[:40]}…: the closer's refusal note filed — no close has decided this run (without a store lock (family static))" in out, out[-3000:]
    assert f"closer refusal note: 1 listed refused run(s) of job {JOB} decided by the closer's refusal-note mode under {sys.executable}" in out and "emit_cli state" not in out, out[-3000:]
    after = {rid: len(_records(spool, rid)) for rid in ids}
    assert after[refused] > before[refused] and all(after[x] == before[x] for x in (sup_a, sup_b, other)), (before, after)
    st = [p for t, p, _pr in _records(spool, refused) if t == "run_state" and p.get("state") == "running"]
    assert st and "closer REFUSED rc=2 under python3: REFUSED pasttense none x: short" in st[-1]["note"], st
    for rid in (sup_a, sup_b):
        assert [p["state"] for t, p, _pr in _records(spool, rid) if t == "run_state"][-1] == "superseded"
    # no list: a loud line, nothing stated
    (pdir / f"{JOB}.json.refused_run_ids.json").unlink()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO), timeout=300); out = r.stdout + r.stderr
    assert r.returncode == 0 and "no refused-run list at" in out and "stating nothing" in out and "state running: wrote" not in out, out[-2000:]
    assert {rid: len(_records(spool, rid)) for rid in ids} == after


def test_a_generated_run_refused_its_nofit_certificate_is_not_listed_and_the_exit_path_states_running_for_the_running_refusal_alone(tmp_path, monkeypatch):
    """The list names the runs the exit path's `running` statement is FOR — the runs the
    closer left RUNNING. Two refusals leave a bon run `generated`: (b) the cell files `generated` in this very close and the no-fit
    CERTIFICATE is then refused because the registration's seed instrument resolves to no judge; (a) a run already `generated` (and not
    certified) whose store now fails the per-slot rule (a differently-stamped proven retry row on one slot: its emptiness is uncertified).
    Both count as refused and neither is listed — listed, the exit path would state `running` over a `generated` run (lifecycle_regresses
    holds the backward move at fold, but a misleading "closer REFUSED" running event lands on it). The run refused `running` in the same
    close (no store at all) is listed, and the exit path states `running` for it alone."""
    _clean_env(monkeypatch)
    root = tmp_path / "reg"; _seeded_registry(root); spool = root / "records"; wave = tmp_path / "wave"; shared = tmp_path / "shared"
    jn = f"f22-stat-{TAG}-3"
    pending = _register(tmp_path, root, job=JOB, job_name=jn, evals="strongreject_bon")
    runs = _runs(pending); assert set(runs) == {("strongreject_bon", "none"), ("strongreject_bon", "prompted__safety_refusal")}
    rid_gen = runs[("strongreject_bon", "none")]["run_id"]; rid_run = runs[("strongreject_bon", "prompted__safety_refusal")]["run_id"]
    goals = int(runs[("strongreject_bon", "none")]["plan"]["n_items_planned"]); cfg = _bon_plan_config(tmp_path, goals=goals, n_max=3)
    nofit = _bon_store(wave, "none", goals=goals, n_max=3, responded=0, refused=goals * 3, config=cfg)     # every slot a typed no-fit; the other cell has no store at all
    # the registration names a seed instrument this checkout's seeds do not carry: the family is still bon (the `bon_` prefix), the judge does not resolve
    doc = json.loads(pending.read_text())
    for r in doc["runs"]:
        if r["route"]["object"] == "none":
            r["plan"]["extra"]["seed_instrument_id"] = "bon_synthetic_seed_this_checkout_does_not_carry"
    pending.write_text(json.dumps(doc, indent=1, sort_keys=True))
    side = Path(str(pending) + ".refused_run_ids.json")
    # (b): filed generated, then the certificate refused for want of a judge — in the refused COUNT, not in the list
    r = _close(pending, spool, wave, job_name=jn)
    assert r.returncode == 2 and "1 filed, 0 skipped, 2 refused of 2 registered run(s)" in r.stdout, r.stdout + r.stderr
    assert f"REFUSED nofit certificate strongreject_bon none {rid_gen[:40]}" in r.stdout and "names no judge_id in instruments.yaml" in r.stdout and "the run stays generated" in r.stdout, r.stdout
    assert f"REFUSED strongreject_bon prompted__safety_refusal {rid_run[:40]}" in r.stdout and "no store directory — the run stays running" in r.stdout, r.stdout
    st, pa = _generated(spool, rid_gen); assert st is not None and pa["n_items_generated"] == 0 and pa["store_uri"] == str(nofit)
    assert _generated(spool, rid_run) == (None, None)
    assert side.exists() and "refused-run list: 1 run id(s)" in r.stdout and json.loads(side.read_text())["refused_run_ids"] == [rid_run], r.stdout
    assert H.ingest_once(str(root), workers=0, min_free=0).refused == 0
    with DB.reading(DB.db_path(str(root))) as conn:
        assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid_gen,)).fetchone()[0] == "generated"
        assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid_run,)).fetchone()[0] == "running"
    # (a) on a run ALREADY generated: a proven retry row under another window on one slot — two geometry stamps, the slot UNDECIDED under the
    # per-slot rule, the certificate refused; the run stays generated — in the refused COUNT, not in the list
    with open(nofit / "bon_errors.jsonl", "a") as fh:
        fh.write(json.dumps(_nofit_row("jbb_harmful_0", 1, 0, effective_window=6000, window=6000, server_window=6000)) + "\n")
    r = _close(pending, spool, wave, job_name=jn)
    assert r.returncode == 2 and "0 filed, 0 skipped, 2 refused of 2 registered run(s)" in r.stdout, r.stdout + r.stderr
    assert f"REFUSED strongreject_bon none {rid_gen[:40]}" in r.stdout and "1 of " in r.stdout and "UNDECIDED under the per-slot rule" in r.stdout, r.stdout
    assert "first jbb_harmful_0/1: 2 geometry stamps" in r.stdout and "— the run stays generated" in r.stdout and "REFUSED nofit certificate" not in r.stdout, r.stdout
    assert json.loads(side.read_text())["refused_run_ids"] == [rid_run] and _generated(spool, rid_gen)[0] == st
    # the exit path over that list: `running` stated for the running refusal alone; the generated run gets no record at all
    pdir = shared / "registry" / "pending"; pdir.mkdir(parents=True)
    (pdir / f"{JOB}.json").write_text(pending.read_text()); (pdir / f"{JOB}.json.refused_run_ids.json").write_text(side.read_text())
    before = {rid: len(_records(spool, rid)) for rid in (rid_gen, rid_run)}
    none_lock_before = (nofit.parent / ".none.close.lock").read_text()           # the generated run's lock, as its CLOSE above left it (holder + time)
    cmd = ["bash", "-c", f"source {REPO}/scripts/sdfbase/full22/common.sh && f22_registry_closer_refused 2 python3 'REFUSED strongreject_bon prompted__safety_refusal x: no store directory'"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=_shell_env(tmp_path, shared, spool, wave), cwd=str(REPO), timeout=300); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"state running: stated for 1 refused run(s) of job {JOB} (1 listed; the refused-run list; D2100 (b))" in out, out[-3000:]
    # round-3 Major on cb9ad5267: a BoN run's note is decided under its store close lock; round-4 Major on 08c90e1cf: the lock lives BESIDE the store
    # (<private run>/.<cell>.close.lock), so the refused cell with no store directory is decided under its lock all the same — the lock file appears
    # beside the absent store and the store stays absent; the generated run is not listed, so its lock is never taken
    assert (f"[registry] running strongreject_bon prompted__safety_refusal {rid_run[:40]}…: the closer's refusal note filed — no close has decided this run "
            "(under the store close lock)") in out and "emit_cli state" not in out and "without a store lock" not in out, out[-3000:]
    assert (nofit.parent / ".prompted__safety_refusal.close.lock").is_file() and not (nofit.parent / "prompted__safety_refusal").exists()
    assert (nofit.parent / ".none.close.lock").read_text() == none_lock_before    # never taken by the note: the holder line is the close's still
    after = {rid: len(_records(spool, rid)) for rid in (rid_gen, rid_run)}
    assert after[rid_run] > before[rid_run] and after[rid_gen] == before[rid_gen], (before, after)
    stated = [p for t, p, _pr in _records(spool, rid_run) if t == "run_state" and p.get("state") == "running"]
    assert stated and "closer REFUSED rc=2 under python3" in stated[-1]["note"], stated
    assert not [p for t, p, _pr in _records(spool, rid_gen) if t == "run_state" and "closer REFUSED" in str(p.get("note") or "")]
    assert [p["state"] for t, p, _pr in _records(spool, rid_gen) if t == "run_state"][-1] == "generated"
