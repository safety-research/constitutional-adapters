"""SERVINGJOBSREG (2026-10-04): a job states the serving jobs of its store at job start.

* `emit_cli register --from-config … --serving-jobs <id,…>` writes `extra.serving_jobs` (sorted, de-duplicated, the registering job's own
  id added) and `extra.serving_jobs_source = "funnel plan D1918"` on every run it registers; a non-numeric id refuses by name and writes
  nothing; no flag = no key (the SERVINGJOBS backfill pass fills siblings unknown at registration).
* common.sh f22_registry_register passes F22_REGISTRY_SERVING_JOBS through as that flag — for a multi-turn TARGET (F22_REGISTRY_MULTITURN_TARGET=1)
  in every case, an empty list stating the job's own id alone.
* reproduce/lib/submit_twin.py `sibling_serving_jobs(subs_ledger, lwid)`: a multi-turn TARGET unit's sibling shard groups' LIVE bindings
  (submitted | registered; never a transition, an unbound or a torn one; never another unit's), and submit_one hands them to the job in
  the job-side environment (the seam, outside the workload spec identity).

Synthetic job ids and bindings; the REAL registration over the checkout's seeds; no cluster, no real spool."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, run_f22_registration, synthetic_checkout  # noqa: E402

sys.path.insert(0, str(REPO / "reproduce" / "lib"))
import stage as ST  # noqa: E402
import submit_twin as SW  # noqa: E402

COMMON = REPO / "scripts/sdfbase/full22/common.sh"


def _checkout(root: Path) -> Path:
    cfg = yaml.safe_dump({"explicit_cells": ["none", "prompted__si_constitution"], "model": "Qwen/Qwen3.5-4B"})
    roster = yaml.safe_dump({"targets": {"qwen35_4b": {"views": {"main": {"cells": {"prompted__si_constitution": None}}}}}})
    return synthetic_checkout(root, configs={"configs/sdfbase_full22_qwen35_4b.yaml": cfg, "configs/sdfbase_full22_roster.yaml": roster,
                                             "configs/sdfbase_full22_pins.json": "{}"})


def _extras(shared: Path, job: str) -> dict:
    doc = json.loads((shared / "registry" / "pending" / f"{job}.json").read_text())
    return {r["route"]["object"]: (r["plan"].get("extra") or {}) for r in doc["runs"]}


def test_the_registration_states_the_serving_jobs_sorted_with_its_own_id_and_nothing_without_the_flag(tmp_path, monkeypatch):
    co = _checkout(tmp_path / "a")
    for s in ("shared1", "shared2"):
        (tmp_path / s).mkdir()
    monkeypatch.setenv("F22_REGISTRY_SERVING_JOBS", "8888,7777,8888")                # the funnel's hand-over: two siblings, one repeated
    r = run_f22_registration(co, tmp_path / "shared1", key="qwen35_4b", view="main", cfg_rel="configs/sdfbase_full22_qwen35_4b.yaml",
                             job_id="4242", job_name="f22-att-qwen35_4b-g01234-crescendo", sha="9c4fd67577dfc6df5e75fc5dd63d2882bba59d7b", evals="xstest")
    assert r.returncode == 0 and "REGISTERED pending=" in r.stdout, r.stdout + r.stderr
    ex = _extras(tmp_path / "shared1", "4242")
    assert set(ex) == {"none", "prompted__si_constitution"}
    for obj, e in ex.items():
        assert e["serving_jobs"] == ["4242", "7777", "8888"] and e["serving_jobs_source"] == "funnel plan D1918", (obj, e)   # sorted, de-duplicated, own id in
        assert e.get("registered_at") == "job_start"                                                                       # the stamp's keys ride beside
    # servingjobsreg-fix-1: the FIRST group of a pass has no live sibling yet — a multi-turn TARGET (the funnel's marker) still states
    # its own id, through the real registration
    monkeypatch.delenv("F22_REGISTRY_SERVING_JOBS", raising=False); monkeypatch.setenv("F22_REGISTRY_MULTITURN_TARGET", "1")
    r = run_f22_registration(co, tmp_path / "shared2", key="qwen35_4b", view="main", cfg_rel="configs/sdfbase_full22_qwen35_4b.yaml",
                             job_id="4343", job_name="f22-att-qwen35_4b-g5678-crescendo", sha="9c4fd67577dfc6df5e75fc5dd63d2882bba59d7b", evals="xstest")
    assert r.returncode == 0 and "REGISTERED pending=" in r.stdout, r.stdout + r.stderr
    for obj, e in _extras(tmp_path / "shared2", "4343").items():
        assert e["serving_jobs"] == ["4343"] and e["serving_jobs_source"] == "funnel plan D1918", (obj, e)
    # without the marker and with no list handed over (every other job kind's env): the key stays absent (unchanged behaviour)
    monkeypatch.delenv("F22_REGISTRY_MULTITURN_TARGET", raising=False); (tmp_path / "shared3").mkdir()
    r = run_f22_registration(co, tmp_path / "shared3", key="qwen35_4b", view="main", cfg_rel="configs/sdfbase_full22_qwen35_4b.yaml",
                             job_id="4444", job_name="f22-att-qwen35_4b-g5678-pair", sha="9c4fd67577dfc6df5e75fc5dd63d2882bba59d7b", evals="xstest")   # another attack: its own run ids (282's job-independent key)
    assert r.returncode == 0, r.stdout + r.stderr
    assert all("serving_jobs" not in e and "serving_jobs_source" not in e for e in _extras(tmp_path / "shared3", "4444").values())


def test_a_non_numeric_serving_job_refuses_by_name_and_writes_nothing(tmp_path, monkeypatch):
    co = _checkout(tmp_path / "a"); (tmp_path / "shared").mkdir()
    monkeypatch.setenv("F22_REGISTRY_SERVING_JOBS", "7777,not-a-job")
    r = run_f22_registration(co, tmp_path / "shared", key="qwen35_4b", view="main", cfg_rel="configs/sdfbase_full22_qwen35_4b.yaml",
                             job_id="4242", job_name="f22-att-qwen35_4b-g01234-crescendo", sha="9c4fd67577dfc6df5e75fc5dd63d2882bba59d7b", evals="xstest")
    out = r.stdout + r.stderr
    assert "self-registration REFUSED" in out and "non-numeric job id" in out and "not-a-job" in out, out
    assert not (tmp_path / "shared" / "registry" / "pending" / "4242.json").exists()


def test_the_register_block_passes_the_env_through_and_submit_one_hands_the_siblings_to_the_job():
    src = COMMON.read_text()
    body = src[src.index("f22_registry_register(){"):]; body = body[:body.index("\n}\n")]
    assert 'if [ "${F22_REGISTRY_MULTITURN_TARGET:-}" = 1 ]; then rung_flags+=(--serving-jobs "${F22_REGISTRY_SERVING_JOBS:-}")' in body       # Always, for a multi-turn target
    assert 'elif [ -n "${F22_REGISTRY_SERVING_JOBS:-}" ]; then rung_flags+=(--serving-jobs "$F22_REGISTRY_SERVING_JOBS"); fi' in body         # other kinds: only when handed a list
    sw = (REPO / "reproduce" / "lib" / "submit_twin.py").read_text()
    i = sw.index('if kind == "target" and "/multiturn/" in lwid:')
    assert sw.index('full_env["F22_REGISTRY_MULTITURN_TARGET"] = "1"') > i                                                  # The marker, every multi-turn target
    assert sw.index('full_env["F22_REGISTRY_SERVING_JOBS"] = ",".join(sibs)') > sw.index("sibs = sibling_serving_jobs(self.subs.path, lwid)") and "if sibs:" not in sw[i:i + 900]
    assert sw.index("sibs = sibling_serving_jobs(self.subs.path, lwid)") > i > sw.index("full_env.update(job_env)")       # after the job-side facts, before the prefix
    assert i < sw.index('prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in full_env.items())')
    cli = (REPO / "src" / "sisteer" / "registry" / "emit_cli.py").read_text()
    assert 'r.add_argument("--serving-jobs"' in cli and 'extra["serving_jobs_source"] = "funnel plan D1918"' in cli


def _bind(subs: str, lwid: str, job_id: str, state: str = "submitted") -> None:
    d = ST.bindings_dir(subs); os.makedirs(d, exist_ok=True)
    rec = {"logical_work_id": lwid, "job_id": job_id, "cluster": "cluster_a", "attempt_id": f"{lwid}@cluster_a@1", "state": state, "utc": "2026-10-04T00:00:00Z"}
    Path(d, f"{ST.safe_id(lwid)}.{state}.json").write_text(json.dumps(rec))


def test_sibling_serving_jobs_names_the_live_sibling_groups_only(tmp_path):
    subs = str(tmp_path / "state" / "submissions.jsonl"); os.makedirs(os.path.dirname(subs)); Path(subs).write_text("")
    me = "q/s2rl/multiturn/crescendo/g01234"; sib = "q/s2rl/multiturn/crescendo/g5678"
    assert SW.sibling_serving_jobs(subs, me) == []                                    # the first group of a pass: no sibling bound yet
    _bind(subs, me, "900001")
    assert SW.sibling_serving_jobs(subs, me) == []                                    # its own binding is not a sibling
    _bind(subs, sib, "900002")
    assert SW.sibling_serving_jobs(subs, me) == ["900002"] and SW.sibling_serving_jobs(subs, sib) == ["900001"]   # the second states the first's job
    _bind(subs, sib, "900002", state="registered")
    assert SW.sibling_serving_jobs(subs, me) == ["900002"]                            # registered counts once
    _bind(subs, "q/s2rl/multiturn/pair/g5678", "900003"); _bind(subs, "q/s2r/multiturn/crescendo/g5678", "900004")
    _bind(subs, "q/s2rl/static/t0", "900005")
    assert SW.sibling_serving_jobs(subs, me) == ["900002"]                            # another attack, another view, another family: not siblings
    ST.write_transition(subs, {"logical_work_id": sib, "nonce": "n", "utc": "2026-10-04T00:01:00Z", "intended_attempt": 2})
    assert SW.sibling_serving_jobs(subs, me) == []                                    # a sibling mid re-run states nothing
    ST.clear_transition(subs, sib)
    assert SW.sibling_serving_jobs(subs, me) == ["900002"]
    assert SW.sibling_serving_jobs(subs, "q/s2rl/static/t0") == [] and SW.sibling_serving_jobs(subs, "not/a/unit") == []   # non-multi-turn units state nothing
