"""scripts/sdfbase/full22/petri_registry_handoff.py — the WRITE side of the native Petri hand-off: the
proxy job registers its (eval, object) runs on the cluster; the grading-host drive imports those identities (the wave config's cell
entries re-spelled to the Petri harness cell ids the stores name — registry_adapter.harness_cell_id), exports them to the runner's
CellStore whose default recorder routes every row onto the registered run of ITS (object, behaviour) and files `store_uri` +
`generated` at results.json, files the POOLED run per cell itself at the cell-done marker (`close-leg`), and pushes at the end
(`finish`). Run against a REAL job-start registration (emit_cli register --from-config over the real seeds), a REAL CellStore built
the way petri_run.py builds it, and the real spool records. Synthetic checkout, TESTHOST-free; no real store or session."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, VENV_PY, fold_spool_runs, read_spool, registry_emitting_env  # noqa: E402

from sisteer.capeval import store as cs  # noqa: E402
from sisteer.registry import emit, emit_cli, hooks  # noqa: E402
from sisteer.registry import identity as RI  # noqa: E402
from sisteer.registry import synth

HANDOFF = REPO / "scripts" / "sdfbase" / "full22" / "petri_registry_handoff.py"
DRIVE = REPO / "scripts" / "sdfbase" / "full22" / "petri_drive_full22.sh"
SEEDS = REPO / "configs" / "registry"
PETRI_EVALS = "petri_pooled,petri_cyber_bio_chem_capabilities,petri_self_preservation,petri_delusion_sycophancy,petri_defer_to_users,petri_reasoning_unfaithfulness,petri_instruction_evasion"
BEHAVIORS = ("cyber_bio_chem_capabilities", "self_preservation", "delusion_sycophancy", "defer_to_users", "reasoning_unfaithfulness", "instruction_evasion")
REV = "0123456789abcdef0123456789abcdef01234567"
JOB = "4242"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in (emit.DISABLED_ENV, emit.SPOOL_ENV, emit.PUSH_SCRIPT_ENV, hooks.EMIT_ENV, hooks.RUN_IDENTITY_ENV, "SLURM_JOB_ID", "SLURM_JOB_NAME"):
        monkeypatch.delenv(k, raising=False)
    hooks._DEFAULT_NOTICED.clear()


def _register(tmp_path: Path, objects=("none", "prompted__safety_refusal", "prompted__si_constitution"), model_key="qwen35_4b", job=JOB) -> dict:
    """The proxy job's REAL job-start registration over the real seeds: a synthetic checkout whose wave config declares `objects`."""
    root = tmp_path / "checkout"; (root / "configs").mkdir(parents=True)
    os.symlink(SEEDS, root / "configs" / "registry")
    cfg = root / "configs" / f"sdfbase_full22_{model_key}.yaml"; cfg.write_text(yaml.safe_dump({"explicit_cells": list(objects)}))
    spool = tmp_path / "cluster_spool"; pending = tmp_path / "cluster_pending"
    rc = emit_cli.main(["register", "--from-config", str(cfg), "--model-key", model_key, "--evals", PETRI_EVALS, "--serving-path", "lora_native",
                        "--campaign", "sdfbase_full22", "--job-id", job, "--job-name", f"f22-petri-{model_key}", "--pending-dir", str(pending),
                        "--spool", str(spool), "--code-sha", synth.code_sha("a"), "--substrate", "cluster_a", "--no-push"])
    assert rc == 0
    pend = pending / f"{job}.json"
    assert pend.is_file() and len(json.loads(pend.read_text())["runs"]) == 7 * len(objects)
    return {"pending": pend, "cluster_spool": spool, "checkout": root}


def _handoff(*args, env=None, cwd=None) -> subprocess.CompletedProcess:
    e = registry_emitting_env(); e.update(env or {})
    return subprocess.run([VENV_PY, str(HANDOFF), *args], capture_output=True, text=True, env=e, timeout=300, cwd=str(cwd or REPO))


def _import(tmp_path: Path, reg: dict, spool: Path, job=JOB) -> Path:
    out = tmp_path / "drive_state" / "registry" / "pending"
    r = _handoff("--spool", str(spool), "import", "--pending", str(reg["pending"]), "--job", job, "--out-dir", str(out), "--sh")
    assert r.returncode == 0, r.stdout + r.stderr
    exports = dict(ln[len("export "):].split("=", 1) for ln in r.stdout.splitlines() if ln.startswith("export "))
    assert exports[hooks.RUN_IDENTITY_ENV].strip("'") == str(out / f"{job}.json") and exports[emit.SPOOL_ENV].strip("'") == str(spool)
    return out / f"{job}.json"


def _petri_store(out_root: Path, run_key: str, cell_id: str, kind: str, ids_by_behavior: dict, *, identity: Path, spool: Path, complete=True) -> Path:
    """A REAL CellStore built the way scripts/petri_run.py builds one (CellStore(out_root, petri_run_key, row_constants=…), the manifest
    with cell_id, the runner's row template: task petri/<b>, item_id = scenario, the three required metrics), driven under the
    drive's exported identity + spool in a fresh interpreter (the default recorder resolves there); results.json only when complete."""
    script = r'''
import json, os, sys, time
from sisteer.capeval.store import CellStore, ResultRow
out_root, run_key, cell_id, kind, ids, complete, job = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], json.loads(sys.argv[5]), sys.argv[6] == "1", sys.argv[7]
const = {"sweep": os.path.basename(out_root), "run_id": f"{os.path.basename(out_root)}.{cell_id}.{run_key[:8]}", "model_id": "Qwen/Qwen3.5-4B",
         "model_revision": "0123456789abcdef0123456789abcdef01234567", "arm_name": cell_id, "kind": kind, "coeff": None,
         "intervention_hash": "2ff9cdff091aef6f", "engine": "proxy_vllm+lora+interleaved", "trial_id": 0, "seed": 0, "attempt": 0,
         "harness_config_hash": "1602b0fd44db6681", "env_digest": "ba9205dffa5ce785", "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
store = CellStore(out_root, run_key, row_constants=const)
resolved = {**const, "row_constants": dict(const), "cell_id": cell_id, "petri_run_key": run_key, "virtual_model_name": f"petri__{cell_id}",
            "scenario_bundle_hash": "8812ba083bb5473c", "resolved_at": time.time()}
store.validate_resume(resolved); store.write_cell_manifest(resolved)
n = 0
for b, sids in ids.items():
    for sid in sids:
        for metric in (b, "eval_awareness", "scenario_realism"):
            store.append_row(ResultRow(**const, tier=-1, task=f"petri/{b}", task_version="8812ba083bb5473c", domain="petri_behavior", item_id=sid,
                                       position_order=None, metric=metric, value=4.0, extra_json=json.dumps({"scenario_id": sid, "epoch": 1, "dimension": metric}),
                                       created_at=time.time(), wall_clock_s=None)); n += 1
cell = store.cell_dir
(cell / "petri_incomplete.json").write_text(json.dumps({"version": 1, "cell_id": cell_id, "petri_run_key": run_key, "unresolved": [] if complete else
    [{"behavior": b, "scenario_id": sids[0], "epoch": 1, "dimension": b, "reason": "no result", "retries": 0} for b, sids in list(ids.items())[:1]]}))
if complete:
    n_items = sum(len(s) for s in ids.values())
    store.write_results({"cell_id": cell_id, "petri_run_key": run_key, "rows_emitted": n, "unscored": 0, "completed_at": time.time(),
                         "audited": {"expected": n_items, "items": n_items},
                         "behaviors": {b: {"n_items": len(s), "n_scenarios": len(s), "incomplete": False} for b, s in ids.items()}})
print("STORE", cell)
'''
    env = registry_emitting_env(PYTHONPATH=str(REPO / "src"), **{hooks.RUN_IDENTITY_ENV: str(identity), emit.SPOOL_ENV: str(spool), "SLURM_JOB_ID": JOB})
    r = subprocess.run([VENV_PY, "-c", script, str(out_root), run_key, cell_id, kind, json.dumps(ids_by_behavior), "1" if complete else "0", JOB],
                       capture_output=True, text=True, env=env, timeout=300, cwd=str(REPO))
    assert r.returncode == 0 and "STORE " in r.stdout, r.stdout + r.stderr
    return out_root / "cells" / run_key


def _items(spool: Path) -> list:
    """Every item_result payload under the spool (the recorder gzips them; the test lib's read_spool reads plain files only)."""
    from sisteer.registry import records as R
    out = []
    for f in R.iter_record_files(str(spool)):
        for _i, rec, _sha in R.read_record_file(f, compute_sha=False):
            if rec.get("record_type") == "item_result":
                out.append(rec["payload"])
    return out


def _ids(n=2):
    return {b: [f"{b[:5]}_scn_{i}" for i in range(n)] for b in BEHAVIORS}


def _rids(identity: Path):
    d = json.loads(identity.read_text())
    return {(r["route"]["eval_id"], r["route"]["object"]): r["run_id"] for r in d["runs"]}


def test_import_translates_the_wave_cells_to_the_harness_ids_and_keeps_every_run(tmp_path):
    reg = _register(tmp_path); spool = tmp_path / "grading_spool"
    ident = _import(tmp_path, reg, spool)
    doc = json.loads(ident.read_text()); src = json.loads(reg["pending"].read_text())
    assert doc["job_id"] == JOB and len(doc["runs"]) == 21 and {r["run_id"] for r in doc["runs"]} == {r["run_id"] for r in src["runs"]}   # every run, the same ids
    assert doc["object_translation"] == {"none": "none", "prompted__safety_refusal": "prompted__safety", "prompted__si_constitution": "prompted__constitution"}
    for r in doc["runs"]:
        assert r["route"]["object"] == doc["object_translation"][r["route"]["object_wave"]] and r["plan"]["producer_object_key_legacy"] == r["route"]["object_wave"]
    assert doc["imported_from"] == str(reg["pending"]) and doc["imported_by"].endswith("petri_registry_handoff")
    # the recorder routes a Petri store by the HARNESS spelling now: `<harness cell>|<behaviour>` for the six behaviour runs of each object
    routed = hooks.routed_identities(doc["runs"])
    assert routed["prompted__constitution|self_preservation"]["run_id"] == _rids(ident)[("petri_self_preservation", "prompted__constitution")]
    assert "prompted__si_constitution|self_preservation" not in routed


def test_import_refuses_a_foreign_job_an_empty_registration_an_untranslatable_cell_and_no_spool(tmp_path):
    reg = _register(tmp_path); spool = tmp_path / "grading_spool"; out = tmp_path / "out"
    r = _handoff("--spool", str(spool), "import", "--pending", str(reg["pending"]), "--job", "9999", "--out-dir", str(out), "--sh")
    assert r.returncode == 2 and "is job '4242', the drive's proxy is job '9999'" in r.stderr and not (out / "9999.json").exists()
    empty = tmp_path / "empty.json"; empty.write_text(json.dumps({"job_id": JOB, "runs": []}))
    r = _handoff("--spool", str(spool), "import", "--pending", str(empty), "--job", JOB, "--out-dir", str(out), "--sh")
    assert r.returncode == 2 and "names no `runs`" in r.stderr and "registered nothing" in r.stderr
    bad = json.loads(reg["pending"].read_text()); bad["runs"][0]["route"]["object"] = "alphakl_ladder__mystery"; bad["runs"][0]["plan"]["producer_object_key_legacy"] = "alphakl_ladder__mystery"
    badp = tmp_path / "bad.json"; badp.write_text(json.dumps(bad))
    r = _handoff("--spool", str(spool), "import", "--pending", str(badp), "--job", JOB, "--out-dir", str(out), "--sh")
    assert r.returncode == 2 and "has no Petri harness cell id" in r.stderr and "alphakl_ladder__mystery" in r.stderr and not (out / f"{JOB}.json").exists()
    # no spool anywhere: refused by name (the drive is the producer)
    r = _handoff("import", "--pending", str(reg["pending"]), "--job", JOB, "--out-dir", str(out), "--sh",
                 env={"HOME": str(tmp_path / "nohome"), "SISTEER_REGISTRY_PROBE_ROOT": str(tmp_path / "nowhere")})
    assert r.returncode == 2 and "no registry spool resolves" in r.stderr, r.stderr


def test_close_leg_files_the_pooled_run_and_leaves_the_recorder_closed_behaviour_runs_alone(tmp_path):
    reg = _register(tmp_path); spool = tmp_path / "grading_spool"
    ident = _import(tmp_path, reg, spool); rids = _rids(ident)
    home = tmp_path / "home"; out_root = home / ".local" / "state" / "petri_native" / "petri_full22_qwen35_4b"
    ids = _ids()
    cell = _petri_store(out_root, "petricell-a1", "prompted__constitution", "prompted", ids, identity=ident, spool=spool)
    # the REAL recorder routed the rows: six behaviour runs of THIS object opened + closed (run payload, items, generated, store_uri patch)
    recs = read_spool(spool)
    gen = {p["run_id"] for rt, p in recs if rt == "run_state" and p["state"] == "generated"}
    assert gen == {rids[(f"petri_{b}", "prompted__constitution")] for b in BEHAVIORS}
    assert rids[("petri_pooled", "prompted__constitution")] not in gen                        # no row spelling routes to the pooled run
    items = _items(spool)                                                                     # item_result records are gzipped: the records reader
    assert len(items) == 6 * 2 * 3 and {p["run_id"] for p in items} == gen
    # close-leg: the pooled run gets the cell's store_uri + generated; the six are left alone (idempotent on the recorder's records)
    r = _handoff("--spool", str(spool), "close-leg", "--identity", str(ident), "--cell-dir", str(cell))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "1 run(s) filed, 6 already generated, 0 recounted, 0 refused, of 7 registered" in r.stdout and "petri_pooled" in r.stdout and str(cell) in r.stdout
    folded = {x["run_id"]: x for x in fold_spool_runs(reg["cluster_spool"], spool)}          # the harvester's one view: the proxy's `run` records + the drive's patches
    mine = {e: folded[rids[(e, "prompted__constitution")]] for e in ("petri_pooled", *(f"petri_{b}" for b in BEHAVIORS))}
    assert all(x["store_uri"] == str(cell) and x["lifecycle_state"] == "ingested" for x in mine.values()), mine
    assert str(cell).endswith("/.local/state/petri_native/petri_full22_qwen35_4b/cells/petricell-a1")   # the REAL grading-host URI shape
    pooled_patch = [p for rt, p in read_spool(spool) if rt == "run_patch" and p["run_id"] == rids[("petri_pooled", "prompted__constitution")]]
    assert pooled_patch and pooled_patch[-1]["fields"]["n_items_generated"] == 12 and pooled_patch[-1]["fields"]["store_completion_sha256"]
    assert all(x["cluster_job_id"] == JOB for x in mine.values() if x.get("cluster_job_id"))
    # a second close-leg is a no-op: every run already generated
    r = _handoff("--spool", str(spool), "close-leg", "--identity", str(ident), "--cell-dir", str(cell))
    assert r.returncode == 0 and "0 run(s) filed, 7 already generated, 0 recounted, 0 refused" in r.stdout
    # the other objects' runs are untouched by this cell: registered (running), no store
    for other in (rids[("petri_pooled", "none")], rids[("petri_self_preservation", "prompted__safety")]):
        assert folded[other]["lifecycle_state"] == "running" and not folded[other].get("store_uri")


def test_close_leg_recounts_a_generated_run_whose_recorded_total_is_stale_against_the_store(tmp_path):
    """A behaviour run already `generated` with a recorded n_items_generated that disagrees with rows.jsonl (the store
    recorder of a resumed pass wrote its increment) is NOT left alone: close-leg files a corrective run_patch with the store total and
    its provenance; a run whose total agrees is left alone, and a second close-leg is a no-op."""
    reg = _register(tmp_path); spool = tmp_path / "grading_spool"
    ident = _import(tmp_path, reg, spool); rids = _rids(ident)
    out_root = tmp_path / "home" / ".local" / "state" / "petri_native" / "petri_full22_qwen35_4b"
    cell = _petri_store(out_root, "petricell-d4", "prompted__constitution", "prompted", _ids(), identity=ident, spool=spool)
    stale = rids[("petri_self_preservation", "prompted__constitution")]
    # the earlier recorder's increment: a later run_patch saying 1 on a run whose store holds 2 items of this behaviour
    from sisteer.registry import identity as RI
    campaign = RI.parse_run_id(stale)["campaign"]
    emit.RecordWriter(str(spool), campaign, stale, emit.producer_block("test", code_sha="b" * 40)).emit_patch(fields={"n_items_generated": 1})
    r = _handoff("--spool", str(spool), "close-leg", "--identity", str(ident), "--cell-dir", str(cell))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "1 run(s) filed, 5 already generated, 1 recounted, 0 refused, of 7 registered" in r.stdout
    assert f"petri_self_preservation {stale[:40]}… n_items_generated 1 -> 2 (increase: the store recorder wrote a resumed pass's increment" in r.stdout
    patches = [p for rt, p in read_spool(spool) if rt == "run_patch" and p["run_id"] == stale]
    rc = patches[-1]["fields"]["extra_json"]["n_items_generated_recount"]
    assert patches[-1]["fields"]["n_items_generated"] == 2 and (rc["previous"], rc["direction"], rc["ledger_settled"]) == (1, "increase", True)
    assert patches[-1]["fields"]["extra_json"]["arm_name_legacy"] == "prompted__constitution"
    folded = {x["run_id"]: x for x in fold_spool_runs(reg["cluster_spool"], spool)}
    assert folded[stale]["n_items_generated"] == 2                                     # the fixture reader folds the patched total (extra_json is checked on the record above)
    assert folded[rids[("petri_defer_to_users", "prompted__constitution")]]["n_items_generated"] == 2
    r = _handoff("--spool", str(spool), "close-leg", "--identity", str(ident), "--cell-dir", str(cell))
    assert r.returncode == 0 and "0 run(s) filed, 7 already generated, 0 recounted, 0 refused" in r.stdout


def _refusals(ident: Path) -> dict:
    """{run_id: refusal} over the per-run files of close_leg_refusals/ beside the identity (the directory verify reads; tombstones and
    the watermark are not refusals)."""
    d = ident.parent / "close_leg_refusals"
    return {json.loads(f.read_text())["run_id"]: json.loads(f.read_text())
            for f in sorted(d.glob("*.json")) if not f.name.endswith(".cleared.json") and not f.name.startswith(".")} if d.is_dir() else {}


def _shrink(cell: Path, behaviour: str) -> str:
    """Cut one scenario's three rows of `behaviour` from the cell's rows.jsonl (the stale-markers shape); returns the full text."""
    rows_p = cell / "rows.jsonl"; full = rows_p.read_text(encoding="utf-8")
    kept = [ln for ln in full.splitlines() if ln.strip() and not (json.loads(ln)["task"] == f"petri/{behaviour}" and json.loads(ln)["item_id"].endswith("_1"))]
    rows_p.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return full


def test_two_concurrent_refusing_legs_both_persist_and_a_third_legs_recount_clears_only_its_own_run(tmp_path):
    """The drive runs distinct legs concurrently under ONE identity. Two cells shrunk under stale markers,
    their close-legs started together: every refusal survives as its own file (no shared read-modify-replace, no shared temp name), verify
    names all four (two behaviour runs + two pooled runs). Restoring ONE cell's rows and closing it clears only that cell's two files; the
    other cell's refusals stand and verify still names them."""
    reg = _register(tmp_path); spool = tmp_path / "grading_spool"
    ident = _import(tmp_path, reg, spool); rids = _rids(ident)
    out_root = tmp_path / "home" / ".local" / "state" / "petri_native" / "petri_full22_qwen35_4b"
    cell_a = _petri_store(out_root, "petricell-f6", "prompted__safety", "prompted", _ids(), identity=ident, spool=spool)
    cell_b = _petri_store(out_root, "petricell-g7", "prompted__constitution", "prompted", _ids(), identity=ident, spool=spool)
    full_a = _shrink(cell_a, "self_preservation"); _shrink(cell_b, "defer_to_users")
    env = registry_emitting_env()
    procs = [subprocess.Popen([VENV_PY, str(HANDOFF), "--spool", str(spool), "close-leg", "--identity", str(ident), "--cell-dir", str(c)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=str(REPO)) for c in (cell_a, cell_b)]
    outs = [p.communicate(timeout=300) for p in procs]
    assert [p.returncode for p in procs] == [3, 3], outs
    # the two behaviour runs are refused (recorded 2, store 1); the pooled runs were never `generated` before, so each is FILED fresh at
    # the shrunk count (11) by the ordinary branch — a first filing is a reading, not a decrease
    want = {rids[("petri_self_preservation", "prompted__safety")], rids[("petri_defer_to_users", "prompted__constitution")]}
    assert set(_refusals(ident)) == want and not list((ident.parent / "close_leg_refusals").glob("*.tmp"))
    v = _handoff("--spool", str(spool), "verify", "--identity", str(ident), "--objects", "prompted__safety,prompted__constitution")
    assert v.returncode == 2 and "2 with a standing close-leg refusal" in v.stderr and v.stdout.count("REFUSED by close-leg") == 2
    # a third leg: cell A's rows restored and closed again — its behaviour run agrees (2 == 2) and its file alone clears, its pooled run
    # recounts upward (11 -> 12); cell B's refusal stands untouched
    (cell_a / "rows.jsonl").write_text(full_a, encoding="utf-8")
    r = _handoff("--spool", str(spool), "close-leg", "--identity", str(ident), "--cell-dir", str(cell_a))
    assert r.returncode == 0 and "1 recounted, 0 refused" in r.stdout and "n_items_generated 11 -> 12 (increase" in r.stdout
    assert set(_refusals(ident)) == {rids[("petri_defer_to_users", "prompted__constitution")]}
    v = _handoff("--spool", str(spool), "verify", "--identity", str(ident), "--objects", "prompted__safety,prompted__constitution")
    assert v.returncode == 2 and "1 with a standing close-leg refusal" in v.stderr and v.stdout.count("REFUSED by close-leg") == 1
    assert _handoff("--spool", str(spool), "verify", "--identity", str(ident), "--objects", "prompted__safety").returncode == 0


def test_a_legacy_single_file_refusal_map_is_migrated_into_per_run_files_and_still_names_the_run(tmp_path):
    """A grading state upgraded with a legacy close_leg_refusals.json standing. The very next `verify`
    migrates it — one per-run file per refusal, the legacy file renamed `.migrated` (never deleted) — and still names the run as
    UNFILED; a later close-leg whose store agrees clears only that run's file; a second legacy file left behind is carried the same way
    without resurrecting the cleared refusal (the per-run file, or its explicit absence after a clear, is not overwritten)."""
    reg = _register(tmp_path); spool = tmp_path / "grading_spool"
    ident = _import(tmp_path, reg, spool); rids = _rids(ident)
    out_root = tmp_path / "home" / ".local" / "state" / "petri_native" / "petri_full22_qwen35_4b"
    cell = _petri_store(out_root, "petricell-h8", "prompted__safety", "prompted", _ids(), identity=ident, spool=spool)
    victim = rids[("petri_self_preservation", "prompted__safety")]
    legacy = ident.parent / "close_leg_refusals.json"
    legacy.write_text(json.dumps({victim: {"cell_id": "prompted__safety", "eval_id": "petri_self_preservation", "recorded": 2, "store_total": 1,
                                           "why": "legacy (E3) refusal standing at upgrade", "rows_sha256": "0" * 64, "utc": "2026-10-01T13:40:00Z"}}), encoding="utf-8")
    v = _handoff("--spool", str(spool), "verify", "--identity", str(ident), "--objects", "prompted__safety")
    assert v.returncode == 2 and "1 with a standing close-leg refusal" in v.stderr and "REFUSED by close-leg (legacy (E3) refusal" in v.stdout
    assert "migrated 1 standing refusal(s)" in v.stderr
    assert not legacy.exists() and (ident.parent / "close_leg_refusals.json.migrated").is_file()         # renamed, never deleted
    migrated = _refusals(ident)
    assert set(migrated) == {victim} and migrated[victim]["migrated_from"] == str(legacy) and migrated[victim]["recorded"] == 2
    v2 = _handoff("--spool", str(spool), "verify", "--identity", str(ident), "--objects", "prompted__safety")
    assert v2.returncode == 2 and "migrated" not in v2.stderr                                              # idempotent: nothing to carry twice
    # the store agrees with the recorded total (2 == 2): close-leg clears that run's file only; verify passes
    r = _handoff("--spool", str(spool), "close-leg", "--identity", str(ident), "--cell-dir", str(cell))
    assert r.returncode == 0 and "1 run(s) filed, 6 already generated, 0 recounted, 0 refused" in r.stdout   # the pooled run files fresh here
    assert _refusals(ident) == {}
    assert _handoff("--spool", str(spool), "verify", "--identity", str(ident), "--objects", "prompted__safety").returncode == 0
    # the clear left a TOMBSTONE (E6): a second, OLDER legacy map re-appearing (another upgrade path, or a copy) does NOT resurrect
    # the cleared refusal — the tombstone is at least as new as its entry — and is still renamed aside, never deleted
    tomb = json.loads((ident.parent / "close_leg_refusals" / (RI.run_dir_name(victim) + ".cleared.json")).read_text())
    assert tomb["run_id"] == victim and tomb["cleared_refusal"]["why"].startswith("legacy (E3) refusal") and len(tomb["agreeing_rows_sha256"]) == 64
    legacy.write_text(json.dumps({victim: {"cell_id": "prompted__safety", "eval_id": "petri_self_preservation", "recorded": 2, "store_total": 1,
                                           "why": "second legacy map", "rows_sha256": "1" * 64, "utc": "2026-10-01T13:50:00Z"}}), encoding="utf-8")
    r = _handoff("--spool", str(spool), "close-leg", "--identity", str(ident), "--cell-dir", str(cell))
    assert r.returncode == 0 and "migrated 0 standing refusal(s)" in r.stderr and "0 run(s) filed, 7 already generated, 0 recounted, 0 refused" in r.stdout and _refusals(ident) == {}
    assert [p.name for p in sorted(ident.parent.glob("close_leg_refusals.json.migrated*"))][0] == "close_leg_refusals.json.migrated"
    assert len(list(ident.parent.glob("close_leg_refusals.json.migrated*"))) == 2
    # the SAME bytes as the first map re-appearing: the watermark imports nothing either
    (ident.parent / "close_leg_refusals.json").write_bytes((ident.parent / "close_leg_refusals.json.migrated").read_bytes())
    v3 = _handoff("--spool", str(spool), "verify", "--identity", str(ident), "--objects", "prompted__safety")
    assert v3.returncode == 0 and "already migrated (watermark)" in v3.stderr and _refusals(ident) == {}
    # a FRESH refusal after the clear supersedes the tombstone: the store shrinks, close-leg refuses, verify names it
    _shrink(cell, "self_preservation")
    r = _handoff("--spool", str(spool), "close-leg", "--identity", str(ident), "--cell-dir", str(cell))
    assert r.returncode == 3 and victim in _refusals(ident) and not (ident.parent / "close_leg_refusals" / (RI.run_dir_name(victim) + ".cleared.json")).exists()


def test_a_verifier_holding_a_stale_legacy_snapshot_cannot_resurrect_a_refusal_a_concurrent_close_leg_cleared(tmp_path):
    """The exact interleaving replayed step by step and then run for real: the verifier LOADS the legacy
    map; a close-leg migrates it, confirms the restored store agrees and clears the run (tombstone); the verifier then continues with
    its stale snapshot — the tombstone blocks the re-import, the refusal stays cleared, verify passes. Then the two processes really
    race (verify ∥ close-leg) on a fresh legacy map over an agreeing store: whatever the order, the end state is cleared and verified."""
    reg = _register(tmp_path); spool = tmp_path / "grading_spool"
    ident = _import(tmp_path, reg, spool); rids = _rids(ident)
    out_root = tmp_path / "home" / ".local" / "state" / "petri_native" / "petri_full22_qwen35_4b"
    cell = _petri_store(out_root, "petricell-i9", "prompted__safety", "prompted", _ids(), identity=ident, spool=spool)
    victim = rids[("petri_self_preservation", "prompted__safety")]
    entry = {"cell_id": "prompted__safety", "eval_id": "petri_self_preservation", "recorded": 2, "store_total": 1,
             "why": "legacy (E3) refusal standing at upgrade", "rows_sha256": "0" * 64, "utc": "2026-10-01T13:40:00Z"}
    legacy = ident.parent / "close_leg_refusals.json"; legacy.write_text(json.dumps({victim: entry}), encoding="utf-8")
    import importlib.util
    spec = importlib.util.spec_from_file_location("ho_mod", str(HANDOFF)); ho = importlib.util.module_from_spec(spec); spec.loader.exec_module(ho)
    # (1) the verifier loaded the map (its snapshot) ... (2) close-leg migrates, agrees (store 2 == recorded 2) and clears → tombstone
    snapshot = json.loads(legacy.read_text())
    r = _handoff("--spool", str(spool), "close-leg", "--identity", str(ident), "--cell-dir", str(cell))
    assert r.returncode == 0 and "migrated 1 standing refusal(s)" in r.stderr and _refusals(ident) == {}
    tomb_p = ident.parent / "close_leg_refusals" / (RI.run_dir_name(victim) + ".cleared.json")
    assert tomb_p.is_file() and json.loads(tomb_p.read_text())["cleared_refusal"]["recorded"] == 2
    # (3) the verifier continues with its stale snapshot: the import half runs under the lock and the tombstone blocks it
    with ho._refusals_lock(str(ident)):
        assert ho._import_legacy_map(str(ident), snapshot, str(legacy)) == 0
    assert _refusals(ident) == {} and _handoff("--spool", str(spool), "verify", "--identity", str(ident), "--objects", "prompted__safety").returncode == 0
    # the real race: a fresh legacy map (newer utc than the tombstone) over an agreeing store; verify and close-leg start together
    entry2 = {**entry, "utc": "2099-01-01T00:00:00Z", "why": "fresh legacy map"}
    legacy.write_text(json.dumps({victim: entry2}), encoding="utf-8")
    env = registry_emitting_env()
    procs = [subprocess.Popen([VENV_PY, str(HANDOFF), "--spool", str(spool), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=str(REPO))
             for args in (["verify", "--identity", str(ident), "--objects", "prompted__safety"], ["close-leg", "--identity", str(ident), "--cell-dir", str(cell)])]
    outs = [p.communicate(timeout=300) for p in procs]
    assert procs[1].returncode == 0, outs                                                         # close-leg: the store agrees, cleared
    assert _refusals(ident) == {} and not legacy.exists()                                          # the map is migrated aside exactly once
    assert _handoff("--spool", str(spool), "verify", "--identity", str(ident), "--objects", "prompted__safety").returncode == 0
    assert not list((ident.parent / "close_leg_refusals").glob("*.tmp"))


def _drop_total_from_spool(spool: Path, run_id: str) -> int:
    """Rewrite the run's run_patch record files WITHOUT any n_items_generated field — the shape a crash between the recorder's
    `run_state generated` and its completion patch leaves. Returns the number of records rewritten."""
    from sisteer.registry import identity as RI, records as R
    d = spool / RI.parse_run_id(run_id)["campaign"] / RI.run_dir_name(run_id)
    n = 0
    for f in sorted(d.iterdir()):
        m = R.RECORD_FILE_RE.match(f.name)
        if not m or m.group("type") != "run_patch":
            continue
        out = []
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            fields = (rec.get("payload") or {}).get("fields") or {}
            if "n_items_generated" in fields:
                fields.pop("n_items_generated"); n += 1
            out.append(json.dumps(rec))
        f.write_text("\n".join(out) + "\n", encoding="utf-8")
    return n


def test_close_leg_recounts_a_generated_run_with_no_recorded_total_and_rules_on_a_decrease(tmp_path):
    """A `generated` run whose spool carries no n_items_generated (the state and the total
    ride separate files) is recounted and patched — provenance names the missing previous — and a second close-leg is a no-op. A store
    SHORTER than the recorded total is lowered ONLY under a completion proof bound to the current rows (results.json's audited footing
    re-verified against the snapshot) and a settled ledger, under direction `decrease` with its cause; stale markers over a truncated
    store are REFUSED by name with exit 3, the refusals persisted beside the identity and read by `verify` as UNFILED, cleared when a
    later close-leg agrees. The count and the recorded rows_sha256 describe one snapshot of rows.jsonl."""
    import hashlib
    reg = _register(tmp_path); spool = tmp_path / "grading_spool"
    ident = _import(tmp_path, reg, spool); rids = _rids(ident)
    out_root = tmp_path / "home" / ".local" / "state" / "petri_native" / "petri_full22_qwen35_4b"
    cell = _petri_store(out_root, "petricell-e5", "prompted__safety", "prompted", _ids(), identity=ident, spool=spool)
    missing = rids[("petri_defer_to_users", "prompted__safety")]
    assert _drop_total_from_spool(spool, missing) == 1
    r = _handoff("--spool", str(spool), "close-leg", "--identity", str(ident), "--cell-dir", str(cell))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "1 run(s) filed, 5 already generated, 1 recounted, 0 refused, of 7 registered" in r.stdout
    assert f"petri_defer_to_users {missing[:40]}… n_items_generated None -> 2 (missing: the run reached `generated` with no n_items_generated recorded" in r.stdout
    patch = [p for rt, p in read_spool(spool) if rt == "run_patch" and p["run_id"] == missing][-1]
    rc = patch["fields"]["extra_json"]["n_items_generated_recount"]
    assert patch["fields"]["n_items_generated"] == 2 and (rc["previous"], rc["direction"], rc["ledger_settled"]) == (None, "missing", True)
    assert rc["rows_sha256"] == hashlib.sha256((cell / "rows.jsonl").read_bytes()).hexdigest()
    folded = {x["run_id"]: x for x in fold_spool_runs(reg["cluster_spool"], spool)}
    assert folded[missing]["n_items_generated"] == 2
    r = _handoff("--spool", str(spool), "close-leg", "--identity", str(ident), "--cell-dir", str(cell))
    assert r.returncode == 0 and "0 run(s) filed, 7 already generated, 0 recounted, 0 refused" in r.stdout
    # the store SHRINKS under a recorded total of 2 for self_preservation (one scenario's three rows cut) while results.json and the
    # settled ledger — markers of a PAST completion — stay in place: the decrease is REFUSED, for the behaviour run AND the pooled
    # run (12 -> 11), exit 3, the refusals persisted beside the identity, and `verify` reads them as UNFILED
    rows_p = cell / "rows.jsonl"; full_rows = rows_p.read_text(encoding="utf-8")
    kept = [ln for ln in full_rows.splitlines() if ln.strip() and not (json.loads(ln)["task"] == "petri/self_preservation" and json.loads(ln)["item_id"].endswith("_1"))]
    rows_p.write_text("\n".join(kept) + "\n", encoding="utf-8")
    shrunk = rids[("petri_self_preservation", "prompted__safety")]; pooled = rids[("petri_pooled", "prompted__safety")]
    r = _handoff("--spool", str(spool), "close-leg", "--identity", str(ident), "--cell-dir", str(cell))
    assert r.returncode == 3, r.stdout + r.stderr
    assert "0 recounted, 2 refused" in r.stdout and "REFUSED to lower n_items_generated 2 -> 1" in r.stdout and "REFUSED to lower n_items_generated 12 -> 11" in r.stdout
    assert "PROOF FAILS" in r.stdout and "ledger settled" in r.stdout and "NOT filed complete" in r.stderr and "exit 3" in r.stderr
    assert [p for rt, p in read_spool(spool) if rt == "run_patch" and p["run_id"] in (shrunk, pooled) and "n_items_generated_recount" in (p["fields"].get("extra_json") or {})] == []
    refusals = _refusals(ident)
    assert set(refusals) == {shrunk, pooled} and refusals[shrunk]["recorded"] == 2 and refusals[shrunk]["store_total"] == 1
    assert not list((ident.parent / "close_leg_refusals").glob("*.tmp"))                       # unique temp names, all replaced
    v = _handoff("--spool", str(spool), "verify", "--identity", str(ident), "--objects", "prompted__safety")
    assert v.returncode == 2 and "2 with a standing close-leg refusal" in v.stderr and v.stdout.count("REFUSED by close-leg") == 2
    # an UNSETTLED ledger over the same shrunk store refuses likewise (exit 3)
    ledger_p = cell / "petri_incomplete.json"; settled = json.loads(ledger_p.read_text())
    ledger_p.write_text(json.dumps({**settled, "unresolved": [{"behavior": "self_preservation", "scenario_id": "x", "epoch": 1, "dimension": "self_preservation", "reason": "no result", "retries": 0}]}))
    r = _handoff("--spool", str(spool), "close-leg", "--identity", str(ident), "--cell-dir", str(cell))
    assert r.returncode == 3 and "0 recounted, 2 refused" in r.stdout and "NOT settled" in r.stdout
    ledger_p.write_text(json.dumps(settled))
    # the rows RESTORED: the proof binds again, the standing refusals clear (both runs agree with their recorded totals), verify passes
    rows_p.write_text(full_rows, encoding="utf-8")
    r = _handoff("--spool", str(spool), "close-leg", "--identity", str(ident), "--cell-dir", str(cell))
    assert r.returncode == 0 and "7 already generated, 0 recounted, 0 refused" in r.stdout
    assert _refusals(ident) == {}
    v = _handoff("--spool", str(spool), "verify", "--identity", str(ident), "--objects", "prompted__safety")
    assert v.returncode == 0 and "REFUSED by close-leg" not in v.stdout and "7 of 7 planned run(s) filed" in v.stdout
    # a BOUND decrease: the recorded total is simply too high (5 on a complete store of 2) — the proof holds against the current rows,
    # so the decrease is filed with its direction, cause and the proof's evidence
    from sisteer.registry import identity as RI
    emit.RecordWriter(str(spool), RI.parse_run_id(shrunk)["campaign"], shrunk, emit.producer_block("test", code_sha="b" * 40)).emit_patch(fields={"n_items_generated": 5})
    r = _handoff("--spool", str(spool), "close-leg", "--identity", str(ident), "--cell-dir", str(cell))
    assert r.returncode == 0 and "1 recounted, 0 refused" in r.stdout and f"n_items_generated 5 -> 2 (decrease: the store holds fewer items than recorded and the completion proof binds" in r.stdout
    patch = [p for rt, p in read_spool(spool) if rt == "run_patch" and p["run_id"] == shrunk][-1]
    rc = patch["fields"]["extra_json"]["n_items_generated_recount"]
    assert (patch["fields"]["n_items_generated"], rc["previous"], rc["direction"], rc["ledger_settled"], rc["completion_proof"]["holds"]) == (2, 5, "decrease", True, True)
    assert rc["rows_sha256"] == hashlib.sha256(rows_p.read_bytes()).hexdigest()


def test_close_leg_files_nothing_for_an_incomplete_cell_or_an_unregistered_control_leg(tmp_path):
    reg = _register(tmp_path); spool = tmp_path / "grading_spool"
    ident = _import(tmp_path, reg, spool); rids = _rids(ident)
    out_root = tmp_path / "home" / ".local" / "state" / "petri_native" / "petri_full22_qwen35_4b"
    cell = _petri_store(out_root, "petricell-b2", "none", "none", _ids(), identity=ident, spool=spool, complete=False)   # no results.json
    r = _handoff("--spool", str(spool), "close-leg", "--identity", str(ident), "--cell-dir", str(cell))
    assert r.returncode == 0 and "INCOMPLETE" in r.stdout and "7 run(s) stay running" in r.stdout
    assert not [p for rt, p in read_spool(spool) if rt == "run_state" and p["state"] == "generated"]      # nobody closed anything
    folded = {x["run_id"]: x for x in fold_spool_runs(reg["cluster_spool"], spool)}
    assert all(x["lifecycle_state"] == "running" for x in folded.values())                                     # nothing reached generated
    assert not folded[rids[("petri_pooled", "none")]].get("store_uri")                                        # the pooled run: no store until close-leg
    assert folded[rids[("petri_self_preservation", "none")]]["store_uri"] == str(cell)                        # the recorder's run payload names the store at OPEN (running)
    # the control leg (none__lora0): no registered object — nothing filed, said
    ctl = _petri_store(out_root, "petricell-c3", "none__lora0", "none", _ids(), identity=ident, spool=spool)
    r = _handoff("--spool", str(spool), "close-leg", "--identity", str(ident), "--cell-dir", str(ctl))
    assert r.returncode == 0 and "no registered run names this cell" in r.stdout
    # finish: pushes every existing run dir final (no push script here: a no-op that says so)
    r = _handoff("--spool", str(spool), "finish", "--identity", str(ident))
    assert r.returncode == 0 and "pushed final" in r.stdout and "no push script" in r.stdout
    # a cell dir that is not a Petri store refuses by name
    r = _handoff("--spool", str(spool), "close-leg", "--identity", str(ident), "--cell-dir", str(tmp_path / "nowhere"))
    assert r.returncode == 2 and "no manifest.json" in r.stderr


def test_the_drive_imports_before_the_freeze_closes_each_leg_and_pushes_at_the_end():
    src = DRIVE.read_text()
    i_import = src.index("petri_registry_handoff.py import --pending")
    assert src.index("[ \"$ST\" = RUNNING ] ||") < i_import < src.index("print-cells --out-root")          # after the proxy is verified RUNNING, before the freeze
    assert 'cat $SHARED/registry/pending/$JOBID.json' in src and "FATAL: proxy job $JOBID registered no runs" in src
    assert 'F22_REGISTRY_HANDOFF' in src and "registry hand-off OFF" in src
    assert 'eval "$REG_SH"; export SISTEER_REGISTRY_RUN_IDENTITY SISTEER_REGISTRY_SPOOL' in src
    assert 'env ${REG_IDENTITY:+SLURM_JOB_ID="$JOBID"} "$PY" scripts/petri_run.py' in src                # the proxy job id rides the runner's row constants
    i_close = src.index("petri_registry_handoff.py close-leg --identity")
    assert src.index("leg_status_write \"$idx\" \"${rc:-unreachable}\"") < i_close < src.index("LEG[$idx] COMPLETE")
    assert '[ "$rc" = 0 ] && [ -f "$OUT/cells/$rk/results.json" ]' in src[i_close - 400:i_close]
    i_finish = src.index("petri_registry_handoff.py finish --identity")
    assert src.index("DRIVE_DONE") < i_finish < src.index("release the proxy when every leg has ENDED")
    assert "--export=" not in src


def test_the_drives_leg_invocation_executes_with_the_hand_off_on_and_off(tmp_path):
    """The leg command form is EXECUTED, not string-matched: a bare `${X:+VAR=v} cmd` is not an
    assignment prefix after expansion — bash would run a command literally named `SLURM_JOB_ID=123` (rc 127) — so the drive
    goes through `env`. Hand-off on: the runner sees the proxy job id; off: it runs unchanged and sees no SLURM_JOB_ID."""
    src = (REPO / "scripts/sdfbase/full22/petri_drive_full22.sh").read_text()
    line = next(l for l in src.splitlines() if 'scripts/petri_run.py "${PETRI_RUN_OPTS[@]}" run --out-root' in l and "env " in l)
    form = line.strip().split(">>")[0]                          # the command up to its log redirect
    fake = tmp_path / "py"; fake.write_text("#!/bin/bash\nprintf 'job=%s args=%s\\n' \"${SLURM_JOB_ID-unset}\" \"$*\"\n"); fake.chmod(0o755)
    def run(reg_identity):
        script = ("PY=%r; JOBID=4242; REG_IDENTITY=%r; OUT=/tmp/o; idx=3; PETRI_RUN_OPTS=(--opt); TARGET_DECODE_ARGS=(); RERUN=; " % (str(fake), reg_identity)) + form
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env={"PATH": os.environ["PATH"]})
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    rc, out, err = run("/tmp/identity.json")
    assert rc == 0 and out.startswith("job=4242 args=scripts/petri_run.py --opt run --out-root /tmp/o --cell-index 3"), (rc, out, err)
    rc, out, err = run("")
    assert rc == 0 and out.startswith("job=unset args=scripts/petri_run.py --opt run"), (rc, out, err)


def _handoff_block() -> str:
    """The drive's hand-off step (1b), from its switch to the tunnel step — executed below, not string-matched."""
    src = DRIVE.read_text()
    return src[src.index('REG_HANDOFF="${F22_REGISTRY_HANDOFF:-on}"'):src.index("# 2. the tunnel")]


def _run_handoff_block(tmp_path: Path, env_extra: dict) -> tuple[int, str]:
    """Run the block under bash with the drive's own options, a `log` that prints, an `ssh` on PATH that answers NOTHING (the
    proxy registered no runs: no pending file on the cluster) and the variables the block reads; REG_IDENTITY is printed last."""
    shims = tmp_path / "bin"; shims.mkdir(exist_ok=True)
    (shims / "ssh").write_text("#!/bin/bash\nexit 1\n"); (shims / "ssh").chmod(0o755)
    state = tmp_path / "state"; state.mkdir(exist_ok=True)
    script = ("set -uo pipefail\nlog(){ echo \"$*\"; }\n"
              f"STATE={str(state)!r}; LOG={str(state / 'drive.log')!r}; HOST=fakehost; SHARED=/shared/TESTUSER; JOBID={JOB}; "
              f"WT={str(REPO)!r}; PY={sys.executable!r}\n" + _handoff_block() + '\necho "REG_IDENTITY=[$REG_IDENTITY]"\n')
    env = {"PATH": f"{shims}:{os.environ['PATH']}", "HOME": str(tmp_path), **env_extra}
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=60)
    return r.returncode, r.stdout + r.stderr


def test_the_drive_stands_the_hand_off_down_when_the_registry_is_disabled_and_keeps_the_fatal_when_it_is_enabled(tmp_path):
    """The proxy-registration FATAL is a defect only of a registry-ENABLED audit. Under the
    emitters' own OFF switch (SISTEER_REGISTRY_DISABLED — the variable the suite's conftest sets for every child, in every spelling
    hooks.emission_enabled reads) the block logs the switch BY NAME and proceeds with no identity, exactly as F22_REGISTRY_HANDOFF=off
    does; with the registry enabled and no pending file the FATAL still fires (rc 1) — `=0` and the empty string included, since
    neither disables the emitters. Before the fix the wave tests (registry off, shimmed ssh) died here."""
    rc, out = _run_handoff_block(tmp_path, {emit.DISABLED_ENV: "1"})
    assert rc == 0 and "registry hand-off OFF (SISTEER_REGISTRY_DISABLED=1" in out and "FATAL" not in out and "REG_IDENTITY=[]" in out, out
    for spelling in ("true", "YES", " on "):
        rc, out = _run_handoff_block(tmp_path, {emit.DISABLED_ENV: spelling})
        assert rc == 0 and "registry hand-off OFF (SISTEER_REGISTRY_DISABLED=" in out and "FATAL" not in out, (spelling, out)
    rc, out = _run_handoff_block(tmp_path, {"F22_REGISTRY_HANDOFF": "off"})
    assert rc == 0 and "registry hand-off OFF (F22_REGISTRY_HANDOFF=off)" in out and "FATAL" not in out and "REG_IDENTITY=[]" in out, out
    for env in ({}, {emit.DISABLED_ENV: "0"}, {emit.DISABLED_ENV: ""}, {emit.DISABLED_ENV: "off"}):
        rc, out = _run_handoff_block(tmp_path, env)
        assert rc == 1 and f"FATAL: proxy job {JOB} registered no runs" in out and "registry hand-off OFF" not in out, (env, out)


def test_close_leg_counts_terminal_judge_refusals_as_generated_transcripts(tmp_path):
    """A terminal judge refusal is a transcript the audit generated and the judge declined -- it carries no rows,
    so counting rows alone undercounted n_items_generated by exactly the refusals (the u0 prompted cells: 9/7/59/57 against planned
    10/10/60/60) and the judge-only re-score of those items then failed the hand-off's coverage check. Now the pooled run is filed with the
    transcript count, a behaviour run whose recorder total counted rows alone is recounted UP with a cause naming the refusals, the other
    behaviour runs agree and are left alone, and the completion proof (audited items / expected) still binds to the rows."""
    reg = _register(tmp_path); spool = tmp_path / "grading_spool"
    ident = _import(tmp_path, reg, spool); rids = _rids(ident)
    home = tmp_path / "home"; out_root = home / ".local" / "state" / "petri_native" / "petri_full22_qwen35_4b"
    ids = _ids()                                                                           # 2 scored scenarios per behaviour -> the recorder files 2 each
    cell = _petri_store(out_root, "petricell-jr", "prompted__constitution", "prompted", ids, identity=ident, spool=spool)
    b0 = BEHAVIORS[0]
    # the sidecar: ONE terminal judge refusal of b0 -- a transcript with no rows; results.json's footing counts it outside `items`
    (cell / "petri_judge_refused.json").write_text(json.dumps({"schema": "sisteer.petri_runner/judge_refused/v1", "version": 1, "cell_id": "prompted__constitution",
        "judge_refused": [{"behavior": b0, "scenario_id": f"{b0[:5]}_scn_refused", "epoch": 1, "outcome": "judge_refused", "judge_model": "anthropic/claude-opus-5",
                           "reason": "judge refused to score (anthropic/claude-opus-5): synthetic", "dimensions": [b0, "eval_awareness", "scenario_realism"]}],
        "rerolled": []}))
    results = json.loads((cell / "results.json").read_text())
    results["audited"] = {"expected": 13, "items": 12}; (cell / "results.json").write_text(json.dumps(results))
    r = _handoff("--spool", str(spool), "close-leg", "--identity", str(ident), "--cell-dir", str(cell))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "1 run(s) filed, 5 already generated, 1 recounted, 0 refused, of 7 registered" in r.stdout, r.stdout
    assert f"petri_{b0} {rids[(f'petri_{b0}', 'prompted__constitution')][:40]}… n_items_generated 2 -> 3 (increase: 1 terminal judge refusal(s) are generated transcripts" in r.stdout, r.stdout
    pooled = [p for rt, p in read_spool(spool) if rt == "run_patch" and p["run_id"] == rids[("petri_pooled", "prompted__constitution")]]
    assert pooled[-1]["fields"]["n_items_generated"] == 13                                      # 12 scored + the declined transcript
    b0_patch = [p for rt, p in read_spool(spool) if rt == "run_patch" and p["run_id"] == rids[(f"petri_{b0}", "prompted__constitution")] and "n_items_generated_recount" in (p["fields"].get("extra_json") or {})]
    rc = b0_patch[-1]["fields"]["extra_json"]["n_items_generated_recount"]
    assert b0_patch[-1]["fields"]["n_items_generated"] == 3 and (rc["previous"], rc["direction"], rc["completion_proof"]["holds"]) == (2, "increase", True), rc
    folded = {x["run_id"]: x for x in fold_spool_runs(reg["cluster_spool"], spool)}
    assert folded[rids[("petri_pooled", "prompted__constitution")]]["n_items_generated"] == 13
    assert folded[rids[(f"petri_{b0}", "prompted__constitution")]]["n_items_generated"] == 3
    assert all(folded[rids[(f"petri_{b}", "prompted__constitution")]]["n_items_generated"] == 2 for b in BEHAVIORS[1:])
    # idempotent: a second close-leg agrees everywhere (the refusal is still counted, nothing moves)
    r2 = _handoff("--spool", str(spool), "close-leg", "--identity", str(ident), "--cell-dir", str(cell))
    assert r2.returncode == 0 and "0 run(s) filed, 7 already generated, 0 recounted, 0 refused" in r2.stdout, r2.stdout


def test_two_refused_epochs_of_one_scenario_are_two_generated_items_and_a_re_score_of_one_leaves_the_count(tmp_path):
    """With bloom_options.epochs 2 the runner's row item ids are scenario#e1 / scenario#e2 and the judge
    may decline both epochs of one scenario -- two terminal records. The hand-off used to dedup refusals on (behavior, scenario) and add
    a bare scenario key: one generated item for two transcripts, so a later re-score could again exceed the count. Now results.json's
    `epochs` keys the item-id format, each refusal record is normalised to the runner's item id before the union, and the completion
    proof counts refusal records: two refused epochs add two; a re-score of one epoch (its rows appear, its refusal leaves the sidecar)
    leaves every count where it was; a record without `epochs` whose behaviours block derives none is refused by name."""
    reg = _register(tmp_path); spool = tmp_path / "grading_spool"
    ident = _import(tmp_path, reg, spool); rids = _rids(ident)
    home = tmp_path / "home"; out_root = home / ".local" / "state" / "petri_native" / "petri_full22_qwen35_4b"
    ids = {b: [f"{b[:5]}_scn_{i}#e{e}" for i in range(2) for e in (1, 2)] for b in BEHAVIORS}      # 2 scenarios x 2 epochs per behaviour
    cell = _petri_store(out_root, "petricell-2e", "prompted__constitution", "prompted", ids, identity=ident, spool=spool)
    b0 = BEHAVIORS[0]; refused_sid = f"{b0[:5]}_scn_refused"
    (cell / "petri_judge_refused.json").write_text(json.dumps({"schema": "sisteer.petri_runner/judge_refused/v1", "version": 1, "cell_id": "prompted__constitution",
        "judge_refused": [{"behavior": b0, "scenario_id": refused_sid, "epoch": e, "outcome": "judge_refused", "judge_model": "anthropic/claude-opus-5",
                           "reason": "judge refused to score (anthropic/claude-opus-5): synthetic", "dimensions": [b0, "eval_awareness", "scenario_realism"]} for e in (1, 2)],
        "rerolled": []}))
    results = json.loads((cell / "results.json").read_text())
    results["epochs"] = 2
    for b in BEHAVIORS:
        results["behaviors"][b].update({"n_scenarios": 2, "n_items": 4})
    results["behaviors"][b0].update({"n_scenarios": 3, "n_items": 6})
    results["audited"] = {"expected": 26, "items": 24}; (cell / "results.json").write_text(json.dumps(results))
    r = _handoff("--spool", str(spool), "close-leg", "--identity", str(ident), "--cell-dir", str(cell))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "1 run(s) filed, 5 already generated, 1 recounted, 0 refused, of 7 registered" in r.stdout, r.stdout
    assert f"n_items_generated 4 -> 6 (increase: 2 terminal judge refusal(s) are generated transcripts" in r.stdout, r.stdout
    pooled = [p for rt, p in read_spool(spool) if rt == "run_patch" and p["run_id"] == rids[("petri_pooled", "prompted__constitution")]]
    assert pooled[-1]["fields"]["n_items_generated"] == 26                                       # 24 scored + two declined transcripts
    folded = {x["run_id"]: x for x in fold_spool_runs(reg["cluster_spool"], spool)}
    assert folded[rids[(f"petri_{b0}", "prompted__constitution")]]["n_items_generated"] == 6
    # the re-score of epoch 1: its three rows appear in rows.jsonl and its refusal leaves the sidecar -- every count stands
    rows = [json.loads(l) for l in (cell / "rows.jsonl").read_text().splitlines() if l.strip()]
    template = next(x for x in rows if x["task"] == f"petri/{b0}")
    with (cell / "rows.jsonl").open("a") as fh:
        for metric in (b0, "eval_awareness", "scenario_realism"):
            fh.write(json.dumps({**template, "item_id": f"{refused_sid}#e1", "metric": metric, "value": 5.0,
                                 "extra_json": json.dumps({"scenario_id": refused_sid, "epoch": 1, "dimension": metric})}) + "\n")
    sidecar = json.loads((cell / "petri_judge_refused.json").read_text()); sidecar["judge_refused"] = [x for x in sidecar["judge_refused"] if x["epoch"] != 1]
    (cell / "petri_judge_refused.json").write_text(json.dumps(sidecar))
    results["audited"] = {"expected": 26, "items": 25}; (cell / "results.json").write_text(json.dumps(results))
    r2 = _handoff("--spool", str(spool), "close-leg", "--identity", str(ident), "--cell-dir", str(cell))
    assert r2.returncode == 0 and "0 run(s) filed, 7 already generated, 0 recounted, 0 refused" in r2.stdout, r2.stdout
    # a record stating no epochs whose behaviours block derives none: refused by name, nothing filed
    results.pop("epochs")
    for b in BEHAVIORS:
        results["behaviors"][b].pop("n_items", None)                                              # no behaviour derives an epoch count
    (cell / "results.json").write_text(json.dumps(results))
    r3 = _handoff("--spool", str(spool), "close-leg", "--identity", str(ident), "--cell-dir", str(cell))
    assert r3.returncode == 3 and "carries no `epochs`" in r3.stderr, r3.stdout + r3.stderr
