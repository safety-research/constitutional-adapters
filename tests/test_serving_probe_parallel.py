"""The serving probe runs its launches in PARALLEL across the job's GPUs.

Attempt 1 of the stage-E probe (one standup at a time on one H200) passed ≈1 launch per 5.3 min — ≈11 h for the plan's
135 launches in a 4 h job. The launches are independent one-GPU standups of one bf16 model, so `serving_probe.py run`
gains `--parallel N` (default `auto` = the job's CUDA_VISIBLE_DEVICES count ÷ the plan's tp under Slurm, 1 outside it):
N launches in flight, each engine pinned — in the CHILD's environment alone — to a STRICT SUBSET of Slurm's device
list (never widened, never a hard-coded index: cluster hard rule 3), on its own API port and under its own launch dir;
the per-launch result and `probe_progress.json` are written under ONE lock as each launch finishes, so a preemption
loses at most the launches in flight and `--resume` re-runs exactly those. The launch order (priority), every record,
the verdict and the EXIT_INCOMPLETE / `--resume` contract are unchanged. The sbatch requests 8 GPUs (half the 16-GPU
`high` budget) and passes `--parallel` through positional 9; its successor keeps the width.

Everything here is CPU-only: fake engines through the `Deps` seam, a fake `nvidia-smi` on PATH, the carrier's text and
`bash -n`, the 03_hpo.sh probe block. Synthetic model ids, no cluster path, no host name."""
from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from sisteer.capeval.serving import standup_manifest as sm
from sisteer.extraction.lora_modules import ArchDims

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
sys.path.insert(0, str(REPO / "tests"))
import serving_probe as sp  # noqa: E402
from front_door_testlib import assert_no_tokens  # noqa: E402
from test_serving_probe_smoke import FakeExporter, _probe  # noqa: E402  (the smoke suite's fake exporter + probe rows)

SBATCH = REPO / "scripts/sbatch/serving_probe.sbatch"
HPO = REPO / "reproduce/03_hpo.sh"
TOY = ArchDims(hidden=64, attention_layers=1, q_out=128, kv_out=16, o_in=64, dense_mlp_layers=2, dense_intermediate=32)
REV = "0123456789abcdef0123456789abcdef01234567"
MODEL = "qwen35_4b_demo"       # the front door's worked example: the probe subject (a key of lora_modules.ARCH_DIMS_BY_MODEL_KEY; a hybrid)
#: the three-launch plan every concurrency test uses: the mlp r4 cell's three servers under the split topology (the
#: 2026-10-01 ruling) — S1cc (control + constitution), S1d (difference) and the shared S2 — at the ladder window, no
#: ladder points: three independent one-GPU standups, all natural (build_manifests emits no fallback)
ONE_WINDOW = (4096,)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """No overlay, no Slurm, no device list reaches a test from the shell; each test sets what it needs."""
    from sisteer.capeval.serving import geometry as g
    for v in (g.SITE_GEOMETRY_DIR_ENV, g.SITE_DIR_ENV, g.SHARED_ROOT_ENV, "SISTEER_MODELS_TABLE",
              sp.VISIBLE_DEVICES_ENV, "SLURM_JOB_ID"):
        monkeypatch.delenv(v, raising=False)


def _strip_union(plan):
    """The plan WITHOUT each pair's union manifest (the served launch class): this suite pins the concurrency
    contract — slots, ordering, the checkpoint — on the pair's three engines (S1cc, S1d, S2); the union's place in
    the plan is tests/test_serving_probe_smoke.py's pin (TestServedClassUnion), and a record's plan file is whatever was written."""
    union = {n for n, m in plan["manifests"].items() if m["topology"] == sm.TOPOLOGY_SINGLE}
    plan["manifests"] = {n: m for n, m in plan["manifests"].items() if n not in union}
    plan["launches"] = [r for r in plan["launches"] if r["manifest"] not in union]
    plan["n_launches"] = len(plan["launches"])
    for pr in plan["pairs"]:
        pr["manifests"] = [n for n in pr["manifests"] if n not in union]
    return plan


def _plan():
    plan = _strip_union(sp.build_plan(TOY, model_key=MODEL, model_id="org/toy", revision=REV, sets=("mlp",), ranks=(4,),
                                      windows=ONE_WINDOW, ladder_caps=()))
    assert plan["n_launches"] == 3 and [r["kind"] for r in plan["launches"]] == ["natural"] * 3
    assert [r["id"] for r in plan["launches"]] == ["mlp_r4_S1cc_cap8_L4096", "mlp_r4_S1d_cap8_L4096", "mlp_r4_S2_cap8_L4096"]
    assert [r["roles"] for r in plan["launches"]] == [["control", "constitution"], ["difference"], []]
    return plan


class _Killed(BaseException):
    """The process dying with a launch in flight (SIGTERM has no handler in the driver; a BaseException is the one thing
    `run_one_launch`'s broad `except Exception` does not turn into a recorded error)."""


def _fake_deps(plan, *, stand_up_hook=None, barrier=None):
    """The smoke suite's clean fake engine, recording per launch the `devices=` every GPU-touching effect received.
    `barrier`: a threading.Barrier every stand_up waits at — proof the launches were in flight TOGETHER.
    `stand_up_hook(launch_id, kw)`: called inside stand_up before the barrier (the kill test's seam)."""
    seen = {"stand_up": {}, "nvidia_smi": [], "wait_gpu_free": [], "order": []}
    lock = threading.Lock()

    def stand_up(**kw):
        lid = Path(kw["out_dir"]).name
        with lock:
            seen["stand_up"][lid] = kw.get("devices")
            seen["order"].append(lid)
        if stand_up_hook is not None:
            stand_up_hook(lid, kw)
        if barrier is not None:
            barrier.wait()
        (Path(kw["out_dir"]) / "vllm_server.log").write_text("INFO engine up\n")

        class L:
            model = "org/toy"
            base_url = "http://127.0.0.1:1"
            auth_headers = {}
            cmd = ("vllm", "serve")
            max_num_seqs = 4
        return L(), object()

    def worker_report(launch, d):
        rec = next(r for r in plan["launches"] if r["id"] == Path(d).name)
        m = sp.plan_manifest(plan, rec["manifest"])
        strides = sm.executing_strides(sp._Dims(plan["dims"]), m.target_class)      # the wrappers of the launch's own class
        return {"max_num_seqs": 4, "kv_num_gpu_blocks": 10, "kv_block_size": 16,
                "lora_execution_module_strides": {f"L0.{k}": dict(v) for k, v in strides.items()},
                "lora_execution_module_lora_buffer_bytes": int(rec["expected_lora_bytes"]["total"]),
                # the attested class a real lora-native engine reports (lora_max_lora_rank / lora_max_loras): the launch's own rank cap and its
                # manifest's adapter count — the fold refuses a passed launch without them (53bd76ed0); the smoke fake carries the same two
                "lora_max_lora_rank": int(rec["cap"]), "lora_max_loras": int(m.max_loras)}

    def chat(launch, body):
        genuine = body["model"] not in ("org/toy", "none__lora0") and "carrier" not in body["model"]
        p = _probe(first_lp=-0.3 if genuine else -0.1)
        return {"choices": [{"token_ids": p["token_ids"], "finish_reason": "length", "message": {"content": "x"},
                             "logprobs": {"content": [{"token": r["token"], "logprob": r["logprob"],
                                                       "top_logprobs": [{"token": t, "logprob": lp} for t, lp in r["top"]]}
                                                      for r in p["rows"]]}}]}

    def nvidia_smi(devices=None):
        with lock:
            seen["nvidia_smi"].append(devices)
        return {"used_mib": 1, "devices": devices}

    def wait_gpu_free(devices=None):
        with lock:
            seen["wait_gpu_free"].append(devices)
        return {"drained": True, "devices": devices}

    def reference(**kw):
        return {"status": "done", "peft_version": "0.20.0", "layouts": {},
                "per_adapter": {item["dir"]: {"status": "pass"} for item in kw["pending"]}}

    deps = sp.Deps(stand_up=stand_up, shut_down=lambda l, p: None, chat=chat,
                   list_models=lambda l: ["org/toy", "control__lora", "constitution__lora", "difference__lora",
                                          "none__lora0"] + [f"{r}__carrier" for r in sm.CARRIER_ROLES],
                   worker_report=worker_report, nvidia_smi=nvidia_smi, throughput=lambda l, **kw: [],
                   read_log=sp._read_log_tail, wait_gpu_free=wait_gpu_free, reference=reference)
    return deps, seen


# --------------------------------------------------------------------------- the device-subset rule


class TestDeviceSubsets:
    def test_visible_devices_reads_slurms_value_as_opaque_tokens(self, monkeypatch):
        assert sp.visible_devices() is None
        monkeypatch.setenv(sp.VISIBLE_DEVICES_ENV, "")
        assert sp.visible_devices() is None
        monkeypatch.setenv(sp.VISIBLE_DEVICES_ENV, "0,1,2,3,4,5,6,7")
        assert sp.visible_devices() == [str(i) for i in range(8)]
        # UUIDs / MIG ids are opaque tokens, spaces tolerated, order kept
        assert sp.visible_devices({sp.VISIBLE_DEVICES_ENV: " GPU-aaaa, GPU-bbbb "}) == ["GPU-aaaa", "GPU-bbbb"]

    def test_device_slots_are_consecutive_disjoint_strict_subsets(self):
        vis = [str(i) for i in range(8)]
        assert sp.device_slots(vis, per_standup=1) == [(str(i),) for i in range(8)]
        assert sp.device_slots(vis, per_standup=2) == [("0", "1"), ("2", "3"), ("4", "5"), ("6", "7")]
        # a remainder shorter than one standup stays idle; nothing is ever invented to fill it
        assert sp.device_slots(["0", "1", "2"], per_standup=2) == [("0", "1")]
        assert sp.device_slots(["0"], per_standup=2) == []
        for slots in (sp.device_slots(vis, per_standup=1), sp.device_slots(vis, per_standup=2)):
            flat = [d for sl in slots for d in sl]
            assert len(set(flat)) == len(flat) and set(flat) <= set(vis)      # disjoint, strict subsets
        with pytest.raises(ValueError, match="duplicate device tokens"):
            sp.device_slots(["0", "0"], per_standup=1)
        with pytest.raises(ValueError, match="per_standup must be >= 1"):
            sp.device_slots(vis, per_standup=0)

    def test_resolve_parallel_auto_is_the_allocation_under_slurm_and_one_outside(self):
        vis8 = [str(i) for i in range(8)]
        # auto under Slurm: one standup per allocated GPU (÷ tp)
        assert sp.resolve_parallel("auto", tp=1, visible=vis8, in_slurm=True) == (8, [(str(i),) for i in range(8)])
        assert sp.resolve_parallel(None, tp=2, visible=vis8, in_slurm=True) == (4, [("0", "1"), ("2", "3"), ("4", "5"), ("6", "7")])
        # auto outside Slurm: ONE standup — pinned to the first subset when a device list exists, unpinned otherwise
        assert sp.resolve_parallel("auto", tp=1, visible=["0", "1"], in_slurm=False) == (1, [("0",)])
        assert sp.resolve_parallel("auto", tp=1, visible=None, in_slurm=False) == (1, None)
        assert sp.resolve_parallel("auto", tp=1, visible=None, in_slurm=True) == (1, None)
        # the documented 2-GPU tp2 submission: auto = 1
        assert sp.resolve_parallel("auto", tp=2, visible=["0", "1"], in_slurm=True) == (1, [("0", "1")])

    def test_resolve_parallel_never_widens_and_never_hard_codes(self):
        vis = ["0", "1", "2"]
        assert sp.resolve_parallel("3", tp=1, visible=vis, in_slurm=True) == (3, [("0",), ("1",), ("2",)])
        assert sp.resolve_parallel(2, tp=1, visible=vis, in_slurm=False) == (2, [("0",), ("1",)])   # an explicit N: a strict subset
        with pytest.raises(ValueError, match="refusing to widen"):
            sp.resolve_parallel("4", tp=1, visible=vis, in_slurm=True)
        with pytest.raises(ValueError, match="refusing to widen"):
            sp.resolve_parallel("2", tp=2, visible=vis, in_slurm=True)                # 2 × tp2 = 4 devices of 3
        with pytest.raises(ValueError, match="nothing to pin to"):
            sp.resolve_parallel("2", tp=1, visible=None, in_slurm=True)
        with pytest.raises(ValueError, match="fewer than the 2 one standup needs"):
            sp.resolve_parallel("auto", tp=2, visible=["0"], in_slurm=True)
        with pytest.raises(ValueError, match="must be >= 1"):
            sp.resolve_parallel("0", tp=1, visible=vis, in_slurm=True)
        with pytest.raises(ValueError, match="integer or 'auto'"):
            sp.resolve_parallel("eight", tp=1, visible=vis, in_slurm=True)
        assert sp.device_tag(("3",)) == " [gpu 3]" and sp.device_tag(("4", "5")) == " [gpu 4,5]" and sp.device_tag(None) == ""

    def test_run_launches_refuses_overlapping_or_too_few_slots(self, tmp_path):
        plan = _plan()
        deps, _seen = _fake_deps(plan)
        with pytest.raises(ValueError, match="only 2 device slot"):
            sp.run_launches(plan, tmp_path, deps, gpu_mem_util=0.9, ready_timeout_s=1, time_budget_s=None,
                            max_launches=None, throughput=False, parallel=3, device_slots=[("0",), ("1",)])
        with pytest.raises(ValueError, match="device slots overlap"):
            sp.run_launches(plan, tmp_path, deps, gpu_mem_util=0.9, ready_timeout_s=1, time_budget_s=None,
                            max_launches=None, throughput=False, parallel=2, device_slots=[("0",), ("0",)])


# --------------------------------------------------------------------------- N concurrent fake launches


class TestConcurrentLaunches:
    def test_three_launches_in_flight_together_each_on_its_own_device_subset_one_checkpoint(self, tmp_path, capsys):
        """N=3 concurrent fake standups: a barrier only three SIMULTANEOUS standups can pass; each saw a distinct,
        disjoint slice of the device list; the record and the checkpoint hold all three exactly as a serial run
        would (same ids, same statuses, `complete`); the log lines carry the device subset."""
        plan = _plan()
        sp.build_adapter_dirs(plan, tmp_path, FakeExporter())
        barrier = threading.Barrier(3, timeout=30)
        deps, seen = _fake_deps(plan, barrier=barrier)
        slots = sp.device_slots(["0", "1", "2"], per_standup=1)
        results = sp.run_launches(plan, tmp_path, deps, gpu_mem_util=0.9, ready_timeout_s=1, time_budget_s=None,
                                  max_launches=None, throughput=False, parallel=3, device_slots=slots)
        ids = [r["id"] for r in plan["launches"]]
        assert sorted(results) == sorted(ids) and all(results[i]["status"] == "passed" for i in ids)
        assert not barrier.broken                                   # all three reached the barrier: in flight together
        # every launch was pinned to its own strict subset; the three subsets are disjoint and cover the slots
        devs = {lid: tuple(seen["stand_up"][lid]) for lid in ids}
        assert sorted(devs.values()) == sorted(slots) and len(set(devs.values())) == 3
        for lid in ids:
            assert results[lid]["devices"] == list(devs[lid])
            assert json.loads((tmp_path / sp.LAUNCHES_DIR / f"{lid}.json").read_text())["devices"] == list(devs[lid])
        # the residency read and the drain wait were on the launch's OWN devices, never a neighbour's
        assert sorted(seen["nvidia_smi"]) == sorted(devs.values())
        assert sorted(seen["wait_gpu_free"]) == sorted(devs.values())
        # ONE checkpoint, all three done, complete — the same record a serial run leaves
        prog = sp.load_progress(tmp_path)
        assert prog["complete"] is True and prog["n_done"] == 3 and prog["outstanding"] == [] and prog["done"] == ids
        assert prog["stopped_reason"] is None
        out = capsys.readouterr().out
        assert "parallel: 3 concurrent standup(s), device slots [0], [1], [2]" in out
        for lid in ids:
            assert re.search(rf"launch {re.escape(lid)} \[gpu {devs[lid][0]}\]: natural ", out), lid
            assert re.search(rf"launch {re.escape(lid)} \[gpu {devs[lid][0]}\]: passed", out), lid
        # the verdict over the parallel record reads exactly as a serial one's: both roles and the S2 passed at the ladder window
        # and certify by the same three launches; the pair itself is excluded for the (8, 4) class this suite's stripped plan never
        # planned (the served launch class — the union's coverage is the smoke suite's pin, not this contract's)
        sp.reference_phase(plan, tmp_path, deps)
        v, allowed = sp.write_verdict(plan, sp.load_results(tmp_path), probe_job_id="7", out=tmp_path)
        assert v["status"] == "complete" and allowed["pairs"] == [] and sp.pair_serving_status(plan, sp.load_results(tmp_path), plan["pairs"][0])["certified_all"]
        e = v["excluded"][0]
        assert e["topology"] == "S1ccS1dS2" and e["certified_by"] == ids and e["decided"] is True
        assert e["reasons"] == ["served_class_unplanned_at_L=4096:(rank cap 8, 4 adapters)"]

    def test_launches_start_in_priority_order_and_the_shared_s2_waits_for_a_freed_slot(self, tmp_path):
        """The identity/order contract under the split topology: with fewer slots than launches the starts follow the
        plan's priority order — the pair's two servers (S1cc, S1d; priority 10 at the ladder window) are in flight
        TOGETHER on the two slots and the shared S2 (priority 30) starts only once one of them frees a slot. No launch
        is conditional any more: build_manifests emits S1cc + S1d + S2 and no VRAM fallback, so nothing is deferred on
        a parent's verdict and nothing is ever `skipped`."""
        plan = _strip_union(sp.build_plan(sp.dims_for(MODEL), model_key=MODEL, model_id="org/toy", revision=REV,
                                          sets=("mlp",), ranks=(64,), windows=(4096,), ladder_caps=()))
        ids = [r["id"] for r in plan["launches"]]
        assert ids == ["mlp_r64_S1cc_cap64_L4096", "mlp_r64_S1d_cap128_L4096", "mlp_r64_S2_cap8_L4096"]
        assert [r["kind"] for r in plan["launches"]] == ["natural"] * 3 and all(r["condition"] is None for r in plan["launches"])
        assert [r["priority"] for r in plan["launches"]] == [10, 10, 30]
        assert [r["cap"] for r in plan["launches"]] == [64, 128, 8]                 # the real dims: cap r, cap 2r, the S2 cap
        sp.build_adapter_dirs(plan, tmp_path, FakeExporter())
        together = threading.Barrier(2, timeout=30)

        def hold_pair(lid, kw):
            if lid != ids[2]:
                together.wait()            # only the two servers reach it: proof they were in flight at the same time

        deps, seen = _fake_deps(plan, stand_up_hook=hold_pair)
        results = sp.run_launches(plan, tmp_path, deps, gpu_mem_util=0.9, ready_timeout_s=1, time_budget_s=None,
                                  max_launches=None, throughput=False, parallel=2, device_slots=[("0",), ("1",)])
        started = seen["order"]
        assert not together.broken and sorted(started[:2]) == sorted(ids[:2]) and started[2] == ids[2]
        assert all(results[i]["status"] == "passed" for i in ids) and not any(r["status"] == "skipped" for r in results.values())
        # the two servers took the two slots; the S2 took whichever one of them freed
        assert {seen["stand_up"][i] for i in ids[:2]} == {("0",), ("1",)} and seen["stand_up"][ids[2]] in {("0",), ("1",)}
        assert sp.load_progress(tmp_path)["complete"] is True

    def test_a_kill_mid_flight_keeps_the_finished_launches_and_resume_reruns_only_the_lost_one(self, tmp_path, monkeypatch):
        """Three launches in flight; the process dies (a BaseException out of one standup) once the other two have
        finished and been checkpointed: `probe_progress.json` names exactly the lost launch as outstanding, its
        result file does not exist, the two finished ones do. `run --resume --parallel 3` (the CLI, under a
        3-device list) re-runs ONLY the lost launch, and the record completes."""
        plan = _plan()
        sp.write_json_atomic(tmp_path / sp.PLAN_FILE, plan)
        sp.build_adapter_dirs(plan, tmp_path, FakeExporter())
        ids = [r["id"] for r in plan["launches"]]
        lost = ids[2]

        def die_last(lid, kw):
            if lid != lost:
                return
            t0 = time.time()
            while time.time() - t0 < 30:
                prog = sp.load_progress(tmp_path)
                if prog and prog.get("n_done", 0) >= 2:
                    raise _Killed(lid)
                time.sleep(0.01)
            raise AssertionError("the other two launches never checkpointed")

        deps, seen = _fake_deps(plan, stand_up_hook=die_last)
        slots = sp.device_slots(["0", "1", "2"], per_standup=1)
        with pytest.raises(_Killed):
            sp.run_launches(plan, tmp_path, deps, gpu_mem_util=0.9, ready_timeout_s=1, time_budget_s=None,
                            max_launches=None, throughput=False, parallel=3, device_slots=slots)
        assert sorted(seen["stand_up"]) == sorted(ids)              # all three were in flight
        prog = sp.load_progress(tmp_path)
        assert prog["complete"] is False and prog["n_done"] == 2 and sorted(prog["done"]) == sorted(ids[:2])
        assert [o["id"] for o in prog["outstanding"]] == [lost] and prog["decision_outstanding"] == [lost]
        assert not (tmp_path / sp.LAUNCHES_DIR / f"{lost}.json").exists()
        for lid in ids[:2]:
            assert json.loads((tmp_path / sp.LAUNCHES_DIR / f"{lid}.json").read_text())["status"] == "passed"
        # a bare `run` over the partial record still refuses; the verdict is still PARTIAL (the contract is unchanged)
        v, _allowed = sp.write_verdict(plan, sp.load_results(tmp_path), probe_job_id="1", out=tmp_path / "v")
        assert v["status"] == "partial"
        # --resume through the CLI under a 3-device list: exactly the lost launch re-runs, pinned to a strict subset
        deps2, seen2 = _fake_deps(plan)
        monkeypatch.setattr(sp, "ProductionExporter", lambda *a, **k: FakeExporter())
        monkeypatch.setattr(sp.Deps, "production", classmethod(lambda cls: deps2))
        monkeypatch.setenv(sp.VISIBLE_DEVICES_ENV, "0,1,2")
        monkeypatch.setenv("SLURM_JOB_ID", "424242")
        base = ["run", "--model-key", MODEL, "--out", str(tmp_path), "--no-throughput"]
        with pytest.raises(SystemExit, match="INCOMPLETE record .* pass --resume"):
            sp.main(base)
        rc = sp.main(base + ["--resume", "--parallel", "3"])
        assert rc == 0
        assert list(seen2["stand_up"]) == [lost] and seen2["stand_up"][lost] in {("0",), ("1",), ("2",)}
        prog = sp.load_progress(tmp_path)
        assert prog["complete"] is True and prog["outstanding"] == [] and prog["done"] == ids
        rc = sp.main(["verdict", "--model-key", MODEL, "--out", str(tmp_path), "--probe-job-id", "2"])
        assert rc == 0 and (tmp_path / sp.ALLOWED_PAIRS_FILE).is_file()

    def test_the_time_budget_stops_new_starts_and_drains_the_launches_in_flight(self, tmp_path):
        """The budget is checked before each START (as before); launches already in flight finish and are recorded;
        the checkpoint says time_budget with the never-started launches outstanding; nothing is killed."""
        plan = _plan()
        sp.build_adapter_dirs(plan, tmp_path, FakeExporter())
        gate = threading.Event()

        def hold(lid, kw):
            gate.wait(30)

        deps, seen = _fake_deps(plan, stand_up_hook=hold)
        clock = {"t": 0.0}
        main = threading.main_thread()

        def now():
            # the driver's budget checks (this thread) advance the clock 1,000 s per read; a worker's own timing reads
            # (run_one_launch's t0 / ready_s / elapsed_s) only read it — the arithmetic below is exact, never a race
            if threading.current_thread() is main:
                clock["t"] += 1000.0
            return clock["t"]

        slow = dataclasses.replace(deps, now=now)
        # two slots: the first two launches start (budget checks at 1,000 s and 2,000 s of a 2,500 s budget); the third is
        # checked only after a slot frees — by then the clock has passed the budget — so it never starts
        threading.Timer(0.2, gate.set).start()
        results = sp.run_launches(plan, tmp_path, slow, gpu_mem_util=0.9, ready_timeout_s=1, time_budget_s=2500.0,
                                  max_launches=None, throughput=False, parallel=2, device_slots=[("0",), ("1",)])
        ids = [r["id"] for r in plan["launches"]]
        assert sorted(results) == sorted(ids[:2]) and all(results[i]["status"] == "passed" for i in ids[:2])
        assert sorted(seen["stand_up"]) == sorted(ids[:2])
        prog = sp.load_progress(tmp_path)
        assert prog["complete"] is False and prog["stopped_reason"] == "time_budget"
        assert [o["id"] for o in prog["outstanding"]] == [ids[2]]

    def test_max_launches_counts_starts(self, tmp_path):
        plan = _plan()
        sp.build_adapter_dirs(plan, tmp_path, FakeExporter())
        deps, seen = _fake_deps(plan)
        results = sp.run_launches(plan, tmp_path, deps, gpu_mem_util=0.9, ready_timeout_s=1, time_budget_s=None,
                                  max_launches=2, throughput=False, parallel=3, device_slots=sp.device_slots(["0", "1", "2"], per_standup=1))
        assert len(results) == 2 and len(seen["stand_up"]) == 2
        assert sp.load_progress(tmp_path)["stopped_reason"] == "max_launches"


# --------------------------------------------------------------------------- the CLI's resolution


class TestCli:
    def _record(self, tmp_path, name):
        out = tmp_path / name
        plan = _plan()
        sp.write_json_atomic(out / sp.PLAN_FILE, plan)
        sp.build_adapter_dirs(plan, out, FakeExporter())
        return out, plan

    def test_auto_is_the_allocation_under_slurm_one_outside_and_a_wider_request_refuses(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(sp, "ProductionExporter", lambda *a, **k: FakeExporter())
        base = ["run", "--model-key", MODEL, "--no-throughput", "--no-reference", "--out"]
        # under Slurm with three devices: auto = 3, one launch per device
        out, plan = self._record(tmp_path, "slurm")
        deps, seen = _fake_deps(plan)
        monkeypatch.setattr(sp.Deps, "production", classmethod(lambda cls: deps))
        monkeypatch.setenv(sp.VISIBLE_DEVICES_ENV, "0,1,2")
        monkeypatch.setenv("SLURM_JOB_ID", "1")
        assert sp.main(base + [str(out)]) == 0
        assert sorted(seen["stand_up"].values()) == [("0",), ("1",), ("2",)]
        log = capsys.readouterr().out
        assert "parallel=3 (requested auto; CUDA_VISIBLE_DEVICES=0,1,2; tp=1) device slots=[0], [1], [2]" in log
        # the same device list OUTSIDE Slurm: auto = 1, every launch pinned to the FIRST device (a strict subset, still)
        out, plan = self._record(tmp_path, "grading_host")
        deps, seen = _fake_deps(plan)
        monkeypatch.setattr(sp.Deps, "production", classmethod(lambda cls: deps))
        monkeypatch.delenv("SLURM_JOB_ID")
        assert sp.main(base + [str(out)]) == 0
        assert set(seen["stand_up"].values()) == {("0",)}
        assert "parallel=1 (requested auto; CUDA_VISIBLE_DEVICES=0,1,2; tp=1) device slots=[0]" in capsys.readouterr().out
        # no device list at all: auto = 1, unpinned (the older single standup)
        out, plan = self._record(tmp_path, "nodev")
        deps, seen = _fake_deps(plan)
        monkeypatch.setattr(sp.Deps, "production", classmethod(lambda cls: deps))
        monkeypatch.delenv(sp.VISIBLE_DEVICES_ENV)
        assert sp.main(base + [str(out)]) == 0
        assert set(seen["stand_up"].values()) == {None} and all(r["devices"] is None for r in sp.load_results(out).values())
        assert "parallel=1 (requested auto; CUDA_VISIBLE_DEVICES=<unset>; tp=1) device slots=[unpinned]" in capsys.readouterr().out
        # refusals happen BEFORE any standup: wider than the allocation; N > 1 with nothing to pin to
        out, plan = self._record(tmp_path, "refuse")
        deps, seen = _fake_deps(plan)
        monkeypatch.setattr(sp.Deps, "production", classmethod(lambda cls: deps))
        monkeypatch.setenv(sp.VISIBLE_DEVICES_ENV, "0,1,2")
        monkeypatch.setenv("SLURM_JOB_ID", "1")
        with pytest.raises(SystemExit, match="--parallel 4 needs 4 device.*refusing to widen"):
            sp.main(base + [str(out), "--parallel", "4"])
        monkeypatch.delenv(sp.VISIBLE_DEVICES_ENV)
        with pytest.raises(SystemExit, match="--parallel 2 needs CUDA_VISIBLE_DEVICES.*nothing to pin to"):
            sp.main(base + [str(out), "--parallel", "2"])
        assert seen["stand_up"] == {} and sp.load_progress(out) is None
        # the parser's default is `auto`
        assert sp.build_parser().parse_args(["run", "--model-key", MODEL]).parallel == sp.PARALLEL_AUTO


# --------------------------------------------------------------------------- the production effects (no vllm)


class TestProductionEffects:
    def test_stand_up_pins_the_subset_in_the_child_env_only_and_records_it(self, tmp_path, monkeypatch):
        """`_production_stand_up(devices=)`: the subset reaches `launch_server` as `extra_env` (the CHILD's environment),
        this process's CUDA_VISIBLE_DEVICES is untouched, launch_cmd.json names the devices; a subset outside the
        visible list or of the wrong width is a LaunchRefused, before any spawn."""
        from sisteer.capeval.serving import native_standup as ns
        from sisteer.capeval.serving import vllm_server as vs
        monkeypatch.setenv(sp.VISIBLE_DEVICES_ENV, "0,1,2,3")
        plan = _plan()
        manifest = sp.plan_manifest(plan, plan["launches"][0]["manifest"])
        calls = {}
        monkeypatch.setattr(sp, "_chat_template_bytes", lambda m, r: b"{{ messages }}")
        monkeypatch.setattr(sp, "_tree_sha", lambda: "deadbeef")
        def write_chat_template(out, b):   # the real one creates the launch dir it writes into
            Path(out).mkdir(parents=True, exist_ok=True)
            return Path(out) / "chat_template.jinja", "0" * 16
        monkeypatch.setattr(ns, "write_chat_template", write_chat_template)
        monkeypatch.setattr(vs, "launch_only_spec", lambda *a, **k: object())
        monkeypatch.setattr(vs, "multimodal_target", lambda m, r: False)

        def build(spec, *, port, **kw):
            # the composer's launch mirrors the manifest's engine facts (the class, the cap, the slot count)
            return vs.ServerLaunch(cmd=("vllm", "serve", "org/toy", "--port", str(port)), env={}, host="127.0.0.1", port=port,
                                   model="org/toy", lora_native=True, lora_target_modules=tuple(manifest.target_class),
                                   max_lora_rank=manifest.max_lora_rank, max_loras=manifest.max_loras, max_num_seqs=kw["max_num_seqs"])
        monkeypatch.setattr(vs, "build_server_launch", build)

        def launch_server(launch, *, log_path, ready_timeout_s, extra_env=None):
            calls["extra_env"] = extra_env
            calls["port"] = launch.port
            return object()
        monkeypatch.setattr(vs, "launch_server", launch_server)
        adapters = {a.name: tmp_path / a.name for a in manifest.adapters}
        launch, _proc = sp._production_stand_up(model_id="org/toy", revision=REV, adapter_dirs=adapters, manifest=manifest, L=4096,
                                                S=4, out_dir=tmp_path / "l1", gpu_mem_util=0.9, ready_timeout_s=1, tp=1,
                                                devices=("3",))
        assert calls["extra_env"] == {"CUDA_VISIBLE_DEVICES": "3"}
        assert os.environ[sp.VISIBLE_DEVICES_ENV] == "0,1,2,3"           # the parent's own environment: never changed
        rec = json.loads((tmp_path / "l1" / "launch_cmd.json").read_text())
        assert rec["devices"] == ["3"] and rec["port"] == launch.port == calls["port"]
        assert "--gpu-memory-utilization" in rec["cmd"]                     # the front-owned extras ride the composed command
        # unpinned: no extra_env, devices recorded as null
        sp._production_stand_up(model_id="org/toy", revision=REV, adapter_dirs=adapters, manifest=manifest, L=4096, S=4,
                                out_dir=tmp_path / "l2", gpu_mem_util=0.9, ready_timeout_s=1, tp=1)
        assert calls["extra_env"] is None and json.loads((tmp_path / "l2" / "launch_cmd.json").read_text())["devices"] is None
        # refusals: a device the allocation does not hold; a subset of the wrong width for tp
        with pytest.raises(sp.LaunchRefused, match="devices_not_visible"):
            sp._production_stand_up(model_id="org/toy", revision=REV, adapter_dirs=adapters, manifest=manifest, L=4096, S=4,
                                    out_dir=tmp_path / "l3", gpu_mem_util=0.9, ready_timeout_s=1, tp=1, devices=("7",))
        with pytest.raises(sp.LaunchRefused, match="devices_tp_mismatch"):
            sp._production_stand_up(model_id="org/toy", revision=REV, adapter_dirs=adapters, manifest=manifest, L=4096, S=4,
                                    out_dir=tmp_path / "l4", gpu_mem_util=0.9, ready_timeout_s=1, tp=2, devices=("3",))
        # concurrent standups never share an API port: the picker hands out distinct ports under one lock
        ports = [sp._pick_port() for _ in range(64)]
        assert len(set(ports)) == 64 and all(1024 <= p <= 65535 for p in ports)

    def test_nvidia_smi_and_the_drain_wait_read_the_launches_own_devices(self, tmp_path, monkeypatch):
        """A fake nvidia-smi on PATH records its arguments: the pinned subset rides `-i <tokens>` (both tokens of a tp2
        slot), `used_mib` sums, `per_device` keeps each, and the drain wait drains on the SUBSET's own residency."""
        bin_dir = tmp_path / "bin"; bin_dir.mkdir()
        argfile = tmp_path / "smi_args"
        (bin_dir / "nvidia-smi").write_text(
            f'#!/bin/bash\necho "$*" >> "{argfile}"\n'
            'case "$*" in *"-i 4,5"*) printf "100, 141000\\n200, 141000\\n";; *"-i 0"*) printf "7000, 141000\\n";; *) printf "1, 141000\\n";; esac\n')
        (bin_dir / "nvidia-smi").chmod(0o755)
        monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
        rep = sp._production_nvidia_smi(devices=("4", "5"))
        assert rep["used_mib"] == 300 and rep["total_mib"] == 282000 and rep["devices"] == ["4", "5"]
        assert rep["per_device"] == [{"used_mib": 100, "total_mib": 141000}, {"used_mib": 200, "total_mib": 141000}]
        assert argfile.read_text().splitlines()[-1].endswith("-i 4,5")
        # unpinned: the first token of this process's own list (the single-standup default), else no -i at all
        monkeypatch.setenv(sp.VISIBLE_DEVICES_ENV, "0,1")
        assert sp._production_nvidia_smi()["used_mib"] == 7000 and argfile.read_text().splitlines()[-1].endswith("-i 0")
        monkeypatch.delenv(sp.VISIBLE_DEVICES_ENV)
        sp._production_nvidia_smi()
        assert "-i" not in argfile.read_text().splitlines()[-1]
        # the drain wait: device 0 holds 7,000 MiB (>= the 6,000 threshold) -> not drained on a 0 s budget; the tp2 slot drains
        w = sp._production_wait_gpu_free(threshold_mib=6000, timeout_s=0.0, devices=("0",))
        assert w["drained"] is False and w["used_mib"] == 7000 and w["devices"] == ["0"]
        w = sp._production_wait_gpu_free(threshold_mib=6000, timeout_s=5.0, devices=("4", "5"))
        assert w["drained"] is True and w["used_mib"] == 200 and w["devices"] == ["4", "5"]


# --------------------------------------------------------------------------- the carrier and the front door


class TestCarrier:
    def test_sbatch_requests_eight_gpus_and_passes_parallel_through(self):
        text = SBATCH.read_text()
        assert subprocess.run(["bash", "-n", str(SBATCH)], capture_output=True, text=True).returncode == 0
        # the header: 8 GPUs, ~16 cpus and ~128G per GPU, never --mem=0, high, no-requeue, the append log
        assert "#SBATCH --gres=gpu:8" in text and "#SBATCH --cpus-per-task=128" in text and "#SBATCH --mem=1024G" in text
        assert "--mem=0" not in text and "#SBATCH --qos=high" in text and "#SBATCH --no-requeue" in text
        assert "#SBATCH --open-mode=append" in text and "#SBATCH --gres=gpu:1" not in text
        # hard rule 1: no --export on any command; hard rule 3: CUDA_VISIBLE_DEVICES is READ (the count, the tp2 guard,
        # the log line), never assigned or exported by the carrier — the driver pins the child engines
        assert re.search(r"^[^#\n]*--export=", text, re.M) is None
        assert re.search(r"^\s*(export\s+)?CUDA_VISIBLE_DEVICES=", text, re.M) is None and "export CUDA_VISIBLE_DEVICES" not in text
        assert 'NGPU="$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr \',\' \'\\n\' | grep -c .)"' in text
        # positional 9 (parallel, default auto), validated, passed to the driver, on the successor's line with THIS job's width
        assert 'PARALLEL="${9:-auto}"' in text and 'case "$PARALLEL" in' in text
        run_line = text[text.index("srun --ntasks=1"):text.index("WORK=$!")]
        assert '--parallel "$PARALLEL"' in run_line and '"${RESUME[@]}"' in run_line
        esc = text[text.index("escalate(){"):text.index("WORK=\"\"\nhandled=0")]
        assert '--gres="gpu:$NGPU"' in esc and '--gres="gpu:$TP"' not in esc
        assert '"$((ATTEMPT + 1))" "$MAX_ATTEMPTS" "$PARALLEL")' in esc
        assert 'if sentinel="$(stop_requested)"' in esc                 # the STOP contract is untouched
        assert "parallel=$PARALLEL gpus=$NGPU" in text                  # the job's log line names both
        assert_no_tokens(text, str(SBATCH))

    def test_the_carriers_width_and_parallel_validation_run_under_bash(self, tmp_path):
        """RUN the carrier's own NGPU / MODE / PARALLEL block: the GPU count is Slurm's device count (read, never set),
        tp2 needs two of them, `parallel` must be auto or a positive integer, and no line sets CUDA_VISIBLE_DEVICES."""
        text = SBATCH.read_text()
        frag = text[text.index('NGPU="$(echo'):text.index('# the run\'s time budget')]
        assert "CUDA_VISIBLE_DEVICES=" not in frag.replace("CUDA_VISIBLE_DEVICES='", "")

        def run(mode, parallel, cvd=None, gpus_on_node=None, token=""):
            env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path)}
            if cvd is not None:
                env["CUDA_VISIBLE_DEVICES"] = cvd
            if gpus_on_node is not None:
                env["SLURM_GPUS_ON_NODE"] = gpus_on_node
            script = (f'set -u; MODE={mode}; PARALLEL={parallel}; REOPEN_TOKEN={token}; FINAL_RC=; log(){{ echo "[log] $*"; }}\n{frag}\n'
                      'echo "NGPU=$NGPU TP=$TP PARALLEL=$PARALLEL CVD=${CUDA_VISIBLE_DEVICES:-unset} REOPEN=${REOPEN[*]:-none}"')
            return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=60)

        r = run("tp1", "auto", cvd="0,1,2,3,4,5,6,7")
        assert r.returncode == 0 and "NGPU=8 TP=1 PARALLEL=auto CVD=0,1,2,3,4,5,6,7" in r.stdout, r.stdout + r.stderr
        r = run("tp2", "auto", cvd="0,1,2,3,4,5,6,7")
        assert r.returncode == 0 and "NGPU=8 TP=2" in r.stdout
        r = run("tp1", "4", cvd="0,1,2,3")
        assert r.returncode == 0 and "NGPU=4 TP=1 PARALLEL=4" in r.stdout
        r = run("tp2", "auto", cvd="0")                                   # the tp2 guard: two GPUs or a refusal
        assert r.returncode == 2 and "mode tp2 needs 2 GPUs" in r.stdout
        r = run("tp1", "eight", cvd="0")                                  # parallel: auto | a positive integer
        assert r.returncode == 2 and "positional 9 parallel must be auto or a positive integer" in r.stdout
        r = run("tp1", "0", cvd="0")
        assert r.returncode == 2
        r = run("tp1", "auto", cvd=None, gpus_on_node="2")               # no device list: Slurm's count, else tp
        assert r.returncode == 0 and "NGPU=2 TP=1" in r.stdout and "CVD=unset" in r.stdout
        r = run("tp1", "auto")
        assert r.returncode == 0 and "NGPU=1 TP=1" in r.stdout
        # positional 10 (the reopen token, re-checks 2/3): empty -> nothing passed; the receipt's 32-hex nonce -> --reopen-token; a
        # 64-hex generation (the typed digest) or anything else refused — and the refusal never echoes the value
        assert "REOPEN=none" in r.stdout
        r = run("tp1", "auto", cvd="0", token="f" * 32)
        assert r.returncode == 0 and f"REOPEN=--reopen-token {'f' * 32}" in r.stdout
        for bad in ("f" * 64, "not-a-nonce"):
            r = run("tp1", "auto", cvd="0", token=bad)
            assert r.returncode == 2 and "positional 10 reopen_token must be the revocation receipt's 32-hex nonce" in r.stdout and bad not in r.stdout

    def test_03_hpo_probe_submission_keeps_the_headers_width(self):
        """`03_hpo.sh --phase probe` has no GPU knob in the cluster block: by DEFAULT its fresh submission passes no --gres and no
        positional `parallel` — the sbatch header's 8 GPUs and `auto` decide the width; `--probe-width N` puts the carrier's
        documented narrower allocation on the command line (probe_width_flags; executed in tests/test_front_door_probe_resume.py)."""
        src = HPO.read_text()
        assert subprocess.run(["bash", "-n", str(HPO)], capture_output=True, text=True).returncode == 0
        probe = src[src.index("# ================================================================================================================ probe"):
                    src.index("# ================================================================================================================ recon")]
        line = next(ln for ln in probe.splitlines() if 'submit "$(unit probe)"' in ln)
        assert "--gres" not in line and line.rstrip().endswith('--env "SERVING_PROBE_OUT=$CL_ROOT"')
        assert '"$MODEL $SRV_REPO $SRV_REV"' in line                     # positionals 4-9 at their defaults
        assert '--flags "$PROBE_FLAGS"' in line and 'PROBE_FLAGS="$(probe_width_flags "${PROBE_WIDTH:-auto}")"' in probe
        assert "the sbatch header's (8 GPUs, `parallel` auto" in probe and "no per-job GPU knob" in probe
        flags = src[src.index("probe_width_flags(){"):src.index("probe_resume(){")]
        assert 'auto|"") return 0' in flags and "--gres=gpu:%d --cpus-per-task=%d --mem=%dG" in flags and "16 * $1" in flags and "128 * $1" in flags
        assert_no_tokens(probe, "the probe block of 03_hpo.sh")
