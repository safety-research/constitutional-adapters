"""Cheap-first scheduling (wave-autopsy 2026-08-22, lesson 3).

In the 2026-08-22 GPU wave, pure longest-first placement put the 31k-step
gradient arms ahead of diffmean on EVERY pod; when the wave was cut at
6.5h, the CHEAPEST cells — minutes of forward passes — were the missing
ones and had to be re-fit on a salvage pod. The scheduler is now a HYBRID:
arms whose static cost falls strictly below
``data_parallel.cheap_first_below`` (the closed-form / capture-only arms)
are scheduled FIRST as a batch — no main arm starts while any cheap arm is
still pending — and the remaining arms keep the exact longest-first order.
ORDERING ONLY: widths, batch sizes, resume-by-skip, the world-size pin and
every fit are untouched.

CPU-only and GPU-free: bare Namespaces + the module's REAL cost table, the
way tests/test_ddp_fit.py already exercises plan_parallel/run_parallel_fit.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_two_si_objects as tt

# The full 13-arm Class A set as configs/two_si.yaml records it (the wave
# this policy exists for), in config order.
CLASS_A_13 = ["psr_kl_all", "psr_mse_all", "lora_r1", "loreft", "lofit",
              "token_bos", "alphakl_null", "alphakl_soft", "diffmean",
              "bidpo", "vector", "bidpo_lora", "token_faithful"]


def _cost(arm: str) -> float:
    return tt._ARM_COST.get(arm, tt._DEFAULT_ARM_COST)


def _wave_cfg(**dp_extra) -> dict:
    """The 2026-08-22 wave's scheduling shape: a 4-GPU pod, width 1 for
    every arm (the measured launch config), so launch order == queue
    order and the placement policy is isolated from width effects."""
    dp = {"gpus": 4, "arm_gpus": {a: 1 for a in CLASS_A_13}}
    dp.update(dp_extra)
    return {"data_parallel": dp,
            "train_kwargs": {"default": {"batch_size": 4}}}


def _old_longest_first_plan(cfg: dict, si_id: str, arms: list[str]):
    """The PRE-cheap-first scheduler, reimplemented verbatim (the exact
    algorithm plan_parallel/run_parallel_fit shipped with on accel-fit):
    sort by (-est_wall, arm), greedy first-fit at every free-up, no phase
    barrier. The regression pin below asserts byte-identical placement
    against it whenever no arm is cheap."""
    dp = cfg.get("data_parallel") or {}
    gpus = int(dp.get("gpus", 1) or 1)
    tasks = []
    for arm in arms:
        w = tt.arm_width(cfg, si_id, arm, gpus)
        cost = _cost(arm)
        tasks.append({"arm": arm, "width": w, "cost": cost,
                      "est_wall": cost / w})
    pending = sorted(tasks, key=lambda t: (-t["est_wall"], t["arm"]))
    free = gpus
    running: list[tuple[float, dict]] = []
    now = 0.0
    order = []
    while pending or running:
        started = True
        while started:
            started = False
            for t in list(pending):
                if t["width"] <= free:
                    free -= t["width"]
                    running.append((now + t["est_wall"], t))
                    order.append({**t, "start": round(now, 3)})
                    pending.remove(t)
                    started = True
                    break
        if not running:
            break
        running.sort(key=lambda e: e[0])
        now, done = running.pop(0)
        free += done["width"]
    makespan = max((o["start"] + o["est_wall"] for o in order), default=0.0)
    return order, round(makespan, 3)


def _project(tasks: list[dict]) -> list[tuple]:
    """A task list on the OLD schema's keys (the new dicts carry an extra
    'phase'), so old and new placements compare byte-for-byte."""
    return [(t["arm"], t["width"], t["cost"], t["est_wall"], t["start"])
            for t in tasks]


# --------------------------------------------------------------------------- #
# (a) The cheap batch owns the earliest slots on the wave's 4-GPU pool          #
# --------------------------------------------------------------------------- #


def test_cheap_arms_occupy_the_earliest_slots_on_a_4gpu_pool():
    cfg = _wave_cfg()
    thr = tt.cheap_first_threshold(cfg)
    cheap = {a for a in CLASS_A_13 if _cost(a) < thr}
    # The threshold's whole point: the closed-form capture arms — and ONLY
    # them — form the cheap batch (diffmean is the arm the autopsy is about).
    assert cheap == {"diffmean", "alphakl_null", "alphakl_soft"}

    plan = tt.plan_parallel(cfg, "safety_refusal", CLASS_A_13)
    order = [t["arm"] for t in plan["tasks"]]
    assert set(order[:len(cheap)]) == cheap, \
        "the cheap closed-form arms must occupy the earliest launch slots"
    for t in plan["tasks"]:
        assert t["phase"] == ("cheap-first" if t["arm"] in cheap else "main")
        if t["arm"] in cheap:
            # Width 1 on a 4-GPU pool: all three cheap arms start at t=0.
            assert t["start"] == 0.0, \
                f"{t['arm']} (cheap) must start immediately, not at " \
                f"{t['start']}"


def test_real_config_classifies_the_same_cheap_batch():
    """configs/two_si.yaml's data_parallel block (the wave's record, now
    carrying cheap_first_below explicitly) yields the same cheap batch and
    puts diffmean in the earliest slots."""
    cfg = tt._load_cfg("configs/two_si.yaml")
    arms = list(cfg["class_a_arms"])
    # + lora_r4 (2026-08-28, the staged rank-4 objects' canonical cell id);
    # cost 7.0 in _ARM_COST, so the cheap batch below is unchanged
    assert set(arms) == set(CLASS_A_13) | {"lora_r4"}
    thr = tt.cheap_first_threshold(cfg)
    cheap = {a for a in arms if _cost(a) < thr}
    assert cheap == {"diffmean", "alphakl_null", "alphakl_soft"}
    plan = tt.plan_parallel(cfg, "safety_refusal", arms)
    order = [t["arm"] for t in plan["tasks"]]
    assert set(order[:3]) == cheap
    starts = {t["arm"]: t["start"] for t in plan["tasks"]}
    assert max(starts[a] for a in cheap) <= \
        min(starts[a] for a in arms if a not in cheap)


def test_no_main_arm_starts_while_a_cheap_arm_is_pending():
    """The phase barrier: every cheap arm starts before (or with) any main
    arm — the starvation that lost the 2026-08-22 wave's diffmean.

    (accel-fit's original exercised the corner where mixed WIDTHS could
    defeat a naive sort — captures at 4, gradients at 2. This trunk packs
    arms 1-per-GPU only — extraction.ddp row-sharding deliberately not
    ported — so every width is pinned 1 and the queue head is always
    launchable; the barrier's observable contract is the start-time
    ordering, asserted on a pool narrower than the cheap batch so the
    barrier actually holds mains back a tick.)"""
    cfg = {"data_parallel": {"gpus": 2},
           "train_kwargs": {"default": {"batch_size": 2}}}
    arms = ["diffmean", "alphakl_null", "alphakl_soft",
            "lofit", "psr_mse_all", "bidpo"]
    plan = tt.plan_parallel(cfg, "safety_refusal", arms)
    by = {t["arm"]: t for t in plan["tasks"]}
    # This trunk's width pin: every arm at 1, whatever its family.
    assert all(t["width"] == 1 for t in plan["tasks"]), \
        f"widths drifted from the 1-per-GPU pin: {by}"
    cheap_starts = [t["start"] for t in plan["tasks"]
                    if t["phase"] == "cheap-first"]
    main_starts = [t["start"] for t in plan["tasks"] if t["phase"] == "main"]
    assert cheap_starts and main_starts
    # 3 cheap arms on a 2-slot pool: the third cheap arm waits for a free
    # slot, and no main may claim one before it.
    assert max(cheap_starts) > 0.0, \
        "corner not exercised: the whole cheap batch fit the pool at t=0"
    assert min(main_starts) >= max(cheap_starts), \
        "a main arm started while a cheap arm was still pending"


# --------------------------------------------------------------------------- #
# (b) The main phase keeps the exact longest-first order                        #
# --------------------------------------------------------------------------- #


def test_main_arms_keep_longest_first_after_the_cheap_batch():
    cfg = _wave_cfg()
    thr = tt.cheap_first_threshold(cfg)
    plan = tt.plan_parallel(cfg, "safety_refusal", CLASS_A_13)
    main_order = [t["arm"] for t in plan["tasks"] if t["phase"] == "main"]
    # Uniform width 1 => launch order == queue order, so the main-phase
    # subsequence must be the historical (-est_wall, arm) sort exactly.
    expected = sorted((a for a in CLASS_A_13 if _cost(a) >= thr),
                      key=lambda a: (-_cost(a), a))
    assert main_order == expected, \
        f"main arms must stay longest-first: {main_order} != {expected}"


# --------------------------------------------------------------------------- #
# (c) A lockstep group is never split by the reordering                         #
# --------------------------------------------------------------------------- #


def test_lockstep_group_is_never_split_by_the_reordering():
    from sisteer.extraction.lockstep import (_LOCKSTEP_COMPATIBLE,
                                             lockstep_compatible)

    # (1) Under the DEFAULT threshold no lockstep-compatible arm is cheap:
    # every lockstep-capable arm is a gradient loop (cost >= 6), so the
    # cheap batch only ever pulls SOLO closed-form arms forward and the
    # sequential driver's co-scheduled teacher-sharing group cannot lose a
    # member to the front of the queue.
    thr = tt.cheap_first_threshold({})
    for arm in sorted(_LOCKSTEP_COMPATIBLE):
        assert _cost(arm) >= thr, \
            f"{arm} is lockstep-compatible but classified cheap — the " \
            "cheap batch would pull it out of a co-scheduled group"

    # (2) The parallel scheduler's unit of work is the whole arm task —
    # each lockstep-compatible arm runs as a SINGLETON lockstep group — so
    # the reordering permutes whole groups and can never split one. Pinned
    # even under a threshold that classifies lockstep arms cheap.
    arms = ["psr_kl_all", "vector", "diffmean", "alphakl_null",
            "lofit", "token_bos"]
    for cheap_below in (None, 99.0):
        dp = {"gpus": 4}
        if cheap_below is not None:
            dp["cheap_first_below"] = cheap_below
        cfg = {"data_parallel": dp,
               "train_kwargs": {"default": {"batch_size": 4}},
               "max_len": 96}
        plan = tt.plan_parallel(cfg, "safety_refusal", arms)
        assert sorted(t["arm"] for t in plan["tasks"]) == sorted(arms), \
            "every task is exactly one arm (singleton lockstep groups)"

        # (3) The sequential partition — the ONE place a multi-arm lockstep
        # group is co-scheduled — is untouched by the cheap-first config:
        # membership and input order are identical whatever the threshold.
        group = tt.plan_si(cfg, "safety_refusal", arms)["lockstep_group"]
        assert group == [a for a in arms if lockstep_compatible(a)], \
            "the lockstep group's membership/order changed with the " \
            "cheap-first config"


# --------------------------------------------------------------------------- #
# (d) Regression pin: no cheap arms => byte-identical to the old scheduler     #
# --------------------------------------------------------------------------- #


def test_no_cheap_arms_is_byte_identical_to_old_longest_first():
    cfg = _wave_cfg()
    gradient_only = [a for a in CLASS_A_13
                     if _cost(a) >= tt.cheap_first_threshold(cfg)]
    plan = tt.plan_parallel(cfg, "safety_refusal", gradient_only)
    assert all(t["phase"] == "main" for t in plan["tasks"])
    old_order, old_makespan = _old_longest_first_plan(
        cfg, "safety_refusal", gradient_only)
    assert _project(plan["tasks"]) == _project(old_order), \
        "with no cheap arms the placement must be byte-identical to the " \
        "old longest-first scheduler"
    assert plan["est_makespan_units"] == old_makespan


def test_threshold_zero_disables_the_phase_entirely():
    """cheap_first_below: 0 is the documented off switch — even WITH cheap
    arms present the placement is byte-identical to the old scheduler."""
    cfg = _wave_cfg(cheap_first_below=0)
    plan = tt.plan_parallel(cfg, "safety_refusal", CLASS_A_13)
    assert all(t["phase"] == "main" for t in plan["tasks"])
    old_order, old_makespan = _old_longest_first_plan(
        cfg, "safety_refusal", CLASS_A_13)
    assert _project(plan["tasks"]) == _project(old_order)
    assert plan["est_makespan_units"] == old_makespan


# --------------------------------------------------------------------------- #
# The LIVE scheduler launches in the same hybrid order                          #
# --------------------------------------------------------------------------- #


def test_run_parallel_fit_launches_cheap_first(tmp_path, monkeypatch):
    """run_parallel_fit's own queue (not just the plan simulation) must
    launch the cheap batch first, then the mains longest-first. Spawning
    is stubbed out — this exercises exactly the ordering + free-list
    bookkeeping, GPU-free."""
    import time

    launched: list[str] = []

    class FakeProc:
        def __init__(self, arm, gpu_ids):
            self.arm = arm
            self.gpu_ids = gpu_ids
            self.logs = [tmp_path / f"{arm}.log"]

        def poll(self):
            return 0  # finishes instantly; no fragment -> recorded failed

        def terminate(self):
            pass

    def fake_spawn(args, arm, width, gpu_ids, logdir):
        launched.append(arm)
        return FakeProc(arm, gpu_ids)

    monkeypatch.setattr(tt, "_spawn_arm_task", fake_spawn)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setenv("SISTEER_FIT_GPU_POOL", "g0,g1,g2,g3")

    cfg = _wave_cfg()
    # The scheduler hashes the pairs FILE for every cell's fit fingerprint
    # (F15) before any spawn, so the pool must exist on disk even though
    # the stubbed arms never read it. (The config path may stay unreadable:
    # its sha falls back to the cfg-dict hash by design.)
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text('{"prompt": "p", "pos": "a", "neg": "b"}\n')
    args = type("A", (), {"si": "safety_refusal", "overwrite": False,
                          "config": "unused.yaml",
                          "pairs": str(pairs_path),
                          "out": "", "arms": "", "token_cache": "off"})()
    rc = tt.run_parallel_fit(args, cfg, list(CLASS_A_13), tmp_path)
    # Every stub exits 0 with no report fragment, so each arm is recorded
    # failed and rc is 1 — failure containment is old behavior; the LAUNCH
    # ORDER is what this test is about.
    assert rc == 1
    assert len(launched) == len(CLASS_A_13)

    thr = tt.cheap_first_threshold(cfg)
    cheap = {a for a in CLASS_A_13 if _cost(a) < thr}
    assert set(launched[:len(cheap)]) == cheap, \
        f"live scheduler launched {launched[:len(cheap)]} first, " \
        f"expected the cheap batch {sorted(cheap)}"
    expected_main = sorted((a for a in CLASS_A_13 if a not in cheap),
                           key=lambda a: (-_cost(a), a))
    assert launched[len(cheap):] == expected_main, \
        "live scheduler's main phase is no longer longest-first"
