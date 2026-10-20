"""The attacker pool MEMBER's own lifecycle (31b): a member closes itself at its LAST CONSUMER,
and a member nothing ever bound closes itself once the group's placement is COMPLETE — decided inside the job from the shared root
(the targets' own attacker_binding.json records, the funnel's attackers/PLACED) and Slurm (sacct/squeue), never a session or a timer.
Execution form: the bash functions of scripts/sdfbase/full22/attacker_pool_lifecycle.sh over a synthetic wave root and the fake slurm."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, set_job_state, write_fake_slurm  # noqa: E402

LIFE = REPO / "scripts" / "sdfbase" / "full22" / "attacker_pool_lifecycle.sh"
COMMON = REPO / "scripts" / "sdfbase" / "full22" / "common.sh"
TARGET = REPO / "scripts" / "sdfbase" / "full22" / "target_attacks.sbatch"
ME = "http://gpu-4:8710"; OTHER = "http://gpu-0:8707"
ATK_MODEL = "attacker-org/atk-model"


def _world(tmp: Path) -> dict:
    root = tmp / "wave"; (root / "attackers").mkdir(parents=True); (root / "serving").mkdir()
    fake = tmp / "fake"; (fake / "jobs").mkdir(parents=True)
    bin_dir = write_fake_slurm(tmp / "bin")
    return {"root": root, "fake": fake, "env": dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", FAKE_SLURM_DIR=str(fake), PY_LIFECYCLE=sys.executable)}


def _bind(root: Path, job: str, urls, attack="crescendo"):
    d = root / "serving" / f"attacks_qwen35_4b_demo_u0_{attack}_{job}"; d.mkdir(parents=True, exist_ok=True)
    (d / "attacker_binding.json").write_text(json.dumps({"declared": len(urls), "alive": len(urls), "serving": len(urls), "waited_s": 2, "urls": list(urls)}))


def _placed(root: Path, jobs, assign=None):
    """The funnel's PLACED record; `assign` = {target job: member job} (31b-fix: the assignment at placement)."""
    assign = assign or {}
    (root / "attackers" / "PLACED").write_text(json.dumps({"group": "mt-u0-cluster_b", "members": sorted({str(v) for v in assign.values()}),
                                                           "targets": [{"job_id": j, "logical_work_id": f"m/s1/multiturn/x/g{j}", **({"assigned_member_job_id": str(assign[j])} if j in assign else {})} for j in jobs],
                                                           "utc": "2026-01-01T00:00:00Z"}))


def _call(w: dict, fn: str, *args) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", f'source "$1"; shift; "$@"', "_", str(LIFE), fn, *args], capture_output=True, text=True, env=w["env"], timeout=120)


def _verdict(w: dict, me: str | None = None):
    r = _call(w, "f22_pool_member_verdict", str(w["root"]), ME, *([me] if me else []))
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout.strip(), r.stderr.strip()


def test_consumers_are_the_targets_whose_binding_names_me_and_the_placed_targets_are_the_funnels(tmp_path):
    w = _world(tmp_path)
    _bind(w["root"], "101", [OTHER, ME]); _bind(w["root"], "102", [OTHER]); _bind(w["root"], "103", [ME], attack="pair")
    r = _call(w, "f22_pool_consumers", str(w["root"]), ME)
    assert r.returncode == 0 and sorted(r.stdout.split()) == ["101", "103"], r.stdout + r.stderr
    assert _call(w, "f22_pool_placed_targets", str(w["root"] / "attackers")).returncode == 1              # no PLACED yet
    _placed(w["root"], ["101", "102", "103", "104"])
    r = _call(w, "f22_pool_placed_targets", str(w["root"] / "attackers"))
    assert r.returncode == 0 and r.stdout.split() == ["101", "102", "103", "104"]


def test_a_member_serves_while_a_consumer_lives_and_exits_at_its_last_consumers_terminal_state(tmp_path):
    w = _world(tmp_path)
    for j, st in (("101", "RUNNING"), ("103", "RUNNING")):
        set_job_state(w["fake"], j, st)
    _bind(w["root"], "101", [OTHER, ME]); _bind(w["root"], "103", [ME], attack="pair")
    _placed(w["root"], ["101", "103"])                                                                   # 31b-fix: a member leaves only once the placement is complete
    v, why = _verdict(w); assert v == "serve" and "consumer(s) active" in why, (v, why)
    set_job_state(w["fake"], "101", "COMPLETED", end="2026-01-01T01:00:00")
    v, why = _verdict(w); assert v == "serve", (v, why)                                                   # one consumer still alive
    set_job_state(w["fake"], "103", "FAILED", end="2026-01-01T01:10:00")
    v, why = _verdict(w); assert v == "exit_last_consumer" and "every consumer terminal" in why, (v, why)
    # a consumer sacct cannot answer about is undecidable: the member stays (never a shrink on a bad probe)
    (w["fake"] / "outage").write_text("1")
    v, why = _verdict(w); assert v == "serve" and "undecidable" in why, (v, why)


def test_an_unbound_member_waits_for_the_placement_and_every_placed_targets_decision_then_exits_unbound(tmp_path):
    w = _world(tmp_path)
    set_job_state(w["fake"], "101", "RUNNING"); set_job_state(w["fake"], "102", "PENDING")
    _bind(w["root"], "101", [OTHER])                                                                     # bound elsewhere
    v, why = _verdict(w); assert v == "serve" and "not complete" in why, (v, why)                       # no PLACED: a target may still come
    _placed(w["root"], ["101", "102"])
    v, why = _verdict(w); assert v == "serve" and "102" in why and "neither bound nor ended" in why, (v, why)   # 102 pending, unbound: wait
    _bind(w["root"], "102", [OTHER]); set_job_state(w["fake"], "102", "RUNNING")
    v, why = _verdict(w); assert v == "exit_unbound" and "none bound me" in why, (v, why)                # every placed target decided elsewhere
    # had 102 bound ME instead, the member serves it to the end
    _bind(w["root"], "102", [ME])
    v, why = _verdict(w); assert v == "serve" and "102" in why, (v, why)
    set_job_state(w["fake"], "102", "CANCELLED", end="2026-01-01T02:00:00")
    v, why = _verdict(w); assert v == "exit_last_consumer", (v, why)
    # a placed target that ended without ever binding anyone counts as decided
    w2 = _world(tmp_path / "w2"); set_job_state(w2["fake"], "201", "TIMEOUT", end="2026-01-01T03:00:00"); _placed(w2["root"], ["201"])
    v, why = _verdict(w2); assert v == "exit_unbound", (v, why)


def test_a_member_with_ended_consumers_keeps_serving_while_the_placement_is_incomplete_or_a_placed_target_is_live_and_unbound(tmp_path):
    """Target A bound the member and ended; target B is placed but queued (A-terminal/B-pending) or starting
    (A-terminal/B-starting) — its binding is written at ITS runtime. The member must stay; before, "every consumer terminal" was its end."""
    w = _world(tmp_path)
    set_job_state(w["fake"], "101", "RUNNING"); set_job_state(w["fake"], "102", "PENDING")
    _bind(w["root"], "101", [ME])
    set_job_state(w["fake"], "101", "COMPLETED", end="2026-01-01T01:00:00")                              # A ended; no PLACED yet
    v, why = _verdict(w); assert v == "serve" and "not complete" in why and "101" in why, (v, why)
    _placed(w["root"], ["101", "102"])                                                                   # A-terminal / B-pending, unbound
    v, why = _verdict(w); assert v == "serve" and "102" in why and "neither bound nor ended" in why, (v, why)
    set_job_state(w["fake"], "102", "RUNNING")                                                           # A-terminal / B-starting, unbound
    v, why = _verdict(w); assert v == "serve" and "102" in why and "neither bound nor ended" in why, (v, why)
    _bind(w["root"], "102", [ME])                                                                        # B bound me at its runtime
    v, why = _verdict(w); assert v == "serve" and "consumer(s) active" in why and "102" in why, (v, why)
    set_job_state(w["fake"], "102", "COMPLETED", end="2026-01-01T02:00:00")
    v, why = _verdict(w); assert v == "exit_last_consumer" and "every consumer terminal" in why, (v, why)
    # had B bound the OTHER member, A's member still leaves only after B decided — and leaves then
    w2 = _world(tmp_path / "w2"); set_job_state(w2["fake"], "201", "COMPLETED", end="2026-01-01T01:00:00"); set_job_state(w2["fake"], "202", "RUNNING")
    _bind(w2["root"], "201", [ME]); _placed(w2["root"], ["201", "202"])
    v, why = _verdict(w2); assert v == "serve" and "202" in why, (v, why)
    _bind(w2["root"], "202", [OTHER])
    v, why = _verdict(w2); assert v == "exit_last_consumer", (v, why)
    # the ASSIGNMENT at placement: a target assigned to me is my consumer before it binds anyone — it keeps me while it lives
    w3 = _world(tmp_path / "w3"); set_job_state(w3["fake"], "301", "PENDING"); set_job_state(w3["fake"], "302", "RUNNING")
    _placed(w3["root"], ["301", "302"], assign={"301": "4242", "302": "4243"}); _bind(w3["root"], "302", [OTHER])
    r = _call(w3, "f22_pool_assigned_targets", str(w3["root"] / "attackers"), "4242"); assert r.returncode == 0 and r.stdout.split() == ["301"], r.stdout + r.stderr
    v, why = _verdict(w3, "4242"); assert v == "serve" and "301" in why and "neither bound nor ended" in why, (v, why)
    _bind(w3["root"], "301", [OTHER]); set_job_state(w3["fake"], "301", "RUNNING")                       # 301 bound the other member while I was down: still assigned me
    v, why = _verdict(w3, "4242"); assert v == "serve" and "assigned" in why and "301" in why, (v, why)
    v, why = _verdict(w3, "4243"); assert v == "serve" and "302" in why, (v, why)
    set_job_state(w3["fake"], "301", "COMPLETED", end="2026-01-01T03:00:00")
    v, why = _verdict(w3, "4242"); assert v == "exit_last_consumer", (v, why)
    v, why = _verdict(w3, "9999"); assert v == "exit_unbound" and "none bound me" in why, (v, why)       # a member nothing bound or was assigned
    # the target side reads its assigned member's URL from the member's record
    (w3["root"] / "attackers" / "4242.json").write_text(json.dumps({"url": ME, "job_id": "4242"}))
    r = _call(w3, "f22_pool_assigned_member_url", str(w3["root"]), "301"); assert r.returncode == 0 and r.stdout.strip() == ME, r.stdout + r.stderr
    assert _call(w3, "f22_pool_assigned_member_url", str(w3["root"]), "302").returncode == 1            # 4243 has no record (not live): none
    assert _call(w3, "f22_pool_assigned_member_url", str(w3["root"]), "999").returncode == 1


def test_the_binding_records_the_consumer_set_so_a_members_exit_follows_its_own_targets_not_the_whole_pool(tmp_path):
    """f22_live_attackers wrote EVERY live member into the binding before the assignment was chosen, so every member
    read every target as its consumer and A's member stayed until B ended. The target job now records the consumer set it actually binds
    (f22_pool_record_binding): A done on ME, B running on OTHER -> ME leaves."""
    w = _world(tmp_path)
    # the writer: the assigned member becomes the one consumer url, the other live members the failover candidates; the fallback keeps all
    d = w["root"] / "serving" / "attacks_qwen35_4b_demo_u0_crescendo_101"; d.mkdir(parents=True)
    (d / "attacker_binding.json").write_text(json.dumps({"declared": 3, "alive": 3, "serving": 3, "waited_s": 7, "urls": [OTHER, ME, "http://gpu-9:8711"]}))
    r = _call(w, "f22_pool_record_binding", str(d), ME, OTHER, ME, "http://gpu-9:8711")
    assert r.returncode == 0 and r.stdout.split() == ["placed", "1", "2"], r.stdout + r.stderr
    b = json.loads((d / "attacker_binding.json").read_text())
    assert b["urls"] == [ME] and b["candidates"] == [OTHER, "http://gpu-9:8711"] and b["assigned_member_url"] == ME and b["assignment"] == "placed"
    assert b["declared"] == 3 and b["alive"] == 3 and b["serving"] == 3 and b["waited_s"] == 7 and b["rewritten_utc"].endswith("Z")
    r = _call(w, "f22_pool_record_binding", str(d), "", OTHER, ME); assert r.returncode == 0 and r.stdout.split() == ["round_robin", "2", "0"], r.stdout + r.stderr
    b = json.loads((d / "attacker_binding.json").read_text()); assert b["urls"] == [OTHER, ME] and b["candidates"] == [] and b["assignment"] == "round_robin"
    r = _call(w, "f22_pool_record_binding", str(d), "http://gpu-5:8700", OTHER, ME)                    # an assigned member not live: the fallback set
    assert r.stdout.split() == ["round_robin", "2", "0"] and json.loads((d / "attacker_binding.json").read_text())["assigned_member_url"] == "http://gpu-5:8700"
    # A (101) bound ME, B (102) bound OTHER, both placed and assigned accordingly: A ends -> ME exits while B still runs; OTHER serves on
    w2 = _world(tmp_path / "w2"); set_job_state(w2["fake"], "101", "RUNNING"); set_job_state(w2["fake"], "102", "RUNNING")
    _placed(w2["root"], ["101", "102"], assign={"101": "4242", "102": "4243"})
    for job, url in (("101", ME), ("102", OTHER)):
        dd = w2["root"] / "serving" / f"attacks_qwen35_4b_demo_u0_crescendo_{job}"; dd.mkdir(parents=True)
        (dd / "attacker_binding.json").write_text(json.dumps({"declared": 2, "alive": 2, "serving": 2, "waited_s": 1, "urls": [OTHER, ME]}))
        assert _call(w2, "f22_pool_record_binding", str(dd), url, OTHER, ME).returncode == 0
    r = _call(w2, "f22_pool_consumers", str(w2["root"]), ME); assert r.stdout.split() == ["101"], r.stdout
    set_job_state(w2["fake"], "101", "COMPLETED", end="2026-01-01T01:00:00")
    v, why = _verdict(w2, "4242"); assert v == "exit_last_consumer" and "101" in why, (v, why)
    r = _call(w2, "f22_pool_member_verdict", str(w2["root"]), OTHER, "4243"); assert r.stdout.strip() == "serve" and "102" in r.stderr, r.stdout + r.stderr


def test_the_pool_sbatch_sources_the_lifecycle_and_writes_the_self_exit_marker():
    sb = (REPO / "scripts" / "sdfbase" / "full22" / "attacker_pool.sbatch").read_text()
    assert "attacker_pool_lifecycle.sh" in sb and "f22_pool_member_verdict" in sb and ".SELF_EXIT" in sb and "exit_last_consumer|exit_unbound" in sb
    assert 'f22_pool_member_verdict "$WAVE_ROOT" "$URL" "$SLURM_JOB_ID"' in sb                             # the member reads its assignment (31b-fix)
    tb = (REPO / "scripts" / "sdfbase" / "full22" / "target_attacks.sbatch").read_text()
    assert "attacker_pool_lifecycle.sh" in tb and 'f22_pool_assigned_member_url "$WAVE_ROOT" "$SLURM_JOB_ID"' in tb and 'if [ "$ASSIGNED_LIVE" = 1 ]; then url="${ATTACKERS[0]}"' in tb
    assert tb.index("f22_live_attackers\n") < tb.index("f22_pool_assigned_member_url") < tb.index('f22_pool_record_binding "$SERVE_DIR"') < tb.index("# Phase C: the drivers")
    assert subprocess.run(["bash", "-n", str(REPO / "scripts" / "sdfbase" / "full22" / "target_attacks.sbatch")], capture_output=True).returncode == 0
    assert subprocess.run(["bash", "-n", str(LIFE)], capture_output=True).returncode == 0 and subprocess.run(["bash", "-n", str(REPO / "scripts" / "sdfbase" / "full22" / "attacker_pool.sbatch")], capture_output=True).returncode == 0


def _injecting_sacct(w: dict, *, targets, urls, each_call: bool = False) -> dict:
    """A `sacct` ahead of the fake one on PATH that, on its FIRST call (or on EVERY call with `each_call`, one target per call), writes the
    binding of the next target in `targets` naming `urls` and marks that target RUNNING — a target binding me WHILE the member's decision
    sits in its slow sacct read — then answers as the fake sacct does. Returns the world with that PATH."""
    shim = w["root"].parent / "shim"; shim.mkdir(exist_ok=True)
    real = Path(w["env"]["PATH"].split(":")[0]) / "sacct"
    (shim / "targets").write_text("\n".join(targets) + "\n"); (shim / "n").write_text("0")
    body = (f'#!/bin/bash\nn=$(cat {shim}/n); t=$(sed -n "$((n + 1))p" {shim}/targets)\n'
            f'if [ -n "$t" ] && {{ [ "$n" = 0 ] || [ "{1 if each_call else 0}" = 1 ]; }}; then\n'
            f'  d={w["root"]}/serving/attacks_qwen35_4b_demo_u0_crescendo_$t; mkdir -p "$d"\n'
            f"  printf '%s\\n' '{json.dumps({'declared': len(urls), 'alive': len(urls), 'serving': len(urls), 'waited_s': 1, 'urls': list(urls)})}' > \"$d/attacker_binding.json\"\n"
            f'  echo RUNNING > {w["fake"]}/jobs/$t.state; echo $((n + 1)) > {shim}/n\nfi\nexec {real} "$@"\n')
    (shim / "sacct").write_text(body); (shim / "sacct").chmod(0o755)
    return dict(w, env=dict(w["env"], PATH=f"{shim}:{w['env']['PATH']}"))


def test_a_target_that_binds_me_during_the_decisions_slow_sacct_read_keeps_the_member(tmp_path):
    """The consumers were read ONCE, before the slow sacct / squeue calls over
    the placed targets and my consumers, and the exit was answered on that stale set — a target whose binding landed meanwhile (its binding is
    written at ITS runtime; a re-minted target is not even in PLACED v1) was missed. The verdict now re-reads the consumers immediately before
    answering exit_unbound / exit_last_consumer and decides again over the fresh set."""
    # (a) the exit_unbound path: placed 101 bound elsewhere and live, placed 102 ended unbound (the slow sacct read is 102's); target 103 — a
    #     re-minted target PLACED v1 does not name — binds ME during that read. Without the shim the same world exits unbound.
    w = _world(tmp_path)
    set_job_state(w["fake"], "101", "RUNNING"); set_job_state(w["fake"], "102", "COMPLETED", end="2026-01-01T01:00:00")
    _bind(w["root"], "101", [OTHER]); _placed(w["root"], ["101", "102"])
    v, why = _verdict(w); assert v == "exit_unbound" and "none bound me" in why, (v, why)
    wi = _injecting_sacct(w, targets=["103"], urls=[ME])
    r = _call(wi, "f22_pool_member_verdict", str(w["root"]), ME); assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip() == "serve" and "consumer(s) active (bound or assigned): 103" in r.stderr.strip().splitlines()[-1], r.stdout + r.stderr
    assert "re-check: consumer set changed under the decision (exit_unbound" in r.stderr and "-> now 103" in r.stderr, r.stderr
    assert (tmp_path / "shim" / "n").read_text().strip() == "1"                                        # injected exactly once, on the first sacct read
    v, why = _verdict(w); assert v == "serve" and "103" in why, (v, why)                               # the binding stands: the plain read sees it too
    set_job_state(w["fake"], "103", "COMPLETED", end="2026-01-01T02:00:00")
    v, why = _verdict(w); assert v == "exit_last_consumer" and "103" in why, (v, why)                  # and its end is my end, as before
    # (b) the exit_last_consumer path: my one consumer 101 ended (the slow read is 101's terminal check); 103 binds me during it
    w2 = _world(tmp_path / "w2"); set_job_state(w2["fake"], "101", "COMPLETED", end="2026-01-01T01:00:00")
    _bind(w2["root"], "101", [ME]); _placed(w2["root"], ["101"])
    v, why = _verdict(w2); assert v == "exit_last_consumer", (v, why)
    r = _call(_injecting_sacct(w2, targets=["103"], urls=[ME]), "f22_pool_member_verdict", str(w2["root"]), ME, "4242"); assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip() == "serve" and "consumer(s) active (bound or assigned): 101 103" in r.stderr.strip().splitlines()[-1], r.stdout + r.stderr
    assert "re-check: consumer set changed under the decision (exit_last_consumer over 101 -> now 101 103" in r.stderr, r.stderr
    # (c) a binding that arrives on EVERY read, each for a target already ended: the re-decision is bounded — three changed sets answer `serve`
    #     for this tick (the next tick decides), never an exit on a stale set and never a spin
    w3 = _world(tmp_path / "w3"); set_job_state(w3["fake"], "101", "RUNNING"); set_job_state(w3["fake"], "102", "COMPLETED", end="2026-01-01T01:00:00")
    for t in ("201", "202", "203", "204"):
        set_job_state(w3["fake"], t, "COMPLETED", end="2026-01-01T01:30:00")
    _bind(w3["root"], "101", [OTHER]); _placed(w3["root"], ["101", "102"])
    w3i = _injecting_sacct(w3, targets=["201", "202", "203", "204"], urls=[ME], each_call=True)
    (w3i["root"].parent / "shim" / "sacct").write_text((w3i["root"].parent / "shim" / "sacct").read_text().replace("echo RUNNING >", "echo COMPLETED >"))
    r = _call(w3i, "f22_pool_member_verdict", str(w3["root"]), ME); assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip() == "serve" and "changed under the decision 3 times" in r.stderr.strip().splitlines()[-1] and r.stderr.count("re-check:") == 2, r.stdout + r.stderr
    # the sbatch consumes the verdict unchanged: ONE verdict word on stdout, the LAST stderr line is the reason it logs
    assert len(r.stdout.split()) == 1
    sb = (REPO / "scripts" / "sdfbase" / "full22" / "attacker_pool.sbatch").read_text()
    assert 'why=$(tail -n 1 "$SERVE_DIR/lifecycle_why.txt")' in sb
    assert subprocess.run(["bash", "-n", str(LIFE)], capture_output=True).returncode == 0


def _advancing_sacct(w: dict, *, versions: list, each_call: bool = False) -> dict:
    """A `sacct` ahead of the fake one on PATH that, on its FIRST call (or on EVERY call with `each_call`, one record per call), REPLACES
    attackers/PLACED with the next record in `versions` (each {version, jobs, assign, states}) and sets the named jobs' fake states — the
    funnel's supersession landing WHILE the member's decision sits in its slow sacct read — then answers as the fake sacct does. The new
    targets get NO binding: the covered case is a v2 whose targets are still queued, so both consumer reads are empty."""
    shim = w["root"].parent / "shim_placed"; shim.mkdir(exist_ok=True)
    real = Path(w["env"]["PATH"].split(":")[0]) / "sacct"
    for i, v in enumerate(versions):
        doc = {"group": "mt-u0-cluster_b", "version": v["version"], "members": sorted({str(m) for m in v["assign"].values()}),
               "targets": [{"job_id": j, "logical_work_id": f"m/s1/multiturn/x/g{j}", **({"assigned_member_job_id": str(v["assign"][j])} if j in v["assign"] else {})} for j in v["jobs"]],
               "utc": "2026-01-01T00:00:00Z"}
        (shim / f"v{i}.json").write_text(json.dumps(doc))
        (shim / f"v{i}.states").write_text("".join(f"{j} {st}\n" for j, st in v["states"].items()))
    (shim / "n").write_text("0")
    body = (f'#!/bin/bash\nn=$(cat {shim}/n)\n'
            f'if [ -f {shim}/v$n.json ] && {{ [ "$n" = 0 ] || [ "{1 if each_call else 0}" = 1 ]; }}; then\n'
            f'  cp {shim}/v$n.json {w["root"]}/attackers/PLACED\n'
            f'  while read -r j st; do [ -n "$j" ] && echo "$st" > {w["fake"]}/jobs/$j.state; done < {shim}/v$n.states\n'
            f'  echo $((n + 1)) > {shim}/n\nfi\nexec {real} "$@"\n')
    (shim / "sacct").write_text(body); (shim / "sacct").chmod(0o755)
    return dict(w, env=dict(w["env"], PATH=f"{shim}:{w['env']['PATH']}"))


def test_a_placement_advanced_during_the_decision_keeps_the_member_even_when_the_new_targets_are_still_queued(tmp_path):
    """placement-fix round 3: the exit re-check compared the CONSUMER reads alone, so a PLACED v2 published during the
    slow decision — assigning me targets still QUEUED, with no binding on either read — was missed: both reads empty, exit_unbound, SELF_EXIT,
    and the supersession assigned targets to a server that was gone. The verdict now fingerprints the placement (version, targets, the ones
    assigned me) before the decision and again before an exit, and decides again over the record it now reads."""
    # (a) the exit_unbound path, the finding's exact shape: I am the REPLACEMENT member 4242 — v1 (101 bound elsewhere and live, 102 ended
    #     unbound) names neither me nor anything for me; v2 lands during the decision's slow sacct read (102's terminal check) with the re-minted
    #     targets 103/104, both PENDING and unbound, so BOTH consumer reads are empty. Without the shim the same world exits unbound.
    #     Two members' shape first (the paper's plan): v2 assigns the queued targets to the OTHER member 4243 — nothing for me, yet a placed target
    #     that has neither bound nor ended keeps EVERY member, and only the placement re-read gets there: the decision's
    #     unbound check ran over v1's targets, its assigned read found nothing, exit_unbound — then the fingerprint moved, and it decides again.
    w = _world(tmp_path)
    set_job_state(w["fake"], "101", "RUNNING"); set_job_state(w["fake"], "102", "COMPLETED", end="2026-01-01T01:00:00")
    _bind(w["root"], "101", [OTHER]); _placed(w["root"], ["101", "102"], assign={"101": "4000", "102": "4000"})
    v, why = _verdict(w, "4242"); assert v == "exit_unbound" and "none bound me" in why, (v, why)
    r = _call(w, "f22_pool_placement_fingerprint", str(w["root"] / "attackers"), "4242"); assert r.stdout.strip() == "v1 targets=101,102 mine=", r.stdout + r.stderr
    wa = _advancing_sacct(w, versions=[{"version": 2, "jobs": ["103", "104"], "assign": {"103": "4243", "104": "4243"}, "states": {"103": "PENDING", "104": "PENDING"}}])
    r = _call(wa, "f22_pool_member_verdict", str(w["root"]), ME, "4242"); assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip() == "serve" and "placed target(s) 103 104 have neither bound nor ended" in r.stderr.strip().splitlines()[-1], r.stdout + r.stderr
    assert "re-check: placement changed under the decision (exit_unbound over v1 targets=101,102 mine= -> now v2 targets=103,104 mine=): deciding again" in r.stderr, r.stderr
    assert "consumer set changed" not in r.stderr                                                     # both consumer reads WERE empty: the placement is what moved
    assert (tmp_path / "shim_placed" / "n").read_text().strip() == "1" and len(r.stdout.split()) == 1
    v, why = _verdict(w, "4242"); assert v == "serve" and "103" in why and "104" in why, (v, why)       # v2 stands: the plain read serves them too
    r = _call(w, "f22_pool_placement_fingerprint", str(w["root"] / "attackers"), "4242"); assert r.stdout.strip() == "v2 targets=103,104 mine=", r.stdout + r.stderr
    # the one-member shape (the demo smoke): v2 assigns the queued targets to ME — the decision's own assigned read, which follows the slow
    # unbound check, already serves them as consumers (assigned, active), so no re-decision is needed; either way the member never leaves
    w1 = _world(tmp_path / "w1"); set_job_state(w1["fake"], "101", "RUNNING"); set_job_state(w1["fake"], "102", "COMPLETED", end="2026-01-01T01:00:00")
    _bind(w1["root"], "101", [OTHER]); _placed(w1["root"], ["101", "102"], assign={"101": "4000", "102": "4000"})
    v, why = _verdict(w1, "4242"); assert v == "exit_unbound", (v, why)
    r = _call(_advancing_sacct(w1, versions=[{"version": 2, "jobs": ["103", "104"], "assign": {"103": "4242", "104": "4242"}, "states": {"103": "PENDING", "104": "PENDING"}}]),
              "f22_pool_member_verdict", str(w1["root"]), ME, "4242"); assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip() == "serve" and "103 104" in r.stderr.strip().splitlines()[-1] and ("neither bound nor ended" in r.stderr or "consumer(s) active (bound or assigned)" in r.stderr), r.stdout + r.stderr
    set_job_state(w1["fake"], "103", "RUNNING"); _bind(w1["root"], "103", [ME])                       # 103 came up and bound me; 104 still queued
    v, why = _verdict(w1, "4242"); assert v == "serve" and "104" in why and "neither bound nor ended" in why, (v, why)
    set_job_state(w1["fake"], "104", "RUNNING"); _bind(w1["root"], "104", [ME])
    v, why = _verdict(w1, "4242"); assert v == "serve" and why.startswith("consumer(s) active (bound or assigned): 103 104"), (v, why)
    for j in ("103", "104"):
        set_job_state(w1["fake"], j, "COMPLETED", end="2026-01-01T02:00:00")
    v, why = _verdict(w1, "4242"); assert v == "exit_last_consumer" and "103" in why and "104" in why, (v, why)   # and their end is my end
    # (b) the exit_last_consumer path: my one consumer 101 (assigned me, bound me) ended; v2 lands during its terminal check, adding 103 (PENDING,
    #     assigned me): the same consumer set on both reads, a changed placement -> decided again -> serve
    w2 = _world(tmp_path / "w2"); set_job_state(w2["fake"], "101", "COMPLETED", end="2026-01-01T01:00:00")
    _bind(w2["root"], "101", [ME]); _placed(w2["root"], ["101"], assign={"101": "4242"})
    v, why = _verdict(w2, "4242"); assert v == "exit_last_consumer", (v, why)
    r = _call(_advancing_sacct(w2, versions=[{"version": 2, "jobs": ["101", "103"], "assign": {"101": "4242", "103": "4242"}, "states": {"103": "PENDING"}}]),
              "f22_pool_member_verdict", str(w2["root"]), ME, "4242"); assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip() == "serve" and "103" in r.stderr.strip().splitlines()[-1] and "neither bound nor ended" in r.stderr.strip().splitlines()[-1], r.stdout + r.stderr
    assert "re-check: placement changed under the decision (exit_last_consumer over v1 targets=101 mine=101 -> now v2 targets=101,103 mine=101,103): deciding again" in r.stderr, r.stderr
    # (c) a placement that advances on EVERY read (the same ended targets, a new version each time): bounded as the consumer flavour is — three
    #     changed reads answer `serve` for this tick, never an exit on a stale record and never a spin
    w3 = _world(tmp_path / "w3"); set_job_state(w3["fake"], "101", "RUNNING"); set_job_state(w3["fake"], "102", "COMPLETED", end="2026-01-01T01:00:00")
    _bind(w3["root"], "101", [OTHER]); _placed(w3["root"], ["101", "102"])
    vs = [{"version": n, "jobs": ["101", "102"], "assign": {}, "states": {}} for n in (2, 3, 4, 5)]
    r = _call(_advancing_sacct(w3, versions=vs, each_call=True), "f22_pool_member_verdict", str(w3["root"]), ME, "4242"); assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip() == "serve" and "changed under the decision 3 times" in r.stderr.strip().splitlines()[-1] and r.stderr.count("re-check: placement changed") == 2, r.stdout + r.stderr
    assert len(r.stdout.split()) == 1


def _transition(root: Path, *, age_s: int = 0, lease_s: int = 600, reason: str = "06 --resubmit pass on abc123"):
    """The pass's marker as submit_twin.py placement-transition open writes it (epoch = the cluster's clock; `age_s` back-dates it)."""
    import time
    (root / "attackers" / "PLACED.transition").write_text(json.dumps({"group": "mt-u0-cluster_b", "from_version": 1, "reason": reason, "lease_s": lease_s,
                                                                      "epoch": int(time.time()) - age_s, "utc": "2026-01-01T00:00:00Z", "by": "test"}))


def test_a_member_holds_every_exit_while_the_groups_placement_is_in_transition_and_ignores_a_lapsed_marker(tmp_path):
    """placement-fix round 3: the --resubmit pass submits every replacement BEFORE it supersedes PLACED, so a replacement
    member that comes up meanwhile (or while a failed mark-placed awaits its retry) reads the record that predates it and leaves exit_unbound. The
    pass now writes attackers/PLACED.transition before its first submission; the member holds exit_unbound AND exit_last_consumer while it is fresh,
    and ignores it — saying so — once its lease lapsed (the bound: a pass nobody completes cannot pin a member for ever) or when it cannot be read."""
    # the replacement member 4242 on v1 (101 bound elsewhere and live, 102 ended unbound): exit_unbound, until the pass's marker says otherwise
    w = _world(tmp_path)
    set_job_state(w["fake"], "101", "RUNNING"); set_job_state(w["fake"], "102", "COMPLETED", end="2026-01-01T01:00:00")
    _bind(w["root"], "101", [OTHER]); _placed(w["root"], ["101", "102"], assign={"101": "4000", "102": "4000"})
    v, why = _verdict(w, "4242"); assert v == "exit_unbound" and "none bound me" in why and "TRANSITION" not in why, (v, why)
    assert _call(w, "f22_pool_placement_in_transition", str(w["root"] / "attackers")).returncode == 1                    # absent
    _transition(w["root"])                                                                                                   # fresh: 600 s left
    r = _call(w, "f22_pool_placement_in_transition", str(w["root"] / "attackers"))
    assert r.returncode == 0 and r.stdout.startswith("since 2026-01-01T00:00:00Z for 600 s (") and "s left): 06 --resubmit pass on abc123" in r.stdout, r.stdout + r.stderr
    v, why = _verdict(w, "4242")
    assert v == "serve" and "the group's placement is in TRANSITION (attackers/PLACED.transition since 2026-01-01T00:00:00Z for 600 s" in why, (v, why)
    assert "06 --resubmit pass on abc123" in why and "would have answered exit_unbound: placement complete (2 target(s))" in why and "none bound me" in why, why
    # the bound: the same marker past its lease is ignored, named in the reason — the member decides on PLACED and its bindings alone again
    _transition(w["root"], age_s=601)
    r = _call(w, "f22_pool_placement_in_transition", str(w["root"] / "attackers")); assert r.returncode == 2 and "(lapsed 1 s ago)" in r.stdout or "(lapsed 2 s ago)" in r.stdout, r.stdout + r.stderr
    v, why = _verdict(w, "4242")
    assert v == "exit_unbound" and why.startswith("placement complete (2 target(s))") and "(a placement-transition marker since 2026-01-01T00:00:00Z for 600 s (lapsed" in why and "lapsed its lease: ignored)" in why, (v, why)
    # an unreadable marker is ignored and named; a removed one leaves the plain verdict
    (w["root"] / "attackers" / "PLACED.transition").write_text("{not json")
    assert _call(w, "f22_pool_placement_in_transition", str(w["root"] / "attackers")).returncode == 3
    v, why = _verdict(w, "4242"); assert v == "exit_unbound" and "marker that cannot be read is ignored" in why, (v, why)
    (w["root"] / "attackers" / "PLACED.transition").unlink()
    v, why = _verdict(w, "4242"); assert v == "exit_unbound" and "transition" not in why.lower(), (v, why)
    # the exit_last_consumer path is held the same way (a re-placing pass may assign a member whose consumers all ended new targets)
    w2 = _world(tmp_path / "w2"); set_job_state(w2["fake"], "101", "COMPLETED", end="2026-01-01T01:00:00")
    _bind(w2["root"], "101", [ME]); _placed(w2["root"], ["101"], assign={"101": "4242"})
    v, why = _verdict(w2, "4242"); assert v == "exit_last_consumer", (v, why)
    _transition(w2["root"], lease_s=120)
    v, why = _verdict(w2, "4242"); assert v == "serve" and "in TRANSITION" in why and "would have answered exit_last_consumer: placement complete and every consumer terminal" in why, (v, why)
    _transition(w2["root"], lease_s=120, age_s=121)
    v, why = _verdict(w2, "4242"); assert v == "exit_last_consumer" and "lapsed its lease: ignored" in why, (v, why)
    # a member with a LIVE consumer reads the plain reason (the fence is applied to exits alone), and an incomplete placement serves as before
    w3 = _world(tmp_path / "w3"); set_job_state(w3["fake"], "101", "RUNNING"); _bind(w3["root"], "101", [ME]); _placed(w3["root"], ["101"], assign={"101": "4242"}); _transition(w3["root"])
    v, why = _verdict(w3, "4242"); assert v == "serve" and why.startswith("consumer(s) active (bound or assigned): 101") and "TRANSITION" not in why, (v, why)
    w4 = _world(tmp_path / "w4"); _transition(w4["root"])
    v, why = _verdict(w4, "4242"); assert v == "serve" and "not complete" in why and "TRANSITION" not in why, (v, why)
    # ONE verdict word on stdout for the sbatch, the fence at BOTH exit sites, the fingerprint read before and after the decision
    r = _call(w, "f22_pool_member_verdict", str(w["root"]), ME, "4242"); assert len(r.stdout.split()) == 1
    life = LIFE.read_text()
    i_dec = life.index("f22_pool_member_decide(){"); i_ver = life.index("f22_pool_member_verdict(){")
    body = life[i_dec:i_ver]
    assert body.count('f22_pool_transition_fence "$pool"') == 2 and body.index("F22_VERDICT=exit_last_consumer") < body.index('f22_pool_transition_fence "$pool"') and body.rindex("F22_VERDICT=exit_unbound") < body.rindex('f22_pool_transition_fence "$pool"')
    verdict = life[i_ver:]
    assert 'placement=$(f22_pool_placement_fingerprint "$root/attackers" "$me")' in verdict and 'fresh_placement=$(f22_pool_placement_fingerprint "$root/attackers" "$me")' in verdict
    assert verdict.index('placement=$(f22_pool_placement_fingerprint') < verdict.index('f22_pool_member_decide "$root" "$url" "$me" "$consumers"') < verdict.index("fresh_placement=$(")
    assert '[ "$fresh" != "$consumers" ] || [ "$fresh_placement" != "$placement" ]' in verdict
    assert subprocess.run(["bash", "-n", str(LIFE)], capture_output=True).returncode == 0


def _target_phase_b(w: dict, *, target: str, serving: tuple = (), wait_s: int = 60) -> tuple:
    """target_attacks.sbatch's Phase B over the World's wave root, from the SHIPPED text: f22_declared_attackers + f22_live_attackers out of
    common.sh and the sbatch's block from `# Phase B` to `# Phase C` (the pool wait, the assignment read, the binding record), with the fake
    slurm on PATH, a `curl` that answers the attacker model for the `serving` URLs only, and a `sleep` that advances SECONDS instead of
    waiting — the wait is exercised exactly (f22_live_attackers polls every 20 s until F22_ATTACKER_WAIT_S) and costs no wall time. Returns
    (the run, the target's serve dir); a `PHASE_C ASSIGNED_LIVE=<0|1> urls=<...>` line on stdout means the job went on to launch drivers."""
    src = COMMON.read_text(); fns = []
    for name in ("f22_declared_attackers", "f22_live_attackers"):
        m = re.search(rf"(?m)^{name}\(\)\{{.*?^\}}\n", src, re.S); assert m, f"{name} not found in {COMMON}"; fns.append(m.group(0))
    tb = TARGET.read_text(); phase_b = tb[tb.index("# Phase B: the shared attacker pool"):tb.index("# Phase C: the drivers")]
    tmp = w["root"].parent; serve = w["root"] / "serving" / f"attacks_qwen35_4b_demo_u0_crescendo_{target}"; serve.mkdir(parents=True, exist_ok=True)
    stubs = tmp / "stubs"; stubs.mkdir(exist_ok=True)
    ok = " ".join(f"{u}/v1/models" for u in serving)
    (stubs / "curl").write_text('#!/bin/bash\nurl="${@: -1}"\n' + f'for u in "" {ok}; do [ -n "$u" ] && [ "$url" = "$u" ] && {{ echo \'{{"data":[{{"id":"{ATK_MODEL}"}}]}}\'; exit 0; }}; done\nexit 0\n')
    (stubs / "curl").chmod(0o755)
    cfg = tmp / "cfg.yaml"; cfg.write_text(f"attacker:\n  model: {ATK_MODEL}\n")
    script = tmp / "phase_b.sh"
    script.write_text("set -uo pipefail\n"                                                                  # the sbatch's own flags
                      'f22_log(){ echo "[log] $*"; }\nf22_fatal(){ f22_log "FATAL: $*"; exit 1; }\n'      # common.sh's, minus the timestamp
                      'sleep(){ echo "SLEPT $1"; SECONDS=$((SECONDS + $1)); }\n'
                      f"MPY={sys.executable!r}\nCFG={str(cfg)!r}\nWAVE_ROOT={str(w['root'])!r}\nSERVE_DIR={str(serve)!r}\nREPO_DIR={str(REPO)!r}\n"
                      f"SLURM_JOB_ID={target!r}\nF22_SERVING_PATH=lora_native\nexport F22_ATTACKER_WAIT_S={wait_s}\n"
                      + "".join(fns) + phase_b + 'echo "PHASE_C ASSIGNED_LIVE=$ASSIGNED_LIVE urls=${ATTACKERS[*]}"\n')
    env = dict(w["env"], PATH=f"{stubs}:{w['env']['PATH']}")
    return subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env, timeout=120), serve


def test_a_placement_committed_to_a_member_that_self_exited_across_the_marker_open_ends_the_target_fatal_and_binds_no_url(tmp_path):
    """An interleaving shipped as a NAMED RESIDUAL (not fenced, provided its
    failure stays LOUD — this test is that condition): a member decides its old consumers are done; a resubmit opens the placement marker; its
    --walk sees the member's job still active and skips replacement; the member writes SELF_EXIT (its EXIT trap removes attackers/<member>.json);
    mark_placed commits v2 assigning a new target to the exited member. The loud outcome, on the SHIPPED target text: the exited member has no
    .json and no .pending, so it is not in the declared pool whatever Slurm says of its job (f22_declared_attackers globs the records; squeue only
    prunes); with no other live member f22_live_attackers polls for F22_ATTACKER_WAIT_S and f22_fatal's "no live attacker" — inside Phase B,
    BEFORE the assignment read, which would miss anyway (f22_pool_assigned_member_url requires the member's .json) — so the target ends FAILED,
    records no attacker_binding.json, launches no driver and binds no URL; the next `06 --family multiturn --resubmit` re-mints member and target.
    The control beside it: the SAME world with the member's record on disk binds the assigned member at once — the record the EXIT trap removed
    is the whole difference. If this test cannot pass, the residual is not loud and the interleaving must be fenced instead."""
    w = _world(tmp_path); member, target = "4242", "301"
    set_job_state(w["fake"], member, "RUNNING"); set_job_state(w["fake"], target, "RUNNING")      # the member's job is still on Slurm, in its exit path
    # v2, the interleaving's commit: the exited member is the group's one member and the re-minted target is assigned to it
    (w["root"] / "attackers" / "PLACED").write_text(json.dumps({"group": "mt-u0-cluster_b", "version": 2, "members": [member], "n_targets": 1,
                                                                "targets": [{"job_id": target, "logical_work_id": "m/s1/multiturn/x/g1", "assigned_member_job_id": member}],
                                                                "supersedes": {"version": 1}, "reason": "06 --resubmit pass", "utc": "2026-01-01T00:10:00Z"}))
    (w["root"] / "attackers" / f"{member}.SELF_EXIT").write_text(json.dumps({"job_id": member, "reason": "exit_last_consumer", "consumers": "201",
                                                                             "why": "placement complete and every consumer terminal", "utc": "2026-01-01T00:09:00Z"}))
    assert not (w["root"] / "attackers" / f"{member}.json").exists() and not (w["root"] / "attackers" / f"{member}.pending").exists()
    # the assignment names the member; the assignment read misses, because the member's record is gone
    r = _call(w, "f22_pool_assigned_targets", str(w["root"] / "attackers"), member); assert r.returncode == 0 and r.stdout.split() == [target], r.stdout + r.stderr
    assert _call(w, "f22_pool_assigned_member_url", str(w["root"]), target).returncode == 1
    # the target's Phase B: the whole F22_ATTACKER_WAIT_S (three 20 s polls at 60 s), then the fatal; no URL, no binding record, no driver
    r, serve = _target_phase_b(w, target=target, serving=(), wait_s=60); out = r.stdout + r.stderr
    assert r.returncode == 1, out
    assert f"FATAL: no live attacker in {w['root'] / 'attackers'} (submit attacker_pool.sbatch first" in out, out
    assert re.search(r"attacker pool still under strength after 6\ds: 0/0 serving \(not yet up: none\)", out) and out.count("SLEPT 20") == 3, out
    assert "PHASE_C" not in out and "assigned member" not in out and "round-robin" not in out and "binding recorded" not in out, out
    assert "http://" not in out and not list(serve.iterdir()), out                                      # never a URL, never attacker_binding.json
    # the fatal precedes the binding record inside f22_live_attackers, and the pool wait precedes the assignment read in the sbatch
    fn = re.search(r"(?m)^f22_live_attackers\(\)\{.*?^\}\n", COMMON.read_text(), re.S).group(0)
    assert fn.index('f22_fatal "no live attacker in') < fn.index('attacker_binding.json')
    tb = TARGET.read_text(); assert tb.index("# Phase B") < tb.index("f22_live_attackers\n") < tb.index("f22_pool_assigned_member_url") < tb.index("# Phase C")
    # the control: the same world with the member's record on disk (what the EXIT trap removed) binds the assigned member at once
    (w["root"] / "attackers" / f"{member}.json").write_text(json.dumps({"url": ME, "job_id": member}))
    assert _call(w, "f22_pool_assigned_member_url", str(w["root"]), target).stdout.strip() == ME
    r2, serve2 = _target_phase_b(w, target=target, serving=(ME,), wait_s=60); out2 = r2.stdout + r2.stderr
    assert r2.returncode == 0 and "SLEPT" not in out2 and f"assigned member {ME} is live: every driver of this job binds it (0 other live member(s) behind it)" in out2, out2
    assert f"PHASE_C ASSIGNED_LIVE=1 urls={ME}" in out2, out2
    b = json.loads((serve2 / "attacker_binding.json").read_text()); assert b["urls"] == [ME] and b["assignment"] == "placed" and b["assigned_member_url"] == ME, b
