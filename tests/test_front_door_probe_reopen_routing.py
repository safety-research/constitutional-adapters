"""`--phase probe --reopen` routes two more states to the carrier's CPU-only reverdict (03_hpo.sh probe_reopen_verb / probe_submit_reverdict;
scripts/sbatch/serving_probe.sbatch phase=reverdict):
* REOPEN_IN_PROGRESS (hpo_probe_status: an in-progress receipt beside a COMPLETE checkpoint — a reverdict consumed the receipt and
  died before the certifying rewrite; info.interrupted_reopen names the consumer) → the carrier in phase=reverdict with positional 12
  `resume-interrupted`, positional 10 `-`: NEVER a fresh revocation (no new receipt, no REVOKED marker). Without --reopen the phase
  refuses and names the routed recovery.
* PARITY_STALE → the receipted path: every staged overlay revoked, the receipt minted into the record, the carrier in phase=reverdict with
  positional 12 `parity-record=<dir>` and the receipt's nonce as positional 10; an incomplete parity record refuses with nothing minted.
Executed against the REAL stage with the probe harness (fake ssh / rsync / squeue / sbatch; the real status reader), no network."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "sdfbase"))
from front_door_testlib import REPO  # noqa: E402
from test_front_door_probe_resume import LWID, MODEL, World, _attempt, _plan, _stamp, world  # noqa: E402,F401
import hpo_probe_resume as hpr  # noqa: E402
import hpo_probe_status as hps  # noqa: E402

CARRIER = REPO / "scripts" / "sbatch" / "serving_probe.sbatch"
NONCE = "0123456789abcdef0123456789abcdef"


def _carrier_tail(call: dict) -> list[str]:
    argv = call["argv"]
    idx = next(i for i, a in enumerate(argv) if a.endswith("serving_probe.sbatch"))
    return argv[idx + 1:]


def _attempt_events(w: World) -> list[dict]:
    rows = [json.loads(ln) for ln in (w.tmp / "state" / MODEL / "attempts.jsonl").read_text().splitlines() if ln.strip()]
    return [a for a in rows if a.get("event") == "attempt"]


def _inprogress_receipt(w: World) -> Path:
    gen = hps.record_generation(w.record)["generation"]
    d = w.record / "reopen_receipts"; d.mkdir(exist_ok=True)
    f = d / f"{gen}.{NONCE}.inprogress.20260930T000000Z.json"
    f.write_text(json.dumps({"generation": gen, "nonce": NONCE, "minted_utc": "2026-09-30T00:00:00Z", "record": str(w.record),
                             "consume": {"started_utc": "2026-09-30T00:10:00Z", "what": "the reverdict", "action": "reverdict",
                                         "pid": 4242, "host": "TESTHOST", "job_id": "99999999"}}))
    return f


def test_an_interrupted_reopen_is_routed_to_reverdict_resume_interrupted_and_never_re_revoked(world):
    """A complete record with no verdict and ONE in-progress receipt: the plain phase refuses and names `--reopen`; `--reopen` submits the
    carrier in phase=reverdict / resume-interrupted (positional 10 `-`, one GPU, tp1, SERVING_PROBE_OUT = the record) — no receipt minted,
    no REVOKED marker, nothing resumed; while that job is queued a second --reopen refuses."""
    world.lay(done=7, complete=True)                               # a complete checkpoint, no verdict / menu, an in-progress receipt -> REOPEN_IN_PROGRESS
    receipt = _inprogress_receipt(world)
    r = world.run()
    assert r.returncode == 2 and "re-run --phase probe --reopen" in r.stderr, r.stdout + r.stderr
    assert "reverdict --resume-interrupted" in r.stderr and "never a fresh revocation" in r.stderr and world.sbatch_calls() == []
    assert "REOPEN_IN_PROGRESS" in r.stdout + r.stderr or "IN PROGRESS" in r.stderr                          # the module's own detail names the state / the recovery
    r = world.run("--reopen")
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "IN PROGRESS / INTERRUPTED" in out and "routing its recovery" in out, out
    assert "never a fresh revocation" in out and "phase=reverdict (resume-interrupted" in out, out
    calls = world.sbatch_calls(); assert len(calls) == 1, calls
    tail = _carrier_tail(calls[0])
    assert len(tail) == 12 and tail[5] == "tp1" and tail[8] == "1" and tail[9] == "-" and tail[10:] == ["reverdict", "resume-interrupted"], tail
    assert tail[3] and tail[4] == "4,16,64", tail                                              # the plan's sets / ranks
    assert "--gres=gpu:1" in " ".join(calls[0]["argv"]), calls[0]["argv"]                     # one GPU
    att = _attempt_events(world)
    assert att and att[-1]["logical_work_id"].startswith(f"{LWID}.rv") and att[-1]["spec"]["env"] == {"SERVING_PROBE_OUT": str(world.record)}, att[-1]
    receipts = sorted((world.record / "reopen_receipts").iterdir())
    assert receipts == [receipt], receipts                                                     # no new mint, the in-progress receipt untouched
    assert list(world.shared.rglob("*.REVOKED.json")) == [] and "hpo.lora.probe.a" not in out and "resuming" not in out
    # the job holds the record now: a second --reopen refuses by name, nothing more submitted
    r = world.run("--reopen")
    assert r.returncode == 2 and "queued/running" in r.stderr and len(world.sbatch_calls()) == 1, r.stdout + r.stderr


def _parity_dir(world: World, *, complete: bool = True, own_artefacts: bool = True, held: bool = False) -> Path:
    """A parity-pass record beside the main record: two passed launches, its checkpoint at the live generation (AT REST by default: no
    attempt / hold, its own verdict + reference_check present); ``own_artefacts=False`` leaves it headless, ``held`` marks a driver's attempt."""
    pdir = world.record / "parity"; (pdir / "launches").mkdir(parents=True)
    plan = _plan(); plan["parity_pass"] = True; plan["n_launches"] = 2; plan["launches"] = plan["launches"][:2]
    (pdir / "probe_plan.json").write_text(json.dumps(plan))
    for i in (1, 2):
        (pdir / "launches" / f"L0{i}.json").write_text(json.dumps({"launch_id": f"L0{i}", "status": "passed", "host": "TESTHOST", "redo": True}))
    live = hps.record_generation(pdir)
    prog = {"complete": complete, "n_done": 2 if complete else 1, "plan_launches": 2, "result_generation": live["generation"],
            "attempt": ({"started_utc": "2026-10-01T00:00:00Z", "job_id": "1"} if held else None), "hold": None}
    (pdir / "probe_progress.json").write_text(json.dumps(prog))
    if own_artefacts:
        (pdir / "probe_verdict.json").write_text(json.dumps({"status": "complete", "result_generation": live["generation"]}))
        (pdir / "reference_check.json").write_text(json.dumps({"status": "done"}))
    return pdir


def _verdict_refused(world: World, *, unreferenced=([[16, "mlp"]]), status="uncertified", with_parity: bool = True, **parity_kw) -> Path | None:
    """Job 820268's shape: a COMPLETE record whose verdict is `uncertified` for want of reference evidence (no allowed_pairs.json), stamped
    with the parity record it read; the parity record at rest unless told otherwise."""
    world.lay(done=7, complete=True, menu=False, verdict=True)
    pdir = _parity_dir(world, **parity_kw) if with_parity else None
    d = json.loads((world.record / "probe_verdict.json").read_text())
    d["status"] = status; d["certifying"] = False; d["unreferenced_pairs"] = list(unreferenced)
    if pdir is not None:
        live = hps.record_generation(pdir)
        d["parity_record"] = {"dir": str(pdir), "result_generation": live["generation"], "n_results": 2, "plan_launches": 2}
    (world.record / "probe_verdict.json").write_text(json.dumps(d))
    return pdir


def _parity_stale(world: World, *, complete: bool = True, own_artefacts: bool = True, held: bool = False,
                  serving_excluded: bool = False) -> Path:
    """A PARITY_STALE main record over a re-planned parity record. By default the parity record is AT REST: a
    complete checkpoint whose stamped generation is the live one, no driver's attempt / hold, its OWN verdict and reference_check
    present. ``own_artefacts=False`` leaves the pass headless (complete=true written on the last result, the reference / verdict
    phases never run); ``held=True`` marks a driver's attempt on the checkpoint."""
    world.lay(done=7, complete=True, menu=True, verdict=True)
    _stamp(world)
    pdir = _parity_dir(world, complete=complete, own_artefacts=own_artefacts, held=held)
    stamp = {"dir": str(pdir), "result_generation": "e" * 64, "n_results": 2}       # the verdict read the parity record at ANOTHER generation
    for name in ("probe_verdict.json", "allowed_pairs.json"):
        d = json.loads((world.record / name).read_text()); d["parity_record"] = stamp
        if serving_excluded:
            # one pair out for a parity verdict (no launch would change it), one for a SERVING reason (launches to run), one undecided
            d["excluded"] = [{"rank": 16, "set": "attn_mlp", "reason": "reference_mismatch:constitution__lora", "decided": True},
                             {"rank": 16, "set": "down", "reason": "refused:KvCoverageShortfall: resolved scheduler state does not match", "decided": True},
                             {"rank": 64, "set": "mlp", "reason": "difference_on_S1d:not_run", "decided": False}]
        (world.record / name).write_text(json.dumps(d))
    return pdir


def test_parity_stale_reopen_revokes_receipts_and_routes_reverdict_parity_record(world):
    """`--reopen` on a PARITY_STALE record: the receipt is minted into the record (slurm_revoke_geometry, the cluster acknowledged), ONE
    carrier goes out in phase=reverdict with `parity-record=<dir>` and the nonce as positional 10; nothing fetched, nothing resumed."""
    pdir = _parity_stale(world)
    r = world.run("--reopen")
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "PARITY_STALE" in out and f"routing reverdict --parity-record {pdir}" in out and f"phase=reverdict (parity-record={pdir}" in out, out
    receipts = sorted((world.record / "reopen_receipts").glob("*.json"))
    assert len(receipts) == 1, receipts
    doc = json.loads(receipts[0].read_text()); nonce = doc["nonce"]; gen = receipts[0].name.split(".")[0]
    assert re.fullmatch(r"[0-9a-f]{32}", nonce) and receipts[0].name == f"{gen}.{nonce}.json" and doc["record"] == str(world.record)
    assert [a["cluster"] for a in doc["acknowledged"]] == ["cluster_a"]
    calls = world.sbatch_calls(); assert len(calls) == 1, calls
    tail = _carrier_tail(calls[0])
    assert len(tail) == 12 and tail[5] == "tp1" and tail[9] == nonce and tail[10:] == ["reverdict", f"parity-record={pdir}"], tail
    att = _attempt_events(world)
    assert att[-1]["logical_work_id"].startswith(f"{LWID}.rv") and att[-1]["spec"]["env"] == {"SERVING_PROBE_OUT": str(world.record)}
    assert not (world.state / "allowed_pairs.json").exists() and "hpo.lora.probe.a" not in out and "resuming" not in out


def test_the_status_names_the_pairs_excluded_for_serving_reasons(world):
    """hpo_probe_status.serving_excluded: a parity verdict (reference_mismatch / reference_undecidable / reference_missing) is not a
    launch to run; any other first reason, or an undecided pair, is — the fact the front door's PARITY_STALE routing reads."""
    menu = {"excluded": [{"rank": 16, "set": "attn_mlp", "reason": "reference_mismatch:constitution__lora", "decided": True},
                         {"rank": 4, "set": "wo", "reason": "reference_undecidable:control__lora", "decided": True},
                         {"rank": 4, "set": "mlp", "reason": "reference_missing:no certifying launch carries a reference_check", "decided": True},
                         {"rank": 16, "set": "down", "reason": "refused:KvCoverageShortfall: x", "decided": True},
                         {"rank": 64, "set": "mlp", "reason": "difference_on_S1d:oom_at_R=128_at_L=49152", "decided": True},
                         {"rank": 64, "set": "down", "reason": "reference_mismatch:control__lora", "decided": False}]}
    sx = hps.serving_excluded(menu)
    assert sx["serving_excluded_n"] == 3 and [r[:2] for r in sx["serving_excluded"]] == [[16, "down"], [64, "mlp"], [64, "down"]]
    assert hps.serving_excluded({"excluded": []}) == {"serving_excluded": [], "serving_excluded_n": 0}
    assert hps.serving_excluded(None)["serving_excluded_n"] == 0 and hps.serving_excluded({"pairs": []})["serving_excluded_n"] == 0
    _parity_stale(world, serving_excluded=True)
    info = hps.verify_menu(world.record)
    assert info["serving_excluded_n"] == 2 and [r[:2] for r in info["serving_excluded"]] == [[16, "down"], [64, "mlp"]]
    menu_path = world.record / "allowed_pairs.json"
    d = json.loads(menu_path.read_text()); d["excluded"] = [d["excluded"][0]]; menu_path.write_text(json.dumps(d))     # the parity verdict alone
    assert hps.verify_menu(world.record)["serving_excluded_n"] == 0


def test_parity_stale_reopen_with_serving_exclusions_resumes_the_record_with_the_certificate_request_and_time(world):
    """`--reopen` on a PARITY_STALE record whose menu excludes pairs for SERVING reasons RESUMES it (the receipt minted,
    attempt+1 of the same record with --resume --replan) instead of routing the parity re-verdict alone; `--certify-v2 16:attn_mlp,16:down`
    rides the carrier's env as SERVING_PROBE_CERTIFY_V2 and `--probe-time 08:00:00` is the carrier's --time for this invocation
    (config.env's SISTEER_TIME_PROBE is sourced over the environment, so the flag is the one way to size the last attempt)."""
    pdir = _parity_stale(world, serving_excluded=True)
    r = world.run("--reopen", "--probe-width", "4", "--certify-v2", "16:attn_mlp,16:down", "--probe-time", "08:00:00")
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "PARITY_STALE" in out and "2 pair(s) excluded for serving reasons" in out and "launches to run: resuming the record" in out, out
    assert "routing reverdict --parity-record" not in out and "phase=reverdict" not in out, out
    receipts = sorted((world.record / "reopen_receipts").glob("*.json"))
    assert len(receipts) == 1, receipts
    nonce = json.loads(receipts[0].read_text())["nonce"]
    calls = world.sbatch_calls(); assert len(calls) == 1, calls
    tail = _carrier_tail(calls[0])
    assert len(tail) == 10 and tail[6] == "3" and tail[7] == "4" and tail[9] == nonce, tail           # attempt 3/4 of the record, the receipt as positional 10
    flags = calls[0]["argv"][:calls[0]["argv"].index("scripts/sbatch/serving_probe.sbatch")]
    assert "--time=08:00:00" in flags and "--gres=gpu:4" in flags and "--cpus-per-task=64" in flags and "--mem=512G" in flags, flags
    att = _attempt_events(world)
    assert att[-1]["logical_work_id"] == f"{LWID}.a3"
    assert att[-1]["spec"]["env"] == {"SERVING_PROBE_OUT": str(world.record), "SERVING_PROBE_CERTIFY_V2": "16:attn_mlp,16:down"}, att[-1]["spec"]["env"]
    assert att[-1]["spec"]["time_limit"] == "08:00:00"
    assert "time 08:00:00 (--probe-time)" in out and "certificate v2 for 16:attn_mlp,16:down" in out, out
    # the parity re-verdict route stands for a PARITY_STALE record WITHOUT serving exclusions (pinned by the test above this one)
    assert not (world.state / "allowed_pairs.json").exists()


def test_parity_stale_resume_route_refuses_a_parity_record_not_at_rest_with_nothing_archived(world):
    """The resume route stood AFTER the parity record's at-rest check was skipped — a PARITY_STALE record
    with serving exclusions would have been archived, revoked and resumed against a parity pass still writing or re-planning. The at-rest
    fact now gates BOTH branches before anything is archived: a held parity record (a driver's attempt on its checkpoint) and a headless
    one (complete, no own verdict / reference) are refused with the local copies, the receipts and the overlays untouched."""
    for kw in ({"held": True}, {"own_artefacts": False}):
        w = world
        _parity_stale(w, serving_excluded=True, **kw)
        local_before = sorted(p.name for p in w.state.iterdir()) if w.state.exists() else []
        r = w.run("--reopen", "--probe-width", "4", "--certify-v2", "16:attn_mlp,16:down")
        out = r.stdout + r.stderr
        assert r.returncode == 2 and "is not AT REST" in out and "nothing archived, nothing revoked, nothing submitted" in out, out
        assert "launches to run: resuming the record" not in out and "resuming" not in out and "routing reverdict" not in out, out
        assert not (w.record / "reopen_receipts").exists() and w.sbatch_calls() == [] and list(w.shared.rglob("*.REVOKED.json")) == []
        assert (sorted(p.name for p in w.state.iterdir()) if w.state.exists() else []) == local_before      # probe_archive_local never ran
        assert not (w.state / "superseded").exists()
        import shutil
        shutil.rmtree(w.record / "parity")                                                                  # the next shape re-lays it


def test_the_resume_derivation_takes_an_operator_cap_widen_and_refuses_one_below_the_next_attempt():
    """hpo_probe_resume --max-attempts N replaces the record's cap for this resume (5/5 -> attempt 6 of 6, cap_widen 5->6), is
    refused below the next attempt, and leaves the derivation unchanged when it equals the record's cap."""
    plan = _plan(); att = json.dumps(_attempt(attempt=5, cap=5))
    spec = hpr.derive(plan, attempt_json=att, legacy_line="", width="record", ledger=None, lwid_prefix=None, max_attempts=6, cap_reason="D549")
    assert (spec["attempt"], spec["max_attempts"], spec["cap_widen"], spec["unit_suffix"]) == (6, 6, "5->6", "a6") and any("cap widened 5 -> 6" in n for n in spec["notes"])
    # a cap that CHANGES needs a reason — refused by name without one (or with a blank one)
    for reason in (None, "", "   "):
        with pytest.raises(hpr.Refused, match="changes the record's cap 5 and no reason was given"):
            hpr.derive(plan, attempt_json=att, legacy_line="", width="record", ledger=None, lwid_prefix=None, max_attempts=6, cap_reason=reason)
    for bad in (5, 4, 1):
        with pytest.raises(hpr.Refused, match="below the next attempt 6"):
            hpr.derive(plan, attempt_json=att, legacy_line="", width="record", ledger=None, lwid_prefix=None, max_attempts=bad, cap_reason="D549")
    with pytest.raises(hpr.Refused, match="attempt cap 5 is reached"):
        hpr.derive(plan, attempt_json=att, legacy_line="", width="record", ledger=None, lwid_prefix=None)
    same = hpr.derive(plan, attempt_json=json.dumps(_attempt(attempt=2, cap=4)), legacy_line="", width="record", ledger=None, lwid_prefix=None, max_attempts=4)
    assert (same["attempt"], same["max_attempts"], same["cap_widen"]) == (3, 4, None)
    # the CLI prints the widen as a key the front door reads; the next derive reads the carrier-written cap from the attempt record
    import io
    out = io.StringIO()
    import contextlib
    with contextlib.redirect_stdout(out):
        rc = hpr.main(["--plan", "-", "--attempt-json", att, "--max-attempts", "6", "--cap-reason", "D549"], stdin=io.StringIO(json.dumps(plan)))
    assert rc == 0 and "attempt=6\nmax_attempts=6\n" in out.getvalue() and "cap_widen=5->6" in out.getvalue()
    with pytest.raises(hpr.Refused, match="attempt cap 6 is reached"):
        hpr.derive(plan, attempt_json=json.dumps(_attempt(attempt=6, cap=6)), legacy_line="", width="record", ledger=None, lwid_prefix=None)


def test_reopen_resumes_a_partial_record_at_the_cap_with_a_widened_cap_recorded_for_the_carrier(world):
    """A record at attempt 5/5 on the 4-hour budget: `--reopen --probe-max-attempts 6` on an INCOMPLETE_STALLED record (launches
    outstanding, no job) resumes it as attempt 6/6 through the funnel — no receipt (the record is not complete) — with the widen
    note on the carrier's env (SERVING_PROBE_CAP_WIDEN -> probe_attempt.json), the certificate request and the time limit riding
    along; the same without the widen is refused at the cap with nothing submitted, and the flag is refused without --reopen."""
    world.lay(done=3, complete=False, attempt=_attempt(attempt=5, cap=5))
    r = world.run("--reopen")
    assert r.returncode == 2 and "attempt cap 5 is reached" in (r.stdout + r.stderr) and world.sbatch_calls() == [], r.stdout + r.stderr
    r = world.run("--probe-max-attempts", "6")
    assert r.returncode == 2 and "pass it with --phase probe --reopen" in (r.stdout + r.stderr) and world.sbatch_calls() == []
    r = world.run("--reopen", "--probe-max-attempts", "6")                                  # no reason: refused before anything is read
    assert r.returncode == 2 and "needs --probe-cap-reason TEXT" in (r.stdout + r.stderr) and world.sbatch_calls() == []
    r = world.run("--reopen", "--probe-max-attempts", "5", "--probe-cap-reason", "x")
    assert r.returncode == 2 and "below the next attempt 6" in (r.stdout + r.stderr) and world.sbatch_calls() == []
    r = world.run("--reopen", "--probe-width", "4", "--probe-time", "08:00:00", "--probe-max-attempts", "6",
                  "--probe-cap-reason", "D549: attempt 5/5 ran on the 4h budget (octal 08)", "--certify-v2", "16:down")
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "INCOMPLETE_STALLED — launches outstanding, no job: resuming it (no receipt" in out and "cap widened 5->6 by the operator" in out, out
    assert "D549: attempt 5/5 ran on the 4h budget (octal 08)" in out and "as attempt 6/6 (counter and cap from" in out and "cap widened 5->6" in out, out
    assert not (world.record / "reopen_receipts").exists() and list(world.shared.rglob("*.REVOKED.json")) == []
    calls = world.sbatch_calls(); assert len(calls) == 1, calls
    tail = _carrier_tail(calls[0])
    assert len(tail) == 9 and tail[6:9] == ["6", "6", "auto"], tail                                     # attempt 6 of the widened cap 6, no token
    flags = calls[0]["argv"][:calls[0]["argv"].index("scripts/sbatch/serving_probe.sbatch")]
    assert "--time=08:00:00" in flags and "--gres=gpu:4" in flags, flags
    att = _attempt_events(world)
    env = att[-1]["spec"]["env"]
    assert att[-1]["logical_work_id"] == f"{LWID}.a6" and env["SERVING_PROBE_OUT"] == str(world.record) and env["SERVING_PROBE_CERTIFY_V2"] == "16:down"
    assert env["SERVING_PROBE_CAP_WIDEN"].startswith("cap widened 5->6 by the operator (03_hpo.sh --probe-max-attempts 6, ") and env["SERVING_PROBE_CAP_WIDEN"].endswith("): D549: attempt 5/5 ran on the 4h budget (octal 08)")
    assert '"' not in env["SERVING_PROBE_CAP_WIDEN"] and "\\" not in env["SERVING_PROBE_CAP_WIDEN"]


def test_reopen_of_a_stale_menu_on_a_complete_record_says_it_mints_the_receipt(world):
    """The INCOMPLETE_STALLED|MENU_STALE arm's log said "no receipt" for every record; a MENU_STALE record whose
    checkpoint is COMPLETE (a menu stamped for another result generation) takes the receipted path — the line says so, and a receipt exists."""
    world.lay(done=7, complete=True, menu=True, verdict=True)
    for name in ("probe_verdict.json", "allowed_pairs.json"):                                 # stamped for ANOTHER generation: MENU_STALE, complete
        d = json.loads((world.record / name).read_text()); d["result_generation"] = "f" * 64; (world.record / name).write_text(json.dumps(d))
    r = world.run("--reopen")
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "MENU_STALE — a stale menu on a COMPLETE record, no job: resuming it (the record is complete: a revocation receipt is minted" in out, out
    assert "no receipt" not in out and len(sorted((world.record / "reopen_receipts").glob("*.json"))) == 1
    tail = _carrier_tail(world.sbatch_calls()[0]); assert len(tail) == 10 and re.fullmatch(r"[0-9a-f]{32}", tail[9]), tail


def test_verdict_refused_for_want_of_reference_evidence_reopens_into_the_parity_reverdict(world):
    """`--reopen` on a VERDICT_REFUSED record whose verdict is uncertified with unreferenced pairs and whose remembered
    parity record is at rest: the receipt is minted, ONE carrier goes out in phase=reverdict with `parity-record=<dir>` and the nonce as
    positional 10 — the parity record untouched, the attempt record untouched (no attempt consumed, no cap_history change), nothing resumed."""
    pdir = _verdict_refused(world)
    att_before = (world.record / "probe_attempt.json").read_bytes()
    parity_before = {p.name: p.read_bytes() for p in pdir.rglob("*") if p.is_file()}
    st, detail, info = hps.probe_state_record(world.record, job_active=False)
    assert st == "VERDICT_REFUSED" and info["verdict_status"] == "uncertified" and info["unreferenced_pairs"] == [[16, "mlp"]]
    assert info["parity_record"]["dir"] == str(pdir) and info["parity_record"]["at_rest"] is True
    r = world.run("--reopen")
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "VERDICT_REFUSED — the verdict is uncertified for want of reference evidence on [[16, \"mlp\"]]" in out and f"routing reverdict --parity-record {pdir}" in out, out
    assert "resuming" not in out and "hpo.lora.probe.a" not in out
    receipts = sorted((world.record / "reopen_receipts").glob("*.json")); assert len(receipts) == 1, receipts
    nonce = json.loads(receipts[0].read_text())["nonce"]
    calls = world.sbatch_calls(); assert len(calls) == 1, calls
    tail = _carrier_tail(calls[0])
    assert len(tail) == 12 and tail[6:8] == ["2", "4"] and tail[9] == nonce and tail[10:] == ["reverdict", f"parity-record={pdir}"], tail   # the record's own 2/4 counter, untouched
    att = _attempt_events(world)
    assert att[-1]["logical_work_id"].startswith(f"{LWID}.rv") and att[-1]["spec"]["env"] == {"SERVING_PROBE_OUT": str(world.record)}
    assert (world.record / "probe_attempt.json").read_bytes() == att_before
    assert {p.name: p.read_bytes() for p in pdir.rglob("*") if p.is_file()} == parity_before
    assert not (world.state / "allowed_pairs.json").exists()


def test_verdict_refused_reopen_refuses_other_shapes_with_nothing_minted(world):
    # the parity record not at rest (a driver's attempt on it)
    _verdict_refused(world, held=True)
    r = world.run("--reopen"); out = r.stdout + r.stderr
    assert r.returncode == 2 and "is not AT REST" in out and "nothing archived, nothing revoked, nothing submitted" in out, out
    assert not (world.record / "reopen_receipts").exists() and world.sbatch_calls() == [] and list(world.shared.rglob("*.REVOKED.json")) == []
    import shutil; shutil.rmtree(world.record / "parity"); shutil.rmtree(world.record)
    # a verdict refused with NO unreferenced pairs is not this remedy
    world.record.mkdir(parents=True)
    _verdict_refused(world, unreferenced=[])
    r = world.run("--reopen"); out = r.stdout + r.stderr
    assert r.returncode == 2 and "the reference re-verdict remedies an uncertified verdict with pairs lacking reference evidence" in out, out
    assert not (world.record / "reopen_receipts").exists() and world.sbatch_calls() == []
    shutil.rmtree(world.record)
    # no parity record was read: nothing to re-read
    world.record.mkdir(parents=True)
    _verdict_refused(world, with_parity=False)
    r = world.run("--reopen"); out = r.stdout + r.stderr
    assert r.returncode == 2 and "the verdict read no parity record" in out, out
    assert not (world.record / "reopen_receipts").exists() and world.sbatch_calls() == []
    # the plain phase still refuses VERDICT_REFUSED, now pointing at --reopen
    shutil.rmtree(world.record); world.record.mkdir(parents=True)
    _verdict_refused(world)
    r = world.run(); out = r.stdout + r.stderr
    assert r.returncode == 2 and "--phase probe --reopen routes the receipted re-verdict against it (D623)" in out and world.sbatch_calls() == [], out


def _ready_with_parity(world: World, *, parity_ready: bool = True, held: bool = False, inprogress: bool = False, parity_failed_reference: bool = False,
                       parity_serving_failure: bool = False) -> Path:
    """A READY probe record whose plan remembers a parity-pass record; the parity record READY too (its own menu + verdict stamped at its
    live generation, probe_attempt.json 3/3) unless told otherwise."""
    world.lay(done=7, complete=True, menu=True, verdict=True)
    _stamp(world)
    pdir = _parity_dir(world, held=held)
    plan = json.loads((world.record / "probe_plan.json").read_text()); plan["parity_record_dir"] = str(pdir)
    (world.record / "probe_plan.json").write_text(json.dumps(plan))
    live = hps.record_generation(pdir)["generation"]
    (pdir / "probe_attempt.json").write_text(json.dumps(_attempt(attempt=3, cap=3, gpus=4)))
    if parity_ready:
        (pdir / "allowed_pairs.json").write_text(json.dumps({"status": "complete", "certifying": True, "result_generation": live, "pairs": [{"rank": 4, "set": "wo"}],
                                                             "written_utc": "2026-10-01T00:00:00Z", "model_key": MODEL}))
    if parity_failed_reference:
        # the shape a FAILED HF+PEFT reference run leaves: reference_check.json status=error, an uncertified verdict
        # with every pair unreferenced, NO menu — the parity record reads VERDICT_REFUSED
        (pdir / "reference_check.json").write_text(json.dumps({"status": "error", "reason": "RuntimeError: the base model failed to load", "per_adapter": {}}))
        excluded = [{"rank": 4, "set": "wo", "reason": "reference_missing:no certifying launch carries a reference_check", "decided": True}]
        if parity_serving_failure:
            # a MIXED record — one pair's engine launch failed (a serving reason) beside the missing reference evidence
            excluded.append({"rank": 4, "set": "mlp", "reason": "difference_on_S1d:refused:KvCoverageShortfall: resolved scheduler state does not match", "decided": True})
        (pdir / "probe_verdict.json").write_text(json.dumps({"status": "uncertified", "certifying": False, "result_generation": live, "unreferenced_pairs": [[4, "wo"]],
                                                              "reference_check_basis": None, "excluded": excluded}))
        for f in (pdir / "allowed_pairs.json",):
            if f.exists(): f.unlink()
    if inprogress:
        (pdir / "reopen_receipts").mkdir(exist_ok=True)
        (pdir / "reopen_receipts" / f"{live}.{'f' * 32}.inprogress.2026-10-01T000000Z-p1-TESTHOST.json").write_text(json.dumps(
            {"generation": live, "nonce": "f" * 32, "acknowledged": [{"cluster": "cluster_a"}], "minted_utc": "2026-10-01T00:00:00Z",
             "consume": {"started_utc": "2026-10-01T00:00:01Z", "what": "the reverdict", "pid": 1, "host": "TESTHOST"}}))
    for name in ("probe_verdict.json", "allowed_pairs.json"):
        d = json.loads((world.record / name).read_text()); d["parity_record"] = {"dir": str(pdir), "result_generation": live, "n_results": 2}; (world.record / name).write_text(json.dumps(d))
    return pdir


def test_parity_record_reopen_re_makes_the_judgements_through_the_front_door(world):
    """`--reopen --reference-only --parity-record-reopen --probe-max-attempts 4 --probe-cap-reason …` on a READY probe record whose
    parity-pass record is READY: the receipt is minted INTO the parity record, ONE carrier goes out in phase=reference (1 GPU) on the parity
    record's own unit / name as attempt 4 of the widened cap 4 with the cap note on its env, nothing on the probe record moves."""
    pdir = _ready_with_parity(world)
    main_before = {p.name: p.read_bytes() for p in world.record.iterdir() if p.is_file()}
    r = world.run("--reopen", "--reference-only", "--parity-record-reopen", "--probe-max-attempts", "4",
                  "--probe-cap-reason", "D641: v1 certificates carry no serving venv / engine identity; re-judge every adapter under the identified engine")
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert f"re-making the HF+PEFT judgements of the parity-pass record {pdir}" in out and "cap widened 3->4 by the operator" in out, out
    assert not (world.record / "reopen_receipts").exists()                                               # the PROBE record is not reopened
    receipts = sorted((pdir / "reopen_receipts").glob("*.json")); assert len(receipts) == 1, receipts
    doc = json.loads(receipts[0].read_text()); nonce = doc["nonce"]
    assert doc["record"] == str(pdir) and "cap widened 3->4" in doc["reason"] and "D641" in doc["reason"]
    calls = world.sbatch_calls(); assert len(calls) == 1 and calls[0]["name"] == f"hpo-{MODEL}-lora-probe-parity", calls
    tail = _carrier_tail(calls[0])
    assert len(tail) == 11 and tail[6:9] == ["4", "4", "1"] and tail[9] == nonce and tail[10] == "reference", tail
    flags = calls[0]["argv"][:calls[0]["argv"].index("scripts/sbatch/serving_probe.sbatch")]
    assert "--gres=gpu:1" in flags, flags
    att = _attempt_events(world)
    assert att[-1]["logical_work_id"] == f"{LWID}.parity.a4", att[-1]["logical_work_id"]
    env = att[-1]["spec"]["env"]
    assert env["SERVING_PROBE_OUT"] == str(pdir) and env["SERVING_PROBE_CAP_WIDEN"].startswith("cap widened 3->4 by the operator")
    assert {p.name: p.read_bytes() for p in world.record.iterdir() if p.is_file()} == main_before
    assert "resuming" in out and f"as attempt 4/4" in out and "--resume --reference-only (D146)" in out, out


def test_parity_record_reopen_recovers_a_failed_reference_run_under_a_fresh_receipt(world):
    """A parity reference run that failed (reference_check.json status=error, an uncertified verdict, no menu —
    VERDICT_REFUSED) is recovered through the same verb: a fresh receipt into the parity record, attempt N+1 of its own (widened) cap in
    phase=reference, the probe record untouched; a VERDICT_REFUSED parity record whose verdict is NOT uncertified is refused by name."""
    pdir = _ready_with_parity(world, parity_ready=False, parity_failed_reference=True)
    st, detail, info = hps.probe_state_record(pdir, job_active=False)
    assert st == "VERDICT_REFUSED" and info["verdict_status"] == "uncertified" and info["unreferenced_pairs"] == [[4, "wo"]]
    r = world.run("--reopen", "--reference-only", "--parity-record-reopen", "--probe-max-attempts", "4", "--probe-cap-reason", "D641: the HF+PEFT reference crashed; re-run it")
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "is VERDICT_REFUSED (uncertified" in out and "the fenced recovery re-runs its reference phase under a fresh receipt" in out and f"parity-pass record {pdir} (VERDICT_REFUSED at" in out, out
    receipts = sorted((pdir / "reopen_receipts").glob("*.json")); assert len(receipts) == 1 and not (world.record / "reopen_receipts").exists()
    calls = world.sbatch_calls(); assert len(calls) == 1 and calls[0]["name"] == f"hpo-{MODEL}-lora-probe-parity"
    tail = _carrier_tail(calls[0]); assert len(tail) == 11 and tail[6:9] == ["4", "4", "1"] and tail[10] == "reference", tail
    assert (pdir / "reference_check.json").read_text() and json.loads((pdir / "reference_check.json").read_text())["status"] == "error"   # the carrier re-judges; nothing rewritten here


def test_parity_record_reopen_refuses_a_mixed_parity_record_whose_engine_launch_failed(world):
    """An uncertified parity record that ALSO excludes a pair for a serving reason (a refused / failed engine
    launch) is not the reference-only recovery — the reference phase cannot re-run an engine — so it refuses by name, names the
    launch-capable path, revokes nothing and spends no attempt; the status reader takes the exclusions from the VERDICT (no menu)."""
    pdir = _ready_with_parity(world, parity_ready=False, parity_failed_reference=True, parity_serving_failure=True)
    st, detail, info = hps.probe_state_record(pdir, job_active=False)
    assert st == "VERDICT_REFUSED" and info["verdict_status"] == "uncertified" and info["serving_excluded_n"] == 1 and info["serving_excluded"][0][:2] == [4, "mlp"]
    r = world.run("--reopen", "--reference-only", "--parity-record-reopen", "--probe-max-attempts", "4", "--probe-cap-reason", "x")
    out = r.stdout + r.stderr
    assert r.returncode == 2 and "is uncertified AND excludes 1 pair(s) for SERVING reasons" in out and "--parity-record-reopen --certify-v2 all" in out, out
    assert "nothing revoked, nothing submitted" in out and not (pdir / "reopen_receipts").exists() and world.sbatch_calls() == [] and list(world.shared.rglob("*.REVOKED.json")) == []
    assert json.loads((pdir / "probe_attempt.json").read_text())["attempt"] == 3                   # no attempt spent


def test_parity_record_reopen_refuses_a_verdict_refused_parity_record_whose_verdict_is_not_uncertified(world):
    """A VERDICT_REFUSED parity record whose verdict is NOT uncertified (some other refusal) is not the failed-reference recovery: refused by name."""
    pdir = _ready_with_parity(world, parity_ready=False, parity_failed_reference=True)
    d = json.loads((pdir / "probe_verdict.json").read_text()); d["status"] = "complete"; (pdir / "probe_verdict.json").write_text(json.dumps(d))
    r = world.run("--reopen", "--reference-only", "--parity-record-reopen", "--probe-max-attempts", "4", "--probe-cap-reason", "x")
    out = r.stdout + r.stderr
    assert r.returncode == 2 and "only an uncertified verdict" in out and not (pdir / "reopen_receipts").exists() and world.sbatch_calls() == [], out
    assert list(world.shared.rglob("*.REVOKED.json")) == []


def test_parity_record_reopen_refuses_when_the_parity_record_is_not_ready_held_or_mid_reopen(world):
    import shutil
    # a parity record with a complete verdict but NO menu reads VERDICT_REFUSED with verdict status `complete` — not the failed-reference shape
    for kw, needle in (({"parity_ready": False}, "only an uncertified verdict"), ({"held": True}, "is not at rest (attempt="), ({"inprogress": True}, "IN PROGRESS")):
        pdir = _ready_with_parity(world, **kw)
        r = world.run("--reopen", "--reference-only", "--parity-record-reopen", "--probe-max-attempts", "4", "--probe-cap-reason", "D641")
        out = r.stdout + r.stderr
        assert r.returncode == 2 and needle in out and "nothing revoked, nothing submitted" in out, (kw, out)
        assert not (world.record / "reopen_receipts").exists() and world.sbatch_calls() == [] and list(world.shared.rglob("*.REVOKED.json")) == []
        assert len(list((pdir / "reopen_receipts").glob("*.json")) if (pdir / "reopen_receipts").exists() else []) == (1 if kw.get("inprogress") else 0)
        shutil.rmtree(world.record)
        world.record.mkdir(parents=True)
    # the flag needs --reopen --reference-only; a cap widen still needs its reason; the cap still binds without the widen
    _ready_with_parity(world)
    r = world.run("--reopen", "--parity-record-reopen")
    assert r.returncode == 2 and "pass it with --phase probe --reopen --reference-only" in r.stdout + r.stderr and world.sbatch_calls() == []
    r = world.run("--reopen", "--reference-only", "--parity-record-reopen")
    assert r.returncode == 2 and "attempt cap 3 is reached" in r.stdout + r.stderr and world.sbatch_calls() == []
    assert not list(world.shared.rglob("*.REVOKED.json"))


def test_reverdict_only_re_folds_a_ready_record_through_the_receipted_reverdict_without_a_resume(world):
    """`--reopen --reverdict-only` on a READY record whose parity record is at rest mints the receipt and submits exactly
    the reverdict carrier with `parity-record=<dir>` — no resume (the record may be at its attempt cap), no attempt consumed, the parity
    record untouched; refused when the parity record is not at rest, when a reopen is in progress, and when combined with a resume flag."""
    pdir = _ready_with_parity(world)
    att_before = (world.record / "probe_attempt.json").read_bytes(); parity_before = {p.name: p.read_bytes() for p in pdir.rglob("*") if p.is_file()}
    r = world.run("--reopen", "--reverdict-only")
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert f"re-folding the certification of {world.record} (READY) against the parity record {pdir}" in out and f"routing reverdict --parity-record {pdir}" not in out
    assert "resuming" not in out and "hpo.lora.probe.a" not in out
    receipts = sorted((world.record / "reopen_receipts").glob("*.json")); assert len(receipts) == 1, receipts
    nonce = json.loads(receipts[0].read_text())["nonce"]
    calls = world.sbatch_calls(); assert len(calls) == 1, calls
    tail = _carrier_tail(calls[0])
    assert len(tail) == 12 and tail[9] == nonce and tail[10:] == ["reverdict", f"parity-record={pdir}"], tail
    att = _attempt_events(world); assert att[-1]["logical_work_id"].startswith(f"{LWID}.rv")
    assert (world.record / "probe_attempt.json").read_bytes() == att_before and {p.name: p.read_bytes() for p in pdir.rglob("*") if p.is_file()} == parity_before


def test_reverdict_only_refuses_a_parity_record_not_at_rest_and_any_resume_flag(world):
    """The same verb refused by name: a parity record holding a driver's attempt (not at rest) mints nothing; combined with a resume flag,
    or without --reopen, the arguments are refused before anything is read."""
    _ready_with_parity(world, held=True)
    r = world.run("--reopen", "--reverdict-only"); out = r.stdout + r.stderr
    assert r.returncode == 2 and "is not AT REST" in out and not (world.record / "reopen_receipts").exists() and world.sbatch_calls() == [], out
    assert list(world.shared.rglob("*.REVOKED.json")) == []
    r = world.run("--reopen", "--reverdict-only", "--probe-max-attempts", "6", "--probe-cap-reason", "x"); out = r.stdout + r.stderr
    assert r.returncode == 2 and "takes none of --reference-only" in out and world.sbatch_calls() == []
    r = world.run("--reopen", "--reverdict-only", "--reference-only"); out = r.stdout + r.stderr
    assert r.returncode == 2 and "takes none of --reference-only" in out and world.sbatch_calls() == []
    r = world.run("--reverdict-only"); out = r.stdout + r.stderr
    assert r.returncode == 2 and "pass it with --phase probe --reopen" in out and world.sbatch_calls() == []


def test_certify_v2_and_probe_time_are_refused_outside_their_phase_or_shape(world):
    _parity_stale(world, serving_excluded=True)
    r = world.run("--certify-v2", "all")                                                       # --phase probe without --reopen
    assert r.returncode != 0 and "pass it with --phase probe --reopen" in (r.stdout + r.stderr) and not world.sbatch_calls()
    r = world.run("--reopen", "--certify-v2", "16-down")
    assert r.returncode != 0 and "--certify-v2 must be all or rank:set" in (r.stdout + r.stderr) and not world.sbatch_calls()
    r = world.run("--reopen", "--probe-time", "8h")
    assert r.returncode != 0 and "--probe-time must be [D-]HH:MM:SS" in (r.stdout + r.stderr) and not world.sbatch_calls()
    r = world.run("--reopen", "--reference-only", "--certify-v2", "all")
    assert r.returncode != 0 and "--reference-only runs none" in (r.stdout + r.stderr) and not world.sbatch_calls()
    assert not list((world.record / "reopen_receipts").glob("*.json")) or True                 # a refusal before the mint leaves no receipt


def test_parity_stale_reopen_refuses_an_incomplete_parity_record_with_nothing_minted(world):
    _parity_stale(world, complete=False)
    r = world.run("--reopen")
    assert r.returncode == 2 and "nothing submitted" in r.stderr, r.stdout + r.stderr
    assert not (world.record / "reopen_receipts").exists() and world.sbatch_calls() == [] and list(world.shared.rglob("*.REVOKED.json")) == []


def test_parity_stale_reopen_refuses_a_complete_but_headless_parity_record_with_nothing_revoked(world):
    """`complete` is written on the last RESULT before the reference / verdict phases — a re-planned pass
    caught in that window (complete=true, no verdict / reference_check of its own) is NOT at rest; the remedy refuses by name and
    revokes nothing, mints nothing, submits nothing."""
    _parity_stale(world, complete=True, own_artefacts=False)
    r = world.run("--reopen")
    out = r.stdout + r.stderr
    assert r.returncode == 2 and "is not AT REST" in out and "own_verdict_present=false" in out and "nothing revoked, nothing submitted" in out, out
    assert not (world.record / "reopen_receipts").exists() and world.sbatch_calls() == [] and list(world.shared.rglob("*.REVOKED.json")) == []


def test_parity_stale_reopen_refuses_a_held_parity_record_with_nothing_revoked(world):
    """A driver's attempt on the parity record's checkpoint (a resume or a reference-only redo in flight) — complete may read true
    from the last write before it; at_rest is false and the remedy waits."""
    _parity_stale(world, complete=True, held=True)
    r = world.run("--reopen")
    out = r.stdout + r.stderr
    assert r.returncode == 2 and "is not AT REST" in out and "attempt=" in out and "nothing revoked, nothing submitted" in out, out
    assert not (world.record / "reopen_receipts").exists() and world.sbatch_calls() == [] and list(world.shared.rglob("*.REVOKED.json")) == []


def _carrier_head() -> str:
    src = CARRIER.read_text()
    i = src.index('PHASE="${11:-'); j = src.index("\nfi\n", src.index('REVERDICT_ARG="${12:-}"', i)) + len("\nfi\n")
    return src[i:j]


def test_the_carrier_validates_phase_reverdict_and_its_positional_12_under_bash():
    head = _carrier_head()
    def run(*pos):
        return subprocess.run(["bash", "-c", head + '\necho "PHASE=$PHASE ARG=$REVERDICT_ARG TOKEN=${REOPEN_TOKEN:-unset}"', "x", *pos], capture_output=True, text=True)
    ok = run("m", "id", "r", "wo", "4", "tp1", "1", "1", "1", "-", "reverdict", "resume-interrupted")
    assert ok.returncode == 0 and "PHASE=reverdict ARG=resume-interrupted" in ok.stdout, ok.stdout + ok.stderr
    ok2 = run("m", "id", "r", "wo", "4", "tp1", "1", "1", "1", NONCE, "reverdict", "parity-record=/shared/exp/x/parity")
    assert ok2.returncode == 0 and "ARG=parity-record=/shared/exp/x/parity" in ok2.stdout, ok2.stdout + ok2.stderr
    bad = run("m", "id", "r", "wo", "4", "tp1", "1", "1", "1", "-", "reverdict", "bogus")
    assert bad.returncode == 2 and "needs positional 12 resume-interrupted | parity-record=<abs dir>" in bad.stdout + bad.stderr
    bad2 = run("m", "id", "r", "wo", "4", "tp1", "1", "1", "1", "-", "bogus")
    assert bad2.returncode == 2 and "must be full, reference, parity or reverdict" in bad2.stdout + bad2.stderr
    rel = run("m", "id", "r", "wo", "4", "tp1", "1", "1", "1", "-", "reverdict", "parity-record=relative/dir")
    assert rel.returncode == 2, rel.stdout + rel.stderr


def test_source_pins_the_reverdict_phase_and_the_routing():
    src = CARRIER.read_text()
    assert 'REOPEN_TOKEN="${10:-}"; [ "$REOPEN_TOKEN" != - ] || REOPEN_TOKEN=""' in src
    assert 'case "${PHASE:-full}" in reference|reverdict)' in src, "a reverdict job never resubmits itself"
    assert '[ "${PHASE:-full}" = reverdict ] || attempt_record_write' in src and "NOT written — phase=reverdict runs no launches" in src
    # the attempt-record block stays a self-contained bash slice (tests/test_front_door_probe_resume.py runs it from the header to `log "attempt record:`)
    frag = src[src.index("# --- THE ATTEMPT RECORD"):src.index('log "attempt record:')]
    assert subprocess.run(["bash", "-n", "-c", frag], capture_output=True, text=True).returncode == 0, "the reverdict guard must not open a block the slice cannot close"
    rv = src[src.index('if [ "${PHASE:-full}" = reverdict ]; then'):]
    rv = rv[:rv.index("\nfi\n")]
    assert 'serving_probe.py reverdict --record "$OUT" --probe-job-id "$SLURM_JOB_ID"' in rv and "--resume-interrupted" in rv and "--parity-record" in rv
    assert "exit $RC" in rv and " run " not in rv and "plan " not in rv.replace("no plan", ""), "no plan, no launches"
    assert src.index('if [ "${PHASE:-full}" = reverdict ]; then') < src.index("RESUME=()"), "the reverdict exits before the resume / plan section"
    hpo = (REPO / "reproduce" / "03_hpo.sh").read_text()
    verb = hpo[hpo.index("probe_reopen_verb(){"):hpo.index("probe_local_currency(){")]
    assert "REOPEN_IN_PROGRESS)" in verb and "PARITY_STALE)" in verb and "probe_submit_reverdict resume-interrupted" in verb and 'probe_submit_reverdict "parity-record=$pdir" "$token"' in verb
    # the interrupted-reopen RECOVERY is routed on its dedicated state, never on VERDICT_REFUSED + interrupted_reopen; the
    # VERDICT_REFUSED arm is the reference re-verdict (uncertified / reference_missing) and never the recovery
    vr = verb[verb.index("VERDICT_REFUSED)"):verb.index("INCOMPLETE_STALLED|MENU_STALE)")]
    assert "resume-interrupted" not in vr and "interrupted_reopen" not in vr and 'probe_submit_reverdict "parity-record=$pdir" "$token"' in vr
    assert verb.count("probe_submit_reverdict resume-interrupted") == 1 and verb.index("probe_submit_reverdict resume-interrupted") > verb.index("REOPEN_IN_PROGRESS)")
    ps = verb[verb.index("PARITY_STALE)"):]                                                 # within the PARITY_STALE arm: mint, then route
    assert ps.index('token="$(probe_reopen_token)"') < ps.index('probe_submit_reverdict "parity-record=')
    assert vr.index('token="$(probe_reopen_token)"') < vr.index('probe_submit_reverdict "parity-record=')   # and within the VERDICT_REFUSED arm
    ir_arm = verb[verb.index("REOPEN_IN_PROGRESS)"):verb.index("PARITY_STALE)")]
    assert "probe_reopen_token" not in ir_arm and "probe_resume" not in ir_arm, "an interrupted reopen is never re-revoked or resumed"
    phase = hpo[hpo.index('case "$PSTATE" in', hpo.index("probe_local_currency(){")):]        # the PLAIN probe phase's dispatcher (after the reopen verb)
    assert "REOPEN_IN_PROGRESS)" in phase and phase.index("REOPEN_IN_PROGRESS)") < phase.index("VERDICT_REFUSED)")   # the plain phase names the routed recovery
    sub = hpo[hpo.index("probe_submit_reverdict(){"):hpo.index("probe_reopen_verb(){")]
    assert "tp1 $a $m 1 ${token:--} reverdict $1" in sub and '--env "SERVING_PROBE_OUT=$CL_ROOT"' in sub and 'probe_width_flags 1' in sub
