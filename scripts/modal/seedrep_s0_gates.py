#!/usr/bin/env python3
"""seedrep_s0_gates.py -- the ONE enumerated S0 runner (the ten proof-set gates + the argv gate + the ledger-vs-quota check +
the exact per-child identity contents), FAIL-CLOSED: S0 used to run five test files and omitted request-evidence
reconciliation, retry binding, controls ancestry / event timing and the accepted-only publication gates; preflight accepted missing
generators as ABSENT; identity files were checked for JSON shape only.

    <venv python> scripts/modal/seedrep_s0_gates.py --tree <W checkout> --plan RS/launch_plan_draft.json --rs RS --quota <containers> \\
        --campaign <registry campaign id> [--s5-references RS/smokes.json] [--launch-set <ids,...|@file>] [--out RS/s0_gates.json] [--lock $STATE/sisteer-pytest/lock]
    <venv python> scripts/modal/seedrep_s0_gates.py --list        # the enumerated gates, no run
    <venv python> scripts/modal/seedrep_s0_gates.py --selftest    # the table's own invariants (pure)

Every gate is a NAMED entry of :data:`GATES` (id, the §8 text, the owning workstream, what runs). A pytest gate PASSES only when its
targets exist, collect, and finish with zero failed / errored / SKIPPED tests (a skip is a gate that did not run -- e.g. the GSM8K reader
gate skipping because the tree predates #196 -- and so FAILS S0 by name); a python gate passes on its own predicate. A missing test
file, a missing peer module, a missing input (quota, campaign) each FAIL the gate that needs it. The report (`RS/s0_gates.json`) lists
every gate's verdict and evidence; the exit code is 0 only when EVERY gate passed. On the grading host every pytest invocation goes through the
shared serialisation lock (`--lock`, default the grading host lock when it exists).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import seedrep_launch_plan as lp  # noqa: E402


def default_lock() -> Path:
    """The grading host pytest serialisation lock: the site's SISTEER_PYTEST_LOCK, else ``$STATE/sisteer-pytest/lock``,
    else ``~/.local/state/sisteer-pytest/lock``."""
    from sisteer import site

    return site.state_path("SISTEER_PYTEST_LOCK", "sisteer-pytest", "lock")


REGISTRY_PLANS_REL = "scripts/sdfbase/seedrep/registry_plans.py"          # the producer (the exact identity contents)
S0_SCHEMA = "seedrep_s0_gates/1"

#: The enumerated S0 gates, in order. `pytest`: node ids / files (with an optional `-k` filter) run under the tree; `python`: a predicate
#: of this module. `depends`: files that must exist in the tree BEFORE the gate runs (a missing one FAILS the gate by name).
GATES: List[Dict[str, Any]] = [
    {"id": "G1", "plan": "(1) every planned GSM8K row through the real post-#196 reader at the common geometry -- admitted; rev 2's 18,432 refused",
     "owner": "WS2-D / WS0", "pytest": ["tests/test_seedrep_standup_argv.py -k gate1"], "depends": ["scripts/tinycap_math500_thinking.py"]},
    {"id": "G2", "plan": "(2) the production headers of EVERY leg through request_evidence_record -> non-null item_id, numeric sample_idx, the seed closure, "
                        "AND the FULL request_evidence_reconciliation per leg against store-derived filed pairs; the gsm8k/none mislabel and the inspect/xstest "
                        "labels REFUSED", "owner": "WS1-D' / WS3-G",
     "pytest": ["tests/test_seedrep_request_evidence.py", "tests/test_seedrep_request_evidence_audit.py"],
     "depends": ["src/sisteer/eval/request_evidence_client.py"]},
    {"id": "G3", "plan": "(3) the retry-binding test (§9.1): a forced first-attempt failure + retry binds to exactly ONE engine-log request", "owner": "WS1-D'",
     "pytest": ["tests/test_seedrep_request_evidence.py -k retry"], "depends": ["src/sisteer/eval/request_evidence_client.py"]},
    {"id": "G4", "plan": "(4) controls ancestry 512 -> 2,048 -> 8,192 -> 16,384 materialises and the first controls_events.jsonl event lands before the fan-out completes",
     "owner": "WS1-C / WS2-E", "pytest": ["tests/test_sampled_controls_drift.py", "tests/test_seedrep_ladder_protocol.py -k controls"],
     "depends": ["scripts/modal/seedrep_ladder.py"]},
    {"id": "G5", "plan": "(5) the two-snapshot election + crash-after-acceptance closure tests", "owner": "WS2-D / WS2-E",
     "pytest": ["tests/test_seedrep_attempts.py"], "depends": []},
    {"id": "G6", "plan": "(6) every Dict name / key carries (W, wave_uuid, leg, view, stage/cap, shard)", "owner": "WS2-D / WS2-E",
     "pytest": ["tests/test_seedrep_control_keys.py"], "depends": []},
    {"id": "G7", "plan": "(7) two shards + one superseded attempt cannot publish outside the accepted union", "owner": "WS2-E",
     # the superseded-attempt closure tests live in tests/test_seedrep_ladder_attempts.py (renamed from
     # test_seedrep_attempts.py at 8ffaebd4) and the accepted-union publication test in tests/test_seedrep_registry_publish.py; an
     # earlier table named tests/test_seedrep_ladder_protocol.py, where `-k superseded` selects nothing (rc 5) --
     # tests/test_seedrep_w3_s0_gate_selection.py holds every pytest gate to a non-empty collection on the real tree
     "pytest": ["tests/test_seedrep_ladder_attempts.py -k superseded", "tests/test_seedrep_registry_publish.py -k superseded"],
     "depends": ["scripts/modal/seedrep_ladder.py"]},
    {"id": "G8", "plan": "(8) planned standups - S5 references = empty (re-asserted after S5; at S0 the roster derives from the plan and, when RS/smokes.json "
                        "exists, the gap is empty -- fc10 / D139 / D146: over the LAUNCHED rows' window classes when --launch-set is given, a class satisfied by its "
                        "own complete reference or one of a larger window (dominated_by), a prior-W / wave reference admitted through its standup's records)",
     "owner": "WS2-D", "python": "gate8_s5_roster", "depends": []},
    {"id": "G9", "plan": "(9) --max-logprobs 64 on every MMLU standup argv + the 20-vs-64 negative", "owner": "WS2-D",
     "pytest": ["tests/test_seedrep_standup_argv.py -k logprobs", "tests/test_seedrep_served_app.py -k logprobs"], "depends": []},
    {"id": "G10", "plan": "(10) adapter reuse enumerated per intervention hash (gate 10)", "owner": "WS2-D",
     "pytest": ["tests/test_seedrep_launch_plan.py -k adapter"], "depends": []},
    {"id": "ARGV", "plan": "tests/test_seedrep_standup_argv.py over every production argv (§4.3)", "owner": "WS2-D",
     "pytest": ["tests/test_seedrep_standup_argv.py"], "depends": ["scripts/sdfbase/mt_serving_up.py"]},
    {"id": "LEDGER", "plan": "the capacity ledger vs the Modal workspace quota (§13): unstaged peak containers <= --quota, else the launch is STAGED by name",
     "owner": "WS2-D", "python": "gate_ledger_vs_quota", "depends": []},
    {"id": "PLAN", "plan": "the plan validates (validate_plan == []) against the tree's geometry table", "owner": "WS2-D", "python": "gate_plan_validates", "depends": []},
    {"id": "DOSE", "plan": "D71: Strength's dose sweep runs on cluster A -- NOT IN THIS WAVE (no dose standups, no ledger line); the D68 peer-roster "
                          "registration path is dormant (--register-peer-dose-standups) and, when switched on, reports REGISTERED / NOT REGISTERED "
                          "(pending_peer_spec) without failing the s1-s3 legs",
     "owner": "WS2-D / Strength", "python": "gate_dose_registration", "depends": []},
    {"id": "IDENTITY", "plan": "every child's registry identity file exists under RS/registry/plans with EXACTLY the contents WS4-H's producer (registry_plans.build) "
                              "renders for this W / campaign / roster / pins (check_plan_set), and loads as the container reads it (load_plan_file)",
     "owner": "WS4-H / WS2-D", "python": "gate_identity_contents", "depends": [REGISTRY_PLANS_REL]},
]
PYTEST_SUMMARY_RE = re.compile(r"(?P<n>\d+) (?P<what>passed|failed|skipped|error|errors|deselected|warnings?)")


def list_gates() -> List[Dict[str, Any]]:
    return [{k: g[k] for k in ("id", "plan", "owner")} | {"runs": g.get("pytest") or [f"python:{g['python']}"]} for g in GATES]


def parse_pytest_summary(text: str) -> Dict[str, int]:
    """The counts off pytest's last summary line (`3 passed, 1 skipped in 2.1s`); every kind absent = 0."""
    last = ""
    for line in text.splitlines():
        if re.search(r"\b(passed|failed|skipped|error|no tests ran)\b", line) and (" in " in line or "no tests ran" in line):
            last = line
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    for m in PYTEST_SUMMARY_RE.finditer(last):
        what = m.group("what")
        key = "errors" if what.startswith("error") else what
        if key in counts:
            counts[key] += int(m.group("n"))
    counts["summary_line"] = last.strip()
    return counts


def pytest_verdict(rc: int, counts: Mapping[str, Any]) -> Optional[str]:
    """None = PASS; else the named reason. A skip is a gate that did not run; zero collected is a gate with no evidence."""
    if counts["failed"] or counts["errors"]:
        return f"{counts['failed']} failed / {counts['errors']} errors"
    if counts["skipped"]:
        return f"{counts['skipped']} SKIPPED test(s) -- a skipped gate did not run (the dependency it names is absent at this tree)"
    if rc != 0 or not counts["passed"]:
        return f"pytest rc {rc} with {counts['passed']} passed -- no evidence collected (an empty -k selection or a collection error)"
    return None


def _split_target(target: str) -> List[str]:
    parts = target.split()
    return parts


def run_pytest_gate(gate: Mapping[str, Any], *, tree: Path, python: str, lock: Optional[Path], runner: Optional[Callable] = None) -> Dict[str, Any]:
    missing = [d for d in gate.get("depends", []) if not (tree / d).is_file()]
    files = sorted({_split_target(t)[0] for t in gate["pytest"]})
    missing += [f for f in files if not (tree / f).is_file()]
    if missing:
        return {"ok": False, "why": f"missing dependency / test file in the tree: {missing}", "missing": missing}
    argv = [python, "-m", "pytest", "-o", "addopts=", "-p", "no:cacheprovider", "-q", "-rs"]
    for t in gate["pytest"]:
        argv += _split_target(t)
    if lock is not None and Path(lock).parent.is_dir():
        argv = ["flock", "-w", "5400", str(lock)] + argv
    env = {**os.environ, "PYTHONPATH": f"{tree}/src"}
    run = runner or (lambda a, **k: subprocess.run(a, capture_output=True, text=True, **k))
    t0 = time.time()
    r = run(argv, cwd=str(tree), env=env, timeout=7200)
    text = (r.stdout or "") + (r.stderr or "")
    counts = parse_pytest_summary(text)
    why = pytest_verdict(r.returncode, counts)
    return {"ok": why is None, "why": why, "rc": r.returncode, "counts": {k: v for k, v in counts.items() if k != "summary_line"},
            "summary_line": counts["summary_line"], "argv": argv, "elapsed_s": round(time.time() - t0, 1), "tail": text[-1500:] if why else ""}


def _lazy_volume_reader(rs: Optional[Path]):
    """The Volume API reader (`seedrep_s5_record._volume_reader`) opened on the FIRST `/vol/...` read only -- G8 needs it solely to admit a
    prior-tree / wave reference through its standup's records; a block of this tree's own references never touches the volume."""
    box: Dict[str, Any] = {}

    def read(path: str):
        if rs is None:
            return None
        if "r" not in box:
            import seedrep_s5_record as s5r  # noqa: PLC0415  (beside this module)
            box["r"] = s5r._volume_reader(Path(rs))
        return box["r"](path)
    return read


def gate8_s5_roster(*, plan: Mapping[str, Any], s5_references: Optional[Path], launch_set: Optional[Sequence[str]] = None, read_text=None,
                    rs: Optional[Path] = None, **_) -> Dict[str, Any]:
    """Gate 8 (fc10 / D139 / D145 / D146): the SAME admission and scoping the certificate applies (`lp.s5_certificate_coverage`, one implementation):
    every reference judged on its content and its basis (this tree's W + the S5 wave, or a prior tree / wave admitted through its `attempt_root`'s
    standup records against this plan's standup for the class); the gap over the LAUNCHED rows' window classes when ``launch_set`` is given
    (the whole roster when None), a class satisfied by its own complete reference or by a complete reference of a larger window at the same
    thinking mode (`dominated_by`); the coverage block recorded on the gate (`s5_coverage`, the certificate's shape) for the launch report."""
    roster = plan.get("s5_roster") or []
    if not roster:
        return {"ok": False, "why": "the plan has no S5 roster"}
    used = {(int(r["geometry"]["L"]), int(r["geometry"]["S"]), bool(r["geometry"]["thinking"])) for r in plan["standups"] + plan["rungs"]}
    have = {(int(r["L"]), int(r["S"]), bool(r["thinking"])) for r in roster}
    if used != have:
        return {"ok": False, "why": f"the S5 roster {sorted(have)} != the windows the executable rows use {sorted(used)}"}
    out: Dict[str, Any] = {"ok": True, "n_windows": len(roster), "s5_references": "pending (RS/smokes.json absent; gate 8 is re-asserted by the FINAL plan)",
                           "launch_set": (sorted({str(x) for x in launch_set}) if launch_set is not None else None)}
    ls_probs = lp.launch_set_problems(plan, out["launch_set"])
    if ls_probs:
        return {**out, "ok": False, "why": f"--launch-set: {ls_probs}"}
    if s5_references is not None and Path(s5_references).is_file():
        s5 = json.loads(Path(s5_references).read_text()).get("S5") or {}
        refs = s5.get("references") or []
        # a reference's `wave_uuid` is the smoke's EXECUTION wave (the standup that measured), named by the block's wrapper;
        # the wrapper itself must target THIS plan (the plan sha is the draft's at this gate and re-asserted by the certificate)
        block = lp.s5_block_problems(s5, plan={**plan, "launch_plan_sha": s5.get("launch_plan_sha")})
        if block:
            return {**out, "ok": False, "why": f"the S5 block is not this plan's: {block}"}
        cov = lp.s5_certificate_coverage(plan, s5, execution_wave=str(s5["execution_wave_uuid"]), launch_set=out["launch_set"],
                                         read_text=read_text if read_text is not None else _lazy_volume_reader(rs))
        out["s5_coverage"] = cov
        bad = {i: a["problems"] for i, a in cov["references"].items() if a["problems"]}
        if bad:                                                # a bare scalar / percentile / partial store is NO reference
            return {**out, "ok": False, "why": f"S5.references[] carries {len(bad)} record(s) that are not S5 references (or not admissible under D139): {bad}"}
        gap = [c for c in cov["classes"] if c.get("gap")]
        dominated = [(tuple(c["class"]), tuple(c["dominated_by"])) for c in cov["classes"] if c.get("satisfied_by") == lp.S5_COVERAGE_DOMINATED]
        prior = sorted({tuple(c["class"]) for c in cov["classes"] if c.get("basis") == lp.S5_REFERENCE_BASIS_PRIOR})
        self_measured = sorted({tuple(c["class"]) for c in cov["classes"] if c.get("satisfied_by") == lp.S5_COVERAGE_RUNG_SELF})
        out["s5_references"] = (f"{len(refs)} reference(s); {len(cov['classes'])} window class(es) {'launched' if launch_set is not None else 'planned'}; "
                                f"gap {[tuple(g['class']) for g in gap]}; dominated (D152) {dominated}; rung-self-measured (D154) {self_measured}; "
                                f"admitted under D139 (prior W / wave) {prior}; multiturn {cov['multiturn']!r}")
        if gap:
            scope = "launched window classes" if launch_set is not None else "planned standups"
            out.update(ok=False, why=(f"{scope} without an S5 reference (own, or dominated by a larger complete class -- D146): "
                                      f"{[(tuple(g['class']), g['why']) for g in gap]} -- exclude the rows that use them by name"))
    return out


def gate_ledger_vs_quota(*, plan: Mapping[str, Any], quota: Optional[int], **_) -> Dict[str, Any]:
    if quota is None:
        return {"ok": False, "why": "--quota <workspace container quota> was not given; the ledger cannot be held to anything"}
    peak = int(plan["capacity_ledger"]["unstaged_peak"]["containers"])
    out = {"ok": True, "unstaged_peak_containers": peak, "quota": int(quota), "staged_launch": peak > int(quota)}
    if peak > int(quota):
        out.update(ok=False, why=f"unstaged peak {peak} containers > quota {quota}: the launch must be STAGED (multi-turn first) -- decide it by name before fan-out")
    return out


def gate_plan_validates(*, plan: Mapping[str, Any], tree: Path, **_) -> Dict[str, Any]:
    table = None
    try:
        sys.path.insert(0, str(tree / "src"))
        from sisteer.capeval.serving import geometry as table  # type: ignore  # noqa: N813
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "why": f"the tree's geometry table is not importable: {type(e).__name__}: {e}"}
    problems = lp.validate_plan(plan, table=table) + lp.config_files_problems(plan, tree)
    return {"ok": not problems, "why": (None if not problems else f"{len(problems)} validate_plan problem(s): {problems[:5]}"), "n_problems": len(problems)}


def load_registry_plans(tree: Path):
    p = tree / REGISTRY_PLANS_REL
    if not p.is_file():
        return None
    spec = importlib.util.spec_from_file_location("seedrep_registry_plans_s0", p)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(tree / "src"))
    sys.path.insert(0, str(p.parent))
    # register BEFORE exec_module: the producer's frozen dataclasses resolve their annotations through
    # sys.modules[cls.__module__] at class-creation time (Python 3.12 dataclasses._is_type), and an unregistered
    # module makes that lookup None -> AttributeError -> the IDENTITY gate would read "failed to import" at S0
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def gate_dose_registration(*, plan: Mapping[str, Any], **_) -> Dict[str, Any]:
    """Informational, never a failure of the s1-s3 legs: which peer dose standups the plan registers, or NOT REGISTERED (pending_peer_spec)."""
    rec = plan.get("dose_standups") or {}
    led = (plan.get("capacity_ledger") or {})
    if rec.get("status") == "not_in_this_wave":
        return {"ok": True, "registered": [], "status": "NOT IN THIS WAVE (D71): Strength's dose sweep runs on cluster A; no dose standups, no ledger line; "
                                                        "the peer-roster registration path is dormant (--register-peer-dose-standups, default off)",
                "why": rec.get("why"), "peer_files_given": rec.get("peer_files_given"), "unstaged_peak": led.get("unstaged_peak"),
                "s5_probe_rows": len(plan.get("s5_probe_rows") or [])}
    if rec.get("status") == "pending_peer_spec":
        return {"ok": True, "registered": [], "status": "NOT REGISTERED (pending_peer_spec): the peer's roster + pin file were not given; the s1-s3 legs are unaffected",
                "containers_when_registered": (rec.get("pending") or {}).get("containers_when_registered"),
                "unstaged_peak_without_dose": led.get("unstaged_peak_without_dose"), "unstaged_peak": led.get("unstaged_peak")}
    return {"ok": True, "registered": list(rec.get("standups") or []), "status": rec.get("status"), "views": sorted(rec.get("views") or {}),
            "pins_source": rec.get("pins_source"), "unstaged_peak_without_dose": led.get("unstaged_peak_without_dose"), "unstaged_peak": led.get("unstaged_peak")}


def gate_identity_contents(*, plan: Mapping[str, Any], tree: Path, rs: Path, campaign: Optional[str], registry_plans=None, **_) -> Dict[str, Any]:
    try:
        RP = registry_plans or load_registry_plans(tree)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "why": f"WS4-H's {REGISTRY_PLANS_REL} failed to import against this tree: {type(e).__name__}: {e}"}
    if RP is None:
        return {"ok": False, "why": f"WS4-H's {REGISTRY_PLANS_REL} is not in the tree -- the exact identity contents cannot be checked"}
    if not campaign:
        return {"ok": False, "why": "--campaign <registry campaign id> was not given (the identities embed it)"}
    out_dir = Path(rs) / "registry" / "plans"
    if not out_dir.is_dir():
        return {"ok": False, "why": f"no identity files under {out_dir} (WS4-H's registry_plans.py --out writes them)"}
    windows = {leg: int(L) for leg, L in (plan.get("s4_decisions") or {}).get("multiturn_window", {}).items()}
    try:
        inp = RP.load_inputs(campaign=campaign, w_sha=plan["W"], policy=tree / "configs/registry/decode_policy.json", roster=tree / lp.ROSTER_REL,
                             pins=tree / lp.PINS_REL, arms=tree / lp.ARMS_REL, arditi_pins=tree / lp.ARDITI_PINS_REL, seeds_dir=tree / "configs/registry",
                             windows=windows or None)
        ps = RP.build(inp)
        problems = list(RP.check_plan_set(ps, out_dir))
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "why": f"registry_plans refused: {type(e).__name__}: {e}"}
    missing, unloadable = [], []
    n_children = 0
    for r in plan["standups"] + plan["rungs"]:
        for k in r["children"]:
            n_children += 1
            p = out_dir / k["registry_identity_rel"][len(lp.PLANS_REL) + 1:]
            if not p.is_file():
                missing.append(str(p))
                continue
            try:
                RP.load_plan_file(p)
            except Exception as e:  # noqa: BLE001
                unloadable.append(f"{p}: {type(e).__name__}: {e}")
    why = None
    if problems or missing or unloadable:
        why = f"{len(problems)} content mismatch(es) vs the producer {problems[:3]}; {len(missing)} child identity file(s) missing {missing[:3]}; {len(unloadable)} unloadable {unloadable[:2]}"
    return {"ok": why is None, "why": why, "n_children": n_children, "n_problems": len(problems), "n_missing": len(missing), "n_unloadable": len(unloadable)}


PYTHON_GATES: Dict[str, Callable[..., Dict[str, Any]]] = {"gate8_s5_roster": gate8_s5_roster, "gate_ledger_vs_quota": gate_ledger_vs_quota, "gate_dose_registration": gate_dose_registration,
                                                          "gate_plan_validates": gate_plan_validates, "gate_identity_contents": gate_identity_contents}


def run_gates(*, tree: Path, plan: Mapping[str, Any], rs: Path, python: str, quota: Optional[int], campaign: Optional[str],
              s5_references: Optional[Path] = None, lock: Optional[Path] = None, runner: Optional[Callable] = None, registry_plans=None,
              only: Optional[Sequence[str]] = None, launch_set: Optional[Sequence[str]] = None, read_text=None) -> Dict[str, Any]:
    report: Dict[str, Any] = {"schema": S0_SCHEMA, "W": plan["W"], "wave_uuid": plan["wave_uuid"], "launch_plan_sha": plan.get("launch_plan_sha"),
                              "tree": str(tree), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gates": {}, "ok": True,
                              # the rows G8's S5 gap was scoped to (None = the whole roster); the certificate records the same set
                              "launch_set": (sorted({str(x) for x in launch_set}) if launch_set is not None else None)}
    for g in GATES:
        if only and g["id"] not in only:
            continue
        if "pytest" in g:
            res = run_pytest_gate(g, tree=tree, python=python, lock=lock, runner=runner)
        else:
            missing = [d for d in g.get("depends", []) if not (tree / d).is_file()]
            if missing:
                res = {"ok": False, "why": f"missing dependency in the tree: {missing}"}
            else:
                res = PYTHON_GATES[g["python"]](plan=plan, tree=tree, rs=rs, quota=quota, campaign=campaign, s5_references=s5_references, registry_plans=registry_plans,
                                                launch_set=launch_set, read_text=read_text)
        report["gates"][g["id"]] = {"plan": g["plan"], "owner": g["owner"], **res}
        report["ok"] = report["ok"] and bool(res.get("ok"))
    report["failed"] = sorted(k for k, v in report["gates"].items() if not v.get("ok"))
    return report


def selftest() -> int:
    ids = [g["id"] for g in GATES]
    assert ids[:10] == [f"G{i}" for i in range(1, 11)], ids
    assert len(set(ids)) == len(ids)
    for g in GATES:
        assert ("pytest" in g) != ("python" in g), g["id"]
        if "python" in g:
            assert g["python"] in PYTHON_GATES, g["id"]
    assert parse_pytest_summary("....\n4 passed, 1 skipped in 1.2s\n") == {"passed": 4, "failed": 0, "skipped": 1, "errors": 0, "summary_line": "4 passed, 1 skipped in 1.2s"}
    assert pytest_verdict(0, {"passed": 4, "failed": 0, "skipped": 1, "errors": 0}) and pytest_verdict(0, {"passed": 4, "failed": 0, "skipped": 0, "errors": 0}) is None
    assert pytest_verdict(5, {"passed": 0, "failed": 0, "skipped": 0, "errors": 0})
    print(f"seedrep_s0_gates selftest OK: {len(GATES)} gates ({', '.join(ids)})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tree", type=Path, default=None, help="the W checkout")
    ap.add_argument("--plan", type=Path, default=None, help="RS/launch_plan_draft.json (S0) or RS/launch_plan.json (the re-run before fan-out)")
    ap.add_argument("--rs", type=Path, default=None)
    ap.add_argument("--quota", type=int, default=None, help="the Modal workspace's container quota (the ledger is held to it)")
    ap.add_argument("--campaign", default=None, help="the registry campaign id the identity files embed (WS4-H)")
    ap.add_argument("--s5-references", type=Path, default=None, help="RS/smokes.json once S5 ran (gate 8's re-assertion)")
    ap.add_argument("--launch-set", default=None,
                    help="fc10 / D146: the standup ids the launch will name (comma-separated, or @<file>: one per line / a JSON list; the certify spelling) -- gate 8's "
                         "S5 gap is scoped to THEIR window classes; omitted = the whole roster (the pre-fc10 behaviour)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--lock", type=Path, default=None, help="the grading host pytest serialisation lock (used when its directory exists; "
                    "default: the site's SISTEER_PYTEST_LOCK, else $STATE/sisteer-pytest/lock)")
    ap.add_argument("--out", type=Path, default=None, help="the report (default <rs>/s0_gates.json)")
    ap.add_argument("--only", default=None, help="comma-separated gate ids (debugging; the S0 verdict needs every gate)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.lock is None:
        a.lock = default_lock()
    if a.selftest:
        return selftest()
    if a.list:
        print(json.dumps(list_gates(), indent=1))
        return 0
    if not (a.tree and a.plan and a.rs):
        raise SystemExit("--tree, --plan and --rs are required to run the gates (--list / --selftest need none)")
    plan = json.loads(a.plan.read_text())
    import seedrep_s5_record as s5r  # noqa: PLC0415  (beside this module: the ONE spelling of --launch-set, certify's)
    rep = run_gates(tree=a.tree, plan=plan, rs=a.rs, python=a.python, quota=a.quota, campaign=a.campaign, s5_references=a.s5_references,
                    lock=a.lock, only=(a.only.split(",") if a.only else None), launch_set=s5r.parse_launch_set(a.launch_set))
    out = a.out or (a.rs / "s0_gates.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    for gid, g in rep["gates"].items():
        print(f"{'PASS' if g.get('ok') else 'FAIL'} {gid:9s} {g.get('why') or g.get('summary_line') or ''}")
    print(f"[s0] {'PASS' if rep['ok'] else 'FAIL'} -- {len(rep['gates']) - len(rep['failed'])}/{len(rep['gates'])} gates passed; report {out}")
    if a.only:
        print("[s0] --only given: this is NOT an S0 verdict (every gate is required)")
        return 0 if rep["ok"] else 1
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
