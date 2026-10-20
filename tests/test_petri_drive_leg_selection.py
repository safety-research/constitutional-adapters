"""petri_leg_select.py + petri_drive_full22.sh's leg selection BY CELL (qwen-base-seed-replicates).

A synthetic frozen enumeration (`none` + six object cells + a prompted cell as `canonical`, the derived `none__lora0` as
`controls`): selecting the six by cell yields exactly their indices and never the `none` / prompted / controls index;
naming `none`, a prompted leg, the controls leg, an unknown id or a duplicate refuses; an explicit index list is held to
the same rule under --objects-only; a record without `cell_class` classifies through the harness grammar. The driver's
wiring (F22_PETRI_CELLS -> the helper -> LEGS; F22_PETRI_OBJECTS_ONLY; F22_PETRI_RELEASE_SCOPE=selected counting the
SELECTED legs) is pinned as text, and the default (unset, legs=ALL) stays today's path.
"""
from __future__ import annotations

import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
F22 = REPO / "scripts" / "sdfbase" / "full22"
for p in (REPO / "src",):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load():
    spec = importlib.util.spec_from_file_location("_petri_leg_select", F22 / "petri_leg_select.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SIX = ["si_constv2_mix_full__psr_ce_all", "si_constv2_mix_full__lora_r4_ce", "si_constv2ctl2_full__psr_ce_all",
       "si_constv2ctl2_full__lora_r4_ce", "si_constv2_mix_full__lora_r4_ce_minus_ctl",
       "si_constv2_mix_full__psr_ce_all_minus_ctl_distilled"]


def _leg(cell_id, cell_class, kind="lora", i=0):
    return {"cell_id": cell_id, "cell_class": cell_class, "kind": kind, "virtual_model_name": f"petri__{cell_id}",
            "serving_identity": f"id-{i}", "intervention_hash": f"h{i:02d}", "petri_run_key": f"rk-{i:02d}", "si_id": None}


def _record(tmp_path, *, drop_class_on=None):
    canonical = [_leg("none", "base", "none", 0)] + [_leg(c, "class_a", "psr" if "psr" in c else "lora", i + 1) for i, c in enumerate(SIX)] \
        + [_leg("prompted__constitution", "prompted", "prompted", 7)]
    controls = [_leg("none__lora0", "control", "lora", 8)]
    if drop_class_on is not None:
        for e in canonical + controls:
            if e["cell_id"] == drop_class_on:
                del e["cell_class"]
    p = tmp_path / "cells.json"
    p.write_text(json.dumps({"canonical": canonical, "controls": controls, "serving_path": "lora_native"}))
    return p


def test_selecting_the_six_by_cell_yields_their_indices_and_nothing_unsteered(tmp_path):
    m = _load()
    legs = m.load_legs(_record(tmp_path))
    idx = m.select_by_cells(legs, SIX)
    assert idx == [1, 2, 3, 4, 5, 6]
    assert 0 not in idx and 7 not in idx and 8 not in idx   # none, prompted, controls
    # order comes from the record, not the argument
    assert m.select_by_cells(legs, list(reversed(SIX))) == idx


@pytest.mark.parametrize("bad, needle", [
    (["none"], "'base' leg"),
    (["prompted__constitution"], "'prompted' leg"),
    (["none__lora0"], "CONTROLS leg"),
    (["si_nowhere__lora_r4"], "not a canonical leg"),
    ([SIX[0], SIX[0]], "twice"),
    ([" "], "names no cell"),
])
def test_unsteered_unknown_and_duplicate_selections_refuse_by_name(tmp_path, bad, needle):
    m = _load()
    legs = m.load_legs(_record(tmp_path))
    with pytest.raises(m.LegSelectError, match=needle):
        m.select_by_cells(legs, bad)


def test_an_explicit_index_list_is_held_to_the_objects_only_rule(tmp_path):
    m = _load()
    legs = m.load_legs(_record(tmp_path))
    assert m.check_indices(legs, [3, 1], objects_only=True) == [1, 3]
    for i, needle in ((0, "'base' leg"), (7, "'prompted' leg"), (8, "CONTROLS leg")):
        with pytest.raises(m.LegSelectError, match=needle):
            m.check_indices(legs, [1, i], objects_only=True)
    with pytest.raises(m.LegSelectError, match="outside"):
        m.check_indices(legs, [9], objects_only=True)
    with pytest.raises(m.LegSelectError, match="twice"):
        m.check_indices(legs, [1, 1], objects_only=True)
    # without --objects-only the list is only range/uniqueness checked (today's explicit-legs behaviour)
    assert m.check_indices(legs, [0, 8], objects_only=False) == [0, 8]


def test_a_record_without_cell_class_classifies_through_the_harness_grammar(tmp_path):
    m = _load()
    legs = m.load_legs(_record(tmp_path, drop_class_on="none"))
    assert legs[0]["_class"] == "base"
    with pytest.raises(m.LegSelectError, match="'base' leg"):
        m.select_by_cells(legs, ["none"])
    legs2 = m.load_legs(_record(tmp_path, drop_class_on=SIX[0]))
    assert legs2[1]["_class"] == "class_a" and m.select_by_cells(legs2, [SIX[0]]) == [1]


def test_the_cli_prints_indices_and_exits_2_on_a_refusal(tmp_path):
    rec = _record(tmp_path)
    r = subprocess.run([sys.executable, str(F22 / "petri_leg_select.py"), str(rec), "--cells", ",".join(SIX)],
                       capture_output=True, text=True, env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "1 2 3 4 5 6"
    r = subprocess.run([sys.executable, str(F22 / "petri_leg_select.py"), str(rec), "--cells", "none"],
                       capture_output=True, text=True, env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 2 and "REFUSED" in r.stderr and "'base' leg" in r.stderr
    r = subprocess.run([sys.executable, str(F22 / "petri_leg_select.py"), str(rec), "--indices", "0 1", "--objects-only"],
                       capture_output=True, text=True, env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 2 and "'base' leg" in r.stderr


def test_the_driver_wires_the_selection_and_the_selected_leg_release():
    drv = (F22 / "petri_drive_full22.sh").read_text()
    assert subprocess.run(["bash", "-n", str(F22 / "petri_drive_full22.sh")]).returncode == 0
    # the knobs, each defaulting to today's driver
    assert 'PETRI_CELLS="${F22_PETRI_CELLS:-}"' in drv and 'OBJECTS_ONLY="${F22_PETRI_OBJECTS_ONLY:-}"' in drv
    assert 'RELEASE_SCOPE="${F22_PETRI_RELEASE_SCOPE:-all}"' in drv and 'RELEASE_SCOPE="${F22_PETRI_RELEASE_SCOPE:-selected}"' in drv
    # BY CELL: the helper resolves the indices from the FROZEN record, after the freeze and before ALL expands
    sel = 'scripts/sdfbase/full22/petri_leg_select.py "$OUT/cells.json" --cells "$PETRI_CELLS"'
    assert sel in drv
    assert drv.index('log "enumeration frozen:') < drv.index(sel) < drv.index('[ "$LEGS" = "ALL" ] && LEGS=$(seq')
    # an explicit list under objects-only is validated by the same helper; ALL refuses under it
    assert '--indices "$LEGS" --objects-only' in drv
    assert 'F22_PETRI_OBJECTS_ONLY=1 with legs=ALL' in drv
    # the release loop counts the SELECTED legs under scope=selected, every leg otherwise
    assert 'if [ "$RELEASE_SCOPE" = selected ]; then SCOPE_LEGS="$SELECTED_LEGS"' in drv
    assert 'for idx in $SCOPE_LEGS; do' in drv
    assert 'SELECTED_LEGS="$LEGS"' in drv
    assert 'ALL $NSCOPE SELECTED LEGS ENDED' in drv and 'ALL $NLEGS LEGS ENDED' in drv
    # exclusive with an explicit legs argument
    assert 'F22_PETRI_CELLS and an explicit legs argument are exclusive' in drv


# ------------------------------------------------------------------------------------- the LEDGERED selector (petri-completion, 2026-10-13)
DEMO_LEGS = [("none", "base", "none"), ("prompted__safety", "prompted", "prompted"), ("prompted__constitution", "prompted", "prompted"), ("arditi_allpos", "class_b", "vector_all"),
             ("si_constv2ctl2_full__lora_r4_ce", "class_a", "lora_all"), ("si_constv2_mix_full__lora_r4_ce", "class_a", "lora_all"),
             ("si_constv2_mix_full__lora_r4_ce_minus_ctl", "class_a", "lora_subtraction"), ("si_constv2ctl2_full__psr_ce_all", "class_a", "psr"),
             ("si_constv2_mix_full__psr_ce_all", "class_a", "psr"), ("si_constv2_mix_full__psr_ce_all_minus_ctl", "class_a", "psr_paired_write")]


def _demo_record(tmp_path):
    """The demo's frozen enumeration shape: ten canonical legs + the native controls twin (control_of none), synthetic hashes."""
    canonical = [{**_leg(c, cc, k, i), "control_of": None} for i, (c, cc, k) in enumerate(DEMO_LEGS)]
    controls = [{**_leg("none__lora0", "control", "none", 10), "control_of": "none"}]
    p = tmp_path / "cells.json"
    p.write_text(json.dumps({"canonical": canonical, "controls": controls, "serving_path": "lora_native"}))
    return p


def _identity(tmp_path, objects):
    """The drive's identity file: one registered run per (object, eval) for `objects` — a control leg is never among them (close-leg's rule)."""
    p = tmp_path / "identity.json"
    p.write_text(json.dumps({"job_id": "11111", "runs": [{"run_id": f"r-{o}-{e}", "route": {"object": o, "eval_id": e}, "plan": {"eval_id": e}} for o in objects for e in ("petri_pooled", "petri_b1")]}))
    return p


def _drive_block(start: str, end: str) -> str:
    """The drive's text from `start` to the first `end` AFTER it (the locks-held refusal has a `RESULT=` and a result copy of its own)."""
    drive = (F22 / "petri_drive_full22.sh").read_text()
    i = drive.index(start)
    return drive[i:drive.index(end, i)]


def _bash(script: str, env: dict, cwd=None):
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", **env}, cwd=cwd)


def test_the_ledgered_selector_needs_the_overflow_pass_and_implies_the_selected_release_scope():
    """The selection block as the daemon's deny-list test extracts it: `ledgered` without F22_OVERFLOW_RERUN is FATAL; with it RELEASE_SCOPE is
    selected; with F22_PETRI_CELLS beside it, FATAL (the ledgers select the legs); the default path is byte-identical."""
    block = _drive_block('PETRI_CELLS="${F22_PETRI_CELLS:-}"', "# THE TARGET DECODE CONTRACT")
    tail = '\necho "PETRI_CELLS=[$PETRI_CELLS] RELEASE_SCOPE=$RELEASE_SCOPE LEDGERED=[$LEDGERED]"\n'
    r = _bash("set -uo pipefail\nLEGS=ledgered\n" + block + tail, {})
    assert r.returncode == 2 and "legs=ledgered is the overflow pass's selector: it needs F22_OVERFLOW_RERUN=1" in r.stdout + r.stderr
    r = _bash("set -uo pipefail\nLEGS=ledgered\nRERUN=1\n" + block + tail, {})
    assert r.returncode == 0 and r.stdout.strip().endswith("PETRI_CELLS=[] RELEASE_SCOPE=selected LEDGERED=[1]"), r.stdout + r.stderr
    r = _bash("set -uo pipefail\nLEGS=ledgered\nRERUN=1\n" + block + tail, {"F22_PETRI_CELLS": "si_x"})
    assert r.returncode == 2 and "exclusive with F22_PETRI_CELLS / F22_PETRI_OBJECTS_ONLY" in r.stdout + r.stderr
    r = _bash("set -uo pipefail\nLEGS=ALL\n" + block + tail, {})
    assert r.returncode == 0 and r.stdout.strip().endswith("PETRI_CELLS=[] RELEASE_SCOPE=all LEDGERED=[]")


def _fake_wt(tmp_path, legs_doc, handoff_rc=0):
    """A worktree stand-in: `scripts/petri_run.py` answers legs-status with `legs_doc`; `scripts/sdfbase/full22/petri_registry_handoff.py` records its
    argv and writes the summary verify would."""
    wt = tmp_path / "wt"; (wt / "scripts/sdfbase/full22").mkdir(parents=True)
    (wt / "legs.json").write_text(json.dumps(legs_doc))
    (wt / "scripts/petri_run.py").write_text(f"import sys\nprint(open({str(wt / 'legs.json')!r}).read())\n")
    (wt / "scripts/sdfbase/full22/petri_registry_handoff.py").write_text(
        "import json, sys\nopen(sys.argv[0] + '.argv', 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "s = sys.argv[sys.argv.index('--summary') + 1] if '--summary' in sys.argv else None\n"
        "s and open(s, 'w').write(json.dumps({'n_planned': 7, 'n_filed': 7}))\n" f"sys.exit({handoff_rc})\n")
    return wt


def _legs_doc(ledgered=(2, 4, 10), done=(), refused=()):
    return {"legs": [{"idx": i, "done": i in done, "refused": i in refused, "overflow_unresolved": 3 if i in ledgered else 0} for i in range(11)]}


def test_ledgered_selects_the_legs_from_legs_status_and_an_empty_set_is_fatal(tmp_path):
    """The post-freeze selection: `petri_run.py legs-status` over the out root -> the legs with an unresolved overflow entry that are neither done
    nor refused, in record order; none -> FATAL."""
    block = _drive_block('if [ -n "$LEDGERED" ]; then\n  # the LEDGERED legs', 'if [ -n "$PETRI_CELLS" ]; then\n  # BY CELL')
    rec = _demo_record(tmp_path)
    pre = f'set -uo pipefail\nlog(){{ echo "$*"; }}\nLEDGERED=1\nPY={sys.executable}\nPETRI_RUN_OPTS=()\nOUT={rec.parent}\nLOG={tmp_path}/log\nLEGS=ALL\n'
    wt = _fake_wt(tmp_path / "a", _legs_doc())
    r = _bash(pre + block + '\necho "LEGS=[$LEGS]"\n', {}, cwd=wt)
    assert r.returncode == 0 and r.stdout.strip().endswith("LEGS=[2 4 10]") and "legs selected as LEDGERED" in r.stdout, r.stdout + r.stderr
    wt = _fake_wt(tmp_path / "b", _legs_doc(ledgered=(2, 4, 10), done=(2,), refused=(4,)))         # a done leg and a refused leg drop out
    r = _bash(pre + block + '\necho "LEGS=[$LEGS]"\n', {}, cwd=wt)
    assert r.returncode == 0 and r.stdout.strip().endswith("LEGS=[10]")
    wt = _fake_wt(tmp_path / "c", _legs_doc(ledgered=()))
    r = _bash(pre + block + '\necho "LEGS=[$LEGS]"\n', {}, cwd=wt)
    assert r.returncode == 1 and "no leg carries an unresolved context-overflow entry" in r.stdout + r.stderr


def test_the_verify_objects_under_ledgered_are_the_selected_legs_registered_objects_and_a_control_only_pass_skips_verify(tmp_path):
    """The drive's side: under `ledgered` the `--objects` are the selected legs' cell ids INTERSECTED with the identity's registered objects
    (the control twin drops out); an empty intersection does not call verify (it would refuse n_planned == 0): verify_scope control_only, no summary;
    the non-ledgered `selected` path's --objects are byte-identical to today's (every selected leg's cell id)."""
    block = _drive_block('VERIFY_SCOPE="$RELEASE_SCOPE"\n', "# THE VERDICT = THE EXIT CODE")
    rec = _demo_record(tmp_path); (tmp_path / "registry").mkdir()      # the drive makes $STATE/registry at the hand-off import; verify writes its summary there
    ident = _identity(tmp_path, [c for c, _cc, _k in DEMO_LEGS])
    base = (f'set -uo pipefail\nlog(){{ echo "$*"; }}\nPY={sys.executable}\nOUT={rec.parent}\nLOG={tmp_path}/log\nSTATE={tmp_path}\nDRIVE_ID=d1\nREG_IDENTITY={ident}\n'
            'NFAIL=0\nNINC=0\nRELEASE_SCOPE=selected\nSKIPPED=""\nHANDOFF_RC=0; VERIFY_RC=""; VERIFY_SUMMARY=""; RUNS_PLANNED=0; RUNS_FILED=0\n')
    tail = '\necho "VERIFY_SCOPE=$VERIFY_SCOPE VERIFY_RC=[$VERIFY_RC] SUMMARY=[$VERIFY_SUMMARY] PLANNED=$RUNS_PLANNED"\n'
    def argv_of(wt):
        p = wt / "scripts/sdfbase/full22/petri_registry_handoff.py.argv"
        return json.loads(p.read_text().splitlines()[-1]) if p.exists() else None
    # legs 2 4 10 under ledgered: the control leg (none__lora0) is not a registered object and drops out
    wt = _fake_wt(tmp_path / "a", {})
    r = _bash(base + 'LEDGERED=1\nSELECTED_LEGS="2 4 10"\nWT=' + str(wt) + "\n" + block + tail, {}, cwd=wt)
    assert r.returncode == 0, r.stdout + r.stderr
    argv = argv_of(wt); assert argv[argv.index("--objects") + 1] == "prompted__constitution,si_constv2ctl2_full__lora_r4_ce" and "verify" in argv
    assert r.stdout.strip().endswith("VERIFY_SCOPE=selected VERIFY_RC=[0] SUMMARY=[" + f"{tmp_path}/registry/verify.d1.json" + "] PLANNED=7")
    # leg 10 alone: a control-only pass — verify is NOT called, the scope says so, no summary
    wt = _fake_wt(tmp_path / "b", {})
    r = _bash(base + 'LEDGERED=1\nSELECTED_LEGS="10"\nWT=' + str(wt) + "\n" + block + tail, {}, cwd=wt)
    assert r.returncode == 0 and argv_of(wt) is None and "every selected leg is a control leg" in r.stdout, r.stdout + r.stderr
    assert r.stdout.strip().endswith("VERIFY_SCOPE=control_only VERIFY_RC=[] SUMMARY=[] PLANNED=0")
    # the non-ledgered selected path: today's list, byte for byte (the control leg's cell id included)
    wt = _fake_wt(tmp_path / "c", {})
    r = _bash(base + 'LEDGERED=""\nSELECTED_LEGS="2 4 10"\nWT=' + str(wt) + "\n" + block + tail, {}, cwd=wt)
    argv = argv_of(wt); assert argv[argv.index("--objects") + 1] == "prompted__constitution,si_constv2ctl2_full__lora_r4_ce,none__lora0"
    assert r.stdout.strip().endswith("VERIFY_SCOPE=selected VERIFY_RC=[0] SUMMARY=[" + f"{tmp_path}/registry/verify.d1.json" + "] PLANNED=7")
    # the release scope `all`: no --objects at all, as today
    wt = _fake_wt(tmp_path / "d", {})
    r = _bash(base.replace("RELEASE_SCOPE=selected", "RELEASE_SCOPE=all") + 'LEDGERED=""\nSELECTED_LEGS="2 4 10"\nWT=' + str(wt) + "\n" + block + tail, {}, cwd=wt)
    assert "--objects" not in argv_of(wt) and r.stdout.strip().endswith("PLANNED=7")
    # the drive refuses an unserved ledgered leg by name, never a skip
    drive = (F22 / "petri_drive_full22.sh").read_text()
    assert 'FATAL: ledgered leg $idx ($vm) is NOT served by this proxy' in drive and drive.index("FATAL: ledgered leg") < drive.index("is NOT served by this proxy (served_models: $SERVED_MODELS) — skipped")


def test_the_result_document_carries_paths_the_verify_scope_and_every_legs_detail(tmp_path):
    """The record 06 --pass overflow and the daemon read: `paths.out_root`, `handoff.verify_scope`, and `legs.detail` — every leg of the frozen
    record with its identity, whether the identity file registers a run for it, this drive's per-leg verdict, and the ledger's unresolved /
    context-overflow counts through petri_run's own readers (the real module, imported from the REAL scripts dir)."""
    # the FINAL result block (the locks-held refusal writes an earlier `RESULT=` of its own): from its comment to the daemon's copy
    block = _drive_block("# the machine-readable outcome, beside the drive's own log", '[ -z "${PETRI_DRIVE_RESULT_FILE:-}" ]')
    rec = _demo_record(tmp_path); out = rec.parent
    ident = _identity(tmp_path, [c for c, _cc, _k in DEMO_LEGS])
    # the overflow prefix the converter writes, read off the real runner module's constant through petri_run's readers in the heredoc
    sys.path.insert(0, str(REPO / "src")); from sisteer.capeval.tiers import petri_runner as pr  # noqa: E402
    for i in (2, 10):
        d = out / "cells" / f"rk-{i:02d}"; d.mkdir(parents=True)
        (d / "petri_incomplete.json").write_text(json.dumps({"version": 1, "cell_id": "x", "petri_run_key": f"rk-{i:02d}", "written_at": 0.0,
                                                             "unresolved": [{"behavior": "b", "scenario_id": "s1", "epoch": 1, "dimension": "b", "reason": pr.overflow_reason("maximum context length"), "retries": 0},
                                                                            {"behavior": "b", "scenario_id": "s2", "epoch": 1, "dimension": "b", "reason": "sample error", "retries": 0}]}))
    env = {"DRIVE_ID": "d1", "KEY": "k", "VIEW": "v", "CLUSTER": "cluster_a", "TAG": "k_v", "WAVE_SHA": "0" * 40, "DRIVE_RC": "3", "VERDICT": "overflow_pending", "NSCOPE": "11", "NCOMP": "8",
           "NINC": "3", "NFAIL": "0", "NSKIP": "0", "RELEASE_SCOPE": "all", "HANDOFF_MODE": "on", "REG_IDENTITY": str(ident), "HANDOFF_RC": "0", "VERIFY_RC": "", "VERIFY_SUMMARY": "",
           "RUNS_PLANNED": "0", "RUNS_FILED": "0", "LOG": str(tmp_path / "log"), "VERIFY_SCOPE": "all", "OUT": str(out), "STATE": str(tmp_path), "REMOTE_OUT": "/remote/serving", "WT": str(REPO),
           "SCOPE_LEGS": " ".join(str(i) for i in range(11)), "STATUS_LINES": "".join(f"{i} {'incomplete_overflow' if i in (2, 4, 10) else 'complete'}\n" for i in range(11)), "PY": sys.executable,
           "PYTHONPATH": str(REPO / "src")}
    r = _bash("set -uo pipefail\n" + "\n".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + "\n" + block, env)   # shlex: STATUS_LINES keeps its real newlines ($'\n' in the drive)
    assert r.returncode == 0, r.stdout + r.stderr
    doc = json.loads((tmp_path / "drive_result.json").read_text())
    assert doc["paths"] == {"out_root": str(out), "cells_json": str(out / "cells.json"), "state": str(tmp_path), "identity": str(ident), "remote_out": "/remote/serving"}
    assert doc["handoff"]["verify_scope"] == "all" and doc["verdict"] == "overflow_pending" and doc["legs"]["incomplete_overflow"] == 3
    detail = doc["legs"]["detail"]; assert len(detail) == 11
    by = {d["idx"]: d for d in detail}
    assert by[2] == {"idx": 2, "cell_id": "prompted__constitution", "control_of": None, "kind": "prompted", "petri_run_key": "rk-02", "registered": True, "in_scope": True,
                     "status": "incomplete_overflow", "unresolved": 2, "overflow_unresolved": 1}
    assert by[10] == {"idx": 10, "cell_id": "none__lora0", "control_of": "none", "kind": "none", "petri_run_key": "rk-10", "registered": False, "in_scope": True,
                      "status": "incomplete_overflow", "unresolved": 2, "overflow_unresolved": 1}
    assert by[0]["status"] == "complete" and by[0]["registered"] is True and by[0]["unresolved"] == 0 and by[0]["overflow_unresolved"] == 0 and "ledger_error" not in by[0]
    # a selected-scope pass: the legs outside the scope read out_of_scope
    env2 = {**env, "RELEASE_SCOPE": "selected", "SCOPE_LEGS": "2 4 10", "NSCOPE": "3", "NCOMP": "0", "STATUS_LINES": "2 incomplete_overflow\n4 incomplete_overflow\n10 incomplete_overflow\n", "VERIFY_SCOPE": "selected"}
    r = _bash("set -uo pipefail\n" + "\n".join(f"{k}={shlex.quote(v)}" for k, v in env2.items()) + "\n" + block, env2)
    assert r.returncode == 0, r.stdout + r.stderr
    by = {d["idx"]: d for d in json.loads((tmp_path / "drive_result.json").read_text())["legs"]["detail"]}
    assert by[0]["status"] == "out_of_scope" and by[0]["in_scope"] is False and by[4]["status"] == "incomplete_overflow" and by[4]["in_scope"] is True
    # the drive's text: the verdict line names the scope; the counts stay the verdict
    drive = (F22 / "petri_drive_full22.sh").read_text()
    assert "verify_scope=$VERIFY_SCOPE" in drive and '"detail": detail' in drive and 'leg_status_note "$idx" incomplete_overflow' in drive
