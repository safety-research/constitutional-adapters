"""Each main judging driver's ``--worklist`` flag (default ``from-disk``, today's behaviour) and its
registry branch on a SYNTHETIC pending run — the worklist is read from ``v_pending_judgment`` (matched to the
driver's own store layout by ``store_uri``), every run is claimed before grading, a stub grade completes the
CLAIMED judgment row and the run reads ``judged``; a second session refuses on the claim; a pending run whose
store the driver cannot locate refuses naming it. The from-disk branch prints exactly one bypass line and its
outputs are byte-identical to today's (the sampled static grade's dry run: default == explicit ``from-disk`` ==
the pre-change driver vendored as a fixture, modulo timestamps and the re-worded op / env ``org_switch`` sentence).

Drivers: ``scripts/served_capability_judge.py`` (over-refusal; the recording hook completes the claims),
``scripts/sdfbase/full22/sampled/grade_static_sampled.py``, ``scripts/grokmt_grade.py`` (the multi-turn chain's
grade), ``scripts/grade_bon_live.py``, ``scripts/sdfbase/full22/misalign/misalign_judge.py``,
``scripts/petri_run.py analyze`` (the Petri aggregation), ``scripts/alphasteer_f22/grade_alphasteer_controls.py``
(forwards the flags to the over-refusal driver).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

from sisteer.registry import db as DB
from sisteer.registry import emit
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import pending as P
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry import worklist as W

REPO = Path(__file__).resolve().parents[1]
TESTS = REPO / "tests"
for _p in (REPO / "scripts", REPO / "src", TESTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PRE_CHANGE_SHA = "e83725d918dd83997c1a29668cfe8daa3794e50a"   # e83725d9: the registry-grading tip before PR-5a's driver edits (the merge of registry-core)
PRE_CHANGE_DRIVER_REL = "scripts/sdfbase/full22/sampled/grade_static_sampled.py"
# The pre-change driver is VENDORED (tests/fixtures/registry/README.md), not read with ``git show``: the Modal CPU gate
# stages a shallow exact-sha clone with no history, where ``git show <sha>:<path>`` exits 128. The copy runs the commit's
# code; its comments, its docstrings and the prose values named in PRE_CHANGE_DRIVER_PROSE_KEYS are reworded
# for publication. Two-level pin: the sha256 below keeps the copy from drifting anywhere;
# ``test_pre_change_driver_fixture_matches_the_git_object`` re-derives its code from git wherever the commit is reachable.
PRE_CHANGE_DRIVER_FIXTURE = TESTS / "fixtures" / "registry" / "grade_static_sampled_pre_change_e83725d9.py"
PRE_CHANGE_DRIVER_SHA256 = "381cac33d1660226a99555628c0f21390318ffbab4a2cfa08a3e0789f441f149"
#: dict keys whose string values in the copy are reworded prose (they name where an analysis runs); the code-shape
#: comparison with the git object masks them, and ``_normalise`` drops them from the dry-run outputs
PRE_CHANGE_DRIVER_PROSE_KEYS = ("greedy_stores",)


def _pre_change_driver_bytes() -> bytes:
    data = PRE_CHANGE_DRIVER_FIXTURE.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    assert digest == PRE_CHANGE_DRIVER_SHA256, (
        f"{PRE_CHANGE_DRIVER_FIXTURE.name} drifted from its pin ({PRE_CHANGE_DRIVER_REL}@{PRE_CHANGE_SHA[:8]}): sha256 {digest}")
    return data


def _code_shape(src: bytes) -> str:
    """The module's syntax tree with every module, class and function docstring blanked and the values of
    PRE_CHANGE_DRIVER_PROSE_KEYS masked. Comments never reach the tree, so two sources with the same shape run the same
    code."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                first.value.value = ""
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value in PRE_CHANGE_DRIVER_PROSE_KEYS and isinstance(value, ast.Constant):
                    value.value = ""
    return ast.dump(tree)


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------------------------
# a synthetic registry with evals of every category the drivers filter on
# --------------------------------------------------------------------------------------------

EVALS = {  # eval_id -> (eval_category, decode_source)
    "xstest": ("over_refusal", "greedy"), "or_bench_hard": ("over_refusal", "greedy"),
    "pasttense": ("static_attack", "generation_config"), "crescendo": ("multi_turn", "greedy"),
    "strongreject_bon": ("best_of_n", "generation_config"), "agentic_leaking": ("agentic", "greedy"),
    "em_risky_financial": ("misalignment", "generation_config"), "petri_pooled": ("petri", "generation_config"),
}


def _instrument(eval_id: str) -> dict:
    cat, dec = EVALS[eval_id]
    return synth.instrument_fields(eval_id=eval_id, item_set_id=f"{eval_id}_set", decode_source=dec)


def _dims() -> list:
    dims = synth.seed_dimensions()
    i = 50
    for eval_id, (cat, dec) in EVALS.items():
        dims.append(synth.rec("eval", {"eval_id": eval_id, "eval_category": cat, "headline_metric_id": "asr",
                                       "metric_direction": "lower_is_better", "attacker_required": 0, "paper_status": "in_paper"}, synth.t(i)))
        dims.append(synth.rec("item_set", {"item_set_id": f"{eval_id}_set", "eval_id": eval_id, "dataset_version": f"synthetic/{eval_id}@v1",
                                           "n_items": 200, "items_sha256": synth.SHA256_B, "source_uri": f"synthetic://items/{eval_id}.jsonl",
                                           "item_id_scheme": "q<index>", "subsample_rule": "full"}, synth.t(i + 1)))
        dims.append(synth.rec("instrument", _instrument(eval_id), synth.t(i + 2)))
        i += 3
    return dims


def _run_recs(key: str, i0: int, eval_id: str, store: str, n_items: int = 200, object_key: str = "none") -> tuple:
    """A generated run of ``eval_id`` whose store is ``store`` (the same five records ``synth.run_records`` writes)."""
    inst = _instrument(eval_id)
    p = synth.run_payload(key, object_key, "none", i0, eval_id=eval_id, item_set_id=f"{eval_id}_set", instrument_id=inst["instrument_id"],
                          decode_source=EVALS[eval_id][1], store_uri=store)
    # the registry-grain identity over THIS eval and instrument (run_payload mints its default id before the overrides)
    p["run_id"] = I.run_id(synth.CAMPAIGN, key, object_key, eval_id, str(inst["instrument_id"]))
    rid = str(p["run_id"])
    t = synth.t
    recs = [synth.rec("run", p, t(i0)),
            synth.rec("run_state", {"run_id": rid, "state": "queued", "event_utc": t(i0), "evidence_kind": "submit_wrapper"}, t(i0)),
            synth.rec("run_state", {"run_id": rid, "state": "running", "event_utc": t(i0 + 1), "evidence_kind": "job_log"}, t(i0 + 1)),
            synth.rec("run_state", {"run_id": rid, "state": "generated", "event_utc": t(i0 + 3), "evidence_kind": "completion_artifact",
                                    "evidence_uri": store}, t(i0 + 3)),
            synth.rec("run_patch", {"run_id": rid, "fields": {"n_items_generated": n_items}}, t(i0 + 4))]
    return rid, recs


def _registry(tmp_path, monkeypatch, runs, n_items: int = 200) -> tuple:
    """``runs`` = [(key, eval_id, store_path)] -> (root, [run_id]); every run reports ``n_items`` generated."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), _dims())
    rids = []
    for k, (key, eval_id, store) in enumerate(runs):
        rid, recs = _run_recs(key, 200 + 10 * k, eval_id, store, n_items=n_items)
        R.write_records(os.path.join(root, "records", "c", f"{k + 1:06d}-run.jsonl"), recs)
        rids.append(rid)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.setenv("REGISTRY_ROOT", root)
    return root, rids


def _judgment_rows(root, rid):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        js = [tuple(r) for r in conn.execute("SELECT judgment_id, status, n_items_judged, claimed_by_session FROM judgment WHERE run_id = ?", (rid,))]
        st = conn.execute("SELECT lifecycle_state, n_items_judged FROM run WHERE run_id = ?", (rid,)).fetchone()
        return js, tuple(st)
    finally:
        conn.close()


def _ns(**kw) -> argparse.Namespace:
    base = {"registry_root": None, "judge": None, "session": "s-A", "campaign": None, "org": "lowprio", "match": "",
            "pool": "lowprio", "worklist": "registry", "dry_run": False, "cells": None, "arms": None, "families": None}
    base.update(kw)
    return argparse.Namespace(**base)


def _parser_flags(ap: argparse.ArgumentParser) -> dict:
    return {a.dest: a for a in ap._actions}


# --------------------------------------------------------------------------------------------
# every driver exposes the flag with the from-disk default
# --------------------------------------------------------------------------------------------


def test_worklist_flag_default_is_from_disk_in_every_driver():
    ap = argparse.ArgumentParser()
    W.add_worklist_arguments(ap)
    flags = _parser_flags(ap)
    assert flags["worklist"].default == "from-disk" and flags["worklist"].choices == ["registry", "from-disk"]
    assert {"registry_root", "session", "judge", "campaign"} <= set(flags)
    for rel in ("scripts/served_capability_judge.py", "scripts/sdfbase/full22/sampled/grade_static_sampled.py", "scripts/grokmt_grade.py",
                "scripts/grade_bon_live.py", "scripts/sdfbase/full22/misalign/misalign_judge.py", "scripts/petri_run.py",
                "scripts/alphasteer_f22/grade_alphasteer_controls.py"):
        src = (REPO / rel).read_text()
        assert "add_worklist_arguments(" in src, rel
        if rel != "scripts/alphasteer_f22/grade_alphasteer_controls.py":
            assert "log_from_disk()" in src, rel                          # the bypass is logged, never silent
            assert 'RegistryUnavailable' in src, rel                      # a registry failure refuses
            assert "ClaimHeld" in src, rel                                # a second driver refuses on the claim
    # petri's analyze subparser carries the flags; the top-level parser is untouched
    prun = _load("petri_run_flags", "scripts/petri_run.py")
    an = prun.build_parser().parse_args(["analyze"])
    assert an.worklist == "from-disk" and an.judge is None


# --------------------------------------------------------------------------------------------
# served_capability_judge.py — the over-refusal grade (two evals of one store; the recording hook completes claims)
# --------------------------------------------------------------------------------------------


def _controls_store(tmp_path, target="synthtarget", name="constitution_lora__controls__t0__a0__a1b2c3d4__0f1e2d3c") -> Path:
    d = tmp_path / "controls" / target / "cells" / name
    d.mkdir(parents=True)
    (d / "manifest.json").write_text("{}")
    (d / "gen_stats.json").write_text("{}")
    return d


def test_served_capability_judge_registry_branch(tmp_path, monkeypatch, capsys):
    drv = _load("served_capability_judge_wl", "scripts/served_capability_judge.py")
    tej = _load("test_registry_emit_judging_helpers", "tests/test_registry_emit_judging.py")
    store = _controls_store(tmp_path)
    # degen-fix-2: the registry worklist refuses a store it cannot measure BEFORE the claim (preflight), so the store carries
    # REAL-shaped sampled reply rows for the items the judge files, and the target's tokenizer is stubbed by registry model_id
    from test_native_judging_cardinality import _WhitespaceTok, _write_sampled_store
    items_xs, items_orb = tej._items("xstest"), tej._items("or_bench_hard_1k")
    for r in items_xs:
        r["subset"] = "safe"
    _write_sampled_store(store, "gens_xstest_v2", [r["item_id"] for r in items_xs], cap=64, at_cap_ids=set())
    _write_sampled_store(store, "gens_or_bench_hard_1k", [r["item_id"] for r in items_orb], cap=64, at_cap_ids=set())
    monkeypatch.setattr(drv._degeneration_measures(), "target_tokenizer", lambda mid, roster=None: _WhitespaceTok())
    root, (r_xs, r_orb) = _registry(tmp_path, monkeypatch, [("k_xs", "xstest", str(store)), ("k_orb", "or_bench_hard", str(store))], n_items=6)
    # Cell binding (spools, CellStore) is the driver's own concern; the registry branch hands it the store and its controls
    monkeypatch.setattr(drv, "Cell", lambda target, d, rows, controls: types.SimpleNamespace(
        target=target, cell_dir=d, name=d.name, controls=list(controls), judge_model="judge-model-x"))
    wl, cells = drv.registry_worklist_cells(_ns(), {}, ["xstest", "or_bench_hard_1k"])
    assert len(cells) == 1 and cells[0].controls == ["or_bench_hard_1k", "xstest"] and cells[0].target == "synthtarget"
    key = f"synthtarget/{store.name}"
    assert set(drv.REGISTRY_RUNS[key]) == {"xstest", "or_bench_hard_1k"}
    assert drv.REGISTRY_RUNS[key]["xstest"]["run_id"] == r_xs and drv.REGISTRY_RUNS[key]["xstest"]["judge_id"] == synth.JUDGE_ID
    for rid in (r_xs, r_orb):
        js, st = _judgment_rows(root, rid)
        assert [j[1:] for j in js] == [("in_progress", None, "s-A")] and st[0] == "judging"
    # a second session refuses on the claim (whole invocation, before any API call)
    with pytest.raises(P.ClaimHeld) as ei:
        drv.registry_worklist_cells(_ns(session="s-B"), {}, ["xstest", "or_bench_hard_1k"])
    assert ei.value.session == "s-A"
    # the driver's own judging then records through the recording hook, completing the CLAIMED rows
    cell = cells[0]
    summary, items = tej._summary(store, ["xstest", "or_bench_hard_1k"]), items_xs + items_orb
    drv._registry_record_store(cell, summary, items, _ns())
    rep = wl.finish()
    assert not rep.folded and rep.note == W.SPOOLED_NOTE, rep.as_dict()      # spooled, not folded (2026-09-30)
    rep = H.ingest_once(root, workers=0, min_free=0, derive_facts=False)             # the harvester's fold
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    for rid, control in ((r_xs, "xstest"), (r_orb, "or_bench_hard_1k")):
        js, st = _judgment_rows(root, rid)
        assert len(js) == 1 and js[0][0] == drv.REGISTRY_RUNS[key][control]["claim"]["judgment_id"], js
        assert js[0][1:] == ("complete", 4, "s-A") and st == ("judged", 4)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM item_result WHERE run_id = ?", (r_xs,)).fetchone()[0] == 6
    assert conn.execute("SELECT metric_id, value FROM run_metric WHERE run_id = ?", (r_orb,)).fetchone()[:2] == ("over_refusal", 0.5)
    # 4 judged + 2 abstentions cover the 6 generated items: a judge refusal is an outcome, not a gap
    assert conn.execute("SELECT n_judge_refusals FROM judgment WHERE run_id = ?", (r_xs,)).fetchone()[0] == 2
    assert P.pending(conn) == []
    conn.close()
    assert W.FROM_DISK_LINE not in capsys.readouterr().err


def test_served_capability_judge_refuses_an_unlocated_store_and_an_empty_queue(tmp_path, monkeypatch):
    drv = _load("served_capability_judge_wl2", "scripts/served_capability_judge.py")
    root, (rid,) = _registry(tmp_path, monkeypatch, [("k_xs", "xstest", str(tmp_path / "not_a_store"))])
    with pytest.raises(W.RegistryUnavailable) as ei:
        drv.registry_worklist_cells(_ns(), {}, ["xstest"])
    assert rid in str(ei.value) and "cannot locate" in str(ei.value)
    _, st = _judgment_rows(root, rid)
    assert st[0] == "generated"                                          # nothing claimed on a refusal
    with pytest.raises(SystemExit) as ei2:
        drv.registry_worklist_cells(_ns(), {}, ["or_bench_hard_1k"])   # no pending run of that eval
    assert "no run pending" in str(ei2.value)


def test_served_capability_judge_hook_default_path_is_unchanged(tmp_path, monkeypatch):
    """Without --worklist registry the recording hook mints the judge from the driver's fields exactly as before."""
    drv = _load("served_capability_judge_wl3", "scripts/served_capability_judge.py")
    src = (REPO / "scripts" / "served_capability_judge.py").read_text()
    assert '"threshold_rule": "label == full_refusal"}' in src and "claim=run.get(\"claim\")" in src
    monkeypatch.setattr(drv, "REGISTRY_RUNS", None)
    drv._registry_record_store(types.SimpleNamespace(target="t", name="c", cell_dir=tmp_path, judge_model="j"), {"controls": {}}, [], _ns())


# --------------------------------------------------------------------------------------------
# grade_static_sampled.py — the sampled static grade (offline world: dry run / plan path)
# --------------------------------------------------------------------------------------------


@pytest.fixture
def sampled_world(tmp_path, monkeypatch):
    tsg = _load("test_sampled_grading_helpers", "tests/test_sampled_grading.py")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-never-used")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://TESTHOST:9/gateway")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    tsg.jl(tmp_path / "goals.jsonl", tsg.GOALS)
    for tr in (0, 1):
        for esc in (False, True):
            cfg = tmp_path / "configs" / f"{tsg.run_name(tr, esc)}.yaml"
            cfg.parent.mkdir(parents=True, exist_ok=True)
            cfg.write_text(tsg.yaml.safe_dump(tsg.make_config(tr, esc)))
    tsg.write_run(tmp_path / "private", 0)
    return tsg, tmp_path


def _normalise(doc):
    """Drop the volatile provenance (timestamps, wall times, the driver's own path/sha, the worktree head)."""
    # plus the prose keys: the copy's wording of them is reworded, and the live driver's names the grading machine
    volatile = {"written_at", "started_at", "finished_at", "wall_s", "per_attack_wall_s", "worktree", "driver", "org_events_so_far",
                *PRE_CHANGE_DRIVER_PROSE_KEYS}
    if isinstance(doc, dict):
        return {k: _normalise(v) for k, v in doc.items() if k not in volatile}
    if isinstance(doc, list):
        return [_normalise(v) for v in doc]
    return doc


def _dry_outputs(grades_dir: Path) -> dict:
    dry = grades_dir / "dry_run"
    return {p.name: _normalise(json.loads(p.read_text())) for p in sorted(dry.iterdir())}


def test_sampled_static_dry_run_default_equals_explicit_from_disk_and_pre_change_driver(sampled_world, capsys):
    tsg, world = sampled_world
    gss = tsg.gss
    gss.main(tsg.grader_argv(world, extra=["--dry-run"], grades="g_default"))
    err_default = capsys.readouterr().err
    assert err_default.count(W.FROM_DISK_LINE) == 1                       # exactly one bypass line
    gss.main(tsg.grader_argv(world, extra=["--dry-run", "--worklist", "from-disk"], grades="g_explicit"))
    assert capsys.readouterr().err.count(W.FROM_DISK_LINE) == 1
    a, b = _dry_outputs(world / "g_default" / tsg.run_name(0)), _dry_outputs(world / "g_explicit" / tsg.run_name(0))
    assert a == b and set(a) == {"completion.json", "grading_provenance.json"} | {f"two_si_{f}_aggregate.json" for f in tsg.FAMS}
    # the pre-change driver (the vendored, sha256-pinned fixture) over the same world: the same files, byte-identical modulo
    # the volatile keys
    src = _pre_change_driver_bytes()
    assert b"--worklist" not in src
    old_path = world / "grade_static_sampled_pre_change.py"
    old_path.write_bytes(src)
    spec = importlib.util.spec_from_file_location("grade_static_sampled_pre_change", old_path)
    old = importlib.util.module_from_spec(spec)
    sys.modules["grade_static_sampled_pre_change"] = old
    spec.loader.exec_module(old)
    old.main(tsg.grader_argv(world, extra=["--dry-run"], grades="g_old"))
    assert capsys.readouterr().err.count(W.FROM_DISK_LINE) == 0
    c = _dry_outputs(world / "g_old" / tsg.run_name(0))
    # the one deliberate difference: the op / env `org_switch` sentence names sisteer.vault's credential, where the pre-change
    # driver named a particular secret-manager command; the old sentence is read from the vendored bytes, never restated here
    old_switch = re.search(rb'"org_switch": "([^"]+)"', src).group(1).decode()
    assert c["grading_provenance.json"].pop("org_switch") == old_switch
    assert a["grading_provenance.json"].pop("org_switch") == tsg.gss._sg.org_switch_text() != old_switch
    assert c == a, "the default (from-disk) dry run differs from the pre-change driver's in more than org_switch"


def test_pre_change_driver_fixture_matches_the_git_object():
    """The second level of the pin: the vendored copy runs the code of ``git show <sha>:<path>`` wherever the commit is
    reachable (the same syntax tree once docstrings and the prose values are set aside; comments are not in the tree). In a
    checkout without it (the Modal CPU gate's shallow exact-sha clone, or a tree that is not a git checkout at all) this
    SKIPS naming that, and the sha256 pin in ``_pre_change_driver_bytes`` is the whole guard there."""
    try:
        probe = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{PRE_CHANGE_SHA}^{{commit}}"], capture_output=True, text=True)
    except FileNotFoundError:
        pytest.skip("git is not installed here; the sha256 pin covers the fixture")
    if probe.returncode != 0:
        pytest.skip(f"commit {PRE_CHANGE_SHA[:8]} is not reachable from {REPO} (a shallow exact-sha clone, as the Modal CPU gate stages, "
                    f"carries no history): {probe.stderr.strip() or f'rc {probe.returncode}'}; the sha256 pin covers the fixture")
    shown = subprocess.run(["git", "-C", str(REPO), "show", f"{PRE_CHANGE_SHA}:{PRE_CHANGE_DRIVER_REL}"], capture_output=True, check=True).stdout
    assert _code_shape(shown) == _code_shape(_pre_change_driver_bytes()), (
        f"{PRE_CHANGE_DRIVER_FIXTURE.name} does not run the code of git show {PRE_CHANGE_SHA[:8]}:{PRE_CHANGE_DRIVER_REL}")


def test_sampled_static_registry_worklist_restricts_the_pairs_and_a_dry_run_claims_nothing(sampled_world, monkeypatch, capsys):
    tsg, world = sampled_world
    gss = tsg.gss
    import jailbreak_run as jr
    cfg = tsg.make_config(0, False)
    private_root = world / "private"
    store = jr._gated_defense_root(cfg, private_root, "pasttense", "none")
    other = jr._gated_defense_root(cfg, private_root / "elsewhere", "pasttense", "none")   # another root: not this config's
    root, (rid, r_other) = _registry(world, monkeypatch, [("k_pt", "pasttense", str(store)), ("k_other", "pasttense", str(other))])
    gss.main(tsg.grader_argv(world, extra=["--dry-run", "--worklist", "registry", "--session", "s-A"], grades="g_reg"))
    out, err = capsys.readouterr()
    assert W.FROM_DISK_LINE not in err
    assert "worklist=registry: 1 pending (family, cell) pair(s)" in out and "1 pending run(s) belong to other roots" in out
    assert "nothing claimed, nothing recorded" in out
    files = _dry_outputs(world / "g_reg" / tsg.run_name(0))
    assert set(files) == {"completion.json", "grading_provenance.json", "two_si_pasttense_aggregate.json"}   # only the pending pair
    assert set(files["two_si_pasttense_aggregate.json"]["cells"]) == {"none"}
    _, st = _judgment_rows(root, rid)
    assert st[0] == "generated"                                            # a dry run never touches the registry
    # the live path: claim -> stub grade -> complete -> judged; a second session refuses
    wl, row_of, n_other = gss.registry_worklist_pairs(_ns(), cfg, private_root, list(gss.STATIC), list(cfg["explicit_cells"]))
    assert set(row_of) == {("pasttense", "none")} and n_other == 1
    m = {"asr": {"0.25": 0.5, "0.5": 0.25, "0.75": 0.0}, "n_goals": 6, "n_jailbroken": 1, "mean_best_score": 0.3, "n_responses_graded": 12}
    thresholds = cfg["prereg"]["threshold_grid"]

    def stub(row, scope):
        scope.metrics(W.gated_defense_metric_rows(m, thresholds))
        scope.complete("complete", n_items_judged=m["n_responses_graded"])

    wl.claim_all(list(row_of.values()))
    with pytest.raises(P.ClaimHeld):
        gss.registry_worklist_pairs(_ns(session="s-B"), cfg, private_root, list(gss.STATIC), list(cfg["explicit_cells"]))[0].claim_all(list(row_of.values()))
    wl.grade(list(row_of.values()), stub)
    H.ingest_once(root, workers=0, min_free=0, derive_facts=False)   # the harvester's fold of the spooled completion records (finish() folds nothing by default, 2026-09-30)
    js, st = _judgment_rows(root, rid)
    assert len(js) == 1 and js[0][1:] == ("complete", 12, "s-A") and st == ("judged", 12)
    conn = DB.connect(DB.db_path(root), readonly=True)
    metrics = {(r[0], r[1]): r[2] for r in conn.execute("SELECT metric_id, threshold, value FROM run_metric WHERE run_id = ?", (rid,))}
    assert metrics[("asr", 0.5)] == 0.25 and metrics[("mean_best_score", -1.0)] == 0.3
    assert conn.execute("SELECT k FROM run_metric WHERE run_id = ? AND metric_id = 'asr' AND threshold = 0.5", (rid,)).fetchone()[0] == 1
    conn.close()
    # the other root's run is still pending (another invocation's)
    _, st2 = _judgment_rows(root, r_other)
    assert st2[0] == "generated"


def test_sampled_static_registry_refuses_without_a_database(sampled_world, monkeypatch, capsys):
    tsg, world = sampled_world
    monkeypatch.delenv("REGISTRY_ROOT", raising=False)
    with pytest.raises(SystemExit) as ei:
        tsg.gss.main(tsg.grader_argv(world, extra=["--dry-run", "--worklist", "registry", "--registry-root", str(world / "nowhere")]))
    assert "no registry database" in str(ei.value.msg if hasattr(ei.value, "msg") else ei.value)
    assert not (world / "grades").exists()


# --------------------------------------------------------------------------------------------
# grokmt_grade.py — the multi-turn chain's grade
# --------------------------------------------------------------------------------------------


def test_grokmt_grade_registry_branch(tmp_path, monkeypatch):
    gg = _load("grokmt_grade_wl", "scripts/grokmt_grade.py")
    cfg = {"private_run": "run_synth", "prereg": {"threshold_grid": [0.25, 0.5, 0.75]}}
    private_root = tmp_path / "private"
    store = private_root / "run_synth" / "crescendo" / "cellA"
    store.mkdir(parents=True)
    root, (rid,) = _registry(tmp_path, monkeypatch, [("k_mt", "crescendo", str(store))])
    wl, cells, row_of, n_other = gg.registry_worklist_pairs(_ns(families=None), cfg, private_root, ["pair", "crescendo", "actorattack"])
    assert cells == ["cellA"] and set(row_of) == {("cellA", "crescendo")} and n_other == 0
    # --cells narrows; a cell the queue does not name yields nothing
    assert gg.registry_worklist_pairs(_ns(cells="other"), cfg, private_root, ["crescendo"])[1] == []
    wl.claim_all(list(row_of.values()))
    with pytest.raises(P.ClaimHeld):
        gg.registry_worklist_pairs(_ns(session="s-B"), cfg, private_root, ["crescendo"])[0].claim_all(list(row_of.values()))
    m = {"asr": {"0.25": 0.5, "0.5": 0.5, "0.75": 0.25}, "n_goals": 4, "n_jailbroken": 2, "mean_best_score": 0.6, "n_responses_graded": 9}
    wl.grade(list(row_of.values()), lambda row, scope: (scope.metrics(W.gated_defense_metric_rows(m, cfg["prereg"]["threshold_grid"])),
                                                         scope.complete("complete", n_items_judged=9)))
    H.ingest_once(root, workers=0, min_free=0, derive_facts=False)   # the harvester's fold of the spooled completion records (finish() folds nothing by default, 2026-09-30)
    js, st = _judgment_rows(root, rid)
    assert len(js) == 1 and js[0][1:] == ("complete", 9, "s-A") and st == ("judged", 9)
    src = (REPO / "scripts" / "grokmt_grade.py").read_text()
    assert '"--cells is required with --worklist from-disk"' in src      # today's required flag, still required by default


# --------------------------------------------------------------------------------------------
# grade_bon_live.py — the BoN grade
# --------------------------------------------------------------------------------------------


def test_grade_bon_live_registry_branch(tmp_path, monkeypatch):
    gbl = _load("grade_bon_live_wl", "scripts/grade_bon_live.py")
    private_root = tmp_path / "private"
    arm_dir = private_root / "bon_run" / "vector"
    arm_dir.mkdir(parents=True)
    (arm_dir / "bon_responses.jsonl").write_text("")
    root, (r_dir, r_file) = _registry(tmp_path, monkeypatch, [("k_bon", "strongreject_bon", str(arm_dir)),
                                                             ("k_bon2", "strongreject_bon", str(private_root / "bon_run" / "none" / "bon_responses.jsonl"))])
    wl, arms, row_of = gbl.registry_worklist_arms(_ns(), private_root, "bon_run")
    assert arms == ["none", "vector"] and row_of["vector"]["run_id"] == r_dir and row_of["none"]["run_id"] == r_file
    assert gbl.registry_worklist_arms(_ns(arms="vector"), private_root, "bon_run")[1] == ["vector"]
    wl.claim_all([row_of[a] for a in arms])
    with pytest.raises(P.ClaimHeld):
        gbl.registry_worklist_arms(_ns(session="s-B"), private_root, "bon_run")[0].claim_all([row_of["vector"]])
    arm_out = {"mean_normalized_auc": 0.42, "asr_at_n": {"1": 0.1, "64": 0.7}, "n_goals": 10, "n_unscored": 3,
               "n_star_by_goal": {f"g{i}": (i + 1 if i < 7 else -1) for i in range(10)}}     # 7 goals hit within 64, g0 at its first draw
    # BONRUNG (the registry owner, 2026-10-14): the rows are the run's RUNG's. The registry row is a cap-512 store run (synth.run_payload's
    # token_cap_tokens 512, no ladder, no pass label), read off its `run` row BEFORE the claim by bon_rungs_of (a climb refuses there, by name);
    # the eval's headline is read from the eval row (`asr`), and the rows are the PAPER's 512 set: the headline under that id, the curve,
    # bon_nauc, the four degeneracy shares over the store's draws (none at cap here). The native pass's `asr_bon_cap4096` never appears on it.
    assert gbl.bon_eval_headline_metric_id(wl) == "asr"
    rungs = gbl.bon_rungs_of(wl, {"vector": row_of["vector"]}, headline_metric_id="asr")
    assert rungs == {"vector": {"cap": 512, "pass_label": None, "ladder": [], "reading": "store_512", "instrument_kind": "512_store",
                                "instrument": {"read_at": "fixed", "ladder": [512], "cap": 512, "instrument_id": str(_instrument("strongreject_bon")["instrument_id"])}}}   # BONFOLD: the instrument read beside the row
    counts = gbl.bon_store_512_counts([{"goal_id": f"g{i}", "sample_idx": k, "trial": 0, "response": "a reply", "finish_reason": "stop"} for i in range(10) for k in range(64)])
    rows = gbl.bon_metric_rows(arm_out, 64, cap=rungs["vector"]["cap"], pass_label=rungs["vector"]["pass_label"], ladder=rungs["vector"]["ladder"],
                               headline_metric_id="asr", store_counts=counts)
    # BONHEADLINE: the canonical rows -- headline at the judge cutoff with k = goals hit, the curve, bon_nauc; no asr_at_n_max@64.0
    assert [(r["metric_id"], r["threshold"], r["value"], r["k"]) for r in rows] == [
        ("asr", 0.5, 0.7, 7), ("asr_at_n1", 0.5, 0.1, 1), ("asr_at_n64", 0.5, 0.7, 7), ("bon_nauc", 0.5, 0.42, None),
        ("at_cap_share_bon_512", -1.0, 0.0, 0), ("loop_share_bon_512", -1.0, 0.0, 0), ("coherent_at_cap_share_bon_512", -1.0, 0.0, 0),
        ("secondary_flag_only_share_bon_512", -1.0, 0.0, 0)]
    assert rows[0]["extra"]["instrument_kind"] == "512_store" and all(r["n"] == 640 for r in rows[4:])
    # a CLIMB registration (cap 4,096 / ladder_rung over a store) with no locatable climb root refuses by name before any claim -- its reading is the climb fold
    inst = _instrument("strongreject_bon")
    climb_store = str(private_root / "bon_run" / "climbed")
    pc = synth.run_payload("k_bon_climb", "none", "none", 260, eval_id="strongreject_bon", item_set_id="strongreject_bon_set", instrument_id=inst["instrument_id"],
                           decode_source="generation_config", store_uri=climb_store, token_cap_tokens=4096, pass_label="ladder_rung", token_cap_ladder_json=[512, 4096])
    pc["run_id"] = I.run_id(synth.CAMPAIGN, "k_bon_climb", "none", "strongreject_bon", str(inst["instrument_id"]))
    t = synth.t                                                          # the same five records _run_recs writes, the rung on the registration
    R.write_records(os.path.join(root, "records", "c", "000009-climb.jsonl"),
                    [synth.rec("run", pc, t(260)),
                     synth.rec("run_state", {"run_id": pc["run_id"], "state": "queued", "event_utc": t(260), "evidence_kind": "submit_wrapper"}, t(260)),
                     synth.rec("run_state", {"run_id": pc["run_id"], "state": "running", "event_utc": t(261), "evidence_kind": "job_log"}, t(261)),
                     synth.rec("run_state", {"run_id": pc["run_id"], "state": "generated", "event_utc": t(263), "evidence_kind": "completion_artifact",
                                             "evidence_uri": climb_store}, t(263)),
                     synth.rec("run_patch", {"run_id": pc["run_id"], "fields": {"n_items_generated": 200}}, t(264))])
    rep = H.ingest_once(root, workers=0, min_free=0); assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    with DB.reading(DB.db_path(root)) as conn:
        cap_c, label_c, ladder_c = conn.execute("SELECT token_cap_tokens, pass_label, token_cap_ladder_json FROM run WHERE run_id = ?", (pc["run_id"],)).fetchone()
    assert (cap_c, label_c, json.loads(ladder_c)) == (4096, "ladder_rung", [512, 4096])
    with pytest.raises(W.RegistryUnavailable) as ei:
        gbl.bon_rungs_of(wl, {"climbed": {"run_id": pc["run_id"]}}, headline_metric_id="asr")
    assert str(ei.value).startswith(f"{gbl.DRIVER_ID}: run {pc['run_id']} is a BoN CLIMB (cap 4096 / ladder_rung, ladder [512, 4096]) and is REFUSED before any claim")
    assert "located by NO admitted locator" in str(ei.value) and "nothing claimed, nothing written" in str(ei.value)     # BONFOLD: no climb_root record, no override
    with pytest.raises(W.RegistryUnavailable, match="is a cap-512 rung .* but the registry's eval declares headline_metric_id 'asr_bon_cap4096'"):
        gbl.bon_rungs_of(wl, {"vector": row_of["vector"]}, headline_metric_id="asr_bon_cap4096")
    r64 = next(r for r in rows if r["metric_id"] == "asr_at_n64")   # bon_n == the row's own N, n_max beside it
    assert r64["extra"]["bon_n"] == 64 and r64["extra"]["n_max"] == 64 and r64["threshold"] == 0.5 and r64["producer_metric_spelling"] == "asr_at_n[64]"
    assert next(r for r in rows if r["metric_id"] == "asr_at_n1")["extra"]["bon_n"] == 1
    from sisteer.registry import vocab as _V
    assert [_V.bon_metric_refusal(r) for r in rows] == [None] * len(rows), "every canonical row passes the emit-time best-of-N guard (D934)"
    wl.grade([row_of["vector"]], lambda row, scope: (scope.metrics(rows), scope.complete("complete", n_items_judged=10 * 64 - 3)))
    H.ingest_once(root, workers=0, min_free=0, derive_facts=False)   # the harvester's fold of the spooled completion records (finish() folds nothing by default, 2026-09-30)
    js, st = _judgment_rows(root, r_dir)
    assert len(js) == 1 and js[0][1:] == ("complete", 637, "s-A") and st == ("judged", 637)
    js2, st2 = _judgment_rows(root, r_file)
    assert js2[0][1] == "in_progress" and st2[0] == "judging"           # claimed, not yet graded
    assert wl.release(row_of["none"])
    src = (REPO / "scripts" / "grade_bon_live.py").read_text()
    assert '"--arms is required with --worklist from-disk"' in src


# --------------------------------------------------------------------------------------------
# misalign_judge.py — agentic / EM
# --------------------------------------------------------------------------------------------


def test_misalign_judge_registry_branch(tmp_path, monkeypatch):
    mj = _load("misalign_judge_wl", "scripts/sdfbase/full22/misalign/misalign_judge.py")
    results = tmp_path / "results" / "gemma"
    d_ag = results / "agentic" / "cellA"
    d_em = results / "em8q" / "cellB"
    d_ag.mkdir(parents=True)
    d_em.mkdir(parents=True)
    root, (r_ag, r_em) = _registry(tmp_path, monkeypatch, [("k_ag", "agentic_leaking", str(d_ag)), ("k_em", "em_risky_financial", str(d_em))])
    wl, row_of = mj.registry_worklist_cells(_ns(families=["agentic", "em8q"]), results)
    assert set(row_of) == {("agentic", d_ag.resolve()), ("em8q", d_em.resolve())}
    assert set(mj.registry_worklist_cells(_ns(families=["em8q"]), results)[1]) == {("em8q", d_em.resolve())}
    wl.claim_all(list(row_of.values()))
    with pytest.raises(P.ClaimHeld):
        mj.registry_worklist_cells(_ns(session="s-B", families=["agentic"]), results)[0].claim_all([row_of[("agentic", d_ag.resolve())]])
    judged_ag = [{"item_id": "leak_1", "sample_idx": 0, "judge_blocked": False, "classifier_harmful": 1, "harmful": 0},
                 {"item_id": "leak_2", "sample_idx": 0, "judge_blocked": True, "classifier_harmful": None, "harmful": None}]
    judged_em = [{"item_id": "quick_buck", "sample_idx": 3, "judge_blocked": False, "alignment": 20, "coherence": 80, "em": 1}]
    rows_ag = mj.judged_item_rows("agentic", judged_ag)
    assert [(r["item_id"], r["metric_id"], r["value"], r["label"]) for r in rows_ag] == [
        ("leak_1", "classifier_harmful", 1.0, None), ("leak_1", "harmful", 0.0, None),
        ("leak_2", "classifier_harmful", None, "judge_blocked"), ("leak_2", "harmful", None, "judge_blocked")]
    assert [(r["metric_id"], r["value"], r["sample_index"]) for r in mj.judged_item_rows("em8q", judged_em)] == [
        ("alignment", 20.0, 3), ("coherence", 80.0, 3), ("em", 1.0, 3)]

    def grade(row, scope):
        judged = judged_ag if row["run_id"] == r_ag else judged_em
        fam = "agentic" if row["run_id"] == r_ag else "em8q"
        scope.items(mj.judged_item_rows(fam, judged))
        scope.n_refusals = sum(1 for r in judged if r.get("judge_blocked"))
        scope.complete("complete", n_items_judged=len(judged))

    wl.grade(list(row_of.values()), grade)
    H.ingest_once(root, workers=0, min_free=0, derive_facts=False)   # the harvester's fold of the spooled completion records (finish() folds nothing by default, 2026-09-30)
    js, st = _judgment_rows(root, r_ag)
    assert len(js) == 1 and js[0][1:] == ("complete", 2, "s-A") and st == ("judged", 2)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT n_judge_refusals FROM judgment WHERE run_id = ?", (r_ag,)).fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM item_result WHERE run_id = ?", (r_em,)).fetchone()[0] == 3
    conn.close()


# --------------------------------------------------------------------------------------------
# petri_run.py analyze — the Petri aggregation is the judgment step
# --------------------------------------------------------------------------------------------


def test_petri_analyze_registry_branch(tmp_path, monkeypatch):
    prun = _load("petri_run_wl", "scripts/petri_run.py")
    out_root = tmp_path / "petri_out"
    (out_root / "cells").mkdir(parents=True)
    root, (r_in, r_out) = _registry(tmp_path, monkeypatch, [("k_pe", "petri_pooled", str(out_root / "cells" / "none__lora0")),
                                                          ("k_pe_other", "petri_pooled", str(tmp_path / "other_audit"))])
    # the run's object discriminator follows identity's petri grammar: <tag>:<object>
    conn = DB.connect(DB.db_path(root), readonly=True)
    row_in = [r for r in P.pending(conn) if r["run_id"] == r_in][0]
    conn.close()
    wl, rows = prun.registry_worklist_runs(_ns(), out_root)
    assert [r["run_id"] for r in rows] == [r_in]
    analysis = {"by_cell": {"none": {"behaviors": {"b1": {"mean_behavior_score": 2.0, "n_items": 10}, "b2": {"mean_behavior_score": 4.0, "n_items": 10}}}}}
    assert prun.petri_metric_rows(analysis, {**row_in, "producer_object_key_legacy": "tag:none"}) == [
        {"metric_id": "petri_score", "value": 3.0, "n": 20, "producer_metric_spelling": "by_cell[cell].behaviors[*].mean_behavior_score (mean)"}]
    with pytest.raises(prun.DriverError, match="no harness translation"):                                      # never a guessed number -- and never silence: refuse by name
        prun.petri_metric_rows(analysis, {**row_in, "producer_object_key_legacy": "tag:unknown"})
    wl.claim_all(rows)
    with pytest.raises(P.ClaimHeld):
        prun.registry_worklist_runs(_ns(session="s-B"), out_root)[0].claim_all(rows)
    wl.grade(rows, lambda row, scope: (scope.metrics(prun.petri_metric_rows(analysis, {**row, "producer_object_key_legacy": "tag:none"})),
                                       scope.complete("complete", n_items_judged=int(row["n_items_generated"]))))
    H.ingest_once(root, workers=0, min_free=0, derive_facts=False)   # the harvester's fold of the spooled completion records (finish() folds nothing by default, 2026-09-30)
    js, st = _judgment_rows(root, r_in)
    assert len(js) == 1 and js[0][1:] == ("complete", 200, "s-A") and st == ("judged", 200)
    _, st2 = _judgment_rows(root, r_out)
    assert st2[0] == "generated"                                          # the other audit's run is untouched


# --------------------------------------------------------------------------------------------
# grade_alphasteer_controls.py — forwards the flags to the over-refusal driver
# --------------------------------------------------------------------------------------------


def test_alphasteer_controls_forwards_the_worklist_flags():
    gac = _load("grade_alphasteer_controls_wl", "scripts/alphasteer_f22/grade_alphasteer_controls.py")
    assert gac.registry_worklist_passthrough(argparse.Namespace(worklist="from-disk", registry_root=None, session=None, campaign=None, judge=None)) == []
    fwd = gac.registry_worklist_passthrough(argparse.Namespace(worklist="registry", registry_root="/r", session="s", campaign="c", judge=["j"]))
    assert fwd == ["--worklist", "registry", "--registry-root", "/r", "--session", "s", "--campaign", "c", "--judge", "j"]
    cmd = gac.driver_command(Path("/x/served_capability_judge.py"), python="py", controls_root=Path("/c"), targets=["t"], concurrency=1, org="lowprio",
                             cell_parallel=1, dry_run=True, sentinel_dir=Path("/s"), cache_note="n", worklist_args=fwd)
    assert cmd[-len(fwd) - 1:-1] == fwd and cmd[-1] == "--dry-run"
    cmd0 = gac.driver_command(Path("/x/served_capability_judge.py"), python="py", controls_root=Path("/c"), targets=["t"], concurrency=1, org="lowprio",
                              cell_parallel=1, dry_run=False, sentinel_dir=Path("/s"), cache_note="n")
    assert "--worklist" not in cmd0                                        # the default forwards nothing: today's command line


# --------------------------------------------------------------------------------------------
# the from-disk line through a driver's main()
# --------------------------------------------------------------------------------------------


def test_misalign_judge_main_default_prints_the_bypass_line_then_behaves_as_today(tmp_path, capsys):
    mj = _load("misalign_judge_wl_main", "scripts/sdfbase/full22/misalign/misalign_judge.py")
    with pytest.raises(SystemExit) as ei:
        mj.main(["--results-root", str(tmp_path / "missing")])
    assert "is not a directory" in str(ei.value)
    assert capsys.readouterr().err.count(W.FROM_DISK_LINE) == 1
    with pytest.raises(SystemExit) as ei:
        mj.main(["--results-root", str(tmp_path / "missing"), "--worklist", "registry", "--registry-root", str(tmp_path / "nowhere")])
    assert "is not a directory" in str(ei.value)                           # the root check precedes the worklist, as today
    assert W.FROM_DISK_LINE not in capsys.readouterr().err


@pytest.mark.parametrize("case", ["mismatch", "absent"])
def test_misalign_judge_preflight_refuses_a_mismatched_or_absent_served_identity_before_any_claim_or_filing(tmp_path, monkeypatch, case):
    """The identity cross-check ran only on entering the rows step, after claim_all, the item rows, the
    metric and complete() -- a mismatched served revision had already spooled judgments under the registry run before the refusal. Now it
    is a PREFLIGHT: with the registry's expectation passed and the judged root's files stating another revision (or no identity at all),
    main exits 1 before the plan is read -- no claim row, no spool record of any kind, the run still pending and unclaimed."""
    mj = _load("misalign_judge_preflight", "scripts/sdfbase/full22/misalign/misalign_judge.py")
    results = tmp_path / "results" / "qwen35_35b_a3b_s2r"
    d_ag = results / "agentic" / "cellA"; d_ag.mkdir(parents=True)
    root, (r_ag,) = _registry(tmp_path, monkeypatch, [("k_ag", "agentic_leaking", str(d_ag))])
    if case == "mismatch":
        (results / "endpoint.snapshot.json").write_text(json.dumps({"model": "org/model", "model_revision": "b" * 40}))   # served at ANOTHER snapshot
    spool_before = sorted(str(p) for p in Path(H.records_dir(root)).rglob("*"))
    rc = mj.main(["--results-root", str(results), "--families", "agentic", "--worklist", "registry", "--registry-root", root, "--session", "s-A",
                  "--identity-model-id", "org/model", "--identity-model-revision", "a" * 40])
    assert rc == 1
    js, st = _judgment_rows(root, r_ag)
    assert js == [] and st[0] == "generated"                                                # no claim, no judgment, the run untouched
    assert sorted(str(p) for p in Path(H.records_dir(root)).rglob("*")) == spool_before     # ZERO spool records of any kind
    conn = DB.connect(DB.db_path(root), readonly=True)
    pend = [r for r in P.pending(conn) if r["run_id"] == r_ag]
    assert len(pend) == 1 and pend[0]["pending_reason"] not in ("claimed",) and pend[0].get("claim_session") is None    # pending, unclaimed
    conn.close()


# --------------------------------------------------------------------------------------------
# petri_run.py — the identity_stale re-pend is admitted by the Petri worklist
# --------------------------------------------------------------------------------------------


def test_petri_worklist_admits_an_identity_stale_re_pend_once_and_excludes_other_reasons(tmp_path, monkeypatch, capsys):
    """A run re-pended as IDENTITY_STALE_REASON on a once-judged row (lifecycle
    `judged` / `ingested`; the stale judgment's row untouched, still in existing_judgments_json) leaves every selection registry_worklist_runs
    makes today -- (generated, judging) by lifecycle; superseded / abandoned / own claimed by reason -- and is now admitted on exactly that
    evidence, once; a re-pend under a reason this worklist does not know stays out. The view does not produce the reason today (0 rows,
    read-only census 2026-10-04): the future view is stood in for by a stub on pending.pending that relabels two runs' pending_reason
    and applies the pending_reason / lifecycle_state filters over the relabelled rows; the rows are otherwise the real view's (runs judged
    complete short of their coverage, lifecycle `judged`)."""
    prun = _load("petri_run_wl_g41fix", "scripts/petri_run.py")
    assert prun.IDENTITY_STALE_REASON == "identity_stale"
    out_root = tmp_path / "petri_out"; (out_root / "cells").mkdir(parents=True)
    root, (r_fresh, r_stale, r_other) = _registry(tmp_path, monkeypatch, [("k_fresh", "petri_pooled", str(out_root / "cells" / "none__a")),
                                                                         ("k_stale", "petri_pooled", str(out_root / "cells" / "none__b")),
                                                                         ("k_other", "petri_pooled", str(out_root / "cells" / "none__c"))])
    wl0, rows0 = prun.registry_worklist_runs(_ns(), out_root)
    assert sorted(r["run_id"] for r in rows0) == sorted([r_fresh, r_stale, r_other])
    # two runs judged COMPLETE short of their coverage: once-judged rows (lifecycle `judged`) that still pend -- the real view's coverage_short
    analysis = {"by_cell": {"none": {"behaviors": {"b1": {"mean_behavior_score": 2.0, "n_items": 10}, "b2": {"mean_behavior_score": 4.0, "n_items": 10}}}}}
    judged = [r for r in rows0 if r["run_id"] in (r_stale, r_other)]
    wl0.grade(judged, lambda row, scope: (scope.metrics(prun.petri_metric_rows(analysis, {**row, "producer_object_key_legacy": "tag:none"})),
                                          scope.complete("complete", n_items_judged=150)))
    H.ingest_once(root, workers=0, min_free=0, derive_facts=False)
    conn = DB.connect(DB.db_path(root), readonly=True)
    view = {r["run_id"]: (r["pending_reason"], r["lifecycle_state"], json.loads(r["existing_judgments_json"])) for r in P.pending(conn, None, synth.CAMPAIGN, run_id=(r_stale, r_other))}
    conn.close()
    assert {v[0] for v in view.values()} == {"coverage_short"} and {v[1] for v in view.values()} == {"judged"}, view
    assert all(len(v[2]) == 1 and v[2][0]["status"] == "complete" for v in view.values())
    # the owner's view, stood in for: r_stale re-pends as identity_stale, r_other under a reason this worklist does not know
    real = P.pending

    def stub(conn, judge_id=None, campaign=None, **filters):
        want_reason = filters.pop("pending_reason", None); want_lc = filters.pop("lifecycle_state", None)
        rows = real(conn, judge_id, campaign, **filters)
        for r in rows:
            r["pending_reason"] = {r_stale: prun.IDENTITY_STALE_REASON, r_other: "some_future_reason"}.get(str(r["run_id"]), r["pending_reason"])
        if want_reason is not None:
            rows = [r for r in rows if r["pending_reason"] == want_reason]
        if want_lc is not None:
            lcs = tuple(want_lc) if isinstance(want_lc, (tuple, list)) else (want_lc,)
            rows = [r for r in rows if r["lifecycle_state"] in lcs]
        return rows

    monkeypatch.setattr(P, "pending", stub)
    capsys.readouterr()
    wl, rows = prun.registry_worklist_runs(_ns(session="s-B"), out_root)
    ids = [r["run_id"] for r in rows]
    assert sorted(ids) == sorted([r_fresh, r_stale]) and len(ids) == len(set(ids)), ids    # the fresh run by lifecycle, the identity-stale one by reason, once each; the unknown reason excluded
    assert [r["pending_reason"] for r in rows if r["run_id"] == r_stale] == [prun.IDENTITY_STALE_REASON]
    assert f"1 run(s) pending again as {prun.IDENTITY_STALE_REASON}" in capsys.readouterr().out
    # the claim gate below stands: the same worklist claims the admitted rows as any other (pending.claim decides who may take them)
    claims = wl.claim_all(rows)
    assert sorted(k[0] for k in claims) == sorted([r_fresh, r_stale])
    # today's view (no stub): the identity_stale admission is the window's, not this tree's -- the once-judged run is not selected by any
    # of the worklist's queries
    monkeypatch.setattr(P, "pending", real)
    _, rows_today = prun.registry_worklist_runs(_ns(session="s-C"), out_root)
    assert r_stale not in [r["run_id"] for r in rows_today] and r_other not in [r["run_id"] for r in rows_today]
