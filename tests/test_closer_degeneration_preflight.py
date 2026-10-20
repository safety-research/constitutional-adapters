"""degen-fix-2: the closers' degeneration preconditions hold BEFORE any claim, and a REAL greedy controls store
files measured union5 rows through registry filing.
  * the static sampled grader: a target whose pinned tokenizer snapshot is absent refuses by name with ZERO claims (every selected judgment
    stays pending; no judgment record is spooled) -- before this (fix-1) the tokenizer resolved after claim_all, so a missing snapshot left
    every selected judgment durably in_progress;
  * the controls judge over a greedy store written by the REAL writer (served_capability_run on the offline world: gens + spool header, no
    .meta.jsonl -- the inventory test_sampled_controls pins): the worklist path refuses BEFORE claiming when a pending control's store has
    no reply rows or the target's snapshot is absent (both pending rows stay pending, nothing spooled), and over the intact store files the
    judgment complete with every union5 member measured over the population (xstest: the safe subset), the canonical at-cap at the spool
    header's common cap, the producer's finish-reason row from gen_stats, and the closer's counts equal to the walker's on the same rows.
Synthetic worlds only (TESTHOST, placeholder ids); a whitespace tokenizer stands in for the target's."""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "tests", REPO / "scripts", REPO / "scripts" / "sdfbase" / "full22", REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sisteer.eval import degeneracy_union as du  # noqa: E402
from sisteer.registry import harvest as H  # noqa: E402
from sisteer.registry import pending as P  # noqa: E402
from sisteer.registry import synth  # noqa: E402
from test_native_judging_cardinality import (  # noqa: E402,F401  (the registry worlds + the static sampled world; the fixture is picked up by name)
    REMOTE_PRIVATE, _WhitespaceTok, _assert_native_judged, _conn, _degeneration, _load, _registry, sampled_world)
from test_served_capability import ORB_ROWS, XST_ROWS, _cell_dir, world  # noqa: E402,F401  (the offline world: the REAL greedy writer)
import degeneracy_gap_fill as GF  # noqa: E402
import served_capability_run as srun  # noqa: E402


def _judgment_records(spool: str) -> list:
    out = []
    for dp, _dn, fns in os.walk(spool):
        out += [os.path.join(dp, f) for f in fns if "judgment" in f]
    return out


# --------------------------------------------------------------------------------------------
# the static sampled grader: a missing snapshot refuses BEFORE the claim -- zero claims, nothing spooled, every row still pending
# --------------------------------------------------------------------------------------------

def test_the_static_grader_refuses_before_any_claim_when_the_targets_snapshot_is_missing(sampled_world, monkeypatch, capsys):
    tsg, w = sampled_world
    gss = tsg.gss
    import jailbreak_run as jr
    cfg = tsg.make_config(0, False)
    private_root = w / "private"
    stores = {c: jr._gated_defense_root(cfg, private_root, "pasttense", c) for c in tsg.CELLS}
    remote = lambda p: REMOTE_PRIVATE + "/" + str(Path(p).relative_to(private_root))      # noqa: E731  -- the recorded compute-node path
    root, (r_none, r_cell) = _registry(w, monkeypatch, [
        ("k_pt_none", "pasttense", remote(stores["none"]), 12, "none", "none", None),
        ("k_pt_cellA", "pasttense", remote(stores["cellA__psr_ce_all"]), 12, "constitution_lora@deadbeef", synth.OBJECT_SHA, None)])

    def missing(target, roster=None):          # the fixture's whitespace stand-in withdrawn: the pinned snapshot is absent from the local cache
        raise RuntimeError(f"tokenizer for 'Qwen/Synthetic' @ deadbeef is not in the local HF cache (OSError); fetch it first (target {target})")
    monkeypatch.setattr(gss.DM, "target_tokenizer", missing)
    with pytest.raises(gss.Refuse) as ei:
        gss.main(tsg.grader_argv(w, extra=["--worklist", "registry", "--session", "s-A", "--run-id", r_none, "--run-id", r_cell,
                                           "--store-root-map", f"cluster_a={REMOTE_PRIVATE}={private_root}"], grades="g_native"))
    assert ei.value.code == 2                                                   # Refuse(SystemExit): the message is printed as REFUSING: … on stderr
    out, err = capsys.readouterr()
    assert "REFUSING:" in err and "refusing before any claim" in err and "not in the local HF cache" in err and "nothing claimed" in err, err
    assert f"target {synth.MODEL_ID}" in err                                    # keyed by the rows' registry model_id, not --target
    assert "claimed by session" not in out + err                                # no claim line was ever printed
    # zero claims: nothing spooled, no judgment row after the harvester's fold, both selected runs still pending
    assert _judgment_records(H.records_dir(root)) == []
    H.ingest_once(root, workers=0, min_free=0, derive_facts=False)
    conn = _conn(root)
    assert {r["run_id"] for r in P.pending(conn)} == {r_none, r_cell}
    assert conn.execute("SELECT COUNT(*) FROM judgment").fetchone()[0] == 0
    conn.close()


# --------------------------------------------------------------------------------------------
# the controls judge over a REAL greedy store: pre-claim refusals, then measured union5 rows through registry filing
# --------------------------------------------------------------------------------------------

def _recs(control: str, rows: list) -> list:
    out = []
    for i, r in enumerate(rows):
        rec = {"control": control, "task": f"{control}_task", "item_id": r["item_id"], "request_key": f"k{i}", "judge_model": "judge-model-x",
               "rubric_sha": synth.SHA256_A, "protocol_sha": synth.SHA256_B, "cache_org": "lowprio", "verdict_created_at": 1.0,
               "cached_before_run": False, "status": "judged", "label": "full_compliance", "label_source": "store"}   # the derived rows' vocabulary
        if control == "xstest":
            rec["subset"] = r["subset"]
        out.append(rec)
    return out


def _blk(n: int) -> dict:
    return {"status": "ok", "task": "t", "n": n, "at_cap_share": 0.0,
            "over_refusal": {"point_estimate_over_judged": 0.0, "wilson95_over_judged": [0.0, 0.5], "n_judged": n, "n": n, "n_abstained": 0,
                             "abstention_rate": 0.0, "abstention_band_full_n": [0.0, 0.0], "full_refusal_n": 0}}


def test_the_judge_refuses_an_unmeasurable_store_before_claiming_and_files_measured_union5_rows_from_a_real_greedy_store(world, tmp_path, monkeypatch, capsys):
    # the REAL greedy writer: the offline world's generate run over xstest + or_bench (greedy decode -> gens + spool header, no sidecar)
    argv = [a if a != "gsm8k,xstest,or_bench_hard_1k" else "xstest,or_bench_hard_1k" for a in world.argv]
    srun.main(argv)
    store = _cell_dir(world.out_root)
    assert not list(store.rglob("*.meta.jsonl")) and (store / "private" / "gens_xstest_v2.jsonl").is_file()
    assert GF.controls_store_shape(store, "xstest") == "greedy" and GF.controls_store_shape(store, "or_bench_hard") == "greedy"
    gen_stats = json.loads((store / "gen_stats.json").read_text())
    n_xs, n_orb = len(XST_ROWS), len(ORB_ROWS)
    assert gen_stats["xstest"]["n_total"] == n_xs and gen_stats["or_bench_hard_1k"]["n_total"] == n_orb
    capsys.readouterr()

    drv = _load("served_capability_judge_native_fix2", "scripts/served_capability_judge.py")
    root, (r_xs, r_orb) = _registry(tmp_path, monkeypatch, [
        ("k_xs", "xstest", str(store), n_xs, "none", "none", None), ("k_orb", "or_bench_hard", str(store), n_orb, "none", "none", None)])
    monkeypatch.setattr(drv, "Cell", lambda target, d, rows, controls: types.SimpleNamespace(
        target=target, cell_dir=d, name=d.name, controls=list(controls), judge_model="judge-model-x", gen_stats=gen_stats))
    DMm = drv._degeneration_measures()
    monkeypatch.setattr(DMm, "target_tokenizer", lambda mid, roster=None: _WhitespaceTok())
    ns = argparse.Namespace(registry_root=None, judge=None, session="s-A", campaign=None, org="lowprio", match="", worklist="registry",
                            run_id=None, model_id=None, store_root_map=None)
    spool = H.records_dir(root)

    # (1) a pending control whose store carries no reply rows refuses BEFORE the claim, by name
    gp = store / "private" / "gens_or_bench_hard_1k.jsonl"
    aside = gp.with_name(gp.name + ".aside")
    gp.rename(aside)
    with pytest.raises(SystemExit) as ei:
        drv.registry_worklist_cells(ns, {}, ["xstest", "or_bench_hard_1k"])
    assert "refusing before any claim" in str(ei.value) and "gens_or_bench_hard_1k.jsonl" in str(ei.value) and "nothing claimed" in str(ei.value), str(ei.value)
    aside.rename(gp)
    # (2) a target without its pinned snapshot refuses BEFORE the claim, keyed by the rows' registry model_id
    def missing(mid, roster=None):
        raise RuntimeError(f"tokenizer for {mid!r} is not in the local HF cache")
    monkeypatch.setattr(DMm, "target_tokenizer", missing)
    with pytest.raises(SystemExit) as ei:
        drv.registry_worklist_cells(ns, {}, ["xstest", "or_bench_hard_1k"])
    assert "not in the local HF cache" in str(ei.value) and repr(synth.MODEL_ID) in str(ei.value), str(ei.value)
    assert _judgment_records(spool) == []
    conn = _conn(root)
    assert {r["run_id"] for r in P.pending(conn)} == {r_xs, r_orb}
    conn.close()
    monkeypatch.setattr(DMm, "target_tokenizer", lambda mid, roster=None: _WhitespaceTok())

    # (3) the intact store: claimed, the scripted verdicts filed, the judgment complete with every union5 member MEASURED
    wl, cells = drv.registry_worklist_cells(ns, {}, ["xstest", "or_bench_hard_1k"])
    cell = cells[0]
    items = _recs("xstest", XST_ROWS) + _recs("or_bench_hard_1k", ORB_ROWS)
    summary = {"target": cell.target, "cell": cell.name, "cell_id": "none", "intervention_hash": "0badf00d0badf00d", "model_id": synth.MODEL_ID,
               "controls": {"xstest": _blk(n_xs), "or_bench_hard_1k": _blk(n_orb)}}
    drv._registry_record_store(cell, summary, items, ns)
    out = capsys.readouterr()
    assert "not recorded" not in out.out + out.err, out.out + out.err           # the scope filed (no swallowed exception)
    assert "REFUSED" not in out.out + out.err, out.out + out.err                # no derived-row refusal (the labels are the derived vocabulary)
    wl.finish()
    rep = H.ingest_once(root, workers=0, min_free=0, derive_facts=False)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    conn = _conn(root)
    ws = _WhitespaceTok()
    for rid, control, gf_key, n in ((r_xs, "xstest", "xstest", n_xs), (r_orb, "or_bench_hard_1k", "or_bench_hard", n_orb)):
        j = _assert_native_judged(conn, rid, n_items_judged=n)
        deg = _degeneration(conn, rid)
        fr = deg[("at_token_cap_finish_reason", "all")]
        assert (fr["state"], fr["numerator"], fr["denominator"]) == ("measured", gen_stats[control]["finish_reasons"].get("length", 0), n), fr
        # the closer's rows == the walker's arithmetic on the same store (the greedy reader, the common cap 64, whitespace tokens)
        rows, facts = GF.read_controls_rows(store, gf_key, label="w")
        assert facts["shape"] == "greedy" and facts["common_cap"] == 64 and len(rows) == n
        wm = GF.measure_rows_by_cap(du, rows, tok=ws, label="w")
        expected = {"union5": wm["degenerate_share"]["n"], "classifier": wm["detectors"]["classifier"]["n"], "contentless": wm["detectors"]["contentless"]["n"],
                    "rep_ratio_loop": wm["detectors"]["rep_ratio_loop"]["n"], "zlib_collapse": wm["detectors"]["zlib_collapse"]["n"],
                    "at_token_cap_retokenized": wm["detectors"]["at_cap"]["at_cap_retokenized"]["n"],
                    "truncation_only": wm["detectors"]["truncation_only"]["k"], "loop_without_cap": wm["detectors"]["loop_without_cap"]["k"]}
        assert wm["detectors"]["at_cap"]["at_cap_retokenized"]["defined"] is True and wm["detectors"]["at_cap"]["at_cap_finish_reason"]["defined"] is False
        for measure, k in expected.items():
            d = deg[(measure, "all")]
            assert (d["state"], d["numerator"], d["denominator"], d["unit"], d["judgment_id"]) == ("measured", k, n, "replies", j["judgment_id"]), (measure, d)
        notes, defn = conn.execute("SELECT notes, definition_version FROM run_degeneration WHERE run_id = ? AND measure = 'union5'", (rid,)).fetchone()
        assert "store shape greedy" in notes and f"tokenizer model_id {synth.MODEL_ID}" in notes and defn == f"{du.SCHEMA}@{du.DEFINITION_SHA}", notes
        assert ("xstest safe subset" in notes) == (control == "xstest")
    assert P.pending(conn) == []
    conn.close()


# --------------------------------------------------------------------------------------------
# degen-fix-3: a real-shaped SAMPLED store whose sidecar is missing refuses BEFORE any claim — zero claims, zero filings
# --------------------------------------------------------------------------------------------

def test_a_sampled_store_missing_its_sidecar_refuses_before_any_claim(tmp_path, monkeypatch):
    from test_native_judging_cardinality import _write_sampled_store
    drv = _load("served_capability_judge_native_fix3", "scripts/served_capability_judge.py")
    store = tmp_path / "controls" / "synthtarget" / "cells" / "constitution_lora__controls__t0__a0__a1b2c3d4__0f1e2d3c"
    store.mkdir(parents=True)
    (store / "manifest.json").write_text(json.dumps({"resolved_config": {"gen": {"do_sample": True, "max_new_tokens": 64}}}))
    (store / "gen_stats.json").write_text("{}")
    ids = [f"xstest-{i}" for i in range(6)]
    _write_sampled_store(store, "gens_xstest_v2", ids, cap=64, at_cap_ids=set())                 # the writer's shape: header do_sample True + sidecar
    (store / "private" / "gens_xstest_v2.meta.jsonl").unlink()                                   # ... and the sidecar is gone
    root, (r_xs,) = _registry(tmp_path, monkeypatch, [("k_xs", "xstest", str(store), 6, "none", "none", None)])
    monkeypatch.setattr(drv, "Cell", lambda target, d, rows, controls: types.SimpleNamespace(
        target=target, cell_dir=d, name=d.name, controls=list(controls), judge_model="judge-model-x", gen_stats={}))
    DMm = drv._degeneration_measures()
    monkeypatch.setattr(DMm, "target_tokenizer", lambda mid, roster=None: _WhitespaceTok())
    ns = argparse.Namespace(registry_root=None, judge=None, session="s-A", campaign=None, org="lowprio", match="", worklist="registry",
                            run_id=None, model_id=None, store_root_map=None)
    with pytest.raises(SystemExit) as ei:
        drv.registry_worklist_cells(ns, {}, ["xstest"])
    assert "refusing before any claim" in str(ei.value) and "sidecar" in str(ei.value) and "nothing claimed" in str(ei.value), str(ei.value)
    assert _judgment_records(H.records_dir(root)) == []                                           # zero filings
    conn = _conn(root)
    assert [r["run_id"] for r in P.pending(conn)] == [r_xs]                                       # still pending, never claimed
    assert conn.execute("SELECT COUNT(*) FROM judgment").fetchone()[0] == 0
    conn.close()
