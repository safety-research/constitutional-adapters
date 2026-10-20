"""reproduce/lib/coverage_probe.py — the gate's per-run coverage reader over the REAL producer layouts,
each laid down synthetically the way its producer writes it: jailbreak_run.py's gated shard stores
(`shardNN/progress.jsonl`, the LAST record per goal, `error:*` / `refused:*` outcomes), bon_served_run.py's append-only
`bon_responses.jsonl` (+ `bon_errors.jsonl`), bon_climb.sbatch's FLAT `out/<cell>/shard<k>.jsonl` beside `plan/climb_manifest.jsonl`
(bon_climb_run.py's `i % nshards == k` slicing of the cell's cap-coherent rows), the tinycap CellStore `cells/<key>/rows.jsonl` +
`manifest.json` (`mmlu_item_set.item_ids` -> the item-set sha256), the misalign runner's `<cell>/responses.jsonl`. Every id here is
synthetic; nothing names a real host or user."""
from __future__ import annotations

import hashlib
import json
import subprocess

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reproduce" / "lib"))
import coverage_probe as CP  # noqa: E402

PROBE = Path(__file__).resolve().parents[1] / "reproduce" / "lib" / "coverage_probe.py"


def _jsonl(p: Path, rows) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_jailbreak_cellstore_last_progress_record_per_goal_errors_apart(tmp_path):
    store = tmp_path / "transcripts_results" / "sdfbase_full22_m_u0_sampled_t0" / "pasttense" / "none"
    _jsonl(store / "shard00" / "progress.jsonl", [
        {"goal_id": "g1", "outcome": "success", "n_target_forwards": 2},
        {"goal_id": "g2", "outcome": "error:PromptOverWindow"},              # errored ...
        {"goal_id": "g2", "outcome": "capped"},                              # ... then retried on resume: the LAST record wins
        {"goal_id": "g3", "outcome": "refused:all_turns_expected_unavailable"},   # terminal, never an error
        {"goal_id": "g4", "outcome": "error:ActorDiscoveryError"},
    ])
    (store / "shard00" / "progress.jsonl").write_text((store / "shard00" / "progress.jsonl").read_text() + '{"goal_id": "torn", "outc')   # a torn tail
    _jsonl(store / "shard00" / "responses.jsonl", [{"goal_id": g, "sample_idx": i} for g in ("g1", "g2", "g3", "g4") for i in range(20)])   # many lines: never counted
    doc = CP.probe_jailbreak_cellstore(str(store), ["shard00", "shard01"])
    assert doc["exists"] and doc["shards"]["shard00"]["present"] and not doc["shards"]["shard01"]["present"]
    s0 = doc["shards"]["shard00"]
    assert s0["completed"] == ["g1", "g2", "g3"] and s0["errored"] == ["g4"] and s0["refused"] == ["g3"]
    assert "no " in doc["shards"]["shard01"]["notes"][0]
    # a run whose recorded store is ONE shard directory: the parent is the store, said so
    doc2 = CP.probe_jailbreak_cellstore(str(store / "shard00"), ["shard00"])
    assert doc2["path"] == str(store) and doc2["shards"]["shard00"]["completed"] == ["g1", "g2", "g3"] and "shard directory" in doc2["notes"][0]


def test_bon_store512_needs_n_max_draws_per_goal_at_the_trial(tmp_path):
    store = tmp_path / "sdfbase_full22_m_u0" / "none"
    rows = [{"goal_id": "g1", "sample_idx": i, "trial": 0} for i in range(75)] + [{"goal_id": "g2", "sample_idx": i, "trial": 0} for i in range(74)]
    rows += [{"goal_id": "g2", "sample_idx": 74, "trial": 1}]                    # another trial's draw never completes trial 0
    rows += [{"goal_id": "g1", "sample_idx": 3, "trial": 0}]                     # a duplicate draw (resume) counts once
    _jsonl(store / "bon_responses.jsonl", rows)
    _jsonl(store / "bon_errors.jsonl", [{"goal_id": "g3", "sample_idx": 0, "trial": 0, "error": "x"}])
    doc = CP.probe_bon_store512(str(store), n_max=75)
    blk = doc["shards"]["shard00"]
    assert blk["present"] and blk["completed"] == ["g1"] and blk["errored"] == ["g3"] and "fewer than n_max=75" in blk["notes"][0]


def _climb_world(tmp_path: Path):
    root = tmp_path / "bon_climb" / "m_u0"
    man = []
    for g, samples in (("gA", (0, 1)), ("gB", (0,)), ("gC", (0, 1, 2)), ("gD", (0,))):
        man += [{"cell_id": "none", "goal_id": g, "sample_idx": s, "trial": 0, "class": "cap_coherent"} for s in samples]
    man.append({"cell_id": "none", "goal_id": "gE", "sample_idx": 0, "trial": 0, "class": "terminated"})     # not climbed
    man.append({"cell_id": "other", "goal_id": "gA", "sample_idx": 0, "trial": 0})                          # another cell
    _jsonl(root / "plan" / "climb_manifest.jsonl", man)
    return root


def test_bon_climb_flat_expected_from_the_manifest_slice_written_draws_per_goal(tmp_path):
    root = _climb_world(tmp_path)
    rows = CP.climb_expected_rows(str(root / "plan" / "climb_manifest.jsonl"), "none")
    assert [(r["goal_id"], r["sample_idx"]) for r in rows] == [("gA", 0), ("gA", 1), ("gB", 0), ("gC", 0), ("gC", 1), ("gC", 2), ("gD", 0)]
    # shard 0 of 2 = rows 0,2,4,6 -> (gA,0) (gB,0) (gC,1) (gD,0); shard 1 = (gA,1) (gC,0) (gC,2)
    _jsonl(root / "out" / "none" / "shard0.jsonl", [{"goal_id": "gA", "sample_idx": 0, "response": "…"}, {"goal_id": "gB", "sample_idx": 0, "response": "…"},
                                                    {"goal_id": "gC", "sample_idx": 1, "response": "…"}])
    _jsonl(root / "out" / "none" / "shard0.errors.jsonl", [{"goal_id": "gD", "sample_idx": 0, "error": "PromptOverWindow"}])
    d0 = CP.probe_bon_climb_flat(str(root), "none", 0, 2)["shards"]["shard00"]
    assert d0["expected"] == ["gA", "gB", "gC", "gD"] and d0["completed"] == ["gA", "gB", "gC"] and d0["errored"] == ["gD"]
    _jsonl(root / "out" / "none" / "shard1.jsonl", [{"goal_id": "gA", "sample_idx": 1, "response": "…"}, {"goal_id": "gC", "sample_idx": 0, "response": "…"}])
    d1 = CP.probe_bon_climb_flat(str(root), "none", 1, 2)["shards"]["shard01"]
    assert d1["expected"] == ["gA", "gC"] and d1["completed"] == ["gA"]                    # gC's shard-1 draws: (0) written, (2) missing
    # no manifest: the expected set is unknown, never guessed
    d = CP.probe_bon_climb_flat(str(tmp_path / "nowhere"), "none", 0, 2)["shards"]["shard00"]
    assert d["expected"] is None and "no climb manifest" in d["notes"][0]


def test_capability_cellstore_unique_items_of_the_task_and_the_item_set_sha(tmp_path):
    store = tmp_path / "cells" / "none__mmlu"
    ids = [f"mmlu-{i:04d}" for i in range(6)]
    rows = [{"task": "s2parity/mmlu_letter", "item_id": i, "metric": "acc", "value": 1.0} for i in ids[:5]]
    rows += [{"task": "s2parity/mmlu_letter", "item_id": ids[5], "metric": "topk_undeterminable", "value": 1.0}]     # a failed read is still a completed item
    rows += [{"task": "s2parity/mmlu_letter", "item_id": ids[0], "metric": "acc", "value": 1.0}]                     # duplicate row
    rows += [{"task": "s2parity/mmlu_letter", "item_id": "__aggregate__", "metric": "acc_aggregate", "value": 0.8}]
    rows += [{"task": "judged/xstest_v2", "item_id": "x1", "metric": "acc", "value": 1.0}]                           # another task in the same store
    _jsonl(store / "rows.jsonl", rows)
    (store / "manifest.json").write_text(json.dumps({"resolved_config": {"mmlu": {"item_set": {"item_set": "full", "n": 6, "seed": 0, "of": 6}}},
                                                     "mmlu_item_set": {"item_ids": ids}}))
    doc = CP.probe_capability_cellstore(str(store), "s2parity/mmlu_letter")
    it = doc["items"]
    assert it["present"] and it["completed"] == ids and it["n_rows"] == 9 and it["tasks_seen"] == {"s2parity/mmlu_letter": 8, "judged/xstest_v2": 1}
    assert it["items_sha256"] == hashlib.sha256("\n".join(ids).encode()).hexdigest() and it["n_attested_items"] == 6
    assert set(it["metrics_seen"]) == {"acc", "topk_undeterminable"}
    doc_x = CP.probe_capability_cellstore(str(store), "judged/xstest_v2")
    assert doc_x["items"]["completed"] == ["x1"]


def test_misalign_responses_items_with_n_samples_of_the_behavior(tmp_path):
    root = tmp_path / "exp" / "misalign" / "m_u0" / "agentic"
    rows = [{"item_id": "leak-1", "sample_idx": i, "behavior": "leaking", "response": "…"} for i in range(10)]
    rows += [{"item_id": "leak-2", "sample_idx": i, "behavior": "leaking", "response": "…"} for i in range(9)]
    rows += [{"item_id": "murder-1", "sample_idx": i, "behavior": "murder", "response": "…"} for i in range(10)]
    _jsonl(root / "none" / "responses.jsonl", rows)
    d = CP.probe_misalign_responses(str(root), "none", "leaking", 10)["shards"]["shard00"]
    assert d["present"] and d["completed"] == ["leak-1"] and "fewer than n_samples=10" in d["notes"][0]
    assert CP.probe_misalign_responses(str(root), "none", "murder", 10)["shards"]["shard00"]["completed"] == ["murder-1"]


def test_cli_prints_one_json_document_and_refuses_a_climb_without_a_cell(tmp_path):
    root = _climb_world(tmp_path)
    r = subprocess.run([sys.executable, str(PROBE), "probe", "--layout", "bon_climb_flat", "--path", str(root), "--cell", "none", "--shard-index", "1", "--nshards", "2"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout)
    assert doc["layout"] == "bon_climb_flat" and doc["shards"]["shard01"]["expected"] == ["gA", "gC"]
    bad = subprocess.run([sys.executable, str(PROBE), "probe", "--layout", "bon_climb_flat", "--path", str(root)], capture_output=True, text=True)
    assert bad.returncode == 2 and "needs --cell" in bad.stderr


# --------------------------------------------------------------------------------------------------- the Petri per-cell store
def _petri_store(tmp_path: Path, *, behaviors=("cyber_bio_chem_capabilities", "self_preservation"), n=3, epochs=1) -> Path:
    """`<audit root>/cells/<run key>/` the way petri_runner writes it: rows.jsonl (task petri/<b>, item_id = scenario [#e<k>], the three
    required metrics, extra_json {scenario_id, epoch, dimension}), petri_incomplete.json (unresolved: []), results.json (n_items /
    n_scenarios per behavior)."""
    cell = tmp_path / "petri_full22_m_u0" / "cells" / "petricell-a1"
    rows = []
    for b in behaviors:
        for i in range(n):
            sid = f"{b[:5]}_scenario_{i}"
            for ep in range(1, epochs + 1):
                iid = sid if epochs == 1 else f"{sid}#e{ep}"
                for metric in (b, "eval_awareness", "scenario_realism"):
                    rows.append({"task": f"petri/{b}", "item_id": iid, "metric": metric, "value": 5.0, "tier": -1, "domain": "petri_behavior",
                                 "extra_json": json.dumps({"scenario_id": sid, "epoch": ep, "dimension": metric, "auditor_chain": "sonnet_only"})})
    _jsonl(cell / "rows.jsonl", rows)
    (cell / "petri_incomplete.json").write_text(json.dumps({"version": 1, "cell_id": "none", "petri_run_key": "petricell-a1", "unresolved": []}))
    (cell / "results.json").write_text(json.dumps({"cell_id": "none", "petri_run_key": "petricell-a1", "rows_emitted": len(rows),
                                                   "behaviors": {b: {"n_items": n * epochs, "n_scenarios": n, "incomplete": False} for b in behaviors}}))
    return cell


def test_petri_cellstore_complete_scenarios_by_behavior_the_ledger_and_terminal_refusals(tmp_path):
    cell = _petri_store(tmp_path)
    doc = CP.probe_petri_cellstore(str(cell))
    blk = doc["shards"]["shard00"]
    assert doc["exists"] and blk["present"] and doc["results_json_present"] and doc["unresolved_n"] == 0
    assert blk["completed"] == sorted(f"{b}/{b[:5]}_scenario_{i}" for b in ("cyber_bio_chem_capabilities", "self_preservation") for i in range(3))
    assert blk["errored"] == [] and blk["refused"] == [] and doc["behaviors_seen"] == {"cyber_bio_chem_capabilities": 9, "self_preservation": 9}
    # one behavior only (a petri_<behavior> eval)
    one = CP.probe_petri_cellstore(str(cell), behavior="self_preservation")["shards"]["shard00"]
    assert one["completed"] == [f"self_preservation/self__scenario_{i}" for i in range(3)]
    # a scenario the runner's failure ledger holds unresolved is NOT complete however many rows it has; a scenario short of one required
    # dimension is not complete; a TERMINAL auditor refusal settles its scenario (covered, reported apart)
    rows = [json.loads(l) for l in (cell / "rows.jsonl").read_text().splitlines()]
    rows = [r for r in rows if not (r["item_id"] == "cyber_scenario_1" and r["metric"] == "scenario_realism")]        # short one dimension
    _jsonl(cell / "rows.jsonl", rows)
    (cell / "petri_incomplete.json").write_text(json.dumps({"version": 1, "unresolved": [
        {"behavior": "self_preservation", "scenario_id": "self__scenario_2", "epoch": 1, "dimension": "self_preservation", "reason": "no result", "retries": 0}]}))
    (cell / "petri_auditor_refused.json").write_text(json.dumps({"version": 1, "auditor_refused": [
        {"behavior": "cyber_bio_chem_capabilities", "scenario_id": "cyber_scenario_9", "epoch": 1, "dimensions": ["cyber_bio_chem_capabilities", "eval_awareness", "scenario_realism"]}]}))
    (cell / "results.json").unlink()
    blk = CP.probe_petri_cellstore(str(cell))["shards"]["shard00"]
    assert "cyber_bio_chem_capabilities/cyber_scenario_1" not in blk["completed"] and "self_preservation/self__scenario_2" not in blk["completed"]
    assert blk["errored"] == ["self_preservation/self__scenario_2"]                                     # the ledger's open item
    assert "cyber_bio_chem_capabilities/cyber_scenario_9" in blk["completed"] and blk["refused"] == ["cyber_bio_chem_capabilities/cyber_scenario_9"]
    assert any("not complete" in n_ and "cyber_scenario_1" in n_ for n_ in blk["notes"]), blk["notes"]
    doc = CP.probe_petri_cellstore(str(cell))
    assert not doc["results_json_present"] and doc["unresolved_n"] == 1 and any("no results.json" in n_ for n_ in doc["notes"])


def test_petri_cellstore_epochs_from_results_json_and_the_cli_expands_a_home_relative_store(tmp_path, monkeypatch):
    cell = _petri_store(tmp_path, behaviors=("defer_to_users",), n=2, epochs=2)
    blk = CP.probe_petri_cellstore(str(cell))["shards"]["shard00"]
    assert blk["completed"] == ["defer_to_users/defer_scenario_0", "defer_to_users/defer_scenario_1"]
    rows = [json.loads(l) for l in (cell / "rows.jsonl").read_text().splitlines()]
    _jsonl(cell / "rows.jsonl", [r for r in rows if not (r["item_id"] == "defer_scenario_0#e2")])      # epoch 2 of scenario 0 never filed
    blk = CP.probe_petri_cellstore(str(cell))["shards"]["shard00"]
    assert blk["completed"] == ["defer_to_users/defer_scenario_1"] and any("e2:" in n_ for n_ in blk["notes"])
    # the adapter records `~/.local/state/petri/.../cells/<key>`: the CLI expands `~` (the gate runs the probe on the grading host)
    monkeypatch.setenv("HOME", str(tmp_path))
    rel = "~/" + str(cell.relative_to(tmp_path))
    r = subprocess.run([sys.executable, str(PROBE), "probe", "--layout", "petri_cellstore", "--path", rel, "--behavior", "defer_to_users"], capture_output=True, text=True, env={**dict(__import__("os").environ), "HOME": str(tmp_path)})
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout)
    assert doc["layout"] == "petri_cellstore" and doc["path"] == str(cell) and doc["exists"] and doc["shards"]["shard00"]["completed"] == ["defer_to_users/defer_scenario_1"]
    # an absent store: exists False, present False, said
    doc = CP.probe_petri_cellstore(str(tmp_path / "nowhere"))
    assert not doc["exists"] and not doc["shards"]["shard00"]["present"] and any("no " in n_ for n_ in doc["shards"]["shard00"]["notes"])


def test_bon_climb_flat_reads_every_shard_asked_in_one_document_and_names_them(tmp_path):
    """The climb array's four tasks register ONE run per cell, so the gate reads every shard of the cell
    in one call — `--shard-index` repeated (or `0,1`): one document, `shards.<shardNN>` per index byte-equal to the single-shard probe's, and
    `shard_indices` naming what was asked (a probe before this read the LAST flag alone and named none). One int, or no flag: as before."""
    root = _climb_world(tmp_path)
    d0 = CP.probe_bon_climb_flat(str(root), "none", 0, 2); d1 = CP.probe_bon_climb_flat(str(root), "none", 1, 2)
    assert set(d0["shards"]) == {"shard00"} and d0["shard_indices"] == [0] and set(d1["shards"]) == {"shard01"}
    both = CP.probe_bon_climb_flat(str(root), "none", [0, 1], 2)
    assert both["shard_indices"] == [0, 1] and both["shards"] == {**d0["shards"], **d1["shards"]} and both["shards"]["shard01"]["expected"] == ["gA", "gC"]
    assert CP.climb_shard_indices(["0,1", "1", " 0 "]) == [0, 1] and CP.climb_shard_indices(3) == [3] and CP.climb_shard_indices(["3"]) == [3]
    for flags in (["--shard-index", "0", "--shard-index", "1"], ["--shard-index", "0,1"]):
        r = subprocess.run([sys.executable, str(PROBE), "probe", "--layout", "bon_climb_flat", "--path", str(root), "--cell", "none", *flags, "--nshards", "2"],
                           capture_output=True, text=True)
        assert r.returncode == 0 and json.loads(r.stdout) == json.loads(json.dumps(both)), (flags, r.stderr)
    r = subprocess.run([sys.executable, str(PROBE), "probe", "--layout", "bon_climb_flat", "--path", str(root), "--cell", "none", "--shard-index", "1", "--nshards", "2"],
                       capture_output=True, text=True)
    assert json.loads(r.stdout)["shards"] == d1["shards"] and json.loads(r.stdout)["shard_indices"] == [1]
    r = subprocess.run([sys.executable, str(PROBE), "probe", "--layout", "bon_climb_flat", "--path", str(root), "--cell", "none", "--nshards", "2"], capture_output=True, text=True)
    assert json.loads(r.stdout)["shards"] == d0["shards"]                                           # no flag: shard 0, as before
    # no manifest: every asked shard says so, the expected set unknown for each
    d = CP.probe_bon_climb_flat(str(tmp_path / "nowhere"), "none", [0, 1], 2)
    assert set(d["shards"]) == {"shard00", "shard01"} and all(b["expected"] is None and "no climb manifest" in b["notes"][0] for b in d["shards"].values())


# --------------------------------------------------------------------------- gate-fixes: the sampled-controls spool, the agentic cell dir
sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import sampled_controls_store  # noqa: E402

XST = {f"xst-{i:04d}": f"text {i}" for i in range(450)}          # the producer generates the 450 superset; the frozen item set scores the 250 safe half
ORB = {f"orb-{i:04d}": f"orb text {i}" for i in range(1319)}


def test_done_predicates_cover_every_layout_and_the_probe_annotations_resolve():
    import typing
    assert set(CP.DONE_PREDICATES) == set(CP.LAYOUTS) and all(v.startswith(f"{k}:v1 — ") for k, v in CP.DONE_PREDICATES.items())
    typing.get_type_hints(CP.petri_scenario_of)                     # `Tuple` is imported (the lazy annotations hid its absence before this lane)


def test_sampled_controls_spool_reads_as_the_producers_complete_store_reader_does(tmp_path):
    cell = sampled_controls_store(tmp_path / "cells" / "none__controls__t0__a0__1fe64a93__9dc1ed90", {"xstest": XST, "or_bench_hard_1k": ORB},
                                  at_cap={"xstest": ["xst-0003", "xst-0010"], "or_bench_hard_1k": ["orb-0001"]})
    assert not (cell / "rows.jsonl").exists() and sorted(p.name for p in (cell / "private").iterdir()) == \
        ["gens_or_bench_hard_1k.caps.jsonl", "gens_or_bench_hard_1k.jsonl", "gens_or_bench_hard_1k.meta.jsonl", "gens_xstest_v2.caps.jsonl", "gens_xstest_v2.jsonl", "gens_xstest_v2.meta.jsonl"]
    doc = CP.probe_capability_cellstore(str(cell), "judged/xstest_v2")
    it = doc["items"]
    assert doc["layout"] == "capability_cellstore" and doc["exists"] and it["present"] and it["refusal"] is None and it["source"] == "private/gens_xstest_v2.jsonl"
    assert it["completed"] == sorted(XST) and it["n_unique"] == 450 and it["n_rows"] == 450 and it["tasks_seen"] == {"judged/xstest_v2": 450}
    assert it["items_sha256"] is None and it["items_sha256_sorted"] == hashlib.sha256("\n".join(sorted(XST)).encode()).hexdigest() \
        and it["items_sha256_spool_order"] == hashlib.sha256("\n".join(XST).encode()).hexdigest() and "not an attestation" in it["items_sha256_source"]
    assert it["gen_stats"] == {"control": "xstest", "n_total": 450, "n_at_cap": 2, "cap": 2048, "decode": "present"} and it["meta_bound"] == 450
    assert any("read the sampled-controls generation spool" in n for n in doc["notes"])
    orb = CP.probe_capability_cellstore(str(cell), "judged/or_bench_hard_1k")["items"]
    assert orb["present"] and len(orb["completed"]) == 1319 and orb["gen_stats"]["control"] == "or_bench_hard_1k"
    # the CLI prints the same document (the shipped probe under a bare python3)
    r = subprocess.run([sys.executable, str(PROBE), "probe", "--layout", "capability_cellstore", "--path", str(cell), "--task", "judged/xstest_v2"], capture_output=True, text=True)
    assert r.returncode == 0 and json.loads(r.stdout)["items"]["completed"] == sorted(XST)
    # a 250-row spool of UNRELATED ids stamped complete at n_total 250: covered BY COUNT (the frozen identity records no ids sha; the record says so)
    other = sampled_controls_store(tmp_path / "cells" / "other", {"xstest": {f"foreign-{i}": "t" for i in range(250)}})
    it2 = CP.probe_capability_cellstore(str(other), "judged/xstest_v2")["items"]
    assert it2["present"] and len(it2["completed"]) == 250 and it2["items_sha256"] is None


def _mutate_spool(cell: Path, name: str, fn):
    p = cell / "private" / name; lines = p.read_text().splitlines(); p.write_text(fn(lines))


@pytest.mark.parametrize("case", ["no_gen_stats", "other_spool", "torn_tail", "no_header", "dup_differs", "count_short", "no_meta", "meta_unbound", "greedy_entry",
                                  "at_cap_not_bool", "tally_disagrees", "two_spools", "n_total_not_int"])
def test_sampled_controls_spool_refuses_exactly_what_the_producer_refuses(tmp_path, case):
    """Precondition for precondition with scripts/sdfbase/full22/sampled/at_cap_controls.py read_control (+ the closer's two-spools check): each
    failure names itself in `items.refusal`, `present` is false, and admission refuses the run by that text."""
    cell = sampled_controls_store(tmp_path / "cells" / "c", {"xstest": dict(list(XST.items())[:12])}, at_cap={"xstest": ["xst-0003"]})
    gs_p = cell / "gen_stats.json"; gs = json.loads(gs_p.read_text())
    if case == "no_gen_stats":
        gs_p.unlink(); want = "no gen_stats.json — the producer stamps it only after every control is complete"
    elif case == "other_spool":
        gs["xstest"]["spool"] = str(cell / "private" / "gens_xstest_v3.jsonl"); gs_p.write_text(json.dumps(gs)); want = "no control entry names spool gens_xstest_v2.jsonl"
    elif case == "torn_tail":
        _mutate_spool(cell, "gens_xstest_v2.jsonl", lambda ls: "\n".join(ls) + '\n{"item_id": "xst-9999", "te'); want = "corrupt or torn line 14; not a complete store"
    elif case == "no_header":
        _mutate_spool(cell, "gens_xstest_v2.jsonl", lambda ls: "\n".join(ls[1:]) + "\n"); want = "first line is not a spool header"
    elif case == "dup_differs":
        _mutate_spool(cell, "gens_xstest_v2.jsonl", lambda ls: "\n".join(ls) + "\n" + json.dumps({"item_id": "xst-0001", "text": "ANOTHER"}) + "\n"); want = "item 'xst-0001' spooled twice with different text"
    elif case == "count_short":
        _mutate_spool(cell, "gens_xstest_v2.jsonl", lambda ls: "\n".join(ls[:-1]) + "\n"); want = "11 spooled items != gen_stats n_total=12; incomplete or inconsistent store"
    elif case == "no_meta":
        (cell / "private" / "gens_xstest_v2.meta.jsonl").unlink(); want = "no meta sidecar — a greedy store, or one written before the sidecar existed"
    elif case == "meta_unbound":
        _mutate_spool(cell, "gens_xstest_v2.meta.jsonl", lambda ls: "\n".join(json.dumps({**json.loads(l), "text_sha256": "0" * 64}) if '"xst-0005"' in l else l for l in ls) + "\n")
        want = "1 spooled item(s) have no bound meta row (first: ['xst-0005'])"
    elif case == "greedy_entry":
        for k in ("decode", "n_at_cap", "at_cap_item_ids"):
            gs["xstest"].pop(k)
        gs_p.write_text(json.dumps(gs)); want = "carries no sampled decode record (decode/n_at_cap/at_cap_item_ids) — a GREEDY store has no per-item at-cap status and never reads as 0"
    elif case == "at_cap_not_bool":
        _mutate_spool(cell, "gens_xstest_v2.meta.jsonl", lambda ls: "\n".join(json.dumps({**json.loads(l), "at_cap": "yes"}) if '"xst-0002"' in l else l for l in ls) + "\n")
        want = "item 'xst-0002' at_cap='yes' is not a bool"
    elif case == "tally_disagrees":
        gs["xstest"]["n_at_cap"] = 2; gs["xstest"]["at_cap_item_ids"] = ["xst-0003", "xst-0004"]; gs_p.write_text(json.dumps(gs))
        want = "meta sidecar says 1 at cap but gen_stats says n_at_cap=2 (2 ids); inconsistent store"
    elif case == "two_spools":
        (cell / "private" / "gens_xstest_v2.jsonl").rename(cell / "private" / "gens_xstest_v2b.jsonl")     # the entry names a spool the cell no longer holds first
        gs["xstest"]["spool"] = str(cell / "private" / "gens_xstest_v2.jsonl"); gs_p.write_text(json.dumps(gs))
        (cell / "private" / "gens_xstest_v2.jsonl").write_text((cell / "private" / "gens_xstest_v2b.jsonl").read_text())
        (cell / "private" / "gens_xstest_v2b.jsonl").unlink(); (cell / "private" / "gens_xstest_v2 .jsonl").write_text("x")
        want = "gen_stats.json names spool 'gens_xstest_v2.jsonl' for xstest, the cell's generation spool is 'gens_xstest_v2 .jsonl' — two spools"
    elif case == "n_total_not_int":
        gs["xstest"]["n_total"] = "twelve"; gs_p.write_text(json.dumps(gs)); want = "n_total='twelve' is not an integer"
    doc = CP.probe_capability_cellstore(str(cell), "judged/xstest_v2")
    it = doc["items"]
    assert not it["present"] and it["refusal"] and want in it["refusal"], (case, it["refusal"])
    assert it["completed"] == []
    # admission refuses the run by the probe's words
    sys.path.insert(0, str(PROBE.parent))
    import admission as A
    frozen = {"coverage": {"layout": "capability_cellstore", "n_items": 250, "task": "judged/xstest_v2", "item_set_id": "xstest_safe_250", "items_sha256": None}}
    probs = A.check_run_coverage(frozen, {"probe": doc, "probe_error": None}, None, "xstest")
    assert len(probs) == 1 and probs[0].startswith("the item artifact is not a complete store: ") and want in probs[0], probs


def test_sampled_controls_admission_covers_by_count_and_a_plain_cellstore_without_any_artifact_is_named(tmp_path):
    sys.path.insert(0, str(PROBE.parent))
    import admission as A
    cell = sampled_controls_store(tmp_path / "cells" / "c", {"xstest": XST}, at_cap={"xstest": ["xst-0003"]})
    doc = CP.probe_capability_cellstore(str(cell), "judged/xstest_v2")
    frozen = {"coverage": {"layout": "capability_cellstore", "n_items": 250, "task": "judged/xstest_v2", "item_set_id": "xstest_safe_250", "items_sha256": None}}
    assert A.check_run_coverage(frozen, {"probe": doc, "probe_error": None}, None, "xstest") == []           # 450 >= 250 by count (the freeze records no ids sha)
    frozen_sha = {**frozen, "coverage": {**frozen["coverage"], "items_sha256": "1" * 64}}
    probs = A.check_run_coverage(frozen_sha, {"probe": doc, "probe_error": None}, None, "xstest")
    assert len(probs) == 1 and "the store attests no item-id list" in probs[0]                               # a future freeze that records a sha is not satisfied by a spool
    empty = tmp_path / "cells" / "empty"; empty.mkdir(parents=True)
    doc2 = CP.probe_capability_cellstore(str(empty), "judged/xstest_v2")
    assert not doc2["items"]["present"] and doc2["items"]["refusal"] is None and any("no " in n and "rows.jsonl" in n for n in doc2["notes"])
    probs2 = A.check_run_coverage(frozen, {"probe": doc2, "probe_error": None}, None, "xstest")
    assert len(probs2) == 1 and probs2[0].startswith("no item artifact (rows.jsonl, or the sampled controls' private/gens_<task>.jsonl") and str(empty) in probs2[0]


def test_misalign_responses_reads_the_recorded_cell_directory_and_the_legacy_root(tmp_path):
    """The closer records `<root>/agentic/<harness cell>` as the run's store (si_<slug>, prompted__safety, none): the probe reads `<path>/responses.jsonl`
    + `meta.json` there; the legacy call (the root + --cell) still reads `<root>/<cell>/responses.jsonl` and says so."""
    root = tmp_path / "exp" / "misalign" / "m_battery" / "agentic"
    cell = root / "si_constv2ctl2_full__psr_ce_all"
    rows = [{"item_id": f"leak-{i}", "sample_idx": k, "behavior": "leaking", "response": "…"} for i in range(3) for k in range(10)]
    rows += [{"item_id": "murder-1", "sample_idx": k, "behavior": "murder", "response": "…"} for k in range(10)]
    _jsonl(cell / "responses.jsonl", rows)
    (cell / "meta.json").write_text(json.dumps({"cell_id": "si_constv2ctl2_full__psr_ce_all", "items": 4, "n_samples": 10, "present": 40, "finished": 1760000000.0, "item_ids": ["leak-0", "leak-1", "leak-2", "murder-1"]}))
    d = CP.probe_misalign_responses(str(cell), "constv2ctl2-full__psr_ce_all", "leaking", 10)
    assert d["exists"] and d["path"] == str(cell) and d["shards"]["shard00"]["present"] and d["shards"]["shard00"]["completed"] == ["leak-0", "leak-1", "leak-2"] and d["notes"] == []
    assert d["meta"] == {"finished": 1760000000.0, "items": 4, "n_samples": 10, "present": 40, "cell_id": "si_constv2ctl2_full__psr_ce_all"}
    legacy = CP.probe_misalign_responses(str(root), "si_constv2ctl2_full__psr_ce_all", "murder", 10)
    assert legacy["shards"]["shard00"]["completed"] == ["murder-1"] and any("legacy layout" in n for n in legacy["notes"])
    absent = CP.probe_misalign_responses(str(root / "none"), "none", "murder", 10)
    assert not absent["shards"]["shard00"]["present"] and absent["meta"] is None
