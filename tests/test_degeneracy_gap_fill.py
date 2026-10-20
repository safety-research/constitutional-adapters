"""scripts/sdfbase/full22/degeneracy_gap_fill.py — the registry-driven store walker for the union classifier.

Pins the PURE parts (no tokenizer, no transformers, no registry): the cell-id -> store-directory spellings, the store
resolution rules per store_uri shape (grades_sampled sibling, pooled glob, grade-lane private trees + cluster fallbacks,
transcripts_results store vs root, the relu_paired relative glob), the multi-turn grades-root reader, the pooled = sum
cross-check, the run_degeneration record shape the registry owner ingests, and the out-of-scope list — so absence from the
output is always a named status, never a silent skip. Synthetic paths and ids only."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "scripts/sdfbase/full22/degeneracy_gap_fill.py"


@pytest.fixture(scope="module")
def m():
    spec = importlib.util.spec_from_file_location("degeneracy_gap_fill_under_test", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cell_dir_candidates_strip_the_lane_prefix_once(m):
    assert m.cell_dir_candidates("none") == ["none"]
    assert m.cell_dir_candidates("constv2-mix-full__lora_r4_ce_minus_ctl") == ["constv2-mix-full__lora_r4_ce_minus_ctl", "lora_r4_ce_minus_ctl"]
    assert m.cell_dir_candidates("d0p75_eb64s2__constv2-mix-full__psr_ce_all_minus_ctl_distilled") == [
        "d0p75_eb64s2__constv2-mix-full__psr_ce_all_minus_ctl_distilled", "constv2-mix-full__psr_ce_all_minus_ctl_distilled"]


def test_pick_cell_dir_prefers_exact_then_stripped_then_single_agreeing(m, tmp_path):
    fam = tmp_path / "grok420"
    (fam / "constv2-mix-full__lora_r4_ce").mkdir(parents=True)
    d, rule = m.pick_cell_dir(fam, "constv2-mix-full__lora_r4_ce", None)
    assert d.name == "constv2-mix-full__lora_r4_ce" and rule == "exact"
    d, rule = m.pick_cell_dir(fam, "paired_eb8__constv2-mix-full__lora_r4_ce", None)
    assert d.name == "constv2-mix-full__lora_r4_ce" and rule == "prefix_stripped"
    # one cell dir + one aggregate cell that agree -> taken and SAID; disagreeing or absent aggregate -> unresolved
    d, rule = m.pick_cell_dir(fam, "something_else", ["constv2-mix-full__lora_r4_ce"])
    assert d.name == "constv2-mix-full__lora_r4_ce" and rule == "single_cell_agrees_with_aggregate"
    assert m.pick_cell_dir(fam, "something_else", ["other"]) == (None, "unresolved")
    assert m.pick_cell_dir(fam, "something_else", None) == (None, "unresolved")
    (fam / "none").mkdir()
    assert m.pick_cell_dir(fam, "something_else", ["constv2-mix-full__lora_r4_ce"]) == (None, "unresolved")   # two dirs: no guess


def test_static_roots_grades_sampled_sibling_and_pooled_glob(m, tmp_path):
    root = tmp_path / "full22" / "sampled"
    (root / "grades_sampled" / "sdfbase_full22_x_d0p1_sampled_t0").mkdir(parents=True)
    agg = root / "grades_sampled" / "sdfbase_full22_x_d0p1_sampled_t0" / "two_si_pasttense_aggregate.json"
    agg.write_text("{}")
    cands, rule = m.static_roots({"eval_id": "pasttense", "store_uri": str(agg)}, "grading_host")
    assert rule == "grades_sampled_sibling" and cands == [("transcripts_sampled", root / "transcripts_sampled" / "sdfbase_full22_x_d0p1_sampled_t0", None)]
    # the bare grades dir spells the same root
    cands2, _ = m.static_roots({"eval_id": "pasttense", "store_uri": str(agg.parent)}, "grading_host")
    assert cands2 == cands
    # a pooled `t*` glob resolves to every matching TRIAL directory under transcripts_sampled, labelled per trial
    for t in ("t0", "t1"):
        (root / "transcripts_sampled" / f"sdfbase_full22_x_sampled_{t}_cap8192").mkdir(parents=True)
    cands3, rule3 = m.static_roots({"eval_id": "grok420", "store_uri": str(root / "grades_sampled" / "sdfbase_full22_x_sampled_t*_cap8192")}, "grading_host")
    assert rule3 == "pooled_glob_under_transcripts_sampled"
    assert [c[0] for c in cands3] == ["trial:sdfbase_full22_x_sampled_t0_cap8192", "trial:sdfbase_full22_x_sampled_t1_cap8192"]
    cands4, rule4 = m.static_roots({"eval_id": "grok420", "store_uri": str(root / "grades_sampled" / "sdfbase_full22_absent_t*")}, "grading_host")
    assert cands4 == [] and "matched nothing" in rule4


def test_static_roots_transcripts_results_store_vs_root_and_cluster_tag(m):
    ch = m.CLUSTER_EXP["cluster_b"]
    store = ch / "sdfbase_full22_native/transcripts_results/sdfbase_full22_q_s2_sampled_t0/codeattack/none"
    cands, rule = m.static_roots({"eval_id": "codeattack", "store_uri": str(store)}, "grading_host")
    assert rule == "transcripts_results_store" and cands == [("transcripts_results:store", store.parent.parent, "cluster_b")]
    root = ch / "sdfbase_full22_native/transcripts_results/sdfbase_full22_q_s2_sampled_t0_cap8192"
    cands, rule = m.static_roots({"eval_id": "codeattack", "store_uri": str(root)}, "grading_host")
    assert rule == "transcripts_results_root" and cands == [("transcripts_results:root", root, "cluster_b")]
    # a grading host transcripts_results tree (seedrep_battery) carries no cluster tag
    local = Path.home() / "x/exp/transcripts_results/sdfbase_full22_q_s1_sampled_t0"
    cands, _ = m.static_roots({"eval_id": "renellm", "store_uri": str(local)}, "grading_host")
    assert cands == [("transcripts_results:root", local, None)]
    assert m.reachable("cluster_b", "grading_host") is False and m.reachable("cluster_b", "cluster_b") is True and m.reachable(None, "grading_host") is True


def test_static_roots_grade_lane_private_trees_then_clusters(m, monkeypatch, tmp_path):
    monkeypatch.setattr(m, "GRADE", tmp_path / "grade")
    agg = tmp_path / "grade/arditi_strength/grades/arditi_strength_g_ar_r0p1_sampled_t0/two_si_grok420_aggregate.json"
    agg.parent.mkdir(parents=True); agg.write_text("{}")
    cands, rule = m.static_roots({"eval_id": "grok420", "store_uri": str(agg)}, "grading_host")
    assert rule == "grade_lane_private_or_cluster"
    labels = [c[0] for c in cands]
    assert labels == ["private", "private_ch", "static_bu", "static_ch", "cluster_a:transcripts_results", "cluster_b:transcripts_results", "cluster_b:private"]
    assert cands[4][1] == m.CLUSTER_EXP["cluster_a"] / "arditi_strength/transcripts_results/arditi_strength_g_ar_r0p1_sampled_t0" and cands[4][2] == "cluster_a"
    assert cands[5][1] == m.CLUSTER_EXP["cluster_b"] / "arditi_strength/transcripts_results/arditi_strength_g_ar_r0p1_sampled_t0" and cands[5][2] == "cluster_b"
    assert cands[6][2] == "cluster_b"
    # both clusters are candidates, so a grading host leg must name BOTH as possible hosts (the first alone sent three Arditi stores to the wrong leg)
    assert {c[2] for c in cands if c[2]} == {"cluster_a", "cluster_b"}


def test_static_roots_relu_paired_relative_glob(m, monkeypatch, tmp_path):
    monkeypatch.setattr(m, "RS", tmp_path / "full22")
    for t in ("t0", "t1"):
        (tmp_path / f"full22/relu_paired/sampled/transcripts_sampled/sdfbase_full22_q_paired_d1_s1_sampled_{t}").mkdir(parents=True)
    cands, rule = m.static_roots({"eval_id": "renellm", "store_uri": "sdfbase_full22_q_paired_d1_s1_sampled_t*"}, "grading_host")
    assert rule == "relu_paired_pooled_glob" and len(cands) == 2 and all(c[2] is None for c in cands)


def test_mt_grades_from_aggregate_file_view_dir_and_root(m, tmp_path):
    root = tmp_path / "grades_anchor4096_g_it"
    (root / "g_it").mkdir(parents=True)
    agg = root / "g_it" / "two_si_pair_aggregate.json"
    agg.write_text(json.dumps({"private_run": "sdfbase_full22_g_it", "cells": {"none": {}}}))
    assert m.mt_grades({"eval_id": "pair", "store_uri": str(agg), "view": "g_it"}) == (root, agg, "g_it")
    assert m.mt_grades({"eval_id": "pair", "store_uri": str(root / "g_it"), "view": "g_it"}) == (root, agg, "g_it")          # the view dir
    assert m.mt_grades({"eval_id": "pair", "store_uri": str(root), "view": "g_it"}) == (root, agg, "g_it")                   # the root
    assert m.mt_grades({"eval_id": "pair", "store_uri": str(root), "view": "other"}) == (root, agg, "g_it")                  # one aggregate: taken
    (root / "h").mkdir(); (root / "h" / "two_si_pair_aggregate.json").write_text("{}")
    assert m.mt_grades({"eval_id": "pair", "store_uri": str(root), "view": "other"}) == (root, None, None)                  # two, none matches: no guess
    assert m.mt_grades({"eval_id": "pair", "store_uri": str(root), "view": "h_it"}) == (root, root / "h" / "two_si_pair_aggregate.json", "h")
    assert m.mt_grades({"eval_id": "pair", "store_uri": str(tmp_path / "transfer_stats_full22_halfdose.json"), "view": "v"}) == (None, None, None)
    assert m.mt_grades({"eval_id": "pair", "store_uri": "relative", "view": "v"}) == (None, None, None)


def test_mt_transcript_roots_sibling_seedrep_and_clusters(m):
    mtu = m.load_mt_runner(REPO)
    roots = m.mt_transcript_roots(Path("/x/full22/grades_obj4096_g_it"), mtu, "grading_host")
    labels = [r[0] for r in roots]
    assert labels[:2] == ["sibling:transcripts_obj4096_g_it", "sibling:transcripts_obj4096_g"]
    assert roots[0][1] == Path("/x/full22/transcripts_obj4096_g_it") and roots[0][2] is None
    assert any(l == "cluster_a:sdfbase_full22_native" and c == "cluster_a" for l, _, c in roots)
    assert any(l == "cluster_b:sdfbase_full22" and c == "cluster_b" for l, _, c in roots)
    seed = m.mt_transcript_roots(Path("/x/full22/seedrep/grades_native_mt"), mtu, "grading_host")
    assert ("seedrep:transcripts_native_mt", Path("/x/full22/seedrep/transcripts_native_mt"), None) in seed


def _measured_cell(n=10, deg=3, cls=2, rep=1, retok=2, fr=1):
    share = lambda k: {"share": k / n, "n": k, "denominator": n}  # noqa: E731
    return {"run_id": "sdfbase_full22:pasttense:aaaa:bbbb", "measured_at": "2026-01-01T00:00:00+00:00", "status": "measured",
            "denominator_unit": "draws", "n_rows": n, "n_goals": 5, "pooling": {"rows_are": "x", "cap": 8192, "n_stores": 1},
            "detectors": {"classifier": share(cls), "rep_ratio_loop": share(rep), "zlib_collapse": share(0), "contentless": share(0),
                          "truncation_only": {"k": 1, "n": n, "p": 1 / n, "rule": "r"}, "loop_without_cap": {"k": 2, "n": n, "p": 2 / n, "rule": "r"},
                          "at_cap": {"canonical": "at_cap_retokenized", "at_cap_retokenized": {**share(retok), "defined": True, "cap": 8192, "fraction": 0.98},
                                     "at_cap_finish_reason": {**share(fr), "defined": True, "n_rows_with_finish_reason": n},
                                     "cross_tab": {"both": 1}, "disagreements": {"n": 0}}},
            "degenerate_share": {**share(deg), "by_reason": {"classifier": 2, "at_cap": 1}}}


def test_records_shape_every_declared_measure_once_with_counts(m):
    cell = _measured_cell()
    rows = m.records_for(cell, "degeneracy_gap_fill/1@deadbeef", "out.json ›cells[x]")
    keys = [(r["measure"], r["scope"]) for r in rows]
    assert keys == [(a, b) for a, b, _ in m.MEASURES] and len(set(keys)) == len(keys)
    by = {r["measure"]: r for r in rows}
    u = by["union5"]
    assert u["measurement_state"] == "measured" and (u["numerator"], u["denominator"], u["denominator_unit"]) == (3, 10, "draws") and u["rate"] == 0.3
    assert u["aggregation"] == "union" and u["extra"]["by_reason"] == {"classifier": 2, "at_cap": 1} and u["judgment_id"] == ""
    assert u["definition_version"] == "degeneracy_gap_fill/1@deadbeef" and u["computed_by"] == "producer" and u["computed_utc"] == cell["measured_at"]
    assert by["at_token_cap_retokenized"]["numerator"] == 2 and by["at_token_cap_finish_reason"]["numerator"] == 1
    assert by["truncation_only"]["numerator"] == 1 and by["loop_without_cap"]["numerator"] == 2 and by["classifier"]["numerator"] == 2
    for r in rows:
        assert r["pooling_json"]["n_rows"] == 10 and r["threshold_params_json"] == m.THRESHOLDS.get(r["measure"], {})


def test_records_undefined_at_cap_is_not_measured_never_zero(m):
    cell = _measured_cell()
    cell["detectors"]["at_cap"]["at_cap_finish_reason"] = {"share": None, "n": None, "denominator": 10, "defined": False, "n_rows_with_finish_reason": 0}
    cell["detectors"]["at_cap"]["at_cap_retokenized"] = {"share": None, "n": None, "denominator": 10, "defined": False}
    by = {r["measure"]: r for r in m.records_for(cell, "d", "s")}
    for k in ("at_token_cap_finish_reason", "at_token_cap_retokenized"):
        assert by[k]["measurement_state"] == "not_measured" and by[k]["numerator"] is None and by[k]["rate"] is None and by[k]["notes"]


def test_sum_measures_adds_per_trial_counts(m):
    a, b = _measured_cell(n=10, deg=3, cls=2, rep=1), _measured_cell(n=7, deg=1, cls=0, rep=1)
    assert m.sum_measures([a, b]) == {"n_rows": 17, "degenerate": 4, "classifier": 2, "rep_ratio_loop": 2, "zlib_collapse": 0, "contentless": 0}


def test_out_of_scope_evals_are_named_not_measured(m):
    for ev in ("mmlu", "strongreject_bon", "bfcl"):
        assert ev in m.OUT_OF_SCOPE and m.OUT_OF_SCOPE[ev]
    assert set(m.STATIC) == {"pasttense", "codeattack", "renellm", "grok420"} and set(m.MT) == {"crescendo", "pair", "actorattack"}
    families = set(m.STATIC) | set(m.MT) | set(m.OVERREFUSAL) | set(m.GSM8K)
    assert not families & set(m.OUT_OF_SCOPE)       # gsm8k moved from out-of-scope to a reader in pass 2


def test_hf_model_for_strips_the_it_suffix_and_refuses_unknown(m):
    models = {"gemma4e4b": {"model": "google/gemma-4-E4B-it", "revision": "ee0e"}, "llama8b": {"model": "NousResearch/Meta-Llama-3.1-8B-Instruct", "revision": None},
              "nemotron3nano4b": {"model": "nvidia/N", "revision": "dfaf"}}
    assert m.hf_model_for("gemma4e4b_it", models) == ("google/gemma-4-E4B-it", "ee0e")
    assert m.hf_model_for("nemotron3nano4b", models) == ("nvidia/N", "dfaf")
    assert m.hf_model_for("llama8b_it", models) == ("NousResearch/Meta-Llama-3.1-8B-Instruct", None)
    with pytest.raises(RuntimeError):
        m.hf_model_for("glm46vflash", models)


def test_load_preresolved_prefers_measured_and_carries_the_aggregate(m, tmp_path):
    f = tmp_path / "results.jsonl"
    rows = [{"run_id": "r1", "status": "remote", "host": "grading_host", "remote_host": "cluster_a", "aggregate": {"private_run": "sdfbase_full22_x", "cells": ["none"]}},
            {"run_id": "r2", "status": "pending", "host": "grading_host"}, {"run_id": "r2", "status": "measured", "host": "grading_host"}]
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    pre = m.load_preresolved(str(f))
    assert pre["r1"]["aggregate"]["private_run"] == "sdfbase_full22_x" and pre["r2"]["status"] == "measured"
    assert m.load_preresolved(None) == {} and m.load_preresolved("") == {}


def test_assemble_merges_hosts_and_names_unreachable(m, tmp_path, monkeypatch):
    out = tmp_path / "out"; out.mkdir()
    cell = _measured_cell(); cell.update({"host": "cluster_b", "eval_id": "pasttense", "model_id": "q", "lane": "seedrep_cluster", "cell": "none", "judgment_ids": [""]})
    dev = [{"run_id": cell["run_id"], "status": "remote", "host": "grading_host", "remote_host": "cluster_b", "note": "transcripts live on cluster_b",
            "eval_id": "pasttense", "model_id": "q", "lane": "seedrep_cluster", "cell": "none", "judgment_ids": [""]},
           {"run_id": "gone", "status": "remote", "host": "grading_host", "remote_host": "cluster_a", "note": "transcripts live on cluster_a",
            "eval_id": "grok420", "model_id": "q", "lane": "arditi_strength", "cell": "none", "judgment_ids": [""]},
           {"run_id": "oos", "status": "out_of_scope", "host": "grading_host", "note": "mmlu", "eval_id": "mmlu", "model_id": "q", "lane": "x", "cell": "none", "judgment_ids": [""]}]
    bu = [{"run_id": "gone", "status": "pending", "host": "cluster_a", "note": "absent", "eval_id": "grok420", "model_id": "q", "lane": "arditi_strength", "cell": "none", "judgment_ids": [""]}]
    (out / "results.jsonl").write_text("\n".join(json.dumps(r) for r in dev) + "\n")
    (out / "results.cluster_a.jsonl").write_text("\n".join(json.dumps(r) for r in bu) + "\n")
    (out / "results.cluster_b.jsonl").write_text(json.dumps(cell) + "\n")
    (out / "run_meta.grading_host.1.json").write_text(json.dumps({"host": "grading_host", "started_at": "t", "union": {"union_src": "s", "DEFINITION_SHA": "abc", "worktree_git_sha": "g", "files": {}}}))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps({"run_id": r, "model_id": "q", "eval_id": "e", "lane": "l", "cell": "c", "judgment_ids": [""], "metric_rows": [{}, {}]}) for r in (cell["run_id"], "gone", "oos", "never")) + "\n")
    class A: pass
    a = A(); a.out = str(out); a.manifest = str(manifest)
    assert m.cmd_assemble(a) == 0
    doc = json.loads((out / "degeneracy_gap_fill.json").read_text())
    cells = doc["cells"]
    assert cells[cell["run_id"]]["status"] == "measured" and cells[cell["run_id"]]["host"] == "cluster_b" and len(cells[cell["run_id"]]["attempts"]) == 2
    assert cells["gone"]["status"] == "unreachable" and "cluster_a: pending" in cells["gone"]["note"] and "grading_host: remote" in cells["gone"]["note"]
    assert cells["oos"]["status"] == "out_of_scope" and cells["never"]["status"] == "not_attempted"
    assert doc["definition_version"] == "degeneracy_gap_fill/1@abc"
    assert doc["coverage"]["runs_by_status"] == {"measured": 1, "unreachable": 1, "out_of_scope": 1, "not_attempted": 1}
    assert doc["coverage"]["metric_rows_by_status"]["measured"] == 2
    assert len(doc["records"]) == len(m.MEASURES) and all(r["run_id"] == cell["run_id"] for r in doc["records"])
    assert len(doc["results_files"]) == 3


def _registry_current(tmp_path, run_ids, rows=None):
    """A synthetic registry whose v_run_current says every named run is current (cmd_export re-reads currency at
    filing time, read-only) and whose `run` table carries each run's identity (the export's ONE snapshot compares store_uri,
    job, instrument, cap, trial, pass and decode with the manifest and the measurement) — `rows` = {run_id: {column: value}}, default the
    identity the export tests' manifests / cells state (store_uri /s/<rid>, instrument "inst", no cap / job). A test names its OWN registry."""
    import sqlite3
    db = tmp_path / "registry.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE v_run_current (run_id TEXT PRIMARY KEY, is_current INTEGER, why TEXT)")
    conn.executemany("INSERT INTO v_run_current VALUES (?, 1, 'current')", [(r,) for r in run_ids])
    conn.execute("CREATE TABLE run (run_id TEXT PRIMARY KEY, store_uri TEXT, cluster_job_id TEXT, instrument_id TEXT, token_cap_tokens INTEGER, reply_cap_tokens INTEGER, "
                 "trial_index INTEGER, pass_label TEXT, decode_source TEXT, extra_json TEXT)")
    for r in run_ids:
        row = {"store_uri": f"/s/{r}", "cluster_job_id": None, "instrument_id": "inst", "token_cap_tokens": None, "reply_cap_tokens": None, "trial_index": None, "pass_label": None, "decode_source": None}
        row.update((rows or {}).get(r) or {})
        conn.execute("INSERT INTO run (run_id, store_uri, cluster_job_id, instrument_id, token_cap_tokens, reply_cap_tokens, trial_index, pass_label, decode_source, extra_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (r, row["store_uri"], row["cluster_job_id"], row["instrument_id"], row["token_cap_tokens"], row["reply_cap_tokens"], row["trial_index"], row["pass_label"], row["decode_source"], "{}"))
    conn.commit(); conn.close()
    return db


def _export_ready(c, store_uri, *, mt=False):
    """degap-fix-5: an export row needs the cell's store_uri to match its manifest row and every store it read to carry the
    attested per-shard identity record (a multi-turn reading also its dated aggregate join) — the shape verify_store writes."""
    ident = {"attested": True, "response_shards": ["shard00"], "shards_with_served_target": ["shard00"], "checks": {"job": {"registry": None, "store": []}}}
    if mt:
        ident["aggregate_join"] = {"attested": True}
    c["store_uri"] = store_uri
    c["store"] = {"paths": [store_uri], "shards": {"shard00": "a"}, "identity": {"transcripts_results:store": ident}}
    return c


def test_export_shape_exclusion_comparison_and_versioning(m, tmp_path):
    out = tmp_path / "out"; out.mkdir()
    a_cell = _measured_cell(); a_cell.update({"host": "grading_host", "eval_id": "pasttense", "model_id": "q", "lane": "l", "cell": "none", "judgment_ids": [""], "reader": "r", "denominator_unit": "draws"})
    b_cell = _measured_cell(n=20, deg=5, cls=3, rep=2, retok=1, fr=2); b_cell.update({"run_id": "sdfbase_full22:pair:cccc:dddd", "host": "cluster_b", "eval_id": "pair", "model_id": "q", "lane": "l", "cell": "x", "judgment_ids": [""], "denominator_unit": "replies"})
    b_cell["degenerate_share"]["by_reason"] = {"classifier": 3, "at_cap": 1, "rep_ratio_loop+at_cap": 1}
    b_cell["detectors"]["at_cap"]["at_cap_finish_reason"] = {"share": None, "n": None, "denominator": 20, "defined": False, "n_rows_with_finish_reason": 7}
    b_cell = _export_ready(b_cell, f"/s/{b_cell['run_id']}", mt=True)          # degap-fix-5: identity attested per shard, aggregate join dated
    doc = {"definition_version": "degeneracy_gap_fill/1@deadbeef", "tool_sha256s": ["t"], "meta": {"runs": []},
           "cells": {a_cell["run_id"]: a_cell, b_cell["run_id"]: b_cell, "oos": {"status": "out_of_scope", "run_id": "oos"}}}
    (out / "degeneracy_gap_fill.json").write_text(json.dumps(doc))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps({"run_id": r, "store_uri": f"/s/{r}", "cell": c, "eval_id": e, "instrument_id": "inst"})
                                  for r, c, e in ((a_cell["run_id"], "none", "pasttense"), (b_cell["run_id"], "x", "pair"), ("oos", "c", "mmlu"))) + "\n")
    filed = tmp_path / "filed.tsv"
    filed.write_text("run_id\tjudgment_id\trate\tnumerator\tdenominator\tdenominator_unit\tdef\tsource_uri\n"
                     f"{a_cell['run_id']}\t\t0.3\t3\t10\tdraws\tolder_instrument\tsrc\n"
                     f"{a_cell['run_id']}\tj1\t0.4\t4\t10\tdraws\tr2 export\tsrc2\n")
    class A: pass
    a = A(); a.out = str(out); a.manifest = str(manifest); a.filed = str(filed); a.hosts = ""; a.batch = "gapfill_test"; a.dir = str(tmp_path / "by_run_id")
    a.registry = str(_registry_current(tmp_path, [a_cell["run_id"], b_cell["run_id"]]))        # degap-fix-5: currency re-read at filing time
    assert m.cmd_export(a) == 0
    p = tmp_path / "by_run_id" / "gapfill_test.v1.json"
    body = json.loads(p.read_text())
    assert body["schema"] == "degeneracy_union_by_run_id/1" and body["instrument_module"] == "sisteer.eval.degeneracy_union" and body["definition_sha"] == "deadbeef"
    assert body["self_check"]["status"] == "passed"
    # the already-filed run is compared (one identical, one different) and NOT re-filed
    fc = body["self_check"]["filed_comparison"]
    assert fc["n_runs_compared"] == 1 and fc["n_rows_compared"] == 2 and fc["n_identical"] == 1 and fc["n_different"] == 1
    assert fc["different"][0]["filed"]["k"] == 4 and fc["different"][0]["mine"]["k"] == 3 and fc["different"][0]["filed_definition"] == "r2 export"
    assert [x["run_id"] for x in body["excluded_already_filed"]] == [a_cell["run_id"]]
    assert [r["run_id"] for r in body["rows"]] == [b_cell["run_id"]]
    r = body["rows"][0]
    assert (r["store_uri"], r["campaign_cell_id_legacy"], r["eval_id"], r["instrument_id"]) == (f"/s/{b_cell['run_id']}", "x", "pair", "inst")   # copied from the registry row
    assert r["judgment_id"] == "" and r["unit"] == "replies" and r["n"] == 20 and r["definition_version"] == "degeneracy_union_by_run_id/1@deadbeef"
    assert set(r["measures"]) == set(m.EXPORT_MEASURES)
    assert r["measures"]["union5"] == {"k": 5, "n": 20, "p": 0.25} and r["by_reason"] == {"classifier": 3, "at_cap": 1, "rep_ratio_loop+at_cap": 1}
    assert r["measures"]["at_token_cap_finish_reason"]["state"] == "not_measured" and "7 of 20" in r["measures"]["at_token_cap_finish_reason"]["reason"]
    assert r["measures"]["at_token_cap_retokenized"] == {"k": 1, "n": 20, "p": 0.05} and r["measures"]["truncation_only"] == {"k": 1, "n": 20, "p": 0.05}
    # a second export never overwrites: it lands beside as v2; a host filter that matches nothing writes an empty batch
    assert m.cmd_export(a) == 0 and (tmp_path / "by_run_id" / "gapfill_test.v2.json").is_file() and p.is_file()
    a.hosts = "cluster_a"
    assert m.cmd_export(a) == 0 and json.loads((tmp_path / "by_run_id" / "gapfill_test.v3.json").read_text())["rows"] == []


def test_export_refuses_a_self_check_failure(m, tmp_path):
    out = tmp_path / "out"; out.mkdir()
    c = _measured_cell(); c.update({"host": "grading_host", "eval_id": "pasttense", "model_id": "q", "lane": "l", "cell": "none", "denominator_unit": "draws"})
    c["degenerate_share"]["by_reason"] = {"classifier": 1}          # 1 != union5.k == 3
    c = _export_ready(c, "/s")                                       # degap-fix-5: identity-ready, so the SELF-CHECK is what refuses
    (out / "degeneracy_gap_fill.json").write_text(json.dumps({"definition_version": "x@abc", "tool_sha256s": [], "meta": {"runs": []}, "cells": {c["run_id"]: c}}))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"run_id": c["run_id"], "store_uri": "/s", "cell": "none", "eval_id": "pasttense", "instrument_id": "i"}) + "\n")
    class A: pass
    a = A(); a.out = str(out); a.manifest = str(manifest); a.filed = ""; a.hosts = ""; a.batch = "b"; a.dir = str(tmp_path / "d")
    a.registry = str(_registry_current(tmp_path, [c["run_id"]], rows={c["run_id"]: {"store_uri": "/s", "instrument_id": "i"}}))   # degap-fix-6: the snapshot agrees, so the self-check is what refuses
    with pytest.raises(SystemExit):
        m.cmd_export(a)
    assert not list((tmp_path / "d").glob("*.json")) if (tmp_path / "d").exists() else True


def test_assemble_flags_two_disagreeing_readings_as_conflict(m, tmp_path):
    out = tmp_path / "out"; out.mkdir()
    a_cell = _measured_cell(); a_cell.update({"host": "grading_host", "eval_id": "pasttense", "model_id": "q", "lane": "l", "cell": "none", "judgment_ids": [""]})
    same = json.loads(json.dumps(a_cell)); same["host"] = "cluster_b"
    diff = json.loads(json.dumps(a_cell)); diff["host"] = "cluster_a"; diff["degenerate_share"]["n"] = 4
    (out / "results.jsonl").write_text(json.dumps(a_cell) + "\n")
    (out / "results.cluster_b.jsonl").write_text(json.dumps(same) + "\n")
    (out / "run_meta.grading_host.1.json").write_text(json.dumps({"host": "grading_host", "started_at": "t", "union": {"union_src": "s", "DEFINITION_SHA": "abc", "worktree_git_sha": "g", "files": {}}}))
    class A: pass
    a = A(); a.out = str(out); a.manifest = ""
    assert m.cmd_assemble(a) == 0
    doc = json.loads((out / "degeneracy_gap_fill.json").read_text())
    kept = doc["cells"][a_cell["run_id"]]
    assert kept["status"] == "measured" and {kept["host"], kept["also_measured_on"][0]["host"]} == {"grading_host", "cluster_b"}   # file order decides which is primary; both are kept
    (out / "results.cluster_a.jsonl").write_text(json.dumps(diff) + "\n")
    assert m.cmd_assemble(a) == 0
    doc = json.loads((out / "degeneracy_gap_fill.json").read_text())
    assert doc["cells"][a_cell["run_id"]]["status"] == "conflict" and doc["coverage"]["runs_by_status"] == {"conflict": 1}
    assert doc["records"] == []


# ---------------------------------------------------------------------------- pass 2: over-refusal + GSM8K readers

import hashlib as _hashlib


def _controls_store(tmp_path, control_stem, items, *, bad_sha_for=None):
    d = tmp_path / "cells" / "none__controls__t0__a0__2ff9cdff__deadbeef"
    (d / "private").mkdir(parents=True)
    gens = [json.dumps({"__spool_header__": {"gen_cfg": {}}})]
    meta = []
    for iid, text, cap, fr, regen in items:
        gens.append(json.dumps({"item_id": iid, "text": text}))
        sha = _hashlib.sha256(text.encode()).hexdigest() if iid != bad_sha_for else "00" * 32
        meta.append(json.dumps({"item_id": iid, "text_sha256": sha, "max_tokens": cap, "source_cap": cap, "finish_reason": fr, "at_cap": fr == "length",
                                "completion_tokens": 7, "regenerated": regen}))
    (d / "private" / f"{control_stem}.jsonl").write_text("\n".join(gens) + "\n")
    (d / "private" / f"{control_stem}.meta.jsonl").write_text("\n".join(meta) + "\n")
    return d


def test_cell_dir_spelling_turns_the_first_double_underscore_into_dashes(m):
    assert m.cell_dir_spelling("constv2-mix-full__lora_r4_ce_minus_ctl") == "constv2-mix-full--lora_r4_ce_minus_ctl"
    assert m.cell_dir_spelling("prompted__si_constitution") == "prompted--si_constitution" and m.cell_dir_spelling("none") == "none"


def test_read_controls_rows_joins_text_and_meta_with_per_row_caps_and_sha_identity(m, tmp_path):
    items = [("xst0000-a", "I can help with that.", 2048, "stop", True), ("xst0001-b", "no " * 300, 512, "length", False), ("xst0002-c", "sure", 512, "stop", False)]
    d = _controls_store(tmp_path, "gens_xstest_v2", items)
    rows, facts = m.read_controls_rows(d, "xstest", label="t0")
    assert [r["goal_id"] for _, r in rows] == ["xst0000-a", "xst0001-b", "xst0002-c"] and all(s == "t0" for s, _ in rows)
    assert [r["cap"] for _, r in rows] == [2048, 512, 512] and rows[1][1]["at_cap"] is True and rows[1][1]["finish_reason"] == "length"
    assert facts["caps"] == {"2048": 1, "512": 2} and facts["n_regenerated"] == 1 and facts["text_sha256_verified"] is True
    # the population filter keeps only the named items and SAYS how many it dropped
    rows2, facts2 = m.read_controls_rows(d, "xstest", item_filter={"xst0002-c"})
    assert [r["goal_id"] for _, r in rows2] == ["xst0002-c"] and facts2["n_dropped_by_item_filter"] == 2
    # a text that does not hash to the meta's text_sha256 refuses; a missing file refuses by name
    d2 = _controls_store(tmp_path / "bad", "gens_or_bench_hard_1k", [("orb0-x", "text", 512, "stop", False)], bad_sha_for="orb0-x")
    with pytest.raises(RuntimeError):
        m.read_controls_rows(d2, "or_bench_hard")
    with pytest.raises(FileNotFoundError):
        m.read_controls_rows(d2, "xstest")


def test_final_rung_pointer_and_trial_store_resolution(m, tmp_path):
    p, view, key = m.parse_final_rung_pointer("~/x/final_rung.json#targets.gemma4e4b_it.controls['xstest|constv2-mix-full__lora_r4_ce']")
    assert p.endswith("/x/final_rung.json") and view == "gemma4e4b_it" and key == "xstest|constv2-mix-full__lora_r4_ce"
    assert m.parse_final_rung_pointer("/y/final_rung.json#s1") == ("/y/final_rung.json", None, None)
    fr = {"targets": {
        "gemma4e4b_it": {"controls": {"xstest|constv2-mix-full__lora_r4_ce": {"status": "ok", "held": False, "pooled": {"final_caps": [8192]},
                                                                                "trials": {"0": {"final_cap": 8192, "store": "/s/controls_cap8192/gemma4e4b_it_t0/cells/c"},
                                                                                           "1": {"final_cap": 8192, "store": "/s/controls_cap8192/gemma4e4b_it_t1/cells/c"}}}}},
        "nemotron3nano4b_paired_d1_nemotron3nano4b_s1": {"controls": {"or_bench_hard_1k|constv2-mix-full__psr_ce_all_minus_ctl": {"status": "ok", "pooled": {"final_caps": [512, 2048]},
                                                                                "trials": {"0": {"final_cap": 512, "store": "/r/controls_cap512/x_t0/cells/c"}, "1": {"final_cap": 2048, "store": "/r/controls_cap2048/x_t1/cells/c"}}}}}}}
    stores, ent = m.final_rung_trial_stores(fr, control="xstest", cell="constv2-mix-full__lora_r4_ce", view="gemma4e4b_it", target_hint="gemma4e4b_it", key_hint="xstest|constv2-mix-full__lora_r4_ce")
    assert [(l, str(s), c) for l, s, c in stores] == [("trial:t0@cap8192", "/s/controls_cap8192/gemma4e4b_it_t0/cells/c", 8192), ("trial:t1@cap8192", "/s/controls_cap8192/gemma4e4b_it_t1/cells/c", 8192)]
    assert ent["final_caps"] == [8192]
    # no fragment: the target that ENDS WITH the registry view, the cell under its prefix-stripped spelling
    stores, ent = m.final_rung_trial_stores(fr, control="or_bench_hard", cell="paired_eb8__constv2-mix-full__psr_ce_all_minus_ctl", view="paired_d1_nemotron3nano4b_s1", target_hint=None, key_hint=None)
    assert [c for _, _, c in stores] == [512, 2048] and ent["target"] == "nemotron3nano4b_paired_d1_nemotron3nano4b_s1"
    with pytest.raises(RuntimeError):
        m.final_rung_trial_stores(fr, control="xstest", cell="constv2-mix-full__psr_ce_all_minus_ctl", view="paired_d1_nemotron3nano4b_s1", target_hint=None, key_hint=None)   # no xstest entry there


def test_alphasteer_benign_rows_assign_redrawn_vs_copied_forward_caps(m, tmp_path):
    d = tmp_path / "lam1"; d.mkdir()
    gens = []
    for i in range(5):
        gens.append({"leg": "none", "slice": "benign", "rung_i": 1, "prompt_id": f"xst{i}", "response": f"r{i}", "finish_reason": "stop", "n_completion_tokens": 3})
    for i in (1, 3):
        gens.append({"leg": "none", "slice": "benign_cap512", "rung_i": 0, "prompt_id": f"xst{i}", "response": "trunc", "finish_reason": "length", "n_completion_tokens": 512})
    gens.append({"leg": "alphasteer", "slice": "benign", "rung_i": 1, "prompt_id": "xst0", "response": "steered", "finish_reason": "stop", "n_completion_tokens": 3})
    gens.append({"leg": "none", "slice": "sampler_probe", "rung_i": 1, "prompt_id": "p", "response": "x", "finish_reason": "stop", "n_completion_tokens": 1})
    (d / "generations.jsonl").write_text("\n".join(json.dumps(g) for g in gens) + "\n")
    (d / "rung_record.json").write_text(json.dumps({"cap_ladder": {"caps": [512, 2048, 8192], "by_leg": {"none": {"2048": {"n_responses": 5, "n_copied_forward": 3}}}}}))
    rows, facts = m.read_alphasteer_benign_rows(d, cell="none")
    assert [(r["goal_id"], r["cap"]) for _, r in rows] == [("xst0", 512), ("xst1", 2048), ("xst2", 512), ("xst3", 2048), ("xst4", 512)]
    assert facts["exit_cap"] == 2048 and facts["n_redrawn_at_exit_cap"] == 2 and facts["n_copied_forward_at_cap"] == 3
    (d / "rung_record.json").write_text(json.dumps({"cap_ladder": {"caps": [512, 2048, 8192], "by_leg": {"none": {"2048": {"n_responses": 5, "n_copied_forward": 4}}}}}))
    with pytest.raises(RuntimeError):
        m.read_alphasteer_benign_rows(d, cell="none")     # the ladder's copied-forward count disagrees with the rows


def test_merge_gsm8k_rows_replaces_the_rung_subset_and_maps_stop_reasons(m):
    src = {"g1": {"text": "a", "stop_reason": "stop", "output_tokens": 10}, "g2": {"text": "b" * 50, "stop_reason": "max_tokens", "output_tokens": 8192},
           "g3": {"text": "c", "stop_reason": "max_tokens", "output_tokens": 8192}}
    rung = {"g2": {"text": "bb", "stop_reason": "stop", "output_tokens": 9000}}
    rows = m.merge_gsm8k_rows(src, rung, {"g2"}, source_cap=8192, rung_cap=16384, stop_map={"max_tokens": "length", "stop": "stop"}, label="x")
    assert [(d, r["goal_id"], r["cap"], r["finish_reason"], r.get("at_cap")) for d, r in rows] == [("source", "g1", 8192, "stop", False), ("rung", "g2", 16384, "stop", False), ("source", "g3", 8192, "length", True)]
    assert rows[1][1]["response"] == "bb" and rows[1][1]["completion_tokens"] == 9000
    with pytest.raises(RuntimeError):
        m.merge_gsm8k_rows(src, {}, {"g2"}, source_cap=8192, rung_cap=16384, stop_map={}, label="x")     # a subset item without a rung draw
    rows0 = m.merge_gsm8k_rows(src, None, None, source_cap=8192, rung_cap=None, stop_map={"max_tokens": "length"}, label="x")
    assert all(d == "source" and r["cap"] == 8192 for d, r in rows0)


def _part(n, deg, cls, rep, zl, ctl, rt, fr, trunc, lwc, by_reason, goals):
    sh = lambda k: {"share": k / n, "n": k, "denominator": n}  # noqa: E731
    return {"n_rows": n, "n_goals": goals, "definition_sha": "abc",
            "detectors": {"classifier": {**sh(cls), "by_reason": {"tail_loop": cls}}, "rep_ratio_loop": sh(rep), "zlib_collapse": sh(zl), "contentless": sh(ctl),
                          "truncation_only": {"k": trunc, "n": n, "p": trunc / n, "rule": "r"}, "loop_without_cap": {"k": lwc, "n": n, "p": lwc / n, "rule": "r"},
                          "at_cap": {"canonical": "at_cap_retokenized", "at_cap_retokenized": {**sh(rt), "defined": True, "cap": 512, "fraction": 0.98, "detector": "d"},
                                     "at_cap_finish_reason": {**sh(fr), "defined": True, "n_rows_with_finish_reason": n},
                                     "cross_tab": {"both": rt, "finish_reason_only": 0, "retokenized_only": 0, "neither": n - rt, "n_rows_compared": n},
                                     "disagreements": {"n": 0, "denominator": n, "examples": []}}},
            "degenerate_share": {**sh(deg), "by_reason": by_reason, "detector": "OR"}, "producer_vs_retokenized_tokens": {"n_compared": n, "equal": n - 1, "max_abs_diff": 2, "mean_abs_diff": 0.5}}


def test_sum_measure_parts_adds_counts_recomputes_shares_and_merges_reasons(m):
    a = _part(10, 4, 2, 1, 1, 0, 3, 3, 1, 1, {"at_cap": 1, "classifier+at_cap": 2, "rep_ratio_loop": 1}, 10)
    b = _part(30, 3, 3, 0, 0, 0, 0, 0, 0, 3, {"classifier": 3}, 30)
    s = m.sum_measure_parts({512: a, 2048: b})
    assert s["n_rows"] == 40 and s["n_goals"] == 40 and s["by_cap"] == {"512": 10, "2048": 30}
    assert s["degenerate_share"] == {"share": 7 / 40, "n": 7, "denominator": 40, "by_reason": {"at_cap": 1, "classifier": 3, "classifier+at_cap": 2, "rep_ratio_loop": 1}, "detector": "OR"}
    assert s["detectors"]["classifier"]["n"] == 5 and s["detectors"]["classifier"]["by_reason"] == {"tail_loop": 5}
    assert s["detectors"]["truncation_only"] == {"k": 1, "n": 40, "p": 1 / 40, "rule": "r"} and s["detectors"]["loop_without_cap"]["k"] == 4
    ac = s["detectors"]["at_cap"]
    assert ac["at_cap_retokenized"]["n"] == 3 and ac["at_cap_retokenized"]["denominator"] == 40 and ac["at_cap_retokenized"]["cap"] == [512, 2048] and ac["at_cap_retokenized"]["defined"]
    assert ac["at_cap_finish_reason"]["n"] == 3 and ac["cross_tab"]["n_rows_compared"] == 40 and ac["cross_tab"]["neither"] == 37
    assert s["producer_vs_retokenized_tokens"] == {"n_compared": 40, "equal": 38, "max_abs_diff": 2, "mean_abs_diff": 0.5}
    # an estimand undefined in one part is undefined in the sum
    b["detectors"]["at_cap"]["at_cap_finish_reason"]["defined"] = False
    assert m.sum_measure_parts({512: a, 2048: b})["detectors"]["at_cap"]["at_cap_finish_reason"]["defined"] is False


def test_xstest_item_filter_by_registry_population(m, tmp_path):
    assert m.xstest_item_filter({"n_items_generated": 450}, [], {"a"}, {})[0] is None
    with pytest.raises(RuntimeError, match="undecidable"):                                  # degen-fix-3: an unknown count never defaults to every row
        m.xstest_item_filter({"n_items_generated": None}, [], {"a"}, {})
    d = tmp_path / "cell"; d.mkdir()
    (d / "judge_items.jsonl").write_text("\n".join(json.dumps({"control": "xstest", "item_id": f"xst{i}", "subset": "safe" if i < 250 else "unsafe"}) for i in range(450)) + "\n")
    ids, prov = m.xstest_item_filter({"n_items_generated": 250}, [d], None, {})
    assert len(ids) == 250 and "xst0" in ids and "xst300" not in ids and prov["source"].endswith("judge_items.jsonl")
    ids2, prov2 = m.xstest_item_filter({"n_items_generated": 249}, [tmp_path / "nojudge"], {"c1", "c2"}, {"source": "canon"})
    assert ids2 == {"c1", "c2"} and prov2["source"] == "canon"
    with pytest.raises(RuntimeError):
        m.xstest_item_filter({"n_items_generated": 250}, [tmp_path / "nojudge"], None, {})


def test_manifest_query_covers_the_over_refusal_headline_and_gsm8k_is_in_scope(m):
    assert "'over_refusal_graded'" in m.GAP_SQL and "'asr'" in m.GAP_SQL and "'acc'" in m.GAP_SQL
    assert "gsm8k" not in m.OUT_OF_SCOPE and "mmlu" in m.OUT_OF_SCOPE and "strongreject_bon" in m.OUT_OF_SCOPE
    assert set(m.OVERREFUSAL) == {"xstest", "or_bench_hard"} and m.CONTROL_FILE["or_bench_hard"] == "gens_or_bench_hard_1k" and m.CONTROL_KEY["or_bench_hard"] == "or_bench_hard_1k"
