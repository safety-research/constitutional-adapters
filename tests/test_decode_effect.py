"""full22 SAMPLED re-run — decode_effect.py: the paired statistics on hand-computed 2x2 tables, the pairing refusals (per trial, missing
metadata), the greedy-controls fallback to the campaign's original directory, the four-way controls decomposition, the XSTest safe-subset
scoping, and the decomposition's final-rung rule (final_rung.final_rung_store, reused, cross-checked against final_rung.json).

Synthetic only (tmp_path, placeholder ids); no store text leaves, no API, no cluster.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
SAMPLED = REPO / "scripts" / "sdfbase" / "full22" / "sampled"
for _p in (REPO / "scripts", REPO / "src", SAMPLED):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
_spec = importlib.util.spec_from_file_location("decode_effect", SAMPLED / "decode_effect.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)
cgs = D.cgs

JUDGE = {"judge_model": "claude-sonnet-5", "rubric_sha": "288e8c08b0c7bbfa", "protocol_sha": "b8d28441cf71caec"}


# ------------------------------------------------------------------------------------------------- McNemar on a hand-computed 2x2
def test_mcnemar_exact_hand_computed_and_log10():
    # discordants b = 5 (sampled-only), c = 1 (greedy-only): n = 6, two-sided exact p = 2 * [C(6,0) + C(6,1)] / 2^6 = 14/64
    m = D.mcnemar_exact_ext(5, 1)
    assert m["b"] == 5 and m["c"] == 1 and math.isclose(m["p_value"], 14 / 64) and math.isclose(m["log10_p"], math.log10(14 / 64))
    assert D.mcnemar_exact_ext(3, 3)["p_value"] == 1.0 and D.mcnemar_exact_ext(3, 3)["log10_p"] == 0.0
    assert D.mcnemar_exact_ext(0, 0)["p_value"] is None
    big = D.mcnemar_exact_ext(494, 2624)          # a float underflow is never printed as 0: the log10 is exact in integer arithmetic
    assert big["p_value"] == 0.0 and big["log10_p"] < -300
    assert D.fmt_p(big).startswith("<1e-300 (log10 p≈-")
    assert D.fmt_p({"p_value": 0.2}) == "0.200" and D.fmt_p({"p_value": 3e-4}) == "3.0e-04" and D.fmt_p(None) == "—"


def test_paired_binary_hand_computed_table():
    ids = [f"g{i}" for i in range(10)]
    x = {i: i in {"g0", "g1", "g2", "g3", "g6"} for i in ids}
    y = {i: i in {"g0", "g1", "g2", "g3", "g7", "g8", "g9"} for i in ids}
    r = D.paired_binary(x, y, "hand", n_boot=200, seed=1)
    assert r["n"] == 10 and r["table"] == {"both": 4, "neither": 2, "a_only": 1, "b_only": 3}
    assert r["a"]["k"] == 5 and r["b"]["k"] == 7 and math.isclose(r["delta_pp"], 20.0)
    assert r["discordant"] == {"n": 4, "a_only": 1, "b_only": 3}
    assert math.isclose(r["mcnemar_exact"]["p_value"], 10 / 16)
    se = math.sqrt(0.036)
    assert math.isclose(r["wald95_pp"]["lo_pp"], 100 * (0.2 - 1.959963984540054 * se))
    assert r["paired_bootstrap95_pp"]["n_paired"] == 10 and r["paired_bootstrap95_pp"]["delta_pp"] == pytest.approx(20.0)


def test_pooled_contrast_sums_discordants_signs_and_clusters_by_id():
    ids = [f"g{i}" for i in range(6)]
    up = D.paired_binary({i: False for i in ids}, {i: i in {"g0", "g1"} for i in ids}, "up")
    down = D.paired_binary({i: i in {"g0"} for i in ids}, {i: False for i in ids}, "down")
    flat = D.paired_binary({i: i == "g0" for i in ids}, {i: i == "g0" for i in ids}, "flat")
    triples = [(i, False, i in {"g0", "g1"}) for i in ids] + [(i, i == "g0", False) for i in ids] + [(i, i == "g0", i == "g0") for i in ids]
    P = D.pooled_contrast([up, down, flat], triples, n_boot=500, seed=2, unit="goal id")
    assert P["n_cells"] == 3 and P["n_pairs"] == 18 and P["discordant"] == {"n": 3, "a_only": 1, "b_only": 2}
    assert P["sign_summary"]["cells_up"] == 1 and P["sign_summary"]["cells_down"] == 1 and P["sign_summary"]["cells_zero"] == 1
    assert math.isclose(P["delta_pp"], 100 * (2 - 1) / 18)
    assert P["clustered_boot95_detail"]["n_clusters"] == 6 and P["clustered_boot95_detail"]["n_pairs"] == 18      # 6 ids recur in 3 cells: 6 clusters, not 18 pairs
    lo, hi = P["clustered_boot95_pp"]
    assert lo <= P["delta_pp"] <= hi and "naive" in P["mcnemar_exact"]["role"]


# ------------------------------------------------------------------------------------------------- the pairing refuses mismatched item sets / instruments
def test_pairing_refuses_mismatched_item_sets():
    x = {"g0": True, "g1": False, "g2": True}
    with pytest.raises(D.PairingMismatch) as exc:
        D.paired_binary(x, {"g0": True, "g1": False}, "lane/fam/cell")
    assert "item sets differ" in str(exc.value) and "g2" in str(exc.value)
    with pytest.raises(D.PairingMismatch):
        D.paired_binary(x, {**x, "g9": True}, "lane/fam/cell")
    assert issubclass(D.PairingMismatch, cgs.Refuse)


def test_pairing_refuses_a_different_or_missing_grader_or_judge():
    g = {"judge_model": "claude-haiku-4-5", "rubric": "strongreject_v1", "threshold": 0.5}
    with pytest.raises(D.PairingMismatch):
        D.check_same_instrument(g, {**g, "rubric": "strongreject_v2"}, "x")
    with pytest.raises(D.PairingMismatch):
        D.check_same_instrument(g, {**g, "threshold": 0.75}, "x")
    with pytest.raises(D.PairingMismatch) as exc:
        D.check_same_instrument(g, {**g, "rubric": None}, "x")
    assert "missing" in str(exc.value)
    D.check_same_instrument(g, dict(g), "x")
    with pytest.raises(D.PairingMismatch):
        D.check_same_judge(JUDGE, {**JUDGE, "protocol_sha": "0000000000000000"}, "x")
    with pytest.raises(D.PairingMismatch):
        D.check_same_judge(JUDGE, {**JUDGE, "judge_model": None}, "x")
    D.check_same_judge(JUDGE, dict(JUDGE), "x")


def _a(**kw):
    return SimpleNamespace(bootstrap=kw.pop("bootstrap", 0), rep_ratio=lambda t: 0.0, _goals=None, _caches={}, **kw)


def test_static_cell_records_a_mismatch_as_unpaired_and_validates_every_trial(tmp_path, monkeypatch):
    greedy = {"status": "ok", "cap_tokens": 4096, "per_item": {"g0": [2, 0, 0, 0, 0.9], "g1": [2, 0, 0, 0, 0.1]},
              "provenance": {"aggregate": {"grader": {"judge_model": "claude-haiku-4-5", "rubric": "strongreject_v1"}}, "flags": {"path": str(tmp_path / "none.flags.jsonl")}},
              "asr": {"threshold": 0.5}}
    ok_grader = {"judge_model": "claude-haiku-4-5", "rubric": "strongreject_v1", "threshold": 0.5, "n_errored": 0}
    monkeypatch.setattr(D, "sampled_static_grades", lambda dest, tag, tr, fam, cell: ({"g0": 0.9, "g7": 0.2}, {"aggregate": {"path": "x"}}, ok_grader))
    rec = D.static_cell(_a(), tmp_path, "qwen35_4b", "pasttense", "none", greedy, [0, 1], [0.25, 0.5, 0.75], False, None, 1, [], {})
    assert rec["status"] == "unpaired" and "goal sets differ" in rec["reason"]
    def grades(dest, tag, tr, fam, cell):     # a grader mismatch on the SECOND trial only is still refused (validated per trial)
        return {"g0": 0.9, "g1": 0.2}, {"aggregate": {"path": "x"}}, (ok_grader if tr == 0 else {**ok_grader, "rubric": "strongreject_v2"})
    monkeypatch.setattr(D, "sampled_static_grades", grades)
    rec = D.static_cell(_a(), tmp_path, "qwen35_4b", "pasttense", "none", greedy, [0, 1], [0.25, 0.5, 0.75], False, None, 1, [], {})
    assert rec["status"] == "unpaired" and "t1" in rec["reason"] and "rubric differs" in rec["reason"]


# ------------------------------------------------------------------------------------------------- static_cell, positive path
def _sampled_store(root: Path, tag: str, tr: int, fam: str, cell: str, rows: list[dict]) -> None:
    d = root / "transcripts_sampled" / f"sdfbase_full22_{tag}_sampled_t{tr}" / fam / cell / "shard00"
    d.mkdir(parents=True)
    (d / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def _srow(gid: str, idx: int, tr: int, text: str, fr: str, ct: int) -> dict:
    return {"goal_id": gid, "sample_idx": idx, "trial": tr, "family": "pasttense", "response": text,
            "decode": {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "cap": 4096, "arm": "vendor_recommended", "seed_base": 20260909, "trial": tr},
            "sampling_seed": 11 + idx, "finish_reason": fr, "completion_tokens": ct, "prompt_tokens": 20, "max_tokens": 4096, "at_cap": fr == "length"}


def test_static_cell_positive_path_pairs_goals_and_measures_looped_goals(tmp_path, monkeypatch):
    goals = ["g0", "g1", "g2", "g3"]
    g_best = {"g0": 0.9, "g1": 0.1, "g2": 0.9, "g3": 0.1}
    t_best = {0: {"g0": 0.1, "g1": 0.1, "g2": 0.9, "g3": 0.9}, 1: dict(g_best)}
    flags = tmp_path / "none.flags.jsonl"
    flags.write_text("".join(json.dumps({"goal_id": g, "order": o, "n_tokens": 100 + 10 * o, "at_cap": False, "loop": (g == "g0" and o == 0), "rep_ratio": 0.0}) + "\n" for g in goals for o in (0, 1)))
    greedy = {"status": "ok", "cap_tokens": 4096, "per_item": {g: [2, int(g == "g0"), 0, int(g == "g0"), g_best[g]] for g in goals},
              "provenance": {"aggregate": {"grader": {"judge_model": "claude-haiku-4-5", "rubric": "strongreject_v1"}}, "flags": {"path": str(flags)}}, "asr": {"threshold": 0.5}}
    for tr in (0, 1):
        rows = [_srow(g, i, tr, f"reply {g} {i} " * 5, "stop", 60 + i) for g in goals for i in (0, 1)]
        if tr == 0:
            rows[-1] = _srow("g3", 1, 0, "x " * 50, "length", 4096)
        _sampled_store(tmp_path, "qwen35_4b", tr, "pasttense", "none", rows)
    monkeypatch.setattr(D, "sampled_static_grades", lambda dest, tag, tr, fam, cell: (dict(t_best[tr]), {"aggregate": {"path": "x"}},
                                                                                    {"judge_model": "claude-haiku-4-5", "rubric": "strongreject_v1", "threshold": 0.5, "n_errored": 0}))
    items: list = []
    rec = D.static_cell(_a(bootstrap=50), tmp_path, "qwen35_4b", "pasttense", "none", greedy, [0, 1], [0.25, 0.5, 0.75], False, None, 7, items, {})
    assert rec["status"] == "ok" and rec["n_goals"] == 4 and rec["grader"]["validated_per_trial"] and rec["variants_per_goal"] == [2]
    c = rec["contrasts"]
    assert c["t0_vs_greedy"]["table"] == {"both": 1, "neither": 1, "a_only": 1, "b_only": 1} and c["t0_vs_greedy"]["delta_pp"] == 0.0
    assert c["t1_vs_greedy"]["discordant"]["n"] == 0 and c["t1_vs_t0"]["discordant"]["n"] == 2
    lg = rec["looped_goals"]
    assert lg["greedy_looped"]["n_goals"] == 1 and lg["greedy_looped"]["greedy_jailbroken"]["k"] == 1 and lg["greedy_looped"]["t0_jailbroken"]["k"] == 0
    assert rec["t0_discordants_by_greedy_loop"] == {"greedy_only_with_loop": 1, "greedy_only_without_loop": 0, "sampled_only_with_loop": 0, "sampled_only_without_loop": 1}
    assert rec["looped_responses"]["status"] == "not_computed"
    L = rec["lengths"]
    assert L["greedy"]["n"] == 8 and L["greedy"]["loop_whole_text_rule"]["k"] == 1 and L["greedy"]["lengths"]["p50"] == 105.0
    assert L["sampled"]["0"]["at_cap"]["k"] == 1 and L["sampled"]["1"]["at_cap"]["k"] == 0
    assert [(r["id"], r["greedy"], r["t0"], r["t1"]) for r in items] == [("g0", True, False, True), ("g1", False, False, False), ("g2", True, True, True), ("g3", False, True, False)]


# ------------------------------------------------------------------------------------------------- judged controls stores (synthetic)
def _judged_store(root: Path, cell: str, items: dict, *, greedy: bool, cap_flags: dict | None = None, subset=None, model=("zai-org/GLM-4.6V-Flash", "411bb4d7"), tag_hex="deadbeef") -> Path:
    """items: item_id -> label for xstest; subset: item_id -> 'safe' | 'unsafe' (default all safe); a complete judged store the readers accept."""
    st = root / f"{cell.replace('__', '--')}__controls__t0__a0__{tag_hex}__{tag_hex}"
    (st / "private").mkdir(parents=True)
    gen = {"max_new_tokens": 512, "batch_size": 8, "do_sample": False} if greedy else {"max_new_tokens": 512, "do_sample": True, "temperature": 0.8, "arm": "vendor_recommended", "trial": 0}
    (st / "manifest.json").write_text(json.dumps({"cell_id": cell, "model_id": model[0], "model_revision": model[1], "resolved_config": {"gen": gen}}))
    n = len(items); n_cap = sum(1 for i in items if (cap_flags or {}).get(i))
    (st / "gen_stats.json").write_text(json.dumps({"xstest": {"control": "xstest", "n_total": n, "finish_reasons": {"length": n_cap, "stop": n - n_cap}, "spool": "private/gens_xstest_v2.jsonl"}}))
    sub = subset or {i: "safe" for i in items}
    safe_j = [i for i in items if sub[i] == "safe"]
    n_ref = sum(1 for i in safe_j if items[i] == "full_refusal")
    (st / "judge_summary.json").write_text(json.dumps({"cell_id": cell, "controls": {"xstest": {"at_cap_share": n_cap / n, "over_refusal": {"point_estimate_over_judged": n_ref / len(safe_j)}}},
                                                       "provenance": {"judge_model": JUDGE["judge_model"], "protocol_sha": {"xstest": JUDGE["protocol_sha"]}}}))
    rows = [{"control": "xstest", "item_id": i, "label": l, "status": "judged", "subset": sub[i], "at_cap": bool((cap_flags or {}).get(i)), "gen_tokens": 512 if (cap_flags or {}).get(i) else 40, **JUDGE}
            for i, l in items.items()]
    (st / "judge_items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if not greedy:
        (st / "private" / "gens_xstest_v2.meta.jsonl").write_text("".join(json.dumps({"item_id": i, "finish_reason": "length" if (cap_flags or {}).get(i) else "stop",
                                                                                     "completion_tokens": 512 if (cap_flags or {}).get(i) else 40}) + "\n" for i in items))
    return st


def test_greedy_controls_fallback_reads_the_original_directory_and_refuses_a_non_greedy_or_foreign_store(tmp_path):
    rs = tmp_path / "run_state"; copy = tmp_path / "copy"
    items = {f"xst{k:04d}": ("full_refusal" if k < 2 else "full_compliance") for k in range(6)}
    _judged_store(rs / "controls_glm" / "glm46vflash" / "cells", "constv2-mix-full__psr_ce_all", items, greedy=True)
    cfg = {"model": "zai-org/GLM-4.6V-Flash", "model_revision": "411bb4d7"}
    gt = {"controls": {"xstest": {}}}
    g, where = D.resolve_greedy_controls(gt, rs, copy, "glm46vflash", "constv2-mix-full__psr_ce_all", "xstest", cfg, lambda t: 0.0, None)
    assert g is not None and g["status"] == "ok" and g["arm"] == "greedy" and "controls_glm" in where and g["cap_tokens"] == 512
    assert set(g["per_item"]) == set(items) and g["judge"]["judge_model"] == JUDGE["judge_model"]
    _judged_store(rs / "controls_glm" / "glm46vflash" / "cells", "none", items, greedy=False, tag_hex="feedface")       # a SAMPLED store is refused as the greedy side
    with pytest.raises(D.PairingMismatch):
        D.resolve_greedy_controls(gt, rs, copy, "glm46vflash", "none", "xstest", cfg, lambda t: 0.0, None)
    _judged_store(rs / "controls_glm" / "glm46vflash" / "cells", "prompted__safety_refusal", items, greedy=True, model=("other/model", "00000000"), tag_hex="0badf00d")
    with pytest.raises(D.PairingMismatch):                                                                                # another served model is refused
        D.resolve_greedy_controls(gt, rs, copy, "glm46vflash", "prompted__safety_refusal", "xstest", cfg, lambda t: 0.0, None)
    assert D.resolve_greedy_controls(gt, rs, copy, "glm46vflash", "absent_cell", "xstest", cfg, lambda t: 0.0, None) == (None, "absent")
    gb = {"controls": {"xstest": {"none": {"status": "ok", "cap_tokens": 512, "per_item": {}}}}}
    assert D.resolve_greedy_controls(gb, rs, copy, "glm46vflash", "none", "xstest", cfg, lambda t: 0.0, None)[1] == "greedy_baseline"


def test_four_way_on_the_common_judged_intersection():
    ids = [f"i{k}" for k in range(8)]
    def labels(refused: set, missing: set = frozenset()):
        return {i: {"label": "full_refusal" if i in refused else "full_compliance", "status": "judged" if i not in missing else "abstained"} for i in ids}
    gi = labels({"i0", "i1", "i2", "i3"}); p1 = labels({"i0", "i1"}); p2 = labels({"i0", "i1", "i2"}, missing={"i7"}); pf = labels({"i0", "i1", "i2", "i5"})
    fw = D.four_way(gi, p1, p2, pf, ids, 2048)
    assert fw["status"] == "ok" and fw["n_common"] == 7 and fw["final_cap"] == 2048
    assert fw["greedy"]["k"] == 4 and fw["sampled_512_pass1"]["k"] == 2 and fw["sampled_512_pass2"]["k"] == 3 and fw["final_rung_pass2"]["k"] == 4
    assert fw["decode_pp"] == pytest.approx(100 * (2 - 4) / 7) and fw["judge_pass_pp"] == pytest.approx(100 / 7) and fw["within_pass_rung_pp"] == pytest.approx(100 / 7)
    assert fw["total_pp"] == pytest.approx(0.0)
    assert D.four_way(gi, p1, None, pf, ids, 2048)["status"] == "pending"


def test_controls_cell_scopes_xstest_lengths_to_the_safe_subset_and_validates_per_trial(tmp_path):
    dest = tmp_path
    safe = {f"xst{k:04d}": "full_compliance" for k in range(4)}
    unsafe_ids = [f"xst{k:04d}" for k in range(4, 8)]
    sub = {**{i: "safe" for i in safe}, **{i: "unsafe" for i in unsafe_ids}}
    caps = {"xst0000": True, "xst0004": True, "xst0005": True}             # one safe item at cap, two unsafe
    def store_items(labels_safe):
        return {**labels_safe, **{i: "full_refusal" for i in unsafe_ids}}
    g_st = _judged_store(tmp_path / "gstore", "none", store_items(safe), greedy=True, cap_flags=caps, subset=sub)
    g = cgs.control_cell_metrics(g_st, "xstest", 512, lambda t: 0.0, None, want_meta=False)
    for tr in (0, 1):
        _judged_store(dest / "controls_sampled" / "controls_cap512" / f"qwen35_4b_t{tr}" / "cells", "none", store_items({**safe, "xst0001": "full_refusal"}), greedy=False, cap_flags=caps, subset=sub, tag_hex=f"{tr:08x}")
    items: list = []
    rec = D.controls_cell(_a(bootstrap=20), dest, "qwen35_4b", "xstest", "none", g, "greedy_baseline", [0, 1], False, None, 3, items, {})
    assert rec["status"] == "ok" and rec["n_items_in_scope"] == 4 and rec["scope"].startswith("the 250 SAFE")
    assert rec["contrasts"]["t0_vs_greedy"]["discordant"] == {"n": 1, "a_only": 0, "b_only": 1}
    L = rec["lengths"]
    assert L["population"].startswith("the 250 SAFE") and L["n_items"] == 4 and L["greedy"]["at_cap"]["k"] == 1 and L["greedy"]["at_cap"]["n"] == 4
    assert L["all_items"]["n_items"] == 8 and L["all_items"]["greedy"]["at_cap"]["k"] == 3
    assert L["sampled"]["0"]["at_cap"]["k"] == 1 and L["sampled"]["0"]["at_cap"]["n"] == 4
    assert rec["four_way"]["status"] == "pending"                          # no flip-pass copies in this world
    assert [r["id"] for r in items] == sorted(safe)
    st1 = cgs.control_store_dirs(dest / "controls_sampled" / "controls_cap512" / "qwen35_4b_t1" / "cells")["none"]      # a judge mismatch on trial 1 only is refused
    rows = [json.loads(l) for l in (st1 / "judge_items.jsonl").read_text().splitlines()]
    for r in rows:
        r["rubric_sha"] = "ffffffffffffffff"
    (st1 / "judge_items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    rec = D.controls_cell(_a(bootstrap=0), dest, "qwen35_4b", "xstest", "none", g, "greedy_baseline", [0, 1], False, None, 3, [], {})
    assert rec["status"] == "unpaired" and "t1" in rec["reason"] and "rubric_sha differs" in rec["reason"]


# ------------------------------------------------------------------------------------------------- the decomposition picks the final rung per (cell, control)
def _controls_root(dest: Path, tag: str, tr: int, cell: str, per_rung: dict) -> None:
    for cap, controls in per_rung.items():
        if not controls:
            continue
        st = dest / "controls_sampled" / f"controls_cap{cap}" / f"{tag}_t{tr}" / "cells" / f"{cell.replace('__', '--')}__controls__t0__a0__{cap:08x}__deadbeef"
        st.mkdir(parents=True)
        (st / "manifest.json").write_text(json.dumps({"cell_id": cell}))
        (st / "gen_stats.json").write_text(json.dumps({c: {"control": c, "n_total": 4, "finish_reasons": {"stop": 4}} for c in controls}))


def test_final_rung_is_per_cell_and_control(tmp_path):
    _controls_root(tmp_path, "qwen35_4b", 0, "none", {512: ["xstest", "or_bench_hard_1k"], 2048: ["xstest", "or_bench_hard_1k"], 8192: ["or_bench_hard_1k"]})
    assert D.final_rung_controls_cap(tmp_path, "qwen35_4b", 0, "none", "xstest") == 2048
    assert D.final_rung_controls_cap(tmp_path, "qwen35_4b", 0, "none", "or_bench_hard_1k") == 8192
    assert D.final_rung_controls_cap(tmp_path, "qwen35_4b", 0, "absent_cell", "xstest") is None


def _one_controls_rec(cell: str, ctl: str) -> dict:
    ids = [f"i{k}" for k in range(4)]
    c = D.paired_binary({i: i == "i0" for i in ids}, {i: i in {"i0", "i1"} for i in ids}, "x")
    return {"status": "ok", "held": False, "cap": 512, "lane": "qwen35_4b", "readout": ctl, "cell": cell,
            "contrasts": {"t0_vs_greedy": c, "t1_vs_greedy": c, "t1_vs_t0": c}, "lengths": {"greedy": {"n": 4}, "n_items": 4}}


def _fr_block(final_caps: dict) -> dict:
    return {"targets": {"qwen35_4b": {"controls": {
        f"{ctl}|none": {"status": "ok", "held": False, "pooled": {"full_refusal": {"p": 0.25, "k": 2, "n": 8, "bootstrap95": {"lo": 0.0, "hi": 0.6}}},
                        "trials": {"0": {"final_cap": cap}, "1": {"final_cap": cap}}, "residual_pooled": {"n_at_cap": 1, "n_responses": 8}}
        for ctl, cap in final_caps.items()}, "static": {}}}}


def _dec_args(tmp_path):
    return SimpleNamespace(bootstrap=50, seed=3, final_rung_path=tmp_path / "final_rung.json", _items_static={}, _items_controls={
        ("qwen35_4b", ctl, "none"): [{"id": f"i{k}", "t0": k == 0, "t1": k in (0, 1)} for k in range(4)] for ctl in cgs.CONTROLS})


def test_decomposition_uses_the_final_rung_store_and_refuses_a_disagreeing_record(tmp_path):
    for tr in (0, 1):
        _controls_root(tmp_path, "qwen35_4b", tr, "none", {512: ["xstest", "or_bench_hard_1k"], 2048: ["xstest", "or_bench_hard_1k"], 8192: ["or_bench_hard_1k"]})
    controls_recs = {"qwen35_4b": {ctl: {"none": _one_controls_rec("none", ctl)} for ctl in cgs.CONTROLS}}
    rows, _ = D.decomposition_rows(tmp_path, _fr_block({"xstest": 2048, "or_bench_hard_1k": 8192}), {}, controls_recs, [0, 1], set(), _dec_args(tmp_path))
    by = {r["readout"]: r for r in rows}
    assert by["xstest"]["sampled_final_rung"]["final_caps"] == [2048] and by["or_bench_hard_1k"]["sampled_final_rung"]["final_caps"] == [8192]
    assert by["xstest"]["greedy_at_cap"]["p"] == 0.25 and by["xstest"]["sampled_same_cap"]["p"] == 0.375
    assert by["xstest"]["decode_term_pp"] == pytest.approx(12.5) and by["xstest"]["final_rung_update_pp"] == pytest.approx(-12.5) and by["xstest"]["total_pp"] == pytest.approx(0.0)
    assert by["xstest"]["sampled_final_rung"]["residual_share_at_cap"] == pytest.approx(1 / 8)
    with pytest.raises(cgs.Refuse) as exc:
        D.decomposition_rows(tmp_path, _fr_block({"xstest": 8192, "or_bench_hard_1k": 8192}), {}, controls_recs, [0, 1], set(), _dec_args(tmp_path))
    assert "final-rung disagreement" in exc.value.msg


def test_split_moved_flags_held_by_lane_and_cell_and_classifies_dominance_with_ties(tmp_path):
    for tr in (0, 1):
        _controls_root(tmp_path, "qwen35_4b", tr, "none", {512: ["xstest", "or_bench_hard_1k"], 2048: ["xstest", "or_bench_hard_1k"], 8192: ["or_bench_hard_1k"]})
    controls_recs = {"qwen35_4b": {ctl: {"none": _one_controls_rec("none", ctl)} for ctl in cgs.CONTROLS}}
    rows, _ = D.decomposition_rows(tmp_path, _fr_block({"xstest": 2048, "or_bench_hard_1k": 8192}), {}, controls_recs, [0, 1], {("qwen35_4b", "none")}, _dec_args(tmp_path))
    assert all(r["held"] for r in rows)
    S = D.split_moved(rows, {("qwen35_4b", "none")})
    assert S["numbers_moved"] == [] and len(S["numbers_moved_held"]) == 2 and S["n_rows_complete"] == 0
    S2 = D.split_moved(rows, {("other_lane", "none")})
    assert len(S2["numbers_moved"]) == 2 and S2["numbers_moved_held"] == []
    assert S2["dominance"] == {"decode": 0, "update": 0, "tie": 2} and all(r["dominant_term"] == "tie" for r in S2["numbers_moved"])   # |12.5| == |-12.5|: a tie


# ------------------------------------------------------------------------------------------------- small helpers
def test_intervals_separate_histogram_and_exact_ecdf_points():
    assert D.intervals_separate({"p": 0.1, "lo": 0.05, "hi": 0.15}, {"p": 0.3, "lo": 0.2, "hi": 0.4}) is True
    assert D.intervals_separate({"p": 0.1, "lo": 0.05, "hi": 0.25}, {"p": 0.3, "lo": 0.2, "hi": 0.4}) is False
    assert D.intervals_separate({"p": 0.1, "lo": None, "hi": None}, {"p": 0.3, "lo": 0.2, "hi": 0.4}) is None
    h = D.histogram([1, 2, 4096, 4096, 20000])
    assert h["n"] == 5 and sum(h["counts"]) == 4 and h["n_above_top_edge"] == 1
    assert D.ecdf_points([512, 40, 512, 7, 512]) == {"values": [7, 40, 512], "counts": [1, 1, 3], "n": 5}     # a step at every observed count; the cap spike AT the cap
    q = D.quantile_block([1, 2, 3, 4, 100])
    assert q["n"] == 5 and q["p50"] == 3.0 and q["max"] == 100.0


# ------------------------------------------------------------------------------------------------- row-level judge identity vs the summary (the guard gap)
def _rewrite_rows(store: Path, **over):
    rows = [json.loads(l) for l in (store / "judge_items.jsonl").read_text().splitlines()]
    for r in rows:
        r.update(over)
    (store / "judge_items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_controls_store_with_missing_or_foreign_row_identity_on_trial_1_is_refused(tmp_path):
    dest = tmp_path
    items = {f"xst{k:04d}": "full_compliance" for k in range(4)}
    g_st = _judged_store(tmp_path / "gstore", "none", items, greedy=True)
    g = cgs.control_cell_metrics(g_st, "xstest", 512, lambda t: 0.0, None, want_meta=False)
    assert g["judge"]["row_identity"]["validated_against_summary"] and g["judge"]["row_identity"]["protocol_sha"] == JUDGE["protocol_sha"]
    for tr in (0, 1):
        _judged_store(dest / "controls_sampled" / "controls_cap512" / f"qwen35_4b_t{tr}" / "cells", "none", items, greedy=False, tag_hex=f"{tr:08x}")
    st1 = cgs.control_store_dirs(dest / "controls_sampled" / "controls_cap512" / "qwen35_4b_t1" / "cells")["none"]
    # (a) trial-1 rows uniformly MISSING judge_model, summary unchanged: refused at the store read, never masked by the summary's provenance
    _rewrite_rows(st1, judge_model=None)
    with pytest.raises(cgs.Refuse) as exc:
        D.controls_cell(_a(bootstrap=0), dest, "qwen35_4b", "xstest", "none", g, "greedy_baseline", [0, 1], False, None, 3, [], {})
    assert "no judge identity" in exc.value.msg
    # (b) trial-1 rows carrying a FOREIGN protocol_sha under an unchanged summary: refused, naming both
    _rewrite_rows(st1, judge_model=JUDGE["judge_model"], protocol_sha="ffffffffffffffff")
    with pytest.raises(cgs.Refuse) as exc:
        D.controls_cell(_a(bootstrap=0), dest, "qwen35_4b", "xstest", "none", g, "greedy_baseline", [0, 1], False, None, 3, [], {})
    assert "foreign rows" in exc.value.msg and "ffffffffffffffff" in exc.value.msg
    # (c) restored: pairs
    _rewrite_rows(st1, protocol_sha=JUDGE["protocol_sha"])
    rec = D.controls_cell(_a(bootstrap=0), dest, "qwen35_4b", "xstest", "none", g, "greedy_baseline", [0, 1], False, None, 3, [], {})
    assert rec["status"] == "ok"


def test_four_way_carries_the_inter_pass_differ_share():
    ids = [f"i{k}" for k in range(8)]
    def labels(refused: set, missing: set = frozenset()):
        return {i: {"label": "full_refusal" if i in refused else "full_compliance", "status": "judged" if i not in missing else "abstained"} for i in ids}
    gi = labels({"i0", "i1", "i2", "i3"}); p1 = labels({"i0", "i1"}); p2 = labels({"i0", "i1", "i2"}, missing={"i7"}); pf = labels({"i0", "i1", "i2", "i5"})
    fw = D.four_way(gi, p1, p2, pf, ids, 2048)
    assert fw["pass1_vs_pass2_label_differ_share"] == pytest.approx(1 / 7)          # i2 differs between the passes on the 7 common items
    assert "not reuse from pass 1" in fw["cached_before_run_meaning"].lower()


# ------------------------------------------------------------------------------------------------- the report's figure links
def test_figure_links_resolve_from_the_report_when_no_mirror_url_is_set(tmp_path):
    dest = tmp_path / "dest"
    md_path = dest / "report" / "analysis" / f"{D.NAME}.md"              # the default --out, .md beside the .json
    fig = dest / "report" / "figs" / D.FIGS_SUBDIR / "decomposition_static.png"   # the default --figs
    fig.parent.mkdir(parents=True); fig.write_bytes(b"png")
    link = D.figure_link(fig, md_path, url_base="")                        # the unset-variable case
    assert link == "../figs/decode/decomposition_static.png"
    assert (md_path.parent / link).resolve() == fig.resolve()
    # a mirror URL: the mirror's own layout, never a local path
    assert D.figure_link(fig, md_path, url_base="https://example/figs/f22-sampled/") == "https://example/figs/f22-sampled/decode/decomposition_static.png"
    # the module default is analysis_sampled's FIGS_URL: the environment's base URL + the mirror's directory, or empty
    assert D.A.FIGS_URL == "" or D.A.FIGS_URL.endswith("/f22-sampled/")
    assert D.A.FIGS_URL != "f22-sampled/"
    # write_md links every figure through it
    import inspect
    assert "figure_link(f['path'], path)" in inspect.getsource(D.write_md)
