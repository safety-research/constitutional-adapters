"""ifeval-canonical-battery — the ANALYSIS step's reader fixes and the ladder-aware FINAL readout (decisions-analysis N1–N2).

The wave and rung agents found the analysis readers refusing every real store on keys the producer never writes (V20 / V36 /
V41): ``off_cap_by_task`` read under ``resolved_config.tinycap`` (the producer writes it at the manifest top level);
``certification_check`` reading the thinking pin at ``endpoint.default_chat_template_kwargs`` and the geometry entry at
``endpoint.geometry.eval`` (the records are ``evidence.serve_header.template_kwargs`` and ``served_geometry.geometry``); the
Nemotron attestation at ``endpoint.nemotron_attestation`` (the record is ``endpoint.mamba_backend_attestation``). Pinned here,
through the synthetic store written in the PRODUCER's shape: the readers take the producer's fields and refuse only real
contract violations; every leg is admitted by ITS OWN pin across several wave roots (a superseded failed store beside the
admitted one is listed, never a leg); the FINAL readout splits the merged reading's residual into loops and truncation by the
record of the leg that last drew each draw; HELD / PENDING rows appear in every table and figure as slots, never as measured;
every figure title is at most two words; the results template is byte-pinned; the canonical ids and the per-leg store pins
are the committed configs.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from sisteer.capeval.tiers import cap_ladder as CL
from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import ifeval_contracts as IC

import ifeval_synthetic_store as G

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def _load(name, rel):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A = _load("tinycap_ifeval_battery", "scripts/tinycap_ifeval_battery.py")
M = _load("tinycap_math500_thinking", "scripts/tinycap_math500_thinking.py")
V = A.V
ROWS = G.matrix_rows()
B_FAST = 150
ESC = ("gemma4e4b", "main", "constv2-mix-full__lora_r4_ce")
NONE_G = ("gemma4e4b", "main", "none")
OTHER_TREE = "1234567890abcdef1234567890abcdef12345678"
THIRD_TREE = "abcdefabcdefabcdefabcdefabcdefabcdefabcd"


def row(*t):
    return G.row_for(ROWS, *t)


def _rows(*triples):
    out = [dict(r) for r in ROWS if (r["model_key"], r["view"], r["cell"]) in set(triples)]
    for i, r in enumerate(out):
        r["index"] = i
    return out


def analyze(root, rows, **kw):
    kw.setdefault("canonical_ids", G.IDS)
    kw.setdefault("n_resamples", B_FAST)
    kw.setdefault("read_eval_logs", False)
    kw.setdefault("analysis_sha", G.FULL_SHA)
    return A.analyze(root, kw.pop("sha", G.SHA7), matrix=rows, **kw)


def _ctx(**kw):
    kw.setdefault("canonical_ids", None)
    kw.setdefault("pins", None)
    kw.setdefault("expected_refusals", {})
    kw.setdefault("read_eval_logs", False)
    return A.Context(**kw)


def _esc(root, **common):
    return G.build_escalated_cell(root, row(*ESC), at_cap={(i, t) for i in range(12) for t in (0, 1)}, loops={(0, 0), (0, 1)},
                                  rung_still_at_cap={(i, t) for i in range(1, 8) for t in (0, 1)}, rung_boundary={(9, 0)},
                                  rung8192=True, rung8192_still_at_cap={(1, 0)}, **common)


# --------------------------------------------------------------------------- (1) off_cap_by_task at the manifest top level
def test_a_rung_leg_states_its_cap_at_the_manifest_top_level_and_the_old_site_is_not_read(tmp_path):
    G.build_store(tmp_path, row(*NONE_G))
    _esc(tmp_path)
    m = _rows(NONE_G, ESC)
    rep = analyze(tmp_path, m)
    assert A.non_log_gates_clear(rep), A.production_refusals(rep)                   # no .eval log read: not production (r3 S1)
    for cap in (4096, 8192):
        leg = rep["legs"][V.leg_key(*ESC, cap)]
        assert leg["status"] == "usable" and leg["identity"] is None
    # a rung manifest whose statement sits ONLY under resolved_config.tinycap (the old fixture's site) does not state the cap
    cd = Path(rep["legs"][V.leg_key(*ESC, 4096)]["out_root"]) / "cells"
    cell_dir = next(p for p in cd.iterdir() if p.is_dir())
    man = json.loads((cell_dir / "manifest.json").read_text())
    assert man[IC.MANIFEST_KEY_OFF_CAP_BY_TASK] == {IC.IFEVAL_TASK_NAME: 4096}
    man["resolved_config"]["tinycap"][IC.MANIFEST_KEY_OFF_CAP_BY_TASK] = man.pop(IC.MANIFEST_KEY_OFF_CAP_BY_TASK)
    (cell_dir / "manifest.json").write_text(json.dumps(man))
    rep2 = analyze(tmp_path, m)
    leg2 = rep2["legs"][V.leg_key(*ESC, 4096)]
    assert leg2["identity"] and "does not state the rung cap 4096" in leg2["identity"] and "top-level" in leg2["identity"]
    assert rep2["gates"]["I2_identity"]["status"] == A.GATE_FAIL
    assert V.off_cap_statement({IC.MANIFEST_KEY_OFF_CAP_BY_TASK: {"ifeval": "8192"}}) == {"ifeval": 8192} and V.off_cap_statement({}) == {}


# --------------------------------------------------------------------------- (2) the certification reads the producer's records
def _one(tmp_path, **kw):
    r = row("qwen35_4b", "main", "none")
    G.build_store(tmp_path, r, **kw)
    return A.collect_leg(tmp_path, r, cap=G.CAP1, ctx=_ctx())["certification"]


def test_the_certification_reads_the_served_geometry_and_the_evidence_serve_header_and_passes_on_the_producers_shape(tmp_path):
    cert = _one(tmp_path)
    assert cert["passed"], cert["failures"]
    assert cert["geometry_eval"] == "capability:ifeval_2048" and cert["max_model_len"] == G.MAX_MODEL_LEN_BY_CAP[2048]
    assert cert["served_seqs"] == 142 and cert["configured_seqs"] == 142 and cert["kv_shrink"] is None
    assert cert["window_source"] == "table" and cert["seqs_source"] == "model_budget"
    assert cert["manifest_geometry_source"] == {"window_source": "override", "seqs_source": "override", "max_model_len": G.MAX_MODEL_LEN_BY_CAP[2048], "max_num_seqs": 142}
    assert cert["template_kwargs"] == {"enable_thinking": False} and cert["thinking_switch"] is True and cert["header_moves"] is True
    assert cert["default_chat_template_kwargs"] == cert["template_kwargs"]          # the old key names the pin AS SERVED now


@pytest.mark.parametrize("family_row, expect_kwargs, expect_switch", [
    (("gemma4e4b", "main", "none"), {"enable_thinking": False}, True),        # the wrapper composes the pin on gemma too (its header does not move)
    (("llama8b", "main", "none"), {}, False),                                   # no switch: nothing pinned, the `not_available` arm
])
def test_the_thinking_pin_is_judged_by_the_templates_switch(tmp_path, family_row, expect_kwargs, expect_switch):
    r = row(*family_row)
    G.build_store(tmp_path, r)
    cert = A.collect_leg(tmp_path, r, cap=G.CAP1, ctx=_ctx())["certification"]
    assert cert["passed"], cert["failures"]
    assert cert["template_kwargs"] == expect_kwargs and cert["thinking_switch"] is expect_switch


@pytest.mark.parametrize("defect, match", [
    ("enable_true", "ENABLE thinking"),
    ("pin_missing_on_switch", "!= the thinking-OFF pin"),
    ("no_probe", "no thinking_header probe"),
    ("hash_mismatch", "served_template_hash"),
    ("wrong_entry", "served geometry.eval"),
    ("window_override", "window_source 'override' != 'table'"),
    ("seqs_source_off", "seqs_source 'override' != 'model_budget'"),
    ("shrink_not_holding", "does not hold"),
    ("endpoint_disagrees", "the endpoint serves"),
    ("concurrency_off", "manifest concurrency"),
    ("manifest_copy_off", "manifest's geometry copy"),
    ("no_served_record", "no served_geometry.json"),
])
def test_a_real_contract_violation_in_the_producers_records_is_refused_by_name(tmp_path, defect, match):
    kw = {}
    if defect == "enable_true":
        kw["evidence_edit"] = lambda ev: ev["serve_header"].__setitem__("template_kwargs", {"enable_thinking": True})
    elif defect == "pin_missing_on_switch":
        kw["evidence_edit"] = lambda ev: ev["serve_header"].__setitem__("template_kwargs", {})
    elif defect == "no_probe":
        kw["endpoint_edit"] = lambda ep: ep.pop("thinking_header")
    elif defect == "hash_mismatch":
        kw["evidence_edit"] = lambda ev: ev["serve_header"].__setitem__("served_template_hash", "0" * 16)
    elif defect == "wrong_entry":
        kw["geometry_edit"] = lambda g: g.__setitem__("eval", "capability:ifeval_4096")
    elif defect == "window_override":
        kw["geometry_edit"] = lambda g: g.__setitem__("window_source", "override")
    elif defect == "seqs_source_off":
        kw["geometry_edit"] = lambda g: g.__setitem__("seqs_source", "override")
    elif defect == "shrink_not_holding":
        shrink = G.kv_shrink(142, 101, G.MAX_MODEL_LEN_BY_CAP[2048])
        shrink["kv_capacity_tokens"] = 10                                      # the capacity no longer divides to the served S
        kw["kv_shrink"] = shrink
    elif defect == "endpoint_disagrees":
        kw["endpoint_edit"] = lambda ep: ep.__setitem__("max_num_seqs", 7)
    elif defect == "concurrency_off":
        kw["manifest_edit"] = lambda man: man.__setitem__(C.MANIFEST_KEY_CONCURRENCY, 9)
    elif defect == "manifest_copy_off":
        kw["manifest_edit"] = lambda man: man[C.MANIFEST_KEY_GEOMETRY].__setitem__("max_num_seqs", 9)
    r = row("qwen35_4b", "main", "none")
    G.build_store(tmp_path, r, **kw)
    if defect == "no_served_record":
        (tmp_path / IC.ifeval_paths(*("qwen35_4b", "main", "none"), 2048).out_root / "served_geometry.json").unlink()
    cert = A.collect_leg(tmp_path, r, cap=G.CAP1, ctx=_ctx())["certification"]
    assert not cert["passed"] and any(match in f for f in cert["failures"]), cert["failures"]


def test_a_holding_kv_shrink_is_the_served_s_and_the_nemotron_attestation_and_certification_are_read(tmp_path):
    r = row("nemotron3nano4b", "main", "none")
    L = G.MAX_MODEL_LEN_BY_CAP[2048]
    G.build_store(tmp_path, r, kv_shrink=G.kv_shrink(256, 168, L))
    certs = {G.FULL_SHA: {"capability:ifeval_2048": {"admitted": True, "attestation_sha256": "c" * 64}}}
    cert = A.collect_leg(tmp_path, r, cap=G.CAP1, ctx=_ctx(nemotron_certifications=certs))["certification"]
    assert cert["passed"], cert["failures"]
    assert cert["served_seqs"] == 168 and cert["configured_seqs"] == 256 and cert["seqs_source"] == "kv_derived" and cert["kv_shrink"]["served_max_num_seqs"] == 168
    assert cert["nemotron_attestation"]["ok"] is True and cert["nemotron_attestation"]["n_reports"] == 1 and cert["nemotron_attestation"]["expected_n_mamba_layers"] == 21
    assert cert["nemotron_certification"] == {"admitted": True, "attestation_sha256": "c" * 64}
    # a 7-hex prefix keys the doc too; a doc without this (tree, entry) refuses; a not-ok attestation refuses; no doc = not asked
    cert7 = A.collect_leg(tmp_path, r, cap=G.CAP1, ctx=_ctx(nemotron_certifications={G.SHA7: certs[G.FULL_SHA]}))["certification"]
    assert cert7["passed"] and cert7["nemotron_certification"]["attestation_sha256"] == "c" * 64
    miss = A.collect_leg(tmp_path, r, cap=G.CAP1, ctx=_ctx(nemotron_certifications={OTHER_TREE: certs[G.FULL_SHA]}))["certification"]
    assert not miss["passed"] and "no Nemotron certification for tree" in miss["failures"][0]
    none_doc = A.collect_leg(tmp_path, r, cap=G.CAP1, ctx=_ctx())["certification"]
    assert none_doc["passed"] and none_doc["nemotron_certification"] is None
    G.build_store(tmp_path / "bad", r, endpoint_edit=lambda ep: ep.__setitem__("mamba_backend_attestation", G.mamba_attestation(ok=False)))
    bad = A.collect_leg(tmp_path / "bad", r, cap=G.CAP1, ctx=_ctx())["certification"]
    assert not bad["passed"] and "is not ok" in bad["failures"][0]
    G.build_store(tmp_path / "layers", r, endpoint_edit=lambda ep: ep["mamba_backend_attestation"]["reports"][0].__setitem__("n_mamba_layers", 20))
    lay = A.collect_leg(tmp_path / "layers", r, cap=G.CAP1, ctx=_ctx())["certification"]
    assert not lay["passed"] and "n_mamba_layers" in lay["failures"][0]


# --------------------------------------------------------------------------- (3) admission by per-row pin across several roots
def test_legs_are_admitted_by_their_own_pin_and_a_superseded_failed_store_under_another_root_is_listed_never_a_leg(tmp_path):
    """The battery: pass-1 stores under two roots at two trees, rungs at two more, and — for two rows — a FAILED first-pin store
    (no summary, RESULT INCOMPLETE) beside the complete re-run under the other root (V21 / V30)."""
    root_a, root_b = tmp_path / "pull_a", tmp_path / "pull_b"
    G.build_store(root_a, row(*NONE_G))                                          # tree FULL_SHA under root a
    G.build_store(root_b, row("gemma4e4b", "it", "none"), tree_sha=OTHER_TREE)  # tree OTHER under root b
    # the failed first-pin store of the it/none row under root a: a manifest at FULL_SHA, no summary, RESULT INCOMPLETE
    failed_cd = G.build_store(root_a, row("gemma4e4b", "it", "none"), tree_sha=G.FULL_SHA)
    failed_root = failed_cd.parents[1]
    for p in failed_cd.rglob("*"):
        if p.is_file() and p.name.startswith("inspect__ifeval") and p.suffix == ".json":
            p.unlink()
    from sisteer.capeval.tinycap import completion
    for p in (completion.task_summary_path(failed_cd, IC.IFEVAL_TASK), completion.done_marker_path(failed_cd, IC.IFEVAL_TASK)):
        if p.exists():
            p.unlink()
    (failed_root / "RESULT").write_text(f"tinycap-modal name=x rc=1 result=INCOMPLETE status=INCOMPLETE sha={G.FULL_SHA} at=t\n")
    m = _rows(NONE_G, ("gemma4e4b", "it", "none"))
    pins = {f"{V.cell_key(*NONE_G)}@2048": G.FULL_SHA, "gemma4e4b/it/none@2048": OTHER_TREE}
    rep = analyze([root_a, root_b], m, sha="e43f81f", store_pins=pins)
    assert A.non_log_gates_clear(rep), A.production_refusals(rep)                   # no .eval log read: not production (r3 S1)
    assert rep["admission"] == "per-row pins" and rep["gen_roots"] == [str(root_a), str(root_b)]
    c_it = rep["cells"]["gemma4e4b/it/none"]
    assert c_it["usable"] and c_it["pin"] == OTHER_TREE and c_it["gen_root"] == str(root_b) and c_it["label"] == "e43f81f"
    assert [(s["producer_tree_sha"], s["result"]["result"], s["admitted"]) for s in c_it["superseded_stores"]] == [(G.FULL_SHA, "INCOMPLETE", False)]
    assert rep["cells"][V.cell_key(*NONE_G)]["pin"] == G.FULL_SHA
    assert rep["gates"]["I7_population"]["status"] == A.GATE_PASS and set(rep["provenance"]["producer_trees"]) == {G.FULL_SHA, OTHER_TREE}
    assert rep["provenance"]["trees_by_cap"] == {"2048": sorted([G.FULL_SHA, OTHER_TREE])}
    assert "superseded stores" in A.supplement_md(rep) and OTHER_TREE[:8] in A._pins_table(rep)
    # without a pins doc the one-tree rule stands: two trees refuse (the legacy admission), and the two stores for one row refuse by name
    rep_legacy = analyze([root_a, root_b], m, sha=G.SHA7)
    assert rep_legacy["legs"]["gemma4e4b/it/none@2048"]["status"] != "usable" and "2 stores for" in rep_legacy["legs"]["gemma4e4b/it/none@2048"]["refused"]
    # a leg whose tree is not its admitted pin is refused by name; a leg without an entry is refused by name
    rep_wrong = analyze([root_a, root_b], m, sha="e43f81f", store_pins={**pins, "gemma4e4b/it/none@2048": THIRD_TREE})
    assert "at the admitted pin" in rep_wrong["legs"]["gemma4e4b/it/none@2048"]["refused"] and rep_wrong["gates"]["I0_completeness"]["status"] == A.GATE_FAIL
    rep_missing = analyze([root_a, root_b], m, sha="e43f81f", store_pins={f"{V.cell_key(*NONE_G)}@2048": G.FULL_SHA})
    assert "no admitted pin" in rep_missing["legs"]["gemma4e4b/it/none@2048"]["refused"]
    # the rung legs are admitted by THEIR pins: a chain whose rung tree differs from the pass-1 tree is a production result
    root_c = tmp_path / "pull_c"
    G.build_store(root_c, row(*NONE_G))
    _esc(root_c)
    pins_c = {f"{V.cell_key(*NONE_G)}@2048": G.FULL_SHA, f"{V.cell_key(*ESC)}@2048": G.FULL_SHA, f"{V.cell_key(*ESC)}@4096": G.FULL_SHA, f"{V.cell_key(*ESC)}@8192": G.FULL_SHA}
    rep_c = analyze(root_c, _rows(NONE_G, ESC), sha="e43f81f", store_pins=pins_c)
    assert A.non_log_gates_clear(rep_c) and rep_c["cells"][V.cell_key(*ESC)]["pins_by_cap"] == {"2048": G.FULL_SHA, "4096": G.FULL_SHA, "8192": G.FULL_SHA}
    pins_bad = {**pins_c, f"{V.cell_key(*ESC)}@8192": OTHER_TREE}
    rep_d = analyze(root_c, _rows(NONE_G, ESC), sha="e43f81f", store_pins=pins_bad)
    assert rep_d["legs"][V.leg_key(*ESC, 8192)]["status"] != "usable" and "not the admitted pin" in rep_d["legs"][V.leg_key(*ESC, 8192)]["refused"]
    with pytest.raises(A.AnalysisError, match="not a tree sha"):
        A.load_store_pins({"a/b/c@2048": "zz"})


# --------------------------------------------------------------------------- (4) the FINAL readout: the residual split by the record that last drew each draw
def test_the_final_readout_splits_the_merged_residual_into_loops_and_truncation_at_the_exit_rung(tmp_path):
    G.build_store(tmp_path, row(*NONE_G))
    _esc(tmp_path)
    rep = analyze(tmp_path, _rows(NONE_G, ESC))
    fr = rep["final_readouts"][V.cell_key(*ESC)]
    c = rep["cells"][V.cell_key(*ESC)]
    # pass 1: 24 at cap of which 2 loops (never re-drawn); 22 re-drawn at 4,096 -> 14 still at cap (the rung record classifies them),
    # those 14 re-drawn at 8,192 -> 1 still at cap; the merged residual = 2 pass-1 loops + the 8,192 rung's 1
    assert fr["exit_cap"] == 8192 and fr["reading"] == "merged" and fr["n_draws"] == 1082 and fr["promoted"] == [4096, 8192]
    assert fr["n_at_cap"] == 3 == c["ladder"]["residual"]["n_at_cap"] and fr["n_unclassified"] == 0
    assert fr["by_source"]["r2048"] == {"loop": 2, "truncated": 0, "unclassified": 0} and set(fr["by_source"]) == {"r2048", "rung8192"}
    assert fr["n_loop"] + fr["n_truncated"] == 3 and fr["share_at_cap"] == pytest.approx(3 / 1082)
    assert fr["share_loop"] == pytest.approx(fr["n_loop"] / 1082) and fr["share_truncated"] == pytest.approx(fr["n_truncated"] / 1082)
    below = rep["final_readouts"][V.cell_key(*NONE_G)]
    # the fixture's `none` cell has NO at-cap draw: the adapter's exit is `clean` (below_threshold names a non-zero share under 1 %)
    assert below == {**below, "exit_cap": 2048, "exit": CL.EXIT_CLEAN, "n_at_cap": 0, "n_loop": 0, "n_truncated": 0, "promoted": []}
    # the final table names the exit rung, the split, the recorder and the pins for every cell
    table = A._final_table(rep)
    assert f"| {V.cell_key(*ESC)} | 8192 | {fr['exit']} |" in table and f"| {fr['n_loop']} / {fr['n_truncated']} |" in table and "AGREES" in table
    assert f"2048: `{G.FULL_SHA[:8]}`; 4096: `{G.FULL_SHA[:8]}`; 8192: `{G.FULL_SHA[:8]}`" in table
    # a residual draw no record classifies fails the ladder gate by name
    rec_path = tmp_path / IC.ifeval_paths(*ESC, 8192).cap_ladder
    rec = json.loads(rec_path.read_text())
    rec["loop_draws"], rec["eligible_draws"] = [], []
    rec_path.write_text(json.dumps(rec))
    rep2 = analyze(tmp_path, _rows(NONE_G, ESC))
    assert rep2["gates"]["I6_ladder"]["status"] == A.GATE_FAIL and "no ladder record classifies" in rep2["gates"]["I6_ladder"]["detail"]


# --------------------------------------------------------------------------- (5) HELD / PENDING slots in every table and figure; the recorder
def test_held_and_pending_rows_are_slots_in_every_table_and_figure_never_measured(tmp_path):
    pytest.importorskip("matplotlib")
    held = ("gemma4e4b", "it", "constv2-mix-full__lora_r4_ce")
    pending = ("gemma4e4b", "it", "alphasteer")
    rows = _rows(("gemma4e4b", "it", "none"), held, pending, NONE_G, ESC)
    G.build_store(tmp_path, row("gemma4e4b", "it", "none"))
    G.build_store(tmp_path, row(*NONE_G))
    _esc(tmp_path)
    rep = analyze(tmp_path, rows)
    assert A.non_log_gates_clear(rep), A.production_refusals(rep)                   # no .eval log read: not production (r3 S1)
    slots = rep["slots"]
    assert set(slots) == {V.cell_key(*held), V.cell_key(*pending)}
    assert slots[V.cell_key(*held)]["label"] == A.HELD_LABEL and "P41" in slots[V.cell_key(*held)]["reason"]
    assert slots[V.cell_key(*pending)]["label"] == A.PENDING_LABEL and "PR #93" in A.PENDING_LABEL and slots[V.cell_key(*pending)]["reason"]
    for k in slots:
        assert rep["cells"][k]["usable"] is False and rep["cells"][k]["slot"] in (A.HELD_LABEL, A.PENDING_LABEL) and "model_key" in rep["cells"][k]
    for table in (A._shares_table(rep), A._contrasts_table(rep), A._final_table(rep), A._slots_table(rep)):
        assert f"**{A.HELD_LABEL}**" in table and "**PENDING" in table, table
    md = A.render_results_md(rep)
    assert "## Appendix A" in md and "HELD and PENDING rows" in md and A.HELD_LABEL in md and "PR #93" in md
    assert str(Path.home()) not in md and "~/" in A._home_relative(Path.home() / "x")      # a committed record never names the machine
    assert "AGREES on every one of the" in md and "The FINAL per-cell readout" in md
    figs = A.make_figures(rep, tmp_path / "figs")
    assert "final_ifeval_exits" in figs["written"]
    nums = json.loads(Path(figs["numbers"]).read_text())
    caps = nums["captions"]
    for stem in ("final_ifeval_metrics_by_cell", "final_ifeval_shares", "final_ifeval_exits", "final_ifeval_contrasts_gemma4e4b"):
        assert "Hatched slots are NOT measured" in caps[stem] and "1 HELD" in caps[stem] and "1 PENDING" in caps[stem], stem
    # the captions state the reading from the report's numbers
    assert "Words:" in caps["final_ifeval_contrasts_gemma4e4b"] and "-> lower than none" in caps["final_ifeval_contrasts_gemma4e4b"]
    assert "Exits:" in caps["final_ifeval_exits"] and "cells exceed it and were promoted" in caps["final_ifeval_shares"]
    # every slot draws in the metrics / shares / exits figures (the figure numbers cover only measured cells)
    fig, _ = A.fig_metrics_by_cell(rep)
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert A.HELD_LABEL in texts and "PENDING" in texts
    fig2, _ = A._fig_contrasts_model(rep, "gemma4e4b")
    assert any("HELD — not measured" in t.get_text() for ax in fig2.axes for t in ax.texts) and any("PENDING — not measured" in t.get_text() for ax in fig2.axes for t in ax.texts)
    import matplotlib.pyplot as plt
    plt.close(fig)
    plt.close(fig2)
    tables = json.loads(Path(figs["tables"]).read_text())
    assert set(tables["slots"]) == set(slots) and tables["final_readouts"][V.cell_key(*ESC)]["exit_cap"] == 8192


def test_a_recorder_that_does_not_agree_fails_i2_and_a_missing_record_is_reported_not_a_pass(tmp_path):
    def disagree(root):
        p = root / IC.ifeval_paths(*NONE_G, 2048).out_root / C.RESOLVED_SAMPLER_FILENAME
        d = json.loads(p.read_text())
        d["verdict"], d["why"] = "DIVERGED", "client top_p 0.9 != --expect 0.8"
        p.write_text(json.dumps(d))
    G.build_store(tmp_path / "ok", row(*NONE_G))
    rep = analyze(tmp_path / "ok", _rows(NONE_G))
    assert rep["provenance"]["recorder"] == {"per_leg": {V.leg_key(*NONE_G, 2048): "AGREES"}, "tally": {"AGREES": 1}, "n_legs": 1, "all_agree": True}
    G.build_store(tmp_path / "bad", row(*NONE_G))
    disagree(tmp_path / "bad")
    rep2 = analyze(tmp_path / "bad", _rows(NONE_G))
    assert rep2["gates"]["I2_identity"]["status"] == A.GATE_FAIL and "recorder verdict 'DIVERGED'" in rep2["gates"]["I2_identity"]["detail"]
    G.build_store(tmp_path / "none", row(*NONE_G), write_recorder=False)
    rep3 = analyze(tmp_path / "none", _rows(NONE_G))
    assert rep3["gates"]["I2_identity"]["status"] == A.GATE_PASS and rep3["provenance"]["recorder"]["tally"] == {"not recorded": 1}
    assert rep3["provenance"]["recorder"]["all_agree"] is False and "not every leg AGREES" in A._recorder_statement(rep3)


# --------------------------------------------------------------------------- (6) titles, the template, the configs
def test_every_figure_title_is_at_most_two_words_and_the_reading_line_fits(tmp_path):
    pytest.importorskip("matplotlib")
    G.build_store(tmp_path, row(*NONE_G))
    G.build_store(tmp_path, row("gemma4e4b", "it", "none"))
    _esc(tmp_path)
    rep = analyze(tmp_path, _rows(NONE_G, ("gemma4e4b", "it", "none"), ESC))
    import matplotlib.pyplot as plt
    for stem, fn in A.figures_for(rep).items():
        fig, _ = fn(rep)
        assert M.figure_title_problem(fig) is None, stem
        bold = [t for t in fig.texts if t.get_fontweight() == "bold"][0].get_text().split("\n")
        assert len(bold) == 2 and len(bold[0].split()) <= 2 and all(len(l) <= 90 for l in bold), (stem, bold)
        assert A.axes_height_fraction(fig) >= A.AXES_MIN_FRACTION, stem
        plt.close(fig)
    with pytest.raises(A.AnalysisError, match="words"):
        A._title2(plt.figure(), "three word title", "reading")
    plt.close("all")


def test_the_results_template_is_pinned_byte_for_byte_and_the_supplement_is_appended_not_written_into_it():
    text = A.RESULTS_TEMPLATE.read_bytes()
    # the immutable skeleton: any edit to the template is a deliberate, reviewed change that re-pins this digest. Re-pinned 2026-10-19 for the
    # figures row alone: it stated one fixed mirror directory, serving path and W&B project for every run, and now carries
    # {{figures_publication}}, which the battery renders from what the run did (the report's `mirrored` / `wandb_url`)
    assert hashlib.sha256(text).hexdigest() == "e75cc08cb05e2af70d0fcb17b68f04c5a1e11a7c11197cdeac341a836e774940"
    assert text.count(b"{{") >= 20
    assert b"## Appendix A" not in text and b"{{noise_floor_table}}" in text


def test_the_canonical_ids_and_the_store_pins_are_the_committed_configs():
    ids = json.loads((REPO / "configs/sdfbase/ifeval/canonical_ids.json").read_text())
    assert len(ids) == IC.IFEVAL_N == 541 and len(set(ids)) == 541 and all(isinstance(i, str) and i.isdigit() for i in ids)
    assert ids == sorted(ids, key=lambda x: (len(x), x)) and A.load_ids(REPO / "configs/sdfbase/ifeval/canonical_ids.json") == ids
    pins = A.load_store_pins(REPO / "configs/sdfbase/ifeval/store_pins_e43f81f.json")
    launch = {V.cell_key(r["model_key"], r["view"], r["cell"]) for r in ROWS if r["status"] == IC.STATUS_LAUNCH}
    assert len(pins) == 52 and {k.rsplit("@", 1)[0] for k in pins} == launch and all(len(v) == 40 for v in pins.values())
    by_cap = {}
    for k, v in pins.items():
        by_cap.setdefault(k.rsplit("@", 1)[1], set()).add(v[:8])
    assert by_cap == {"2048": {"4d389d87", "c629e506"}, "4096": {"44395207", "5c816cf7"}, "8192": {"e43f81f2"}}
    assert sum(1 for k in pins if k.endswith("@2048")) == 30 and sum(1 for k in pins if k.endswith("@4096")) == 14 and sum(1 for k in pins if k.endswith("@8192")) == 8
    certs = A.load_certifications(REPO / "configs/sdfbase/ifeval/nemotron_certifications_e43f81f.json")
    assert {t[:8] for t in certs} == {"4d389d87", "5c816cf7"} and all(rec["admitted"] for by in certs.values() for rec in by.values())
