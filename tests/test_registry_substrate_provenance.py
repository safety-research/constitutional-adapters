"""Substrate provenance across the legacy-store readers (2026-09-21).

The defect: ``derive.substrate_from_path(path, recorded=<bare value>)`` returned the value stamped ``recorded`` -- the strongest
label -- and the readers handed it constants (``_substrate="cluster_a"`` on every alphasteer / gsm8k / math500 / relu_native row,
``"cluster_b"`` on every Llama row, ``"modal"`` on relu_paired's), whether or not any record said so. Several were wrong: every
bcb_bfcl store manifest names a Modal volume, nine gsm8k entries and one math500 entry ran on cluster B, the alphasteer ``ladder/``
rungs stood up on cluster B while ``ladder_canon/`` did on cluster A. The rule now (a null makes a reader check, a wrong value does
not): ``recorded`` only for a producer FIELD that literally names the substrate; a path / routing-table / job-id prefix is
``derived``; a reader's constant is an ``assumed_default`` with its basis in ``field_confidence_json``; nothing decides
-> ``substrate = grading_host`` with ``substrate_source = unverified`` and the reason (the column admits no ``unverified`` value).

Per changed reader, one test on its synthetic fixture asserts the run's ``substrate``, ``substrate_source`` and the confidence
entry's evidence; a repo-wide AST test forbids a bare string literal in the recorded slot in every reader.
"""
from __future__ import annotations

import ast
import glob
import json
import os

import pytest

from sisteer.registry import derive as D
from sisteer.registry.adapters import declared_common as DC
from sisteer.registry import synth

import registry_declared_fixtures as F  # noqa: E402
from test_registry_adapter_declared import EM_PANEL, NOW, _by_type, _read_all  # noqa: E402
import _anchor_frame_fixtures as AF  # noqa: E402  (emitter 4: in-memory reads need a primed anchor frame)

ADAPTERS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "sisteer", "registry", "adapters")


def _runs(adapter):
    recs, rep = _read_all(adapter)
    return _by_type(recs)["run"], rep


def _sub(run):
    """(substrate, substrate_source, the confidence entry's evidence text)."""
    fc = run["field_confidence_json"]
    entry = fc.get("substrate") or {}
    return run["substrate"], run["substrate_source"], str(entry.get("source") or entry.get("reason") or "")


def _edit_json(path: str, fn) -> None:
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    fn(d)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(d, fh)


# --------------------------------------------------------------------------------------------
# the emitter: every run carries the evidence; a bare constant is an assumed default; unknown is grading_host / unverified + reason
# --------------------------------------------------------------------------------------------

def test_every_emitted_run_carries_its_substrate_evidence_and_a_bare_constant_is_never_recorded(tmp_path):
    """The shared emitter files a `substrate` confidence entry on EVERY run (v2 wrote one only for unverified), a bare string in the
    `_substrate` slot as `assumed_default` (its `_substrate_basis` the evidence), and a grading-host-only store path as unknown."""
    from sisteer.registry.adapters import gsm8k_p200 as G8
    fx = F.build_gsm8k_report(str(tmp_path / "g8k"))
    ad = G8.Gsm8kP200Adapter(report=fx["path"], later_reports=(), expect_pin=G8.STORE_PIN)
    rows, _ = ad.long_rows(fx["path"])
    # drive the emitter directly with the three spellings on real rows: a bare constant, a grading host mirror path, nothing
    from sisteer.registry.adapters.base import AdapterContext, RecordBatch
    ctx = AdapterContext(root="", now=NOW, producer={"script": "tests.substrate", "code_sha": synth.code_sha("a"), "session": None, "host": "TESTHOST", "substrate": "grading_host"})
    measured = [dict(r) for r in rows if r["status"] == "measured"]
    assert len(measured) == 2
    bare, mirror = measured
    bare.update(_substrate="cluster_a", _substrate_basis="the submit scripts of this source go to cluster A", _substrate_derived=None, _substrate_path="")
    mirror.update(_substrate=None, _substrate_derived=None, _substrate_default=None, _substrate_path="/home/TESTUSER/example-grade/mirror/x", _substrate_reason=None)
    batch = RecordBatch("substrate:test", "test")
    ctx = AF.primed_ctx_for_rows([bare, mirror], ctx, adapter="gsm8k_p200", campaign="test", reader="tests", seed_provenance="none", code_sha="unverified")
    DC.emit_rows(batch, ctx, [bare, mirror], adapter="gsm8k_p200", campaign="test", reader="tests", seed_provenance="none", code_sha="unverified")
    runs = {r["payload"]["campaign_cell_id_legacy"]: r["payload"] for r in batch.records if r["record_type"] == "run"}
    b, m = runs[bare["cell_id"]], runs[mirror["cell_id"]]
    assert (b["substrate"], b["substrate_source"]) == ("cluster_a", "assumed_default")
    assert b["field_confidence_json"]["substrate"]["source"].startswith("the submit scripts of this source go to cluster A")
    assert (m["substrate"], m["substrate_source"]) == ("grading_host", "unverified")
    assert "the grading host mirror" in m["field_confidence_json"]["substrate"]["reason"] and "substrate_source" in m["field_confidence_json"]


# --------------------------------------------------------------------------------------------
# gsm8k_p200: per served store, from the launch / ledger paths; no witness -> unknown (9 of 142 live entries are cluster B)
# --------------------------------------------------------------------------------------------

def test_gsm8k_substrate_is_read_per_leg_from_its_launch_paths(tmp_path):
    from sisteer.registry.adapters import gsm8k_p200 as G8
    fx = F.build_gsm8k_report(str(tmp_path / "g8k"))

    def witness(d):
        entries = d["legs"]
        none_entry = [k for k in entries if k.endswith("/none@8192")][0]
        steered_entry = [k for k in entries if not k.endswith("/none@8192")][0]
        entries[none_entry]["launch"]["serving_config"] = f"/workspace-vast/TESTUSER/sisteer-eval-repo/{fx['pin']}/configs/x.yaml"
        entries[steered_entry]["launch"]["serving_config"] = f"/workspace/TESTUSER/sisteer-eval-repo/{fx['pin']}/configs/x.yaml"
    _edit_json(fx["path"], witness)
    ad = G8.Gsm8kP200Adapter(report=fx["path"], later_reports=(), expect_pin=G8.STORE_PIN)
    runs, rep = _runs(ad)
    assert rep.runs == 2 and rep.refusals == []
    by = {r["campaign_cell_id_legacy"]: _sub(r) for r in runs}
    assert by["none"][:2] == ("cluster_a", "derived") and "launch.serving_config" in by["none"][2] and "/workspace-vast/" in by["none"][2]
    assert by[fx["cell"]][:2] == ("cluster_b", "derived") and "launch.serving_config" in by[fx["cell"]][2]
    # no witness at all: unknown, with the reader's reason and the emitter's; never the old constant
    fx2 = F.build_gsm8k_report(str(tmp_path / "g8k2"))
    ad2 = G8.Gsm8kP200Adapter(report=fx2["path"], later_reports=(), expect_pin=G8.STORE_PIN)
    runs2, _ = _runs(ad2)
    for r in runs2:
        s, src, why = _sub(r)
        assert (s, src) == ("grading_host", "unverified") and "fields name no path on a cluster's shared root" in why


# --------------------------------------------------------------------------------------------
# math500: per served store, from the ledger binding's routing table
# --------------------------------------------------------------------------------------------

def test_math500_substrate_is_read_per_leg_from_its_ledger_binding(tmp_path):
    from sisteer.registry.adapters import math500 as M5
    fx = F.build_math500_report(str(tmp_path / "m5"))
    old = F.build_math500_report(str(tmp_path / "m5_old"), key_suffix="_r4")

    def witness(d):
        for k, entry in d["legs"].items():
            root = "/workspace-vast/TESTUSER" if entry["cell"] == "none" else "/workspace/TESTUSER"
            entry["ledger"] = {"binding": {"routing_table": f"{root}/exp/math500_thinking_on_full22/serving/m5t_x/math500/routing_table.json"}}
    _edit_json(fx["path"], witness)
    ad = M5.Math500Adapter(report=fx["path"], superseded_report=old["path"])
    runs, rep = _runs(ad)
    live = {r["campaign_cell_id_legacy"]: _sub(r) for r in runs if not r["producer_run_key_legacy"].endswith("_r4")}
    assert live["none"][:2] == ("cluster_a", "derived") and "ledger.binding.routing_table" in live["none"][2]
    assert live["constv2-mix-full__lora_r4_ce"][:2] == ("cluster_b", "derived")
    # the superseded revision's entries carry no witness: unknown with the reason
    sup = [_sub(r) for r in runs if r["producer_run_key_legacy"].endswith("_r4")]
    assert sup and all(s[:2] == ("grading_host", "unverified") and "ledger binding / launch fields name no path" in s[2] for s in sup)


# --------------------------------------------------------------------------------------------
# bcb_bfcl: Modal by the source's default (the paper rows' provenance is the mirror); derived from a read manifest's usage ledger
# --------------------------------------------------------------------------------------------

def test_bcb_bfcl_substrate_is_modal_by_default_and_derived_from_a_read_manifest(tmp_path):
    from sisteer.registry.adapters import bcb_bfcl as BB
    import test_registry_declared_bcb_d60 as D60
    F.build_bcb_parquet(str(tmp_path / "an"), "paper_rows_bcb_bfcl_200_synth.parquet")
    ad = BB.BcbBfclAdapter(pattern=os.path.join(str(tmp_path / "an"), "paper_rows_*.parquet"))
    runs, rep = _runs(ad)
    assert rep.runs == 2 and rep.refusals == []
    for r in runs:
        s, src, why = _sub(r)
        # the fixture's `provenance` is on a shared root, but the row's provenance is the mirror and is NOT read: the default stands
        assert (s, src) == ("modal", "assumed_default") and why.startswith("tinycap's Agentic runs serve on Modal") and "no per-run path" in why
    # a fitted row without a sha reads its store manifest: the manifest's usage_ledger on the Modal volume is the per-run witness
    store = str(tmp_path / "store")
    D60.build_parquet(str(tmp_path / "an2"), store, sha=None)
    D60.build_manifest(store, usage_ledger="/vol/tinycap/gen/gemma4e4b-f22/x/thinking_off/synthwave/usage_ledger.jsonl",
                       route={"bcb_sandbox": {"provider": "modal_owned"}})
    ad2 = BB.BcbBfclAdapter(pattern=os.path.join(str(tmp_path / "an2"), "paper_rows_*.parquet"), store_root=store)
    runs2, rep2 = _runs(ad2)
    steered = [r for r in runs2 if r["campaign_cell_id_legacy"] == D60.CELL]
    assert len(steered) == 2 and rep2.refusals == []
    for r in steered:
        s, src, why = _sub(r)
        assert (s, src) == ("modal", "derived") and "Modal volume path" in why and "agrees with the reader's declared default" in why
        assert r["extra"]["store_usage_ledger"].startswith("/vol/tinycap/") and r["extra"]["store_sandbox_provider"] == "modal_owned"
    anchors = [r for r in runs2 if r["campaign_cell_id_legacy"] == "none"]
    assert anchors and all(_sub(r)[:2] == ("modal", "assumed_default") for r in anchors)


# --------------------------------------------------------------------------------------------
# llama8b_wave: derived from the view sidecar's serving / store dir keys; a run the sidecar does not key takes the source's default
# --------------------------------------------------------------------------------------------

def test_llama8b_substrate_is_derived_from_the_sidecar_keys_else_the_source_default(tmp_path):
    from sisteer.registry.adapters import llama8b_wave as LW
    fx = F.build_llama8b(str(tmp_path / "llama"))
    ad = LW.Llama8bWaveAdapter(stage=fx["stage"], mmlu_cache=fx["cache"])
    runs, rep = _runs(ad)
    petri = [r for r in runs if r["eval_id"] == "petri_pooled"]
    mmlu = [r for r in runs if r["eval_id"] == "mmlu"]
    assert petri and mmlu
    for r in petri:
        s, src, why = _sub(r)
        assert (s, src) == ("cluster_b", "derived") and "view_sidecar.json:serving_dirs" in why and "/workspace/<user>" in why
    for r in mmlu:
        s, src, why = _sub(r)
        assert (s, src) == ("cluster_b", "derived") and "view_sidecar.json:store_dirs" in why
    # drop the sidecar's store_dirs: the MMLU runs fall to the source's declared default, said so; the Petri runs still derive
    _edit_json(os.path.join(fx["stage"], "view_sidecar.json"), lambda d: d.__setitem__("store_dirs", {}))
    runs2, _ = _runs(LW.Llama8bWaveAdapter(stage=fx["stage"], mmlu_cache=fx["cache"]))
    for r in runs2:
        s, src, why = _sub(r)
        if r["eval_id"] == "mmlu":
            assert (s, src) == ("cluster_b", "assumed_default") and why.startswith("the Llama runs were served on cluster B") and "not among the sidecar's keys" in why
        else:
            assert (s, src) == ("cluster_b", "derived")


# --------------------------------------------------------------------------------------------
# relu_native: the grades root's cluster tag (derived); the anchors unknown; the GSM8K cards an assumed default with its basis
# --------------------------------------------------------------------------------------------

def test_relu_native_substrate_by_root_tag_anchor_unknown_gsm8k_card_default(tmp_path):
    from sisteer.registry.adapters import relu_native as RN
    allowance, _ = RN.passthrough_anchor_allowance()
    fx = F.build_relu_native(str(tmp_path / "relu"), allowance)
    # the GSM8K native card's root is the untagged `gsm8k_native/` mirror (as the real paired records name it)
    rec = os.path.join(fx["paired"], f"{fx['model']}__gsm8k_native_none__{fx['cell']}.json")
    _edit_json(rec, lambda d: (d["steered"].__setitem__("root", f"{fx['rs']}/gsm8k_native/gsm8k_paired200_paper_objects_llama_psr"),
                               d["anchor"].__setitem__("root", f"{fx['rs']}/gsm8k_native/gsm8k_paired200_paper_objects_llama_psr")))
    ad = RN.ReluNativeAdapter(rs=fx["rs"], paired_root=fx["paired"])
    runs, rep = _runs(ad)
    mt_steered = [r for r in runs if r["eval_id"] == "actorattack" and r["object_id"] not in (None, "none")]
    mt_anchor = [r for r in runs if r["eval_id"] == "actorattack" and r["object_id"] in (None, "none")]
    g8k = [r for r in runs if r["eval_id"] == "gsm8k" and r["object_id"] not in (None, "none")]
    g8k_anchor = [r for r in runs if r["eval_id"] == "gsm8k" and r["object_id"] in (None, "none")]
    assert mt_steered and mt_anchor and g8k and g8k_anchor
    for r in mt_steered:
        s, src, why = _sub(r)
        assert (s, src) == ("cluster_a", "derived") and "cluster tag _bu_" in why and "grade_relu.sh" in why
    for r in mt_anchor:
        s, src, why = _sub(r)
        assert (s, src) == ("grading_host", "unverified") and "carries no cluster tag" in why and "anchor's generating substrate is not on the record" in why
    for r in g8k:
        s, src, why = _sub(r)
        assert (s, src) == ("cluster_a", "assumed_default") and why.startswith("the ReLU native GSM8K ladder was submitted to cluster A")
    for r in g8k_anchor:
        assert _sub(r)[:2] == ("grading_host", "unverified")
    assert RN.substrate_of_root("/home/TESTUSER/rs/grades_relu_native_ch_qwen35_4b", "the grades root")[0] == "cluster_b"
    assert RN.substrate_of_root("/home/TESTUSER/rs/grades_anchor4096_gemma4e4b_it", "the grades root") is None
    assert RN.substrate_of_root("/workspace-vast/TESTUSER/exp/grades_x", "the grades root")[0] == "cluster_a"


# --------------------------------------------------------------------------------------------
# em_misalign: the store's `substrate` field (recorded), its cells_served manifest path (derived), else the source default (assumed)
# --------------------------------------------------------------------------------------------

def _em_adapter(tmp_path, tag: str, **store_over):
    from sisteer.registry.adapters import em_misalign as EM
    root, grade = str(tmp_path / f"figs_{tag}"), str(tmp_path / f"grade_{tag}")
    F.build_em_panel(root)
    pins = F.build_em_pins(str(tmp_path / f"pins_{tag}.json"))
    p = F.build_em_store(grade, "gemma4e4b", legs=[("none", "proxy_vllm+passthrough+interleaved", "aaaa000011112222"),
                                                    ("si_constv2_mix_full__lora_r4_ce", "proxy_vllm+lens+interleaved", "bbbb000011112222")],
                         bank_objects={"constv2-mix-full__lora_r4_ce": F.SHA_C})
    if store_over:
        def over(d):
            for k, v in store_over.items():
                if k == "cells_served_manifest":
                    d["cells_served"]["manifest"] = v
                else:
                    d[k] = v
        _edit_json(p, over)
    return EM.EmMisalignAdapter(root=root, full22_pins=pins, baseline_pins=None, behaviour_source={}, panels=EM_PANEL, grade_root=grade)


def test_em_substrate_recorded_from_the_store_field_derived_from_its_manifest_else_assumed(tmp_path):
    runs, rep = _runs(_em_adapter(tmp_path, "default"))
    assert rep.runs == 2 and rep.refusals == []
    for r in runs:
        s, src, why = _sub(r)
        assert (s, src) == ("cluster_a", "assumed_default") and why.startswith("the EM agentic / organism generations were served on cluster A")
    runs, _ = _runs(_em_adapter(tmp_path, "modal", substrate="modal"))
    for r in runs:
        s, src, why = _sub(r)
        assert (s, src) == ("modal", "recorded") and "cells_full.json:substrate" in why
    runs, _ = _runs(_em_adapter(tmp_path, "manifest", cells_served_manifest="/workspace-vast/TESTUSER/sisteer-eval-repo/x/configs/capeval/m.yaml"))
    for r in runs:
        s, src, why = _sub(r)
        assert (s, src) == ("cluster_a", "derived") and "cells_served.manifest" in why and "/workspace-vast/" in why


# --------------------------------------------------------------------------------------------
# alphasteer_rebake: the rung records' routing tables and the defence summary's sampler-resolution paths (derived); the refit
# multi-turn store unknown (staged on both clusters)
# --------------------------------------------------------------------------------------------

def test_alphasteer_substrate_from_routing_tables_and_sampler_paths_else_unknown(tmp_path):
    from sisteer.registry.adapters import alphasteer_rebake as AS
    fx = F.build_alphasteer(str(tmp_path / "as"), moved=True)
    ad = AS.AlphaSteerRebakeAdapter(store=fx["store"], defence_store=fx["defence_store"], graded_reread=fx["graded"], orb_grades=str(tmp_path / "no_orb"), refit=str(tmp_path / "no_refit"))
    runs, rep = _runs(ad)
    xs = [r for r in runs if r["eval_id"] == "xstest"]
    static = [r for r in runs if r["eval_id"] == "pasttense"]
    assert len(xs) == 3 and static
    for r in xs:
        s, src, why = _sub(r)
        assert (s, src) == ("cluster_a", "derived") and "rung_record.json:served.routing_table" in why
    for r in static:
        s, src, why = _sub(r)
        assert (s, src) == ("cluster_a", "derived") and "alphasteer_defense_summary.json:dose.resolved_sampler.engine_defaults.files_searched" in why
    # no witness in the records: unknown with each site's reason -- never the old constant
    fx2 = F.build_alphasteer(str(tmp_path / "as2"), moved=True, substrate_witness=False)
    ad2 = AS.AlphaSteerRebakeAdapter(store=fx2["store"], defence_store=fx2["defence_store"], graded_reread=fx2["graded"], orb_grades=str(tmp_path / "no_orb"), refit=str(tmp_path / "no_refit"))
    runs2, _ = _runs(ad2)
    for r in runs2:
        s, src, why = _sub(r)
        assert (s, src) == ("grading_host", "unverified"), (r["eval_id"], s, src)
        assert ("names no routing table" in why) or ("sampler resolution read no file" in why), why
    # the refit multi-turn rows: staged on both clusters, so unknown with that reason, the refit default withdrawn
    refit = str(tmp_path / "refit")
    F.build_refit_mt_summary(refit, complete=True)
    ad3 = AS.AlphaSteerRebakeAdapter(store=fx["store"], defence_store=fx["defence_store"], graded_reread=fx["graded"], orb_grades=str(tmp_path / "no_orb"), refit=refit)
    common = dict(_confidence={}, _object_id=F.SHA_REF, _when=NOW, _code_sha="unverified", _strength_coefficient=AS.REFIT_LAMBDA, _strength_nominal_source="derived",
                  _is_operating_point=1, _substrate_default=AS.REFIT_SUBSTRATE_DEFAULT, _extra={})
    mt = ad3._refit_mt_rows(F.SHA_REF, AS.REFIT_LAMBDA, common, {"harness": "bespoke_sisteer", "token_cap_tokens": 4096, "decode": {}, "harness_version": "unverified",
                                                                 "thinking_mode": "not_available", "serving_path": "lens"})
    assert len(mt) == 3 and all(r["_substrate_default"] is None and r["_substrate_reason"] == AS.REFIT_MT_SUBSTRATE_REASON and r["_substrate_path"] == "" for r in mt)
    assert D.substrate_from_path("", None, derived=None, default=None)[1] == "unverified"
    # the claims by kind
    assert AS._routing_claim("/workspace/TESTUSER/exp/objects/x/ladder/lam1/serve/routing_table.json", "rung_record.json:served.routing_table")[0] == "cluster_b"
    assert AS._routing_claim("/home/TESTUSER/mirror/routing_table.json", "x") is None and AS._routing_claim(None, "x") is None
    assert AS._cluster_field_claim({"cluster": "cluster_a"}, "orbench_graded.json:cluster") == ("orbench_graded.json:cluster", "cluster_a")
    assert AS._cluster_field_claim({"cluster": "node-8"}, "x") is None and AS._cluster_field_claim({}, "x") is None


# --------------------------------------------------------------------------------------------
# strength_ladder: the node / job / config rule is a derivation
# --------------------------------------------------------------------------------------------

def test_strength_ladder_substrate_rule_is_derived(tmp_path):
    from sisteer.registry.adapters import strength_ladder as SL
    import test_registry_declared_strength_d60 as S60
    root = tmp_path / "strength"
    fx = F.build_strength(str(root / "lora"))                 # a record directory under the reader's root, as the strength tests build it
    S60._d60ify(fx["path"], S60.LINEAGE_ONE)
    runs, rep = _runs(SL.StrengthLadderAdapter(root=str(root)))
    assert runs and rep.refusals == []
    for r in runs:
        s, src, why = _sub(r)
        if r["eval_id"] == "mmlu":
            continue     # the MMLU dose table's rows are read from another record (its own witness, tested where that record is built)
        assert (s, src) == ("modal", "derived") and "name Modal" in why, (r["eval_id"], s, src, why)
    assert SL._substrate_of({"node": "node-13", "config": "/workspace-vast/TESTUSER/configs/x.yaml"}) == ("cluster_a", "the battery entry's node node-13 with its config on the /workspace-vast/ shared root")
    assert SL._substrate_of({"node": "node-13", "config": "/workspace/TESTUSER/configs/x.yaml"})[0] == "cluster_b"
    assert SL._substrate_of({"node": "node-13", "config": "/home/TESTUSER/x.yaml"}) is None and SL._substrate_of({}) is None


# --------------------------------------------------------------------------------------------
# repo-wide: no reader passes a bare string literal into the recorded slot
# --------------------------------------------------------------------------------------------

def _bare_literal_sites():
    """``{file: [line, ...]}`` of every `_substrate=<str literal>` keyword and every `substrate_from_path(<path>, <str literal>)` /
    `recorded=<str literal>` call under adapters/."""
    out = {}
    for path in sorted(glob.glob(os.path.join(ADAPTERS_DIR, "*.py"))):
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
        hits = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg == "_substrate" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    hits.append(kw.value.lineno)
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else "")
            if name == "substrate_from_path":
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                    hits.append(node.args[1].lineno)
                for kw in node.keywords:
                    if kw.arg == "recorded" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        hits.append(kw.value.lineno)
        if hits:
            out[os.path.basename(path)] = sorted(hits)
    return out


def test_no_reader_passes_a_bare_string_literal_into_the_recorded_slot():
    """Every reader, relu_paired included (its own branch landed as #304 and moved onto this contract at v9)."""
    assert _bare_literal_sites() == {}


def test_helper_signature_never_returns_recorded_for_a_bare_value():
    for value in ("cluster_a", "cluster_b", "modal", "grading_host", "runpod"):
        assert D.substrate_from_path("", value)[1] == "assumed_default"
        assert D.substrate_from_path("/home/TESTUSER/mirror", value)[1] == "assumed_default"
        assert D.substrate_from_path("", recorded=value)[1] == "assumed_default"
    assert D.substrate_from_path("", ("drive.log:cluster", "modal"))[1] == "recorded"
