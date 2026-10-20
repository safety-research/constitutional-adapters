"""The site record's per-window S is sized from the launches whose adapter class DOMINATES the served standup, not from every launch:
the s2r pair target 831867 served max_num_seqs=1 at 16,384 and died at `selection_honored` (a mixed batch is impossible at one
sequence per step) because the LoRA lane's rank-64 probe generation (821376) had written `*_r64_S1_cap128_L16384` launches admitted at
S=1 into the record and `unsteered_site_set` took the least admitted S over EVERY launch — a rank-64 x 128-adapter ceiling applied to
a rank-1 x 3-adapter standup, while the r4 / cap8 launches at that window admitted 14.

Pinned here: the probe fold writes each passed launch's engine class (`max_lora_rank`, `max_loras`) beside S / L and refuses by name a
passed launch without one; `least_admitted_by_window` with a served class takes the least admitted S over the tightest DOMINATING class
(16,384 -> 14; 49,152 stays 4 — conservative across sets), stays record-wide with no served class or on an unclassed record (noted),
and refuses a window nothing dominates or a half-classed record; `budget_for` / `resolve_geometry` carry `served_lora`; the native
standup records a `launch_class` adjustment (the one that may RAISE S) and leaves an explicit S alone; `is_hybrid` spans both tuples and
the model-facts snippet reads it. Synthetic record (the toy probe of test_site_geometry_overlay.py)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402

from sisteer.capeval.serving import geometry as g  # noqa: E402
from sisteer.capeval.serving import native_standup as ns  # noqa: E402

KEY = "qwen35_4b_demo"
GEN = "f" * 64
WO, QKVO = ("o_proj",), ("k_proj", "o_proj", "q_proj", "v_proj")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in (g.SITE_GEOMETRY_DIR_ENV, g.SITE_DIR_ENV, g.SHARED_ROOT_ENV):
        monkeypatch.delenv(v, raising=False)


def _budget(S=16, L=16384) -> g.ModelBudget:
    return g.ModelBudget(model_key=KEY, batched_token_budget=S * L, certified=True, source=g.MEASURED_BUDGET_SOURCE,
                         basis=g.BudgetBasis(method=f"{g.SITE_GEOMETRY_METHOD}: test", max_model_len=L, max_num_seqs=S, gpu_memory_utilization=0.9,
                                             kv_tokens_reported=S * L + 16, probe_job_id="11111111", measured_on="2026-10-02", source="test",
                                             first_refused=None, note="synthetic"))


def _launch(L, S, rank=None, loras=None):
    d = {"L": L, "S": S, "batched_tokens": S * L}
    if rank is not None:
        d["max_lora_rank"] = rank
    if loras is not None:
        d["max_loras"] = loras
    return d


def _set(sid, modules, D, launches, *, budget_S=16, budget_L=16384) -> dict:
    """One set's record; the budget must be one of its launches' products (the loader checks), so it is named per set."""
    first = next(iter(launches))
    return {"lora_modules": list(modules), "lora_max_module_dim": D, "budget": _budget(budget_S, budget_L),
            "allowed_pairs": [[4, sid]], "certified_by": [first], "module_dim_launches": {first: D}, "budget_launches": dict(launches),
            "adapter_parity": g.adapter_parity_block([{"rank": 4, "set": sid, "status": g.PARITY_VERIFIED, "basis": "hf_peft", "mismatched_adapters": []}])}


# the live record's shape in miniature: at 16,384 the r4/cap8 class admitted 14-16 and the r64/cap128 class 1; at 49,152 the r4/cap8 class
# admitted 4 (qkvo) / 5 (wo) and the r64/cap64 class 4
CLASSED = {
    "wo": _set("wo", WO, 4096, {"wo_r4_single_cap8_L16384": _launch(16384, 16, 8, 4), "wo_r64_single_cap128_L16384": _launch(16384, 1, 128, 128),
                                "wo_r4_single_cap8_L49152": _launch(49152, 5, 8, 4), "wo_r64_single_cap64_L49152": _launch(49152, 4, 64, 64)}),
    "qkvo": _set("qkvo", QKVO, 9216, {"qkvo_r4_S1_cap8_L16384": _launch(16384, 14, 8, 4), "qkvo_r64_S1_cap128_L16384": _launch(16384, 1, 128, 128),
                                      "qkvo_r4_S1_cap8_L49152": _launch(49152, 4, 8, 4), "qkvo_r64_S1a_cap64_L49152": _launch(49152, 4, 64, 64)},
                 budget_S=14),       # 14 x 16,384 = 229,376: the set's largest launch product (the live qkvo / attn_mlp sets' number)
}
UNCLASSED = {sid: _set(sid, s["lora_modules"], s["lora_max_module_dim"], {lid: {k: v for k, v in bl.items() if k in ("L", "S", "batched_tokens")}
                                                                            for lid, bl in s["budget_launches"].items()},
                       budget_S=s["budget"].batched_token_budget // 16384) for sid, s in CLASSED.items()}


def _record(sets) -> dict:
    rec = g.site_geometry_record(model_key=KEY, hf_id="org/toy", revision="0" * 40, sets=sets, written_utc="2026-10-02T00:00:00Z",
                                 provenance={"probe_job_id": "11111111", "probe_record": "/shared/exp/multilora_cert/toy_probe_11111111",
                                             "probe_cluster": "cluster_a", g.RESULT_GENERATION_KEY: GEN,
                                             "attestation_sha256": {"wo_r4_single_cap8_L16384": {"file": None, "sha256": "ab" * 32}}, "host": "TESTHOST"})
    rec[g.STAGING_VERIFICATION_KEY] = {"verified_utc": "2026-10-02T00:00:01Z", "record_generation": GEN, "verifier_host": "TESTHOST"}
    return rec


def _install(tmp_path: Path, monkeypatch, sets) -> Path:
    d = tmp_path / "geometry"; d.mkdir(parents=True, exist_ok=True)
    (d / f"{KEY}.json").write_text(json.dumps(_record(sets)))
    monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(d))
    return d


def _all_launches(sets):
    return {lid: bl for s in sets.values() for lid, bl in s["budget_launches"].items()}


def test_launch_class_reads_only_positive_ints_and_never_guesses():
    assert g.launch_class({"L": 1, "S": 1, "max_lora_rank": 8, "max_loras": 4}) == (8, 4)
    for bad in ({"L": 1, "S": 1}, {"max_lora_rank": 8}, {"max_lora_rank": 0, "max_loras": 4}, {"max_lora_rank": True, "max_loras": 4}, {"max_lora_rank": "8", "max_loras": 4}):
        assert g.launch_class(bad) is None, bad


def test_the_tightest_dominating_class_sizes_each_window_and_the_record_wide_rule_stays_for_no_class():
    L = _all_launches(CLASSED)
    ws, src, chosen = g.least_admitted_by_window(KEY, L, (1, 3))
    assert ws == {16384: 14, 49152: 4} and src == "launch_class:1x3"
    assert chosen[16384] == ("qkvo_r4_S1_cap8_L16384",) and chosen[49152] == ("qkvo_r4_S1_cap8_L49152",)       # conservative ACROSS sets at the tight class
    ws0, src0, chosen0 = g.least_admitted_by_window(KEY, L, None)
    assert ws0 == {16384: 1, 49152: 4} and src0 == g.WINDOW_SEQS_RECORD_WIDE and set(chosen0[16384]) == {"wo_r64_single_cap128_L16384", "qkvo_r64_S1_cap128_L16384"}
    L16 = {lid: bl for lid, bl in L.items() if int(bl["L"]) == 16384}
    ws64, src64, _ = g.least_admitted_by_window(KEY, L16, (64, 128))
    assert ws64 == {16384: 1} and src64 == "launch_class:64x128"                  # the big class IS capped by its own launches
    with pytest.raises(g.SiteGeometryError, match="no probed launch at window 49152 dominates the served adapter class"):
        g.least_admitted_by_window(KEY, L, (64, 128))                              # at 49,152 only cap64 launches exist: 128 adapters are not certified there
    with pytest.raises(g.SiteGeometryError, match="no probed launch at window 16384 dominates the served adapter class"):
        g.least_admitted_by_window(KEY, L, (256, 3))
    # an UNCLASSED record with a served class falls back to the record-wide rule, and SAYS so
    wsu, srcu, _ = g.least_admitted_by_window(KEY, _all_launches(UNCLASSED), (1, 3))
    assert wsu == {16384: 1, 49152: 4} and srcu.startswith(g.WINDOW_SEQS_RECORD_WIDE) and "unrecorded" in srcu
    # a HALF-classed record refuses by name
    half = dict(L); half["wo_r4_single_cap8_L49152"] = _launch(49152, 5); half["wo_r64_single_cap64_L49152"] = _launch(49152, 4)
    half["qkvo_r4_S1_cap8_L49152"] = _launch(49152, 4); half["qkvo_r64_S1a_cap64_L49152"] = _launch(49152, 4)
    with pytest.raises(g.SiteGeometryError, match="no launch at window 49152 carries an adapter class"):
        g.least_admitted_by_window(KEY, half, (1, 3))
    with pytest.raises(ValueError, match="served_lora must be"):
        g.least_admitted_by_window(KEY, L, (0, 3))


def test_the_served_class_reaches_the_budget_and_the_resolver_and_the_record_wide_row_is_unchanged(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, CLASSED)
    rb0 = g.budget_for(KEY)
    assert rb0.batched_token_budget == 229_376          # the MINIMUM over the sets: qkvo's 14 x 16,384
    assert rb0.module_set == "wo" and rb0.entry.window_seqs == {16384: 1, 49152: 4} and rb0.entry.window_seqs_source == g.WINDOW_SEQS_RECORD_WIDE
    assert rb0.entry.seqs_at(16384) == 1 and rb0.entry.seqs_at(49152) == 4 and "record-wide" in rb0.entry.basis.method
    rb = g.budget_for(KEY, served_lora=(1, 3))
    assert rb.entry.window_seqs == {16384: 14, 49152: 4} and rb.entry.window_seqs_source == "launch_class:1x3"
    assert rb.entry.seqs_at(16384) == 14 and rb.entry.seqs_at(49152) == 4 and rb.entry.window_seqs_launches[16384] == ("qkvo_r4_S1_cap8_L16384",)
    assert "launch_class:1x3" in rb.entry.basis.method
    assert "window_seqs_source" not in rb.entry.as_dict() and "window_seqs_launches" not in rb.entry.as_dict()     # read-side provenance only
    geom0 = g.resolve_geometry("multiturn_attacks", max_model_len=16384, model_key=KEY, strict_identity=True)
    geom = g.resolve_geometry("multiturn_attacks", max_model_len=16384, model_key=KEY, strict_identity=True, served_lora=(1, 3))
    assert (geom0.max_num_seqs, geom0.seqs_source) == (1, "model_budget") and (geom.max_num_seqs, geom.seqs_source) == (14, "model_budget")
    assert g.resolve_geometry("multiturn_attacks", max_model_len=49152, model_key=KEY, strict_identity=True, served_lora=(1, 3)).max_num_seqs == 4
    with pytest.raises(g.SiteGeometryError, match="dominates the served adapter class"):
        g.budget_for(KEY, served_lora=(256, 3))


def test_the_native_standup_records_a_launch_class_adjustment_and_leaves_an_explicit_S_alone(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, CLASSED)
    geom0 = g.resolve_geometry("multiturn_attacks", max_model_len=16384, model_key=KEY, strict_identity=True)
    geom, rec = ns.launch_class_geometry(geom0, served_lora=(1, 3), eval_name="multiturn_attacks", model_key=KEY)
    assert geom.max_num_seqs == 14 and geom.seqs_source == "override" and geom.max_model_len == 16384
    assert rec == {"kind": g.ADJUSTMENT_LAUNCH_CLASS, "from_S": 1, "to_S": 14, "max_lora_rank": 1, "max_loras": 3, "L": 16384,
                   "window_seqs_source": "launch_class:1x3", "launches": ["qkvo_r4_S1_cap8_L16384"]}
    assert g.ADJUSTMENT_LAUNCH_CLASS == "launch_class" and rec["kind"] not in (ns.ADJUSTMENT_KERNEL, ns.ADJUSTMENT_KV)
    # the same S -> no step; an explicit S -> the caller's, untouched; nothing dominates -> refused by name
    same, rec2 = ns.launch_class_geometry(g.resolve_geometry("multiturn_attacks", max_model_len=49152, model_key=KEY, strict_identity=True),
                                          served_lora=(1, 3), eval_name="multiturn_attacks", model_key=KEY)
    assert rec2 is None and same.max_num_seqs == 4
    explicit = g.resolve_geometry("multiturn_attacks", max_model_len=16384, max_num_seqs=2, model_key=KEY, strict_identity=False)
    assert ns.launch_class_geometry(explicit, served_lora=(1, 3), eval_name="multiturn_attacks", model_key=KEY) == (explicit, None)
    assert ns.launch_class_geometry(geom0, served_lora=None, eval_name="multiturn_attacks", model_key=KEY) == (geom0, None)
    with pytest.raises(ns.StandupRefused, match="dominates the served adapter class"):
        ns.launch_class_geometry(geom0, served_lora=(256, 3), eval_name="multiturn_attacks", model_key=KEY)
    # an UNCLASSED record: the same S comes back (no step) — the standup's S < 2 refusal then speaks (pinned on the source text below)
    _install(tmp_path, monkeypatch, UNCLASSED)
    g1 = g.resolve_geometry("multiturn_attacks", max_model_len=16384, model_key=KEY, strict_identity=True)
    assert ns.launch_class_geometry(g1, served_lora=(1, 3), eval_name="multiturn_attacks", model_key=KEY) == (g1, None) and g1.max_num_seqs == 1
    src = (REPO / "src/sisteer/capeval/serving/native_standup.py").read_text()
    assert "if int(geom.max_num_seqs) < 2 and len(all_blocks) >= 2:" in src and "selection_honored) needs two" in src
    assert src.index("THE LAUNCH CLASS, BEFORE THE KERNEL CEILING") < src.index("THE LORA KERNEL INDEX CEILING, BEFORE THE LAUNCH (the second")


def _serving_probe():
    name = "serving_probe_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts/sdfbase/serving_probe.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod            # dataclasses resolve the defining module through sys.modules; an unregistered module breaks them
    spec.loader.exec_module(mod)      # type: ignore[union-attr]
    return mod


def test_the_probe_fold_writes_the_launch_class_beside_S_and_L_and_refuses_a_classless_passed_launch():
    sp = _serving_probe()
    probs: list = []
    assert sp.launch_class_fields({"engine": {"max_lora_rank": 8, "max_loras": 4, "max_num_seqs_engine": 14}}, "x_L16384", probs) == {"max_lora_rank": 8, "max_loras": 4} and probs == []
    assert sp.launch_class_fields({"engine": {"max_lora_rank": 8, "max_loras": None}}, "y_L16384", probs) is None
    assert probs == ["y_L16384: passed but the engine reported no positive max_loras (None) — the launch's adapter class cannot be folded, and is never guessed"]
    src = (REPO / "scripts/sdfbase/serving_probe.py").read_text()
    assert 'st["budget_launches"][rec["id"]] = {"S": int(eng), "L": L, "batched_tokens": int(eng) * L, **cls}' in src


def test_is_hybrid_spans_both_tuples_and_the_model_facts_snippet_reads_it(monkeypatch):
    assert g.is_hybrid("qwen35_4b") and g.is_hybrid("nemotron3nano4b") and not g.is_hybrid("llama8b") and g.is_hybrid(KEY)   # the demo serves the paper's Qwen3.5-4B: a hybrid (ONBOARDED_HYBRID_MODEL_KEYS)
    assert g.ONBOARDED_HYBRID_MODEL_KEYS == (KEY,) and KEY in g.ONBOARDED_MODEL_KEYS and KEY not in g.HYBRID_MODEL_KEYS   # listed as an ONBOARDED hybrid, never appended to the sweep's tuple
    # an onboarded hybrid is listed in ONBOARDED_HYBRID_MODEL_KEYS, never appended to the sweep's tuple, and is_hybrid reads BOTH: a synthetic
    # onboarded key is dense until that tuple names it
    assert not g.is_hybrid("synth_onboarded")
    monkeypatch.setattr(g, "ONBOARDED_HYBRID_MODEL_KEYS", (KEY, "synth_onboarded"))
    assert g.is_hybrid("synth_onboarded") and g.is_hybrid(KEY) and "synth_onboarded" not in g.HYBRID_MODEL_KEYS and KEY not in g.HYBRID_MODEL_KEYS
    sh = (REPO / "scripts/sdfbase/full22/common.sh").read_text()
    assert "from sisteer.capeval.serving.geometry import HYBRID_SEQS_CAP, is_hybrid" in sh and "hybrid = is_hybrid(key)" in sh
    assert "hybrid = key in HYBRID_MODEL_KEYS" not in sh


def test_a_seven_slot_class_never_dominates_the_eight_slot_served_class_and_every_probed_window_must_be_covered():
    """The probe owner's ruling of 2026-10-11 (the battery's steered standups, three units refused at standup in a row): a steered ``wo``
    standup serves (rank cap 8, 8 adapters) — control r4, constitution r4, difference r8, the four carriers and ``none__lora0`` — and the
    paper-pair probe had folded only the split servers' classes, (8, 7) and (8, 2). Dominance is rank cap >= AND slots >= at the SAME
    window: (8, 7) never dominates (8, 8), a wider cap buys no slot (the cap-128 ladder re-pin keeps 7), and the rule covers EVERY
    probed window of the record — an (8, 8) launch at four windows leaves the fifth refused by name, whatever the standup's own
    window. The union's launches at every window are picked at every window; the record-wide row is unchanged."""
    W = (4096, 16384, 49152, 65536, 98304)
    split = {}
    for L in W:
        split[f"wo_r4_S1cc_cap8_L{L}"] = _launch(L, 64 if L == 4096 else 5, 8, 7)
        split[f"wo_r4_S1d_cap8_L{L}"] = _launch(L, 64 if L == 4096 else 5, 8, 2)
    with pytest.raises(g.SiteGeometryError, match=r"no probed launch at window 4096 dominates the served adapter class \(rank cap 8, 8 adapters\): "
                                                  r"probed classes \[\(8, 2\), \(8, 7\)\]"):
        g.least_admitted_by_window(KEY, split, (8, 8))
    with pytest.raises(g.SiteGeometryError, match="no probed launch at window 4096 dominates"):
        g.least_admitted_by_window(KEY, {**split, "wo_r4_S1cc_cap128_L4096": _launch(4096, 64, 128, 7)}, (8, 8))
    # the split servers DO size the classes they certify
    ws7, src7, chosen7 = g.least_admitted_by_window(KEY, split, (8, 7))
    assert ws7 == {4096: 64, 16384: 5, 49152: 5, 65536: 5, 98304: 5} and src7 == "launch_class:8x7"
    assert chosen7 == {L: (f"wo_r4_S1cc_cap8_L{L}",) for L in W}
    # the union (8, 8) at every window: the tightest dominating class, picked at every window
    union = {f"wo_r4_single_cap8_L{L}": _launch(L, 60 if L == 4096 else 4, 8, 8) for L in W}
    ws, src, chosen = g.least_admitted_by_window(KEY, {**split, **union}, (8, 8))
    assert ws == {4096: 60, 16384: 4, 49152: 4, 65536: 4, 98304: 4} and src == "launch_class:8x8"
    assert chosen == {L: (f"wo_r4_single_cap8_L{L}",) for L in W}
    ws0, src0, _ = g.least_admitted_by_window(KEY, {**split, **union}, None)
    assert ws0 == ws and src0 == g.WINDOW_SEQS_RECORD_WIDE
    # four of five windows covered: the fifth refuses by name (no window transfer)
    four = {lid: bl for lid, bl in union.items() if int(bl["L"]) != 65536}
    with pytest.raises(g.SiteGeometryError, match="no probed launch at window 65536 dominates the served adapter class"):
        g.least_admitted_by_window(KEY, {**split, **four}, (8, 8))
    ws4, _, chosen4 = g.least_admitted_by_window(KEY, {**split, **four}, (8, 7))        # the (8, 7) class is still sized everywhere
    assert ws4 == ws7 and chosen4 == chosen7
