"""The SITE geometry overlay, keyed (model_key, module_set).

The FD 24 serving probe's verdict folds, PER ALLOWED MODULE SET, the measured widest executing-module dim and the certified
batched-token budget (with provenance) into ``$SISTEER_GEOMETRY_DIR/<model_key>.json`` (else the front door's
``$SISTEER_SITE_DIR/geometry`` or ``$SISTEER_SHARED_ROOT/exp/reproduce/geometry``); ``geometry.rows_for(key, module_set)`` /
``budget_for(key, module_set)`` / ``kernel_ceiling_adjustment(key, L, S, module_set=)`` resolve the SELECTED set's record
BEFORE the built-in tables. A native standup's ceiling is the D of the class it registered, so a model-wide maximum over
the allowed sets would make the padded preflight and the post-standup pins expect the WRONG S for every design narrower
than the widest allowed set — the CLOSE finding. CPU-only, synthetic records (placeholder ids, hostname TESTHOST).

Covered here (the probe → overlay → native standup → both pins path itself is in tests/test_serving_probe_smoke.py):
- the directory rule (explicit > SISTEER_GEOMETRY_DIR > SISTEER_SITE_DIR/geometry > SISTEER_SHARED_ROOT/exp/reproduce/geometry
  > none), never $HOME;
- absent file = None (the built-in tables stand); a present file that is malformed / another model's / version 1 (the retired
  model-wide shape) / inconsistent / short on provenance is a loud SiteGeometryError from every resolver;
- the SELECTED set's record shadows the built-in row for ANY known key; no set named = the built-in tables (an onboarded
  model has no row of its own: the Llama-certified default, which steered padding refuses); a set the file does not certify REFUSES by name; the summary never
  sizes anything; the built-in tables are untouched;
- the padded preflight selects the set from the ONE source (the models table's promoted `lora_architecture`; a pending pick
  refuses by name; a model outside the table sizes from the built-in tables), and both padded launchers' geometry pins
  (rendered from the sbatch heredocs, run against synthetic endpoint records) resolve the selected set's record and refuse a
  provisional budget / an uncertified set / a standup of another class;
- geometry_adjustment_problem re-derives a per-set model's kernel step from the step's OWN D, which must be a certified one.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sisteer.capeval.serving import geometry as g
from sisteer.capeval.serving import serving_attestation as sa

REPO = Path(__file__).resolve().parents[1]
PAD = REPO / "scripts/sdfbase/full22/static_padding"
sys.path.insert(0, str(PAD))
import padding_preflight as pf  # noqa: E402

KEY = "qwen35_4b_demo"   # the one ONBOARDED key on this tree (geometry.ONBOARDED_MODEL_KEYS): no built-in budget row, its geometry is the overlay's
LIMIT = 2**31 - 1
WO, QKVO = ("o_proj",), ("k_proj", "o_proj", "q_proj", "v_proj")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in (g.SITE_GEOMETRY_DIR_ENV, g.SITE_DIR_ENV, g.SHARED_ROOT_ENV, pf.MODELS_TABLE_ENV):
        monkeypatch.delenv(v, raising=False)


def _budget(key=KEY, S=8, L=49152, *, certified=True, source=g.MEASURED_BUDGET_SOURCE) -> g.ModelBudget:
    return g.ModelBudget(model_key=key, batched_token_budget=S * L, certified=certified, source=source,
                         basis=g.BudgetBasis(method=f"{g.SITE_GEOMETRY_METHOD}: test", max_model_len=L, max_num_seqs=S,
                                             gpu_memory_utilization=0.9, kv_tokens_reported=S * L + 16,
                                             probe_job_id="11111111", measured_on="2026-09-30", source="test",
                                             first_refused=None, note="synthetic"))


GEN = "f" * 64   # the synthetic result generation every fixture overlay was "folded over"


def _set(sid, modules, D, S=8, L=49152, parity=g.PARITY_VERIFIED, **kw) -> dict:
    """One set's record over its one pair (4, sid); ``parity`` is that pair's adapter-parity status — `verified` by default, the
    state every earlier fixture was implicitly in (the menu only ever held verified pairs), so the steered pins below pass."""
    return {"lora_modules": list(modules), "lora_max_module_dim": D, "budget": _budget(KEY if "key" not in kw else kw.pop("key"), S, L, **kw),
            "allowed_pairs": [[4, sid]], "certified_by": [f"{sid}_r4_x_L4096"],
            "module_dim_launches": {f"{sid}_r4_x_L4096": D}, "budget_launches": {f"{sid}_r4_x_L{L}": {"S": S, "L": L, "batched_tokens": S * L}},
            "adapter_parity": g.adapter_parity_block([{"rank": 4, "set": sid, "status": parity, "basis": (None if parity == g.PARITY_UNVERIFIED else "hf_peft"),
                                                        "mismatched_adapters": (["control__lora"] if parity == g.PARITY_MISMATCH else [])}])}


def _record(key=KEY, sets=None) -> dict:
    sets = sets if sets is not None else {"wo": _set("wo", WO, 4096), "qkvo": _set("qkvo", QKVO, 9216, S=2, L=98304)}
    if key != KEY:
        for s_ in sets.values():
            b = s_["budget"]
            s_["budget"] = g.ModelBudget(model_key=key, batched_token_budget=b.batched_token_budget, certified=b.certified, source=b.source, basis=b.basis)
    rec = g.site_geometry_record(model_key=key, hf_id="org/toy", revision="0" * 40, sets=sets, written_utc="2026-09-30T00:00:00Z",
                                 provenance={"probe_job_id": "11111111", "probe_record": "/shared/exp/multilora_cert/toy_probe_11111111",
                                             "probe_cluster": "cluster_a", g.RESULT_GENERATION_KEY: GEN,
                                             "attestation_sha256": {"wo_r4_x_L4096": {"file": None, "sha256": "ab" * 32}}, "host": "TESTHOST"})
    # a STAGED copy: its record lives on another cluster (unreachable from here), so it carries the staging verification for
    # its own generation — the only form an unreachable-record overlay may take
    rec[g.STAGING_VERIFICATION_KEY] = {"verified_utc": "2026-09-30T00:00:01Z", "record_generation": GEN, "verifier_host": "TESTHOST"}
    return rec


def _write(tmp_path: Path, rec: dict, key=KEY) -> Path:
    d = tmp_path / "geometry"; d.mkdir(parents=True, exist_ok=True)
    (d / f"{key}.json").write_text(json.dumps(rec)); return d


def _table(tmp_path: Path, modules, *, selected_by="manual", key=KEY) -> Path:
    mods = "null" if modules is None else "[" + ", ".join(modules) + "]"
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "models.yaml"
    p.write_text(f"schema: 1\nmodels:\n  {key}:\n    lora_architecture: {{rank: 4, modules: {mods}, per_expert_rank: null, "
                 f"selected_by: {selected_by}, selection_ruling_id: r1}}\n")
    return p


# --------------------------------------------------------------------------------------------- the directory rule


def test_directory_rule_explicit_env_site_shared_then_none(monkeypatch, tmp_path):
    assert g.site_geometry_dir() is None and g.site_geometry_path(KEY) is None
    monkeypatch.setenv(g.SHARED_ROOT_ENV, "/shared/u")
    assert g.site_geometry_dir() == Path("/shared/u/exp/reproduce/geometry")
    monkeypatch.setenv(g.SITE_DIR_ENV, "/site")
    assert g.site_geometry_dir() == Path("/site/geometry")
    monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, "/explicit")
    assert g.site_geometry_dir() == Path("/explicit") and g.site_geometry_path(KEY) == Path(f"/explicit/{KEY}.json")
    assert g.site_geometry_dir(tmp_path) == tmp_path
    # never $HOME
    for v in (g.SITE_GEOMETRY_DIR_ENV, g.SITE_DIR_ENV, g.SHARED_ROOT_ENV):
        monkeypatch.delenv(v)
    monkeypatch.setenv("HOME", str(tmp_path)); (tmp_path / ".config/sisteer/geometry").mkdir(parents=True)
    (tmp_path / ".config/sisteer/geometry" / f"{KEY}.json").write_text(json.dumps(_record()))
    assert g.site_geometry_dir() is None and g.load_site_geometry(KEY) is None
    assert g.budget_for(KEY, "wo").source == g.DEFAULT_BUDGET_SOURCE


def test_absent_overlay_leaves_every_built_in_answer_unchanged(monkeypatch, tmp_path):
    monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(tmp_path / "empty"))
    assert g.load_site_geometry(KEY) is None and g.site_geometry_for(KEY, "wo") is None and g.rows_for(KEY, "wo").site is None
    assert g.budget_for(KEY).source == g.DEFAULT_BUDGET_SOURCE and g.budget_for(KEY, "wo").source == g.DEFAULT_BUDGET_SOURCE
    assert KEY in g.ONBOARDED_MODEL_KEYS and KEY not in g.ONBOARDED_BUDGET_BY_MODEL and KEY not in g.BUDGET_BY_MODEL     # no built-in row of its own
    assert g.budget_for("qwen35_4b", "wo").measured and g.lora_max_module_dim_for("qwen35_4b", "wo") == 4096 == g.lora_max_module_dim_for("qwen35_4b")
    assert g.lora_max_module_dim_for(KEY) is None and g.lora_max_module_dim_for(KEY, "wo") is None and g.certified_module_dims(KEY) == {}
    with pytest.raises(g.UnknownModelKey, match="no pinned LoRA max module dim for module set 'wo'.*site geometry overlay absent at"):
        g.kernel_ceiling_adjustment(KEY, 4096, 10, module_set="wo")
    with pytest.raises(g.UnknownModelKey, match="unknown model key"):
        g.rows_for("no_such_model", "wo")
    with pytest.raises(g.SiteGeometryError, match="module_set must be a non-empty module set id"):
        g.site_geometry_for(KEY, "")


# --------------------------------------------------------------------------------------------- the record + its validation


def test_record_round_trips_per_set_and_the_selected_set_shadows_the_built_in_row(monkeypatch, tmp_path):
    d = _write(tmp_path, _record())
    monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(d))
    site = g.load_site_geometry(KEY)
    assert site is not None and sorted(site.sets) == ["qkvo", "wo"] and site.path == str(d / f"{KEY}.json")
    assert site.summary["lora_max_module_dim_max"] == 9216 and site.summary["batched_token_budget_max"] == 8 * 49152
    assert site.summary["batched_token_budget_min"] == 2 * 98304 and "never sizes a standup" in site.summary["note"]
    assert site.as_record() == _record()
    wo = g.site_geometry_for(KEY, "wo")
    assert wo.module_set == "wo" and wo.lora_modules == WO and wo.lora_max_module_dim == 4096 and wo.budget.certified
    assert wo.allowed_pairs == ((4, "wo"),) and wo.provenance["probe_job_id"] == "11111111" and wo.path == site.path
    # the SELECTED set's record shadows the built-in default; the other set's record is another number
    rows = g.rows_for(KEY, "wo")
    assert rows.site == wo and rows.budget == wo.budget and rows.onboarded and rows.hybrid and not rows.provisional      # the demo: a qwen3_5 hybrid
    rb = g.budget_for(KEY, "wo")
    assert rb.measured and rb.per_model and rb.batched_token_budget == 8 * 49152 and rb.entry.certified
    assert g.budget_for(KEY, "qkvo").batched_token_budget == 2 * 98304
    assert g.lora_max_module_dim_for(KEY, "wo") == 4096 and g.lora_max_module_dim_for(KEY, "qkvo") == 9216
    assert g.certified_module_dims(KEY) == {"wo": 4096, "qkvo": 9216}
    L = 98304
    assert g.kernel_ceiling_adjustment(KEY, L, 8, module_set="wo") == {"kind": g.ADJUSTMENT_KERNEL, "from_S": 8, "to_S": 5, "D": 4096, "ceiling": LIMIT // 4096}
    assert g.kernel_ceiling_adjustment(KEY, L, 8, module_set="qkvo")["to_S"] == 2 and g.adjusted_seqs(KEY, L, 8, module_set="qkvo") == 2
    # NO set named — an UNSTEERED caller (every production standup's budget_for(model_key)): the record's `wo` row, the class of the
    # zero adapter an unsteered native standup registers (2026-09-30) — never the provisional row while a record exists, never the summary
    rows0 = g.rows_for(KEY)
    assert rows0.site.module_set == "wo" and rows0.site.lora_max_module_dim == 4096 and rows0.site_module_set_source == g.MODULE_SET_SOURCE_UNSTEERED
    assert rows.site_module_set_source == g.MODULE_SET_SOURCE_NAMED
    rb0 = g.budget_for(KEY)
    # CONSERVATIVE across the certified sets — the minimum budget (qkvo's 2 x 98,304 here, not wo's 8 x 49,152), tagged with the record
    assert rb0.measured and rb0.batched_token_budget == 2 * 98304 and rb0.module_set == "wo" and rb0.module_set_source == g.MODULE_SET_SOURCE_UNSTEERED
    assert rb0.entry.basis.source == f"site_record:{GEN}" and rb0.entry.seqs_at(98304) == 2 and rb0.entry.seqs_at(49152) == 4
    assert rb.module_set == "wo" and rb.module_set_source == g.MODULE_SET_SOURCE_NAMED
    assert g.lora_max_module_dim_for(KEY) == 4096 and g.kernel_ceiling_adjustment(KEY, L, 8) == g.kernel_ceiling_adjustment(KEY, L, 8, module_set="wo")
    # a set the file does not certify REFUSES by name (never a neighbour's record, never the summary)
    for call in (lambda: g.site_geometry_for(KEY, "mlp"), lambda: g.rows_for(KEY, "mlp"), lambda: g.budget_for(KEY, "mlp"),
                 lambda: g.lora_max_module_dim_for(KEY, "mlp"), lambda: g.kernel_ceiling_adjustment(KEY, L, 8, module_set="mlp")):
        with pytest.raises(g.SiteGeometryError, match="selected module set 'mlp' has NO certified geometry record.*certified module sets: \\['qkvo', 'wo'\\]"):
            call()
    # a built-in MEASURED row is shadowed too, per set; the tables never move
    d2 = _write(tmp_path / "two", _record(key="qwen35_4b", sets={"wo": _set("wo", WO, 8192, S=3, L=16384, key="qwen35_4b")}), key="qwen35_4b")
    monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(d2))
    assert g.budget_for("qwen35_4b", "wo").batched_token_budget == 3 * 16384 and g.lora_max_module_dim_for("qwen35_4b", "wo") == 8192
    assert g.budget_for("qwen35_4b").batched_token_budget != 3 * 16384 and g.LORA_MAX_MODULE_DIM_BY_MODEL["qwen35_4b"] == 4096
    assert KEY not in g.ONBOARDED_BUDGET_BY_MODEL                                  # the tables never move: still no row of its own
    # writing through the API: atomic, validated, read back
    p = g.write_site_geometry(_record(), tmp_path / "three")
    assert p == tmp_path / "three" / f"{KEY}.json" and sorted(g.load_site_geometry(KEY, tmp_path / "three").sets) == ["qkvo", "wo"]
    assert not list((tmp_path / "three").glob("*.tmp.*"))
    monkeypatch.delenv(g.SITE_GEOMETRY_DIR_ENV)
    with pytest.raises(g.SiteGeometryError, match="no site geometry directory"):
        g.write_site_geometry(_record())


def _v1_record() -> dict:
    return {"version": 1, "model_key": KEY, "hf_id": "org/toy", "revision": "0" * 40, "written_utc": "x", "lora_max_module_dim": 4096,
            "executing_module_dim_by_set": {"wo": 4096}, "provenance": {"probe_job_id": "1", "probe_record": "/r", "attestation_sha256": {"a": {}}},
            "budget": {"batched_token_budget": 8 * 49152, "certified": True, "source": "measured", "basis": {}}}


@pytest.mark.parametrize("mutate, needle", [
    (lambda r: r.update(version=1), "version 1 != 2.*model-wide record is refused"),
    (lambda r: r.update(model_key="no_such_model"), "not a known model key"),
    (lambda r: r.__setitem__("sets", {}), "sets must be a non-empty object"),
    (lambda r: r["sets"]["wo"].__setitem__("lora_max_module_dim", 2048), "lora_max_module_dim 2048 != the widest measured launch dim 4096"),
    (lambda r: r["sets"]["wo"]["budget"].__setitem__("batched_token_budget", 5), "the budget IS the passing probe's product"),
    (lambda r: r["sets"]["wo"]["budget"].__setitem__("source", "guess"), "source must be one of"),
    (lambda r: r["sets"]["wo"]["budget"]["basis"].pop("measured_on"), "budget.basis lacks \\['measured_on'\\]"),
    (lambda r: r["sets"]["wo"].__setitem__("allowed_pairs", [[4, "qkvo"]]), "allowed_pairs entry \\[4, 'qkvo'\\] is not \\[rank, 'wo'\\]"),
    (lambda r: r["sets"]["wo"].__setitem__("certified_by", []), "certified_by must be a non-empty list"),
    (lambda r: r["sets"]["wo"].__setitem__("budget_launches", {"x": {"S": 1, "L": 4096, "batched_tokens": 4096}}), "is not one of its launches' products"),
    (lambda r: r["sets"]["wo"].pop("lora_modules"), "lacks \\['lora_modules'\\]"),
    # the record says BOTH facts — a set without its parity block, a record without the derived block, a block whose status is
    # not its pairs' aggregate, and a record-level block that is not the sets' aggregate are each refused by name
    (lambda r: r["sets"]["wo"].pop("adapter_parity"), "lacks \\['adapter_parity'\\]"),
    (lambda r: r.pop("adapter_parity"), "no adapter_parity block"),
    (lambda r: r["sets"]["wo"]["adapter_parity"].__setitem__("status", "mismatch"), "adapter_parity.status 'mismatch' disagrees with its per_pair"),
    (lambda r: (r["adapter_parity"].__setitem__("status", "mismatch"), r["adapter_parity"]["per_pair"][0].__setitem__("status", "mismatch")),
     "adapter_parity \\(mismatch.*disagrees with the sets' blocks"),
    (lambda r: r["summary"].__setitem__("lora_max_module_dim_max", 1), "summary.lora_max_module_dim_max 1 disagrees with the sets"),
    (lambda r: r["provenance"].pop("attestation_sha256"), "provenance.attestation_sha256 is missing or empty"),
    (lambda r: r.pop("hf_id"), "hf_id must be a non-empty string"),
])
def test_a_present_but_unusable_overlay_is_loud_from_every_resolver(monkeypatch, tmp_path, mutate, needle):
    rec = _record(); mutate(rec)
    d = _write(tmp_path, rec)
    monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(d))
    for call in (lambda: g.load_site_geometry(KEY), lambda: g.site_geometry_for(KEY, "wo"), lambda: g.rows_for(KEY, "wo"),
                 lambda: g.budget_for(KEY, "wo"), lambda: g.lora_max_module_dim_for(KEY, "wo"),
                 lambda: g.kernel_ceiling_adjustment(KEY, 4096, 10, module_set="wo"), lambda: pf.geometry_expectation(KEY, 4096, module_set="wo")):
        with pytest.raises(g.SiteGeometryError, match=needle):
            call()
    assert isinstance(g.SiteGeometryError("x"), ValueError)


def test_version_one_unparseable_and_another_models_file_are_refused(monkeypatch, tmp_path):
    d = tmp_path / "geometry"; d.mkdir()
    monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(d))
    (d / f"{KEY}.json").write_text(json.dumps(_v1_record()))
    with pytest.raises(g.SiteGeometryError, match="version 1 != 2"):
        g.budget_for(KEY, "wo")
    (d / f"{KEY}.json").write_text("{not json")
    with pytest.raises(g.SiteGeometryError, match="unreadable"):
        g.budget_for(KEY, "wo")
    (d / f"{KEY}.json").write_text(json.dumps(_record(key="qwen35_4b")))
    with pytest.raises(g.SiteGeometryError, match=f"names model 'qwen35_4b', asked for '{KEY}'"):
        g.rows_for(KEY, "wo")


# --------------------------------------------------------------------------------------------- the ONE selection source


def test_selected_module_set_comes_from_the_models_table_alone(monkeypatch, tmp_path):
    # the real table: the paper's models are `paper` picks on o_proj -> wo, and so is the onboarded demo's (the paper's pick on the same
    # checkpoint); a PENDING pick (a synthetic table) is refused by name
    assert pf.selected_module_set("qwen35_4b")[0] == "wo" and pf.selected_module_set("gemma4e4b")[0] == "wo" and pf.selected_module_set(KEY)[0] == "wo"
    sid, info = pf.selected_module_set("qwen35_4b")
    assert info["selected_by"] == "paper" and info["modules"] == ["o_proj"] and info["source"].endswith("reproduce/models.yaml")
    assert pf.selected_module_set(KEY)[1]["selected_by"] == "paper" and pf.selected_module_set(KEY)[1]["modules"] == ["o_proj"]
    monkeypatch.setenv(pf.MODELS_TABLE_ENV, str(_table(tmp_path / "pending", None, selected_by="pending_user_pick")))
    with pytest.raises(ValueError, match=f"{KEY}: no promoted LoRA architecture pick in .*selected_by='pending_user_pick'.*--phase pick"):
        pf.selected_module_set(KEY)
    monkeypatch.delenv(pf.MODELS_TABLE_ENV)
    # a key outside the table: None (the built-in geometry tables stand), with the reason
    sid, info = pf.selected_module_set("llama3b")
    assert sid is None and "not a row of" in info["reason"]
    # another copy of the SAME table (SISTEER_MODELS_TABLE — reproduce/lib/models.py's own seam), the promoted pick's modules -> set id
    monkeypatch.setenv(pf.MODELS_TABLE_ENV, str(_table(tmp_path, QKVO)))
    assert pf.selected_module_set(KEY) == ("qkvo", {"source": str(tmp_path / "models.yaml"), "selected_by": "manual", "selection_ruling_id": "r1",
                                                    "rank": 4, "modules": list(QKVO), "module_set": "qkvo"})
    monkeypatch.setenv(pf.MODELS_TABLE_ENV, str(_table(tmp_path, ("q_proj", "gate_proj"))))
    with pytest.raises(ValueError, match="not one of the module sets the serving probe certifies"):
        pf.selected_module_set(KEY)
    monkeypatch.setenv(pf.MODELS_TABLE_ENV, str(tmp_path / "missing.yaml"))
    with pytest.raises(ValueError, match="cannot be read"):
        pf.selected_module_set(KEY)


def test_padded_preflight_sizes_the_selected_set_and_refuses_provisional_pending_and_uncertified(monkeypatch, tmp_path):
    """At the preflight: the set comes from the models table; the record is that set's; a provisional or default
    budget, a pending pick and an uncertified set all refuse by name; the paper's model is unchanged."""
    d = _write(tmp_path, _record(sets={"wo": _set("wo", WO, 4096, S=8, L=98304), "qkvo": _set("qkvo", QKVO, 9216, S=8, L=98304)}))
    monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(d))
    # a PENDING pick (a synthetic table): the STEERED preflight refuses by name for want of a pick; the UNSTEERED one needs no pick — it
    # sizes from the zero adapter's class (wo) (2026-09-30: job 807664 died here demanding one)
    monkeypatch.setenv(pf.MODELS_TABLE_ENV, str(_table(tmp_path / "pending", None, selected_by="pending_user_pick")))
    with pytest.raises(ValueError, match="no promoted LoRA architecture pick"):
        pf.geometry_expectation(KEY, 98304, steered=True)
    geo = pf.geometry_expectation(KEY, 98304)
    assert geo["module_set"] == "wo" and geo["module_set_selection"]["selected_by"] == "unsteered"
    # the REAL table: the demo's pick is the paper's (wo) and this record certifies wo — the STEERED preflight sizes from that set's row
    monkeypatch.delenv(pf.MODELS_TABLE_ENV)
    geo = pf.geometry_expectation(KEY, 98304, steered=True)
    assert geo["module_set"] == "wo" and geo["module_set_selection"]["selected_by"] == "paper" and geo["served_S"] == 5 and geo["steered"] is True   # wo's D 4,096 kernel ceiling binds at 98,304
    assert geo["budget_source"] == g.MEASURED_BUDGET_SOURCE and geo["module_set_source"] == "named"
    # a PROMOTED pick on the MLP set (a synthetic table): the STEERED preflight follows the pick and refuses by name because this record
    # certifies wo/qkvo only (an UNCERTIFIED set, before any number); the UNSTEERED one REPORTS the promoted pick and keeps the zero class
    monkeypatch.setenv(pf.MODELS_TABLE_ENV, str(_table(tmp_path / "mlp0", ("down_proj", "gate_proj", "up_proj"))))
    with pytest.raises(ValueError, match="selected module set 'mlp' has NO certified geometry record"):
        pf.geometry_expectation(KEY, 98304, steered=True)
    geo = pf.geometry_expectation(KEY, 98304)
    assert geo["module_set"] == "wo" and geo["module_set_selection"]["selected_by"] == "unsteered"
    assert geo["module_set_selection"]["promoted_pick"]["module_set"] == "mlp" and geo["module_set_selection"]["promoted_pick"]["rank"] == 4
    assert geo["module_set_source"] == g.MODULE_SET_SOURCE_UNSTEERED and geo["served_S"] == 5 and geo["steered"] is False and geo["budget_from_site_overlay"] is True
    # the pick selects wo: D 4,096 -> S 5 at 98,304; selecting qkvo: D 9,216 -> S 2 — the SAME model, the STEERED design decides; the
    # UNSTEERED standup keeps the zero adapter's class whatever the pick (a qkvo pick is reported, never its size)
    monkeypatch.setenv(pf.MODELS_TABLE_ENV, str(_table(tmp_path, WO)))
    geo = pf.geometry_expectation(KEY, 98304, steered=True)
    assert geo["module_set"] == "wo" and geo["module_set_selection"]["selected_by"] == "manual" and geo["budget_from_site_overlay"] is True
    assert geo["identity_S"] == 8 and geo["lora_max_module_dim"] == 4096 and geo["kernel_adjustment"]["to_S"] == 5 and geo["served_S"] == 5
    assert geo["module_set_source"] == g.MODULE_SET_SOURCE_NAMED
    monkeypatch.setenv(pf.MODELS_TABLE_ENV, str(_table(tmp_path, QKVO)))
    geo = pf.geometry_expectation(KEY, 98304, steered=True)
    assert geo["module_set"] == "qkvo" and geo["lora_max_module_dim"] == 9216 and geo["served_S"] == 2
    geo = pf.geometry_expectation(KEY, 98304)
    assert geo["module_set"] == "wo" and geo["served_S"] == 5 and geo["module_set_selection"]["promoted_pick"]["module_set"] == "qkvo"
    # an explicit module_set (a test's) beats the table; an uncertified set refuses by name
    assert pf.geometry_expectation(KEY, 98304, module_set="wo")["served_S"] == 5
    with pytest.raises(g.SiteGeometryError, match="selected module set 'mlp' has NO certified geometry record"):
        pf.geometry_expectation(KEY, 98304, module_set="mlp")
    monkeypatch.setenv(pf.MODELS_TABLE_ENV, str(_table(tmp_path, ("down_proj", "gate_proj", "up_proj"))))
    with pytest.raises(g.SiteGeometryError, match="selected module set 'mlp' has NO certified geometry record"):
        pf.geometry_expectation(KEY, 98304, steered=True)
    assert pf.geometry_expectation(KEY, 98304)["module_set"] == "wo"                                     # unsteered: the pick is not its class
    # no overlay: the Llama-certified default is refused by name — steered and unsteered alike (an onboarded key has no row of its own)
    monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(tmp_path / "none"))
    monkeypatch.setenv(pf.MODELS_TABLE_ENV, str(_table(tmp_path, WO)))
    with pytest.raises(ValueError, match=f"{KEY}: batched-token budget source is 'default_llama_certified'"):
        pf.geometry_expectation(KEY, 98304, steered=True)
    with pytest.raises(ValueError, match=f"{KEY}: batched-token budget source is 'default_llama_certified'"):
        pf.geometry_expectation(KEY, 98304)
    # a provisional row INSIDE the overlay loads but steered padding refuses it
    d = _write(tmp_path / "p", _record(sets={"wo": _set("wo", WO, 4096, S=4, L=49152, certified=False, source=g.PROVISIONAL_BUDGET_SOURCE)}))
    monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(d)); monkeypatch.setenv(pf.MODELS_TABLE_ENV, str(_table(tmp_path, WO)))
    assert g.budget_for(KEY, "wo").provisional
    with pytest.raises(ValueError, match="'provisional-conservative' \\(196,608\\), not a measurement"):
        pf.geometry_expectation(KEY, 49152)
    # the paper's measured model is unchanged by all of this (its pick is `paper` on o_proj -> wo; no overlay)
    monkeypatch.delenv(pf.MODELS_TABLE_ENV)
    geo = pf.geometry_expectation("qwen35_4b", 98304)
    assert geo["served_S"] == 5 and geo["module_set"] == "wo" and geo["budget_from_site_overlay"] is False


def _pin_program(sbatch: Path) -> str:
    src = sbatch.read_text()
    i = src.index('"$MPY" - "$SERVE_DIR/static/endpoint.json" "$KEY" "$WINDOW" "$TARGET_S"')
    start = src.index("\n", src.index("<<'PY'", i)) + 1
    return src[start:src.index("\nPY\n", start)]


@pytest.mark.parametrize("sbatch, extra", [(PAD / "target_static_padded.sbatch", []), (PAD / "target_static_padded_obj.sbatch", [""])])
def test_both_padded_launchers_geometry_pins_resolve_the_selected_set(tmp_path, sbatch, extra):
    """Both launchers' pins (RUN from their heredocs against a synthetic endpoint record): the selected set comes from the models
    table; the pin derives S from THAT set's D and budget and holds the attested record to it — a standup of another class (the
    chain's D) is refused by name; no pick, an uncertified set and a provisional budget refuse by name."""
    L, S_budget = 98304, 8
    d = _write(tmp_path, _record(sets={"wo": _set("wo", WO, 4096, S=S_budget, L=L), "qkvo": _set("qkvo", QKVO, 9216, S=S_budget, L=L)}))
    to_S = (LIMIT // 4096) // L                                                  # 5 for wo; qkvo would give 2
    ep_wo = {"engine": "lora_native", "id_name_map": {"1": "none__lora0", "2": "si_x__lora_r4"},
             "geometry": {"adjustments": [{"kind": sa.LORA_KERNEL_CEILING_REASON, "from_S": S_budget, "to_S": to_S, "D": 4096, "ceiling": LIMIT // 4096}]}}
    ep_qkvo = {**ep_wo, "geometry": {"adjustments": [{"kind": sa.LORA_KERNEL_CEILING_REASON, "from_S": S_budget, "to_S": 2, "D": 9216, "ceiling": LIMIT // 9216}]}}

    def run(ep, S, *, table, geometry=d):
        p = tmp_path / "endpoint.json"; p.write_text(json.dumps(ep))
        e = {"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"}
        if geometry is not None:
            e[g.SITE_GEOMETRY_DIR_ENV] = str(geometry)
        if table is not None:
            e[pf.MODELS_TABLE_ENV] = str(table)
        r = subprocess.run([sys.executable, "-", str(p), KEY, str(L), str(S)] + extra, input=_pin_program(sbatch), capture_output=True, text=True, cwd=REPO, env=e)
        return r.returncode, r.stdout + r.stderr

    steered = sbatch.name.endswith("_obj.sbatch")
    wo_table, qkvo_table = _table(tmp_path / "wo", WO), _table(tmp_path / "qkvo", QKVO)
    rc, out = run(ep_wo, to_S, table=wo_table)
    assert rc == 0 and "[geometry-pin] OK" in out and f"S {to_S} x L {L} = identity {S_budget} -> kernel ceiling {to_S} at D 4096 for module set 'wo'" in out, out
    if steered:
        rc, out = run(ep_qkvo, 2, table=qkvo_table)
        assert rc == 0 and "kernel ceiling 2 at D 9216 for module set 'qkvo'" in out, out
    else:
        # the UNSTEERED launcher registers only the o_proj zero adapter: a qkvo pick does not change its class — it still attests wo's
        # S 5 (the pick is reported, not the size), and a standup that attested qkvo's S 2 is refused (2026-09-30)
        rc, out = run(ep_wo, to_S, table=qkvo_table)
        assert rc == 0 and "for module set 'wo'" in out and "unsteered standup: module set 'wo' (unsteered_default)" in out and "'module_set': 'qkvo'" in out, out
        rc, out = run(ep_qkvo, 2, table=qkvo_table)
        assert rc == 1 and "attested max_num_seqs 2 != 5" in out, out
    # the CLOSE trigger: the pick is wo but the pin were to hold the arms to qkvo's D — the attested wo standup (S 5) must NOT be refused,
    # and a standup that registered the OTHER class is refused by name
    rc, out = run(ep_qkvo, 2, table=wo_table)
    assert rc == 1 and "attested max_num_seqs 2 != 5" in out and "certifies D=4096" in out, out
    rc, out = run({**ep_wo, "geometry": {"adjustments": [dict(ep_wo["geometry"]["adjustments"][0], D=9216, ceiling=LIMIT // 9216)]}}, to_S, table=wo_table)
    assert rc == 1 and "certifies D=4096" in out and "another class" in out, out
    # a PENDING pick (a synthetic table) and an explicitly uncertified pick: the STEERED pin refuses by name (no pick; no certified mlp
    # record), the UNSTEERED one needs no pick (the zero adapter's class sizes it); a default budget (no overlay) refuses both; never a
    # Traceback. The REAL table's pick is the paper's (wo): the steered pin passes on this wo record like the synthetic wo table above
    rc, out = run(ep_wo, to_S, table=None)
    assert rc == 0 and "[geometry-pin] OK" in out and "for module set 'wo'" in out and "Traceback" not in out, out
    rc, out = run(ep_wo, to_S, table=_table(tmp_path / "pending", None, selected_by="pending_user_pick"))
    if steered:
        assert rc == 1 and "[geometry-pin] REFUSED" in out and "no promoted LoRA architecture pick" in out and "Traceback" not in out, out
    else:
        assert rc == 0 and "[geometry-pin] OK" in out and "unsteered standup: module set 'wo' (unsteered_default)" in out, out
    rc, out = run(ep_wo, to_S, table=_table(tmp_path / "mlp", ("down_proj", "gate_proj", "up_proj")))
    if steered:
        assert rc == 1 and "selected module set 'mlp' has NO certified geometry record" in out and "Traceback" not in out, out
    else:
        assert rc == 0 and "unsteered standup: module set 'wo'" in out, out
    rc, out = run(ep_wo, to_S, table=wo_table, geometry=None)
    assert rc == 1 and "default_llama_certified" in out and "Traceback" not in out, out


# --------------------------------------------------------------------------------------------- the adjustment re-derivation


def test_geometry_adjustment_problem_rederives_a_per_set_models_step_from_its_own_certified_D(monkeypatch, tmp_path):
    L, S = 98304, 8
    wo_step = {"kind": g.ADJUSTMENT_KERNEL, "from_S": S, "to_S": (LIMIT // 4096) // L, "D": 4096, "ceiling": LIMIT // 4096}
    qkvo_step = {"kind": g.ADJUSTMENT_KERNEL, "from_S": S, "to_S": (LIMIT // 9216) // L, "D": 9216, "ceiling": LIMIT // 9216}
    prob = lambda blk, served: g.geometry_adjustment_problem(blk, model_key=KEY, L=L, S_identity=S, S_served=served, native=True)
    # no overlay: nothing to re-derive against
    why = prob({g.ADJUSTMENTS_KEY: [wo_step]}, wo_step["to_S"])
    assert why and "no pinned LoRA max module dim" in why and "site geometry overlay not configured" in why
    d = _write(tmp_path, _record(sets={"wo": _set("wo", WO, 4096, S=S, L=L), "qkvo": _set("qkvo", QKVO, 9216, S=S, L=L)}))
    monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(d))
    # a wo standup's record and a qkvo standup's record BOTH re-derive (each from its own certified D)
    assert prob({g.ADJUSTMENTS_KEY: [wo_step]}, wo_step["to_S"]) is None
    assert prob({g.ADJUSTMENTS_KEY: [qkvo_step]}, qkvo_step["to_S"]) is None
    # a D no certified set has is refused by name; a certified D with the wrong arithmetic too
    why = prob({g.ADJUSTMENTS_KEY: [dict(wo_step, D=5120, ceiling=LIMIT // 5120)]}, wo_step["to_S"])
    assert why and "records D=5120, which is not a certified module set's width" in why and "'wo': 4096" in why
    why = prob({g.ADJUSTMENTS_KEY: [dict(wo_step, to_S=4)]}, 4)
    assert why and "records to_S=4; the formula gives 5" in why
    # the paper's model keeps its single pinned D (no overlay for it)
    assert g.geometry_adjustment_problem({g.ADJUSTMENTS_KEY: []}, model_key="qwen35_4b", L=4096, S_identity=10, S_served=10, native=True) is None


def test_one_serializer_writes_every_copy_and_the_canonical_sha_ignores_formatting(tmp_path):
    """The probe's record, the overlay and every copy are written by ONE serializer (byte-identical,
    one trailing newline); the canonical content sha is the same for the same content in any formatting and differs on content."""
    rec = _record()
    b = g.site_geometry_bytes(rec)
    assert b.endswith(b"\n") and not b.endswith(b"\n\n") and b == (json.dumps(rec, indent=2, sort_keys=True) + "\n").encode()
    p1 = g.write_site_geometry(rec, tmp_path / "overlay")
    p2 = g.write_site_geometry_file(rec, tmp_path / "rec" / "geometry_record.json")
    assert p1.read_bytes() == p2.read_bytes() == b and not list((tmp_path / "rec").glob("*.tmp.*"))
    canon = g.canonical_site_geometry_sha256(rec)
    assert canon == g.canonical_site_geometry_sha256(p1) == g.canonical_site_geometry_sha256(b) == g.canonical_site_geometry_sha256(b.decode())
    # formatting-only variants (another indent, no trailing newline, another key order) hash the same content
    (tmp_path / "v1.json").write_text(json.dumps(rec, indent=4))
    (tmp_path / "v2.json").write_text(json.dumps({k: rec[k] for k in reversed(list(rec))}, separators=(",", ":")))
    assert g.canonical_site_geometry_sha256(tmp_path / "v1.json") == canon == g.canonical_site_geometry_sha256(tmp_path / "v2.json")
    assert (tmp_path / "v1.json").read_bytes() != b                              # the bytes differ; the content does not
    # ANY content difference hashes differently
    other = json.loads(json.dumps(rec)); other["sets"]["wo"]["lora_max_module_dim"] = 4095; other["sets"]["wo"]["module_dim_launches"]["wo_r4_x_L4096"] = 4095
    assert g.canonical_site_geometry_sha256(other) != canon
    # the shell one-liner lib/slurm.sh uses computes the SAME digest
    sh = (REPO / "reproduce/lib/slurm.sh").read_text()
    one = sh[sh.index("local canon='") + len("local canon='"):]
    one = one[:one.index("'")]
    r = subprocess.run([sys.executable, "-c", one, str(tmp_path / "v1.json")], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == canon, (r.stdout, r.stderr)
    with pytest.raises(g.SiteGeometryError, match="not JSON"):
        g.canonical_site_geometry_sha256(b"{not json")
    with pytest.raises(g.SiteGeometryError):
        g.write_site_geometry_file({"version": 1}, tmp_path / "x.json")          # validated before any byte lands
    assert not (tmp_path / "x.json").exists()


def test_the_public_surface_is_exported():
    for name in ("SITE_GEOMETRY_DIR_ENV", "SITE_DIR_ENV", "SHARED_ROOT_ENV", "SITE_GEOMETRY_VERSION", "SITE_GEOMETRY_METHOD",
                 "SITE_GEOMETRY_PROVENANCE_REQUIRED", "SITE_GEOMETRY_SET_REQUIRED", "SITE_GEOMETRY_SUMMARY_NOTE", "SiteGeometryError",
                 "SiteGeometry", "SiteGeometrySet", "site_geometry_dir", "site_geometry_path", "parse_site_geometry", "site_geometry_record",
                 "load_site_geometry", "site_geometry_for", "write_site_geometry", "lora_max_module_dim_for", "certified_module_dims",
                 "site_geometry_bytes", "write_site_geometry_file", "canonical_site_geometry_sha256", "SITE_GEOMETRY_CANONICAL_SEPARATORS"):
        assert name in g.__all__, name
    assert g.SITE_GEOMETRY_VERSION == 2 and g.SITE_GEOMETRY_DIR_ENV == "SISTEER_GEOMETRY_DIR" and pf.MODELS_TABLE_ENV == "SISTEER_MODELS_TABLE"
    assert g.SITE_GEOMETRY_SHARED_SUBDIR == ("exp", "reproduce", "geometry")


# --------------------------------------------------------------------------------------------- the unsteered default (2026-09-30)


def test_an_unsteered_caller_sizes_from_the_records_zero_adapter_class_and_never_the_provisional_row_while_a_record_exists(monkeypatch, tmp_path):
    """The u0 wave's second death (2026-09-30): every production standup calls budget_for(model_key) with NO module set, and no set meant
    the built-in row even with the staged record present (262,144 for every set: S 24).
    Now: no set = the record's `wo` row (the class of the zero adapter an unsteered native standup registers); a record without `wo`
    refuses by name; a present-but-unusable record refuses as ever; no record at all = the built-in tables, unchanged."""
    # the staged file's shape: five sets, one budget (64 x 4,096 = 262,144), D 4,096 / 9,216, adapter parity mismatch
    mlp_mods = ("down_proj", "gate_proj", "up_proj")
    sets = {sid: _set(sid, mods, D, S=64, L=4096, parity=g.PARITY_MISMATCH) for sid, mods, D in
            (("wo", WO, 4096), ("qkvo", QKVO, 9216), ("attn_mlp", QKVO + mlp_mods, 9216), ("mlp", mlp_mods, 4096), ("down", ("down_proj",), 4096))}
    d = _write(tmp_path, _record(sets=sets))
    monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(d))
    rb = g.budget_for(KEY)
    assert rb.measured and rb.batched_token_budget == 262_144 and rb.module_set == "wo" and rb.module_set_source == g.MODULE_SET_SOURCE_UNSTEERED
    assert rb.entry.seqs_at(10752) == 24 and rb.entry.seqs_at(4864) == 53 and rb.entry.seqs_at(16384) == 16 and rb.entry.seqs_at(1536) == 170
    assert rb.entry.seqs_at(4096) == 64 and rb.entry.basis.source == f"site_record:{GEN}" and "unsteered default" in rb.entry.basis.method
    # at a PROBED window the least admitted S over the sets caps the quotient — one set came up at only S 3 x 16,384
    sets2 = json.loads(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "budget"} for k, v in sets.items()}))
    for k in sets2:
        sets2[k]["budget"] = sets[k]["budget"]
    sets2["down"]["budget_launches"]["down_r4_x_L16384"] = {"S": 3, "L": 16384, "batched_tokens": 3 * 16384}
    d3 = _write(tmp_path / "probed16k", _record(sets=sets2))
    monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(d3))
    rb3 = g.budget_for(KEY)
    assert rb3.batched_token_budget == 262_144 and rb3.entry.seqs_at(16384) == 3 and rb3.entry.seqs_at(10752) == 24 and rb3.entry.window_seqs == {4096: 64, 16384: 3}
    assert g.budget_for(KEY, "wo").entry.seqs_at(16384) == 16                                          # a NAMED set keeps its own row
    monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(d))
    assert rb.as_dict()["module_set"] == "wo" and rb.as_dict()["module_set_source"] == "unsteered_default" and rb.as_dict()["provisional"] is False
    rows = g.rows_for(KEY)
    assert rows.site.module_set == "wo" and rows.site_module_set_source == g.MODULE_SET_SOURCE_UNSTEERED and not rows.provisional and rows.hybrid
    assert g.lora_max_module_dim_for(KEY) == 4096 and g.kernel_ceiling_adjustment(KEY, 10752, 24) is None       # the ceiling does not bind at D 4,096
    assert g.budget_for(KEY, "qkvo").module_set_source == g.MODULE_SET_SOURCE_NAMED and g.budget_for(KEY, "qkvo").module_set == "qkvo"
    # the record certifies sets but NOT wo: refuse by name — never the provisional row while a record exists
    d2 = _write(tmp_path / "noworow", _record(sets={"qkvo": sets["qkvo"]}))
    monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(d2))
    for call in (lambda: g.budget_for(KEY), lambda: g.rows_for(KEY), lambda: g.lora_max_module_dim_for(KEY), lambda: g.kernel_ceiling_adjustment(KEY, 10752, 24),
                 lambda: pf.geometry_expectation(KEY, 10752)):
        with pytest.raises(g.SiteGeometryError, match="does not certify 'wo'.*zero adapter.*while a record exists"):
            call()
    assert g.budget_for(KEY, "qkvo").measured                                                          # a caller naming the set is unchanged
    # a present but UNUSABLE record refuses too (never a silent fallback): a REVOKED marker beside it
    monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(d))
    g.revoked_marker_path(d / f"{KEY}.json").write_text(json.dumps({"generation": "f" * 64, "revoked_utc": "x", "reason": "test"}))
    with pytest.raises(g.SiteGeometryError, match="REVOKED"):
        g.budget_for(KEY)
    g.revoked_marker_path(d / f"{KEY}.json").unlink()
    # the unsteered preflight and the resolver agree; the parity of the record is REPORTED (unsteered reads a mismatch record)
    geo = pf.geometry_expectation(KEY, 10752)
    assert geo["module_set"] == "wo" and geo["served_S"] == 24 and geo["batched_token_budget"] == 262_144 and geo[g.ADAPTER_PARITY_KEY]["status"] == "mismatch"
    # no record: the built-in tables, as before (the onboarded key has no row: the Llama-certified default; the sweep's measured rows untouched)
    monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(tmp_path / "none"))
    assert g.budget_for(KEY).source == g.DEFAULT_BUDGET_SOURCE and g.budget_for(KEY).module_set is None and g.budget_for("qwen35_4b").measured
    assert g.budget_for("qwen35_4b").module_set is None and g.lora_max_module_dim_for("qwen35_4b") == 4096
    # the constant IS native_standup's zero-class rule: lora_export.launch_target_class's default when no genuine adapter (and no
    # residual carrier) selects a class — the standup reads that one rule (352956c83: a carrier never decides the class)
    assert g.UNSTEERED_MODULE_SET == "wo"
    assert "zero_class = lx.launch_target_class(classes)" in (REPO / "src/sisteer/capeval/serving/native_standup.py").read_text()
    assert 'default: Tuple[str, ...] = ("o_proj",)) -> Tuple[str, ...]:' in (REPO / "src/sisteer/capeval/serving/lora_export.py").read_text()
    from sisteer.capeval.serving import lora_export as _lx
    assert _lx.launch_target_class({}) == ("o_proj",) == _lx.modules_of_class(("o_proj",))


def test_a_revoked_marker_with_the_overlay_absent_refuses_every_resolver_and_never_falls_to_the_built_in_row(monkeypatch, tmp_path):
    """A receipted reopen ARCHIVES the overlay and leaves `<model>.REVOKED.json` where it stood (lib/slurm.sh
    slurm_revoke_geometry) — before this, load_site_geometry returned None on the absent file before looking for the marker, so a queued
    or retried unpadded u0 standup during a re-probe took the built-in row again. Now the marker alone refuses
    by name from every resolver (named set or none), the preflight and the adjustment re-derivation; the built-in tables stand only when
    NEITHER the file NOR the marker exists."""
    d = tmp_path / "revoked"; d.mkdir()
    g.revoked_marker_path(d / f"{KEY}.json").write_text(json.dumps({"model_key": KEY, "generation": "c" * 64, "revoked_utc": "2026-09-30T14:30:00Z",
                                                                     "reason": "reopen of the record (03_hpo.sh --phase probe)", "by": "front door"}))
    monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(d))
    assert not (d / f"{KEY}.json").exists()
    for call in (lambda: g.load_site_geometry(KEY), lambda: g.site_geometry_for(KEY, "wo"), lambda: g.rows_for(KEY), lambda: g.rows_for(KEY, "wo"),
                 lambda: g.budget_for(KEY), lambda: g.budget_for(KEY, "wo"), lambda: g.lora_max_module_dim_for(KEY), lambda: g.lora_max_module_dim_for(KEY, "wo"),
                 lambda: g.kernel_ceiling_adjustment(KEY, 10752, 24), lambda: g.certified_module_dims(KEY),
                 lambda: pf.geometry_expectation(KEY, 10752), lambda: pf.geometry_expectation(KEY, 10752, module_set="wo", steered=True)):
        with pytest.raises(g.SiteGeometryError, match="ABSENT and .*REVOKED.json stands where it stood.*reopened this record.*never stand in for a revoked record"):
            call()
    assert "REVOKED — absent at" in g._site_overlay_where(KEY) and "reopened" in g._site_overlay_where(KEY, "wo")
    why = g.geometry_adjustment_problem({g.ADJUSTMENTS_KEY: []}, model_key=KEY, L=10752, S_identity=24, S_served=24, native=True)
    assert why and "refuses" in why and "REVOKED" in why                                                 # a problem string, never a raise
    # the marker gone (a verified staging cleared it) with the file still absent: NO record — the built-in tables, as before
    g.revoked_marker_path(d / f"{KEY}.json").unlink()
    assert g.load_site_geometry(KEY) is None and g.budget_for(KEY).source == g.DEFAULT_BUDGET_SOURCE and g.rows_for(KEY).site is None
    # a sweep model's measured row stands for a no-set caller (bound (i)), but a NAMED set on a marker-only state refuses
    d2 = tmp_path / "revoked_paper"; d2.mkdir()
    g.revoked_marker_path(d2 / "qwen35_4b.json").write_text(json.dumps({"generation": "d" * 64, "revoked_utc": "x", "reason": "test"}))
    monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(d2))
    assert g.budget_for("qwen35_4b").measured and g.budget_for("qwen35_4b").module_set is None
    with pytest.raises(g.SiteGeometryError, match="ABSENT and .*REVOKED"):
        g.budget_for("qwen35_4b", "wo")
