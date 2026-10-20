"""EM misalign adapter, on SYNTHETIC fixtures: panels are ENUMERATED from the root and
parsed by name (a new ``<stem>_vendor_leak`` directory is read without a code edit; an unknown suffix refuses BY NAME in its own
group); the ``_vendor_leak`` engine columns land on the run (serving identity, cross-checked against the store leg) and on the
contrast (``engine`` / ``anchor_engine`` / ``cross_engine`` + the cross-engine pairing allowance); a ``killed_alphasteer`` entry
under a judged store is skipped by name and listed in the store's source_state membership; a sharded ``cells_full`` store is
merged in memory (and a conflicting shard refuses); ``baseline_pin`` maps to the operating-point id and the ``_bl`` arms carry
their dose unit (alpha / lambda) with the pinned coefficient, while LoRA rows at the nominal leave the unit unset.

Everything here is invented (tests/registry_declared_fixtures.py + post-edits of the CSV / JSON it writes); nothing reads a real store.
"""
from __future__ import annotations

import json
import os

import pytest

from sisteer.registry import validate as VAL
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import em_misalign as EM
from sisteer.registry.adapters.base import AdapterContext, AdapterRefusal
from sisteer.registry import synth

import registry_declared_fixtures as F  # noqa: E402
import _anchor_frame_fixtures as AF  # noqa: E402  (emitter 4: in-memory reads need a primed anchor frame)

NOW = "2026-09-17T12:00:00.000Z"
PRODUCER = {"script": "tests.declared_em_d60", "code_sha": synth.code_sha("a"), "session": None, "host": "TESTHOST", "substrate": "grading_host"}
PASS, LENS, LORA = "proxy_vllm+passthrough+interleaved", "proxy_vllm+lens+interleaved", "proxy_vllm+lora+interleaved"
LORA_CELL, PSR_CELL = "si_constv2_mix_full__lora_r4_ce", "si_constv2_mix_full__psr_ce_all"
LORA_ID, PSR_ID = "constv2-mix-full__lora_r4_ce", "constv2-mix-full__psr_ce_all"
ALPHA, LAMBDA = 0.05, 0.1369
ARDITI_PIN = {"alpha": ALPHA, "object_sha256": F.SHA_A, "selection_outcome": "ruled", "source": "arditi_strength/gemma4e4b/strength/cells_view/ar_r0p05/arditi"}
ALPHA_PIN = {"rung": "lam1", "rung_i": 1, "lambda": LAMBDA, "object_sha256": F.SHA_B, "selection_outcome": "ruled_2026-09-14",
             "source": "alphasteer_f22_rebake/gemma4e4b/ladder_canon/lam1/cells_view/it/alphasteer"}


def _ctx() -> AdapterContext:
    return AdapterContext(root="", now=NOW, producer=dict(PRODUCER))


def _read_all(adapter):
    """discover -> comparison groups -> read; (records, report) with every record validated (as test_registry_adapter_declared._read_all)."""
    inputs = adapter.discover()
    groups = {}
    for i in inputs:
        groups.setdefault(adapter.comparison_group(i), []).append(i)
    recs = []
    ctx = AF.primed_ctx(adapter, now=NOW, producer=PRODUCER, groups=groups)     # the frame of the pass AFTER the fold: every own anchor current
    for g, ins in sorted(groups.items()):
        for batch in adapter.read(ctx, g, ins):
            recs.extend(batch.records)
    bad = [str(v) for v in (VAL.validate_record(r, path="<test>") for r in recs) if isinstance(v, VAL.Malformed)]
    assert not bad, bad[:5]
    return recs, adapter.last_report


def _by_type(recs):
    out = {}
    for r in recs:
        out.setdefault(r["record_type"], []).append(r["payload"])
    return out


def _add_engine_columns(panel_dir: str, engine_of: dict, anchor_engine: str = PASS, cross: dict | None = None) -> None:
    """Post-edit a fixture panel into the ``_vendor_leak`` shape: summary ``engine``, paired ``engine`` / ``anchor_engine`` / ``cross_engine``."""
    import pandas as pd
    s = pd.read_csv(os.path.join(panel_dir, "summary.csv"))
    s["engine"] = s.arm.map(engine_of)
    s.to_csv(os.path.join(panel_dir, "summary.csv"), index=False)
    p = pd.read_csv(os.path.join(panel_dir, "paired.csv"))
    p["engine"] = p.arm.map(engine_of)
    p["anchor_engine"] = anchor_engine
    p["cross_engine"] = p.arm.map(lambda a: bool((cross or {}).get(a, False)))
    p.to_csv(os.path.join(panel_dir, "paired.csv"), index=False)


def _adapter(root, grade, pins=None, bpins=None, **kw):
    return EM.EmMisalignAdapter(root=root, full22_pins=pins, baseline_pins=bpins, behaviour_source=kw.pop("behaviour_source", {}), grade_root=grade, **kw)


def _vendor_leak_world(tmp_path, csv_engine_for_lora: str = LENS):
    """A ``qwen35_4b_vendor_leak`` panel (none / lens LoRA / native PSR) with its store under misalign_vendor_leak/qwen35_4b."""
    root, grade = str(tmp_path / "figs"), str(tmp_path / "grade")
    fx = F.build_em_panel(root, stem="qwen35_4b_vendor_leak", arms=(("none", 0.36, 510), (LORA_CELL, 0.2, 510), (PSR_CELL, 0.1, 510)))
    _add_engine_columns(fx["dir"], {"none": PASS, LORA_CELL: csv_engine_for_lora, PSR_CELL: LORA}, cross={PSR_CELL: True})
    F.build_em_store(grade, "qwen35_4b", legs=[("none", PASS, "aaaa000011112222"), (LORA_CELL, LENS, "bbbb000011112222"), (PSR_CELL, LORA, "cccc000011112222")],
                     bank_objects={LORA_ID: F.SHA_A, PSR_ID: F.SHA_B}, subdir="misalign_vendor_leak")
    return root, grade, fx


# ============================================================================================
# 1. enumeration
# ============================================================================================

def test_enumeration_finds_a_new_vendor_leak_directory_without_a_code_edit(tmp_path):
    root, grade, fx = _vendor_leak_world(tmp_path)
    ad = _adapter(root, grade)                                                    # the default stem vocabulary; no directory list anywhere
    inputs = ad.discover()
    assert {os.path.basename(os.path.dirname(i.uri)) for i in inputs if not AB.is_listing_input(i)} == {"qwen35_4b_vendor_leak"}
    assert {ad.comparison_group(i) for i in inputs} == {"qwen35_4b"}             # a variant of the qwen35_4b stem: same refresh boundary (its store's listing included)
    rows, facts = ad.long_rows("qwen35_4b")
    assert len(rows) == 3 and {r["source"] for r in rows} == {"em/agentic/qwen35_4b_vendor_leak"}
    assert facts["stems"] == [dict(stem="qwen35_4b", dir="qwen35_4b_vendor_leak", laddered=False, live=True, variant="vendor_leak")]
    arm = [r for r in rows if r["cell_id"] == LORA_ID][0]
    # read as a panel in its own right: its own file sha recorded, NOT pinned; the vendor decode instrument; measured (not the rung-1 "pending")
    assert arm["_extra"]["panel_summary_sha256"] == F.sha256_file(os.path.join(fx["dir"], "summary.csv")) and arm["_extra"]["panel_pinned"] is False
    assert arm["_instrument"]["decode_source"] == "vendor_recommendation" and arm["_instrument"]["token_cap_tokens"] == 16384 and arm["status"] == "measured"
    assert arm["_extra"]["judged_store_source"] == "cells_full.json" and arm["_producer_key"] == "misalign_vendor_leak/qwen35_4b/bbbb000011112222"
    assert facts["skipped"] == [] and facts["refused"] == []


def test_unknown_suffix_refuses_by_name_in_its_own_group(tmp_path):
    root, grade = str(tmp_path / "figs"), str(tmp_path / "grade")
    F.build_em_panel(root)                                                        # gemma4e4b: a known stem
    F.build_em_panel(root, stem="gemma4e4b__weird")                               # an unknown suffix
    F.build_em_panel(root, stem="frobnicator_bl")                                 # an unknown stem
    pins = F.build_em_pins(str(tmp_path / "pins.json"))
    F.build_em_store(grade, "gemma4e4b", legs=[("none", PASS, "a" * 16), (LORA_CELL, LENS, "b" * 16)], bank_objects={LORA_ID: F.SHA_C})
    ad = _adapter(root, grade, pins=pins)
    inputs = ad.discover()
    groups = {ad.comparison_group(i) for i in inputs}
    assert groups == {"gemma4e4b", "unknown/gemma4e4b__weird", "unknown/frobnicator_bl"}   # visible: each unknown directory is its own refused group
    assert ad.last_facts["<discover>"]["unknown"].keys() == {"gemma4e4b__weird", "frobnicator_bl"}
    with pytest.raises(AdapterRefusal, match="gemma4e4b__weird"):
        ad.long_rows("unknown/gemma4e4b__weird")
    with pytest.raises(AdapterRefusal, match="frobnicator_bl.*not in the paper's EM panel vocabulary"):
        list(ad.read(_ctx(), "unknown/frobnicator_bl", [i for i in inputs if "frobnicator" in i.uri]))
    # the known stem still reads: the unknown directories cost nothing but their own group
    rows, facts = ad.long_rows("gemma4e4b")
    assert len(rows) == 2 and facts["panels"] == ["gemma4e4b"]
    # no list of panel directories remains in the module: the stem table is the paper's vocabulary
    assert not hasattr(EM, "EM_AGENTIC_PANELS") and not hasattr(EM, "PANEL_ONLY_ARMS")
    assert set(EM.EM_STEMS) >= {"gemma4e4b", "qwen35_4b_bl", "llama8b_main", "llama8b_it"} and all(len(v) == 4 for v in EM.EM_STEMS.values())


def test_vendor_pin_is_still_checked_on_the_directory_it_names(tmp_path):
    root, grade = str(tmp_path / "figs"), str(tmp_path / "grade")
    F.build_em_panel(root, stem="qwen35_4b_it_ladder", behaviour="murder")
    F.build_em_panel(root, stem="qwen35_4b_it_vendor", behaviour="murder")
    F.build_em_store(grade, "qwen35_4b_it", legs=[("none", PASS, "a" * 16), (LORA_CELL, LENS, "b" * 16)], bank_objects={LORA_ID: F.SHA_A}, subdir="misalign_ladder")
    F.build_em_store(grade, "qwen35_4b_it", legs=[("none", PASS, "c" * 16), (LORA_CELL, LENS, "d" * 16)], bank_objects={LORA_ID: F.SHA_A}, subdir="misalign_vendor")
    good = F.sha256_file(os.path.join(root, "qwen35_4b_it_vendor", "summary.csv"))
    ad = _adapter(root, grade, behaviour_source={("qwen35_4b_it", "murder"): dict(dir="qwen35_4b_it_vendor", sha256=good)})
    rows, facts = ad.long_rows("qwen35_4b_it")
    live = [r for r in rows if not r["_superseded"]]
    assert {r["_extra"]["panel_dir"] for r in live} == {"qwen35_4b_it_vendor"} and all(r["_extra"]["panel_pinned"] and r["_extra"]["panel_pin_sha256"] == good for r in live)
    assert all(r["_superseded"] and r["_superseded_by_key"].endswith(f"|qwen35_4b_it_vendor|murder|{r['_extra']['raw_arm']}") for r in rows if r["_extra"]["panel_dir"] == "qwen35_4b_it_ladder")
    assert any(s.get("pinned_instrument") for s in facts["stems"])                 # the vendor directory is listed as the pinned instrument, not read twice
    bad = _adapter(root, grade, behaviour_source={("qwen35_4b_it", "murder"): dict(dir="qwen35_4b_it_vendor", sha256="0" * 64)})
    with pytest.raises(AdapterRefusal, match="pin"):
        bad.long_rows("qwen35_4b_it")


# ============================================================================================
# 2. engine columns
# ============================================================================================

def test_engine_columns_land_on_the_run_and_the_contrast(tmp_path):
    root, grade, fx = _vendor_leak_world(tmp_path)
    ad = _adapter(root, grade)
    rows, facts = ad.long_rows("qwen35_4b")
    lora = [r for r in rows if r["cell_id"] == LORA_ID][0]
    psr = [r for r in rows if r["cell_id"] == PSR_ID][0]
    none = [r for r in rows if r["cell_id"] == "none"][0]
    # summary `engine` -> the row's serving identity (+ serving path), agreeing with the store leg
    assert lora["_instrument"]["serving_identity"] == LENS and lora["_instrument"]["serving_path"] == "lens" and lora["_extra"]["leg_serving_identity_source"] == "summary.csv engine"
    assert psr["_instrument"]["serving_identity"] == LORA and psr["_instrument"]["serving_path"] == "lora_native" and psr["_extra"]["leg_serving_path"] == "lora_native"
    assert none["_instrument"]["serving_path"] == "passthrough" and "serving_path" not in none["_confidence"]
    # paired engine / anchor_engine / cross_engine -> the contrast's extra; a cross-engine anchor is declared, the anchor stays the panel's none
    assert lora["_contrast_extra"] == {"engine": LENS, "anchor_engine": PASS, "cross_engine": False}
    assert psr["_contrast_extra"] == {"engine": LORA, "anchor_engine": PASS, "cross_engine": True, "pairing_allowance": "cross_engine_anchor"}
    assert psr["_anchor_kind"] == "same_standup_none" and lora["_anchor_kind"] == "same_standup_none"
    assert psr["delta_method"].startswith("exact McNemar + percentile bootstrap over scenarios; cross-engine anchor") and "cross_engine_anchor" in psr["delta_method"]
    assert lora["delta_method"] == "exact McNemar + percentile bootstrap over scenarios"
    recs, rep = _read_all(ad)
    bt = _by_type(recs)
    assert rep.refusals == [] and rep.runs == 3 and rep.contrasts == 2
    runs = {r["object_id"]: r for r in bt["run"]}
    assert runs[F.SHA_A]["serving_identity"] == LENS and runs[F.SHA_A]["serving_path"] == "lens"
    assert runs[F.SHA_B]["serving_identity"] == LORA and runs[F.SHA_B]["serving_path"] == "lora_native" and runs[F.SHA_B]["applied_positions_mode"] == "all"
    assert runs["none"]["serving_path"] == "passthrough" and runs[F.SHA_A]["extra"]["panel_variant"] == "vendor_leak"
    contrasts = {c["run_id"]: c for c in bt["run_contrast"]}
    c_psr, c_lora = contrasts[runs[F.SHA_B]["run_id"]], contrasts[runs[F.SHA_A]["run_id"]]
    assert c_psr["anchor_run_id"] == runs["none"]["run_id"] and c_psr["anchor_kind"] == "same_standup_none"
    assert c_psr["extra"]["engine"] == LORA and c_psr["extra"]["anchor_engine"] == PASS and c_psr["extra"]["cross_engine"] is True and c_psr["extra"]["pairing_allowance"] == "cross_engine_anchor"
    assert c_lora["extra"]["cross_engine"] is False and "pairing_allowance" not in c_lora["extra"] and "cross_engine_anchor" in c_psr["delta_method"]
    # LoRA / ReLU at the nominal: no declared dose unit -> the emitter files nominal_only
    assert "_strength_unit" not in lora and "_strength_unit" not in psr
    assert runs[F.SHA_A]["strength_unit"] == "nominal_only" and runs[F.SHA_B]["strength_unit"] == "nominal_only" and runs["none"]["strength_unit"] == "unverified"


def test_engine_column_disagreeing_with_the_store_leg_refuses_the_cell_by_name(tmp_path):
    root, grade, fx = _vendor_leak_world(tmp_path, csv_engine_for_lora=LORA)      # the CSV says native LoRA; the store leg says lens
    ad = _adapter(root, grade)
    rows, facts = ad.long_rows("qwen35_4b")
    lora = [r for r in rows if r["cell_id"] == LORA_ID][0]
    assert lora["status"].startswith("refused: ") and LORA in lora["status"] and LENS in lora["status"]
    assert facts["refused"] == [f"qwen35_4b_vendor_leak/qwen35_4b_vendor_leak/leaking/{LORA_CELL}: " + lora["status"][len("refused: "):]]
    recs, rep = _read_all(ad)
    bt = _by_type(recs)
    assert rep.runs == 2 and rep.dispositions == 1 and bt["disposition"][0]["state"] == "held" and bt["disposition"][0]["object_id"] == F.SHA_A
    assert LORA in bt["disposition"][0]["reason"] and {r["object_id"] for r in bt["run"]} == {"none", F.SHA_B}


# ============================================================================================
# 3. killed_alphasteer under the grade root
# ============================================================================================

def test_killed_alphasteer_under_the_grade_root_is_skipped_by_name(tmp_path):
    root, grade, fx = _vendor_leak_world(tmp_path)
    store = os.path.join(grade, "misalign_vendor_leak", "qwen35_4b")
    killed = os.path.join(store, EM.KILLED_DIR, "agentic_alphasteer")
    os.makedirs(killed)
    with open(os.path.join(killed, "meta.json"), "w") as fh:
        json.dump({"leg": "alphasteer", "status": "killed", "n_rows": 7}, fh)
    with open(os.path.join(killed, "responses.jsonl"), "w") as fh:
        fh.write(json.dumps({"scenario": "s_alpha", "text": "lorem"}) + "\n")
    ad = _adapter(root, grade)
    rows, facts = ad.long_rows("qwen35_4b")
    assert facts["skipped"] == [dict(path=os.path.join(store, EM.KILLED_DIR), reason=EM.KILLED_REASON)]
    assert not any(r["cell_id"] == "alphasteer_rebake" for r in rows)              # never an arm, never a run
    inputs = ad.discover()
    # the store's listing is an INPUT of the group (base.ListingInput; the runner's listings-are-inputs rule): its URI names the group and the store's
    # path under the grade root, its sha is the listing's fingerprint, and the killed directory is one of its entries -- so its arrival is a revision change
    (listing,) = [i for i in inputs if AB.is_listing_input(i)]
    assert listing.uri == AB.listing_uri("em_misalign:qwen35_4b", os.path.join("misalign_vendor_leak", "qwen35_4b")) and ad.comparison_group(listing) == "qwen35_4b"
    assert listing.sha256 == AB.listing_fingerprint(AB.listing_entries(store)) and listing.entries == ("cells_full.json", EM.KILLED_DIR)
    states = ad.source_states(_ctx(), inputs)
    store_rows = [s for s in states if s.input_uri == listing.uri]
    assert len(store_rows) == 1 and store_rows[0].json_path == "$.entries" and store_rows[0].source_id == "em_misalign:qwen35_4b"
    assert EM.KILLED_DIR in store_rows[0].current_members and "cells_full.json" in store_rows[0].current_members
    assert store_rows[0].input_sha256 == listing.sha256 and not any(s.input_uri == store for s in states)
    assert [s for s in states if s.json_path == "$.arm"][0].current_members == ["none", LORA_CELL, PSR_CELL]
    recs, rep = _read_all(ad)
    assert rep.runs == 3 and rep.refusals == []


# ============================================================================================
# 4. sharded cells_full
# ============================================================================================

def _shard_store(grade: str, stem: str, subdir: str, split: dict, node_job: dict) -> str:
    """Split the fixture's cells_full.json into shards (legs by name), each shard with its own node / job id; remove the merged file."""
    d = os.path.join(grade, subdir, stem)
    full = json.load(open(os.path.join(d, "cells_full.json")))
    os.remove(os.path.join(d, "cells_full.json"))
    for shard, cells in split.items():
        if shard.endswith("_bank"):
            continue
        s = json.loads(json.dumps(full))
        s["legs"] = [l for l in full["legs"] if l["cell_id"] in cells]
        s["endpoint"]["node"], s["endpoint"]["slurm_job_id"] = node_job[shard]
        keep = split.get(f"{shard}_bank")                                              # a per-shard bank subset when given, else the whole bank
        s["endpoint"]["bank_objects"] = {c: e for c, e in full["endpoint"]["bank_objects"].items() if keep is None or c in keep}
        s["written_at"] = f"2026-09-13T0{shard[-1]}:00:00+00:00"
        for l in s["legs"]:
            l["object_path"] = f"/vol/{shard}/objects/{l['cell_id']}/object.pt"    # a per-substrate path: never a conflict
        with open(os.path.join(d, f"{shard}.json"), "w") as fh:
            json.dump(s, fh)
    return d


def test_sharded_cells_full_is_merged_in_memory(tmp_path):
    root, grade = str(tmp_path / "figs"), str(tmp_path / "grade")
    F.build_em_panel(root, arms=(("none", 0.2, 100), (LORA_CELL, 0.1, 100), (PSR_CELL, 0.15, 100)))
    F.build_em_store(grade, "gemma4e4b", legs=[("none", PASS, "aaaa000011112222"), (LORA_CELL, LENS, "bbbb000011112222"), (PSR_CELL, LORA, "cccc000011112222")],
                     bank_objects={LORA_ID: F.SHA_A, PSR_ID: F.SHA_B})
    d = _shard_store(grade, "gemma4e4b", "misalign", {"cells_full_shard0": {"none", LORA_CELL}, "cells_full_psr_shard1": {"none", PSR_CELL},
                                                      "cells_full_shard0_bank": {LORA_ID}, "cells_full_psr_shard1_bank": {PSR_ID}},
                     {"cells_full_shard0": ("gpu-3", "111"), "cells_full_psr_shard1": ("gpu-9", "222")})
    assert not os.path.exists(os.path.join(d, "cells_full.json"))
    merged = EM.merge_cells_full(sorted(os.path.join(d, f) for f in os.listdir(d)))
    assert sorted(l["cell_id"] for l in merged["legs"]) == sorted(["none", LORA_CELL, PSR_CELL])          # union of legs (the shared none once)
    assert set(merged["endpoint"]["bank_objects"]) == {LORA_ID, PSR_ID}                                 # union of bank_objects
    assert merged["endpoint"]["node"] is None and merged["endpoint"]["slurm_job_id"] is None             # per-shard scalars that differ are not invented
    assert [(s["file"], s["node"], s["slurm_job_id"]) for s in merged["shards"]] == [("cells_full_psr_shard1.json", "gpu-9", "222"), ("cells_full_shard0.json", "gpu-3", "111")]
    assert merged["wave_sha"] == synth.code_sha("e") and merged["written_at"] == "2026-09-13T01:00:00+00:00" and merged["cells_served"] == {"manifest": "synthetic"}
    ad = _adapter(root, grade)
    rows, facts = ad.long_rows("gemma4e4b")
    by_cell = {r["cell_id"]: r for r in rows}
    assert by_cell[LORA_ID]["_producer_key"] == "misalign/gemma4e4b/bbbb000011112222" and by_cell[PSR_ID]["_producer_key"] == "misalign/gemma4e4b/cccc000011112222"
    assert by_cell[PSR_ID]["_extra"]["judged_store_source"] == "merged in memory from 2 shard(s)" and len(by_cell[PSR_ID]["_extra"]["store_shards"]) == 2
    assert by_cell[PSR_ID]["_extra"]["leg_serving_path"] == "lora_native" and by_cell[LORA_ID]["_extra"]["leg_serving_path"] == "lens"
    assert by_cell[PSR_ID]["_code_sha"] == synth.code_sha("e") and by_cell[PSR_ID]["_instrument"]["job_id"] is None
    recs, rep = _read_all(ad)
    bt = _by_type(recs)
    assert rep.runs == 3 and rep.refusals == []
    run = [r for r in bt["run"] if r["object_id"] == F.SHA_B][0]
    assert run["cluster_job_id"] is None and [s["slurm_job_id"] for s in run["extra"]["store_shards"]] == ["222", "111"] and run["serving_path"] == "lora_native"


def test_conflicting_shards_refuse_by_name(tmp_path):
    root, grade = str(tmp_path / "figs"), str(tmp_path / "grade")
    F.build_em_panel(root)
    F.build_em_store(grade, "gemma4e4b", legs=[("none", PASS, "aaaa000011112222"), (LORA_CELL, LENS, "bbbb000011112222")], bank_objects={LORA_ID: F.SHA_A})
    d = _shard_store(grade, "gemma4e4b", "misalign", {"cells_full_shard0": {"none", LORA_CELL}, "cells_full_shard1": {"none", LORA_CELL}},
                     {"cells_full_shard0": ("gpu-3", "111"), "cells_full_shard1": ("gpu-9", "222")})
    # the same cell served two ways across shards
    s1 = json.load(open(os.path.join(d, "cells_full_shard1.json")))
    [l for l in s1["legs"] if l["cell_id"] == LORA_CELL][0]["intervention_hash"] = "f" * 16
    json.dump(s1, open(os.path.join(d, "cells_full_shard1.json"), "w"))
    with pytest.raises(AdapterRefusal, match=f"cells_full_shard1.json: shard conflicts with cells_full_shard0.json on leg '{LORA_CELL}'.*intervention_hash"):
        EM.merge_cells_full([os.path.join(d, "cells_full_shard0.json"), os.path.join(d, "cells_full_shard1.json")])
    with pytest.raises(AdapterRefusal, match="shard conflicts"):
        _adapter(root, grade).long_rows("gemma4e4b")                                   # the stem refuses; nothing falls back to the pins
    # a conflicting bank object refuses too
    s0 = json.load(open(os.path.join(d, "cells_full_shard0.json")))
    s1 = json.load(open(os.path.join(d, "cells_full_shard1.json")))
    [l for l in s1["legs"] if l["cell_id"] == LORA_CELL][0]["intervention_hash"] = [l for l in s0["legs"] if l["cell_id"] == LORA_CELL][0]["intervention_hash"]
    s1["endpoint"]["bank_objects"][LORA_ID] = {"object_sha256": F.SHA_C}
    json.dump(s1, open(os.path.join(d, "cells_full_shard1.json"), "w"))
    with pytest.raises(AdapterRefusal, match=f"bank_objects\\['{LORA_ID}'\\]"):
        EM.merge_cells_full([os.path.join(d, "cells_full_shard0.json"), os.path.join(d, "cells_full_shard1.json")])
    # a producer-merged cells_full.json beside the shards is read instead of re-merging
    F.build_em_store(grade, "gemma4e4b", legs=[("none", PASS, "aaaa000011112222"), (LORA_CELL, LENS, "bbbb000011112222")], bank_objects={LORA_ID: F.SHA_A})
    rows, facts = _adapter(root, grade).long_rows("gemma4e4b")
    assert {r["_extra"]["judged_store_source"] for r in rows} == {"cells_full.json"} and rows[0]["_extra"]["store_shards"] is None


# ============================================================================================
# 5. baseline_pin and the dose unit on the _bl panels
# ============================================================================================

def _bl_world(tmp_path, csv_pin: str | None = None, ruled_alphasteer_sha: str = F.SHA_B):
    root, grade = str(tmp_path / "figs"), str(tmp_path / "grade")
    fx = F.build_em_panel(root, stem="gemma4e4b_bl", arms=(("none", 0.3, 100), ("arditi", 0.2, 100), ("alphasteer", 0.1, 100)))
    F.build_em_store(grade, "gemma4e4b_bl", legs=[("none", PASS, "a" * 16), ("arditi", LENS, "b" * 16), ("alphasteer", LENS, "c" * 16)],
                     bank_objects={"arditi": F.SHA_A, "alphasteer": F.SHA_B})
    store = os.path.join(grade, "misalign", "gemma4e4b_bl", "cells_full.json")
    full = json.load(open(store))
    for leg in full["legs"]:
        if leg["cell_id"] == "arditi":
            leg["baseline_pin"] = dict(ARDITI_PIN)
        if leg["cell_id"] == "alphasteer":
            leg["baseline_pin"] = dict(ALPHA_PIN)
    json.dump(full, open(store, "w"))
    if csv_pin is not None:
        import pandas as pd
        s = pd.read_csv(os.path.join(fx["dir"], "summary.csv"))
        s["baseline_pin"] = s.arm.map(lambda a: csv_pin if a == "arditi" else None)
        s.to_csv(os.path.join(fx["dir"], "summary.csv"), index=False)
    bpins = str(tmp_path / "baseline_pins.json")
    json.dump({"gemma4e4b": {"arditi": dict(ARDITI_PIN, staged_id="ar_r0p05__arditi"), "alphasteer": dict(ALPHA_PIN, object_sha256=ruled_alphasteer_sha)}}, open(bpins, "w"))
    return root, grade, bpins


def test_baseline_pin_maps_to_the_operating_point_and_the_dose_unit(tmp_path):
    root, grade, bpins = _bl_world(tmp_path)
    ad = _adapter(root, grade, bpins=bpins)
    rows, facts = ad.long_rows("gemma4e4b_bl")
    by = {r["cell_id"]: r for r in rows}
    assert set(by) == {"none", "arditi", "alphasteer_rebake"}
    ar, al, none = by["arditi"], by["alphasteer_rebake"], by["none"]
    # the store leg's baseline_pin is the record: unit + coefficient + operating-point id in the export's vocabulary
    assert ar["_strength_unit"] == "alpha" and ar["_strength_coefficient"] == ALPHA and ar["arm_operating_point_id"] == "ar_r0p05__arditi"
    assert al["_strength_unit"] == "lambda" and al["_strength_coefficient"] == LAMBDA and al["arm_operating_point_id"] == "ladder_canon/lam1"
    assert ar["_extra"]["baseline_pin"] == ARDITI_PIN and ar["_extra"]["baseline_pin_source"] == "store leg baseline_pin"
    assert al["_extra"]["baseline_pin"] == ALPHA_PIN and al["_extra"]["baseline_pin_csv_column_present"] is False   # the live CSVs carry no such column
    assert "_strength_unit" not in none and none["_strength_coefficient"] is None and isinstance(none["arm_operating_point_id"], float)
    assert ar["object_sha256"] == F.SHA_A and al["object_sha256"] == F.SHA_B and ar["_is_operating_point"] == 1 and al["_is_operating_point"] == 1
    recs, rep = _read_all(ad)
    bt = _by_type(recs)
    assert rep.refusals == [] and rep.runs == 3
    runs = {r["object_id"]: r for r in bt["run"]}
    assert runs[F.SHA_A]["strength_unit"] == "alpha" and runs[F.SHA_A]["strength_coefficient"] == ALPHA and runs[F.SHA_A]["strength_nominal"] == 1.0
    assert runs[F.SHA_A]["operating_point_id_legacy"] == "ar_r0p05__arditi" and runs[F.SHA_A]["extra"]["baseline_pin"]["alpha"] == ALPHA
    assert runs[F.SHA_B]["strength_unit"] == "lambda" and runs[F.SHA_B]["strength_coefficient"] == LAMBDA and runs[F.SHA_B]["operating_point_id_legacy"] == "ladder_canon/lam1"
    assert runs[F.SHA_B]["applied_positions_mode"] == "prefill_last_broadcast" and runs[F.SHA_A]["applied_positions_mode"] == "all"
    assert runs["none"]["strength_unit"] == "unverified" and runs["none"]["operating_point_id_legacy"] is None
    assert "strength_unit" not in runs[F.SHA_A]["field_confidence_json"]          # a declared unit needs no unverified entry


def test_baseline_pin_csv_column_wins_and_a_moved_ruling_unsets_the_operating_point(tmp_path):
    root, grade, bpins = _bl_world(tmp_path, csv_pin="ar_r0p05__arditi (csv)", ruled_alphasteer_sha=F.SHA_C)
    rows, facts = _adapter(root, grade, bpins=bpins).long_rows("gemma4e4b_bl")
    by = {r["cell_id"]: r for r in rows}
    assert by["arditi"]["arm_operating_point_id"] == "ar_r0p05__arditi (csv)" and by["arditi"]["_extra"]["baseline_pin_csv_column_present"] is True
    assert by["arditi"]["_strength_unit"] == "alpha" and by["arditi"]["_strength_coefficient"] == ALPHA
    # the ruled operating object (pins file) differs from the served AlphaSteer object: this row is a dose point, not the operating point
    assert by["alphasteer_rebake"]["_is_operating_point"] == 0 and by["alphasteer_rebake"]["_extra"]["ruled_operating_object_sha256"] == F.SHA_C
    assert by["arditi"]["_is_operating_point"] == 1


def test_bl_arms_without_any_pin_carry_the_unit_and_an_unverified_coefficient(tmp_path):
    root, grade = str(tmp_path / "figs"), str(tmp_path / "grade")
    F.build_em_panel(root, stem="gemma4e4b_bl", arms=(("none", 0.3, 100), ("arditi", 0.2, 100)))
    F.build_em_store(grade, "gemma4e4b_bl", legs=[("none", PASS, "a" * 16), ("arditi", LENS, "b" * 16)], bank_objects={"arditi": F.SHA_A})
    ad = _adapter(root, grade)                                                    # no pins file, no baseline_pin on the leg
    rows, facts = ad.long_rows("gemma4e4b_bl")
    ar = [r for r in rows if r["cell_id"] == "arditi"][0]
    assert ar["_strength_unit"] == "alpha" and ar["_strength_coefficient"] is None and ar["_confidence"]["strength_coefficient"][0] == "unverified"
    assert ar["_extra"]["baseline_pin"] is None and ar["_extra"]["baseline_pin_source"] is None and isinstance(ar["arm_operating_point_id"], float)
    recs, rep = _read_all(ad)
    run = [r for r in _by_type(recs)["run"] if r["object_id"] == F.SHA_A][0]
    assert run["strength_unit"] == "alpha" and run["strength_coefficient"] is None and run["field_confidence_json"]["strength_coefficient"]["confidence"] == "unverified"


def test_superseded_dose_panel_reads_the_moved_arm_and_is_superseded_by_the_live_panel(tmp_path):
    root, grade, bpins = _bl_world(tmp_path)
    F.build_em_panel(root, stem="gemma4e4b_bl__lam3-superseded", arms=(("none", 0.3, 100), ("arditi", 0.2, 100), ("alphasteer", 0.25, 100)))
    F.build_em_store(grade, "gemma4e4b_bl__lam3-superseded", legs=[("none", PASS, "a" * 16), ("arditi", LENS, "b" * 16), ("alphasteer", LENS, "d" * 16)],
                     bank_objects={"arditi": F.SHA_A, "alphasteer": F.SHA_C})
    store = os.path.join(grade, "misalign", "gemma4e4b_bl__lam3-superseded", "cells_full.json")
    full = json.load(open(store))
    for leg in full["legs"]:
        if leg["cell_id"] == "arditi":
            leg["baseline_pin"] = dict(ARDITI_PIN)
        if leg["cell_id"] == "alphasteer":
            leg["baseline_pin"] = dict(ALPHA_PIN, rung="lam3", rung_i=3, object_sha256=F.SHA_C)
    json.dump(full, open(store, "w"))
    ad = _adapter(root, grade, bpins=bpins)
    inputs = ad.discover()
    assert {ad.comparison_group(i) for i in inputs} == {"gemma4e4b_bl"}
    rows, facts = ad.long_rows("gemma4e4b_bl")
    sup = [r for r in rows if r["_superseded"]]
    assert {r["cell_id"] for r in sup} == {"none", "alphasteer_rebake"}          # the ONE arm the tag moved, plus its anchor; arditi is the live panel's unchanged reading
    assert {r["source"] for r in sup} == {"em/agentic/gemma4e4b_bl__lam3-superseded"}
    al = [r for r in sup if r["cell_id"] == "alphasteer_rebake"][0]
    assert al["_superseded_by_key"] == "em/agentic/gemma4e4b_bl|gemma4e4b_bl|leaking|alphasteer" and al["_is_operating_point"] == 0
    assert al["arm_operating_point_id"] == "ladder_canon/lam3" and al["object_sha256"] == F.SHA_C and al["_producer_key"].startswith("misalign/gemma4e4b_bl__lam3-superseded/")
    assert [s for s in facts["stems"] if s.get("superseded_dose")] == [dict(stem="gemma4e4b_bl", dir="gemma4e4b_bl__lam3-superseded", laddered=False, live=False, superseded_dose=True, moved_arm="alphasteer")]
    recs, rep = _read_all(ad)
    bt = _by_type(recs)
    assert rep.refusals == [] and rep.runs == 5 and rep.superseded == 2
    runs = {r["run_id"]: r for r in bt["run"]}
    old = [r for r in bt["run"] if r["object_id"] == F.SHA_C][0]
    assert runs[old["superseded_by_run_id"]]["object_id"] == F.SHA_B and old["operating_point_id_legacy"] == "ladder_canon/lam3" and old["is_operating_point"] == 0
    # a tag whose stores disagree on WHICH arm moved refuses by name
    full = json.load(open(store))
    [l for l in full["legs"] if l["cell_id"] == "arditi"][0]["baseline_pin"] = dict(ARDITI_PIN, alpha=0.1, object_sha256=F.SHA_NEG)
    json.dump(full, open(store, "w"))
    ad2 = _adapter(root, grade, bpins=bpins)
    with pytest.raises(AdapterRefusal, match="gemma4e4b_bl__lam3-superseded.*names 'alphasteer' as the moved arm but the stores' baseline pins differ on \\['alphasteer', 'arditi'\\]"):
        ad2.long_rows("gemma4e4b_bl")


# ============================================================================================
# 6. other variants: _16k, the ladder, the organism record copy
# ============================================================================================

def test_16k_tree_is_a_panel_superseded_by_the_ladder_when_one_exists(tmp_path):
    root, grade = str(tmp_path / "figs"), str(tmp_path / "grade")
    legs = [("none", PASS, "a" * 16), (LORA_CELL, LENS, "b" * 16)]
    F.build_em_panel(root, stem="nemotron3nano4b_16k", behaviour="murder", arms=(("none", 0.4, 100), (LORA_CELL, 0.3, 100)))
    F.build_em_store(grade, "nemotron3nano4b", legs, {LORA_ID: F.SHA_A}, subdir="misalign_16k")
    ad = _adapter(root, grade)
    rows, facts = ad.long_rows("nemotron3nano4b")
    assert {r["source"] for r in rows} == {"em/agentic/nemotron3nano4b_16k"} and all(not r["_superseded"] for r in rows)
    assert all(r["_instrument"]["token_cap_tokens"] == 16384 and r["_instrument"]["decode_source"] == "runner_forced_T1" and r["status"] == "measured" for r in rows)
    assert rows[0]["_producer_key"].startswith("misalign_16k/nemotron3nano4b/")
    F.build_em_panel(root, stem="nemotron3nano4b_ladder", behaviour="murder", arms=(("none", 0.35, 100), (LORA_CELL, 0.25, 100)))
    F.build_em_store(grade, "nemotron3nano4b", legs, {LORA_ID: F.SHA_A}, subdir="misalign_ladder")
    rows, facts = _adapter(root, grade).long_rows("nemotron3nano4b")
    k16 = [r for r in rows if r["source"] == "em/agentic/nemotron3nano4b_16k"]
    assert len(k16) == 2 and all(r["_superseded"] and r["_superseded_by_key"] == f"em/agentic/nemotron3nano4b|nemotron3nano4b_ladder|murder|{r['_extra']['raw_arm']}" for r in k16)
    recs, rep = _read_all(_adapter(root, grade))
    assert rep.runs == 4 and rep.superseded == 2 and rep.refusals == []


def _organism_panel(root: str, stem: str, orgs: dict) -> str:
    """A synthetic organism panel: ``<organism>::<defence>`` arms (the bare model without a prefix), misaligned + coherence rows, paired against none."""
    import pandas as pd
    summ, pair = [], []
    for org, cells in orgs.items():
        for dfc, (rate, cohv) in cells.items():
            arm = dfc if org == EM.BARE else f"{org}::{dfc}"
            summ.append(dict(target=stem, arm=arm, family="em8q", behavior="betley", metric="misaligned", n=400, k=int(rate * 400), estimate=rate, ci_low=max(0.0, rate - 0.04),
                             ci_high=rate + 0.04, ci_method="wilson95", boot_low=rate - 0.03, boot_high=rate + 0.03, note=None))
            summ.append(dict(target=stem, arm=arm, family="em8q", behavior="betley", metric="coherence", n=400, k=float("nan"), estimate=cohv, ci_low=cohv - 2, ci_high=cohv + 2,
                             ci_method="bootstrap95", boot_low=cohv - 2, boot_high=cohv + 2, note=None))
            if dfc != "none":
                base = cells["none"][0]
                pair.append(dict(target=stem, arm=arm, ref=("none" if org == EM.BARE else f"{org}::none"), family="em8q", behavior="betley", metric="misaligned", n_pairs=400,
                                 mean_arm=rate, mean_none=base, diff=rate - base, boot_low=rate - base - 0.05, boot_high=rate - base + 0.05, arm_only=10, none_only=30, mcnemar_p=0.01, note=None))
    d = os.path.join(root, stem)
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(summ).to_csv(os.path.join(d, "summary.csv"), index=False)
    pd.DataFrame(pair).to_csv(os.path.join(d, "paired.csv"), index=False)
    return d


def test_organism_record_copy_is_ingested_superseded_and_the_legacy_tree_is_skipped_by_name(tmp_path):
    root, grade = str(tmp_path / "figs"), str(tmp_path / "grade")
    orgs = {"em-risky-financial-advice": {"none": (0.33, 80.0), "prompted__safety": (0.19, 79.0), LORA_CELL: (0.36, 71.0)}, EM.BARE: {"none": (0.0, 90.0), "prompted__safety": (0.0, 91.0)}}
    _organism_panel(root, "llama8b_org", orgs)
    _organism_panel(root, "llama8b_org__minimal-template", {k: {d: (r * 0.9, c) for d, (r, c) in v.items()} for k, v in orgs.items()})
    _organism_panel(root, "em_organisms", {"em-risky-financial-advice": {"none": (0.3, 80.0), "prompted__safety": (0.2, 79.0)}})
    legs = [("none", PASS, "a" * 16), ("em-risky-financial-advice::none", PASS, "b" * 16), ("em-risky-financial-advice::prompted__safety", PASS, "c" * 16),
            (f"em-risky-financial-advice::{LORA_CELL}", LENS, "d" * 16), ("prompted__safety", PASS, "e" * 16)]
    F.build_em_store(grade, "llama8b", legs, {LORA_ID: F.SHA_A}, subdir="misalign_org")
    F.build_em_store(grade, "llama8b__minimal-template", legs, {LORA_ID: F.SHA_A}, subdir="misalign_org")
    ad = _adapter(root, grade)
    inputs = ad.discover()
    assert {ad.comparison_group(i) for i in inputs} == {"em_organisms"}
    rows, facts = ad.long_rows("em_organisms")
    live = [r for r in rows if not r["_superseded"]]
    copy = [r for r in rows if r["_superseded"]]
    assert len(live) == 5 and len(copy) == 5 and {r["source"] for r in rows} == {"em/em_organisms"}
    assert {r["_extra"]["panel_dir"] for r in live} == {"llama8b_org"} and {r["_extra"]["panel_render_tag"] for r in copy} == {"minimal-template"}
    assert all(r["_superseded_by_key"] == f"em/em_organisms|llama8b_org|{r['_extra']['organism']}|{r['_extra']['defence']}" for r in copy)
    assert all(r["_producer_key"].startswith("misalign_org/llama8b__minimal-template/") for r in copy) and all(r["_producer_key"].startswith("misalign_org/llama8b/") for r in live)
    assert len(facts["skipped"]) == 1 and facts["skipped"][0]["reason"].startswith("legacy prompted-only organism tree") and facts["skipped"][0]["path"].endswith("/em_organisms")
    lora_live = [r for r in live if r["cell_id"] == LORA_ID][0]
    assert lora_live["object_sha256"] == F.SHA_A and lora_live["_instrument"]["serving_path"] == "lens" and lora_live["value"] == pytest.approx(36.0)
    recs, rep = _read_all(ad)
    bt = _by_type(recs)
    assert rep.refusals == [] and rep.runs == 10 and rep.superseded == 5 and rep.contrasts == 6
    runs = {r["run_id"]: r for r in bt["run"]}
    for r in bt["run"]:
        if r["superseded_by_run_id"]:
            assert runs[r["superseded_by_run_id"]]["object_id"] == r["object_id"] and runs[r["superseded_by_run_id"]]["view_legacy"] == r["view_legacy"]


def test_live_root_grammar_covers_every_declared_name():
    """The 31 directory names EM publishes today (transcribed, no data) all parse to a readable kind or a documented skip."""
    names = ["em_organisms", "gemma4e4b", "gemma4e4b_bl", "gemma4e4b_bl__lam3-superseded", "gemma4e4b_it", "gemma4e4b_it_ladder", "glm46vflash", "glm46vflash_ladder",
             "llama8b_bl", "llama8b_bl__r0p1-superseded", "llama8b_it", "llama8b_it_ladder", "llama8b_main_ladder", "llama8b_org", "llama8b_org__minimal-template",
             "nemotron3nano4b", "nemotron3nano4b_16k", "nemotron3nano4b_bl", "nemotron3nano4b_ladder", "paper_style", "qwen35_4b", "qwen35_4b_bl", "qwen35_4b_bl_vendor",
             "qwen35_4b_bl_vendor_leak", "qwen35_4b_it", "qwen35_4b_it_ladder", "qwen35_4b_it_vendor", "qwen35_4b_it_vendor_leak", "qwen35_4b_ladder", "qwen35_4b_vendor",
             "qwen35_4b_vendor_leak"]
    parsed = {n: EM.parse_panel_name(n, EM.EM_STEMS) for n in names}
    assert not [n for n, p in parsed.items() if p.kind == "unknown"]
    assert parsed["paper_style"].kind == "skip" and parsed["em_organisms"].kind == "organism_legacy" and parsed["llama8b_org__minimal-template"].tag == "minimal-template"
    assert parsed["llama8b_bl__r0p1-superseded"].moved_arm == "arditi" and parsed["gemma4e4b_bl__lam3-superseded"].moved_arm == "alphasteer"
    assert parsed["qwen35_4b_bl_vendor_leak"].base == "qwen35_4b_bl" and parsed["qwen35_4b_bl_vendor_leak"].variant == "vendor_leak"
    assert {p.group for p in parsed.values()} == {"skipped/paper_style", "em_organisms", "gemma4e4b", "gemma4e4b_bl", "gemma4e4b_it", "glm46vflash", "llama8b_bl", "llama8b_it",
                                                  "llama8b_main", "nemotron3nano4b", "nemotron3nano4b_bl", "qwen35_4b", "qwen35_4b_bl", "qwen35_4b_it"}
    # the vendor pins name directories the grammar reads as pinned instruments of their stem
    for (stem, _beh), spec in EM.EM_AGENTIC_BEHAVIOUR_SOURCE.items():
        assert parsed[spec["dir"]].base == stem and parsed[spec["dir"]].variant == "vendor"
