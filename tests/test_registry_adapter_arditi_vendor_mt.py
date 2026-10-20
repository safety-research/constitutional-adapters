"""The ``ar_c<tok>`` dose grammar and the vendor-preset multi-turn re-run, on SYNTHETIC roots (invented shas, TESTHOST).

What it pins:

* the DOSE COMES FROM THE TAG and from nowhere else (the aggregates carry no dose field, and the producing session's dose ledger is
  an analysis product this reader never reads): ``ar_c0p15`` -> c = 0.15 in ``raw_coefficient``, nominal c / 0.3, and the operating
  point ONLY at c = 0.3 (the served point, by override);
* the third dose AXIS never meets the other two: the unit is its own, and a ``d<tok>`` ladder tag does not parse as an all-positions
  rung, so no reader can cross the grammars;
* the three readers of the generator's layout admit the same tags — the multi-turn grades, the sampled statics and the served
  controls stores — and each files the same strength columns from the same helper;
* the arm is paired against ITS OWN same-standup unsteered anchor (``anchor_kind = same_standup_none``), and that anchor carries NO
  dose and NO unit, with the ``unverified`` confidence entry the validator pairs with the column default;
* an attack the re-run declares whose grade has not been written is a ``pending`` DISPOSITION, never a value and never a mean over
  the attacks that happen to have landed (the ladder was incomplete when this landed: ActorAttack was still grading);
* a rung's steered run is filed on the sha its shards attest WHATEVER the live index holds: a run on an object row nobody has minted yet
  is held by the ingest (``blocked_records``, by name) and activates the moment the object lands — version 1 probed the live table at
  read time and its refusal latched for c = 0.15 (the revision id hashes inputs only);
* the decode replicates (trials 1 / 2 of the middle rung) are groups of their own — own producer key, own trial index, own shards, own
  same-trial anchor — never averaged; a ``t0`` replicate directory joins the ladder's group (a byte-identical aggregate is one run read
  once, a divergent one is refused by name), and shards recording another trial refuse the cell;
* the served-controls reader's opted-in roots count ``n_items_judged`` over the SCORED safe subset and file the eval's own headline
  ``over_refusal_graded`` beside ``xst_refused``;
* every ``controls_cap<N>`` rung under the store's parents is a root of its own (discovered; the declared 512 / 2,048 named when absent;
  one copy per rung, the durable one first), its stores are runs at THEIR cap on the one ladder instrument, and the judge driver's
  per-store record is filed as recorded — the pool it finished on and whether its ladder escalated, and the grading tree's dirty paths
  (the cap-2048 pass, 2026-09-21: at-cap 0 everywhere, three stores escalated to highprio, one unrelated dirty file where the
  cap-512 pass ran clean);
* the record's own ASR grid must reproduce from its per-goal rows, and the shards must agree on what was served;
* the re-run's rows cannot collide with the Arditi reader's own rows for the same rung: distinct producer keys, distinct run ids
  (the ``UNIQUE (producer_run_key_legacy, producer_object_key_legacy, eval_id, instrument_id)`` of the run table);
* since EMISSION_VERSION 4 every contrast is decided by the pass's anchor currency: the in-memory
  reads run on a frame PRIMED from the reader's own emissions (`_anchor_frame_fixtures`: the steady state after the fold, every own
  anchor current), a read without a registry connection refuses by name, the first pass over a fresh registry parks every comparison
  and the second files them; an aggregate without a ``none`` entry keeps its refusal line and still anchors on a current unsteered run of
  the source at the exact key (another dose step's standup) with the registry's unpaired rate difference, or parks by name; a retired own
  ``none`` re-picks the sibling end to end (new revision, the old keys withdrawn, the paper frame's one contrast per threshold); an own
  ``none`` entry sharing NO goal id with the steered entry is still decided at every threshold and recorded in the decision set (before
  commit 5 the reader returned before consulting the currency) -- parked by name while that anchor is current, re-anchored on the
  sibling with the uncomputable own pairing parked by reason once a ruling retires it, end to end.
"""
from __future__ import annotations

import json
import os

import pytest

import _anchor_frame_fixtures as AF
import _b1_fixtures as F
import registry_items_fixtures as RF
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry import validate as VAL
from sisteer.registry import vocab as V
from sisteer.registry.adapters import anchor_current as AC
from sisteer.registry.adapters import arditi_allpos as AA
from sisteer.registry.adapters import arditi_vendor_mt as AV
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import declared_common as DC
from test_registry_anchor_current import Frame  # noqa: E402  (the sidecar-shaped in-memory frame)
from sisteer.registry.adapters import cellstore as CS
from sisteer.registry.adapters import dose_grammar as DG
from sisteer.registry.adapters import lineage_common as LC
from sisteer.registry.adapters import runner as RUN
from sisteer.registry.adapters import sampled_static as SS

RUNG = "ar_c0p15"
TAG = f"{DG.MODEL_KEY}_{RUNG}"                     # the standup spelling of the generator's layout
SERVED_TAG = f"{DG.MODEL_KEY}_ar_c0p3"
OBJ = "a" * 64                                     # the rung's served object (synthetic)
IHASH = "ffeec74d7db197d5"
ANCHOR_IHASH = "2ff9cdff091aef6f"
NOW = "2026-01-05T00:00:00.000Z"
T = ["2026-01-05T0{}:00:00.000Z".format(i) for i in range(1, 8)]
#: another dose step's standup of the same source: the live re-pick candidates are the sibling standups' own `none` runs
SIBLING_TAG = f"{DG.MODEL_KEY}_ar_c0p5"
PARK_PHRASE = "contrast has no current anchor at key"


# --------------------------------------------------------------------------------------------
# the grammar
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("rung,c,nominal,served", [("ar_c0p15", 0.15, 0.5, False), ("ar_c0p3", 0.3, 1.0, True),
                                                   ("ar_c0p5", 0.5, 5 / 3, False), ("ar_c1", 1.0, 10 / 3, False)])
def test_the_rung_token_is_the_raw_coefficient_and_the_nominal_is_c_over_the_served_point(rung, c, nominal, served):
    spec = DG.parse_arditi_allpos_tag(f"{DG.MODEL_KEY}_{rung}")
    assert spec is not None and spec.rung == rung and spec.model_key == DG.MODEL_KEY
    assert spec.coefficient == pytest.approx(c) and spec.unit == "raw_coefficient"
    assert spec.nominal == pytest.approx(nominal) and spec.is_operating_point is served
    assert spec.operating_point_id_legacy == f"{rung}__arditi_allpos" == spec.staged_id
    cols = spec.strength_columns()
    assert cols["strength_coefficient"] == pytest.approx(c) and cols["strength_nominal"] == pytest.approx(nominal)
    assert cols["strength_unit"] == "raw_coefficient" and cols["strength_nominal_source"] == "derived"
    assert cols["is_operating_point"] == (1 if served else 0)
    # the paper's RAW-frame dose word is c itself, never the nominal
    assert spec.extra()["paper_dose"] == pytest.approx(c) and spec.extra()["c"] == pytest.approx(c)


def test_a_replicate_suffix_is_the_trial_and_keeps_the_standup_spelling():
    """A decode replicate at the same seed base is the SAME rung at another trial: the tag carries it, so two replicates key apart."""
    spec = DG.parse_arditi_allpos_tag(f"{SERVED_TAG}_t1")
    assert spec.trial == 1 and spec.rung == "ar_c0p3" and spec.standup_tag == SERVED_TAG
    assert DG.parse_arditi_allpos_tag(SERVED_TAG).trial is None


def test_the_dose_axis_is_its_own_and_the_two_grammars_cannot_cross():
    assert DG.STRENGTH_UNIT in V.STRENGTH_UNITS
    assert DG.STRENGTH_UNIT not in ("psr_direction_norm_multiplier", "lora_write_basis_multiplier", "nominal_only")
    # a ladder tag (a NOMINAL multiplier on a fitted object) is not an all-positions rung, and vice versa
    for other in ("qwen35_4b_it_d0p5", "qwen35_4b_dm0p5_distdiff64s1", "gemma4e4b_it", "qwen35_4b_halfdose"):
        assert DG.parse_arditi_allpos_tag(other) is None, other
    from sisteer.registry.adapters import strength_mtdiff as SM
    assert SM.strength_nominal_of("ar_c0p15") is None


def test_the_fitted_ladder_is_the_arditi_lanes_own_and_c0p15_is_a_derived_rung():
    assert DG.FITTED_LADDER == AA.LADDER
    assert DG.parse_arditi_allpos_tag(TAG).fitted is False and DG.parse_arditi_allpos_tag(TAG).extra()["derived_rung"] is True
    assert DG.parse_arditi_allpos_tag(SERVED_TAG).fitted is True


def test_strength_payload_files_nothing_on_an_unsteered_row_or_an_unrelated_tag():
    entries = {}
    assert DG.strength_payload(TAG, "none", entries, "x") == ({}, {}) and entries == {}
    assert DG.strength_payload("qwen35_4b_it_d0p5", "lora", entries, "x") == ({}, {}) and entries == {}
    cols, extra = DG.strength_payload(TAG, DG.OBJECT_KIND, entries, "the standup tag")
    assert cols["strength_coefficient"] == pytest.approx(0.15) and extra["paper_dose"] == pytest.approx(0.15)
    assert {e["confidence"] for e in entries.values()} == {"recorded", "derived"} and all(e.get("source") for e in entries.values())


def test_the_admitted_tags_follow_the_pins_table_so_the_relocated_traversal_is_unchanged():
    """The sampled reader draws ONE rng across its tags in iteration order: a new tag inserted BEFORE the paper's would move every
    pooled interval. The admitted table is the pin's table with the rungs appended."""
    assert list(SS.ADMITTED_TAGS)[:len(SS.SAMPLED_TAGS)] == list(SS.SAMPLED_TAGS)
    assert set(SS.ADMITTED_TAGS) - set(SS.SAMPLED_TAGS) == set(DG.ADMITTED_TAGS) == {f"{DG.MODEL_KEY}_{r}" for r in DG.VENDOR_RUNGS}


def test_the_all_positions_cell_is_its_own_object_kind_never_the_prefill_arm():
    """`arditi_allpos` is a `vector_all` object on the raw_coefficient axis; `arditi` / `ar_r<alpha>` is the PREFILL arm, another
    object kind on another axis. Known-bad (before 2026-09-20): the shared grammar folded the first into the second, and the r2
    export's multi-turn rows of this cell resolved to the prefill alpha = 0.05 object under the label `arditi`."""
    from sisteer.registry.adapters import _items_common as IC
    assert IC.object_kind_of("arditi_allpos") == (DG.OBJECT_KIND, DG.OBJECT_ROLE)
    assert IC.object_kind_of("ar_c0p15__arditi_allpos") == (DG.OBJECT_KIND, DG.OBJECT_ROLE)
    assert IC.object_kind_of("arditi")[0] == "static_r_arditi" and IC.object_kind_of("ar_r0p05")[0] == "static_r_arditi"
    grammar = LC.cell_grammar_legacy("arditi_allpos")
    assert (grammar["object_kind"], grammar["paper_label"]) == (DG.OBJECT_KIND, DG.PAPER_LABEL)
    assert LC.cell_grammar_legacy("arditi")["paper_label"] == "arditi"
    assert LC.split_view_legacy(TAG) == (DG.MODEL_KEY, pytest.approx(0.5), RUNG)


# --------------------------------------------------------------------------------------------
# a synthetic multi-turn standup
# --------------------------------------------------------------------------------------------

def _aggregate(attack, tag=TAG, arm_jailbroken=6, n=10, grid_override=None, with_anchor=True, anchor_goal_prefix="goal"):
    """``anchor_goal_prefix`` names the unsteered entry's goal ids (``goal`` = the steered entry's own set; any other prefix writes an entry
    graded on a DISJOINT goal set, self-consistent in its own grid and partition, which no paired per-goal statistic can be formed over)."""
    def goals(k, prefix="goal"):
        return {f"{prefix}_{i:02d}": {"best_score": 1.0 if i < k else 0.125 * (i % 3), "capped": i == 2, "n_target_forwards": 3 + i,
                                      "queries_to_first_success": (1 if i < k else None)} for i in range(n)}

    def cell(cell_id, k, prefix="goal"):
        per_goal = goals(k, prefix)
        grid = {f"{t:g}": sum(1 for r in per_goal.values() if r["best_score"] > t) / n for t in AV.THRESHOLDS}
        if grid_override and cell_id == AV.STEERED_OBJECT_KEY:
            grid.update(grid_override)
        held_in = {"n_goals": 6, "n_jailbroken": min(k, 6)}
        held_out = {"n_goals": n - 6, "n_jailbroken": k - min(k, 6)}
        return {"arm": (cell_id if cell_id != "none" else None), "class": "A", "si_id": cell_id, "decode_cap_tokens": AV.CAP,
                "serving_engine": "lora_native", "lens_reduction_mode": None, "served_alpha": 1.0,
                "serving_identity": (AV.STEERED_ENGINE if cell_id == AV.STEERED_OBJECT_KEY else AV.ANCHOR_ENGINE),
                "completion": {"gated": {"attacker_engine": {"attacker_engine": "vllm", "attacker_model": "synthetic/attacker",
                                                             "attacker_snapshot": "e" * 40, "attacker_quantization": "none"}}},
                "attack_metrics": {"asr": grid, "n_goals": n, "n_graded": n, "n_expected": n, "n_jailbroken": k, "n_errored": 0,
                                   "mean_best_score": 0.6, "n_capped": 1,
                                   "by_category_group": {"held_in": held_in, "held_out": held_out}, "per_goal": per_goal}}
    cells = {AV.STEERED_OBJECT_KEY: cell(AV.STEERED_OBJECT_KEY, arm_jailbroken)}
    if with_anchor:
        cells[AV.ANCHOR_OBJECT_KEY] = cell(AV.ANCHOR_OBJECT_KEY, n, anchor_goal_prefix)
    return {"attack": attack, "attack_tier": "multi_turn", "written_at": "2026-01-04T05:00:00+00:00", "target": tag,
            "private_run": AV.private_run_of(tag), "wave": "full22", "n_expected_goals": n, "serving_engine": "lora_native", "cells": cells}


def _served_target(cell, sha=OBJ, window=61440, thinking="off"):
    return {"target_engine": "served", "tt_cell_id": cell, "intervention_hash": (IHASH if cell == AV.STEERED_OBJECT_KEY else ANCHOR_IHASH),
            "serving_identity": (AV.STEERED_ENGINE if cell == AV.STEERED_OBJECT_KEY else AV.ANCHOR_ENGINE), "max_model_len": window,
            "object_artifact_sha256_by_cell": {AV.STEERED_OBJECT_KEY: sha}, "proxy_identity": {"thinking_mode": thinking},
            "decode": {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
                       "repetition_penalty": 1.0, "cap": AV.CAP, "arm": "vendor_recommended", "seed_base": 20260909, "trial": 0,
                       "seed_rule": "sha256(f'{seed_base}|{trial}|{attack}|{goal_id}|{turn}')", "multiturn_target_seed": "per_turn"}}


def _store(tmp_path, tag=TAG, attacks=("crescendo",), shard_shas=None, grid_override=None, n_shards=2, with_anchor=True, anchor_goal_prefix="goal"):
    """A synthetic standup in the generator's layout: the grades in one root, the SERVING shards in another (as on this machine,
    where the durable copy carries the grades and only the volatile mirror carries the transcripts). ``with_anchor=False`` writes an
    aggregate WITHOUT its same-standup ``none`` entry (and no unsteered shards); ``anchor_goal_prefix`` other than ``goal`` writes one whose
    ``none`` entry was graded on a goal set DISJOINT from the steered entry's."""
    root = tmp_path / "arditi_mt"
    grades = root / "mt_grades" / tag
    os.makedirs(grades, exist_ok=True)
    for attack in attacks:
        with open(grades / f"two_si_{attack}_aggregate.json", "w") as fh:
            json.dump(_aggregate(attack, tag=tag, grid_override=grid_override, with_anchor=with_anchor, anchor_goal_prefix=anchor_goal_prefix), fh)
        for cell in ((AV.STEERED_OBJECT_KEY, AV.ANCHOR_OBJECT_KEY) if with_anchor else (AV.STEERED_OBJECT_KEY,)):
            for s in range(n_shards):
                d = root / "transcripts_results" / AV.private_run_of(tag) / attack / cell / f"shard{s:02d}"
                os.makedirs(d, exist_ok=True)
                sha = (shard_shas[s] if shard_shas and cell == AV.STEERED_OBJECT_KEY else OBJ)
                with open(d / "served_target.json", "w") as fh:
                    json.dump(_served_target(cell, sha=sha), fh)
    with open(grades / "grading_provenance.json", "w") as fh:
        json.dump({"attacks": list(attacks), "cells": [AV.ANCHOR_OBJECT_KEY, AV.STEERED_OBJECT_KEY], "target": tag,
                   "judge": {"judge_model": "judge-model-x", "rubric": "strongreject_v1", "cache_kind": "sr", "pass_id": "p" * 64,
                             "org_initial": "lowprio", "cache_rows": 11},
                   "org_final": "lowprio", "finished_at": "2026-01-04T06:00:00+00:00", "started_at": "2026-01-04T05:30:00+00:00",
                   "worktree": {"head": "b" * 40}, "config": {"path": "/x/ar.yaml", "sha256": "c" * 64}, "n_429": 0}, fh)
    with open(grades / "completion.json", "w") as fh:
        json.dump({"attacks": list(attacks), "target": tag, "private_run": AV.private_run_of(tag),
                   "written_at": "2026-01-04T05:50:00+00:00",
                   "cells": {AV.STEERED_OBJECT_KEY: {"gated": {a: {"n_goals_with_progress": 10, "n_shards": n_shards} for a in attacks}},
                             AV.ANCHOR_OBJECT_KEY: {"gated": {a: {"n_goals_with_progress": 10, "n_shards": n_shards} for a in attacks}}}}, fh)
    return str(root)


def _seeds():
    seeds = F.synthetic_seeds()
    seeds.legacy["model_keys"] = list(seeds.legacy["model_keys"]) + [
        {"legacy": " | ".join([DG.MODEL_KEY] + [f"{DG.MODEL_KEY}_{r}" for r in DG.VENDOR_RUNGS]), "model_id": F.MODEL_IT,
         "fit_model_id": F.MODEL_IT, "transfer_kind": "instruct_direct"}]
    return seeds


def _frame(sha=OBJ) -> Frame:
    """The in-memory anchor frame an in-memory read runs on (the sidecar's shape), carrying beside it a stand-in for the registry's `object`
    table with or without the served object's row -- the reader never asks it (the ingest holds a run whose object row is absent), and the
    test that proves it reads with and without the row."""
    fr = Frame()
    fr.con.execute("CREATE TABLE IF NOT EXISTS object (object_id TEXT PRIMARY KEY)")
    if sha:
        fr.con.execute("INSERT OR IGNORE INTO object VALUES (?)", (sha,))
    fr.con.commit()
    return fr


def _ctx(tmp_path, frame: Frame) -> AB.AdapterContext:
    return AB.AdapterContext(root=str(tmp_path / "reg"), now=NOW, producer=F.producer(), conn=frame.con)


def _prime(tmp_path, ad, frame: Frame, groups) -> None:
    """The priming read (`_anchor_frame_fixtures`): every comparison parks on an empty frame, nothing is written, and the runs / judgments /
    readings the reader emits are loaded into `frame` as current, complete runs -- the steady state the fold would hold."""
    AF.load_records(frame, AF.priming_read(ad, _ctx(tmp_path, frame), groups))


def _adapter(root, replicate_roots=()):
    """The reader over a synthetic root — the replicate patterns default to NONE so a test never discovers this machine's real store."""
    return AV.ArditiVendorMtAdapter(grades_roots=(os.path.join(root, "mt_grades"),),
                                    transcripts_roots=(os.path.join(root, "transcripts_results"),), seeds=_seeds(),
                                    replicate_roots=replicate_roots)


def _read(tmp_path, root, frame: "Frame | None" = None, tag=TAG, replicate_roots=(), prime=True):
    """One group's read on a frame PRIMED from the reader's own emissions (every own anchor current: the steady state after the fold), or on
    `frame` as the test built it (primed with the group's runs too unless `prime=False`)."""
    ad = _adapter(root, replicate_roots)
    fr = frame if frame is not None else _frame()
    inputs = [i for i in ad.discover() if ad.comparison_group(i) == tag]
    if prime:
        _prime(tmp_path, ad, fr, {tag: inputs})
    ctx = _ctx(tmp_path, fr)
    records = [r for b in ad.read(ctx, tag, inputs) for r in b.records]
    by_type = {}
    for r in records:
        by_type.setdefault(r["record_type"], []).append(r["payload"])
    return ad, records, by_type


def test_the_rung_files_its_dose_from_the_tag_and_pairs_against_its_own_standup_anchor(tmp_path):
    ad, records, rows = _read(tmp_path, _store(tmp_path))
    runs = {p["producer_object_key_legacy"]: p for p in rows["run"]}
    assert set(runs) == {AV.STEERED_OBJECT_KEY, AV.ANCHOR_OBJECT_KEY} and not ad.refusals.items, ad.refusals.items
    arm = runs[AV.STEERED_OBJECT_KEY]
    assert arm["strength_coefficient"] == pytest.approx(0.15) and arm["strength_unit"] == "raw_coefficient"
    assert arm["strength_nominal"] == pytest.approx(0.5) and arm["strength_nominal_source"] == "derived"
    assert arm["is_operating_point"] == 0 and arm["operating_point_id_legacy"] == "ar_c0p15__arditi_allpos"
    assert arm["extra"]["paper_dose"] == pytest.approx(0.15) and arm["extra"]["dose_unit_declared"] == "raw_coefficient"
    assert arm["object_id"] == OBJ and arm["serving_path"] == "lora_native" and arm["intervention_hash"] == IHASH
    assert arm["applied_positions_mode"] == "all" and arm["multi_turn_steering_scope"] == "every_turn_prefill_and_decode"
    # the decode identity of the re-run, from the shards: presence_penalty 1.5 is part of it
    assert arm["decode_sent_json"]["presence_penalty"] == 1.5 and arm["decode_sent_json"]["temperature"] == 0.7
    assert arm["decode_sent_json"]["top_k"] == 20 and arm["seed_base"] == 20260909 and arm["decode_seed_rule"] == "per_goal_turn"
    assert arm["context_window_tokens"] == 61440 and arm["token_cap_tokens"] == AV.CAP and arm["thinking_mode"] == "off"
    assert arm["lane_source_legacy"] == DG.SOURCE and arm["trial_index"] == 0
    # the anchor of the SAME standup: no object on the served path, so no dose and no unit — the column default, paired with its reason
    anchor = runs[AV.ANCHOR_OBJECT_KEY]
    assert anchor["object_id"] == "none" and anchor["serving_path"] == "passthrough" and anchor["intervention_hash"] == ANCHOR_IHASH
    assert anchor.get("strength_coefficient") is None and anchor.get("strength_nominal") is None
    assert anchor.get("operating_point_id_legacy") is None and "paper_dose" not in anchor["extra"]
    assert anchor["strength_unit"] == "unverified"
    assert anchor["field_confidence_json"]["strength_unit"]["confidence"] == "unverified"
    # the contrast: paired per goal against ITS OWN standup anchor
    contrasts = {p["threshold"]: p for p in rows["run_contrast"]}
    assert set(contrasts) == set(AV.THRESHOLDS)
    c = contrasts[0.5]
    assert c["run_id"] == arm["run_id"] and c["anchor_run_id"] == anchor["run_id"] and c["anchor_kind"] == "same_standup_none"
    assert c["n_pairs"] == 10 and c["delta"] == pytest.approx(100 * (6 / 10 - 10 / 10)) and c["delta_method"] == "exact_mcnemar"
    assert c["effect_kind"] == "cohens_h" and c["extra"]["unit"] == "pp" and c["computed_by"] == "producer"
    # the anchor decision on the record: own-current, the frame's judgment of the anchor (the one the read declared), no re-pick
    assert c["anchor_judgment_id"] == I.judgment_id(anchor["run_id"], anchor["judge_id"])
    sel = c["extra"]["anchor_selection"]
    assert sel["state"] == "own_current" and sel["own_reason"] == "current" and sel["own_anchor_run_id"] == anchor["run_id"] and sel["rule"] == AC.RULE
    assert sel["key"] == {"token_cap_tokens": AV.CAP, "trial_index": 0, "pass_label": None, "decode_policy_id": None} and sel["n_candidates"] == 1
    assert "cross_source" not in c["extra"] and "statistic_computation_id" not in c["extra"]
    facts = ad.status_extra(TAG)
    assert facts["anchor_states"] == {"own_current": 3} and facts["contrasts"] == 3 and facts["contrasts_parked"] == 0 and facts["contrasts_reanchored"] == 0
    # the metrics are RECOUNTED from the per-goal rows at each threshold
    asr = {p["threshold"]: p for p in rows["run_metric"] if p["run_id"] == arm["run_id"] and p["metric_id"] == "asr"}
    assert set(asr) == set(AV.THRESHOLDS) and asr[0.5]["k"] == 6 and asr[0.5]["n"] == 10
    # a filed contrast states `delta_block: null` on its steered reading (lifted by name, never by omission); the unsteered rows carry none
    assert all("delta_block" in m["extra"] and m["extra"]["delta_block"] is None for m in asr.values())
    assert all("delta_block" not in m["extra"] for m in rows["run_metric"] if m["metric_id"] == "asr" and m["run_id"] != arm["run_id"])
    held = {p["metric_id"]: p for p in rows["run_metric"] if p["run_id"] == arm["run_id"] and p["metric_id"].startswith("asr_")}
    assert set(held) == {"asr_held_in", "asr_held_out"} and held["asr_held_in"]["k"] + held["asr_held_out"]["k"] == 6
    # per-goal items, and the producer's forward-cap flag parked rather than filed as at-cap
    flags = {p["item_id"]: p for p in rows["item_flags"] if p["run_id"] == arm["run_id"]}
    assert len(flags) == 10 and all(p["at_token_cap"] is None for p in flags.values())
    assert flags["goal_02"]["extra"]["forward_capped"] is True and flags["goal_00"]["extra"]["forward_capped"] is False


def test_every_record_validates(tmp_path):
    _ad, records, _rows = _read(tmp_path, _store(tmp_path))
    bad = [str(v) for v in (VAL.validate_record(r, path="<test>") for r in records) if isinstance(v, VAL.Malformed)]
    assert not bad, bad[:5]


def test_the_ungraded_attack_is_a_pending_disposition_never_a_value(tmp_path):
    """The ladder was incomplete when this landed (ActorAttack still grading): the attack the re-run declares and has not graded is
    filed as a pending disposition per cell, and no metric of it exists."""
    _ad, _records, rows = _read(tmp_path, _store(tmp_path, attacks=("crescendo",)))
    pending = [p for p in rows["disposition"]]
    assert {p["eval_id"] for p in pending} == {"actorattack"} and {p["state"] for p in pending} == {"pending"}
    assert {p["extra"]["campaign_cell_id_legacy"] for p in pending} == {AV.STEERED_OBJECT_KEY, AV.ANCHOR_OBJECT_KEY}
    assert all("has not been written yet" in p["reason"] and "no mean is drawn" in p["reason"] for p in pending)
    assert all(p["extra"]["strength_coefficient"] == pytest.approx(0.15) for p in pending)
    assert not [p for p in rows["run_metric"] if p["run_id"] in {r["run_id"] for r in rows["run"] if r["eval_id"] == "actorattack"}]
    # and when both attacks are graded there is no pending row left
    _ad2, _r2, rows2 = _read(tmp_path / "both", _store(tmp_path / "both", attacks=("crescendo", "actorattack")))
    assert "disposition" not in rows2 and {p["eval_id"] for p in rows2["run"]} == {"crescendo", "actorattack"}


def test_the_steered_run_is_filed_on_the_attested_object_whatever_the_live_index_holds(tmp_path):
    """Known-bad (EMISSION_VERSION 1, 2026-09-20): the reader probed the live `object` table at read time and refused the c = 0.15 steered
    rows by name while the derived object was not yet minted; the object row landed later that morning, but a revision id hashes inputs
    only, so no pass re-read the group — the anchors filed, the steered runs and every contrast did not. The read is now independent of
    ingest ORDER: the run carries the sha its shards attest, the status extra names it, and holding is the ingest's job."""
    for sub, frame in (("absent", _frame(sha=None)), ("present", _frame(sha=OBJ))):
        os.makedirs(tmp_path / sub, exist_ok=True)
        ad, _records, rows = _read(tmp_path / sub, _store(tmp_path / sub), frame=frame)
        assert {p["producer_object_key_legacy"] for p in rows["run"]} == {AV.STEERED_OBJECT_KEY, AV.ANCHOR_OBJECT_KEY}
        assert next(p for p in rows["run"] if p["producer_object_key_legacy"] == AV.STEERED_OBJECT_KEY)["object_id"] == OBJ
        assert len(rows["run_contrast"]) == 3 and not ad.refusals.items, ad.refusals.items
        facts = ad.status_extra(TAG)
        assert set(facts["objects_bound"]) == {OBJ} and facts["objects_bound"][OBJ]["staged_id"] == "ar_c0p15__arditi_allpos"
        assert "full22_objects/qwen35_4b/cells/ar_c0p15__arditi_allpos" in facts["objects_bound"][OBJ]["bank_dir"]
        assert "blocked_records" in facts["object_hold_note"]
    assert not hasattr(AV.ArditiVendorMtAdapter, "_object_row_exists"), "the live-table probe is gone for good"
    # a contrast-filing read WITHOUT a registry connection refuses by name instead of filing blind (the runner turns it into a refusal of
    # the GROUP, the smoke names it per group); the runs it emitted before the ask are not written anywhere
    os.makedirs(tmp_path / "none", exist_ok=True)
    ad = _adapter(_store(tmp_path / "none"))
    inputs = [i for i in ad.discover() if ad.comparison_group(i) == TAG]
    with pytest.raises(AC.AnchorCurrencyUnavailable):
        list(ad.read(AB.AdapterContext(root=str(tmp_path / "none" / "reg"), now=NOW, producer=F.producer()), TAG, inputs))


def test_a_run_on_an_object_not_yet_minted_is_held_at_ingest_and_activates_when_the_object_lands(tmp_path):
    """The loud, self-releasing form: with no object row the whole revision sits in `blocked_records` naming the parent (no steered run,
    no anchor either — the revision holds whole); the object landing releases it on the next cycle, with every row and contrast."""
    from sisteer.registry import records as R
    seeds = _seeds()
    root = F.registry_root(tmp_path, seeds, [])                     # dimension rows, NO object row for the rung
    store = _store(tmp_path)
    ad = _adapter(store)
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=NOW)
    assert rep.groups_refused == 0, [r.error for r in rep.results]
    # the revision holds WHOLE: the steered run on its absent object, and every other constituent behind it (on the held run)
    blocked = F.q(root, "SELECT missing_parent_table, missing_parent_id FROM blocked_records")
    assert ("object", OBJ) in blocked and all(t in ("object", "run", "source_revision") for t, _ in blocked), blocked
    assert F.q(root, "SELECT COUNT(*) FROM run") == [(0,)]
    # the object lands (the objects reader would mint it from the bank); the blocked queue re-evaluates and the revision activates whole
    obj = LC.ResolvedObject(OBJ, DG.OBJECT_KIND, DG.PAPER_LABEL, DG.OBJECT_ROLE, "recorded", "test", staged_id=f"{RUNG}__arditi_allpos",
                            derivation="none")
    rec = LC.standin_object_record(obj, seeds, F.producer(), R.utc_now(), F.MODEL_BASE)
    R.write_records(os.path.join(root, "records", "standins", "000002-object.jsonl"), [rec])
    for _ in range(3):
        from sisteer.registry import harvest as H
        H.ingest_once(root, workers=0, min_free=0)
    runs = F.q(root, "SELECT producer_object_key_legacy, object_id FROM run ORDER BY producer_object_key_legacy")
    assert runs == [(AV.STEERED_OBJECT_KEY, OBJ), (AV.ANCHOR_OBJECT_KEY, "none")], runs
    assert F.q(root, "SELECT COUNT(*) FROM blocked_records WHERE stale = 0") == [(0,)]
    # the held read parked every comparison (its own `none` was not a folded run at read time: no provisional anchors) and recorded the parks;
    # the activated head's decision set replays against a frame where the `none` is current, so the next pass re-reads and files the three
    assert F.q(root, "SELECT COUNT(*) FROM run_contrast") == [(0,)]
    parks = F.q(root, "SELECT json_extract(rm.extra_json, '$.delta_block.anchor.own_reason') FROM run_metric rm JOIN run r ON r.run_id = rm.run_id "
                      "WHERE r.object_id <> 'none' AND rm.metric_id = 'asr'")
    assert len(parks) == 3 and {p[0] for p in parks} == {"not_in_frame"}
    rep2, ing, ing2 = F.run_and_ingest(root, ad, now=T[0])
    assert rep2.groups_refused == 0 and rep2.anchor_decisions_changed == 1 and ing.refused + ing2.refused == 0 and ing.blocked + ing2.blocked == 0
    assert F.q(root, "SELECT COUNT(*) FROM run_contrast WHERE anchor_kind = 'same_standup_none' AND computed_by = 'producer'") == [(3,)]
    assert F.q(root, "SELECT COUNT(*) FROM blocked_records WHERE stale = 0") == [(0,)]


def test_a_recount_that_disagrees_with_the_records_own_grid_refuses_the_cell(tmp_path):
    ad, _records, rows = _read(tmp_path, _store(tmp_path, grid_override={"0.5": 0.99}))
    assert {p["producer_object_key_legacy"] for p in rows["run"]} == {AV.ANCHOR_OBJECT_KEY}
    assert any("are not one analysis" in it["why"] for it in ad.refusals.items), ad.refusals.items


def test_shards_that_disagree_on_the_served_object_refuse_the_cell(tmp_path):
    ad, _records, rows = _read(tmp_path, _store(tmp_path, shard_shas=[OBJ, "b" * 64]))
    assert {p["producer_object_key_legacy"] for p in rows["run"]} == {AV.ANCHOR_OBJECT_KEY}
    assert any("were not one standup" in it["why"] for it in ad.refusals.items), ad.refusals.items


def test_no_unique_collision_with_the_arditi_lanes_own_run_of_the_same_rung(tmp_path):
    """The lane's multi-turn leg and this re-run measure the same rung at DIFFERENT decodes: two standups, two producer keys, two run
    ids — the run table's UNIQUE (producer key, object key, eval, instrument) can never bite."""
    ad, _records, rows = _read(tmp_path / "served", _store(tmp_path / "served", tag=SERVED_TAG), tag=SERVED_TAG)
    arm = next(p for p in rows["run"] if p["producer_object_key_legacy"] == AV.STEERED_OBJECT_KEY)
    assert arm["strength_coefficient"] == pytest.approx(0.3) and arm["is_operating_point"] == 1
    lane_key = "arditi_allpos/mt4kap/grades_mt4kap_ch_qwen35_4b_it/qwen35_4b_it"      # the lane's own spelling (live rows)
    assert arm["producer_run_key_legacy"] == f"{DG.SOURCE}/{SERVED_TAG}" != lane_key
    lane_run = I.run_id("arditi_allpos", lane_key, "arditi_allpos@ar_c0p3", arm["eval_id"], arm["instrument_id"])
    assert arm["run_id"] != lane_run
    # the two rows agree on the ONE thing they should: the rung's operating-point id
    assert arm["operating_point_id_legacy"] == "ar_c0p3__arditi_allpos"


def test_end_to_end_pass_and_ingest(tmp_path):
    seeds = _seeds()
    obj = LC.ResolvedObject(OBJ, DG.OBJECT_KIND, DG.PAPER_LABEL, DG.OBJECT_ROLE, "recorded", "test", staged_id=f"{RUNG}__arditi_allpos",
                            derivation="none")
    root = F.registry_root(tmp_path, seeds, [obj])
    store = _store(tmp_path)
    ad = AV.ArditiVendorMtAdapter(grades_roots=(os.path.join(store, "mt_grades"),),
                                  transcripts_roots=(os.path.join(store, "transcripts_results"),), seeds=seeds, replicate_roots=())
    # pass 1 over a fresh registry parks every comparison (the own `none` is emitted by this very read: `not_in_frame`); the fold makes it
    # current and the anchor decision input moves, so pass 2 re-reads the group under a new revision id and files the three contrasts
    rep1, ing1, ing1b = F.run_and_ingest(root, ad, now=NOW)
    assert rep1.groups_refused == 0 and ing1.refused + ing1b.refused == 0, ([r.error for r in rep1.results], ing1.refusals, ing1b.refusals)
    assert F.q(root, "SELECT COUNT(*) FROM run_contrast") == [(0,)] and ad.status_extra(TAG)["anchor_states"] == {"no_current_anchor": 3}
    assert sum(1 for it in ad.refusals.items if PARK_PHRASE in it["why"] and ": not_in_frame]" in it["why"]) == 3
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=T[0])
    assert rep.groups_refused == 0 and rep.anchor_decisions_changed == 1, [r.error for r in rep.results]
    assert ing.refused + ing2.refused == 0, ing.refusals + ing2.refusals
    assert ad.status_extra(TAG)["anchor_states"] == {"own_current": 3} and not ad.refusals.items
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        rows = [tuple(r) for r in conn.execute(
            "SELECT producer_object_key_legacy, strength_coefficient, strength_unit, strength_nominal, is_operating_point, "
            "operating_point_id_legacy, json_extract(extra_json, '$.paper_dose') FROM run ORDER BY producer_object_key_legacy")]
        assert rows == [(AV.STEERED_OBJECT_KEY, 0.15, "raw_coefficient", 0.5, 0, "ar_c0p15__arditi_allpos", 0.15),
                        (AV.ANCHOR_OBJECT_KEY, None, "unverified", None, None, None, None)], rows
        assert conn.execute("SELECT COUNT(*) FROM run_contrast WHERE anchor_kind = 'same_standup_none'").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM disposition WHERE state = 'pending'").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0
        assert conn.execute("SELECT DISTINCT lane_source_legacy FROM run").fetchone()[0] == DG.SOURCE
    finally:
        conn.close()


# --------------------------------------------------------------------------------------------
# the anchor currency decides every contrast
# --------------------------------------------------------------------------------------------

def _sibling_frame(steered_run, *, k, n, standup=SIBLING_TAG, source=DG.SOURCE):
    """A frame holding ONE current unsteered run of `source` at the steered run's EXACT key under another standup, complete, off-roster, with
    one `asr` reading per threshold (k of n) under the steered run's judge: tier 1b's one eligible candidate (`sibling_none`)."""
    fr = _frame()
    rid = I.run_id(LC.CAMPAIGN, f"{source}/{standup}", "none", str(steered_run["eval_id"]), str(steered_run["instrument_id"]))
    fr.run(rid, model=str(steered_run["model_id"]), eval_id=str(steered_run["eval_id"]), inst=str(steered_run["instrument_id"]),
           cap=steered_run["token_cap_tokens"], trial=steered_run["trial_index"], pass_label=steered_run.get("pass_label"), decode=steered_run.get("decode_policy_id"),
           source=source, standup=standup, key=f"{source}/{standup}", in_paper=0, roster=False, reading=False)
    jid = fr.judgment(rid, judge=steered_run["judge_id"])
    for t in AV.THRESHOLDS:
        fr.metric(rid, jid, metric="asr", thr=t, k=k, n=n, value=k / n)
    return fr, rid, jid


def test_an_aggregate_without_a_none_entry_parks_by_name_or_anchors_on_a_same_key_unsteered_run_of_the_source(tmp_path):
    """The aggregate carries no same-standup `none` entry: its refusal line stands (no entry is filled from elsewhere) and the contrast is
    still a comparison of the read -- `none_declared`, the exact-key search runs. With nothing at the key the three comparisons park by name
    on the steered `asr` rows (no own pairing to park: the block says so); with a current unsteered run of this source at the exact key under
    another dose step's standup, the registry files its unpaired rate difference from the steered recount and the candidate's reading
    (`sibling_none`, `computed_by registry`, its own computation row), `delta_block: null` on the steered rows."""
    store = _store(tmp_path, with_anchor=False)
    missing_line = "the aggregate carries no 'none' cell"
    # ---- nothing at the key: parked by name, the missing-entry refusal kept
    ad, records, rows = _read(tmp_path, store)
    assert {p["producer_object_key_legacy"] for p in rows["run"]} == {AV.STEERED_OBJECT_KEY} and "run_contrast" not in rows
    assert sum(1 for it in ad.refusals.items if missing_line in it["why"]) == 1
    parks = [it for it in ad.refusals.items if PARK_PHRASE in it["why"]]
    assert len(parks) == 3 and all(": none_declared]" in it["why"] and "no unsteered run of" in it["why"] for it in parks) and len(ad.refusals.items) == 4
    facts = ad.status_extra(TAG)
    assert facts["anchor_states"] == {"no_current_anchor": 3} and facts["contrasts_parked"] == 3 and facts["contrasts"] == 0
    steered = rows["run"][0]
    asr = {m["threshold"]: m for m in rows["run_metric"] if m["metric_id"] == "asr"}
    assert set(asr) == set(AV.THRESHOLDS)
    for m in asr.values():
        blk = m["extra"]["delta_block"]
        assert blk["anchor"]["state"] == "no_current_anchor" and blk["anchor"]["own_reason"] == "none_declared" and blk["anchor"]["own_anchor_run_id"] is None
        assert blk["delta"] is None and blk["n_pairs"] == 0 and "no own pairing" in blk["delta_method"] and blk["rule"] == AC.RULE
        assert blk["anchor_selection"]["state"] == "no_current_anchor" and blk["anchor_selection"]["n_candidates"] == 0
    bad = [str(v) for v in (VAL.validate_record(r, path="<test>") for r in records) if isinstance(v, VAL.Malformed)]
    assert not bad, bad[:5]
    # ---- a current unsteered run of the source at the exact key under another standup: the registry's unpaired rate difference
    k_a, n_a = 2, 10
    fr, sib, sib_jid = _sibling_frame(steered, k=k_a, n=n_a)
    ad, records, rows = _read(tmp_path / "sib", store, frame=fr)
    assert sum(1 for it in ad.refusals.items if missing_line in it["why"]) == 1 and len(ad.refusals.items) == 1, ad.refusals.items
    facts = ad.status_extra(TAG)
    assert facts["anchor_states"] == {"sibling_same_source": 3} and facts["contrasts_reanchored"] == 3 and facts["contrasts"] == 3 and facts["contrasts_parked"] == 0
    assert facts["reanchor_computations"] == 1
    contrasts = {p["threshold"]: p for p in rows["run_contrast"]}
    assert set(contrasts) == set(AV.THRESHOLDS)
    comps = {p["computation_id"]: p for p in rows["computation"]}
    for t, c in contrasts.items():
        assert c["run_id"] == steered["run_id"] and c["anchor_run_id"] == sib and c["anchor_kind"] == "sibling_none" and c["anchor_judgment_id"] == sib_jid
        assert c["computed_by"] == "registry" and c["delta_method"] == DC.UNPAIRED_RATE_METHOD and c["n_pairs"] == 0 and c["computation_id"] == ""
        assert c["delta"] == pytest.approx(100 * (6 / 10 - k_a / n_a)) and c["effect_kind"] == "cohens_h"
        ex = c["extra"]
        assert (ex["k_steered"], ex["n_steered"], ex["k_anchor"], ex["n_anchor"], ex["paired"]) == (6, 10, k_a, n_a, False)
        assert ex["anchor_selection"]["state"] == "sibling_same_source" and ex["anchor_selection"]["own_reason"] == "none_declared"
        assert ex["anchor_selection"]["own_anchor_run_id"] is None and ex["anchor_selection"]["candidates"] == [sib] and ex["anchor_selection"]["rejected"] == []
        assert ex["cross_source"] is False and ex["anchor_standup"] == {"lane_source_legacy": DG.SOURCE, "view_legacy": SIBLING_TAG, "cluster_job_id": None}
        assert ex["contrast_caps"] == {"steered": AV.CAP, "anchor": AV.CAP} and ex["producer_delta_parked_on_metric"] is False
        assert ex["anchor_reading"]["judgment_id"] == sib_jid and ex["statistic_computation_id"] in comps
        assert comps[ex["statistic_computation_id"]]["algorithm"] == "registry.unpaired_rate_contrast" and comps[ex["statistic_computation_id"]]["version"] == AC.RULE
    asr = {m["threshold"]: m for m in rows["run_metric"] if m["metric_id"] == "asr"}
    assert all("delta_block" in m["extra"] and m["extra"]["delta_block"] is None for m in asr.values())
    bad = [str(v) for v in (VAL.validate_record(r, path="<test>") for r in records) if isinstance(v, VAL.Malformed)]
    assert not bad, bad[:5]


def _steered_run(root):
    rows = F.q(root, "SELECT r.run_id, r.campaign, r.model_id, r.eval_id, r.item_set_id, r.instrument_id, r.token_cap_tokens, r.trial_index, r.pass_label, "
                     "r.decode_policy_id, r.view_legacy, r.judge_id FROM run r WHERE r.object_id <> 'none' AND r.eval_id = 'crescendo'")
    assert len(rows) == 1, rows
    return dict(zip(("run_id", "campaign", "model_id", "eval_id", "item_set_id", "instrument_id", "token_cap_tokens", "trial_index", "pass_label",
                     "decode_policy_id", "view_legacy", "judge_id"), rows[0]))


def _sibling_none_records(root, steered, *, i0, k, n, view_legacy=SIBLING_TAG):
    """A complete, judged unsteered run of source `arditi_vendor_mt` at the steered run's EXACT key under another dose step's standup,
    carrying one `asr` reading per threshold (k of n) under the steered run's judge -- filed as records and folded. Returns `(run_id, judgment_id)`."""
    run_key = f"{DG.SOURCE}/{view_legacy}"
    rid = I.run_id(str(steered["campaign"]), run_key, "none", str(steered["eval_id"]), str(steered["instrument_id"]))
    over = {c: steered[c] for c in ("campaign", "model_id", "eval_id", "item_set_id", "instrument_id", "token_cap_tokens", "trial_index", "pass_label", "decode_policy_id")}
    over.update(run_id=rid, lane_source_legacy=DG.SOURCE, view_legacy=view_legacy)
    judge = str(steered["judge_id"])
    _rid, recs = synth.run_records(run_key, i0, "none", "none", n_items=n, judge=judge, prod=F.producer(), **over)
    assert _rid == rid
    jid = I.judgment_id(rid, judge)
    recs = [r for r in recs if r["record_type"] != "run_metric"]
    for j, t in enumerate(AV.THRESHOLDS):
        recs.append(synth.rec("run_metric", {"run_id": rid, "judgment_id": jid, "metric_id": "asr", "threshold": t, "value": k / n, "k": k, "n": n, "computed_by": "producer"},
                              synth.t(i0 + 9 + j), F.producer()))
    R.write_records(os.path.join(root, "records", "sibling", f"{i0:06d}-sibling-none.jsonl"), recs)
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and ing.blocked == 0, (ing.refused, ing.blocked, ing.refusals)
    assert F.q(root, "SELECT is_current, why FROM v_run_current WHERE run_id = ?", (rid,)) == [(1, None)]
    return rid, jid


def _exclude(root, ruling_id, run_id, i):
    payload = {"ruling_id": ruling_id, "round_id": LC.ROUND_ID, "action": "exclude", "scope_kind": "run", "scope_json": {"run_id": run_id},
               "target_json": {}, "reason": f"synthetic: exclude {run_id}", "ruled_utc": synth.t(i), "ruled_utc_precision": "datetime",
               "ruled_by": "test", "source_ref": "synthetic"}
    R.write_records(os.path.join(root, "records", "rulings", f"{i:06d}-{ruling_id}.jsonl"), [synth.rec("ruling", payload, synth.t(i))])
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and ing.blocked == 0, (ing.refused, ing.blocked, ing.refusals)


def _current_contrast_keys(root, source_id):
    return {r[0] for r in F.q(root, "SELECT observation_key FROM observation WHERE source_id = ? AND table_name = 'run_contrast' AND current = 1", (source_id,))}


def test_a_retired_own_unsteered_run_re_picks_the_same_sources_sibling_end_to_end(tmp_path):
    """Through the runner and the harvester on a temp registry: the two-pass steady state files the three own contrasts; an `exclude` ruling
    retires the standup's own `none` while a current unsteered run of this source stands at the exact key under another dose step's standup
    -- the next pass re-reads the group (a new revision id, no file change, no bump: the §4 decision input moved), files the registry's
    unpaired rate difference against the sibling at each threshold (`sibling_none`, keyed on the reading's computation so the paper frame
    shows it as the row's one contrast), parks the own pairing's McNemar numbers on the steered `asr` rows, withdraws the old keys; then quiet."""
    seeds = _seeds()
    obj = LC.ResolvedObject(OBJ, DG.OBJECT_KIND, DG.PAPER_LABEL, DG.OBJECT_ROLE, "recorded", "test", staged_id=f"{RUNG}__arditi_allpos", derivation="none")
    root = F.registry_root(tmp_path, seeds, [obj])
    store = _store(tmp_path)
    ad = AV.ArditiVendorMtAdapter(grades_roots=(os.path.join(store, "mt_grades"),), transcripts_roots=(os.path.join(store, "transcripts_results"),),
                                  seeds=seeds, replicate_roots=())
    source_id = RUN.source_id_for(AV.NAME, TAG)
    for now in (NOW, T[0]):
        rep, ing, ing2 = F.run_and_ingest(root, ad, now=now)
        assert rep.groups_refused == 0 and ing.refused + ing2.refused == 0 and ing.blocked + ing2.blocked == 0, ([r.error for r in rep.results], ing.refusals, ing2.refusals)
    steered = _steered_run(root)
    own = F.q(root, "SELECT anchor_run_id, delta, computed_by, delta_method FROM run_contrast WHERE run_id = ? AND threshold = 0.5", (steered["run_id"],))
    assert len(own) == 1 and own[0][1] == pytest.approx(-40.0) and own[0][2] == "producer" and own[0][3] == "exact_mcnemar"
    own_none = own[0][0]
    keys2 = _current_contrast_keys(root, source_id)
    assert len(keys2) == 3
    k_a, n_a = 2, 10
    sib, sib_jid = _sibling_none_records(root, steered, i0=300, k=k_a, n=n_a)
    _exclude(root, "r_synth_exclude_vendor_none", own_none, 10)
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=T[1])
    # the moved anchor-decision input IS a changed input of the group (inputs_changed counts it); no FILE moved and nothing was bumped
    assert rep.groups_refreshed == 1 and rep.anchor_decisions_changed == 1 and rep.inputs_changed == 1, [r.error for r in rep.results]
    assert ing.refused + ing2.refused == 0 and ing.blocked + ing2.blocked == 0, (ing.refusals, ing2.refusals)
    facts = ad.status_extra(TAG)
    assert facts["anchor_states"] == {"sibling_same_source": 3} and facts["contrasts_reanchored"] == 3 and facts["contrasts_parked"] == 0 and not ad.refusals.items
    for t in AV.THRESHOLDS:
        rows = F.q(root, "SELECT anchor_run_id, anchor_kind, delta, delta_method, computed_by, computation_id, anchor_judgment_id, n_pairs, extra_json "
                         "FROM run_contrast WHERE run_id = ? AND metric_id = 'asr' AND threshold = ? ORDER BY anchor_run_id", (steered["run_id"], t))
        assert sorted(r[0] for r in rows) == sorted([own_none, sib])          # no run-scale row is deleted: the old row stays ...
        new = next(r for r in rows if r[0] == sib)
        assert new[1] == "sibling_none" and new[3] == DC.UNPAIRED_RATE_METHOD and new[4] == "registry" and new[5] == "" and new[6] == sib_jid and new[7] == 0
        assert new[2] == pytest.approx(100 * (6 / 10 - k_a / n_a))
        ex = json.loads(new[8])
        assert ex["anchor_selection"]["state"] == "sibling_same_source" and ex["anchor_selection"]["own_reason"] == "excluded" and ex["anchor_selection"]["own_anchor_run_id"] == own_none
        assert ex["anchor_selection"]["candidates"] == [sib] and [r[0] for r in ex["anchor_selection"]["rejected"]] == [own_none] and "exclude" in ex["anchor_selection"]["rejected"][0][1]
        assert ex["anchor_standup"]["view_legacy"] == SIBLING_TAG and ex["anchor_standup"]["lane_source_legacy"] == DG.SOURCE and ex["cross_source"] is False
        assert ex["contrast_caps"] == {"steered": AV.CAP, "anchor": AV.CAP} and ex["producer_delta_parked_on_metric"] is True
        assert F.q(root, "SELECT algorithm, version FROM computation WHERE computation_id = ?", (ex["statistic_computation_id"],)) == [("registry.unpaired_rate_contrast", AC.RULE)]
        # the own pairing's numbers park on the steered reading beside the filed contrast
        blk = json.loads(F.q(root, "SELECT extra_json FROM run_metric WHERE run_id = ? AND metric_id = 'asr' AND threshold = ?", (steered["run_id"], t))[0][0])["delta_block"]
        assert blk["delta"] == pytest.approx(-40.0) and blk["delta_method"] == "exact_mcnemar" and blk["computed_by"] == "producer" and blk["n_pairs"] == 10
        assert blk["anchor"]["own_anchor_run_id"] == own_none and blk["anchor"]["state"] == "sibling_same_source" and blk["anchor"]["registry_contrast_filed_against"] == sib
        assert "stands" in blk["why_not_a_contrast_row"] and blk["producer_contrast_extra"]["discordant_anchor_only"] == 4
        # the paper frame: ONE current contrast on the row, the sibling, the registry's delta, the cap matched
        frame = F.q(root, "SELECT anchor_run_id, anchor_kind, delta, delta_method, n_current_contrasts, cap_matched, metric_extra_json FROM v_paper_long "
                          "WHERE run_id = ? AND metric_id = 'asr' AND threshold = ?", (steered["run_id"], t))
        assert len(frame) == 1 and frame[0][:2] == (sib, "sibling_none") and frame[0][2] == pytest.approx(100 * (6 / 10 - k_a / n_a))
        assert frame[0][3] == DC.UNPAIRED_RATE_METHOD and frame[0][4] == 1 and frame[0][5] == 1 and json.loads(frame[0][6])["delta_block"]["delta"] == pytest.approx(-40.0)
    # ... but the old keys are withdrawn: the source's current contrast keys are the sibling's, one per threshold
    keys3 = _current_contrast_keys(root, source_id)
    assert len(keys3) == 3 and {(json.loads(k)["anchor_run_id"], json.loads(k)["threshold"]) for k in keys3} == {(sib, t) for t in AV.THRESHOLDS}
    assert not (keys2 & keys3)
    # ---- quiet afterwards
    rep5, _, _ = F.run_and_ingest(root, ad, now=T[2])
    assert rep5.inputs_changed == 0 and rep5.groups_refreshed == 0 and rep5.anchor_decisions_changed == 0


def _read_logged(tmp_path, ad, fr, inputs, tag=TAG):
    """One group's read on `fr` with the group's decision log bound (what the runner does per group), returning the records by type and the log."""
    ctx = _ctx(tmp_path, fr)
    ctx.begin_group(tag)
    records = [r for b in ad.read(ctx, tag, inputs) for r in b.records]
    by_type = {}
    for r in records:
        by_type.setdefault(r["record_type"], []).append(r["payload"])
    return records, by_type, list(ctx.decision_log)


def test_an_own_anchor_sharing_no_goal_id_is_still_decided_at_every_threshold(tmp_path):
    """The aggregate's own `none` entry was graded on a goal set DISJOINT from the steered entry's: no paired per-goal statistic can be formed.
    Until commit 5 the reader returned at that point, before consulting the currency -- no contrast, no park on the steered
    `asr` rows, and the comparison ABSENT from the decision set, so a later currency change alone could never re-read the group. The
    comparison is the read's whatever the goal overlap: with the own anchor current it parks by name at every threshold (the own anchor stands;
    no other anchor is picked over a current own one) with the reason on the steered rows and three entries in the decision log; with the own
    anchor retired and a current unsteered run of the source at the exact key under another dose step's standup, the registry's unpaired rate
    difference files against the sibling and the uncomputable own pairing parks beside it by reason, no numbers."""
    store = _store(tmp_path, anchor_goal_prefix="other")
    disjoint = "share no goal id"
    # ---- the own anchor current: parked by name, recorded
    ad = _adapter(store)
    inputs = [i for i in ad.discover() if ad.comparison_group(i) == TAG]
    fr = _frame()
    _prime(tmp_path, ad, fr, {TAG: inputs})
    records, rows, log = _read_logged(tmp_path, ad, fr, inputs)
    runs = {p["producer_object_key_legacy"]: p for p in rows["run"]}
    assert set(runs) == {AV.STEERED_OBJECT_KEY, AV.ANCHOR_OBJECT_KEY} and "run_contrast" not in rows
    steered, own = runs[AV.STEERED_OBJECT_KEY], runs[AV.ANCHOR_OBJECT_KEY]
    assert sum(1 for it in ad.refusals.items if disjoint in it["why"] and it["key"].endswith("|contrast")) == 1
    parks = [it for it in ad.refusals.items if it["why"].endswith("filed without a contrast")]
    assert len(parks) == 3 and all("is current" in it["why"] and "no shared goal id" in it["why"] for it in parks) and len(ad.refusals.items) == 4, ad.refusals.items
    facts = ad.status_extra(TAG)
    assert facts["anchor_states"] == {"own_current": 3} and facts["contrasts_parked"] == 3 and facts["contrasts"] == 0 and facts["contrasts_reanchored"] == 0
    asr = {m["threshold"]: m for m in rows["run_metric"] if m["metric_id"] == "asr" and m["run_id"] == steered["run_id"]}
    assert set(asr) == set(AV.THRESHOLDS)
    for m in asr.values():
        blk = m["extra"]["delta_block"]
        assert blk["anchor"]["state"] == "own_current" and blk["anchor"]["own_reason"] == "current" and blk["anchor"]["own_anchor_run_id"] == own["run_id"]
        assert blk["anchor"]["anchor_run_id"] == own["run_id"] and blk["anchor"]["registry_contrast_filed_against"] is None
        assert blk["delta"] is None and blk["n_pairs"] == 0 and disjoint in blk["delta_method"] and "no shared goal id" in blk["why_not_a_contrast_row"]
        assert blk["anchor_selection"]["state"] == "own_current" and blk["anchor_selection"]["own_anchor_run_id"] == own["run_id"] and blk["rule"] == AC.RULE
    assert all("delta_block" not in m["extra"] for m in rows["run_metric"] if m["metric_id"] == "asr" and m["run_id"] == own["run_id"])
    # the three comparisons ARE in the group's decision set, decided on the own anchor (the park is the reader's: the statistic, not the currency)
    assert len(log) == 3 and {e["comparison"]["threshold"] for e in log} == set(AV.THRESHOLDS)
    assert all(e["comparison"]["run_id"] == steered["run_id"] and e["comparison"]["key"]["own_anchor_run_id"] == own["run_id"] for e in log)
    assert all(e["decision"]["anchor_run_id"] == own["run_id"] and e["decision"]["state"] == "own_current" for e in log)
    bad = [str(v) for v in (VAL.validate_record(r, path="<test>") for r in records) if isinstance(v, VAL.Malformed)]
    assert not bad, bad[:5]
    # ---- the own anchor retired, a current sibling at the exact key: the registry's statistic against the sibling, the own pairing parked by reason
    k_a, n_a = 2, 10
    fr2, sib, sib_jid = _sibling_frame(steered, k=k_a, n=n_a)
    ad2 = _adapter(store)
    inputs2 = [i for i in ad2.discover() if ad2.comparison_group(i) == TAG]
    _prime(tmp_path / "sib", ad2, fr2, {TAG: inputs2})
    AF.retire(fr2, own["run_id"], "selection_state excluded (decisive ruling exclude r_synth_exclude_vendor_none)")
    records, rows, log = _read_logged(tmp_path / "sib", ad2, fr2, inputs2)
    assert len(ad2.refusals.items) == 1 and disjoint in ad2.refusals.items[0]["why"], ad2.refusals.items
    facts = ad2.status_extra(TAG)
    assert facts["anchor_states"] == {"sibling_same_source": 3} and facts["contrasts_reanchored"] == 3 and facts["contrasts"] == 3 and facts["contrasts_parked"] == 0
    assert facts["reanchor_computations"] == 1
    contrasts = {p["threshold"]: p for p in rows["run_contrast"]}
    assert set(contrasts) == set(AV.THRESHOLDS)
    comps = {p["computation_id"]: p for p in rows["computation"]}
    for t, c in contrasts.items():
        assert c["run_id"] == steered["run_id"] and c["anchor_run_id"] == sib and c["anchor_kind"] == "sibling_none" and c["anchor_judgment_id"] == sib_jid
        assert c["computed_by"] == "registry" and c["delta_method"] == DC.UNPAIRED_RATE_METHOD and c["n_pairs"] == 0 and c["computation_id"] == ""
        assert c["delta"] == pytest.approx(100 * (6 / 10 - k_a / n_a))
        ex = c["extra"]
        assert (ex["k_steered"], ex["n_steered"], ex["k_anchor"], ex["n_anchor"], ex["paired"]) == (6, 10, k_a, n_a, False)
        assert ex["anchor_selection"]["state"] == "sibling_same_source" and ex["anchor_selection"]["own_reason"] == "excluded"
        assert ex["anchor_selection"]["own_anchor_run_id"] == own["run_id"] and ex["anchor_selection"]["candidates"] == [sib]
        assert ex["producer_delta_parked_on_metric"] is True and ex["statistic_computation_id"] in comps
    asr = {m["threshold"]: m for m in rows["run_metric"] if m["metric_id"] == "asr" and m["run_id"] == steered["run_id"]}
    for m in asr.values():
        blk = m["extra"]["delta_block"]
        assert blk["delta"] is None and blk["n_pairs"] == 0 and disjoint in blk["delta_method"] and blk["computed_by"] == "producer"
        assert blk["anchor"]["state"] == "sibling_same_source" and blk["anchor"]["own_reason"] == "excluded" and blk["anchor"]["own_anchor_run_id"] == own["run_id"]
        assert blk["anchor"]["anchor_run_id"] == sib and blk["anchor"]["registry_contrast_filed_against"] == sib
        assert "no own pairing was formed" in blk["why_not_a_contrast_row"] and "stands" in blk["why_not_a_contrast_row"]
    assert len(log) == 3 and all(e["decision"]["anchor_run_id"] == sib and e["decision"]["own_reason"] == "excluded" for e in log)
    bad = [str(v) for v in (VAL.validate_record(r, path="<test>") for r in records) if isinstance(v, VAL.Malformed)]
    assert not bad, bad[:5]


def test_an_own_anchor_sharing_no_goal_id_re_anchors_when_its_currency_moves_end_to_end(tmp_path):
    """Through the runner and the harvester on a temp registry: the two-pass steady state of a standup whose own `none` entry shares no goal id
    with the steered entry files NO contrast -- three parks on the steered `asr` rows and three recorded comparisons on the head's marker, all
    decided on the own anchor; an `exclude` ruling retires that own `none` while a current unsteered run of this source stands at the exact key
    under another dose step's standup, and the next pass re-reads the group on the moved decision input ALONE (no file change, no bump), files the
    registry's unpaired rate difference against the sibling at each threshold with the uncomputable own pairing parked by reason, the paper
    frame showing the sibling as the row's one contrast; then quiet."""
    seeds = _seeds()
    obj = LC.ResolvedObject(OBJ, DG.OBJECT_KIND, DG.PAPER_LABEL, DG.OBJECT_ROLE, "recorded", "test", staged_id=f"{RUNG}__arditi_allpos", derivation="none")
    root = F.registry_root(tmp_path, seeds, [obj])
    store = _store(tmp_path, anchor_goal_prefix="other")
    ad = AV.ArditiVendorMtAdapter(grades_roots=(os.path.join(store, "mt_grades"),), transcripts_roots=(os.path.join(store, "transcripts_results"),),
                                  seeds=seeds, replicate_roots=())
    source_id = RUN.source_id_for(AV.NAME, TAG)
    head_sql = "SELECT extra_json FROM source_revision WHERE source_id = ? AND accepted = 1 ORDER BY observed_utc DESC, revision_id DESC LIMIT 1"
    for now in (NOW, T[0]):
        rep, ing, ing2 = F.run_and_ingest(root, ad, now=now)
        assert rep.groups_refused == 0 and ing.refused + ing2.refused == 0 and ing.blocked + ing2.blocked == 0, ([r.error for r in rep.results], ing.refusals, ing2.refusals)
    steered = _steered_run(root)
    own_none = F.q(root, "SELECT run_id FROM run WHERE object_id = 'none' AND eval_id = 'crescendo'")
    assert len(own_none) == 1
    own_none = own_none[0][0]
    # the steady state: no contrast, the park on every steered reading, the three comparisons recorded on the head
    assert F.q(root, "SELECT COUNT(*) FROM run_contrast WHERE run_id = ?", (steered["run_id"],)) == [(0,)] and _current_contrast_keys(root, source_id) == set()
    facts = ad.status_extra(TAG)
    assert facts["anchor_states"] == {"own_current": 3} and facts["contrasts_parked"] == 3 and facts["contrasts"] == 0
    assert len(ad.refusals.items) == 4 and sum(1 for it in ad.refusals.items if "share no goal id" in it["why"]) == 1
    for t in AV.THRESHOLDS:
        blk = json.loads(F.q(root, "SELECT extra_json FROM run_metric WHERE run_id = ? AND metric_id = 'asr' AND threshold = ?", (steered["run_id"], t))[0][0])["delta_block"]
        assert blk["anchor"]["state"] == "own_current" and blk["anchor"]["own_anchor_run_id"] == own_none and blk["anchor"]["registry_contrast_filed_against"] is None
        assert blk["delta"] is None and blk["n_pairs"] == 0 and "share no goal id" in blk["delta_method"]
    decisions = json.loads(F.q(root, head_sql, (source_id,))[0][0] or "{}")["anchor_decisions"]
    assert decisions["n_comparisons"] == 3 and decisions["n_parked"] == 0 and decisions["rule"] == AC.DECISION_RULE
    assert all(e["comparison"]["run_id"] == steered["run_id"] and e["decision"]["anchor_run_id"] == own_none and e["decision"]["state"] == "own_current"
               for e in decisions["comparisons"])
    # ---- the own `none` excluded, a current sibling at the exact key: the decision input moves, the group re-reads, the sibling contrast files
    k_a, n_a = 2, 10
    sib, sib_jid = _sibling_none_records(root, steered, i0=300, k=k_a, n=n_a)
    _exclude(root, "r_synth_exclude_vendor_none_disjoint", own_none, 10)
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=T[1])
    assert rep.groups_refreshed == 1 and rep.anchor_decisions_changed == 1 and rep.inputs_changed == 1, [r.error for r in rep.results]
    assert ing.refused + ing2.refused == 0 and ing.blocked + ing2.blocked == 0, (ing.refusals, ing2.refusals)
    facts = ad.status_extra(TAG)
    assert facts["anchor_states"] == {"sibling_same_source": 3} and facts["contrasts_reanchored"] == 3 and facts["contrasts_parked"] == 0
    assert len(ad.refusals.items) == 1 and "share no goal id" in ad.refusals.items[0]["why"], ad.refusals.items
    for t in AV.THRESHOLDS:
        rows = F.q(root, "SELECT anchor_run_id, anchor_kind, delta, delta_method, computed_by, computation_id, anchor_judgment_id, n_pairs, extra_json "
                         "FROM run_contrast WHERE run_id = ? AND metric_id = 'asr' AND threshold = ?", (steered["run_id"], t))
        assert len(rows) == 1 and rows[0][:2] == (sib, "sibling_none") and rows[0][3] == DC.UNPAIRED_RATE_METHOD and rows[0][4] == "registry"
        assert rows[0][5] == "" and rows[0][6] == sib_jid and rows[0][7] == 0 and rows[0][2] == pytest.approx(100 * (6 / 10 - k_a / n_a))
        ex = json.loads(rows[0][8])
        assert ex["anchor_selection"]["state"] == "sibling_same_source" and ex["anchor_selection"]["own_reason"] == "excluded" and ex["anchor_selection"]["own_anchor_run_id"] == own_none
        assert ex["anchor_selection"]["candidates"] == [sib] and [r[0] for r in ex["anchor_selection"]["rejected"]] == [own_none]
        assert ex["producer_delta_parked_on_metric"] is True and ex["cross_source"] is False
        assert F.q(root, "SELECT algorithm, version FROM computation WHERE computation_id = ?", (ex["statistic_computation_id"],)) == [("registry.unpaired_rate_contrast", AC.RULE)]
        blk = json.loads(F.q(root, "SELECT extra_json FROM run_metric WHERE run_id = ? AND metric_id = 'asr' AND threshold = ?", (steered["run_id"], t))[0][0])["delta_block"]
        assert blk["delta"] is None and blk["n_pairs"] == 0 and "share no goal id" in blk["delta_method"] and blk["computed_by"] == "producer"
        assert blk["anchor"]["state"] == "sibling_same_source" and blk["anchor"]["own_anchor_run_id"] == own_none and blk["anchor"]["registry_contrast_filed_against"] == sib
        assert "no own pairing was formed" in blk["why_not_a_contrast_row"] and "stands" in blk["why_not_a_contrast_row"]
        frame = F.q(root, "SELECT anchor_run_id, anchor_kind, delta, delta_method, n_current_contrasts, cap_matched FROM v_paper_long "
                          "WHERE run_id = ? AND metric_id = 'asr' AND threshold = ?", (steered["run_id"], t))
        assert len(frame) == 1 and frame[0][:2] == (sib, "sibling_none") and frame[0][2] == pytest.approx(100 * (6 / 10 - k_a / n_a))
        assert frame[0][3] == DC.UNPAIRED_RATE_METHOD and frame[0][4] == 1 and frame[0][5] == 1
    keys = _current_contrast_keys(root, source_id)
    assert len(keys) == 3 and {(json.loads(k)["anchor_run_id"], json.loads(k)["threshold"]) for k in keys} == {(sib, t) for t in AV.THRESHOLDS}
    decisions = json.loads(F.q(root, head_sql, (source_id,))[0][0] or "{}")["anchor_decisions"]
    assert decisions["n_comparisons"] == 3 and all(e["decision"]["anchor_run_id"] == sib and e["decision"]["own_reason"] == "excluded" for e in decisions["comparisons"])
    # ---- quiet afterwards
    rep5, _, _ = F.run_and_ingest(root, ad, now=T[2])
    assert rep5.inputs_changed == 0 and rep5.groups_refreshed == 0 and rep5.anchor_decisions_changed == 0


def test_the_smoke_names_the_unavailable_anchor_currency_for_the_standup(tmp_path):
    out = DC.smoke(_adapter(_store(tmp_path)))
    assert out["groups"] == 1 and out["anchor_unavailable_groups"] == 1
    assert len(out["refused_groups"]) == 1 and out["refused_groups"][0].startswith(f"{TAG}: anchor currency unavailable") and "no registry connection" in out["refused_groups"][0]


# --------------------------------------------------------------------------------------------
# the decode replicates: trials of their own
# --------------------------------------------------------------------------------------------

def _replicate_store(tmp_path, tag=SERVED_TAG, trials=(1, 2), attacks=("crescendo",), t0_twin=True, t0_divergent=False, t0_extra=("pair",),
                     shard_trial=None):
    """The ladder's standup at trial 0 plus the replicate trees: `<root>/replicate_grades/t<T>/<tag>/` (grades) and
    `<root>/rep_t<T>/transcripts_results/` (the trial's shards) — the two live shapes. A `t0` directory repeats the ladder's crescendo
    aggregate byte for byte (`t0_twin`) or as a divergent regrade (`t0_divergent`), and carries PAIR, which the ladder never graded."""
    root = _store(tmp_path, tag=tag, attacks=attacks)
    def write_trial(trial, directory, tx_root, attacks_here, divergent=False):
        os.makedirs(directory, exist_ok=True)
        for attack in attacks_here:
            agg = _aggregate(attack, tag=tag, arm_jailbroken=(3 if divergent else 6 - trial))
            with open(os.path.join(directory, f"two_si_{attack}_aggregate.json"), "w") as fh:
                json.dump(agg, fh)
            for cell in (AV.STEERED_OBJECT_KEY, AV.ANCHOR_OBJECT_KEY):
                for s in range(2):
                    d = os.path.join(tx_root, AV.private_run_of(tag), attack, cell, f"shard{s:02d}")
                    os.makedirs(d, exist_ok=True)
                    st = _served_target(cell)
                    st["decode"]["trial"] = trial if shard_trial is None else shard_trial
                    st["routing_table"] = f"/cluster/exp/sdfbase_full22_native_ar_c0p3_t{trial}/serving/attacks_{tag}_{attack}_58{trial}/routing_table.json"
                    with open(os.path.join(d, "served_target.json"), "w") as fh:
                        json.dump(st, fh)
        with open(os.path.join(directory, "grading_provenance.json"), "w") as fh:
            json.dump({"attacks": list(attacks_here), "target": tag, "judge": {"judge_model": "judge-model-x", "rubric": "strongreject_v1",
                       "cache_kind": "sr", "pass_id": "p" * 64, "org_initial": "lowprio"}, "org_final": "lowprio",
                       "finished_at": "2026-01-05T06:00:00+00:00", "started_at": "2026-01-05T05:30:00+00:00", "worktree": {"head": "b" * 40},
                       "config": {"path": "/x/ar.yaml", "sha256": "c" * 64}, "n_429": 0}, fh)
    for trial in trials:
        write_trial(trial, os.path.join(root, "replicate_grades", f"t{trial}", tag), os.path.join(root, f"rep_t{trial}", "transcripts_results"), attacks)
    t0_dir = os.path.join(root, "replicate_grades", "t0", tag)
    if t0_twin or t0_divergent or t0_extra:
        write_trial(0, t0_dir, os.path.join(root, "rep_t0", "transcripts_results"), t0_extra, divergent=False)
        for attack in attacks:
            src = os.path.join(root, "mt_grades", tag, f"two_si_{attack}_aggregate.json")
            if t0_divergent:
                with open(os.path.join(t0_dir, f"two_si_{attack}_aggregate.json"), "w") as fh:
                    json.dump(_aggregate(attack, tag=tag, arm_jailbroken=2), fh)
            elif t0_twin:
                with open(src, "rb") as a, open(os.path.join(t0_dir, f"two_si_{attack}_aggregate.json"), "wb") as b:
                    b.write(a.read())
    patterns = ((os.path.join(root, "replicate_grades", "t<T>"), os.path.join(root, "rep_t<T>", "transcripts_results")),)
    return root, patterns


def _read_all(tmp_path, root, patterns):
    """Every group's read on one frame primed from every group's emissions; `(adapter, {group: (rows by type, refusals, facts)}, {group: records})`."""
    ad = _adapter(root, patterns)
    inputs = ad.discover()
    groups = {}
    for i in inputs:
        groups.setdefault(ad.comparison_group(i), []).append(i)
    fr = _frame()
    _prime(tmp_path, ad, fr, groups)
    ctx = _ctx(tmp_path, fr)
    out = {}
    records = {}
    for g in sorted(groups):
        recs = [r for b in ad.read(ctx, g, groups[g]) for r in b.records]
        by = {}
        for r in recs:
            by.setdefault(r["record_type"], []).append(r["payload"])
        out[g] = (by, list(ad.refusals.items), ad.status_extra(g))
        records[g] = recs
    return ad, out, records


def test_decode_replicates_file_as_trials_of_their_own_groups_never_averaged(tmp_path):
    """Trials 1 and 2 of the middle rung are groups `<standup>_t1` / `<standup>_t2`: own producer key, own run ids, `trial_index` T, the
    trial's own shards (decode.trial attested) and the trial's OWN anchor; the ladder's trial-0 rows keep their ids. Nothing pools them."""
    root, patterns = _replicate_store(tmp_path, t0_twin=False, t0_extra=())
    ad, out, records = _read_all(tmp_path, root, patterns)
    assert set(out) == {SERVED_TAG, f"{SERVED_TAG}_t1", f"{SERVED_TAG}_t2"}
    assert ad.replicate_grades_root == {1: os.path.join(root, "replicate_grades", "t1"), 2: os.path.join(root, "replicate_grades", "t2")}
    assert ad.transcripts_roots_for(1) == (os.path.join(root, "rep_t1", "transcripts_results"),) and ad.transcripts_roots_for(0) == ad.transcripts_roots
    ids = set()
    for trial in (0, 1, 2):
        g = AV.group_for(SERVED_TAG, trial)
        rows, refusals, facts = out[g]
        assert not refusals, (g, refusals)
        runs = {p["producer_object_key_legacy"]: p for p in rows["run"]}
        assert set(runs) == {AV.STEERED_OBJECT_KEY, AV.ANCHOR_OBJECT_KEY}
        for p in runs.values():
            assert p["trial_index"] == trial and p["producer_run_key_legacy"] == f"{DG.SOURCE}/{g}" and p["view_legacy"] == g
            assert p["extra"]["standup_legacy"] == g and p["extra"]["replicate_of_standup_legacy"] == SERVED_TAG
            assert p["extra"]["decode_trial_attested"] == trial and p["seed_base"] == 20260909 and p["decode_seed_rule"] == "per_goal_turn"
            assert p["field_confidence_json"]["trial_index"]["confidence"] == "recorded"
            ids.add(p["run_id"])
        steered = runs[AV.STEERED_OBJECT_KEY]
        assert steered["strength_coefficient"] == pytest.approx(0.3) and steered["is_operating_point"] == 1 and steered["object_id"] == OBJ
        if trial:
            assert steered["extra"]["cluster_stores_legacy"] == [f"sdfbase_full22_native_ar_c0p3_t{trial}"]
        c = next(p for p in rows["run_contrast"] if p["threshold"] == 0.5)
        assert c["anchor_run_id"] == runs[AV.ANCHOR_OBJECT_KEY]["run_id"] and c["anchor_kind"] == "same_standup_none"
        assert c["delta"] == pytest.approx(100 * ((6 - trial) / 10 - 1.0))
        assert facts["trial_index"] == trial and facts["group"] == g and facts["standup"] == SERVED_TAG
        assert all(p["trial_index"] == trial for p in rows["item_result"])
    assert len(ids) == 6, "three trials x two cells: six distinct runs"
    # the trial-0 ids are the ladder's own (unchanged by the replicates' arrival)
    t0 = next(p for p in out[SERVED_TAG][0]["run"] if p["producer_object_key_legacy"] == AV.STEERED_OBJECT_KEY)
    assert t0["producer_run_key_legacy"] == f"{DG.SOURCE}/{SERVED_TAG}"
    assert t0["run_id"] == I.run_id(LC.CAMPAIGN, f"{DG.SOURCE}/{SERVED_TAG}", AV.STEERED_OBJECT_KEY, "crescendo", t0["instrument_id"])
    # every record validates (the records of the primed reads above)
    for g, recs in records.items():
        bad = [str(v) for v in (VAL.validate_record(r, path="<test>") for r in recs) if isinstance(v, VAL.Malformed)]
        assert not bad, (g, bad[:3])


def test_a_t0_replicate_directory_joins_the_ladders_group_a_twin_once_a_divergent_copy_refused(tmp_path):
    """A `t0` replicate directory IS the ladder's standup: it joins the bare group under the ladder's key. An aggregate it repeats byte for
    byte is the same run read ONCE (named as a twin); PAIR, which it alone carries, is a new run of that standup with the pair shards;
    an aggregate that DIFFERS from the ladder's is refused by name and the ladder's copy stands."""
    root, patterns = _replicate_store(tmp_path, trials=(), t0_twin=True, t0_extra=("pair",))
    ad, out, _records = _read_all(tmp_path, root, patterns)
    assert set(out) == {SERVED_TAG}
    rows, refusals, facts = out[SERVED_TAG]
    assert not refusals, refusals
    runs = {(p["eval_id"], p["producer_object_key_legacy"]): p for p in rows["run"]}
    assert set(runs) == {("crescendo", AV.STEERED_OBJECT_KEY), ("crescendo", AV.ANCHOR_OBJECT_KEY), ("pair", AV.STEERED_OBJECT_KEY), ("pair", AV.ANCHOR_OBJECT_KEY)}
    assert all(p["trial_index"] == 0 and p["producer_run_key_legacy"] == f"{DG.SOURCE}/{SERVED_TAG}" for p in runs.values())
    assert len(rows["run"]) == 4, "the twin crescendo copy minted no second run"
    assert facts["twins_skipped"] == [LC.rel_home(os.path.join(root, "replicate_grades", "t0", SERVED_TAG, "two_si_crescendo_aggregate.json"))]
    assert [os.path.basename(d) for d in facts["grades_dirs"]] == [SERVED_TAG, SERVED_TAG] and facts["grades_dirs"][0].endswith(f"mt_grades/{SERVED_TAG}")
    assert runs[("pair", AV.STEERED_OBJECT_KEY)]["extra"]["grades_dir"].endswith(f"replicate_grades/t0/{SERVED_TAG}")
    assert runs[("crescendo", AV.STEERED_OBJECT_KEY)]["extra"]["grades_dir"].endswith(f"mt_grades/{SERVED_TAG}")
    assert "disposition" not in rows or {p["eval_id"] for p in rows["disposition"]} == {"actorattack"}     # PAIR is declared by nobody
    # a divergent regrade in the t0 directory: refused by name, the ladder's crescendo stands
    root2, patterns2 = _replicate_store(tmp_path / "div", trials=(), t0_twin=False, t0_divergent=True, t0_extra=())
    ad2, out2, _records2 = _read_all(tmp_path / "div", root2, patterns2)
    rows2, refusals2, facts2 = out2[SERVED_TAG]
    cres = [p for p in rows2["run"] if p["eval_id"] == "crescendo" and p["producer_object_key_legacy"] == AV.STEERED_OBJECT_KEY]
    assert len(cres) == 1 and cres[0]["extra"]["grades_dir"].endswith(f"mt_grades/{SERVED_TAG}")
    assert next(p for p in rows2["run_metric"] if p["run_id"] == cres[0]["run_id"] and p["threshold"] == 0.5)["k"] == 6, "the ladder's grade, not the regrade"
    assert any("two grades of one (standup, trial, attack)" in it["why"] and "replicate_grades/t0" in it["why"] for it in refusals2), refusals2
    assert facts2["twins_skipped"] == []


def test_shards_recording_another_trial_refuse_the_cell(tmp_path):
    """A mirror filed under the wrong trial is not this standup's replicate: `decode.trial` is part of the attestation."""
    root, patterns = _replicate_store(tmp_path, trials=(1,), t0_twin=False, t0_extra=(), shard_trial=2)
    _ad, out, _records = _read_all(tmp_path, root, patterns)
    rows, refusals, _facts = out[f"{SERVED_TAG}_t1"]
    assert "run" not in rows and any("decode trial 2, not this group's trial 1" in it["why"] for it in refusals), refusals


def test_a_shard_under_two_mirrors_of_one_trial_is_read_once():
    """The same shard mirrored twice is ONE shard (`n_shards` is a fact about the store, not about how many copies this machine holds)."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        for root in ("a", "b"):
            for s in range(2):
                p = os.path.join(d, root, AV.private_run_of(SERVED_TAG), "crescendo", "none", f"shard{s:02d}")
                os.makedirs(p)
                with open(os.path.join(p, "served_target.json"), "w") as fh:
                    json.dump(_served_target("none"), fh)
        files = AV._shard_files((os.path.join(d, "a"), os.path.join(d, "b")), AV.private_run_of(SERVED_TAG), "crescendo", "none")
        assert len(files) == 2 and all(f.startswith(os.path.join(d, "a")) for f in files)
        assert AV.served_attestation(files)[0]["n_shards"] == 2


# --------------------------------------------------------------------------------------------
# the other two readers of the same layout
# --------------------------------------------------------------------------------------------

def test_the_sampled_statics_reader_admits_the_tag_and_files_the_rungs_dose(tmp_path):
    """The statics leg is out of scope for this re-run, but the tag is admitted so the rows land the moment a static grade exists."""
    root = str(tmp_path / "sampled")
    RF.write_sampled_root(root, tag=SERVED_TAG, cells=("none", AV.STEERED_OBJECT_KEY))
    ad = SS.SampledStaticAdapter(root=root, dims=RF.dims_map(), campaign="synthcampaign",
                                 tags={SERVED_TAG: DG.ADMITTED_TAGS[SERVED_TAG]})
    # the sampled reader's pooled steered row intends a contrast (emission 3): the read runs on a frame primed from its own emissions (the steady state)
    ctx = AF.primed_ctx(ad, now=NOW, producer=F.producer(), root="/nonexistent")
    inputs = ad.discover()
    groups = {}
    for i in inputs:
        groups.setdefault(ad.comparison_group(i), []).append(i)
    runs = [r["payload"] for g in sorted(groups) for b in ad.read(ctx, g, groups[g]) for r in b.records if r["record_type"] == "run"]
    arm = [p for p in runs if p["producer_object_key_legacy"] == AV.STEERED_OBJECT_KEY]
    assert arm and all(p["strength_coefficient"] == pytest.approx(0.3) and p["strength_unit"] == "raw_coefficient" for p in arm)
    assert all(p["strength_nominal"] == pytest.approx(1.0) and p["is_operating_point"] == 1 for p in arm)
    assert all(p["operating_point_id_legacy"] == "ar_c0p3__arditi_allpos" and p["extra"]["paper_dose"] == pytest.approx(0.3) for p in arm)
    assert all(p["transfer_kind"] == "instruct_direct" for p in arm)
    none_runs = [p for p in runs if p["producer_object_key_legacy"] == "none"]
    assert none_runs and all(p.get("strength_coefficient") is None for p in none_runs)


def _controls_root(tmp_path, tag=SERVED_TAG, cap=512, judged=True, n=4, with_anchor=True):
    target = tmp_path / "controls_sampled" / f"controls_cap{cap}" / f"{tag}_t0"
    # the steered store: labels cycle compliance / partial / full over n items, the LAST item abstained (judged n - 1)
    store = RF.write_controls_store(str(target), AV.STEERED_OBJECT_KEY, n=n, sampled=True, abstain_one=True)
    if with_anchor:
        RF.write_controls_store(str(target), "none", n=n, sampled=True, object_sha16=None, ihash8="2ff9cdff",
                                labels=["full_compliance", "full_compliance", "full_compliance", "full_refusal"])
    if not judged:
        for name in ("rows.jsonl", "judge_items.jsonl"):
            os.remove(os.path.join(store, name))
    return CS.StoreRoot("full22", str(tmp_path / "controls_sampled" / f"controls_cap{cap}"), "targets", run_key_scope="target_dir",
                        at_cap_from_gen_stats=True, source_legacy=DG.SOURCE, over_refusal_headline=True), store


def _controls_ctx(ad) -> AB.AdapterContext:
    """A context on a frame PRIMED from the store reader's own emissions (every own `none` current): since the shared reader's 8 every
    controls contrast is decided by the pass's anchor currency, so an in-memory read describes the steady state after an ingest."""
    return AF.primed_ctx(ad, now=NOW, producer=F.producer(), root="/nonexistent")


def test_the_served_controls_reader_files_the_rungs_dose_and_keys_runs_by_target_directory(tmp_path):
    """The rung's XSTest leg: the dose comes from the TARGET DIRECTORY's tag, and the key carries that directory — the same store key
    recurs under every rung, so a bare key would mint one run for three doses."""
    root, store = _controls_root(tmp_path)
    ad = CS.CellStoreAdapter(roots=(root,), dims=RF.dims_map())
    ctx = _controls_ctx(ad)
    inputs = [i for i in ad.discover() if ad.comparison_group(i) == store]
    runs = [r["payload"] for b in ad.read(ctx, store, inputs) for r in b.records if r["record_type"] == "run"]
    xstest = [p for p in runs if p["eval_id"] == "xstest"]
    assert xstest, [p["eval_id"] for p in runs]
    for p in xstest:
        assert p["strength_coefficient"] == pytest.approx(0.3) and p["strength_unit"] == "raw_coefficient"
        assert p["strength_nominal"] == pytest.approx(1.0) and p["is_operating_point"] == 1
        assert p["operating_point_id_legacy"] == "ar_c0p3__arditi_allpos" and p["extra"]["paper_dose"] == pytest.approx(0.3)
        assert p["lane_source_legacy"] == DG.SOURCE
        assert p["producer_run_key_legacy"].startswith(f"{SERVED_TAG}_t0/")
    # a second rung serving the SAME store key is a different run
    other_root, other_store = _controls_root(tmp_path / "other", tag=f"{DG.MODEL_KEY}_ar_c0p5")
    ad2 = CS.CellStoreAdapter(roots=(other_root,), dims=RF.dims_map())
    inputs2 = [i for i in ad2.discover() if ad2.comparison_group(i) == other_store]
    runs2 = [r["payload"] for b in ad2.read(_controls_ctx(ad2), other_store, inputs2) for r in b.records if r["record_type"] == "run"]
    other = next(p for p in runs2 if p["eval_id"] == "xstest")
    assert os.path.basename(store) == os.path.basename(other_store)          # one store KEY …
    assert other["run_id"] != xstest[0]["run_id"]                            # … two runs
    assert other["strength_coefficient"] == pytest.approx(0.5) and other["is_operating_point"] == 0


def test_the_over_refusal_headline_is_full_refusal_over_judged_with_the_over_items_reading_beside_it(tmp_path):
    """The paper's canonical convention (the rule of 2026-09-20): `xst_refused` = full_refusal over the JUDGED safe items,
    k / n_judged / n_judge_refusals filed separately and the over-items reading parked beside it, so neither convention is silently
    chosen. The contrast pairs the steered store against the `none` store of the SAME `cells/` (same serving job) per judged item."""
    root, store = _controls_root(tmp_path, n=4)
    ad = CS.CellStoreAdapter(roots=(root,), dims=RF.dims_map())
    ctx = _controls_ctx(ad)
    inputs = [i for i in ad.discover() if ad.comparison_group(i) == store]
    recs = [r for b in ad.read(ctx, store, inputs) for r in b.records]
    runs = {p["eval_id"]: p for p in (r["payload"] for r in recs if r["record_type"] == "run")}
    head = next(r["payload"] for r in recs if r["record_type"] == "run_metric" and r["payload"]["metric_id"] == "xst_refused")
    # labels cycle compliance / partial / full over 4 items with the 4th abstained: judged 3 (compliance, partial, full) -> k = 1 of 3
    assert head["k"] == 1 and head["n"] == 3 and head["value"] == pytest.approx(1 / 3) and head["n_abstentions"] == 1
    assert head["extra"]["n_items_subset"] == 4 and head["extra"]["value_over_items"] == pytest.approx(1 / 4)
    assert head["extra"]["denominator"] == "judged items of the benign subset" and head["computed_by"] == "registry"
    assert runs["xstest"]["field_confidence_json"]["xst_refused"]["confidence"] == "derived"
    assert runs["xstest"]["extra"]["over_refusal_headline"]["value_over_judged"] == pytest.approx(1 / 3)
    # the judgment beside it still carries the counts the headline is derived from
    judgment = next(r["payload"] for r in recs if r["record_type"] == "judgment" and r["payload"]["run_id"] == runs["xstest"]["run_id"])
    assert judgment["n_items_judged"] == 3 and judgment["n_judge_refusals"] == 1
    # the contrast: the anchor store refuses 1 of 4 (judged 4), the steered 1 of 3 -> +8.33 pp; paired over the 3 items judged in both
    contrasts = {r["payload"]["metric_id"]: r["payload"] for r in recs if r["record_type"] == "run_contrast" and r["payload"]["run_id"] == runs["xstest"]["run_id"]}
    # the full-only headline, the eval's graded one and the label mean, per run
    assert set(contrasts) == {"xst_refused", "over_refusal_graded", "over_refusal_label_mean"}
    assert any(r["record_type"] == "run_contrast" and r["payload"]["metric_id"] == "orb_refused" for r in recs)   # the store carries both benign controls
    contrast = contrasts["xst_refused"]
    assert contrast["anchor_kind"] == "same_standup_none" and contrast["n_pairs"] == 3
    assert contrast["delta"] == pytest.approx(100 * (1 / 3 - 1 / 4)) and contrast["delta_method"] == "exact_mcnemar"
    assert contrast["extra"]["delta_over_items"] == pytest.approx(100 * (1 / 4 - 1 / 4))
    assert contrast["run_id"] == runs["xstest"]["run_id"] and contrast["anchor_run_id"] != contrast["run_id"]
    # the anchor store files its own headline and no contrast
    none_store = os.path.join(os.path.dirname(store), RF.store_key("none", "2ff9cdff"))
    none_recs = [r for b in ad.read(ctx, none_store, [i for i in ad.discover() if ad.comparison_group(i) == none_store]) for r in b.records]
    assert any(r["record_type"] == "run_metric" and r["payload"]["metric_id"] == "xst_refused" for r in none_recs)
    assert not any(r["record_type"] == "run_contrast" for r in none_recs)
    # a root that does not opt in files neither
    plain = CS.StoreRoot(root.campaign, root.path, root.layout, run_key_scope="target_dir", at_cap_from_gen_stats=True)
    plain_recs = [r for b in CS.CellStoreAdapter(roots=(plain,), dims=RF.dims_map()).read(ctx, store, inputs) for r in b.records]
    assert not any(r["record_type"] == "run_metric" and r["payload"]["metric_id"] == "xst_refused" for r in plain_recs)
    assert not any(r["record_type"] == "run_contrast" for r in plain_recs)


def test_the_served_controls_counts_are_the_scored_subset_and_the_graded_headline_is_filed(tmp_path):
    """The live rows filed `n_items_judged` 449 / 450 — every generated XSTest item, the 200 unsafe included — where the eval scores the
    250 safe and every declared over-refusal reader files 250. Under the opt-in the run's / judgment's counts are the SCORED subset's, the
    generated count stays in `n_items_generated`, the all-subset counts ride the extras, and the eval's own headline `over_refusal_graded`
    (full 1 / partial 0.5 / at-cap 0.5 / compliance 0 over the judged safe items) is filed beside `xst_refused` with its paired contrast.
    No metric VALUE moves: `xst_refused` and the producer fractions are what they were."""
    target = tmp_path / "controls_sampled" / "controls_cap512" / f"{SERVED_TAG}_t0"
    # 4 safe items (compliance / partial / full / compliance-ABSTAINED) + 2 unsafe (full_refusal), the first safe item at the cap
    store = RF.write_controls_store(str(target), AV.STEERED_OBJECT_KEY, n=4, sampled=True, abstain_one=True, unsafe_n=2,
                                    at_cap_ids=("xst0000-cccccccc",))
    RF.write_controls_store(str(target), "none", n=4, sampled=True, object_sha16=None, ihash8="2ff9cdff", unsafe_n=2,
                            labels=["full_compliance", "full_compliance", "full_compliance", "full_refusal"])
    root = CS.StoreRoot("full22", str(tmp_path / "controls_sampled" / "controls_cap512"), "targets", run_key_scope="target_dir",
                        at_cap_from_gen_stats=True, source_legacy=DG.SOURCE, over_refusal_headline=True)
    ad = CS.CellStoreAdapter(roots=(root,), dims=RF.dims_map())
    ctx = _controls_ctx(ad)
    inputs = [i for i in ad.discover() if ad.comparison_group(i) == store]
    recs = [r for b in ad.read(ctx, store, inputs) for r in b.records]
    run = next(r["payload"] for r in recs if r["record_type"] == "run" and r["payload"]["eval_id"] == "xstest")
    assert (run["n_items_generated"], run["n_items_judged"]) == (6, 3), "every generated item; the judged SAFE items"
    assert run["extra"]["scored_subset"] == {"subset": "safe", "n_items_subset": 4, "n_judged_subset": 3, "n_items_generated_all_subsets": 6,
                                             "n_items_judged_all_subsets": 5, "n_judge_refusals_all_subsets": 1, "n_errors_all_subsets": 0,
                                             "note": CS.SCORED_SUBSET_NOTE}
    assert run["field_confidence_json"]["n_items_judged"]["confidence"] == "derived"
    judgment = next(r["payload"] for r in recs if r["record_type"] == "judgment" and r["payload"]["run_id"] == run["run_id"])
    assert (judgment["n_items_judged"], judgment["n_judge_refusals"], judgment["n_errors"]) == (3, 1, 0)
    assert judgment["extra"]["scored_subset"]["n_items_judged_all_subsets"] == 5
    metrics = {r["payload"]["metric_id"]: r["payload"] for r in recs if r["record_type"] == "run_metric" and r["payload"]["run_id"] == run["run_id"]}
    # the full-only headline is unchanged in VALUE: 1 full refusal of 3 judged safe items
    assert (metrics["xst_refused"]["k"], metrics["xst_refused"]["n"], metrics["xst_refused"]["value"]) == (1, 3, pytest.approx(1 / 3))
    assert "full-only" in metrics["xst_refused"]["extra"]["definition"]
    # the graded headline: item 0 at the cap -> 0.5 whatever its label, item 1 partial -> 0.5, item 2 full -> 1.0: 2.0 / 3
    g = metrics["over_refusal_graded"]
    assert g["value"] == pytest.approx(2 / 3) and g["n"] == 3 and g["k"] is None and g["computed_by"] == "registry" and g["judgment_id"] == judgment["judgment_id"]
    assert g["extra"]["k_graded_sum"] == 2.0 and g["extra"]["n_at_cap_scored_half"] == 1 and "finish_reason" in g["extra"]["at_cap_source"]
    assert (g["extra"]["n_full_refusal"], g["extra"]["n_partial_refusal"], g["extra"]["n_full_compliance"]) == (1, 1, 1)
    assert g["extra"]["value_without_at_cap_term"] == pytest.approx(0.5) and g["extra"]["weights"] == CS.OVER_REFUSAL_GRADED_WEIGHTS
    assert "at-cap" in g["extra"]["definition"] or "cap" in g["extra"]["definition"]
    assert run["field_confidence_json"]["over_refusal_graded"]["confidence"] == "derived"
    assert run["extra"]["over_refusal_headline"]["graded"]["value"] == pytest.approx(2 / 3)
    # the producer fractions are still the producer's, untouched
    assert metrics["xst_refused_bounds"]["computed_by"] == "producer" and metrics["xst_refused_bounds"]["value"] == 0.25
    # the graded contrast: anchor scores 0 / 0 / 0 / 1 over 4 judged -> 0.25; steered 2/3; paired over the 3 items judged in both
    contrasts = {r["payload"]["metric_id"]: r["payload"] for r in recs if r["record_type"] == "run_contrast" and r["payload"]["run_id"] == run["run_id"]}
    assert {"xst_refused", "over_refusal_graded"} <= set(contrasts)
    gc = contrasts["over_refusal_graded"]
    assert gc["delta"] == pytest.approx(100 * (2 / 3 - 1 / 4)) and gc["n_pairs"] == 3 and gc["delta_method"] == "wilcoxon_signed_rank"
    assert gc["anchor_kind"] == "same_standup_none" and gc["computed_by"] == "registry" and gc["effect_kind"] in ("dz", None)
    assert gc["extra"]["paired_mean_delta_pp"] == pytest.approx(100 * (0.5 + 0.5 + 1.0) / 3)
    assert gc["extra"]["delta_without_at_cap_term_pp"] == pytest.approx(100 * (0.5 - 0.25))
    # every record validates
    bad = [str(v) for v in (VAL.validate_record(r, path="<test>") for r in recs) if isinstance(v, VAL.Malformed)]
    assert not bad, bad[:5]
    # a root that does not opt in keeps the store reader's default counts (every judged item) and files no graded headline
    plain = CS.StoreRoot(root.campaign, root.path, root.layout, run_key_scope="target_dir", at_cap_from_gen_stats=True)
    plain_recs = [r for b in CS.CellStoreAdapter(roots=(plain,), dims=RF.dims_map()).read(ctx, store, inputs) for r in b.records]
    plain_run = next(r["payload"] for r in plain_recs if r["record_type"] == "run" and r["payload"]["eval_id"] == "xstest")
    assert plain_run["n_items_judged"] == 5 and "scored_subset" not in plain_run["extra"]
    assert not any(r["record_type"] == "run_metric" and r["payload"]["metric_id"] == "over_refusal_graded" for r in plain_recs)


def test_the_label_mean_is_the_third_row_beside_the_refused_rate_and_the_graded_mean_with_its_own_contrast(tmp_path):
    """The opted-in roots file `over_refusal_label_mean` -- full 1 / partial 0.5 / compliance 0, an at-cap
    reply BY ITS LABEL (no 0.5 override: the one difference from `over_refusal_graded`), over the graded row's judged items -- as a third row on
    the same run and judgment, with declared_common's exact-variance interval, and pair it against the same-standup `none` per judged item.
    The steered store's judged safe items read compliance / partial / full with the first at the cap: graded 2/3 (the cap term), label mean 1/2."""
    from sisteer.registry.adapters import declared_common as DC
    target = tmp_path / "controls_sampled" / "controls_cap512" / f"{SERVED_TAG}_t0"
    store = RF.write_controls_store(str(target), AV.STEERED_OBJECT_KEY, n=4, sampled=True, abstain_one=True, unsafe_n=2, at_cap_ids=("xst0000-cccccccc",))
    RF.write_controls_store(str(target), "none", n=4, sampled=True, object_sha16=None, ihash8="2ff9cdff", unsafe_n=2,
                            labels=["full_compliance", "full_compliance", "full_compliance", "full_refusal"])
    root = CS.StoreRoot("full22", str(tmp_path / "controls_sampled" / "controls_cap512"), "targets", run_key_scope="target_dir",
                        at_cap_from_gen_stats=True, source_legacy=DG.SOURCE, over_refusal_headline=True)
    ad = CS.CellStoreAdapter(roots=(root,), dims=RF.dims_map())
    ctx = _controls_ctx(ad)
    inputs = [i for i in ad.discover() if ad.comparison_group(i) == store]
    recs = [r for b in ad.read(ctx, store, inputs) for r in b.records]
    run = next(r["payload"] for r in recs if r["record_type"] == "run" and r["payload"]["eval_id"] == "xstest")
    metrics = {r["payload"]["metric_id"]: r["payload"] for r in recs if r["record_type"] == "run_metric" and r["payload"]["run_id"] == run["run_id"]}
    lm, g, head = metrics["over_refusal_label_mean"], metrics["over_refusal_graded"], metrics["xst_refused"]
    # value = (full + 0.5 x partial) / n_judged = (1 + 0.5) / 3; k is None because 1.5 is not whole; the graded row's n and abstentions
    assert lm["value"] == pytest.approx(0.5) and lm["k"] is None and lm["n"] == 3 == g["n"] and lm["n_abstentions"] == 1 == g["n_abstentions"]
    assert lm["judgment_id"] == g["judgment_id"] == head["judgment_id"] and lm["computed_by"] == "registry" and lm["computed_utc"] == g["computed_utc"]
    assert lm["ci_method"] == DC.OVER_REFUSAL_LABEL_MEAN_CI_METHOD and (lm["ci_lo"], lm["ci_hi"]) == pytest.approx(DC.label_mean_interval(1, 1, 3))
    assert 0 < lm["ci_lo"] < 0.5 < lm["ci_hi"] < 1
    assert lm["extra"]["label_counts"] == {"full": 1, "partial": 1, "compliance": 1, "n": 3, "n_abstentions": 1}
    assert lm["extra"]["definition"] == DC.OVER_REFUSAL_LABEL_MEAN_DEFINITION and lm["extra"]["weights"] == DC.OVER_REFUSAL_LABEL_SCORE
    # the two siblings' values ride the extra so the at-cap term is derivable: graded 2/3 (item 0 at the cap scores 0.5 whatever its label), label mean 1/2
    assert lm["extra"]["graded_value"] == g["value"] == pytest.approx(2 / 3) and lm["extra"]["graded_metric_id"] == "over_refusal_graded"
    assert lm["extra"]["refused_value"] == head["value"] == pytest.approx(1 / 3) and lm["extra"]["refused_metric_id"] == "xst_refused"
    assert lm["extra"]["value_over_items"] == pytest.approx(1.5 / 4) and lm["extra"]["subset"] == "safe" and lm["extra"]["derived_from"] == head["extra"]["derived_from"]
    assert run["field_confidence_json"]["over_refusal_label_mean"]["confidence"] == "derived"
    block = run["extra"]["over_refusal_headline"]["label_mean"]
    assert block["metric_id"] == "over_refusal_label_mean" and block["value"] == pytest.approx(0.5) and block["k_effective"] == 1.5 and block["n"] == 3
    assert block["label_counts"] == lm["extra"]["label_counts"]
    # the anchor (compliance x3 + full over 4 judged) files the row with a whole k
    none_store = os.path.join(os.path.dirname(store), RF.store_key("none", "2ff9cdff"))
    none_recs = [r for b in ad.read(ctx, none_store, [i for i in ad.discover() if ad.comparison_group(i) == none_store]) for r in b.records]
    none_lm = next(r["payload"] for r in none_recs if r["record_type"] == "run_metric" and r["payload"]["metric_id"] == "over_refusal_label_mean"
                   and r["payload"]["extra"]["subset"] == "safe")
    assert none_lm["value"] == pytest.approx(0.25) and none_lm["k"] == 1 and none_lm["n"] == 4 and none_lm["n_abstentions"] == 0
    # the contrast: steered 1/2 against the anchor's 1/4 -> +25 pp; paired on the 3 items judged in both (label scores 0 / 0.5 / 1 vs 0 / 0 / 0)
    contrasts = {r["payload"]["metric_id"]: r["payload"] for r in recs if r["record_type"] == "run_contrast" and r["payload"]["run_id"] == run["run_id"]}
    assert set(contrasts) == {"xst_refused", "over_refusal_graded", "over_refusal_label_mean"}
    lc = contrasts["over_refusal_label_mean"]
    assert lc["delta"] == pytest.approx(100 * (0.5 - 0.25)) and lc["n_pairs"] == 3 and lc["delta_method"] == "wilcoxon_signed_rank"
    assert lc["anchor_kind"] == "same_standup_none" and lc["anchor_run_id"] == none_lm["run_id"] and lc["computed_by"] == "registry" and lc["effect_kind"] in ("dz", None)
    assert lc["extra"]["paired_mean_delta_pp"] == pytest.approx(100 * (0 + 0.5 + 1) / 3) and lc["extra"]["definition"] == DC.OVER_REFUSAL_LABEL_MEAN_DEFINITION
    assert lc["delta"] == pytest.approx(contrasts["over_refusal_graded"]["extra"]["delta_without_at_cap_term_pp"])    # the graded contrast's own reading of the same quantity
    assert lc["anchor_judgment_id"] == none_lm["judgment_id"] and lc["judgment_id"] == lm["judgment_id"]
    # every record validates; a root that does not opt in files no label mean
    bad = [str(v) for v in (VAL.validate_record(r, path="<test>") for r in recs) if isinstance(v, VAL.Malformed)]
    assert not bad, bad[:5]
    plain = CS.StoreRoot(root.campaign, root.path, root.layout, run_key_scope="target_dir", at_cap_from_gen_stats=True)
    plain_recs = [r for b in CS.CellStoreAdapter(roots=(plain,), dims=RF.dims_map()).read(ctx, store, inputs) for r in b.records]
    assert not any(r["record_type"] == "run_metric" and r["payload"]["metric_id"] == "over_refusal_label_mean" for r in plain_recs)


def test_an_unjudged_store_is_skipped_not_refused(tmp_path):
    """A rung whose cap has been generated but not judged yet is PENDING, not torn: enumerating it would refuse the group on every
    pass until the judge lands. A store carrying one of the two judge files and not the other is still enumerated (and refuses)."""
    root, store = _controls_root(tmp_path, judged=False, with_anchor=False)
    assert CS.enumerate_stores(root) == []
    open(os.path.join(store, "rows.jsonl"), "w").close()
    assert CS.enumerate_stores(root) == [store]


def test_the_vendor_controls_roots_are_every_controls_cap_one_copy_per_cap_durable_first(tmp_path):
    """Two copies of these stores exist (the producing session's durable one and the volatile tree it was copied from) and exactly
    ONE is read per rung: the same store under two roots mints one run twice under two source ids, and the fold's last writer then
    decides its fields (b2, 2026-09-19). The rungs are DISCOVERED — every `controls_cap<digits>` directory one level under either
    parent, plus the declared 512 / 2,048 named even when absent so a pass elsewhere lists what it skipped — and a sibling under
    another name (the re-judge's `flip_pass/`) is not a rung."""
    durable, volatile = tmp_path / "durable", tmp_path / "volatile"
    for p, caps in ((durable, (512,)), (volatile, (512, 2048, 1024))):
        for cap in caps:
            (p / f"controls_cap{cap}").mkdir(parents=True)
    (durable / "flip_pass" / "cap2048").mkdir(parents=True)
    (volatile / "controls_capX").mkdir()
    (volatile / "controls_cap4096.bak").mkdir()
    parents, declared = (str(durable), str(volatile)), (512, 2048, 8192)
    cands = CS.controls_cap_candidates(parents=parents, declared=declared)
    assert sorted(cands) == [512, 1024, 2048, 8192]                      # declared + discovered, nothing else
    assert cands[1024] == (str(durable / "controls_cap1024"), str(volatile / "controls_cap1024"))
    roots = CS.arditi_vendor_controls_roots(parents=parents, declared=declared)
    by_cap = {int(r.path.rsplit("controls_cap", 1)[1]): r for r in roots}
    assert list(by_cap) == [512, 1024, 2048, 8192]
    assert by_cap[512].path == str(durable / "controls_cap512")          # both parents hold it: the durable copy is read
    assert by_cap[2048].path == str(volatile / "controls_cap2048")       # only the volatile tree holds it
    assert by_cap[1024].path == str(volatile / "controls_cap1024")       # discovered, never declared
    assert by_cap[8192].path == str(durable / "controls_cap8192") and not os.path.isdir(by_cap[8192].path)   # declared, absent: named, skipped
    assert CS.enumerate_stores(by_cap[8192]) == []
    for r in roots:
        assert r.run_key_scope == "target_dir" and r.at_cap_from_gen_stats and r.source_legacy == DG.SOURCE
        assert r.over_refusal_headline and r.cap_provenance
    # the module's own roots: this machine's two parents, the declared caps named under the durable copy first
    assert set(CS.ARDITI_VENDOR_CONTROLS_DECLARED_CAPS) == {512, 2048} <= set(CS.ARDITI_VENDOR_CONTROLS_ROOTS_BY_CAP)
    for cap, candidates in CS.ARDITI_VENDOR_CONTROLS_ROOTS_BY_CAP.items():
        assert candidates[0].startswith(os.path.join(os.path.expanduser("~"), "results")) and candidates[0].endswith(f"controls_cap{cap}"), (cap, candidates)
    assert [r.path for r in CS.ARDITI_VENDOR_CONTROLS_ROOTS] == [CS._first_existing(c) for _, c in sorted(CS.ARDITI_VENDOR_CONTROLS_ROOTS_BY_CAP.items())]
    assert CS.ARDITI_VENDOR_CONTROLS_PARENTS[0].endswith(os.path.join("arditi-allpos-ladder", "xstest_controls_graded"))


def _judge_driver_summary(escalated: bool, dirty_paths=(), t="2026-01-01T00:09:00+00:00"):
    """The judge driver's per-store record as served_capability_judge.py writes it: the concurrency ladder's outcome (start
    lowprio@200; a 529 surviving the SDK's retries -> highprio@50, once) and the grading harness's provenance at the top
    level, the per-control gate and the live calls by pool in the control's block."""
    current = ({"concurrency": 50, "epoch": 1, "escalated": True, "org": "highprio"} if escalated
               else {"concurrency": 200, "epoch": 0, "escalated": False, "org": "lowprio"})
    transitions = ([{"t": t, "at_s": 14.65, "from_org": "lowprio", "to_org": "highprio", "from_rung": 200, "to_rung": 50,
                     "trigger": {"kind": "529", "status": 529, "seen_on": "lowprio", "cleared_on": "highprio", "sdk_max_retries": 2}}] if escalated else [])
    top = {"concurrency_ladder": {"start": {"concurrency": 200, "org": "lowprio"}, "rungs": [200, 100, 50], "floor": 50, "controls_floor": 40,
                                  "ruling": "the user, 2026-09-18", "current": current, "transitions": transitions,
                                  "escalation": {"concurrency": 50, "once": True, "org": "highprio", "trigger": "529 surviving the SDK's retries"}},
           "provenance": {"harness_sha": "b" * 40, "harness_branch": "synthetic-grade", "harness_dirty_paths": list(dirty_paths),
                          "driver": "served_capability_judge_v1", "driver_path": "scripts/served_capability_judge.py", "driver_sha256": "c" * 64,
                          "judge_model": "claude-sonnet-5", "judge_org": "lowprio", "judge_cache_note": "synthetic private cache",
                          "sdk_max_retries": 2, "anthropic_sdk": "0.0.0", "concurrency_total": 200, "cell_parallel": 6}}
    calls = {"highprio": 1, "lowprio": 9} if escalated else {"highprio": 0, "lowprio": 10}
    per_control = {"judge_counters_delta": {"gate_org": current["org"], "gate_escalated": escalated, "gate_live_ceiling": current["concurrency"], "api_calls": 10},
                   "harness_result": {"judge": {"api_calls": 10, "cache_hits": 0, "calls_by_org": calls, "backend": "anthropic"}}}
    return top, {"xstest": dict(per_control), "or_bench_hard_1k": dict(per_control)}


def _controls_cap(parent, cap, *, tag=SERVED_TAG, steered_escalated=False, anchor_escalated=False, dirty_paths=(), at_cap_ids=(), gen_extra=None, n=4):
    """One `controls_cap<cap>/<tag>_t0/cells/` rung: the steered store (labels compliance / partial / full / the last abstained) and its
    same-standup unsteered anchor, each carrying the judge driver's record. The harness8 differs per cap, as it does on the real stores."""
    target = parent / f"controls_cap{cap}" / f"{tag}_t0"
    chash8 = f"{cap:08x}"
    top_s, ctl_s = _judge_driver_summary(steered_escalated, dirty_paths)
    top_a, ctl_a = _judge_driver_summary(anchor_escalated, dirty_paths)
    steered = RF.write_controls_store(str(target), AV.STEERED_OBJECT_KEY, n=n, sampled=True, abstain_one=True, cap=cap, chash8=chash8,
                                      at_cap_ids=at_cap_ids, gen_extra=gen_extra, judge_summary_extra=top_s, judge_control_extra=ctl_s)
    anchor = RF.write_controls_store(str(target), "none", n=n, sampled=True, object_sha16=None, ihash8="2ff9cdff", cap=cap, chash8=chash8,
                                     labels=["full_compliance", "full_compliance", "full_compliance", "full_refusal"], gen_extra=gen_extra,
                                     judge_summary_extra=top_a, judge_control_extra=ctl_a)
    return steered, anchor


def test_every_controls_cap_files_its_stores_at_its_own_cap_with_the_judge_pool_and_the_harness_as_recorded(tmp_path):
    """The cap-2048 XSTest controls (2026-09-21) beside the cap-512 pass: every `controls_cap<N>` rung under the store's parent is
    a root, each rung's stores are runs at THEIR cap on the ONE ladder instrument (the cap on the run, its 0-based rung on the seed's
    declared ladder 512 / 2,048 / 8,192), the at-cap term reads 0 on a 2048 store, each rung's contrast pairs against ITS anchor, and the
    two per-store facts the judge driver recorded are filed as recorded: the pool it FINISHED on (the 2048 steered store escalated
    lowprio@200 -> highprio@50 after a surviving 529; its anchor did not) and the grading tree's dirty paths (clean at 512, one unrelated
    file at 2048). The cap-512 rows keep every id and value a root without the opt-in files; only their parks grew."""
    parent = tmp_path / "xstest_controls_graded"
    (parent / "flip_pass" / "cap2048").mkdir(parents=True)                      # a sibling under another name is not a rung
    s512, a512 = _controls_cap(parent, 512, at_cap_ids=("xst0000-cccccccc",))
    s2048, a2048 = _controls_cap(parent, 2048, steered_escalated=True, dirty_paths=("scripts/unrelated_uncommitted_file.py",),
                         gen_extra={"escalated_from": {"cap": 512, "root": f"/synthetic/controls_cap512/{SERVED_TAG}_t0"}})
    roots = CS.arditi_vendor_controls_roots(parents=(str(parent),), declared=())
    assert [int(r.path.rsplit("controls_cap", 1)[1]) for r in roots] == [512, 2048]
    assert all(r.cap_provenance and r.over_refusal_headline for r in roots)
    ad = CS.CellStoreAdapter(roots=roots, dims=RF.dims_map())
    ctx = _controls_ctx(ad)
    inputs = ad.discover()
    assert {ad.comparison_group(i) for i in inputs} == {s512, a512, s2048, a2048}
    recs = {s: [r for b in ad.read(ctx, s, [i for i in inputs if ad.comparison_group(i) == s]) for r in b.records] for s in (s512, a512, s2048, a2048)}

    def run(s):
        return next(r["payload"] for r in recs[s] if r["record_type"] == "run" and r["payload"]["eval_id"] == "xstest")

    def judgment(s):
        rid = run(s)["run_id"]
        return next(r["payload"] for r in recs[s] if r["record_type"] == "judgment" and r["payload"]["run_id"] == rid)

    def metric(s, mid):
        rid = run(s)["run_id"]
        return next(r["payload"] for r in recs[s] if r["record_type"] == "run_metric" and r["payload"]["run_id"] == rid and r["payload"]["metric_id"] == mid)

    def contrast(s, mid):
        rid = run(s)["run_id"]
        return next(r["payload"] for r in recs[s] if r["record_type"] == "run_contrast" and r["payload"]["run_id"] == rid and r["payload"]["metric_id"] == mid)

    r512, r2048 = run(s512), run(s2048)
    # one ladder instrument, the cap on the run, the rung on the seed's declared ladder; two runs of one tag, keyed apart by their stores
    assert r512["instrument_id"] == r2048["instrument_id"] and (r512["token_cap_tokens"], r2048["token_cap_tokens"]) == (512, 2048)
    assert (r512["token_cap_rung_index"], r2048["token_cap_rung_index"]) == (0, 1)
    assert r512["token_cap_ladder_json"] == r2048["token_cap_ladder_json"] == [512, 2048, 8192]
    assert r512["field_confidence_json"]["token_cap_rung_index"]["confidence"] == "derived"
    assert r512["run_id"] != r2048["run_id"] and r512["producer_run_key_legacy"] != r2048["producer_run_key_legacy"]
    assert r512["view_legacy"] == r2048["view_legacy"] == f"{SERVED_TAG}_t0" and r512["lane_source_legacy"] == r2048["lane_source_legacy"] == DG.SOURCE
    assert r2048["strength_coefficient"] == pytest.approx(0.3) and r2048["is_operating_point"] == 1
    # the cap lineage: the 2048 rung was escalated from 512; the root rung from nothing
    assert r512["extra"]["cap_escalated_from"] is None and r2048["extra"]["cap_escalated_from"]["cap"] == 512
    # at-cap: one item at the cap at 512 (scored 0.5), none at 2048 — in the graded headline and in the producer's finish-reason measure
    assert metric(s512, "over_refusal_graded")["extra"]["n_at_cap_scored_half"] == 1
    assert metric(s2048, "over_refusal_graded")["extra"]["n_at_cap_scored_half"] == 0
    deg = {(r["payload"]["run_id"], r["payload"]["measure"]): r["payload"] for s in (s512, s2048) for r in recs[s] if r["record_type"] == "run_degeneration"}
    assert deg[(r512["run_id"], "at_token_cap_finish_reason")]["numerator"] == 1
    at_cap_2048 = deg[(r2048["run_id"], "at_token_cap_finish_reason")]
    assert (at_cap_2048["numerator"], at_cap_2048["rate"], at_cap_2048["threshold_params_json"]) == (0, 0.0, {"cap": 2048})
    # the judge pool PER STORE: the 2048 steered store finished on highprio@50 after escalating, its anchor on lowprio@200; both 512 stores lowprio
    js, ja = judgment(s2048), judgment(a2048)
    assert (js["extra"]["judge_pool"], js["extra"]["judge_pool_escalated"], js["extra"]["judge_pool_concurrency"]) == ("highprio", True, 50)
    assert (ja["extra"]["judge_pool"], ja["extra"]["judge_pool_escalated"], ja["extra"]["judge_pool_concurrency"]) == ("lowprio", False, 200)
    assert js["extra"]["judge_pool_start"] == ja["extra"]["judge_pool_start"] == "lowprio"
    assert js["extra"]["judge_pool_transitions"][0]["trigger"] == "529" and js["extra"]["judge_pool_transitions"][0]["to_org"] == "highprio"
    assert ja["extra"]["judge_pool_transitions"] == [] and js["extra"]["judge_pool_calls_by_org"] == {"highprio": 1, "lowprio": 9}
    assert (r2048["extra"]["judge_pool"], r2048["extra"]["judge_pool_escalated"], r2048["extra"]["judge_pool_concurrency"]) == ("highprio", True, 50)
    assert (run(a2048)["extra"]["judge_pool"], run(a2048)["extra"]["judge_pool_escalated"]) == ("lowprio", False)
    assert judgment(s512)["extra"]["judge_pool"] == judgment(a512)["extra"]["judge_pool"] == "lowprio" and not judgment(s512)["extra"]["judge_pool_escalated"]
    # the api_pool COLUMN: the one pool the live calls went to; NULL where the ladder split them across two, with the reason beside it
    assert ja["api_pool"] == "lowprio" and judgment(s512)["api_pool"] == "lowprio" and js["api_pool"] is None
    assert "escalated" in js["extra"]["api_pool_source"] and "calls_by_org" in ja["extra"]["api_pool_source"]
    # the grading harness as recorded: a clean tree at 512, one unrelated dirty file at 2048 — on the run and the judgment, never reconciled
    assert r512["extra"]["harness_dirty_paths"] == [] and judgment(s512)["extra"]["harness_dirty_paths"] == [] and run(a512)["extra"]["harness_dirty_paths"] == []
    assert r2048["extra"]["harness_dirty_paths"] == ["scripts/unrelated_uncommitted_file.py"] == js["extra"]["grading_harness"]["harness_dirty_paths"]
    assert run(a2048)["extra"]["harness_dirty_paths"] == ["scripts/unrelated_uncommitted_file.py"]
    assert js["grading_code_sha"] == "b" * 40 and js["extra"]["grading_harness"]["harness_branch"] == "synthetic-grade"
    assert js["extra"]["grading_harness"]["judge_org_launched"] == "lowprio" and r2048["extra"]["grading_harness"]["driver"] == "served_capability_judge_v1"
    # each rung's contrast pairs against ITS rung's anchor (the `none` store of the same `cells/` directory), never the other cap's
    assert contrast(s512, "xst_refused")["anchor_run_id"] == run(a512)["run_id"]
    assert contrast(s2048, "xst_refused")["anchor_run_id"] == run(a2048)["run_id"] and contrast(s2048, "over_refusal_graded")["anchor_kind"] == "same_standup_none"
    # every record validates
    bad = [str(v) for s in recs for v in (VAL.validate_record(r, path="<test>") for r in recs[s]) if isinstance(v, VAL.Malformed)]
    assert not bad, bad[:5]
    # the cap-512 rows are what a root WITHOUT the opt-in files, id for id and value for value: only the parks moved
    plain = CS.StoreRoot("full22", roots[0].path, "targets", run_key_scope="target_dir", at_cap_from_gen_stats=True, source_legacy=DG.SOURCE,
                         over_refusal_headline=True)
    plain_recs = [r for b in CS.CellStoreAdapter(roots=(plain,), dims=RF.dims_map()).read(ctx, s512, [i for i in inputs if ad.comparison_group(i) == s512])
                  for r in b.records]
    pr = next(r["payload"] for r in plain_recs if r["record_type"] == "run" and r["payload"]["eval_id"] == "xstest")
    assert pr["run_id"] == r512["run_id"] and pr["token_cap_tokens"] == 512 and "token_cap_rung_index" not in pr and "judge_pool" not in pr["extra"]
    for mid in ("xst_refused", "over_refusal_graded"):
        pm = next(r["payload"] for r in plain_recs if r["record_type"] == "run_metric" and r["payload"]["metric_id"] == mid and r["payload"]["run_id"] == pr["run_id"])
        assert (pm["value"], pm["n"]) == (metric(s512, mid)["value"], metric(s512, mid)["n"])
    pj = next(r["payload"] for r in plain_recs if r["record_type"] == "judgment" and r["payload"]["run_id"] == pr["run_id"])
    assert pj["judgment_id"] == judgment(s512)["judgment_id"] and pj["api_pool"] == "lowprio" and pj.get("grading_code_sha") is None


def test_the_grades_are_read_from_one_copy_and_the_shards_from_every_local_mirror(tmp_path):
    """The durable copy carries the grades but not the transcripts, so the two roots are chosen by different rules: the FIRST
    existing grades copy (never two), and EVERY transcripts mirror that exists."""
    store = _store(tmp_path)
    absent = str(tmp_path / "nope")
    ad = AV.ArditiVendorMtAdapter(grades_roots=(absent, os.path.join(store, "mt_grades")),
                                  transcripts_roots=(absent, os.path.join(store, "transcripts_results")), seeds=_seeds(), replicate_roots=())
    assert ad.grades_root == os.path.join(store, "mt_grades")
    assert ad.transcripts_roots == (os.path.join(store, "transcripts_results"),)
    assert {ad.comparison_group(i) for i in ad.discover()} == {TAG}
    # no copy on this machine: no group, no row — never a refusal (the roots are machine-local)
    gone = AV.ArditiVendorMtAdapter(grades_roots=(absent,), transcripts_roots=(absent,), seeds=_seeds(), replicate_roots=())
    assert gone.grades_root == "" and gone.discover() == []
    # the live defaults name the durable copy first and a `<T>` pattern per replicate root shape
    assert AV.GRADES_ROOTS[0].startswith(os.path.join(os.path.expanduser("~"), "results")) and AV.TRANSCRIPTS_ROOTS[0].startswith(AV.DURABLE_ROOT)
    assert all(g.split(os.sep)[-1].endswith("<T>") or g.split(os.sep)[-2].endswith("<T>") for g, _ in AV.REPLICATE_ROOTS)
    with pytest.raises(ValueError, match="<T>"):
        AV.replicate_roots_by_trial(((os.path.join(absent, "no_trial_here"), None),))


# --------------------------------------------------------------------------------------------
# the object the rung serves: what the sidecars say and what the registry can mint today
# --------------------------------------------------------------------------------------------

PARENT = "f" * 64


def _arditi_rung_sidecars(sha=OBJ, parent=PARENT, scale=1.5):
    """The Arditi lane's staging of a DERIVED rung (scripts/arditi/make_arditi_allpos_objects.py, 2026-09-19), synthetic shas: the same
    `derivation` / `parent_object_ids` / `params.base_scale` record in all three sidecars, `fit_provenance.json` being the PARENT's fit
    record copied and re-stamped, `object.pt.provenance.json` carrying the kind, the layers and the `derived` block."""
    record = {"derivation": "dose_scale", "parent_object_ids": [parent], "params": {"base_scale": scale}, "kind_composed": "dose_scale"}
    return {
        "staged_from.json": dict(record, c=0.15, cell_id=AV.STEERED_OBJECT_KEY, dose_unit=DG.STRENGTH_UNIT, lane="arditi_allpos",
                                 object_sha256=sha, staged_id=f"{RUNG}__arditi_allpos", target=DG.MODEL_KEY, view=RUNG,
                                 source="DERIVED, not fitted: the c=0.1 rung's direction x1.5"),
        "fit_provenance.json": dict(record, c=0.1, cell_id=AV.STEERED_OBJECT_KEY, dose_unit=DG.STRENGTH_UNIT, object_sha256=sha,
                                    positions_at_fit="all", resolved_model_id="synthetic/instruct-4b", resolved_revision="c" * 40,
                                    staged_id="ar_c0p1__arditi_allpos", target=DG.MODEL_KEY, chat_template_hash="a4aee8afcf2e0711",
                                    derived_rung={"c": 0.15, "from": "ar_c0p1", "scale": scale, "fit_is_unchanged": "reuses the parent's fit"}),
        "object.pt.provenance.json": dict(record, kind=DG.OBJECT_KIND, c=0.15, dose_unit=DG.STRENGTH_UNIT, positions="all", fit_positions="all",
                                          layers=[12], stored_layer=12,
                                          derived={"how": "the c=0.1 rung's stored direction multiplied by 1.5", "parent_view": "ar_c0p1",
                                                   "parent_object_sha256": parent, "verified": {"cosine_to_parent": 1.0},
                                                   "built_at": "2026-09-19T23:35:41Z", "built_by": "synthetic"}),
    }


def test_the_staged_dose_scale_spelling_mints_the_derived_rung_on_its_parent(tmp_path):
    """The Arditi lane's dose-scale spelling is a dose stamp: the derived rung mints as a `vector_all` object under the label
    `arditi_allpos`, `fit_target_kind derived` / `derivation dose_scale` on its parent, `base_scale` the factor, its campaign the staging
    lane's (so it sits beside its fitted siblings' rows), and the payload validates. Before 2026-09-20 this spelling was no stamp
    (`fit_target_kind undecidable`) and the rung's object could not be minted."""
    from sisteer.registry import derive as D
    from sisteer.registry import records as R
    from sisteer.registry.adapters import fit_sidecars as FS
    import registry_fit_fixtures as FF
    sidecars = _arditi_rung_sidecars()
    assert D.dose_scale_derivation(sidecars)[0] == "staged_from.json"
    assert "derivation=dose_scale" in D._dose_stamp(sidecars)
    assert D.fit_target_kind(DG.OBJECT_KIND, sidecars, {})[0] == "derived"
    # the word alone is not a stamp: no parent list, no dose copy
    assert D._dose_stamp({"staged_from.json": {"derivation": "dose_scale"}}) is None
    d = tmp_path / f"{RUNG}__arditi_allpos"
    os.makedirs(d, exist_ok=True)
    for name, payload in sidecars.items():
        with open(d / name, "w") as fh:
            json.dump(payload, fh)
    bundle = FS.read_bundle(str(d), read_blob=False, bank="full22_objects", campaign_hint="sdfbase_full22", target_hint=DG.MODEL_KEY)
    p = FS.build_object_payload(bundle, FF.seeds(), NOW, blob_reads_enabled=False)
    assert p["object_id"] == OBJ and p["object_kind"] == DG.OBJECT_KIND and p["paper_label"] == DG.PAPER_LABEL
    assert p["object_role"] == DG.OBJECT_ROLE and p["campaign"] == "arditi_allpos" and p["campaign_cell_id_legacy"] == f"{RUNG}__arditi_allpos"
    assert p["fit_target_kind"] == "derived" and p["derivation"] == "dose_scale" and p["parent_object_ids_json"] == [PARENT]
    assert p["base_scale"] == 1.5 and p["derivation_params_json"]["base_scale"] == 1.5 and p["fit_objective"] == "not_applicable"
    assert p["fit_positions"] == "all" and p["fit_layers_json"] == [12] and p["fit_model_id"] == FF.synth.MODEL_ID
    assert p["derivation_params_json"]["parent_identity"]["parent_view"] == "ar_c0p1"
    assert "params.base_scale" in p["field_confidence_json"]["base_scale"]["source"]
    rec = R.make_record("object", p, F.producer(), "00000000-0000-5000-8000-000000000001", NOW)
    assert not isinstance(VAL.validate_record(rec, path="<test>"), VAL.Malformed)
