"""The over-refusal per-item source (`overrefusal_items`) on SYNTHETIC target dirs: the relocated
reader's rows (graded score, at-cap share, paired bootstrap vs the unsteered cell) under the pinned computation id,
the producer's paired verdict as run_contrast rows, the newest-wins rule across roots, determinism of the shared-RNG
traversal, and the end-to-end pass."""
from __future__ import annotations

import os

import numpy as np
import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry.adapters import _items_common as IC
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import overrefusal_items as OR
from sisteer.registry.adapters import runner as RUN

import registry_items_fixtures as F

NOW = "2026-01-05T00:00:00.000Z"


def _ctx(conn=None):
    return AB.AdapterContext(root="/nonexistent", now=NOW, producer={"script": "tests", "code_sha": "unverified", "session": None, "host": "TESTHOST", "substrate": "grading_host"}, conn=conn)


def _view(tmp_path, root="root", labels_cell=("full_refusal", "full_refusal", "full_compliance", "partial_refusal")):
    view = str(tmp_path / root / "synthmodel")
    none = F.write_controls_store(view, "none", ihash8="2ff9cdff", object_sha16=None, n=4, labels=["full_compliance", "full_compliance", "full_refusal", "partial_refusal"])
    cell = F.write_controls_store(view, F.LORA_CELL, n=4, labels=list(labels_cell), at_cap_ids=["orb0003-cccccccc"])
    F.write_paired_vs_none(view, os.path.basename(none), [os.path.basename(cell)])
    return view, none, cell


def test_relocated_graded_matches_the_pinned_estimand(tmp_path):
    view, none, cell = _view(tmp_path)
    files = OR.reader_files([str(tmp_path / "root")])
    assert set(files) == {("synthmodel", "none"), ("synthmodel", F.LORA_CELL)}
    stats = OR.relocated_graded(files)
    st = stats[("synthmodel", F.LORA_CELL, "or_bench_hard_1k")]
    # cell scores: FR 1, FR 1, FC 0, PR 0.5 -> but item 3 is at cap -> 0.5 ; mean = (1+1+0+0.5)/4
    assert st["value"] == pytest.approx((1 + 1 + 0 + 0.5) / 4) and st["n"] == 4 and st["at_cap_share"] == pytest.approx(0.25)
    base = [0.0, 0.0, 1.0, 0.5]
    d = np.array([1, 1, 0, 0.5]) - np.array(base)
    assert st["delta"] == pytest.approx(100 * d.mean()) and st["n_pairs"] == 4
    assert st["delta_lo"] <= st["delta"] <= st["delta_hi"]
    assert "delta" not in stats[("synthmodel", "none", "or_bench_hard_1k")]
    # the traversal is deterministic (one rng, seed 0, dict order)
    again = OR.relocated_graded(OR.reader_files([str(tmp_path / "root")]))
    assert again[("synthmodel", F.LORA_CELL, "xstest")]["delta_lo"] == stats[("synthmodel", F.LORA_CELL, "xstest")]["delta_lo"]


def test_newest_judge_items_wins_across_roots(tmp_path):
    view1, _, cell1 = _view(tmp_path, "root1")
    view2, _, cell2 = _view(tmp_path, "root2")
    old = os.path.join(cell1, "judge_items.jsonl"); new = os.path.join(cell2, "judge_items.jsonl")
    os.utime(old, (1_700_000_000, 1_700_000_000)); os.utime(new, (1_800_000_000, 1_800_000_000))
    files = OR.reader_files([str(tmp_path / "root1"), str(tmp_path / "root2")])
    assert files[("synthmodel", F.LORA_CELL)] == new


def test_adapter_emits_relocated_rows_and_producer_contrasts(tmp_path):
    view, none, cell = _view(tmp_path)
    ad = OR.OverrefusalItemsAdapter(roots=[str(tmp_path / "root")], dims=F.dims_map(), campaign="synthcampaign")
    inputs = ad.discover()
    groups = {}
    for i in inputs:
        groups.setdefault(ad.comparison_group(i), []).append(i)
    assert set(groups) == {view, none, cell}
    recs = {}
    for g, ins in groups.items():
        for b in ad.read(_ctx(), g, ins):
            for r in b.records:
                recs.setdefault(r["record_type"], []).append(r["payload"])
    comp = OR.relocated_computation_id()
    metrics = [p for p in recs["run_metric"] if p.get("computation_id") == comp]
    assert len(metrics) == 4  # 2 stores x 2 controls
    # the producer key is `<target dir>/<store key>` (StoreReader(run_key_scope="target_dir")), never the store key alone
    m = next(p for p in metrics if p["run_id"] == I.run_id("synthcampaign", f"synthmodel/{os.path.basename(cell)}", f"{F.LORA_CELL.replace('__', '--')}@{F.IHASH8}", "or_bench_hard",
                                                             F.dims_map().instrument("overrefusal_greedy_512_controls", "or_bench_hard", F.MODEL_IT)[0]))
    assert all(r["producer_run_key_legacy"] == f"synthmodel/{os.path.basename(r['store_uri'])}" for r in recs["run"])
    assert m["metric_id"] == "over_refusal_graded" and m["value"] == pytest.approx(0.625) and m["frac_at_token_cap"] == pytest.approx(0.25) and m["computed_by"] == "producer"
    assert m["extra"]["at_cap_estimand"] == IC.AT_CAP_RETOKENIZED      # the reader's at-cap share is the judge driver's re-tokenized flag
    contrasts = [p for p in recs["run_contrast"] if p["computation_id"] == comp]
    assert len(contrasts) == 2 and all(p["anchor_kind"] == "same_standup_none" and p["delta_method"] == "paired_bootstrap" and p["n_pairs"] == 4 for p in contrasts)
    producer = [p for p in recs["run_contrast"] if p["computation_id"] == OR.producer_paired_computation_id()]
    assert len(producer) == 2 and {p["delta_method"] for p in producer} == {"exact_mcnemar"}
    pc = next(p for p in producer if p["metric_id"] == "orb_refused")
    assert pc["delta"] == pytest.approx(-0.2) and pc["delta_lo"] == pytest.approx(-0.5) and pc["delta_p"] == pytest.approx(0.25) and pc["n_pairs"] == 3
    assert {p["algorithm"] for p in recs["computation"]} == {OR.RELOCATED_ALGORITHM, OR.PRODUCER_PAIRED_ALGORITHM}
    assert ad.rng_scope == "reader_traversal"
    # the runs the pinned reader files (its two benign controls) carry its source — `run.lane_source_legacy = overrefusal_items` — so the
    # paper frame spells them (paper.JUDGE_SPELLING_LEGACY / SERVED_KEY_DOSE_SOURCES key on it; a NULL source matched no pin key)
    filed = [r for r in recs["run"] if r["eval_id"] in ("xstest", "or_bench_hard")]
    assert filed and all(r["lane_source_legacy"] == OR.ADAPTER_NAME for r in filed)
    assert all(r["lane_source_legacy"] is None for r in recs["run"] if r["eval_id"] not in ("xstest", "or_bench_hard"))


def test_paired_file_naming_an_absent_none_store_refuses(tmp_path):
    view = str(tmp_path / "root" / "synthmodel")
    cell = F.write_controls_store(view, F.LORA_CELL, n=2)
    F.write_paired_vs_none(view, "none__controls__t0__a0__deadbeef__deadbeef", [os.path.basename(cell)])
    ad = OR.OverrefusalItemsAdapter(roots=[str(tmp_path / "root")], dims=F.dims_map(), campaign="synthcampaign")
    inputs = [i for i in ad.discover() if ad.comparison_group(i) == view]
    with pytest.raises(AB.AdapterRefusal, match="none_cell"):
        list(ad.read(_ctx(), view, inputs))


def test_end_to_end_pass_lands_side_by_side_free_rows(tmp_path):
    root = str(tmp_path / "reg")
    F.ingest_dimensions(root)
    _view(tmp_path, "stores")
    ad = OR.OverrefusalItemsAdapter(roots=[str(tmp_path / "stores")], campaign="synthcampaign")
    conn = DB.connect(DB.db_path(root), readonly=True)
    rep = RUN.run_pass(root, [ad], conn=conn, now=NOW)
    conn.close()
    assert rep.groups_refused == 0, [r.error for r in rep.results]
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0, ing.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM run_contrast").fetchone()[0] == 4
    assert conn.execute("SELECT COUNT(*) FROM run_metric WHERE metric_id = 'over_refusal_graded'").fetchone()[0] == 4
    assert conn.execute("SELECT COUNT(*) FROM computation WHERE algorithm LIKE 'paperlib.%'").fetchone()[0] == 1
    assert [tuple(r) for r in conn.execute("SELECT DISTINCT priority FROM source_revision WHERE adapter = ? AND accepted = 1", (ad.name,)).fetchall()] == [(IC.PRIORITY_PAPER_SOURCE,)]
    assert IC.PRIORITY_PAPER_SOURCE > IC.PRIORITY_GENERIC_READER > 0 > IC.PRIORITY_MIRROR
    assert conn.execute("SELECT COUNT(*) FROM v_item_conflict").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0
    conn.close()


def test_unsteered_store_repeated_under_several_target_dirs_is_one_run_per_target_dir(tmp_path):
    """b2 (2026-09-19): one model's unsteered store is served again under its `_halfdose` (and `_it`) target dir under the SAME store key
    — `none__controls__t0__a0__2ff9cdff__<harness8>`: no object, so the intervention hash is constant, and the harness hash names
    neither the dose nor the served model. Keyed by the bare store key the stores minted ONE run per eval and the fold's last writer
    mixed their fields (`n_items_judged` from one store, `store_uri` from another). The run is keyed `<target dir>/<store key>`, so
    each store's control is its own run, and every steered run's contrasts — the relocated paired bootstrap and the producer's
    paired verdict — anchor on the control of its OWN target dir."""
    root = str(tmp_path / "root")
    base, base_none, base_cell = _view(tmp_path)                                    # root/synthmodel: none + the lora cell
    half = os.path.join(root, "synthmodel_halfdose")                                # the same none store key, a different dose behind it
    half_none = F.write_controls_store(half, "none", ihash8="2ff9cdff", object_sha16=None, n=4, labels=["full_refusal", "full_refusal", "full_refusal", "partial_refusal"])
    half_cell = F.write_controls_store(half, F.LORA_CELL, ihash8="a72ddf98", n=4, labels=["full_compliance", "full_compliance", "full_compliance", "partial_refusal"])
    F.write_paired_vs_none(half, os.path.basename(half_none), [os.path.basename(half_cell)])
    assert os.path.basename(base_none) == os.path.basename(half_none)               # the collision the bare key made
    ad = OR.OverrefusalItemsAdapter(roots=[root], dims=F.dims_map(), campaign="synthcampaign")
    groups = {}
    for i in ad.discover():
        groups.setdefault(ad.comparison_group(i), []).append(i)
    recs = {}
    for g, ins in groups.items():
        for b in ad.read(_ctx(), g, ins):
            for r in b.records:
                recs.setdefault(r["record_type"], []).append(r["payload"])
    runs = recs["run"]
    inst = F.dims_map().instrument("overrefusal_greedy_512_controls", "or_bench_hard", F.MODEL_IT)[0]
    none_key = os.path.basename(base_none)
    by_dir = {d: [r for r in runs if r["eval_id"] == "or_bench_hard" and r["campaign_cell_id_legacy"] == "none" and r["view_legacy"] == d]
              for d in ("synthmodel", "synthmodel_halfdose")}
    assert all(len(v) == 1 for v in by_dir.values()), {d: len(v) for d, v in by_dir.items()}
    base_ctl, half_ctl = by_dir["synthmodel"][0], by_dir["synthmodel_halfdose"][0]
    assert base_ctl["run_id"] != half_ctl["run_id"]
    assert base_ctl["run_id"] == I.run_id("synthcampaign", f"synthmodel/{none_key}", "none@2ff9cdff", "or_bench_hard", inst)
    assert half_ctl["run_id"] == I.run_id("synthcampaign", f"synthmodel_halfdose/{none_key}", "none@2ff9cdff", "or_bench_hard", inst)
    assert base_ctl["producer_run_key_legacy"] == f"synthmodel/{none_key}" and half_ctl["producer_run_key_legacy"] == f"synthmodel_halfdose/{none_key}"
    assert base_ctl["store_uri"] == base_none and half_ctl["store_uri"] == half_none
    # the run table's UNIQUE (producer_run_key_legacy, producer_object_key_legacy, eval_id, instrument_id) admits both
    assert len({(r["producer_run_key_legacy"], r["producer_object_key_legacy"], r["eval_id"], r["instrument_id"]) for r in runs}) == len(runs)
    # every steered run's contrasts anchor on the control of its own target dir, under both computations
    steered = {r["view_legacy"]: r["run_id"] for r in runs if r["eval_id"] == "or_bench_hard" and r["campaign_cell_id_legacy"] == F.LORA_CELL}
    anchors = {}
    for c in recs["run_contrast"]:
        if c["metric_id"] in ("over_refusal_graded", "orb_refused") and c["run_id"] in steered.values():
            anchors.setdefault(c["run_id"], set()).add(c["anchor_run_id"])
    assert anchors[steered["synthmodel"]] == {base_ctl["run_id"]} and anchors[steered["synthmodel_halfdose"]] == {half_ctl["run_id"]}
    assert all(c["anchor_kind"] == "same_standup_none" for c in recs["run_contrast"] if c["run_id"] in steered.values())
    # the relocated reader's own paired delta per target dir is what the contrast carries (each cell against ITS dir's control)
    comp = OR.relocated_computation_id()
    rel = {c["run_id"]: c for c in recs["run_contrast"] if c["computation_id"] == comp and c["metric_id"] == "over_refusal_graded"}
    stats = ad.relocated()
    assert rel[steered["synthmodel"]]["delta"] == pytest.approx(stats[("synthmodel", F.LORA_CELL, "or_bench_hard_1k")]["delta"])
    assert rel[steered["synthmodel_halfdose"]]["delta"] == pytest.approx(stats[("synthmodel_halfdose", F.LORA_CELL, "or_bench_hard_1k")]["delta"])
    assert rel[steered["synthmodel"]]["delta"] != rel[steered["synthmodel_halfdose"]]["delta"]


def test_fitted_runs_carry_the_target_directory_s_dose_and_the_operating_point(tmp_path):
    """The per-item over-refusal source's fitted runs had `strength_nominal` / `is_operating_point` NULL on every target directory (2026-09-20).
    Through the shared store reader a fitted object's run now files the dose its target directory spells — the base directory the
    campaign nominal 1.0 (the operating point), `_halfdose` 0.5 (not) — and the unsteered runs of both directories file no strength."""
    root = str(tmp_path / "root")
    _view(tmp_path)                                                                 # root/synthmodel: none + the lora cell
    half = os.path.join(root, "synthmodel_halfdose")
    half_none = F.write_controls_store(half, "none", ihash8="2ff9cdff", object_sha16=None, n=4, labels=["full_refusal", "full_refusal", "full_refusal", "partial_refusal"])
    half_cell = F.write_controls_store(half, F.LORA_CELL, ihash8="a72ddf98", n=4, labels=["full_compliance", "full_compliance", "full_compliance", "partial_refusal"])
    F.write_paired_vs_none(half, os.path.basename(half_none), [os.path.basename(half_cell)])
    ad = OR.OverrefusalItemsAdapter(roots=[root], dims=F.dims_map(), campaign="synthcampaign")
    groups = {}
    for i in ad.discover():
        groups.setdefault(ad.comparison_group(i), []).append(i)
    runs = []
    for g, ins in groups.items():
        for b in ad.read(_ctx(), g, ins):
            runs += [r["payload"] for r in b.records if r["record_type"] == "run"]
    lora = {p["view_legacy"]: p for p in runs if p["campaign_cell_id_legacy"] == F.LORA_CELL and p["eval_id"] == "or_bench_hard"}
    assert set(lora) == {"synthmodel", "synthmodel_halfdose"}
    assert (lora["synthmodel"]["strength_nominal"], lora["synthmodel"]["is_operating_point"]) == (1.0, 1)
    assert (lora["synthmodel_halfdose"]["strength_nominal"], lora["synthmodel_halfdose"]["is_operating_point"]) == (0.5, 0)
    assert all(p["strength_nominal_source"] == "derived" and p["strength_unit"] == "nominal_only" for p in lora.values())
    # every control of the store, the gsm8k one the paper reads from no source included, carries the same dose
    assert {(p["view_legacy"], p["strength_nominal"]) for p in runs if p["campaign_cell_id_legacy"] == F.LORA_CELL} == {("synthmodel", 1.0), ("synthmodel_halfdose", 0.5)}
    none = [p for p in runs if p["campaign_cell_id_legacy"] == "none"]
    assert len(none) == 8 and all("strength_nominal" not in p and "is_operating_point" not in p for p in none)
