"""The paper notebook's item-clustered pooled intervals, as the landing gate sees them.

The intervals' own tests are paper/tests/test_registry_nb_pooled_ci.py (synthetic frames and a fixture unit provider) and the opt-in
paper/tests/test_registry_nb_pooled_ci_slow.py (the committed paper frame), both outside the suite's `testpaths`, so the gate runs
them only when a landing names them. This file sits in tests/ and pins what every pooled interval rests on, so a change that moves it
is caught by every gate run:
  * THE REAL ITEM READERS: `registry_nb.item_units` and the SQL behind it (`rule_item_rows`, the judges' item rows, the Petri judge's
    transcripts), run against a registry holding only the rows they read for three of Qwen's unsteered anchor pools, copied from the
    committed paper frame (paper/tests/data/pooled_ci_item_fixture.json; item ids as opaque tokens): GSM8K (rule-scored, both
    unscored-item conventions), Past-tense (StrongREJECT re-derived under minimax-m3 v1) and Petri self-preservation (glm-5.3). Every
    run's units give its reading (value and n) and each pool's se is the anchor se the committed render drew. A reader that stops
    finding its rows turns every pooled cell into the marked bound, which nothing else in the gate would see;
  * THE FIT-SEED POOLS (2026-11-21): four of Qwen's appendix seed pools from the same frame (Control LoRA s1-s3 on GSM8K and on
    Past-tense, Wilson-form rates; Difference ReLU s1-s3 on Petri self-preservation and on the pooled Petri score, stratified by
    behaviour, the seeds without an interval of their own), their seeds' and anchor runs' rows and item rows in the same fixture: every
    run's units give its reading, and the seed pool the real readers build (`registry_nb.seed_pool`: the seeds' runs item-clustered
    across the seeds, the between-fit variance, the anchor once) is the se the committed render drew; and on the Control LoRA seeds,
    the other seed-pool readers: a mean column over two evals is one pool (equal to the category path's mean mark), a transfer
    difference on the same anchor runs nets the anchor away, a padding mean pools the seeds by goal at the per-cell effective n;
  * the control-pool file's own rows and the generator file's rows: every payload's units (`judge_rules.payload_reading_units`, through
    `ctlstop_judging.units`) give the reading `ctlstop_judging.reduce` takes from it, value and n, at both StrongREJECT thresholds the
    appendix pins (a seed pool of those rows reads these units; a drifted rule would stop the render);
  * the sign-decomposition file's per-unit payloads (2026-11-22; data/appendix_geometry_evenodd.json, schema 2), through the unit sources
    its reader registers: every reading's units give its stored rate and n, all replies and the coherent recount, held by the same
    `row_units` a render holds them by, and every anchor's runs' units average to its value (a drifted unit path would stop the render);
  * the committed frame's manifest carries both item passes the readers read: the verbatim pass (every row of the in-paper runs, the
    rule-scored evals' rows among them) and the paper judges' item rows.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO / "paper" / "notebooks"
if str(NOTEBOOKS) not in sys.path:
    sys.path.append(str(NOTEBOOKS))
import matplotlib  # noqa: E402

matplotlib.use("Agg")

FIXTURE = REPO / "paper" / "tests" / "data" / "pooled_ci_item_fixture.json"
SNAPSHOT = REPO / "paper" / "snapshot"
#: the paper's judges whose item rows the frame must carry: minimax-m3 StrongREJECT v1 and glm-5.3 on Petri
PAPER_ITEM_JUDGES = {"192df392a676e6b046e86c08ebb96484d31cad7573736d10e7c5351a5128605c",
                     "d4dc693904dcf80f559f07cfd861d80315b85af67714c157f7fc56b17bad6043"}


def _column(spec):
    if isinstance(spec, dict) and "const" in spec:
        return [spec["const"]] * spec["n"]
    if isinstance(spec, dict):
        return [None if c is None else spec["levels"][c] for c in spec["codes"]]
    return spec


@pytest.fixture
def frame_items(tmp_path, monkeypatch):
    """The fixture's rows as a registry the notebook reads as its live source, with no query cache, through the same SQL and the same
    readers a render runs. `registry_nb` is imported FRESH from this checkout (another module may have left a mirror copy in
    sys.modules), with no judge-comparison switch set, and the previous module is put back afterwards."""
    import importlib
    for k in ("REGISTRY_NB_JUDGES", "REGISTRY_NB_SR_THRESHOLD", "REGISTRY_NB_PETRI_JUDGE", "REGISTRY_NB_PETRI_EXCLUDE", "REGISTRY_NB_FIGS",
              "REGISTRY_NB_STRENGTH_READS", "REGISTRY_NB_SNAPSHOT_ROOT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("REGISTRY_NB_SOURCE", "snapshot")                 # resolved at import only; the readers below read the fixture
    monkeypatch.syspath_prepend(str(NOTEBOOKS))
    saved_mods = {m: sys.modules.pop(m, None) for m in ("registry_nb", "pooled_ci", "judge_rules")}
    R = importlib.import_module("registry_nb")
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    db = tmp_path / "registry.sqlite"
    con = sqlite3.connect(db)
    for name, cols in data["tables"].items():
        t = pd.DataFrame({c: _column(v) for c, v in cols.items()})
        if name == "item_result":
            t["record_sha256"] = [bytes.fromhex(h) for h in t["record_sha256"]]
            t["withdrawn_by_record_sha256"] = None
            t["extra_json"] = None
        t.to_sql(name, con, index=False)
    con.execute("CREATE TABLE record_ingest (ingest_seq INTEGER)")
    con.execute("INSERT INTO record_ingest VALUES (?)", (data["head"],))
    con.commit()
    con.close()
    monkeypatch.setattr(R, "SOURCE", "live")
    monkeypatch.setattr(R, "DB", str(db))
    monkeypatch.setattr(R, "CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(R, "PETRI_EXCLUDE", "")
    monkeypatch.setenv("NB_CACHE", "0")
    R.reset_pooled_ci(None)
    yield R, data
    for m, mod in saved_mods.items():
        if mod is None:
            sys.modules.pop(m, None)
        else:
            sys.modules[m] = mod


def test_the_item_readers_reproduce_the_committed_frames_readings_and_anchor_intervals(frame_items):
    R, data = frame_items
    got = {}
    for p in data["pools"]:
        for r in p["rows"]:
            u = R.item_units(pd.Series(r))
            assert u.value == pytest.approx(r["value"], abs=1e-9) and u.n == r["n"], (p["eval_id"], r["run_id"])
        got[p["eval_id"]] = R.anchor_pool_se(pd.DataFrame(p["rows"]))
        assert got[p["eval_id"]].se == pytest.approx(p["anchor_se"], rel=1e-9), p["eval_id"]
        assert got[p["eval_id"]].method.startswith("item-clustered")
    assert len({r["n"] for r in data["pools"][0]["rows"]}) > 1          # both unscored-item conventions are in the GSM8K pool
    assert not R.POOLED_CI_REPORT["bound"] and R.POOLED_CI_REPORT["pools"] == 3
    assert got["gsm8k"].se == pytest.approx(1.2840, abs=5e-5) and got["petri_self_preservation"].se == pytest.approx(0.1407, abs=5e-5)


#: (object, eval) -> the seed pool's se the committed render drew (pp; the Petri pools in score points)
SEED_POOL_SE = {("control_lora", "gsm8k"): 1.8367, ("control_lora", "pasttense"): 2.7713,
                ("difference_relu", "petri_self_preservation"): 0.3135, ("difference_relu", "petri_pooled"): 0.4048}


def _members(R, p) -> list:
    """The fixture pool's seeds as the seed pool's members (`registry_nb.SeedMember`), each seed's run and anchor runs registered as a
    cell registers them; every run's units are read through the real readers and held to its reading."""
    members = []
    for s in p["seeds"]:
        row = pd.Series(s["row"])
        u = R.item_units(row)
        assert u.value == pytest.approx(row["value"], abs=1e-9) and u.n == row["n"], (p["eval_id"], row["run_id"])
        arows = [pd.Series(a) for a in s["anchor"]]
        for a in arows:
            R._UNIT_ROWS[R.unit_key(a)] = a
        steer = R._steer_weights([row])                                     # registers the seed's run as a cell does
        anchor = {R.unit_key(a): 1.0 / len(arows) for a in arows}
        members.append(R.SeedMember(R.fit_key(row), s["d_y"], ((p["eval_id"], tuple(sorted(steer.items()))),),
                                    ((p["eval_id"], tuple(sorted(anchor.items()))),), np.nan))
    return members


def test_the_item_readers_give_the_committed_frames_seed_pools(frame_items):
    R, data = frame_items
    assert len(data["seed_pools"]) == len(SEED_POOL_SE)
    for p in data["seed_pools"]:
        members = _members(R, p)
        res = R.seed_pool(members)
        assert res.method.startswith(R.SEED_CLUSTERED), (p["eval_id"], res.method)
        assert res.se == pytest.approx(p["se"], rel=1e-9), (p["object_label"], p["eval_id"])
        assert res.se == pytest.approx(SEED_POOL_SE[(p["object_label"], p["eval_id"])], abs=1e-4)
        assert float(np.mean([s["d_y"] for s in p["seeds"]])) == pytest.approx(p["d_y"], abs=1e-12)
    assert not R.POOLED_CI_REPORT["bound"] and R.POOLED_CI_REPORT["seed_pools"] == len(SEED_POOL_SE)
    assert not any(s["own_interval"] for s in data["seed_pools"][2]["seeds"])            # Petri seeds read through their transcripts


#: the pools below on the committed frame's Control LoRA seeds (pp): the mean column over GSM8K and Past tense, the transfer difference of
#: the Past-tense seeds against a one-fit lane on their anchor runs, and a padding mean of the Past-tense seeds against one anchor run
MEAN_COLUMN_SE, TRANSFER_SE, PADDING_BOUNDS = 1.5382, 2.7846, (0.1803, 12.1713)


def test_a_mean_column_a_transfer_difference_and_a_padding_mean_on_the_committed_frames_seeds(frame_items):
    """The other seed-pool readers on real units, where the gate runs (their synthetic tests are paper/tests/test_registry_nb_seed_ci.py,
    which a landing names in its focused run): a panel's MEAN COLUMN over two evals is one seed pool over the same fits, equal to the
    category path's mean mark, not the per-eval pools' intervals combined; a TRANSFER difference whose lanes use the same anchor runs
    nets the anchor to nothing, hypot of the lanes' steered parts; a PADDING mean is the seeds item-clustered by goal with the between-fit
    variance, Wilson at the per-cell effective n, combined with the anchor by Newcombe's hybrid."""
    R, data = frame_items
    P = R._pooled_ci()
    pools = {(p["object_label"], p["eval_id"]): p for p in data["seed_pools"]}
    mg, mt = _members(R, pools[("control_lora", "gsm8k")]), _members(R, pools[("control_lora", "pasttense")])
    se_g, se_t = R.seed_pool(mg).se, R.seed_pool(mt).se
    # (1) the mean column over the two evals, through the appendix's reader, against the category means' mean mark
    pts = pd.DataFrame([dict(model_id=R.MODEL_ID, object_label="control_lora", eval_id=e, y=float(np.mean([m.y for m in ms])),
                             lo=float(np.mean([m.y for m in ms])) - 1.96 * se, hi=float(np.mean([m.y for m in ms])) + 1.96 * se,
                             seed_parts=tuple(ms), n_pooled=3) for e, ms, se in (("gsm8k", mg, se_g), ("pasttense", mt, se_t))])
    col = R.eval_mean_column(pts, ["gsm8k", "pasttense"]).iloc[0]
    half = lambda terms: tuple((e, tuple((k_, w_ / 2) for k_, w_ in kw)) for e, kw in terms)
    cat = R.seed_pool_se([(a.y + b.y) / 2 for a, b in zip(mg, mt)], [np.nan] * 3, [()] * 3,
                         [half(a.steer + b.steer) for a, b in zip(mg, mt)], [half(a.anchor + b.anchor) for a, b in zip(mg, mt)])
    se_col = float((col.hi - col.lo) / 3.92)
    assert se_col == pytest.approx(cat.se, rel=1e-12) and col.se_method.startswith(R.SEED_CLUSTERED)
    assert se_col != pytest.approx(np.hypot(se_g, se_t) / 2, rel=1e-3)          # not the per-eval pools combined
    assert se_col == pytest.approx(MEAN_COLUMN_SE, abs=1e-4)
    # (2) a transfer difference: the Past-tense seeds against a one-fit lane (seed 1's run) on the same anchor runs, weighted alike
    it = R.SeedMember("it", mt[0].y, mt[0].steer, mt[0].anchor, np.nan)
    rb = pd.Series(dict(model_id=R.MODEL_ID, object_label="control_lora", metric_id="asr", d_lo=-1.0, d_hi=1.0))
    se, how = R.transfer_se([tuple(mt)], [(it,)], [("pasttense", rb, rb)])
    lane = R.seed_pool(mt, anchor=False)
    s1 = pd.Series(pools[("control_lora", "pasttense")]["seeds"][0]["row"])
    assert how == R.TRANSFER_NETTED and lane.v_anchor == 0.0
    assert se == pytest.approx(np.hypot(np.sqrt(lane.v_items + lane.v_fits), R.own_se(s1)), rel=1e-12)
    assert se < np.hypot(se_t, np.hypot(R.own_se(s1), R.combo_se(dict(mt[0].anchor[0][1]))))   # the anchor counted twice, before
    assert se == pytest.approx(TRANSFER_SE, abs=1e-4)
    # (3) a padding mean of the Past-tense seeds against one of their anchor runs
    seeds = [pd.Series(x["row"]) for x in pools[("control_lora", "pasttense")]["seeds"]]
    a0 = pools[("control_lora", "pasttense")]["seeds"][0]["anchor"][0]
    k0, n0 = int(a0["k"]), int(a0["n"])
    rows = pd.DataFrame([dict(object_label="control_lora", seed=r["seed"], pad=0.0, k1=int(r["k"]), n1=int(r["n"]), k0=k0, n0=n0,
                              delta=100.0 * (k0 / n0 - r["k"] / r["n"]), steer_terms={R.unit_key(r): 1.0}) for r in seeds])
    m = R.padding_pooled(rows).iloc[0]
    us = [R.item_units(r) for r in seeds]
    ps = np.array([u.value for u in us])
    v1 = P.cluster_var(us, [1 / 3] * 3) + P.fit_var(ps)
    n_eff = float(np.mean([u.n * P.cells_var(u) for u in us])) / v1
    p1 = rows.k1.sum() / rows.n1.sum()
    l1, u1 = R.wilson(p1 * n_eff, n_eff)
    l0, u0 = R.wilson(k0, n0)
    lo, hi = P.newcombe_bounds(k0 / n0, l0, u0, p1, l1, u1)
    assert (m.lo, m.hi) == (pytest.approx(100 * lo, rel=1e-12), pytest.approx(100 * hi, rel=1e-12)) and m.se_method.startswith("seed pool")
    assert (m.lo, m.hi) == (pytest.approx(PADDING_BOUNDS[0], abs=1e-4), pytest.approx(PADDING_BOUNDS[1], abs=1e-4))
    assert m.delta == pytest.approx(100.0 * (k0 / n0 - p1), abs=1e-12)
    assert not R.POOLED_CI_REPORT["bound"] and R.POOLED_CI_REPORT["seed_overlap_min"] == 1.0


@pytest.mark.parametrize("path,want", [("qwen_ctlstop_rows.json", 194), ("appendix_generator_rows.json", 80)])
def test_every_file_payloads_units_give_its_reading(frame_items, path, want):
    R, _data = frame_items
    f = json.loads((NOTEBOOKS / "data" / path).read_text(encoding="utf-8"))
    checked = set()
    for r in f["rows"]:
        entry = f["items"].get(r["run_id"])
        if entry is None:
            continue                                                        # a reading not yet in the file (pending)
        fam = R.CTLSTOP_EVALS[r["eval_id"]]["family"]
        for t in ((0.5, 0.75) if fam == "strongreject" else (None,)):
            red = R.ctlstop_judging.reduce(fam, entry, t)
            if not red.n:
                continue
            ru = R.ctlstop_judging.units(fam, entry, t)
            assert float(ru.b.sum()) == red.n and float(ru.a.sum() / ru.b.sum()) == pytest.approx(red.value, abs=1e-12), (path, r["run_id"], t)
        checked.add(r["run_id"])
    assert len(checked) == want


#: the sign-decomposition file's runs with a payload (readings and anchor runs) and those with a coherent reading
EVENODD_RUNS, EVENODD_COHERENT_RUNS = 151, 151


def test_the_sign_decomposition_files_payload_units_give_every_runs_reading(frame_items):
    """The sign decomposition's payloads, where the gate runs (their contract tests are paper/tests/test_appendix_geometry_evenodd.py):
    the reader (appendix_geometry_evenodd, imported fresh against this `registry_nb` and put back afterwards) registers its unit sources,
    and each run's units, read through `registry_nb.row_units` as a render reads them, give its reading: a reading's stored rate and n
    in both modes (all replies: every unit of its item set, an ungraded goal not attacked; coherent: the graded units without a degenerate
    reply; UnitsMismatch otherwise), an anchor's runs their anchor's value on average and, all replies, the whole item set each."""
    R, _data = frame_items
    saved = {m: sys.modules.pop(m, None) for m in ("appendix_geometry_evenodd", "appendix_common")}
    try:
        import appendix_geometry_evenodd as GE
        assert GE.R is R and set(GE.UNIT_SOURCE.values()) <= set(R.UNIT_SOURCES)
        f = GE.load_geometry_evenodd()                                       # the file's contract, its payloads' own recount included
        assert GE.register_payloads(f) == len(f["payloads"])
        checked = {"all": set(), "coherent": set()}
        for r in f["rows"]:
            for mode, rate, n in (("all", r["rate"], r["n_units"] + r["n_ungraded"]), ("coherent", r["rate_coherent"], r["n_coherent"])):
                if rate is None:
                    continue
                u = R.row_units(GE._ROW_OF[(r["run_id"], mode)])
                assert 100.0 * u.value == pytest.approx(rate, abs=1e-9) and u.n == n, (r["run_id"], mode)
                checked[mode].add(r["run_id"])
        for k, a in f["anchors"].items():
            for mode, used, val in (("all", a["runs_used"], a["value"]), ("coherent", a["runs_used_coherent"], a["value_coherent"])):
                us = [R.row_units(GE._ROW_OF[(x, mode)]) for x in used]
                assert 100.0 * float(np.mean([u.value for u in us])) == pytest.approx(val, abs=1e-9), (k, mode)
                assert mode == "coherent" or {u.n for u in us} == {f["item_sets"][a["eval"]]["n_items"]}, k
                checked[mode] |= set(used)
        assert len(checked["all"]) == len(f["payloads"]) == EVENODD_RUNS and len(checked["coherent"]) == EVENODD_COHERENT_RUNS
        assert not R.POOLED_CI_REPORT["bound"]
    finally:
        for m, mod in saved.items():
            if mod is None:
                sys.modules.pop(m, None)
            else:
                sys.modules[m] = mod


def test_the_committed_frame_ships_the_item_rows_the_pooled_intervals_read():
    cur = (SNAPSHOT / "CURRENT").read_text(encoding="utf-8").strip()
    m = json.loads((SNAPSHOT / "snapshots" / cur / "MANIFEST.json").read_text(encoding="utf-8"))
    assert m["item_result_passes"]["verbatim"] > 0 and m["item_result_passes"]["item_judges"] > 0
    assert PAPER_ITEM_JUDGES <= set(m["item_projection"]["judges"])
