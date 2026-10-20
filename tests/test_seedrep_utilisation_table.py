"""The Qwen seed-replicate native battery's utilisation table (the U bars) on synthetic samplers: the three
bars read from `vllm_metrics.jsonl` × `driver_inflight.jsonl` × `gpu_dmon.log` against S and the S5 reference, each
`true` / `false` / `unmeasured(<why>)`; H200-h actual vs sized; the kernel-ceiling haircut; every missing input refuses.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "scripts" / "sdfbase" / "seedrep", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import audit_common as ac  # noqa: E402
import seedrep_audit_fixtures as fx  # noqa: E402
import utilisation_table as ut  # noqa: E402


def _root(tmp_path: Path, leg="pasttense", **samplers):
    root = fx.make_root(tmp_path / "stores", leg=leg, view="s1", arms=("none",))
    fx.populate_root(root, s=1, arms=("none",), n_items=1)
    if samplers:
        root.write_samplers(**samplers)
    return root, ac.read_attempt_root(root.path)


def _row(ar, *, s5=None, expected=None, period=10.0):
    return ut.utilisation_row(ar, s5=s5 if s5 is not None else fx.s5_references(), expected=expected or {"pasttense": 3.0}, period_s=period)


def test_bars_met_and_every_column_measured(tmp_path: Path):
    root, ar = _root(tmp_path)  # defaults: Running S, in-flight S, SM 92, 50 tok/s/seq
    r = _row(ar)
    assert r["S"] == 45 and r["L"] == 11520 and r["window"] == "11520x45" and r["thinking"] == "off"
    assert r["n_scrapes"] == 12 and r["n_decode_scrapes"] == 11 and r["n_saturated_scrapes"] == 11
    assert r["median_running_saturated"] == 45.0 and r["running_bar"] == "true"
    assert r["median_sm_decode"] == 92.0 and r["sm_bar"] == "true"
    assert r["tok_s_median"] == 2250.0 and r["tok_s_mean"] == 2250.0 and r["s5_tok_s"] == 2000.0 and r["tok_s_ratio_s5"] == 1.125
    assert r["tok_s_aggregate"] == 2250.0 and r["n_eligible_intervals"] == 10 and r["sampling_coverage"] == 1.0
    assert r["tok_bar"] == "true" and r["hours_bar"] == "true" and r["bars_met"] is True
    assert r["h200_h_actual"] == 2.0 and r["h200_h_expected"] == 3.0 and r["h200_h_ratio"] == 0.667
    assert r["kernel_from_S"] == 56 and r["kernel_to_S"] == 45 and r["kernel_haircut"] == 0.196
    assert r["dmon_ts_basis"] == "dmon:-o DT" and r["problems"] == []
    assert set(r) == {c.rstrip("?") for c in ut.COLUMNS}


def test_running_bar_false_when_engine_side_short(tmp_path: Path):
    root, ar = _root(tmp_path, running=20)  # in-flight S but Running 20 < 0.8 S → engine-side cause (§8 re-size rule)
    r = _row(ar)
    assert r["median_running_saturated"] == 20.0 and r["running_bar"] == "false" and r["bars_met"] is False


def test_running_bar_unmeasured_when_client_never_saturates(tmp_path: Path):
    root, ar = _root(tmp_path, inflight=10)
    r = _row(ar)
    assert r["running_bar"] == "unmeasured(in-flight never >= S)" and r["median_running_saturated"] == "unmeasured"
    assert r["n_saturated_scrapes"] == 0 and r["bars_met"] is False
    assert any("never reached S=45" in p for p in r["problems"])


def test_sm_and_tok_bars_false(tmp_path: Path):
    root, ar = _root(tmp_path, sm=70)
    r = _row(ar)
    assert r["median_sm_decode"] == 70.0 and r["sm_bar"] == "false"
    r2 = _row(ar, s5=fx.s5_references(tok_s=10000.0))
    assert r2["tok_s_ratio_s5"] == 0.225 and r2["tok_bar"] == "false" and r2["bars_met"] is False


def test_s5_reference_schema_rejects_scalars_percentiles_partials_and_mirrors(tmp_path: Path):
    """`s5_reference` took a bare number or any dict with `tok_s` — a `method=arbitrary_percentile`,
    `store_complete=false`, `sampling_coverage=0.01` record passed as the ceiling. Now the record must be the `S5` smoke's
    measurement (`S5_REFERENCE_KEYS`) on THIS standup's tree and wave; every other shape refuses by name."""
    root, ar = _root(tmp_path)
    good = fx.s5_references(tok_s=2250.0)
    assert _row(ar, s5=good)["tok_s_ratio_s5"] == 1.0 and _row(ar, s5=good)["s5_tok_s"] == 2250.0
    with pytest.raises(ac.AuditRefusal, match=re.escape("a `<L>x<S>: tok_s` scalar map is not a reference")):
        _row(ar, s5={"11520x45:off": 4500.0})
    with pytest.raises(ac.AuditRefusal, match="not RS/smokes.json's S5 block"):
        _row(ar, s5={"windows": {"11520x45": {"tok_s": 2250.0, "thinking": "off"}}})

    def bad(**over):
        d = json.loads(json.dumps(good))
        for r in d["S5"]["references"]:
            if (r["L"], r["S"]) == (11520, 45):
                r.update(over)
        return d
    for over, text in [({"method": "arbitrary_percentile"}, "method 'arbitrary_percentile' is not 'aggregate_decode_tok_s'"),
                       ({"store_complete": False}, "store_complete is not true — a partial store is a floor"),
                       ({"sampling_coverage": 0.01}, "sampling_coverage 0.01 < 0.8 — not steady state"),
                       ({"duration_s": 600}, "duration_s 600 < 1200"),
                       ({"n_eligible_intervals": 0}, "n_eligible_intervals 0 is not a positive int"),
                       ({"W": "0" * 40}, f"W {'0' * 40} is not the standup's tree {fx.W_FIXTURE}"),
                       ({"wave_uuid": "22222222-3333-4444-5555-666666666666"}, "is not the standup's wave"),
                       ({"source": "manual"}, "source 'manual' — a manual or mirrored number is not a measurement"),
                       ({"tok_s_aggregate": "2250"}, "tok_s_aggregate '2250' is not a positive number"),
                       ({"thinking": True}, "carry thinking [True], not the standup's False")]:
        with pytest.raises(ac.AuditRefusal, match=re.escape(text)):
            _row(ar, s5=bad(**over))
    d = bad()
    for r in d["S5"]["references"]:
        r.pop("n_eligible_intervals", None)
    with pytest.raises(ac.AuditRefusal, match=re.escape("lacks ['n_eligible_intervals']")):
        _row(ar, s5=d)
    with pytest.raises(ac.AuditRefusal, match=re.escape("no S5 decode-rate reference for window 11520x45 (off)")):
        _row(ar, s5={"S5": {"references": [r for r in good["S5"]["references"] if r["L"] != 11520]}})
    two = json.loads(json.dumps(good))
    two["S5"]["references"].append(dict(next(r for r in good["S5"]["references"] if r["L"] == 11520)))
    with pytest.raises(ac.AuditRefusal, match="2 S5 references for window 11520x45"):
        _row(ar, s5=two)
    # the block alone (a file holding only the S5 block) reads the same
    assert _row(ar, s5={"references": good["S5"]["references"]})["tok_s_ratio_s5"] == 1.0


def test_expected_hours_lookup_and_refusal(tmp_path: Path):
    root, ar = _root(tmp_path)
    assert _row(ar, expected={"pasttense/s1/base/shard00": 1.0})["h200_h_ratio"] == 2.0
    assert _row(ar, expected={"pasttense/s1/base": 4.0})["h200_h_ratio"] == 0.5
    with pytest.raises(ac.AuditRefusal, match="no entry for pasttense/s1/base/shard00"):
        _row(ar, expected={"bon": 3.0})


def test_dmon_without_timestamps_needs_samplers_started_utc(tmp_path: Path):
    root, ar = _root(tmp_path, dmon_timestamps=False)
    r = _row(ar)
    assert r["dmon_ts_basis"] == "samplers_started_utc+k*period" and r["median_sm_decode"] == 92.0
    att = json.loads((root.path / ac.ATTEMPT_FILE).read_text())
    att.pop("samplers_started_utc")
    (root.path / ac.ATTEMPT_FILE).write_text(json.dumps(att))
    # fu2 (a): without samplers_started_utc the rows are aligned to the metrics cadence (the served app's shape) -- the refusal stands only
    # when there are no scrapes to align to (the reader called without metrics)
    r2 = _row(ac.read_attempt_root(root.path))
    assert r2["dmon_ts_basis"] == ut.DMON_BASIS_METRICS and r2["median_sm_decode"] == 92.0
    with pytest.raises(ac.AuditRefusal, match="dmon rows carry no timestamps"):
        ut.read_dmon(root.path / ac.DMON_FILE, period_s=10.0, start_utc=None, metrics=None)


def test_missing_inputs_refuse(tmp_path: Path):
    root, ar = _root(tmp_path)
    (root.path / ac.COMPLETE_FILE).unlink()
    with pytest.raises(ac.AuditRefusal, match="COMPLETE.json missing"):
        _row(ar)
    root.flush_pipeline()
    (root.path / ac.METRICS_FILE).unlink()
    with pytest.raises(ac.AuditRefusal, match="no 'vllm_metrics.jsonl'"):
        _row(ar)
    root.write_samplers()
    (root.path / ac.INFLIGHT_FILE).write_text("")
    with pytest.raises(ac.AuditRefusal, match="driver recorded no in-flight count"):
        _row(ar)


def test_metric_key_spellings_and_timestamps(tmp_path: Path):
    root, ar = _root(tmp_path)
    rows = ac.read_jsonl(root.path / ac.METRICS_FILE)
    plain = [{"t": ut.parse_ts(r["ts"], "x"), "num_requests_running": r["vllm:num_requests_running"],
              "num_requests_waiting": r["vllm:num_requests_waiting"], "generation_tokens_total": r["vllm:generation_tokens_total"]} for r in rows]
    (root.path / ac.METRICS_FILE).write_text("".join(json.dumps(r) + "\n" for r in plain))
    assert _row(ar)["tok_s_median"] == 2250.0
    with pytest.raises(ac.AuditRefusal, match="neither ISO-8601 nor epoch"):
        ut.parse_ts("yesterday", "x")


def test_cli_fail_on_bars_and_outputs(tmp_path: Path):
    root, ar = _root(tmp_path)
    s5 = fx.s5_file(tmp_path / "smokes.json")
    exp = tmp_path / "hours.json"
    exp.write_text(json.dumps({"pasttense": 3.0}))
    out = tmp_path / "tables" / "utilisation.csv"
    argv = ["--stores", str(tmp_path / "stores"), "--out", str(out), "--s5", str(s5), "--expected-hours", str(exp), "--fail-on-bars"]
    assert ac.run_main(ut.main, argv) == 0
    with open(out, newline="") as f:
        (row,) = list(csv.DictReader(f))
    assert row["bars_met"] == "True" and row["median_running_saturated"] == "45.0" and (out.with_suffix(".json")).is_file()
    root.write_samplers(sm=50)
    assert ac.run_main(ut.main, argv) == ac.EXIT_FAILED
    assert ac.run_main(ut.main, argv[:-1]) == 0  # without --fail-on-bars the table is written and exit 0
    s5.write_text(json.dumps({"1x1:off": 1.0}))
    assert ac.run_main(ut.main, argv) == ac.EXIT_REFUSED


# --------------------------------------------------------------------------- #
# P1-7: the tok/s bar is the AGGREGATE throughput (a stall lowers it; a median of interval rates cannot see it), and the  #
# projected-hours bar is part of the verdict                                                                            #
# --------------------------------------------------------------------------- #

def test_stalls_fail_the_aggregate_bar_while_the_median_stays_high(tmp_path: Path):
    """10 decode intervals, 4 of them stalls (Running > 0, no tokens): the median interval rate is still 2250 tok/s
    (ratio 1.125 — the old bar passed), the aggregate is 6 × 22,500 / 100 s = 1350 tok/s (ratio 0.675) → the bar FAILS."""
    root, ar = _root(tmp_path, stall_scrapes=[2, 4, 6, 8])
    r = _row(ar)
    assert r["tok_s_median"] == 2250.0 and r["n_eligible_intervals"] == 10 and r["sampling_coverage"] == 1.0
    assert r["tok_s_aggregate"] == 1350.0 and r["tok_s_ratio_s5"] == 0.675 and r["tok_bar"] == "false"
    assert r["running_bar"] == "true" and r["sm_bar"] == "true" and r["hours_bar"] == "true" and r["bars_met"] is False


def test_projected_hours_bar_is_in_the_verdict(tmp_path: Path):
    root, ar = _root(tmp_path)
    r = _row(ar, expected={"pasttense": 0.1})  # 2.0 h actual vs 0.1 h sized → 20x
    assert r["h200_h_ratio"] == 20.0 and r["hours_bar"] == "false" and r["bars_met"] is False
    assert r["running_bar"] == r["sm_bar"] == r["tok_bar"] == "true"
    assert any("is 20.00x the sized 0.1 (bar: <= 1.5x)" in p for p in r["problems"])
    assert _row(ar, expected={"pasttense": 1.4})["hours_bar"] == "true"  # 1.43x: within the bar
    r0 = _row(ar, expected={"pasttense": 0.0})
    assert r0["hours_bar"] == "unmeasured(expected hours 0)" and r0["h200_h_ratio"] == "expected 0" and r0["bars_met"] is False


def test_counter_resets_lower_the_sampling_coverage_and_unmeasure_the_bar(tmp_path: Path):
    root, ar = _root(tmp_path, reset_scrapes=[3, 5, 7, 9, 11])  # the interval INTO each reset scrape is non-monotone: 5 of 10
    r = _row(ar)
    assert r["n_eligible_intervals"] == 5 and r["sampling_coverage"] == 0.5
    assert r["tok_bar"] == f"unmeasured(sampling coverage 0.50 < {ut.MIN_SAMPLING_COVERAGE})" and r["bars_met"] is False
    assert any("dropped 5 of 10" in p for p in r["problems"])


def test_cli_unmet_line_names_the_hours_bar(tmp_path: Path, capsys):
    root, ar = _root(tmp_path)
    s5 = fx.s5_file(tmp_path / "smokes.json")
    exp = tmp_path / "hours.json"
    exp.write_text(json.dumps({"pasttense": 0.1}))
    out = tmp_path / "tables" / "utilisation.csv"
    argv = ["--stores", str(tmp_path / "stores"), "--out", str(out), "--s5", str(s5), "--expected-hours", str(exp), "--fail-on-bars"]
    assert ac.run_main(ut.main, argv) == ac.EXIT_FAILED
    assert "hours=false" in capsys.readouterr().out
    with open(out, newline="") as f:
        (row,) = list(csv.DictReader(f))
    assert row["bars_met"] == "False" and row["hours_bar"] == "false" and row["tok_bar"] == "true" and set(row) == {c.rstrip("?") for c in ut.COLUMNS}


# --------------------------------------------------------------------------- #
# fu1 / fu2: driver_inflight.jsonl in the SERVED app's shape (child start / exit events) -- DRIVER DEMAND = live children x the  #
# concurrency the ROOT records per child (children/<child>.argv.json --concurrency); two DISJOINT schemas validated over the  #
# whole file; dmon rows aligned to the metrics cadence; the served COMPLETE.json; the cluster shape unchanged                  #
# --------------------------------------------------------------------------- #

import subprocess  # noqa: E402
import types  # noqa: E402
from datetime import timedelta  # noqa: E402

PRE_FU1 = "3409826296bf282e5fd1c024d51611fa6a641362"          # the launch tree: utilisation_table.py before fu1 / fu2 (byte-identical on main / fj3)
INFO_COLUMNS = ("inflight_basis", "inflight_counts", "dmon_alignment_uncertainty_s", "complete_ts_basis")   # fu1 / fu2's informational columns


def _pre_fu1_module():
    src = subprocess.run(["git", "-C", str(REPO), "show", f"{PRE_FU1}:scripts/sdfbase/seedrep/utilisation_table.py"], capture_output=True, text=True, check=True).stdout
    mod = types.ModuleType("utilisation_table_pre_fu1")
    mod.__file__ = str(REPO / "scripts" / "sdfbase" / "seedrep" / "utilisation_table.py")
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def _sha_exists(sha):
    return subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True).returncode == 0


def _t(k, period_s=10):
    return fx._iso(fx.T0 + timedelta(seconds=period_s * k))


def _served_root(root, *, children, concurrency=1, exits=None, argv_for=None):
    """The served app's driver_inflight.jsonl (`start` events at scrape 0 for every child, `exit` events at the given scrape index per child,
    `live` = concurrent children) and the ROOT's record of each child's argv (`children/<child>.argv.json` naming `--concurrency N`) --
    synthetic child names, the fixture's clock. ``argv_for``: per-child argv override (None = no record for that child)."""
    exits = exits or {}
    rows, live = [], 0
    for c in children:
        live += 1
        rows.append({"t": _t(0), "event": "start", "child": c, "live": live})
    for c, k in sorted(exits.items(), key=lambda kv: kv[1]):
        live -= 1
        rows.append({"t": _t(k), "event": "exit", "child": c, "rc": 0, "live": live})
    (root.path / ac.INFLIGHT_FILE).write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root.path / "children").mkdir(exist_ok=True)
    for c in children:
        argv = (argv_for or {}).get(c, ["/venv/bin/python", "driver.py", "--concurrency", str(concurrency), "--out", "x"]) if argv_for is None or c in argv_for else None
        p = root.path / "children" / f"{c}.argv.json"
        if argv is None:
            if p.exists():
                p.unlink()
            continue
        p.write_text(json.dumps({"argv": argv, "env_registry": {}}))
    return rows


def _set_running(root, values):
    rows = ac.read_jsonl(root.path / ac.METRICS_FILE)
    for r, v in zip(rows, values):
        r["vllm:num_requests_running"] = v
    (root.path / ac.METRICS_FILE).write_text("".join(json.dumps(r) + "\n" for r in rows))


def _cli_argv(tmp_path, root, out):
    s5 = fx.s5_file(tmp_path / "smokes.json")
    exp = tmp_path / "hours.json"
    exp.write_text(json.dumps({"pasttense": 3.0}))
    return ["--roots", str(root.path), "--out", str(out), "--s5", str(s5), "--expected-hours", str(exp), "--fail-on-bars"]


@pytest.mark.skipif(not _sha_exists(PRE_FU1), reason="the launch tree's sha is not in this clone")
def test_fu2_sol_p1_driver_demand_from_live_children_gives_the_cluster_selectors_verdict(tmp_path: Path):
    """The scenario: S=45, twelve scrapes, Running 20 on nine and 45 on three (waiting 0). Cluster tick log (in-flight 45 throughout): 12
    selected, median 20, bar FALSE. fu1's running+waiting selector picked the three 45s (median 45, bar true) -- the served basis now rebuilds
    driver demand (45 children live x concurrency 1 = 45 >= S on every scrape): 12 selected, median 20, bar FALSE -- the same verdict."""
    running = [45, 20, 20, 20, 45, 20, 20, 20, 45, 20, 20, 20]
    root, ar = _root(tmp_path)
    _set_running(root, running)
    r_cluster = _row(ar)                                                               # the fixture's cluster tick log: in-flight 45 on scrapes 1..11
    assert r_cluster["inflight_basis"] == "driver_inflight" and r_cluster["n_saturated_scrapes"] == 11 and r_cluster["median_running_saturated"] == 20.0 and r_cluster["running_bar"] == "false"
    children = [f"pasttense__arm{i:02d}__s1" for i in range(45)]
    _served_root(root, children=children, concurrency=1)
    old = _pre_fu1_module()
    with pytest.raises(ac.AuditRefusal, match=re.escape("row lacks any of ['inflight', 'in_flight', 'n_inflight']")):
        old.utilisation_row(ar, s5=fx.s5_references(), expected={"pasttense": 3.0}, period_s=10.0)
    r = _row(ac.read_attempt_root(root.path))
    assert r["inflight_basis"] == ut.INFLIGHT_BASIS_SERVED == "served_children_live_concurrency"
    assert r["n_saturated_scrapes"] == 12 and r["median_running_saturated"] == 20.0 and r["running_bar"] == "false" and r["bars_met"] is False
    cnt = r["inflight_counts"]
    assert cnt["n_children"] == 45 and cnt["max_demand"] == 45.0 and set(cnt["per_child_concurrency"].values()) == {1} and set(cnt["concurrency_source"].values()) == {"--concurrency"}
    assert [d for _, d in cnt["demand_series"]] == [45.0] * 12 and cnt["events"] == {"start": 45} and cnt["children_without_recorded_concurrency"] == []
    assert set(r) == {c.rstrip("?") for c in ut.COLUMNS}
    # children exiting mid-run LOWER the selection honestly: 10 of 45 exit at scrape 6 -> demand 35 < S from scrape 6 on -> 6 selected (Running 45,20,20,20,45,20)
    _served_root(root, children=children, concurrency=1, exits={c: 6 for c in children[:10]})
    r2 = _row(ac.read_attempt_root(root.path))
    assert r2["n_saturated_scrapes"] == 6 and [d for _, d in r2["inflight_counts"]["demand_series"]] == [45.0] * 6 + [35.0] * 6 and r2["median_running_saturated"] == 20.0
    # demand never reaching S (20 children x 2 = 40 < 45): unmeasured, the problem names the driver-demand basis
    _served_root(root, children=children[:20], concurrency=2)
    r3 = _row(ac.read_attempt_root(root.path))
    assert r3["running_bar"] == "unmeasured(in-flight never >= S)" and r3["n_saturated_scrapes"] == 0
    assert any("never reached S=45 (max driver demand (served_children_live_concurrency) 40)" in p for p in r3["problems"])
    # a live child whose concurrency the root does NOT record: unmeasured by name, no engine-occupancy fallback, no scrape selected
    _served_root(root, children=children, concurrency=1, argv_for={c: ["/venv/bin/python", "driver.py", "--concurrency", "1"] for c in children[1:]})
    r4 = _row(ac.read_attempt_root(root.path))
    assert r4["running_bar"] == "unmeasured(driver demand not recorded on the root)" and r4["n_saturated_scrapes"] == 0 and r4["bars_met"] is False
    assert r4["inflight_counts"]["children_without_recorded_concurrency"] == [children[0]] and r4["inflight_counts"]["per_child_concurrency"][children[0]] is None
    assert any("driver demand not recorded on the root: 1 child(ren)" in p and children[0] in p for p in r4["problems"])
    # a child whose argv names no concurrency flag (the static driver) is likewise unrecorded
    _served_root(root, children=children, concurrency=1, argv_for={**{c: ["/venv/bin/python", "driver.py", "--concurrency", "1"] for c in children[1:]},
                                                                    children[0]: ["/venv/bin/python", "jailbreak_run.py", "generate", "--shard", "0"]})
    r5 = _row(ac.read_attempt_root(root.path))
    assert r5["running_bar"] == "unmeasured(driver demand not recorded on the root)" and "names no" in r5["inflight_counts"]["concurrency_source"][children[0]]
    # the CLI over a served-shaped root: exit 0 without --fail-on-bars, the basis in the CSV
    _served_root(root, children=children, concurrency=1)
    out = tmp_path / "tables" / "utilisation.csv"
    argv = _cli_argv(tmp_path, root, out)
    assert ac.run_main(ut.main, argv[:-1]) == 0
    with open(out, newline="") as f:
        (row,) = list(csv.DictReader(f))
    assert row["inflight_basis"] == "served_children_live_concurrency" and json.loads(row["inflight_counts"])["n_children"] == 45 and row["running_bar"] == "false"


def test_fu2_sol_p2_two_disjoint_schemas_over_the_whole_file(tmp_path: Path):
    root, ar = _root(tmp_path)
    bad = [
        ([{"event": "unrelated"}], "lacks ['t', 'child', 'live']"),                                                         # selected the served basis before
        ([{"t": _t(0), "event": "start", "child": "c", "live": 1, "inflight": 3}], "is a hybrid"),                          # served rows with a stray inflight
        ([{"ts": _t(0), "inflight": 45}, {"t": _t(1), "event": "start", "child": "c", "live": 1}], "rows mix the cluster tick log and the served event log"),  # a mixture
        ([{"ts": _t(0), "inflight": 45}, {"ts": _t(1), "inflight": 45, "event": "start", "child": "c", "live": 1}], "is a hybrid"),
        ([{"t": _t(0), "event": "teardown_after_failure", "child": "c", "live": 0}], "is not one of ['start', 'exit']"),
        ([{"t": _t(0), "event": "start", "child": "c", "live": "1"}], "live '1' is not an integer"),
        ([{"t": _t(0), "event": "start", "child": 7, "live": 1}], "child 7 is not a name"),
        ([{"inflight": 45}], "lacks a timestamp key"),
        ([{"ts": _t(0), "inflight": 45, "in_flight": 45}], "carries 2 in-flight keys"),
        ([{"t": _t(0), "children": 3}], "lacks any of ['inflight', 'in_flight', 'n_inflight'] and is not a served child event"),
        (["not-an-object"], "is not an object"),
    ]
    for rows, needle in bad:
        (root.path / ac.INFLIGHT_FILE).write_text("".join(json.dumps(r) + "\n" for r in rows))
        with pytest.raises(ac.AuditRefusal, match=re.escape(needle)) as ei:
            _row(ar)
        assert "not an object" in needle or any(json.dumps(r, default=str)[:40] in str(ei.value) for r in rows if isinstance(r, dict)), (needle, str(ei.value))   # the offending row shown
    (root.path / ac.INFLIGHT_FILE).write_text("")
    with pytest.raises(ac.AuditRefusal, match="driver recorded no in-flight count"):
        _row(ar)


@pytest.mark.skipif(not _sha_exists(PRE_FU1), reason="the launch tree's sha is not in this clone")
def test_fu2_a_cluster_shaped_root_yields_the_same_table_as_the_pre_fix_module(tmp_path: Path):
    root, ar = _root(tmp_path)                       # the fixture's driver_inflight.jsonl is the cluster driver's {ts, inflight}; dmon -o DT; COMPLETE written_utc
    old = _pre_fu1_module()
    out_old, out_new = tmp_path / "old" / "utilisation.csv", tmp_path / "new" / "utilisation.csv"
    assert ac.run_main(old.main, _cli_argv(tmp_path, root, out_old)) == 0
    assert ac.run_main(ut.main, _cli_argv(tmp_path, root, out_new)) == 0
    with open(out_new, newline="") as f:
        new_rows = list(csv.DictReader(f))
    with open(out_old, newline="") as f:
        old_rows = list(csv.DictReader(f))
    assert sorted(c for c in new_rows[0] if c not in old_rows[0]) == sorted(INFO_COLUMNS)
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(old_rows[0].keys()), lineterminator="\n")
    w.writeheader()
    for r in new_rows:
        w.writerow({k: r[k] for k in old_rows[0]})
    assert buf.getvalue() == out_old.read_text(encoding="utf-8")                          # byte for byte, minus the informational columns
    jo, jn = json.loads(out_old.with_suffix(".json").read_text()), json.loads(out_new.with_suffix(".json").read_text())
    assert [{k: v for k, v in r.items() if k not in INFO_COLUMNS} for r in jn] == jo
    assert jn[0]["inflight_basis"] == "driver_inflight" and jn[0]["inflight_counts"] == {"n_rows": 12, "max_inflight": 45.0}
    assert jn[0]["dmon_alignment_uncertainty_s"] == {"uncertainty_s": 0.0} and jn[0]["complete_ts_basis"] == "written_utc"


def test_fu2_a_served_dmon_log_without_timestamps_is_aligned_to_the_metrics_cadence_and_the_served_complete_json_is_read(tmp_path: Path):
    root, ar = _root(tmp_path, dmon_timestamps=False)                 # `nvidia-smi dmon -i 0 -s um -d 10` without -o DT, as the served app runs it
    att = json.loads((root.path / ac.ATTEMPT_FILE).read_text()); att.pop("samplers_started_utc")
    (root.path / ac.ATTEMPT_FILE).write_text(json.dumps(att))          # ... and no samplers_started_utc recorded
    comp = json.loads((root.path / ac.COMPLETE_FILE).read_text())
    comp["completed_utc"] = comp.pop("written_utc"); comp["elapsed_s"] = 7200.0
    (root.path / ac.COMPLETE_FILE).write_text(json.dumps(comp))        # the served app's COMPLETE.json {completed_utc, elapsed_s}
    r = _row(ac.read_attempt_root(root.path))
    assert r["dmon_ts_basis"] == ut.DMON_BASIS_METRICS == "dmon_rows_aligned_to_metrics_cadence" and r["median_sm_decode"] == 92.0 and r["sm_bar"] == "true"
    assert r["dmon_alignment_uncertainty_s"] == {"uncertainty_s": 10.0, "n_rows": 12, "n_rows_span_needs": 12, "n_rows_beyond_span": 0}
    assert r["complete_ts_basis"] == "completed_utc" and r["h200_h_actual"] == 2.0 and r["hours_bar"] == "true" and r["bars_met"] is True
    # trailing dmon rows (the sampler outliving the scrape loop at teardown) are counted, not placed; two rows short of the span refuse by name
    text = (root.path / ac.DMON_FILE).read_text()
    (root.path / ac.DMON_FILE).write_text(text + "0    90    40     0     0     0     0  90000     4     0\n" * 3)
    r2 = _row(ac.read_attempt_root(root.path))
    assert r2["dmon_alignment_uncertainty_s"]["n_rows_beyond_span"] == 3 and r2["median_sm_decode"] == 92.0
    lines = [l for l in text.splitlines() if l.strip()]
    (root.path / ac.DMON_FILE).write_text("\n".join(lines[:-2]) + "\n")   # 10 data rows for a 12-period span
    with pytest.raises(ac.AuditRefusal, match="10 dmon rows cannot cover the 12 periods"):
        _row(ac.read_attempt_root(root.path))
    (root.path / ac.DMON_FILE).write_text("\n".join(lines[:-1]) + "\n")   # one short: within a cadence, placed
    assert _row(ac.read_attempt_root(root.path))["dmon_alignment_uncertainty_s"]["n_rows_beyond_span"] == -1
    # a COMPLETE.json with neither timestamp refuses by name
    comp.pop("completed_utc"); (root.path / ac.COMPLETE_FILE).write_text(json.dumps(comp))
    with pytest.raises(ac.AuditRefusal, match="without written_utc \\(nor the served app's completed_utc\\)"):
        _row(ac.read_attempt_root(root.path))
