"""The paper parity gate on a synthetic pair: a synthetic pinned checkout (a mini paperlib whose
readers emit the long schema from a fixture) beside a synthetic registry → L0 / L1 / L1b / L1c pass; one moved value
fails naming the key; one input sha moved between the legs refuses the gate; the legacy-read audit hook raises by name
on the registry leg; figdata extraction covers lines / bars / collections / text and the comparison names the path
that moved; the disagreements are listed first in the report; `parity_gate` records are written per level.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest
import yaml

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import paper
from sisteer.registry import parity as P
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry.adapters import roster as RO

COLUMNS = ["source", "model", "lane_kind", "view", "dose", "cell_id", "arm", "family", "object_type", "component", "eval_id", "eval", "metric", "threshold",
           "bon_n", "value", "k", "n", "ci_lo", "ci_hi", "delta", "delta_lo", "delta_hi", "delta_p", "delta_method", "effect_size", "effect_kind",
           "decode_cap_tokens", "decode", "judge", "status", "note", "provenance", "thinking_mode", "object_sha256", "held_out", "arm_operating_point_id"]

MINI_LOAD = textwrap.dedent('''
    """A synthetic paperlib.load: the readers emit the long schema from a JSON fixture named by config."""
    import json, os
    from dataclasses import dataclass, field
    import pandas as pd
    from . import config as C
    COLUMNS = %s
    @dataclass
    class LaneStatus:
        name: str; owner: str; integrated: bool; where: str; detail: str; cells_measured: int = 0; extra: dict = field(default_factory=dict)
    def _rows():
        with open(C.R2_PARQUET) as fh:
            return pd.DataFrame(json.load(fh)).reindex(columns=COLUMNS)
    def load_r2():
        return _rows(), {"built_utc": "synthetic", "n_rows": 0}
    def load_overrefusal_graded():
        return _rows().iloc[0:0], LaneStatus("over-refusal", "S", False, C.R2_PARQUET, "none")
    SECONDARY = []
    def load_all(guard=True, return_lane_map=False):
        rows, meta = load_r2()
        rows.attrs["non_canonical"] = {}
        return rows, meta, [LaneStatus("r2 tidy export (primary)", "Scale", True, C.R2_PARQUET, "synthetic")]
    def paper_lanes(rows):
        return rows
    def canonical_view(rows, keep_second_pass=False, keep_same_source_passes=False):
        return rows
''')

MINI_GUARD = textwrap.dedent('''
    import json, pathlib
    import pandas as pd
    WATERMARKS = pathlib.Path(__file__).with_name("lane_watermarks.json")
    KEY = ("source", "model", "lane_kind", "component", "dose")
    def _render(v):
        if v is None or (isinstance(v, float) and v != v):
            return "NA"
        if isinstance(v, (int, float)):
            return f"{float(v):g}"
        return str(v)
    def measured_counts(rows):
        m = rows[rows.value.notna() & rows.source.notna()]
        out = {}
        for vals, n in m.groupby(list(KEY), dropna=False).size().items():
            out.setdefault(_render(vals[0]), {})["|".join(_render(v) for v in vals[1:])] = int(n)
        return out
    def load_watermarks(path=WATERMARKS):
        return json.load(open(path))
''')

MINI_FIGS = textwrap.dedent('''
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGDIR = Path(__file__).resolve().parent.parent / "figures"
    def save(fig, name):
        FIGDIR.mkdir(exist_ok=True)
        fig.savefig(FIGDIR / f"{name}.png")
        return fig
    def fig_synth(rows):
        fig, ax = plt.subplots()
        r = rows[rows.value.notna()].sort_values(["arm", "eval_id"])
        ax.plot(range(len(r)), r.value.values, marker="o", label="value")
        ax.bar(range(len(r)), r.n.values / 1000.0, alpha=0.3)
        ax.scatter(range(len(r)), r.value.values * 0.5)
        for i, a in enumerate(r.arm.values):
            ax.text(i, 0.0, str(a))
        ax.set_title("synthetic figure")
        save(fig, "fig_synth")
        return fig
''')

NOTEBOOK = {"cells": [{"cell_type": "code", "source": ["import sys\n", "from paperlib import load as L, figs as F\n", "rows, meta, statuses = L.load_all(guard=False)\n",
                                                    "F.fig_synth(rows)\n"], "metadata": {}, "outputs": [], "execution_count": None}],
            "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


def _git(cwd, *args):
    return subprocess.check_output(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-C", cwd] + list(args), text=True, stderr=subprocess.STDOUT).strip()


def _write(root, sub, name, recs):
    return R.write_records(os.path.join(root, "records", sub, name), recs)


def _registry(tmp_path):
    """A synthetic registry: two judged runs (none + constitution_lora) on one source, a roster matching both."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    dims = synth.seed_dimensions()
    R.write_records(os.path.join(root, "records", "dimensions", "seed", "000001-dimensions.jsonl"), dims)
    run_ids = []
    for k, (key, okey, oid) in enumerate((("synthstore_000", "none", "none"), ("synthstore_001", "constitution_lora@deadbeef", synth.OBJECT_SHA))):
        rid, recs = synth.run_records(key, 100 + 20 * k, okey, oid, 4)
        for rec in recs:
            if rec["record_type"] == "run":
                rec["payload"]["lane_source_legacy"] = "synth_source"
                rec["payload"]["view_legacy"] = "synth_view"
        synth.write_run_dir(root, synth.CAMPAIGN, rid, [(str(r["record_type"]), [r]) for r in recs], gz_items=False)
        run_ids.append(rid)
    roster = {"meta": {"schema": RO.ROSTER_SCHEMA, "roster_version": "v1", "round_id": "r_synth",
                       "paper_round": {"round_id": "r_synth", "opened_utc": "2026-01-01", "description": "synthetic", "paper_build_commit": "e" * 40}},
              "vocabulary": {"eval_id_from_eval_legacy": {}},
              "models": {synth.MODEL_ID: {"paper_lane": {"transfer_kind": "base_to_instruct"}}},
              "objects": {"none": {"paper_label": "none", "object_kind": "none", "object_role": "baseline_none"},
                          "constitution_lora": {"paper_label": "constitution_lora", "object_kind": "lora", "object_role": "constitution", "arm_legacy": "const_lora"}},
              "judges": {"j": {"judge_model": "judge-model-x"}},
              "evals": {synth.EVAL_ID: {"eval_category": "static", "component_legacy": "static", "judge": "j", "instrument_selector_json": {"caps": [512]}}},
              "comparisons": [{"comparison_id_template": "fig_synth/{transfer_kind}", "figure_legacy": "fig_synth",
                               "models": [{"model_id": synth.MODEL_ID, "transfer_kind": "base_to_instruct"}], "objects": ["none", "constitution_lora"],
                               "evals": [synth.EVAL_ID], "strength_rules": ["operating_point"], "source_citation": "paperlib.config.SYNTHETIC @ eeeeeeee",
                               "exceptions": [{"kind": "annotate", "object": "constitution_lora", "inferred": True, "disagreement": "D-synth: a written ruling disagrees with the build"}]}]}
    ry = tmp_path / "roster.yaml"
    ry.write_text(yaml.safe_dump(roster, sort_keys=False))
    RO.write_records(root, RO.load(str(ry)), written_utc=synth.t(50))
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    return root, run_ids


def _checkout(tmp_path, fixture_rows, watermarks):
    co = tmp_path / "paper-pin"
    lib = co / "paper" / "paperlib"
    lib.mkdir(parents=True)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    fixture = inputs / "results.json"
    fixture.write_text(json.dumps(fixture_rows, default=str))
    (lib / "__init__.py").write_text("")
    (lib / "config.py").write_text(f'R2_PARQUET = {str(fixture)!r}\nR2_SOURCES = {str(inputs / "sources.json")!r}\nR2_TRANSFER_ITEMS = None\nR2_TRANSFER_STATS = None\nSTRENGTH_RECORDS = []\n')
    (inputs / "sources.json").write_text(json.dumps({"built_utc": "synthetic"}))
    (lib / "load.py").write_text(MINI_LOAD % json.dumps(COLUMNS))
    (lib / "lane_guard.py").write_text(MINI_GUARD)
    (lib / "figs.py").write_text(MINI_FIGS)
    (lib / "lane_watermarks.json").write_text(json.dumps(watermarks))
    (co / "paper" / "constitutional_adapters_figures.ipynb").write_text(json.dumps(NOTEBOOK))
    _git(str(co), "init", "-q")
    _git(str(co), "add", "-A")
    _git(str(co), "commit", "-q", "-m", "synthetic pin")
    return str(co), _git(str(co), "rev-parse", "HEAD"), str(fixture)


@pytest.fixture
def pair(tmp_path):
    root, run_ids = _registry(tmp_path)
    conn = DB.connect(DB.db_path(root), readonly=True)
    frame = paper.paper_long_frame(conn, "r_synth", columns=COLUMNS)
    conn.close()
    rows = json.loads(frame[COLUMNS].to_json(orient="records"))
    # the watermarks are the measured counts of this very frame (lane_guard's key), so L1c must hold on the identical pair
    keys = {}
    for r in rows:
        key = "|".join(("NA" if r[c] is None else (f"{float(r[c]):g}" if isinstance(r[c], (int, float)) else str(r[c]))) for c in ("model", "lane_kind", "component", "dose"))
        keys[key] = keys.get(key, 0) + 1
    marks = {"written_at": "synthetic", "sources": {"synth_source": {"lane": "synthetic", "owner": "t", "rows": sum(keys.values()), "keys": keys}}}   # lane_guard's file shape
    co, sha, fixture = _checkout(tmp_path, rows, marks)
    return root, co, sha, fixture, rows


def test_identical_pair_passes_every_level_and_writes_records_and_report(pair, tmp_path):
    root, co, sha, fixture, rows = pair
    res = P.check_paper_parity(co, sha, root, round_id="r_synth", levels=("L0", "L1", "L1b", "L1c", "L2"), paper_python=sys.executable,
                               work_dir=str(tmp_path / "work"), now=synth.t(9000))
    assert res.refused is None, res.refused
    assert res.result_of("L0") == "pass" and res.result_of("L1") == "pass" and res.result_of("L1b") == "pass" and res.result_of("L1c") == "pass"
    # L2's registry leg needs the cutover branch's paperlib: skipped with the reason, the legacy figdata produced
    assert res.levels["L2"]["skipped"] and res.levels["L2"]["legacy_figures"] == ["fig_synth.figdata.json"]
    assert res.result_of("L2") == "refused"
    # the input set was pinned (fixture + sources.json + watermarks) and the disagreements come first in the report
    assert fixture in res.input_shas and all(len(v) == 64 for v in res.input_shas.values())
    report = open(res.report_path).read()
    assert report.index("## Disagreements first") < report.index("## Expected breaks") < report.index("## L0")
    assert "D-synth" in report and "test_fig5_frontier_tweaks.py" in report
    # one parity_gate record per level, ingestable
    assert len(res.record_paths) == 5
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rows_ = conn.execute("SELECT level, result FROM parity_gate ORDER BY level").fetchall()
    assert [tuple(r) for r in rows_] == [("L0", "pass"), ("L1", "pass"), ("L1b", "pass"), ("L1c", "pass"), ("L2", "refused")]
    conn.close()
    # the figdata extraction covered lines, bars, a collection and texts
    fd = json.load(open(os.path.join(os.path.dirname(res.report_path), "figdata_legacy", "fig_synth.figdata.json")))
    ax = fd["axes"][0]
    assert ax["lines"] and ax["patches"] and ax["collections"] and ax["texts"] and ax["title"] == "synthetic figure"


def test_one_moved_value_fails_naming_the_key(pair, tmp_path):
    root, co, sha, fixture, rows = pair
    mutated = json.loads(json.dumps(rows))
    target = next(r for r in mutated if r["arm"] == "const_lora")
    target["value"] = float(target["value"]) + 0.01
    with open(fixture, "w") as fh:
        json.dump(mutated, fh)
    subprocess.check_call(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-C", co, "commit", "-q", "--allow-empty", "-m", "fixture moved"])
    sha2 = subprocess.check_output(["git", "-C", co, "rev-parse", "HEAD"], text=True).strip()
    res = P.check_paper_parity(co, sha2, root, round_id="r_synth", levels=("L0", "L1", "L1b"), paper_python=sys.executable, work_dir=str(tmp_path / "work"),
                               write_records=False)
    assert res.refused is None
    assert res.result_of("L0") == "fail" and res.result_of("L1") == "fail"
    diff = res.levels["L1"]["assembled"]["differing"]
    assert len(diff) == 1 and "const_lora" in diff[0]["key"] and "value" in diff[0]["diffs"]
    assert abs(diff[0]["diffs"]["value"]["legacy"] - diff[0]["diffs"]["value"]["registry"] - 0.01) < 1e-9
    report = open(res.report_path).read()
    assert "## L1: FAIL" in report and "const_lora" in report


def test_input_moved_between_legs_is_refused(pair, tmp_path):
    root, co, sha, fixture, rows = pair

    def move():
        with open(fixture, "a") as fh:
            fh.write("\n")   # the bytes change, the frame does not: exactly the case the input-set pin exists for
    res = P.check_paper_parity(co, sha, root, round_id="r_synth", levels=("L0",), paper_python=sys.executable, work_dir=str(tmp_path / "work"),
                               between_sides=move, write_records=True, now=synth.t(9100))
    assert res.refused and "input set moved between the legs" in res.refused and fixture in res.refused
    assert res.result_of("L0") == "refused" and "L0" not in res.levels
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT result FROM parity_gate").fetchone()[0] == "refused"
    conn.close()


def test_wrong_pin_and_missing_venv_are_refused_before_any_leg(pair, tmp_path):
    root, co, sha, fixture, rows = pair
    with pytest.raises(P.ParityRefused, match="not the pin"):
        P.check_paper_parity(co, "0" * 40, root, round_id="r_synth", paper_python=sys.executable)
    with pytest.raises(P.ParityRefused, match="missing"):
        P.check_paper_parity(co, sha, root, round_id="r_synth", paper_python=str(tmp_path / "no-such-python"))


def test_legacy_read_hook_raises_by_name_in_a_fresh_process(tmp_path):
    forbidden = tmp_path / "legacy-root"
    forbidden.mkdir()
    (forbidden / "store.json").write_text("{}")
    allowed = tmp_path / "ok.json"
    allowed.write_text("{}")
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    code = textwrap.dedent(f'''
        import os
        from sisteer.registry import parity as P
        P.install_legacy_read_hook([{str(forbidden)!r}])
        open({str(allowed)!r}).read()
        try:
            open({str(forbidden / "store.json")!r}).read()
        except P.LegacyReadForbidden as exc:
            print("REFUSED", exc)
        try:
            os.listdir({str(forbidden)!r})
        except P.LegacyReadForbidden as exc:
            print("REFUSED2", exc)
    ''')
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=src))
    assert out.returncode == 0, out.stderr
    assert "REFUSED " in out.stdout and "store.json" in out.stdout and "REFUSED2" in out.stdout


def test_figdata_and_frame_comparison_units():
    a = {"axes": [{"lines": [{"label": "v", "xy": [[0, 0.5], [1, 0.25]]}], "patches": [{"x": 0, "y": 0, "w": 0.8, "h": 0.2}], "collections": [{"offsets": [[0, 0.25]]}], "texts": [{"x": 0, "y": 0, "s": "a"}]}]}
    b = json.loads(json.dumps(a))
    assert P.compare_figdata(a, b) == []
    b["axes"][0]["lines"][0]["xy"][1][1] = 0.25 + 1e-12
    assert P.compare_figdata(a, b) == []                       # inside 1e-9
    b["axes"][0]["lines"][0]["xy"][1][1] = 0.26
    d = P.compare_figdata(a, b)
    assert len(d) == 1 and d[0].startswith("/axes[0]/lines[0]/xy[1][1]")
    b["axes"][0]["texts"][0]["s"] = "b"
    assert any("texts[0]/s" in x for x in P.compare_figdata(a, b))
    rows_a = [{"source": "s", "model": "m", "lane_kind": "l", "arm": "a", "eval_id": "e", "metric": "asr", "threshold": 0.5, "decode_cap_tokens": 512, "judge": "j", "pass": None, "value": 0.1, "n": 4}]
    rows_b = [dict(rows_a[0], value=float("nan"))]
    cmp = P.compare_frames(rows_a, rows_b)
    assert not cmp["pass"] and cmp["differing"][0]["diffs"]["value"]["registry"] != cmp["differing"][0]["diffs"]["value"]["registry"]
    assert P.compare_frames(rows_a, [dict(rows_a[0], value=0.1 + 1e-10)])["pass"]
    assert P.compare_watermarks({"s": {"k": 2}}, {"s": {"k": 3}})["shortfalls"] == {"s": {"k": {"watermark": 3, "measured": 2}}}
    assert P.compare_watermarks({"s": {"k": 3}}, {"s": {"k": 3}})["pass"]
    assert P.compare_watermarks({"s": {"k": 2}}, {"sources": {"s": {"lane": "x", "rows": 3, "keys": {"k": 3}}}})["shortfalls"] == {"s": {"k": {"watermark": 3, "measured": 2}}}
    assert P.compare_watermarks({"s": {"k": 3}}, {"sources": {"s": {"lane": "x", "rows": 3, "keys": {"k": 3}}}})["pass"]
