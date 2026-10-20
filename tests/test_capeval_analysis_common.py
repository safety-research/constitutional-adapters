"""The shared analysis helpers (``scripts/capeval_analysis_common.py``) the
BCB / BFCL-200 analysis imports: the Wilson interval and ``pass_rate`` at their
pinned values, the exact McNemar test, the JSON / JSONL readers' refusals, and
the house figure rules (title ≤ 2 lines; axes keep ≥ 80 % of the height; a tile
is exempt). Pure: no Modal, no network — matplotlib on Agg."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "capeval_analysis_common.py"


def _load():
    name = "capeval_analysis_common"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


A = _load()


def test_wilson_interval_pinned_values():
    assert tuple(round(v, 4) for v in A.wilson_interval(0, 89)) == (0.0, 0.0414)
    assert tuple(round(v, 4) for v in A.wilson_interval(8, 89)) == (0.0463, 0.1675)
    assert A.wilson_interval(89, 89)[1] == 1.0 and A.wilson_interval(0, 89)[0] == 0.0
    for bad in ((0, 0), (-1, 5), (6, 5), (True, 5), (1, 5.0)):
        with pytest.raises(A.AnalysisError):
            A.wilson_interval(*bad)


def test_pass_rate_carries_the_interval_and_is_null_at_zero_trials():
    pr = A.pass_rate(8, 89)
    assert pr["rate"] == pytest.approx(8 / 89) and (pr["k"], pr["n"]) == (8, 89)
    assert (pr["wilson_low"], pr["wilson_high"]) == A.wilson_interval(8, 89)
    assert A.pass_rate(0, 0) == {"k": 0, "n": 0, "rate": None, "wilson_low": None, "wilson_high": None}


def test_mcnemar_exact_p_pinned_values():
    assert A.mcnemar_exact_p(0, 0) == 1.0
    assert A.mcnemar_exact_p(1, 0) == 1.0 and A.mcnemar_exact_p(0, 1) == 1.0
    assert A.mcnemar_exact_p(0, 3) == pytest.approx(0.25)
    assert A.mcnemar_exact_p(1, 4) == pytest.approx(0.375)
    assert A.mcnemar_exact_p(0, 10) == pytest.approx(2 / 1024)
    assert A.mcnemar_exact_p(5, 5) == 1.0                        # capped at 1


def test_readers_name_the_file_and_line_they_refuse(tmp_path):
    good = tmp_path / "a.json"
    good.write_text(json.dumps({"k": 1}))
    assert A.read_json(good) == {"k": 1}
    with pytest.raises(A.AnalysisError, match="unreadable"):
        A.read_json(tmp_path / "missing.json")
    bad = tmp_path / "b.json"
    bad.write_text("{not json")
    with pytest.raises(A.AnalysisError, match="not JSON"):
        A.read_json(bad)
    rows = tmp_path / "r.jsonl"
    rows.write_text('{"a": 1}\n\n{"a": 2}\n')
    assert A.read_jsonl(rows) == [{"a": 1}, {"a": 2}]
    rows.write_text('{"a": 1}\n[1, 2]\n')
    with pytest.raises(A.AnalysisError, match=r"r\.jsonl:2: row is a list"):
        A.read_jsonl(rows)
    rows.write_text('{"a": 1}\nnope\n')
    with pytest.raises(A.AnalysisError, match=r"r\.jsonl:2: not JSON"):
        A.read_jsonl(rows)


def test_summary_stats_counts_nulls_and_nans_apart():
    s = A.summary_stats([3.0, None, 1.0, float("nan"), 2.0])
    assert (s["n"], s["n_null"], s["min"], s["median"], s["mean"], s["max"]) == (3, 2, 1.0, 2.0, 2.0, 3.0)
    assert A.summary_stats([None])["n"] == 0 and A.summary_stats([])["median"] is None


def test_title_and_layout_rules_are_enforced():
    import matplotlib.pyplot as plt
    with pytest.raises(A.AnalysisError):
        A.wrap_title("x " * 200)                          # > 2 lines at 90 chars
    assert A.wrap_title("short title").count("\n") == 0
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.subplots_adjust(top=0.5, bottom=0.1)              # axes keep 40 % of the height
    with pytest.raises(A.AnalysisError, match="axes keep"):
        A.finish_figure(fig, "t")
    plt.close(fig)
    fig = plt.figure(figsize=(6, 4))
    A.finish_figure(fig, "tile", tile=True)               # a tile figure has no axes and is exempt
    plt.close(fig)
    with pytest.raises(A.AnalysisError, match="no axes"):
        A.finish_figure(plt.figure(figsize=(6, 4)), "no axes, not a tile")
    plt.close("all")


def test_save_figure_writes_png_and_svg_and_closes(tmp_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [0, 1])
    written = A.save_figure(fig, tmp_path / "figs", "demo")
    assert [Path(w).name for w in written] == ["demo.png", "demo.svg"]
    assert all(Path(w).is_file() for w in written) and plt.get_fignums() == []
