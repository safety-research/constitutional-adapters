"""The HPO table's header carries the design's lr box and its width in decades (the review must see a box narrower than the
protocol's advisory 2.5 decades), the clip constant, and the reconnaissance's certified lr top with its stable / unstable rungs — read
from r1_trials.json and recon.json in the search dir; a search dir without them renders as before."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


def test_header_names_the_box_width_clip_and_certified_top(tmp_path):
    from test_tune_arm import tune_arm  # noqa: F401  — the table imports ONE tune_arm instance
    hpo_table = _load("hpo_table_hdr", REPO / "scripts" / "sdfbase" / "hpo_table.py")
    (tmp_path / "r1_trials.json").write_text(json.dumps({
        "arm": "psr_ce_all", "axes": ["lr"], "box": {"lr": [3.1622776601683795e-05, 0.005623413251903491], "max_grad_norm": [1.5, 1.5]},
        "warmup_levels": [0.15], "init_scheme": "random_lecun_1e-3", "si": "constv2-mix-10m", "round": 1, "search_id": "hpo-native:0123456789abcdef",
        "trials": [{"index": i, "lr": 1e-4, "max_grad_norm": 1.5, "warmup_frac": 0.15} for i in range(16)]}))
    (tmp_path / "recon.json").write_text(json.dumps({"clip": 1.5, "percentile": 95.0, "certified_lr_max": 0.0056234, "certified_lr_log10_max": -2.25,
                                                     "stable_lrs": [3.1623e-05, 0.00017783, 0.001, 0.0056234], "unstable_lrs": [0.031623]}))
    out = tmp_path / "table"
    hpo_table.render(tmp_path, out)
    md = (out / "table.md").read_text(); html = (out / "table.html").read_text(); js = json.loads((out / "table.json").read_text())
    for text in (md, html):
        assert "lr box [3.162e-05, 5.623e-03] (log10 [-4.500, -2.250]; 2.25 decades)" in text, text[:800]
        assert "below the HPO protocol" in text and "the bottom stays (D157)" in text
        assert "clip constant 1.5 (the reconnaissance" in text
        assert "certified lr top 5.6234e-03 (log10 -2.250)" in text and "3.162e-02" in text
    assert js["design"][0].startswith("lr box [3.162e-05, 5.623e-03]") and any("round-1 trials 16" in ln for ln in js["design"])
    # without the design and the verdict the table renders as before (no header block)
    (tmp_path / "r1_trials.json").unlink(); (tmp_path / "recon.json").unlink()
    hpo_table.render(tmp_path, tmp_path / "table2")
    md2 = (tmp_path / "table2" / "table.md").read_text()
    assert "## design" not in md2 and json.loads((tmp_path / "table2" / "table.json").read_text())["design"] == []
