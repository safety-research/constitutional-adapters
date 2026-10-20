"""Review round 1 (2026-09-10), P2-6 — the XSTest over-refusal control is the 250 SAFE prompts, selected BEFORE
the specs, the item hashes and every cap-ladder input; the 200 contrast rows never enter a llama8b store.

The selection is one config key (`xstest_subset: safe`, emitted by the generator from the roster target), read
by both controls drivers (cell_controls_run / served_capability_run) through ONE helper, honoured by the served
grader when it binds a store, and recorded in the store identity ONLY when set — every historical 450-row
config and store is byte-identical and keys the same.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from sisteer.capeval.tiers import judged
from sisteer.capeval.tiers.judged import JudgedError, TASK_XSTEST

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests"))

import cell_controls_run as ccr  # noqa: E402
import served_capability_run as srun  # noqa: E402
import test_capeval_judged as tcj  # noqa: E402  (FakeJudge, _store, _spec, _spool, _cpi_label_of)
from full22_wave_configs import wave_configs  # noqa: E402  (the wave configs, minus the tables sharing their prefix)

LLAMA_CFGS = [REPO / f"configs/sdfbase_full22_llama8b{s}.yaml" for s in ("", "_it", "_fullconst")]


def _frame():
    import pandas as pd
    prompts = [f"synthetic safe prompt {i}" for i in range(250)] + [f"synthetic contrast prompt {i}" for i in range(200)]
    types = ["homonyms_synth"] * 250 + ["contrast_synth"] * 200
    return pd.DataFrame({"prompt": prompts, "type": types})


@pytest.fixture()
def pinned(monkeypatch):
    """judged.xstest_rows over the synthetic frame (no hub), honouring `subset` exactly as the real loader."""
    df = _frame()
    monkeypatch.setattr(judged, "xstest_rows", lambda pins=None, *, subset=None: judged.xstest_rows_from_frame(df, subset=subset))
    return df


# --------------------------------------------------------------------------- the loader

def test_the_safe_half_is_selected_after_the_gate_with_the_ids_of_the_full_load():
    df = _frame()
    full = judged.xstest_rows_from_frame(df)
    safe = judged.xstest_rows_from_frame(df, subset="safe")
    assert len(full) == 450 and len(safe) == 250 and all(r["subset"] == "safe" for r in safe)
    assert [r["item_id"] for r in safe] == [r["item_id"] for r in full if r["subset"] == "safe"]   # ids unchanged
    assert not any(r["type"].startswith("contrast_") for r in safe)
    assert len(judged.xstest_rows_from_frame(df, subset="unsafe")) == 200
    with pytest.raises(JudgedError, match="unknown subset"):
        judged.xstest_rows_from_frame(df, subset="benign")
    with pytest.raises(JudgedError, match="450|250"):            # the load gate still fires first
        judged.xstest_rows_from_frame(df.iloc[:300], subset="safe")
    with pytest.raises(JudgedError, match="no 'unsafe' rows"):
        judged.xstest_subset_rows([r for r in full if r["subset"] == "safe"], "unsafe")


# --------------------------------------------------------------------------- the drivers

def _cfg(path, **over):
    cfg = yaml.safe_load(path.read_text())
    cfg.update(over)
    return cfg


def test_the_config_key_is_read_once_and_only_safe_is_legal():
    assert ccr.XSTEST_SUBSET_KEY == "xstest_subset"
    assert ccr.xstest_subset_of({}) is None and ccr.xstest_subset_of({"xstest_subset": "safe"}) == "safe"
    with pytest.raises(SystemExit, match="only 'safe'"):
        ccr.xstest_subset_of({"xstest_subset": "unsafe"})
    with pytest.raises(SystemExit, match="only 'safe'"):
        ccr.xstest_subset_of({"xstest_subset": True})


def test_both_drivers_build_the_xstest_spec_over_the_250_safe_rows_when_the_config_says_so(pinned):
    cfg = _cfg(LLAMA_CFGS[0])
    assert cfg["xstest_subset"] == "safe"
    hist = _cfg(REPO / "configs/sdfbase_full22_gemma4e4b.yaml")
    assert "xstest_subset" not in hist
    template = {"t": 1}
    # the HF driver
    s_llama = ccr._judged_specs(cfg, ["xstest"], template, None, 0, "judge")["xstest"]
    s_hist = ccr._judged_specs(hist, ["xstest"], template, None, 0, "judge")["xstest"]
    assert len(s_llama.rows) == 250 and {r["subset"] for r in s_llama.rows} == {"safe"}
    assert len(s_hist.rows) == 450 and {r["subset"] for r in s_hist.rows} == {"safe", "unsafe"}
    # the served driver (the same pinned loader, the same seeded subset)
    v_llama = srun._judged_specs(cfg, ["xstest"], template, None, 0, "judge", None, 0)["xstest"]
    v_hist = srun._judged_specs(hist, ["xstest"], template, None, 0, "judge", None, 0)["xstest"]
    assert [r["item_id"] for r in v_llama.rows] == [r["item_id"] for r in s_llama.rows]
    assert len(v_hist.rows) == 450
    # a --limit seeded subset is drawn FROM the safe half (the selection precedes the subset)
    v_lim = srun._judged_specs(cfg, ["xstest"], template, None, 40, "judge", None, 0)["xstest"]
    assert len(v_lim.rows) == 40 and {r["subset"] for r in v_lim.rows} == {"safe"}


def test_the_store_identity_carries_the_key_only_when_set():
    cell = SimpleNamespace(id="none")
    hist = _cfg(REPO / "configs/sdfbase_full22_gemma4e4b.yaml")
    llama = _cfg(LLAMA_CFGS[0])
    r_hist = ccr._resolved_config(cell, hist, None, 0, "judge")
    r_llama = ccr._resolved_config(cell, llama, None, 0, "judge")
    assert "xstest_subset" not in r_hist and r_llama["xstest_subset"] == "safe"
    assert {k for k in r_llama if k not in r_hist} == {"xstest_subset"}
    route = SimpleNamespace(virtual_model="petri__base", serving_identity="proxy_vllm+passthrough+interleaved")
    v_hist = srun._resolved_config(cell, hist, None, 0, "judge", route=route, model_id="m", model_revision="r")
    v_llama = srun._resolved_config(cell, llama, None, 0, "judge", route=route, model_id="m", model_revision="r")
    assert "xstest_subset" not in v_hist and v_llama["xstest_subset"] == "safe"
    assert {k for k in v_llama if k not in v_hist} == {"xstest_subset"}


# --------------------------------------------------------------------------- the judge + the served grader

def test_judge_xstest_over_a_safe_only_spec_emits_the_safe_aggregates_and_no_unsafe_block(tmp_path):
    rows = [{"item_id": "xst0000-11111111", "prompt": "synthetic safe 0", "subset": "safe", "type": "homonyms_synth"},
            {"item_id": "xst0001-22222222", "prompt": "synthetic safe 1", "subset": "safe", "type": "homonyms_synth"}]
    spec = tcj._spec(TASK_XSTEST, rows)
    spool = tcj._spool(tmp_path, spec)
    spool.append("xst0000-11111111", "sure, here it is")
    spool.append("xst0001-22222222", "I REFUSE")
    store = tcj._store(tmp_path)
    summary = judged.judge_xstest(spool, tcj.FakeJudge(cpi_of=tcj._cpi_label_of), spec, store)
    assert summary["subsets"] == ["safe"] and "unsafe" not in summary
    assert summary["safe"]["n"] == 2 and summary["safe"]["full_refusal"] == pytest.approx(0.5)
    metrics = {r["metric"] for r in store._parsed_rows() if r["item_id"] == "__aggregate__"}
    assert "xst_safe_full_refusal_frac_aggregate" in metrics and "xst_safe_refused_bounds_aggregate" in metrics
    assert not any(m.startswith("xst_unsafe_") for m in metrics)
    # a spec with no SAFE rows is not an XSTest spec
    bad = tcj._spec(TASK_XSTEST, [{"item_id": "xst0002-33333333", "prompt": "c", "subset": "unsafe", "type": "contrast_synth"}])
    bspool = tcj._spool(tmp_path / "b", bad); bspool.append("xst0002-33333333", "sure")
    with pytest.raises(JudgedError, match="no 'safe' items"):
        judged.judge_xstest(bspool, tcj.FakeJudge(cpi_of=tcj._cpi_label_of), bad, tcj._store(tmp_path / "b"))


def test_the_served_grader_binds_a_safe_store_to_the_safe_rows():
    src = (REPO / "scripts/served_capability_judge.py").read_text()
    assert "self.xstest_subset = rc.get(ccr.XSTEST_SUBSET_KEY)" in src
    assert "rows = judged.xstest_subset_rows(rows, self.xstest_subset)" in src
    assert 'blk["xstest_subset"] = getattr(cell, "xstest_subset", None)' in src
    assert "resolved_config.xstest_subset: safe" in src


# --------------------------------------------------------------------------- the configs

def test_every_llama8b_view_config_selects_the_safe_half_and_no_historical_config_does():
    for p in LLAMA_CFGS:
        cfg = yaml.safe_load(p.read_text())
        assert cfg["xstest_subset"] == "safe", p.name
        assert "xstest_subset: safe" in p.read_text() and "the 250 SAFE XSTest prompts only" in p.read_text()
    # the WAVE configs, through the one enumerator (tests/full22_wave_configs.py) that names the tables sharing the
    # prefix; the generated sampled configs are their wave config as data except named keys, `xstest_subset` not
    # among them (test_sdfbase_full22_sampled), so the frozen historical ones are covered there
    for p in wave_configs(REPO):
        if "llama8b" in p.name:
            continue
        assert "xstest_subset" not in yaml.safe_load(p.read_text()), p.name     # the frozen historical configs are untouched
    # the sampled configs embed the wave config: the llama8b ones carry the key, the historical ones do not
    for p in sorted((REPO / "configs").glob("sdfbase_full22_llama8b*_sampled_t*.yaml")):
        assert yaml.safe_load(p.read_text())["xstest_subset"] == "safe", p.name
    roster = yaml.safe_load((REPO / "configs/sdfbase_full22_roster.yaml").read_text())
    assert roster["targets"]["llama8b"]["xstest_subset"] == "safe"
    assert all("xstest_subset" not in t for k, t in roster["targets"].items() if k != "llama8b")


def test_the_roster_reader_refuses_any_selection_but_safe():
    sys.path.insert(0, str(REPO / "scripts/sdfbase/full22"))
    import f22_roster as fr
    roster = yaml.safe_load((REPO / "configs/sdfbase_full22_roster.yaml").read_text())
    assert fr.target_xstest_subset(roster, "llama8b") == "safe" and fr.target_xstest_subset(roster, "gemma4e4b") is None
    bad = json.loads(json.dumps(roster)); bad["targets"]["llama8b"]["xstest_subset"] = "unsafe"
    with pytest.raises(ValueError, match="only 'safe'"):
        fr.target_xstest_subset(bad, "llama8b")
