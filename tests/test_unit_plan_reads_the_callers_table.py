"""unit_plan's predicates read the CALLER's table (05 and 06 pass --models-table): steered_wave_view / baseline_view / the wave
config's view answer from the same table as the model row, never the default or env table (helper S's question, ruled 2026-10-08:
a caller with a non-default table — every tmp-table test, any second table — used to get predicates from one roster and the row from
another). Shown on a tmp copy of the table in which the demo row RUNS IN PHASES (u0 = none + the prompted arms, s1 = every arm):
under it s1 is a steered wave and u0 the baseline; under the default table the demo is single-phase (s1 is the baseline wave)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "reproduce" / "lib"))
import models as M  # noqa: E402
import stage as ST  # noqa: E402

KEY = "qwen35_4b_demo"
SHA = "a" * 40


def _phased_table(tmp_path) -> str:
    t = yaml.safe_load((REPO / "reproduce" / "models.yaml").read_text(encoding="utf-8"))
    row = t["models"][KEY]
    s1 = list(row["views"]["battery"]["cells"])              # the demo's declared wave; the synthetic phased row spells its steered wave in the grammar (s1)
    row["views"] = {"u0": {"cells": [c for c in s1 if c == "none" or c.startswith("prompted__")]}, "s1": {"cells": s1}}
    row.pop("wave", None)                                      # a phased row declares no wave (reproduce/lib/models.check refuses one beside u0)
    p = tmp_path / "models.yaml"
    p.write_text(yaml.safe_dump(t, sort_keys=False), encoding="utf-8")
    return str(p)


def test_the_two_tables_answer_the_single_phase_question_differently(tmp_path, monkeypatch):
    monkeypatch.delenv("SISTEER_MODELS_TABLE", raising=False)
    pt = _phased_table(tmp_path)
    assert M.single_phase(KEY) and M.baseline_view(KEY) == "battery" and not M.steered_wave_view("battery", KEY)
    assert not M.single_phase(KEY, pt) and M.baseline_view(KEY, pt) == "u0" and M.steered_wave_view("s1", KEY, pt)


def test_unit_plan_predicates_follow_the_table_argument_not_the_default(tmp_path, monkeypatch):
    monkeypatch.delenv("SISTEER_MODELS_TABLE", raising=False)
    monkeypatch.setenv("SISTEER_PLAN_SKIP_ARTEFACT_CHECK", "1")     # the tmp phased row has no generated artefacts; the predicates are under test
    pt = _phased_table(tmp_path)
    default_rows = ST.unit_plan(KEY, "battery", families=["gsm8k"], sha=SHA)
    tmp_rows = ST.unit_plan(KEY, "s1", table=pt, families=["gsm8k"], sha=SHA)
    cells = M.model(KEY)["views"]["battery"]["cells"]
    n_unsteered = sum(1 for c in cells if c in M.UNSTEERED_LEGS)             # `none` (the prompted arms are not unsteered legs)
    assert n_unsteered == 1 and len(default_rows) == len(cells)              # single-phase: the one wave carries its unsteered leg
    assert len(tmp_rows) == len(cells) - n_unsteered                          # the tmp table: a steered wave plans no unsteered leg
    ids = lambda rows: {r["args"][2] for r in rows}                           # the roster cell id positional
    assert "none" in ids(default_rows) and "none" not in ids(tmp_rows) and ids(tmp_rows) == ids(default_rows) - {"none"}
    # the wave config the freeze reads is the baseline's — the table's baseline, not the default's
    assert ST._wave_config(KEY)[0].endswith(f"sdfbase_full22_{KEY}_battery.yaml")
    assert ST._wave_config(KEY, table=pt)[0].endswith(f"sdfbase_full22_{KEY}_u0.yaml")


STEERED = [c for c in yaml.safe_load((REPO / "reproduce" / "models.yaml").read_text(encoding="utf-8"))["models"][KEY]["views"]["battery"]["cells"]
           if c != "none" and not c.startswith("prompted__")]


def _roster_world(root: Path, config_cells) -> Path:
    """A roster naming a wave config for KEY/s1 (what view_cells_of reads: the roster's `config`, then its `explicit_cells`)."""
    root.mkdir(parents=True, exist_ok=True)
    cfg = root / f"sdfbase_full22_{KEY}_s1.yaml"
    cfg.write_text(yaml.safe_dump({"explicit_cells": list(config_cells)}), encoding="utf-8")
    roster = root / "sdfbase_full22_roster.yaml"
    roster.write_text(yaml.safe_dump({"targets": {KEY: {"views": {"s1": {"config": str(cfg), "cells": {c: f"base__{c}" for c in config_cells if c != "none"}}}}}}),
                      encoding="utf-8")
    return roster


def test_the_wave_configs_cells_are_read_under_the_callers_table(tmp_path, monkeypatch):
    """The two view_cells_of reads on the plan path (the steered-wave Petri proxy's declared cells; the --cells selection's universe)
    carry the caller's table too: under the tmp phased table s1 is a steered wave, so the proxy's cells are the config's steered
    cells, a config listing `none` refuses by name, and `--cells none` refuses — while the default table (single-phase) keeps the
    proxy on the whole baseline. Before the fix the inner predicate read the default table and `none` rode the proxy's cells."""
    monkeypatch.delenv("SISTEER_MODELS_TABLE", raising=False)
    monkeypatch.setenv("SISTEER_PLAN_SKIP_ARTEFACT_CHECK", "1")
    pt = _phased_table(tmp_path)
    monkeypatch.setenv("F22_ROSTER", str(_roster_world(tmp_path / "clean", STEERED)))
    proxy = [r for r in ST.unit_plan(KEY, "s1", table=pt, families=["petri"], sha=SHA) if r["script"].endswith("petri_proxy.sbatch")]
    assert len(proxy) == 1 and proxy[0]["cells"] == STEERED and "none" not in proxy[0]["cells"]
    assert proxy[0]["args"][ST.CELLS_POSITIONAL[proxy[0]["script"]]] == ",".join(STEERED)
    default = [r for r in ST.unit_plan(KEY, "battery", families=["petri"], sha=SHA) if r["script"].endswith("petri_proxy.sbatch")]
    assert len(default) == 1 and "cells" not in default[0]                       # single-phase: the proxy audits the whole baseline
    with pytest.raises(ValueError):                                                # `none` is not among the steered wave's cells
        ST.unit_plan(KEY, "s1", table=pt, families=["petri"], sha=SHA, cells=["none"])
    monkeypatch.setenv("F22_ROSTER", str(_roster_world(tmp_path / "stale", ["none", *STEERED])))
    with pytest.raises(ValueError, match=r"lists the unsteered leg\(s\) \['none'\] on a steered-wave view"):
        ST.unit_plan(KEY, "s1", table=pt, families=["petri"], sha=SHA)
    src = (REPO / "reproduce" / "lib" / "stage.py").read_text(encoding="utf-8")
    assert "view_cells_of(model_key, view)" not in src and "view_cells_of(model_key, view, roster_path=roster_path)" not in src
