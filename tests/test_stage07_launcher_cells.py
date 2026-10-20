"""Stage 07's `--cells` are reproduce/models.yaml's view-cell NAMES (control_lora, difference_relu, …); the Modal launcher plans with
its own cell grammar (contracts.parse_launcher_cell: `none | prompted__<si> | <si>__<arm> | a bare Class B id`). The ONE mapping between
them is lib/models.py launcher_cells — roster_cell_id, the rule 05 writes the wave roster's cells with (the table's `objects.<cell>.cell_id`
is spelled in the launcher's grammar; the arditi cell's rung is stripped), every token checked against the launcher's grammar. Pinned here:
every cell of every phase view of every table model maps to a token the launcher accepts — a DOSE view's object cell to the SAME token as the
wave's parent, served from the dose view's own registry row (`launcher_slug` = `<key>-<view>`, the paper's dose rows' shape) — `none` /
`prompted__*` round-trip unchanged, the demo's mapping is the literal one below, an unknown cell refuses by name, and the CLI prints
`<cell>\t<token>` and the slug."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "reproduce/lib"))
sys.path.insert(0, str(REPO / "src"))
import models as M  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402

DEMO = "qwen35_4b_demo"
#: the worked example's mapping, view cell -> the launcher's --cells token (the table's objects block, through roster_cell_id). A
#: deliberate change to a demo object's cell_id updates this pin with it (difference_relu moved to the paired-write spelling, demo-import M2).
DEMO_TOKENS = {
    "none": "none",
    "prompted__safety_refusal": "prompted__safety_refusal",
    "prompted__si_constitution": "prompted__si_constitution",
    "refusal_direction": "arditi_allpos",                                               # ar_c0p3__arditi_allpos: the rung rides the staged id only
    "control_lora": "constv2ctl2-full__lora_r4_ce",
    "constitution_lora": "constv2-mix-full__lora_r4_ce",
    "difference_lora": "constv2-mix-full__lora_r4_ce_minus_ctl",
    "control_relu": "constv2ctl2-full__psr_ce_all",
    "constitution_relu": "constv2-mix-full__psr_ce_all",
    "difference_relu": "constv2-mix-full__psr_ce_all_minus_ctl",                      # the composed paired write (demo-import M2; the distilled placeholder is gone)
}


def _is_dose_view(m: dict, view: str) -> bool:
    return "dose" in ((m.get("views") or {}).get(view) or {})


def test_every_cell_of_every_phase_view_maps_to_a_token_the_launcher_accepts():
    """The pin the gap closes on: for EVERY table model and every view that lists cells, launcher_cells returns one token per cell and
    contracts.parse_launcher_cell accepts each; `none` is the base cell, a `prompted__*` cell a prompted one, every object cell a FITTED
    one whose arm the launcher names (the bank directory is the token: object_path_for keys on the cell id)."""
    seen_object_cells = 0
    for key, m in M.load_models().items():
        for view, vd in (m.get("views") or {}).items():
            cells = [str(c) for c in ((vd or {}).get("cells") or [])]
            if not cells:
                continue
            mapped = M.launcher_cells(key, view)
            assert [c for c, _t in mapped] == cells, (key, view)
            for cell, tok in mapped:
                parsed = C.parse_launcher_cell(tok)                     # the launcher's own validator: accepts, or the test is red
                if cell == "none":
                    assert tok == "none" and parsed.kind == C.CELL_KIND_BASE
                elif cell.startswith("prompted__"):
                    assert tok == cell and parsed.kind == C.CELL_KIND_PROMPTED and parsed.si == cell[len("prompted__"):]
                else:
                    assert parsed.kind == C.CELL_KIND_FITTED and parsed.arm, (key, view, cell, tok)
                    assert tok == M.roster_cell_id(m, cell)
                    seen_object_cells += 1
    assert seen_object_cells >= 7, "the worked example's seven object cells were expected among the mapped cells"


def test_the_demos_mapping_is_the_pinned_one_in_the_views_order():
    m = M.model(DEMO)
    view = M.baseline_view(DEMO)                                        # the one steered wave (never spelled here: a peer renames it)
    mapped = M.launcher_cells(DEMO, view)
    assert dict(mapped) == DEMO_TOKENS and [c for c, _t in mapped] == [str(c) for c in m["views"][view]["cells"]]
    assert len({t for _c, t in mapped}) == len(mapped), "two cells on one token"
    # a subset keeps the table's spelling per cell and the caller's order
    assert M.launcher_cells(DEMO, view, ["difference_lora", "none"]) == [("difference_lora", DEMO_TOKENS["difference_lora"]), ("none", "none")]


def test_none_and_prompted_cells_round_trip_unchanged_and_the_object_cells_translate():
    view = M.baseline_view(DEMO)
    for cell, tok in M.launcher_cells(DEMO, view):
        if cell == "none" or cell.startswith("prompted__"):
            assert tok == cell
        else:
            assert tok != cell and "__" in tok or tok in C.launchable_class_b_ids(), (cell, tok)


def test_an_unknown_cell_refuses_by_name_before_any_launcher_call():
    view = M.baseline_view(DEMO)
    with pytest.raises(ValueError, match=r"'nope' is not a cell of the view"):
        M.launcher_cells(DEMO, view, ["control_lora", "nope"])
    # a raw launcher token is not a roster cell either: 07 speaks the table's names
    with pytest.raises(ValueError, match=r"'constv2ctl2-full__lora_r4_ce' is not a cell of the view"):
        M.launcher_cells(DEMO, view, ["constv2ctl2-full__lora_r4_ce"])
    with pytest.raises(KeyError, match=r"no view 'nope'"):
        M.launcher_cells(DEMO, "nope")


def test_a_view_cell_without_an_objects_entry_and_a_cell_id_outside_the_grammar_refuse_by_name(tmp_path):
    import yaml
    table = yaml.safe_load((REPO / "reproduce/models.yaml").read_text())
    m = table["models"][DEMO]
    view = M.baseline_view(DEMO)
    # (a) a view names a cell the objects block does not carry
    m["views"][view]["cells"].append("mystery_arm")
    p = tmp_path / "a.yaml"; p.write_text(yaml.safe_dump(table, sort_keys=False))
    with pytest.raises(KeyError, match=r"view cell 'mystery_arm' has no `objects` entry"):
        M.launcher_cells(DEMO, view, ["mystery_arm"], str(p))
    # (b) an objects entry spelled outside the launcher's grammar (an arm the harness never audited)
    m["views"][view]["cells"].pop()
    m["objects"]["control_lora"]["cell_id"] = "constv2ctl2-full__alphakl_0"
    p = tmp_path / "b.yaml"; p.write_text(yaml.safe_dump(table, sort_keys=False))
    with pytest.raises(ValueError, match=r"'control_lora' -> 'constv2ctl2-full__alphakl_0' is outside the launcher's cell grammar.*outside the harness's audited set"):
        M.launcher_cells(DEMO, view, ["control_lora"], str(p))


def test_a_dose_views_object_cells_map_to_the_parents_token_and_the_view_is_served_from_its_own_row():
    """The launcher's grammar carries no dose and a row serves ONE bank under the cell id; a dose view's object cell has the SAME cell id as
    the wave's parent (the SCALED COPY `d<tok>__<cell_id>` is the cluster bank's spelling of the view's bytes, 09_dose_ladder.sh), so the view is
    served the way the paper served its dose rows on Modal — from a registry row of its own: `launcher_slug(key, view)` = `<key>-<view>`,
    whose bank is `<the key's bank>/<view>/` and whose config is the view's front-door file (contracts.MODELS; demo-import 2026-10-12). Every
    dose view of every table model maps its cells (the object cell's token the wave's), and the baseline wave's slug is the key itself."""
    dose_views = 0
    for key, m in M.load_models().items():
        base = M.baseline_view(key) if key in M.front_door_keys() else None
        if base is not None:
            assert M.launcher_slug(key, base) == key
        for view in (m.get("views") or {}):
            if not _is_dose_view(m, view):
                continue
            dose_views += 1
            cells = [str(c) for c in m["views"][view]["cells"]]
            objects = [c for c in cells if c != "none" and not c.startswith("prompted__")]
            assert objects, (key, view)
            mapped = dict(M.launcher_cells(key, view))
            assert list(mapped) == cells
            for c in objects:
                assert mapped[c] == M.roster_cell_id(m, c) and mapped[c] in dict(M.launcher_cells(key, base)).values(), (key, view, c)
                assert M.staged_id(m, c, view).startswith(f"d{M.dose_token_of_view(view)}") and M.staged_id(m, c, view).endswith(f"__{m['objects'][c]['cell_id']}")
            if "none" in cells:
                assert mapped["none"] == "none"
            slug = M.launcher_slug(key, view)
            assert slug == f"{key}-{view}" and slug in C.MODELS, (key, view, slug)
            row = C.model_row(slug)
            assert row.objects_dir == f"{C.model_row(key).objects_dir}/{view}" and row.cfg == C.frontdoor_config_path(key, view)
    assert dose_views >= 2, "the worked example's two dose rungs were expected"
    with pytest.raises(KeyError, match="no view 'nope'"):
        M.launcher_slug(DEMO, "nope")


def test_cli_prints_cell_tab_token_and_refuses_by_name_with_exit_2():
    py = sys.executable
    view = M.baseline_view(DEMO)
    out = subprocess.run([py, str(REPO / "reproduce/lib/models.py"), "launcher-cells", DEMO, view, "all"], capture_output=True, text=True, check=True).stdout
    assert [ln.split("\t") for ln in out.splitlines()] == [[c, t] for c, t in DEMO_TOKENS.items()]
    out = subprocess.run([py, str(REPO / "reproduce/lib/models.py"), "launcher-cells", DEMO, view, "none,difference_lora"], capture_output=True, text=True, check=True).stdout
    assert out == f"none\tnone\ndifference_lora\t{DEMO_TOKENS['difference_lora']}\n"
    r = subprocess.run([py, str(REPO / "reproduce/lib/models.py"), "launcher-cells", DEMO, view, "control_lora,nope"], capture_output=True, text=True)
    assert r.returncode == 2 and r.stdout == "" and "REFUSED: models.yaml:" in r.stderr and "'nope' is not a cell of the view" in r.stderr
    # launcher-slug: the key for the wave, <key>-<view> for a dose view, a refusal by name for a view the key lacks
    assert subprocess.run([py, str(REPO / "reproduce/lib/models.py"), "launcher-slug", DEMO, view], capture_output=True, text=True, check=True).stdout == f"{DEMO}\n"
    m = M.model(DEMO)
    for dose_view in [v for v, vd in m["views"].items() if "dose" in (vd or {})]:
        assert subprocess.run([py, str(REPO / "reproduce/lib/models.py"), "launcher-slug", DEMO, dose_view], capture_output=True, text=True, check=True).stdout == f"{DEMO}-{dose_view}\n"
        out = subprocess.run([py, str(REPO / "reproduce/lib/models.py"), "launcher-cells", DEMO, dose_view, "all"], capture_output=True, text=True, check=True).stdout
        assert [ln.split("\t") for ln in out.splitlines()] == [[c, DEMO_TOKENS[c]] for c in m["views"][dose_view]["cells"]]
    r = subprocess.run([py, str(REPO / "reproduce/lib/models.py"), "launcher-slug", DEMO, "nope"], capture_output=True, text=True)
    assert r.returncode == 2 and "no view 'nope'" in r.stderr
