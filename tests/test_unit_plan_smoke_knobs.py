"""The ONE smoke knob the production planner emits (demo-import M10; Q9 ruled): `OVERRIDE_SMOKE_GOALS=<lo>-<hi>/N` in the MODEL's
overrides document narrows the multi-turn targets (target_attacks.sbatch `[nshards]=N [shards]=<lo..hi>`) and the per-view padding
levels (target_static_padded_obj.sbatch `shards=<lo>-<hi>/N`) to that shard range, and touches no other row. `unit_plan` emits the
subset and `expected_units` / `attach_expected` freeze the SAME subset (`coverage.expected` = goals[s::N] over the same N), so the frozen
inventory and the submission stay one plan. It is a PLAN KNOB: `freeze_plan_knobs()` pins it in the inventory header (its unset value is
exactly None, as SISTEER_MT_ATTACKER_GPUS is), `plan --inventory` refuses a differing live value and `refreeze_refusal` refuses a silent
change once attempts exist. Its value is read from the DOCUMENT (the 30-fix reader: OVERRIDE_MODEL scoping), never the bare environment.

Six conditions, in order: (1) no override -> rows byte-identical to today, for every front-door key and the phased
(u0 + s1 + dose rungs) table shape the 35B lane runs under — that lane's table is site-side, so its SHAPE is modelled on the demo key, as
tests/test_unit_plan_reads_the_callers_table.py models it; (2) the unset value is None and a header lacking the key still plans;
(3) every refusal by name; (4) freeze and plan agree, and `plan --inventory` refuses a knob changed after the freeze; (5) the header records
`plan_knobs.OVERRIDE_SMOKE_GOALS` and the document's id / sha256; (6) the Petri, static, controls, mmlu, gsm8k, BoN store, BoN climb and
agentic rows are unchanged under the knob. The views are read through models.baseline_view / models.views, never a literal, so the pins
hold across the demo wave's rename."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, REPRO, VENV_PY, petri_bundle, run_stage  # noqa: E402
from test_admission import goal_fixture, wave_config  # noqa: E402
import test_front_door_wave_integration as WI  # noqa: E402

sys.path.insert(0, str(REPRO / "lib"))
import models as M  # noqa: E402
import stage as ST  # noqa: E402

TABLE = str(REPRO / "models.yaml")
MODEL = "qwen35_4b_demo"                                   # the front-door model of reproduce/models.yaml
VIEW = M.baseline_view(MODEL, TABLE)                       # the inventory's view (the single-phase recipe's one wave), never a literal
SHA = "c" * 40
AB = ["cluster_a", "cluster_b"]
KEY = ST.SMOKE_GOALS_KEY
DEMO_SMOKE = REPRO / "overrides" / "demo_smoke.env"
PAPER = REPRO / "overrides" / "paper_defaults.env"
OTHER_FAMILIES = ("static", "bon", "petri", "agentic", "controls", "mmlu", "gsm8k")      # every family but the two the knob narrows
GOALS = [f"goal-{i:03d}" for i in range(259)]              # test_admission.goal_fixture's pool (the same ids WI.GOAL_IDS spells)


def _doc(tmp: Path, name: str = "t_smoke.env", **kv) -> Path:
    """An overrides document: the smoke's three lines unless a test overrides or drops one (`None` drops the key)."""
    vals = {"OVERRIDE_ID": "t_smoke", ST.HOLD_OUT_MODEL_KEY: MODEL, KEY: "0-0/99"}
    vals.update(kv)
    p = tmp / name
    p.write_text("# synthetic overrides document\n" + "".join(f"{k}={v}\n" for k, v in vals.items() if v is not None))
    return p


def _cli(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    """stage.py as the stages call it, with no overrides document inherited from the runner's shell."""
    e = {k: v for k, v in os.environ.items() if k != ST.RUN_OVERRIDES_ENV}
    e["PYTHONPATH"] = str(REPO / "src") + (os.pathsep + e["PYTHONPATH"] if e.get("PYTHONPATH") else "")
    e.update(env or {})
    return subprocess.run([sys.executable, str(REPRO / "lib" / "stage.py"), *args], capture_output=True, text=True, env=e, cwd=str(REPO), timeout=600)


#: the phased recipe's steered wave is a GRAMMAR name (`s<N>`): a row that declares u0 declares no `wave:` (reproduce/lib/models.py check
#: refuses one — the single-phase recipe's declared wave `battery` cannot sit beside u0), so the phased table below spells its wave `s1`
PHASED_WAVE = "s1"


def _phased_table(tmp: Path) -> str:
    """The demo row re-declared in PHASES (u0 = none + the prompted arms; the steered wave `s1` = every arm; the dose rungs kept; no
    `wave:`) — the shape the 35B lane's site table has (tests/test_unit_plan_reads_the_callers_table.py's construction), under the same
    key so the decode-policy and roster lookups the multi-turn rows make still resolve."""
    t = yaml.safe_load((REPRO / "models.yaml").read_text(encoding="utf-8"))
    row = t["models"][MODEL]
    row.pop("wave", None)
    wave = list(row["views"][VIEW]["cells"])
    row["views"] = {"u0": {"cells": [c for c in wave if c == "none" or c.startswith("prompted__")]}, PHASED_WAVE: {"cells": wave},
                    **{v: dict(d) for v, d in row["views"].items() if v != VIEW}}
    p = tmp / "models_phased.yaml"
    p.write_text(yaml.safe_dump(t, sort_keys=False), encoding="utf-8")
    return str(p)


def _targets(rows):
    return [r for r in rows if r["family"] == "multiturn" and r["kind"] == "target"]


def _families_of(key: str, view: str, table: str):
    """Every family — except the Petri proxy of a STEERED-WAVE view, which declares the roster's cells for the view (view_cells_of) and the
    committed roster carries no demo views until 05 --step roster renders them; the proxy row is pinned on the baseline view instead."""
    return None if not M.steered_wave_view(view, key, table) else [f for f in ST.FAMILIES if f != "petri"]


def _attacks_of(key: str, view: str, table: str):
    """The multi-turn attacks a view plans: a DOSE view (a `views` entry declaring `dose:`) the Appendix D.2 ladder's two, crescendo +
    actorattack, never pair (stage 09 runs the dose view's multiturn family through 06, so the rung's pool is sized from the rung's own
    legs); every other view the wave's three."""
    return ST.DOSE_LADDER_MT_ATTACKS if ST.is_dose_view_row(M.model(key, table), view) else ST.MT_ATTACKS


def _pin_todays_spellings(rows, view: str, attacks=ST.MT_ATTACKS):
    """The paper's rows as they read today (tests/test_shard_groups.py:46-56 and the 06 dry-run pins): nshards 9 on the fifth multi-turn
    positional, the four shard groups a disjoint cover of 0..8, the padding `3` and its three shards, static's nshards 9 — over the
    view's own attacks (_attacks_of), and no target of any other."""
    assert sorted({r["args"][2] for r in _targets(rows)}) == sorted(attacks), (view, sorted({r["args"][2] for r in _targets(rows)}))
    for atk in attacks:
        mine = [r for r in _targets(rows) if r["args"][2] == atk]
        assert [r["args"][4] for r in mine] == ["9"] * 4, mine
        groups = [tuple(int(x) for x in r["args"][5].split(",")) for r in mine]
        assert sorted(s for g in groups for s in g) == list(range(9)) and groups == [tuple(int(x) for x in g.split(",")) for g in ST.DEFAULT_SHARD_GROUPS]
        assert all(r["expected_shards"][atk] == [f"shard{s:02d}" for s in g] for r, g in zip(mine, groups))
    for r in (r for r in rows if r["family"] == "padding"):
        if view == "u0":
            assert r["args"][5] == "3" and r["expected_shards"] == {e: ["shard00", "shard01", "shard02"] for e in ST.STATIC_EVALS}
        else:
            assert r["args"][5] == "3" and r["args"][6:] == ["all", view] and r["expected_shards"] == {"grok420": ["shard00", "shard01", "shard02"]}
    for r in (r for r in rows if r["family"] == "static"):
        assert r["args"][3] == "9" and all(len(v) == 9 for v in r["expected_shards"].values())


# ---------------------------------------------------------------------------------------------------- (1) no override: byte-identical
def test_no_override_plans_every_front_door_view_byte_identically_to_today(tmp_path, monkeypatch):
    monkeypatch.delenv(ST.RUN_OVERRIDES_ENV, raising=False)
    monkeypatch.setenv("SISTEER_PLAN_SKIP_ARTEFACT_CHECK", "1")        # the dose views' generated artefacts are not under test here
    keys = M.front_door_keys(TABLE)
    assert MODEL in keys
    for key in keys:
        for view in M.views(key, TABLE):
            fams = _families_of(key, view, TABLE)
            plain = ST.unit_plan(key, view, table=TABLE, sha=SHA, clusters=AB, families=fams)
            assert plain == ST.unit_plan(key, view, table=TABLE, sha=SHA, clusters=AB, families=fams, smoke_goals=None)
            assert not any(KEY in json.dumps(r) or "/99" in json.dumps(r["args"]) for r in plain)
            _pin_todays_spellings(plain, view, _attacks_of(key, view, TABLE))
    # the CLI, as 06 calls it: no document, the paper's document, and a smoke-free document naming the model all print the SAME bytes
    base = _cli("plan", "--model", MODEL, "--view", VIEW, "--sha", SHA, "--clusters", ",".join(AB), "--models-table", TABLE, "--json")
    assert base.returncode == 0, base.stderr[-1500:]
    for doc in (str(PAPER), str(_doc(tmp_path, "no_knob.env", **{KEY: None, "OVERRIDE_ID": "no_knob"}))):
        r = _cli("plan", "--model", MODEL, "--view", VIEW, "--sha", SHA, "--clusters", ",".join(AB), "--models-table", TABLE, "--json", "--overrides", doc)
        assert r.returncode == 0 and r.stdout == base.stdout, (doc, r.stderr[-1500:])
    # $SISTEER_RUN_OVERRIDES is the default document (the stage's export): the paper's path through it is the same bytes too
    r = _cli("plan", "--model", MODEL, "--view", VIEW, "--sha", SHA, "--clusters", ",".join(AB), "--models-table", TABLE, "--json", env={ST.RUN_OVERRIDES_ENV: str(PAPER)})
    assert r.returncode == 0 and r.stdout == base.stdout, r.stderr[-1500:]


def test_no_override_plans_the_phased_table_shape_byte_identically_to_today(tmp_path, monkeypatch):
    """The 35B lane's shape: a phased table (u0 baseline, the steered wave, the dose rungs). Its table is site-side, so the shape is
    modelled on the demo key (test_unit_plan_reads_the_callers_table.py's construction)."""
    monkeypatch.delenv(ST.RUN_OVERRIDES_ENV, raising=False)
    monkeypatch.setenv("SISTEER_PLAN_SKIP_ARTEFACT_CHECK", "1")
    pt = _phased_table(tmp_path)
    assert M.check(pt) == [] and M.baseline_view(MODEL, pt) == "u0" and M.steered_wave_view(PHASED_WAVE, MODEL, pt)
    for view in M.views(MODEL, pt):
        fams = _families_of(MODEL, view, pt)
        plain = ST.unit_plan(MODEL, view, table=pt, sha=SHA, clusters=AB, families=fams)
        assert plain == ST.unit_plan(MODEL, view, table=pt, sha=SHA, clusters=AB, families=fams, smoke_goals=None)
        _pin_todays_spellings(plain, view, _attacks_of(MODEL, view, pt))
    u0 = ST.unit_plan(MODEL, "u0", table=pt, sha=SHA, clusters=AB, families=["padding"])
    assert [r["script"] for r in u0] == ["scripts/sdfbase/full22/static_padding/target_static_padded.sbatch"] * 5        # the u0 launcher, count 3


# ---------------------------------------------------------------------------------------------------- (2) unset = None; a header without the key plans
def test_the_knobs_unset_value_is_exactly_none_and_a_header_lacking_the_key_still_plans(tmp_path, monkeypatch):
    monkeypatch.delenv(ST.RUN_OVERRIDES_ENV, raising=False)
    knobs = ST.freeze_plan_knobs()
    assert knobs[KEY] is None and knobs["SISTEER_MT_ATTACKER_GPUS"] is None and ST.freeze_plan_knobs(None) == knobs
    assert ST.smoke_goals_header(None) is None
    sg = ST.smoke_goals(str(DEMO_SMOKE), MODEL)
    assert ST.freeze_plan_knobs(sg)[KEY] == "0-0/99" and {k: v for k, v in ST.freeze_plan_knobs(sg).items() if k != KEY} == {k: v for k, v in knobs.items() if k != KEY}
    # a header frozen before the knob existed: the three knobs, no OVERRIDE_SMOKE_GOALS key — the union diff is empty (None == absent)
    legacy = {k: v for k, v in knobs.items() if k != KEY}
    assert ST.plan_knob_diff(legacy, ST.freeze_plan_knobs()) == {}
    assert ST.plan_knob_diff(legacy, ST.freeze_plan_knobs(sg)) == {KEY: (None, "0-0/99")}
    state = tmp_path / "state"; (state / "gate").mkdir(parents=True)
    inv = state / "gate" / "expected_units.json"
    ST.write_atomic(str(inv), {"schema": 2, "model": MODEL, "view": VIEW, "code_sha": SHA, "frozen_utc": "2026-01-02T00:00:00Z", "plan_knobs": legacy, "units": []})
    ok = _cli("plan", "--model", MODEL, "--view", VIEW, "--families", "mmlu", "--sha", SHA, "--models-table", TABLE, "--inventory", str(inv), "--json")
    assert ok.returncode == 0 and json.loads(ok.stdout)[0]["family"] == "mmlu", ok.stderr[-1500:]
    # … and the same header refuses a live plan that TURNS the knob on (the inventory pins nothing of the sort)
    bad = _cli("plan", "--model", MODEL, "--view", VIEW, "--families", "mmlu", "--sha", SHA, "--models-table", TABLE, "--inventory", str(inv), "--json", "--overrides", str(DEMO_SMOKE))
    assert bad.returncode == 2 and f"{KEY} pinned None, live '0-0/99'" in bad.stderr and "freeze a NEW wave" in bad.stderr, bad.stderr[-1500:]


# ---------------------------------------------------------------------------------------------------- (3) every refusal by name
REFUSALS = [
    ({"OVERRIDE_ID": "paper_defaults"}, r"OVERRIDE_ID='paper_defaults' names no smoke"),
    ({"OVERRIDE_ID": None}, r"OVERRIDE_ID='\(unset\)' names no smoke"),
    ({ST.HOLD_OUT_MODEL_KEY: None}, r"OVERRIDE_MODEL is empty — a smoke is ONE model's recipe"),
    ({ST.HOLD_OUT_MODEL_KEY: "gemma4e4b"}, r"OVERRIDE_MODEL='gemma4e4b' but this plan / inventory / stage runs model 'qwen35_4b_demo'"),
    ({KEY: "0-0/259"}, r"N=259 > 99 — the padding store names shards shard00\.\.shard99"),
    ({KEY: "0-0/100"}, r"N=100 > 99"),
    ({KEY: "1-0/99"}, r"lo 1 > hi 0 \(inverted range\)"),
    ({KEY: "0-99/99"}, r"hi 99 is out of range for N=99 \(the shards are 0\.\.98\)"),
    ({KEY: "0-0/0"}, r"N must be >= 1"),
    ({KEY: "99"}, r"is not `<lo>-<hi>/N`"),
    ({KEY: "0-0"}, r"is not `<lo>-<hi>/N`"),
    ({KEY: "a-b/c"}, r"is not `<lo>-<hi>/N`"),
    ({KEY: "0,1/99"}, r"is not `<lo>-<hi>/N`"),
]


@pytest.mark.parametrize("kv,match", REFUSALS, ids=[f"{list(kv)[0]}={list(kv.values())[0]}" for kv, _ in REFUSALS])
def test_the_reader_refuses_by_name(tmp_path, kv, match):
    with pytest.raises(ValueError, match=match):
        ST.smoke_goals(str(_doc(tmp_path, **kv)), MODEL)


def test_the_reader_reads_nothing_from_a_knob_free_document_and_refuses_a_missing_file(tmp_path):
    assert ST.smoke_goals(None) is None and ST.smoke_goals("") is None and ST.smoke_goals(str(PAPER), MODEL) is None
    assert ST.smoke_goals(str(_doc(tmp_path, **{KEY: None})), MODEL) is None                       # the model named, no knob: the paper's rows
    assert ST.smoke_goals(str(_doc(tmp_path, **{KEY: None, ST.HOLD_OUT_MODEL_KEY: "gemma4e4b"})), MODEL) is None   # another model's knob-free document stamps nothing
    with pytest.raises(ValueError, match="does not exist"):
        ST.smoke_goals(str(tmp_path / "absent.env"), MODEL)
    sg = ST.smoke_goals(str(_doc(tmp_path, **{KEY: " 0-2/16 ", "OVERRIDE_ID": "lane_SMOKE_v2"})), MODEL)     # the id check is case-insensitive; the spec is normalised
    assert sg["spec"] == "0-2/16" and (sg["lo"], sg["hi"], sg["nshards"]) == (0, 2, 16) and sg["shards"] == ["shard00", "shard01", "shard02"]
    assert sg["override_id"] == "lane_SMOKE_v2" and sg["model"] == MODEL and sg["overrides_path"] == "t_smoke.env" and len(sg["overrides_sha256"]) == 64
    assert ST.smoke_goals(str(_doc(tmp_path)), None)["spec"] == "0-0/99"                       # no caller model: the document's own scoping stands


def test_the_knob_does_not_reach_the_u0_padding_launcher_and_shard_groups_relax_only_under_the_knob(tmp_path, monkeypatch):
    monkeypatch.setenv("SISTEER_PLAN_SKIP_ARTEFACT_CHECK", "1")
    sg = ST.smoke_goals(str(DEMO_SMOKE), MODEL)
    pt = _phased_table(tmp_path)
    with pytest.raises(ValueError, match=r"does not reach the u0 padding launcher .*target_static_padded\.sbatch takes a shard count, not a range"):
        ST.unit_plan(MODEL, "u0", table=pt, sha=SHA, clusters=AB, families=["padding"], smoke_goals=sg)
    u0_mt = _targets(ST.unit_plan(MODEL, "u0", table=pt, sha=SHA, clusters=AB, families=["multiturn"], smoke_goals=sg))
    assert [r["args"][4:6] for r in u0_mt] == [["99", "0"]] * 3                                   # u0's multi-turn targets take it
    # shard_groups: the cover rule holds without the knob and is relaxed to the smoke's ONE range with it, at the smoke's N only
    with pytest.raises(ValueError, match="not a disjoint cover"):
        ST.shard_groups(99, "0")
    assert ST.shard_groups(99, "0", smoke=sg) == [(0,)] and ST.shard_groups(99, smoke=sg) == [(0,)]
    with pytest.raises(ValueError, match=r"nshards 9 is not the smoke knob's N 99"):
        ST.shard_groups(9, smoke=sg)
    sg3 = ST.smoke_goals(str(_doc(tmp_path, **{KEY: "2-4/16"})), MODEL)
    assert ST.shard_groups(16, smoke=sg3) == [(2, 3, 4)]
    rows = ST.unit_plan(MODEL, VIEW, table=TABLE, sha=SHA, clusters=AB, families=["multiturn", "padding"], smoke_goals=sg3)
    assert {r["args"][5] for r in _targets(rows)} == {"2,3,4"} and {r["unit"].split("/")[1] for r in _targets(rows)} == {"g234"}
    assert all(r["expected_shards"] == {r["args"][2]: ["shard02", "shard03", "shard04"]} for r in _targets(rows))
    assert all(r["args"][5] == "2-4/16" and r["expected_shards"] == {"grok420": ["shard02", "shard03", "shard04"]} for r in rows if r["family"] == "padding")


def test_every_cli_entry_point_refuses_a_bad_document_by_name(tmp_path, monkeypatch):
    monkeypatch.delenv(ST.RUN_OVERRIDES_ENV, raising=False)
    bad = str(_doc(tmp_path, **{KEY: "0-0/259"}))
    r = _cli("plan", "--model", MODEL, "--view", VIEW, "--sha", SHA, "--models-table", TABLE, "--json", "--overrides", bad)
    assert r.returncode == 2 and r.stderr.startswith("REFUSED: plan: ") and "N=259 > 99" in r.stderr, r.stderr[-1500:]
    r = _cli("freeze-inventory", "--model", MODEL, "--models-table", TABLE, "--code-sha", SHA, "--out", str(tmp_path / "gate" / "expected_units.json"), "--dry-run", "--overrides", bad)
    assert r.returncode == 2 and r.stderr.startswith("REFUSED: freeze-inventory: ") and "N=259 > 99" in r.stderr, r.stderr[-1500:]
    r = _cli("placement-manifest", "ensure", "--model", MODEL, "--view", VIEW, "--clusters", ",".join(AB), "--manifest", str(tmp_path / "placement.json"), "--dry-run", "--overrides", bad)
    assert r.returncode == 2 and r.stderr.startswith("REFUSED: placement-manifest: ") and "N=259 > 99" in r.stderr, r.stderr[-1500:]
    # the same three under $SISTEER_RUN_OVERRIDES (the stage's export), with a document for another model
    other = str(_doc(tmp_path, "other.env", **{ST.HOLD_OUT_MODEL_KEY: "gemma4e4b"}))
    for args in (("plan", "--model", MODEL, "--view", VIEW, "--sha", SHA, "--models-table", TABLE, "--json"),
                 ("freeze-inventory", "--model", MODEL, "--models-table", TABLE, "--code-sha", SHA, "--out", str(tmp_path / "gate" / "expected_units.json"), "--dry-run"),
                 ("placement-manifest", "ensure", "--model", MODEL, "--view", VIEW, "--clusters", ",".join(AB), "--manifest", str(tmp_path / "placement.json"), "--dry-run")):
        r = _cli(*args, env={ST.RUN_OVERRIDES_ENV: other})
        assert r.returncode == 2 and "never travels to another model" in r.stderr, (args[0], r.stderr[-1500:])


# ---------------------------------------------------------------------------------------------------- (4) + (6) freeze and plan agree (offline)
def test_the_freeze_and_the_plan_emit_the_same_subset_and_every_other_family_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv(ST.RUN_OVERRIDES_ENV, raising=False)
    fx = dict(goals_jsonl=str(goal_fixture(tmp_path)), wave_config=str(wave_config(tmp_path)), petri_bundle=str(petri_bundle(tmp_path)), labels=AB)
    paper_units = ST.expected_units(MODEL, SHA, TABLE, **fx)
    assert paper_units == ST.expected_units(MODEL, SHA, TABLE, overrides=str(PAPER), **fx)             # no override: the frozen units are byte-identical
    smoke_units = ST.expected_units(MODEL, SHA, TABLE, overrides=str(DEMO_SMOKE), **fx)
    sg = ST.smoke_goals(str(DEMO_SMOKE), MODEL)
    plan = {r["logical_work_id"]: r for r in ST.unit_plan(MODEL, VIEW, table=TABLE, sha=SHA, clusters=AB, smoke_goals=sg) if r["registers"]}
    assert set(plan) == {u["logical_work_id"] for u in smoke_units}
    for u in smoke_units:
        row = plan[u["logical_work_id"]]
        assert u["workload"]["args"] == ST.workload_identity_of_row(row, SHA)["args"] and u["workload"]["env"] == row["env"]
        assert {e: ev["expected_shards"] for e, ev in u["evals"].items()} == row["expected_shards"]
    by_fam = lambda units, fam: sorted((u for u in units if u["family"] == fam), key=lambda u: u["logical_work_id"])     # noqa: E731
    mt = by_fam(smoke_units, "multiturn")
    assert [u["unit"] for u in mt] == [f"{atk}/g0" for atk in sorted(ST.MT_ATTACKS)]                     # ONE target per attack, shard 0 of 99
    for u in mt:
        atk = u["unit"].split("/")[0]; ev = u["evals"][atk]
        assert ev["expected_shards"] == ["shard00"] and ev["coverage"]["expected"] == {"shard00": GOALS[0::99]} and len(GOALS[0::99]) == 3
        assert "goals[s::99]" in ev["coverage"]["expected_source"] and u["workload"]["args"][4:6] == ["99", "0"]
    pad = by_fam(smoke_units, "padding")
    assert sorted(u["padding_tokens"] for u in pad) == sorted(ST.PADDING_LEVELS)
    for u in pad:
        ev = u["evals"]["grok420"]
        assert ev["expected_shards"] == ["shard00"] and ev["coverage"]["expected"] == {"shard00": GOALS[0::99]} and "goals[s::99]" in ev["coverage"]["expected_source"]
        assert u["workload"]["args"][5:] == ["0-0/99", "all", VIEW]
    # the paper's freeze slices goals[s::9] over nine shards (multi-turn) and goals[s::3] over three (padding) — today's spellings
    for u in by_fam(paper_units, "multiturn"):
        atk = u["unit"].split("/")[0]
        assert "goals[s::9]" in u["evals"][atk]["coverage"]["expected_source"] and set(u["evals"][atk]["coverage"]["expected"]) == set(u["evals"][atk]["expected_shards"])
    for u in by_fam(paper_units, "padding"):
        assert "goals[s::3]" in u["evals"]["grok420"]["coverage"]["expected_source"] and u["evals"]["grok420"]["expected_shards"] == ["shard00", "shard01", "shard02"]
    # (6) every other family's frozen units are UNCHANGED under the knob (the BoN store and climb, static's nine shards, Petri, agentic, …)
    for fam in OTHER_FAMILIES:
        assert by_fam(smoke_units, fam) == by_fam(paper_units, fam), fam
    for u in by_fam(smoke_units, "static"):
        assert all(ev["expected_shards"] == [f"shard{i:02d}" for i in range(9)] and "goals[s::9]" in ev["coverage"]["expected_source"] for ev in u["evals"].values())


def test_every_other_familys_plan_rows_are_unchanged_under_the_knob():
    sg = ST.smoke_goals(str(DEMO_SMOKE), MODEL)
    for fam in OTHER_FAMILIES:
        assert ST.unit_plan(MODEL, VIEW, table=TABLE, sha=SHA, clusters=AB, families=[fam], smoke_goals=sg) == ST.unit_plan(MODEL, VIEW, table=TABLE, sha=SHA, clusters=AB, families=[fam]), fam
    full = ST.unit_plan(MODEL, VIEW, table=TABLE, sha=SHA, clusters=AB, smoke_goals=sg)
    plain = ST.unit_plan(MODEL, VIEW, table=TABLE, sha=SHA, clusters=AB)
    other = lambda rows: [r for r in rows if r["family"] not in ("multiturn", "padding")]      # noqa: E731
    assert other(full) == other(plain) and len(full) < len(plain)
    store = next(r for r in full if r["family"] == "bon" and r["unit"] == "store512"); climb = next(r for r in full if r["unit"] == "climb")
    assert store["args"][4] == "9" and climb["array"] == "0-3" and climb["args"][3] == "4"                   # the BoN 512 store and the WHOLE climb (no --array 0-0)
    assert next(r for r in full if r["family"] == "petri")["args"] == [MODEL, VIEW, SHA]                     # Petri runs whole, on the production drive
    assert all(r["args"][3] == "9" for r in full if r["family"] in ("static", "controls"))                  # static / controls keep nshards 9
    # the pool is sized from the smoke's targets: three targets over two labels -> one member per label (one per three concurrent targets)
    pool = [r for r in full if r["kind"] == "attacker"]
    assert sorted(r["cluster"] for r in pool) == AB and len(_targets(full)) == 3
    assert sorted(r["cluster"] for r in plain if r["kind"] == "attacker") == ["cluster_a", "cluster_a", "cluster_b", "cluster_b"]


# ---------------------------------------------------------------------------------------------------- the knob renames no store; the daemon dispatches it
def _paired_smoke_and_paper_rows():
    """(smoke row, the paper row of the same store) for every multi-turn target and padding level the knob narrows: the paper's shard-0
    group of the attack, and the paper's level of the same pad."""
    sg = ST.smoke_goals(str(DEMO_SMOKE), MODEL)
    smoke = ST.unit_plan(MODEL, VIEW, table=TABLE, sha=SHA, clusters=AB, smoke_goals=sg)
    plain = ST.unit_plan(MODEL, VIEW, table=TABLE, sha=SHA, clusters=AB)
    pairs = []
    for r in _targets(smoke):
        atk = r["args"][2]
        pairs.append((r, next(p for p in _targets(plain) if p["args"][2] == atk and p["args"][5].split(",")[0] == "0")))
    for r in (r for r in smoke if r["family"] == "padding"):
        pairs.append((r, next(p for p in plain if p["family"] == "padding" and p["padding_tokens"] == r["padding_tokens"])))
    assert len(pairs) == len(ST.MT_ATTACKS) + len(ST.PADDING_LEVELS)
    return pairs


def test_the_knob_emits_the_launchers_shard_positionals_into_the_same_stores_under_a_different_registry_identity():
    """The smoke writes the SAME physical stores the paper's rows do — `<wave root>/transcripts_results/sdfbase_full22_<tag>/<attack>/<cell>`
    and `<wave root>/static_padding/transcripts_results/pad_static_<key>_<view>_p<pad>/grok420/<cell>` — and differs from them in exactly the
    launchers' shard positionals (target_attacks.sbatch `[nshards] [shards]`; target_static_padded_obj.sbatch `shards=<lo>-<hi>/N`), the
    shards it expects and the goals it freezes. That is the design, not an omission: the store path is the PRODUCER's and
    job-independent by construction (target_attacks.sbatch's resume: `$WAVE_ROOT/transcripts_results/<private_run>/
    <attack>/<cell>/shard*`); what tells a smoke's data from a confirmatory run's is the REGISTRY identity the sbatch bakes from these very
    positionals (`sh<shards>of<nshards>`, target_attacks.sbatch: "a one-goal smoke (nshards 259, shard 0) and a confirmatory
    run (nshards 9, shard 0) are different data, so they are different runs"); a store laid out under
    another N is refused BY NAME on the cluster (the padding launcher's shard leases; the closer's shard / goal-set checks) and a re-plan under
    one state root by `refreeze_refusal`. The planner renames nothing: a store namespace would be a producer, config, closer, grader,
    coverage and daemon change — the runner-change class Q9 refused."""
    for smoke, paper in _paired_smoke_and_paper_rows():
        same = ("script", "family", "stage_family", "model", "view", "kind", "env", "evals", "padding_tokens", "registers")
        assert {k: smoke.get(k) for k in same} == {k: paper.get(k) for k in same}, smoke["logical_work_id"]
        for e, cov in smoke["coverage"].items():
            assert cov["private_run"] == paper["coverage"][e]["private_run"] and cov["path"] == paper["coverage"][e]["path"], e
            assert cov["layout"] == paper["coverage"][e]["layout"] and cov["host"] == paper["coverage"][e]["host"]
        if smoke["family"] == "multiturn":
            assert smoke["args"][:4] == paper["args"][:4] and smoke["args"][6:] == paper["args"][6:]
            assert smoke["args"][4:6] == ["99", "0"] and paper["args"][4:6] == ["9", "0,1,2"]            # the sbatch's documented smoke / confirmatory forms
            assert smoke["logical_work_id"] != paper["logical_work_id"]                                   # <atk>/g0 vs <atk>/g012: a different unit
            atk = smoke["args"][2]
            assert smoke["coverage"][atk]["private_run"] == f"sdfbase_full22_{MODEL}_{VIEW}"
            assert smoke["coverage"][atk]["path"]["root_rel"] == f"{ST.WAVE_ROOT_REL}/{ST.PRIVATE_ROOT_REL}/sdfbase_full22_{MODEL}_{VIEW}/{atk}/<cell>"
        else:
            assert smoke["args"][:5] == paper["args"][:5] and smoke["args"][6:] == paper["args"][6:] and smoke["name"] == paper["name"]
            assert smoke["args"][5] == "0-0/99" and paper["args"][5] == "3" and smoke["logical_work_id"] == paper["logical_work_id"]
            prun = f"pad_static_{MODEL}_{VIEW}_p{smoke['padding_tokens']}"
            assert smoke["coverage"]["grok420"]["private_run"] == prun
            assert smoke["coverage"]["grok420"]["path"]["root_rel"] == f"{ST.WAVE_ROOT_REL}/{ST.PADDING_PRIVATE_ROOT_REL}/{prun}/grok420/<cell>"
        assert smoke["expected_shards"] != paper["expected_shards"]


def test_every_store_the_knob_narrows_dispatches_through_the_grading_daemon_to_its_generators_config():
    """Every private run the smoke's rows name — the multi-turn wave run and the five padding levels — is one `grade_daemon.target_for` /
    `config_for` resolve to the `--target` and the config the generators spell for it (make_full22_configs names the wave config after
    `private_run`; make_padding_configs.padded_config / private_run name the padded pair), so 08's daemon invokes the two graders the
    smoke made subset-aware with the right config (the demo's padded private run is split by the daemon into the key and the view it
    pads). The view is read through models.baseline_view, never a literal: on a tree whose demo wave is a grammar name (`s1`) the
    daemon's `_PADDED_RUN_RE` splits it; on the tree that declares the wave (`battery`, M3) the daemon consults the models table —
    either way the resolution must land on the generator's spelling, or the smoke's padded levels stay pending for ever."""
    import importlib.util
    for p in (REPO / "scripts", REPO / "scripts" / "sdfbase" / "full22" / "static_padding"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location("grade_daemon_smoke_pin", REPO / "scripts" / "registry" / "grade_daemon.py")
    gd = importlib.util.module_from_spec(spec); spec.loader.exec_module(gd)
    import make_padding_configs as gen
    configs_root = str(REPO / "configs")
    sg = ST.smoke_goals(str(DEMO_SMOKE), MODEL)
    rows = ST.unit_plan(MODEL, VIEW, table=TABLE, sha=SHA, clusters=AB, smoke_goals=sg)
    seen = set()
    for r in rows:
        if r["family"] not in ("multiturn", "padding") or r["kind"] != "target":
            continue
        for e, cov in r["coverage"].items():
            prun = cov["private_run"]
            # store_root_of's reading of the row's store: the private run is the component right under the family's root
            assert cov["path"]["root_rel"].split("/")[-3] == prun, cov["path"]
            fam = gd.family_of({"eval_category": "multi_turn" if r["family"] == "multiturn" else "static_attack", "padding_tokens": r.get("padding_tokens")})
            if r["family"] == "multiturn":
                assert fam == "multiturn" and gd.target_for(fam, prun, MODEL) == f"{MODEL}_{VIEW}"
                assert gd.config_for(fam, configs_root=configs_root, private_run=prun) == os.path.join(configs_root, f"{prun}.yaml")
            else:
                assert fam == "padding" and gd.target_for(fam, prun, MODEL) == MODEL, prun
                want = os.path.join(configs_root, "sdfbase", "static_padding", gen.padded_config(MODEL, r["padding_tokens"], VIEW).name)
                assert gd.config_for(fam, configs_root=configs_root, private_run=prun) == want, prun
                assert prun == gen.private_run(MODEL, r["padding_tokens"], VIEW)
            seen.add(prun)
    assert len(seen) == 1 + len(ST.PADDING_LEVELS)


# ---------------------------------------------------------------------------------------------------- (4) + (5) through the stages: 05 freezes, 06 plans
def test_05_freezes_the_knob_into_the_header_and_06_refuses_a_dropped_or_changed_knob(tmp_path):
    w = WI.World(tmp_path)
    doc_rel = "reproduce/overrides/demo_smoke.env"
    doc = w.repo / doc_rel
    r = w.stage("05_stage_roster.sh", "--model", MODEL, "--step", "freeze", "--overrides", doc_rel, SISTEER_RUN_OVERRIDES="")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    inv = w.inventory()
    # (5) the header: the pinned knob beside the three knobs, and the document's id / sha256 / basename / model
    assert inv["plan_knobs"][KEY] == "0-0/99" and set(inv["plan_knobs"]) == {"SISTEER_MT_SEED_BASE", "SISTEER_MT_ATTACKER_GPUS", "SISTEER_SHARD_GROUPS", KEY}
    assert inv["smoke_goals"] == {"spec": "0-0/99", "lo": 0, "hi": 0, "nshards": 99, "shards": ["shard00"], "override_id": "demo_smoke", "overrides_path": "demo_smoke.env",
                                  "overrides_sha256": hashlib.sha256(doc.read_bytes()).hexdigest(), "model": MODEL}
    assert inv["shard_groups"] == [[0]] and inv["nshards"] == 9 and inv["view"] == VIEW
    mt = [u for u in inv["units"] if u["family"] == "multiturn"]
    assert sorted(u["unit"] for u in mt) == sorted(f"{atk}/g0" for atk in ST.MT_ATTACKS)
    for u in mt:
        atk = u["unit"].split("/")[0]
        assert u["evals"][atk]["coverage"]["expected"] == {"shard00": WI.GOAL_IDS[0::99]} and u["workload"]["args"][4:6] == ["99", "0"]
    for u in (u for u in inv["units"] if u["family"] == "padding"):
        assert u["evals"]["grok420"]["coverage"]["expected"] == {"shard00": WI.GOAL_IDS[0::99]} and u["workload"]["args"][5] == "0-0/99"
    man = json.load(open(ST.placement_path(str(w.state / "submissions.jsonl"), VIEW)))
    assert sorted(lw.split("/", 3)[3] for lw, p in man["units"].items() if p["kind"] == "target") == sorted(f"{atk}/g0" for atk in ST.MT_ATTACKS)
    assert sorted(lw.rsplit("/", 1)[1] for lw, p in man["units"].items() if p["kind"] == "attacker") == ["1", "1"]      # one pool member per label
    # (4) 06 under the SAME document: the live plan equals the frozen inventory and the smoke positionals go out
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--view", VIEW, "--family", "multiturn,padding", "--dry-run", "--overrides", doc_rel, SISTEER_RUN_OVERRIDES="")
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
    assert "plan == frozen inventory" in r.stdout and "placement OK" in r.stdout and "overrides=demo_smoke" in r.stdout
    assert re.search(rf"target_attacks\.sbatch {MODEL} {VIEW} crescendo [0-9a-f]{{40}} 99 0'", r.stdout) and f" grok420 0-0/99 all {VIEW}" in r.stdout
    assert sum(1 for ln in r.stdout.splitlines() if ln.startswith("submitted ") and ".attacker" in ln) == 2
    # … the knob DROPPED (the paper's defaults): refused by name, nothing planned
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--view", VIEW, "--family", "multiturn", "--dry-run", SISTEER_RUN_OVERRIDES="")
    assert r.returncode != 0 and f"{KEY} pinned '0-0/99', live None" in r.stderr and "submitted " not in r.stdout, r.stdout[-1500:] + r.stderr[-1500:]
    # … the knob CHANGED after the freeze (another smoke document for the same model): refused by name
    alt = _doc(tmp_path, "demo_smoke_alt.env", OVERRIDE_ID="demo_smoke_alt", **{KEY: "0-1/99"})
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--view", VIEW, "--family", "multiturn", "--dry-run", "--overrides", str(alt), SISTEER_RUN_OVERRIDES="")
    assert r.returncode != 0 and f"{KEY} pinned '0-0/99', live '0-1/99'" in r.stderr, r.stdout[-1500:] + r.stderr[-1500:]
    # an unchanged re-freeze under the same document is a no-op; once an attempt exists, a freeze that drops the knob is refused
    r = w.stage("05_stage_roster.sh", "--model", MODEL, "--step", "freeze", "--overrides", doc_rel, SISTEER_RUN_OVERRIDES="")
    assert r.returncode == 0 and "inventory unchanged" in r.stdout, r.stdout[-1500:] + r.stderr[-1500:]
    lw = mt[0]["logical_work_id"]
    with open(w.state / "attempts.jsonl", "a") as fh:
        fh.write(json.dumps({"event": "attempt", "logical_work_id": lw, "attempt_id": f"{lw}@cluster_a@1", "cluster": "cluster_a", "job_id": "4242", "sha": inv["code_sha"],
                             "kind": "target", "spec": {**mt[0]["workload"], "sha": inv["code_sha"]}, "utc": "2026-01-02T00:00:00Z"}) + "\n")
    r = w.stage("05_stage_roster.sh", "--model", MODEL, "--step", "freeze", SISTEER_RUN_OVERRIDES="")
    assert r.returncode != 0 and "frozen ONCE" in r.stderr and f"{KEY} '0-0/99' -> None" in r.stderr, r.stdout[-1500:] + r.stderr[-1500:]
    assert w.inventory()["plan_knobs"][KEY] == "0-0/99"                                            # the pinned inventory stands


# ---------------------------------------------------------------------------------------------------- the committed tree's 06 dry run (the plan's verify line)
def test_06_dry_run_on_the_committed_tree_prints_the_smoke_positionals_from_the_shipped_document(tmp_path):
    root = tmp_path / "fd"; site = root / "site"
    shutil.copytree(REPRO / "site.example", site)
    with open(site / "config.env", "a") as fh:
        fh.write(f"\nSISTEER_STATE_DIR={root / 'state'}\nSISTEER_PYTHON={VENV_PY}\nSISTEER_REGISTRY_ROOT={root / 'registry'}\n")
    env = {ST.RUN_OVERRIDES_ENV: ""}
    smoke = run_stage("06_eval_waves.sh", "--dry-run", "--model", MODEL, "--view", VIEW, "--family", "multiturn,padding", "--overrides", "reproduce/overrides/demo_smoke.env", site=site, env=env, timeout=600)
    assert smoke.returncode == 0, smoke.stdout[-2500:] + smoke.stderr[-2500:]
    out = smoke.stdout
    assert re.search(rf"target_attacks\.sbatch {MODEL} {VIEW} crescendo [0-9a-f]{{40}} 99 0'", out) and re.search(rf"target_attacks\.sbatch {MODEL} {VIEW} pair [0-9a-f]{{40}} 99 0'", out)
    assert re.search(rf"target_attacks\.sbatch {MODEL} {VIEW} actorattack [0-9a-f]{{40}} 99 0 .{{0,12}}49152", out)       # the empty cells positional, then the window
    assert all(f" {pad} grok420 0-0/99 all {VIEW}" in out for pad in ST.PADDING_LEVELS) and "overrides=demo_smoke" in out
    units = [ln.split()[3] for ln in out.splitlines() if ln.startswith("submitted ")]
    assert sorted(u for u in units if u.startswith("multiturn.") and ".attacker" not in u) == sorted(f"multiturn.{atk}.g0" for atk in ST.MT_ATTACKS)
    assert sum(1 for u in units if ".attacker" in u) == 2
    paper = run_stage("06_eval_waves.sh", "--dry-run", "--model", MODEL, "--view", VIEW, "--family", "multiturn,padding", site=site, env=env, timeout=600)
    assert paper.returncode == 0 and " 9 0,1,2'" in paper.stdout and f" grok420 3 all {VIEW}" in paper.stdout and "overrides=paper_defaults" in paper.stdout
    assert sum(1 for ln in paper.stdout.splitlines() if ln.startswith("submitted ") and ".attacker" in ln) == 4
