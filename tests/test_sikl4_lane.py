"""The non-production roster ``configs/sdfbase_full22_roster_sikl4.yaml``: the four static single-turn families, best-of-N, the capability
and over-refusal controls and the multi-turn attacks on the eighteen SI-distilled Qwen3.5-4B objects refit with recorded fit seeds (the safety SI, the constitution SI and the short irrelevant control SI C16, each as a rank-4
o_proj LoRA and an all-layer ReLU steer, fit seeds 1-3), fit on and served on the instruct checkpoint.

The production-roster tests never open this file, so its contract is pinned here:

* every top-level key but ``targets`` is the production roster's as of the roster's landing, pinned by the sha256 of its canonical JSON
  (so a later change to the production roster does not redden this file; a change here is deliberate), and the target's model, revision
  and body config are pinned likewise;
* exactly three views ``sikl4s<k>`` (k in 1..3), none a production view (common.sh refuses a non-production roster that names one), no
  view name under a grammar another reader keys on (the patterns are imported from their modules, never retyped), and each view name
  read as fit seed ``s<k>`` by the notebook's seed reading;
* each view runs every leg but Petri (static, bon, capability, controls, attacks), omits Petri with a reason, declares the per-view
  best-of-N cap ``caps: {bon: 4096}`` (the seed views' native best-of-75 pass) and no other cap, carries no dose, family or paired key, and
  binds the same six cells, every one in the pin form against ``configs/sikl4_pins.json`` with the pin ``qwen35_4b.sikl4s<k>__<cell>``;
* each cell translates through the harness vocabulary to its object kind and SI: ``lora_r4`` -> ``lora_all``, ``psr_kl_all`` -> ``psr``;
  ``safety_refusal`` <-> ``safety``, ``si_constitution`` <-> ``constitution``, ``C16`` <-> ``irrelevant_c16`` (a distilled-only SI: Class A
  cells only, outside the Petri canonical matrix, refused by name as a prompted cell by the harness hash's behavior-dir inventory, and
  never the long control SI ``si_irrelevant``);
* the C16 cells' SI resolves in the TrainTest cell registry of each rendered config only under the candidate-SI loader (a synthetic
  snapshot here), which every job that builds these views runs with;
* ``roster_view.py ... has_leg`` admits each view's five legs under this roster and refuses Petri;
* the sampled leg renders exactly the three views' nine sampled configs (the 4,096 base pass and its 8,192 / 16,384 rungs) under this
  roster only, each at its fit seed k with one trial, the published seed views' decode block but for the seed base;
* the job-start registration over each committed wave config at each static rung's producer key, cap, ladder and pass label plans the
  four static families on their canonical instruments for ``none`` and the six pinned cells, one run id per (cell, family, rung);
* the model key a job registers under (``qwen35_4b_sikl4s<k>``) resolves through ``legacy_spellings.yaml`` to the instruct checkpoint,
  instruct-direct, with no replicate label;
* the job-start registration over each committed rendered config and the real lane pins plans exactly each view's six pinned cells,
  and the wave's defender job names ``f22-att-sikl-s<k>{all,saf,con,c16}-<attack>`` register their attack under a job-independent
  producer key, while the serving smoke's name ``sikl-smoke-s<k>-crescendo`` is no registering job kind;
* the multi-turn grader resolves the three view tags by name;
* the wave-config enumerator names this file as a table;
* the opt-in serving smoke in ``target_attacks.sbatch`` (``F22_STANDUP_SMOKE=1``) is two marked blocks at their two anchors, inert when
  unset, refused without registration off, and, set, runs the positive control on every other object cell under its own tag, runs the
  job's cleanup once and exits 0 before Phase B with the smoke line as the job's last line (by execution, the job's functions stubbed).

The pins themselves, and ``make_full22_configs.py --roster ... --check`` on the committed configs, are ``tests/test_sikl4_pins.py``'s."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "scripts" / "sdfbase", REPO / "scripts" / "sdfbase" / "full22", REPO / "scripts" / "sdfbase" / "full22" / "sampled",
           REPO / "scripts" / "sdfbase" / "seedrep", REPO / "src", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import f22_roster as fr  # noqa: E402
import eval_seed as es  # noqa: E402
import make_full22_configs as gen  # noqa: E402
import make_sampled_configs as msc  # noqa: E402
from full22_wave_configs import EXCLUDED_TABLES  # noqa: E402
from sisteer.capeval import manifest as cm  # noqa: E402
from sisteer.capeval.petri import defense_cells as dc  # noqa: E402
from sisteer.capeval.petri import petri_stats  # noqa: E402
from sisteer.capeval.petri import registry_adapter as ra  # noqa: E402
from sisteer.data.candidate_sis import PATH_ENV as CANDIDATE_SIS_PATH_ENV  # noqa: E402
from sisteer.eval.defense_cell import arm_of, si_of  # noqa: E402
from sisteer.registry import jobplan as JP  # noqa: E402
from sisteer.registry import synth  # noqa: E402
from sisteer.registry.adapters import seedrep_cluster as SRC  # noqa: E402
from sisteer.registry.adapters import strength_mtdiff as SMD  # noqa: E402

LANE_ROSTER_PATH = REPO / "configs/sdfbase_full22_roster_sikl4.yaml"
LANE_PINS_REL = "configs/sikl4_pins.json"
PROD_ROSTER_PATH = REPO / "configs/sdfbase_full22_roster.yaml"
PROD_ROSTER = yaml.safe_load(PROD_ROSTER_PATH.read_text())
LANE_ROSTER = yaml.safe_load(LANE_ROSTER_PATH.read_text())
TARGET = "qwen35_4b"
VIEWS = ("sikl4s1", "sikl4s2", "sikl4s3")
#: cell -> (object kind, TrainTest SI, harness SI): per SI the LoRA then the ReLU, the order the rendered configs list them
CELLS = {"safety_refusal__lora_r4": ("lora_all", "safety_refusal", "safety"),
         "safety_refusal__psr_kl_all": ("psr", "safety_refusal", "safety"),
         "si_constitution__lora_r4": ("lora_all", "si_constitution", "constitution"),
         "si_constitution__psr_kl_all": ("psr", "si_constitution", "constitution"),
         "C16__lora_r4": ("lora_all", "C16", "irrelevant_c16"),
         "C16__psr_kl_all": ("psr", "C16", "irrelevant_c16")}
#: the legs each view runs, in the roster grammar's launch order (f22_roster.LEGS minus Petri), and the one it omits
LEGS = ("static", "bon", "capability", "controls", "attacks")
OMITTED = ("petri",)
#: the per-view best-of-N cap: the native best-of-75 pass at 4,096, as on the seed views s<k> (make_full22_configs.view_declared_bon_cap)
VIEW_CAPS = {"bon": 4096}
#: sha256 of the canonical JSON (sorted keys, compact separators) of the roster's top level minus ``targets``: the production roster's
#: serving contract when this roster was written (serving_impl_commit 94e294c4…, caps, controls window, bon climb, attacker flags, lane flags)
TOP_LEVEL_SHA256 = "12de193b1412a428271407d856c0fc18dffb9356adbd63a2839e9e0e1d37ee15"
TARGET_SBATCH = REPO / "scripts/sdfbase/full22/target_attacks.sbatch"
REGISTRY_NB = REPO / "paper/notebooks/registry_nb.py"


def _seed(view: str) -> int:
    assert view in VIEWS, view
    return int(view[-1])


def _top_level_sha256(roster: dict) -> str:
    top = {k: v for k, v in roster.items() if k != "targets"}
    return hashlib.sha256(json.dumps(top, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _env() -> dict:
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    env.pop(JP.PIN_FORM_ENV, None)
    return env


# --------------------------------------------------------------------------------------------------------------------- the roster

def test_the_roster_carries_the_production_contract_and_exactly_the_three_views():
    assert _top_level_sha256(LANE_ROSTER) == TOP_LEVEL_SHA256
    assert LANE_ROSTER["serving_impl_commit"] == "94e294c445dc5eec6ebe0ce3a3e25553004cd4b0" and LANE_ROSTER["target_max_new_tokens"] == 4096
    assert list(LANE_ROSTER["targets"]) == [TARGET]
    t, pt = LANE_ROSTER["targets"][TARGET], PROD_ROSTER["targets"][TARGET]
    assert set(t) == {"model", "model_revision", "body_config", "views"}            # no TARGET caps table (a view's own caps.bon is
                                                                                    # admitted only then), no lanes, no dose families
    assert (t["model"], t["model_revision"], t["body_config"]) == ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                                                                   "configs/sdfbase_transfer_qwen35_4b_r3.yaml")
    assert tuple(t["views"]) == VIEWS
    assert not set(VIEWS) & set(pt["views"])                         # none is a production view (common.sh f22_bank_view refuses one)
    assert LANE_ROSTER_PATH.name in EXCLUDED_TABLES                   # a roster under the wave-config glob, named as a table


@pytest.mark.parametrize("view", VIEWS)
def test_no_view_name_falls_under_another_readers_grammar_and_each_reads_as_its_fit_seed(view):
    tag = f"{TARGET}_{view}"
    assert not gen.PHASE_VIEW_RE.match(view)                          # the phase views
    assert not gen._DOSE_VIEW_RE.match(view)                          # d<tok> / it_d<tok>[_<family>]
    assert not gen._PAIRED_VIEW_RE.match(view)                        # the paired standups s<k>p
    assert not fr.STEERED_WAVE_VIEW_RE.match(view) and not fr.steered_wave_view(view, TARGET)
    assert not SRC.TAG_RE.match(tag) and not SRC.MT_STORE_RE.match(f"sdfbase_full22_{tag}")   # the seed-replicate tags and stores
    assert not SMD.STANDUP_RE.match(tag)                              # the strength standups
    assert gen.view_paired_of(LANE_ROSTER, TARGET, view) is None
    # the notebook reads a run's seed off its view (registry_nb, the `from_view` extraction): both patterns are read from its source,
    # never retyped, and the view name must read as the fit seed
    src = REGISTRY_NB.read_text()
    strip = re.search(r'_v = d\["view_legacy"\]\.fillna\(""\)\.str\.replace\(r"([^"]+)", "", regex=True\)', src)
    seed = re.search(r'from_view = _v\.str\.extract\(r"([^"]+)"\)\[0\]', src)
    assert strip and seed, "the notebook's seed reading moved; re-point this check at it"
    m = re.search(seed.group(1), re.sub(strip.group(1), "", tag))
    assert m and m.group(1) == f"s{_seed(view)}", (tag, m)


@pytest.mark.parametrize("view", VIEWS)
def test_each_view_runs_every_leg_but_petri_at_the_operating_point_and_binds_six_pinned_cells(view):
    v = LANE_ROSTER["targets"][TARGET]["views"][view]
    assert set(v) == {"legs", "legs_omitted", "caps", "config", "manifest", "cells"}   # no dose / dose_unit / object_family / paired_of
    assert fr.view_legs(LANE_ROSTER, TARGET, view) == LEGS
    assert v["caps"] == VIEW_CAPS and gen.view_declared_bon_cap(LANE_ROSTER, TARGET, view) == 4096
    omitted = fr.view_legs_omitted(LANE_ROSTER, TARGET, view)
    assert set(omitted) == set(OMITTED) and all(isinstance(why, str) and why.strip() for why in omitted.values())
    assert gen.view_dose(LANE_ROSTER, TARGET, view) is None
    assert fr.view_cell_universe(LANE_ROSTER, TARGET, view) == ["none", *CELLS]
    assert v["config"] == f"configs/sdfbase_full22_{TARGET}_{view}.yaml"
    assert v["manifest"] == f"configs/capeval/manifests/petri_full22_{TARGET}_{view}.yaml"
    assert tuple(v["cells"]) == tuple(CELLS)
    for cell, mapping in v["cells"].items():
        assert fr.is_pin_form(mapping) and mapping == {"pins": LANE_PINS_REL, "pin": f"{TARGET}.{view}__{cell}"}, (view, cell, mapping)
    assert fr.staged_cells(LANE_ROSTER, TARGET, view) == dict(v["cells"])


# --------------------------------------------------------------------------------------------------------------------- the cells

@pytest.mark.parametrize("cell", list(CELLS))
def test_each_cell_translates_to_its_object_kind_and_si(cell):
    kind, tt_si, h_si = CELLS[cell]
    assert ra.ARM_OBJECT_KINDS[arm_of(cell)] == kind and si_of(cell) == tt_si
    harness = ra.harness_cell_id(cell)
    assert harness == gen.harness_cell_id(cell) == f"si_{h_si}__{arm_of(cell)}"           # the generator and the adapter agree
    parsed = dc.parse_cell_id(harness)
    assert (parsed.cell_class, parsed.si_id, parsed.method) == ("class_a", h_si, arm_of(cell))
    assert ra.HARNESS_TO_TRAINTEST_SI[h_si] == tt_si and ra.TRAINTEST_TO_HARNESS_SI[tt_si] == h_si
    assert ra.traintest_cell_id(harness) == cell
    assert gen.TT_TO_HARNESS_SI[tt_si] == h_si


def test_c16_is_a_distilled_only_si_never_the_long_control_si():
    assert ra.HARNESS_TO_TRAINTEST_SI["irrelevant_c16"] == "C16"
    assert ra.HARNESS_TO_TRAINTEST_SI["irrelevant"] == "si_irrelevant"                     # the long control SI keeps its own id
    assert cm.PETRI_DISTILLED_ONLY_SI_IDS == ("irrelevant_c16",)
    assert "irrelevant_c16" in cm.PETRI_SI_IDS
    assert "irrelevant_c16" not in cm.PETRI_PROMPTABLE_SI_IDS and "irrelevant_c16" not in cm.PETRI_OBJECT_ONLY_SI_IDS
    assert "irrelevant_c16" not in cm.PETRI_SI_VARIANT_DIRS
    # Class A only: no prompted or stacked leg, in either vocabulary
    for bad in ("prompted__irrelevant_c16", "stacked__irrelevant_c16__lora_r4", "stacked__irrelevant_c16__psr_kl_all"):
        with pytest.raises(dc.DefenseCellError, match="DISTILLED-ONLY"):
            dc.parse_cell_id(bad)
    with pytest.raises(ra.RegistryAdapterError, match="DISTILLED-ONLY"):
        ra.harness_cell_id("prompted__C16")
    # the Petri canonical matrix is unchanged: no C16 cell is expected of any audit
    assert not any("irrelevant_c16" in c for c in petri_stats.canonical_cell_ids())


@pytest.mark.parametrize("view", VIEWS)
def test_the_harness_inventory_refuses_a_prompted_c16_cell_by_name(view):
    """A distilled-only SI is in ``PETRI_SI_IDS`` but has no SI-variant behavior dir, so the harness hash's behavior-dir inventory must
    refuse ``prompted__irrelevant_c16`` with a named DISTILLED-ONLY ManifestError, never a bare KeyError off ``PETRI_SI_VARIANT_DIRS``
    (before C16 joined ``PETRI_SI_IDS`` the same cell was refused by name as an unknown prompted SI). The lane's own manifest, whose C16
    cells are Class A only, adds no variant dir."""
    spec = cm.load_petri_spec(REPO / f"configs/capeval/manifests/petri_full22_{TARGET}_{view}.yaml")
    assert cm.petri_expected_behavior_keys(spec) == sorted(spec.behaviors)
    bad = dataclasses.replace(spec, cells=[*spec.cells, "prompted__irrelevant_c16"])
    with pytest.raises(cm.ManifestError, match=r"prompts the DISTILLED-ONLY SI 'irrelevant_c16'.*only 'si_irrelevant_c16__<method>'"):
        cm.petri_expected_behavior_keys(bad)


# --------------------------------------------------------------------------------------------------------------------- the readers

@pytest.mark.parametrize("view", VIEWS)
def test_roster_view_admits_every_leg_but_petri(view):
    env = _env()
    tool = REPO / "scripts/sdfbase/full22/roster_view.py"
    for leg in LEGS:
        ok = subprocess.run([sys.executable, str(tool), TARGET, view, "has_leg", leg, "--roster", str(LANE_ROSTER_PATH)],
                            capture_output=True, text=True, env=env, cwd=REPO, timeout=120)
        assert ok.returncode == 0, (leg, ok.stderr)
    for leg in OMITTED:
        r = subprocess.run([sys.executable, str(tool), TARGET, view, "has_leg", leg, "--roster", str(LANE_ROSTER_PATH)],
                           capture_output=True, text=True, env=env, cwd=REPO, timeout=120)
        assert r.returncode != 0, (leg, r.stdout, r.stderr)
    count = REPO / "scripts/sdfbase/full22/roster_cell_count.py"
    assert subprocess.run([sys.executable, str(count), TARGET, view, "--roster", str(LANE_ROSTER_PATH)], capture_output=True, text=True,
                          env=env, cwd=REPO, timeout=120).stdout.strip() == "6"
    # the production roster names no such view: the collision guard of common.sh f22_bank_view stays quiet
    assert subprocess.run([sys.executable, str(count), TARGET, view, "--roster", str(PROD_ROSTER_PATH)], capture_output=True, text=True,
                          env=env, cwd=REPO, timeout=120).returncode != 0


@pytest.mark.parametrize("view", VIEWS)
def test_the_registration_model_key_of_each_view_is_an_instruct_direct_legacy_spelling(view):
    rows = yaml.safe_load((REPO / "configs/registry/legacy_spellings.yaml").read_text())["model_keys"]
    assert JP.model_for_key(rows, f"{TARGET}_{view}") == ("qwen35_4b_it", "instruct_direct")
    hits = [r for r in rows if f"{TARGET}_{view}" in [s.strip() for s in str(r["legacy"]).split("|")]]
    assert len(hits) == 1, hits
    assert hits[0]["fit_model_id"] == "qwen35_4b_it" and "fit_replicate_label" not in hits[0]


def test_the_grader_resolves_the_three_view_tags_by_name():
    code = ("import importlib.util, json\n"
            f"spec = importlib.util.spec_from_file_location('sikl4_lane_grader', {str(REPO / 'scripts/sdfbase/seedrep/grade_full22_native.py')!r})\n"
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
            "t, _ = m.resolve_targets()\n"
            f"views = {list(VIEWS)!r}\n"
            "print(json.dumps({'targets': {k: [v['run'], list(v['default_attacks'])] for k, v in t.items() if k.startswith('qwen35_4b_sikl4')},\n"
            "                  'phase': [bool(m.PHASE_VIEW_RE.match(v)) for v in views], 'seedrep': [bool(m.SEEDREP_VIEW_RE.match(v)) for v in views],\n"
            "                  'cells': {c: m.parse_cell(c) for c in " + repr(list(CELLS)) + "}}))")
    env = dict(_env(), HF_HUB_OFFLINE="1",
               PYTHONPATH=os.pathsep.join(str(REPO / p) for p in ("src", "scripts", "scripts/sdfbase/seedrep")))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert got["targets"] == {f"{TARGET}_{v}": [f"sdfbase_full22_{TARGET}_{v}", ["crescendo", "actorattack"]] for v in VIEWS}
    assert got["phase"] == [False] * 3 and got["seedrep"] == [False] * 3          # no grammar is widened for them
    assert got["cells"] == {c: {"class": "A", "si_id": tt_si, "arm": arm_of(c)} for c, (_, tt_si, _) in CELLS.items()}


# ------------------------------------------------------------------------------------------------- job-start registration

#: the canonical multi-turn instruments of the served instruct checkpoint on the native LoRA path
INSTRUMENT = {"crescendo": "33f80ed1d4f536e34c870fa36c2d88d90c7623ea17a9d4db178fea008c12ac85",
              "actorattack": "f8832b53b08131258d0e5c9e15a16a2888f8a977271d52dde270d52b1af00203"}
ITEM_SET = "multiturn_goals_259"
CAMPAIGN = "si_target_refit"
ATTACKS = ("crescendo", "actorattack")
SEEDS = REPO / "configs/registry"


@pytest.fixture(scope="module")
def _seed_index():
    return JP.load_seed_index(str(SEEDS))


@pytest.fixture
def seeds_parsed_once(monkeypatch, _seed_index):
    """``build_plans`` re-parses the registry seeds on every call (~1.4 s); hand it one parse per module, a fresh copy per call."""
    def load(seeds_dir: str):
        assert os.path.realpath(seeds_dir) == os.path.realpath(SEEDS), seeds_dir
        return copy.deepcopy(_seed_index)
    monkeypatch.setattr(JP, "load_seed_index", load)


def _plans(cfg: Path, view: str, attack: str, *, objects=None, pin_form=True):
    shards = "_".join(str(s) for s in range(8))
    return JP.build_plans(config_path=str(cfg), model_key=f"{TARGET}_{view}", roster_selection=view, evals=[attack],
                          serving_path="lora_native", campaign=CAMPAIGN, job_id="4242", job_name=f"f22-att-synthetic-{attack}",
                          roster_path=str(LANE_ROSTER_PATH), substrate="cluster_a", code_sha=synth.code_sha("a"), objects=objects,
                          producer_key=f"sdfbase_full22-att-{attack}-{TARGET}_{view}-sh{shards}of8", pin_form=pin_form)


def _check_plan_facts(b, view: str, attack: str, shas: dict) -> None:
    assert b.skipped == [], b.skipped
    assert {r.route["object"]: r.plan["object_id"] for r in b.runs} == shas
    assert len({r.run_id for r in b.runs}) == len(shas)
    for r in b.runs:
        p = r.plan
        assert (p["eval_id"], p["instrument_id"], p["item_set_id"]) == (attack, INSTRUMENT[attack], ITEM_SET)
        assert (p["model_id"], p["campaign"], p["serving_path"]) == ("qwen35_4b_it", CAMPAIGN, "lora_native")
        assert p["transfer_kind"] == ("none" if p["object_id"] == "none" else "instruct_direct")
        assert p["extra"]["model_key"] == f"{TARGET}_{view}" and p["extra"]["view_legacy"] == view
        assert not {"trial_index", "pass_label", "token_cap_rung_index"} & set(p)
        assert p["token_cap_tokens"] == 4096


@pytest.mark.parametrize("attack", ATTACKS)
@pytest.mark.parametrize("view", VIEWS)
def test_the_job_start_registration_over_the_rendered_config_plans_exactly_the_six_pinned_cells(view, attack, monkeypatch,
                                                                                                 seeds_parsed_once):
    """Over the committed rendered config and the real lane pins: the registration over the view's cells plans exactly its six objects,
    each the sha of the pin its roster entry names, with nothing skipped; the defender's cell list (no `none`) plans no unsteered run;
    without the pin-form opt-in every entry is skipped by name, as for any pin-form roster."""
    monkeypatch.delenv(JP.PIN_FORM_ENV, raising=False)
    pins = json.loads((REPO / LANE_PINS_REL).read_text())[TARGET]
    shas = {c: pins[f"{view}__{c}"]["object_sha256"] for c in CELLS}
    cfg = REPO / f"configs/sdfbase_full22_{TARGET}_{view}.yaml"
    _check_plan_facts(_plans(cfg, view, attack), view, attack, {"none": "none", **shas})
    _check_plan_facts(_plans(cfg, view, attack, objects=list(CELLS)), view, attack, shas)
    off = _plans(cfg, view, attack, objects=list(CELLS), pin_form=False)
    assert not off.runs and sorted(what for what, _ in off.skipped) == sorted(f"object {c}" for c in CELLS), off.skipped


# ------------------------------------------------------------------------------------------------- the static leg (sampled lane)

#: the canonical static instruments of the served instruct checkpoint (the vendor-preset sampled lane, trial 0; read_at ladder_final),
#: with their item sets: each family's 259 goals x its draws per goal
STATIC_INSTRUMENT = {"pasttense": ("14942170c71578e7a58daee14ebb6cb074e7575a8399f5a17ac63596456010d3", "static_pasttense_259x2"),
                     "codeattack": ("7c6751db529f9f637926fbfcc0cac21254ddd9adb8e8cdb23c8724234e7de286", "static_codeattack_259x3"),
                     "renellm": ("58ad64eb969e23e3959fcf3832348f0cf99fc89dd3879cdc9e22dc2f8576b0e8", "static_renellm_259x3"),
                     "grok420": ("f71ed06cf77fea63ad450bff882f4f7097ed99ba4643c6b9658f57ef8e3ffc04", "static_grok420_259x2")}
#: (rung suffix of the private run, cap, rung index, pass label) -- common_sampled.sh f22s_parse_phase's static_base / static_esc /
#: static_esc16 registration exports, the ladder F22S_STATIC_RUNGS_EXT
STATIC_RUNGS = (("", 4096, 0, "base_cap"), ("_cap8192", 8192, 1, "ladder_rung"), ("_cap16384", 16384, 2, "ladder_rung"))
STATIC_LADDER = (4096, 8192, 16384)


def test_the_sampled_leg_renders_the_three_views_under_the_lane_roster_only_at_their_fit_seed_one_trial(monkeypatch):
    presets = msc.load_presets()
    # the production scope is untouched: not in SAMPLED_VIEWS, not in scope, not among the production TAGs
    assert not set(VIEWS) & set(msc.SAMPLED_VIEWS[TARGET])
    assert all(not msc.in_scope(TARGET, v) for v in VIEWS) and not set(msc.lane_replicate_tags()) & set(msc.sampled_tags())
    # the seed rule rows: literal data in the rule table AND the data file, at the fit seed, one trial
    assert set(VIEWS) <= set(msc.SEEDED_LANE_VIEWS[TARGET])
    for view in VIEWS:
        k, tag = _seed(view), msc.tag_of(TARGET, view)
        assert msc.seed_of_lane_view(view) == k
        assert msc.SEED_BASE_RULE_BY_TAG[tag] == presets["seed_base_by_view"][tag] == es.seed_base(k) == k != presets["seed_base"]
        assert msc.seed_base_of_tag(presets, tag) == k and msc.n_trials_of_tag(presets, tag) == presets["n_trials_by_view"][tag] == 1
    # under the LANE roster the scope is exactly the three views and the committed sampled configs are the generator's
    monkeypatch.setattr(msc, "_ROSTER_PATH", LANE_ROSTER_PATH.resolve())
    assert msc._private() and msc.scope_views(TARGET) == VIEWS and all(msc.scope_views(k) == () for k in msc.SAMPLED_VIEWS if k != TARGET)
    assert all(msc.in_scope(TARGET, v) for v in VIEWS) and not msc.in_scope(TARGET, "s1") and not msc.in_scope(TARGET, "main")
    msc.check_pins(presets, msc.load_roster())
    outs = msc.outputs(presets)
    lane_tags = {msc.tag_of(TARGET, v) for v in VIEWS}
    assert len(outs) == 9 == 3 * 1 * 3 and {t for _p, t, *_ in outs} == lane_tags < set(msc.lane_replicate_tags())
    for path, tag, trial, esc in outs:
        assert trial == 0 and path.is_file(), path.name
        assert path.read_text() == msc.render(presets, tag, trial, esc)
        doc = yaml.safe_load(path.read_text())
        sd, k = doc["sampled_decode"], _seed(tag.split(f"{TARGET}_", 1)[1])
        assert sd["seed_base"] == k and sd["n_trials"] == 1 and sd["trial"] == 0 and sd["arm"] == "vendor_recommended"
        # the published seed view's decode block, rung for rung, but for the seed base and the store it escalates from
        rung = path.name.split("_sampled_t0", 1)[1].removesuffix(".yaml")
        ref = yaml.safe_load((REPO / f"configs/sdfbase_full22_{TARGET}_s2_sampled_t0{rung}.yaml").read_text())["sampled_decode"]
        drop = ("seed_base", "escalated_from")
        assert {a: b for a, b in sd.items() if a not in drop} == {a: b for a, b in ref.items() if a not in drop}, path.name
        assert doc["explicit_cells"] == ["none", *CELLS]
    assert msc.stale_outputs(presets) == []
    # the lane roster's seeded views need their rows: a data file without them refuses by name
    stripped = copy.deepcopy(presets)
    stripped["n_trials_by_view"] = {t: n for t, n in presets["n_trials_by_view"].items() if t not in lane_tags}
    with pytest.raises(SystemExit, match="seeded lane view"):
        msc.check_pins(stripped, msc.load_roster())


def _static_plans(cfg: Path, view: str, rung: tuple, *, objects=None):
    suffix, cap, index, label = rung
    return JP.build_plans(config_path=str(cfg), model_key=f"{TARGET}_{view}", roster_selection=view, evals=list(STATIC_INSTRUMENT),
                          serving_path="lora_native", campaign=CAMPAIGN, job_id="4242", job_name="sikl4-static-synthetic",
                          roster_path=str(LANE_ROSTER_PATH), substrate="cluster_a", code_sha=synth.code_sha("a"), objects=objects,
                          producer_key=f"sdfbase_full22_{TARGET}_{view}_sampled_t0{suffix}", token_cap=cap,
                          token_cap_rung_index=index, ladder=list(STATIC_LADDER), pass_label=label, trial_index=0, pin_form=True)


@pytest.mark.parametrize("rung", STATIC_RUNGS, ids=[f"cap{r[1]}" for r in STATIC_RUNGS])
@pytest.mark.parametrize("view", VIEWS)
def test_the_static_registration_plans_the_four_families_for_none_and_the_six_pinned_cells_at_each_rung(view, rung, monkeypatch,
                                                                                                        seeds_parsed_once):
    """The job-start (base) and gate (rung) registration of the sampled static lane over the committed wave config: every (cell, family)
    pair lands on the family's canonical static instrument and item set, at the rung's cap / index / pass label, trial 0, under the
    campaign and the view's instruct-direct model key; the run id is the identity rule over the rung's private-run producer key."""
    monkeypatch.delenv(JP.PIN_FORM_ENV, raising=False)
    pins = json.loads((REPO / LANE_PINS_REL).read_text())[TARGET]
    shas = {"none": "none", **{c: pins[f"{view}__{c}"]["object_sha256"] for c in CELLS}}
    b = _static_plans(REPO / f"configs/sdfbase_full22_{TARGET}_{view}.yaml", view, rung)
    assert b.skipped == [], b.skipped
    suffix, cap, index, label = rung
    pk = f"sdfbase_full22_{TARGET}_{view}_sampled_t0{suffix}"
    got = {(r.route["object"], r.plan["eval_id"]): r for r in b.runs}
    assert set(got) == {(c, e) for c in shas for e in STATIC_INSTRUMENT} and len(b.runs) == 28
    for (cell, ev), r in got.items():
        p, (inst, iset) = r.plan, STATIC_INSTRUMENT[ev]
        assert p["object_id"] == shas[cell]
        assert (p["instrument_id"], p["item_set_id"], p["campaign"], p["model_id"]) == (inst, iset, CAMPAIGN, "qwen35_4b_it")
        assert p["transfer_kind"] == ("none" if cell == "none" else "instruct_direct") and p["serving_path"] == "lora_native"
        assert (p["token_cap_tokens"], p["token_cap_rung_index"], p.get("pass_label"), p.get("trial_index", 0)) == (cap, index, label, 0)
        assert p["producer_run_key_legacy"] == pk and p["producer_object_key_legacy"] == cell
        want_id = f"{CAMPAIGN}:{ev}:{inst[:12]}:{hashlib.sha256(f'{pk}|{cell}'.encode()).hexdigest()[:16]}"
        assert r.run_id == want_id, (r.run_id, want_id)
    # a task that serves a subset registers that subset only (F22_REGISTRY_CELLS narrows the declared cells)
    sub = _static_plans(REPO / f"configs/sdfbase_full22_{TARGET}_{view}.yaml", view, rung, objects=["C16__lora_r4", "C16__psr_kl_all"])
    assert sub.skipped == [] and {r.route["object"] for r in sub.runs} == {"C16__lora_r4", "C16__psr_kl_all"} and len(sub.runs) == 8


# ------------------------------------------------------------------------------------------------- the C16 cells under the loader

#: the cell registry the standup and the drivers build from a rendered config (defense_cell.registry_from_cfg, the explicit branch),
#: printed per view as the per-cell refusal reasons over an empty bank view: the SI check runs before the staged-artifact check, so a
#: cell whose SI resolves refuses only for its absent artifact. One interpreter per loader state (the loader reads its switch once, at
#: import), the three views in it (the bank view dir is read per call)
_REGISTRY_PROBE = """
import json, os, sys, traceback, yaml
from sisteer.eval import defense_cell as D
out = {}
for view, cfg, view_dir in json.loads(sys.argv[1]):
    os.environ["TT_TWO_SI_DEFENSES_DIR"] = view_dir
    try:
        D.registry_from_cfg(yaml.safe_load(open(cfg)))
        out[view] = {"built": True}
    except ValueError as exc:
        out[view] = {"built": False, "message": str(exc)}
    except Exception:
        out[view] = {"crashed": traceback.format_exc()}
print(json.dumps(out))
"""


@pytest.fixture(scope="module")
def registry_probes(tmp_path_factory) -> tuple[Path, dict]:
    """(the bank views' root, {"off" | "on": {view: probe}}): each committed rendered config over an empty bank view, without and with
    the candidate-SI loader on a synthetic snapshot."""
    from test_candidate_si_registry import _synth_entries, _write
    root = tmp_path_factory.mktemp("sikl4_c16")
    snapshot = _write(root, _synth_entries())
    views = [(v, str(REPO / f"configs/sdfbase_full22_{TARGET}_{v}.yaml"), str(root / "cells_view" / v)) for v in VIEWS]
    env = _env()
    for k in ("SISTEER_LOAD_CANDIDATE_SIS", CANDIDATE_SIS_PATH_ENV):
        env.pop(k, None)
    probes = {}
    for state, extra in (("off", {}), ("on", {"SISTEER_LOAD_CANDIDATE_SIS": snapshot})):
        r = subprocess.run([sys.executable, "-c", _REGISTRY_PROBE, json.dumps(views)], capture_output=True, text=True, cwd=REPO,
                           env={**env, **extra}, timeout=300)
        assert r.returncode == 0, r.stderr[-3000:]
        probes[state] = json.loads(r.stdout.strip().splitlines()[-1])
    return root, probes


@pytest.mark.parametrize("view", VIEWS)
def test_the_c16_cells_resolve_in_the_cell_registry_only_under_the_candidate_si_loader(view, registry_probes):
    """Over the committed rendered config and an empty bank view: without the candidate-SI loader the two C16 cells refuse because C16
    does not resolve (the other four refuse only for their absent artifacts); with the loader on a synthetic snapshot every one of the
    six refuses only for its absent artifact, so the C16 SI resolves wherever the jobs run with the loader."""
    root, probes = registry_probes
    off, on = probes["off"][view], probes["on"][view]
    assert "crashed" not in off and "crashed" not in on, (off.get("crashed"), on.get("crashed"))
    assert not off["built"] and not on["built"]
    view_dir = root / "cells_view" / view
    for cell in CELLS:
        absent = f"{cell!r}: no staged artifact at {view_dir / cell}"
        unresolved = f"{cell!r}: SI id 'C16' does not resolve through resolve_instruction"
        assert absent in on["message"] and unresolved not in on["message"], (cell, on["message"][-1500:])
        if cell.startswith("C16__"):
            assert unresolved in off["message"] and absent not in off["message"], (cell, off["message"][-1500:])
        else:
            assert absent in off["message"], (cell, off["message"][-1500:])


COMMON_SH = REPO / "scripts/sdfbase/full22/common.sh"
#: the wave's defender job names `f22-att-sikl-s<k><tag>-<attack>` (the wave tools' layout): `all` = the six cells on Crescendo, one SI's
#: LoRA + ReLU pair on ActorAttack; and the serving smokes `sikl-smoke-s<k>-crescendo`, which must not read as a registering job kind
DEFENDER_TAGS = {"all": "crescendo", "saf": "actorattack", "con": "actorattack", "c16": "actorattack"}
NSHARDS_BY_ATTACK = {"crescendo": 8, "actorattack": 12}


def _job_kind(job_name: str, key_tag: str, nshards: int) -> tuple[str, str]:
    """(evals, producer key) the job forms, by the checkout's OWN shell: target_attacks.sbatch's ``export F22_REGISTRY_KEY_SEGMENTS=…``
    line evaluated with SHARDS / NSHARDS, then common.sh ``f22_registry_evals_for_job`` and ``f22_registry_producer_key``."""
    seg = [x for x in TARGET_SBATCH.read_text().splitlines() if x.startswith("export F22_REGISTRY_KEY_SEGMENTS=")]
    assert len(seg) == 1, seg
    shards = ",".join(str(s) for s in range(nshards))
    script = ('set -u; SHARDS="$1"; NSHARDS="$2"; unset F22_REGISTRY_KEY_SEGMENTS; eval "$3"; '
              'source <(sed -n "/^f22_registry_evals_for_job()/,/^}/p; /^f22_registry_producer_key()/,/^}/p" "$4"); '
              'echo "EVALS=$(f22_registry_evals_for_job "$5")"; echo "KEY=$(f22_registry_producer_key "$5" "$6" "$F22_REGISTRY_KEY_SEGMENTS")"')
    r = subprocess.run(["bash", "-c", script, "_", shards, str(nshards), seg[0], str(COMMON_SH), job_name, key_tag],
                       capture_output=True, text=True, env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C"}, timeout=60)
    assert r.returncode == 0, r.stderr
    out = dict(ln.split("=", 1) for ln in r.stdout.splitlines() if "=" in ln)
    return out["EVALS"], out["KEY"]


@pytest.mark.parametrize("tag", list(DEFENDER_TAGS))
@pytest.mark.parametrize("view", VIEWS)
def test_the_defender_job_names_register_their_attack_under_a_job_independent_key(view, tag):
    attack, n = DEFENDER_TAGS[tag], NSHARDS_BY_ATTACK[DEFENDER_TAGS[tag]]
    name, key_tag = f"f22-att-sikl-s{_seed(view)}{tag}-{attack}", f"{TARGET}_{view}"
    evals, key = _job_kind(name, key_tag, n)
    assert evals == attack, (name, evals)                              # common.sh f22_registry_register reads the eval off the name
    assert key == f"sdfbase_full22-att-{attack}-{key_tag}-sh{'_'.join(str(s) for s in range(n))}of{n}", key
    assert not name.startswith("scanary")                              # never the steered canary's name pattern (common.sh f22_canary_name)


@pytest.mark.parametrize("view", VIEWS)
def test_the_serving_smoke_name_is_no_registering_job_kind(view):
    evals, key = _job_kind(f"sikl-smoke-s{_seed(view)}-crescendo", f"{TARGET}_{view}", 8)
    assert (evals, key) == ("", "")


# ----------------------------------------------------------------------------------- the opt-in serving smoke (F22_STANDUP_SMOKE=1)

_SOURCE_ANCHOR = re.compile(r'^  \|\| source "\$\{SISTEER_EVAL_REPO_ROOT:\?.*/scripts/sdfbase/full22/common\.sh"$')
_BREAK_ANCHOR = "  break; done   # the first OBJECT cell: a prompted baseline has no scale-0 twin"
_PHASE_B_ANCHOR = "# Phase B: the shared attacker pool (LIVE members only)"


def _smoke_blocks() -> tuple[list[str], list[tuple[int, int]]]:
    """The script's lines and the (start, end) line index of each ``# >>> F22_STANDUP_SMOKE (i of 2)`` .. ``# <<< …`` block, in order."""
    lines = TARGET_SBATCH.read_text().splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.startswith("# >>> F22_STANDUP_SMOKE")]
    ends = [i for i, ln in enumerate(lines) if ln.startswith("# <<< F22_STANDUP_SMOKE")]
    assert len(starts) == len(ends) == 2, (starts, ends)
    assert lines[starts[0]].startswith("# >>> F22_STANDUP_SMOKE (1 of 2)") and lines[starts[1]].startswith("# >>> F22_STANDUP_SMOKE (2 of 2)")
    blocks = list(zip(starts, ends))
    assert all(s < e for s, e in blocks) and blocks[0][1] < blocks[1][0]
    return lines, blocks


def test_the_smoke_is_two_marked_blocks_at_their_anchors_and_nothing_else_mentions_it():
    lines, ((s1, e1), (s2, e2)) = _smoke_blocks()
    # block 1 sits between the common.sh source and f22_attempt_wait: before the job's first registry emission (f22_init)
    assert _SOURCE_ANCHOR.match(lines[s1 - 1]), lines[s1 - 1]
    assert lines[e1 + 1].startswith("f22_attempt_wait")
    assert next(i for i, ln in enumerate(lines) if ln.startswith("f22_init ")) > e1
    # block 2 sits between the positive-control loop (which breaks after the first object cell) and Phase B
    assert lines[s2 - 1] == _BREAK_ANCHOR and lines[e2 + 1] == _PHASE_B_ANCHOR
    assert lines[e2 + 2] == "f22_live_attackers"
    # removed, the anchors are adjacent again, as they were before the smoke existed; nothing outside the blocks names it
    rest = lines[:s1] + lines[e1 + 1:s2] + lines[e2 + 1:]
    assert _SOURCE_ANCHOR.match(rest[s1 - 1]) and rest[s1].startswith("f22_attempt_wait")
    assert rest[rest.index(_BREAK_ANCHOR) + 1] == _PHASE_B_ANCHOR
    assert not any("F22_STANDUP_SMOKE" in ln or "standup-smoke" in ln or "attacks_smoke_" in ln for ln in rest)
    # each block's code is ONE `if` keyed on F22_STANDUP_SMOKE == 1 (so unset is a no-op), its comments aside
    for s, e in ((s1, e1), (s2, e2)):
        code = [ln for ln in lines[s + 1:e] if not ln.lstrip().startswith("#")]
        assert code[0].startswith('if [ "${F22_STANDUP_SMOKE:-0}" = 1 ]') and code[0].endswith("; then"), code[0]
        assert code[-1] == "fi" and sum(1 for ln in code if ln == "fi") == 1, code


def _run(script: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C", **(env_extra or {})}
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=60)


def _region() -> str:
    """The positive-control loop and smoke block 2, verbatim: from ``f22s_low_dose_liveness`` through ``# <<< … (2 of 2)``."""
    lines, (_, (_, e2)) = _smoke_blocks()
    start = lines.index("f22s_low_dose_liveness")
    assert start < e2
    return "\n".join(lines[start:e2 + 1])


_STUBS = ('set -uo pipefail\n'
          'declare -A F22S_LIVENESS_RECORD=([cellB]=/tmp/recB.json)\n'
          'CELLS=(none prompted__safety_refusal cellA cellB cellC); RC=0\n'
          'cleanup(){ echo "CLEANUP RC=$RC"; }; trap cleanup EXIT SIGTERM SIGINT SIGQUIT\n'   # the job's own trap, as the script arms it
          'f22s_low_dose_liveness(){ :; }\n'
          'f22_log(){ echo "LOG $*"; }\n'
          'f22_positive_control(){ echo "PC cell=$1 tag=$2 rec=${SISTEER_LOW_DOSE_LIVENESS_RECORD:-NONE}"; '
          '[ "$1" != "${FAIL_CELL:-}" ]; }\n')


def _run_region(env_extra: dict) -> subprocess.CompletedProcess:
    """The region under the stubs, then a marker for reaching Phase B."""
    return _run(_STUBS + _region() + '\necho "PHASE B reached"\n', env_extra)


@pytest.mark.parametrize("value", [None, "0", "", "true", "yes"])
def test_unset_or_any_value_but_1_leaves_the_job_on_its_old_path(value):
    env = {} if value is None else {"F22_STANDUP_SMOKE": value}
    p = _run_region(env)
    assert p.returncode == 0, p.stdout + p.stderr
    pcs = [ln for ln in p.stdout.splitlines() if ln.startswith("PC ")]
    assert pcs == ["PC cell=cellA tag=attacks rec=NONE"], p.stdout          # the loop's one check, as before
    assert "PHASE B reached" in p.stdout and "standup-smoke" not in p.stdout
    assert [ln for ln in p.stdout.splitlines() if ln.startswith("CLEANUP")] == ["CLEANUP RC=0"] and p.stdout.rstrip().endswith("CLEANUP RC=0")


def test_set_it_checks_every_other_object_cell_under_its_own_tag_then_exits_0_before_phase_b():
    p = _run_region({"F22_STANDUP_SMOKE": "1"})
    assert p.returncode == 0, p.stdout + p.stderr
    pcs = [ln for ln in p.stdout.splitlines() if ln.startswith("PC ")]
    # the first object cell under the loop's tag; every other object cell under its own tag, the liveness record to its own check only
    assert pcs == ["PC cell=cellA tag=attacks rec=NONE",
                   "PC cell=cellB tag=attacks_smoke_cellB rec=/tmp/recB.json",
                   "PC cell=cellC tag=attacks_smoke_cellC rec=NONE"], p.stdout
    assert "PHASE B reached" not in p.stdout
    # the cleanup (stop the target, the RESULT line) runs once, BEFORE the smoke line, which is the job's last line
    out = [ln for ln in p.stdout.splitlines() if ln.strip()]
    assert [ln for ln in out if ln.startswith("CLEANUP")] == ["CLEANUP RC=0"]
    assert out[-2:] == ["CLEANUP RC=0", "LOG [standup-smoke] standup + positive controls passed; exiting before Phase B"], out


@pytest.mark.parametrize("fail_cell", ["cellB", "cellC"])
def test_a_failed_cell_ends_the_smoke_non_zero_and_never_on_the_smoke_line(fail_cell):
    p = _run_region({"F22_STANDUP_SMOKE": "1", "FAIL_CELL": fail_cell})
    assert p.returncode == 1, p.stdout + p.stderr
    assert "standup-smoke" not in p.stdout and "PHASE B reached" not in p.stdout
    assert [ln for ln in p.stdout.splitlines() if ln.startswith("CLEANUP")] == ["CLEANUP RC=1"] and p.stdout.rstrip().endswith("CLEANUP RC=1")
    if fail_cell == "cellB":                                                # the job stops at the first failure
        assert "cellC" not in p.stdout


def test_a_smoke_with_registration_on_is_refused_before_any_emission():
    lines, ((s1, e1), _) = _smoke_blocks()
    block = "\n".join(lines[s1:e1 + 1])
    head = 'set -uo pipefail\nf22_log(){ echo "LOG $*"; }\nf22_fatal(){ f22_log "FATAL: $*"; exit 1; }\n'
    for env, rc in (({}, 0), ({"F22_STANDUP_SMOKE": "1"}, 1), ({"F22_STANDUP_SMOKE": "1", "SISTEER_REGISTRY_DISABLED": "0"}, 1),
                    ({"F22_STANDUP_SMOKE": "1", "SISTEER_REGISTRY_DISABLED": "1"}, 0), ({"SISTEER_REGISTRY_DISABLED": "0"}, 0)):
        p = _run(head + block + '\necho "INIT reached"\n', env)
        assert p.returncode == rc, (env, p.stdout, p.stderr)
        assert ("INIT reached" in p.stdout) == (rc == 0), (env, p.stdout)
        if rc:
            assert "FATAL: F22_STANDUP_SMOKE=1 needs SISTEER_REGISTRY_DISABLED=1" in p.stdout, p.stdout
