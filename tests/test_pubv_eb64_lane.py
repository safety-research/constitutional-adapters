"""The non-production roster ``configs/sdfbase_full22_roster_pubv_eb64.yaml``: the multi-turn attacks on the three PUBLISHED composed
difference ReLU objects (constitution minus control, paired write; batch 64, one per fit seed) and on the unsteered model, in the
evaluation setup of the early-stopped control objects' roster (``configs/sdfbase_full22_roster_ctlstop_eb64.yaml``).

The production-roster tests never open this file, so its contract is pinned here:

* every top-level key but ``targets`` is the production roster's as of the roster's landing, pinned by the sha256 of its canonical JSON
  (so a later change to the production roster does not redden this file; a change here is deliberate), and the target's model, revision
  and body config are pinned likewise;
* exactly three views ``pubv64s<k>`` (k in 1..3), none a production view (common.sh refuses a non-production roster that names one), and
  no view name falls under a grammar another reader keys on (the phase views, the dose views, the paired views, the steered waves, the
  seed-replicate tags and the strength standups; the patterns are imported from their modules, never retyped);
* each view runs the attacks leg alone, omits the other five legs with a reason, and carries no dose, family or paired key;
* each view binds ONE cell in the roster's string form, the staged id the production paired view ``s<k>p`` serves, which resolves through
  the production pins to the published object of that seed (the full sha below) and is a paired-write ReLU difference;
* ``roster_view.py ... has_leg attacks`` admits each view under this roster and refuses every other leg;
* the rendered configs and manifests are the generator's (``make_full22_configs.py --check --roster ...`` exits 0, and each text is
  ``render_config`` / ``render_manifest``'s), each config serving ``none`` and the one cell, with the attacker, attacks and grader blocks of
  the early-stopped control views of the same seed;
* the model key a job registers under (``qwen35_4b_pubv64s<k>``) resolves through ``legacy_spellings.yaml``, with no replicate label;
* the job-start registration over each rendered config plans exactly ``none`` and the published object on both attacks' instruments, with
  run ids that are a pure function of the producer key the job forms and the object key (never of the scheduler job id or the pin-form
  opt-in), and equal to the eight ids pinned as literals;
* the multi-turn grader resolves the three view tags by name;
* the wave-config enumerator names this file as a table."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "scripts" / "sdfbase", REPO / "scripts" / "sdfbase" / "full22", REPO / "src", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import f22_roster as fr  # noqa: E402
import make_full22_configs as gen  # noqa: E402
from full22_wave_configs import EXCLUDED_TABLES  # noqa: E402
from sisteer.capeval.petri.registry_adapter import ARM_OBJECT_KINDS  # noqa: E402
from sisteer.eval.defense_cell import arm_of  # noqa: E402
from sisteer.registry import identity as I  # noqa: E402
from sisteer.registry import jobplan as JP  # noqa: E402
from sisteer.registry import synth  # noqa: E402
from sisteer.registry.adapters import seedrep_cluster as SRC  # noqa: E402
from sisteer.registry.adapters import strength_mtdiff as SMD  # noqa: E402

LANE_ROSTER_PATH = REPO / "configs/sdfbase_full22_roster_pubv_eb64.yaml"
PROD_ROSTER_PATH = REPO / "configs/sdfbase_full22_roster.yaml"
PROD_PINS_REL = "configs/sdfbase_full22_pins.json"
PROD_ROSTER = yaml.safe_load(PROD_ROSTER_PATH.read_text())
LANE_ROSTER = yaml.safe_load(LANE_ROSTER_PATH.read_text())
PROD_PINS = json.loads((REPO / PROD_PINS_REL).read_text())
TARGET = "qwen35_4b"
VIEWS = ("pubv64s1", "pubv64s2", "pubv64s3")
#: the composed ReLU difference (constitution minus control, paired write): the one cell every view serves
CELL = "constv2-mix-full__psr_ce_all_minus_ctl"
#: the published object of each fit seed (production pins `qwen35_4b.paired_eb64s<k>__<CELL>`)
PUBLISHED = {1: "f98f0349673b7cb0cf85f54aee71f74bf9f0bbf981e989b52c8ba6599a492c2d",
             2: "9c639c91d2ef5b8fc482cb039e8ae65deaa79389e1eab7daaa0c45a5caacfc66",
             3: "0718a3f9edfb033ad84c3322f717dc62b32b6298fe48f0a6c42e9f0a7c031647"}
OMITTED = ("static", "bon", "capability", "controls", "petri")
ATTACKS = ("crescendo", "actorattack")
#: the multi-turn instruments of the served instruct checkpoint on the native LoRA path (the seeds' canonical instruments)
INSTRUMENT = {"crescendo": "33f80ed1d4f536e34c870fa36c2d88d90c7623ea17a9d4db178fea008c12ac85",
              "actorattack": "f8832b53b08131258d0e5c9e15a16a2888f8a977271d52dde270d52b1af00203"}
ITEM_SET = "multiturn_goals_259"
#: sha256 of the canonical JSON (sorted keys, compact separators) of the roster's top level minus ``targets``: the production roster's
#: serving contract when this roster was written (serving_impl_commit 94e294c4…, caps, controls window, bon climb, attacker flags, lane flags)
TOP_LEVEL_SHA256 = "12de193b1412a428271407d856c0fc18dffb9356adbd63a2839e9e0e1d37ee15"
#: the shard count every job of the re-measurement serves (target_attacks.sbatch positional 5); it is in the producer key
NSHARDS = 18
#: the eight runs the re-measurement registers, by (view, attack, cell), as LITERALS: the registry names these runs by id before any of
#: them exists (a run-scoped exclusion over all eight, a coexist over the two unsteered ones), so a change to the derivation, the producer
#: key, the instruments or the cell spelling must fail here rather than leave those ids naming nothing. The unsteered model is measured
#: on pubv64s1 only.
PLANNED_RUN_IDS = {
    ("pubv64s1", "crescendo", "none"): "reseed_ctlstop:crescendo:33f80ed1d4f5:c205b827fc1bdeab",
    ("pubv64s1", "crescendo", CELL): "reseed_ctlstop:crescendo:33f80ed1d4f5:a3883e53266d76cd",
    ("pubv64s1", "actorattack", "none"): "reseed_ctlstop:actorattack:f8832b53b081:ff9345e54abe3974",
    ("pubv64s1", "actorattack", CELL): "reseed_ctlstop:actorattack:f8832b53b081:6fdf1c76dcbc09a4",
    ("pubv64s2", "actorattack", CELL): "reseed_ctlstop:actorattack:f8832b53b081:333e305d8bf35407",
    ("pubv64s2", "crescendo", CELL): "reseed_ctlstop:crescendo:33f80ed1d4f5:fb385e893c2dde6a",
    ("pubv64s3", "actorattack", CELL): "reseed_ctlstop:actorattack:f8832b53b081:c3284772c4644579",
    ("pubv64s3", "crescendo", CELL): "reseed_ctlstop:crescendo:33f80ed1d4f5:c77141ad69a37c4b",
}
TARGET_SBATCH = REPO / "scripts/sdfbase/full22/target_attacks.sbatch"
COMMON_SH = REPO / "scripts/sdfbase/full22/common.sh"


def _seed(view: str) -> int:
    assert view in VIEWS, view
    return int(view[-1])


def _staged(view: str) -> str:
    return f"paired_eb64s{_seed(view)}__{CELL}"


def _top_level_sha256(roster: dict) -> str:
    top = {k: v for k, v in roster.items() if k != "targets"}
    return hashlib.sha256(json.dumps(top, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _env() -> dict:
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    env.pop("SISTEER_REGISTRY_PIN_FORM", None)
    return env


def test_the_roster_carries_the_production_contract_and_exactly_the_three_views():
    assert _top_level_sha256(LANE_ROSTER) == TOP_LEVEL_SHA256
    assert LANE_ROSTER["serving_impl_commit"] == "94e294c445dc5eec6ebe0ce3a3e25553004cd4b0" and LANE_ROSTER["target_max_new_tokens"] == 4096
    assert list(LANE_ROSTER["targets"]) == [TARGET]
    t, pt = LANE_ROSTER["targets"][TARGET], PROD_ROSTER["targets"][TARGET]
    assert set(t) == {"model", "model_revision", "body_config", "views"}
    assert (t["model"], t["model_revision"], t["body_config"]) == ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                                                                   "configs/sdfbase_transfer_qwen35_4b_r3.yaml")
    assert tuple(t["views"]) == VIEWS
    assert not set(VIEWS) & set(pt["views"])                         # none is a production view (common.sh f22_bank_view refuses one)
    assert LANE_ROSTER_PATH.name in EXCLUDED_TABLES                   # a roster under the wave-config glob, named as a table


@pytest.mark.parametrize("view", VIEWS)
def test_no_view_name_falls_under_another_readers_grammar(view):
    tag = f"{TARGET}_{view}"
    assert not gen.PHASE_VIEW_RE.match(view)                          # the phase views (byte-equal to the grader's copy, pinned elsewhere)
    assert not gen._DOSE_VIEW_RE.match(view)                          # d<tok> / it_d<tok>[_<family>]
    assert not gen._PAIRED_VIEW_RE.match(view)                        # the paired standups s<k>p
    assert not fr.STEERED_WAVE_VIEW_RE.match(view) and not fr.steered_wave_view(view, TARGET)
    assert not SRC.TAG_RE.match(tag) and not SRC.MT_STORE_RE.match(f"sdfbase_full22_{tag}")   # the seed-replicate adapter's tags and stores
    assert not SMD.STANDUP_RE.match(tag)                              # the strength standups
    assert gen.view_paired_of(LANE_ROSTER, TARGET, view) is None


@pytest.mark.parametrize("view", VIEWS)
def test_each_view_runs_the_attacks_alone_at_the_operating_point_and_binds_the_published_object(view):
    k = _seed(view)
    v = LANE_ROSTER["targets"][TARGET]["views"][view]
    assert set(v) == {"legs", "legs_omitted", "config", "manifest", "cells"}       # no dose / dose_unit / object_family / paired_of, no Petri knobs
    assert fr.view_legs(LANE_ROSTER, TARGET, view) == ("attacks",)
    omitted = fr.view_legs_omitted(LANE_ROSTER, TARGET, view)
    assert set(omitted) == set(OMITTED) and all(isinstance(why, str) and why.strip() for why in omitted.values())
    assert gen.view_dose(LANE_ROSTER, TARGET, view) is None
    assert fr.view_cell_universe(LANE_ROSTER, TARGET, view) == ["none", CELL]
    assert v["config"] == f"configs/sdfbase_full22_{TARGET}_{view}.yaml"
    assert v["manifest"] == f"configs/capeval/manifests/petri_full22_{TARGET}_{view}.yaml"
    # ONE cell, in the string form: the staged id the production paired view of the same seed serves
    assert v["cells"] == {CELL: _staged(view)} and not fr.is_pin_form(v["cells"][CELL])
    assert fr.staged_cells(LANE_ROSTER, TARGET, view) == {CELL: _staged(view)}
    assert PROD_ROSTER["targets"][TARGET]["views"][f"s{k}p"]["cells"][CELL] == _staged(view)
    # the production pins name the published object, a paired-write ReLU difference composed on the seed's batch-64 fits
    pin = PROD_PINS[TARGET][_staged(view)]
    assert pin["object_sha256"] == PUBLISHED[k]
    assert ARM_OBJECT_KINDS[arm_of(CELL)] == "psr_paired_write"
    assert pin["source"].endswith(f"/derived/psr_paired_write/qwen35_4b_base_eb64_s{k}/{CELL}")
    # the job-start registration reads the same object, with and without the pin-form opt-in (a string cell never needs it)
    for pins_root in (None, str(REPO)):
        assert JP.object_id_for(CELL, model_key=TARGET, roster_selection=view, roster=LANE_ROSTER, pins=PROD_PINS, si_sha={},
                                pins_root=pins_root) == PUBLISHED[k]


def test_the_three_objects_are_the_three_published_seeds():
    assert len(set(PUBLISHED.values())) == 3
    assert {PROD_PINS[TARGET][_staged(v)]["object_sha256"] for v in VIEWS} == set(PUBLISHED.values())


@pytest.mark.parametrize("view", VIEWS)
def test_roster_view_admits_the_attacks_leg_only(view):
    env = _env()
    tool = REPO / "scripts/sdfbase/full22/roster_view.py"
    ok = subprocess.run([sys.executable, str(tool), TARGET, view, "has_leg", "attacks", "--roster", str(LANE_ROSTER_PATH)],
                        capture_output=True, text=True, env=env, cwd=REPO, timeout=120)
    assert ok.returncode == 0, ok.stderr
    for leg in OMITTED:
        r = subprocess.run([sys.executable, str(tool), TARGET, view, "has_leg", leg, "--roster", str(LANE_ROSTER_PATH)],
                           capture_output=True, text=True, env=env, cwd=REPO, timeout=120)
        assert r.returncode != 0, (leg, r.stdout, r.stderr)
    count = REPO / "scripts/sdfbase/full22/roster_cell_count.py"
    assert subprocess.run([sys.executable, str(count), TARGET, view, "--roster", str(LANE_ROSTER_PATH)], capture_output=True, text=True,
                          env=env, cwd=REPO, timeout=120).stdout.strip() == "1"
    # the production roster names no such view: the collision guard of common.sh f22_bank_view stays quiet
    assert subprocess.run([sys.executable, str(count), TARGET, view, "--roster", str(PROD_ROSTER_PATH)], capture_output=True, text=True,
                          env=env, cwd=REPO, timeout=120).returncode != 0


def test_the_generator_check_passes_on_the_committed_configs():
    r = subprocess.run([sys.executable, str(REPO / "scripts/sdfbase/make_full22_configs.py"), "--roster", str(LANE_ROSTER_PATH), "--check"],
                       capture_output=True, text=True, cwd=REPO, env=_env(), timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert r.stdout.count("ok: ") == 2 * len(VIEWS) and "DRIFT" not in r.stdout


@pytest.mark.parametrize("view", VIEWS)
def test_the_rendered_config_and_manifest_are_the_generators_and_serve_none_plus_the_published_object(view):
    k = _seed(view)
    v = LANE_ROSTER["targets"][TARGET]["views"][view]
    cfg_path, man_path = REPO / v["config"], REPO / v["manifest"]
    text = cfg_path.read_text()
    assert text == gen.render_config(LANE_ROSTER, gen.load_pins(), TARGET, view, roster_path=LANE_ROSTER_PATH)
    assert man_path.read_text() == gen.render_manifest(LANE_ROSTER, TARGET, view, roster_path=LANE_ROSTER_PATH)
    cfg = yaml.safe_load(text)
    assert cfg["explicit_cells"] == ["none", CELL]
    assert cfg["private_run"] == f"sdfbase_full22_{TARGET}_{view}" and cfg["defenses_dir"] == f"out/full22/{TARGET}/cells_view/{view}"
    assert cfg["attacks"]["target_max_new_tokens"] == LANE_ROSTER["target_max_new_tokens"] == 4096
    assert f"from\n# configs/{LANE_ROSTER_PATH.name} (target {TARGET}, view {view})" in text
    assert f"{CELL} <- {_staged(view)}\n" in text and PUBLISHED[k] in text
    # the evaluation setup of the early-stopped control views of the same seed: the attacker, the attacks and the grader, as data
    ctl = yaml.safe_load((REPO / f"configs/sdfbase_full22_{TARGET}_ctlstop64s{k}.yaml").read_text())
    for block in ("attacker", "attacks", "grader"):
        assert cfg[block] == ctl[block], (view, block)
    # ... and `bon` in ONE key: the control-pool-stopped views declare their own best-of-N cap (`caps: {bon: 4096}`, the seed views'
    # native pass, for their best-of-N leg), while these views run the attacks alone and keep the body's 512
    assert sorted(x for x in set(cfg) | set(ctl) if cfg.get(x) != ctl.get(x)) == ["bon", "defenses_dir", "explicit_cells", "out_dir", "private_run"]
    assert [x for x in sorted(set(cfg["bon"]) | set(ctl["bon"])) if cfg["bon"].get(x) != ctl["bon"].get(x)] == ["max_new_tokens"]
    assert (cfg["bon"]["max_new_tokens"], ctl["bon"]["max_new_tokens"]) == (gen.BON_MAX_NEW_TOKENS, 4096) == (512, 4096)


@pytest.mark.parametrize("view", VIEWS)
def test_the_registration_model_key_of_each_view_is_a_legacy_spelling(view):
    rows = yaml.safe_load((REPO / "configs/registry/legacy_spellings.yaml").read_text())["model_keys"]
    assert JP.model_for_key(rows, f"{TARGET}_{view}") == ("qwen35_4b_it", "base_to_instruct")
    row = next(r for r in rows if r["legacy"] == f"{TARGET}_{view}")
    assert row["fit_model_id"] == "qwen35_4b_base" and "fit_replicate_label" not in row


def _producer_key(job_name: str, key_tag: str, shards_csv: str, nshards: int) -> str:
    """The key the job forms, by the checkout's OWN shell: target_attacks.sbatch's ``export F22_REGISTRY_KEY_SEGMENTS=…`` line evaluated
    with SHARDS / NSHARDS, then common.sh ``f22_registry_producer_key``."""
    seg = [x for x in TARGET_SBATCH.read_text().splitlines() if x.startswith("export F22_REGISTRY_KEY_SEGMENTS=")]
    assert len(seg) == 1, seg
    script = ('set -u; SHARDS="$1"; NSHARDS="$2"; unset F22_REGISTRY_KEY_SEGMENTS; eval "$3"; '
              'source <(sed -n "/^f22_registry_producer_key()/,/^}/p" "$4"); '
              'f22_registry_producer_key "$5" "$6" "$F22_REGISTRY_KEY_SEGMENTS"')
    r = subprocess.run(["bash", "-c", script, "_", shards_csv, str(nshards), seg[0], str(COMMON_SH), job_name, key_tag],
                       capture_output=True, text=True, env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C"}, timeout=60)
    assert r.returncode == 0 and r.stdout.strip(), r.stderr
    return r.stdout.strip()


def _plans(view: str, attack: str, *, job_id: str, pin_form: bool, pkey: str, code_sha: str):
    return JP.build_plans(config_path=str(REPO / f"configs/sdfbase_full22_{TARGET}_{view}.yaml"), model_key=f"{TARGET}_{view}",
                          roster_selection=view, evals=[attack], serving_path="lora_native", campaign="reseed_ctlstop", job_id=job_id,
                          job_name=f"f22-att-synthetic-{attack}", roster_path=str(LANE_ROSTER_PATH),
                          pins_path=str(REPO / PROD_PINS_REL), code_sha=code_sha, substrate="cluster_a", producer_key=pkey,
                          pin_form=pin_form)


@pytest.mark.parametrize("attack", ATTACKS)
@pytest.mark.parametrize("view", VIEWS)
def test_the_job_start_registration_plans_none_and_the_published_object_with_deterministic_run_ids(view, attack, monkeypatch):
    monkeypatch.delenv("SISTEER_REGISTRY_PIN_FORM", raising=False)
    k, tag = _seed(view), f"{TARGET}_{view}"
    shards = ",".join(str(s) for s in range(NSHARDS))
    pkey = _producer_key(f"f22-att-synthetic-{attack}", tag, shards, NSHARDS)
    assert pkey == f"sdfbase_full22-att-{attack}-{tag}-sh{shards.replace(',', '_')}of{NSHARDS}"
    b = _plans(view, attack, job_id="4242", pin_form=False, pkey=pkey, code_sha=synth.code_sha("a"))
    assert b.skipped == [], b.skipped
    assert {r.route["object"]: r.plan["object_id"] for r in b.runs} == {"none": "none", CELL: PUBLISHED[k]}
    for r in b.runs:
        p, obj = r.plan, r.route["object"]
        assert (p["eval_id"], p["instrument_id"], p["item_set_id"]) == (attack, INSTRUMENT[attack], ITEM_SET)
        assert (p["model_id"], p["view_legacy"], p["campaign"], p["serving_path"]) == ("qwen35_4b_it", tag, "reseed_ctlstop", "lora_native")
        assert p["transfer_kind"] == ("none" if obj == "none" else "base_to_instruct")
        assert p["token_cap_tokens"] == 4096 and not {"trial_index", "pass_label", "token_cap_rung_index"} & set(p)
        assert (p["producer_run_key_legacy"], p["producer_object_key_legacy"]) == (pkey, obj)
        assert r.run_id == I.run_id("reseed_ctlstop", pkey, obj, attack, INSTRUMENT[attack])
        if (view, attack, obj) in PLANNED_RUN_IDS:                    # every steered run, and the unsteered runs of pubv64s1
            assert r.run_id == PLANNED_RUN_IDS[(view, attack, obj)], (view, attack, obj, r.run_id)
    # the ids are the measurement's: another scheduler job, another code sha, or the pin-form opt-in plans the same runs
    ids = sorted(r.run_id for r in b.runs)
    for job_id, pin_form, sha in (("9999999", False, synth.code_sha("b")), ("1", True, synth.code_sha("a"))):
        again = _plans(view, attack, job_id=job_id, pin_form=pin_form, pkey=pkey, code_sha=sha)
        assert again.skipped == [] and sorted(r.run_id for r in again.runs) == ids, (job_id, pin_form)


def test_the_pinned_run_ids_are_the_eight_units_of_the_re_measurement():
    """Six steered runs (three objects x two attacks) and two unsteered runs (pubv64s1, both attacks); every id names its campaign, attack
    and instrument, and none repeats. The registration test above checks each one against the plans."""
    assert set(PLANNED_RUN_IDS) == {(v, a, CELL) for v in VIEWS for a in ATTACKS} | {("pubv64s1", a, "none") for a in ATTACKS}
    assert len(set(PLANNED_RUN_IDS.values())) == 8
    for (view, attack, cell), rid in PLANNED_RUN_IDS.items():
        assert I.parse_run_id(rid)["campaign"] == "reseed_ctlstop" and rid.startswith(f"reseed_ctlstop:{attack}:{INSTRUMENT[attack][:12]}:")


def test_the_grader_resolves_the_three_view_tags_by_name():
    code = ("import importlib.util, json\n"
            f"spec = importlib.util.spec_from_file_location('pubv_lane_grader', {str(REPO / 'scripts/sdfbase/seedrep/grade_full22_native.py')!r})\n"
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
            "t, _ = m.resolve_targets()\n"
            f"views = {list(VIEWS)!r}\n"
            "print(json.dumps({'targets': {k: [v['run'], list(v['default_attacks'])] for k, v in t.items() if k.startswith('qwen35_4b_pubv')},\n"
            "                  'phase': [bool(m.PHASE_VIEW_RE.match(v)) for v in views], 'seedrep': [bool(m.SEEDREP_VIEW_RE.match(v)) for v in views]}))")
    env = dict(_env(), HF_HUB_OFFLINE="1",
               PYTHONPATH=os.pathsep.join(str(REPO / p) for p in ("src", "scripts", "scripts/sdfbase/seedrep")))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert got["targets"] == {f"{TARGET}_{v}": [f"sdfbase_full22_{TARGET}_{v}", ["crescendo", "actorattack"]] for v in VIEWS}
    assert got["phase"] == [False] * 3 and got["seedrep"] == [False] * 3          # no grammar is widened for them
