"""The SI-distilled refit views ``sikl4s<k>`` (configs/sdfbase_full22_roster_sikl4.yaml) carry best-of-N, the over-refusal and capability
controls and GSM8K beside the static families and the multi-turn attacks, on the instruments and draw procedure of the published
seed-view rows:

* the eighteen objects are the lane pins, six per view, in the rendered configs' order;
* the nine sampled configs ``configs/sdfbase_full22_qwen35_4b_sikl4s<k>_sampled_t0{,_cap8192,_cap16384}.yaml`` are the generator's output
  under the lane roster, byte for byte, each at seed base k (the objects' fit seed), ONE trial, trial 0 and the Qwen vendor preset
  (T 0.7 / top_p 0.8 / top_k 20 / min_p 0 / presence 1.5 / repetition 1.0), the static cap of its rung and the controls cap 512 (the root of
  the over-refusal ladder 512 -> 2048 -> 8192 the controls phase climbs in-job);
* each wave config draws best-of-N natively at 4,096 (the view's ``caps: {bon: 4096}``) and still serves ``[none, the six cells]``;
* ``roster_view.py ... controls`` lists mmlu, xstest and or_bench_hard_1k on every view (the capability and controls legs);
* the seeded-lane seed rule is ONE rule: the sampled generator's table, the decode-preset file and the GSM8K views module name the same
  three views at the same seeds;
* the job-start registration (``jobplan.build_plans``, pin form on, campaign ``si_target_refit``) binds pasttense / codeattack / renellm /
  grok420 / pair / xstest / or_bench_hard / gsm8k to the published instruments with the eighteen object ids, while best-of-N binds the
  canonical 512-store instrument and MMLU is refused (two canonical seeds): those two register in-job through the lane's own seeds;
* the wave's job names register their families: the PAIR defenders ``f22-att-sibat-s<k>all-pair`` under a job-independent producer key,
  the over-refusal array ``f22s-sibat-orxs_t0`` as the sampled kind with the trial its stores' close reads;
* the ``sikl4`` GSM8K matrix is registered in every table and launcher, regenerates byte-identically from the lane roster + pins its
  views file sha-pins, and holds 18 launch rows (no ``none`` row), each at seed base k under the replicate rule, the common geometry
  class and the view's constitution-SI LoRA as the standup companion, with its own overlay pair;
* no ``sikl4`` row is DECLARED: the two safety cells share their names with the header-located scale-12 objects of ``main`` but act at
  every position, so the header-located rule is scoped to the ``main`` views (``g8k_cells.HEADER_LOCATED_VIEWS``) and every other
  matrix is unchanged;
* the six cells run thinking-ON under their OWN out-of-support pair ``gsm8k_paired200_qwen35_4b_native_sikl4_<cap>`` (native identity,
  undeclared), beside the native battery's and the lens-era declared pair, which keep their doors;
* the GSM8K driver admits a row's ``--bank-view`` only from the roster it is handed: the launcher exports the job's roster for it, and
  every row's argv — the 16,384 rung form included — parses under it and is refused at the parser without it.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts" / "sdfbase", REPO / "scripts" / "sdfbase" / "full22", REPO / "scripts" / "sdfbase" / "full22" / "sampled",
           REPO / "scripts" / "sdfbase" / "seedrep", REPO / "scripts" / "sdfbase" / "gsm8k", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import eval_seed as ES  # noqa: E402
import g8k_cells as M  # noqa: E402
import g8k_views as V  # noqa: E402
import make_gsm8k_matrix as MK  # noqa: E402
import make_gsm8k_overlays as OV  # noqa: E402
import make_sampled_configs as msc  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402
from sisteer.registry import jobplan as JP  # noqa: E402
from sisteer.registry import synth  # noqa: E402

TARGET = "qwen35_4b"
VIEWS = ("sikl4s1", "sikl4s2", "sikl4s3")
#: per SI the LoRA then the ReLU steer, the rendered configs' order after `none`
CELLS = ("safety_refusal__lora_r4", "safety_refusal__psr_kl_all", "si_constitution__lora_r4", "si_constitution__psr_kl_all",
         "C16__lora_r4", "C16__psr_kl_all")
COMPANION = "si_constitution__lora_r4"                    # the view's constitution-SI LoRA (rank 4 -> --max-lora-rank 8)
LANE_ROSTER_REL = "configs/sdfbase_full22_roster_sikl4.yaml"
LANE_ROSTER_PATH = REPO / LANE_ROSTER_REL
PROD_ROSTER_PATH = REPO / "configs/sdfbase_full22_roster.yaml"
LANE_PINS_REL = "configs/sikl4_pins.json"
LANE_PINS = json.loads((REPO / LANE_PINS_REL).read_text())
CAMPAIGN = "si_target_refit"
#: the eighteen objects, by (view, cell): the pins file is the authority, these literals the cross-check
OBJECTS = {
    ("sikl4s1", "safety_refusal__lora_r4"): "e3d61a63cfd0e4d4ff1462a5760513a5dacf459d3352f29b81834fa08b122445",
    ("sikl4s2", "safety_refusal__lora_r4"): "0952ae4f12e3f4f4bc63bcefe609d621df5f01f0b13ac9ae9293cc8f720b2bad",
    ("sikl4s3", "safety_refusal__lora_r4"): "627cfb19664cdc510be08c6282cc324d456d0d53776e807bb929c492b5ebb950",
    ("sikl4s1", "safety_refusal__psr_kl_all"): "b37fb015b4b0f71f8785df76826cff5d738d1dee6eaac5759a30ae3d496caa04",
    ("sikl4s2", "safety_refusal__psr_kl_all"): "b2882f69628ecfe0e6846d57fe9f358ef569d8cbc93fa1c3f10a4249dcf9d29a",
    ("sikl4s3", "safety_refusal__psr_kl_all"): "b073c88aef7a422e1d7e932e09968e3135987e3c4972aa5ca38e60ca1e0f125e",
    ("sikl4s1", "si_constitution__lora_r4"): "10c3f366540ee7c1642974ff9653badcdbb8f1979c530865c8de7a4e11d9172c",
    ("sikl4s2", "si_constitution__lora_r4"): "11d790ce754fadc68057d98d1f24f6cc54732a10306be890c88e416658cf7931",
    ("sikl4s3", "si_constitution__lora_r4"): "0888b24b140a8969daf41a47e291ea06ce817133e136b44be8dd339cca9ab5d2",
    ("sikl4s1", "si_constitution__psr_kl_all"): "55edaec8a4db665e8547a7b94257582a1405cfcfeb5feedf6e2ab5dbabd5ce68",
    ("sikl4s2", "si_constitution__psr_kl_all"): "4ae1341a8a29f7967137adcb243fadd45963f968c15db34505d31af19e10ca3b",
    ("sikl4s3", "si_constitution__psr_kl_all"): "790188c523fd1114abd6695af5d55b8aa8b9b553f5ba8126f59537d516f963ce",
    ("sikl4s1", "C16__lora_r4"): "80a8f416a634aa7a49dca1e9da9f68c14889d3a6be2a6177ce2ef3a5b2cf2db8",
    ("sikl4s2", "C16__lora_r4"): "eb719f4222051e3bfd5104214c03769ed007080ddd14f75e13a29068f89b110f",
    ("sikl4s3", "C16__lora_r4"): "332845ea44c7a5538c896ff653229f8046858bae0fa8b5f0a4ff2af57346be70",
    ("sikl4s1", "C16__psr_kl_all"): "a473f52ddcd72931112030f70164d8cd9d9ea2716e84e564d2d4afcf01f21bb2",
    ("sikl4s2", "C16__psr_kl_all"): "0590e9dec5bdd293c414e3529c926d4884d86111d1f8691874fe3d7fbe6a33ae",
    ("sikl4s3", "C16__psr_kl_all"): "85bc02be19e48b1b666e015c0e0f09801b4c1d8a4c74908a73b83e4cb4cbb61e",
}
#: the published instruments the seed-view rows sit on (the registry's instrument table)
INSTRUMENT = {
    "pasttense": "14942170c71578e7a58daee14ebb6cb074e7575a8399f5a17ac63596456010d3",
    "codeattack": "7c6751db529f9f637926fbfcc0cac21254ddd9adb8e8cdb23c8724234e7de286",
    "renellm": "58ad64eb969e23e3959fcf3832348f0cf99fc89dd3879cdc9e22dc2f8576b0e8",
    "grok420": "f71ed06cf77fea63ad450bff882f4f7097ed99ba4643c6b9658f57ef8e3ffc04",
    "pair": "373118ba258c81ebc306e1d7d3ace8acdc53807a96b7277d3931ef961b2311ed",
    "xstest": "28166871df5f55f26ea3ec1fff110ab272e16b7c93f5cf8f32bf034bf408bbe9",
    "or_bench_hard": "e5804e74d4dd459a9911940f95d55a018e0dd53d3ddf3e0b51de8faec0117b87",
    "gsm8k": "5ea36b254426cba37be52b4c9eb07a740e23d48a2dff82336f40685e6322f5c5",
}
#: the canonical best-of-N instrument native registration binds (the 512-store climb) — NOT the seed views' native 4,096 pass
CANONICAL_BON = "78b4699fb41a79a40fc9029f72c9a1bf4b711a8e4fa28aa9671559a24eb38c57"
SAMPLED_EVALS = ["pasttense", "codeattack", "renellm", "grok420", "xstest", "or_bench_hard"]   # common.sh f22_registry_evals_for_job f22s-*
VENDOR = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}
RUNG_CAPS = {"": 4096, "_cap8192": 8192, "_cap16384": 16384}
MATRIX_NAME = "sikl4"
G8K = REPO / "scripts/sdfbase/gsm8k"
COMMON_SH = REPO / "scripts/sdfbase/full22/common.sh"
TARGET_ATTACKS = REPO / "scripts/sdfbase/full22/target_attacks.sbatch"
ENV = {**os.environ, "PYTHONPATH": str(REPO / "src")}
#: the wave's job names (the lane's wave tools): the PAIR defender per view, the over-refusal array
DEFENDER_NAME = "f22-att-sibat-s{k}all-pair"
CONTROLS_NAME = "f22s-sibat-orxs_t0"


def _k(view: str) -> int:
    return int(view[-1])


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_the_eighteen_objects_are_the_lane_pins():
    assert set(LANE_PINS[TARGET]) == {f"{v}__{c}" for v in VIEWS for c in CELLS}
    for (view, cell), sha in OBJECTS.items():
        assert LANE_PINS[TARGET][f"{view}__{cell}"]["object_sha256"] == sha, (view, cell)


# --------------------------------------------------------------------------- the sampled configs (static + over-refusal)
def test_the_nine_sampled_configs_are_the_generators_at_the_fit_seed_one_trial_and_the_vendor_preset(monkeypatch):
    presets = msc.load_presets()
    monkeypatch.setattr(msc, "_ROSTER_PATH", LANE_ROSTER_PATH.resolve())
    assert msc._private() and msc.scope_views(TARGET) == VIEWS and all(msc.in_scope(TARGET, v) for v in VIEWS)
    msc.check_pins(presets, msc.load_roster())
    outs = msc.outputs(presets)
    want = {REPO / f"configs/sdfbase_full22_{TARGET}_{v}_sampled_t0{sfx}.yaml" for v in VIEWS for sfx in RUNG_CAPS}
    assert len(outs) == 9 and {p for p, *_ in outs} == want
    for path, tag, trial, esc in outs:
        assert path.is_file() and path.read_text() == msc.render(presets, tag, trial, esc), path.name   # byte-identical (--check)
        view = tag.split(f"{TARGET}_", 1)[1]
        doc = yaml.safe_load(path.read_text())
        sd = doc["sampled_decode"]
        sfx = path.stem.split("_sampled_t0", 1)[1]
        assert (sd["arm"], sd["trial"], sd["n_trials"], sd["seed_base"]) == ("vendor_recommended", 0, 1, _k(view)), path.name
        assert {x: sd[x] for x in VENDOR} == VENDOR and sd["source_kind"] == "model_card", path.name
        assert sd["static_cap"] == RUNG_CAPS[sfx] and sd["controls_cap"] == 512, path.name
        assert doc["explicit_cells"] == ["none", *CELLS] and doc["bon"]["max_new_tokens"] == 4096, path.name
    assert msc.stale_outputs(presets) == []


def test_the_seeded_lane_rule_names_the_three_views_at_their_fit_seed_in_every_table():
    presets = msc.load_presets()
    assert set(VIEWS) <= set(msc.SEEDED_LANE_VIEWS[TARGET])
    for view in VIEWS:
        k, tag = _k(view), msc.tag_of(TARGET, view)
        assert msc.seed_of_lane_view(view) == k == ES.seed_base(k)
        assert msc.SEED_BASE_RULE_BY_TAG[tag] == presets["seed_base_by_view"][tag] == k != presets["seed_base"]
        assert presets["n_trials_by_view"][tag] == 1 != presets["n_trials"]
        # the GSM8K views module carries the SAME rule for the same views (one rule, two stdlib-free readers)
        assert V.SEEDED_LANE_VIEW_SEEDS[view] == k == V.replicate_seed_of_view(view)
    assert {v for v in V.SEEDED_LANE_VIEW_SEEDS if v.startswith("sikl4")} == set(VIEWS)
    with pytest.raises(ValueError):
        V.replicate_seed_of_view("sikl4s4")


# --------------------------------------------------------------------------- the wave configs and the roster's legs
@pytest.mark.parametrize("view", VIEWS)
def test_each_wave_config_draws_best_of_n_natively_at_4096_and_serves_the_six_cells(view):
    cfg = yaml.safe_load((REPO / f"configs/sdfbase_full22_{TARGET}_{view}.yaml").read_text())
    assert cfg["bon"]["max_new_tokens"] == 4096 and cfg["explicit_cells"] == ["none", *CELLS]
    assert cfg["attacks"]["static_max_new_tokens"] == 4096 and cfg["attacks"]["target_max_new_tokens"] == 4096
    assert "pair" in cfg["attacks"]["attacker_sampling"]                      # the PAIR attacker's sampler rides the wave config


@pytest.mark.parametrize("view", VIEWS)
def test_roster_view_lists_the_capability_and_over_refusal_controls(view):
    r = subprocess.run([sys.executable, str(REPO / "scripts/sdfbase/full22/roster_view.py"), TARGET, view, "controls",
                        "--roster", str(LANE_ROSTER_PATH)], capture_output=True, text=True, env=ENV, cwd=REPO, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().split(",") == ["mmlu", "xstest", "or_bench_hard_1k"]          # gsm8k is retired from the controls leg


# --------------------------------------------------------------------------- the job-start registration
def _plans(cfg_rel: str, evals: list, view: str, *, objects=None, producer_key=None):
    return JP.build_plans(config_path=str(REPO / cfg_rel), model_key=f"{TARGET}_{view}", roster_selection=view, evals=evals,
                          serving_path="lora_native", campaign=CAMPAIGN, job_id="4242", job_name="synthetic-sikl4-battery",
                          roster_path=str(LANE_ROSTER_PATH), pins_path=str(REPO / "configs/sdfbase_full22_pins.json"),
                          code_sha=synth.code_sha("a"), substrate="cluster_a", producer_key=producer_key, objects=objects, pin_form=True)


def _bound(bundle, view: str) -> dict:
    """``{(eval, object cell): (instrument id, object id)}`` of a plan bundle that skipped nothing, every run on the lane's campaign,
    the native path, the served instruct model and the view's own legacy spelling."""
    assert bundle.skipped == [], bundle.skipped
    out = {}
    for r in bundle.runs:
        p = r.plan
        assert (p["campaign"], p["serving_path"], p["model_id"], p["view_legacy"]) == (CAMPAIGN, "lora_native", "qwen35_4b_it", f"{TARGET}_{view}")
        out[(p["eval_id"], r.route["object"])] = (p["instrument_id"], p["object_id"])
    return out


@pytest.mark.parametrize("view", VIEWS)
def test_the_job_start_registration_binds_the_published_instruments_with_the_eighteen_objects(view, monkeypatch):
    monkeypatch.delenv("SISTEER_REGISTRY_PIN_FORM", raising=False)
    objs = {c: OBJECTS[(view, c)] for c in CELLS}
    # static + over-refusal: the sampled lane's evals from the sampled config (the f22s-* job kind)
    got = _bound(_plans(f"configs/sdfbase_full22_{TARGET}_{view}_sampled_t0.yaml", SAMPLED_EVALS, view,
                        producer_key=f"sdfbase_full22_{TARGET}_{view}_sampled_t0"), view)
    # PAIR: the wave config (the f22-att-*-pair job kind)
    got.update(_bound(_plans(f"configs/sdfbase_full22_{TARGET}_{view}.yaml", ["pair"], view), view))
    for ev in SAMPLED_EVALS + ["pair"]:
        assert got[(ev, "none")] == (INSTRUMENT[ev], "none")
        for c, sha in objs.items():
            assert got[(ev, c)] == (INSTRUMENT[ev], sha), (ev, c)
    # GSM8K: one row per job, registered from the row's own overlay narrowed to its cell
    for c, sha in objs.items():
        g = _bound(_plans(f"configs/sdfbase/gsm8k/sdfbase_full22_{TARGET}_{view}.gsm8k_{M.cell_slug(c)}.yaml", ["gsm8k"], view, objects=[c]), view)
        assert g == {("gsm8k", c): (INSTRUMENT["gsm8k"], sha)}
    # best-of-N natively binds the canonical 512-store instrument, never the seed views' native pass; MMLU natively refuses (two canonical
    # seeds): both families register through the lane's own seeds in-job, never through this path
    bon = _plans(f"configs/sdfbase_full22_{TARGET}_{view}.yaml", ["strongreject_bon"], view)
    assert bon.skipped == [] and {r.plan["instrument_id"] for r in bon.runs} == {CANONICAL_BON}
    mmlu = _plans(f"configs/sdfbase_full22_{TARGET}_{view}.yaml", ["mmlu"], view)
    assert mmlu.runs == [] and len(mmlu.skipped) == 1 and "2 canonical seeded instruments for eval 'mmlu'" in mmlu.skipped[0][1]


def _job_kind(job_name: str, key_tag: str, nshards: int) -> tuple[str, str]:
    """(evals, producer key) the job forms, by the checkout's OWN shell: target_attacks.sbatch's ``export F22_REGISTRY_KEY_SEGMENTS=…``
    line evaluated with SHARDS / NSHARDS, then common.sh ``f22_registry_evals_for_job`` and ``f22_registry_producer_key``."""
    seg = [x for x in TARGET_ATTACKS.read_text().splitlines() if x.startswith("export F22_REGISTRY_KEY_SEGMENTS=")]
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


@pytest.mark.parametrize("view", VIEWS)
def test_the_pair_defender_name_registers_pair_under_a_job_independent_key(view):
    name, key_tag, n = DEFENDER_NAME.format(k=_k(view)), f"{TARGET}_{view}", 27
    evals, key = _job_kind(name, key_tag, n)
    assert evals == "pair", (name, evals)                               # common.sh f22_registry_register reads the eval off the name
    assert key == f"sdfbase_full22-att-pair-{key_tag}-sh{'_'.join(str(s) for s in range(n))}of{n}", key
    assert not name.startswith("scanary")                               # never the steered canary's name pattern


def test_the_over_refusal_array_name_is_the_sampled_kind_and_carries_the_trial_its_close_reads():
    """The over-refusal array registers as the sampled kind (its evals include xstest / or_bench_hard), and its name carries `_t0`:
    registry_close_stores.py's controls layout takes the trial from the job name when no --trial is passed, and common.sh passes none."""
    evals, key = _job_kind(CONTROLS_NAME, f"{TARGET}_sikl4s1", 9)
    assert evals.split() == SAMPLED_EVALS and key == ""                  # the sampled kind keeps jobplan's default key
    import registry_close_stores as RCS                                   # scripts/sdfbase/full22 is on sys.path (module top)
    assert RCS.trial_of(CONTROLS_NAME, None) == 0 and RCS.trial_of("f22s-sibat-orxs", None) is None


# --------------------------------------------------------------------------- the GSM8K matrix
def test_the_sikl4_matrix_is_registered_in_every_table_and_launcher():
    assert M.MATRICES[MATRIX_NAME] == REPO / "configs/sdfbase/gsm8k/matrix_sikl4.json"
    assert M.VIEWS_BY_MATRIX[MATRIX_NAME] == REPO / "configs/sdfbase/gsm8k/views_sikl4.yaml"
    assert M.WAVE_SIKL4 == MATRIX_NAME and M.out_root_prefix_for(M.WAVE_SIKL4) == "controls_sikl4"
    assert M.WAVE_SIKL4 not in M.COMPLETED_WAVES
    for name in ("gsm8k_thinking.sbatch", "ladder_plan.sbatch", "dry_run_prod.sbatch", "smoke_not_available.sbatch"):
        m = re.search(r'case "\$G8K_MATRIX" in ([a-z0-9_|]+)\) ;;', (G8K / name).read_text())
        assert m and MATRIX_NAME in m.group(1).split("|"), name
    regen = (REPO / "scripts/release/regen_configs.sh").read_text()
    for line in ('"scripts/sdfbase/gsm8k/make_gsm8k_matrix.py --matrix-name sikl4"', '"scripts/sdfbase/gsm8k/make_gsm8k_overlays.py --matrix sikl4"',
                 '"scripts/sdfbase/full22/sampled/make_sampled_configs.py --roster configs/sdfbase_full22_roster_sikl4.yaml"',
                 '"scripts/sdfbase/make_full22_configs.py --roster configs/sdfbase_full22_roster_sikl4.yaml"'):
        assert line in regen, line


def test_the_sikl4_views_file_pins_the_lane_roster_and_pins_and_the_matrix_regenerates_byte_identically():
    spec = V.load_views_yaml(M.VIEWS_BY_MATRIX[MATRIX_NAME])
    assert spec.schema == M.MATRIX_SCHEMA_2 and spec.wave == M.WAVE_SIKL4 and spec.row_overlays
    assert spec.views == {TARGET: list(VIEWS)} and spec.holds == []
    assert [(e["view"], e["cells"]) for e in spec.exclude] == [(v, ["none"]) for v in VIEWS]
    assert spec.inspect_seed_policy == M.INSPECT_SEED_POLICY_REPLICATE and spec.inspect_seed_base_by_view == {v: _k(v) for v in VIEWS}
    assert spec.standup_companions["cell"] == COMPANION and spec.geometry_class == M.GEOMETRY_CLASS_COMMON_F22
    assert spec.trigger_threshold == M.TRIGGER_THRESHOLD and spec.wave_root_rel == "exp/gsm8k_paired200_paper_objects_sikl4"
    assert V.frozen_input_paths(spec, REPO) == (LANE_ROSTER_PATH, REPO / LANE_PINS_REL)
    text, doc = MK.build(configs_root=REPO, views_yaml=None, views_spec=None, matrix_name=MATRIX_NAME)
    assert M.MATRICES[MATRIX_NAME].read_text() == text
    # the document names the lane-private files it was derived from, never the production roster
    assert doc["source"] == {"views_yaml": "configs/sdfbase/gsm8k/views_sikl4.yaml",
                             "roster": LANE_ROSTER_REL, "roster_sha256": _sha256(LANE_ROSTER_PATH),
                             "pins": LANE_PINS_REL, "pins_sha256": _sha256(REPO / LANE_PINS_REL)}


def test_the_sikl4_matrix_holds_eighteen_undeclared_launch_rows_at_the_fit_seed_with_the_companion_and_no_unsteered_row():
    doc = M.load_matrix(M.MATRICES[MATRIX_NAME])
    rows = doc["rows"]
    assert len(rows) == 18 and [r["index"] for r in rows] == list(range(18))
    assert [(r["view"], r["cell"]) for r in rows] == [(v, c) for v in VIEWS for c in CELLS]          # no `none` row
    for r in rows:
        view, cell, k = r["view"], r["cell"], _k(r["view"])
        assert r["model_key"] == TARGET and r["tag"] == f"{TARGET}_{view}" and r["status"] == M.STATUS_LAUNCH and r["fitted"]
        assert r["staged_id"] == f"{view}__{cell}" and r["object_sha256"] == OBJECTS[(view, cell)]
        assert r["wave"] == M.WAVE_SIKL4 and r["own_overlay"] and r["out_root"].startswith(f"controls_sikl4/{TARGET}_{view}/")
        assert (r["inspect_seed_policy"], r["inspect_seed_base"]) == (M.INSPECT_SEED_POLICY_REPLICATE, k) and M.row_inspect_seed(r) == ("replicate", k)
        assert r["thinking_mode"] == M.THINKING_ON and r["sampling_mode"] == M.SAMPLING_THINKING_CARD
        assert M.row_geometry_class(r) == M.GEOMETRY_CLASS_COMMON_F22
        assert M.geometry_for_cell(cell, M.CAP, model_key=TARGET, geometry_class=M.row_geometry_class(r)) == ("capability:gen_long_gsm8k_common_f22", 16640)
        # every object acts at every position: no row is declared, the safety cells included (their names are the header-located
        # scale-12 objects' of `main` only)
        assert r["declared"] is False and r["served_positions_expected"] == "all" and r["standup_cell"] == cell, (view, cell)
        assert r["tuple_name"] == "gsm8k_paired200_qwen35_4b_8192" and "thinking_on" in r["expected_stamps"]
        if cell == COMPANION:
            assert M.row_companions(r) == []                     # the companion's own row registers itself
        else:
            assert M.row_companions(r) == [(COMPANION, V.harness_cell_id(COMPANION), f"{view}__{COMPANION}", OBJECTS[(view, COMPANION)])]
        assert M.launch_cells_for(r)[0] == "none"                # the standup serves the base route, unmeasured
    assert [r["index"] for r in M.eligible_rows(M.WAVE_SIKL4, rows)] == list(range(18))


def test_the_header_located_rule_is_scoped_to_the_main_views_and_the_published_declared_rows_are_unchanged():
    assert M.HEADER_LOCATED_VIEWS == (M.VIEW_MAIN,) and M.HEADER_LOCATED_CELLS == C.G8K_HEADER_LOCATED_CELLS
    published = M.load_matrix(M.MATRICES[M.MATRIX_DEFAULT_NAME])["rows"]
    declared = [(r["model_key"], r["view"], r["cell"]) for r in published if r.get("declared")]
    assert declared == [("gemma4e4b", "main", "safety_refusal__psr_kl_all"), ("gemma4e4b", "main", "safety_refusal__lora_r4"),
                        ("qwen35_4b", "main", "safety_refusal__psr_kl_all"), ("qwen35_4b", "main", "safety_refusal__lora_r4")]
    for r in published:
        if r.get("fitted"):
            assert r["declared"] is (r["cell"] in M.HEADER_LOCATED_CELLS and r["view"] in M.HEADER_LOCATED_VIEWS), (r["view"], r["cell"])
    # every committed matrix regenerates byte-identically under the scoped rule (a declared row would move a document)
    for name in (M.MATRIX_DEFAULT_NAME, M.MATRIX_LLAMA_NAME, M.MATRIX_ITDOC_NAME, M.MATRIX_QWEN_SEEDREP_NAME, M.MATRIX_ALLPOS_NAME,
                 M.MATRIX_CTLSTOP_NAME, MATRIX_NAME):
        text, _doc = MK.build(configs_root=REPO, views_yaml=None, views_spec=None, matrix_name=name)
        assert M.MATRICES[name].read_text() == text, name


def test_every_sikl4_row_serves_through_its_own_overlay_pair_cut_from_the_campaign_config():
    rows = M.load_matrix(M.MATRICES[MATRIX_NAME])["rows"]
    outs = list(OV.outputs(REPO, rows))
    assert len(outs) == 36
    for path, text in outs:
        assert path.is_file() and path.read_text() == text, path.name                                   # byte-identical (--check)
    for r in rows:
        tag, slug = r["tag"], M.cell_slug(r["cell"])
        cfg = yaml.safe_load((REPO / f"configs/sdfbase/gsm8k/sdfbase_full22_{tag}.gsm8k_{slug}.yaml").read_text())
        man = yaml.safe_load((REPO / f"configs/capeval/manifests/petri_full22_{tag}.gsm8k_{slug}.yaml").read_text())["petri"]
        companions = [] if r["cell"] == COMPANION else [COMPANION]
        assert cfg["explicit_cells"] == ["none", r["cell"], *companions]
        assert man["cells"] == ["none", r["harness_cell"], *[V.harness_cell_id(c) for c in companions]]
        campaign = yaml.safe_load((REPO / f"configs/sdfbase_full22_{tag}.yaml").read_text())
        assert {k: v for k, v in cfg.items() if k != "explicit_cells"} == {k: v for k, v in campaign.items() if k != "explicit_cells"}


def test_the_views_loader_holds_each_lane_views_base_to_its_fit_seed(tmp_path):
    text = M.VIEWS_BY_MATRIX[MATRIX_NAME].read_text()
    bad = tmp_path / "views_sikl4.yaml"
    bad.write_text(text.replace("  sikl4s2: 2\n", "  sikl4s2: 1\n", 1))
    with pytest.raises(SystemExit, match=r"inspect_seed_base_by_view.sikl4s2 = 1 but eval_seed.seed_base\(2\) = 2"):
        V.load_views_yaml(bad)


# --------------------------------------------------------------------------- the out-of-support pair the six cells run under
def test_the_sikl4_cells_have_their_own_native_gsm8k_pair_beside_every_other_tuple():
    names = [C.g8k_native_sikl4_tuple_name(TARGET, cap) for cap in C.G8K_CAPS]
    assert names == ["gsm8k_paired200_qwen35_4b_native_sikl4_8192", "gsm8k_paired200_qwen35_4b_native_sikl4_16384"]
    assert C.G8K_NATIVE_SIKL4_MODELS == (TARGET,) and C.G8K_NATIVE_SIKL4_CELLS == CELLS and C.G8K_NATIVE_SIKL4_TUPLE_INFIX == "_native_sikl4"
    # a NEW set of cells: none is a lane cell of the native battery or the paired arm, none an all-position published cell
    assert not set(CELLS) & (set(C.G8K_NATIVE_CELLS) | set(C.G8K_NATIVE_PAIRED_CELLS) | set(C.G8K_ALL_POSITION_CELLS))
    for name, cap in zip(names, C.G8K_CAPS):
        exp = C.AUTHORISED_OOS_EXPERIMENTS[name]
        base = C.AUTHORISED_OOS_EXPERIMENTS[C.g8k_native_tuple_name(TARGET, cap)]
        # every field the native battery's pair at this cap has, but the cells and the prose
        assert exp._replace(cells=base.cells, what=base.what) == base
        assert exp.reduction_mode == C.REDUCTION_MODE_NOT_APPLICABLE and exp.header_located is False and exp.declares is False
        assert exp.max_tokens_on == {C.GSM8K_PAIRED_TASK_NAME: cap} and name in C.G8K_EXPERIMENT_NAMES
    C.check_authorised_render_facts()


@pytest.mark.parametrize("cell", CELLS)
def test_each_sikl4_cell_is_authorised_at_each_cap_under_the_native_identity_only(cell):
    for cap, other in ((8192, 16384), (16384, 8192)):
        for spelling in (cell, C.parse_launcher_cell(cell).harness_id):
            req = dict(model=TARGET, cell=spelling, controls=["gsm8k"], mode="on", sampling_mode="thinking_card", samples=1,
                       reduction_mode=C.REDUCTION_MODE_NOT_APPLICABLE)
            assert C.check_out_of_support_thinking_authorised(**req, max_tokens_by_task={"gsm8k_paired200": cap}) == \
                C.g8k_native_sikl4_tuple_name(TARGET, cap)
            assert C.check_out_of_support_thinking_authorised(**req, max_tokens_by_task={"gsm8k_paired200": other}) == \
                C.g8k_native_sikl4_tuple_name(TARGET, other)
            stacked = {**req, "reduction_mode": C.REDUCTION_MODE_STACKED}
            if cell in C.G8K_HEADER_LOCATED_CELLS:
                # the lens-era DECLARED door of `main`'s header-located objects keeps its own run: the stated identity tells them apart
                assert C.check_out_of_support_thinking_authorised(**stacked, max_tokens_by_task={"gsm8k_paired200": cap}) == \
                    C.g8k_tuple_name(TARGET, cap, declared=True)
                unstated = {k: v for k, v in req.items() if k != "reduction_mode"}
                with pytest.raises(C.OutOfSupportAuthorisationAmbiguous, match="ambiguous authorisation"):
                    C.check_out_of_support_thinking_authorised(**unstated, max_tokens_by_task={"gsm8k_paired200": cap})
            else:
                with pytest.raises(C.OutOfSupportThinkingUnauthorised):          # the lens-era identity is never this door
                    C.check_out_of_support_thinking_authorised(**stacked, max_tokens_by_task={"gsm8k_paired200": cap})
    # another model never borrows the pair
    with pytest.raises(C.OutOfSupportThinkingUnauthorised):
        C.check_out_of_support_thinking_authorised(model="gemma4e4b", cell=cell, controls=["gsm8k"], mode="on", sampling_mode="thinking_card",
                                                   samples=1, reduction_mode=C.REDUCTION_MODE_NOT_APPLICABLE,
                                                   max_tokens_by_task={"gsm8k_paired200": 8192})


# --------------------------------------------------------------------------- the GSM8K driver admits the lane views from the job's roster
def _driver_plans(tmp_path: Path, rows: str, *, rung_rows: str | None = None, roster: str | None = LANE_ROSTER_REL) -> list:
    """The dry run's DRIVER: lines for the given rows of the matrix (the launcher's ONE argv function, ``g8k_driver_argv``), each with
    the ROSTER_ENV: line printed after it: ``[(argv, variable name, value)]``. ``roster`` is the submit line's ``F22_ROSTER`` — the
    lane roster, as every job of this matrix is submitted; ``None`` leaves it off. The driver's own variable is never inherited."""
    env = {k: v for k, v in os.environ.items() if k not in (C.ENV_FULL22_ROSTER, "F22_ROSTER")}
    env["PYTHONPATH"] = str(REPO / "src")
    if roster is not None:
        env["F22_ROSTER"] = roster
    r = subprocess.run([sys.executable, str(G8K / "g8k_dry_run.py"), "--stage", "pre", "--matrix", MATRIX_NAME, "--rows", rows,
                        *(["--rung-rows", rung_rows] if rung_rows else []), "--tmp", str(tmp_path / "t"), "--python", sys.executable],
                       capture_output=True, text=True, env=env, cwd=REPO, timeout=900)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    lines, out = r.stdout.splitlines(), []
    for i, line in enumerate(lines):
        if not line.startswith("DRIVER:"):
            continue
        argv = shlex.split(line[len("DRIVER:"):])
        assert argv[:3] == [sys.executable, "scripts/served_capability_run.py", "generate"], argv[:3]
        env_line = next((x for x in lines[i + 1:] if x.startswith(("ROSTER_ENV:", "DRIVER:"))), "")
        assert env_line.startswith("ROSTER_ENV:"), f"no ROSTER_ENV: line after {line[:120]}"
        name, _, value = env_line[len("ROSTER_ENV:"):].strip().partition("=")
        out.append((argv, name, "".join(shlex.split(value))))
    return out


def _parse_driver(argv: list, monkeypatch, *, roster: str | None):
    """A DRIVER argv through the driver's OWN parser, under the roster variable given (``None`` = unset):
    ``(namespace, None)``, or ``(None, (exit code, stderr))`` when argparse refuses it."""
    if str(REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts"))
    import served_capability_run as D
    if roster is None:
        monkeypatch.delenv(C.ENV_FULL22_ROSTER, raising=False)
    else:
        monkeypatch.setenv(C.ENV_FULL22_ROSTER, roster)
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            return D.build_parser().parse_args(argv[2:]), None
    except SystemExit as e:
        return None, (e.code, err.getvalue())


def _bank_view_of(argv: list) -> str:
    return argv[argv.index(C.FLAG_BANK_VIEW) + 1]


def test_the_gsm8k_driver_admits_every_row_under_the_jobs_roster_and_refuses_it_without(tmp_path, monkeypatch):
    """Every job of this matrix is submitted with ``F22_ROSTER`` naming the lane roster; the launcher hands the DRIVER the same roster
    and the dry run prints it. Under it all eighteen rows' argv and the 16,384 rung form parse, each stamped with its own view, pinned
    to its own object, at its view's seed base and with the out-of-support acknowledgement; without it the driver refuses every one at
    its parser — the refusal a job would have met after its standup."""
    common = (G8K / "g8k_common.sh").read_text()
    assert re.search(r"^G8K_ROSTER_ENV=(\S+)$", common, re.M).group(1) == C.ENV_FULL22_ROSTER
    plans = _driver_plans(tmp_path, "0-17", rung_rows="7")
    rows = M.load_matrix(M.MATRICES[MATRIX_NAME])["rows"]
    got = []
    for argv, name, value in plans:
        assert name == C.ENV_FULL22_ROSTER and Path(value).is_absolute() and Path(value).resolve() == LANE_ROSTER_PATH.resolve(), (name, value)
        ns, refusal = _parse_driver(argv, monkeypatch, roster=value)
        assert refusal is None, refusal
        assert ns.bank_view in VIEWS and ns.defenses_dir.endswith(f"/{TARGET}/cells_view/{ns.bank_view}"), (ns.bank_view, ns.defenses_dir)
        assert ns.expect_object_sha256 == OBJECTS[(ns.bank_view, ns.cell)] and int(ns.inspect_seed_base) == _k(ns.bank_view)
        assert ns.allow_out_of_support_thinking and C.FLAG_ALLOW_OOS_THINKING in argv
        got.append((ns.bank_view, ns.cell, bool(ns.item_subset)))
        ns, refusal = _parse_driver(argv, monkeypatch, roster=None)
        assert ns is None and refusal[0] == 2, (ns, refusal)
        assert f"argument {C.FLAG_BANK_VIEW}: {C.FLAG_BANK_VIEW} must be one of" in refusal[1], refusal[1][-400:]
        assert f"got {_bank_view_of(argv)!r}" in refusal[1], refusal[1][-400:]
    want = [(r["view"], r["cell"], False) for r in rows] + [(rows[7]["view"], rows[7]["cell"], True)]
    assert sorted(got) == sorted(want) and len(got) == 19


def test_a_submit_line_without_the_lane_roster_is_exposed_by_the_plans_own_lines(tmp_path, monkeypatch):
    """The known-bad twin: left off the submit line, ``F22_ROSTER`` is the production roster, which names none of these views, and the
    driver refuses the row under it — the same two lines that prove a correct launch expose this one."""
    (argv, name, value), = _driver_plans(tmp_path, "0", roster=None)
    assert name == C.ENV_FULL22_ROSTER and Path(value).resolve() == PROD_ROSTER_PATH.resolve()
    ns, refusal = _parse_driver(argv, monkeypatch, roster=value)
    assert ns is None and refusal[0] == 2 and f"got {_bank_view_of(argv)!r}" in refusal[1], (ns, refusal)
