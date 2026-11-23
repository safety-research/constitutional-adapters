"""The 1M-token generator comparison's views ``qg1ms<k>`` (configs/sdfbase_full22_roster_qwengen1m.yaml) carry best-of-N, the
over-refusal and capability controls, GSM8K and the multi-turn attacks beside the static families they first ran, on the instruments
and draw procedure of the control-pool-stopped views (tests/test_ctlstop64_views_all_families.py):

* each view serves exactly its two seed-k LoRAs (the Opus-written and the Qwen-written 1M pools) through configs/qwengen1m_pins.json;
  its roster entry runs every leg but Petri with ONE per-view cap ``caps: {bon: 4096}``, and its wave config draws best-of-N natively
  at 4,096 with the attacker, attacks and grader blocks of the control-pool-stopped view of the same seed;
* the nine sampled configs are the generator's output under the lane roster, each at seed base k, ONE trial and the Qwen vendor
  preset; every re-rendered config equals the static wave commit's copy AS DATA except the best-of-N cap, and the Petri manifests
  are its bytes (the static readings and the new families share one harness): asserted against digests pinned here, so it holds in
  a checkout without that commit, and cross-checked against the commit itself where the clone has it;
* the seeded-lane seed rule is ONE rule: the sampled generator's table, the decode-preset file and the GSM8K views module name the
  three views at their training seeds;
* the job-start registration (``jobplan.build_plans``, pin form on, campaign ``qwengen1m``) with the LANE SEEDS ROOT, narrowed to
  the two cells, binds the six object ids to Best-of-75 7b676184, PAIR, Crescendo, ActorAttack, XSTest, OR-Bench, MMLU 1447cf1b and
  GSM8K and plans no unsteered run. The narrowing is the submit path's, not the configs': every config serves ``[none, the two
  cells]``, so an un-narrowed registration of any family also plans the unsteered ``none`` run (the known-bad twin pins this), and
  only the lane tools' refusal of an empty or ``none``-bearing cell list keeps it out. The seeds root is built here, in a temporary
  checkout-shaped directory, by the one edit the wave applies (one added best-of-N seed minting 7b676184's full id, and two seeds
  flagged non-canonical), never committed: under the repository's own seeds best-of-N binds the canonical 512-store instrument and
  MMLU is refused;
* the ``qwengen1m`` GSM8K matrix is registered in every table and launcher and in the regeneration script, regenerates
  byte-identically from the lane roster + pins its views file sha-pins, and holds 6 launch rows (no ``none`` row, no standup
  companion) under wave ``qwengen1m`` with out roots under ``controls_qwengen1m/``, each at seed base k, with its own overlay pair;
* the GSM8K driver admits a row's ``--bank-view`` only from the lane roster the launcher hands it;
* every row's driver argv, at 8,192 and in the 16,384 rung form, passes the driver's out-of-support authorisation door under the lane's
  NATIVE served identity, matched to the lane's own pair ``gsm8k_paired200_qwen35_4b_native_qwengen1m_<cap>`` (the two 1M-pool cells, by
  name; each row's bytes bound by ``--expect-object-sha256`` to its pin), never to the all-positions Arditi pair. Canary 1484284 died at
  that door after its standup because no tuple named the cells; its refusal quoted the Arditi pair only as the NEAREST tuple.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import shlex
import shutil
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
VIEWS = ("qg1ms1", "qg1ms2", "qg1ms3")
CELLS = ("constv2-mix-1m__lora_r4_ce", "constqwen38-mix-1m__lora_r4_ce")     # Opus-written, Qwen-written
LANE_ROSTER_REL = "configs/sdfbase_full22_roster_qwengen1m.yaml"
LANE_ROSTER_PATH = REPO / LANE_ROSTER_REL
PROD_ROSTER_PATH = REPO / "configs/sdfbase_full22_roster.yaml"
CTLSTOP_ROSTER_PATH = REPO / "configs/sdfbase_full22_roster_ctlstop_eb64.yaml"
LANE_PINS_REL = "configs/qwengen1m_pins.json"
LANE_PINS = json.loads((REPO / LANE_PINS_REL).read_text())
PROD_PINS_REL = "configs/sdfbase_full22_pins.json"
CAMPAIGN = "qwengen1m"
#: the six objects, by (view, cell): the pins file is the authority, these literals the cross-check
OBJECTS = {
    ("qg1ms1", CELLS[0]): "812ca9a2dcd7aabf4d7c08877427ca38a8e5b98ab6b67bb6c54c3f39ae3d5780",
    ("qg1ms2", CELLS[0]): "a0d9971b292a02a8513f8fdb7aa22ef50131c576976d9fa03525708975605472",
    ("qg1ms3", CELLS[0]): "3d93ed6aff39ba1a74440dc7a0f7c597a05f8b32b375fabba6676b347aec7e63",
    ("qg1ms1", CELLS[1]): "e1439afaa9d2829d463f9360b199a8df188802d02c4ed36974de253423b2392f",
    ("qg1ms2", CELLS[1]): "229aa4b74fd9d23869905ac7fa183392493c096589b483b6bcc213921d0a9cad",
    ("qg1ms3", CELLS[1]): "a10be211ce17bf1180e995b6b6642c2de0287c2e12d7a4aaec0eb90d50a96c85",
}
#: the instruments the eight families must bind (the registry's instrument table; the control-pool-stopped rows' instruments)
INSTRUMENT = {
    "strongreject_bon": "7b6761844d0dbcc49159a29c6ba1d0f07f5edea1d59212342830cba5e3bc7dcc",   # Best-of-75, native 4,096 pass
    "pair": "373118ba258c81ebc306e1d7d3ace8acdc53807a96b7277d3931ef961b2311ed",
    "crescendo": "33f80ed1d4f536e34c870fa36c2d88d90c7623ea17a9d4db178fea008c12ac85",
    "actorattack": "f8832b53b08131258d0e5c9e15a16a2888f8a977271d52dde270d52b1af00203",
    "xstest": "28166871df5f55f26ea3ec1fff110ab272e16b7c93f5cf8f32bf034bf408bbe9",
    "or_bench_hard": "e5804e74d4dd459a9911940f95d55a018e0dd53d3ddf3e0b51de8faec0117b87",
    "mmlu": "1447cf1b8538ad07fa41a77e4122a5712d07e3ea645e57e49afe6cb5d12016f2",
    "gsm8k": "5ea36b254426cba37be52b4c9eb07a740e23d48a2dff82336f40685e6322f5c5",
}
#: the canonical best-of-N instrument the repository's OWN seeds bind (the 512-store climb) -- never this lane's
CANONICAL_BON = "78b4699fb41a79a40fc9029f72c9a1bf4b711a8e4fa28aa9671559a24eb38c57"
CONTROL_EVALS = ["xstest", "or_bench_hard"]                     # the over-refusal controls, registered from the sampled config
ATTACK_EVALS = ["pair", "crescendo", "actorattack"]               # registered from the wave config
VENDOR = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}
RUNG_CAPS = {"": 4096, "_cap8192": 8192, "_cap16384": 16384}
MATRIX_NAME = "qwengen1m"
G8K = REPO / "scripts/sdfbase/gsm8k"
ENV = {**os.environ, "PYTHONPATH": str(REPO / "src")}
LAUNCHERS = ("gsm8k_thinking.sbatch", "ladder_plan.sbatch", "dry_run_prod.sbatch", "smoke_not_available.sbatch")
#: the static wave's commit: the 28 static readings ran from its configs
STATIC_WAVE_COMMIT = "e9d8f4262745b47582dd91239c64d98e88f6167e"
#: the static wave commit's best-of-N cap (the 512 base store the canonical instrument climbs from); the views now carry 4,096
STATIC_WAVE_BON_CAP = 512
#: the static wave commit's copy of each config as ``_harness_digest`` reads it (its data, best-of-N cap left out), pinned so the
#: harness identity is asserted in a checkout that lacks the commit (an exact-sha staged checkout, an exported gate tree)
STATIC_WAVE_HARNESS_DIGEST = {
    "configs/sdfbase_full22_qwen35_4b_qg1ms1.yaml": "a499359e3181300066ab3db53741ba06828a51b1d171cec8f1011289a9526962",
    "configs/sdfbase_full22_qwen35_4b_qg1ms1_sampled_t0.yaml": "b81ae1e68d54a67d2f05b9cb3ca945c0ef8fe1ebda33df78e591c8f4aa5c7838",
    "configs/sdfbase_full22_qwen35_4b_qg1ms1_sampled_t0_cap8192.yaml": "3ac05ff9e0578e0fadf2866f057be174b3e97145071a7e95be0ef00d749da64d",
    "configs/sdfbase_full22_qwen35_4b_qg1ms1_sampled_t0_cap16384.yaml": "19fcef600a8b714e79ea3ee1ce30fb80b12e05e69c6c8a0faf88a249d00322d1",
    "configs/sdfbase_full22_qwen35_4b_qg1ms2.yaml": "1e88d438bacebfc8f729f74768550213259127c0176bf8d5702988bac80885e8",
    "configs/sdfbase_full22_qwen35_4b_qg1ms2_sampled_t0.yaml": "83e96e236926f1903f25228c959c53ae7efba483f47bba96b5555c6b8aec4153",
    "configs/sdfbase_full22_qwen35_4b_qg1ms2_sampled_t0_cap8192.yaml": "300d7c84fa2c872c28979981ae21decc164933b4d809d4b1089f4854b9940023",
    "configs/sdfbase_full22_qwen35_4b_qg1ms2_sampled_t0_cap16384.yaml": "75d813ccbe0d872acaf084a66680a3e29fa0e008614129b4148a71a1756b853e",
    "configs/sdfbase_full22_qwen35_4b_qg1ms3.yaml": "1b855ca6910853113d4651373505659f848c61f8fd895d0e0fb747292dc356f5",
    "configs/sdfbase_full22_qwen35_4b_qg1ms3_sampled_t0.yaml": "366e70f1b72a265aa69372f45eb383524002ea51bc6d976d760241843ea5a052",
    "configs/sdfbase_full22_qwen35_4b_qg1ms3_sampled_t0_cap8192.yaml": "67ab0428bd2d6768241245a2bf8a7e8a2d95e6072efde1aecf7b21a2f358afd4",
    "configs/sdfbase_full22_qwen35_4b_qg1ms3_sampled_t0_cap16384.yaml": "9ea0b82f45c4bed30a400afe35ece27c82c92dbe3e8c5116200f98f4af3dbf3f",
}
#: the static wave commit's Petri manifests, sha256 of their bytes
STATIC_WAVE_MANIFEST_BYTES = {
    "configs/capeval/manifests/petri_full22_qwen35_4b_qg1ms1.yaml": "a0fa9d73f5afbf94c5ea36856e2bcfaf44d166df0d29d13a50685a8daf35cad4",
    "configs/capeval/manifests/petri_full22_qwen35_4b_qg1ms2.yaml": "9f607029840ff5aaec2cf8e40a312a7205346776089a13bb5164e1f21464772a",
    "configs/capeval/manifests/petri_full22_qwen35_4b_qg1ms3.yaml": "febe3df66bcd54129ec656a53e2fc6f4dff271c6d66c637ff787adb5ed73a7b2",
}

#: the LANE SEEDS ROOT's one edit (the wave's seeds-root builder applies the same): ONE added best-of-N seed, the seeded native-4,096
#: vendor-preset instrument with exactly the four fields where the registry row of 7b676184 differs, its decode refs narrowed to the
#: Qwen 4B; and paper_status canonical -> sensitivity on the repository's canonical best-of-N seed and on the second canonical MMLU seed
BON_SEED_TEMPLATE = "bon_native4096_vendor_preset"
BON_SEED_NAME = "bon_native4096_vendor_preset_sr2_seedrep"
BON_ROW_FIELDS = {
    "harness_version": "src/sisteer/eval/attacks/bon_served.py + bon_climb.py; scripts/sdfbase/full22/bon_served_run.py, bon_climb_plan.py"
                       " @ seedrep_battery W e526edf",
    "ladder_promotion_rule": "fixed cap 4,096 on every draw (no 512 store, no climb): the VENDOR-preset BoN instrument (D0; not comparable"
                             " with the paper's T=1.0 rows)",
    "context_window_tokens": 11520,
    "judge_id": "strongreject_v2_haiku45",
}
BON_DECODE_REF = "qwen35_4b_it|bon|off|vendor_preset"
FLAGGED_SEEDS = ("bon_512store_4096climb_T1", "mmlu_letter_prefill_200")
MMLU_CANONICAL_SEED = "mmlu_letter_prefill_full_1531"


def _k(view: str) -> int:
    return int(view[-1])


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_the_six_objects_are_the_lane_pins():
    assert set(LANE_PINS[TARGET]) == {f"{v}__{c}" for v in VIEWS for c in CELLS}
    for (view, cell), sha in OBJECTS.items():
        assert LANE_PINS[TARGET][f"{view}__{cell}"]["object_sha256"] == sha, (view, cell)


# --------------------------------------------------------------------------- the roster and the wave configs
def test_the_roster_runs_every_leg_but_petri_with_the_native_best_of_n_cap_on_the_production_contract():
    lane, ctl = yaml.safe_load(LANE_ROSTER_PATH.read_text()), yaml.safe_load(CTLSTOP_ROSTER_PATH.read_text())
    # every top-level value is the control-pool-stopped roster's (itself the production roster's), as data
    assert {k: v for k, v in lane.items() if k != "targets"} == {k: v for k, v in ctl.items() if k != "targets"}
    tgt = lane["targets"][TARGET]
    assert {k: v for k, v in tgt.items() if k != "views"} == {k: v for k, v in ctl["targets"][TARGET].items() if k != "views"}
    assert list(tgt["views"]) == list(VIEWS)
    for view in VIEWS:
        v = tgt["views"][view]
        assert v["legs"] == ["static", "bon", "capability", "controls", "attacks"] and list(v["legs_omitted"]) == ["petri"]
        assert v["caps"] == {"bon": 4096} and not {"dose", "dose_unit", "object_family"} & set(v)
        assert v["config"] == f"configs/sdfbase_full22_{TARGET}_{view}.yaml"
        assert v["manifest"] == f"configs/capeval/manifests/petri_full22_{TARGET}_{view}.yaml"
        assert v["cells"] == {c: {"pins": LANE_PINS_REL, "pin": f"{TARGET}.{view}__{c}"} for c in CELLS}   # pin form, explicit cells only


@pytest.mark.parametrize("view", VIEWS)
def test_each_wave_config_draws_best_of_n_at_4096_and_shares_the_control_pool_stopped_views_attack_setup(view):
    cfg = yaml.safe_load((REPO / f"configs/sdfbase_full22_{TARGET}_{view}.yaml").read_text())
    assert cfg["bon"]["max_new_tokens"] == 4096 and cfg["explicit_cells"] == ["none", *CELLS]
    assert cfg["attacks"]["static_max_new_tokens"] == 4096 and cfg["attacks"]["target_max_new_tokens"] == 4096
    ctl = yaml.safe_load((REPO / f"configs/sdfbase_full22_{TARGET}_ctlstop64s{_k(view)}.yaml").read_text())
    for block in ("attacker", "attacks", "grader", "bon"):
        assert cfg[block] == ctl[block], (view, block)
    # the two views differ only in what names the view and its cells
    assert sorted(x for x in set(cfg) | set(ctl) if cfg.get(x) != ctl.get(x)) == ["defenses_dir", "explicit_cells", "out_dir", "private_run"]
    assert cfg["private_run"] == f"sdfbase_full22_{TARGET}_{view}" and cfg["defenses_dir"] == f"out/full22/{TARGET}/cells_view/{view}"


@pytest.mark.parametrize("view", VIEWS)
def test_roster_view_lists_the_capability_and_over_refusal_controls(view):
    r = subprocess.run([sys.executable, str(REPO / "scripts/sdfbase/full22/roster_view.py"), TARGET, view, "controls",
                        "--roster", str(LANE_ROSTER_PATH)], capture_output=True, text=True, env=ENV, cwd=REPO, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().split(",") == ["mmlu", "xstest", "or_bench_hard_1k"]          # gsm8k is retired from the controls leg


# --------------------------------------------------------------------------- the sampled configs (static + over-refusal)
def test_the_nine_sampled_configs_are_the_generators_at_the_training_seed_one_trial_and_the_vendor_preset(monkeypatch):
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
        sd = yaml.safe_load(path.read_text())["sampled_decode"]
        sfx = path.stem.split("_sampled_t0", 1)[1]
        assert (sd["arm"], sd["trial"], sd["n_trials"], sd["seed_base"]) == ("vendor_recommended", 0, 1, _k(view)), path.name
        assert {x: sd[x] for x in VENDOR} == VENDOR and sd["source_kind"] == "model_card", path.name
        assert sd["static_cap"] == RUNG_CAPS[sfx] and sd["controls_cap"] == 512, path.name
    assert msc.stale_outputs(presets) == []


def _flat(d, p=""):
    if isinstance(d, dict):
        return {k2: v2 for k, v in d.items() for k2, v2 in _flat(v, f"{p}.{k}" if p else str(k)).items()}
    if isinstance(d, list):
        return {k2: v2 for i, v in enumerate(d) for k2, v2 in _flat(v, f"{p}[{i}]").items()}
    return {p: d}


def _harness_digest(data) -> tuple:
    """``(best-of-N cap, sha256)`` of a config's data: the cap, and the digest of every OTHER value as sorted flattened JSON (key path
    -> value). Two configs with one digest share a harness whatever their comments and key order."""
    flat = _flat(data)
    cap = flat.pop("bon.max_new_tokens")
    return cap, hashlib.sha256(json.dumps(flat, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _view_config_rels(view: str) -> list:
    return [f"configs/sdfbase_full22_{TARGET}_{view}{sfx}.yaml" for sfx in ("", *(f"_sampled_t0{s}" for s in RUNG_CAPS))]


def _at_static_wave_commit(rel: str) -> bytes | None:
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{STATIC_WAVE_COMMIT}:{rel}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def test_the_pinned_static_wave_digests_name_every_lane_config_and_manifest():
    assert set(STATIC_WAVE_HARNESS_DIGEST) == {rel for v in VIEWS for rel in _view_config_rels(v)}
    assert set(STATIC_WAVE_MANIFEST_BYTES) == {f"configs/capeval/manifests/petri_full22_{TARGET}_{v}.yaml" for v in VIEWS}


@pytest.mark.parametrize("view", VIEWS)
def test_the_new_families_share_the_static_readings_harness(view):
    """The 28 static readings ran from the static wave commit's configs. Re-rendered with every leg, each config of the view equals that
    commit's AS DATA except the best-of-N cap (512 -> the view's native 4,096), and the Petri manifest is its bytes. Asserted against
    the digests pinned above, so no git history is needed; a red here names the file, and ``git show <commit>:<file>`` against it
    shows the difference."""
    man = f"configs/capeval/manifests/petri_full22_{TARGET}_{view}.yaml"
    assert _sha256(REPO / man) == STATIC_WAVE_MANIFEST_BYTES[man], man
    for rel in _view_config_rels(view):
        assert _harness_digest(yaml.safe_load((REPO / rel).read_text())) == (4096, STATIC_WAVE_HARNESS_DIGEST[rel]), rel


@pytest.mark.parametrize("view", VIEWS)
def test_the_pinned_digests_are_the_static_wave_commits_where_the_commit_is_present(view):
    """The cross-check of the pins above against the commit itself, with the full difference on a red. Skips by name where the commit
    is not in the clone (a shallow exact-sha checkout); the pinned assertion above runs everywhere."""
    if _at_static_wave_commit(LANE_PINS_REL) is None:
        pytest.skip(f"the static wave commit {STATIC_WAVE_COMMIT[:9]} is not in this clone")
    man = f"configs/capeval/manifests/petri_full22_{TARGET}_{view}.yaml"
    assert hashlib.sha256(_at_static_wave_commit(man)).hexdigest() == STATIC_WAVE_MANIFEST_BYTES[man], man
    for rel in _view_config_rels(view):
        old = yaml.safe_load(_at_static_wave_commit(rel))
        assert _harness_digest(old) == (STATIC_WAVE_BON_CAP, STATIC_WAVE_HARNESS_DIGEST[rel]), rel
        old, new = _flat(old), _flat(yaml.safe_load((REPO / rel).read_text()))
        assert ({k: (old.get(k), new.get(k)) for k in set(old) | set(new) if old.get(k) != new.get(k)}
                == {"bon.max_new_tokens": (STATIC_WAVE_BON_CAP, 4096)}), rel


def test_the_seeded_lane_rule_names_the_three_views_at_their_training_seed_in_every_table():
    presets = msc.load_presets()
    assert msc.QG1M_VIEWS == VIEWS and msc.SEEDED_LANE_VIEWS[TARGET][-len(VIEWS):] == VIEWS
    for view in VIEWS:
        k, tag = _k(view), msc.tag_of(TARGET, view)
        assert msc.seed_of_lane_view(view) == k == ES.seed_base(k)
        assert msc.SEED_BASE_RULE_BY_TAG[tag] == presets["seed_base_by_view"][tag] == k != presets["seed_base"]
        assert presets["n_trials_by_view"][tag] == 1 != presets["n_trials"]
        # the GSM8K views module carries the SAME rule for the same views (one rule, two stdlib-free readers)
        assert V.SEEDED_LANE_VIEW_SEEDS[view] == k == V.replicate_seed_of_view(view)
    assert {v for v in V.SEEDED_LANE_VIEW_SEEDS if v.startswith("qg1m")} == set(VIEWS)
    with pytest.raises(ValueError):
        V.replicate_seed_of_view("qg1ms4")


# --------------------------------------------------------------------------- the job-start registration
def _block_span(lines: list, seed_id: str) -> tuple:
    pat = re.compile(rf"^  - instrument_id: {re.escape(seed_id)}\s*$")
    starts = [i for i, ln in enumerate(lines) if pat.match(ln)]
    assert len(starts) == 1, (seed_id, starts)
    e = starts[0] + 1
    while e < len(lines) and not (lines[e].startswith("  - instrument_id:") or (lines[e] and not lines[e].startswith(" "))
                                  or lines[e].startswith("  #") or lines[e].strip() == ""):
        e += 1
    return starts[0], e


def _lane_seed_block(template: list) -> list:
    out, seen = [], set()
    for ln in template:
        m = re.match(r"^    ([a-z_0-9]+): ", ln)
        key = m.group(1) if m else None
        seen.add(key)
        if ln.startswith("  - instrument_id: "):
            out.append(f"  - instrument_id: {BON_SEED_NAME}")
        elif key in ("harness_version", "ladder_promotion_rule"):
            out.append(f"    {key}: {json.dumps(BON_ROW_FIELDS[key])}")
        elif key == "read_at":
            out += [ln, f"    context_window_tokens: {BON_ROW_FIELDS['context_window_tokens']}"]
        elif key == "decode_policy_refs":
            out.append(f"    decode_policy_refs: [{json.dumps(BON_DECODE_REF)}]")
        elif key == "judge_id":
            out.append(f"    judge_id: {BON_ROW_FIELDS['judge_id']}")
        elif key == "paper_status":
            out.append("    paper_status: canonical")
        else:
            out.append(ln)
    assert {"harness_version", "ladder_promotion_rule", "read_at", "decode_policy_refs", "judge_id", "paper_status"} <= seen
    assert "context_window_tokens" not in seen
    return out


def _canonical(seeds: Path, eval_id: str) -> list:
    """(seed id, instrument id) of every seeded instrument jobplan weighs as canonical for (eval, qwen35_4b_it, lora_native)."""
    idx, out = JP.load_seed_index(str(seeds)), []
    for payload, extra in idx.instruments:
        if str(payload.get("eval_id")) != eval_id or JP._paper_status(payload, extra) != "canonical":
            continue
        models = JP._instrument_models(extra)
        if models and "qwen35_4b_it" not in models:
            continue
        sp = payload.get("serving_path")
        if isinstance(sp, str) and sp not in (JP.UNVERIFIED, "") and sp != "lora_native":
            continue
        out.append((str(extra.get("seed_instrument_id")), str(payload.get("instrument_id"))))
    return sorted(set(out))


@pytest.fixture(scope="module")
def lane_seeds_root(tmp_path_factory) -> Path:
    """A checkout-shaped root: configs/registry (all of it, ONE file edited), the lane pins and the lane roster. With pin form on,
    jobplan resolves a pin-form cell's `pins` path relative to the seeds directory's grandparent, so the pins sit beside the seeds."""
    root = tmp_path_factory.mktemp("qg1m_seeds_root")
    shutil.copytree(REPO / "configs/registry", root / "configs/registry")
    for rel in (LANE_PINS_REL, LANE_ROSTER_REL):
        shutil.copy2(REPO / rel, root / rel)
    inst = root / "configs/registry/instruments.yaml"
    lines = inst.read_text().split("\n")
    s, e = _block_span(lines, BON_SEED_TEMPLATE)
    lines = lines[:e] + ["", "  # the lane seeds root's best-of-N seed (this temporary root only)"] + _lane_seed_block(lines[s:e]) + lines[e:]
    for seed in FLAGGED_SEEDS:
        s2, e2 = _block_span(lines, seed)
        hits = [i for i in range(s2, e2) if lines[i] == "    paper_status: canonical"]
        assert len(hits) == 1, (seed, hits)
        lines[hits[0]] = "    paper_status: sensitivity"
    inst.write_text("\n".join(lines))
    return root


def test_the_lane_seeds_root_adds_one_seed_minting_the_native_best_of_n_instrument_and_leaves_one_canonical_mmlu(lane_seeds_root):
    repo_seeds, lane_seeds = REPO / "configs/registry", lane_seeds_root / "configs/registry"
    # the repository's own seeds: the 512-store climb is the canonical best-of-N, and MMLU has two canonical seeds (native refusal)
    assert _canonical(repo_seeds, "strongreject_bon") == [("bon_512store_4096climb_T1", CANONICAL_BON)]
    assert sorted(s for s, _ in _canonical(repo_seeds, "mmlu")) == sorted(("mmlu_letter_prefill_200", MMLU_CANONICAL_SEED))
    # the lane root, under the REAL loader: exactly one record added, and it mints 7b676184's full id
    before = {(str(x.get("seed_instrument_id")), str(p.get("instrument_id"))) for p, x in JP.load_seed_index(str(repo_seeds)).instruments}
    after = {(str(x.get("seed_instrument_id")), str(p.get("instrument_id"))) for p, x in JP.load_seed_index(str(lane_seeds)).instruments}
    assert after - before == {(BON_SEED_NAME, INSTRUMENT["strongreject_bon"])} and before <= after
    assert _canonical(lane_seeds, "strongreject_bon") == [(BON_SEED_NAME, INSTRUMENT["strongreject_bon"])]
    assert _canonical(lane_seeds, "mmlu") == [(MMLU_CANONICAL_SEED, INSTRUMENT["mmlu"])]
    # never committed: the repository's seeds name no seed minting it
    assert INSTRUMENT["strongreject_bon"] not in {i for _, i in before}


def _plans(cfg_rel: str, evals: list, view: str, *, seeds_root: Path | None, pairs=None, objects=None, producer_key=None):
    root = seeds_root if seeds_root is not None else REPO
    return JP.build_plans(config_path=str(REPO / cfg_rel), model_key=f"{TARGET}_{view}", roster_selection=view, evals=evals,
                          serving_path="lora_native", campaign=CAMPAIGN, job_id="4242", job_name="synthetic-qwengen1m",
                          seeds_dir=str(root / "configs/registry"), roster_path=str(root / LANE_ROSTER_REL),
                          pins_path=str(REPO / PROD_PINS_REL), code_sha=synth.code_sha("a"), substrate="cluster_a",
                          producer_key=producer_key, pairs=pairs, objects=objects, pin_form=True)


def _pairs(evals: list) -> list:
    """The (eval, object) pairs a lane submit path hands the registration: the two cells, never `none` (the configs serve `none`
    too, so this narrowing, enforced by the lane tools' cell-list gate, is what keeps the unsteered run out)."""
    return [",".join(f"{ev}:{c}" for ev in evals for c in CELLS)]


def _bound(bundle, view: str) -> dict:
    """``{(eval, object cell): (instrument id, object id)}`` of a NARROWED plan bundle that skipped nothing, every run on the lane's
    campaign, the native path, the served instruct model and the view's own legacy spelling, and no unsteered run."""
    assert bundle.skipped == [], bundle.skipped
    out = {}
    for r in bundle.runs:
        p = r.plan
        assert (p["campaign"], p["serving_path"], p["model_id"], p["view_legacy"]) == (CAMPAIGN, "lora_native", "qwen35_4b_it", f"{TARGET}_{view}")
        assert r.route["object"] != "none" and p["object_id"] != "none", p
        out[(p["eval_id"], r.route["object"])] = (p["instrument_id"], p["object_id"])
    return out


@pytest.mark.parametrize("view", VIEWS)
def test_the_job_start_registration_binds_the_six_objects_to_the_eight_instruments_with_the_lane_seeds_root(view, lane_seeds_root, monkeypatch):
    monkeypatch.delenv("SISTEER_REGISTRY_PIN_FORM", raising=False)
    objs = {c: OBJECTS[(view, c)] for c in CELLS}
    wave_cfg = f"configs/sdfbase_full22_{TARGET}_{view}.yaml"
    got = {}
    # over-refusal: from the sampled config (the f22s-* job kind), its explicit pairs
    got.update(_bound(_plans(f"configs/sdfbase_full22_{TARGET}_{view}_sampled_t0.yaml", CONTROL_EVALS, view, seeds_root=lane_seeds_root,
                             pairs=_pairs(CONTROL_EVALS), producer_key=f"sdfbase_full22_{TARGET}_{view}_sampled_t0"), view))
    # best-of-N and MMLU: from the wave config, through the lane seeds root
    for ev in ("strongreject_bon", "mmlu"):
        got.update(_bound(_plans(wave_cfg, [ev], view, seeds_root=lane_seeds_root, pairs=_pairs([ev])), view))
    # the attacks: from the wave config; a defender serves the cells positional, which narrows the registration to those cells
    for ev in ATTACK_EVALS:
        got.update(_bound(_plans(wave_cfg, [ev], view, seeds_root=lane_seeds_root, objects=list(CELLS)), view))
    # GSM8K: one row per job, registered from the row's own overlay narrowed to its cell
    for c in CELLS:
        got.update(_bound(_plans(f"configs/sdfbase/gsm8k/sdfbase_full22_{TARGET}_{view}.gsm8k_{M.cell_slug(c)}.yaml", ["gsm8k"], view,
                                 seeds_root=lane_seeds_root, objects=[c]), view))
    assert got == {(ev, c): (INSTRUMENT[ev], sha) for ev in INSTRUMENT for c, sha in objs.items()}


def test_an_un_narrowed_registration_of_every_family_plans_the_unsteered_run(lane_seeds_root, monkeypatch):
    """The known-bad twin of the narrowing. Every config serves ``[none, the cells]``, so a submit path that handed the registration no
    cell list would plan, under campaign ``qwengen1m`` and on the paper instruments, an unsteered run no lane ruling covers. Nothing in
    the configs prevents it; the lane tools' refusal of an empty or ``none``-bearing cell list does, and this pins that dependency."""
    monkeypatch.delenv("SISTEER_REGISTRY_PIN_FORM", raising=False)
    view = "qg1ms1"
    wave_cfg = f"configs/sdfbase_full22_{TARGET}_{view}.yaml"
    jobs = [(f"configs/sdfbase_full22_{TARGET}_{view}_sampled_t0.yaml", CONTROL_EVALS, CELLS, f"sdfbase_full22_{TARGET}_{view}_sampled_t0"),
            (wave_cfg, ["strongreject_bon"], CELLS, None), (wave_cfg, ["mmlu"], CELLS, None),
            *((wave_cfg, [ev], CELLS, None) for ev in ATTACK_EVALS),
            *((f"configs/sdfbase/gsm8k/sdfbase_full22_{TARGET}_{view}.gsm8k_{M.cell_slug(c)}.yaml", ["gsm8k"], (c,), None) for c in CELLS)]
    for cfg, evals, cells, producer_key in jobs:
        b = _plans(cfg, evals, view, seeds_root=lane_seeds_root, producer_key=producer_key)
        assert b.skipped == [], (cfg, evals, b.skipped)
        got = {(r.plan["eval_id"], r.route["object"]): (r.plan["instrument_id"], r.plan["object_id"]) for r in b.runs}
        want = {(ev, "none"): (INSTRUMENT[ev], "none") for ev in evals}
        want.update({(ev, c): (INSTRUMENT[ev], OBJECTS[(view, c)]) for ev in evals for c in cells})
        assert got == want, (cfg, evals)


def test_under_the_repositorys_own_seeds_best_of_n_binds_the_climb_instrument_and_mmlu_is_refused(monkeypatch):
    """The known-bad twin: registering without the lane seeds root would put the Best-of-75 runs on the 512-store climb instrument and
    register no MMLU run at all (two canonical seeds)."""
    monkeypatch.delenv("SISTEER_REGISTRY_PIN_FORM", raising=False)
    cfg = f"configs/sdfbase_full22_{TARGET}_qg1ms1.yaml"
    bon = _plans(cfg, ["strongreject_bon"], "qg1ms1", seeds_root=None, pairs=_pairs(["strongreject_bon"]))
    assert bon.skipped == [] and {r.plan["instrument_id"] for r in bon.runs} == {CANONICAL_BON}
    mmlu = _plans(cfg, ["mmlu"], "qg1ms1", seeds_root=None, pairs=_pairs(["mmlu"]))
    assert mmlu.runs == [] and len(mmlu.skipped) == 1 and "2 canonical seeded instruments for eval 'mmlu'" in mmlu.skipped[0][1]


# --------------------------------------------------------------------------- the GSM8K matrix
def test_the_qwengen1m_matrix_is_registered_in_every_table_launcher_and_the_regeneration_script():
    assert M.MATRIX_QWENGEN1M_NAME == MATRIX_NAME
    assert M.MATRICES[MATRIX_NAME] == REPO / "configs/sdfbase/gsm8k/matrix_qwengen1m.json"
    assert M.VIEWS_BY_MATRIX[MATRIX_NAME] == REPO / "configs/sdfbase/gsm8k/views_qwengen1m.yaml"
    assert M.WAVE_QWENGEN1M == MATRIX_NAME and M.OUT_ROOT_PREFIX_BY_WAVE[M.WAVE_QWENGEN1M] == "controls_qwengen1m"
    assert M.out_root_prefix_for(M.WAVE_QWENGEN1M) == "controls_qwengen1m" and M.WAVE_QWENGEN1M not in M.COMPLETED_WAVES
    for name in LAUNCHERS:
        m = re.search(r'case "\$G8K_MATRIX" in ([a-z0-9_|]+)\) ;;', (G8K / name).read_text())
        assert m and MATRIX_NAME in m.group(1).split("|"), name
    regen = (REPO / "scripts/release/regen_configs.sh").read_text()
    for line in (f'"scripts/sdfbase/make_full22_configs.py --roster {LANE_ROSTER_REL}"',
                 f'"scripts/sdfbase/full22/sampled/make_sampled_configs.py --roster {LANE_ROSTER_REL}"',
                 '"scripts/sdfbase/gsm8k/make_gsm8k_matrix.py --matrix-name qwengen1m"',
                 '"scripts/sdfbase/gsm8k/make_gsm8k_overlays.py --matrix qwengen1m"'):
        assert regen.count(line) == 1, line
    # dependency order: the wave configs before the sampled configs, the matrix before its overlays
    assert regen.index(f"make_full22_configs.py --roster {LANE_ROSTER_REL}") < regen.index(f"make_sampled_configs.py --roster {LANE_ROSTER_REL}")
    assert regen.index("make_gsm8k_matrix.py --matrix-name qwengen1m") < regen.index("make_gsm8k_overlays.py --matrix qwengen1m")


def test_the_qwengen1m_views_file_pins_the_lane_roster_and_pins_and_the_matrix_regenerates_byte_identically():
    spec = V.load_views_yaml(M.VIEWS_BY_MATRIX[MATRIX_NAME])
    assert spec.schema == M.MATRIX_SCHEMA_2 and spec.wave == M.WAVE_QWENGEN1M and spec.row_overlays
    assert spec.views == {TARGET: list(VIEWS)} and spec.holds == []
    assert [(e["view"], e["cells"]) for e in spec.exclude] == [(v, ["none"]) for v in VIEWS]
    assert spec.inspect_seed_policy == M.INSPECT_SEED_POLICY_REPLICATE and spec.inspect_seed_base_by_view == {v: _k(v) for v in VIEWS}
    assert not spec.standup_companions and spec.geometry_class == M.GEOMETRY_CLASS_COMMON_F22
    assert spec.trigger_threshold == M.TRIGGER_THRESHOLD and spec.wave_root_rel == "exp/gsm8k_paired200_paper_objects_qwengen1m"
    frozen = V.frozen_input_paths(spec, REPO)
    assert frozen == (LANE_ROSTER_PATH, REPO / LANE_PINS_REL)
    text, doc = MK.build(configs_root=REPO, views_yaml=None, views_spec=None, matrix_name=MATRIX_NAME)
    assert M.MATRICES[MATRIX_NAME].read_text() == text
    # the document names the lane-private files it was derived from, never the production roster
    assert doc["source"] == {"views_yaml": "configs/sdfbase/gsm8k/views_qwengen1m.yaml",
                             "roster": LANE_ROSTER_REL, "roster_sha256": _sha256(LANE_ROSTER_PATH),
                             "pins": LANE_PINS_REL, "pins_sha256": _sha256(REPO / LANE_PINS_REL)}


def test_the_qwengen1m_matrix_holds_six_launch_rows_at_the_training_seed_with_no_companion_and_no_unsteered_row():
    doc = M.load_matrix(M.MATRICES[MATRIX_NAME])
    rows = doc["rows"]
    assert len(rows) == 6 and [r["index"] for r in rows] == list(range(6))
    assert [(r["view"], r["cell"]) for r in rows] == [(v, c) for v in VIEWS for c in CELLS]          # no `none` row
    for r in rows:
        view, cell, k = r["view"], r["cell"], _k(r["view"])
        assert r["model_key"] == TARGET and r["tag"] == f"{TARGET}_{view}" and r["status"] == M.STATUS_LAUNCH and r["fitted"]
        assert r["staged_id"] == f"{view}__{cell}" and r["object_sha256"] == OBJECTS[(view, cell)]
        assert r["wave"] == M.WAVE_QWENGEN1M == M.row_wave(r) == "qwengen1m" and r["own_overlay"]
        assert r["out_root"].split("/", 1)[0] == "controls_qwengen1m" == M.out_root_prefix_for(M.row_wave(r))
        assert r["out_root"].startswith(f"controls_qwengen1m/{TARGET}_{view}/")
        assert (r["inspect_seed_policy"], r["inspect_seed_base"]) == (M.INSPECT_SEED_POLICY_REPLICATE, k) and M.row_inspect_seed(r) == ("replicate", k)
        assert r["thinking_mode"] == M.THINKING_ON and r["sampling_mode"] == M.SAMPLING_THINKING_CARD
        assert M.row_geometry_class(r) == M.GEOMETRY_CLASS_COMMON_F22
        assert M.geometry_for_cell(cell, M.CAP, model_key=TARGET, geometry_class=M.row_geometry_class(r)) == ("capability:gen_long_gsm8k_common_f22", 16640)
        # the launcher's SWITCH for --allow-out-of-support-thinking (non-empty = the flag, g8k_common.sh g8k_driver_argv), as on the ctlstop
        # and seed-view rows; the tuple that AUTHORISES the run is the registry's match at the driver's door (the OOS tests below)
        assert r["tuple_name"] == "gsm8k_paired200_qwen35_4b_8192"
        assert M.row_companions(r) == []                         # both cells are rank-4 LoRAs: no companion
        assert M.launch_cells_for(r)[0] == "none"                # the standup serves the base route, unmeasured
    assert [r["index"] for r in M.eligible_rows(M.WAVE_QWENGEN1M, rows)] == list(range(6))


def test_every_qwengen1m_row_serves_through_its_own_overlay_pair_cut_from_the_campaign_config():
    rows = M.load_matrix(M.MATRICES[MATRIX_NAME])["rows"]
    outs = list(OV.outputs(REPO, rows))
    assert len(outs) == 12
    for path, text in outs:
        assert path.is_file() and path.read_text() == text, path.name                                   # byte-identical (--check)
    for r in rows:
        tag, slug = r["tag"], M.cell_slug(r["cell"])
        cfg = yaml.safe_load((REPO / f"configs/sdfbase/gsm8k/sdfbase_full22_{tag}.gsm8k_{slug}.yaml").read_text())
        man = yaml.safe_load((REPO / f"configs/capeval/manifests/petri_full22_{tag}.gsm8k_{slug}.yaml").read_text())["petri"]
        assert cfg["explicit_cells"] == ["none", r["cell"]] and man["cells"] == ["none", r["harness_cell"]]
        campaign = yaml.safe_load((REPO / f"configs/sdfbase_full22_{tag}.yaml").read_text())
        assert {k: v for k, v in cfg.items() if k != "explicit_cells"} == {k: v for k, v in campaign.items() if k != "explicit_cells"}


def test_the_views_loader_holds_each_lane_views_base_to_its_training_seed(tmp_path):
    text = M.VIEWS_BY_MATRIX[MATRIX_NAME].read_text()
    bad = tmp_path / "views_qwengen1m.yaml"
    bad.write_text(text.replace("  qg1ms2: 2\n", "  qg1ms2: 1\n", 1))
    with pytest.raises(SystemExit, match=r"inspect_seed_base_by_view.qg1ms2 = 1 but eval_seed.seed_base\(2\) = 2"):
        V.load_views_yaml(bad)


# --------------------------------------------------------------------------- the GSM8K driver admits the lane views from the job's roster
def test_the_bank_view_vocabulary_admits_the_lane_views_only_from_the_lane_roster(monkeypatch):
    monkeypatch.delenv(C.ENV_FULL22_ROSTER, raising=False)
    prod = C.bank_views_admitted(PROD_ROSTER_PATH)
    assert not set(VIEWS) & set(prod)
    for v in VIEWS:
        assert f"{C.FLAG_BANK_VIEW} {v!r} is not one of" in C.bank_view_problem(v, PROD_ROSTER_PATH)
    monkeypatch.setenv(C.ENV_FULL22_ROSTER, str(LANE_ROSTER_PATH))          # the variable the launcher exports, read with no argument
    assert all(C.bank_view_problem(v) is None and C.check_bank_view_arg(v) == v for v in VIEWS)
    assert C.bank_view_problem("ctlstop64s1") is not None                   # ONE roster at a time: another lane's view is not this job's


def _driver_plans(tmp_path: Path, rows: str, *, rung_rows: str | None = None) -> list:
    """The dry run's DRIVER: lines for the given rows (the launcher's ONE argv function), each with the ROSTER_ENV: line after it,
    submitted with ``F22_ROSTER`` naming the lane roster as every job of this matrix is: ``[(argv, variable name, value)]``."""
    env = {k: v for k, v in os.environ.items() if k not in (C.ENV_FULL22_ROSTER, "F22_ROSTER")}
    env.update(PYTHONPATH=str(REPO / "src"), F22_ROSTER=LANE_ROSTER_REL)
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


def test_the_gsm8k_driver_admits_every_row_under_the_lane_roster_and_refuses_it_without(tmp_path, monkeypatch):
    if str(REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts"))
    import served_capability_run as D
    plans = _driver_plans(tmp_path, "0-5", rung_rows="1")
    rows = M.load_matrix(M.MATRICES[MATRIX_NAME])["rows"]
    got = []
    for argv, name, value in plans:
        assert name == C.ENV_FULL22_ROSTER and Path(value).is_absolute() and Path(value).resolve() == LANE_ROSTER_PATH.resolve(), (name, value)
        monkeypatch.setenv(C.ENV_FULL22_ROSTER, value)
        ns = D.build_parser().parse_args(argv[2:])
        assert ns.bank_view in VIEWS and ns.defenses_dir.endswith(f"/{TARGET}/cells_view/{ns.bank_view}"), (ns.bank_view, ns.defenses_dir)
        assert ns.expect_object_sha256 == OBJECTS[(ns.bank_view, ns.cell)] and int(ns.inspect_seed_base) == _k(ns.bank_view)
        got.append((ns.bank_view, ns.cell, bool(ns.item_subset)))
        monkeypatch.delenv(C.ENV_FULL22_ROSTER, raising=False)
        err = io.StringIO()
        with contextlib.redirect_stderr(err), pytest.raises(SystemExit) as exc:
            D.build_parser().parse_args(argv[2:])
        assert exc.value.code == 2 and f"argument {C.FLAG_BANK_VIEW}" in err.getvalue(), err.getvalue()[-400:]
    want = [(r["view"], r["cell"], False) for r in rows] + [(rows[1]["view"], rows[1]["cell"], True)]
    assert sorted(got) == sorted(want) and len(got) == 7


# --------------------------------------------------------------------------- the GSM8K out-of-support authorisation
#: the lane's own native pair: the tuple each row's driver door must match, per cap
LANE_OOS_TUPLES = {cap: C.g8k_native_qwengen1m_tuple_name(TARGET, cap) for cap in C.G8K_CAPS}
#: GSM8K canary 1484284's refusal (tree 6090779e6), as the driver printed it after the standup
CANARY_REFUSAL = ("cell differs — requested 'constv2-mix-1m__lora_r4_ce', authorised ['none', 'arditi_allpos']")


def _driver():
    if str(REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts"))
    import served_capability_run as D
    return D


def _oos_door(D, argv: list, *, reduction_mode: str = C.REDUCTION_MODE_NOT_APPLICABLE):
    """The driver's FIRST out-of-support door over one dry-run argv, fed what the driver feeds it (served_capability_run.py, the
    `if args.allow_out_of_support_thinking:` block): its own parser, the config the argv names, the parsed controls, the thinking arm,
    the sampling row it resolves (model key from the config's model), the cap STATEMENT and the SERVED reduction mode — the native
    engine's (`contracts.served_reduction_mode` over a lora_native record) unless stated. -> (namespace, the matched tuple's name)."""
    ns = D.build_parser().parse_args(argv[2:])
    cfg = D.jailbreak_run._load_cfg(ns.config)
    controls = D.parse_controls(ns.controls)
    atoms = D.tinycap_atoms(controls)
    sampling = D._resolve_sampling(ns, cfg, atoms, D._resolve_thinking_by_task(atoms, ns.thinking))
    stmt = C.parse_max_tokens_by_task(ns.max_tokens_by_task or [])
    C.check_authorised_render_facts()
    name = C.check_out_of_support_thinking_authorised(
        model=sampling.model_key, cell=ns.cell, controls=controls, mode=ns.thinking, sampling_mode=sampling.mode,
        samples=sampling.samples, reduction_mode=reduction_mode, max_tokens_by_task=(dict(stmt) if stmt else None), where="driver run")
    return ns, name


def test_every_qwengen1m_row_is_authorised_at_the_drivers_door_by_the_lanes_own_native_pair_at_both_caps(tmp_path, monkeypatch):
    """Canary 1484284 (row 0, tree 6090779e6) stood its server up and died at this door: `OOS_UNAUTHORISED … cell differs`. Every row,
    at 8,192 and in the 16,384 rung form, must match a tuple whose cells include the row's own cell — the lane's pair, at the rung's
    cap — with the row's pin bound by --expect-object-sha256; under the lens-era identity (`stacked`) the lane cells match nothing."""
    D = _driver()
    native = C.served_reduction_mode({C.ENDPOINT_KEY_ENGINE: C.SERVING_ENGINE_LORA_NATIVE}, source="the lane's native standup")
    assert native == C.REDUCTION_MODE_NOT_APPLICABLE
    rows = M.load_matrix(M.MATRICES[MATRIX_NAME])["rows"]
    plans = _driver_plans(tmp_path, "0-5", rung_rows="0-5")
    monkeypatch.chdir(REPO)                                      # the argv's --config is checkout-relative, as in the job
    got = []
    for argv, _name, roster in plans:
        monkeypatch.setenv(C.ENV_FULL22_ROSTER, roster)
        ns, name = _oos_door(D, argv, reduction_mode=native)
        cap = int(dict(C.parse_max_tokens_by_task(ns.max_tokens_by_task))[C.GSM8K_PAIRED_TASK_NAME])
        exp = C.AUTHORISED_OOS_EXPERIMENTS[name]
        assert ns.allow_out_of_support_thinking and name == LANE_OOS_TUPLES[cap], (ns.bank_view, ns.cell, cap, name)
        assert ns.cell in exp.cell_spellings() and exp.cells == C.G8K_NATIVE_QWENGEN1M_CELLS == CELLS
        assert exp.max_tokens_on == {C.GSM8K_PAIRED_TASK_NAME: cap} and exp.reduction_mode == native and exp.model == TARGET
        assert C.ALLPOS_TUPLE_SUFFIX not in name and not set(exp.cells) & {C.ALLPOS_CELL}
        # the bytes: the row's pin, the lane pins file's entry, the literal
        pin = LANE_PINS[TARGET][f"{ns.bank_view}__{ns.cell}"]["object_sha256"]
        assert ns.expect_object_sha256 == pin == OBJECTS[(ns.bank_view, ns.cell)]
        with pytest.raises(C.OutOfSupportThinkingUnauthorised, match="cell differs"):
            _oos_door(D, argv, reduction_mode=C.REDUCTION_MODE_STACKED)
        got.append((ns.bank_view, ns.cell, cap))
    assert sorted(got) == sorted((r["view"], r["cell"], cap) for r in rows for cap in C.G8K_CAPS) and len(got) == 12


def test_no_arditi_tuple_authorises_a_lane_cell_and_without_the_lane_pair_the_door_refuses_as_the_canary_did(tmp_path, monkeypatch):
    """The canary's refusal NAMED the all-positions Arditi pair, which never authorised a lane row: it is the NEAREST tuple (every field
    but the cell agrees) and is registered first. The lane pair is the only tuple naming a lane cell (no existing tuple was widened),
    no Arditi tuple names one, and with the lane pair removed the door refuses row 0 exactly as canary 1484284 was refused."""
    D = _driver()
    naming = {n for n, exp in C.AUTHORISED_OOS_EXPERIMENTS.items() if set(exp.cells) & set(CELLS)}
    assert naming == set(LANE_OOS_TUPLES.values()), naming
    allpos = [n for n in C.AUTHORISED_OOS_EXPERIMENTS if C.ALLPOS_TUPLE_SUFFIX in n]
    assert set(allpos) >= {C.g8k_allpos_tuple_name(TARGET, cap) for cap in C.G8K_CAPS}
    for n in allpos:
        assert not set(C.AUTHORISED_OOS_EXPERIMENTS[n].cells) & set(CELLS), n
    # registered LAST: the nearest tuple an existing refusal quotes is unchanged
    assert list(C.AUTHORISED_OOS_EXPERIMENTS)[-len(LANE_OOS_TUPLES):] == [LANE_OOS_TUPLES[c] for c in C.G8K_CAPS]
    (argv, _name, roster), = _driver_plans(tmp_path, "0")
    monkeypatch.chdir(REPO)
    monkeypatch.setenv(C.ENV_FULL22_ROSTER, roster)
    without = {n: e for n, e in C.AUTHORISED_OOS_EXPERIMENTS.items() if n not in LANE_OOS_TUPLES.values()}
    monkeypatch.setattr(C, "AUTHORISED_OOS_EXPERIMENTS", without)
    with pytest.raises(C.OutOfSupportThinkingUnauthorised) as e:
        _oos_door(D, argv)
    assert CANARY_REFUSAL in str(e.value) and C.g8k_allpos_tuple_name(TARGET, M.CAP) in str(e.value)
