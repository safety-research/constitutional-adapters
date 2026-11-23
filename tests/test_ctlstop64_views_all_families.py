"""The control-pool-stopped views ``ctlstop64s<k>`` (configs/sdfbase_full22_roster_ctlstop_eb64.yaml) carry the single-turn families,
best-of-N, the over-refusal and capability controls and GSM8K beside the multi-turn attacks, on the instruments and draw procedure of
the published seed-view rows:

* the nine sampled configs ``configs/sdfbase_full22_qwen35_4b_ctlstop64s<k>_sampled_t0{,_cap8192,_cap16384}.yaml`` are the generator's
  output under the lane roster, byte for byte, each at seed base k (the objects' training seed), ONE trial, trial 0 and the Qwen vendor
  preset (T 0.7 / top_p 0.8 / top_k 20 / min_p 0 / presence 1.5 / repetition 1.0), the static cap of its rung and the controls cap 512;
* each wave config draws best-of-N natively at 4,096 (the view's ``caps: {bon: 4096}``) and still serves ``[none, the four cells]``;
* ``roster_view.py ... controls`` lists mmlu, xstest and or_bench_hard_1k on every view (the capability and controls legs);
* the seeded-lane seed rule is ONE rule: the sampled generator's table, the decode-preset file and the GSM8K views module name the same
  three views at the same seeds;
* the job-start registration (``jobplan.build_plans``, pin form on, campaign ``reseed_ctlstop``) binds pasttense / codeattack / renellm /
  grok420 / pair / xstest / or_bench_hard / gsm8k to the published instruments with the twelve object ids, while best-of-N binds the
  canonical 512-store instrument and MMLU is refused (two canonical seeds): those two register through the lane's own seeds instead;
* the ``ctlstop`` GSM8K matrix is registered in every table and launcher, regenerates byte-identically from the lane roster + pins its
  views file sha-pins, and holds 12 launch rows (no ``none`` row), each at seed base k under the replicate rule, the common geometry
  class and the view's rank-8 LoRA difference as the standup companion, with its own overlay pair;
* the GSM8K driver admits a row's ``--bank-view`` only from the roster it is handed: the launcher exports the job's roster for it
  (``g8k_roster_env``), the dry run prints that variable beside the DRIVER: line, and every row's argv — the 16,384 rung form
  included — parses under it and is refused at the parser without it;
* the pin-form resolution of ``g8k_views`` refuses every malformed or foreign pin by name, and leaves a string cell unchanged.
"""
from __future__ import annotations

import contextlib
import copy
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
VIEWS = ("ctlstop64s1", "ctlstop64s2", "ctlstop64s3")
CELLS = ("constv2ctl2-full__lora_r4_ce", "constv2ctl2-full__psr_ce_all",
         "constv2-mix-full__lora_r4_ce_minus_ctl", "constv2-mix-full__psr_ce_all_minus_ctl")
COMPANION = "constv2-mix-full__lora_r4_ce_minus_ctl"       # the view's rank-8 LoRA difference
LANE_ROSTER_REL = "configs/sdfbase_full22_roster_ctlstop_eb64.yaml"
LANE_ROSTER_PATH = REPO / LANE_ROSTER_REL
PROD_ROSTER_PATH = REPO / "configs/sdfbase_full22_roster.yaml"
LANE_PINS_REL = "configs/ctlstop_eb64_pins.json"
LANE_PINS = json.loads((REPO / LANE_PINS_REL).read_text())
CAMPAIGN = "reseed_ctlstop"
#: the twelve objects, by (view, cell): the pins file is the authority, these literals the cross-check (control LoRA, control ReLU,
#: LoRA difference, composed ReLU difference; seeds 1..3)
OBJECTS = {
    ("ctlstop64s1", CELLS[0]): "296e68cb226b798c09da5c7614e2d18157965ec84a357b96f3367bac6d32cd4f",
    ("ctlstop64s2", CELLS[0]): "bcf577784f1e63ef7afb0c22f80f75d3329b8bd00a2ceaf506725e823bfa7f17",
    ("ctlstop64s3", CELLS[0]): "7d10e3f4264a7438f278228b6533c9cc71a8eb96a3ceb0fbb71627bc7ab2f8c9",
    ("ctlstop64s1", CELLS[1]): "5a6debc0d544ad884bc02b6ca50fe0bc415f4ddab5c3b3d1d8b204b05fdbb3df",
    ("ctlstop64s2", CELLS[1]): "5304858d6e9848e1d4d770401ea9a3e19a36b8cbd2e9072b55bb50647939d59c",
    ("ctlstop64s3", CELLS[1]): "5b8f0d2f21057e511f9aa256de6400622617d134b6b08e84e4a14b294d79bc66",
    ("ctlstop64s1", CELLS[2]): "ef09dfc52403920405147067b42db4f9da287cf2190ba4e4d3470341d0934203",
    ("ctlstop64s2", CELLS[2]): "2332a70ca2e017d159901676c8d7b43d19b636dad269296379a21176214e5b82",
    ("ctlstop64s3", CELLS[2]): "d9d8fedcc4afdf990e2eb917fb8ba9d990d14e70a65092cf5baa2112ac57b0ba",
    ("ctlstop64s1", CELLS[3]): "919692f0a0815f5756a367f349b3c26d790e450069ffdd6b7845564041bb379f",
    ("ctlstop64s2", CELLS[3]): "2f8dba7f964c970b09a8d67d0c2bc3127f6bf4b3a5c0e3eb2e3da5cfa89e82eb",
    ("ctlstop64s3", CELLS[3]): "2353cf807c1ce503ba0ec7e0823d49c8dced2de038238d5d406365559f4004d6",
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
MATRIX_NAME = "ctlstop"
G8K = REPO / "scripts/sdfbase/gsm8k"
ENV = {**os.environ, "PYTHONPATH": str(REPO / "src")}


def _k(view: str) -> int:
    return int(view[-1])


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_the_twelve_objects_are_the_lane_pins():
    for (view, cell), sha in OBJECTS.items():
        assert LANE_PINS[TARGET][f"{view}__{cell}"]["object_sha256"] == sha, (view, cell)


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


def test_the_seeded_lane_rule_names_the_three_views_at_their_training_seed_in_every_table():
    presets = msc.load_presets()
    assert set(VIEWS) <= set(msc.SEEDED_LANE_VIEWS[TARGET])
    for view in VIEWS:
        k, tag = _k(view), msc.tag_of(TARGET, view)
        assert msc.seed_of_lane_view(view) == k == ES.seed_base(k)
        assert msc.SEED_BASE_RULE_BY_TAG[tag] == presets["seed_base_by_view"][tag] == k != presets["seed_base"]
        assert presets["n_trials_by_view"][tag] == 1 != presets["n_trials"]
        # the GSM8K views module carries the SAME rule for the same views (one rule, two stdlib-free readers)
        assert V.SEEDED_LANE_VIEW_SEEDS[view] == k == V.replicate_seed_of_view(view)
    # this lane's entries are exactly its three views; the SI-distilled refit views' sikl4s<k> entries are pinned by
    # tests/test_sikl4_lane.py, and the generator comparison's qg1ms<k> entries by
    # tests/test_qwengen1m_more_evals_lane.py
    assert {v for v in V.SEEDED_LANE_VIEW_SEEDS if v.startswith("ctlstop")} == set(VIEWS)
    assert set(VIEWS) < set(V.SEEDED_LANE_VIEW_SEEDS) and all(V.SEEDED_LANE_VIEW_SEEDS[v] == _k(v) for v in VIEWS)
    # the seed views keep eval_seed's own rule
    assert [V.replicate_seed_of_view(v) for v in ("s1", "s2p", "s3")] == [1, 2, 3]
    with pytest.raises(ValueError):
        V.replicate_seed_of_view("ctlstop64s4")


# --------------------------------------------------------------------------- the wave configs and the roster's legs
@pytest.mark.parametrize("view", VIEWS)
def test_each_wave_config_draws_best_of_n_natively_at_4096_and_serves_the_four_cells(view):
    cfg = yaml.safe_load((REPO / f"configs/sdfbase_full22_{TARGET}_{view}.yaml").read_text())
    assert cfg["bon"]["max_new_tokens"] == 4096 and cfg["explicit_cells"] == ["none", *CELLS]
    assert cfg["attacks"]["static_max_new_tokens"] == 4096 and cfg["attacks"]["target_max_new_tokens"] == 4096


@pytest.mark.parametrize("view", VIEWS)
def test_roster_view_lists_the_capability_and_over_refusal_controls(view):
    r = subprocess.run([sys.executable, str(REPO / "scripts/sdfbase/full22/roster_view.py"), TARGET, view, "controls",
                        "--roster", str(LANE_ROSTER_PATH)], capture_output=True, text=True, env=ENV, cwd=REPO, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().split(",") == ["mmlu", "xstest", "or_bench_hard_1k"]          # gsm8k is retired from the controls leg


# --------------------------------------------------------------------------- the job-start registration
def _plans(cfg_rel: str, evals: list, view: str, *, objects=None, producer_key=None):
    return JP.build_plans(config_path=str(REPO / cfg_rel), model_key=f"{TARGET}_{view}", roster_selection=view, evals=evals,
                          serving_path="lora_native", campaign=CAMPAIGN, job_id="4242", job_name="synthetic-ctlstop",
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
def test_the_job_start_registration_binds_the_published_instruments_with_the_twelve_objects(view, monkeypatch):
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


# --------------------------------------------------------------------------- the GSM8K matrix
def test_the_ctlstop_matrix_is_registered_in_every_table_and_launcher():
    assert M.MATRICES[MATRIX_NAME] == REPO / "configs/sdfbase/gsm8k/matrix_ctlstop.json"
    assert M.VIEWS_BY_MATRIX[MATRIX_NAME] == REPO / "configs/sdfbase/gsm8k/views_ctlstop.yaml"
    assert M.WAVE_CTLSTOP == MATRIX_NAME and M.out_root_prefix_for(M.WAVE_CTLSTOP) == "controls_ctlstop"
    assert M.WAVE_CTLSTOP not in M.COMPLETED_WAVES
    for name in ("gsm8k_thinking.sbatch", "ladder_plan.sbatch", "dry_run_prod.sbatch", "smoke_not_available.sbatch"):
        m = re.search(r'case "\$G8K_MATRIX" in ([a-z0-9_|]+)\) ;;', (G8K / name).read_text())
        assert m and MATRIX_NAME in m.group(1).split("|"), name
    regen = (REPO / "scripts/release/regen_configs.sh").read_text()
    for line in ('"scripts/sdfbase/gsm8k/make_gsm8k_matrix.py --matrix-name ctlstop"', '"scripts/sdfbase/gsm8k/make_gsm8k_overlays.py --matrix ctlstop"',
                 '"scripts/sdfbase/full22/sampled/make_sampled_configs.py --roster configs/sdfbase_full22_roster_ctlstop_eb64.yaml"'):
        assert line in regen, line


def test_the_ctlstop_views_file_pins_the_lane_roster_and_pins_and_the_matrix_regenerates_byte_identically():
    spec = V.load_views_yaml(M.VIEWS_BY_MATRIX[MATRIX_NAME])
    assert spec.schema == M.MATRIX_SCHEMA_2 and spec.wave == M.WAVE_CTLSTOP and spec.row_overlays
    assert spec.views == {TARGET: list(VIEWS)} and spec.holds == []
    assert [(e["view"], e["cells"]) for e in spec.exclude] == [(v, ["none"]) for v in VIEWS]
    assert spec.inspect_seed_policy == M.INSPECT_SEED_POLICY_REPLICATE and spec.inspect_seed_base_by_view == {v: _k(v) for v in VIEWS}
    assert spec.standup_companions["cell"] == COMPANION and spec.geometry_class == M.GEOMETRY_CLASS_COMMON_F22
    assert spec.trigger_threshold == M.TRIGGER_THRESHOLD and spec.wave_root_rel == "exp/gsm8k_paired200_paper_objects_ctlstop"
    frozen = V.frozen_input_paths(spec, REPO)
    assert frozen == (LANE_ROSTER_PATH, REPO / LANE_PINS_REL)
    text, doc = MK.build(configs_root=REPO, views_yaml=None, views_spec=None, matrix_name=MATRIX_NAME)
    assert M.MATRICES[MATRIX_NAME].read_text() == text
    # the document names the lane-private files it was derived from, never the production roster
    assert doc["source"] == {"views_yaml": "configs/sdfbase/gsm8k/views_ctlstop.yaml",
                             "roster": "configs/sdfbase_full22_roster_ctlstop_eb64.yaml", "roster_sha256": _sha256(LANE_ROSTER_PATH),
                             "pins": LANE_PINS_REL, "pins_sha256": _sha256(REPO / LANE_PINS_REL)}


def test_the_ctlstop_matrix_holds_twelve_launch_rows_at_the_training_seed_with_the_companion_and_no_unsteered_row():
    doc = M.load_matrix(M.MATRICES[MATRIX_NAME])
    rows = doc["rows"]
    assert len(rows) == 12 and [r["index"] for r in rows] == list(range(12))
    assert [(r["view"], r["cell"]) for r in rows] == [(v, c) for v in VIEWS for c in CELLS]          # no `none` row
    for r in rows:
        view, cell, k = r["view"], r["cell"], _k(r["view"])
        assert r["model_key"] == TARGET and r["tag"] == f"{TARGET}_{view}" and r["status"] == M.STATUS_LAUNCH and r["fitted"]
        assert r["staged_id"] == f"{view}__{cell}" and r["object_sha256"] == OBJECTS[(view, cell)]
        assert r["wave"] == M.WAVE_CTLSTOP and r["own_overlay"] and r["out_root"].startswith(f"controls_ctlstop/{TARGET}_{view}/")
        assert (r["inspect_seed_policy"], r["inspect_seed_base"]) == (M.INSPECT_SEED_POLICY_REPLICATE, k) and M.row_inspect_seed(r) == ("replicate", k)
        assert r["thinking_mode"] == M.THINKING_ON and r["sampling_mode"] == M.SAMPLING_THINKING_CARD
        assert M.row_geometry_class(r) == M.GEOMETRY_CLASS_COMMON_F22
        assert M.geometry_for_cell(cell, M.CAP, model_key=TARGET, geometry_class=M.row_geometry_class(r)) == ("capability:gen_long_gsm8k_common_f22", 16640)
        assert r["tuple_name"] == "gsm8k_paired200_qwen35_4b_8192"
        if cell == COMPANION:
            assert M.row_companions(r) == []                     # the companion's own row registers itself
        else:
            assert M.row_companions(r) == [(COMPANION, V.harness_cell_id(COMPANION), f"{view}__{COMPANION}", OBJECTS[(view, COMPANION)])]
        assert M.launch_cells_for(r)[0] == "none"                # the standup serves the base route, unmeasured
    assert [r["index"] for r in M.eligible_rows(M.WAVE_CTLSTOP, rows)] == list(range(12))


def test_every_ctlstop_row_serves_through_its_own_overlay_pair_cut_from_the_campaign_config():
    rows = M.load_matrix(M.MATRICES[MATRIX_NAME])["rows"]
    outs = list(OV.outputs(REPO, rows))
    assert len(outs) == 24
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
    """The ``--bank-view`` value an argv carries."""
    return argv[argv.index(C.FLAG_BANK_VIEW) + 1]


def test_the_bank_view_vocabulary_admits_the_lane_views_only_from_the_lane_roster(monkeypatch):
    """The ONE vocabulary the driver's flag, its per-row identity stamp and the GSM8K reader read (``contracts.bank_view_problem``):
    `main` | `it`, the views of ONE roster, and the source banks the matrices declare. The production roster names none of these
    views; the lane roster names exactly them and keeps the instrument views and every declared source bank."""
    monkeypatch.delenv(C.ENV_FULL22_ROSTER, raising=False)
    prod = C.bank_views_admitted(PROD_ROSTER_PATH)
    assert not set(VIEWS) & set(prod)
    for v in VIEWS:
        assert f"{C.FLAG_BANK_VIEW} {v!r} is not one of" in C.bank_view_problem(v, PROD_ROSTER_PATH)
    monkeypatch.setenv(C.ENV_FULL22_ROSTER, str(LANE_ROSTER_PATH))          # the variable the launcher exports, read with no argument
    lane = C.bank_views_admitted()
    assert all(C.bank_view_problem(v) is None and C.check_bank_view_arg(v) == v for v in VIEWS)
    declared = C.declared_bank_views(PROD_ROSTER_PATH)
    assert declared and set(lane) == set(C.BANK_VIEWS) | set(VIEWS) | set(declared) and lane[:len(C.BANK_VIEWS)] == tuple(C.BANK_VIEWS)
    assert C.bank_view_problem("s1") is not None           # ONE roster at a time: a production lane view is not this job's


def test_the_gsm8k_driver_admits_every_row_under_the_jobs_roster_and_refuses_it_without(tmp_path, monkeypatch):
    """Every job of this matrix is submitted with ``F22_ROSTER`` naming the lane roster. The launcher hands the DRIVER that same
    roster (``g8k_roster_env``, inside the one argv function, so the job and the dry run cannot differ), and the dry run prints
    the variable as a child process reads it. Under it all twelve rows' argv and the 16,384 rung form parse, each stamped with its
    own view and pinned to its own object; without it the driver refuses every one at its parser — the refusal a job would have
    met after its standup."""
    common = (G8K / "g8k_common.sh").read_text()
    assert re.search(r"^G8K_ROSTER_ENV=(\S+)$", common, re.M).group(1) == C.ENV_FULL22_ROSTER          # one name in the shell and the driver
    plans = _driver_plans(tmp_path, "0-11", rung_rows="5")
    rows = M.load_matrix(M.MATRICES[MATRIX_NAME])["rows"]
    got = []
    for argv, name, value in plans:
        assert name == C.ENV_FULL22_ROSTER and Path(value).is_absolute() and Path(value).resolve() == LANE_ROSTER_PATH.resolve(), (name, value)
        ns, refusal = _parse_driver(argv, monkeypatch, roster=value)
        assert refusal is None, refusal
        assert ns.bank_view in VIEWS and ns.defenses_dir.endswith(f"/{TARGET}/cells_view/{ns.bank_view}"), (ns.bank_view, ns.defenses_dir)
        assert ns.expect_object_sha256 == OBJECTS[(ns.bank_view, ns.cell)] and int(ns.inspect_seed_base) == _k(ns.bank_view)
        got.append((ns.bank_view, ns.cell, bool(ns.item_subset)))
        ns, refusal = _parse_driver(argv, monkeypatch, roster=None)
        assert ns is None and refusal[0] == 2, (ns, refusal)
        assert f"argument {C.FLAG_BANK_VIEW}: {C.FLAG_BANK_VIEW} must be one of" in refusal[1], refusal[1][-400:]
        assert f"got {_bank_view_of(argv)!r}" in refusal[1], refusal[1][-400:]
    want =[(r["view"], r["cell"], False) for r in rows] + [(rows[5]["view"], rows[5]["cell"], True)]
    assert sorted(got) == sorted(want) and len(got) == 13


def test_a_parse_only_run_of_the_driver_line_passes_under_the_plans_roster_alone(tmp_path):
    """The cheap gate, run the way a job execs the driver (the plan's interpreter and ``scripts/served_capability_run.py`` from the
    checkout, ``PYTHONPATH`` src:scripts): the DRIVER: argv with ``--help`` appended exits 0 under the plan's ROSTER_ENV: line and
    2 without it. One base row and its 16,384 rung form."""
    plans = _driver_plans(tmp_path, "5", rung_rows="5")
    assert [C.FLAG_ITEM_SUBSET in argv for argv, _, _ in plans] == [False, True]
    base = {k: v for k, v in os.environ.items() if k not in (C.ENV_FULL22_ROSTER, "F22_ROSTER")}
    base.update(CUDA_VISIBLE_DEVICES="", PYTHONPATH=f"{REPO / 'src'}:{REPO / 'scripts'}")
    for argv, name, value in plans:
        ok = subprocess.run(argv + ["--help"], cwd=REPO, env={**base, name: value}, capture_output=True, text=True, timeout=600)
        assert ok.returncode == 0 and ok.stdout.lstrip().startswith("usage:"), (ok.returncode, ok.stderr[-800:])
        bad = subprocess.run(argv + ["--help"], cwd=REPO, env=base, capture_output=True, text=True, timeout=600)
        assert bad.returncode == 2 and f"argument {C.FLAG_BANK_VIEW}" in bad.stderr and f"got {_bank_view_of(argv)!r}" in bad.stderr, \
            (bad.returncode, bad.stderr[-800:])


def test_a_submit_line_without_the_lane_roster_is_exposed_by_the_plans_own_lines(tmp_path, monkeypatch):
    """The known-bad twin. Left off the submit line, ``F22_ROSTER`` is the production roster, which names none of these views: the
    plan then names the production roster for the driver, and the driver refuses the row under it — the same two lines that
    prove a correct launch expose this one."""
    (argv, name, value), = _driver_plans(tmp_path, "0", roster=None)
    assert name == C.ENV_FULL22_ROSTER and Path(value).resolve() == PROD_ROSTER_PATH.resolve()
    ns, refusal = _parse_driver(argv, monkeypatch, roster=value)
    assert ns is None and refusal[0] == 2 and f"got {_bank_view_of(argv)!r}" in refusal[1], (ns, refusal)


# --------------------------------------------------------------------------- the pin-form resolution (known-bad twins)
def test_pin_form_cells_resolve_through_the_matrix_pins_file_and_refuse_every_foreign_shape():
    pins = copy.deepcopy(LANE_PINS)
    ok = {"pins": LANE_PINS_REL, "pin": f"{TARGET}.ctlstop64s2__{CELLS[1]}"}
    assert V.staged_id_of(ok, key=TARGET, cell=CELLS[1], pins=pins, pins_file=LANE_PINS_REL) == f"ctlstop64s2__{CELLS[1]}"
    assert V.staged_id_of("base64s1__x", key=TARGET, cell="x") == "base64s1__x" and V.staged_id_of(None, key=TARGET, cell="p") is None
    with pytest.raises(SystemExit, match="needs the matrix's pins document"):
        V.staged_id_of(ok, key=TARGET, cell=CELLS[1])
    with pytest.raises(SystemExit, match="one pins authority per matrix"):
        V.staged_id_of(ok, key=TARGET, cell=CELLS[1], pins=pins, pins_file="configs/sdfbase_full22_pins.json")
    with pytest.raises(SystemExit, match="is not `qwen35_4b.<entry>`"):
        V.staged_id_of(dict(ok, pin=f"gemma4e4b.ctlstop64s2__{CELLS[1]}"), key=TARGET, cell=CELLS[1], pins=pins, pins_file=LANE_PINS_REL)
    with pytest.raises(SystemExit, match="has no"):
        V.staged_id_of(dict(ok, pin=f"{TARGET}.ctlstop64s9__{CELLS[1]}"), key=TARGET, cell=CELLS[1], pins=pins, pins_file=LANE_PINS_REL)
    moved = copy.deepcopy(LANE_PINS)
    moved[TARGET]["alias"] = dict(moved[TARGET][f"ctlstop64s2__{CELLS[1]}"])
    with pytest.raises(SystemExit, match="under another key"):
        V.staged_id_of(dict(ok, pin=f"{TARGET}.alias"), key=TARGET, cell=CELLS[1], pins=moved, pins_file=LANE_PINS_REL)
    with pytest.raises(SystemExit, match="neither a staged id nor the roster's pin form"):
        V.staged_id_of({"pins": LANE_PINS_REL}, key=TARGET, cell=CELLS[1], pins=pins, pins_file=LANE_PINS_REL)


def test_the_views_loader_holds_each_lane_views_base_to_its_training_seed(tmp_path):
    text = M.VIEWS_BY_MATRIX[MATRIX_NAME].read_text()
    bad = tmp_path / "views_ctlstop.yaml"
    bad.write_text(text.replace("  ctlstop64s2: 2\n", "  ctlstop64s2: 1\n", 1))
    with pytest.raises(SystemExit, match=r"inspect_seed_base_by_view.ctlstop64s2 = 1 but eval_seed.seed_base\(2\) = 2"):
        V.load_views_yaml(bad)
