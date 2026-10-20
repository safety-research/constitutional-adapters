"""The AlphaSteer strength SCREEN launcher (scripts/alphasteer_f22/screen): the scope table, the
sampled variants, and the sbatch wrapper's identity with the wave's controls phase.

The launcher's claim is that an AlphaSteer strength view and a LoRA dose view are screened on ONE instrument. This file
pins the three places where that claim could silently break: the sampled config (must be the serving config as data
except the wave's permitted keys, with the wave generator's block for the served model), the wrapper (must carry the
wave sbatch's controls phase BYTE-FOR-BYTE, and must not carry a `--export=` or a `--mem=0`), and the scope (fifteen
paper views; glm46vflash refused by name).
"""
from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SCREEN = REPO / "scripts" / "alphasteer_f22" / "screen"
SAMPLED = REPO / "scripts" / "sdfbase" / "full22" / "sampled"
WRAPPER = SCREEN / "asscreen_controls.sbatch"
for p in (SCREEN, SCREEN.parent, SAMPLED):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import as22_screen as scr  # noqa: E402
import as22_strength as st  # noqa: E402
import make_sampled_configs as msc  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = _load("make_strength_sampled_configs", SCREEN / "make_strength_sampled_configs.py")
WAVE_SBATCH_REL = "scripts/sdfbase/full22/sampled/target_sampled.sbatch"


def _wave_sbatch_text() -> str:
    """The LoRA screens' sbatch at the pinned wave sha, from this repository's object store; skipped on a clone without it."""
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "show", f"{scr.LORA_SCREEN_WAVE_SHA}:{WAVE_SBATCH_REL}"],
                                       text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        pytest.skip(f"{scr.LORA_SCREEN_WAVE_SHA[:12]} is not in this clone's object store (fetch strength-wave)")


def _region(text: str, start: str, end: str, *, inclusive_end: bool) -> str:
    lines = text.splitlines(keepends=True)
    a = next(i for i, l in enumerate(lines) if l.startswith(start))
    b = next(i for i, l in enumerate(lines) if l.startswith(end))
    return "".join(lines[a:(b + 1) if inclusive_end else b]).rstrip("\n")


# ------------------------------------------------------------------------------------------------------------ #
# scope + grammar                                                                                               #
# ------------------------------------------------------------------------------------------------------------ #

def test_scope_is_the_fifteen_paper_views_and_glm_is_refused_by_name():
    assert scr.SCREEN_TARGETS == ("gemma4e4b", "qwen35_4b", "nemotron3nano4b")
    assert scr.DESCOPED_TARGETS == ("glm46vflash",)
    assert all(scr.SCREEN_VIEWS[t] == ("as_dm0p5", "as_d0p1", "as_d0p3", "as_d0p5", "as_d1") for t in scr.SCREEN_TARGETS)
    assert len(scr.screen_tags()) == 15 and scr.screen_tags()[0] == "gemma4e4b_as_dm0p5"
    with pytest.raises(scr.ScreenRefusal, match="DESCOPED.*D62 / D70"):
        scr.check_scope("glm46vflash", "as_d0p3")
    with pytest.raises(scr.ScreenRefusal, match="not a strength view"):
        scr.check_scope("nemotron3nano4b", "d0p3")          # the LoRA dose spelling is not ours
    with pytest.raises(scr.ScreenRefusal, match="unknown target"):
        scr.check_scope("llama8b", "as_d0p3")
    assert scr.main(["--is-in-scope", "glm46vflash", "as_d1"]) == 3
    assert scr.main(["--is-in-scope", "gemma4e4b", "as_d1"]) == 0


def test_tag_job_name_and_sampled_naming_follow_the_wave_grammar():
    assert scr.tag_of("nemotron3nano4b", "as_d0p3") == "nemotron3nano4b_as_d0p3"
    assert scr.split_tag("gemma4e4b_as_dm0p5") == ("gemma4e4b", "as_dm0p5")
    assert scr.job_name("nemotron3nano4b", "as_d0p3") == "asscr-nemotron3nano4b_as_d0p3-ctl0"
    assert scr.job_name("qwen35_4b", "as_d1", 1) == "asscr-qwen35_4b_as_d1-ctl1"
    assert scr.sampled_name("gemma4e4b", "as_d0p1", 0) == "alphasteer_f22_gemma4e4b_as_d0p1_sampled_t0"
    assert scr.sampled_name("gemma4e4b", "as_d0p1", 1, 8192) == "alphasteer_f22_gemma4e4b_as_d0p1_sampled_t1_cap8192"
    assert scr.sampled_config_rel("gemma4e4b", "as_d0p1", 0, 16384) == "configs/alphasteer_f22_serve_gemma4e4b_as_d0p1_sampled_t0_cap16384.yaml"
    # the LoRA screens' code sha, spelled once
    assert re.fullmatch(r"[0-9a-f]{40}", scr.LORA_SCREEN_WAVE_SHA) and scr.LORA_SCREEN_WAVE_SHA.startswith("f38d055d")


def test_resolve_prints_absolute_paths_that_exist_and_the_roster_agrees_with_the_table():
    d = scr.resolve("nemotron3nano4b", "as_d0p3", 0, repo=REPO, shared="/workspace/TESTUSER")
    assert d["KEY"] == "nemotron3nano4b" and d["TAG"] == "nemotron3nano4b_as_d0p3" and d["STRENGTH"] == "0.3"
    assert d["CELL"] == "alphasteer" and d["STAGED_ID"] == "as_d0p3__alphasteer"
    for k in ("WAVE_CFG", "CFG", "CFG_ESC", "CFG_ESC16", "MANIFEST", "AS_ROSTER", "AS_PINS"):
        assert Path(d[k]).is_absolute() and Path(d[k]).is_file(), (k, d[k])
    assert d["AS_VIEW_DIR"] == "/workspace/TESTUSER/exp/objects/alphasteer_f22/nemotron3nano4b/strength/cells_view/as_d0p3"
    assert d["AS_BANK"] == "/workspace/TESTUSER/exp/objects/alphasteer_f22/nemotron3nano4b/strength"
    assert d["PRIVATE_RUN_BASE"] == "alphasteer_f22_nemotron3nano4b_as_d0p3_sampled_t0" and d["SIBLING_TRIAL"] == "1"
    assert d["AS_SERVED_CELLS"] == "none,alphasteer" and d["JOB_NAME"] == "asscr-nemotron3nano4b_as_d0p3-ctl0"
    # shell-safe: every value single-quoted or a plain word; one assignment per line
    text = scr.shell_assignments(d)
    assert all(re.fullmatch(r"[A-Z_0-9]+=(?:'[^']*'|[A-Za-z0-9_./:@%+=,-]+)", ln) for ln in text.splitlines()), text
    for t in scr.SCREEN_TARGETS:
        for v in scr.SCREEN_VIEWS[t]:
            scr.resolve(t, v, 0, repo=REPO, shared="/workspace/TESTUSER")   # every in-scope view resolves


# ------------------------------------------------------------------------------------------------------------ #
# the sampled variants                                                                                          #
# ------------------------------------------------------------------------------------------------------------ #

def test_the_ninety_sampled_configs_are_byte_identical_to_the_generator():
    presets = msc.load_presets()
    gen.check_pins(presets, scr.load_roster(REPO))
    outs = gen.outputs()
    assert len(outs) == 15 * 2 * 3
    drift = [str(dst) for dst, t, v, trial, cap in outs if not dst.is_file() or dst.read_text(encoding="utf-8") != gen.render(presets, t, v, trial, cap)]
    assert drift == [], drift


@pytest.mark.parametrize("dst,target,view,trial,cap", [(d, t, v, tr, c) for d, t, v, tr, c in gen.outputs()],
                         ids=[d.name for d, *_ in gen.outputs()])
def test_sampled_config_is_the_serving_config_as_data_except_the_permitted_keys(dst, target, view, trial, cap):
    got = yaml.safe_load(dst.read_text(encoding="utf-8"))
    src = yaml.safe_load((REPO / st.serve_config_rel(target, st.strength_of_view(view))).read_text(encoding="utf-8"))
    changed = {k for k in set(got) | set(src) if got.get(k) != src.get(k)}
    assert changed == {"out_dir", "private_run", "sampled_decode"} | ({"attacks"} if cap else set()), sorted(changed)
    assert got["explicit_cells"] == ["none", "alphasteer"] == src["explicit_cells"]
    name = scr.sampled_name(target, view, trial, cap)
    assert got["out_dir"] == f"out/{name}" and got["private_run"] == name
    sd = got["sampled_decode"]
    # the judged controls' cap is its OWN key since main's 2026-09-11 multi-turn ruling: 512, while the multi-turn
    # target cap is the roster's wave-wide 4,096 — the campaign config's values, which the strength view holds (never moves)
    assert sd["trial"] == trial and sd["controls_cap"] == 512 == got["attacks"]["controls_max_new_tokens"]
    assert got["attacks"]["target_max_new_tokens"] == 4096
    assert sd["static_cap"] == got["attacks"]["static_max_new_tokens"] == (cap or 4096)
    assert sd["readouts"]["gsm8k"]["thinking"] == "on" and sd["readouts"]["gsm8k"]["cap"] == 8192
    if cap:
        prev_cap, prev_esc = msc.previous_rung(cap)
        assert sd["escalated_from"] == {"kind": "static", "cap": prev_cap,
                                        "private_run": scr.sampled_name(target, view, trial, None if prev_esc is False else prev_cap)}
    else:
        assert "escalated_from" not in sd


def test_the_block_is_the_wave_generators_for_the_served_model_except_the_escalation_chain_name():
    presets = msc.load_presets()
    for target in scr.SCREEN_TARGETS:
        for view in scr.SCREEN_VIEWS[target]:
            for trial in range(int(presets["n_trials"])):
                ours = gen.expected_block(presets, target, view, trial, None)
                theirs = msc.expected_block(presets, target, trial, False)        # the model's main-view TAG is the bare key
                assert ours == theirs, (target, view, trial)
                for cap in msc.STATIC_RUNGS:
                    ours = gen.expected_block(presets, target, view, trial, cap)
                    theirs = msc.expected_block(presets, target, trial, cap)
                    assert {k: v for k, v in ours.items() if k != "escalated_from"} == {k: v for k, v in theirs.items() if k != "escalated_from"}
                    assert ours["escalated_from"]["private_run"].startswith(f"alphasteer_f22_{target}_{view}_sampled_t{trial}")
                    assert {k: v for k, v in ours["escalated_from"].items() if k != "private_run"} == \
                           {k: v for k, v in theirs["escalated_from"].items() if k != "private_run"}


def test_a_strength_view_and_a_lora_dose_view_of_one_target_differ_as_data_in_exactly_the_four_serving_keys():
    """The comparability pin: the LoRA dose view's sampled config vs ours, same target, same trial, differ AS DATA in
    exactly the four serving keys. Read against THIS tree's LoRA dose config (the same generators, the same roster caps):
    since main restated the wave-wide caps (target 4,096 / controls 512 / the BoN climb keys) both views carry the
    campaign's current values, so the frozen-sha copy of the LoRA config (scr.LORA_SCREEN_WAVE_SHA, the 512 era) is no
    longer the like-for-like partner — the screen records that ran at that sha keep their own provenance."""
    lora = yaml.safe_load((REPO / "configs/sdfbase_full22_nemotron3nano4b_d0p25_sampled_t0.yaml").read_text(encoding="utf-8"))
    ours = yaml.safe_load((REPO / scr.sampled_config_rel("nemotron3nano4b", "as_d0p3", 0)).read_text(encoding="utf-8"))
    changed = {k for k in set(lora) | set(ours) if lora.get(k) != ours.get(k)}
    assert changed == {"defenses_dir", "explicit_cells", "out_dir", "private_run"}, sorted(changed)
    assert lora["sampled_decode"] == ours["sampled_decode"]


def test_generator_refuses_a_preset_that_is_not_the_rosters_snapshot_and_a_descoped_target():
    presets = msc.load_presets()
    roster = scr.load_roster(REPO)
    bad = yaml.safe_load(yaml.safe_dump(roster))
    bad["targets"]["nemotron3nano4b"]["model_revision"] = "0" * 40
    with pytest.raises(scr.ScreenRefusal, match="strength roster serves"):
        gen.check_pins(presets, bad)
    with pytest.raises(scr.ScreenRefusal):
        gen.render(presets, "glm46vflash", "as_d0p3", 0, None)


# ------------------------------------------------------------------------------------------------------------ #
# the wrapper                                                                                                   #
# ------------------------------------------------------------------------------------------------------------ #

def test_wrapper_parses_and_carries_the_cluster_rules():
    text = WRAPPER.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    assert "--export=" not in text.replace("never --export=", "").replace("never `--export=`", "")
    assert "--mem=0" not in text and "#SBATCH --mem=200G" in text and "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --qos=high" in text and "#SBATCH --no-requeue" in text and "#SBATCH --open-mode=append" in text
    assert 'source "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$WAVE_SHA/scripts/sdfbase/full22/sampled/common_sampled.sh"' in text
    assert 'f22_init "$WAVE_SHA"' in text and 'f22s_check_config "$MPY"' in text and "f22s_assert_vllm_mamba_kernels" in text
    assert "f22s_parse_phase" in text and 'as22_screen.py" --resolve' in text or 'AS_SCREEN" --resolve' in text
    # our two replacements, and nothing of the wave's that they replace
    assert "asscr_bank_view" in text and "f22_bank_view" not in text.split("asscr_bank_view(){")[1].split("}")[0]
    assert "f22s_resolve_config" not in text.replace("f22s_resolve_config ->", "").replace("(1) f22s_resolve_config", "")
    assert "make_bank_view.py --check" in text and '--roster "$AS_ROSTER" --pins "$AS_PINS" --bank "$AS_BANK"' in text
    assert '[ "${ALL_CELLS[*]}" = "none $CELL" ]' in text
    # the positive control runs, on the object cell, at every rung (inside the verbatim phase)
    assert "f22s_positive_control controls_512" in text and 'f22s_positive_control "controls_${RUNG}"' in text
    # the nominal-grid and 512-ladder notes travel with the launcher
    assert "NOMINAL STRENGTHS ARE LABELS" in text and "AT-CAP AT 512 IS THE LADDER WORKING" in text
    assert "INSTRUMENT DIFFERS" in text and "asscreen_job.json" in text


def test_wrapper_carries_the_wave_sbatchs_controls_phase_and_helpers_verbatim():
    wave = _wave_sbatch_text()
    ours = WRAPPER.read_text(encoding="utf-8")
    helpers = _region(wave, "ROOT_OF(){", 'TARGET_GPU="$CUDA_VISIBLE_DEVICES"', inclusive_end=True)
    phase = _region(wave, "# ---------------------------------------------------------------- controls_t<T>",
                    "# ---------------------------------------------------------------- ladder_extend_t<T>", inclusive_end=False)
    assert helpers in ours, "the phase helpers (ROOT_OF, f22s_ctl_top_logprobs, cleanup, trap) drifted from the wave sbatch"
    assert phase in ours, "the controls phase drifted from the wave sbatch — the two screens would no longer share a driver invocation"
    # the block keeps its `elif` and sits behind our guard `if`, so the shell shape is the wave's
    assert 'if [ "$PHASE_KIND" != controls ]; then f22_fatal' in ours
    assert ours.index('if [ "$PHASE_KIND" != controls ]') < ours.index('elif [ "$PHASE_KIND" = controls ]; then')
    # the phase is the whole of the wave's driver contract: the standup geometry, the driver, the ladder
    for needle in ('f22_standup attacks "$SERVE_DIR/controls_512" "" capability:short 2048 "" "$CTL_TOP_LOGPROBS"',
                   "scripts/served_capability_run.py generate", '--controls "$CTL_LIST" --concurrency "$CONC" --top-logprobs "$CTL_TOP_LOGPROBS"',
                   "for RUNG in 2048 8192; do", '--controls-cap "$RUNG" --escalate-from-root "$PREV_ROOT" --floor-sample "$F22S_FLOOR_SAMPLE_K"',
                   'f22s_write_ladder controls "$LADDER_DIR/ladder.json"'):
        assert needle in phase and needle in ours, needle


def test_the_wrapper_wires_the_d87_numeric_gate_without_editing_the_verbatim_phase():
    """The greedy-text probe is an OBSERVATION at low dose. The gate runs before every standup's positive control —
    implemented by RENAMING the wave's f22s_positive_control and wrapping it, so the phase text stays byte-identical."""
    t = WRAPPER.read_text(encoding="utf-8")
    assert "eval \"$(declare -f f22s_positive_control | sed '1s/^f22s_positive_control/asscr_wave_positive_control/')\"" in t
    assert "asscr_wave_positive_control \"$ptag\"" in t          # the wave's own control still runs, unmodified
    assert "asscr_liveness.py" in t and "SISTEER_LOW_DOSE_LIVENESS_RECORD" in t
    # the three verdicts: admit (export), genuine refusal (return 1), unevaluable (unset -> the text gate governs)
    assert "export SISTEER_LOW_DOSE_LIVENESS_RECORD=" in t and "unset SISTEER_LOW_DOSE_LIVENESS_RECORD" in t
    assert "REFUSING TRAFFIC: low-dose liveness" in t and "return 1" in t
    assert "LOW-DOSE LIVENESS (D87" in t
    # the phase still calls the (now overridden) name, and the call sites are untouched
    assert "f22s_positive_control controls_512" in t and 'f22s_positive_control "controls_${RUNG}"' in t


def test_the_code_sha_is_the_lora_sha_plus_two_bounded_commits():
    """The screens run at LOW_DOSE_LIVENESS_WAVE_SHA = the LoRA screens' sha + TWO commits, both needed before an
    AlphaSteer object can be screened at all: the liveness seam (scripts-only, inert without the policy env) and the
    Class B provenance registration (canary 93844 died without it). The claim under test is that NEITHER touches the
    serving stack the manifests pin (`serving_impl_commit`), and that the file set is exactly the expected one — so a
    third, unexamined change cannot ride along silently."""
    assert re.fullmatch(r"[0-9a-f]{40}", scr.LOW_DOSE_LIVENESS_WAVE_SHA)
    assert set(scr.COMPARABLE_WAVE_SHAS) == {scr.LORA_SCREEN_WAVE_SHA, scr.LOW_DOSE_LIVENESS_WAVE_SHA}
    try:
        chain = subprocess.check_output(["git", "-C", str(REPO), "rev-list",
                                         f"{scr.LORA_SCREEN_WAVE_SHA}..{scr.LOW_DOSE_LIVENESS_WAVE_SHA}"],
                                        text=True, stderr=subprocess.DEVNULL).split()
        files = subprocess.check_output(["git", "-C", str(REPO), "diff", "--name-only",
                                         scr.LORA_SCREEN_WAVE_SHA, scr.LOW_DOSE_LIVENESS_WAVE_SHA], text=True,
                                        stderr=subprocess.DEVNULL).split()
    except subprocess.CalledProcessError:
        pytest.skip("the code sha is not in this clone's object store (fetch asscreen-d87-liveness)")
    assert len(chain) == 2, f"expected exactly two commits on top of the LoRA sha, got {chain}"
    # the ONE invariant that keeps the manifests honest
    assert not [f for f in files if f.startswith("src/sisteer/capeval/serving/") or f.endswith("tiers/petri_runner.py")], files
    assert set(files) == {"scripts/sdfbase/full22/low_dose_liveness_policy.py", "scripts/served_capability_run.py",
                          "scripts/sdfbase/full22/smoke_positive_control.py", "tests/test_low_dose_liveness_policy.py",
                          "src/sisteer/eval/sweep.py", "tests/test_class_b_fit_provenance.py"}, sorted(files)


def test_readme_carries_the_contract_sections():
    text = (SCREEN / "README.md").read_text(encoding="utf-8")
    for needle in ("Two shas, one instrument", "same_instrument_as_lora_screens", "Comparability with the LoRA screens",
                   "steering_bites", "Nominal strengths are labels", "-0.5, 0.1, 0.3, 0.5, 1.0", "realised_dose_median",
                   "The 512 rung is a ladder rung, not a verdict", "asscr-nemotron3nano4b_as_d0p3-ctl0",
                   "screen_verdict.py --local-root", "--roster configs/alphasteer_f22_strength_roster.yaml",
                   scr.LOW_DOSE_LIVENESS_WAVE_SHA, "an observation at low dose rather than a gate",
                   "asscr_liveness.py", "SISTEER_LOW_DOSE_LIVENESS_RECORD"):
        assert needle in text, needle
