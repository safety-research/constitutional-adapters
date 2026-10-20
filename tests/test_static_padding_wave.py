"""The PADDED STATIC leg's cluster staging (docs/static_padding.md; scripts/sdfbase/full22/static_padding/):

  * the generated config + manifest pair is byte-identical to its generator (`--check`), is the campaign pair AS
    DATA except the named keys, lists exactly the three prompted-side arms, pins the pad level in the CONFIG (one
    private_run per level: one padding contract per store) and never the filler path (the argv's), and passes the
    standup's own manifest/config gate.
  * the cap of EVERY level's config is the static ladder's CEILING (16,384; make_padding_configs.STATIC_CAP ==
    ladder_decide.CEILING) — the leg runs once and never needs the rung the rung driver would refuse a padded store — and
    the preflight refuses any other cap by name.
  * the pair lives OUTSIDE the wave-config enumerator's glob (tests/full22_wave_configs.py) — a subdirectory, the
    math500 overlays' precedent — so the byte-identity test of the generated wave configs never sees it.
  * the sbatch parses, carries the cluster's hard rules (positional only, never --export=, never --mem=0, the high QoS,
    append, the disarming cleanup trap), declares its cell source before f22_init, runs the lane preflight BEFORE the
    standup, stands up ONCE at the window positional with S derived, and its rendered driver argv carries the filler
    path on a padded level only, never --static-padding-tokens, never a greedy flag.
  * the preflight's arithmetic (worst-case pad, rendered need, the smallest aligned window) and its contract reader.
  * the STEERED family's two ADDED RUNGS (8,192 / 32,768; 2026-09-24) exist for that family only, at the same window and the
    same S, while the six FILED configs keep their exact sha256 (the results registry pins them); and the steered sbatch's
    SHARDED shape — the `<lo>-<hi>/N` positional, the per-(family, shard) leases that refuse overlap and re-laid-out stores,
    and the driver loop that issues exactly the single-job argv for the owned indices.
"""
from __future__ import annotations

import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
LANE = REPO / "scripts" / "sdfbase" / "full22" / "static_padding"
SBATCH = LANE / "target_static_padded.sbatch"
CONFIG_DIR = REPO / "configs" / "sdfbase" / "static_padding"
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase", REPO / "tests", LANE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import make_padding_configs as gen  # noqa: E402
import padding_preflight as pf  # noqa: E402
from full22_wave_configs import wave_configs  # noqa: E402
from front_door_testlib import MATCHERS, SCAN, token_hits  # noqa: E402

#: one tier of the publication scan's token table, matched as the scan matches it (the tokens are never spelled here)
REGION_TOKENS = SCAN.Matcher([r for r in MATCHERS["rows"] if r.tier == "region"])

CELLS = ("none", "prompted__safety_refusal", "prompted__si_constitution")
#: the prompted-only family's FIVE rungs since 2026-09-24: the filed three + the two REFERENCE standups the registry files the
#: steered rungs on (make_padding_configs.py, THE ADDED RUNGS)
PADS = (0, 8192, 16384, 32768, 65536)
PADS_FILED = (0, 16384, 65536)


# --- the generated pair --------------------------------------------------------------------------------------------

def test_generator_check_passes_and_names_its_outputs():
    rc = subprocess.run([sys.executable, str(LANE / "make_padding_configs.py"), "--check"], capture_output=True, text=True, cwd=REPO,
                        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"})
    assert rc.returncode == 0, rc.stdout + rc.stderr
    rc = subprocess.run([sys.executable, str(LANE / "make_padding_configs.py"), "--list"], capture_output=True, text=True, cwd=REPO,
                        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"})
    listed = rc.stdout.split()
    # the prompted-only family FIRST — five configs (the filed three + the two reference standups of 2026-09-24) and its one
    # manifest, in level order; the steered family's follow (its own test below)
    assert listed[:6] == [f"configs/sdfbase/static_padding/sdfbase_full22_qwen35_4b.padstatic_p{p}.yaml" for p in PADS] \
        + ["configs/capeval/manifests/petri_full22_qwen35_4b.padstatic.yaml"]
    # 27 = the hand-listed key's: 5 + 1 prompted-only; 5 + 1 + 1 steered; 2 x (5 + 1 + 1) seed replicates — PLUS, for every front-door key the
    # production roster names (paper-companion-repo; none until stage 05 lands the block), 5 + 1 prompted-only and 5 + 1 per steered phase
    # view the roster carries — roster-conditional, so stage 05 on the real site does not turn this suite red (the fix-1 owner follow-up 10)
    # a phased front-door key whose roster block carries u0 adds its prompted-only 5 + 1 (KEYS); every front-door key adds 5 + 1 per steered
    # phase view the roster carries (families_of; the single-phase demo's three views, 18 outputs, since 05 --step roster landed its block)
    front_door = tuple(k for k in gen.FRONT_DOOR_KEYS if gen._roster_names(k, gen.FRONT_DOOR_SOURCE_VIEW))
    n_front_door = sum(6 * len([f for f in gen.families_of(k)]) for k in gen.FRONT_DOOR_KEYS)
    assert len(listed) == 27 + n_front_door and n_front_door == 18
    assert all((REPO / p).is_file() for p in listed)
    assert gen.PADS == PADS and gen.CELLS == CELLS and gen.KEYS == ("qwen35_4b", *front_door) and gen.FAMILIES == (None, "obj", "obj_s2", "obj_s3")
    assert gen.pads_of(None) == PADS == gen.OBJ_PADS                           # one level list for every family (the reference rule)


@pytest.mark.parametrize("pad", PADS)
def test_each_level_is_the_campaign_config_as_data_except_the_named_keys(pad):
    src = yaml.safe_load((REPO / "configs/sdfbase_full22_qwen35_4b.yaml").read_text())
    cfg = yaml.safe_load(gen.padded_config("qwen35_4b", pad).read_text())
    assert cfg["explicit_cells"] == list(CELLS)
    assert cfg["private_run"] == f"pad_static_qwen35_4b_p{pad}" and cfg["out_dir"] == f"out/pad_static_qwen35_4b_p{pad}"
    assert (cfg["model"], cfg["model_revision"]) == ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    assert cfg["attacks"]["static_max_new_tokens"] == gen.STATIC_CAP == 16384   # the ladder's CEILING at every level: single-pass
    assert "sampled_decode" not in cfg and "target_decode" not in cfg["attacks"]  # the vendor-preset DEFAULT decodes it
    block = cfg["attacks"].get("static_padding")
    if pad == 0:
        assert block is None                                                       # the control: 0 = off, no block
    else:
        assert block == {"tokens": pad, "placement": "before", "mode": "fixed", "reserve": 8192}
        assert "filler_cache" not in block                                          # the argv's, never stated twice
        assert set(block) <= {"tokens", "placement", "mode", "filler_cache", "tolerance", "reserve"}   # the closed key set
    # everything else is the campaign config's data
    want = dict(src)
    want.update(explicit_cells=list(CELLS), private_run=cfg["private_run"], out_dir=cfg["out_dir"])
    want["attacks"] = {**src["attacks"], "static_max_new_tokens": gen.STATIC_CAP}
    if pad:
        want["attacks"]["static_padding"] = block
    assert cfg == want
    assert src["attacks"]["static_max_new_tokens"] == 4096                          # the campaign's cap, which this leg raises
    assert gen.pad_of(gen.padded_config("qwen35_4b", pad)) == pad


def test_the_manifest_is_the_campaign_manifest_as_data_except_the_cells_and_matches_every_level():
    import mt_serving_up as m
    from sisteer.capeval import manifest as cm
    from sisteer.capeval.petri import registry_adapter as ra

    src = yaml.safe_load((REPO / "configs/capeval/manifests/petri_full22_qwen35_4b.yaml").read_text())
    man = gen.padded_manifest("qwen35_4b")
    doc = yaml.safe_load(man.read_text())
    assert doc["petri"]["cells"] == ["none", "prompted__safety", "prompted__constitution"] == [ra.harness_cell_id(c) for c in CELLS]
    want = dict(src); want["petri"] = {**src["petri"], "cells": doc["petri"]["cells"]}
    assert doc == want
    spec = cm.load_petri_spec(man)
    assert (spec.target_model, spec.target_revision) == ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    assert set(spec.prompted_prefix_hashes) == {"safety", "constitution", "irrelevant"}   # the served SIs' hashes ride
    assert "--limit-mm-per-prompt" in spec.serving_launch_flags                            # the text-only pin (common.sh's belt)
    for pad in PADS:
        cfg = yaml.safe_load(gen.padded_config("qwen35_4b", pad).read_text())
        assert m.assert_manifest_matches_config(spec, cfg) == {"none": "none", "prompted__safety": "prompted__safety_refusal",
                                                               "prompted__constitution": "prompted__si_constitution"}


def test_the_pair_is_outside_the_wave_config_enumerator_and_carries_no_lens_sentence():
    names = {p.name for p in wave_configs(REPO)}
    assert not any(".padstatic" in n for n in names)
    assert not list((REPO / "configs").glob("sdfbase_full22_*padstatic*"))      # never at the top level, where the glob is
    for path in [gen.padded_config("qwen35_4b", p) for p in PADS] + [gen.padded_manifest("qwen35_4b")]:
        text = path.read_text()
        assert "GENERATED by scripts/sdfbase/full22/static_padding/" in text
        assert "every lora cell serves via the LENS o_proj site" not in text


def test_generator_refuses_a_hand_edited_output(tmp_path, monkeypatch):
    """A drifted output is named with the remedy; --check never rewrites."""
    monkeypatch.setattr(gen, "CONFIG_DIR", tmp_path / "configs")
    monkeypatch.setattr(gen, "MANIFEST_DIR", tmp_path / "manifests")
    assert gen.main([]) == 0
    p = gen.padded_config("qwen35_4b", 16384)
    p.write_text(p.read_text().replace("tokens: 16384", "tokens: 16000"))
    assert gen.main(["--check"]) == 1
    assert "tokens: 16000" in p.read_text()                                    # --check writes nothing
    with pytest.raises(ValueError):
        gen.pad_of(Path("sdfbase_full22_qwen35_4b.yaml"))


def test_the_cap_is_the_static_ladders_ceiling_and_the_preflight_refuses_any_other():
    """The leg runs ONCE at the ceiling: it is prefill-bound, a rung would re-pay the whole pad prefill, and the rung driver
    refuses a padded base (escalate_static.assert_base_unpadded) — so a cap below the ceiling leaves a >1 % at-cap share
    nowhere to go but a whole-level regeneration. Pinned to the ladder's OWN ceiling, and refused by the preflight by name."""
    sampled = REPO / "scripts" / "sdfbase" / "full22" / "sampled"
    if str(sampled) not in sys.path:
        sys.path.insert(0, str(sampled))
    import ladder_decide as ld
    assert gen.STATIC_CAP == ld.CEILING == ld.LADDERS["static"][-1] == 16384
    assert "def assert_base_unpadded" in (sampled / "escalate_static.py").read_text()   # the guard the reasoning rests on
    for pad in PADS:
        path = gen.padded_config("qwen35_4b", pad)
        text = path.read_text()
        c = pf.contract_of(yaml.safe_load(text), path)
        assert c["cap"] == gen.STATIC_CAP and pf.check_cap(c) == []
        # the generated file carries the WHY beside the key, so it is not "optimised" away by someone who sees 16,384 as waste
        assert "PADDED LEG (docs/static_padding.md): the cap is the static ladder's CEILING (16,384" in text
        assert re.search(r"^  static_max_new_tokens: 16384$", text, re.M) and not re.search(r"^  static_max_new_tokens: 4096$", text, re.M)
    below = pf.check_cap({"level": 65536, "cap": 4096})
    assert len(below) == 1 and "16,384" in below[0] and "ceiling" in below[0] and "prefill-bound" in below[0]
    assert pf.check_cap({"level": 0, "cap": 8192})                                   # the control is held to the same cap
    assert "THE CAP IS THE LADDER'S CEILING" in SBATCH.read_text()
    assert "### The cap is the ladder's ceiling, so the wave runs once" in (REPO / "docs" / "static_padding.md").read_text()


# --- the preflight -------------------------------------------------------------------------------------------------

#: the wave's window (raise the window, never shrink the pad; re-derived on 2026-09-22 for
#: the 16,384 ceiling cap — the 4,096-cap sizing's 81,920 is 7,723 short at it) and the MEASURED floor — the longest rendered
#: [constitution SI + payload] over EVERY variant of every goal of all four families on the pinned Qwen3.5-4B tokenizer
#: (grok420, cluster B 2026-09-22; the char-longest proxy under-read it at 6,403)
WINDOW = 98304
OLD_WINDOW = 81920           # the 4,096-cap sizing, kept as the pin of WHY the window moved
MEASURED_PROMPT_MAX = 6412


def test_preflight_arithmetic_names_the_constitution_shortfall_and_the_decided_window():
    assert pf.PG19_FILLER_SHA256.startswith("7351d518dc71c53b") and len(pf.PG19_FILLER_SHA256) == 64   # the sha256 the doc records
    assert pf.PG19_FILLER_BYTES == 2_881_719
    assert pf.pad_worst_case(65536, 0.02) == 65536 + 1311 and pf.pad_worst_case(16384, 0.02) == 16384 + 328 and pf.pad_worst_case(0, 0.02) == 0
    need = pf.rendered_need(MEASURED_PROMPT_MAX, pad_tokens=65536, tolerance=0.02, cap=gen.STATIC_CAP)
    assert need == 6412 + 66847 + 16384 == 89643 > OLD_WINDOW                     # the 4,096-cap sizing is 7,723 short at the ceiling cap
    rule = 65536 + gen.STATIC_CAP + gen.RESERVE
    assert rule == 90112 == pf.align_up(need, 1024) == pf.align_up(rule, 1024)      # the harness rule IS the smallest aligned window
    assert pf.align_up(1, 1024) == 1024
    assert need <= WINDOW and WINDOW - need == 8661 and WINDOW - rule == 8192 == gen.RESERVE   # one reserve of headroom over the rule
    assert pf.rendered_need(MEASURED_PROMPT_MAX, pad_tokens=16384, tolerance=0.02, cap=gen.STATIC_CAP) == 39508 < 16384 + gen.STATIC_CAP + gen.RESERVE == 40960 < WINDOW
    assert pf.rendered_need(MEASURED_PROMPT_MAX, pad_tokens=65536, tolerance=0.02, cap=4096) == 77355 <= OLD_WINDOW   # what 81,920 was sized for
    assert gen.RESERVE >= MEASURED_PROMPT_MAX + 1311                               # the pinned reserve stays honest for the floor
    with pytest.raises(ValueError):
        pf.align_up(5, 0)


@pytest.mark.parametrize("pad", PADS)
def test_preflight_contract_reader_agrees_with_the_harness_rule(pad):
    from sisteer.eval.attacks import static_padding as sp
    path = gen.padded_config("qwen35_4b", pad)
    cfg = yaml.safe_load(path.read_text())
    c = pf.contract_of(cfg, path)
    assert (c["level"], c["tokens"], c["cap"]) == (pad, pad, gen.STATIC_CAP)
    assert (c["placement"], c["mode"], c["tolerance"]) == ("before", "fixed", sp.DEFAULT_TOLERANCE)
    assert c["reserve"] == (8192 if pad else sp.DEFAULT_RESERVE_TOKENS)
    # the harness's own rule: p65536 needs 90,112 (tokens + cap + reserve) — the 4,096-cap sizing's 81,920 refuses it, the
    # smallest aligned window and the decided window both carry every level (p16384 needs 40,960 and fits all three)
    problems = pf.check_contract(c, OLD_WINDOW)
    assert bool(problems) == (pad == 65536), problems
    assert not pf.check_contract(c, 90112) and not pf.check_contract(c, WINDOW)


def test_preflight_contract_reader_refuses_a_level_that_disagrees_with_its_name(tmp_path):
    cfg = yaml.safe_load(gen.padded_config("qwen35_4b", 16384).read_text())
    bad = tmp_path / "sdfbase_full22_qwen35_4b.padstatic_p65536.yaml"
    with pytest.raises(ValueError, match="pins level 65536"):
        pf.contract_of(cfg, bad)
    cfg0 = yaml.safe_load(gen.padded_config("qwen35_4b", 0).read_text())
    cfg0["attacks"]["static_padding"] = {"tokens": 5}
    with pytest.raises(ValueError, match="CONTROL"):
        pf.contract_of(cfg0, tmp_path / "sdfbase_full22_qwen35_4b.padstatic_p0.yaml")


def test_preflight_refuses_a_wrong_filler_and_a_missing_snapshot_before_any_render(tmp_path):
    filler = tmp_path / "pg19_filler.json"
    filler.write_text('{"text": "some neutral words here"}')
    rec, probs = pf.check_filler(filler, want_sha=pf.PG19_FILLER_SHA256, want_bytes=pf.PG19_FILLER_BYTES)
    assert rec["bytes"] == filler.stat().st_size and len(rec["sha256"]) == 64 and rec["chars"] > 0
    assert any("sha256" in p for p in probs) and any("bytes" in p for p in probs)
    assert pf.check_filler(tmp_path / "absent.json", want_sha=pf.PG19_FILLER_SHA256, want_bytes=None)[1]
    rec, probs = pf.check_filler(filler, want_sha=rec["sha256"], want_bytes=None)
    assert probs == []                                                           # the right bytes pass
    rec, probs = pf.check_model("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", tmp_path / "hf")
    assert probs and "no snapshot" in probs[0] and rec["tokenizer_ok"] is False
    # the CLI: a wrong filler + an empty cache refuses (3) with the render not run; a bad family is a usage error (2)
    out = tmp_path / "pre.json"
    rc = pf.main(["--window", "73728", "--config", str(gen.padded_config("qwen35_4b", 65536)), "--filler", str(filler),
                  "--hf-home", str(tmp_path / "hf"), "--out", str(out)])
    assert rc == pf.EXIT_REFUSED
    import json
    d = json.loads(out.read_text())
    assert d["ok"] is False and "render" not in d and any(p.startswith("filler:") for p in d["problems"]) \
        and any(p.startswith("model:") for p in d["problems"]) and any(p.startswith("contract p65536") for p in d["problems"])
    assert pf.main(["--window", "73728", "--config", str(gen.padded_config("qwen35_4b", 0)), "--hf-home", str(tmp_path / "hf"),
                    "--families", "bon"]) == pf.EXIT_USAGE


# --- the sbatch ----------------------------------------------------------------------------------------------------

def test_sbatch_parses_and_carries_the_cluster_rules():
    rc = subprocess.run(["bash", "-n", str(SBATCH)], capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr
    src = SBATCH.read_text()
    for line in src.splitlines():
        if "--export=" in line:
            assert "never" in line.lower(), line                              # only the rule's own statement names the flag
        if re.search(r"(?<![\w$\"{])CUDA_VISIBLE_DEVICES=", line):
            pytest.fail(f"assigns CUDA_VISIBLE_DEVICES: {line}")
        assert "scancel" not in line.lower()
    assert "--mem=0" not in src
    for m in ("#SBATCH --partition=general", "#SBATCH --qos=high", "#SBATCH --gres=gpu:1", "#SBATCH --cpus-per-task=32",
              "#SBATCH --mem=200G", "#SBATCH --time=04:00:00", "#SBATCH --output=/workspace/%u/exp/logs/%x_%j.out",
              "#SBATCH --open-mode=append", "#SBATCH --no-requeue", "#SBATCH --signal=B:TERM@120"):
        assert m in src, m
    mem = [l for l in src.splitlines() if l.startswith("#SBATCH --mem=")][0]
    assert int(mem.split("=")[1].rstrip("G")) <= 250
    assert "positional only" in src and "never --export=" in src
    assert '|| source "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$WAVE_SHA/scripts/sdfbase/full22/common.sh"' in src
    assert 'source "/workspace' not in src
    assert 'KEY="${1:?key}"; WAVE_SHA="${2:?wave sha}"; WINDOW="${3:?window (the standup max_model_len)}"' in src
    # ONE LEVEL PER JOB: the pad positional is required and a comma list is refused by name — the
    # refusal and the registration block are pinned by tests/test_padding_registration.py
    assert 'PADS="${4:?pad (ONE level: 0 | 8192 | 16384 | 32768 | 65536; one job per level, a comma list is refused)}"; ' \
           'FAMILIES="${5:-pasttense,codeattack,renellm,grok420}"; NSHARDS="${6:-3}"; CELLS_ARG="${7:-}"' in src
    # the cleanup trap: the script's status first, the signal traps DISARMED before any child is signalled, the RESULT line
    assert "trap cleanup EXIT SIGTERM SIGINT SIGQUIT" in src and "f22_result" in src
    body = src[src.index("cleanup(){"):src.index("trap cleanup EXIT")]
    assert body.index("local ec=$?") < body.index("trap '' SIGTERM SIGINT SIGQUIT") < body.index("kill -TERM")
    assert 'TARGET_GPU="$CUDA_VISIBLE_DEVICES"' in src and "f22_stop_requested" in src
    assert "source $SHARED/scripts/hf_env.sh" not in src                    # f22_init sources it (and HF_HUB_OFFLINE=1)
    assert "srun " not in src.replace("srun-WRAPPED", "").replace("in srun", "")   # the lineage's shape, stated in the header


def test_sbatch_declares_its_cell_source_runs_the_preflight_and_stands_up_once_at_the_window():
    src = SBATCH.read_text()
    i_decl = src.index('F22_LAUNCH_MANIFEST="$MANIFEST"')
    i_init = src.index('f22_init "$WAVE_SHA"')
    i_venv = src.index("f22_driver_venv; f22_serving_venv")
    i_view = src.index("pad_bank_view\n")
    i_pre = src.index('"$MPY" "$PAD_DIR/padding_preflight.py" --window "$WINDOW"')
    i_up = src.index('f22_standup attacks "$SERVE_DIR/static" "" families_served:short "$WINDOW" || { RC=1; exit 1; }')
    i_drv = src.index("scripts/jailbreak_run.py generate")
    i_stop = src.index("f22_standup_stop")
    assert i_decl < i_init < i_venv < i_view < i_pre < i_up < i_drv < i_stop
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))              # the header names what it avoids
    assert code.count("f22_standup attacks") == 1 and code.count("f22_standup_stop") == 1      # ONE standup for every level
    assert '"" families_served:short "$WINDOW"' in code and "--max-num-seqs" not in code         # S derived, never forced
    assert "f22_bank_view " not in code and "f22_bank_view_check" not in code                     # the scratch prompted-only view
    assert 'make_bank_view.py --target "$KEY" --view "$VIEW" --roster "$roster" --bank "$bank"' in src
    assert 'make_bank_view.py --check --target "$KEY" --view "$VIEW" --roster "$roster" --bank "$bank"' in src
    assert 'export TT_TWO_SI_DEFENSES_DIR="$bank/cells_view/$VIEW"' in src
    assert 'FILLER="${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}/data/pg19_filler.json"' in src and "--filler \"$FILLER\"" in src
    assert "f22_positive_control" not in code                                                    # no steered cell to probe
    assert "F22_LOAD_CHECK_SYSTEM_SI" in code and "f22_standup_extra_reset --enable-log-requests" in code
    assert "f22_served_cells_filter CELLS \"$CELLS_ARG\"" in code
    assert 'PRIVATE_ROOT="$WAVE_ROOT/static_padding/transcripts_results"' in code                  # a lane root, never the campaign's
    # registered PER LEVEL (tests/test_padding_registration.py pins the block): the evals and the level block are
    # exported before the ONE f22_registry_register call, which precedes the standup
    assert 'F22_REGISTRY_EVALS="${FAMILIES//,/ }"; export F22_REGISTRY_EVALS' in code
    assert code.index("f22_registry_register || true") < code.index("f22_standup attacks")
    assert "DRIVER_CAP=96" in code and "MAX_DRIVERS=$(( TARGET_S < DRIVER_CAP ? TARGET_S : DRIVER_CAP ))" in code
    for forbidden in ("--greedy", "--target-greedy", "--static-padding-tokens", "--static-padding-placement", "--static-padding-mode"):
        assert forbidden not in code, forbidden
    # the preflight derives the geometry for the config's model key and the standup's utilization
    assert '--window "$WINDOW" --model-key "$KEY"' in code and '${GPU_MEM_UTIL:+--gpu-mem-util "$GPU_MEM_UTIL"}' in code
    # THE GEOMETRY PIN: after the standup and before any driver, the AUTHORITATIVE record must show the native engine, the
    # zero adapter, and an attested S equal to the kernel-ceiling S the geometry module derives at the window (the S a
    # steered standup resolves) — never an explicit serving path, never a literal S
    i_pin = src.index('"$MPY" - "$SERVE_DIR/static/endpoint.json" "$KEY" "$WINDOW" "$TARGET_S"')
    assert i_up < i_pin < i_drv
    pin = src[i_pin:i_drv]
    for needle in ("from sisteer.capeval.petri.defense_cells import NOOP_LORA_CELL_ID", "from sisteer.capeval.serving import geometry as g",
                   "rb.entry.seqs_at(L)", "g.kernel_ceiling_adjustment(key, L, identity, module_set=SET)",   # the SELECTED set's D
                   'ep.get("engine") != "lora_native"', "NOOP_LORA_CELL_ID not in names", "if S != expect:",
                   '(ep.get("geometry") or {}).get("adjustments")', "REFUSING TRAFFIC: the standup's geometry is not the steered arms' geometry"):
        assert needle in pin, needle
    assert "F22_SERVING_PATH=" not in code                                                          # the kind-aware default selects native; the pin verifies it
    assert not re.search(r"(?<![\w.])none__lora0", code)                                             # the zero adapter's id comes from defense_cells, never a literal
    assert "98,304" in src and "89,643" in src and "90,112" in src and "6,412" in src and "32 KiB" in src and "128 KiB PER CONTEXT TOKEN ON THIS MODEL, NOT 128" not in src
    assert "#SBATCH --time=04:00:00" in src and "PREFILL-bound" in src and "the leg is DECODE-bound" not in src   # the cost model the ceiling cap rests on
    assert "lora_kernel_index_ceiling" in src and "none__lora0" in src                              # the header names WHY the geometry is pinned


def _render_driver_argv(pads: str, extra_vars: str = "") -> list[list[str]]:
    """Run the sbatch's driver loop under bash with stubs: every f22_queue_launch argv, one list per driver."""
    src = SBATCH.read_text()
    loop = src[src.index("# --- drivers: begin"):src.index("# --- drivers: end")]
    stubs = ('f22_queue_launch(){ shift; for a in "$@"; do printf "%s\\n" "$a"; done; printf "@@\\n"; }; '
             'f22_stop_requested(){ return 1; }; f22_queue_wait(){ NFAIL=0; }; f22_log(){ :; }; ')
    vars_ = (f'set -u; IFS="," read -ra PAD_LIST <<< "{pads}"; declare -A CFG_BY_PAD=(); '
             'for p in "${PAD_LIST[@]}"; do CFG_BY_PAD[$p]="configs/sdfbase/static_padding/sdfbase_full22_qwen35_4b.padstatic_p${p}.yaml"; done; '
             'FAMS=(pasttense grok420); CELLS=(none prompted__si_constitution); NSHARDS=1; MPY=/venv/bin/python; '
             'PRIVATE_ROOT=/wave/static_padding/transcripts_results; PROXY_URL=http://127.0.0.1:4321/v1; TABLE=/wave/serving/routing_table.json; '
             f'TARGET_L={WINDOW}; FILLER=/scratch/proj/data/pg19_filler.json; MAX_DRIVERS=6; declare -a DRIVER_PIDS=(); TOTAL_FAIL=0; '
             'sleep(){ :; }; ' + extra_vars)
    rc = subprocess.run(["bash", "-c", vars_ + stubs + loop], capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr
    return [blk.split("\n") for blk in rc.stdout.strip().split("\n@@") if blk.strip()]


def test_rendered_driver_argv_carries_the_level_in_the_config_and_the_filler_only_when_padded():
    argvs = _render_driver_argv("0,16384")
    assert len(argvs) == 2 * 2 * 2                                               # 2 levels x 2 families x 2 cells x 1 shard
    for argv in argvs:
        argv = [a for a in argv if a]
        cfg = argv[argv.index("--config") + 1]
        pad = int(re.search(r"padstatic_p(\d+)\.yaml$", cfg).group(1))
        assert argv[:3] == ["/venv/bin/python", "scripts/jailbreak_run.py", "generate"]
        assert "--confirmatory" in argv and argv[argv.index("--target-engine") + 1] == "served"
        assert argv[argv.index("--target-max-model-len") + 1] == str(WINDOW)
        assert argv[argv.index("--private-root") + 1] == "/wave/static_padding/transcripts_results"
        assert "--static-padding-tokens" not in argv and "--greedy" not in argv and "--target-greedy" not in argv
        if pad == 0:
            assert not any(a.startswith("--static-padding") for a in argv)      # the control: no padding flag at all
        else:
            assert argv[argv.index("--static-padding-filler") + 1] == "/scratch/proj/data/pg19_filler.json"
            assert sum(a.startswith("--static-padding") for a in argv) == 1    # the filler ONLY; the level is the config's
    # levels run in the order given: every p0 driver before every p16384 driver
    order = [int(re.search(r"padstatic_p(\d+)", " ".join(a)).group(1)) for a in argvs]
    assert order == sorted(order)


# --- the geometry the wave stands up on -----------------------------------------------------------------------------

def test_geometry_expectation_is_the_launchers_own_arithmetic_and_the_steered_arms_count():
    """At the decided window the identity S (the model's measured budget // L) is 7 and the LoRA kernels' int32 index
    ceiling — which the native standup applies to EVERY launch because it always registers the zero adapter — lowers it
    to 5, recorded as the same adjustment a steered standup at this window records. Pure arithmetic through the geometry
    module, so a moved budget or a moved pinned module width moves this test, never a literal in the sbatch."""
    from sisteer.capeval.serving import geometry as g, serving_attestation as sa
    geo = pf.geometry_expectation("qwen35_4b", WINDOW)
    mb = g.BUDGET_BY_MODEL["qwen35_4b"]
    assert geo["identity_S"] == mb.seqs_at(WINDOW) == mb.batched_token_budget // WINDOW == 7
    assert geo["lora_max_module_dim"] == g.LORA_MAX_MODULE_DIM_BY_MODEL["qwen35_4b"] == 4096
    ceiling = sa.lora_kernel_index_ceiling(4096)
    assert geo["kernel_adjustment"] == {"kind": "lora_kernel_index_ceiling", "from_S": 7, "to_S": ceiling // WINDOW, "D": 4096, "ceiling": ceiling}
    assert geo["served_S"] == ceiling // WINDOW == 5 and geo["batched_tokens"] == 5 * WINDOW == 491520
    # the SAME count a steered standup resolves: resolve_geometry sizes the identity S the way f22_standup does, and the
    # kernel step is the model's pinned width, not the cell set's
    r = g.resolve_geometry("families_served:short", max_model_len=WINDOW, model_key="qwen35_4b")
    assert (r.max_num_seqs, r.max_model_len) == (7, WINDOW)
    assert pf.geometry_expectation("qwen35_4b", 90112)["served_S"] == 5                # the smallest aligned window buys the same S
    assert pf.geometry_expectation("qwen35_4b", 104448)["served_S"] == 5               # ... as does every window up to the ceiling's next step
    assert pf.geometry_expectation("qwen35_4b", 106496)["served_S"] == 4               # where the headroom would stop being free
    assert pf.geometry_expectation("qwen35_4b", OLD_WINDOW)["served_S"] == 6            # the 4,096-cap sizing: one more row, the same 491,520 tokens
    assert pf.geometry_expectation("qwen35_4b", OLD_WINDOW)["batched_tokens"] == pf.geometry_expectation("qwen35_4b", WINDOW)["batched_tokens"]
    with pytest.raises(KeyError):
        pf.geometry_expectation("no_such_model", WINDOW)


def test_kv_bytes_per_token_reads_the_layer_roster_not_the_layer_count():
    """Qwen3.5-4B's config: 32 layers of which 8 are full attention and 24 gated deltanet -> 32 KiB per context token, a
    QUARTER of the 128 KiB a num_hidden_layers count gives (the mistake the first sizing note made)."""
    hybrid = {"num_hidden_layers": 32, "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256, "hidden_size": 2560,
              "intermediate_size": 9216, "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 8}
    kv = pf.kv_bytes_per_token(hybrid)
    assert (kv["kv_layers"], kv["recurrent_layers"], kv["bytes_per_token"]) == (8, 24, 32 * 1024)
    assert kv["bytes_per_token"] * 4 == 2 * 32 * 4 * 256 * 2                            # the dense count is exactly 4x
    dense = {k: v for k, v in hybrid.items() if k != "layer_types"}
    assert pf.kv_bytes_per_token(dense)["bytes_per_token"] == 128 * 1024 and pf.kv_bytes_per_token(dense)["layer_types_present"] is False
    nested = {"text_config": hybrid, "model_type": "qwen3_5"}
    assert pf.kv_bytes_per_token(nested)["bytes_per_token"] == 32 * 1024                 # a VLM wrapper's text_config is read
    with pytest.raises(ValueError):
        pf.kv_bytes_per_token({**hybrid, "layer_types": ["full_attention"] * 31})


def test_check_geometry_sums_weights_kv_and_the_warmup_intermediate(tmp_path):
    import json
    snap = tmp_path / "snap"; snap.mkdir()
    cfg = {"num_hidden_layers": 32, "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256, "hidden_size": 2560,
           "intermediate_size": 9216, "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 8}
    (snap / "config.json").write_text(json.dumps(cfg))
    (snap / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 9_319_737_856}, "weight_map": {}}))
    rec, probs = pf.check_geometry("qwen35_4b", WINDOW, snap)
    assert probs == [] and rec["served_S"] == 5 and rec["kv"]["bytes_per_token"] == 32768
    mem = rec["memory"]
    assert abs(mem["weights_gib"] - 8.68) < 0.01 and abs(mem["kv_gib"] - 15.0) < 0.01 and abs(mem["mlp_intermediate_gib"] - 8.44) < 0.01
    assert abs(mem["total_gib"] - 32.1) < 0.1 and abs(mem["reserved_gib"] - 0.9 * 139.81) < 1e-6
    # a card too small for the sum refuses, naming the numbers; a snapshot without config.json records "not computed"
    rec2, probs2 = pf.check_geometry("qwen35_4b", WINDOW, snap, util=0.2)
    assert probs2 and "geometry:" in probs2[0] and "32768 B/token" in probs2[0]
    rec3, probs3 = pf.check_geometry("qwen35_4b", WINDOW, tmp_path / "absent")
    assert probs3 == [] and "not computed" in rec3["memory"]["note"] and rec3["served_S"] == 5
    # a window the ceiling admits no row at refuses through the geometry module's own ValueError
    rec4, probs4 = pf.check_geometry("qwen35_4b", 600_000, snap)
    assert probs4 and "no max_num_seqs" in probs4[0]


def test_token_max_renders_every_variant_and_beats_the_char_longest_proxy():
    """The floor-setting number is the max IN TOKENS over every variant: a shorter string can render to more tokens."""
    variants = [("g1", "a", "x" * 100), ("g2", "b", "yz yz yz yz yz"), ("g3", "c", "q" * 40)]
    def count(system, user):                         # a counter that charges 1 per char for x/q, 2 per char for the yz text
        return (len(system or "") + (2 * len(user) if "yz" in user else len(user) // 4))
    n, gid, name = pf.token_max(count, None, variants)
    assert (n, gid, name) == (28, "g2", "b")                                              # not g1, the char-longest
    assert pf.token_max(count, "S" * 10, variants)[0] == 38
    assert pf.token_max(count, None, []) == (0, None, None)
    cfg = {"explicit_cells": ["none", "prompted__safety_refusal"]}
    contract = {"level": 65536, "tokens": 65536, "tolerance": 0.02, "cap": 16384}
    rows = pf.render_rows(tok=None, cfg=cfg, contract=contract, window=WINDOW, variants_by_family={"grok420": variants},
                          families=("grok420",), count=lambda s, u: (len(s or "") // 50) + count(None, u))
    assert [(r["cell"], r["goal_id"], r["variant"], r["n_variants"]) for r in rows] == [("none", "g2", "b", 3), ("prompted__safety_refusal", "g2", "b", 3)]
    assert all(r["need"] == r["prompt_tokens_rendered"] + 66847 + 16384 and r["fits"] for r in rows)
    assert rows[1]["system_tokens_rendered"] > 0 and rows[1]["prompt_tokens_rendered"] >= rows[0]["prompt_tokens_rendered"]


def test_generator_names_carry_the_model_key_and_the_preflight_refuses_a_mismatched_key(tmp_path):
    assert gen.key_of(gen.padded_config("qwen35_4b", 65536)) == "qwen35_4b" and gen.pad_of(gen.padded_config("qwen35_4b", 65536)) == 65536
    with pytest.raises(ValueError):
        gen.key_of(Path("configs/sdfbase_full22_qwen35_4b.yaml"))
    out = tmp_path / "pre.json"
    rc = pf.main(["--window", str(WINDOW), "--model-key", "gemma4e4b", "--config", str(gen.padded_config("qwen35_4b", 0)),
                  "--hf-home", str(tmp_path / "hf"), "--skip-render", "--out", str(out)])
    assert rc == pf.EXIT_REFUSED
    import json
    d = json.loads(out.read_text())
    assert any("--model-key gemma4e4b" in p for p in d["problems"]) and d["geometry"]["model_key"] == "gemma4e4b"
    # the default key is the config's own; its geometry rides the record even when the snapshot is absent
    rc = pf.main(["--window", str(WINDOW), "--config", str(gen.padded_config("qwen35_4b", 0)), "--hf-home", str(tmp_path / "hf"),
                  "--skip-render", "--out", str(out)])
    assert rc == pf.EXIT_REFUSED                                                            # the missing snapshot still refuses
    d = json.loads(out.read_text())
    assert d["geometry"]["model_key"] == "qwen35_4b" and d["geometry"]["served_S"] == 5 and d["geometry"]["identity_S"] == 7
    assert "not computed" in d["geometry"]["memory"]["note"] and not any(p.startswith("geometry:") for p in d["problems"])


# --- the STEERED family (2026-09-23): ten cells, the padobj view, the cluster A sbatch ------------------------------------

OBJ_SBATCH = LANE / "target_static_padded_obj.sbatch"
#: the steered family's five levels: the prompted-only three plus the two ADDED RUNGS (2026-09-24), one comparable series
OBJ_PADS = (0, 8192, 16384, 32768, 65536)
OBJ_PADS_ADDED = (8192, 32768)
#: sha256 of the six FILED configs (the results registry holds their stores, 2026-09-23): a change to any is a regression
FILED_SHA256 = {
    "sdfbase_full22_qwen35_4b.padstatic_p0.yaml": "bab37897acdfe6d6397ba500df5566b08dcbaace2719171ce5894e8bacc7565a",
    "sdfbase_full22_qwen35_4b.padstatic_p16384.yaml": "8195c24c677d5bd1a75b4f72bb378a8775c168f3b7dadc10437c362a0c943a5f",
    "sdfbase_full22_qwen35_4b.padstatic_p65536.yaml": "7a4e2d848a4a8261657e6ee09c59c6bab92be45168abffefa6eaea1587774f5b",
    "sdfbase_full22_qwen35_4b.padstatic_obj_p0.yaml": "5fda042105b8347f54812d3eabef0cf6da1fde16834a5a2d15d892db31473f7f",
    "sdfbase_full22_qwen35_4b.padstatic_obj_p16384.yaml": "43c2e4fe1a2fa8adf98e99be9a116749a7a16216351aeca7c854003fbdc1a25f",
    "sdfbase_full22_qwen35_4b.padstatic_obj_p65536.yaml": "4778ebece38969508ce9e43db1e374c2d03eaff0f29550d8e0ce9d0841a8c244",
}
OBJ_CELLS = ("none", "prompted__safety_refusal", "prompted__si_constitution",
             "constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce", "constv2ctl2-full__psr_ce_all",
             "constv2ctl2-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl",
             "arditi_allpos", "constv2-mix-full__psr_ce_all_minus_ctl")
OBJ_HARNESS = ["none", "prompted__safety", "prompted__constitution",
               "si_constv2_mix_full__psr_ce_all", "si_constv2_mix_full__lora_r4_ce", "si_constv2ctl2_full__psr_ce_all",
               "si_constv2ctl2_full__lora_r4_ce", "si_constv2_mix_full__lora_r4_ce_minus_ctl",
               "arditi_allpos", "si_constv2_mix_full__psr_ce_all_minus_ctl"]
PAIRED_SHA = "f98f0349673b7cb0cf85f54aee71f74bf9f0bbf981e989b52c8ba6599a492c2d"
ARDITI_SHA = "ba2212f0c2f9ffef2997bdd24831182052560cf980054bcee9cf95ea0db085b2"


def test_generator_lists_the_steered_family_after_the_prompted_one_and_its_names_parse():
    rc = subprocess.run([sys.executable, str(LANE / "make_padding_configs.py"), "--list"], capture_output=True, text=True, cwd=REPO,
                        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"})
    listed = rc.stdout.split()
    assert listed[6:13] == [f"configs/sdfbase/static_padding/sdfbase_full22_qwen35_4b.padstatic_obj_p{p}.yaml" for p in OBJ_PADS] \
        + ["configs/capeval/manifests/petri_full22_qwen35_4b.padstatic_obj.yaml",
           "configs/sdfbase/static_padding/sdfbase_full22_qwen35_4b.padobj_roster.yaml"]
    # the SEED REPLICATES (2026-09-24) follow, one block per seed, in the same shape (their own test below) — the hand-listed key's 27
    assert listed[13:27] == [x for n in (2, 3) for x in
                             [f"configs/sdfbase/static_padding/sdfbase_full22_qwen35_4b.padstatic_obj_s{n}_p{p}.yaml" for p in OBJ_PADS]
                             + [f"configs/capeval/manifests/petri_full22_qwen35_4b.padstatic_obj_s{n}.yaml",
                                f"configs/sdfbase/static_padding/sdfbase_full22_qwen35_4b.padobj_s{n}_roster.yaml"]]
    # ... then, per FRONT-DOOR key the production roster names (the worked example's three views, since 05 --step roster landed its
    # block), its prompted-only five + manifest when it declares u0 and per steered phase view its five + manifest — no lane roster
    tail = []
    for k in gen.FRONT_DOOR_KEYS:
        for fam in gen.families_of(k):
            tail += [f"configs/sdfbase/static_padding/sdfbase_full22_{k}.padstatic{gen.seg_of(fam)}_p{p}.yaml" for p in gen.pads_of(fam)]
            tail.append(f"configs/capeval/manifests/petri_full22_{k}.padstatic{gen.seg_of(fam)}.yaml")
    assert listed[27:] == tail and len(tail) == 18
    assert all((REPO / p).is_file() for p in listed)
    assert gen.OBJ_CELLS == OBJ_CELLS and gen.OBJ_FAMILY == "obj" and gen.OBJ_VIEW == "padobj"
    assert gen.OBJ_PADS == OBJ_PADS == tuple(sorted(PADS_FILED + OBJ_PADS_ADDED)) == PADS and gen.OBJ_PADS_ADDED == OBJ_PADS_ADDED
    assert gen.pads_of("obj") == OBJ_PADS and gen.pads_of(None) == PADS         # the added rungs are every family's (the reference rule)
    for pad in OBJ_PADS:
        p = gen.padded_config("qwen35_4b", pad, "obj")
        assert (gen.key_of(p), gen.pad_of(p), gen.family_of(p)) == ("qwen35_4b", pad, "obj")
        assert gen.family_of(gen.padded_config("qwen35_4b", pad)) is None                    # the prompted-only names are unchanged
        assert gen.private_run("qwen35_4b", pad, "obj") == f"pad_static_qwen35_4b_obj_p{pad}" != gen.private_run("qwen35_4b", pad)
    with pytest.raises(ValueError, match="does not know"):
        gen.pad_of(Path("sdfbase_full22_qwen35_4b.padstatic_zzz_p0.yaml"))
    with pytest.raises(ValueError):
        gen.padded_config("qwen35_4b", 0, "zzz")
    # the steered family lives OUTSIDE the wave-config enumerator's glob too, roster included
    names = {p.name for p in wave_configs(REPO)}
    assert not any("padobj" in n or "padstatic" in n for n in names)
    assert not list((REPO / "configs").glob("sdfbase_full22_*pad*"))


@pytest.mark.parametrize("pad", OBJ_PADS)
def test_each_steered_level_is_the_campaign_config_as_data_except_the_named_keys(pad):
    src = yaml.safe_load((REPO / "configs/sdfbase_full22_qwen35_4b.yaml").read_text())
    path = gen.padded_config("qwen35_4b", pad, "obj")
    cfg = yaml.safe_load(path.read_text())
    assert cfg["explicit_cells"] == list(OBJ_CELLS)
    assert cfg["defenses_dir"] == "out/full22/qwen35_4b/cells_view/padobj"
    assert cfg["private_run"] == f"pad_static_qwen35_4b_obj_p{pad}" and cfg["out_dir"] == f"out/pad_static_qwen35_4b_obj_p{pad}"
    assert cfg["attacks"]["static_max_new_tokens"] == gen.STATIC_CAP == 16384
    block = cfg["attacks"].get("static_padding")
    assert block == (None if pad == 0 else {"tokens": pad, "placement": "before", "mode": "fixed", "reserve": 8192})
    want = dict(src)
    want.update(explicit_cells=list(OBJ_CELLS), private_run=cfg["private_run"], out_dir=cfg["out_dir"],
                defenses_dir="out/full22/qwen35_4b/cells_view/padobj")
    want["attacks"] = {**src["attacks"], "static_max_new_tokens": gen.STATIC_CAP}
    if pad:
        want["attacks"]["static_padding"] = block
    assert cfg == want
    # the preflight reads the same contract off it, at the same window rule, and refuses a cap below the ceiling
    c = pf.contract_of(cfg, path)
    assert (c["level"], c["tokens"], c["cap"]) == (pad, pad, gen.STATIC_CAP) and pf.check_cap(c) == []
    assert not pf.check_contract(c, WINDOW)
    text = path.read_text()
    assert "STEERED family" in text and "target_static_padded_obj.sbatch" in text and "GENERATED by scripts/sdfbase/full22/static_padding/" in text
    # the ADDED RUNGS carry one extra header paragraph naming the sharded job shape; the filed three carry no such text
    assert ("# ADDED RUNG (2026-09-24)" in text) == (pad in OBJ_PADS_ADDED)
    if pad in OBJ_PADS_ADDED:
        assert "<lo>-<hi>/N" in text and "shard00..shard<N-1>" in text and "98,304" in text and "S = 5" in text


def test_the_filed_configs_are_byte_identical_to_what_the_registry_holds():
    """Adding the steered family's two rungs may not move a byte of the six FILED configs: their stores are graded and filed
    in the results registry, and a config change would be a regression against it. Pinned by sha256, not by --check alone
    (--check pins the files to the generator; this pins the generator's output to the filed bytes, or to their certified
    rewrite in configs/SCRUB_MAP.tsv)."""
    import hashlib
    from sisteer import scrub_map
    for name, want in FILED_SHA256.items():
        got = hashlib.sha256((CONFIG_DIR / name).read_bytes()).hexdigest()
        # the filed sha stays as the registry filed it; the public release's rule rewrite of the file is certified by
        # configs/SCRUB_MAP.tsv (the filed sha -> today's), and anything else is a changed filed config
        verdict = scrub_map.check(want, got, path=f"configs/sdfbase/static_padding/{name}",
                                  map_path=REPO / "configs" / "SCRUB_MAP.tsv")
        assert verdict in (scrub_map.EXACT, scrub_map.VIA_MAP), (
            f"{name}: sha256 {got[:12]} != the filed {want[:12]}, and no scrub-map row certifies it — a filed config changed")
    for pad in OBJ_PADS_ADDED:
        assert gen.padded_config("qwen35_4b", pad, "obj").name not in FILED_SHA256          # the added rungs are not filed
        assert gen.padded_config("qwen35_4b", pad).name not in FILED_SHA256                 # nor their prompted-only REFERENCE configs
        assert gen.padded_config("qwen35_4b", pad).is_file()                                # which exist since 2026-09-24 (the reference rule)
    for pad in PADS_FILED:
        assert (gen.padded_config("qwen35_4b", pad).name in FILED_SHA256
                and gen.padded_config("qwen35_4b", pad, "obj").name in FILED_SHA256)
    assert set(FILED_SHA256) == {gen.padded_config("qwen35_4b", p, f).name for p in PADS_FILED for f in (None, "obj")}


def test_preflight_arithmetic_carries_the_added_rungs_at_the_same_window():
    """The window stays 98,304 for the five-rung series: at 32,768 the measured floor is
    6,412 + 33,424 (the pad at its 2 % worst case: ceil(655.36) = 656 over 32,768) + 16,384 = 56,220 and the harness rule
    32,768 + 16,384 + 8,192 = 57,344; at 8,192, 31,152 and 32,768. Both far inside the window, at the same derived S = 5.
    (The render itself ran on the pinned tokenizer over every grok420 variant of every goal, 2026-09-24: the same 6,412.)"""
    assert pf.pad_worst_case(32768, 0.02) == 32768 + 656 == 33424 and pf.pad_worst_case(8192, 0.02) == 8192 + 164 == 8356
    need32 = pf.rendered_need(MEASURED_PROMPT_MAX, pad_tokens=32768, tolerance=0.02, cap=gen.STATIC_CAP)
    need8 = pf.rendered_need(MEASURED_PROMPT_MAX, pad_tokens=8192, tolerance=0.02, cap=gen.STATIC_CAP)
    assert need32 == 6412 + 33424 + 16384 == 56220 and need8 == 6412 + 8356 + 16384 == 31152
    assert 32768 + gen.STATIC_CAP + gen.RESERVE == 57344 and 8192 + gen.STATIC_CAP + gen.RESERVE == 32768
    assert need32 < 57344 < WINDOW and need8 < 32768 < WINDOW
    for pad in OBJ_PADS_ADDED:
        path = gen.padded_config("qwen35_4b", pad, "obj")
        c = pf.contract_of(yaml.safe_load(path.read_text()), path)
        assert (c["level"], c["tokens"], c["cap"], c["reserve"]) == (pad, pad, gen.STATIC_CAP, gen.RESERVE)
        assert pf.check_contract(c, WINDOW) == [] and pf.check_cap(c) == []
        assert pf.geometry_expectation("qwen35_4b", WINDOW)["served_S"] == 5                   # one window, one width, five rungs


def test_the_steered_manifest_names_the_ten_harness_ids_and_selects_the_native_path():
    import mt_serving_up as m
    from sisteer.capeval import manifest as cm
    from sisteer.capeval.petri import registry_adapter as ra

    src = yaml.safe_load((REPO / "configs/capeval/manifests/petri_full22_qwen35_4b.yaml").read_text())
    man = gen.padded_manifest("qwen35_4b", "obj")
    doc = yaml.safe_load(man.read_text())
    assert doc["petri"]["cells"] == OBJ_HARNESS == [ra.harness_cell_id(c) for c in OBJ_CELLS]
    want = dict(src); want["petri"] = {**src["petri"], "cells": OBJ_HARNESS}
    assert doc == want
    spec = cm.load_petri_spec(man)
    for pad in OBJ_PADS:
        cfg = yaml.safe_load(gen.padded_config("qwen35_4b", pad, "obj").read_text())
        assert list(m.assert_manifest_matches_config(spec, cfg)) == OBJ_HARNESS
    # the kind-aware serving-path default: seven natively servable objects, three passthroughs, NOTHING skipped
    rc = subprocess.run([sys.executable, str(REPO / "scripts/sdfbase/full22/serving_path_select.py"), "--manifest", str(man), "--json"],
                        capture_output=True, text=True, cwd=REPO, env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"})
    assert rc.returncode == 0, rc.stderr
    import json
    sel = json.loads(rc.stdout)
    assert sel["selected"] == "lora_native" and sel["cells_by_disposition"]["skip"] == []
    assert len(sel["cells_by_disposition"]["serve"]) == 7 and len(sel["cells_by_disposition"]["passthrough"]) == 3


def test_the_lane_roster_names_the_padobj_view_through_the_wave_pins_and_the_two_pin_files():
    """The view's mechanism: five same-bank staged ids (the production s1 view's, pinned), two CROSS-BANK PINS (no bytes
    copied), two prompted nulls; a view the production roster does not name (f22_bank_view's rule); the paired pin is
    the SAME bytes the production s1p view stages; the Arditi pin is the c = 0.3 all-positions rung."""
    import json
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))
    import f22_roster as fr
    from sisteer.capeval.petri.registry_adapter import ARM_OBJECT_KINDS

    roster_p = gen.padded_roster("qwen35_4b")
    assert roster_p == REPO / "configs/sdfbase/static_padding/sdfbase_full22_qwen35_4b.padobj_roster.yaml"
    roster = fr.load_roster(roster_p)
    prod = fr.load_roster(REPO / "configs/sdfbase_full22_roster.yaml")
    assert "padobj" not in prod["targets"]["qwen35_4b"]["views"]                     # a lane view, never a production one
    assert (roster["wave"], roster["bank_root"]) == (prod["wave"], prod["bank_root"]) == ("full22", "exp/objects/sdfbase/full22")
    t = roster["targets"]["qwen35_4b"]
    assert (t["model"], t["model_revision"]) == (prod["targets"]["qwen35_4b"]["model"], prod["targets"]["qwen35_4b"]["model_revision"])
    assert list(roster["targets"]) == ["qwen35_4b"] and list(t["views"]) == ["padobj"] and roster["lane_flags"] == {}
    v = t["views"]["padobj"]
    assert v["config"] == "configs/sdfbase/static_padding/sdfbase_full22_qwen35_4b.padstatic_obj_p0.yaml"
    assert v["manifest"] == "configs/capeval/manifests/petri_full22_qwen35_4b.padstatic_obj.yaml"
    served = fr.served_cells(roster, "qwen35_4b", "padobj"); staged = fr.staged_cells(roster, "qwen35_4b", "padobj")
    assert list(served) == list(OBJ_CELLS[1:]) and len(served) == 9 and len(staged) == 7
    assert served["prompted__safety_refusal"] is None and served["prompted__si_constitution"] is None
    # the five s1 cells: the production s1 view's staged ids, byte for byte, every one pinned in the wave pins
    s1 = fr.staged_cells(prod, "qwen35_4b", "s1")
    wave_pins = json.loads((REPO / "configs/sdfbase_full22_pins.json").read_text())["qwen35_4b"]
    for c in gen.OBJ_S1_CELLS:
        assert staged[c] == s1[c] and isinstance(staged[c], str) and wave_pins[staged[c]]["object_sha256"]
    assert staged["constv2-mix-full__psr_ce_all_minus_ctl"] == {"pins": "configs/reseed_paired_pins.json",
                                                               "pin": "qwen35_4b.constv2-mix-full__psr_ce_all_minus_ctl"}
    assert staged["arditi_allpos"] == {"pins": "configs/arditi_allpos_pins.json", "pin": "qwen35_4b.ar_c0p3__arditi_allpos"}
    assert all(fr.is_pin_form(staged[c]) for c in ("arditi_allpos", "constv2-mix-full__psr_ce_all_minus_ctl"))
    assert fr.every_staged_id(roster, "qwen35_4b") == [s1[c] for c in gen.OBJ_S1_CELLS]   # a pin names no staged id of THIS bank
    # the pins the form points at: the paired write (full dose, seed 1, all positions, served_layer 0 — the pin's own) whose
    # bytes are the production s1p view's staged copy; the Arditi rung c = 0.3 (vector_all, served layer 12, positions all)
    paired = json.loads((REPO / "configs/reseed_paired_pins.json").read_text())["qwen35_4b"]["constv2-mix-full__psr_ce_all_minus_ctl"]
    assert paired["object_sha256"] == PAIRED_SHA == wave_pins["paired_eb64s1__constv2-mix-full__psr_ce_all_minus_ctl"]["object_sha256"]
    assert (paired["kind"], paired["positions"], paired["served_layer"], paired["seed"], paired["scale_matched"]) == ("psr_paired_write", "all", 0, 1, True)
    assert paired["derived_id"] == "constv2-mix-full__psr_ce_all_minus_ctl" and paired["source"].startswith("paired_write/qwen35_4b/cells_view/")
    assert paired["staged_id"] == "paired_s1__constv2-mix-full__psr_ce_all_minus_ctl" and paired["identity_sidecar"] == "derived_meta.json"
    ard = json.loads((REPO / "configs/arditi_allpos_pins.json").read_text())["qwen35_4b"]["ar_c0p3__arditi_allpos"]
    assert (ard["object_sha256"], ard["kind"], ard["c"], ard["positions"], ard["served_layer"]) == (ARDITI_SHA, "vector_all", 0.3, "all", 12)
    assert ard["identity_sidecar"] == "fit_provenance.json" and ard["source"] == "arditi_allpos/qwen35_4b/ladder/cells_view/ar_c0p3/arditi_allpos"
    assert ARM_OBJECT_KINDS["arditi_allpos"] == "vector_all" and ARM_OBJECT_KINDS["psr_ce_all_minus_ctl"] == "psr_paired_write"
    # the OUT rulings hold by construction: no half-dose view, no *_distilled object anywhere in the family
    assert not any("distilled" in c or "halfdose" in str(v) for c, v in served.items())
    text = roster_p.read_text()
    assert text.startswith("# sdfbase_full22_qwen35_4b.padobj_roster.yaml") and "GENERATED by scripts/sdfbase/full22/static_padding/" in text


def test_the_generator_refuses_a_roster_whose_pins_disagree(monkeypatch):
    """The roster renderer holds the paired pin's bytes to the wave pins' staged copy and every pin's kind to the arm's."""
    import json
    good = json.loads(gen.WAVE_PINS.read_text())
    bad = json.loads(json.dumps(good)); bad["qwen35_4b"]["paired_eb64s1__constv2-mix-full__psr_ce_all_minus_ctl"]["object_sha256"] = "0" * 64
    monkeypatch.setattr(gen, "_load_json", lambda p: bad if p == gen.WAVE_PINS else json.loads(Path(p).read_text()))
    with pytest.raises(SystemExit, match="two spellings of one object"):
        gen._obj_roster_cells("qwen35_4b")
    monkeypatch.setattr(gen, "OBJ_PIN_REFS", {**gen.OBJ_PIN_REFS, "arditi_allpos": ("configs/arditi_allpos_pins.json", "ar_c1__arditi_allpos")})
    monkeypatch.setattr(gen, "_load_json", lambda p: json.loads(Path(p).read_text()))
    with pytest.raises(SystemExit, match="c = 0.3"):
        gen._obj_roster_cells("qwen35_4b")


def test_preflight_system_text_follows_the_drivers_rule():
    """Only a prompted baseline (or a stacked cell) carries a system message; `none` and every compiled cell — Class A
    `<si>__<arm>` or a bare Class B id — are served object-only (jailbreak.build_defense_cell), so the render check must
    not resolve a corpus pseudo-instruction as a system prompt (it would render megabytes and refuse every window)."""
    from sisteer.eval.jailbreak import resolve_instruction
    assert pf.si_text_of("none") is None
    for c in OBJ_CELLS[3:]:
        assert pf.si_text_of(c) is None, c
    assert pf.si_text_of("prompted__safety_refusal") == resolve_instruction("safety_refusal").text
    assert pf.si_text_of("prompted__si_constitution") == resolve_instruction("si_constitution").text
    assert pf.si_text_of("stacked__safety_refusal__psr_kl_all") == resolve_instruction("safety_refusal").text


def test_obj_sbatch_parses_and_carries_the_cluster_a_rules():
    rc = subprocess.run(["bash", "-n", str(OBJ_SBATCH)], capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr
    src = OBJ_SBATCH.read_text()
    for line in src.splitlines():
        if "--export=" in line:
            assert "never" in line.lower(), line
        if re.search(r"(?<![\w$\"{])CUDA_VISIBLE_DEVICES=", line):
            pytest.fail(f"assigns CUDA_VISIBLE_DEVICES: {line}")
        assert "scancel" not in line.lower()
    assert "--mem=0" not in src
    for m in ("#SBATCH --partition=general", "#SBATCH --qos=high\n", "#SBATCH --gres=gpu:1", "#SBATCH --cpus-per-task=32",
              "#SBATCH --mem=200G", "#SBATCH --time=04:00:00", "#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out",
              "#SBATCH --open-mode=append", "#SBATCH --no-requeue", "#SBATCH --signal=B:TERM@120"):
        assert m in src, m
    assert not token_hits(src.split("set -uo pipefail")[1], REGION_TOKENS)       # no cluster-specific QoS in the body
    mem = [l for l in src.splitlines() if l.startswith("#SBATCH --mem=")][0]
    assert 128 <= int(mem.split("=")[1].rstrip("G")) <= 250
    assert "positional only" in src and "never --export=" in src
    assert '|| source "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$WAVE_SHA/scripts/sdfbase/full22/common.sh"' in src
    assert 'KEY="${1:?key}"; WAVE_SHA="${2:?wave sha}"; WINDOW="${3:?window (the standup max_model_len)}"; PAD="${4:?pad (ONE level: 0 | 8192 | 16384 | 32768 | 65536)}"' in src
    assert 'FAMILIES="${5:-grok420}"; SHARDS_SPEC="${6:-3}"; CELLS_ARG="${7:-}"' in src
    assert 'CONFIG_FAMILY="${8:-obj}"' in src and '[ "$CELLS_ARG" != all ] || CELLS_ARG=""' in src     # the eighth positional (2026-09-24)
    assert "--time=03:00:00" not in src                                      # the walltime rule: 4 h on every job
    for needle in ("SHARDING ACROSS JOBS", "WHY IT IS SAFE, AND WHERE THE SEAM IS", "goals WITHIN a cell", "within-standup",
                   "--time=04:00:00 on EVERY job", "AS MEASURED on the filed stores", "10.66 M completion tokens", "N = 16"):
        assert needle in src, needle
    assert "trap cleanup EXIT SIGTERM SIGINT SIGQUIT" in src and "f22_result" in src
    body = src[src.index("cleanup(){"):src.index("trap cleanup EXIT")]
    assert body.index("local ec=$?") < body.index("trap '' SIGTERM SIGINT SIGQUIT") < body.index("kill -TERM")
    assert 'TARGET_GPU="$CUDA_VISIBLE_DEVICES"' in src and "f22_stop_requested" in src
    assert "source $SHARED/scripts/hf_env.sh" not in src                    # f22_init sources the SHARED copy
    assert "srun " not in src.replace("srun-WRAPPED", "").replace("in srun", "")
    # the header carries the degeneracy-writer rationale
    for needle in ("PER-ITEM DEGENERACY WRITER IS NOT OPTIONAL", "DIFFERENTLY BY MODEL", "FLIPS SIGN", "BETWEEN RUBRICS",
                   "looping and truncation RISE", "the steering decayed", "the generations degenerated", "degeneracy_items.py"):
        assert needle in src, needle


def test_obj_sbatch_uses_the_campaign_bank_view_over_the_lane_roster_and_pins_the_geometry_per_level():
    src = OBJ_SBATCH.read_text()
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    i_roster = src.index('F22_ROSTER="configs/sdfbase/static_padding/sdfbase_full22_${KEY}.${VIEW}_roster.yaml"')
    i_decl = src.index('F22_LAUNCH_MANIFEST="$MANIFEST"')
    i_init = src.index('f22_init "$WAVE_SHA"')
    i_venv = src.index("f22_driver_venv; f22_serving_venv")
    i_view = src.index('f22_bank_view "$MPY"')
    i_pre = src.index('"$MPY" "$PAD_DIR/padding_preflight.py" --window "$WINDOW"')
    i_up = src.index('f22_standup attacks "$SERVE_DIR/static" "" families_served:short "$WINDOW" || { RC=1; exit 1; }')
    i_pin = src.index('"$MPY" - "$SERVE_DIR/static/endpoint.json" "$KEY" "$WINDOW" "$TARGET_S"')
    i_pc = src.index('f22_positive_control "$pc" "static_p${PAD}"')
    i_drv = src.index("scripts/jailbreak_run.py generate")
    i_stop = src.index("f22_standup_stop")
    assert i_roster < i_decl < i_init < i_venv < i_view < i_pre < i_up < i_pin < i_pc < i_drv < i_stop
    assert code.count("f22_standup attacks") == 1 and code.count("f22_standup_stop") == 1
    assert '"" families_served:short "$WINDOW"' in code and "--max-num-seqs" not in code
    # the VIEW PATH: the campaign's helper over the lane roster — never the prompted-only wave's scratch view
    assert "pad_bank_view" not in src and "prompted_only" not in code and "make_bank_view.py" not in code
    # the paths follow the CONFIG FAMILY positional (2026-09-24): `obj` renders the seed-1 spellings byte for byte (test below)
    assert 'obj)    VIEW=padobj;;' in code and 'TAG="${KEY}_${VIEW}"' in code
    assert 'MANIFEST="configs/capeval/manifests/petri_full22_${KEY}.padstatic_${CONFIG_FAMILY}.yaml"' in code
    assert 'CFG="configs/sdfbase/static_padding/sdfbase_full22_${KEY}.padstatic_${CONFIG_FAMILY}_p${PAD}.yaml"' in code
    assert 'FILLER="${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}/data/pg19_filler.json"' in src and "--filler \"$FILLER\"" in src
    assert "F22_LOAD_CHECK_SYSTEM_SI" in code and "f22_standup_extra_reset --enable-log-requests" in code
    assert "f22_served_cells_filter CELLS \"$CELLS_ARG\"" in code
    assert 'PRIVATE_ROOT="$WAVE_ROOT/static_padding/transcripts_results"' in code
    assert 'F22_REGISTRY_EVALS="${FAMILIES//,/ }"; export F22_REGISTRY_EVALS' in code and "F22_SERVING_PATH=" not in code   # registered per level
    assert code.index("f22_registry_register || true") < code.index('f22_bank_view "$MPY"')
    assert "DRIVER_CAP=96" in code and "MAX_DRIVERS=$(( TARGET_S < DRIVER_CAP ? TARGET_S : DRIVER_CAP ))" in code
    for forbidden in ("--greedy", "--target-greedy", "--static-padding-tokens", "--static-padding-placement", "--static-padding-mode"):
        assert forbidden not in code, forbidden
    # ONE level per job: the positional is one integer, a comma list is refused by name, and the drivers loop over no level list
    assert "a comma list is refused" in code and "for pad in" not in code and 'PAD_FLAGS=(); [ "$PAD" -gt 0 ]' in code
    # the GEOMETRY PIN, extended with the D check (the module-dim premise is load-bearing on a steered standup)
    pin = src[i_pin:i_pc]
    for needle in ("from sisteer.capeval.petri.defense_cells import NOOP_LORA_CELL_ID", "from sisteer.capeval.serving import geometry as g",
                   "rb.entry.seqs_at(L)", "g.kernel_ceiling_adjustment(key, L, identity, module_set=SET)",   # the SELECTED set's D
                   'ep.get("engine") != "lora_native"', "NOOP_LORA_CELL_ID not in names", "if S != expect:",
                   '(ep.get("geometry") or {}).get("adjustments")', 'int(a["D"]) != int(adj["D"])',
                   "REFUSING TRAFFIC: the standup's geometry is not the pinned geometry"):
        assert needle in pin, needle
    assert not re.search(r"(?<![\w.])none__lora0", code)
    assert "4,096" in src and "lora_kernel_ceiling_geometry" in src and "adapter_module_dims" in src


def _render_obj_driver_argv(pad: int, extra_vars: str = "") -> list[list[str]]:
    src = OBJ_SBATCH.read_text()
    loop = src[src.index("# --- drivers: begin"):src.index("# --- drivers: end")]
    stubs = ('f22_queue_launch(){ shift; for a in "$@"; do printf "%s\\n" "$a"; done; printf "@@\\n"; }; '
             'f22_stop_requested(){ return 1; }; f22_queue_wait(){ NFAIL=0; }; f22_log(){ :; }; ')
    vars_ = (f'set -u; PAD={pad}; CFG="configs/sdfbase/static_padding/sdfbase_full22_qwen35_4b.padstatic_obj_p{pad}.yaml"; '
             'FAMS=(grok420); CELLS=(none arditi_allpos constv2-mix-full__psr_ce_all_minus_ctl); NSHARDS=2; SHARD_LO=0; SHARD_HI=1; MPY=/venv/bin/python; '
             'PRIVATE_ROOT=/wave/static_padding/transcripts_results; PROXY_URL=http://127.0.0.1:4321/v1; TABLE=/wave/serving/routing_table.json; '
             f'TARGET_L={WINDOW}; FILLER=/scratch/proj/data/pg19_filler.json; MAX_DRIVERS=5; declare -a DRIVER_PIDS=(); TOTAL_FAIL=0; '
             'sleep(){ :; }; ' + extra_vars)
    rc = subprocess.run(["bash", "-c", vars_ + stubs + loop], capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr
    return [blk.split("\n") for blk in rc.stdout.strip().split("\n@@") if blk.strip()]


@pytest.mark.parametrize("pad", OBJ_PADS)
def test_obj_rendered_driver_argv_carries_the_level_in_the_config_and_the_filler_only_when_padded(pad):
    argvs = _render_obj_driver_argv(pad)
    assert len(argvs) == 1 * 3 * 2                                              # 1 family x 3 cells x 2 shards, ONE level
    for argv in argvs:
        argv = [a for a in argv if a]
        assert argv[:3] == ["/venv/bin/python", "scripts/jailbreak_run.py", "generate"]
        assert argv[argv.index("--config") + 1] == f"configs/sdfbase/static_padding/sdfbase_full22_qwen35_4b.padstatic_obj_p{pad}.yaml"
        assert argv[argv.index("--attack") + 1] == "grok420" and "--confirmatory" in argv
        assert argv[argv.index("--target-engine") + 1] == "served" and argv[argv.index("--target-max-model-len") + 1] == str(WINDOW)
        assert argv[argv.index("--private-root") + 1] == "/wave/static_padding/transcripts_results"
        assert "--static-padding-tokens" not in argv and "--greedy" not in argv
        if pad == 0:
            assert not any(a.startswith("--static-padding") for a in argv)
        else:
            assert argv[argv.index("--static-padding-filler") + 1] == "/scratch/proj/data/pg19_filler.json"
            assert sum(a.startswith("--static-padding") for a in argv) == 1
    assert sorted(a[a.index("--defense") + 1] for a in argvs) == sorted(["none"] * 2 + ["arditi_allpos"] * 2 + ["constv2-mix-full__psr_ce_all_minus_ctl"] * 2)


# --- the SHARDED shape (2026-09-24): the range positional, the leases, the driver loop -------------------------------------

def _shard_block() -> str:
    src = OBJ_SBATCH.read_text()
    return src[src.index("# --- shards: begin"):src.index("# --- shards: end")]


def _parse_shards(spec: str) -> tuple[int, str]:
    """Run the sbatch's shard-spec parser under bash with a stubbed f22_fatal: (rc, stdout)."""
    prog = ('set -u; f22_fatal(){ echo "FATAL: $*"; exit 1; }; SHARDS_SPEC=' + shlex.quote(spec) + "\n" + _shard_block()
            + '\necho "ok lo=$SHARD_LO hi=$SHARD_HI N=$NSHARDS count=$SHARD_COUNT"')
    rc = subprocess.run(["bash", "-c", prog], capture_output=True, text=True)
    return rc.returncode, (rc.stdout + rc.stderr).strip()


def test_shard_spec_parses_a_bare_count_and_a_range_and_refuses_every_malformation():
    """`N` is the single-job shape (all N shards); `<lo>-<hi>/N` is this job's range. Out-of-range, inverted, malformed and
    too-wide specs refuse BY NAME before anything else runs — the driver would otherwise slice an empty goal list for a
    shard index >= N and write an empty shard directory without a word."""
    assert _parse_shards("3") == (0, "ok lo=0 hi=2 N=3 count=3")                    # the filed rungs' shape
    assert _parse_shards("0-3/16") == (0, "ok lo=0 hi=3 N=16 count=4")              # the added rungs' four jobs
    assert _parse_shards("12-15/16") == (0, "ok lo=12 hi=15 N=16 count=4")
    assert _parse_shards("7-7/8") == (0, "ok lo=7 hi=7 N=8 count=1")
    assert _parse_shards("00-01/08") == (0, "ok lo=0 hi=1 N=8 count=2")             # leading zeros are decimal, never octal
    for spec, phrase in (("16-19/16", "out of range"), ("16-16/16", "out of range"), ("3-2/8", "inverted"), ("0-1", "not a positive integer"),
                         ("a-b/8", "lo 'a'"), ("0-b/8", "hi 'b'"), ("0-1/0", ">= 1"), ("0-1/100", "> 99"), ("1/2/3", "not a positive integer"),
                         ("", "not a positive integer"), ("-/8", "lo ''"), ("0-/8", "hi ''"), ("5/8", "<lo>-<hi>")):
        rc, out = _parse_shards(spec)
        assert rc == 1 and out.startswith("FATAL: shards") and phrase in out, (spec, out)


def test_obj_rendered_driver_argv_covers_exactly_the_owned_range_for_every_cell():
    """A job owning 4..7 of 16 issues --shard 4,5,6,7 --nshards 16 for EVERY cell (the same split per arm: the within-standup
    comparison stays balanced), and nothing outside its range — the argv a single job at N=16 would have issued for those s."""
    argvs = _render_obj_driver_argv(32768, "NSHARDS=16; SHARD_LO=4; SHARD_HI=7; ")
    assert len(argvs) == 1 * 3 * 4                                                  # 1 family x 3 cells x 4 owned shards
    by_cell: dict = {}
    for argv in argvs:
        argv = [a for a in argv if a]
        by_cell.setdefault(argv[argv.index("--defense") + 1], []).append((int(argv[argv.index("--shard") + 1]), int(argv[argv.index("--nshards") + 1])))
        assert argv[argv.index("--config") + 1].endswith("padstatic_obj_p32768.yaml")
        assert argv[argv.index("--static-padding-filler") + 1] == "/scratch/proj/data/pg19_filler.json"
    assert set(by_cell) == {"none", "arditi_allpos", "constv2-mix-full__psr_ce_all_minus_ctl"}
    assert all(sorted(v) == [(4, 16), (5, 16), (6, 16), (7, 16)] for v in by_cell.values())
    # the loop reads the range variables, never a literal 0..N-1
    loop = OBJ_SBATCH.read_text()
    loop = loop[loop.index("# --- drivers: begin"):loop.index("# --- drivers: end")]
    assert 'for s in $(seq "$SHARD_LO" "$SHARD_HI")' in loop and "seq 0 $((NSHARDS-1))" not in loop


def _lease_block() -> str:
    src = OBJ_SBATCH.read_text()
    return src[src.index("# --- shard leases: begin"):src.index("# --- shard leases: end")]


def _run_leases(tmp: Path, *, job: str, lo: int, hi: int, n: int, squeue: str, sacct: str, setup: str = "", cells="none arditi_allpos") -> tuple[int, str]:
    """Run the sbatch's lease block under bash with stubs: squeue/sacct are functions printing the given text (empty = no
    answer, exit 1 for squeue), f22_fatal prints and exits 1. Returns (rc, output)."""
    prog = ("set -u; f22_fatal(){ echo \"FATAL: $*\"; exit 1; }; f22_log(){ echo \"LOG: $*\"; }; "
            f"squeue(){{ [ -n {shlex.quote(squeue)} ] && echo {shlex.quote(squeue)} || return 1; }}; "
            f"sacct(){{ echo {shlex.quote(sacct)}; }}; "
            f"KEY=qwen35_4b; PAD=32768; CONFIG_FAMILY=obj; WAVE_ROOT={shlex.quote(str(tmp))}; PRIVATE_ROOT={shlex.quote(str(tmp / 'transcripts_results'))}; "
            f"FAMS=(grok420); CELLS=({cells}); NSHARDS={n}; SHARD_LO={lo}; SHARD_HI={hi}; MPY={shlex.quote(sys.executable)}; "
            f"SLURM_JOB_ID={job}; SLURM_JOB_NAME=t-{job}; " + setup + "\n" + _lease_block() + "\necho DONE")
    rc = subprocess.run(["bash", "-c", prog], capture_output=True, text=True)
    return rc.returncode, (rc.stdout + rc.stderr).strip()


def test_leases_are_taken_per_family_and_shard_beside_the_store_and_released_as_exited(tmp_path):
    import json
    rc, out = _run_leases(tmp_path, job="101", lo=0, hi=3, n=16, squeue="", sacct="")
    assert rc == 0 and out.endswith("DONE") and "shard leases taken" in out, out
    lease_dir = tmp_path / "static_padding" / "shard_leases" / "pad_static_qwen35_4b_obj_p32768"
    assert sorted(p.name for p in lease_dir.iterdir()) == [f"grok420__s{i:02d}.json" for i in range(4)]
    rec = json.loads((lease_dir / "grok420__s02.json").read_text())
    assert (rec["run"], rec["family"], rec["shard"], rec["nshards"], rec["range"], rec["job_id"], rec["state"], rec["rc"]) == \
        ("pad_static_qwen35_4b_obj_p32768", "grok420", 2, 16, "0-3", "101", "running", None)
    assert not (tmp_path / "transcripts_results").exists()                       # nothing is written inside the store tree
    # the release the cleanup trap performs: state exited with the rc, same files
    prog = "\npad_lease_write exited 2\n"
    rc, out = _run_leases(tmp_path, job="101", lo=0, hi=3, n=16, squeue="", sacct="", setup="")
    assert rc == 0                                                                # our own running lease re-enters cleanly
    rc = subprocess.run(["bash", "-c", "set -u; f22_fatal(){ exit 1; }; f22_log(){ :; }; " +
                         f"KEY=qwen35_4b; PAD=32768; CONFIG_FAMILY=obj; WAVE_ROOT={shlex.quote(str(tmp_path))}; PRIVATE_ROOT=/nonexistent; FAMS=(grok420); CELLS=(none); "
                         f"NSHARDS=16; SHARD_LO=0; SHARD_HI=3; MPY={shlex.quote(sys.executable)}; SLURM_JOB_ID=101; SLURM_JOB_NAME=t; RC=2\n"
                         + _lease_block().replace("pad_lease_check\npad_lease_write running null; LEASE_WRITTEN=1", "pad_lease_write exited \"$RC\"")],
                        capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr
    rec = json.loads((lease_dir / "grok420__s00.json").read_text())
    assert rec["state"] == "exited" and rec["rc"] == 2


def test_leases_refuse_overlap_with_a_live_job_another_layout_and_unvouched_stores_but_free_a_dead_holder(tmp_path):
    # job 101 holds 0..3 of 16, live per squeue: an overlapping 2..5 refuses; a disjoint 4..7 passes
    assert _run_leases(tmp_path, job="101", lo=0, hi=3, n=16, squeue="", sacct="")[0] == 0
    rc, out = _run_leases(tmp_path, job="202", lo=2, hi=5, n=16, squeue="RUNNING", sacct="")
    assert rc == 1 and "held by LIVE job 101" in out and "overlapping" in out and "shard02" in out, out
    rc, out = _run_leases(tmp_path, job="202", lo=4, hi=7, n=16, squeue="RUNNING", sacct="")
    assert rc == 0 and "shard leases taken" in out, out
    # a holder squeue no longer lists but sacct reports terminal (TIMEOUT / CANCELLED by ...) is dead: taken over, per goal
    rc, out = _run_leases(tmp_path, job="303", lo=0, hi=1, n=16, squeue="", sacct="TIMEOUT")
    assert rc == 0 and "stale lease from job 101" in out and "taken over" in out, out
    rc, out = _run_leases(tmp_path, job="404", lo=2, hi=2, n=16, squeue="", sacct="CANCELLED by 12345")
    assert rc == 0 and "stale lease from job 101" in out, out
    # a holder squeue does not list and sacct says is still PENDING is live; one neither tool answers for is unknown: both refuse
    rc, out = _run_leases(tmp_path, job="505", lo=3, hi=3, n=16, squeue="", sacct="PENDING")
    assert rc == 1 and "held by LIVE job 101" in out, out
    rc, out = _run_leases(tmp_path, job="606", lo=3, hi=3, n=16, squeue="", sacct="")
    assert rc == 1 and "neither squeue nor sacct" in out and "refusing" in out, out
    # another N over a leased shard: two layouts of one store — refused even when the holder is dead
    rc, out = _run_leases(tmp_path, job="707", lo=0, hi=0, n=8, squeue="", sacct="TIMEOUT")
    assert rc == 1 and "leased under N=16" in out and "N=8" in out and "two layouts" in out, out
    # an exited lease is free without asking the scheduler at all
    (tmp_path / "static_padding/shard_leases/pad_static_qwen35_4b_obj_p32768/grok420__s08.json").write_text(
        '{"job_id": "1", "nshards": 16, "state": "exited"}')
    rc, out = _run_leases(tmp_path, job="808", lo=8, hi=8, n=16, squeue="RUNNING", sacct="RUNNING")
    assert rc == 0 and "stale" not in out, out
    # the on-disk belt: a shard directory numbered beyond N refuses; one in range with no lease refuses; a leased one passes
    store = tmp_path / "transcripts_results" / "pad_static_qwen35_4b_obj_p32768" / "grok420" / "none"
    (store / "shard09").mkdir(parents=True)
    rc, out = _run_leases(tmp_path, job="909", lo=9, hi=9, n=16, squeue="", sacct="")
    assert rc == 1 and "shard09 exists with no lease" in out and "cannot vouch for" in out, out
    (store / "shard09").rmdir(); (store / "shard16").mkdir()
    rc, out = _run_leases(tmp_path, job="910", lo=10, hi=10, n=16, squeue="", sacct="")
    assert rc == 1 and "shard16 exists but this job says N=16" in out and "larger N" in out, out
    (store / "shard16").rmdir(); (store / "shard04").mkdir()                       # leased above by job 202 (live): its own resume
    rc, out = _run_leases(tmp_path, job="202", lo=4, hi=7, n=16, squeue="RUNNING", sacct="")
    assert rc == 0, out                                                             # the same job re-entering its own lease
    # an unreadable lease refuses rather than being taken over
    (tmp_path / "static_padding/shard_leases/pad_static_qwen35_4b_obj_p32768/grok420__s11.json").write_text("not json")
    rc, out = _run_leases(tmp_path, job="911", lo=11, hi=11, n=16, squeue="", sacct="")
    assert rc == 1 and "unreadable lease" in out, out


def test_obj_sbatch_orders_the_leases_before_the_standup_and_releases_them_in_cleanup():
    src = OBJ_SBATCH.read_text()
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    i_pre = src.index('"$MPY" "$PAD_DIR/padding_preflight.py" --window "$WINDOW"')
    i_lease = src.index("pad_lease_check\npad_lease_write running null; LEASE_WRITTEN=1")
    i_trap = src.index("trap cleanup EXIT SIGTERM SIGINT SIGQUIT")
    i_up = src.index('f22_standup attacks "$SERVE_DIR/static" "" families_served:short "$WINDOW" || { RC=1; exit 1; }')
    i_shards = src.index("# --- shards: begin")
    i_init = src.index('f22_init "$WAVE_SHA"')
    assert i_init < i_shards < i_pre < i_lease < i_trap < i_up                     # parsed after init, leased after the preflight, before any GPU minute
    body = src[src.index("cleanup(){"):src.index("trap cleanup EXIT")]
    assert 'pad_lease_write exited "${RC:-1}"' in body and body.index("pad_lease_write exited") < body.index("f22_result")
    assert 'LEASE_DIR="$WAVE_ROOT/static_padding/shard_leases/$RUN"' in code and 'RUN="pad_static_${KEY}_${CONFIG_FAMILY}_p${PAD}"' in code
    assert "shards=$SHARD_LO-$SHARD_HI/$NSHARDS" in code                           # the RESULT and phase lines name the range
    assert "for pad in" not in code


# --- the SEED REPLICATES (2026-09-24): obj_s2 / obj_s3, six cells, the padobj_s<N> views, the family positional -------------

#: SEVEN cells: `none` first — the native standup derives its zero-adapter leg from it (29 six-cell jobs refused without it,
#: 2026-09-24) — then the seed-1 family's six fitted objects at the seed
OBJ_SEED_CELLS = ("none", "constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce", "constv2ctl2-full__psr_ce_all",
                  "constv2ctl2-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl", "constv2-mix-full__psr_ce_all_minus_ctl")
OBJ_SEED_HARNESS = ["none", "si_constv2_mix_full__psr_ce_all", "si_constv2_mix_full__lora_r4_ce", "si_constv2ctl2_full__psr_ce_all",
                    "si_constv2ctl2_full__lora_r4_ce", "si_constv2_mix_full__lora_r4_ce_minus_ctl", "si_constv2_mix_full__psr_ce_all_minus_ctl"]
#: the seed-2/3 objects' pins (configs/sdfbase_full22_pins.json), re-hashed on cluster A 2026-09-24 — a drift here is a different object
SEED_STAGED_SHA = {
    2: {"base64s2__constv2-mix-full__psr_ce_all": "2d43a58e55e674d3e33c6cbbaf7bb0ad9b06021d308b9e924213acf7c0609fd8",
        "base64s2__constv2-mix-full__lora_r4_ce": "724fe6ddb52cb679656442b2537057900f3c523ddc3c45b3edde50bdc4a487d2",
        "base64s2__constv2ctl2-full__psr_ce_all": "900fbabb6d56a61f37acf71a28482f8dd843a6f27809f12f09ca38a1328bbecb",
        "base64s2__constv2ctl2-full__lora_r4_ce": "900938b516d6e266eacd5d53aef2663367fe0974fcb19757ead908b1e1a68e63",
        "derived_base64s2__constv2-mix-full__lora_r4_ce_minus_ctl": "06f81d946969c1863103d6856841b7fc8c8ddb0475c9851b1860c438387e979f",
        "paired_eb64s2__constv2-mix-full__psr_ce_all_minus_ctl": "9c639c91d2ef5b8fc482cb039e8ae65deaa79389e1eab7daaa0c45a5caacfc66"},
    3: {"base64s3__constv2-mix-full__psr_ce_all": "cc1ec068380ed8fe51de1e91e8b8fe6b32c95fa9a00bcf381a536007c075bf09",
        "base64s3__constv2-mix-full__lora_r4_ce": "a9934f6a796c238d58f4da10615c9f1c211d8b067afcf1330241cdcff967ce42",
        "base64s3__constv2ctl2-full__psr_ce_all": "c5686c7316b7f79ba31edeea4ecf5d3b237724e399cafb41295478b457471c34",
        "base64s3__constv2ctl2-full__lora_r4_ce": "33e7b327049aa28605818fe02970d918a86a2c6eea8f6bbf38a01779b84f9d32",
        "derived_base64s3__constv2-mix-full__lora_r4_ce_minus_ctl": "8952728fa0f7fb6a0b48557e9c491073c5080f63178fc64c4f113088e1f2f974",
        "paired_eb64s3__constv2-mix-full__psr_ce_all_minus_ctl": "0718a3f9edfb033ad84c3322f717dc62b32b6298fe48f0a6c42e9f0a7c031647"},
}


def test_seed_replicate_family_names_parse_and_the_seed_1_spellings_are_unchanged():
    assert gen.OBJ_SEEDS == (2, 3) and gen.OBJ_SEED_FAMILIES == ("obj_s2", "obj_s3") and gen.OBJ_SEED_CELLS == OBJ_SEED_CELLS
    assert gen.FAMILIES == (None, "obj", "obj_s2", "obj_s3") and gen.DECODE_SEED_BASE_HELD == 20260909 and gen.GEOMETRY_PIN_S == 5
    assert (gen.seed_of(None), gen.seed_of("obj"), gen.seed_of("obj_s2"), gen.seed_of("obj_s3")) == (None, None, 2, 3)
    assert (gen.view_of("obj"), gen.view_of("obj_s2"), gen.view_of("obj_s3")) == ("padobj", "padobj_s2", "padobj_s3")
    assert gen.padded_roster("qwen35_4b") == gen.padded_roster("qwen35_4b", "obj")                 # the seed-1 default stands
    for n in (2, 3):
        fam = f"obj_s{n}"
        assert gen.cells_of(fam) == OBJ_SEED_CELLS and gen.pads_of(fam) == OBJ_PADS                 # seven cells, the SAME five rungs
        assert gen.padded_roster("qwen35_4b", fam) == CONFIG_DIR / f"sdfbase_full22_qwen35_4b.padobj_s{n}_roster.yaml"
        assert gen.padded_manifest("qwen35_4b", fam).name == f"petri_full22_qwen35_4b.padstatic_obj_s{n}.yaml"
        for pad in OBJ_PADS:
            p = gen.padded_config("qwen35_4b", pad, fam)
            assert p.name == f"sdfbase_full22_qwen35_4b.padstatic_obj_s{n}_p{pad}.yaml" and p.is_file()
            assert (gen.key_of(p), gen.pad_of(p), gen.family_of(p)) == ("qwen35_4b", pad, fam)
            assert gen.private_run("qwen35_4b", pad, fam) == f"pad_static_qwen35_4b_obj_s{n}_p{pad}"
            assert gen.private_run("qwen35_4b", pad, fam) not in (gen.private_run("qwen35_4b", pad), gen.private_run("qwen35_4b", pad, "obj"))
    assert gen.cells_of("obj") == OBJ_CELLS and gen.cells_of(None) == CELLS                          # the older families are untouched
    with pytest.raises(ValueError, match="does not know"):
        gen.pad_of(Path("sdfbase_full22_qwen35_4b.padstatic_obj_s9_p0.yaml"))                       # a seed this lane does not run
    with pytest.raises(ValueError):
        gen.view_of(None)


@pytest.mark.parametrize("n", (2, 3))
@pytest.mark.parametrize("pad", OBJ_PADS)
def test_each_seed_replicate_level_is_the_campaign_config_as_data_except_the_named_keys(n, pad):
    fam = f"obj_s{n}"
    src = yaml.safe_load((REPO / "configs/sdfbase_full22_qwen35_4b.yaml").read_text())
    path = gen.padded_config("qwen35_4b", pad, fam)
    cfg = yaml.safe_load(path.read_text())
    assert cfg["explicit_cells"] == list(OBJ_SEED_CELLS) and cfg["explicit_cells"][0] == "none"   # seven cells: none first, no prompted, no Arditi
    assert not any(c.startswith("prompted__") or c == "arditi_allpos" for c in cfg["explicit_cells"])
    assert cfg["defenses_dir"] == f"out/full22/qwen35_4b/cells_view/padobj_s{n}"
    assert cfg["private_run"] == f"pad_static_qwen35_4b_obj_s{n}_p{pad}" and cfg["out_dir"] == f"out/pad_static_qwen35_4b_obj_s{n}_p{pad}"
    assert cfg["attacks"]["static_max_new_tokens"] == gen.STATIC_CAP == 16384
    block = cfg["attacks"].get("static_padding")
    assert block == (None if pad == 0 else {"tokens": pad, "placement": "before", "mode": "fixed", "reserve": 8192})
    assert cfg.get("sampled_decode") is None                                                          # the vendor-preset default, as seed 1
    want = dict(src)
    want.update(explicit_cells=list(OBJ_SEED_CELLS), private_run=cfg["private_run"], out_dir=cfg["out_dir"],
                defenses_dir=f"out/full22/qwen35_4b/cells_view/padobj_s{n}")
    want["attacks"] = {**src["attacks"], "static_max_new_tokens": gen.STATIC_CAP}
    if pad:
        want["attacks"]["static_padding"] = block
    assert cfg == want
    c = pf.contract_of(cfg, path)
    assert (c["level"], c["tokens"], c["cap"]) == (pad, pad, gen.STATIC_CAP) and pf.check_cap(c) == [] and not pf.check_contract(c, WINDOW)
    text = path.read_text()
    # the header records the two deliberate choices: the HELD decode seed base and the absent `none` anchor
    for needle in (f"SEED-{n} REPLICATE", "SEED REPLICATE (2026-09-24", "DECODE SEED BASE IS HELD AT 20260909", "NOT set to the training seed",
                   "`none` IS LISTED", "enumerate_cells derived no none__lora0 leg", "IDENTITY S = 7", "THE WIDTH IS PINNED", "GEOMETRY_PIN_S",
                   "S = 5", f"padobj_s{n}", "target_static_padded_obj.sbatch", f"`obj_s{n}`", "GENERATED by scripts/sdfbase/full22/static_padding/"):
        assert needle in text, needle
    assert "CROSS-STANDUP" not in text and "NO `none` CELL" not in text                              # the six-cell design is gone, not hedged
    assert "# ADDED RUNG (2026-09-24)" not in text                                                   # the seed-1 family's history, not this one's


@pytest.mark.parametrize("n", (2, 3))
def test_the_seed_replicate_manifest_names_seven_harness_ids_six_natively_served_and_the_none_passthrough(n):
    import json
    import mt_serving_up as m
    from sisteer.capeval import manifest as cm
    from sisteer.capeval.petri import registry_adapter as ra

    fam = f"obj_s{n}"
    src = yaml.safe_load((REPO / "configs/capeval/manifests/petri_full22_qwen35_4b.yaml").read_text())
    man = gen.padded_manifest("qwen35_4b", fam)
    doc = yaml.safe_load(man.read_text())
    assert doc["petri"]["cells"] == OBJ_SEED_HARNESS == [ra.harness_cell_id(c) for c in OBJ_SEED_CELLS]
    want = dict(src); want["petri"] = {**src["petri"], "cells": OBJ_SEED_HARNESS}
    assert doc == want
    spec = cm.load_petri_spec(man)
    for pad in OBJ_PADS:
        cfg = yaml.safe_load(gen.padded_config("qwen35_4b", pad, fam).read_text())
        assert list(m.assert_manifest_matches_config(spec, cfg)) == OBJ_SEED_HARNESS
    # the kind-aware serving-path default: six natively servable objects, NOTHING skipped, and the ONE passthrough leg `none` the
    # native standup derives its zero-adapter leg from — a skipped cell would silently drop an arm, so the selection is pinned here
    rc = subprocess.run([sys.executable, str(REPO / "scripts/sdfbase/full22/serving_path_select.py"), "--manifest", str(man), "--json"],
                        capture_output=True, text=True, cwd=REPO, env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"})
    assert rc.returncode == 0, rc.stderr
    sel = json.loads(rc.stdout)
    assert sel["selected"] == "lora_native" and sel["cells_by_disposition"]["skip"] == []
    assert len(sel["cells_by_disposition"]["serve"]) == 6 and sel["cells_by_disposition"]["passthrough"] == ["none"]


@pytest.mark.parametrize("n", (2, 3))
def test_the_seed_replicate_roster_reads_the_production_seed_views_in_the_string_form_and_is_pinned(n):
    """Six same-bank staged ids: the five fitted cells from the production `s<N>` view, the paired write from its `s<N>p` view —
    every one carrying the seed, pinned in the wave pins to the sha re-hashed on cluster A; no pin form (the paired_write bank
    stages no seed-2/3 copy); a view the production roster does not name; nothing half-dose or distilled."""
    import json
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))
    import f22_roster as fr

    fam, view = f"obj_s{n}", f"padobj_s{n}"
    roster_p = gen.padded_roster("qwen35_4b", fam)
    assert roster_p == REPO / f"configs/sdfbase/static_padding/sdfbase_full22_qwen35_4b.padobj_s{n}_roster.yaml"
    roster = fr.load_roster(roster_p)
    prod = fr.load_roster(REPO / "configs/sdfbase_full22_roster.yaml")
    assert view not in prod["targets"]["qwen35_4b"]["views"] and "padobj" not in prod["targets"]["qwen35_4b"]["views"]
    assert (roster["wave"], roster["bank_root"], roster["staging_manifest"]) == (prod["wave"], prod["bank_root"], prod["staging_manifest"])
    t = roster["targets"]["qwen35_4b"]
    assert (t["model"], t["model_revision"]) == (prod["targets"]["qwen35_4b"]["model"], prod["targets"]["qwen35_4b"]["model_revision"])
    assert list(roster["targets"]) == ["qwen35_4b"] and list(t["views"]) == [view] and roster["lane_flags"] == {}
    v = t["views"][view]
    assert v["config"] == f"configs/sdfbase/static_padding/sdfbase_full22_qwen35_4b.padstatic_obj_s{n}_p0.yaml"
    assert v["manifest"] == f"configs/capeval/manifests/petri_full22_qwen35_4b.padstatic_obj_s{n}.yaml"
    served = fr.served_cells(roster, "qwen35_4b", view); staged = fr.staged_cells(roster, "qwen35_4b", view)
    assert list(served) == list(OBJ_SEED_CELLS[1:]) and len(staged) == 6 and not any(fr.is_pin_form(s) for s in staged.values())   # none is a config cell, never a roster cell
    fitted = fr.staged_cells(prod, "qwen35_4b", f"s{n}"); paired = fr.staged_cells(prod, "qwen35_4b", f"s{n}p")
    wave_pins = json.loads((REPO / "configs/sdfbase_full22_pins.json").read_text())["qwen35_4b"]
    for c in gen.OBJ_S1_CELLS:
        assert staged[c] == fitted[c] and f"s{n}__" in staged[c]
    assert staged["constv2-mix-full__psr_ce_all_minus_ctl"] == paired["constv2-mix-full__psr_ce_all_minus_ctl"] \
        == f"paired_eb64s{n}__constv2-mix-full__psr_ce_all_minus_ctl"
    assert {sid: wave_pins[sid]["object_sha256"] for sid in staged.values()} == SEED_STAGED_SHA[n]
    assert all(wave_pins[sid]["verified_cluster_a"] is True for sid in staged.values())
    assert fr.every_staged_id(roster, "qwen35_4b") == list(staged.values())
    assert not any("distilled" in sid or "halfdose" in sid or "d0p5" in sid for sid in staged.values())
    text = roster_p.read_text()
    assert text.startswith(f"# sdfbase_full22_qwen35_4b.padobj_s{n}_roster.yaml") and f"SEED-{n} REPLICATE" in text and "STRING form" in text
    assert "GENERATED by scripts/sdfbase/full22/static_padding/" in text


def test_the_seed_roster_renderer_refuses_a_wrong_seed_a_half_dose_id_and_an_unpinned_object(monkeypatch):
    import json
    good_pins = json.loads(gen.WAVE_PINS.read_text())
    prod = yaml.safe_load(gen.PRODUCTION_ROSTER.read_text())
    # the production s2 view mapping a cell to a seed-1 id: refused by name
    bad = json.loads(json.dumps(prod)); bad["targets"]["qwen35_4b"]["views"]["s2"]["cells"]["constv2-mix-full__psr_ce_all"] = "base64s1__constv2-mix-full__psr_ce_all"
    with pytest.raises(SystemExit, match="does not carry seed 2"):
        gen._seed_roster_cells("qwen35_4b", "obj_s2", bad, good_pins["qwen35_4b"])
    # a half-dose paired id: OUT
    bad = json.loads(json.dumps(prod)); bad["targets"]["qwen35_4b"]["views"]["s2p"]["cells"]["constv2-mix-full__psr_ce_all_minus_ctl"] = "d0p5_eb64s2__constv2-mix-full__psr_ce_all_minus_ctl"
    with pytest.raises(SystemExit, match="half-dose or distilled"):
        gen._seed_roster_cells("qwen35_4b", "obj_s2", bad, good_pins["qwen35_4b"])
    # an unpinned object, and one not verified on cluster A
    pins = json.loads(json.dumps(good_pins))["qwen35_4b"]; del pins["derived_base64s3__constv2-mix-full__lora_r4_ce_minus_ctl"]["object_sha256"]
    with pytest.raises(SystemExit, match="no object_sha256 pin"):
        gen._seed_roster_cells("qwen35_4b", "obj_s3", prod, pins)
    pins = json.loads(json.dumps(good_pins))["qwen35_4b"]; pins["paired_eb64s3__constv2-mix-full__psr_ce_all_minus_ctl"]["verified_cluster_a"] = False
    with pytest.raises(SystemExit, match="not verified on cluster A"):
        gen._seed_roster_cells("qwen35_4b", "obj_s3", prod, pins)


def _family_block() -> str:
    src = OBJ_SBATCH.read_text()
    return src[src.index('case "$CONFIG_FAMILY" in'):src.index("esac", src.index('case "$CONFIG_FAMILY" in')) + 4]


def test_obj_sbatch_family_positional_selects_the_view_and_every_derived_path_and_refuses_an_unknown_family():
    """`obj` (the default) renders the seed-1 spellings byte for byte; `obj_s<N>` the seed replicate's; anything else refuses
    by name before a roster is read. The literal `all` in the cells positional means every cell, so the family can follow."""
    src = OBJ_SBATCH.read_text()
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    # the default arm asks the generator (`--resolve-family`): the harness supplies what the sbatch has by then — f22_fatal, a no-op
    # f22_driver_venv, $MPY, $PAD_DIR, f22_log. REPO_DIR is bound HERE only to point the resolver at this tree: the job has NOT bound it
    # yet (f22_init does, after this block), so the block reads `${REPO_DIR:-$PWD}`; tests/test_padding_front_door_view_families.py
    # renders it with REPO_DIR unset, as the job runs it (cluster A 981270 died on a bare `$REPO_DIR` under set -u)
    harness = ('set -u; f22_fatal(){ echo "FATAL: $*"; exit 1; }; f22_driver_venv(){ :; }; f22_log(){ echo "$*"; }; '
               f'MPY={sys.executable}; REPO_DIR={REPO}; PAD_DIR=scripts/sdfbase/full22/static_padding; export PYTHONPATH={REPO / "src"}; ')
    for fam, view in (("", "padobj"), ("obj", "padobj"), ("obj_s2", "padobj_s2"), ("obj_s3", "padobj_s3")):
        prog = (harness + 'set -- qwen35_4b sha 98304 0 grok420 0-3/12 all ' + fam + '\n'
                'CONFIG_FAMILY="${8:-obj}"\nCELLS_ARG="${7:-}"\n[ "$CELLS_ARG" != all ] || CELLS_ARG=""\nKEY="$1"\n' + _family_block()
                + '\necho "fam=$CONFIG_FAMILY view=$VIEW cells=[$CELLS_ARG]"')
        rc = subprocess.run(["bash", "-c", prog], capture_output=True, text=True)
        assert rc.returncode == 0 and rc.stdout.strip() == f"fam={fam or 'obj'} view={view} cells=[]", (fam, rc.stdout, rc.stderr)
    prog = harness + 'KEY=qwen35_4b; CONFIG_FAMILY=obj_s9\n' + _family_block()
    rc = subprocess.run(["bash", "-c", prog], capture_output=True, text=True)
    assert rc.returncode == 1 and "not a config family" in rc.stdout and "obj_s9" in rc.stdout, rc.stdout + rc.stderr
    # ... and a front door's phase view for the hand-listed key is refused the same way (the generator's answer, by name: the hand-listed
    # key's families are obj | obj_s2 | obj_s3; a phase view is a FRONT-DOOR key's family)
    prog = harness + 'KEY=qwen35_4b; CONFIG_FAMILY=s1\n' + _family_block()
    rc = subprocess.run(["bash", "-c", prog], capture_output=True, text=True)
    assert rc.returncode == 1 and "not a config family" in rc.stdout and "a front-door key's phase view" in rc.stdout and "obj | obj_s2 | obj_s3" in rc.stdout, rc.stdout + rc.stderr
    # every path the job reads follows the family / view — the seed-1 default renders the filed spellings
    for needle in ('F22_ROSTER="configs/sdfbase/static_padding/sdfbase_full22_${KEY}.${VIEW}_roster.yaml"',
                   'MANIFEST="configs/capeval/manifests/petri_full22_${KEY}.padstatic_${CONFIG_FAMILY}.yaml"',
                   'CFG="configs/sdfbase/static_padding/sdfbase_full22_${KEY}.padstatic_${CONFIG_FAMILY}_p${PAD}.yaml"',
                   'RUN="pad_static_${KEY}_${CONFIG_FAMILY}_p${PAD}"', 'TAG="${KEY}_${VIEW}"'):
        assert needle in code, needle
    i_case = src.index('case "$CONFIG_FAMILY" in'); i_roster = src.index('F22_ROSTER="configs/sdfbase/static_padding/')
    assert src.index("common.sh") < i_case < i_roster < src.index('f22_init "$WAVE_SHA"')          # after f22_fatal exists, before any reader
    assert '[ -z "${FD_ROSTER:-}" ] || F22_ROSTER="$FD_ROSTER"' in code and 'PAD_REGISTRY_VIEW_DEFAULT="${FD_REGISTRY_VIEW:-main}"' in code   # the front door's answer
    assert '"$MPY" "$PAD_DIR/make_padding_configs.py" --resolve-family "$KEY" "$CONFIG_FAMILY"' in code and 'GEOMETRY_PIN_S=""' in code
    for needle in ("SEED REPLICATES (2026-09-24", "DECODE SEED BASE IS HELD AT 20260909", "WHY `none` IS RUN PER SEED",
                   "enumerate_cells derived no none__lora0 leg", "GEOMETRY_PIN_S = 5", "obj_s2", "obj_s3",
                   "THREE jobs per (seed, rung), N = 12", "all obj_s3", "[family=obj]", "CLUSTER_QOS"):
        assert needle in src, needle
    # the rendered lease directory and store follow the family too (no seed store can collide with the seed-1 stores)
    import json
    rc, out = _run_leases(Path(subprocess.run(["mktemp", "-d"], capture_output=True, text=True).stdout.strip()),
                          job="1", lo=0, hi=3, n=12, squeue="", sacct="", setup="CONFIG_FAMILY=obj_s2; ")
    assert rc == 0 and "pad_static_qwen35_4b_obj_s2_p32768" in out, out


def test_bank_view_count_guard_subtracts_the_none_anchor_only_when_the_config_lists_one():
    """common.sh f22_bank_view's short-view guard (2026-09-24): the campaign configs list `none`; a seed-replicate config lists its
    six objects only, and `none` is never a roster cell — so the served count is explicit_cells minus `none` WHEN present."""
    common = (REPO / "scripts/sdfbase/full22/common.sh").read_text()
    line = 'have=${#ALL_CELLS[@]}; case " ${ALL_CELLS[*]} " in *" none "*) have=$((have - 1));; esac'
    assert line in common and "have=$(( ${#ALL_CELLS[@]} - 1 ))" not in common
    for cells, want in (("none a b c", 3), ("a b c d e f", 6), ("none", 0), ("nonesuch a", 2)):
        rc = subprocess.run(["bash", "-c", f"set -u; ALL_CELLS=({cells}); {line}; echo $have"], capture_output=True, text=True)
        assert rc.returncode == 0 and rc.stdout.strip() == str(want), (cells, rc.stdout, rc.stderr)


# --- the WIDTH PIN (2026-09-24): the literal S = 5 in the preflight and in the post-standup geometry pin ----------------------

def test_preflight_width_pin_refuses_a_derivation_off_the_pinned_width_or_without_the_kernel_step():
    geo = pf.geometry_expectation("qwen35_4b", WINDOW)
    assert (geo["identity_S"], geo["served_S"], geo["kernel_adjustment"]["to_S"]) == (7, 5, 5)
    assert pf.check_served_s(geo, None) == [] and pf.check_served_s(geo, 5) == []
    bad = pf.check_served_s(geo, 7)
    assert len(bad) == 1 and "derived served S 5" in bad[0] and "pinned width 7" in bad[0]
    identity_only = {**geo, "served_S": 7, "kernel_adjustment": None}                    # what a standup without the zero adapter would derive
    bad = pf.check_served_s(identity_only, 5)
    assert len(bad) == 2 and "!= the lane's pinned width 5" in bad[0] and "no kernel-ceiling step lowers it" in bad[1]
    assert pf.check_served_s({"window": WINDOW}, 5) == []                                 # no geometry record (an earlier problem): nothing to add
    # the flag is wired into main and recorded
    rc = subprocess.run([sys.executable, str(LANE / "padding_preflight.py"), "--window", str(WINDOW), "--model-key", "qwen35_4b",
                         "--config", str(gen.padded_config("qwen35_4b", 0, "obj_s2")), "--hf-home", "/nonexistent", "--skip-render",
                         "--families", "grok420", "--expect-served-s", "7", "--out", "/dev/stdout"],
                        capture_output=True, text=True, cwd=REPO, env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"})
    assert rc.returncode == pf.EXIT_REFUSED and "pinned width 7" in rc.stdout and '"expect_served_s": 7' in rc.stdout


def _pin_block() -> str:
    src = OBJ_SBATCH.read_text()
    block = src[src.index("# --- geometry pin: begin"):src.index("# --- geometry pin: end")]
    start = block.index("\n", block.index("<<'PY'")) + 1          # the program begins on the line AFTER the heredoc marker
    return block[start:block.rindex("\nPY")]


def _run_pin(tmp_path: Path, ep: dict, *, S: int, pin: int = 5, L: int = WINDOW) -> tuple[int, str]:
    import json
    p = tmp_path / "endpoint.json"; p.write_text(json.dumps(ep))
    rc = subprocess.run([sys.executable, "-", str(p), "qwen35_4b", str(L), str(S), str(pin)], input=_pin_block(),
                        capture_output=True, text=True, cwd=REPO, env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"})
    return rc.returncode, rc.stdout + rc.stderr


def test_geometry_pin_holds_the_attested_width_to_the_literal_and_refuses_the_identity_seven(tmp_path):
    """The post-standup pin (rendered from the sbatch's heredoc) takes GEOMETRY_PIN_S as its fifth argument: the seed-1 wave's
    attested record (engine lora_native, the zero adapter registered, S 5 through a kernel step at D 4,096) passes; the identity
    S = 7 the six-cell attempt printed — with or without the zero adapter, with or without a chain — refuses by name."""
    good = {"engine": "lora_native", "id_name_map": {"1": "none__lora0", "2": "si_constv2_mix_full__psr_ce_all"},
            "geometry": {"adjustments": [{"kind": "lora_kernel_index_ceiling", "from_S": 7, "to_S": 5, "D": 4096, "ceiling": 524287}]}}
    rc, out = _run_pin(tmp_path, good, S=5)
    assert rc == 0 and "[geometry-pin] OK" in out and "== the literal pin 5" in out, out
    # the identity width, no zero adapter, no chain: what a standup without `none` would attest
    rc, out = _run_pin(tmp_path, {"engine": "lora_native", "id_name_map": {"2": "si_constv2_mix_full__psr_ce_all"}, "geometry": {"adjustments": []}}, S=7)
    assert rc == 1 and "REFUSED" in out and "attested max_num_seqs 7 != the lane's pinned width 5" in out \
        and "none__lora0 is not among the attested adapters" in out and "carries no lora_kernel_index_ceiling step ending at the pinned width 5" in out, out
    # the identity width WITH the zero adapter and a chain that says 5: the attested S still disagrees -> refused
    rc, out = _run_pin(tmp_path, good, S=7)
    assert rc == 1 and "attested max_num_seqs 7 != the lane's pinned width 5" in out, out
    # a pin the derivation does not reach (someone re-pins to 7): the derivation says 5 -> refused, the pin names the instrument
    rc, out = _run_pin(tmp_path, {**good, "geometry": {"adjustments": [{"kind": "lora_kernel_index_ceiling", "from_S": 7, "to_S": 7, "D": 4096, "ceiling": 524287}]}}, S=7, pin=7)
    assert rc == 1 and "derives S 5 at L=98304" in out and "!= the pinned width 7" in out, out
    # the right S reached at a wider window (a different instrument): the derivation at THAT L is not 5 -> refused
    rc, out = _run_pin(tmp_path, good, S=5, L=131072)
    assert rc == 1 and "derives S 3 at L=131072" in out, out                        # 524,287 // 131,072 = 3
    # a chain whose kernel step recorded a wider D than pinned: the existing D check still refuses
    rc, out = _run_pin(tmp_path, {**good, "geometry": {"adjustments": [{"kind": "lora_kernel_index_ceiling", "from_S": 7, "to_S": 5, "D": 8192, "ceiling": 262143}]}}, S=5)
    assert rc == 1 and "widest module D=8192 != the pinned 4096" in out, out
    # the sbatch wires the literal into both the preflight call and the pin's argv
    src = OBJ_SBATCH.read_text()
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert "GEOMETRY_PIN_S=5" in code and '--expect-served-s "$GEOMETRY_PIN_S"' in code
    assert '"$MPY" - "$SERVE_DIR/static/endpoint.json" "$KEY" "$WINDOW" "$TARGET_S" "$GEOMETRY_PIN_S"' in code
    assert code.index("GEOMETRY_PIN_S=5") < code.index('--expect-served-s "$GEOMETRY_PIN_S"') < code.index('"$TARGET_S" "$GEOMETRY_PIN_S"')


# ---------------------------------------------------------------------------------------- the warmup MLP-intermediate width

MOE_TEXT_CFG = {"num_hidden_layers": 24, "num_attention_heads": 12, "num_key_value_heads": 4, "head_dim": 128, "hidden_size": 1536,
                "moe_intermediate_size": 384, "shared_expert_intermediate_size": 768, "num_experts": 32, "num_experts_per_tok": 4,
                "mlp_only_layers": [], "layer_types": (["linear_attention"] * 2 + ["full_attention"]) * 8}   # a SYNTHETIC MoE hybrid (no table entry)
MOE_CFG = {"architectures": ["SynthMoeForCausalLM"], "model_type": "synth_moe", "text_config": MOE_TEXT_CFG}
DENSE_CFG = {"num_hidden_layers": 32, "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256, "hidden_size": 2560,
             "intermediate_size": 9216, "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 8}
DEMO_CFG = {**DENSE_CFG}      # the worked example's Qwen3.5-4B shape (reproduce/models.yaml architecture): the dims table's one entry (QWEN35_4B_DIMS)


def test_mlp_intermediate_width_reads_the_verified_dims_first_then_the_config_by_architecture_and_refuses_by_name():
    """Pass #3's five padding units died at 45 s on `int(txt["intermediate_size"])` against an MoE checkpoint's config (which carries
    `moe_intermediate_size`, `shared_expert_intermediate_size`, `num_experts_per_tok` and NO dense key). The width now comes from the
    verified dims table the serving probe plans from, the config is read by architecture behind it, and a config that yields nothing
    is a problem named — never a KeyError."""
    from sisteer.capeval.serving import standup_manifest as sm
    from sisteer.extraction import lora_modules as lm
    # the table's arithmetic: a dense block's intermediate is the per-token width (the worked example's Qwen3.5-4B: 32 dense MLPs of 9,216); a dims
    # with no dense blocks yields no table width (`elements` None), and the config-side reader below sizes such a model by architecture
    g = sm.mlp_intermediate_elements(lm.QWEN35_4B_DIMS)
    assert g["elements"] == 9_216 and g["dense_block"] == 9_216 and g["terms"]["dense_mlp_layers"] == 32 and g["terms"]["dense_intermediate"] == 9_216
    assert sm.mlp_intermediate_elements(lm.ArchDims(hidden=8, attention_layers=1, q_out=8, kv_out=8, o_in=8))["elements"] is None
    assert lm.arch_dims_for("qwen35_4b_demo") is lm.QWEN35_4B_DIMS and lm.arch_dims_for("qwen35_4b") is None and set(lm.ARCH_DIMS_BY_MODEL_KEY) == {"qwen35_4b_demo"}
    # the real shape: table + a config that agrees -> the table's number, source named
    w = pf.mlp_intermediate_width("qwen35_4b_demo", DEMO_CFG)
    assert w["problem"] is None and w["elements"] == 9_216 and w["source"] == "lora_modules.ARCH_DIMS_BY_MODEL_KEY['qwen35_4b_demo']"
    assert w["config"]["kind"] == "dense" and w["config"]["elements"] == 9_216 and w["table"]["elements"] == 9_216
    # no table entry: the config by architecture — MoE (Qwen3-MoE keys: num_experts_per_tok x moe_intermediate_size + the shared expert's),
    # Mixtral's expert-wide intermediate_size, DeepSeek's shared experts, dense blocks beside MoE blocks take the wider; a dense model reads
    # intermediate_size
    m = pf.mlp_intermediate_width("synth_moe", MOE_CFG)
    assert m["problem"] is None and m["elements"] == 2304 and m["source"] == "config:moe" and m["table"] is None
    assert m["config"]["kind"] == "moe" and m["config"]["elements"] == 2304 and m["config"]["terms"]["num_experts_per_tok"] == 4 and m["config"]["terms"]["shared_intermediate"] == 768
    assert pf.mlp_intermediate_width("mixtral", {"num_local_experts": 8, "num_experts_per_tok": 2, "intermediate_size": 14336, "hidden_size": 4096})["elements"] == 2 * 14336
    assert pf.mlp_intermediate_width("dsv", {"n_routed_experts": 64, "num_experts_per_tok": 6, "moe_intermediate_size": 1408, "n_shared_experts": 2,
                                             "first_k_dense_replace": 1, "intermediate_size": 10944})["elements"] == max(6 * 1408 + 2 * 1408, 10944)
    assert pf.mlp_intermediate_width("qwen3moe_dense_blocks", {**MOE_TEXT_CFG, "mlp_only_layers": [0, 1], "intermediate_size": 6144})["elements"] == 6144
    d = pf.mlp_intermediate_width("qwen35_4b", DENSE_CFG)
    assert d["elements"] == 9216 and d["source"] == "config:dense" and d["problem"] is None and d["table"] is None
    # refusals BY NAME, never an exception: no key at all; an MoE without its per-token count; a table the config disagrees with
    none = pf.mlp_intermediate_width("qwen35_4b", {"hidden_size": 2560, "num_hidden_layers": 32, "num_attention_heads": 16})
    assert none["elements"] is None and "no per-token MLP intermediate width for qwen35_4b" in none["problem"] and "neither an MoE key" in none["problem"] and "ARCH_DIMS_BY_MODEL_KEY" in none["problem"]
    half = pf.mlp_intermediate_width("synth_moe", {"num_experts": 64, "moe_intermediate_size": 512})
    assert half["elements"] is None and "num_experts_per_tok" in half["problem"]
    other = pf.mlp_intermediate_width("qwen35_4b_demo", {**DEMO_CFG, "intermediate_size": 10240})
    assert other["elements"] is None and "9,216" in other["problem"] and "10,240" in other["problem"] and "another checkpoint" in other["problem"]
    assert pf.mlp_intermediate_width("qwen35_4b_demo", None)["elements"] == 9_216                        # the table alone (no config to disagree)


def test_check_geometry_sizes_the_warmup_term_of_an_moe_config_and_refuses_a_width_it_cannot_derive(tmp_path, monkeypatch):
    """check_geometry over an MoE config shape with no dims-table entry (the width read from the config by architecture): the memory sum is
    computed (no KeyError), the width and its source are recorded, the certified per-set module widths ride beside the derivation; a config
    that yields no width refuses by name with the sum recorded as not computed. The derivation itself is pinned by the tests above; here it
    is held to a fixed served S."""
    import json
    snap = tmp_path / "snap"; snap.mkdir()
    (snap / "config.json").write_text(json.dumps(MOE_CFG))
    (snap / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 20_000_000_000}, "weight_map": {}}))
    S = 5
    geo = {"model_key": "synth_moe", "window": WINDOW, "served_S": S, "identity_S": 8, "batched_tokens": S * WINDOW, "kernel_adjustment": None,
           "module_set": "wo", "lora_max_module_dim": 4096}
    from sisteer.capeval.serving import geometry as g
    monkeypatch.setattr(pf, "geometry_expectation", lambda model_key, window, **kw: dict(geo))
    monkeypatch.setattr(g, "certified_module_dims", lambda model_key: {"wo": 4096, "mlp": 2048})     # the preflight imports g per call
    rec, probs = pf.check_geometry("synth_moe", WINDOW, snap)
    assert probs == [] and rec["certified_module_dims"] == {"wo": 4096, "mlp": 2048}
    mem = rec["memory"]
    toks = S * WINDOW
    kv_bpt = 2 * 8 * 4 * 128 * 2                 # 2 (K, V) x the config's 8 full-attention layers x 4 KV heads x head_dim 128 x 2 B
    assert mem["mlp_intermediate"]["elements"] == 2304 and mem["mlp_intermediate"]["source"] == "config:moe"
    assert abs(mem["mlp_intermediate_gib"] - toks * 2304 * 2 / 2**30) < 1e-9 and abs(mem["kv_gib"] - toks * kv_bpt / 2**30) < 1e-9
    assert rec["kv"]["bytes_per_token"] == kv_bpt and abs(mem["total_gib"] - (mem["weights_gib"] + mem["kv_gib"] + mem["mlp_intermediate_gib"])) < 1e-9
    # a config that yields no width: refused by name, the sum not computed, nothing raised
    (snap / "config.json").write_text(json.dumps({"num_hidden_layers": 24, "num_attention_heads": 12, "num_key_value_heads": 4, "head_dim": 128, "hidden_size": 1536}))
    rec2, probs2 = pf.check_geometry("some_model", WINDOW, snap)
    assert len(probs2) == 1 and probs2[0].startswith("geometry: no per-token MLP intermediate width for some_model") and rec2["memory"]["total_gib"] is None
    assert rec2["memory"]["mlp_intermediate_gib"] is None and rec2["memory"]["mlp_intermediate"]["problem"]
    # the source pin: the dense key is never indexed unguarded again
    src = (LANE / "padding_preflight.py").read_text()
    assert 'txt["intermediate_size"]' not in src and "mlp_intermediate_width(model_key, model_cfg)" in src
