"""T13 — pilot dependency order (twosi-model-sweeps §8; review fix P1-1).

``pod_pilot.sh --dry-run`` emits one machine-readable ``PILOT-STEP`` line per
step. These tests pin the §4 order and the artifact graph it fixes: the
naively-ordered pilot had a dependency CYCLE (the acceptance probe needs
``--pairs`` and a config batch size, but the production batch table is
Phase D and a pool needs the model on a GPU). Specifically:

- the eight steps run in the §4 order;
- every step's inputs exist among EARLIER steps' outputs (or are external
  files / declared synthetic self-made inputs);
- bootstrap's fit-driver gate runs on synthetic pairs at an explicit bs;
- the memprobe precedes the acceptance probe, which consumes the TEMPORARY
  pilot batch table + the pilot pool — never before either exists;
- no step consumes the production batch table, and no pilot artifact is
  written into ``configs/``.

CPU-only: --dry-run runs no GPU work and creates nothing.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

SCRIPT = REPO / "scripts" / "pod_pilot.sh"
CONFIG = "configs/two_si.yaml"
GEMMA_CONFIG = "configs/two_si_gemma4e4b.yaml"

STEP_RE = re.compile(
    r"^PILOT-STEP idx=(\d+) name=(\S+) inputs=(\S+) outputs=(\S+)$")

EXPECTED_ORDER = [
    "bootstrap",
    "onboard_full",
    "vllm_gate",
    "memprobe",
    "pilot_pool",
    "acceptance_probe",
    "training",
    "verdict",
]


def _dry_run(tmp_path, extra_env=None, config=CONFIG):
    import os

    env = dict(os.environ)
    env["WORKSPACE_DIR"] = str(tmp_path / "ws")
    env.update(extra_env or {})
    return subprocess.run(
        ["bash", str(SCRIPT), "--config", config,
         "--pilot-dir", str(tmp_path / "pilot"), "--dry-run"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=60)


@pytest.fixture(scope="module")
def plan(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pilot-dry")
    proc = _dry_run(tmp)
    assert proc.returncode == 0, proc.stderr
    steps = []
    for line in proc.stdout.splitlines():
        m = STEP_RE.match(line)
        if m:
            steps.append({
                "idx": int(m.group(1)),
                "name": m.group(2),
                "inputs": m.group(3).split(","),
                "outputs": m.group(4).split(","),
            })
    return {"steps": steps, "stdout": proc.stdout, "tmp": tmp}


def test_dry_run_emits_the_eight_steps_in_section4_order(plan):
    steps = plan["steps"]
    assert [s["name"] for s in steps] == EXPECTED_ORDER
    assert [s["idx"] for s in steps] == list(range(1, 9))
    assert "PILOT-PLAN-END steps=8" in plan["stdout"]


def test_every_input_exists_among_earlier_outputs_or_externally(plan):
    """The load-bearing property: no step consumes an artifact no earlier
    step produced. External inputs (the per-model config) must exist on
    disk; pilot artifacts must appear in a STRICTLY earlier step's outputs;
    synthetic:/explicit: pseudo-inputs are the step's own to make."""
    produced: set[str] = set()
    for s in plan["steps"]:
        for inp in s["inputs"]:
            if inp.startswith(("synthetic:", "explicit:")):
                continue
            if inp in produced:
                continue
            p = Path(inp) if Path(inp).is_absolute() else REPO / inp
            assert p.exists(), (
                f"step {s['idx']} ({s['name']}) input {inp!r} is neither an "
                f"existing external file nor an earlier step's output")
        produced.update(s["outputs"])


def test_probe_never_before_the_pilot_pool(plan):
    by_name = {s["name"]: s for s in plan["steps"]}
    pool, probe = by_name["pilot_pool"], by_name["acceptance_probe"]
    assert pool["idx"] < probe["idx"]
    pool_jsonl = [o for o in pool["outputs"] if o.endswith("pilot_pairs.jsonl")]
    assert pool_jsonl, "pilot_pool must output the pilot pairs jsonl"
    assert pool_jsonl[0] in probe["inputs"], (
        "the acceptance probe must consume the pilot pool")


def test_memprobe_before_the_probe_and_its_table_consumed(plan):
    by_name = {s["name"]: s for s in plan["steps"]}
    mem, probe = by_name["memprobe"], by_name["acceptance_probe"]
    assert mem["idx"] < probe["idx"]
    tables = [o for o in mem["outputs"] if "pilot_batch_table" in o]
    assert tables, "memprobe must output the TEMPORARY pilot batch table"
    assert any(t in probe["inputs"] for t in tables) or any(
        "pilot_config" in i for i in probe["inputs"]), (
        "the probe must run at the measured (pilot-table) batch size")


def test_no_step_consumes_the_production_batch_table(plan):
    """The pilot's whole point is running BEFORE Phase D measures the real
    table — nothing here may touch the committed measured table."""
    assert "two_si_budget_measured" not in plan["stdout"]
    for s in plan["steps"]:
        for inp in s["inputs"]:
            assert "budget_measured" not in inp, (s["name"], inp)


def test_no_pilot_artifact_lands_in_configs(plan):
    """The pilot batch table and derived configs are TEMPORARY: every output
    lives under the pilot dir, never the repo's configs/."""
    for s in plan["steps"]:
        for out in s["outputs"]:
            p = Path(out)
            assert not (not p.is_absolute() and out.startswith("configs/")), (
                s["name"], out)
            assert "pilot" in out, (
                f"{s['name']} output {out!r} is not under the pilot dir")


def test_bootstrap_gate_declares_synthetic_pairs_and_explicit_bs(plan):
    boot = plan["steps"][0]
    assert boot["name"] == "bootstrap"
    assert "synthetic:pairs" in boot["inputs"]
    assert "explicit:bs=2" in boot["inputs"]
    # The per-model config is the only real file bootstrap consumes.
    assert CONFIG in boot["inputs"]


def test_training_gated_on_probe_and_verdict_gated_on_training(plan):
    by_name = {s["name"]: s for s in plan["steps"]}
    probe_out = by_name["acceptance_probe"]["outputs"]
    assert any(o in by_name["training"]["inputs"] for o in probe_out), (
        "the 200-300-step training must not start on an unaccepted pod")
    train_out = by_name["training"]["outputs"]
    assert any(o in by_name["verdict"]["inputs"] for o in train_out)


def test_training_declares_the_sustained_duty_tripwire(plan):
    """The step-7 duty gate (gpu-util directive, 2026-08-25): training must
    OUTPUT the sustained-duty measurement (dmon SM% over a >=20 s stepping
    window, mean asserted >= PILOT_MIN_DUTY) and the verdict must consume
    it — a pilot that never measures sustained duty cannot certify a pod
    for the 16-GPU fan-out (the 33-45%-duty ragged-shape/cuDNN-SDPA case
    is exactly what the 1-GPU pilot exists to catch)."""
    by_name = {s["name"]: s for s in plan["steps"]}
    duty = [o for o in by_name["training"]["outputs"] if "duty" in o]
    assert duty, "training must output the sustained-duty measurement"
    assert any(d in by_name["verdict"]["inputs"] for d in duty), (
        "the verdict must embed the sustained-duty measurement")


def test_smoke_bs_env_is_carried_into_the_declared_plan(tmp_path):
    proc = _dry_run(tmp_path, extra_env={"SMOKE_BS": "4"})
    assert proc.returncode == 0, proc.stderr
    assert "explicit:bs=4" in proc.stdout


def test_missing_config_fails_loudly(tmp_path):
    import os

    env = dict(os.environ)
    env["WORKSPACE_DIR"] = str(tmp_path / "ws")
    proc = subprocess.run(
        ["bash", str(SCRIPT), "--config", "configs/no_such_model.yaml",
         "--dry-run"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0
    assert "not found" in proc.stderr


# --------------------------------------------------------------------------- #
# Parser-level integration (review fix F10): the dry-run declarations used to  #
# advertise a CLI (--model-config/--mode) onboard_model_smoke.py did not       #
# define, so T13 passed while the real step 2 crashed at argparse. These tests #
# therefore take the EXACT command text the pilot emits and parse/run it       #
# against the real CLIs — mocked model/tokenizer ops, real argument parsing.   #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def gemma_cmds(tmp_path_factory):
    """Step name -> the exact ``cmd:`` line the pilot prints for it, from a
    dry-run against a real PER-MODEL config (the pilot's production
    spelling — two_si.yaml's Llama pin is not a MODEL_SPECS entry, so the
    onboarding steps resolve only against a per-model config)."""
    tmp = tmp_path_factory.mktemp("pilot-dry-gemma")
    proc = _dry_run(tmp, config=GEMMA_CONFIG)
    assert proc.returncode == 0, proc.stderr
    cmds: dict[str, str] = {}
    lines = proc.stdout.splitlines()
    for i, line in enumerate(lines):
        m = STEP_RE.match(line)
        if m and i + 1 < len(lines) and lines[i + 1].startswith("  cmd: "):
            cmds[m.group(2)] = lines[i + 1][len("  cmd: "):]
    assert set(cmds) == set(EXPECTED_ORDER)
    return cmds


def _argv_after(cmd: str, script_name: str) -> list[str]:
    assert script_name in cmd, cmd
    return cmd.split(script_name, 1)[1].split()


def test_pilot_onboard_commands_parse_and_run_under_mocks(
        gemma_cmds, monkeypatch):
    """Both onboarding commands the pilot generates must PARSE AND RUN
    (exit 0 — the gate the .ok artifacts hang on), not merely print."""
    from test_model_onboarding import mock_onboard_checks

    monkeypatch.chdir(REPO)  # the emitted config path is repo-relative
    oms, calls, specs = mock_onboard_checks(monkeypatch)

    rc = oms.main(_argv_after(gemma_cmds["onboard_full"],
                              "onboard_model_smoke.py"))
    assert rc == 0, "the pilot's onboard_full command must run green"
    assert "model_load" in calls, "step 2 is the FULL tier (real weights)"
    assert "vllm" not in calls
    assert specs and specs[0].slug == "gemma4e4b", (
        "--model-config must resolve the model from the config's pins")

    calls.clear()
    rc = oms.main(_argv_after(gemma_cmds["vllm_gate"],
                              "onboard_model_smoke.py"))
    assert rc == 0, "the pilot's vllm_gate command must run green"
    assert calls == ["vllm"], (
        "step 3 runs in the DEDICATED vllm venv: only the vLLM gate — no "
        "lockfile-pin checks, no double weight load")


def test_pilot_probe_command_parses_against_the_probe_cli(gemma_cmds):
    import pod_acceptance_probe as pap

    argv = _argv_after(gemma_cmds["acceptance_probe"],
                       "pod_acceptance_probe.py")
    args = pap.build_parser().parse_args(argv)
    assert args.si == "si_constitution"
    assert args.arm == "lora_r1_all"
    assert args.config.endswith("pilot_config.yaml"), (
        "the probe must run at the MEASURED pilot-table batch size")


def test_pilot_pool_command_is_the_agreed_fixd_cli(gemma_cmds):
    """Step 5 is coded against the FIX-D generate-pos CLI: --config <cfg>
    --si si_constitution --max-prompts <N> --pilot-pairs-out <jsonl>."""
    cmd = gemma_cmds["pilot_pool"]
    assert "generate-pos" in cmd
    assert f"--config {GEMMA_CONFIG}" in cmd
    assert "--si si_constitution" in cmd
    assert "--max-prompts" in cmd
    assert "--pilot-pairs-out" in cmd


def test_pool_generate_pos_defines_the_pilot_flags(capsys, monkeypatch):
    """Dependency marker: FIX-D is landing --max-prompts/--pilot-pairs-out
    on generate-pos. SKIPS loudly until it does; once landed, pins that the
    pilot's flags actually parse there."""
    import build_two_si_pool as pool

    monkeypatch.setattr(sys, "argv",
                        ["build_two_si_pool.py", "generate-pos", "--help"])
    with pytest.raises(SystemExit):
        pool.main()
    text = capsys.readouterr().out
    for flag in ("--max-prompts", "--pilot-pairs-out"):
        if flag not in text:
            pytest.skip(
                f"FIX-D dependency not landed yet: generate-pos defines no "
                f"{flag}; pod_pilot.sh step 5 is coded against it "
                "(PILOT_POOL_CMD overrides if the landed CLI differs)")


def test_trainer_train_defines_profile(capsys, monkeypatch):
    """Dependency marker: the driver owner is landing --profile on `train`.
    SKIPS loudly until it does; the pilot emits exactly
    ``train ... --profile <dir>`` (step 7's TB trace)."""
    import train_two_si_objects as t2

    monkeypatch.setattr(sys, "argv",
                        ["train_two_si_objects.py", "train", "--help"])
    with pytest.raises(SystemExit):
        t2.main()
    if "--profile" not in capsys.readouterr().out:
        pytest.skip(
            "driver-owner dependency not landed yet: `train` defines no "
            "--profile; pod_pilot.sh step 7 emits `train ... --profile "
            "<dir>`")


def test_pilot_training_command_declares_profile(gemma_cmds):
    assert "--profile" in gemma_cmds["training"]


# --------------------------------------------------------------------------- #
# Script-level pins for the shell-only fixes (F11 duty window, F13 privacy,    #
# bootstrap arm_gpus) — the behaviors live in bash, so the pure logic is       #
# unit-tested in tests/test_pod_probe_units.py and these pin the wiring.       #
# --------------------------------------------------------------------------- #


def test_pilot_duty_window_rides_the_probes_step_classifier():
    """F11: the duty window opens on a true OPTIMIZER-STEP record and its
    verdict comes from pod_acceptance_probe's shared helpers (one step-line
    classifier for pilot and probe); step advance across the window is
    recorded; trainer liveness comes from /proc state, never kill -0 on the
    trainer (a zombie passes kill -0)."""
    src = SCRIPT.read_text()
    assert "from pod_acceptance_probe import count_step_lines" in src
    assert ("from pod_acceptance_probe import duty_verdict, "
            "parse_dmon_sm_max" in src)
    assert "STEPS_BEFORE" in src and "STEPS_AFTER" in src
    assert "/proc/$TRAIN_PID/stat" in src
    assert 'kill -0 "$TRAIN_PID"' not in src


def test_pilot_writes_are_owner_only():
    """F13: raw pairs/completions land on /workspace, readable by other
    users on shared mounts — the pilot sets a private umask before anything
    is created and re-enforces modes on pre-existing paths."""
    src = SCRIPT.read_text()
    assert re.search(r"^umask 077$", src, re.M)
    assert src.index("umask 077") < src.index('mkdir -p "$PILOT_DIR"')
    assert 'chmod 700 "$PILOT_DIR"' in src
    assert 'chmod 600 "$A_POOL"' in src


def test_bootstrap_smoke_never_requests_row_sharding():
    """The driver refuses arm_gpus > 1 loudly (row-sharding is off; the
    scheduler packs arms 1-per-GPU) — the bootstrap smoke must not set it
    to the pod's GPU count."""
    src = (REPO / "scripts" / "runpod_bootstrap.sh").read_text()
    assert '"arm_gpus": {"diffmean": 1}' in src
