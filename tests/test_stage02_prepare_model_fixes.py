"""Stage 02 (`reproduce/02_prepare_model.sh`) — pinned on its `--dry-run` lines against the
NEUTRAL site example (no cluster, no ssh, nothing written outside the state dir):

* `--smoke-mode vllm` is a GPU srun in the SERVING venv (the requirements-hash vllm venv
  scripts/sbatch/serving_probe.sbatch builds, bootstrapped on the node from configs/capeval/vllm-requirements.txt),
  NEVER the prebuilt fit venv; `--smoke-mode full` stays on the fit venv; the vllm line runs the smoke's explicit `--mode
  vllm` tier (which FAILS without vllm — tests/test_onboard_smoke_native_launch.py).
* the CPU steps honour `SISTEER_CPU_SRUN` exactly as 04_fit_objects.sh does (set = your wrapper verbatim, even
  empty; unset = the site's dev/CPU partition line), and no remote call bypasses `repro_ssh` (a bare `ssh` used to read the
  memprobe record).
* the tokenizer staging line passes `--pins <checkout>/corpora/TOKENIZERS.json` so the vendored
  CLI verifies repo, revision, file set and every sha256 before copying.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, REPRO, VENV_PY, assert_no_tokens, run_stage  # noqa: E402

STAGE = "02_prepare_model.sh"
MODEL = "qwen35_4b"


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    root = tmp_path_factory.mktemp("s02")
    site = root / "site"
    shutil.copytree(REPRO / "site.example", site)
    with open(site / "config.env", "a") as fh:
        fh.write(f"\nSISTEER_STATE_DIR={root / 'state'}\nSISTEER_PYTHON={VENV_PY}\nSISTEER_REGISTRY_ROOT={root / 'registry'}\n")
    return site


def _dry(site, *args, env=None):
    r = run_stage(STAGE, "--dry-run", "--model", MODEL, *args, site=site, env=env, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    # the runner's OWN paths (this checkout, its interpreter, the pytest tmp root, HOME) are not hard-coded identifiers
    scan = (out.replace(str(REPO), "<repo>").replace(VENV_PY, "<python>").replace(str(site.parent), "<tmp>")
               .replace(os.path.expanduser("~"), "<home>"))
    assert_no_tokens(scan, STAGE)
    return out


def _ssh_lines(out):
    return [ln for ln in out.splitlines() if ln.startswith("DRY-RUN: ssh ")]


def test_vllm_smoke_runs_in_the_serving_venv_as_a_gpu_srun_never_the_fit_venv(site):
    out = _dry(site, "--step", "smoke", "--smoke-mode", "vllm")
    gpu = [ln for ln in _ssh_lines(out) if "--gres=gpu:1" in ln]
    assert len(gpu) == 1, _ssh_lines(out)
    line = gpu[0]
    # the smoke's EXPLICIT tier, from the serving venv's interpreter with `vllm` on PATH for launch_server
    assert "scripts/onboard_model_smoke.py" in line and "--mode vllm" in line
    assert "$VENV/bin/python scripts/onboard_model_smoke.py" in line and "PATH=$VENV/bin:$PATH" in line
    # the venv is the lanes' requirements-hash one, bootstrapped by the same recipe (hash of the requirements bytes,
    # sentinel + flock, pinned --exclude-newer), and its vllm import is asserted before the gate runs
    assert "configs/capeval/vllm-requirements.txt" in line and "sisteer-petri-vllm-${SREQ:0:8}" in line
    assert "uv venv --python 3.12" in line and "--exclude-newer 2026-08-03T00:00:00Z" in line and "flock -x 9" in line
    # peft rides the import guard with vllm and torch: REQUIRED for the probe's HF+PEFT parity check
    assert re.search(r'python" -c "import vllm, torch, peft', line), line
    # never the prebuilt fit venv on this line
    assert "envs/sdfbase-fit/bin/python" not in line
    # the fit venv is still what `full` runs on
    full = [ln for ln in _ssh_lines(_dry(site, "--step", "smoke", "--smoke-mode", "full")) if "--gres=gpu:1" in ln]
    assert len(full) == 1 and "envs/sdfbase-fit/bin/python scripts/onboard_model_smoke.py" in full[0] and "--mode full" in full[0]
    assert "sisteer-petri-vllm" not in full[0]


def test_cpu_steps_honour_sisteer_cpu_srun_like_stage_04(site):
    default = _ssh_lines(_dry(site, "--step", "pools"))
    assert default and all("--job-name=D_$USER-02prep" in ln for ln in default), default
    custom = _ssh_lines(_dry(site, "--step", "pools", env={"SISTEER_CPU_SRUN": "srun --partition=cpu_only --qos=cpu_q -c 2"}))
    assert custom and all("srun --partition=cpu_only --qos=cpu_q -c 2 bash -c" in ln for ln in custom), custom
    assert not any("D_$USER-02prep" in ln for ln in custom)
    # SET but EMPTY = no wrapper at all (04_fit_objects.sh's `${SISTEER_CPU_SRUN-<default>}` semantics)
    empty = _ssh_lines(_dry(site, "--step", "pools", env={"SISTEER_CPU_SRUN": ""}))
    assert empty and not any("srun" in ln for ln in empty), empty
    # the tokenizer staging line (a CPU srun too) follows the same rule
    corp = [ln for ln in _ssh_lines(_dry(site, "--step", "corpora", env={"SISTEER_CPU_SRUN": "srun -p cpu_only"})) if "stage-tokenizers" in ln]
    assert len(corp) == 1 and "srun -p cpu_only uv run --directory corpus_generator" in corp[0]


def test_no_bare_ssh_bypasses_repro_ssh():
    text = (REPRO / STAGE).read_text()
    assert re.search(r"^[^#\n]*\bssh -o BatchMode", text, re.M) is None, "a remote call bypasses repro_ssh"
    assert "repro_ssh \"$PRIMARY\" \"cat $STATE/memprobe.$arm.json\"" in text


def test_tokenizer_staging_passes_the_pins_file(site):
    out = _dry(site, "--step", "corpora")
    lines = [ln for ln in _ssh_lines(out) if "corpusgen stage-tokenizers" in ln]
    assert len(lines) == 1, _ssh_lines(out)
    line = lines[0]
    # --pins names the CHECKOUT's committed pins (absolute: `uv run --directory corpus_generator` runs from corpus/), beside the
    # per-family --revision pins and the offline --hf-home
    m = re.search(r"--pins (\S+)/corpora/TOKENIZERS\.json", line)
    assert m, line
    assert "/scratch/example/shared/sisteer-eval-repo/" in m.group(1) and "--hf-home /scratch/example/shared/hf_cache" in line
    assert line.count("--revision ") == 3
    assert "--pins" in line.split("--revision")[0]   # the pins ride before the revisions (one argv shape)
    assert (REPO / "corpora/TOKENIZERS.json").is_file()
