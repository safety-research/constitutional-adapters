"""Hygiene pins for the sdf-base-steering Slurm fit lane's sbatch surface
(2026-09-06): the two scripts under scripts/sbatch/ that put the campaign's
fit cells on the Slurm clusters. Static text checks — the cluster hard
rules that have each caused a real outage (no ``--export=``, no ``--mem=0``,
never ``dev`` in sbatch, no manual ``CUDA_VISIBLE_DEVICES``), the lane's
preemption-tolerance contract (positional attempt counter, STOP sentinel,
self-resubmit carrying the sha pin, checkpoint/resume through the trainer),
the identity plumbing the Modal parity depends on (``SISTEER_VOL_ROOT``,
``--expect-config-sha`` via the driver, offline W&B with the Modal-form run
name), and the RESULT + mirror-figs EXIT path."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SBATCH_DIR = REPO / "scripts" / "sbatch"
FIT = SBATCH_DIR / "sdfbase_fit_cell.sbatch"
ENV = SBATCH_DIR / "sdfbase_fit_env.sbatch"


def _hard_rules(txt: str) -> None:
    assert not re.search(r"^#SBATCH --export", txt, re.M)     # rule 1
    assert "--export" not in txt.replace("never --export", "").replace(
        "NEVER --export", "")                                 # nor in the body
    assert not re.search(r"^#SBATCH --mem=0", txt, re.M)      # rule 2
    assert "CUDA_VISIBLE_DEVICES" not in txt                  # rule 3
    assert not re.search(r"^#SBATCH --qos=dev", txt, re.M)    # dev is srun-only
    assert re.search(r"^#SBATCH --time=", txt, re.M)          # never the 24h default
    assert re.search(r"^#SBATCH --open-mode=append", txt, re.M)
    assert re.search(r"^#SBATCH --qos=high$", txt, re.M)


def test_both_scripts_parse():
    for p in (FIT, ENV):
        subprocess.run(["bash", "-n", str(p)], check=True)


def test_fit_cell_sbatch_hygiene():
    txt = FIT.read_text()
    _hard_rules(txt)
    assert re.search(r"^#SBATCH --gres=gpu:1$", txt, re.M)
    assert re.search(r"^#SBATCH --mem=200G$", txt, re.M)
    assert re.search(r"^#SBATCH --cpus-per-task=16$", txt, re.M)
    assert re.search(r"^#SBATCH --no-requeue$", txt, re.M)   # we resubmit ourselves
    assert re.search(r"^#SBATCH --signal=B:TERM@\d+$", txt, re.M)
    # the workload is srun-wrapped, backgrounded and waited on (the trap
    # must run inside the grace window)
    assert re.search(r"^srun --ntasks=1 .*slurm_fit_cell\.py", txt, re.M)
    assert 'SRUN_PID=$!' in txt and 'wait "$SRUN_PID"' in txt
    # cluster env lines
    for line in ('source "$SHARED/scripts/hf_env.sh"',
                 "export HF_HUB_OFFLINE=1",
                 "export NCCL_NVLS_ENABLE=0",
                 'export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"'):
        assert line in txt, line
    # the shared root and the escalation QoS come from the job env (the front door's, from the site's cluster block)
    assert 'SHARED="$SISTEER_SHARED_ROOT"' in txt and 'CLUSTER_QOS="${SISTEER_CLUSTER_QOS:-high}"' in txt
    assert '[ -d "/workspace' not in txt and "SISTEER_SHARED_ROOT is unset" in txt
    # NCCL's socket interface is the SITE's (its nccl.conf, else NCCL's own detection): neither the fit nor the parity smoke
    # exports or picks one (an exported name overrides the site's setting; one the node lacks fails the first collective), and
    # the log line records an inherited value without tripping `set -u`
    smoke = (SBATCH_DIR / "ddp_parity_smoke.sbatch").read_text()
    for t in (txt, smoke):
        assert "NCCL_SOCKET_IFNAME=" not in t and "nccl_ifname" not in t
        assert "nccl_if=${NCCL_SOCKET_IFNAME:-unset}" in t


def test_fit_cell_sbatch_reads_the_width_from_the_config_and_sizes_the_step():
    """The config's top-level `fit_width:` line is THE width; the job
    refuses an allocation that differs (SLURM_GPUS_ON_NODE, else the cgroup's
    devices) and a job env that disagrees; the wrapper gets --fit-width and
    launches torchrun inside the ONE srun task; the header stays the width-1
    allocation (test_fit_cell_sbatch_hygiene) — a wide submission adds the
    flags on the sbatch line (fit_width_flags, empty at width 1)."""
    txt = FIT.read_text()
    assert re.search(r'^FIT_WIDTH="\$\(sed -n .s/\^fit_width: \*.*"\$CONFIG".*; FIT_WIDTH="\$\{FIT_WIDTH:-1\}"$', txt, re.M)
    assert re.search(r"^fit_width_flags\(\) \{", txt, re.M)
    assert '[ "$n" -gt 1 ] 2>/dev/null || return 0' in txt
    assert "--gres=gpu:%d --cpus-per-task=%d --mem=%dG" in txt and '"$((16 * n))" "$((128 * n))"' in txt
    assert 'NGPU="${SLURM_GPUS_ON_NODE:-}"' in txt and 'nvidia-smi -L' in txt
    assert '[ "$NGPU" = "$FIT_WIDTH" ] ||' in txt and 'exit 3; }' in txt
    assert '[ -z "${SDFBASE_FIT_WIDTH:-}" ] || [ "$SDFBASE_FIT_WIDTH" = "$FIT_WIDTH" ] ||' in txt
    assert re.search(r'^srun --ntasks=1 "\$PY" "\$REPO/scripts/sdfbase/slurm_fit_cell\.py" \\$', txt, re.M)
    assert '--python "$PY" --fit-width "$FIT_WIDTH" &' in txt
    assert "torchrun" in txt and "nproc_per_node" in txt      # the how is stated where the step runs


def test_fit_cell_escalate_carries_the_width_flags_and_env(tmp_path):
    """The self-resubmit of a width-4 job asks for the width-4 allocation and
    SDFBASE_FIT_WIDTH=4; a width-1 job's successor line is unchanged."""
    out = "/workspace/nobodyx/exp/logs/reseed-fit-s1-mixfull-psr_4242.out"
    argv, envline = _run_escalate(FIT, tmp_path / "w4", stdout_path=out, positionals=FIT_POSITIONALS, extra_env="FIT_WIDTH=4")
    assert "--gres=gpu:4 --cpus-per-task=64 --mem=512G" in argv, argv
    assert argv.index("--gres=gpu:4") < argv.index("--comment=sha:") and "--export" not in argv and "--mem=0" not in argv
    assert envline == "ENV=sdfbase-fit-tf515 WIDTH=4"
    argv1, envline1 = _run_escalate(FIT, tmp_path / "w1", stdout_path=out, positionals=FIT_POSITIONALS, extra_env="FIT_WIDTH=1")
    assert "--gres" not in argv1 and "--cpus-per-task" not in argv1 and "--mem=" not in argv1
    assert envline1 == "ENV=sdfbase-fit-tf515 WIDTH=1"


def test_parity_smoke_sbatch_tees_rank_0_both_streams():
    """The parity smoke's torchrun leg carries the same rank-0 tee spec as slurm_fit_cell.py: "0:3",
    never a bare "0" (Std.NONE)."""
    from torch.distributed.elastic.multiprocessing.api import Std

    txt = (SBATCH_DIR / "ddp_parity_smoke.sbatch").read_text()
    m = re.search(r"--redirects 3 --tee (\S+) --log-dir", txt)
    assert m, "the smoke's torchrun line"
    assert m.group(1) == "0:3" and Std.from_str(m.group(1)) == {0: Std.ALL}
    assert "--tee 0 " not in txt and "--tee 0\n" not in txt
    _hard_rules_smoke = txt
    assert "--export" not in _hard_rules_smoke.replace("never --export", "") and "--mem=0" not in _hard_rules_smoke
    subprocess.run(["bash", "-n", str(SBATCH_DIR / "ddp_parity_smoke.sbatch")], check=True)


def test_fit_cell_sbatch_preemption_contract():
    txt = FIT.read_text()
    # positional attempt counter (never --export), STOP sentinel, cap
    assert 'ATTEMPT="${5:-1}"' in txt and 'MAX_ATTEMPTS="${6:-3}"' in txt
    assert '[ -e "$STOP_DIR/STOP" ]' in txt
    assert '[ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ]' in txt
    # the resubmit re-passes EVERY positional + the sha pin + the time limit
    assert re.search(r'sbatch --qos="\$ESCALATE_QOS" --job-name="\$RUN_NAME" '
                     r'\$\{TIMELIMIT:\+--time="\$TIMELIMIT"\}', txt)
    # pin-fix: the successor carries the job's WHOLE comment (sha field + the funnel's nonce), never a re-minted `sha:$HEAD_SHA` alone
    assert '--comment="$COMMENT" "$SCRIPT_PATH"' in txt and '--comment="sha:$HEAD_SHA"' not in txt
    assert '"$MODEL_KEY" "$SI" "$ARM" "$BANK" "$((ATTEMPT + 1))" "$MAX_ATTEMPTS"' in txt
    # $0 is a spool copy: the real path comes from scontrol
    assert "scontrol show job" in txt and "Command=" in txt
    assert "trap on_term SIGTERM SIGINT SIGQUIT" in txt
    # checkpoint/resume rides the trainer's own --checkpoint-dir (the driver
    # passes it); the sbatch never invents a second resume mechanism
    assert "--checkpoint-every 100" in txt
    assert "resume_from_checkpoint" not in txt


def test_fit_cell_sbatch_identity_plumbing():
    txt = FIT.read_text()
    # the config's /vol/... held-out path resolves under $SHARED/vol
    assert 'export SISTEER_VOL_ROOT="$SHARED/vol"' in txt
    assert 'PAIRS="$SHARED/vol/pool/$POOL_KEY/pairs_${SI}.jsonl"' in txt
    # the pool dir is READ from the config's held-out path — a `_full` key
    # and its further siblings share the parent's dir; the former suffix
    # strip (`${MODEL_KEY%_full}`) read the layout off the key's spelling
    # and would have sent `_full_r2t12` to a pool dir that does not exist
    assert 'POOL_KEY="${MODEL_KEY%_full}"' not in txt
    assert re.search(r'^POOL_KEY="\$\(sed -n .*heldout_pool: /vol/pool/'
                     r'.*"\$CONFIG".*\)"$', txt, re.M)
    assert '[ -n "$POOL_KEY" ] ||' in txt
    # submit-time sha pin, fail-closed
    # pin-fix: the pin is the `sha` FIELD found by name among `,`/`;`-separated key:value fields (the funnel appends `,nonce:<n>`)
    assert 'sha:*) PIN="${_f#sha:}"' in txt and 'PIN="${COMMENT#sha:}"' not in txt
    assert '[ "$PIN" = "$HEAD_SHA" ] ||' in txt
    # offline W&B under the site's own entity (no default entity: none -> W&B off), Modal-form run name via TT_WANDB_FIT
    assert "export WANDB_MODE=offline" in txt
    assert 'export WANDB_ENTITY="${WANDB_ENTITY:-${SISTEER_WANDB_ENTITY:-}}"' in txt and "export WANDB_MODE=disabled" in txt
    assert 'export TT_WANDB_FIT="{\\"name\\": \\"fit-${MODEL_SLUG}-${SI}-${ARM}-all\\"' in txt
    # no key of any kind is read or written on the cluster
    assert "WANDB_API_KEY" not in txt and "HF_TOKEN=" not in txt
    # preflight uses -s (present AND non-empty) on every external input
    assert '[ -s "$f" ] ||' in txt
    for f in ('"$CONFIG"', '"$PAIRS"', '"$HELDOUT"',
              '"$SNAP"', '"$VENV/BUILD-COMPLETE"'):
        assert f in txt, f
    # the manifest: PRESENCE in either spelling (the pool builder ships the
    # stem-swapped one); validation — cell, pool_sha256 — is the wrapper's,
    # through the resolver the trainer shares, never a -s that skips an
    # empty file the trainer would hash (2026-09-10)
    assert '[ -e "$PAIRS.manifest.json" ] || [ -e "${PAIRS%.jsonl}.manifest.json" ]' in txt
    assert '[ -s "$PAIRS.manifest.json" ]' not in txt
    # bank layout = the volume's out/<key>/ layout, keyed by the CONFIG's
    # identity (fit_cell_identity out-key), never the pool key: configs
    # sharing a pool must never share a root (2026-09-10)
    assert 'OUT_ROOT="$SHARED/exp/objects/sdfbase/$BANK/$POOL_KEY"' not in txt
    assert re.search(r'^OUT_KEY="\$\(python3 "\$REPO/scripts/sdfbase/'
                     r'fit_cell_identity\.py" out-key --bank "\$BANK" '
                     r'--config "\$CONFIG"\)" \|\| OUT_KEY=""$', txt, re.M)
    assert '[ -n "$OUT_KEY" ] ||' in txt
    assert 'OUT_ROOT="$SHARED/exp/objects/sdfbase/$BANK/$OUT_KEY"' in txt
    assert 'CELL_DIR="$OUT_ROOT/cells_all/$CELL_ID"' in txt
    # the offline W&B dir follows the same namespace; the pool key names
    # INPUTS only
    assert 'WANDB_DIR="$SHARED/exp/wandb/$BANK/$OUT_KEY/$CELL_ID"' in txt
    assert '$BANK/$POOL_KEY' not in txt


def test_fit_cell_sbatch_result_and_mirror_on_exit():
    txt = FIT.read_text()
    assert "trap 'finish' EXIT" in txt
    assert 'echo "$line" > "$FIGS_DIR/RESULT"' in txt
    assert '[ -d "$CELL_DIR" ] && echo "$line" >> "$CELL_DIR/RESULT"' in txt
    assert '"$SHARED/scripts/mirror-figs.sh" "$FIGS_DIR"' in txt
    # the RESULT line names the outcome, the cell, the fingerprint and the commit
    for tok in ("state", "cell=$OUT_KEY/$CELL_ID", "fingerprint=$fp",
                "head=${HEAD_SHA:0:12}", "attempt=$ATTEMPT/$MAX_ATTEMPTS"):
        assert tok in txt, tok


def test_env_build_sbatch_hygiene_and_pins():
    txt = ENV.read_text()
    _hard_rules(txt)
    assert not re.search(r"^#SBATCH --gres", txt, re.M)   # CPU-only: no GPU budget
    assert re.search(r"^srun --ntasks=1 bash -c build_env$", txt, re.M)
    # the Modal image_nemotron recipe, pin for pin
    for pin in ("triton==3.7.1", "fla-core==0.5.2",
                "flash-linear-attention==0.5.2", "ninja==1.13.0",
                "causal-conv1d==1.6.2.post1", "mamba-ssm==2.3.2.post1",
                "CAUSAL_CONV1D_FORCE_BUILD=TRUE", "MAMBA_FORCE_BUILD=TRUE",
                "--no-build-isolation --no-deps mamba-ssm",
                "--extra-index-url https://download.pytorch.org/whl/cu128",
                "TORCH_CUDA_ARCH_LIST='9.0;10.0'"):
        assert pin in txt, pin
    # the bake assertions the image runs, and the requirements versions
    assert "from mamba_ssm.ops.triton.ssd_combined import" in txt
    for v in ('"torch": "2.11.0+cu128"', '"transformers": "5.15.0"',
              '"accelerate": "1.14.0"', '"wandb": "0.28.1"'):
        assert v in txt, v
    # no /tmp for GB-scale build intermediates; a positive completion artifact
    assert 'export TMPDIR="$SHARED/tmp/' in txt
    assert 'BUILD-COMPLETE' in txt


def test_env_pins_match_requirements_txt():
    """The version assertions the build runs are the lockfile export's pins
    (scripts/modal/requirements.txt = the Modal image's tree)."""
    req = (REPO / "scripts" / "modal" / "requirements.txt").read_text()
    for name, ver in (("torch", "2.11.0+cu128"), ("transformers", "5.15.0"),
                      ("accelerate", "1.14.0"), ("wandb", "0.28.1")):
        assert not re.search(r"^peft==", req, re.M)   # none pinned: none asserted
        assert re.search(rf"^{re.escape(name)}=={re.escape(ver)}\b", req,
                         re.M), (name, ver)


def test_fit_cell_sbatch_pool_key_is_read_from_the_config():
    """Run the sbatch's own POOL_KEY line against the committed configs:
    the frozen `_doc` lane, its `_full` sibling and the `_full_r2t12`
    refit (2026-09-07) all resolve to the parent's pool dir, nemotron's
    full lane to its retired ladder's dir; a file without the held-out
    line resolves to nothing, which the sbatch refuses on."""
    txt = FIT.read_text()
    line = re.search(r'^POOL_KEY="\$\(sed .*\)"$', txt, re.M).group(0)

    def pool_key(path: Path) -> str:
        return subprocess.run(
            ["bash", "-c", f'CONFIG="$1"; {line}; printf %s "$POOL_KEY"',
             "_", str(path)], capture_output=True, text=True,
            check=True).stdout

    cfgs = REPO / "configs"
    for name, want in (
            ("sdfbase_glm46vflash_doc_fits.yaml", "glm46vflash_doc"),
            ("sdfbase_glm46vflash_doc_full_fits.yaml", "glm46vflash_doc"),
            ("sdfbase_glm46vflash_doc_full_r2t12_fits.yaml",
             "glm46vflash_doc"),
            ("sdfbase_nemotron3nano4b_doc_full_fits.yaml",
             "nemotron3nano4b_doc")):      # the retired ladder lane's pool dir (RETIRED_POOL_PARENTS)
        assert pool_key(cfgs / name) == want, name
    assert pool_key(FIT) == ""


def _run_sbatch_line(pattern: str, **env) -> subprocess.CompletedProcess:
    """Run ONE assignment line of the fit sbatch (matched by ``pattern`` at
    line start) under bash with the named variables set; stdout is the
    variable the line assigns."""
    txt = FIT.read_text()
    line = re.search(pattern, txt, re.M).group(0)
    var = line.split("=", 1)[0]
    exports = " ".join(f"{k}={subprocess.list2cmdline([v])}"
                       for k, v in env.items())
    return subprocess.run(
        ["bash", "-c", f'{exports}; log() {{ echo "$*" >&2; }}; {line}; '
                       f'printf %s "${var}"'],
        capture_output=True, text=True)


def test_fit_cell_sbatch_out_key_is_the_configs_not_the_pools():
    """Run the sbatch's own OUT_KEY line (fit_cell_identity out-key, under
    the node's python3) against the committed configs and the banks on
    cluster A (2026-09-10): the two frozen `_full` lanes resolve to the
    pool-key dirs they were published under, dir for dir; the `_full_r2t12`
    refit sibling — same pool, same POOL_KEY — gets a root of ITS OWN; and the
    small glm config has no root in newmodels_full/ (its own key IS the frozen
    lane's dir), so the line assigns nothing and the sbatch refuses."""
    cfgs = REPO / "configs"
    cases = (
        ("newmodels_full", "sdfbase_glm46vflash_doc_full_fits.yaml",
         "glm46vflash_doc"),
        ("newmodels_full", "sdfbase_nemotron3nano4b_doc_full_fits.yaml",
         "nemotron3nano4b_doc"),
        ("newmodels_full", "sdfbase_glm46vflash_doc_full_r2t12_fits.yaml",
         "glm46vflash_doc_full_r2t12"),
        ("newmodels_full_gate", "sdfbase_glm46vflash_doc_full_fits.yaml",
         "glm46vflash_doc_full"),
    )
    for bank, name, want in cases:
        r = _run_sbatch_line(r'^OUT_KEY=.*$', REPO=str(REPO), BANK=bank,
                             CONFIG=str(cfgs / name))
        assert r.returncode == 0 and r.stdout == want, (bank, name, r)
    r = _run_sbatch_line(r'^OUT_KEY=.*$', REPO=str(REPO),
                         BANK="newmodels_full",
                         CONFIG=str(cfgs / "sdfbase_glm46vflash_doc_fits.yaml"))
    assert r.stdout == "" and "frozen namespace" in r.stderr, r
    # the two keys that share glm46vflash_doc's pool never share a root
    a = _run_sbatch_line(r'^OUT_KEY=.*$', REPO=str(REPO), BANK="newmodels_full",
                         CONFIG=str(cfgs / "sdfbase_glm46vflash_doc_full_fits.yaml"))
    b = _run_sbatch_line(r'^OUT_KEY=.*$', REPO=str(REPO), BANK="newmodels_full",
                         CONFIG=str(cfgs / "sdfbase_glm46vflash_doc_full_r2t12_fits.yaml"))
    assert a.stdout and b.stdout and a.stdout != b.stdout


# --------------------------------------------------------------------------- #
# qwen-base-seed-replicates (2026-09-15): the escalate() --output passthrough,   #
# the env-dir NAME knob (O6) on all three sbatch files, the derived sbatch's    #
# hygiene                                                                       #
# --------------------------------------------------------------------------- #

DERIVED = SBATCH_DIR / "sdfbase_derived_build.sbatch"
ENV_KNOB_TEXT = (
    'FIT_ENV_NAME="${SDFBASE_FIT_ENV_NAME:-$( [ -s "{bank}/FIT_ENV_NAME" ] && head -1 '
    '"{bank}/FIT_ENV_NAME" | tr -d \'[:space:]\' )}"\n'
    'FIT_ENV_NAME="${FIT_ENV_NAME:-sdfbase-fit}"\n'
    'VENV="${SDFBASE_FIT_ENV:-$SHARED/envs/$FIT_ENV_NAME}"\n')


def test_derived_sbatch_parses_and_keeps_the_hard_rules():
    subprocess.run(["bash", "-n", str(DERIVED)], check=True)
    txt = DERIVED.read_text()
    _hard_rules(txt)
    assert re.search(r"^#SBATCH --gres=gpu:1$", txt, re.M)
    assert re.search(r"^#SBATCH --no-requeue$", txt, re.M)
    assert "trap on_term SIGTERM SIGINT SIGQUIT" in txt
    assert 'case "$STAGE" in prep|calibrate|assess|finish) ;;' in txt


def _extract_function(txt: str, name: str) -> str:
    m = re.search(rf"^{name}\(\) \{{\n(.*?)^\}}\n", txt, re.M | re.S)
    assert m, name
    return m.group(0)


def _run_escalate(script: Path, tmp_path: Path, *, stdout_path: str, positionals: str,
                  extra_env: str = "") -> list[str]:
    """Run the sbatch's own escalate() under bash with a recording `sbatch`
    stub; returns the recorded sbatch lines (argv + the env the stub saw)."""
    txt = script.read_text()
    fn = _extract_function(txt, "escalate")
    helpers = _extract_function(txt, "fit_width_flags") if re.search(r"^fit_width_flags\(\) \{", txt, re.M) else ""
    tmp_path.mkdir(parents=True, exist_ok=True)
    log = tmp_path / "sbatch.log"
    (tmp_path / "script.sbatch").write_text("#!/bin/bash\n")
    prog = (
        f'sbatch() {{ printf "%s\\n" "$*" >> {log}; printf "ENV=%s WIDTH=%s\\n" "${{SDFBASE_FIT_ENV_NAME:-}}" "${{SDFBASE_FIT_WIDTH:-}}" >> {log}; }}\n'
        'log() { echo "$*" >&2; }\n'
        f'ESCALATE_QOS=high; STOP_DIR={tmp_path}/stop; ATTEMPT=1; MAX_ATTEMPTS=3; '
        f'SCRIPT_PATH={tmp_path}/script.sbatch; RUN_NAME=reseed-fit-test; TIMELIMIT=1-02:00:00; '
        f'HEAD_SHA={"e" * 40}; COMMENT=sha:{"e" * 40},nonce:{NONCE}; FIT_ENV_NAME=sdfbase-fit-tf515; STDOUT_PATH={subprocess.list2cmdline([stdout_path])}; '
        f'{positionals}; {extra_env}\n'
        f'{helpers}\n{fn}\nescalate\n')
    r = subprocess.run(["bash", "-c", prog], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return log.read_text().splitlines() if log.exists() else []


NONCE = "0123456789ab"      # the funnel's reservation nonce as the job's comment carries it (`sha:<sha>,nonce:<n>`)
FIT_POSITIONALS = 'MODEL_KEY=qwen35_4b_base_eb64_s1; SI=constv2-mix-full; ARM=psr_ce_all; BANK=reseed'
DERIVED_POSITIONALS = ('STAGE=calibrate; MODEL_KEY=qwen35_4b_base_eb64_s1; MIX_SI=constv2-mix-full; '
                       'CTL_SI=constv2ctl2-full; BANK=reseed; LR_MULT=0.1; OBJECTS=both; BUILD=chain')    # positional 9 since aac3d4079 (the requested objects) + 10 (the build)


def test_fit_cell_escalate_passes_its_stdout_as_output_and_the_env_name(tmp_path):
    """The self-resubmit carries this job's own resolved log path as
    --output (the header's $SHARED/exp/logs/ does not exist on
    cluster B) and the venv name as an env prefix — never --export="""
    txt = FIT.read_text()
    assert re.search(r'^STDOUT_PATH="\$\(scontrol show job "\$SLURM_JOB_ID" .*StdOut=', txt, re.M)
    assert '${STDOUT_PATH:+--output="$STDOUT_PATH"}' in txt
    assert 'SDFBASE_FIT_ENV_NAME="$FIT_ENV_NAME" SDFBASE_FIT_WIDTH="${FIT_WIDTH:-1}" \\\n  sbatch --qos="$ESCALATE_QOS"' in txt
    out = "/workspace/nobodyx/exp/logs/reseed-fit-s1-mixfull-psr_4242.out"
    lines = _run_escalate(FIT, tmp_path, stdout_path=out, positionals=FIT_POSITIONALS)
    assert len(lines) == 2
    argv, envline = lines
    assert f"--output={out}" in argv and "--qos=high" in argv and "--time=1-02:00:00" in argv
    assert argv.endswith(f"{tmp_path}/script.sbatch qwen35_4b_base_eb64_s1 constv2-mix-full psr_ce_all reseed 2 3")
    assert f"--comment=sha:{'e' * 40},nonce:{NONCE}" in argv and "--export" not in argv     # the FULL original comment rides the successor
    assert envline == "ENV=sdfbase-fit-tf515 WIDTH=1"
    # no StdOut resolved -> no --output (the header applies), nothing else changes
    lines = _run_escalate(FIT, tmp_path / "b", stdout_path="", positionals=FIT_POSITIONALS)
    assert "--output" not in lines[0] and lines[0].endswith("reseed 2 3")
    # STOP sentinel / attempt cap still stand down
    (tmp_path / "c" / "stop").mkdir(parents=True)
    (tmp_path / "c" / "stop" / "STOP").write_text("")
    assert _run_escalate(FIT, tmp_path / "c", stdout_path=out, positionals=FIT_POSITIONALS) == []


def test_derived_escalate_passes_output_env_name_and_its_multiplier(tmp_path):
    txt = DERIVED.read_text()
    assert '${STDOUT_PATH:+--output="$STDOUT_PATH"}' in txt
    out = "/workspace/nobodyx/exp/logs/reseed-derived-s1-calib-x0.1_99.out"
    lines = _run_escalate(DERIVED, tmp_path, stdout_path=out, positionals=DERIVED_POSITIONALS)
    argv, envline = lines
    assert f"--output={out}" in argv and envline == "ENV=sdfbase-fit-tf515 WIDTH="
    assert f"--comment=sha:{'e' * 40},nonce:{NONCE}" in argv                                 # the derived successor keeps the nonce too
    assert argv.endswith("script.sbatch calibrate qwen35_4b_base_eb64_s1 constv2-mix-full "
                         "constv2ctl2-full reseed 0.1 2 3 both chain")                               # … + the OBJECTS positional (aac3d4079) + the BUILD positional


def _run_knob(script: Path, tmp_path: Path, bank_expr: str, *, env_name: str | None = None,
              bank_file: str | None = None, full_path: str | None = None) -> str:
    """Run the sbatch's own FIT_ENV_NAME / VENV lines and print VENV."""
    txt = script.read_text()
    lines = [l for l in txt.splitlines()
             if l.startswith(('FIT_ENV_NAME="', 'VENV="'))]
    assert len(lines) == 3, lines
    shared = tmp_path / "shared"
    bank = tmp_path / "bank"
    bank.mkdir(parents=True, exist_ok=True)
    if bank_file is not None:
        (bank / "FIT_ENV_NAME").write_text(bank_file)
    pre = f'SHARED={shared}; BANK=reseed; ROOT={bank}; '
    exports = ""
    if env_name is not None:
        exports += f"SDFBASE_FIT_ENV_NAME={env_name} "
    if full_path is not None:
        exports += f"SDFBASE_FIT_ENV={full_path} "
    body = "\n".join(lines).replace('"$SHARED/exp/objects/sdfbase/$BANK/FIT_ENV_NAME"', f'"{bank}/FIT_ENV_NAME"')
    r = subprocess.run(["bash", "-c", f'{exports}{pre}\n{body}\nprintf %s "$VENV|$FIT_ENV_NAME"'],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


@pytest.mark.parametrize("script,bank_expr", [(FIT, "$SHARED/exp/objects/sdfbase/$BANK"),
                                               (DERIVED, "$ROOT")])
def test_env_dir_name_knob_defaults_to_todays_path(tmp_path, script, bank_expr):
    """O6: SDFBASE_FIT_ENV_NAME (implicit export) > the bank's FIT_ENV_NAME
    file > sdfbase-fit; SDFBASE_FIT_ENV (a full path) beats all. The default
    resolves to exactly today's $SHARED/envs/sdfbase-fit."""
    txt = script.read_text()
    assert ENV_KNOB_TEXT.replace("{bank}", bank_expr) in txt, script.name
    assert 'VENV="${SDFBASE_FIT_ENV:-$SHARED/envs/sdfbase-fit}"' not in txt
    shared = tmp_path / "shared"
    assert _run_knob(script, tmp_path, bank_expr) == f"{shared}/envs/sdfbase-fit|sdfbase-fit"
    assert _run_knob(script, tmp_path, bank_expr, env_name="sdfbase-fit-tf515") == \
        f"{shared}/envs/sdfbase-fit-tf515|sdfbase-fit-tf515"
    assert _run_knob(script, tmp_path, bank_expr, bank_file="sdfbase-fit-tf515\n") == \
        f"{shared}/envs/sdfbase-fit-tf515|sdfbase-fit-tf515"
    assert _run_knob(script, tmp_path, bank_expr, env_name="from-env", bank_file="from-file\n") == \
        f"{shared}/envs/from-env|from-env"
    assert _run_knob(script, tmp_path, bank_expr, bank_file="") == f"{shared}/envs/sdfbase-fit|sdfbase-fit"
    assert _run_knob(script, tmp_path, bank_expr, env_name="x", full_path="/elsewhere/venv") == \
        "/elsewhere/venv|x"
    # the preflight and the log line name the resolved env
    assert '"$VENV/BUILD-COMPLETE"' in txt and "env=$FIT_ENV_NAME ($VENV)" in txt


def test_env_build_sbatch_name_knob(tmp_path):
    """sdfbase_fit_env.sbatch builds INTO the named dir: positional 1 >
    SDFBASE_FIT_ENV_NAME > sdfbase-fit, build dir <name>.build beside it;
    today's default is byte-identical."""
    txt = ENV.read_text()
    lines = [l for l in txt.splitlines() if l.startswith(("export SDFBASE_FIT_ENV_NAME=",
                                                            "export SDFBASE_BUILD_DIR=",
                                                            "export SDFBASE_FIT_ENV="))]
    assert len(lines) == 3
    body = "\n".join(lines)

    def run(*positional, **env):
        exports = " ".join(f"{k}={v}" for k, v in env.items())
        r = subprocess.run(["bash", "-c", f'{exports} SHARED=/S\n{body}\n'
                                          'printf %s "$SDFBASE_FIT_ENV_NAME|$SDFBASE_BUILD_DIR|$SDFBASE_FIT_ENV"',
                            "_", *positional], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return r.stdout
    assert run() == "sdfbase-fit|/S/envs/sdfbase-fit.build|/S/envs/sdfbase-fit"
    assert run("sdfbase-fit-tf515") == "sdfbase-fit-tf515|/S/envs/sdfbase-fit-tf515.build|/S/envs/sdfbase-fit-tf515"
    assert run(SDFBASE_FIT_ENV_NAME="viaenv") == "viaenv|/S/envs/viaenv.build|/S/envs/viaenv"
    assert run("pos", SDFBASE_FIT_ENV_NAME="viaenv") == "pos|/S/envs/pos.build|/S/envs/pos"
    assert run(SDFBASE_FIT_ENV="/x/venv") == "sdfbase-fit|/S/envs/sdfbase-fit.build|/x/venv"
    _hard_rules(txt)


def test_derived_sbatch_reads_the_seed_from_the_config_default_zero(tmp_path):
    """--seed comes from train_kwargs.default.seed (the generator emits the
    line only for a declared seed), default 0 — byte-identical for every
    existing key — and rides with --holdout-frac 0.1 on capture-set, calibrate
    and distill; the RESULT line names seed=."""
    txt = DERIVED.read_text()
    line = re.search(r'^SEED="\$\(sed -n .*\)"; SEED="\$\{SEED:-0\}"$', txt, re.M).group(0)

    def seed_of(cfg_text: str) -> str:
        p = tmp_path / "cfg.yaml"
        p.write_text(cfg_text)
        return subprocess.run(["bash", "-c", f'CONFIG={p}; {line}; printf %s "$SEED"'],
                              capture_output=True, text=True, check=True).stdout
    assert seed_of("model: x\ntrain_kwargs:\n  default:\n    holdout_frac: 0.0\n") == "0"
    assert seed_of("# seed: 1 (train_kwargs.default.seed) — a SEED REPLICATE\n"
                   "train_kwargs:\n  default:\n    probe_n: null\n    seed: 2\n") == "2"
    assert seed_of("train_kwargs:\n  default:\n    seed: 13\n") == "13"
    # the committed configs carry no seed line except a declared replicate's
    for cfg in sorted((REPO / "configs").glob("sdfbase_*_fits.yaml")):
        got = seed_of(cfg.read_text())
        declared = re.search(r"^ {4}seed: (\d+)\s*$", cfg.read_text(), re.M)
        assert got == (declared.group(1) if declared else "0"), cfg.name
    assert 'SEED_ARGS=(--seed "$SEED" --holdout-frac "$HOLDOUT_FRAC")' in txt and "HOLDOUT_FRAC=0.1" in txt
    for stage in ("capture-set", "calibrate", "distill"):
        assert re.search(rf'{stage} "\$\{{RESID_COMMON\[@\]\}}"( "\$\{{RESID_MODEL\[@\]\}}")? "\$\{{SEED_ARGS\[@\]\}}"', txt), stage
    assert "pair=$POOL_KEY/${MIX_SI}-${CTL_SI} seed=$SEED bank=$BANK" in txt
