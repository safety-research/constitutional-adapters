# capeval sbatch scaffolding (sourced, not run).
#
# Sourced by every capeval_*.sbatch AFTER its #SBATCH block and BEFORE
# s2_init. It (1) points the stage2_common.sh extension variables at
# capeval's OWN shared checkout + venv (the peers' checkout and
# EXPECTED_SHA are NEVER touched — this file pins to
# ${SISTEER_PROJECT_ROOT}-capeval-repo, whose gate file is
# $S2_REPO_DIR/EXPECTED_SHA inside the capeval checkout), (2) exports the
# CAPEVAL_* cell-enumeration constants (the sbatch-side MIRROR of
# scripts/capeval_run.py enumerate_cells — one canonical copy in Python,
# pinned against this file by tests/test_capeval_driver.py, F15), and
# (3) sources stage2_common.sh from the capeval checkout, inheriting
# preemption tolerance, STOP sentinel, QoS escalation, offline W&B, and
# figure mirroring.
#
# Submission procedure (every submission, §11.2 — the submitter runs this,
# never the job):
#   1. bring the repo mirror up to date; clone/fetch ${SISTEER_PROJECT_ROOT}-capeval-repo
#      from the mirror; checkout the capeval-v0 commit;
#      echo <sha> > ${SISTEER_PROJECT_ROOT}-capeval-repo/EXPECTED_SHA
#   2. gpu_avail.py --gpus <array width>  -> take its QoS
#   3. widths/ranges from:  uv run python scripts/capeval_run.py \
#        --manifest configs/capeval/manifests/<m>.yaml print-cells
#   4. sbatch --comment=sha:$(git -C ${SISTEER_PROJECT_ROOT}-capeval-repo rev-parse HEAD) \
#      --qos=<from 2> --array=<from 3> scripts/sbatch/capeval_<x>.sbatch
#      (the sha pin is MANDATORY — s2_init FATALs an unpinned submission;
#      it must name the commit checked out + recorded in step 1)
# Order: fetch-data -> calib index 0 ALONE -> calib 1-5 ∥ vllm -> controls
# -> s2 pilot (0) -> s2 1-6; EvalPlus waves (the grading host) after each
# generation-producing array completes.
#
# Env-var seams (set by a job script BEFORE sourcing this file; never via
# sbatch --export, hard rule 1):
#   CAPEVAL_REPO_DIR      override the capeval checkout (default below)
#   CAPEVAL_UV_ENV        override the lane venv — the vllm job sets its
#                         OWN venv here
#   CAPEVAL_UV_SYNC_ARGS  override the uv sync extra args
# shellcheck shell=bash

# The shared root is SISTEER_SHARED_ROOT, which the front door exports from the cluster's site block; a job
# without it refuses by name (there is no built-in root).
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"

# --- capeval's own checkout ---------------------------------------------------
export S2_REPO_DIR="${CAPEVAL_REPO_DIR:-${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}-capeval-repo}"
if [ ! -d "$S2_REPO_DIR" ]; then
  echo "[capeval] FATAL: capeval checkout '$S2_REPO_DIR' does not exist." >&2
  echo "[capeval] Create it from the mirror (submission step 1, plan §11.2):" >&2
  echo "[capeval]   git clone <the repo mirror> $S2_REPO_DIR  # the repo-mirror tool keeps that mirror current" >&2
  echo "[capeval]   git -C $S2_REPO_DIR checkout <capeval-v0 sha> && echo <sha> > $S2_REPO_DIR/EXPECTED_SHA" >&2
  echo "[capeval] The peers' checkout \$SISTEER_PROJECT_ROOT/repo is NEVER used for capeval jobs (D9)." >&2
  exit 1
fi

# --- lane venv + uv ----------------------------------------------------------
export S2_UV_ENV="${CAPEVAL_UV_ENV:-/home/$USER/venvs/sisteer-capeval}"
export S2_UV_SYNC_ARGS="${CAPEVAL_UV_SYNC_ARGS:---python 3.12 --group capeval}"
# 3.12 interpreter fetched ONCE onto shared storage, not per-node (§3.2):
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$SHARED/.uv-pythons}"

# --- CAPEVAL_* cell enumerations (sbatch-side mirror; F15) --------------------
# Canonical copy: scripts/capeval_run.py enumerate_cells(). These constants
# exist for array-width validation and submission ranges; the driver itself
# decodes --cell-index from the SAME enumeration. A manifest edit that
# changes any of these must update BOTH copies or the mirrored-constants
# test fails (tests/test_capeval_driver.py).
export CAPEVAL_CALIB_TRIALS=12         # §7.1 k=6 + growth batches 1+2 (final, 2026-08-16); trial 0 = reference
export CAPEVAL_CALIB_CELLS=12
export CAPEVAL_CALIB_ARRAY_REFERENCE="0"   # runs ALONE first (F5)
export CAPEVAL_CALIB_ARRAY_REST="1-11"
export CAPEVAL_CONTROLS_CELLS=15       # §8/§11.2 expected width
# controls order (enumerate_cells): 0 zero_coeff(T0+1), 1 sham_hook(T0+1),
# 2-5 random_vector T0 @ [0.5,1,2,4], 6-7 random_vector T1 @ [1,4],
# 8-11 noise_lora T0 @ [1e-4,3e-4,1e-3,3e-3], 12-13 noise_lora T1 @
# [3e-4,3e-3], 14 weight_noise(T0+1).
export CAPEVAL_CONTROLS_ARRAY_T0ONLY="2-5,8-11"   # cheap rungs: --time=0-02:00:00
export CAPEVAL_CONTROLS_ARRAY_FULL="0-1,6-7,12-14"
export CAPEVAL_S2_CELLS=7              # §9: 6 divergent + 1 gentle contrast
export CAPEVAL_S2_ARRAY_PILOT="0"      # number_words/bidpo, the −0.84 cell
export CAPEVAL_S2_ARRAY_REST="1-6"
export CAPEVAL_VLLM_CELLS=2            # Tier-1 ×2 repeats in ONE job

# --- vllm own-venv bootstrap --------------------------------------------------
# The vLLM stack installs from configs/capeval/vllm-requirements.txt into
# the vllm job's OWN venv, never the shared capeval lane venv — the lane
# venv stays lockfile-governed (uv sync --frozen), and the vllm stack is
# a uv-pip install outside the lockfile (isolation; the pin moved to
# vllm==0.26.0 + torch 2.11 cu128 after the 0.27.x wheel check came
# back negative under driver 570, then to the OFFICIAL
# vllm 0.25.1+cu129 release wheel + torch 2.11 cu129 after that set
# proved unsatisfiable on protobuf AND its PyPI wheel proved
# cu13-linked — cu129 runs on driver 570 via CUDA 12.x minor-version
# compatibility, probe-verified). The vllm job sets
# CAPEVAL_UV_ENV=/home/$USER/venvs/sisteer-capeval-vllm and calls
# this AFTER s2_init (same per-venv flock; install-at-bootstrap only —
# job steps use `uv run --no-sync` so nothing later can strip it).
capeval_vllm_bootstrap() {
  case "$(basename "$UV_PROJECT_ENVIRONMENT")" in
    *vllm*) ;;
    *)
      echo "[capeval] FATAL: capeval_vllm_bootstrap refuses to install the" >&2
      echo "[capeval] out-of-lockfile vllm stack into '$UV_PROJECT_ENVIRONMENT' —" >&2
      echo "[capeval] that is not a dedicated vllm venv (D12). Set" >&2
      echo "[capeval] CAPEVAL_UV_ENV=/home/\$USER/venvs/sisteer-capeval-vllm before sourcing." >&2
      return 1
      ;;
  esac
  local lockdir rc reqs sentinel req_hash
  lockdir="$(dirname "$UV_PROJECT_ENVIRONMENT")"
  reqs="$S2_REPO_DIR/configs/capeval/vllm-requirements.txt"
  sentinel="$lockdir/.$(basename "$UV_PROJECT_ENVIRONMENT").vllm-bootstrap"
  req_hash="$(sha256sum "$reqs" | cut -c1-16)"
  (
    flock -x 9 || exit 99
    # Sentinel skip (review infra #12): once THIS requirements content is
    # installed, later bootstraps (an overlapping job on the same node)
    # do not touch the venv at all — no mutation under a running workload.
    if [ -f "$sentinel" ] && [ "$(cat "$sentinel")" = "$req_hash" ]; then
      echo "[capeval] vllm stack already bootstrapped for $(basename "$reqs") @ ${req_hash:0:8} — skipping (sentinel)"
    else
      # Same supply-chain cutoff as the repo's tool.uv exclude-newer:
      # every registry pin in the requirements file predates it (vllm
      # 0.25.1, 2026-07-14; its direct-URL wheel is sha256-pinned
      # instead), so transitive resolution is bounded identically.
      # --python is REQUIRED: `uv pip` ignores UV_PROJECT_ENVIRONMENT (a
      # `uv sync`-family variable), so a bare install finds no venv and
      # dies with "No virtual environment found" — job 114815's exact
      # failure (2026-08-16).
      # --index-strategy unsafe-best-match is REQUIRED: the requirements
      # file adds a pytorch extra index for the +cu torch builds, and
      # uv's default first-index strategy confines any name that index
      # also serves (it served `vllm` at a stale version) to that index
      # — "no version of vllm==0.26.0" — job 116213's exact failure
      # (2026-08-16). Safe here: both indexes are trusted and every
      # top-level pin is exact.
      if uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
           --index-strategy unsafe-best-match \
           --exclude-newer 2026-08-03T00:00:00Z -r "$reqs"; then
        echo "$req_hash" > "$sentinel"
      else
        exit 1
      fi
    fi
  ) 9>"$lockdir/.$(basename "$UV_PROJECT_ENVIRONMENT").uv-sync.lock"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[capeval] FATAL: vllm-lane bootstrap failed (rc=$rc) — see" >&2
    echo "[capeval] configs/capeval/vllm-requirements.txt (D12); NOT continuing." >&2
    return "$rc"
  fi
}

# --- REVISE-TIME check (§11.2) ------------------------------------------------
# Call after the workload with the script's header --time limit in seconds
# as the FALLBACK; the EFFECTIVE limit is read from the live job when
# possible (scontrol TimeLimit — review infra #16: T0-only controls submit
# with a --time override, so comparing against the header constant would
# silently skip the warning for them). Warns when measured wall-clock
# exceeded 60% of the effective limit (re-time before wide arrays).

# [D-]HH:MM:SS | HH:MM:SS | MM:SS | MM -> seconds (Slurm TimeLimit forms);
# empty output on anything unparseable (caller falls back).
capeval_timelimit_to_s() {
  local tl="$1" days=0 rest h m s
  case "$tl" in
    ''|UNLIMITED|Partition_Limit|*[!0-9:-]*) return 0;;
  esac
  case "$tl" in
    *-*) days="${tl%%-*}"; rest="${tl#*-}";;
    *)   rest="$tl";;
  esac
  IFS=: read -r h m s <<< "$rest"
  # 10#: force base-10 (a leading zero like "08" would otherwise parse octal)
  if [ -z "${m:-}" ]; then           # bare minutes
    echo $(( 10#$days * 86400 + 10#$h * 60 ))
  elif [ -z "${s:-}" ]; then
    if [ "$tl" = "$rest" ]; then     # MM:SS (no days part)
      echo $(( 10#$h * 60 + 10#$m ))
    else                             # D-HH:MM
      echo $(( 10#$days * 86400 + 10#$h * 3600 + 10#$m * 60 ))
    fi
  else                               # [D-]HH:MM:SS
    echo $(( 10#$days * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s ))
  fi
}

capeval_revise_time_check() {
  local limit_s="$1" elapsed="$SECONDS" tl effective
  if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
    tl="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
          | sed -n 's/.*[[:space:]]TimeLimit=\([^ ]*\).*/\1/p' | head -1)"
    effective="$(capeval_timelimit_to_s "$tl")"
    if [ -n "$effective" ] && [ "$effective" -gt 0 ]; then
      limit_s="$effective"
    fi
  fi
  if [ "$elapsed" -gt $((limit_s * 60 / 100)) ]; then
    echo "[capeval] REVISE-TIME: cell took ${elapsed}s > 60% of the ${limit_s}s" >&2
    echo "[capeval] effective --time limit — re-size --time before submitting the wide array (§11.2)." >&2
  fi
}

# --- inherit the stage-2 scaffolding from the CAPEVAL checkout ----------------
source "$S2_REPO_DIR/scripts/sbatch/stage2_common.sh"
