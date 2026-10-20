#!/bin/bash
# scripts/sbatch/cuda_toolkit.sh -- the CUDA toolkit the `mamba` extra's kernels
# are compiled with. SOURCED (never executed) by stage2_common.sh's s2_init when
# a job requests the extra and by mamba_build.sbatch, the once-per-cluster cache
# warm-up, so the fit path and the warm-up select the SAME toolkit the SAME way.
#
# Why a selection rather than /usr/local/cuda: the lockfile's torch is
# 2.11.0+cu128, and torch.utils.cpp_extension REFUSES to build an extension
# with a toolkit of another MAJOR version -- `RuntimeError: The detected CUDA
# version (13.0) mismatches the version that was used to compile PyTorch
# (12.8)`, raised before the first nvcc call (a minor mismatch only warns).
# Cluster A's /usr/local/cuda is 12.8; cluster B's is 13.0 and carries no 12.x
# toolkit anywhere (its login host and a compute node, checked 2026-10-12), so a
# cold-cache sync there needs a toolkit of its own: mamba_build.sbatch installs
# CUDA 12.8.1 under $SHARED/cuda-12.8 (user-space, toolkit only, no driver) when
# this selection finds none, and that path is a candidate below. It replaces an
# unconditional `CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}`, which on cluster B
# named the CUDA 13 toolkit and failed the frozen sync inside uv's build before
# the fit started.
#
# Variables (plain shell, read when a function runs; a job sets them in its
# body, never via sbatch --export -- hard rule 1):
#   S2_CUDA_MAJOR             12 -- the major the locked torch was built with
#                             (cu128), the only nvcc major torch accepts.
#   S2_CUDA_SHARED_TOOLKIT    where mamba_build.sbatch installs when the system
#                             has no CUDA 12: $SHARED/cuda-12.8 when SHARED is
#                             set; nothing otherwise.
#   S2_CUDA_RELEASE           OUTPUT of a successful selection: the `release
#                             X.Y` of the chosen nvcc, for the job log.
#
# s2_cuda12_select
#   Exports a VALIDATED CUDA_HOME and puts $CUDA_HOME/bin first on PATH (compute
#   nodes ship their toolkit with nvcc OFF PATH, 471132); returns 0. A caller's
#   own CUDA_HOME is honoured when its nvcc reports the right major and REFUSED
#   (return 1) when it does not -- never silently replaced: the caller named a
#   toolkit on purpose, and a wrong one must fail where it can be read, not
#   minutes later inside uv's build. With no CUDA_HOME, the first candidate
#   whose nvcc reports `release $S2_CUDA_MAJOR.` wins, in this order:
#     /usr/local/cuda-12.8   /usr/local/cuda   /usr/local/cuda-12   $S2_CUDA_SHARED_TOOLKIT
#   (an explicit list, no glob: a non-matching glob is a literal under bash and
#   an error under zsh, and the probe list must read the same in every shell).
#   Returns 1, with every probe and its verdict on stderr, when none validates.
# s2_cuda_release <dir>
#   Prints the `release X.Y` of <dir>/bin/nvcc; prints nothing (returns 1) when
#   there is no runnable nvcc there.

S2_CUDA_MAJOR="${S2_CUDA_MAJOR:-12}"

s2_cuda_release() {
  local nvcc="$1/bin/nvcc"
  [ -x "$nvcc" ] || return 1
  "$nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1
}

s2_cuda12_select() {
  local rel cand
  local probed=()
  local shared_tk="${S2_CUDA_SHARED_TOOLKIT:-${SHARED:+$SHARED/cuda-12.8}}"
  if [ -n "${CUDA_HOME:-}" ]; then
    rel="$(s2_cuda_release "$CUDA_HOME")"
    case "$rel" in
      "$S2_CUDA_MAJOR".*)
        export CUDA_HOME
        export PATH="$CUDA_HOME/bin:$PATH"
        S2_CUDA_RELEASE="$rel"
        return 0 ;;
      *)
        echo "cuda_toolkit.sh: CUDA_HOME=$CUDA_HOME is not a CUDA $S2_CUDA_MAJOR toolkit (nvcc: ${rel:-none runnable}); refusing to build the mamba extra's kernels with it (the locked torch is cu128, and torch.utils.cpp_extension refuses another major)" >&2
        return 1 ;;
    esac
  fi
  for cand in /usr/local/cuda-12.8 /usr/local/cuda /usr/local/cuda-12 ${shared_tk:+"$shared_tk"}; do
    if [ ! -d "$cand" ]; then
      probed+=("$cand: absent")
      continue
    fi
    rel="$(s2_cuda_release "$cand")"
    case "$rel" in
      "$S2_CUDA_MAJOR".*)
        export CUDA_HOME="$cand"
        export PATH="$CUDA_HOME/bin:$PATH"
        S2_CUDA_RELEASE="$rel"
        return 0 ;;
      "") probed+=("$cand: no runnable bin/nvcc") ;;
      *)  probed+=("$cand: release $rel") ;;
    esac
  done
  echo "cuda_toolkit.sh: no CUDA $S2_CUDA_MAJOR toolkit for the mamba extra's source build (the locked torch is cu128; torch.utils.cpp_extension refuses another major); probed: ${probed[*]}" >&2
  return 1
}
