"""Proof-only determinism wrapper (P4 twin run).

Job 112688 (k3) showed that same-node, same-job, same-code greedy decode
still flips ~26% of long generations at near-tie argmax points (238/900
records across ALL arms, baseline/prompted included; one judgment flip) —
run-to-run kernel nondeterminism (bf16 reduction-order variance in
attention/matmul epilogues), not an env property. So every proof cell runs
under the standard torch determinism knobs, applied HERE — the proof's own
wrapper — because the guarantee under test is the ENV'S, and sisteer/
production code must not change to make the proof pass:

  * CUBLAS_WORKSPACE_CONFIG=:4096:8  (must be set before cuBLAS init; the
    sbatch also exports it job-wide, this is the backstop)
  * torch.use_deterministic_algorithms(True)  — STRICT: if some op on this
    path has no deterministic implementation, the cell dies loudly with
    torch's RuntimeError naming the op. That is deliberate (fail-fast); the
    documented fallback is to rerun with TSPLIT_DET=warn (warn_only=True —
    knobs best-effort, the A/A compare then decides empirically) and say so
    in the verdict. Never silently downgrade.
  * cudnn.benchmark off, TF32 off (matmul + cudnn)

Usage (from torch_split_proof.sbatch):
  python scripts/_tsplit_det_wrapper.py scripts/stage2_run.py <args...>

The wrapped script runs as __main__ with its own argv, exactly as if
invoked directly.
"""

from __future__ import annotations

import os
import runpy
import sys


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: _tsplit_det_wrapper.py <script.py> [args...]")

    # Backstop only: cuBLAS reads this at init, so the sbatch exports it
    # before python starts; setdefault keeps a job-wide value authoritative.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch  # after the env var, before any model code

    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    mode = os.environ.get("TSPLIT_DET", "strict")
    if mode == "warn":
        torch.use_deterministic_algorithms(True, warn_only=True)
    elif mode == "strict":
        torch.use_deterministic_algorithms(True)
    else:
        sys.exit(f"FATAL: unknown TSPLIT_DET={mode!r} (strict|warn)")
    print(f"[tsplit-det] determinism knobs ON (mode={mode}): "
          f"use_deterministic_algorithms, cudnn.benchmark=False, TF32=off, "
          f"CUBLAS_WORKSPACE_CONFIG="
          f"{os.environ['CUBLAS_WORKSPACE_CONFIG']}", flush=True)

    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
