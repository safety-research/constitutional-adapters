"""Sub-env seam for the attack families that need a SEPARATE ``transformers<4.56``
interpreter (AmpleGCG-plus).

Both families run a GENERATOR model whose reference implementations pin an older
transformers generation API that is incompatible with the repo's pinned stack —
so the generation step runs in a dedicated venv (built the way ``fa_build.sbatch``
/ the capeval vllm lane build a second env: a distinct ``UV_PROJECT_ENVIRONMENT``,
installed once, invoked with its own interpreter), and the main process only
REPLAYS the prompts that env produced.

Two seams live here, kept small and side-effect-free so importing an attack
module (for the registry) never pulls the sub-env or transformers:

- ``subenv_available`` — the AVAILABILITY GATE. This is a DIFFERENT axis from
  ``GATED_ATTACKS`` (which marks families needing the repo's attacker model +
  PyRIT at run time). AmpleGCG-plus needs neither the attacker model
  nor PyRIT — they need the sub-env — so they are NOT in ``GATED_ATTACKS``;
  they gate on this check instead, and when it fails they SKIP LOUDLY rather
  than degrade silently (the authoritative-source rule).
- ``run_worker`` — invoke a worker module (``sisteer.eval.attacks._*_worker``)
  under the sub-env interpreter over stdin/stdout JSON. The worker's heavy
  imports live inside its ``main`` so it is import-safe in the main env too.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def resolve_subenv_python(env_var: str, explicit: str | None = None) -> str | None:
    """The sub-env interpreter path from ``explicit`` or ``$<env_var>``, or None
    if unset / not an executable file."""
    cand = explicit or os.environ.get(env_var)
    if not cand:
        return None
    p = Path(cand).expanduser()
    return str(p) if p.is_file() and os.access(p, os.X_OK) else None


def subenv_available(
    env_var: str,
    explicit: str | None = None,
    *,
    model_env: str | None = None,
    model_path: str | None = None,
) -> tuple[bool, str]:
    """Whether the sub-env is provisioned. Returns ``(True, interpreter_path)``
    when ready, else ``(False, human_reason)``. Checks the interpreter exists +
    is executable and (when ``model_env`` is given) that a generator checkpoint
    is configured; a configured LOCAL absolute path must exist (an HF repo id is
    accepted — the sub-env fetches it)."""
    py = resolve_subenv_python(env_var, explicit)
    if py is None:
        return False, (
            f"sub-env interpreter not found — set ${env_var} (or pass "
            f"subenv_python=) to a transformers<4.56 python")
    if model_env is not None:
        mp = model_path or os.environ.get(model_env)
        if not mp:
            return False, (
                f"generator checkpoint not configured — set ${model_env} (or "
                f"pass generator_model=) to the checkpoint path or HF id")
        mpath = Path(str(mp)).expanduser()
        if mpath.is_absolute() and not mpath.exists():
            return False, f"generator checkpoint path does not exist: {mp}"
    return True, py


def loud_skip(family: str, reason: str, *, subenv: bool = True) -> dict:
    """Emit a LOUD skip to stderr and return a skipped result (no generation,
    no files written) — never a silent degrade. The caller returns this.
    ``subenv=False`` is a skip for a missing input, not a missing sub-env, so the
    sub-env hint is left out (the reason names the input and how to build it)."""
    hint = (" This family needs a separate transformers<4.56 sub-env (see the "
            "module docstring for provisioning)." if subenv else "")
    print(
        f"[{family}] SKIPPED — {reason}.{hint} Nothing was generated.",
        file=sys.stderr, flush=True,
    )
    return {"skipped": True, "reason": reason, "family": family,
            "per_goal": [], "n_goals_run": 0, "n_goals_total": 0}


def run_worker(python: str, module: str, payload: dict, *,
               timeout: float | None = None) -> dict:
    """Invoke ``python -m <module>`` under the sub-env, feeding ``payload`` as
    JSON on stdin and parsing the worker's JSON stdout. Raises loudly (with the
    worker's stderr tail) on a non-zero exit — a failed generation must never
    look like an empty one."""
    proc = subprocess.run(
        [python, "-m", module], input=json.dumps(payload),
        text=True, capture_output=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"sub-env worker {module} failed (rc={proc.returncode}): "
            f"{proc.stderr[-2000:]}")
    return json.loads(proc.stdout)
