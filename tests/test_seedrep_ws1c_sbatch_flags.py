"""The Qwen seed-replicate battery: the two sbatch flag changes, evaluated by bash, not grepped only.

  * scripts/sdfbase/full22/misalign/misalign_proxy.sbatch: the runner's --seed-policy / --seed-base ride
    F22_MIS_SEED_POLICY / F22_MIS_SEED_BASE from the launch environment; both unset = the AGENTIC_ARGS
    array exactly as before (flags only, no structural change to the script);
  * scripts/sdfbase/full22/sampled/target_sampled.sbatch: the static phase's MAX_DRIVERS defaults to the
    server width S (no DRIVER_CAP 96); F22S_DRIVER_CAP caps it."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MIS = REPO / "scripts/sdfbase/full22/misalign/misalign_proxy.sbatch"
SAMPLED = REPO / "scripts/sdfbase/full22/sampled/target_sampled.sbatch"


def _bash(script: str, env: dict) -> str:
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True,
                          env={"PATH": "/usr/bin:/bin", **env}).stdout.strip()


def _agentic_args_block() -> str:
    src = MIS.read_text()
    start = src.index("AGENTIC_ARGS=(--n 10 --behaviors leaking murder --max-tokens 3000)")
    end = src.index("# D2: nemotron3nano4b is EXCLUDED FROM LEAKING")
    return src[start:end]


def test_misalign_sbatch_passes_the_seed_flags_through_the_environment_and_is_unchanged_without_them():
    src = MIS.read_text()
    assert '[ -n "${F22_MIS_SEED_POLICY:-}" ] && AGENTIC_ARGS+=(--seed-policy "$F22_MIS_SEED_POLICY")' in src
    assert '[ -n "${F22_MIS_SEED_BASE:-}" ] && AGENTIC_ARGS+=(--seed-base "$F22_MIS_SEED_BASE")' in src
    block = _agentic_args_block()
    script = block + '\nprintf "%s\\n" "${AGENTIC_ARGS[@]}"'
    base = {"MODE": "vendor", "BEHAVIORS": "leaking murder", "ONLY_CELLS": "", "SRC_ROOT": "/s"}
    # the array must not depend on bash arrays from outside the block: BEHAVIORS is an array in the script
    script = 'BEHAVIORS=(leaking murder)\n' + script
    unset = _bash(script, base).splitlines()
    assert unset == ["--n", "10", "--behaviors", "leaking", "murder", "--max-tokens", "16384", "--sampler", "vendor"]
    seeded = _bash(script, {**base, "F22_MIS_SEED_POLICY": "per_response", "F22_MIS_SEED_BASE": "2"}).splitlines()
    assert seeded == unset + ["--seed-policy", "per_response", "--seed-base", "2"]
    only_base = _bash(script, {**base, "F22_MIS_SEED_BASE": "3"}).splitlines()
    assert only_base == unset + ["--seed-base", "3"]                # the runner refuses this pair itself
    full = _bash(script, {**base, "MODE": "full"}).splitlines()
    assert full == ["--n", "10", "--behaviors", "leaking", "murder", "--max-tokens", "3000"]
    # the runner line itself is untouched (flags only): concurrency = the attested S, the four fixed args
    assert '--concurrency "$TARGET_S" "$@" 2>&1 | tee "$log" ) &' in src
    assert src.count("misalign.runner") >= 2


def test_target_sampled_static_drivers_default_to_the_server_width_and_the_env_caps_them():
    src = SAMPLED.read_text()
    assert "DRIVER_CAP=96" not in src
    m = re.search(r'\n  DRIVER_CAP="\$\{F22S_DRIVER_CAP:-\$TARGET_S\}"\n  MAX_DRIVERS=\$\(\( TARGET_S < DRIVER_CAP \? TARGET_S : DRIVER_CAP \)\)\n', src)
    assert m, "the static phase's driver-width lines"
    lines = m.group(0).strip("\n")
    for env, want in (({"TARGET_S": "45"}, "45 45"), ({"TARGET_S": "200"}, "200 200"),
                      ({"TARGET_S": "45", "F22S_DRIVER_CAP": "96"}, "96 45"),
                      ({"TARGET_S": "200", "F22S_DRIVER_CAP": "96"}, "96 96"),
                      ({"TARGET_S": "45", "F22S_DRIVER_CAP": "16"}, "16 16")):
        assert _bash(lines + '\necho "$DRIVER_CAP $MAX_DRIVERS"', env) == want, env
    # the escalation phases' driver counts are untouched
    assert src.count("MAX_DRIVERS=$(( (TARGET_S + F22S_ESC_CONC - 1) / F22S_ESC_CONC ))") == 3
    assert "MAX_DRIVERS=$((NCELLS + 1))" in src and "MAX_DRIVERS=$((NRUNG + 1))" in src
