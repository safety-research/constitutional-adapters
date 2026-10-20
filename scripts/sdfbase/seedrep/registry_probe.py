#!/usr/bin/env python3
"""qwen-seedrep-native-battery — the registry emission PROBE a container runs once per row.

Under the DRIVER venv, with ONE child's registry env (`SISTEER_REGISTRY_RUN_IDENTITY` / `SISTEER_REGISTRY_SPOOL`, exactly as
`seedrep_launch_plan.resolve_spawn` composes it), print ONE JSON line from THE REAL hooks — never a re-derivation:

    {"enabled": <hooks.emission_enabled()>, "spool": <hooks.resolve_spool()>, "identity_routes": [<routes of hooks.identity_from_env()>]}

`seedrep_served.registry_row_preflight` refuses the row by name (`REGISTRY_EMISSION_OFF`) unless this says ON with the ATTEMPT spool
and the identity's routes read — a silent "not recorded" is the defect (the archived attempt roots carried no `registry/` at all).
Stdlib + `sisteer.registry.hooks` (stdlib-only) only; exit 0 with the line, non-zero with the error on stderr.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

try:
    _REPO = Path(__file__).resolve().parents[3]                  # <tree>/scripts/sdfbase/seedrep/<this> -> <tree>
except IndexError:                                                # a flat copy has no parents[3]; the env's PYTHONPATH must serve
    _REPO = None
if _REPO is not None and (_REPO / "src" / "sisteer").is_dir() and str(_REPO / "src") not in sys.path \
        and not any(p.endswith("/src") and os.path.isdir(os.path.join(p, "sisteer")) for p in sys.path if os.path.isdir(p)):
    sys.path.insert(0, str(_REPO / "src"))

from sisteer.registry import hooks  # noqa: E402


def probe() -> dict:
    ident = hooks.identity_from_env()
    return {"enabled": bool(hooks.emission_enabled()), "spool": hooks.resolve_spool(),
            "identity_routes": sorted(hooks.normalise_identities(ident)) if ident else []}


def main() -> int:
    print(json.dumps(probe(), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
