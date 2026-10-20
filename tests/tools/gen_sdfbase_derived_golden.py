#!/usr/bin/env python
"""Freeze the sdfbase_derived 100k golden (tests/derived_golden_replay.py
collect()) from a module file — by default the PRE-CHANGE module taken
from git (``--from-git a9bf57f``), the sdf-base-steering tip the six 100k
derived objects were published from. Re-run only to regenerate the
fixture from THAT commit; regenerating it from a later module would
make the byte-identity test tautological.

  python tests/tools/gen_sdfbase_derived_golden.py --from-git a9bf57f \\
      --out tests/fixtures/sdfbase_derived_100k_golden.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TESTS = REPO / "tests"
MODULE_REL = "scripts/modal/sdfbase_derived.py"
for _p in (TESTS, REPO / "scripts" / "modal", REPO / "scripts" / "sdfbase"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def load_module(src_path: Path):
    from test_watch_successor import make_fake_modal
    if "modal" not in sys.modules:
        sys.modules["modal"] = make_fake_modal()
    # the W&B secret's name is a site setting read at import (the replay records its place, not the name)
    os.environ.setdefault("SISTEER_MODAL_WANDB_SECRET", "example-wandb-secret")
    sys.modules.pop("sdfbase_fits", None)
    spec = importlib.util.spec_from_file_location("_sdfbase_derived_golden_src",
                                                  src_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-git", default="a9bf57f",
                    help="commit whose scripts/modal/sdfbase_derived.py is "
                         "replayed (default a9bf57f, the 100k publish tree)")
    ap.add_argument("--module", default=None,
                    help="replay THIS module file instead of a git blob")
    ap.add_argument("--out", default=str(TESTS / "fixtures" /
                                         "sdfbase_derived_100k_golden.json"))
    args = ap.parse_args(argv)
    from derived_golden_replay import collect  # noqa: E402  (tests/ on path)

    with tempfile.TemporaryDirectory() as td:
        if args.module:
            src = Path(args.module)
            source = {"module": str(src)}
        else:
            full = subprocess.run(["git", "-C", str(REPO), "rev-parse",
                                   args.from_git], capture_output=True,
                                  text=True, check=True).stdout.strip()
            blob = subprocess.run(["git", "-C", str(REPO), "show",
                                   f"{full}:{MODULE_REL}"], capture_output=True,
                                  text=True, check=True).stdout
            # the module resolves sdfbase_fits from ITS OWN dir first, so the
            # blob is materialized beside the live scripts (and removed)
            src = REPO / "scripts" / "modal" / "_golden_src_sdfbase_derived.py"
            src.write_text(blob, encoding="utf-8")
            source = {"git_sha": full, "module": MODULE_REL}
        try:
            text = src.read_bytes()
            source["module_sha256"] = hashlib.sha256(text).hexdigest()
            sd = load_module(src)
            golden = collect(sd, Path(td) / "replay")
        finally:
            if not args.module:
                src.unlink(missing_ok=True)
    doc = {"source": source, "golden": golden}
    Path(args.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n",
                              encoding="utf-8")
    n_cmds = sum(len(v) for v in golden["commands"].values())
    print(f"wrote {args.out}: {n_cmds} command entries, "
          f"{sum(len(v) for v in golden['score_traces'].values())} score "
          f"traces, {len(golden['spawns'])} spawn lists; source {source}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
