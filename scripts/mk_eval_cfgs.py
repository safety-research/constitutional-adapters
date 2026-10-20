#!/usr/bin/env python
"""Write the narrowed eval config copies + counts-only pool manifest for the
lora_r1_all complete-eval (DSR + over-refusal). One arg: the absolute
lora_all_cells root on the pod (holds cells_all/ and cells_resp/).

Produces, under that root:
  two_si_all.yaml   class_a_arms=[lora_r1_all], class_b_arms=[], defenses_dir=
                    cells_all, private_run=two_si_all, pool_manifest set,
                    bon.n_max=64 (full curve to 64; the 256 tail is dropped
                    for tractability across 14 cells).
  two_si_resp.yaml  same but defenses_dir=cells_resp, private_run=two_si_resp.
  pool_manifest.json  counts-only (matched_intersection, all pools 23,002) —
                    two_si_run.py aggregate hard-aborts without it; it carries
                    no prompt/response data.
"""
import json
import sys
from pathlib import Path

import yaml

root = Path(sys.argv[1])
base = yaml.safe_load(open(Path(__file__).resolve().parent.parent
                          / "configs" / "two_si.yaml"))
base["class_a_arms"] = ["lora_r1_all"]
base["class_b_arms"] = []
base.setdefault("bon", {})["n_max"] = 64

for leg in ("all", "resp"):
    c = dict(base)
    c["defenses_dir"] = str(root / f"cells_{leg}")
    c["private_run"] = f"two_si_{leg}"
    c["pool_manifest"] = str(root / "pool_manifest.json")
    yaml.safe_dump(c, open(root / f"two_si_{leg}.yaml", "w"))

json.dump({"regime": "matched_intersection",
           "pool_sizes": {"safety_refusal": 23002,
                          "si_constitution": 23002,
                          "si_irrelevant": 23002}},
          open(root / "pool_manifest.json", "w"))
print(f"wrote two_si_all.yaml, two_si_resp.yaml, pool_manifest.json -> {root}")
