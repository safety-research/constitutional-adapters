"""(Re)build the committed training-fingerprint META attestation from the
authoritative realised store.

The held-out leakage gate checks each family's corpus against the stage-3
TRAINING pool by normalised-exact match. The full sha SET is large and machine-
local (not committed); this CLI reads it from the config-declared pool
(``pool.store_dir`` + ``pool.train_families``) and writes a small, leak-free META
attestation (``fingerprint_sha`` + ``n_shas`` + provenance — no prompt text) to
``configs/heldout/stage3_training_fingerprint.meta.json``, so the gate's result
can be pinned and drift caught (``heldout_leakage.verify_fingerprint``).

Re-run whenever the training pool legitimately changes.

    python scripts/build_heldout_fingerprint.py --config configs/two_si.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/two_si.yaml")
    ap.add_argument("--out",
                    default="configs/heldout/stage3_training_fingerprint.meta.json")
    ap.add_argument("--include-benign", action="store_true",
                    help="also include the benign training corpora in the pool")
    args = ap.parse_args()

    import yaml

    from sisteer.eval.attacks.heldout_leakage import (
        load_training_fingerprint, write_fingerprint_meta)

    cfg = yaml.safe_load(Path(args.config).read_text())
    pool = cfg["pool"]
    benign = pool.get("benign_families", []) if args.include_benign else ()
    fp = load_training_fingerprint(pool["train_families"],
                                   store_dir=pool.get("store_dir"),
                                   benign_families=benign)
    meta = write_fingerprint_meta(
        fp, args.out,
        provenance={"store_dir": str(pool.get("store_dir")),
                    "train_families": ",".join(pool["train_families"]),
                    "source": "realised_labelled_store", "field": "attack_prompt"})
    print(f"[fingerprint] {meta['n_shas']} shas, sha {meta['fingerprint_sha']} "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
