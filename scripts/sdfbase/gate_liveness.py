#!/usr/bin/env python
"""Per-layer PSR gate liveness for SAVED objects — the round-2 telemetry
the analysis reads as ``gate_liveness.json`` ({layer: live_frac}) next to a
derived object (V, P_resid; the "gate liveness" column) or next to a
gate_reg trial's ``summary.json`` (min-per-layer live-gate
fraction per trial — without it the r3 liveness column is PENDING).

Liveness = the fraction of valid (non-pad) probe positions at which a gate
is OPEN (ReLU(w·h + b) > 0) under a real forward with the object applied
(``psr.psr_gate_liveness`` for a ``psr`` object — the same function the
fit-time guard uses; ``paired_write.paired_gate_liveness`` for a
``psr_paired_write`` object, whose flat file carries the ANY-term-open
fraction — the positions where the paired write is nonzero — with the
per-parent fractions in ``gate_liveness_detail.json``). The probe pool is
a doc pool (the data cell the object was fit on — mix-1m for the r3 trials,
the capture union for the derived objects), rendered verbatim
(docs_to_ce_examples), the first ``--n-probe`` docs in file order.

Identity: a derived dir (``derived_meta.json``) or a staged cell validates
through ``sisteer.eval.object_identity`` before loading; an HPO trial dir
carries only ``summary.json`` + ``object.pt`` (tune_arm's layout, no
identity sidecar), so trial dirs need ``--allow-bare`` — liveness is
telemetry about a saved object, never a served artifact, and the object's
sha256 is recorded in the output so the reading is bound to the bytes.

  python scripts/sdfbase/gate_liveness.py --model <hf id> --revision <rev> \\
      --pairs /vol/pool/<key>/pairs_constv2-mix-1m.jsonl --n-probe 32 \\
      --dirs /vol/out/hpo/<key>/psr_ce_all_all/r3_t00,... --allow-bare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

OUT_NAME = "gate_liveness.json"
DETAIL_NAME = "gate_liveness_detail.json"


def _examples(pairs_path, n_probe: int):
    from sisteer.data.pairs import Pair, docs_to_ce_examples

    rows = [json.loads(l) for l in Path(pairs_path).read_text(
        encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"{pairs_path}: no rows")
    rows = rows[:n_probe] if n_probe > 0 else rows
    pairs = [Pair(prompt=r.get("prompt", ""), pos=r["pos"], neg="",
                  instruction_id=r.get("instruction_id", "")) for r in rows]
    return docs_to_ce_examples(None, pairs)


def paired_liveness_detail(live: dict, kind: str) -> tuple[dict, dict]:
    """({layer: live_frac}, detail) from ``paired_gate_liveness``'s raw
    result: the flat file carries the ANY-term-open fraction (the positions
    where the paired write is nonzero), the detail the per-parent
    fractions. The one spelling of this shape — the residual builder
    writes it beside V through this same function."""
    flat = {str(int(l)): float(v) for l, v in live["any"].items()}
    detail = {"kind": kind,
              "definition": "any-term-open fraction (positions where the "
                            "paired write is nonzero)",
              "terms": {str(i): {str(int(l)): float(v) for l, v in d.items()}
                        for i, d in live["terms"].items()}}
    return flat, detail


def liveness_for(model, tok, obj, examples, *, max_len: int) -> tuple[dict, dict]:
    """({layer: live_frac}, detail) for a psr or psr_paired_write object."""
    from sisteer.extraction.paired_write import (SharedStateWriteObject,
                                                 paired_gate_liveness)
    from sisteer.extraction.psr import PSRObject, psr_gate_liveness

    if isinstance(obj, SharedStateWriteObject):
        live = paired_gate_liveness(model, tok, obj=obj, examples=examples,
                                    max_len=max_len)
        flat, detail = paired_liveness_detail(live, obj.kind)
        return flat, {**detail, "n_valid_positions": live["n_valid_positions"]}
    if isinstance(obj, PSRObject):
        live = psr_gate_liveness(model, tok, obj=obj, examples=examples,
                                 max_len=max_len, n_probe=len(examples))
        flat = {str(int(l)): float(v) for l, v in live.items()}
        return flat, {"kind": obj.kind,
                      "definition": "fraction of valid probe positions with "
                                    "ReLU(w.h + b) > 0 under the object's "
                                    "own forward"}
    raise SystemExit(f"gate liveness is defined for psr gates; got "
                     f"{type(obj).__name__} (kind {getattr(obj, 'kind', None)!r})")


def write_liveness(out_dir, flat: dict, *, detail: dict) -> dict:
    """Write ``gate_liveness.json`` ({str(layer): live_frac}, sorted) and
    ``gate_liveness_detail.json`` (``detail`` plus the min/mean live
    fraction over layers) beside an object; returns the detail document
    as written. The one writer of these files."""
    out_dir = Path(out_dir)
    flat = {str(int(l)): float(v) for l, v in flat.items()}
    (out_dir / OUT_NAME).write_text(json.dumps(flat, indent=1, sort_keys=True)
                                    + "\n", encoding="utf-8")
    vals = list(flat.values())
    doc = {**detail,
           "min_live_frac": min(vals) if vals else None,
           "mean_live_frac": (sum(vals) / len(vals)) if vals else None}
    (out_dir / DETAIL_NAME).write_text(
        json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8")
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--pairs", required=True, help="probe doc pool (jsonl)")
    ap.add_argument("--n-probe", type=int, default=32,
                    help="first N docs of the pool (0 = all)")
    ap.add_argument("--dirs", required=True,
                    help="comma list of object dirs (each holds object.pt)")
    ap.add_argument("--allow-bare", action="store_true",
                    help="accept dirs with no identity sidecar (HPO trial "
                         "dirs) — telemetry only, sha256 recorded")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    args = ap.parse_args(argv)

    from sisteer.eval.defense_cell import sha256_file
    from sisteer.eval.object_identity import validate_artifact_identity

    dirs = [Path(d) for d in args.dirs.split(",") if d]
    targets = []
    for d in dirs:
        obj_path = d / "object.pt"
        if not obj_path.is_file():
            raise SystemExit(f"{obj_path} missing")
        kind = validate_artifact_identity(obj_path, None, consumer="gate-liveness",
                                          allow_bare=args.allow_bare)
        targets.append((d, obj_path, kind))

    from sisteer.extraction import tokcache
    from sisteer.extraction.fitmon import pin_attention_backends
    from sisteer.methods import load_object
    from sisteer.models import load_model_and_tokenizer

    # device/dtype None -> the loader's own defaults (cuda when available;
    # bf16 on cuda, fp32 otherwise); the loader also puts the model in
    # eval mode.
    model, tok = load_model_and_tokenizer(args.model, device=args.device,
                                          dtype=args.dtype,
                                          revision=args.revision)
    pin_attention_backends()
    examples = _examples(args.pairs, args.n_probe)
    pairs_sha = sha256_file(args.pairs)
    # One in-process token memo for the run: every --dirs entry re-encodes
    # the same probe docs, and a cache hit is a copy of what the tokenizer
    # returned (byte-neutral — the fit scripts' convention).
    prev = tokcache.active()
    tokcache.set_active(tokcache.TokCache())
    try:
        for d, obj_path, kind in targets:
            obj = load_object(obj_path, model=model, tok=tok)
            flat, detail = liveness_for(model, tok, obj, examples,
                                        max_len=args.max_len)
            doc = write_liveness(d, flat, detail={
                **detail, "object": str(obj_path),
                "object_sha256": sha256_file(obj_path), "identity_kind": kind,
                "pairs": str(args.pairs), "pairs_sha256": pairs_sha,
                "n_probe_docs": len(examples), "max_len": args.max_len,
                "model": args.model, "revision": args.revision})
            print(f"[gate-liveness] {d}: min={doc['min_live_frac']:.3f} "
                  f"mean={doc['mean_live_frac']:.3f} over {len(flat)} "
                  f"layers -> {d / OUT_NAME}", flush=True)
    finally:
        tokcache.set_active(prev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
