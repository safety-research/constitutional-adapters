#!/usr/bin/env python
"""Emit the round-2 gate_reg mini-sweep design (user directive 2026-09-02).

Six trials over gate_reg ∈ {0.01, 0.03, 0.1, 0.3, 1.0, 3.0} — 0.1 is the
campaign default, re-run as the sweep's in-design control — at ONE cell:
gemma4e4b_base × constv2-mix-1m × psr_ce_all, with lr / max_grad_norm /
warmup_frac FIXED at the D22 winners, read verbatim from the committed
(folded) config so the design can never drift from what the fits ran.

This is deliberately NOT tune_arm's Sobol emit-trials: a 6-point grid over
one axis with every continuous axis held fixed is not a space-filling
design, and build_trials (correctly) refuses constant lr/clip. The payload
keeps the exact schema run_trial consumes — model/model_revision/
config_sha identity (validated before any model load), round, position,
trials rows — plus:

  * ``si``: the pinned data cell. run_trial refuses an --si that
    contradicts it, and sdfbase_fits.py's hpo-spawn/_hpo_impl read it so
    the wave trains on constv2-mix-1m's pool with that cell's epoch cap
    (train_kwargs_by_si.constv2-mix-1m, the round-2 24-epoch table).
  * per-row ``gate_reg``: threaded into the PSR trainer's reg_coeff by
    CESoloAdapter.train (overriding the config's 0.1) and into the trial
    fit fingerprint (a re-emitted design with different values at the
    same index must never alias to a finished trial).

Known analyze-side gap (stated, not worked around): ``tune_arm.py
analyze`` is the Sobol design's consumer — its quadratic-winner path reads
the (lr, warmup_frac, max_grad_norm) coordinates only (the analyze-side
tuples deliberately stay 3-wide; ``TRIAL_EXTRA_AXES`` folds gate_reg into
the TRIAL side alone: fingerprint, TrialHP, banner, summary echo). Run on
this round it would see six trials at ONE coordinate and could neither
rank them by gate_reg nor fit a surface, so it must NOT be used to pick a
gate_reg "winner". The §4.6 read (per-trial gate liveness, best held-out
CE/ratio, probe margin where trial objects were retained, all against
log-gate_reg) is computed by the analysis from the trial summaries, which
echo ``gate_reg`` first-class — no analyze-side fold is implemented here,
and none is needed for that read.

Launch (follow-on operator, after commit + volume put + deploy):

  python scripts/sdfbase/emit_gatereg_trials.py          # writes the file
  modal volume put twosi-fits \
    <staging dir>/sdfbase/hpo/trials/r3_gemma4e4b_base_psr_ce_all_all.json \
    hpo/trials/r3_gemma4e4b_base_psr_ce_all_all.json
  modal run scripts/modal/sdfbase_fits.py --mode hpo-spawn \
    --model gemma4e4b_base --arms psr_ce_all --round-no 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]

#: The swept grid (0.1 = the campaign default, the in-design control).
GATE_REG_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)

DEFAULT_MODEL_KEY = "gemma4e4b_base"
DEFAULT_ARM = "psr_ce_all"
DEFAULT_SI = "constv2-mix-1m"
DEFAULT_ROUND = 3
DEFAULT_OUT_DIR = "$STAGE/sdfbase/hpo/trials"   # under the staging directory; expanded through sisteer.site when used


def build_gatereg_trials(cfg_path: Path, *, arm: str = DEFAULT_ARM,
                         si: str = DEFAULT_SI, round_no: int = DEFAULT_ROUND,
                         grid: tuple[float, ...] = GATE_REG_GRID) -> dict:
    """The 6-trial payload. Fixed lr/clip/warmup come from the committed
    config's train_kwargs.<arm> block (the folded winners) — a
    sentinel there (unfolded lane) float()-crashes, never emits."""
    if arm != "psr_ce_all":
        raise SystemExit(f"[emit-gatereg] gate_reg is a psr_ce_all axis "
                         f"only, not {arm!r} (no gate to regularize)")
    raw = cfg_path.read_bytes()  # one read: parsed, and hashed for identity
    cfg = yaml.safe_load(raw)
    block = (cfg.get("train_kwargs") or {}).get(arm) or {}
    lr = float(block["lr"])                      # sentinel -> ValueError
    clip = float(block["max_grad_norm"])
    warmup = float(block["warmup_frac"])
    si_caps = (cfg.get("train_kwargs_by_si") or {}).get(si)
    if not si_caps:
        raise SystemExit(f"[emit-gatereg] {cfg_path.name} carries no "
                         f"train_kwargs_by_si.{si} — the pinned cell must "
                         "be a registered config cell")
    if si not in (cfg.get("si_ids") or []):
        raise SystemExit(f"[emit-gatereg] {si!r} is not in the config's "
                         "si_ids — refusing a sweep at an unregistered cell")
    values = sorted(set(float(g) for g in grid))
    if len(values) != len(grid):
        raise SystemExit(f"[emit-gatereg] duplicate grid values in {grid}")
    return {
        "round": int(round_no),
        "design": "gate_reg_grid",
        "axes": ["gate_reg"],
        "grid": values,
        # Degenerate by design (single-axis grid; the continuous axes are
        # the winners of the continuous sweep, held fixed) — recorded for
        # the fingerprint's box slot, not consumed as a search box.
        "box": {"lr": [lr, lr], "max_grad_norm": [clip, clip]},
        "warmup_levels": [warmup],
        "fixed_from": f"D22 winner (config train_kwargs.{arm})",
        "arm": arm,
        "si": si,
        "epochs_cap": dict(si_caps),
        "model": cfg["model"],
        "model_revision": cfg.get("model_revision"),
        # tune_arm._config_file_sha's convention (16-hex sha256 of the
        # config FILE BYTES), re-stated here rather than imported: tune_arm
        # pulls torch at import and this emitter is yaml-only.
        "config_sha": hashlib.sha256(raw).hexdigest()[:16],
        "position": "all",
        "trials": [
            {"index": i, "lr": lr, "warmup_frac": warmup,
             "max_grad_norm": clip, "gate_reg": v}
            for i, v in enumerate(values)
        ],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-key", default=DEFAULT_MODEL_KEY)
    ap.add_argument("--arm", default=DEFAULT_ARM)
    ap.add_argument("--si", default=DEFAULT_SI)
    ap.add_argument("--round-no", type=int, default=DEFAULT_ROUND)
    ap.add_argument("--config", type=Path, default=None,
                    help="defaults to configs/sdfbase_<model-key>_fits.yaml")
    ap.add_argument("--out", type=Path, default=None,
                    help="defaults to $STAGE/sdfbase/hpo/trials/"
                         "r<round>_<model-key>_<arm>_all.json ($STAGE: the "
                         "site's staging directory)")
    args = ap.parse_args(argv)
    cfg_path = args.config or (
        REPO / "configs" / f"sdfbase_{args.model_key}_fits.yaml")
    payload = build_gatereg_trials(cfg_path, arm=args.arm, si=args.si,
                                   round_no=args.round_no)
    if args.out is None:
        if str(REPO / "scripts") not in sys.path:
            sys.path.append(str(REPO / "scripts"))
        from site_paths import site_path   # the site config's staging directory; an unset $STAGE refuses by name
        out = (site_path(DEFAULT_OUT_DIR, "--out")
               / f"r{args.round_no}_{args.model_key}_{args.arm}_all.json")
    else:
        out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    t0 = payload["trials"][0]
    print(f"[emit-gatereg] {len(payload['trials'])} trials over gate_reg="
          f"{payload['grid']} at {args.si} (lr={t0['lr']:.4g} "
          f"clip={t0['max_grad_norm']:.4g} warmup={t0['warmup_frac']:g} "
          f"fixed) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
