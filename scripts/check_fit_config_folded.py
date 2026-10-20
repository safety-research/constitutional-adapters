#!/usr/bin/env python
"""Refuse a twosi fit launch whose LEG still carries HPO-winner sentinels.

Usage: check_fit_config_folded.py <fit-config.yaml> <arm[,arm...]>

The fit configs fold per-arm HPO winners as each cell's HPO lands
(make_fit_configs.py — partial folds are the normal case, since the
Llama cells land on their own clocks), so a config legitimately carries
PENDING_LLAMA_HPO_WINNERS for arms whose rounds are still running. The
old whole-file ``grep -q`` gate in twosi_fits_llama.sbatch therefore
refused legs whose own arms were fully folded (the token_faithful wave
would block on psr/lora sentinels it never reads). This checks ONLY the
arms the invocation trains: exit 0 when none of their train_kwargs carry
a sentinel; exit 1 naming the unfolded arms otherwise.
"""
import sys

import yaml

SENTINEL = "PENDING_LLAMA_HPO_WINNERS"


def unfolded_arms(cfg: dict, arms: list[str]) -> list[str]:
    tk = cfg.get("train_kwargs") or {}
    return sorted(a for a in arms
                  if any(v == SENTINEL for v in (tk.get(a) or {}).values()))


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    path, arms_csv = argv[1], argv[2]
    arms = [a for a in arms_csv.split(",") if a]
    if not arms:
        print(f"FATAL: no arms given to check against {path}",
              file=sys.stderr)
        return 2
    with open(path) as f:
        cfg = yaml.safe_load(f)
    bad = unfolded_arms(cfg, arms)
    if bad:
        print(f"FATAL: {path} still carries {SENTINEL} for "
              f"{','.join(bad)} — fold those arms' HPO winners "
              "(make_fit_configs.py), commit, re-stage.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
