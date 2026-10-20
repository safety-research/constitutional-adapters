#!/usr/bin/env python
"""Generate the sdfbase-mt wave's transfer configs from the `_r3` configs.

`configs/sdfbase_transfer_<slug>_mt1.yaml` = `configs/sdfbase_transfer_<slug>_r3.yaml`
byte-for-byte as DATA except exactly four keys (test-pinned, the `_r3`/`_it`
precedent): `explicit_cells` (none + the four full-corpus cells: the
constitutional corpus and the register-matched control corpus, both CE arms),
`defenses_dir` (the per-wave bank), `out_dir` / `private_run` (the `_mt1`
namespace). The header records the target pin and the OBJECT IDENTITY the
wave serves: the fit fingerprints (known from the fits' spawn manifests) and,
once staged, each object's sha256 — passed in as JSON so the header is
regenerated, never hand-edited, when a continuation publishes a new object.

Usage:
  python scripts/sdfbase/make_mt_wave_configs.py --pins <pins.json> [--wave mt1]
  pins.json: {"<slug>": {"<cell_id>": {"fit_fingerprint": ..., "object_sha256": ... | null,
                                        "source": "...", "published_at": ... | null}}}
The tests regenerate the committed files with the committed pins and require
byte-identity, so a drift between generator and file is caught.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# QWEN ONLY (scope change 2026-09-04): gemma4e4b_base is dropped from this wave.
SLUGS = ("qwen35_4b",)
BASE_KEY = {"qwen35_4b": "qwen35_4b_base"}
CELLS = ["constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce",
         "constv2ctl2-full__psr_ce_all", "constv2ctl2-full__lora_r4_ce"]
PINS_DEFAULT = REPO / "configs" / "sdfbase_mt_pins.json"


def _target_line(r3_text: str) -> str:
    m = re.search(r"^# serving target .*$", r3_text, re.M)
    if not m:
        raise SystemExit("r3 config has no '# serving target' header line")
    return m.group(0)


def render(slug: str, wave: str, r3_text: str, pins: dict,
           cells: list[str] | None = None) -> str:
    """``cells`` (default: every full-corpus cell) narrows the enumeration —
    the SMOKE wave lists only the cells that are published at smoke time."""
    cells = list(CELLS if cells is None else cells)
    target = _target_line(r3_text)
    # body = everything from the '# -- everything below' marker on
    marker = "# -- everything below:"
    i = r3_text.index(marker)
    body = r3_text[i:]
    body = re.sub(r"^defenses_dir: .*$", f"defenses_dir: out/transfer_{slug}_{wave}/cells",
                  body, count=1, flags=re.M)
    body = re.sub(r"^out_dir: .*$", f"out_dir: out/sdfbase_transfer_{slug}_{wave}",
                  body, count=1, flags=re.M)
    body = re.sub(r"^private_run: .*$", f"private_run: sdfbase_transfer_{slug}_{wave}",
                  body, count=1, flags=re.M)
    body = body.replace(
        "# -- everything below: the _r2 config's data, unchanged, EXCEPT defenses_dir/out_dir/private_run --",
        "# -- everything below: the _r3 config's data, unchanged, EXCEPT defenses_dir/out_dir/private_run --")
    pin_lines = []
    for cid in cells:
        p = pins.get(slug, {}).get(cid, {})
        fp = p.get("fit_fingerprint") or "UNPINNED"
        sha = p.get("object_sha256")
        pin_lines.append(f"#       {cid:<32} fit_fingerprint {fp}")
        pin_lines.append(f"#       {'':<32} object_sha256   "
                         f"{sha if sha else 'PENDING (staged + pinned by mt_stage_bank.py at launch)'}")
    pins_block = "\n".join(pin_lines)
    header = f"""# sdfbase_transfer_{slug}_{wave}.yaml — the SLURM MULTI-TURN + PETRI wave of the
# instruct-transfer leg (sdf-base-steering-r2): the FULL-CORPUS
# base-fit objects — the constitutional corpus (constv2-mix-full) and the
# register-matched control corpus (constv2ctl2-full), both CE arms — applied to
# the instruct target and attacked by Crescendo / ActorAttack / PAIR (5x3) on a
# vLLM-served lens target, plus the Petri misalignment audit, with a FRESH
# same-wave bare anchor. DERIVED from configs/sdfbase_transfer_{slug}_r3.yaml
# (scripts/sdfbase/make_mt_wave_configs.py): every key below is that file's,
# byte-for-byte as data, EXCEPT —
#
#   * `explicit_cells` is [none, the four full-corpus cells]. `none` first is
#     the SAME-WAVE bare anchor: greedy generations differ run-to-run from
#     numeric noise alone, so every contrast is paired against a bare run
#     generated in THIS wave (same target serving path, same attacker, same
#     goals), never against the Modal batteries' or the two-SI campaign's
#     transcripts. The cell ids are the fitting lane's (<pseudo-instruction>__<arm>).
#   * `defenses_dir` names the per-wave bank out/transfer_{slug}_{wave}/cells —
#     on the cluster it is the TT_TWO_SI_DEFENSES_DIR export every job sets
#     (the defenses-dir resolution order), staged from the FITTING bank
#     out/{BASE_KEY[slug]}/cells_all/<cell> (the COMPLETE publication) as a
#     byte-verified copy by $STAGE/sdfbase/mt/mt_stage_bank.py. The
#     objects this wave serves are pinned here by identity — the fit
#     fingerprint their spawn manifests recorded (fit_calls_cont4_full.json /
#     fit_calls_ctl2.json expect_fingerprint) and the published object sha256
#     once known; the launch gate refuses when a staged object's sha differs
#     from the LIVE object_sha256 in the fitting bank:
{pins_block}
#     A staged object is never overwritten: a re-publication after staging
#     re-stages into a NEW wave suffix with a regenerated header.
#   * out_dir / private_run get the fresh `_{wave}` namespace, so this wave's
#     transcripts, aggregates and judge cache can never mix with, or clobber,
#     any battery wave's trees. Nothing here is compared against the HF-target
#     two-SI multi-turn numbers: this wave is its own compared set.
#
# Serving: --target-engine served (lens-routed RoutingProxy, one vLLM per model
# serving all five cells as virtual models; objects at their recorded fit leg,
# positions=all, so the drivers pass NO --positions). Attacker: the config's
# huihui-ai/Huihui-Qwen3.6-27B-abliterated on vLLM at --max-model-len 49152,
# thinking stripped; PAIR_N_STREAMS=5 asserted per driver. Goals: the ~87-goal
# stratified subset (goal index i with i % 9 in {{0,1,2}} of the confirmatory
# pool; generated as --nshards 18 shards {{0,1,2,9,10,11}}).
#
{target}

# -- the explicit cell enumeration (ordered; the registry is EXACTLY this) ---
explicit_cells:
- none
"""
    notes = {"constv2-mix-full__psr_ce_all": "# the constitutional full corpus (42.8M official tokens), both CE arms",
             "constv2ctl2-full__psr_ce_all": "# the register-matched CONTROL full corpus (budget-matched), both CE arms"}
    for cid in cells:
        if cid in notes:
            header += notes[cid] + "\n"
        header += f"- {cid}\n"
    header += "\n"
    if cells != CELLS:
        header = header.replace(
            "# -- the explicit cell enumeration (ordered; the registry is EXACTLY this) ---",
            "# SCRATCH SMOKE WAVE: a SUBSET of the cells (the ones published at smoke time), served\n"
            "# from an interim bank; results under this namespace are plumbing evidence only and\n"
            "# may be superseded — never a results row.\n"
            "# -- the explicit cell enumeration (ordered; the registry is EXACTLY this) ---")
    return header + body


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", default="mt1")
    ap.add_argument("--pins", type=Path, default=PINS_DEFAULT)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the committed files differ from the render")
    ap.add_argument("--cells", default=None,
                    help="comma list of fitted cells (smoke waves); default all four")
    args = ap.parse_args(argv)
    cells = args.cells.split(",") if args.cells else None
    pins = json.loads(args.pins.read_text()) if args.pins.is_file() else {}
    rc = 0
    for slug in SLUGS:
        r3 = (REPO / "configs" / f"sdfbase_transfer_{slug}_r3.yaml").read_text()
        text = render(slug, args.wave, r3, pins, cells)
        dst = REPO / "configs" / f"sdfbase_transfer_{slug}_{args.wave}.yaml"
        if args.check:
            if not dst.is_file() or dst.read_text() != text:
                print(f"DRIFT: {dst}"); rc = 1
            else:
                print(f"ok: {dst}")
        else:
            dst.write_text(text); print(f"wrote {dst}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
