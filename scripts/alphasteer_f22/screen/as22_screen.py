"""alphasteer-full22 STRENGTH FRONTIER — the SCREEN launcher's one table (paper Figure 5, the AlphaSteer strength
lane).

The screen is the judged-controls leg at trial 0 (XSTest + OR-Bench-Hard, the 512 -> 2048 -> 8192 cap ladder) run on
the LoRA dose views' EXACT instrument — the wave checkout `f38d055d` every LoRA dose screen ran from — for one
(target, strength view) of the AlphaSteer ladder. This module is what the shell wrapper (`asscreen_controls.sbatch`),
the sampled-config generator (`make_strength_sampled_configs.py`) and the tests share, so that the scope, the TAG
grammar and every path are spelled ONCE.

SCOPE: the three paper targets x the five nominal strengths of `as22_strength.STRENGTHS` — fifteen views.
glm46vflash has strength objects, configs and bank views (as22_strength / the roster still name it) but is
dropped from the paper; this launcher REFUSES it rather than screening a descoped target by accident.
A view enters the screen by being written here (the wave's `SAMPLED_VIEWS` rule), never by existing in the roster.

TAG GRAMMAR: `<target>_<view>` (e.g. `gemma4e4b_as_d0p3`) — the wave's `tag_of` applied to the strength view, so
every store / ladder / serving path keyed on TAG lands beside the LoRA screens' under the SAME wave root:
    $WAVE_ROOT/sampled/controls_cap<rung>/<TAG>_t<T>/cells/<store>/{gen_stats,manifest,evidence}.json
    $WAVE_ROOT/sampled/serving/controls_t<T>_<TAG>_<jobid>/positive_control_controls_512.json(.verdict)
    $WAVE_ROOT/sampled/ladders/<TAG>_t<T>/{events.jsonl,ladder.json}
and screen_verdict.py reads them with `--roster configs/alphasteer_f22_strength_roster.yaml` (the roster's
`strength:` key is the dose; the job prefix below is the screen job's). Job names: `asscr-<TAG>-ctl<T>`
(the `-ctl<T>` suffix is the sampled submitter's grammar, kept so the verdict tool's screen/leg split works).

TWO CHECKOUTS, ONE INSTRUMENT: a screen job takes TWO shas. `<wave_sha>` is the CODE — common.sh,
common_sampled.sh, mt_serving_up.py, served_capability_run.py, smoke_positive_control.py, src/sisteer — and it is
`f38d055d59a7102f930e30d3abac64adbcf1270b`, the sha the LoRA dose screens ran from, so the two ladders are read on
one instrument — in practice plus the two commits of `LOW_DOSE_LIVENESS_WAVE_SHA`, both of which an AlphaSteer object
needs before it can be screened at all (see `COMPARABLE_WAVE_SHAS`). `<as_sha>` is a checkout of THIS branch and supplies DATA only: the serving config, its sampled
variants, the manifest, the roster and the pins. No code runs from it (the resolver below is stdlib + yaml and is
invoked by the wrapper before any venv exists, exactly as the wave's `--is-in-scope` gate is). The wrapper refuses a
dirty or wrong-HEAD checkout on either side.

NOMINAL STRENGTHS ARE LABELS. The grid -0.5, 0.1, 0.3, 0.5, 1.0 is `base_scale` in the served op's own units and
is UNCHANGED for the re-baked objects (the paper-sized layer sets): the same five views, the
same TAGs, the same launcher — only the object bytes behind `cells_view/<view>/alphasteer` change. A nominal
strength is NOT comparable across object types (a LoRA write-basis multiplier, an AlphaSteer base_scale, an Arditi
residual fraction are three different knobs) nor across two AlphaSteer objects with different layer sets: the arm
keys `alphasteer_as_fit` and `alphasteer_rebake` must never be pooled. The physical comparison is the
separately measured `realised_dose_median`, never these labels.

THE 512 RUNG IS A LADDER RUNG, NOT A VERDICT: at-cap responses at 512 are the ladder
working — the controls phase re-draws exactly those items at 2048 and then 8192 (window 4864 / 8704 on nemotron,
4096 / 10240 elsewhere), and a pair leaves the ladder when its previous-rung count is zero. Nothing in this launcher
reads a 512 at-cap count as a defect; amending the verdict tool's clause (b) is out of scope here.
"""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

_HERE = Path(__file__).resolve().parent
_LANE = _HERE.parent                      # scripts/alphasteer_f22
REPO = _HERE.parents[2]
for _p in (str(_LANE),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from as22_strength import (BANK_SUBDIR, CELL_ID, STRENGTHS, manifest_rel, serve_config_rel,  # noqa: E402
                           strength_of_view, view_name)
from as22_targets import LADDER_CELLS, TARGETS as LANE_TARGETS  # noqa: E402

#: the paper's targets (glm46vflash dropped). The lane still bakes / stages / rosters glm; the SCREEN does not run it.
SCREEN_TARGETS: Tuple[str, ...] = ("gemma4e4b", "qwen35_4b", "nemotron3nano4b")
DESCOPED_TARGETS: Tuple[str, ...] = tuple(t for t in LANE_TARGETS if t not in SCREEN_TARGETS)
#: the strength views, in ladder order — THE ONE LIST the shell gate asks (`--is-in-scope`), never restated
SCREEN_VIEWS: Dict[str, Tuple[str, ...]] = {t: tuple(view_name(s) for s in STRENGTHS) for t in SCREEN_TARGETS}
#: the wave sha the LoRA dose screens ran from (cluster B standup_jobs.jsonl `sha` of f22s-<TAG>-ctl0 jobs 90195, 90198,
#: 93739-93742): the CODE side of every screen job. A different sha is a different instrument — the wrapper records it
#: and the report must say so.
LORA_SCREEN_WAVE_SHA = "f38d055d59a7102f930e30d3abac64adbcf1270b"
#: the code sha the screens RUN at: LORA_SCREEN_WAVE_SHA + ONE commit (branch asscreen-d87-liveness) — at low
#: dose liveness is NUMERIC and a text-divergence refusal alone does not stop a cell. The commit touches two scripts and adds
#: scripts/sdfbase/full22/low_dose_liveness_policy.py; NOTHING under src/ (so the manifests' serving_impl_commit stays true),
#: and it is INERT unless SISTEER_LOW_DOSE_LIVENESS_RECORD names a `live_numeric` record for the cell and standup in hand. So
#: the generation path, the decode and every other refusal are byte-identical to the LoRA screens'; what differs is only
#: whether a live-and-small cell is allowed to generate at all.
LOW_DOSE_LIVENESS_WAVE_SHA = "ec72674809050dad3f0fec4c6b428ee50fc47b7c"
#: the shas whose screens are comparable with the LoRA dose screens', and what to say about each
COMPARABLE_WAVE_SHAS = {
    LORA_SCREEN_WAVE_SHA: "the LoRA dose screens' own sha — one instrument, no deviation",
    LOW_DOSE_LIVENESS_WAVE_SHA: ("the LoRA dose screens' sha + TWO commits, neither touching src/sisteer/capeval/serving/ "
                                 "or tiers/petri_runner.py (so the manifests' serving_impl_commit stays true): (1) the D87 "
                                 "low-dose numeric-liveness seam, scripts-only and inert without "
                                 "SISTEER_LOW_DOSE_LIVENESS_RECORD; (2) registering the Class B fitter in the provenance "
                                 "READER (sweep.CLASS_B_FIT_WRITERS, matched on the writer head) — without it the driver "
                                 "refuses EVERY AlphaSteer object at _mirror_provenance, as canary 93844 measured. The "
                                 "generation path, the decode and every readout are byte-identical to the LoRA screens'; "
                                 "what differs is only WHICH objects are admitted to serve at all"),
}
#: the screen job's name prefix (the task's `asscr-`); `-ctl<T>` is the sampled submitter's suffix for the controls phase
JOB_PREFIX = "asscr"
ROSTER_REL = "configs/alphasteer_f22_strength_roster.yaml"
PINS_REL = "configs/alphasteer_f22_strength_pins.json"
#: the wave-side names the sampled configs carry (make_sampled_configs.py's caps / rungs, read from that module by the generator)
RUN_TAG_PREFIX = "alphasteer_f22"


class ScreenRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"as22_screen: REFUSED — {msg}")


# --- scope -------------------------------------------------------------------------------------------------------

def tag_of(target: str, view: str) -> str:
    """The wave's TAG for a non-main view: `<target>_<view>` (common.sh f22_resolve_config)."""
    return f"{target}_{view}"


def in_scope(target: str, view: str) -> bool:
    return view in SCREEN_VIEWS.get(target, ())


def check_scope(target: str, view: str) -> None:
    """Refuse loudly, naming WHY: a descoped target, an unknown target, or a view that is not one of the five."""
    if target in DESCOPED_TARGETS:
        raise ScreenRefusal(f"target {target!r} is DESCOPED from the paper (strength-frontier D62 / D70) — its strength views "
                            f"exist but the screen does not run them; screen targets: {list(SCREEN_TARGETS)}")
    if target not in SCREEN_TARGETS:
        raise ScreenRefusal(f"unknown target {target!r}; screen targets: {list(SCREEN_TARGETS)}")
    if not in_scope(target, view):
        raise ScreenRefusal(f"view {view!r} is not a strength view of {target} ({list(SCREEN_VIEWS[target])})")
    strength_of_view(view)   # the canonical spelling, or a refusal


def split_tag(tag: str) -> Tuple[str, str]:
    for target, views in SCREEN_VIEWS.items():
        for v in views:
            if tag == tag_of(target, v):
                return target, v
    raise ScreenRefusal(f"{tag!r} is not a (target, view) in the screen scope {screen_tags()}")


def screen_tags() -> List[str]:
    return [tag_of(t, v) for t, vs in SCREEN_VIEWS.items() for v in vs]


def job_name(target: str, view: str, trial: int = 0) -> str:
    return f"{JOB_PREFIX}-{tag_of(target, view)}-ctl{int(trial)}"


# --- the sampled configs (the generator writes them; the wrapper reads them) ---------------------------------------

def sampled_name(target: str, view: str, trial: int, cap: Optional[int] = None) -> str:
    """out_dir / private_run of a sampled variant: `alphasteer_f22_<target>_<view>_sampled_t<T>[_cap<rung>]` — the wave's
    `sdfbase_full22_<TAG>_sampled_t<T>[_cap<rung>]` with the lane's run-tag prefix."""
    return f"{RUN_TAG_PREFIX}_{tag_of(target, view)}_sampled_t{int(trial)}" + (f"_cap{int(cap)}" if cap else "")


def sampled_config_rel(target: str, view: str, trial: int, cap: Optional[int] = None) -> str:
    """`configs/alphasteer_f22_serve_<target>_<view>_sampled_t<T>[_cap<rung>].yaml` — beside the serve config it derives from."""
    return f"configs/alphasteer_f22_serve_{tag_of(target, view)}_sampled_t{int(trial)}" + (f"_cap{int(cap)}" if cap else "") + ".yaml"


# --- the roster (the data the wrapper's bank-view check binds to) ---------------------------------------------------

def load_roster(repo: Path = REPO) -> dict:
    return yaml.safe_load((Path(repo) / ROSTER_REL).read_text(encoding="utf-8"))


def roster_view(roster: dict, target: str, view: str) -> dict:
    try:
        v = roster["targets"][target]["views"][view]
    except KeyError as e:
        raise ScreenRefusal(f"{ROSTER_REL} names no view {target}/{view} ({e})") from None
    s = strength_of_view(view)
    if float(v.get("strength")) != s:
        raise ScreenRefusal(f"{ROSTER_REL} {target}/{view}: strength {v.get('strength')!r} != the view name's {s!r}")
    if v.get("config") != serve_config_rel(target, s) or v.get("manifest") != manifest_rel(target, s):
        raise ScreenRefusal(f"{ROSTER_REL} {target}/{view}: config/manifest {v.get('config')} / {v.get('manifest')} are not "
                            f"as22_strength's {serve_config_rel(target, s)} / {manifest_rel(target, s)}")
    if list(v.get("cells") or {}) != [CELL_ID]:
        raise ScreenRefusal(f"{ROSTER_REL} {target}/{view}: cells {list(v.get('cells') or {})} != [{CELL_ID!r}]")
    return v


def resolve(target: str, view: str, trial: int, *, repo: Path, shared: str) -> Dict[str, str]:
    """Everything the wrapper needs, as ABSOLUTE paths under the two roots: the AlphaSteer-screen checkout `repo`
    (configs / manifest / roster / pins) and `shared` ($SHARED: the bank + view dir). Refuses a missing file so the job
    dies before a GPU minute, with the path named."""
    check_scope(target, view)
    repo = Path(repo)
    s = strength_of_view(view)
    roster = load_roster(repo)
    v = roster_view(roster, target, view)
    tag = tag_of(target, view)
    cfg = repo / serve_config_rel(target, s)
    sampled = {c: repo / sampled_config_rel(target, view, trial, c) for c in (None, 8192, 16384)}
    manifest = repo / manifest_rel(target, s)
    bank = Path(shared) / roster["bank_root"] / target / BANK_SUBDIR
    view_dir = bank / "cells_view" / view
    for p in (cfg, manifest, *sampled.values(), repo / ROSTER_REL, repo / PINS_REL):
        if not p.is_file():
            raise ScreenRefusal(f"missing {p} (sampled variants: scripts/alphasteer_f22/screen/make_strength_sampled_configs.py)")
    served = list(yaml.safe_load(cfg.read_text(encoding="utf-8"))["explicit_cells"])
    if served != list(LADDER_CELLS):
        raise ScreenRefusal(f"{cfg}: explicit_cells {served} != {list(LADDER_CELLS)}")
    if roster.get("bank_subdir") != BANK_SUBDIR:
        raise ScreenRefusal(f"{ROSTER_REL}: bank_subdir {roster.get('bank_subdir')!r} != {BANK_SUBDIR!r}")
    return {
        "KEY": target, "TARGET": target, "VIEW": view, "TAG": tag, "TRIAL": str(int(trial)),
        "STRENGTH": repr(s), "CELL": CELL_ID, "STAGED_ID": v["cells"][CELL_ID],
        "WAVE_CFG": str(cfg), "CFG": str(sampled[None]), "CFG_ESC": str(sampled[8192]), "CFG_ESC16": str(sampled[16384]),
        "MANIFEST": str(manifest),
        "PRIVATE_RUN_BASE": sampled_name(target, view, trial), "PRIVATE_RUN_ESC": sampled_name(target, view, trial, 8192),
        "PRIVATE_RUN_ESC16": sampled_name(target, view, trial, 16384), "SIBLING_TRIAL": str(1 - int(trial)),
        "AS_ROSTER": str(repo / ROSTER_REL), "AS_PINS": str(repo / PINS_REL), "AS_BANK": str(bank), "AS_VIEW_DIR": str(view_dir),
        "AS_SERVED_CELLS": ",".join(served), "AS_STAGED_CELLS": CELL_ID,
        "JOB_NAME": job_name(target, view, trial), "LORA_SCREEN_WAVE_SHA": LORA_SCREEN_WAVE_SHA,
        "LOW_DOSE_LIVENESS_WAVE_SHA": LOW_DOSE_LIVENESS_WAVE_SHA,
    }


def shell_assignments(d: Dict[str, str]) -> str:
    return "\n".join(f"{k}={shlex.quote(v)}" for k, v in d.items())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--is-in-scope", nargs=2, metavar=("TARGET", "VIEW"),
                    help="exit 0 (printing the TAG) when (TARGET, VIEW) is a screen view, 3 when not (why, on stderr); reads no file")
    ap.add_argument("--tags", action="store_true", help="print the fifteen screen TAGs, one per line; reads no file")
    ap.add_argument("--job-name", nargs=2, metavar=("TARGET", "VIEW"), help="print the screen job name for (TARGET, VIEW) at --trial")
    ap.add_argument("--resolve", nargs=2, metavar=("TARGET", "VIEW"),
                    help="print shell assignments (KEY=… one per line, shlex-quoted) for the wrapper: absolute config / manifest / "
                         "roster / pins / bank / view-dir paths under --repo and --shared")
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--repo", default=str(REPO), help="the AlphaSteer-screen checkout (default: this file's)")
    ap.add_argument("--shared", default=None, help="$SHARED on the cluster (its shared root: the site config's CLUSTER_SHARED_ROOT)")
    args = ap.parse_args(argv)
    if args.is_in_scope:
        t, v = args.is_in_scope
        try:
            check_scope(t, v)
        except ScreenRefusal as e:
            print(str(e), file=sys.stderr)
            return 3
        print(tag_of(t, v))
        return 0
    if args.tags:
        print("\n".join(screen_tags()))
        return 0
    if args.job_name:
        t, v = args.job_name
        check_scope(t, v)
        print(job_name(t, v, args.trial))
        return 0
    if args.resolve:
        if not args.shared:
            raise ScreenRefusal("--resolve needs --shared")
        t, v = args.resolve
        print(shell_assignments(resolve(t, v, args.trial, repo=Path(args.repo), shared=args.shared)))
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
