"""alphasteer-full22 STRENGTH FRONTIER (paper Figure 5, the strength lane): the ONE table of nominal
strengths, view names, LANES and paths the lane's tooling shares.

Three consumers must agree on which strength lives in which view dir, or a standup resolves a dir the
bake never wrote: `bake_strength.py` (lays the per-strength cell dirs out under the target's
`strength/cells/`), `make_strength_configs.py` (names each view in `defenses_dir` and derives the
serving config / manifest per (target, strength)) and `make_strength_pins.py` (pins each staged id's
object sha for make_bank_view.py). So the assignment lives here once, and
`tests/test_alphasteer_f22_strength.py` pins it against each lane's roster.

THE LADDER: nominal `base_scale` values -0.5, 0.1, 0.3, 0.5, 1.0 — read as the served op's coefficient
directly, NOT as multiples of the campaign's selected rung. `1.0` IS the lambda=1 parent object
bit-for-bit (`<root>/<target>/cells/alphasteer/`) and is served from its own view as a same-pass anchor
— never re-saved, never re-scaled. Nominal strengths are not comparable across object types, and that
does not matter: the physical comparison comes from the separately measured realised dose.

VIEW NAMES parse the dose the way the LoRA lane's do: `as_` + `d` + the strength with
`.` -> `p` and a leading `m` for a negative sign — `as_dm0p5 as_d0p1 as_d0p3 as_d0p5 as_d1`. The staged
id is `<view>__alphasteer`; the cell id the code sees is still the bare Class B `alphasteer`.

LANES — the SAME ladder over a different set of lambda=1 parents. A lane is ONE roster file: its TARGETS
are that roster's `targets` keys (adding a target is a roster entry plus an argument, never an edit here
or in any script), and its bank root, parent lineage and each target's template config / manifest all
come from that file. Two lanes today:
  * `alphasteer_f22`         the campaign's Phase B parents (gemma 20 / qwen 13 / nemotron 19 / glm 18
                             layers) under $SHARED/exp/objects/alphasteer_f22/<target>/;
  * `alphasteer_f22_rebake`  the RE-BAKED parents (paper-sized, threshold-selected, NON-CONTIGUOUS layer
                             sets: gemma 13 = 12-22,24,25; qwen 9 = 9,12-19; nemotron 13 = 13-25; Llama
                             10-19 pending) under the PARALLEL root
                             $SHARED/exp/objects/alphasteer_f22_rebake/<target>/ — the original objects
                             were never mutated; both roots exist side by side.
The strength views of the two lanes share their NAMES (they live under different bank roots) and differ in
their generated config / manifest paths and run tags (`<lane>_<target>_<view>`).

LAYOUT, per target (a bank of its own so nothing of the parent lane's is touched; make_bank_view.py's
shape, `--bank <root>/<target>/strength`):
    <root>/<target>/strength/cells/<view>__alphasteer/     the staged copy (bake_strength.py)
    <root>/<target>/strength/cells_view/<view>/alphasteer  -> ../../cells/<view>__alphasteer  (make_bank_view.py)
    <root>/<target>/strength/strength_bake.json            the bake record

Import-light on purpose (stdlib, plus yaml only when a lane's roster is read): the bake runs on a cluster
CPU allocation and the generators run on the grading host; neither needs torch to know where things go.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from as22_targets import CELL_ID  # noqa: F401  (re-exported)

REPO = Path(__file__).resolve().parents[2]

#: the nominal base_scale ladder, in the order the views are listed
STRENGTHS: Tuple[float, ...] = (-0.5, 0.1, 0.3, 0.5, 1.0)
#: the strength that IS the parent: served from its own view, never baked (same-pass anchor)
ANCHOR_STRENGTH = 1.0
VIEW_PREFIX = "as_"
#: the per-target bank under <root>/<target>/ (a sibling of the parent lane's cells/, cells_view/, ladder/)
BANK_SUBDIR = "strength"
#: the decode caps the strength views carry. BoN is the campaign's 512 STORE, which the campaign's 512 -> 4,096
#: climb tooling then escalates (superseding a native 4,096: the frontier's three method lanes must share ONE
#: best-of-N estimand or the methods are not comparable with each other, which is the comparison Figure 5
#: exists to make; the LoRA lane moved to 512 + climb, and Arditi is built to it).
#: The multi-turn target replies and the static single-shot families keep the campaign's caps.
BON_MAX_NEW_TOKENS = 512
TARGET_MAX_NEW_TOKENS = 512
STATIC_MAX_NEW_TOKENS = 4096


class StrengthRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"as22_strength: REFUSED — {msg}")


# --- lanes -------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Lane:
    name: str                 #: the lane key (`--lane`), also the run-tag prefix
    bank_root_rel: str        #: under $SHARED on either cluster: <bank_root_rel>/<target>/strength/
    roster_rel: str           #: the ONE place the lane's targets live
    pins_rel: str             #: GENERATED by make_strength_pins.py from the bake records
    config_prefix: str        #: configs/<config_prefix><target>_<view>.yaml
    manifest_prefix: str      #: configs/capeval/manifests/<manifest_prefix><target>_<view>.yaml
    records_subdir: str       #: the grading host: $GRADE/<records_subdir>/<target>/strength_bake.json
    parent_lineage: str       #: `campaign_lambda1` (cross-checked against the campaign pins) or `rebake`

    @property
    def roster_path(self) -> Path:
        return REPO / self.roster_rel

    @property
    def pins_path(self) -> Path:
        return REPO / self.pins_rel

    @property
    def bank_root_name(self) -> str:
        return Path(self.bank_root_rel).name


LANES: Dict[str, Lane] = {
    "alphasteer_f22": Lane(
        name="alphasteer_f22", bank_root_rel="exp/objects/alphasteer_f22",
        roster_rel="configs/alphasteer_f22_strength_roster.yaml", pins_rel="configs/alphasteer_f22_strength_pins.json",
        config_prefix="alphasteer_f22_serve_", manifest_prefix="alphasteer_f22_",
        records_subdir="alphasteer_f22/strength", parent_lineage="campaign_lambda1"),
    "alphasteer_f22_rebake": Lane(
        name="alphasteer_f22_rebake", bank_root_rel="exp/objects/alphasteer_f22_rebake",
        roster_rel="configs/alphasteer_f22_rebake_strength_roster.yaml",
        pins_rel="configs/alphasteer_f22_rebake_strength_pins.json",
        config_prefix="alphasteer_f22_rebake_serve_", manifest_prefix="alphasteer_f22_rebake_",
        records_subdir="alphasteer_f22_rebake/strength", parent_lineage="rebake"),
}
DEFAULT_LANE = "alphasteer_f22"
PARENT_LINEAGES = ("campaign_lambda1", "rebake")


def lane(name: Optional[str] = None) -> Lane:
    key = name or DEFAULT_LANE
    try:
        return LANES[key]
    except KeyError:
        raise StrengthRefusal(f"unknown lane {name!r}; lanes: {sorted(LANES)}")


def load_roster(ln: Lane) -> dict:
    import yaml

    if not ln.roster_path.is_file():
        raise StrengthRefusal(f"lane {ln.name}: roster {ln.roster_rel} missing")
    r = yaml.safe_load(ln.roster_path.read_text(encoding="utf-8"))
    if r.get("bank_root") != ln.bank_root_rel or r.get("parent_lineage") != ln.parent_lineage:
        raise StrengthRefusal(f"lane {ln.name}: roster bank_root / parent_lineage {r.get('bank_root')!r} / "
                              f"{r.get('parent_lineage')!r} != the lane's {ln.bank_root_rel!r} / {ln.parent_lineage!r}")
    if not r.get("targets"):
        raise StrengthRefusal(f"lane {ln.name}: roster names no targets")
    return r


def lane_targets(ln: Lane, roster: Optional[dict] = None) -> List[str]:
    """THE target set of a lane: the roster's `targets` keys, in roster order — never a literal."""
    return list((roster or load_roster(ln))["targets"])


def require_target(ln: Lane, target: str, roster: Optional[dict] = None) -> dict:
    r = roster or load_roster(ln)
    if target not in r["targets"]:
        raise StrengthRefusal(f"lane {ln.name}: target {target!r} is not in {ln.roster_rel} "
                              f"({list(r['targets'])}) — add it there deliberately, never guess")
    return r["targets"][target]


def roster_parent_layers(tinfo: dict) -> Optional[List[int]]:
    """The roster's DECLARED parent layer set for a target (`parent_layers`), validated: ints, strictly
    ascending, no contiguity assumed. None when the roster does not declare one."""
    pl = tinfo.get("parent_layers")
    if pl is None:
        return None
    if not isinstance(pl, list) or not pl or not all(isinstance(l, int) and not isinstance(l, bool) for l in pl):
        raise StrengthRefusal(f"parent_layers {pl!r} is not a non-empty list of ints")
    if any(b <= a for a, b in zip(pl, pl[1:])):
        raise StrengthRefusal(f"parent_layers {pl!r} is not strictly ascending")
    return list(pl)


def layer_gaps(layers: List[int]) -> List[int]:
    """The integers missing between min and max of a layer set — recorded, never filled (the threshold's
    selection is allowed to skip layers: gemma's re-bake skips 23, qwen's skips 10 and 11)."""
    ls = sorted(int(l) for l in layers)
    if not ls:
        return []
    present = set(ls)
    return [l for l in range(ls[0], ls[-1] + 1) if l not in present]


# --- strengths ---------------------------------------------------------------------------------

def validate_strength(s) -> float:
    """A nominal strength is a finite, non-zero real number. 0 is refused (a no-op edit is `none`,
    not a dose); bools, NaN and infinities are refused; anything that is not a number is refused."""
    if isinstance(s, bool) or not isinstance(s, (int, float)):
        raise StrengthRefusal(f"strength {s!r} is not a number")
    f = float(s)
    if not math.isfinite(f):
        raise StrengthRefusal(f"strength {s!r} is not finite")
    if f == 0.0:
        raise StrengthRefusal("strength 0 is refused — a zero edit is the `none` anchor, not a dose")
    return f


def parse_strength(text: str) -> float:
    try:
        return validate_strength(float(str(text).strip()))
    except ValueError as e:
        raise StrengthRefusal(f"strength {text!r} does not parse as a number") from e


def format_strength(s) -> str:
    """`1.0` -> `1`, `0.1` -> `0p1`, `-0.5` -> `m0p5`, `1.5` -> `1p5`, `0.25` -> `0p25`."""
    f = validate_strength(s)
    a = abs(f)
    text = f"{a:.10g}"
    if "e" in text or "E" in text:
        raise StrengthRefusal(f"strength {s!r} has no positional decimal spelling ({text}); refusing to name a view for it")
    return ("m" if f < 0 else "") + text.replace(".", "p")


def view_name(s) -> str:
    return f"{VIEW_PREFIX}d{format_strength(s)}"


def strength_of_view(view: str) -> float:
    """The inverse of :func:`view_name`; refuses a view that is not the CANONICAL spelling of its
    strength (`as_d0p10`, `as_d+0p1`, a foreign prefix), so a typo never serves a dose by accident."""
    if not isinstance(view, str) or not view.startswith(VIEW_PREFIX + "d"):
        raise StrengthRefusal(f"view {view!r} is not a strength view ({VIEW_PREFIX}d...)")
    body = view[len(VIEW_PREFIX) + 1:]
    neg = body.startswith("m")
    if neg:
        body = body[1:]
    if not body or not all(c.isdigit() or c == "p" for c in body):
        raise StrengthRefusal(f"view {view!r}: dose spelling {body!r} is not digits with one `p`")
    s = parse_strength(("-" if neg else "") + body.replace("p", "."))
    if view_name(s) != view:
        raise StrengthRefusal(f"view {view!r} is not the canonical spelling of {s!r} ({view_name(s)})")
    return s


def is_anchor(s) -> bool:
    return validate_strength(s) == ANCHOR_STRENGTH


def staged_id(s) -> str:
    return f"{view_name(s)}__{CELL_ID}"


def strength_of_staged_id(sid: str) -> float:
    suffix = f"__{CELL_ID}"
    if not isinstance(sid, str) or not sid.endswith(suffix):
        raise StrengthRefusal(f"staged id {sid!r} does not end in {suffix!r}")
    return strength_of_view(sid[: -len(suffix)])


def run_tag(target: str, s, ln: Optional[Lane] = None) -> str:
    """`<lane>_<target>_<view>` (spec: Tag = <target>_<view>, namespaced by the lane)."""
    ln = ln or lane()
    return f"{ln.name}_{target}_{view_name(s)}"


# --- the objects root layout -------------------------------------------------------------------

def parent_cell_dir(root: Path, target: str) -> Path:
    """`<root>/<target>/cells/alphasteer/` — the lambda=1 parent of whichever lane `root` belongs to
    (the same path shape as as22_targets.parent_cell_dir; restated so a lane needs no campaign import)."""
    return Path(root) / target / "cells" / CELL_ID


def strength_bank_dir(root: Path, target: str) -> Path:
    """`<root>/<target>/strength/` — the bank make_bank_view.py is pointed at (`--bank`)."""
    return Path(root) / target / BANK_SUBDIR


def strength_cells_dir(root: Path, target: str) -> Path:
    return strength_bank_dir(root, target) / "cells"


def strength_cell_dir(root: Path, target: str, s) -> Path:
    return strength_cells_dir(root, target) / staged_id(s)


def strength_view_dir(root: Path, target: str, s) -> Path:
    """The per-strength DEFENSES DIR a standup is pointed at (`--defenses-dir` /
    `TT_TWO_SI_DEFENSES_DIR`): `<root>/<target>/strength/cells_view/<view>` (built by make_bank_view.py)."""
    return strength_bank_dir(root, target) / "cells_view" / view_name(s)


def bake_record_path(root: Path, target: str) -> Path:
    return strength_bank_dir(root, target) / "strength_bake.json"


# --- the generated files -----------------------------------------------------------------------

def serve_config_rel(target: str, s, ln: Optional[Lane] = None) -> str:
    ln = ln or lane()
    return f"configs/{ln.config_prefix}{target}_{view_name(s)}.yaml"


def manifest_rel(target: str, s, ln: Optional[Lane] = None) -> str:
    ln = ln or lane()
    return f"configs/capeval/manifests/{ln.manifest_prefix}{target}_{view_name(s)}.yaml"


def defenses_dir_placeholder(target: str, s, ln: Optional[Lane] = None) -> str:
    """The config's `defenses_dir` value: the wave's `out/...` placeholder shape, resolved on the
    cluster by the job's TT_TWO_SI_DEFENSES_DIR export, which for a lane is
    $SHARED/<bank_root>/<target>/strength/cells_view/<view>."""
    ln = ln or lane()
    return f"out/{ln.bank_root_name}/{target}/{BANK_SUBDIR}/cells_view/{view_name(s)}"
