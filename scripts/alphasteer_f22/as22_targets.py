"""alphasteer-full22: the ONE table of targets, views and paths the Phase C tooling shares.

Two consumers — `bake_rungs.py` (lays out the per-rung defenses dirs under `cells_view/<view>/`)
and `make_ladder_configs.py` (names that view in `defenses_dir` and derives each serving config /
manifest from the wave's config for the same (target, view)) — must agree on which VIEW a target
serves its AlphaSteer cell from, or the standup resolves a dir the bake never wrote. So the
assignment lives here once, and `tests/test_alphasteer_f22_bake.py` pins it against the wave
roster (`configs/sdfbase_full22_roster.yaml`: the view must exist for the target and the derived
config must be the roster's config for that view).

View assignment (plan Phase F, spec A2): `it` for gemma4e4b and qwen35_4b — AlphaSteer is fit on
and served by the INSTRUCT model, which is the model the wave's `_it` configs serve — and `main`
for glm46vflash and nemotron3nano4b, whose wave has only a `main` view (their served model IS the
fit model). The table is keyed generically: a target absent from it is a loud refusal in both
consumers, never a guessed view.

Import-light on purpose (stdlib only): the bake runs on a cluster login shell and the config
generator runs on the grading host; neither needs torch to know where things go.
"""
from __future__ import annotations

from pathlib import Path

#: the four served targets of the campaign, in the wave's order
TARGETS = ("gemma4e4b", "qwen35_4b", "glm46vflash", "nemotron3nano4b")
#: wave-roster targets this table does NOT BAKE — the roster names what exists, TARGETS names what this lane's MINT
#: (bake_rungs.py / make_ladder_configs.py) lays out; tests/test_alphasteer_f22_bake.py pins TARGETS + these == the roster's
#: targets, so a new roster target is a loud decision here, never a silent omission. llama8b (a roster target) is
#: NOT minted here — its AlphaSteer object comes from the OFF-ROSTER wave source pinned in
#: scripts/arditi/arditi_targets.py (OFF_ROSTER_WAVE_SOURCES: branch sdfbase-llama8b-wave @ d358287b, per-file sha256s,
#: so the AlphaSteer and Arditi lanes derive llama8b from the same bytes). The lane's lambda ladder DOES serve it, at view
#: `it` (ladder_select.py; $GRADE/alphasteer_f22_rebake/ladder_canon/llama8b/lambda_ladder.json; its rung-1
#: object 4cf8e1641a596d2d… is the paper's llama AlphaSteer operating point) — the exclusion is the bake table, not the lane.
#: A front-door target of reproduce/models.yaml that joins the roster is declared here, not omitted: the front door
#: bakes no AlphaSteer ladder for its model (the table declares no alphasteer object; its cells are the ReLU, LoRA and difference
#: objects), and the table test goes red the day the roster gains a target this tuple does not name. qwen35_4b_demo (the worked
#: example's roster block, c7e73cd04, 2026-10-10) is that target: its battery / d0p75 / d0p75_relu views carry the refusal direction,
#: the LoRA and ReLU objects and their differences from the front-door object banks, never an AlphaSteer object — declared, not baked.
ROSTER_TARGETS_NOT_BAKED_HERE = ("llama8b", "qwen35_4b_demo")
#: target -> the bank VIEW its AlphaSteer cell is served from (plan Phase F / spec A2)
VIEW_OF = {"gemma4e4b": "it", "qwen35_4b": "it", "glm46vflash": "main", "nemotron3nano4b": "main"}
#: the bare Class B cell id (defense_cells.CLASS_B_IDS; registry_adapter.traintest_cell_id and
#: harness_cell_id are both the identity on it)
CELL_ID = "alphasteer"
#: the canonical trio of every ladder standup is [none, none__lens0 (DERIVED by
#: defense_cells.enumerate_cells, never listed), alphasteer] — what a config / manifest LISTS:
LADDER_CELLS = ("none", CELL_ID)
#: the wave's production serving-implementation commit (configs/sdfbase_full22_roster.yaml
#: `serving_impl_commit`). The ladder runs the PATCHED tree W_L and must NEVER record this sha
#: in its manifests; the generator refuses it.
PRODUCTION_SERVING_IMPL_COMMIT = "94e294c445dc5eec6ebe0ce3a3e25553004cd4b0"
#: the marker the W_L-cutting agent greps for to fill the ladder manifests' serving_impl_commit
LADDER_SERVING_IMPL_COMMIT_MARKER = "AS22_LADDER_SERVING_IMPL_COMMIT"


def view_of(target: str) -> str:
    try:
        return VIEW_OF[target]
    except KeyError:
        raise SystemExit(f"target {target!r} has no view assignment in "
                         f"scripts/alphasteer_f22/as22_targets.VIEW_OF ({sorted(VIEW_OF)}) — "
                         "add it deliberately (plan Phase F names the rule), never guess a view")


def wave_tag(target: str) -> str:
    """The wave's file tag for (target, its AlphaSteer view): `<target>` for `main`,
    `<target>_<view>` otherwise (make_full22_configs.py's `tag`)."""
    view = view_of(target)
    return target + ("" if view == "main" else f"_{view}")


def wave_config_rel(target: str) -> str:
    return f"configs/sdfbase_full22_{wave_tag(target)}.yaml"


def wave_manifest_rel(target: str) -> str:
    return f"configs/capeval/manifests/petri_full22_{wave_tag(target)}.yaml"


def serve_config_rel(target: str) -> str:
    """NOTE the name deviates from `configs/alphasteer_f22_<target>.yaml`: that path is
    Phase B's FIT config on branch alphasteer-multilayer (cherry-picked by PR-B), so the SERVING
    config carries `_serve_`."""
    return f"configs/alphasteer_f22_serve_{target}.yaml"


def manifest_rel(target: str) -> str:
    return f"configs/capeval/manifests/alphasteer_f22_{target}.yaml"


# --- the objects root layout (plan Phase F + §C.2; LADDER_CONTRACT.md "Per-rung directory") ----

def parent_cell_dir(root: Path, target: str) -> Path:
    """The Phase B fit: `<root>/<target>/cells/alphasteer/` (object.pt, fit_meta.json,
    fit_provenance.json, meta.json)."""
    return Path(root) / target / "cells" / CELL_ID


def ladder_dir(root: Path, target: str) -> Path:
    return Path(root) / target / "ladder"


def rung_dir(root: Path, target: str, i: int) -> Path:
    return ladder_dir(root, target) / f"lam{int(i)}"


def rung_view_dir(root: Path, target: str, i: int) -> Path:
    """The per-rung DEFENSES DIR the standup is pointed at (`--defenses-dir` /
    `TT_TWO_SI_DEFENSES_DIR`): `<root>/<target>/ladder/lam<i>/cells_view/<view>`."""
    return rung_dir(root, target, i) / "cells_view" / view_of(target)


def rung_cell_dir(root: Path, target: str, i: int) -> Path:
    return rung_view_dir(root, target, i) / CELL_ID
