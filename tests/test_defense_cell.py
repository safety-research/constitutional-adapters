"""DefenseCell — the two-SI defense identity refactor.

CPU-only, no network, no harmful text (synthetic prompts only). Proves the
identity layer that stops the two SIs' objects for one arm silently
overwriting each other:

- every cell id is unique, and the matrix count DERIVES from the registry
  (never a hard-coded 16/19).
- the regression: the two SI versions of ONE arm produce DISTINCT object
  paths AND DISTINCT aggregate entries — written so it would FAIL against the
  old arm-keyed scheme (which is reproduced inline to show the collision).
- a Class B cell has ``si_id is None``; the two prompted baselines are distinct.
- the completeness check cannot silently drop a cell.
- backwards compatibility: a single-SI-lane bare name resolves byte-identically
  to the old arm-keyed path.
- the sbatch array decode, meta.json, and the over-refusal reference
  plumbing all key on ``cell.id``.
- ``build_defense_cell`` serves a compiled cell's object from its cell-id-keyed
  path and gives each prompted baseline its OWN SI text.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from sisteer.eval.defense_cell import (  # noqa: E402
    CLASS_A_ARMS,
    CLASS_B_ARMS,
    DEFAULT_SI_IDS,
    ControlComparison,
    DefenseCell,
    array_decode,
    arm_of,
    assert_matrix_complete,
    build_registry,
    by_id,
    cell_for_array_index,
    cell_meta,
    matrix_counts,
    missing_cells,
    object_path_for,
    over_refusal_comparisons,
    registry_from_cfg,
    si_of,
)
from sisteer.methods import ARMS, object_artifact_name  # noqa: E402

DDIR = Path("/defenses")
CONTROLS = ("xstest", "or_bench_hard_1k", "mmlu", "gsm8k")


def _reg():
    return build_registry(DDIR)


# --------------------------------------------------------------------------- #
# ids are unique, and every count derives from the registry                    #
# --------------------------------------------------------------------------- #


def test_all_cell_ids_unique():
    reg = _reg()
    ids = [c.id for c in reg]
    assert len(ids) == len(set(ids)), "duplicate cell id"
    # by_id also guards against dups (a dup silently drops a column)
    assert set(by_id(reg)) == set(ids)


def test_counts_derive_from_registry_not_a_literal():
    """The matrix count comes from si_ids × the REGISTERED Class A/B arms, so
    the count DERIVES from the registry — never a hard-coded literal."""
    reg = _reg()
    n_si = len(DEFAULT_SI_IDS)
    reg_a = [a for a in CLASS_A_ARMS if a in ARMS]
    reg_b = [b for b in CLASS_B_ARMS if b in ARMS]
    counts = matrix_counts(reg)
    assert counts["class_A"] == len(reg_a) * n_si
    assert counts["class_B"] == len(reg_b)
    assert counts["baselines"] == 1 + n_si  # none + one prompted per SI
    assert counts["objects"] == counts["class_A"] + counts["class_B"]
    assert counts["total"] == counts["objects"] + counts["baselines"]
    # cast / circuit_breaker left methods.ARMS ENTIRELY (2026-08-20, user
    # instruction), and left the CLASS_A_ARMS/CLASS_B_ARMS defaults with it —
    # so the derivation shrank to exclude them WITHOUT a count literal
    # changing anywhere: no cell carries either arm, and a STALE config that
    # still names them is filtered out by require_registered.
    assert "cast" not in ARMS and "circuit_breaker" not in ARMS
    assert "cast" not in reg_a and "circuit_breaker" not in reg_b
    assert sum(1 for c in reg if c.arm == "cast") == 0
    assert sum(1 for c in reg if c.arm == "circuit_breaker") == 0
    stale = build_registry(DDIR, class_a_arms=("psr_kl_all", "cast"),
                           class_b_arms=("antidote", "circuit_breaker"))
    assert matrix_counts(stale)["class_A"] == 1 * n_si   # cast filtered out
    assert matrix_counts(stale)["class_B"] == 1          # cb filtered out
    # The derivation is arm-LIST driven: an explicit unfiltered list still
    # grows it (require_registered=False is the only path that can name an
    # unregistered arm, and it is opt-in).
    forced = build_registry(DDIR, class_a_arms=("psr_kl_all", "cast"),
                            class_b_arms=("antidote", "circuit_breaker"),
                            require_registered=False)
    assert matrix_counts(forced)["class_A"] == 2 * n_si  # psr_kl_all + cast
    assert matrix_counts(forced)["class_B"] == 2          # antidote + cb


# --------------------------------------------------------------------------- #
# THE REGRESSION — distinct paths AND distinct aggregate entries              #
# --------------------------------------------------------------------------- #


def test_two_si_versions_of_one_arm_are_distinct_paths_and_entries():
    """Both SI versions of ``psr_kl_all`` must get DISTINCT object paths AND
    DISTINCT aggregate entries. Written to FAIL against the old arm-keyed
    scheme, which is reproduced inline: keyed on the ARM, the two cells collide
    on one path and one dict key, and the second silently overwrites the first
    (exactly the defect cell-id keying fixes)."""
    reg = by_id(_reg())
    a = reg["safety_refusal__psr_kl_all"]
    b = reg["si_constitution__psr_kl_all"]
    assert a.arm == b.arm == "psr_kl_all"     # SAME arm
    assert a.si_id != b.si_id                  # DIFFERENT SI
    assert a.id != b.id                        # DIFFERENT cell id

    # --- distinct object paths (keyed on cell.id, filename on cell.arm) ------
    assert a.object_path != b.object_path
    assert a.object_path == DDIR / a.id / object_artifact_name("psr_kl_all")
    assert b.object_path == DDIR / b.id / object_artifact_name("psr_kl_all")

    # The OLD arm-keyed resolution COLLIDES — demonstrate the bug directly.
    old_path = lambda cell: DDIR / cell.arm / object_artifact_name(cell.arm)
    assert old_path(a) == old_path(b)          # <- the collision

    # --- distinct aggregate entries (keyed on cell.id) -----------------------
    per_cell: dict[str, str] = {}
    per_cell[a.id] = "result_A"
    per_cell[b.id] = "result_B"
    assert len(per_cell) == 2 and per_cell[a.id] != per_cell[b.id]

    # keyed on the ARM (the old scheme) the two collapse to ONE entry and the
    # second overwrites the first while the aggregate still "looks complete".
    per_arm: dict[str, str] = {}
    per_arm[a.arm] = "result_A"
    per_arm[b.arm] = "result_B"     # silently overwrites result_A
    assert len(per_arm) == 1 and per_arm["psr_kl_all"] == "result_B"


# --------------------------------------------------------------------------- #
# class labels, si_id, prompted baselines                                       #
# --------------------------------------------------------------------------- #


def test_class_b_cells_have_no_si():
    reg = _reg()
    b_cells = [c for c in reg if c.class_ == "B"]
    assert b_cells, "expected at least one registered Class B object"
    for c in b_cells:
        assert c.si_id is None          # SI-agnostic, fit once
        assert c.arm is not None
        assert c.id == c.arm            # bare arm id (degenerates to old-lane name)


def test_class_a_cells_carry_both_arm_and_si():
    reg = _reg()
    a_cells = [c for c in reg if c.class_ == "A"]
    assert a_cells
    for c in a_cells:
        assert c.arm is not None and c.si_id is not None
        assert c.id == f"{c.si_id}__{c.arm}"


def test_two_prompted_baselines_are_distinct():
    reg = by_id(_reg())
    p1 = reg["prompted__safety_refusal"]
    p2 = reg["prompted__si_constitution"]
    assert p1.id != p2.id
    assert p1.si_id == "safety_refusal" and p2.si_id == "si_constitution"
    assert p1.class_ == p2.class_ == "baseline"
    assert p1.arm is None and p2.arm is None
    # exactly two prompted baselines (one per SI), plus the undefended base
    baselines = [c for c in reg.values() if c.class_ == "baseline"]
    prompted = [c for c in baselines if c.id.startswith("prompted__")]
    assert len(prompted) == len(DEFAULT_SI_IDS)
    assert any(c.id == "none" for c in baselines)


def test_defense_cell_validation_rejects_malformed_cells():
    with pytest.raises(ValueError):  # Class A needs si_id
        DefenseCell("x", "A", "psr_kl_all", None)
    with pytest.raises(ValueError):  # Class A needs arm
        DefenseCell("x", "A", None, "safety_refusal")
    with pytest.raises(ValueError):  # Class B must have si_id None
        DefenseCell("x", "B", "antidote", "safety_refusal")
    with pytest.raises(ValueError):  # unknown class
        DefenseCell("x", "C", None, None)


# --------------------------------------------------------------------------- #
# completeness — a cell can never silently drop                                 #
# --------------------------------------------------------------------------- #


def test_completeness_enumerates_by_cell_id():
    reg = _reg()
    all_ids = [c.id for c in reg]
    # full set present -> OK
    assert_matrix_complete(reg, all_ids)
    assert missing_cells(reg, all_ids) == []


def test_completeness_raises_on_a_silently_dropped_cell():
    reg = _reg()
    all_ids = [c.id for c in reg]
    dropped = "si_constitution__psr_kl_all"
    present = [i for i in all_ids if i != dropped]
    assert missing_cells(reg, present) == [dropped]
    with pytest.raises(ValueError, match=dropped):
        assert_matrix_complete(reg, present)
    # a refused-with-reason cell is a first-class outcome, not a silent drop
    assert_matrix_complete(reg, present, refused={dropped: "gate: positive control moved nothing"})
    assert_matrix_complete(reg, present, refused=[dropped])


# --------------------------------------------------------------------------- #
# backwards compatibility — the single-SI lane resolves byte-identically        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", list(ARMS))
def test_bare_arm_path_matches_old_scheme(name):
    """A single-SI-lane bare arm resolves to <defenses_dir>/<arm>/object...
    exactly as the old _defense_object_path did."""
    old = DDIR / name / object_artifact_name(name)
    assert object_path_for(DDIR, name) == old


def test_baselines_have_no_object_path():
    assert object_path_for(DDIR, "none") is None
    assert object_path_for(DDIR, "prompted") is None
    assert object_path_for(DDIR, "prompted__safety_refusal") is None


def test_id_component_parsing_roundtrips():
    assert arm_of("si_constitution__psr_kl_all") == "psr_kl_all"
    assert si_of("si_constitution__psr_kl_all") == "si_constitution"
    assert arm_of("antidote") == "antidote" and si_of("antidote") is None
    assert arm_of("none") is None and si_of("none") is None
    assert arm_of("prompted__safety_refusal") is None
    assert si_of("prompted__safety_refusal") == "safety_refusal"
    # an arm with underscores is recovered whole (split on the FIRST __ only)
    assert arm_of("safety_refusal__psr_kl_all_hard") == "psr_kl_all_hard"


# --------------------------------------------------------------------------- #
# sbatch array decode — the index enumerates cell.ids                           #
# --------------------------------------------------------------------------- #


def test_array_decode_enumerates_every_cell_without_collision():
    reg = _reg()
    n_shards = 4
    seen: set[tuple[str, int]] = set()
    for cell_idx in range(len(reg)):
        for shard in range(n_shards):
            task_id = cell_idx * n_shards + shard
            got_idx, got_shard = array_decode(task_id, n_shards)
            assert (got_idx, got_shard) == (cell_idx, shard)
            cell, sh = cell_for_array_index(reg, task_id, n_shards)
            assert cell is reg[cell_idx] and sh == shard
            seen.add((cell.id, sh))
    # every (cell, shard) is hit exactly once -> no two SI cells collide
    assert len(seen) == len(reg) * n_shards


def test_array_decode_out_of_range_is_loud():
    reg = _reg()
    n_shards = 4
    stale = len(reg) * n_shards  # one past the last valid task
    with pytest.raises(IndexError):
        cell_for_array_index(reg, stale, n_shards)
    with pytest.raises(ValueError):
        array_decode(0, 0)


# --------------------------------------------------------------------------- #
# meta.json sidecar — keyed on cell.id                                          #
# --------------------------------------------------------------------------- #


def test_cell_meta_carries_identity_and_extra():
    reg = by_id(_reg())
    cell = reg["si_constitution__psr_kl_all"]
    meta = cell_meta(cell, fit_data="matched_pairs", completion_tokens=512,
                     positive_control={"held_in_delta": 0.4})
    assert meta["cell_id"] == "si_constitution__psr_kl_all"
    assert meta["class"] == "A"
    assert meta["si_id"] == "si_constitution"
    assert meta["arm"] == "psr_kl_all"
    assert meta["fit_data"] == "matched_pairs"
    assert meta["completion_tokens"] == 512
    assert meta["positive_control"] == {"held_in_delta": 0.4}
    # a Class B object records si_id=None so it can't be read back as an SI cell
    bmeta = cell_meta(reg["antidote"], fit_data="harmful_benign")
    assert bmeta["class"] == "B" and bmeta["si_id"] is None


# --------------------------------------------------------------------------- #
# over-refusal non-inferiority reference plumbing (identity only)               #
# --------------------------------------------------------------------------- #


def test_class_a_ni_reference_is_the_matching_prompted_arm():
    reg = by_id(_reg())
    assert reg["safety_refusal__psr_kl_all"].ni_reference_ids == ("prompted__safety_refusal",)
    assert reg["si_constitution__psr_kl_all"].ni_reference_ids == ("prompted__si_constitution",)


def test_class_b_ni_reference_is_both_prompted_arms_not_the_base():
    reg = by_id(_reg())
    cell = reg["antidote"]
    assert set(cell.ni_reference_ids) == {"prompted__safety_refusal",
                                          "prompted__si_constitution"}
    assert "none" not in cell.ni_reference_ids  # the reference is not the base


def test_over_refusal_comparisons_express_cell_reference_control():
    reg = _reg()
    comps = over_refusal_comparisons(reg, CONTROLS)
    # baselines are references, never compared subjects
    assert all(c.cell_id not in ("none", "prompted__safety_refusal",
                                 "prompted__si_constitution") for c in comps)
    # each is a (cell, reference, control) triple over the SIX controls
    assert all(isinstance(c, ControlComparison) for c in comps)
    assert {c.control for c in comps} == set(CONTROLS)
    n_a = sum(1 for c in reg if c.class_ == "A")
    n_b = sum(1 for c in reg if c.class_ == "B")
    # Class A: 1 reference each; Class B: 2 references each; × 6 controls
    assert len(comps) == (n_a * 1 + n_b * 2) * len(CONTROLS)
    # a concrete Class B triple against BOTH prompted arms exists
    b_refs = {c.reference_id for c in comps if c.cell_id == "antidote"
              and c.control == "xstest"}
    assert b_refs == {"prompted__safety_refusal", "prompted__si_constitution"}


# --------------------------------------------------------------------------- #
# registry_from_cfg — reads si_ids + defenses_dir off the two-SI config          #
# --------------------------------------------------------------------------- #


def test_registry_from_cfg_uses_config_si_ids_and_dir():
    import yaml

    repo = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((repo / "configs" / "two_si.yaml").read_text())
    reg = registry_from_cfg(cfg, repo_root=repo)
    ids = {c.id for c in reg}
    assert "safety_refusal__psr_kl_all" in ids
    assert "si_constitution__psr_kl_all" in ids
    assert "prompted__safety_refusal" in ids and "prompted__si_constitution" in ids
    # object paths root under REPO / defenses_dir
    a = by_id(reg)["si_constitution__psr_kl_all"]
    assert a.object_path == (repo / cfg["defenses_dir"]
                             / a.id / object_artifact_name(a.arm))


# --------------------------------------------------------------------------- #
# CLI surface — the valid --defense set keys on cell.id for the two-SI lane      #
# --------------------------------------------------------------------------- #


def test_cli_valid_defense_ids_two_si_vs_old_lane():
    import yaml

    import jailbreak_run

    repo = Path(__file__).resolve().parents[1]
    two_si = yaml.safe_load((repo / "configs" / "two_si.yaml").read_text())
    valid = jailbreak_run._valid_defense_ids(two_si)
    # the two-SI cell ids are accepted...
    assert "si_constitution__psr_kl_all" in valid
    assert "prompted__si_constitution" in valid
    # ...alongside the single-SI lane's own names (superset)
    from sisteer.eval.jailbreak import DEFENSES

    assert set(DEFENSES) <= valid

    # the old lane (no si_ids in its config) accepts ONLY DEFENSES — a two-SI
    # cell id is correctly rejected there.
    old = yaml.safe_load((repo / "configs" / "stage2_jailbreak.yaml").read_text())
    vold = jailbreak_run._valid_defense_ids(old)
    assert vold == set(DEFENSES)
    assert "si_constitution__psr_kl_all" not in vold


# --------------------------------------------------------------------------- #
# build_defense_cell — serving keys object on cell.id, SI on cell.si_id          #
# --------------------------------------------------------------------------- #


def test_build_defense_cell_prompted_baselines_send_their_own_si():
    """Each of the TWO prompted baselines sends its OWN SI text — the single
    `prompted` identity the old lane had could not distinguish them."""
    from sisteer.eval.jailbreak import build_defense_cell, resolve_instruction

    reg = by_id(_reg())
    seam = lambda rendered, **kw: ["x"] * len(rendered)
    d_safety = build_defense_cell(None, None, reg["prompted__safety_refusal"],
                                  positions="all", generate_fn=seam)
    d_const = build_defense_cell(None, None, reg["prompted__si_constitution"],
                                 positions="all", generate_fn=seam)
    assert d_safety.mode == "prompted" and d_const.mode == "prompted"
    assert d_safety.system == resolve_instruction("safety_refusal").text
    assert d_const.system == resolve_instruction("si_constitution").text
    assert d_safety.system != d_const.system   # distinct SI texts

    d_none = build_defense_cell(None, None, reg["none"], positions="all",
                                generate_fn=seam)
    assert d_none.mode == "none" and d_none.system is None


def test_build_defense_cell_serves_object_from_the_cell_id_path(tiny_llama, tmp_path):
    """A compiled cell's object is loaded from its cell-id-keyed path; two SI
    cells for one arm read DISTINCT artifacts, so one never shadows the other."""
    from sisteer.data.instructions import BY_ID
    from sisteer.eval.jailbreak import build_defense_cell

    model, tok = tiny_llama
    reg = by_id(build_registry(tmp_path))
    a = reg["safety_refusal__psr_kl_all"]
    b = reg["si_constitution__psr_kl_all"]
    assert a.object_path != b.object_path   # Distinct paths

    # train + save ONE object at cell a's path only, with the fit-provenance
    # sidecar the two-SI publish contract always writes beside it
    # (train_two_si_objects stamps positions_at_fit; the serve path reads it
    # to auto-match fit — an all-token psr_kl_all fit here).
    import json as _json
    a.object_path.parent.mkdir(parents=True, exist_ok=True)
    obj = ARMS["psr_kl_all"].train(model, tok, BY_ID["french"], _pairs(),
                                   layer=1, epochs=1, batch_size=2, max_len=64)
    obj.save(a.object_path)
    (a.object_path.parent / "train_budget_meta.json").write_text(
        _json.dumps({"cell_id": a.id, "arm": "psr_kl_all",
                     "positions_at_fit": "all"}))

    d = build_defense_cell(model, tok, a, positions="all")
    assert d.mode == "psr_kl_all" and d._obj is not None   # object from cell.arm
    assert d.positions_at_fit == "all" and d.positions_served == "all"
    out = d.generate(["hello there", "two plus two"], max_new_tokens=3)
    assert isinstance(out, list) and len(out) == 2

    # cell b's object (and its sidecar) were never written -> serving it is a
    # loud miss, proving the artifacts are isolated per cell (no silent
    # overwrite/shadow). The missing fit-provenance record refuses first
    # (ValueError), before any load of the absent object.
    assert not b.object_path.exists()
    with pytest.raises((FileNotFoundError, OSError, ValueError)):
        build_defense_cell(model, tok, b)


def _pairs():
    from sisteer.data.pairs import Pair

    return [
        Pair(prompt="Say something nice.", pos="Bonjour le monde",
             neg="Hello world", instruction_id="french"),
        Pair(prompt="Describe a tree.", pos="C'est un grand arbre",
             neg="It is a tall tree", instruction_id="french"),
    ]


# --------------------------------------------------------------------------- #
# resolve_defenses_dir: readers move to the commit-independent dir              #
# --------------------------------------------------------------------------- #


def test_registry_from_cfg_env_moves_relative_dir_off_the_checkout(
        tmp_path, monkeypatch):
    """Cluster jobs export TT_TWO_SI_DEFENSES_DIR (the ONE absolute,
    commit-independent objects dir); the committed RELATIVE config default
    then resolves there — never inside the per-commit checkout — while an
    ABSOLUTE config value (a submit-time --out) stays authoritative."""
    import yaml

    repo = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((repo / "configs" / "two_si.yaml").read_text())
    shared = tmp_path / "exp" / "two_si_defenses" / "cells"
    monkeypatch.setenv("TT_TWO_SI_DEFENSES_DIR", str(shared))
    a = by_id(registry_from_cfg(cfg, repo_root=repo))[
        "si_constitution__psr_kl_all"]
    assert a.object_path == shared / a.id / object_artifact_name(a.arm)
    # an absolute config value beats the env override
    cfg["defenses_dir"] = str(tmp_path / "explicit")
    b = by_id(registry_from_cfg(cfg, repo_root=repo))[
        "si_constitution__psr_kl_all"]
    assert b.object_path == (tmp_path / "explicit" / b.id
                             / object_artifact_name(b.arm))


# --------------------------------------------------------------------------- #
# Stacked cells — SI system prompt + compiled object together (opt-in, and      #
# byte-identical to the historical matrix when absent)                          #
# --------------------------------------------------------------------------- #

_STACK_BASE = "safety_refusal__lora_r1"
_STACK_ID = f"stacked__{_STACK_BASE}"


def test_stacked_id_roundtrips_to_si_and_arm():
    """A three-segment id: the stacked__ prefix strips FIRST, then <si>__<arm>
    splits on its own first __ — so a raw two-__ id still recovers (si, arm)."""
    assert si_of(_STACK_ID) == "safety_refusal"
    assert arm_of(_STACK_ID) == "lora_r1"
    # existing ids unaffected (the exact pre-stacked behaviour)
    assert si_of(_STACK_BASE) == "safety_refusal"
    assert arm_of(_STACK_BASE) == "lora_r1"
    assert arm_of("prompted__safety_refusal") is None
    assert si_of("prompted__safety_refusal") == "safety_refusal"
    assert arm_of("antidote") == "antidote" and si_of("antidote") is None
    assert arm_of("none") is None and si_of("none") is None


def test_stacked_reuses_the_underlying_class_a_object_path():
    """No new artifact: stacked__X and X resolve to the SAME object.pt, so the
    stack serves the exact bytes the Class A single served."""
    assert object_path_for(DDIR, _STACK_ID) == object_path_for(DDIR, _STACK_BASE)
    assert object_path_for(DDIR, _STACK_ID) == (
        DDIR / _STACK_BASE / object_artifact_name("lora_r1"))


def test_stacked_cells_are_opt_in_and_additive():
    """Absent -> the registry is byte-for-byte the historical matrix. Present ->
    exactly one cell appended at the END (every earlier array index unchanged)."""
    reg0 = build_registry(DDIR)
    reg1 = build_registry(DDIR, stacked_cells=(_STACK_BASE,))
    assert [c.id for c in reg1[:len(reg0)]] == [c.id for c in reg0]
    assert len(reg1) == len(reg0) + 1
    cell = reg1[-1]
    assert cell.id == _STACK_ID and cell.class_ == "stacked"
    assert cell.arm == "lora_r1" and cell.si_id == "safety_refusal"
    # references BOTH singles it stacks (the compounded-over-refusal check)
    assert cell.ni_reference_ids == ("prompted__safety_refusal", _STACK_BASE)
    counts = matrix_counts(reg1)
    assert counts["stacked"] == 1
    # publishes no new object (reuses the Class A artifact)
    assert counts["objects"] == matrix_counts(reg0)["objects"]


def test_stacked_cell_requires_both_arm_and_si():
    with pytest.raises(ValueError):
        DefenseCell(_STACK_ID, "stacked", "lora_r1", None)     # si_id missing
    with pytest.raises(ValueError):
        DefenseCell(_STACK_ID, "stacked", None, "safety_refusal")  # arm missing


def test_stacked_cell_completeness_and_over_refusal_comparisons():
    reg = build_registry(DDIR, stacked_cells=(_STACK_BASE,))
    comps = over_refusal_comparisons(reg, ("xstest",))
    stacked_refs = {c.reference_id for c in comps if c.cell_id == _STACK_ID}
    assert stacked_refs == {"prompted__safety_refusal", _STACK_BASE}
    # it is a first-class cell: dropping it makes the matrix incomplete
    with pytest.raises(ValueError):
        assert_matrix_complete(
            reg, present_ids=[c.id for c in reg if c.id != _STACK_ID])


def test_stacked_cell_from_cfg():
    cfg = {"si_ids": ["safety_refusal"], "defenses_dir": str(DDIR),
           "class_a_arms": ["lora_r1"], "class_b_arms": [],
           "stacked_cells": [_STACK_BASE]}
    reg = registry_from_cfg(cfg)
    assert [c.id for c in reg] == [
        "none", "prompted__safety_refusal", _STACK_BASE, _STACK_ID]


def test_build_defense_cell_stacked_serves_object_plus_si(tiny_llama, tmp_path):
    """The run path: a stacked cell builds mode='stacked' with the SI text AND
    the reused Class A object — object loaded from the base <si>__<arm> path."""
    from sisteer.data.instructions import BY_ID
    from sisteer.eval.jailbreak import build_defense_cell, resolve_instruction

    model, tok = tiny_llama
    reg = by_id(build_registry(tmp_path, si_ids=("safety_refusal",),
                               class_a_arms=("lora_r1",), class_b_arms=(),
                               stacked_cells=(_STACK_BASE,)))
    base = reg[_STACK_BASE]
    stacked = reg[_STACK_ID]
    assert stacked.object_path == base.object_path   # reuse, no new artifact

    import json as _json
    base.object_path.parent.mkdir(parents=True, exist_ok=True)
    obj = ARMS["lora_r1"].train(model, tok, BY_ID["french"], _pairs(),
                                layer=1, epochs=1, batch_size=2, max_len=64)
    obj.save(base.object_path)
    # the publish contract's fit-phase record: the stacked path serves the
    # underlying object at its fit leg, so the sidecar must exist (the same
    # fit==serve guard the compiled path enforces)
    (base.object_path.parent / "train_budget_meta.json").write_text(
        _json.dumps({"cell_id": base.id, "arm": "lora_r1",
                     "positions_at_fit": "all"}))

    d = build_defense_cell(model, tok, stacked)
    assert d.mode == "stacked" and d._obj is not None
    assert d.positions_at_fit == "all" and d.positions_served == "all"
    assert d.system == resolve_instruction("safety_refusal").text
    out = d.generate(["hello there"], max_new_tokens=3)
    assert isinstance(out, list) and len(out) == 1
