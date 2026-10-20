"""The selection guard's NAMED discrete axes: `guard_selection(...,
discrete_axes={name: {levels, selected}})` files `discrete_extreme:<name>`
when the selected level is an END of an ordered menu with >= 2 levels — rank 4
and rank 64, the narrowest and the widest module set, a reduced 3-level menu's
ends, and BOTH ends of a two-level menu — while the warm-up factor keeps its
2-level exemption BY NAME, the set order is DERIVED from trainable-parameter
counts, and `guard["selected"]` carries the named axes. Every record written
without named axes is byte-identical.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from test_tune_arm import tune_arm  # noqa: E402

guard = tune_arm.hpo_selection_guard
CLIP = 1.3
BOX = {"lr": [10 ** -4.5, 10 ** -2.0], "max_grad_norm": [CLIP, CLIP]}
SETS = ["wo", "qkvo", "down", "mlp", "attn_mlp"]
COUNTS = {"wo": 245_760, "qkvo": 860_160, "down": 26_624_000, "mlp": 58_900_480, "attn_mlp": 59_760_640}


def _rows(n=16):
    rows = []
    for i in range(n):
        u = -4.4 + 2.3 * i / (n - 1)
        rows.append({"index": i, "round": 1, "lr": 10 ** u, "max_grad_norm": CLIP, "warmup_frac": 0.15,
                     "best_heldout_loss": 1.9 + 0.3 * (u + 3.2) ** 2 + 0.001 * i,
                     "lora_rank": (4, 16, 64)[i % 3], "lora_set_id": SETS[i % 5]})
    return rows


def _guard(selected_rank, selected_set, *, ranks=(4, 16, 64), sets=SETS, rows=None, warmup=0.15,
           warmup_levels=(0.15,)):
    rows = rows or _rows()
    sel = {"lr": 10 ** -3.2, "max_grad_norm": CLIP, "warmup_frac": warmup, "lora_rank": selected_rank,
           "lora_set_id": selected_set}
    fit = {"positive_definite": True, "hessian_eigs": [0.6], "rms_log_y": 0.01, "mad_log_y": 0.01}
    return guard.guard_selection(
        selected=sel, selected_from="fitted_argmin", rows=rows, box=BOX, levels=list(warmup_levels),
        fit=fit, y_key="best_heldout_loss", sigma_log_y=0.01, fitted_min_log_y=0.64,
        discrete_axes={"lora_rank": {"levels": list(ranks), "selected": selected_rank},
                       "lora_modules": {"levels": list(sets), "selected": selected_set,
                                        "order": "trainable_params"}})


def codes(g):
    return sorted(guard.alarm_code(a) for a in g["alarms"])


@pytest.mark.parametrize("rank,expect", [(4, True), (16, False), (64, True)])
def test_rank_ends_alarm_by_name(rank, expect):
    g = _guard(rank, "down")
    assert ("discrete_extreme:lora_rank" in codes(g)) is expect, codes(g)
    assert g["selected"]["lora_rank"] == rank and g["selected"]["lora_modules"] == "down"
    assert g["dims"]["lora_rank"]["level_index"] == [4, 16, 64].index(rank)
    assert g["discrete_axes"]["lora_rank"]["levels"] == [4, 16, 64]


@pytest.mark.parametrize("sid,expect", [("wo", True), ("qkvo", False), ("down", False), ("mlp", False),
                                        ("attn_mlp", True)])
def test_narrowest_and_widest_set_alarm_by_name(sid, expect):
    g = _guard(16, sid)
    assert ("discrete_extreme:lora_modules" in codes(g)) is expect, codes(g)
    assert "discrete_extreme:lora_rank" not in codes(g)
    assert "discrete_extreme:warmup_frac" not in codes(g)


def test_reduced_three_level_menu_keeps_its_own_ends():
    menu = ["qkvo", "down", "mlp"]            # the probe excluded wo and attn_mlp
    assert "discrete_extreme:lora_modules" in codes(_guard(16, "qkvo", sets=menu))
    assert "discrete_extreme:lora_modules" not in codes(_guard(16, "down", sets=menu))
    assert "discrete_extreme:lora_modules" in codes(_guard(16, "mlp", sets=menu))


def test_two_level_menu_alarms_at_either_end_unlike_the_warmup_factor():
    # ranks {4, 16} after rank 64 was excluded: BOTH ends alarm (review D P2)
    assert "discrete_extreme:lora_rank" in codes(_guard(4, "down", ranks=(4, 16)))
    assert "discrete_extreme:lora_rank" in codes(_guard(16, "down", ranks=(4, 16)))
    # two surviving sets: both ends alarm
    assert "discrete_extreme:lora_modules" in codes(_guard(16, "qkvo", sets=["qkvo", "mlp"]))
    assert "discrete_extreme:lora_modules" in codes(_guard(16, "mlp", sets=["qkvo", "mlp"]))
    # the warm-up DESIGN factor: a 2-level menu is recorded, never alarmed — exempt BY NAME
    rows = _rows()
    for i, r in enumerate(rows):
        r["warmup_frac"] = (0.05, 0.15)[i % 2]
    g = _guard(16, "down", rows=rows, warmup=0.15, warmup_levels=(0.05, 0.15))
    assert "discrete_extreme:warmup_frac" not in codes(g)
    assert g["dims"]["warmup_frac"]["note"] == "2-level factor: no interior exists by design; recorded, not alarmed"
    with pytest.raises(ValueError, match="warmup_frac"):
        guard.guard_selection(selected={"lr": 1e-3, "max_grad_norm": CLIP, "warmup_frac": 0.15}, selected_from="fitted_argmin",
                              rows=_rows(), box=BOX, levels=[0.15], fit={"positive_definite": True, "hessian_eigs": [1.0]},
                              y_key="best_heldout_loss", sigma_log_y=None, fitted_min_log_y=0.6,
                              discrete_axes={"warmup_frac": {"levels": [0.05, 0.15], "selected": 0.15}})


def test_single_level_menu_is_recorded_not_alarmed_and_a_foreign_level_is_a_mismatch():
    g = _guard(16, "down", ranks=(16,))
    assert "discrete_extreme:lora_rank" not in codes(g)
    assert g["dims"]["lora_rank"]["note"].startswith("single level")
    g2 = _guard(32, "down")
    assert "fixed_axis_mismatch:lora_rank" in codes(g2)


def test_tune_arm_mirrors_the_module_set_vocabulary():
    """tune_arm's torch-free mirror of the module-set / init-scheme vocabulary
    equals the library's (the design path must not import torch, so the
    mirror is pinned here instead of asserted at import)."""
    from sisteer.extraction import lora_modules as lm

    assert tune_arm._MODULE_SETS_FALLBACK == {k: tuple(v) for k, v in lm.MODULE_SETS.items()}
    assert tune_arm._INIT_SCHEMES_FALLBACK == tuple(lm.INIT_SCHEMES)
    # with the library loaded, the helpers read it and agree with the mirror
    assert tune_arm._module_sets() == tune_arm._MODULE_SETS_FALLBACK
    for sid, mods in lm.MODULE_SETS.items():
        assert tune_arm.set_id_of(mods) == sid == lm.set_id_for(mods)
        assert tune_arm.modules_of_set(sid) == sorted(mods)


def test_set_order_is_derived_from_parameter_counts_not_hand_written():
    from sisteer.extraction.lora_modules import set_order_by_parameter_count
    order = set_order_by_parameter_count(COUNTS)
    assert order == SETS
    assert tune_arm.set_order_from_counts(COUNTS) == order
    # a menu whose counts put a set elsewhere reorders the ends accordingly
    weird = {"mlp": 10, "wo": 20, "down": 30}
    assert set_order_by_parameter_count(weird) == ["mlp", "wo", "down"]
    assert "discrete_extreme:lora_modules" in codes(_guard(16, "mlp", sets=set_order_by_parameter_count(weird)))
    assert "discrete_extreme:lora_modules" not in codes(_guard(16, "wo", sets=set_order_by_parameter_count(weird)))
    # from an allowed_pairs document: trainable_params / rank orders the sets
    allowed = {"pairs": [{"rank": 4, "set": "mlp", "lora_modules": ["down_proj", "gate_proj", "up_proj"], "trainable_params": 58_900_480},
                         {"rank": 16, "set": "wo", "lora_modules": ["o_proj"], "trainable_params": 4 * 245_760},
                         {"rank": 4, "set": "down", "lora_modules": ["down_proj"], "trainable_params": 26_624_000}],
               "excluded": []}
    assert tune_arm.set_order_for_pairs(allowed) == ["wo", "down", "mlp"]
    # a pair without a count refuses by name: the order is never hand-written
    del allowed["pairs"][1]["trainable_params"]
    with pytest.raises(SystemExit, match="no trainable_params recorded for module set.*'wo'"):
        tune_arm.set_order_for_pairs(allowed)


def test_records_without_named_axes_are_byte_identical():
    rows = _rows()
    sel = {"lr": 10 ** -3.2, "max_grad_norm": CLIP, "warmup_frac": 0.15}
    fit = {"positive_definite": True, "hessian_eigs": [0.6]}
    kw = dict(selected=sel, selected_from="fitted_argmin", rows=rows, box=BOX, levels=[0.15], fit=fit,
              y_key="best_heldout_loss", sigma_log_y=0.01, fitted_min_log_y=0.64)
    a = guard.guard_selection(**kw)
    b = guard.guard_selection(**kw, discrete_axes=None)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert "discrete_axes" not in a and set(a["selected"]) == {"lr", "max_grad_norm", "warmup_frac"}
