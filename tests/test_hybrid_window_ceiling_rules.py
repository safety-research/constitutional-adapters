"""The per-window ceiling's six live rules (src/sisteer/capeval/serving/geometry.py), pinned on the shipped nemotron3nano4b row
(`("nemotron3nano4b", 4608)`, the allocator's verdict on a refused standup) and synthetic variants of it: a verdict under TP > 1 is
refused by name; a ceiling must name the profile's budget; `_check_window_ceilings` refuses a util mismatch, a budget that moved
underneath the ceiling and a ceiling that does not LOWER the quotient; a site-overlay row (by the loader's stamp) is never shadowed; and
`seqs_at(..., budget=)` applies the ceiling only under the profile it was measured under. Nothing here names a model outside the menu."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
import sisteer.capeval.serving.geometry as sg  # noqa: E402

KEY = "nemotron3nano4b"
L = 4608


def _row() -> sg.ModelBudget:
    return sg.BUDGET_BY_MODEL[KEY]


def _shipped() -> sg.WindowSeqsCeiling:
    return sg.HYBRID_WINDOW_SEQS_CEILINGS[(KEY, L)]


def _quotient(budget: int) -> int:
    return min(sg.identity_seqs(L, budget=budget), sg.HYBRID_SEQS_CAP)


def test_the_shipped_ceiling_lowers_the_rows_quotient_and_seqs_at_applies_it():
    b, c = _row(), _shipped()
    assert c.max_num_seqs == 167 < _quotient(b.batched_token_budget)
    assert b.window_ceiling(L) is c and b.seqs_at(L) == 167
    assert c.batched_token_budget == b.batched_token_budget and c.gpu_memory_utilization == b.basis.gpu_memory_utilization


def test_a_verdict_under_tensor_parallel_above_one_is_refused_by_name():
    with pytest.raises(ValueError, match="serves every hybrid TP = 1"):
        dataclasses.replace(_shipped(), tensor_parallel=2)


def test_a_ceiling_must_name_the_profiles_budget():
    with pytest.raises(ValueError, match="name the batched_token_budget"):
        dataclasses.replace(_shipped(), batched_token_budget=0)


def test_check_window_ceilings_refuses_a_util_mismatch_a_moved_budget_and_a_ceiling_that_does_not_lower(monkeypatch):
    b, c0 = _row(), _shipped()                                                               # the shipped row, read once (the dict is patched below)
    monkeypatch.setitem(sg.HYBRID_WINDOW_SEQS_CEILINGS, (KEY, L), dataclasses.replace(c0, gpu_memory_utilization=b.basis.gpu_memory_utilization + 0.05))
    with pytest.raises(ValueError, match="not comparable"):
        sg._check_window_ceilings({KEY: b})
    monkeypatch.setitem(sg.HYBRID_WINDOW_SEQS_CEILINGS, (KEY, L), dataclasses.replace(c0, batched_token_budget=b.batched_token_budget + L))
    with pytest.raises(ValueError, match="the budget moved underneath it"):
        sg._check_window_ceilings({KEY: b})
    q = _quotient(b.batched_token_budget)                                                   # a verdict AT the quotient lowers nothing (its own fields consistent)
    monkeypatch.setitem(sg.HYBRID_WINDOW_SEQS_CEILINGS, (KEY, L), dataclasses.replace(c0, max_num_seqs=q, kv_max_concurrency=q + 0.55, kv_tokens_reported=int((q + 0.55) * L)))
    with pytest.raises(ValueError, match="does not LOWER"):
        sg._check_window_ceilings({KEY: b})
    monkeypatch.setitem(sg.HYBRID_WINDOW_SEQS_CEILINGS, (KEY, L), c0)
    sg._check_window_ceilings({KEY: b})                                                     # the shipped row passes the import-time check


def test_a_site_overlay_row_is_never_shadowed_by_a_table_ceiling():
    b = _row()
    site = dataclasses.replace(b, site_record_generation="g1")                               # the loader's stamp, never a string prefix
    assert site.window_ceiling(L) is None and site.seqs_at(L) == _quotient(site.batched_token_budget) > 167


def test_seqs_at_applies_the_ceiling_only_under_the_profile_it_was_measured_under():
    b = _row(); other = b.batched_token_budget + L
    assert b.window_ceiling(L, budget=other) is None and b.seqs_at(L, budget=other) == _quotient(other)
    assert b.window_ceiling(L, budget=b.batched_token_budget) is _shipped() and b.seqs_at(L, budget=b.batched_token_budget) == 167
