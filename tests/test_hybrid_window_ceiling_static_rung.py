"""The sampled lane's static 8192 rung serves families_served:short at the per-model rung window
(8,192 + 2,048 headroom = 10,240), and a hybrid model's budget quotient can compose an S its allocator refuses at readiness — a hybrid
KV cache's token capacity is not S x L. When no site-record rung exists at that window (the probe writes the production windows only),
the allocator's verdict is recorded as a per-window ceiling (`HYBRID_WINDOW_SEQS_CEILINGS`) applied inside `seqs_at`, so the standup's
resolved geometry, identity_S, the composed record and submit_sampled's print all size the SAME S with the identity holding (no
adjustment step), while the budget row and every other window's S are untouched (the nemotron-4608 pattern). Pinned on a
SYNTHETIC ceiling for a hybrid menu key, installed under the table row's own profile (the precedent of test_model_budget)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
import sisteer.capeval.serving.geometry as sg  # noqa: E402

KEY = "qwen35_4b"
L_RUNG = 10240


@pytest.fixture(autouse=True)
def _no_site_overlay(monkeypatch):
    for k in ("SISTEER_GEOMETRY_DIR", "SISTEER_SITE_DIR", "SISTEER_SHARED_ROOT"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def ceiling(monkeypatch) -> sg.WindowSeqsCeiling:
    """A synthetic allocator verdict at the rung window for the hybrid key, under its OWN budget row's profile (budget + util): the
    quotient the row buys at 10,240 (under the hybrid cap) is shadowed by 6 = floor(6.08x) over 62,227 KV tokens."""
    b = sg.BUDGET_BY_MODEL[KEY]
    quotient = min(b.batched_token_budget // L_RUNG, sg.HYBRID_SEQS_CAP)
    assert quotient > 6 and b.window_ceiling(L_RUNG) is None                               # the shipped table has no rung ceiling here
    c = sg.WindowSeqsCeiling(model_key=KEY, max_model_len=L_RUNG, max_num_seqs=6, kv_max_concurrency=6.08, kv_tokens_reported=62_227,
                             gpu_memory_utilization=b.basis.gpu_memory_utilization, probe_job_id="11111", measured_on="2026-10-02",
                             tensor_parallel=1, serving_path="lora_native", batched_token_budget=b.batched_token_budget,
                             engine_version="test", note="synthetic: the allocator reported 62,227 tokens = 6.08x at 10,240")
    monkeypatch.setitem(sg.HYBRID_WINDOW_SEQS_CEILINGS, (KEY, L_RUNG), c)
    sg._check_window_ceilings({**sg.BUDGET_BY_MODEL, **sg.ONBOARDED_BUDGET_BY_MODEL})      # a ceiling must LOWER the quotient it shadows
    return c


def test_the_rung_window_sizes_from_the_allocators_verdict_and_nothing_else_moves(ceiling):
    b = sg.BUDGET_BY_MODEL[KEY]
    quotient = min(b.batched_token_budget // L_RUNG, sg.HYBRID_SEQS_CAP)
    assert b.window_ceiling(L_RUNG) is ceiling
    assert sg.identity_seqs(L_RUNG, budget=b.batched_token_budget) == b.batched_token_budget // L_RUNG   # the quotient that would be refused
    assert b.seqs_at(L_RUNG) == ceiling.max_num_seqs == 6 == int(ceiling.kv_max_concurrency)
    assert 6 * L_RUNG == 61_440 <= ceiling.kv_tokens_reported == 62_227 < quotient * L_RUNG
    assert ceiling.batched_token_budget == b.batched_token_budget and ceiling.gpu_memory_utilization == b.basis.gpu_memory_utilization
    # nothing else moves: the static base window, the Petri window, the shipped nemotron ceiling
    assert b.window_ceiling(4864) is None and b.seqs_at(4864) == min(b.batched_token_budget // 4864, sg.HYBRID_SEQS_CAP)
    assert b.window_ceiling(16384) is None and b.seqs_at(16384) == min(b.batched_token_budget // 16384, sg.HYBRID_SEQS_CAP)
    assert sg.HYBRID_WINDOW_SEQS_CEILINGS[("nemotron3nano4b", 4608)].max_num_seqs == 167
    assert f"{KEY} @ {L_RUNG}: S 6" in sg.describe_budgets()                                  # rendered in the ceilings section


def test_the_standups_resolved_geometry_and_the_planners_print_both_size_six_with_the_identity_holding(ceiling):
    g = sg.resolve_geometry("families_served:short", max_model_len=L_RUNG, model_key=KEY, strict_identity=True)
    assert g.max_model_len == L_RUNG and g.max_num_seqs == 6 and g.seqs_source == "model_budget", g
    ident = g.identity_S() if callable(getattr(g, "identity_S", None)) else g.identity_S
    assert ident == 6                                                                       # identity holds: no adjustment step is needed
    rb = sg.budget_for(KEY)
    assert rb.per_model and rb.entry.seqs_at(L_RUNG) == 6                                   # submit_sampled._Budget.seqs_at is exactly this call
    src = (REPO / "scripts" / "sdfbase" / "full22" / "sampled" / "submit_sampled.py").read_text()
    assert re.search(r'"static_esc":\s*10240', src) and "seqs_at(int(L))" in src            # the planner prints 10240 x seqs_at(10240)
    # the rung window itself is the per-model table's 10,240 for an 8192 rung (common_sampled.sh f22s_rung_window)
    cs = (REPO / "scripts" / "sdfbase" / "full22" / "sampled" / "common_sampled.sh").read_text()
    assert "*:8192) echo 10240;;" in cs
