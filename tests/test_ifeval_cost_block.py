"""ifeval-canonical-battery: a leg's actual GPU-hours rest on the ACCEPTED attempt's ledger span.

``cost_block`` read ``summary.wall_clock_s`` — the FINALISATION's wall — while its note said "ledger wall + standup
adder": a reused-log finalisation of a 600 s generation read ~1 s there and understated the leg (0.070 vs 0.236 GPU-h).
Now the actual cost is the accepted generation attempt's ``generation_span.elapsed_s`` + the standup adder with its
``basis`` named; the finalisation wall is used only when the ledger spans nothing, LABELLED as the fallback; and the
BILLED cost over every recorded attempt (an abandoned attempt paid for its standup and its rows) is reported beside it.
"""
from __future__ import annotations

import pytest

import tinycap_ifeval_battery as A
import tinycap_ifeval_validators as V
import ifeval_synthetic_store as G
from test_tinycap_ifeval_analysis import ESC, small_report, small_store  # noqa: F401


def _leg(*, wall: float, span_s: float | None, by_attempt=None):
    return {"key": "k", "status": A.LEG_STATUS_USABLE, "model_key": "gemma4e4b", "cell": "none", "summary": {"wall_clock_s": wall},
            "ledger": {"generation_span": ({"first_start": "t0", "last_end": "t1", "elapsed_s": span_s} if span_s is not None else None),
                       "generation_span_by_attempt": by_attempt, "accepted_attempt": "a2",
                       "decode_tok_s_aggregate": 1000.0, "decode_tok_s_per_request": 40.0, "completion_tokens": None}}


def test_the_reviewers_reproduction_a_600s_generation_finalised_in_1s_costs_0_236_not_0_070():
    cells = {"c": {"legs": {"2048": _leg(wall=1.0, span_s=600.0,
                                         by_attempt={"a1": {"elapsed_s": 300.0, "n_rows": 37}, "a2": {"elapsed_s": 600.0, "n_rows": 1082}})}}}
    block = A.cost_block(cells)
    leg = block["legs"]["k"]
    assert leg["gpu_h_actual"] == pytest.approx((600.0 + A.STANDUP_ADDER_S) / 3600.0, abs=1e-4) == pytest.approx(0.2361, abs=1e-4)
    assert leg["gpu_h_actual"] != pytest.approx((1.0 + A.STANDUP_ADDER_S) / 3600.0, abs=1e-4)         # the old 0.0697 reading
    assert leg["basis"] == A.COST_BASIS_LEDGER_SPAN and leg["fallback"] is False and leg["generation_elapsed_s"] == 600.0
    assert leg["finalisation_wall_clock_s"] == 1.0 and leg["accepted_attempt"] == "a2"
    # billed over EVERY attempt: each attempt's span + one standup adder each — distinct from, and above, the accepted cost
    assert leg["gpu_h_billed_all_attempts"] == pytest.approx((300.0 + 250.0 + 600.0 + 250.0) / 3600.0, abs=1e-4) and leg["n_attempts_billed"] == 2
    assert leg["gpu_h_billed_all_attempts"] > leg["gpu_h_actual"]
    assert block["total_gpu_h_actual"] == pytest.approx(0.236, abs=1e-3) and block["total_gpu_h_billed_all_attempts"] == pytest.approx(0.389, abs=1e-3)
    assert block["n_legs_on_fallback_basis"] == 0 and block["basis"] == A.COST_BASIS_LEDGER_SPAN and "ACCEPTED" in block["note"]


def test_a_leg_whose_ledger_spans_nothing_falls_back_to_the_finalisation_wall_labelled_as_such():
    block = A.cost_block({"c": {"legs": {"2048": _leg(wall=1.0, span_s=None)}}})
    leg = block["legs"]["k"]
    assert leg["fallback"] is True and leg["basis"] == A.COST_BASIS_FALLBACK_WALL and "FALLBACK" in leg["basis"]
    assert leg["gpu_h_actual"] == pytest.approx((1.0 + A.STANDUP_ADDER_S) / 3600.0, abs=1e-4) and leg["generation_elapsed_s"] is None
    assert leg["gpu_h_billed_all_attempts"] is None and block["total_gpu_h_billed_all_attempts"] is None
    assert block["n_legs_on_fallback_basis"] == 1


def test_the_real_report_costs_every_leg_on_the_ledger_span_and_bills_the_abandoned_attempt(small_report, tmp_path):
    cost = small_report["cost"]
    assert cost["n_legs_on_fallback_basis"] == 0 and cost["total_gpu_h_billed_all_attempts"] >= cost["total_gpu_h_actual"] > 0
    for k, leg in cost["legs"].items():
        assert leg["basis"] == A.COST_BASIS_LEDGER_SPAN and leg["fallback"] is False, k
        assert leg["gpu_h_actual"] == pytest.approx((leg["generation_span"]["elapsed_s"] + A.STANDUP_ADDER_S) / 3600.0, abs=1e-4), k
        assert leg["accepted_attempt"] in leg["generation_span_by_attempt"], k
    # a store with an abandoned attempt: the ledger check files both attempts' spans; billed > actual by the abandoned
    # attempt's span + its standup adder
    row = G.row_for(G.matrix_rows(), "gemma4e4b", "main", "none")
    cell = G.build_store(tmp_path / "gen", row, abandoned_attempt=37)
    root = cell.parents[1]
    import json
    from sisteer.capeval.tinycap import completion, contracts as C
    manifest = json.loads((cell / "manifest.json").read_text())
    recs = V.iter_jsonl(root / C.REQUEST_EVIDENCE_FILENAME)
    by_id, _ = V.evidence_index(recs)
    outcomes = completion.read_task_outcomes(cell, V.TASK)
    summary = completion.read_task_summary(cell, V.TASK)
    led = V.ledger_check(root / V.IC.LEDGER_FILENAME, header=outcomes, summary=summary, cap=2048, max_model_len=2528,
                         manifest=manifest, evidence_by_id=by_id)
    assert led["passed"], led["failures"]
    spans = led["generation_span_by_attempt"]
    assert set(spans) == {"a1", "a2"} and spans["a1"]["n_rows"] == 37 and spans["a2"]["n_rows"] == 1082
    assert spans["a2"]["elapsed_s"] == led["generation_span"]["elapsed_s"]
    leg = A.leg_cost({"key": "x", "status": A.LEG_STATUS_USABLE, "model_key": "gemma4e4b", "cell": "none", "summary": summary, "ledger": led})
    assert leg["n_attempts_billed"] == 2 and leg["gpu_h_billed_all_attempts"] == pytest.approx(
        leg["gpu_h_actual"] + (spans["a1"]["elapsed_s"] + A.STANDUP_ADDER_S) / 3600.0, abs=1e-4)
