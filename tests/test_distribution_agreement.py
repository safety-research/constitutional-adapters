"""Two top-k readouts of one model served two ways: what the agreement module
certifies, what it refuses to certify, and why a small coarsened KL is not
equivalence evidence (the readout can only bound the full KL from below).

CPU-only, stdlib module; the same code re-analyses a recorded probe's raw
readouts on the grading host.
"""
from __future__ import annotations

import math

from sisteer.eval.attacks import distribution_agreement as da


def _dist(probs: dict) -> dict:
    return {k: math.log(v) for k, v in probs.items()}


# a peaked position: 64-token readout would capture ~all mass; here 4 tokens do
PEAKED = {"a": 0.90, "b": 0.06, "c": 0.03, "d": 0.009}          # residual 0.001


def test_identical_readouts_certify_and_have_zero_kl():
    rec = da.compare_position(_dist(PEAKED), _dist(PEAKED))
    assert rec["status"] == da.STATUS_CERTIFIED
    assert rec["membership_identical"] and rec["argmax_agree"]
    assert rec["max_abs_delta_logp_common"] == 0.0
    assert rec["coarsened_kl_lower_bound_nats"] == 0.0
    assert abs(rec["captured_mass_left"] - 0.999) < 1e-6


def test_small_coarsened_kl_with_a_membership_change_is_divergent_not_equivalent():
    # the reviewer's case: the shared mass barely moves (tiny KL) while a token
    # LEAVES the top-k on one side -- a lower bound cannot see that as equivalence
    right = {"a": 0.90, "b": 0.06, "c": 0.03, "e": 0.009}          # 'd' -> 'e'
    rec = da.compare_position(_dist(PEAKED), _dist(right))
    assert rec["coarsened_kl_lower_bound_nats"] < 1e-3
    assert not rec["membership_identical"]
    assert rec["n_left_only"] == 1 and rec["n_right_only"] == 1
    assert rec["max_mass_left_only"] == 0.009
    assert rec["status"] == da.STATUS_DIVERGENT
    assert da.verdict_of([rec]) == da.VERDICT_DIVERGENT


def test_common_token_delta_above_tolerance_is_divergent():
    right = {"a": 0.88, "b": 0.08, "c": 0.03, "d": 0.009}           # b moved by ln(8/6)=0.29
    rec = da.compare_position(_dist(PEAKED), _dist(right), delta_tol=0.01)
    assert rec["membership_identical"]
    assert rec["max_abs_delta_logp_common"] > 0.01
    assert rec["status"] == da.STATUS_DIVERGENT
    # the same readout under a looser tolerance certifies: the tolerance is part
    # of the claim and is recorded in the summary
    assert da.compare_position(_dist(PEAKED), _dist(right), delta_tol=0.5)["status"] == da.STATUS_CERTIFIED
    assert da.summarize([rec], delta_tol=0.01)["criteria"]["delta_tol_nats"] == 0.01


def test_unbounded_tail_is_inconclusive_even_with_identical_membership():
    flat = {"a": 0.30, "b": 0.20, "c": 0.10, "d": 0.05}             # residual 0.35
    rec = da.compare_position(_dist(flat), _dist(flat))
    assert rec["membership_identical"] and rec["max_abs_delta_logp_common"] == 0.0
    assert rec["status"] == da.STATUS_UNBOUNDED
    assert da.verdict_of([rec]) == da.VERDICT_INCONCLUSIVE


def test_no_common_token_is_skipped_and_inconclusive():
    rec = da.compare_position(_dist({"a": 0.5, "b": 0.5}), _dist({"c": 0.5, "d": 0.5}))
    assert rec["status"] == da.STATUS_SKIPPED
    assert rec["coarsened_kl_lower_bound_nats"] is None
    assert not rec["argmax_left_in_right_topk"]
    assert da.verdict_of([rec]) == da.VERDICT_INCONCLUSIVE


def test_verdict_is_equivalent_only_when_every_position_certifies():
    left_rows = [[_dist(PEAKED)] * 3, [_dist(PEAKED)] * 2]
    right_rows = [[_dist(PEAKED)] * 3, [_dist(PEAKED)] * 2]
    out = da.compare(left_rows, right_rows)
    assert out["summary"]["verdict"] == da.VERDICT_EQUIVALENT
    assert out["summary"]["n_positions"] == 5
    assert [(r["row"], r["t"]) for r in out["positions"]] == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    # one unbounded position among certified ones -> inconclusive, never equivalent
    flat = {"a": 0.30, "b": 0.20, "c": 0.10, "d": 0.05}
    right_rows[1][1] = _dist(flat)
    left_rows[1][1] = _dist(flat)
    s = da.summarize(da.compare_rows(left_rows, right_rows))
    assert s["verdict"] == da.VERDICT_INCONCLUSIVE
    assert s["by_status"][da.STATUS_CERTIFIED] == 4 and s["by_status"][da.STATUS_UNBOUNDED] == 1
    # and one divergent position dominates the verdict
    right_rows[0][0] = _dist({"a": 0.90, "b": 0.06, "c": 0.03, "e": 0.009})
    s = da.summarize(da.compare_rows(left_rows, right_rows))
    assert s["verdict"] == da.VERDICT_DIVERGENT
    assert s["divergent_positions"] == [{
        "row": 0, "t": 0, "membership_identical": False, "max_abs_delta_logp_common": 0.0,
        "max_mass_left_only": 0.009, "max_mass_right_only": 0.009, "argmax_agree": True}]


def test_summary_labels_the_kl_as_a_lower_bound_and_reports_the_decidables():
    s = da.summarize(da.compare_rows([[_dist(PEAKED)]], [[_dist(PEAKED)]]))
    assert "LOWER bound" in s["coarsened_kl_lower_bound_nats"]["note"]
    for key in ("membership_identical_rate", "argmax_agree_rate", "max_abs_delta_logp_common",
                "max_abs_delta_logp_above_mass_floor",
                "residual_mass_worse_side", "max_mass_of_a_token_ranked_by_one_side_only",
                "membership_identical_above_mass_floor_rate"):
        assert key in s, key
    assert s["membership_identical_rate"] == 1.0 and s["argmax_agree_rate"] == 1.0


def test_keys_may_be_token_ids_or_strings():
    ids = {1: math.log(0.9), 2: math.log(0.099)}
    assert da.compare_position(ids, ids)["status"] == da.STATUS_CERTIFIED
    strs = {"x": math.log(0.9), "y": math.log(0.099)}
    assert da.compare_position(strs, strs)["status"] == da.STATUS_CERTIFIED


def test_mass_floored_delta_ignores_quantized_tail_tokens():
    # a repeat of the same serving: one zero-mass tail token's logprob moves by a
    # bf16 step (0.5 nats); the strict criterion flags it, the floored companion
    # reports what the sampled-from tokens did (nothing)
    left = dict(_dist(PEAKED), z=-40.0)
    right = dict(_dist(PEAKED), z=-40.5)
    rec = da.compare_position(left, right)
    assert rec["max_abs_delta_logp_common"] == 0.5 and rec["status"] == da.STATUS_DIVERGENT
    assert rec["max_abs_delta_logp_above_mass_floor"] == 0.0


def test_outer_row_count_mismatch_refuses_instead_of_zipping():
    import pytest
    left = [[_dist(PEAKED)], [_dist(PEAKED)]]
    right = [[_dist(PEAKED)]]
    with pytest.raises(da.AlignmentError, match="2 vs 1 rows") as e:
        da.compare_rows(left, right)
    assert e.value.row is None and (e.value.left_len, e.value.right_len) == (2, 1)
    with pytest.raises(da.AlignmentError):
        da.compare(left, right)


def test_inner_position_count_mismatch_refuses_instead_of_zipping():
    import pytest
    # the reviewer's reproduction: a two-position left row vs a one-position right
    # row used to certify the surviving position and read "equivalent"
    left = [[_dist(PEAKED), _dist(PEAKED)]]
    right = [[_dist(PEAKED)]]
    with pytest.raises(da.AlignmentError, match="row 0 has 2 vs 1 positions") as e:
        da.compare_rows(left, right)
    assert (e.value.row, e.value.left_len, e.value.right_len) == (0, 2, 1)
    # the recorded form of a refused comparison certifies nothing
    s = da.inconclusive_summary("row 0 has 2 vs 1 positions")
    assert s["verdict"] == da.VERDICT_INCONCLUSIVE and s["by_status"][da.STATUS_CERTIFIED] == 0
    assert "alignment_error" in s


def test_probe_records_a_misaligned_contrast_as_inconclusive():
    import importlib.util
    import pathlib
    import sys
    tests_dir = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(tests_dir))
    from test_grokmt_served_resume import _stub_modal
    sys.modules.setdefault("modal", _stub_modal())
    spec = importlib.util.spec_from_file_location(
        "_attacker_latency_scope_under_test",
        tests_dir.parent / "scripts" / "modal" / "attacker_latency_scope.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod._coarsened_kl([[_dist(PEAKED), _dist(PEAKED)]], [[_dist(PEAKED)]])
    assert out["summary"]["verdict"] == da.VERDICT_INCONCLUSIVE
    assert "2 vs 1 positions" in out["summary"]["alignment_error"]
    assert out["positions"] == []
    # aligned readouts still go through unchanged
    ok = mod._coarsened_kl([[_dist(PEAKED)]], [[_dist(PEAKED)]])
    assert ok["summary"]["verdict"] == da.VERDICT_EQUIVALENT


def test_matched_empty_row_is_inconclusive_and_accounted_never_certified():
    # the reviewer's reproduction: a prompt with an empty continuation is an empty
    # row on BOTH legs; the shapes align, one position certifies, and the verdict
    # used to read "equivalent" while that prompt was never compared
    left = [[], [_dist(PEAKED)]]
    right = [[], [_dist(PEAKED)]]
    out = da.compare(left, right)
    s = out["summary"]
    assert s["verdict"] == da.VERDICT_INCONCLUSIVE
    assert s["by_status"][da.STATUS_EMPTY_ROW] == 1 and s["by_status"][da.STATUS_CERTIFIED] == 1
    assert s["rows"] == {"n_rows": 2, "n_rows_compared": 1, "n_rows_empty": 1, "empty_rows": [0]}
    assert s["n_positions"] == 1                       # rates are over compared positions only
    assert s["membership_identical_rate"] == 1.0 and s["argmax_agree_rate"] == 1.0
    empty_rec = [r for r in out["positions"] if r["status"] == da.STATUS_EMPTY_ROW]
    assert empty_rec and empty_rec[0]["row"] == 0 and empty_rec[0]["t"] is None
    # wholly empty inputs certify nothing either
    assert da.summarize(da.compare_rows([[]], [[]]))["verdict"] == da.VERDICT_INCONCLUSIVE
    assert da.summarize(da.compare_rows([], []))["verdict"] == da.VERDICT_INCONCLUSIVE
    # and an empty row beside a divergent one still reads divergent (the stronger finding)
    right2 = [[], [_dist({"a": 0.90, "b": 0.06, "c": 0.03, "e": 0.009})]]
    assert da.summarize(da.compare_rows(left, right2))["verdict"] == da.VERDICT_DIVERGENT


def test_probe_summaries_show_the_uncompared_prompt():
    import importlib.util
    import pathlib
    import sys
    tests_dir = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(tests_dir))
    from test_grokmt_served_resume import _stub_modal
    sys.modules.setdefault("modal", _stub_modal())
    spec = importlib.util.spec_from_file_location(
        "_attacker_latency_scope_under_test_rows",
        tests_dir.parent / "scripts" / "modal" / "attacker_latency_scope.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tf = mod._teacher_forced_kl([[], [_dist(PEAKED)]], [[], [_dist(PEAKED)]])["summary"]
    assert tf["n_prompts"] == 2 and tf["n_prompts_compared"] == 1 and tf["n_prompts_empty"] == 1
    assert tf["verdict"] == da.VERDICT_INCONCLUSIVE and tf["rows"]["empty_rows"] == [0]
    ck = mod._coarsened_kl([[], [_dist(PEAKED)]], [[], [_dist(PEAKED)]])["summary"]
    assert ck["verdict"] == da.VERDICT_INCONCLUSIVE and ck["rows"]["n_rows_empty"] == 1
