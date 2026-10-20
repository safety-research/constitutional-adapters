"""The steered smoke's direct-selection GATING seam
(scripts/petri_steered_smoke.py, ``selection_direct_failure``) —
CPU-only, no vllm.

The contract under test: a ``FAILED`` selection_direct verdict (the
worker's own record shows a request bound to the wrong object, or an
unselected request bound to one) is a routing failure and must enter
the gating failures — failure marker + nonzero exit — while
``MISSING``/``PARTIAL`` verdicts stay ADVISORY, because an instrument
that did not observe a row is not a row that resolved wrongly. Before
this seam existed the record's own docstring already called a direct
FAIL "a real routing bug, outranking everything else", but nothing fed
it into the failure list, so a mis-routed run could still exit 0 with
smoke_record.json on disk.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
_spec = importlib.util.spec_from_file_location(
    "_petri_steered_smoke", ROOT / "scripts" / "petri_steered_smoke.py")
smoke = importlib.util.module_from_spec(_spec)
sys.modules["_petri_steered_smoke"] = smoke
_spec.loader.exec_module(smoke)


# --- the pure gating rule ------------------------------------------------------


def test_failed_verdict_produces_a_gating_failure():
    msg = smoke.selection_direct_failure({
        "verdict": "FAILED", "n_rows": 12, "n_mismatched": 2,
        "aggregate_n_observations": 24, "aggregate_n_mismatched": 3})
    assert msg is not None
    assert "selection_direct FAILED" in msg
    assert "2" in msg and "3" in msg  # both halves' counts are named


def test_missing_and_partial_verdicts_stay_advisory():
    for verdict in (
            "passed",
            "PARTIAL",
            "MISSING: the worker recorded no observations for this batch "
            "at all",
            "MISSING: not one selected row could be correlated to the "
            "worker's record by request id, so the per-row half did not "
            "run. The aggregate half below still applies.",
            "match_none"):
        assert smoke.selection_direct_failure({"verdict": verdict}) is None
    assert smoke.selection_direct_failure(None) is None
    assert smoke.selection_direct_failure({}) is None


# --- end-to-end with the report builder ----------------------------------------


def _plan():
    # three prompts: one unselected, two selected
    return [(0, None), (1, "arm_a"), (2, "arm_b")]


def _rows():
    return [{"req_id": "r0"}, {"req_id": "r1"}, {"req_id": "r2"}]


def _record(requests, n_obs, n_bad, pairs=None):
    return {"requests": requests, "n_observations": n_obs,
            "n_mismatched": n_bad, "n_record_errors": 0,
            "n_unstable_requests": 0, "pairs": pairs or {}}


def test_a_misbound_row_gates_through_the_report():
    """A request bound to the WRONG object -> verdict FAILED -> a gating
    failure string."""
    record = _record({
        "r1": {"applied_cell": "arm_b"},   # asked arm_a, got arm_b
        "r2": {"applied_cell": "arm_b"},
    }, n_obs=2, n_bad=0)
    rep = smoke.selection_direct_report(_plan(), _rows(), record, "armed")
    assert rep["verdict"] == "FAILED"
    assert smoke.selection_direct_failure(rep) is not None


def test_an_unselected_row_bound_to_an_object_gates():
    record = _record({
        "r0": {"applied_cell": "arm_a"},   # asked NOTHING, got arm_a
        "r1": {"applied_cell": "arm_a"},
        "r2": {"applied_cell": "arm_b"},
    }, n_obs=3, n_bad=0)
    rep = smoke.selection_direct_report(_plan(), _rows(), record, "armed")
    assert rep["verdict"] == "FAILED"
    assert smoke.selection_direct_failure(rep) is not None


def test_lost_correlation_is_partial_or_missing_and_advisory():
    # one selected row unobserved -> PARTIAL; all unobserved -> MISSING —
    # neither produces a gating failure (lost telemetry, not mis-routing)
    partial = _record({"r1": {"applied_cell": "arm_a"}}, n_obs=2, n_bad=0,
                      pairs={"arm_a": ["arm_a"], "arm_b": ["arm_b"]})
    rep = smoke.selection_direct_report(_plan(), _rows(), partial, "armed")
    assert rep["verdict"] == "PARTIAL"
    assert smoke.selection_direct_failure(rep) is None
    rep_missing = smoke.selection_direct_report(_plan(), _rows(),
                                                _record({}, 2, 0), "armed")
    assert rep_missing["verdict"].startswith("MISSING")
    assert smoke.selection_direct_failure(rep_missing) is None
    rep_none = smoke.selection_direct_report(_plan(), _rows(), None,
                                             "reset failed")
    assert rep_none["verdict"].startswith("MISSING")
    assert smoke.selection_direct_failure(rep_none) is None


def test_an_aggregate_mismatch_gates_even_without_row_correlation():
    """The request-id-independent half: the worker itself counted a
    mis-binding -> FAILED -> gates, even when every row correlates."""
    record = _record({
        "r1": {"applied_cell": "arm_a"},
        "r2": {"applied_cell": "arm_b"},
    }, n_obs=4, n_bad=1)
    rep = smoke.selection_direct_report(_plan(), _rows(), record, "armed")
    assert rep["verdict"] == "FAILED"
    assert smoke.selection_direct_failure(rep) is not None


def test_a_clean_record_passes_and_produces_no_failure():
    record = _record({
        "r1": {"applied_cell": "arm_a"},
        "r2": {"applied_cell": "arm_b"},
    }, n_obs=2, n_bad=0)
    rep = smoke.selection_direct_report(_plan(), _rows(), record, "armed")
    assert rep["verdict"] == "passed"
    assert smoke.selection_direct_failure(rep) is None


# --- the install gate runs the PRODUCTION comparison ------------------------------


def test_production_install_check_is_the_production_comparison(tmp_path):
    """Both GPU smokes gate their install through
    ``production_install_check`` — vllm_server.expected_install_state
    vs validate_install_reports, the exact readiness comparison the
    wave's launch_server runs. Before this seam the LoRA smoke asserted
    a bespoke subset of the report and went green (job 232307) while
    the wave's readiness refused the identical install (job 232410):
    the smoke could not fail when production was broken."""
    import pytest

    from sisteer.capeval.serving import vllm_server as vs
    from test_lora_lens_serving import HEADER, _blob_bytes, _lora_all_blob

    reg = tmp_path / "reg"
    smoke.build_registry(reg, [{"cell_id": "si_safety__lora_r4",
                                "kind": "lora_all",
                                "bytes": _blob_bytes(_lora_all_blob())}],
                         HEADER)
    expected = vs.expected_install_state(reg)
    ok = {k: (list(v) if isinstance(v, list) else v)
          for k, v in expected.items()}
    out = smoke.production_install_check(reg, [ok])
    assert out["expected"] == expected
    assert out["reports"] == [ok]
    # the wave's failure mode — a worker serving NOTHING — refuses
    # through the smoke's gate with the production error, and so does an
    # empty RPC result
    unhooked = dict(ok, layers=[], oproj_layers=[])
    with pytest.raises(vs.VllmServerError, match="must not pass readiness"):
        smoke.production_install_check(reg, [unhooked])
    with pytest.raises(vs.VllmServerError, match="PER WORKER"):
        smoke.production_install_check(reg, [])


def test_render_ids_honors_thinking_pin():
    """The smoke's local render seam (Qwen3.5 onboarding, 2026-08-30):
    a thinking-ON render on a reasoning template derives the WRONG
    assistant header (the open <think> instead of the empty
    <think></think> block the fits counted, H=7 on Qwen3.5) and
    false-fails every response-mask arm at prefill. render_ids must
    honor the models.pin_reasoning_template_off pin, exactly like the
    real chat_markers.render_ids."""

    class FakeTok:
        chat_template = "... enable_thinking is defined ..."

        def apply_chat_template(self, messages, tokenize=True,
                                add_generation_prompt=True, **kw):
            ids = [1, 2, 3]
            if add_generation_prompt:
                # thinking-ON header (open think) vs OFF (empty block)
                ids += ([9, 8] if kw.get("enable_thinking", True)
                        else [9, 7, 7, 8])
            return ids

    tok = FakeTok()
    on = smoke.render_ids(tok, [{"role": "user", "content": "x"}], True)
    assert on == [1, 2, 3, 9, 8]  # unpinned: template default (thinking ON)

    tok._sisteer_template_kwargs = {"enable_thinking": False}
    off = smoke.render_ids(tok, [{"role": "user", "content": "x"}], True)
    assert off == [1, 2, 3, 9, 7, 7, 8]  # pinned: the fit's header shape


def test_canonical_model_identity_path_vs_hub_id():
    """Jobs 255287/255288 (2026-08-30): with a warm offline cache vLLM
    resolves a hub id to its snapshot PATH, and a string comparison can
    never pass. Canonical identity must equate the two — while a
    different repo, or the same repo at the WRONG snapshot, still
    refuses."""
    path = ("/shared/hf_cache/hub/models--Qwen--Qwen3.5-4B/"
            "snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    rid, rev = smoke.canonical_model_identity(path)
    assert rid == "Qwen/Qwen3.5-4B"
    assert rev == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    # a hub id canonicalizes to itself, claiming no revision
    assert smoke.canonical_model_identity("Qwen/Qwen3.5-4B") == (
        "Qwen/Qwen3.5-4B", None)
    # a different model stays different
    other, _ = smoke.canonical_model_identity(
        "/x/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/" + "0" * 40)
    assert other != rid
