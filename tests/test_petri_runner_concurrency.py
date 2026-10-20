"""Petri eval concurrency + target response cap are COUPLED to the
serving geometry (petri_runner.target_serving_concurrency /
target_response_cap).

The defect these pin down: ``eval_set`` ran with NO ``max_connections``,
which at the pinned inspect-ai 0.3.251 resolves to the ADAPTIVE
concurrency default (min=10, start=20, ramping to max=100 on success).
vLLM never pushes back above ``max_num_seqs`` — it queues and preempts —
so the controller would pile ~3x the 32-sequence serving geometry onto
the proxy: an API-bound run silently converted into a proxy-queued one.
And the target role carried no ``max_tokens`` anywhere, so vLLM granted
every response the remaining window — one degenerate reply could swallow
the whole 16k window mid-audit.

Every assertion here compares against the vllm_server CONSTANTS, never a
literal, so a serving-geometry change (e.g. the 8 -> 32 max_num_seqs
raise) moves these expectations automatically — the coupling IS the
test. CPU-only; no vLLM, no inspect_ai."""

from inspect import signature
from types import SimpleNamespace

import pytest

from sisteer.capeval.serving import vllm_server as vsrv
from sisteer.capeval.tiers import petri_runner as pr


def _engine(identity=None):
    return SimpleNamespace(proxy_identity=identity)


def _spec(mode):
    return SimpleNamespace(serving_mode=mode)


def test_interleaved_width_equals_the_concurrent_geometry():
    width = pr.target_serving_concurrency(_spec("interleaved"),
                                          _engine(None))
    assert width == vsrv.DEFAULT_MAX_NUM_SEQS  # the coupling, not a literal


def test_interleaved_engine_without_identity_still_sizes():
    # interleaved has no strategy to read; an engine with no identity
    # block (e.g. petri_target-style handles) sizes to the concurrent
    # geometry rather than refusing
    width = pr.target_serving_concurrency(
        _spec("interleaved"), SimpleNamespace())
    assert width == vsrv.DEFAULT_MAX_NUM_SEQS


def test_serialize_strategy_serializes_the_eval_side_too():
    ident = {"exact_replay_strategy": "serialize"}
    width = pr.target_serving_concurrency(_spec("exact_replay"),
                                          _engine(ident))
    assert width == vsrv.DEFAULT_MAX_NUM_SEQS_SERIALIZE
    # the proxy holds a global lock under serialize: >1 connection just
    # queues at the lock, so the eval side must not fan out either
    assert width == 1


def test_batch_invariant_strategy_runs_the_full_geometry():
    ident = {"exact_replay_strategy": "batch_invariant"}
    width = pr.target_serving_concurrency(_spec("exact_replay"),
                                          _engine(ident))
    assert width == vsrv.DEFAULT_MAX_NUM_SEQS


def test_exact_replay_without_an_identity_refuses_loudly():
    # serialize serves ONE sequence, batch_invariant the full geometry:
    # guessing either way wrecks the run, so no identity -> refusal,
    # never a default (the repo's authoritative-sources rule)
    with pytest.raises(pr.PetriRunnerError, match="exact_replay_strategy"):
        pr.target_serving_concurrency(_spec("exact_replay"), _engine(None))
    with pytest.raises(pr.PetriRunnerError, match="exact_replay_strategy"):
        pr.target_serving_concurrency(_spec("exact_replay"),
                                      SimpleNamespace())


def test_unrecognized_strategy_refuses_rather_than_guessing():
    ident = {"exact_replay_strategy": "mystery_mode"}
    with pytest.raises(pr.PetriRunnerError, match="mystery_mode"):
        pr.target_serving_concurrency(_spec("exact_replay"), _engine(ident))


def test_target_response_cap_derives_from_the_launch_window():
    window = signature(vsrv.build_server_launch).parameters[
        "max_model_len"].default
    cap = pr.target_response_cap()
    assert isinstance(cap, int)
    assert cap == window // 16  # derived from the composer's own default
    # sanity at the current geometry: bounds a degenerate turn to 1/16
    # of the window while leaving ~4x the projected median reply
    assert cap >= 256


# --- the context-overflow re-run geometry ---------------


def test_rerun_geometry_doubles_window_and_halves_seqs():
    window = signature(vsrv.build_server_launch).parameters[
        "max_model_len"].default
    geo = pr.rerun_serving_geometry()
    assert geo == {"max_model_len": window * 2,
                   "max_num_seqs": vsrv.DEFAULT_MAX_NUM_SEQS // 2}


def test_rerun_geometry_preserves_the_certified_token_product():
    # max_num_batched_tokens = max_num_seqs * max_model_len is a
    # CORRECTNESS identity (no-chunking, vllm_server module docstring)
    # and S*L tokens * 128 KiB/token is the KV ceiling the standard
    # launch already certified — the re-run pass must not exceed it.
    window = signature(vsrv.build_server_launch).parameters[
        "max_model_len"].default
    geo = pr.rerun_serving_geometry()
    assert (geo["max_num_seqs"] * geo["max_model_len"]
            <= vsrv.DEFAULT_MAX_NUM_SEQS * window)


def test_rerun_geometry_follows_the_constant_at_call_time(monkeypatch):
    # the serving agent may re-certify DEFAULT_MAX_NUM_SEQS (e.g. 32 ->
    # 24); the re-run pass must follow automatically, never a snapshot
    monkeypatch.setattr(vsrv, "DEFAULT_MAX_NUM_SEQS", 24)
    assert pr.rerun_serving_geometry()["max_num_seqs"] == 12


def test_rerun_geometry_refuses_a_non_widening_scale():
    with pytest.raises(pr.PetriRunnerError, match="scale"):
        pr.rerun_serving_geometry(scale=1)
    with pytest.raises(pr.PetriRunnerError, match="scale"):
        pr.rerun_serving_geometry(scale=True)


def test_response_cap_does_not_scale_with_the_rerun_window(monkeypatch):
    # the cap is measurement IDENTITY (what the target may emit); the
    # window is capacity. Re-run rows must measure the same bounded
    # target as first-pass rows, so the cap reads the composer's DEFAULT
    # window even while the re-run pass is armed.
    before = pr.target_response_cap()
    monkeypatch.setenv(pr.PETRI_OVERFLOW_RERUN_ENV, "1")
    assert pr.target_response_cap() == before


def test_eval_concurrency_caps_to_rerun_geometry_when_armed(monkeypatch):
    spec, engine = _spec("interleaved"), _engine(None)
    assert pr.eval_concurrency(spec, engine) == vsrv.DEFAULT_MAX_NUM_SEQS
    monkeypatch.setenv(pr.PETRI_OVERFLOW_RERUN_ENV, "1")
    assert (pr.eval_concurrency(spec, engine)
            == pr.rerun_serving_geometry()["max_num_seqs"])
    # "0" / empty mean NOT armed (unset-like spellings, not a third state)
    monkeypatch.setenv(pr.PETRI_OVERFLOW_RERUN_ENV, "0")
    assert pr.eval_concurrency(spec, engine) == vsrv.DEFAULT_MAX_NUM_SEQS
    monkeypatch.setenv(pr.PETRI_OVERFLOW_RERUN_ENV, "")
    assert pr.eval_concurrency(spec, engine) == vsrv.DEFAULT_MAX_NUM_SEQS


def test_eval_concurrency_never_raises_a_serialize_width(monkeypatch):
    # under a serialize strategy the proxy holds a global lock; the
    # re-run cap may only ever NARROW the width
    ident = {"exact_replay_strategy": "serialize"}
    monkeypatch.setenv(pr.PETRI_OVERFLOW_RERUN_ENV, "1")
    assert pr.eval_concurrency(_spec("exact_replay"), _engine(ident)) == 1
