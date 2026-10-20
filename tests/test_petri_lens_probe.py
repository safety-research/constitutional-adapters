"""CPU tests for the vllm-lens x vllm compatibility probe's pure logic
(scripts/petri_lens_probe.py) — the evidence ASSEMBLY that feeds the
existing Phase-0 checkers (lens_steer.run_phase0), never a
reimplementation of the checks themselves.

Covers: the fire-log -> per-generated-token attribution (prefill
produces token 1; decode pass i-1 produces token i; at most ONE trailing
scheduler-committed extra pass tolerated and bounded), the tick
interleaving co-batching proof, the returned-modification relative
residual metric, the full phase-0 evidence assembly round-tripped
through the REAL run_phase0 (a healthy gathering passes; the documented
silent-nothing failure mode fails NAMING the dead assertions), and the
probe-level checks. The GPU gathering itself runs on a cluster node via
scripts/sbatch/petri_probe.sbatch.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

from sisteer.capeval.serving import lens_steer as ls

ROOT = Path(__file__).parents[1]
_spec = importlib.util.spec_from_file_location(
    "_petri_lens_probe", ROOT / "scripts" / "petri_lens_probe.py")
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

N_PROMPT = 12
N_GEN = 32
LAYER = 16


def _fires(*, prefill_chunks=(N_PROMPT,), decode_passes=N_GEN - 1,
           layer=LAYER, start_tick=0):
    """A synthetic temporal fire log: prefill chunk(s) then decode passes."""
    fires, tick = [], start_tick
    for seq in prefill_chunks:
        fires.append({"layer": layer, "seq_len": seq, "tick": tick,
                      "num_reqs": 1})
        tick += 1
    for _ in range(decode_passes):
        fires.append({"layer": layer, "seq_len": 1, "tick": tick,
                      "num_reqs": 1})
        tick += 1
    return fires


# --- build_saved_sentinel: attribution + accounting ---------------------------


def test_clean_world_passes_the_real_checker():
    ev, acc = probe.build_saved_sentinel(
        _fires(), n_prompt=N_PROMPT, n_gen=N_GEN, layer=LAYER)
    assert ev == {"prefill_fires": 1, "decode_fires": N_GEN,
                  "generated_tokens": N_GEN,
                  "decode_step_fires": [1] * N_GEN}
    assert acc["ok"] and acc["post_final_extra_passes"] == 0
    ls._check_saved_sentinel(ev)  # the REAL checker accepts it


def test_one_trailing_extra_pass_is_tolerated_and_bounded():
    # vLLM v1 can commit one decode pass after the final token (vllm-lens
    # _trim_activations documents it); it produces no token and must not
    # corrupt per-token attribution.
    ev, acc = probe.build_saved_sentinel(
        _fires(decode_passes=N_GEN), n_prompt=N_PROMPT, n_gen=N_GEN,
        layer=LAYER)
    assert ev["decode_step_fires"] == [1] * N_GEN
    assert acc["ok"] and acc["post_final_extra_passes"] == 1
    ls._check_saved_sentinel(ev)


def test_two_extra_passes_fail_accounting():
    _, acc = probe.build_saved_sentinel(
        _fires(decode_passes=N_GEN + 1), n_prompt=N_PROMPT, n_gen=N_GEN,
        layer=LAYER)
    assert not acc["ok"]


def test_missing_decode_fires_fail_the_real_checker():
    ev, _ = probe.build_saved_sentinel(
        _fires(decode_passes=N_GEN - 5), n_prompt=N_PROMPT, n_gen=N_GEN,
        layer=LAYER)
    assert 0 in ev["decode_step_fires"]
    with pytest.raises(ls.Phase0Failure, match="EVERY generated token"):
        ls._check_saved_sentinel(ev)


def test_hook_never_fired_fails_the_real_checker():
    ev, _ = probe.build_saved_sentinel(
        [], n_prompt=N_PROMPT, n_gen=N_GEN, layer=LAYER)
    assert ev["prefill_fires"] == 0 and ev["decode_fires"] == 0
    with pytest.raises(ls.Phase0Failure, match="never fired at"):
        ls._check_saved_sentinel(ev)


def test_chunked_prefill_covers_the_prompt():
    ev, acc = probe.build_saved_sentinel(
        _fires(prefill_chunks=(7, 5)), n_prompt=N_PROMPT, n_gen=N_GEN,
        layer=LAYER)
    assert ev["prefill_fires"] == 2
    assert acc["ok"] and acc["prefill_token_coverage"] == N_PROMPT
    ls._check_saved_sentinel(ev)


def test_partial_prefill_coverage_fails_accounting():
    _, acc = probe.build_saved_sentinel(
        _fires(prefill_chunks=(7,)), n_prompt=N_PROMPT, n_gen=N_GEN,
        layer=LAYER)
    assert not acc["ok"]


def test_off_layer_fires_fail_accounting():
    fires = _fires() + [{"layer": LAYER + 1, "seq_len": 1, "tick": 99,
                         "num_reqs": 1}]
    _, acc = probe.build_saved_sentinel(
        fires, n_prompt=N_PROMPT, n_gen=N_GEN, layer=LAYER)
    assert acc["off_layer_fires"] == 1 and not acc["ok"]


# --- ticks_interleave: the co-batching proof -----------------------------------


def test_interleaved_ticks_prove_cobatching():
    assert probe.ticks_interleave([0, 2, 4], [1, 3, 5])
    assert probe.ticks_interleave([0, 10], [5])  # range overlap suffices


def test_serialized_or_vacuous_ticks_never_pass():
    assert not probe.ticks_interleave([0, 1, 2], [3, 4, 5])
    assert not probe.ticks_interleave([], [1, 2])
    assert not probe.ticks_interleave([1, 2], [])


# --- max_relative_residual: the returned-modification metric --------------------


def test_landed_add_measures_zero_and_missing_add_measures_one():
    base = torch.randn(N_PROMPT, 64)
    v = torch.randn(64) * 5.0
    assert probe.max_relative_residual(base, base + v, v) < 1e-6
    # the silent-nothing failure mode: hidden states unmodified.
    missing = probe.max_relative_residual(base, base, v)
    assert abs(missing - 1.0) < 1e-6
    assert missing > probe.RETMOD_ATOL


def test_misaligned_rows_refuse():
    with pytest.raises(ValueError, match="misaligned"):
        probe.max_relative_residual(torch.zeros(3, 8), torch.zeros(4, 8),
                                    torch.zeros(8))


# --- full assembly round-tripped through the REAL run_phase0 --------------------

PKGS = {"vllm": "0.25.1+cu129", "vllm-lens": "1.2.1",
        "torch": "2.11.0+cu129"}
BASE = list(range(100, 100 + N_GEN))
STEERED = list(range(200, 200 + N_GEN))
ALT = list(range(300, 300 + N_GEN))


def _run(tokens, req, n_prompt=N_PROMPT):
    return {"request_id": req, "tokens": list(tokens),
            "text_head": "", "n_prompt": n_prompt}


def _healthy_active():
    sent_ev, acc = probe.build_saved_sentinel(
        _fires(), n_prompt=N_PROMPT, n_gen=N_GEN, layer=LAYER)
    return {
        "packages": dict(PKGS),
        "plugin": {"generate_patched": True, "chat_patched": True,
                   "engine_args_patched": True},
        "engine": {"enforce_eager": True, "prefix_caching_enabled": False,
                   "worker_extension_cls":
                       "vllm_lens._worker_ext.HiddenStatesExtension"},
        "rpc_counts": {"install_hooks": 1, "set_persistent_hooks": 1},
        "requests_observed_with_hook": 6,
        "runs": {
            "baseline": _run(BASE, "0"),
            "baseline_repeat": _run(BASE, "1"),
            "sentinel_only": _run(BASE, "2"),
            "steered": _run(STEERED, "3"),
            "scale0": _run(BASE, "4"),
            "steered_alt_solo": _run(ALT, "5"),
            "mixed_unhooked": _run(BASE, "6"),
            "mixed_hooked": _run(ALT, "7"),
            "post_clear_baseline": _run(BASE, "8"),
        },
        "fires_by_run": {
            "steered": _fires(),
            "mixed_unhooked": _fires(start_tick=0),
            "mixed_hooked": _fires(start_tick=1),  # interleaved ticks
        },
        "saved_sentinel_evidence": sent_ev,
        "saved_sentinel_accounting": acc,
    }


def _healthy_disabled():
    return {"plugin": {"generate_patched": False,
                       "engine_args_patched": False},
            "saved_count": 0,
            "runs": {"base": _run(BASE, "0"),
                     "steer_attempt": _run(BASE, "1")}}


def test_healthy_gathering_passes_the_real_run_phase0():
    ev = probe.assemble_phase0_evidence(
        _healthy_active(), _healthy_disabled(), expected_packages=PKGS,
        retmod_delta=0.003, retmod_atol=probe.RETMOD_ATOL)
    report = ls.run_phase0(ev)
    assert report.passed == ls.PHASE0_ASSERTIONS
    checks = probe.evaluate_probe_checks(
        _healthy_active(), _healthy_disabled(),
        _healthy_active()["saved_sentinel_accounting"])
    assert all(c["ok"] for c in checks), [c for c in checks if not c["ok"]]


def test_silent_nothing_fails_naming_the_dead_assertions():
    """vllm-lens's documented failure mode: generation fluent, hook never
    fired, output unmodified. The probe must fail naming saved_sentinel
    and returned_modification — never pass on fluent generation alone."""
    active = _healthy_active()
    sent_ev, acc = probe.build_saved_sentinel(
        [], n_prompt=N_PROMPT, n_gen=N_GEN, layer=LAYER)
    active["saved_sentinel_evidence"] = sent_ev
    active["saved_sentinel_accounting"] = acc
    active["runs"]["steered"] = _run(BASE, "3")  # output NOT modified
    active["fires_by_run"] = {}
    ev = probe.assemble_phase0_evidence(
        active, _healthy_disabled(), expected_packages=PKGS,
        retmod_delta=1.0, retmod_atol=probe.RETMOD_ATOL)
    with pytest.raises(ls.Phase0Failure) as ei:
        ls.run_phase0(ev)
    msg = str(ei.value)
    assert "saved_sentinel" in msg and "returned_modification" in msg
    checks = {c["name"]: c for c in probe.evaluate_probe_checks(
        active, _healthy_disabled(), acc)}
    assert not checks["steering_changed_output"]["ok"]
    assert not checks["mixed_batch_cobatched"]["ok"]  # vacuous fires


def test_version_drift_fails_provenance():
    active = _healthy_active()
    active["packages"]["vllm"] = "0.19.0"
    ev = probe.assemble_phase0_evidence(
        active, _healthy_disabled(), expected_packages=PKGS,
        retmod_delta=0.003, retmod_atol=probe.RETMOD_ATOL)
    with pytest.raises(ls.Phase0Failure, match="server_provenance"):
        ls.run_phase0(ev)


def test_missing_gathering_is_a_failure_never_a_pass():
    # a probe that crashed before gathering (empty worker output) must
    # fail every assertion by NAME, not pass vacuously.
    ev = probe.assemble_phase0_evidence(
        {}, {}, expected_packages=PKGS, retmod_delta=None,
        retmod_atol=probe.RETMOD_ATOL)
    with pytest.raises(ls.Phase0Failure) as ei:
        ls.run_phase0(ev)
    assert f"{len(ls.PHASE0_ASSERTIONS)}/{len(ls.PHASE0_ASSERTIONS)}" \
        in str(ei.value)


def test_disabled_process_showing_patches_fails_the_probe_check():
    disabled = _healthy_disabled()
    disabled["plugin"]["generate_patched"] = True
    checks = {c["name"]: c for c in probe.evaluate_probe_checks(
        _healthy_active(), disabled,
        _healthy_active()["saved_sentinel_accounting"])}
    assert not checks["disable_flag_unpatched"]["ok"]


def test_contaminated_mixed_batch_fails_isolation():
    active = _healthy_active()
    # the unhooked request's output changed when batched with a hooked one
    active["runs"]["mixed_unhooked"] = _run(STEERED, "6")
    ev = probe.assemble_phase0_evidence(
        active, _healthy_disabled(), expected_packages=PKGS,
        retmod_delta=0.003, retmod_atol=probe.RETMOD_ATOL)
    with pytest.raises(ls.Phase0Failure, match="mixed_batch_isolation"):
        ls.run_phase0(ev)


def test_internal_id_matching_accepts_both_schemes_refuses_lookalikes():
    assert probe.match_internal_id("3", "3")
    assert probe.match_internal_id("3-a1b2", "3")
    assert not probe.match_internal_id("31-a1b2", "3")
    assert not probe.match_internal_id("13", "3")
