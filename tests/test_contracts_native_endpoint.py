"""The tinycap CONTRACTS on a NATIVE endpoint, the proxy-identity engine
helper and the engine-aware endpoint readers.

Pinned here:

- ``proxy_engine.served_engine_problem`` over the REAL ``/v1/models``
  ``petri_proxy`` shapes: a lens proxy attesting ``stacked`` / ``per_row``, a
  pre-landing lens proxy (no key), a native proxy (the ``engine`` block
  ``serving_attestation.engine_identity`` builds) — for all three expectations;
  a torn identity (native AND a lens mode) refuses; an unknown expectation is a
  ``ValueError``.
- ``contracts.twin_cell_id_for`` / ``twin_vmodel_for`` for both steered
  identities (a string, a parsed identity, a ``RouteEntry``, a route block) and
  the refusal for passthrough; ``TWIN_CELL_ID_LORA`` pinned equal to the
  registry's zero cell and the exporter's adapter name.
- ``contracts.endpoint_is_native`` on capctrl's, mt's and the native core's
  endpoint shapes (their real key sets, synthetic values), on the ``/v1/models``
  identity block, on a torn record, on non-mappings.
- ``expected_stamp_policy`` / ``check_serve_attestation`` on a native fixture
  endpoint (``not-consulted``; steps 1/3/4 ``not_applicable``, steps 2 and 5
  run; a declared configuration refuses naming ``--serving-path lens``) and on
  the lens fixtures BYTE-IDENTICAL to the pre-change semantics (every value
  pinned as a literal derived by hand from the lens rules).
- ``contracts.served_reduction_mode`` agrees with
  ``lens_worker.endpoint_reduction_mode`` case by case over the whole record
  vocabulary INCLUDING the native case (the engine read first); the literals
  the two spell are pinned equal.
- ``served_identity_fields(protocol_sampling, native)``: the lens key set is
  byte-identical; ``serving_engine`` present iff native.
- the two new RECs, ``__all__`` membership, and that ``not_applicable`` is NOT a
  member of ``REDUCTION_MODES``.

CPU-only."""

from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pytest

from sisteer.capeval import manifest as cm
from sisteer.capeval.petri import defense_cells as dc
from sisteer.capeval.serving import proxy as px
from sisteer.capeval.serving import proxy_engine as pe
from sisteer.capeval.serving import serving_attestation as sa
from sisteer.capeval.tinycap import contracts as C
from _env import host_site_path  # noqa: E402  (tests/ helper: a host-only data path through the site config)

LORA_CELL = "constv2-mix-full__lora_r4_ce"                 # the launcher's TrainTest spelling
LORA_HARNESS = "si_constv2_mix_full__lora_r4_ce"
SHA = "ab" * 32
SHA2 = "cd" * 32
TEMPLATE_HASH = "a" * 16

# ---------------------------------------------------------------------------
# fixtures: the three writers' endpoint shapes (real key sets, synthetic values)
# ---------------------------------------------------------------------------


def native_engine_block(*, genuine=LORA_HARNESS) -> dict:
    """The ``engine`` block the proxy echoes, through the REAL
    ``serving_attestation.engine_identity`` over a minimal attestation shape."""
    att = {"kind": sa.SERVING_ATTESTATION_KIND,
           "engine": {"enforce_eager": False, "cudagraph_mode": "FULL_AND_PIECEWISE"},
           "lora_config": {"max_loras": 2, "max_lora_rank": 8, "target_modules": ["o_proj"]},
           "adapters": {
               dc.NOOP_LORA_CELL_ID: {"config_sha256": "e" * 64, "weights_sha256": "f" * 64,
                                      "fingerprint_bf16": "c" * 64, "lora_int_id": 1,
                                      "petri_intervention_hash": None, "petri_zero_adapter": True},
               genuine: {"config_sha256": "1" * 64, "weights_sha256": "2" * 64,
                         "fingerprint_bf16": "3" * 64, "lora_int_id": 2,
                         "petri_intervention_hash": "c3170d1425ca6576", "petri_zero_adapter": False}}}
    return sa.engine_identity(att)


def ident_lens(mode) -> dict:
    """A lens proxy's ``petri_proxy`` block (``RoutingProxy.models_payload``):
    the mode echoed ONLY when the launcher attested one (``None`` = pre-landing:
    no key at all)."""
    out = {"version": 1, "serving_mode": "interleaved", "exact_replay_strategy": None,
           "base_model": "google/gemma-4-E4B-it", "petri_agent_loop_entry": True,
           "agent_loop_render": {"mode": "auto", "loaded": True, "source": "x"},
           "resumed_prefill_policy_cells": {"si_safety__psr_kl_all": "conditional"},
           "thinking_mode": "off"}
    if mode is not None:
        out[px.LENS_REDUCTION_MODE_KEY] = mode
    return out


def ident_native() -> dict:
    out = ident_lens(None)
    out.pop("resumed_prefill_policy_cells")
    out[px.ENGINE_KEY] = native_engine_block()
    return out


def capctrl_endpoint(*, mode="stacked", mode_error=None) -> dict:
    """capctrl_serving_up.Standup.endpoint()'s key set (a LENS standup)."""
    return {"proxy_url": "http://127.0.0.1:1/v1", "routing_table": "<TMP>/routing_table.json",
            "cell": LORA_CELL, "harness_cell": LORA_HARNESS, "cells": ["none", LORA_CELL],
            "harness_cells": ["none", LORA_HARNESS], "model": "google/gemma-4-E4B-it",
            "pid": 1, "node": "TESTHOST", "eval": "capability:ifeval_4096",
            "max_model_len": 4576, "max_num_seqs": 128,
            "geometry": {"max_model_len": 4576, "max_num_seqs": 128, "identity": True,
                         "seqs_source": "model_budget", "window_source": "table"},
            "kv_shrink": None, "enable_log_requests": False, "tool_calls": False,
            "reasoning_parser": None, "agent_loop_entry": True,
            "object_artifact_sha256": SHA, "object_artifact_sha256_by_cell": {LORA_HARNESS: SHA},
            "proxy_upstream_timeout_s": None, "lens_salt_policy": "attested",
            "thinking_header": None, "thinking_header_error": None,
            "lens_served_positions": {LORA_HARNESS: ["all"]},
            "chat_template_hash": TEMPLATE_HASH,
            "lens_reduction_mode": mode, "lens_reduction_mode_error": mode_error,
            "resumed_prefill_policy_cells": {LORA_HARNESS: "always"},
            "lens_serve_declarations": {}, "lens_serve_declarations_error": None}


def mt_endpoint(*, mode="stacked") -> dict:
    """mt_serving_up's lens endpoint: the wave's shape (a superset of capctrl's
    common keys, several cells, no single ``cell``)."""
    ep = capctrl_endpoint(mode=mode)
    ep.pop("cell"); ep.pop("harness_cell")
    ep.update({"cells": ["none", "prompted__constitution", LORA_CELL], "mode": "static",
               "served_models": ["petri__base", "petri__prompted__constitution",
                                 f"petri__{LORA_HARNESS}", "petri__none__lens0"],
               "slurm_job_id": "1", "require_attested_mode": mode, "started": "2026-09-13T00:00:00Z"})
    return ep


def core_endpoint(*, with_row_spelling=True) -> dict:
    """lora_serving_up.LoraStandup.endpoint()'s key set: ``engine`` (and, per
    §1.2, the row-facing ``serving_engine``); NO lens key of any kind."""
    ep = {"proxy_url": "http://127.0.0.1:1/v1", "proxy_port": 1, "vllm_port": 2,
          "routing_table": "<TMP>/routing_table.json", "mode": "static", "eval": "petri",
          "model": "google/gemma-4-E4B-it", "model_key": "gemma4e4b", "model_revision": "4" * 40,
          "revision": "4" * 40, "cells": ["none", LORA_CELL], "harness_cells": ["none", LORA_HARNESS],
          "cells_requested": ["none", LORA_CELL],
          "served_models": ["petri__base", f"petri__{LORA_HARNESS}", "petri__none__lora0"],
          "pid": 1, "node": "TESTHOST", "slurm_job_id": None,
          "max_model_len": 4864, "max_num_seqs": 107, "seqs_source": "override",
          "attested_max_num_seqs": 107, "attested_max_model_len": 4864,
          "geometry": {"max_model_len": 4864, "max_num_seqs": 107, "identity": False,
                       "seqs_source": "override", "window_source": "table"},
          "geometry_override_reason": sa.LORA_KERNEL_CEILING_REASON,
          "geometry_lowered": {"reason": sa.LORA_KERNEL_CEILING_REASON, "requested_S": 121, "S": 107},
          sa.LORA_KERNEL_CEILING_KEY: {"D": 4096, "ceiling": 524_287, "L": 4864, "S": 107},
          "tool_calls": False, "reasoning_parser": None, "agent_loop_entry": True,
          "petri_agent_loop_entry": True, "proxy_token": None, "proxy_upstream_timeout_s": None,
          "chat_template_hash": TEMPLATE_HASH, "chat_template_sha256": "a" * 64,
          "object_artifact_sha256": SHA, "object_artifact_sha256_by_cell": {LORA_HARNESS: SHA},
          "max_logprobs": None, "written_at": "2026-09-13T00:00:00Z", "started": "2026-09-13T00:00:00Z",
          "engine": "lora_native", "attempt_id": "a" * 16,
          "serving_attestation": "<TMP>/serving_attestation.json",
          "cells_served": "<TMP>/cells_served.json", "positive_controls": "<TMP>/positive_controls.json",
          "adapter_bank": "<TMP>/bank", "adapters": {}, "zero_adapter": {}, "adapters_reused": {},
          "attested_enforce_eager": False, "attested_cudagraph_mode": "FULL_AND_PIECEWISE",
          "attested_model_runner": "v2", "lora_config": {}, "id_name_map": {"1": "none__lora0"},
          "vmodels": {"none": "petri__base", LORA_HARNESS: f"petri__{LORA_HARNESS}"},
          "lora_cells": [LORA_HARNESS], "prompted_cells_admitted": [],
          "manifest_flags_dropped": ["--enforce-eager"], "routing_table_version": 2,
          "probe_max_tokens": 48}
    if with_row_spelling:
        ep[C.SERVED_ENGINE_FIELD] = "lora_native"
    return ep


# ---------------------------------------------------------------------------
# 1. served_engine_problem over the real identity shapes
# ---------------------------------------------------------------------------


class TestServedEngineProblem:
    def test_vocabulary(self):
        assert pe.ENGINE_EXPECTATIONS == ("lens:stacked", "lens:per_row", "lora_native")
        assert pe.ENGINE_EXPECT_LORA_NATIVE == px.ENGINE_KIND_LORA_NATIVE == sa.SERVING_ATTESTATION_KIND
        for name in ("served_engine_problem", "ENGINE_EXPECTATIONS"):
            assert name in pe.__all__
        with pytest.raises(ValueError, match="expect must be one of"):
            pe.served_engine_problem(ident_lens("stacked"), expect="stacked")

    @pytest.mark.parametrize("mode", ["stacked", "per_row"])
    def test_a_lens_proxy_matches_its_attested_mode_only(self, mode):
        ident = ident_lens(mode)
        assert pe.served_engine_problem(ident, expect=f"lens:{mode}") is None
        other = "per_row" if mode == "stacked" else "stacked"
        why = pe.served_engine_problem(ident, expect=f"lens:{other}")
        assert why and f"lens_reduction_mode={mode!r}" in why and other in why
        why = pe.served_engine_problem(ident, expect="lora_native")
        assert why and "attests no 'engine' block" in why

    def test_a_pre_landing_lens_proxy_is_neither_stacked_nor_per_row_by_default(self):
        """The four compares this replaces were ``ident.get(key) == mode`` — an
        absent key never read as per_row there, and does not here."""
        ident = ident_lens(None)
        assert px.LENS_REDUCTION_MODE_KEY not in ident
        for expect in ("lens:stacked", "lens:per_row"):
            why = pe.served_engine_problem(ident, expect=expect)
            assert why and "the key is absent" in why
        assert "no 'engine' block" in pe.served_engine_problem(ident, expect="lora_native")

    def test_a_native_proxy_matches_lora_native_only(self):
        ident = ident_native()
        assert ident[px.ENGINE_KEY]["kind"] == "lora_native"
        assert set(ident[px.ENGINE_KEY]) == set(px.ENGINE_ATTESTATION_KEYS)
        assert pe.served_engine_problem(ident, expect="lora_native") is None
        for expect in ("lens:stacked", "lens:per_row"):
            why = pe.served_engine_problem(ident, expect=expect)
            assert why and "native multi-LoRA engine" in why and "not the lens" in why

    def test_a_torn_identity_refuses_and_odd_shapes_are_named(self):
        torn = {**ident_native(), px.LENS_REDUCTION_MODE_KEY: "stacked"}
        assert "torn identity" in pe.served_engine_problem(torn, expect="lora_native")
        assert "native multi-LoRA engine" in pe.served_engine_problem(torn, expect="lens:stacked")
        wrong_kind = {**ident_lens(None), px.ENGINE_KEY: {**native_engine_block(), "kind": "lens_plugin"}}
        assert "engine.kind is 'lens_plugin'" in pe.served_engine_problem(wrong_kind, expect="lora_native")
        for bad in (None, [], "petri_proxy"):
            assert "no petri_proxy identity block" in pe.served_engine_problem(bad, expect="lora_native")

    def test_the_real_pulled_identity_when_present(self):
        """On the grading host the pulled certification attestations are the REAL
        engine blocks; the check holds on them. Skipped where they are absent
        (Modal's CPU suite)."""
        pulled = sorted(host_site_path("$STATE/romp/multilora-cert").glob("*/unit/lora_native/serving_attestation.json"))
        if not pulled:
            pytest.skip("no pulled certification attestations on this machine")
        for path in pulled:
            att = json.loads(path.read_text())
            ident = {**ident_lens(None), px.ENGINE_KEY: sa.engine_identity(att)}
            ident.pop("resumed_prefill_policy_cells")
            assert pe.served_engine_problem(ident, expect="lora_native") is None, path
            assert pe.served_engine_problem(ident, expect="lens:stacked"), path


# ---------------------------------------------------------------------------
# 2. the twin per engine
# ---------------------------------------------------------------------------


class TestTwinPerEngine:
    def test_constants(self):
        assert C.TWIN_CELL_ID == "none__lens0" == dc.NOOP_LENS_CELL_ID
        assert C.TWIN_CELL_ID_LORA == "none__lora0" == dc.NOOP_LORA_CELL_ID
        from sisteer.capeval.serving import lora_export as lx
        assert C.TWIN_CELL_ID_LORA == lx.ZERO_ADAPTER_NAME
        assert C.twin_vmodel() == "petri__none__lens0"          # unchanged
        for name in ("TWIN_CELL_ID_LORA", "twin_cell_id_for", "twin_vmodel_for"):
            assert name in C.__all__

    @pytest.mark.parametrize("mode", ["interleaved", "exact_replay"])
    def test_resolution_for_both_steered_identities_in_every_spelling(self, mode):
        lens_id, lora_id = f"proxy_vllm+lens+{mode}", f"proxy_vllm+lora+{mode}"
        route_lens = px.RouteEntry(virtual_model="petri__x", cell_id="x", cell_class="class_a",
                                   serving_identity=lens_id, intervention_hash="0" * 16)
        route_lora = px.RouteEntry(virtual_model="petri__y", cell_id="y", cell_class="class_a",
                                   serving_identity=lora_id, intervention_hash="0" * 16)
        for spelled in (lens_id, cm.parse_serving_identity(lens_id), route_lens, {"serving_identity": lens_id}):
            assert C.twin_cell_id_for(spelled) == "none__lens0"
            assert C.twin_vmodel_for(spelled) == "petri__none__lens0" == C.twin_vmodel()
        for spelled in (lora_id, cm.parse_serving_identity(lora_id), route_lora, {"serving_identity": lora_id}):
            assert C.twin_cell_id_for(spelled) == "none__lora0"
            assert C.twin_vmodel_for(spelled) == "petri__none__lora0" == dc._vmodel("none__lora0", "control")

    def test_a_passthrough_route_has_no_twin(self):
        for ident in ("proxy_vllm+passthrough+interleaved", "hf_shim+apply+interleaved"):
            with pytest.raises(ValueError, match="no scale-0 twin"):
                C.twin_cell_id_for(ident)
            with pytest.raises(ValueError, match="no scale-0 twin"):
                C.twin_vmodel_for(ident)
        with pytest.raises(cm.ManifestError):
            C.twin_cell_id_for("not+an+identity+at+all")

    def test_no_switch_probe_inventory_counts_the_engines_twin(self):
        """The no-switch class's EXACT probe inventory counts the
        bites probes on the ENGINE's twin — ``petri__none__lens0`` on a lens
        route, ``petri__none__lora0`` on a native one — through the one twin
        rule; the default (no identity) is the historical lens reading,
        byte-identical; a passthrough identity on a fitted cell has no twin and
        refuses; an unfitted cell is the render probe alone under any identity."""
        NA, RENDER, BITES = C.THINKING_NOT_AVAILABLE, C.PROBE_KIND_RENDER, C.PROBE_KIND_BITES
        vm = "petri__si_constv2_mix_full__lora_r4_ce"
        lens_id, lora_id = "proxy_vllm+lens+interleaved", "proxy_vllm+lora+interleaved"
        sig = inspect.signature(C.no_switch_probe_inventory)
        p = sig.parameters["serving_identity"]
        assert p.default is None and p.kind is inspect.Parameter.KEYWORD_ONLY
        historical = C.no_switch_probe_inventory(vm, fitted=True, n_probes=5)
        assert historical == {(vm, NA, RENDER): 1, (vm, NA, BITES): 5, ("petri__none__lens0", NA, BITES): 5}
        route_lora = px.RouteEntry(virtual_model=vm, cell_id="x", cell_class="class_a",
                                   serving_identity=lora_id, intervention_hash="0" * 16)
        for spelled in (lens_id, cm.parse_serving_identity(lens_id), {"serving_identity": lens_id}):
            assert C.no_switch_probe_inventory(vm, fitted=True, n_probes=5, serving_identity=spelled) == historical
        for spelled in (lora_id, cm.parse_serving_identity(lora_id), route_lora, {"serving_identity": lora_id}):
            native = C.no_switch_probe_inventory(vm, fitted=True, n_probes=5, serving_identity=spelled)
            assert native == {(vm, NA, RENDER): 1, (vm, NA, BITES): 5, ("petri__none__lora0", NA, BITES): 5}
            assert ("petri__none__lens0", NA, BITES) not in native
        # the twin's vmodel is the registry's rule, never spelled by hand
        assert {k[0] for k in C.no_switch_probe_inventory(vm, fitted=True, n_probes=1, serving_identity=lora_id)} == {
            vm, C.twin_vmodel_for(lora_id)}
        # a fitted cell on a passthrough identity has no twin to count
        with pytest.raises(ValueError, match="no scale-0 twin"):
            C.no_switch_probe_inventory(vm, fitted=True, n_probes=5, serving_identity="proxy_vllm+passthrough+interleaved")
        # ... but an unfitted cell never resolves a twin, whatever its identity
        for ident in (None, lens_id, lora_id, "proxy_vllm+passthrough+interleaved"):
            assert C.no_switch_probe_inventory("petri__base", fitted=False, n_probes=None, serving_identity=ident) == {
                ("petri__base", NA, RENDER): 1}
        # the n_probes refusal precedes the twin resolution (unchanged wording)
        with pytest.raises(ValueError, match="must be a positive count"):
            C.no_switch_probe_inventory(vm, fitted=True, n_probes=0, serving_identity=lora_id)

    def test_the_twin_agrees_with_the_proxy_engines_path_matched_twin(self):
        """``proxy_engine.scale0_twin`` resolves the same zero cell for a route
        whose own ``<cell>__lora0`` is not exported — one rule, two readers."""
        routes = [px.RouteEntry("petri__base", "none", "base", "proxy_vllm+passthrough+interleaved", "0" * 16),
                  px.RouteEntry("petri__y", "y", "class_a", "proxy_vllm+lora+interleaved", "1" * 16),
                  px.RouteEntry("petri__none__lora0", "none__lora0", "control", "proxy_vllm+lora+interleaved",
                                "0" * 16, control_of="none"),
                  px.RouteEntry("petri__x", "x", "class_a", "proxy_vllm+lens+interleaved", "2" * 16),
                  px.RouteEntry("petri__none__lens0", "none__lens0", "control", "proxy_vllm+lens+interleaved",
                                "0" * 16, control_of="none")]
        by = {r.cell_id: r for r in routes}
        assert pe.scale0_twin(by["y"], routes).cell_id == C.twin_cell_id_for(by["y"]) == "none__lora0"
        assert pe.scale0_twin(by["x"], routes).cell_id == C.twin_cell_id_for(by["x"]) == "none__lens0"
        assert pe.scale0_twin(by["none"], routes) is None


# ---------------------------------------------------------------------------
# 3. endpoint_is_native on the three writers' shapes
# ---------------------------------------------------------------------------


class TestEndpointIsNative:
    def test_the_three_writers(self):
        assert C.endpoint_is_native(capctrl_endpoint()) is False
        assert C.endpoint_is_native(mt_endpoint()) is False
        assert C.endpoint_is_native(core_endpoint()) is True
        assert C.endpoint_is_native(core_endpoint(with_row_spelling=False)) is True
        # the row-facing spelling alone (a record that copied only serving_engine)
        assert C.endpoint_is_native({C.SERVED_ENGINE_FIELD: "lora_native"}) is True
        assert C.endpoint_is_native({C.SERVED_ENGINE_FIELD: "lens"}) is False
        # a lens endpoint never carries an engine key of either spelling
        for ep in (capctrl_endpoint(), mt_endpoint()):
            assert C.ENDPOINT_KEY_ENGINE not in ep and C.SERVED_ENGINE_FIELD not in ep

    def test_the_identity_block_and_odd_shapes(self):
        assert C.endpoint_is_native(ident_native()) is True       # the /v1/models block (a mapping engine)
        assert C.endpoint_is_native(ident_lens("stacked")) is False
        for odd in (None, [], "endpoint.json", 3, {}, {"engine": None}, {"engine": "vllm"},
                    {"engine": {"kind": "lens_plugin"}}):
            assert C.endpoint_is_native(odd) is False
        with pytest.raises(ValueError, match="torn record"):
            C.endpoint_is_native({"engine": "lora_native", C.SERVED_ENGINE_FIELD: "lens"})

    def test_vocabulary(self):
        assert C.SERVED_ENGINE_FIELD == "serving_engine"
        assert C.SERVING_ENGINES == ("lens", "lora_native")
        assert C.SERVING_ENGINE_LORA_NATIVE == px.ENGINE_KIND_LORA_NATIVE == sa.SERVING_ATTESTATION_KIND
        assert C.ENDPOINT_KEY_ENGINE == px.ENGINE_KEY == "engine"
        assert C.NOT_APPLICABLE == C.REDUCTION_MODE_NOT_APPLICABLE == "not_applicable"
        assert C.REDUCTION_MODE_NOT_APPLICABLE not in C.REDUCTION_MODES     # a lens vocabulary
        assert C.REDUCTION_MODES == ("per_row", "stacked")
        assert C.REC_SERVING_ENGINE_MISMATCH == "SERVING_ENGINE_MISMATCH"
        assert C.REC_SERVING_ENGINE_CONFLICT == "SERVING_ENGINE_CONFLICT"
        assert C.REC_SERVING_ENGINE_MISMATCH not in C.OOS_REC_NAMES      # that set is pinned
        for name in ("NOT_APPLICABLE", "REDUCTION_MODE_NOT_APPLICABLE", "SERVED_ENGINE_FIELD",
                     "SERVING_ENGINES", "endpoint_is_native", "served_reduction_mode",
                     "REC_SERVING_ENGINE_MISMATCH", "REC_SERVING_ENGINE_CONFLICT",
                     "SERVED_IDENTITY_NATIVE_FIELDS", "reduction_mode_mismatch"):
            assert name in C.__all__, name


# ---------------------------------------------------------------------------
# 4. served_identity_fields
# ---------------------------------------------------------------------------


def test_served_identity_fields_matrix():
    base = C.SERVED_IDENTITY_FIELDS
    assert base == ("reasoning_parser", "tool_calls", "max_model_len", "max_num_seqs", "concurrency",
                    "agent_loop_entry", "reduction_mode")
    assert C.served_identity_fields() is base                             # byte-identical, by identity
    assert C.served_identity_fields(protocol_sampling=True) == (*base, "sampling_clamp_override")
    assert C.served_identity_fields(native=True) == (*base, "serving_engine")
    assert C.served_identity_fields(native=True, protocol_sampling=True) == (
        *base, "serving_engine", "sampling_clamp_override")
    assert C.SERVED_IDENTITY_NATIVE_FIELDS == ("serving_engine",)
    sig = inspect.signature(C.served_identity_fields)
    assert sig.parameters["native"].default is False and sig.parameters["native"].kind is inspect.Parameter.KEYWORD_ONLY
    # "serving_engine" never enters a lens key set
    assert "serving_engine" not in C.served_identity_fields(protocol_sampling=True)


# ---------------------------------------------------------------------------
# 5. expected_stamp_policy / check_serve_attestation: native vs lens
# ---------------------------------------------------------------------------

BUDGET = {"chat_template_hash": TEMPLATE_HASH}


def _budget_ok(monkeypatch):
    """The prompt-budget compare, scripted to its two verdicts so step 5 is
    observable without the real artifact shape."""
    def fake(budget, served_hash, *, served_source):
        return None if budget.get("chat_template_hash") == served_hash else f"{served_source} differs"
    monkeypatch.setattr(C, "prompt_budget_template_mismatch", fake)


class TestPoliciesOnANativeEndpoint:
    def test_expected_stamp_policy_is_not_consulted_for_every_cell(self):
        ep = core_endpoint()
        for cell in ("none", C.CELL_PROMPTED_CONSTITUTION, LORA_CELL, C.CELL_PSRKL_SAFETY):
            for arm in ("off", "on"):
                assert C.expected_stamp_policy(ep, cell=cell, arm=arm, out_of_support=False,
                                               model_key="gemma4e4b") == C.POLICY_NOT_CONSULTED
        # ... the same the base route already returns on a lens endpoint
        assert C.expected_stamp_policy(capctrl_endpoint(), cell="none", arm="off", out_of_support=False,
                                       model_key="gemma4e4b") == C.POLICY_NOT_CONSULTED

    def test_expected_stamp_policy_on_the_lens_fixture_is_unchanged(self):
        """A lens fitted cell reads its ATTESTED per-cell label (``always`` here)
        — the pre-change rule, spelled out."""
        for ep in (capctrl_endpoint(), mt_endpoint()):
            assert C.expected_stamp_policy(ep, cell=LORA_CELL, arm="off", out_of_support=False,
                                           model_key="gemma4e4b") == "always"
            # the probe twin's own attested label: the fixture's per-cell map does not
            # name ``none__lens0``, and ``attested_stamp_policy`` spells that ``absent``
            # (a literal there, not a module constant) — pinned so the lens reading of
            # a cell the map lacks is byte-identical too
            assert C.expected_stamp_policy(ep, cell=LORA_CELL, arm="off", out_of_support=False,
                                           model_key="gemma4e4b", harness_key=C.TWIN_CELL_ID) == "absent"
            # and the base route on a lens endpoint is what a native endpoint returns for EVERY cell
            assert C.expected_stamp_policy(ep, cell="none", arm="off", out_of_support=False,
                                           model_key="gemma4e4b") == C.POLICY_NOT_CONSULTED

    def test_check_serve_attestation_native_in_support(self):
        ep = core_endpoint()
        out = C.check_serve_attestation(ep, cell=LORA_CELL, arm="off", out_of_support=False,
                                        model_key="gemma4e4b", expected_object_sha256=SHA, budget=None)
        decl = out["declarations"]
        assert decl["status"] == C.NOT_APPLICABLE and decl["own_key"] == LORA_HARNESS
        assert {k: decl[k] for k in ("expected", "own", "twin_key", "twin", "raw")} == {
            "expected": None, "own": None, "twin_key": None, "twin": None, "raw": None}
        assert decl["error_present"] is False and "no serve-declaration surface" in decl["reason"]
        # step 2 ran: the digest compared and recorded
        assert out["object_digest"] == {"scalar": SHA, "by_cell": {LORA_HARNESS: SHA}}
        assert out[C.ENDPOINT_KEY_LENS_SALT_POLICY] == C.NOT_APPLICABLE
        assert out[C.ROUTE_KEY_RESUMED_PREFILL_POLICY] == C.NOT_APPLICABLE
        assert out[C.SERVED_ENGINE_FIELD] == "lora_native"
        assert C.LENS_SERVED_POSITIONS_KEY not in out           # a lens fact; not invented
        # the base route on the same endpoint: no digest compare, same not_applicable facts
        base = C.check_serve_attestation(ep, cell="none", arm="off", out_of_support=False,
                                         model_key="gemma4e4b", expected_object_sha256=None, budget=None)
        assert base["object_digest"] is None and base["declarations"]["own_key"] == "none"
        assert base[C.ENDPOINT_KEY_LENS_SALT_POLICY] == C.NOT_APPLICABLE

    def test_step_2_refuses_a_digest_mismatch_on_a_native_endpoint(self):
        ep = core_endpoint()
        with pytest.raises(C.OutOfSupportAttestationRefusal) as exc:
            C.check_serve_attestation(ep, cell=LORA_CELL, arm="off", out_of_support=False,
                                      model_key="gemma4e4b", expected_object_sha256=SHA2, budget=None)
        assert exc.value.rec == C.REC_OOS_DECLARATION_MISMATCH
        # an out-of-support fitted run told no digest refuses as on the lens
        with pytest.raises(C.OutOfSupportAttestationRefusal) as exc:
            C.check_serve_attestation(ep, cell=LORA_CELL, arm="off", out_of_support=True,
                                      model_key="gemma4e4b", expected_object_sha256=None, budget=BUDGET,
                                      header_located=False)
        assert exc.value.rec == C.REC_OOS_DECLARATION_UNATTESTED

    def test_step_5_runs_on_a_native_endpoint(self, monkeypatch):
        _budget_ok(monkeypatch)
        ep = core_endpoint()
        # an all-position fitted cell thinking-ON out of support is UNDECLARED
        # (header_located False) and admissible; the budget compare runs
        out = C.check_serve_attestation(ep, cell=LORA_CELL, arm="on", out_of_support=True,
                                        model_key="gemma4e4b", expected_object_sha256=SHA, budget=BUDGET,
                                        header_located=False)
        assert out["object_digest"]["scalar"] == SHA
        with pytest.raises(C.OutOfSupportAttestationRefusal) as exc:
            C.check_serve_attestation(ep, cell=LORA_CELL, arm="on", out_of_support=True,
                                      model_key="gemma4e4b", expected_object_sha256=SHA, budget=None,
                                      header_located=False)
        assert exc.value.rec == C.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH
        with pytest.raises(C.OutOfSupportAttestationRefusal) as exc:
            C.check_serve_attestation({**ep, "chat_template_hash": "b" * 16}, cell=LORA_CELL, arm="on",
                                      out_of_support=True, model_key="gemma4e4b", expected_object_sha256=SHA,
                                      budget=BUDGET, header_located=False)
        assert exc.value.rec == C.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH
        # the off-cap override path too
        out = C.check_serve_attestation(ep, cell="none", arm="off", out_of_support=False, model_key="gemma4e4b",
                                        expected_object_sha256=None, budget=BUDGET, off_cap_override=True)
        assert out["declarations"]["status"] == C.NOT_APPLICABLE

    def test_a_declared_configuration_refuses_naming_the_escape_hatch(self):
        """A header-located fitted cell asked thinking-ON out of support is a
        DECLARED configuration; the native engine has no declaration surface —
        a REAL limitation, refused by name (spec concern 7)."""
        ep = core_endpoint()
        with pytest.raises(C.OutOfSupportAttestationRefusal) as exc:
            C.check_serve_attestation(ep, cell=LORA_CELL, arm="on", out_of_support=True,
                                      model_key="gemma4e4b", expected_object_sha256=SHA, budget=BUDGET,
                                      header_located=True)
        assert exc.value.rec == C.REC_OOS_DECLARATION_UNATTESTED
        assert "--serving-path lens" in str(exc.value) and "DECLARED configuration" in str(exc.value)
        # a declared SPELLING of the cell: under the override it is the same declared
        # configuration (refused naming the hatch); in support there is no declared
        # native cell to spell, so the spelling itself refuses
        with pytest.raises(C.OutOfSupportAttestationRefusal, match="--serving-path lens"):
            C.check_serve_attestation(ep, cell=f"{LORA_CELL}@thinking", arm="on", out_of_support=True,
                                      model_key="gemma4e4b", expected_object_sha256=SHA, budget=BUDGET)
        with pytest.raises(C.CellNameError):
            C.check_serve_attestation(ep, cell=f"{LORA_CELL}@thinking", arm="off", out_of_support=False,
                                      model_key="gemma4e4b", expected_object_sha256=SHA, budget=None)

    def test_the_lens_fixtures_read_exactly_as_before(self, monkeypatch):
        """The lens path's outputs, pinned as literals derived from the lens
        rules (in support: declarations tolerated, digest compared, the salt
        policy and the attested per-cell policy read, served positions
        recorded)."""
        _budget_ok(monkeypatch)
        for ep in (capctrl_endpoint(), mt_endpoint()):
            out = C.check_serve_attestation(ep, cell=LORA_CELL, arm="off", out_of_support=False,
                                            model_key="gemma4e4b", expected_object_sha256=SHA, budget=None)
            assert out == {
                "declarations": {"expected": None, "own_key": LORA_HARNESS, "own": None, "twin_key": None,
                                 "twin": None, "raw": {}, "error_present": True},
                "object_digest": {"scalar": SHA, "by_cell": {LORA_HARNESS: SHA}},
                C.ENDPOINT_KEY_LENS_SALT_POLICY: "attested",
                C.ROUTE_KEY_RESUMED_PREFILL_POLICY: "always",
                C.LENS_SERVED_POSITIONS_KEY: ["all"],
            }
            assert C.SERVED_ENGINE_FIELD not in out
            base = C.check_serve_attestation(ep, cell="none", arm="off", out_of_support=False,
                                             model_key="gemma4e4b", expected_object_sha256=None, budget=None)
            assert base == {"declarations": {"expected": None, "own_key": "none", "own": None, "twin_key": None,
                                             "twin": None, "raw": {}, "error_present": True},
                            "object_digest": None, C.ENDPOINT_KEY_LENS_SALT_POLICY: "attested",
                            C.ROUTE_KEY_RESUMED_PREFILL_POLICY: None}
            # and the lens refusals still fire
            with pytest.raises(C.OutOfSupportAttestationRefusal) as exc:
                C.check_serve_attestation({**ep, "lens_salt_policy": "request_unique_all"}, cell=LORA_CELL,
                                          arm="off", out_of_support=False, model_key="gemma4e4b",
                                          expected_object_sha256=SHA, budget=None)
            assert exc.value.rec == C.REC_OOS_SALT_POLICY_UNEXPECTED

    def test_native_serve_declarations_record_shape(self):
        rec = C.native_serve_declarations_record(LORA_CELL)
        assert set(rec) == {"expected", "own_key", "own", "twin_key", "twin", "raw", "error_present",
                            "status", "reason"}
        assert rec["own_key"] == LORA_HARNESS and rec["status"] == C.NOT_APPLICABLE


# ---------------------------------------------------------------------------
# 6. served_reduction_mode == lens_worker.endpoint_reduction_mode, case by case
# ---------------------------------------------------------------------------


class TestServedReductionModeAgreesWithTheWorkerReader:
    def test_the_literals_agree(self):
        from sisteer.capeval.serving import lens_worker as lw
        assert lw.LENS_REDUCTION_MODE_KEY == C.PROXY_IDENTITY_REDUCTION_MODE == px.LENS_REDUCTION_MODE_KEY
        assert lw.LENS_REDUCTION_MODE_ERROR_KEY == C.LENS_REDUCTION_MODE_ERROR_KEY == "lens_reduction_mode_error"
        assert lw.ENDPOINT_ENGINE_KEY == C.ENDPOINT_KEY_ENGINE == px.ENGINE_KEY
        assert lw.ENGINE_KIND_LORA_NATIVE == C.SERVING_ENGINE_LORA_NATIVE == sa.SERVING_ATTESTATION_KIND
        assert lw.REDUCTION_MODE_NOT_APPLICABLE == C.REDUCTION_MODE_NOT_APPLICABLE
        assert lw.ENGINE_KIND_LENS == C.SERVING_ENGINE_LENS == "lens"
        assert lw.SERVED_ENGINE_FIELD == C.SERVED_ENGINE_FIELD == "serving_engine"

    def test_case_by_case_including_the_native_case(self):
        from sisteer.capeval.serving import lens_worker as lw
        KEY, ERR = lw.LENS_REDUCTION_MODE_KEY, lw.LENS_REDUCTION_MODE_ERROR_KEY
        why = "scheduler-config worker 0: no lens_reduction_mode in its scheduler-config report"
        records = [{KEY: "per_row"}, {KEY: "stacked"},
                   {KEY: "per_row", ERR: None}, {KEY: "stacked", ERR: ""},
                   {}, {"proxy_url": "http://127.0.0.1:1/v1"},
                   {KEY: None, ERR: why}, {ERR: why}, {KEY: None, ERR: ""},
                   {KEY: None}, {KEY: None, ERR: None},
                   {KEY: "vectorized"}, {KEY: "STACKED"}, {KEY: ""}, {KEY: 1}, {KEY: True},
                   None, [], "endpoint.json", 3,
                   # native-serving-default: the engine read FIRST
                   {"engine": "lora_native"}, core_endpoint(), core_endpoint(with_row_spelling=False),
                   {"engine": "lora_native", KEY: None}, {"engine": "lora_native", KEY: None, ERR: None},
                   {"engine": "lora_native", KEY: "stacked"},                 # torn -> both refuse
                   {"engine": "lora_native", KEY: "per_row", ERR: why},
                   {"engine": "vllm", KEY: "stacked"},                       # an UNKNOWN engine: both refuse
                   {"engine": None, KEY: "per_row"},
                   {"engine": "lens", KEY: "stacked"}, {"engine": "lens"},   # the lens spelled out
                   {C.SERVED_ENGINE_FIELD: "lens", KEY: "per_row"},
                   {C.SERVED_ENGINE_FIELD: "lora_native"},                   # the row spelling alone
                   {C.SERVED_ENGINE_FIELD: "lora_native", KEY: "stacked"},
                   {"engine": "lora_native", C.SERVED_ENGINE_FIELD: "lens"},  # torn: two spellings disagree
                   {"engine": "lens", C.SERVED_ENGINE_FIELD: "lora_native"},
                   {"engine": "LORA_NATIVE"}, {"engine": ""}, {"engine": 1},
                   ident_native(), ident_lens("stacked"), ident_lens(None),   # the /v1/models block
                   {**ident_native(), KEY: "stacked"},                       # a torn identity
                   {"engine": {"kind": "lens_plugin"}, KEY: "stacked"},
                   capctrl_endpoint(), mt_endpoint(mode="per_row"),
                   capctrl_endpoint(mode=None, mode_error="workers disagree")]

        def read(fn, record, exc):
            try:
                return fn(record), False
            except exc:
                return None, True

        for record in records:
            mine = read(C.served_reduction_mode, record, ValueError)
            theirs = read(lw.endpoint_reduction_mode, record, lw.ReductionModeUnattested)
            assert mine == theirs, (record, mine, theirs)
        assert read(C.served_reduction_mode, core_endpoint(), ValueError) == ("not_applicable", False)
        assert read(C.served_reduction_mode, {"engine": "lora_native", KEY: "stacked"}, ValueError) == (None, True)
        assert read(C.served_reduction_mode, capctrl_endpoint(), ValueError) == ("stacked", False)
        assert read(C.served_reduction_mode, {}, ValueError) == ("per_row", False)
        assert read(C.served_reduction_mode, capctrl_endpoint(mode=None, mode_error="x"), ValueError) == (None, True)
        # the identity block's mapping spelling is native; the lens spelled out reads its mode;
        # an unknown engine and a torn pair of spellings refuse
        assert read(C.served_reduction_mode, ident_native(), ValueError) == ("not_applicable", False)
        assert read(C.served_reduction_mode, {"engine": "lens", KEY: "stacked"}, ValueError) == ("stacked", False)
        assert read(C.served_reduction_mode, {"engine": "vllm", KEY: "stacked"}, ValueError) == (None, True)
        assert read(C.served_reduction_mode, {"engine": "lora_native", C.SERVED_ENGINE_FIELD: "lens"},
                    ValueError) == (None, True)

    def test_the_worker_reader_reads_the_engine_first(self):
        from sisteer.capeval.serving import lens_worker as lw
        assert lw.endpoint_reduction_mode({"engine": "lora_native"}) == "not_applicable"
        with pytest.raises(lw.ReductionModeUnattested, match="torn record"):
            lw.endpoint_reduction_mode({"engine": "lora_native", "lens_reduction_mode": "stacked"})
        # the /v1/models identity block's mapping spelling, the row spelling alone, the lens spelled out
        assert lw.endpoint_reduction_mode(ident_native()) == "not_applicable"
        assert lw.endpoint_reduction_mode({C.SERVED_ENGINE_FIELD: "lora_native"}) == "not_applicable"
        assert lw.endpoint_reduction_mode({"engine": "lens", "lens_reduction_mode": "stacked"}) == "stacked"
        with pytest.raises(lw.ReductionModeUnattested, match="two spellings of one engine that disagree"):
            lw.endpoint_reduction_mode({"engine": "lora_native", C.SERVED_ENGINE_FIELD: "lens"})
        with pytest.raises(lw.ReductionModeUnattested, match="unknown engine"):
            lw.endpoint_reduction_mode({"engine": "vllm", "lens_reduction_mode": "stacked"})
        src = inspect.getsource(lw.endpoint_reduction_mode)
        assert src.index("ENDPOINT_ENGINE_KEY") < src.index("why_none = ")   # the engine before the lens keys
        # the pre-change readings are byte-identical
        assert lw.endpoint_reduction_mode({}) == "per_row" and lw.endpoint_reduction_mode({"lens_reduction_mode": "stacked"}) == "stacked"

    def test_copy_of_the_fixture_is_not_mutated_by_the_readers(self):
        ep = core_endpoint()
        before = copy.deepcopy(ep)
        C.served_reduction_mode(ep); C.endpoint_is_native(ep)
        C.check_serve_attestation(ep, cell=LORA_CELL, arm="off", out_of_support=False, model_key="gemma4e4b",
                                  expected_object_sha256=SHA, budget=None)
        assert ep == before


# ---------------------------------------------------------------------------
# 7. the native core's seam vocabulary == its sources (native_standup re-exports
#    the seam names; three literals stay so it gains no contracts / proxy_engine import)
# ---------------------------------------------------------------------------


class TestNativeStandupSeamsAgreeWithTheirSources:
    """``native_standup`` re-exports ``registry_adapter`` / ``defense_cells``'
    seam names as the SOURCE objects, wraps ``proxy_engine.served_engine_problem``
    and ``contracts.endpoint_is_native`` behind call-time imports, and keeps
    THREE literals (``REDUCTION_MODE_NOT_APPLICABLE`` / ``NOT_APPLICABLE``,
    ``ENGINE_EXPECTATIONS``) so the core gains no ``contracts`` / ``proxy_engine``
    import — each literal pinned equal to its source here, the way section 6
    pins the worker reader's."""

    def test_the_literals_agree(self):
        from sisteer.capeval.petri import registry_adapter as ra
        from sisteer.capeval.serving import native_standup as ns
        assert ns.REDUCTION_MODE_NOT_APPLICABLE == C.REDUCTION_MODE_NOT_APPLICABLE == ns.NOT_APPLICABLE == C.NOT_APPLICABLE
        assert ns.ENGINE_EXPECTATIONS == pe.ENGINE_EXPECTATIONS == ("lens:stacked", "lens:per_row", "lora_native")
        assert ns.NATIVE_DISPOSITIONS is dc.NATIVE_DISPOSITIONS
        assert ns.DEFAULT_STANDUP_SERVING_PATH == ra.DEFAULT_STANDUP_SERVING_PATH == "lora_native"
        # the two standup paths are the two served engines, the native one first (the entry-point default)
        assert set(ns.SERVING_PATHS) == set(C.SERVING_ENGINES) and ns.SERVING_PATHS[0] == ns.DEFAULT_STANDUP_SERVING_PATH

    def test_the_re_exports_are_the_source_objects_and_the_wrappers_answer_as_their_sources(self):
        from sisteer.capeval.petri import registry_adapter as ra
        from sisteer.capeval.serving import native_standup as ns
        assert ns.SkippedCell is ra.SkippedCell
        assert ns.partition_cells_for_native is ra.partition_cells_for_native
        assert ns.native_serving_disposition is dc.native_serving_disposition
        native = {"engine": {"kind": "lora_native", "adapters": {"none__lora0": {}}}, "thinking_mode": "off"}
        lens = {"lens_reduction_mode": "stacked", "thinking_mode": "off"}
        for ident, expect in ((native, "lora_native"), (lens, "lens:stacked"), (lens, "lora_native"),
                              (native, "lens:stacked"), ({**native, "lens_reduction_mode": "stacked"}, "lora_native")):
            assert ns.served_engine_problem(ident, expect=expect) == pe.served_engine_problem(ident, expect=expect)
        for ep in (native, lens, {"engine": "lora_native"}, {C.SERVED_ENGINE_FIELD: "lens"}, {}, None):
            assert ns.endpoint_is_native(ep) is C.endpoint_is_native(ep)
        for name in ("DEFAULT_STANDUP_SERVING_PATH", "SkippedCell", "partition_cells_for_native",
                     "native_serving_disposition", "NATIVE_DISPOSITIONS", "served_engine_problem",
                     "endpoint_is_native", "ENGINE_EXPECTATIONS", "REDUCTION_MODE_NOT_APPLICABLE",
                     "NOT_APPLICABLE", "SERVING_PATHS"):
            assert name in ns.__all__, name

    def test_the_core_imports_neither_contracts_nor_proxy_engine_at_module_level(self):
        """The call-time imports exist for this: importing the core must not pull ``proxy`` / ``vllm_server``."""
        import subprocess
        import sys
        heavy = ("sisteer.capeval.serving.proxy_engine", "sisteer.capeval.tinycap.contracts",
                 "sisteer.capeval.serving.vllm_server", "sisteer.capeval.serving.proxy")
        code = ("import sys\nimport sisteer.capeval.serving.native_standup\n"
                f"print(sorted(m for m in sys.modules if m in {heavy!r}))")
        out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout.strip()
        assert out == "[]", out
