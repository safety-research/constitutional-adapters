"""native-psr-serving step 2 — the NATIVE POSITION MODE and the residual kinds'
DISPOSITION (``defense_cells.native_position_mode``, ``NATIVE_LORA_SERVABLE_
KINDS``, ``native_serving_disposition``).

Pinned here:

- the kind gate: the five residual kinds are natively servable and the
  exporter's set equals it (strict);
- ``native_position_mode`` per kind x record shape, read as the object's own
  loader reads it: psr / diffmean from the FIT leg ``positions`` (missing ->
  ``all``, the loaders' default), ``response`` needing its header width;
  vector_all's deployment ``positions`` with the ``fit_positions`` cross-check
  (a masked fit deployed at every position REFUSES); vector's sidecar
  ``positions`` vs the persisted ``vector_object`` record; static_r's
  ``steer_positions`` / ``site`` and the cross-phase ``positions == response``;
  every malformed record raises ``DefenseCellError`` naming the fact;
- the disposition in BOTH worlds (``NATIVE_STAGE2_ARMED`` True since step 7,
  False monkeypatched for the stage-1 world): mode ``all`` -> serve; another
  mode -> SERVE once armed, skip with the mode named while unarmed; a record
  the predicate cannot judge -> refuse; static_r ``pre`` at block 0 -> refuse;
  an absent artifact on a residual kind -> serve (ABSENT downstream, like
  a LoRA cell);
- the fact tags (``UNSERVABLE_FACTS`` / ``NATIVE_SKIP_FACTS``) and the
  partition over the synthetic bank (the response-fit psr cell SERVED under the
  armed flag, skipped with the stage-2 reason while it is not).

CPU-only; torch-free specs."""

from __future__ import annotations

import pytest

from sisteer.capeval import manifest as cm
from sisteer.capeval.petri import defense_cells as dc
from sisteer.capeval.petri import registry_adapter as ra
from sisteer.capeval.serving import lora_export as lx
from sisteer.capeval.serving import native_residual as nr

PSR_ALL = {"kind": "psr", "layers": [0, 2], "train_direction": False, "positions": "all",
           "response_header_len": None}
PSR_RESP = {**PSR_ALL, "positions": "response", "response_header_len": 3}
VA_ALL = {"kind": "vector_all", "positions": "all", "skip_bos": True, "fit_positions": "all",
          "fit_response_header_len": None}
DM_ALL = {"kind": "diffmean", "skip_bos": True, "positions": "all", "response_header_len": None,
          "metadata": {}}
VEC_ALL = {"kind": "vector", "method": "caa", "layers": [3], "norms": {"3": 1.0},
           "metadata": {"vector_object": {"positions": "all", "skip_bos": True, "norm_relative": False}}}
SR = {"kind": "static_r", "base_scale": 0.05, "steer_positions": "prefill_all", "site": "pre",
      "layers": [19], "metadata": {}}


def test_kind_gate():
    assert dc.NATIVE_LORA_SERVABLE_KINDS == ("circuit_breaker", "lora", "lora_all", "lora_subtraction",
                                             "psr", "vector", "vector_all", "diffmean", "static_r",
                                             "psr_paired_write")   # the paired write (relu-differenced-native-qwen)
    assert set(dc.NATIVE_LORA_SERVABLE_KINDS) == set(lx.GENUINE_EXPORT_KINDS)
    assert dc.RESIDUAL_SLOT_KINDS == nr.RESIDUAL_SLOT_KINDS == lx.RESIDUAL_EXPORT_KINDS
    assert set(dc.RESIDUAL_SLOT_KINDS) < set(dc.NATIVE_LORA_SERVABLE_KINDS)
    assert dc.NATIVE_POSITION_MODES == nr.POSITION_MODES == ("all", "response", "prompt_last", "prefill_all")
    # step 7 ARMS the position class. The two flags must agree (the worker refuses to LOAD
    # a restricted payload while its own is False) — pinned equal by
    # tests/test_native_lora_constants_agree.py, which is where the pair lives; here the
    # gate's own value is what the disposition below reads.
    assert dc.NATIVE_STAGE2_ARMED is True


@pytest.mark.parametrize("kind,spec,want", [
    ("psr", PSR_ALL, ("all", None)),
    ("psr", {k: v for k, v in PSR_ALL.items() if k != "positions"}, ("all", None)),   # the loader's default
    ("psr", PSR_RESP, ("response", 3)),
    ("diffmean", DM_ALL, ("all", None)),
    ("diffmean", {**DM_ALL, "positions": "response", "response_header_len": 4}, ("response", 4)),
    ("vector_all", VA_ALL, ("all", None)),
    ("vector_all", {**VA_ALL, "positions": "response", "fit_positions": "response",
                    "fit_response_header_len": 3}, ("response", 3)),
    ("vector_all", {**VA_ALL, "positions": "response", "response_header_len": 5}, ("response", 5)),  # cross-phase
    ("vector", VEC_ALL, ("all", None)),
    ("vector", {"kind": "vector", "method": "caa"}, ("all", None)),
    ("vector", {**VEC_ALL, "positions": "all"}, ("all", None)),
    ("static_r", SR, ("prefill_all", None)),
    ("static_r", {**SR, "steer_positions": "prompt_last", "site": "post"}, ("prompt_last", None)),
    ("static_r", {**SR, "positions": "response"}, ("response", None)),                 # the cross-phase Arditi cell
    ("static_r", {"kind": "static_r", "base_scale": 1.0}, ("prompt_last", None)),      # the legacy defaults
])
def test_native_position_mode(kind, spec, want):
    assert dc.native_position_mode(kind, spec) == want


@pytest.mark.parametrize("kind,spec,needle", [
    ("psr", {**PSR_ALL, "positions": "last"}, "not a native residual-slot mode"),
    ("psr", {**PSR_ALL, "positions": "response_all_turns"}, "not a native residual-slot mode"),
    ("psr", {**PSR_ALL, "positions": "response", "response_header_len": 0}, "positive int"),
    ("diffmean", {**DM_ALL, "positions": "prompt"}, "not a native residual-slot mode"),
    ("vector_all", {**VA_ALL, "fit_positions": "response"}, "fit=response / serve=all"),
    ("vector_all", {**VA_ALL, "fit_positions": "last"}, "not a fit leg"),
    ("vector", {**VEC_ALL, "positions": "response"}, "disagree"),
    ("vector", {"kind": "vector", "method": "caa", "positions": "front"}, "not a native residual-slot mode"),
    ("static_r", {**SR, "steer_positions": "prefill"}, "unknown steer_positions"),
    ("static_r", {**SR, "site": "mid"}, "unknown hook site"),
    ("static_r", {**SR, "positions": "all"}, "ambiguous"),
])
def test_native_position_mode_refuses_malformed_records(kind, spec, needle):
    with pytest.raises(dc.DefenseCellError, match=needle):
        dc.native_position_mode(kind, spec)
    disposition, reason = dc.native_serving_disposition(kind, spec)
    assert disposition == "refuse" and "cannot be judged" in reason and needle.split(" ")[0] in reason
    assert dc._native_lora_verdict(kind, spec)[1] == dc.UNSERVABLE_RESIDUAL_RECORD


@pytest.mark.parametrize("kind,spec", [
    ("psr", {**PSR_ALL, "positions": "response", "response_header_len": None}),
    ("vector_all", {**VA_ALL, "positions": "response"}),
    ("diffmean", {**DM_ALL, "positions": "response"}),
])
def test_a_response_record_without_its_header_width_skips_unarmed_and_refuses_armed(kind, spec, monkeypatch):
    """The width is needed to SERVE the response mode, not to decide the stage.
    ARMED (this tree, step 7) the incomplete record REFUSES: the mask's boundary
    is the fit's own and is never guessed. UNARMED the mode alone skipped the
    cell by name and the lens served it. The strict reading (the exporter's)
    raises either way."""
    with pytest.raises(dc.DefenseCellError, match="no response_header_len"):
        dc.native_position_mode(kind, spec)
    assert dc.native_position_mode(kind, spec, header_required=False) == ("response", None)
    d, why = dc.native_serving_disposition(kind, spec)
    assert d == "refuse" and "no response_header_len" in why
    monkeypatch.setattr(dc, "NATIVE_STAGE2_ARMED", False)        # the stage-1 world
    d, why = dc.native_serving_disposition(kind, spec)
    assert d == "skip" and "native position mode 'response'" in why and "stage-2" in why


def test_native_position_mode_refuses_a_non_residual_kind_and_a_missing_spec():
    with pytest.raises(dc.DefenseCellError, match="not a residual-slot kind"):
        dc.native_position_mode("lora_all", {"positions": "all"})
    with pytest.raises(dc.DefenseCellError, match="needs the resolved spec"):
        dc.native_position_mode("psr", None)


def test_disposition_in_both_worlds(monkeypatch):
    # STAGE 2 (this tree): mode `all` serves, and so does every other mode the lens names —
    # the per-token position class serves them, so the `position_mode_stage2` skip is retired
    assert dc.native_serving_disposition("psr", PSR_ALL) == ("serve", None)
    assert dc.native_serving_disposition("vector_all", VA_ALL) == ("serve", None)
    assert dc.native_serving_disposition("diffmean", DM_ALL) == ("serve", None)
    assert dc.native_serving_disposition("vector", VEC_ALL) == ("serve", None)
    STAGE2_SHAPES = (
        ("psr", PSR_RESP, "response"),
        ("static_r", SR, "prefill_all"),
        ("static_r", {**SR, "steer_positions": "prompt_last", "site": "post"}, "prompt_last"),
        ("static_r", {**SR, "positions": "response"}, "response"),
        ("vector_all", {**VA_ALL, "positions": "response", "response_header_len": 3}, "response"))
    for kind, spec, _mode in STAGE2_SHAPES:
        assert dc.native_serving_disposition(kind, spec) == ("serve", None), (kind, spec)
        assert dc.native_lora_unservable_reason(kind, spec) is None
        assert dc._native_lora_verdict(kind, spec) == (None, None)
    # an absent artifact on a residual kind: serve (ABSENT downstream), as for a LoRA kind
    for kind in dc.RESIDUAL_SLOT_KINDS:
        assert dc.native_serving_disposition(kind, None) == ("serve", None)
        assert dc.native_lora_unservable_reason(kind, None) is None
    # static_r pre at block 0 refuses — a rule about the SITE, not the stage, so it is
    # unchanged by the flag (armed here, unarmed below)
    d, why = dc.native_serving_disposition("static_r", {**SR, "layers": [0, 19]})
    assert d == "refuse" and "resid_pre of layer 0" in why and "D6" in why
    assert dc._native_lora_verdict("static_r", {**SR, "layers": [0, 19]})[1] == dc.UNSERVABLE_RESIDUAL_RECORD
    assert dc.native_serving_disposition("static_r", {**SR, "site": "post", "layers": [0]}) == ("serve", None)
    # a mode OUTSIDE the vocabulary still refuses by name (never a silent all-position serve)
    assert dc.native_serving_disposition("psr", {**PSR_ALL, "positions": "last"})[0] == "refuse"
    # a serve declaration refuses before the mode is read
    from sisteer.capeval.serving import serve_declaration as sd
    d, why = dc.native_serving_disposition("psr", sd.apply_variant(dict(PSR_RESP), sd.VARIANT_ALL_POSITIONS))
    assert d == "refuse" and "serve declaration" in why
    # STAGE 1 (the flag as it was): every restricted mode is SKIPPED by name with the mode in
    # the sentence and the lens serves it; `all` is unaffected; block 0 ``pre`` still refuses
    monkeypatch.setattr(dc, "NATIVE_STAGE2_ARMED", False)
    for kind, spec, mode in STAGE2_SHAPES:
        d, why = dc.native_serving_disposition(kind, spec)
        assert d == "skip", (kind, spec)
        assert f"native position mode '{mode}'" in why and "stage-2" in why and "the lens serves it" in why
        assert dc._native_lora_verdict(kind, spec)[1] == dc.UNSERVABLE_POSITION_MODE_STAGE2
        assert dc.native_lora_unservable_reason(kind, spec) == why
    assert dc.native_serving_disposition("psr", PSR_ALL) == ("serve", None)
    assert dc.native_serving_disposition("static_r", {**SR, "layers": [0, 19]})[0] == "refuse"


def test_fact_tags():
    assert set(dc.UNSERVABLE_FACTS) == {"stacked", "kind", "serve_declaration", "lora_scaling",
                                        "positions_missing", "positions_masked", "residual_record",
                                        "position_mode_stage2"}
    assert set(dc.NATIVE_SKIP_FACTS) == {"stacked", "kind", "positions_masked", "position_mode_stage2"}
    assert set(dc.NATIVE_SKIP_FACTS) < set(dc.UNSERVABLE_FACTS)
    # the o_proj rules never fire on a residual kind: no positions -> the loader's default, serve
    assert dc._native_lora_verdict("psr", {"kind": "psr", "layers": [0]}) == (None, None)
    # the LoRA kinds are unchanged
    assert dc._native_lora_verdict("lora_all", {"layers": [0], "rank": 4})[1] == "positions_missing"
    assert dc._native_lora_verdict("lora_all", {"layers": [0], "rank": 4, "positions": "response"})[1] == "positions_masked"
    assert dc._native_lora_verdict("loreft", {"positions": "all"})[1] == "kind"


def test_partition_serves_the_response_fit_psr_armed_and_skips_it_unarmed(tmp_path, monkeypatch):
    from sisteer.eval import defense_cell as tt

    from test_lora_native_enumeration import (H_LORA, H_PSR, H_SUB, TT_LORA, TT_PSR, TT_SUB, _bank,
                                              _explicit_source, _stage_budget_cell)

    bank = _bank(tmp_path)                                   # the synthetic psr cell is fit at 'response'
    lens = _explicit_source(bank, [TT_LORA, TT_PSR, TT_SUB])
    assert lens.get(H_PSR).resolved_spec["positions"] == "response"
    # ARMED (this tree): the response-fit cell is served natively, as a lora route
    served, skipped = ra.partition_cells_for_native(["none", H_PSR, H_LORA, H_SUB], lens)
    assert served == ["none", H_PSR, H_LORA, H_SUB] and skipped == []
    native = _explicit_source(bank, [TT_LORA, TT_PSR, TT_SUB], serving_path_policy="lora_native")
    assert native.get(H_PSR).serving_path == "lora"
    # UNARMED (stage 1): skipped by name, and the lora_native backstop refuses it by name
    monkeypatch.setattr(dc, "NATIVE_STAGE2_ARMED", False)
    served, skipped = ra.partition_cells_for_native(["none", H_PSR, H_LORA, H_SUB], lens)
    assert served == ["none", H_LORA, H_SUB]
    assert [(s.cell_id, s.kind) for s in skipped] == [(H_PSR, "psr")]
    assert "native position mode 'response'" in skipped[0].reason and "stage-2" in skipped[0].reason
    native_unarmed = _explicit_source(bank, [TT_LORA, TT_PSR, TT_SUB], serving_path_policy="lora_native")
    with pytest.raises(ra.RegistryAdapterError, match="stage-2"):
        native_unarmed.get(H_PSR)
    monkeypatch.undo()
    # an ALL-positions psr cell in the same bank is SERVED (a lora route)
    bank2 = tmp_path / "bank2"
    import torch
    g = torch.Generator().manual_seed(4)
    blob = {"kind": "psr", "layers": [0, 2], "train_direction": False, "positions": "all",
            "response_header_len": None,
            "directions": {str(l): torch.randn(10, generator=g) for l in (0, 2)},
            "gate_w": {str(l): torch.randn(10, generator=g) for l in (0, 2)},
            "gate_b": {str(l): torch.randn(1, generator=g) for l in (0, 2)}}
    _stage_budget_cell(bank2, TT_PSR, blob)
    lens2 = _explicit_source(bank2, [TT_PSR])
    assert ra.partition_cells_for_native(["none", H_PSR], lens2) == (["none", H_PSR], [])
    native2 = _explicit_source(bank2, [TT_PSR], serving_path_policy="lora_native")
    assert native2.get(H_PSR).serving_path == "lora"


# ---------------------------------------------------------------------------
# the CONTRACT side of the same ruling (native-psr-serving §1.8): a psr cell
# served NATIVELY is not a lens leg, so the §2.5.11 lens rows do not apply to
# it. Pinned here — beside the disposition that makes such a cell servable in
# the first place — because contracts.py has no stage-2 test home of its own.
# ---------------------------------------------------------------------------


def test_the_authorised_psr_predicates_are_engine_aware():
    from sisteer.capeval.tinycap import contracts as tc

    cell = "safety_refusal__psr_kl_all"      # the authorised experiment's fitted cell
    model_key = next(iter(tc.AUTHORISED_OOS_THINKING.model_spellings()))
    # unstated engine: every existing caller's reading, byte-identical
    assert tc.is_authorised_psr_cell(cell) is True
    assert tc.is_authorised_psr_leg(cell, model_key) is True
    assert tc.is_authorised_psr_cell(cell, engine=tc.SERVING_ENGINE_LENS) is True
    # NATIVE: the same object is served as a residual adapter, not on the lens hook — the
    # contract's lens rows describe a leg that does not exist there
    assert tc.is_authorised_psr_cell(cell, engine=tc.SERVING_ENGINE_LORA_NATIVE) is False
    assert tc.is_authorised_psr_leg(cell, model_key, engine=tc.SERVING_ENGINE_LORA_NATIVE) is False
    with pytest.raises(ValueError, match="is not one of"):
        tc.is_authorised_psr_cell(cell, engine="vllm")


def test_the_authorised_stamp_policy_is_engine_aware():
    from sisteer.capeval.tinycap import contracts as tc

    # unchanged without an engine (the five §2.5.11 rows)
    assert tc.authorised_stamp_policy(lens=True, arm=tc.THINKING_ON) == tc.DECLARED_SALT_POLICY_LABEL
    assert tc.authorised_stamp_policy(lens=True, arm=tc.THINKING_OFF) == tc.RESUMED_PREFILL_CONDITIONAL
    assert tc.authorised_stamp_policy(lens=False, arm=tc.THINKING_OFF) == tc.POLICY_NOT_CONSULTED
    # NATIVE: no policy is consulted off the lens path, whatever the arm
    for arm in (tc.THINKING_ON, tc.THINKING_OFF):
        assert tc.authorised_stamp_policy(lens=False, arm=arm,
                                          engine=tc.SERVING_ENGINE_LORA_NATIVE) == tc.POLICY_NOT_CONSULTED
    # ... and a LENS leg on the native engine is a contradiction, refused rather than answered
    with pytest.raises(ValueError, match="serves no lens route"):
        tc.authorised_stamp_policy(lens=True, arm=tc.THINKING_OFF,
                                   engine=tc.SERVING_ENGINE_LORA_NATIVE)
    with pytest.raises(ValueError, match="is not one of"):
        tc.authorised_stamp_policy(lens=False, arm=tc.THINKING_OFF, engine="vllm")


def test_a_restricted_native_position_mode_expects_a_request_unique_salt():
    """A native leg's KV namespace is decided by the residual slot's POSITION MODE —
    `all` (and a plain LoRA, which has none) keep the cell-stable namespace, a restricted
    mode is request-unique. The regime is the proxy's own rule, asked for here, never
    recomputed; a lens leg has no position mode and refuses one."""
    from sisteer.capeval.tinycap import contracts as tc

    base = tc.expected_stamp_regimes(lens=False, arm=tc.THINKING_OFF, policy=tc.POLICY_NOT_CONSULTED)
    assert base.salt_regime == tc.SALT_REGIME_CELL_STABLE
    for mode in ("all", None):
        got = tc.expected_stamp_regimes(lens=False, arm=tc.THINKING_OFF,
                                        policy=tc.POLICY_NOT_CONSULTED, position_mode=mode)
        assert got == base, mode
    for mode in ("response", "prompt_last", "prefill_all"):
        got = tc.expected_stamp_regimes(lens=False, arm=tc.THINKING_OFF,
                                        policy=tc.POLICY_NOT_CONSULTED, position_mode=mode)
        assert got.salt_regime == tc.SALT_REGIME_REQUEST_UNIQUE, mode
        assert got.resumed_policy == tc.POLICY_NOT_CONSULTED
    with pytest.raises(ValueError, match="on a LENS leg"):
        tc.expected_stamp_regimes(lens=True, arm=tc.THINKING_OFF,
                                  policy=tc.RESUMED_PREFILL_CONDITIONAL, position_mode="response")
