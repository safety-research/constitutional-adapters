"""The fit/serve position-regime guard on the LENS serving path
(lens_adapters._check_fit_serve_positions, installed by register_adapter
around every adapter).

Mirrors the PEFT/multi-LoRA export path's existing guard
(serving/lora_export.py: _gate_positions + _check_sidecar): the two-SI
steering objects come in two INCOMPATIBLE position regimes — all-token
(Class A/B) and decode-only (Class C, ``positions='response'``) — and an
object fit under one must never be served under the other. Before this
guard, ``_adapt_psr`` (and every other lens adapter) ignored the fit leg
the blob itself records, so a decode-only-fit psr blob would silently
serve all-position under the cell's hash. What these tests pin:

- a MISMATCHED fit/serve pair refuses, naming both regimes (the arm is
  named by the export wrapper, pinned here too);
- a MATCHED pair serves unchanged;
- the ABSENT case passes deliberately (surveyed 2026-08-27: 0/284
  on-disk blobs carry ``positions_at_fit``; only the PositionMaskedFit
  family persists a fit leg at all, and the classes pin an absent key to
  the historical all-position fit) and is RECORDED: the export index
  stamps ``positions_at_fit`` per cell, explicit null included;
- the fit's own records disagreeing with each other refuses (the
  _check_sidecar mirror);
- the DELIBERATE cross-phase re-mask (fit "all", CELL spec requesting
  "response") still serves — the Evals-lane decision _served_positions
  documents must not regress;
- the guard also fires on the worker-registry load path, so an index
  that never went through the capeval exporter cannot bypass it.
"""

import io
import json

import pytest
import torch

from sisteer.capeval import manifest as cm
from sisteer.capeval.serving import lens_adapters as la
from sisteer.capeval.serving import lens_steer as ls
from sisteer.vectors import SteeringVector

HIDDEN = 8
HDR = (7, 8, 9)


def _psr_blob_bytes(extra=None) -> bytes:
    layers = (0, 2)
    g = torch.Generator().manual_seed(9)
    blob = {
        "kind": "psr", "layers": list(layers),
        "directions": {str(l): torch.randn(HIDDEN, generator=g) * 0.05
                       for l in layers},
        "gate_w": {str(l): torch.randn(HIDDEN, generator=g) * 0.3
                   for l in layers},
        "gate_b": {str(l): torch.randn(1, generator=g) * 0.5
                   for l in layers},
    }
    blob.update(extra or {})
    buf = io.BytesIO()
    torch.save(blob, buf)
    return buf.getvalue()


def _vector_all_blob_bytes(positions) -> bytes:
    buf = io.BytesIO()
    torch.save({"kind": "vector_all", "vectors": {"0": torch.ones(4)},
                "positions": positions, "skip_bos": True}, buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# the adapter-level guard (every serving path traverses get_adapter)
# ---------------------------------------------------------------------------


def test_response_fit_blob_refuses_all_position_serving():
    """THE HEADLINE CASE: a Class C (decode-only-fit) psr blob must never
    adapt to all-position ops — before the guard it did so silently (76
    such blobs were on cluster disk when this landed). The mainline has
    since evolved the SERVE side: with the served header ids present the
    blob serves the response MASK (the cross-phase re-mask), and without
    them the adapter refuses loudly — either way, no op is ever
    positions='all'."""
    blob = _psr_blob_bytes({"positions": "response",
                            "response_header_len": len(HDR)})
    # no served header ids -> the mask cannot be armed -> loud refusal,
    # never a silent all-position adaptation
    with pytest.raises(la.AdapterError) as ei:
        la.get_adapter("psr")({"kind": "psr"}, blob, b"")
    assert "response" in str(ei.value)
    assert "header" in str(ei.value)
    # served header ids present -> every op wears the response mask
    ops = la.get_adapter("psr")({"kind": "psr"}, blob, b"",
                                header_ids=HDR)
    assert [o.positions for o in ops] == [la.POSITIONS_RESPONSE] * 2
    assert la.POSITIONS_ALL not in {o.positions for o in ops}


def test_matched_all_fit_blob_serves_unchanged():
    ref = la.get_adapter("psr")({"kind": "psr"}, _psr_blob_bytes(), b"")
    ops = la.get_adapter("psr")({"kind": "psr"},
                                _psr_blob_bytes({"positions": "all"}), b"")
    assert [o.positions for o in ops] == [la.POSITIONS_ALL] * 2
    # guard only — the recorded-fit blob serves the SAME ops as the
    # record-less one (extra blob keys never leak into the translation)
    assert [(o.op, o.layer, o.phase) for o in ops] == \
        [(o.op, o.layer, o.phase) for o in ref]


def test_matched_response_fit_vector_all_serves_the_response_mask():
    ops = la.get_adapter("vector_all")(
        {"kind": "vector_all"}, _vector_all_blob_bytes("response"), b"",
        header_ids=HDR)
    assert [o.positions for o in ops] == [la.POSITIONS_RESPONSE]


def test_absent_fit_record_passes_and_reads_none():
    """The DELIBERATE absent-case decision: pass (0/284 surveyed on-disk
    blobs carry positions_at_fit; refusing absence would refuse the
    whole servable set, including classes whose trainers refuse masked
    fits and never persist a leg), reading None for the caller to
    record."""
    blob = _psr_blob_bytes()
    assert la.recorded_fit_positions("psr", blob, b"") is None
    ops = la.get_adapter("psr")({"kind": "psr"}, blob, b"")
    assert [o.positions for o in ops] == [la.POSITIONS_ALL] * 2


def test_fit_records_disagreeing_with_each_other_refuse():
    """The _check_sidecar mirror: two records of the fit that contradict
    each other are never resolved by picking one."""
    blob = _psr_blob_bytes({"positions": "all",
                            "positions_at_fit": "response"})
    with pytest.raises(la.AdapterError, match="records disagree"):
        la.get_adapter("psr")({"kind": "psr"}, blob, b"")


def test_blob_positions_at_fit_alone_is_honoured():
    blob = _psr_blob_bytes({"positions_at_fit": "response"})
    with pytest.raises(la.AdapterError, match="DECODE-ONLY"):
        la.get_adapter("psr")({"kind": "psr"}, blob, b"")


def test_sidecar_positions_at_fit_guards_the_vector_kind(tmp_path):
    """The safetensors 'vector' kind takes its fit record from the JSON
    sidecar (the lora_export convention): a decode-only record refuses
    all-position serving and passes response serving."""
    sv = SteeringVector(vectors={1: torch.ones(HIDDEN) * 0.05},
                        method="caa", model_name="TESTMODEL",
                        instruction_id="synthetic")
    path = tmp_path / "vec.safetensors"
    sv.save(path)
    artifact = path.read_bytes()
    side = json.loads(path.with_suffix(".json").read_text())
    side["positions_at_fit"] = "response"
    sidecar = json.dumps(side).encode("utf-8")
    with pytest.raises(la.AdapterError, match="DECODE-ONLY"):
        la.get_adapter("vector")({"kind": "vector", "positions": "all"},
                                 artifact, sidecar)
    ops = la.get_adapter("vector")(
        {"kind": "vector", "positions": "response"}, artifact, sidecar,
        header_ids=HDR)
    assert [o.positions for o in ops] == [la.POSITIONS_RESPONSE]


def test_deliberate_cross_phase_re_mask_still_serves():
    """REGRESSION PIN (via _served_positions): an all-fit artifact
    served under the response mask BY THE CELL SPEC is the deliberate,
    hashed cross-phase cell — the guard must not make it unservable
    again."""
    ops = la.get_adapter("vector_all")(
        {"kind": "vector_all", "positions": "response"},
        _vector_all_blob_bytes("all"), b"", header_ids=HDR)
    assert [o.positions for o in ops] == [la.POSITIONS_RESPONSE]


def test_response_fit_with_spec_override_to_all_refuses():
    """The reverse cross is the audited hazard itself (a decode-only fit
    served all-token) and stays refused even when the spec requests it —
    a deliberate transfer cell needs its own carve-out, never a mask
    override fallout."""
    with pytest.raises(la.AdapterError, match="DECODE-ONLY"):
        la.get_adapter("vector_all")(
            {"kind": "vector_all", "positions": "all"},
            _vector_all_blob_bytes("response"), b"")


def test_unknown_or_malformed_fit_records_refuse():
    with pytest.raises(la.AdapterError, match="not lens-served"):
        la.get_adapter("psr")({"kind": "psr"},
                              _psr_blob_bytes({"positions": "front"}), b"")
    with pytest.raises(la.AdapterError, match="not lens-served"):
        la.get_adapter("psr")({"kind": "psr"},
                              _psr_blob_bytes({"positions": 3}), b"")


# ---------------------------------------------------------------------------
# the worker-registry load path (unbypassable: guard rides get_adapter)
# ---------------------------------------------------------------------------


def _registry_with_response_fit_entry(tmp_path, header_len):
    blob = _psr_blob_bytes({"positions": "response",
                            "response_header_len": header_len})
    reg = tmp_path / "reg"
    (reg / "arm").mkdir(parents=True)
    (reg / "arm" / "artifact.bin").write_bytes(blob)
    resolved = {"kind": "psr", "coeff": 1.0}
    ih = cm.petri_intervention_hash(resolved, artifact_bytes=blob,
                                    sidecar_bytes=b"")
    entry = {"cell_id": "arm__c1", "kind": "psr",
             "resolved_spec": resolved, "expect_hash": ih,
             "artifact": "arm/artifact.bin", "sidecar": None,
             "artifact_sha16": cm.sha256_hex16(blob),
             "artifact_intervention_hash": ih, "forced_scale": 1.0,
             "donor_cell": None, "validated_by": None}
    (reg / ls.REGISTRY_INDEX).write_text(json.dumps(
        {"version": 1, "assistant_header_ids": list(HDR),
         "entries": [entry]}))
    return reg


def test_worker_registry_load_refuses_a_mismatched_response_fit_entry(
        tmp_path):
    """A registry index that never went through an exporter (hand-built,
    or a foreign producer) still cannot arm a decode-only fit whose
    recorded header disagrees with the served template's: the width
    cross-check rides get_adapter, where the guard lives."""
    reg = _registry_with_response_fit_entry(tmp_path,
                                            header_len=len(HDR) + 1)
    with pytest.raises(la.AdapterError, match="DIFFERENT regions"):
        ls.WorkerObjectRegistry.load(reg)


def test_worker_registry_load_serves_a_response_fit_entry_masked(tmp_path):
    """The matched case: a decode-only-fit entry loads, and every op it
    serves wears the response mask — the registry's header ids armed it;
    all-position serving of a Class C blob is unreachable."""
    reg = _registry_with_response_fit_entry(tmp_path, header_len=len(HDR))
    loaded = ls.WorkerObjectRegistry.load(reg)
    (obj,) = loaded.objects()
    assert obj.ops and {o.positions for o in obj.ops} == \
        {la.POSITIONS_RESPONSE}


# ---------------------------------------------------------------------------
# per-kind blob vocabularies (_BLOB_POSITION_SCHEMAS): the same key name
# records DIFFERENT things across the save schemas — the PositionMaskedFit
# leg (psr/lora/lora_all), vector_all's serve-time deployment attr, and
# loreft's intrinsic prompt/response policy — and the guard must read each
# through its own kind's declaration. Regression (2026-09-01): reading
# loreft's intrinsic "prompt" as a malformed PositionMaskedFit leg refused
# every loreft artifact ever saved.
# ---------------------------------------------------------------------------


def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(hidden_size=HIDDEN, intermediate_size=2 * HIDDEN,
                      num_hidden_layers=3, num_attention_heads=2,
                      num_key_value_heads=2, vocab_size=32)
    return LlamaForCausalLM(cfg)


def _loreft_blob_bytes(tmp_path, fit_positions=None) -> bytes:
    from sisteer.extraction.reft import LoReFTObject
    torch.manual_seed(7)
    obj = LoReFTObject(_tiny_llama(), [1], rank=2)
    if fit_positions is not None:
        obj.set_fit_positions(fit_positions, len(HDR))
    path = tmp_path / "loreft.pt"
    obj.save(path)
    return path.read_bytes()


def test_intrinsic_prompt_record_accepted_and_response_fit_psr_refused(
        tmp_path):
    """THE INVARIANT PAIR (regression, 2026-09-01), both directions in
    one place so neither fix can quietly cost the other:

    - a loreft blob's ``positions`` is the class's INTRINSIC policy —
      "prompt" is the method's identity fit, in a vocabulary of its own
      (LoReFTObject.save) — and must be ACCEPTED, serving the class's
      prefill-only translation;
    - a response-fit psr blob served all-position — the corner the guard
      exists for (it invalidated seven scale-12 conditions) — must stay
      REFUSED, even when the cell spec requests the widening."""
    blob = _loreft_blob_bytes(tmp_path)
    assert la.recorded_fit_positions("loreft", blob, b"") \
        == la.POSITIONS_PROMPT
    ops = la.get_adapter("loreft")(
        {"kind": "loreft", "layers": [1], "rank": 2}, blob, b"")
    assert ops and all(op.phase == la.PHASE_PREFILL for op in ops)
    assert la.POSITIONS_RESPONSE not in {op.positions for op in ops}

    psr = _psr_blob_bytes({"positions": "response",
                           "response_header_len": len(HDR)})
    with pytest.raises(la.AdapterError, match="DECODE-ONLY"):
        la.get_adapter("psr")({"kind": "psr", "positions": "all"}, psr,
                              b"")


def test_loreft_response_record_normalizes_through_its_own_vocabulary(
        tmp_path):
    """The OTHER loreft value: "response" in loreft's vocabulary IS the
    decode-only PositionMaskedFit leg, and the record must read as such
    (the adapter's own response refusal is pinned in
    test_petri_lens_adapters; this pins the record)."""
    blob = _loreft_blob_bytes(tmp_path, fit_positions="response")
    assert la.recorded_fit_positions("loreft", blob, b"") \
        == la.POSITIONS_RESPONSE


def test_prompt_record_with_decode_reaching_ops_refuses(tmp_path):
    """The backstop's teeth for the intrinsic regime: if a translation
    ever serves a prompt-fit record with decode-reaching ops (the
    loreft_both arithmetic under the loreft kind — the psr bug's shape
    on this class), the guard refuses rather than trusting the
    adapter."""
    blob = _loreft_blob_bytes(tmp_path)
    both_phases = (la.HookOp(op=la.OP_ADD_VECTOR, layer=1,
                             vector=torch.ones(HIDDEN),
                             phase=la.PHASE_BOTH),)
    with pytest.raises(la.AdapterError, match="PROMPT-POSITIONS"):
        la._check_fit_serve_positions("loreft", {"kind": "loreft"}, blob,
                                      b"", both_phases)


def test_vector_all_explicit_fit_leg_outranks_the_deployment_attr():
    """MultiLayerVectorObject.save keeps the fit leg under
    ``fit_positions``, DISTINCT from the serve-time deployment attr
    ``positions`` — so a Class-C blob whose deployment default reads
    "all" still records a decode-only fit and must refuse an
    all-position serve (the hole the generic single-key read left
    open)."""
    buf = io.BytesIO()
    torch.save({"kind": "vector_all", "vectors": {"0": torch.ones(4)},
                "positions": "all", "skip_bos": True,
                "fit_positions": "response"}, buf)
    blob = buf.getvalue()
    assert la.recorded_fit_positions("vector_all", blob, b"") \
        == la.POSITIONS_RESPONSE
    with pytest.raises(la.AdapterError, match="DECODE-ONLY"):
        la.get_adapter("vector_all")({"kind": "vector_all"}, blob, b"")


def test_undeclared_position_vocabulary_refuses():
    """A kind with no declared save schema whose blob carries a
    position-record key refuses rather than guessing which vocabulary
    the key speaks — either mis-read is the loreft bug again, in one
    direction or the other."""
    buf = io.BytesIO()
    torch.save({"kind": "someday_kind", "positions": "all"}, buf)
    with pytest.raises(la.AdapterError, match="does not declare"):
        la.recorded_fit_positions("someday_kind", buf.getvalue(), b"")


def test_stacked_wrapper_reads_the_object_legs_vocabulary(tmp_path):
    """A stacked cell's artifact IS its object leg's blob (the stacked
    adapter delegates), so the guard must evaluate it in the OBJECT
    kind's vocabulary — a loreft leg's intrinsic "prompt" under
    kind='stacked' is the same mis-read through the wrapper path."""
    blob = _loreft_blob_bytes(tmp_path)
    spec = {"kind": la.STACKED_KIND, "si_id": "si",
            "si_prefix_sha16": "0" * 16,
            "object": {"kind": "loreft", "layers": [1], "rank": 2}}
    ops = la.get_adapter(la.STACKED_KIND)(spec, blob, b"")
    assert ops and all(op.phase == la.PHASE_PREFILL for op in ops)


# ---------------------------------------------------------------------------
# THE NAMED CARVE-OUTS: a resolved spec
# carrying ``serve_experiment`` (hashed, visible in the cell name as
# ``@all_positions`` / ``@all_turns``) names exactly one (recorded fit,
# served positions) pair; the guard admits the pair BY NAME and refuses a
# name the deployment does not match. The unnamed cases are pinned above,
# verbatim, and must not move.
# ---------------------------------------------------------------------------

from sisteer.capeval.serving import serve_declaration as sd  # noqa: E402

TURN_END = (7, 9)


def _response_fit_blob():
    return _psr_blob_bytes({"positions": "response", "response_header_len": len(HDR)})


def test_named_all_positions_carve_out_passes_the_collapse_corner():
    """The headline: fit-response / serve-all — refused for every unnamed
    cell (test_response_fit_with_spec_override_to_all_refuses, kept above) —
    PASSES under the hashed name, and the ops serve `all`."""
    spec = sd.apply_variant({"kind": "psr", "positions": "response"}, "all_positions")
    assert spec[sd.SERVE_EXPERIMENT_KEY] == sd.EXPERIMENT_FIT_RESPONSE_SERVE_ALL_POSITIONS
    ops = la.get_adapter("psr")(spec, _response_fit_blob(), b"", header_ids=HDR)
    assert {o.positions for o in ops} == {la.POSITIONS_ALL}
    assert la._check_fit_serve_positions("psr", spec, _response_fit_blob(), b"", ops) == "response"


def test_named_all_turns_carve_out_passes_and_the_unnamed_case_is_unchanged():
    spec = sd.apply_variant({"kind": "psr", "positions": "response"}, "all_turns")
    ops = la.get_adapter("psr")(spec, _response_fit_blob(), b"", header_ids=HDR,
                                turn_end_ids=TURN_END)
    assert {o.positions for o in ops} == {la.POSITIONS_RESPONSE_ALL_TURNS}
    # the UNNAMED all-turns cell keeps passing exactly as at d1eb1064
    unnamed = la.get_adapter("psr")({"kind": "psr", "positions": "response_all_turns"},
                                    _response_fit_blob(), b"", header_ids=HDR,
                                    turn_end_ids=TURN_END)
    assert {o.positions for o in unnamed} == {la.POSITIONS_RESPONSE_ALL_TURNS}


def test_unnamed_response_fit_served_all_still_refuses_and_names_the_carve_out():
    """The refusal text now says HOW a deliberate transfer cell is admitted."""
    with pytest.raises(la.AdapterError, match="DECODE-ONLY") as ei:
        la.get_adapter("psr")({"kind": "psr", "positions": "all"}, _response_fit_blob(), b"")
    assert sd.EXPERIMENT_FIT_RESPONSE_SERVE_ALL_POSITIONS in str(ei.value)
    assert "@all_positions" in str(ei.value)


@pytest.mark.parametrize("blob_positions,variant,served", [
    ("all", "all_positions", la.POSITIONS_ALL),           # a name on an ALL-fit
    ("all", "all_turns", la.POSITIONS_RESPONSE_ALL_TURNS),
    (None, "all_positions", la.POSITIONS_ALL),            # a name on an ABSENT record
    (None, "all_turns", la.POSITIONS_RESPONSE_ALL_TURNS),
])
def test_a_name_whose_fit_does_not_match_refuses(blob_positions, variant, served):
    extra = {} if blob_positions is None else {"positions": blob_positions}
    if blob_positions == "all":
        extra["response_header_len"] = len(HDR)
    blob = _psr_blob_bytes(extra)
    spec = sd.apply_variant({"kind": "psr"}, variant)
    with pytest.raises(la.AdapterError):
        la.get_adapter("psr")(spec, blob, b"", header_ids=HDR, turn_end_ids=TURN_END)


def test_a_name_whose_served_ops_do_not_match_refuses():
    """The guard is evaluated over the OPS, so a name whose pair the ops do
    not express refuses even though the fit record matches."""
    blob = _response_fit_blob()
    resp_ops = la.get_adapter("psr")({"kind": "psr", "positions": "response"}, blob, b"",
                                     header_ids=HDR)
    for variant in ("all_positions", "all_turns"):
        spec = sd.apply_variant({"kind": "psr"}, variant)
        with pytest.raises(la.AdapterError, match="must describe the deployment"):
            la._check_fit_serve_positions("psr", spec, blob, b"", resp_ops)


def test_an_unknown_or_malformed_experiment_name_refuses():
    blob = _response_fit_blob()
    for bad in ({"kind": "psr", "positions": "all", sd.SERVE_EXPERIMENT_KEY: "fit_all_serve_everything"},
                {"kind": "psr", "positions": "all", sd.SERVE_EXPERIMENT_KEY: 3},
                {"kind": "psr", "positions": "all", sd.SERVE_EXPERIMENT_KEY: sd.EXPERIMENT_FIT_RESPONSE_SERVE_ALL_POSITIONS,
                 sd.SERVE_OUT_OF_SUPPORT_KEY: ["thinking"]}):
        with pytest.raises(la.AdapterError, match="malformed serve declaration"):
            la.get_adapter("psr")(bad, blob, b"", header_ids=HDR)


def test_the_guards_pairs_are_serve_declarations_and_the_literals_agree():
    """The torch-free module pins the position literals by hand; the guard
    reads them — they must be lens_adapters' own."""
    assert sd.EXPERIMENT_FIT_SERVE_PAIRS == {
        sd.EXPERIMENT_FIT_RESPONSE_SERVE_ALL_TURNS: (la.POSITIONS_RESPONSE, la.POSITIONS_RESPONSE_ALL_TURNS),
        sd.EXPERIMENT_FIT_RESPONSE_SERVE_ALL_POSITIONS: (la.POSITIONS_RESPONSE, la.POSITIONS_ALL)}


def _registry_with_named_entry(tmp_path, variant, header_len=len(HDR)):
    blob = _psr_blob_bytes({"positions": "response", "response_header_len": header_len})
    reg = tmp_path / "reg"
    (reg / "arm").mkdir(parents=True)
    (reg / "arm" / "artifact.bin").write_bytes(blob)
    resolved = sd.apply_variant({"kind": "psr", "coeff": 1.0, "positions": "response"}, variant)
    ih = cm.petri_intervention_hash(resolved, artifact_bytes=blob, sidecar_bytes=b"")
    entry = {"cell_id": f"arm__c1@{variant}", "kind": "psr", "resolved_spec": resolved,
             "expect_hash": ih, "artifact": "arm/artifact.bin", "sidecar": None,
             "artifact_sha16": cm.sha256_hex16(blob), "artifact_intervention_hash": ih,
             "forced_scale": None, "donor_cell": None, "validated_by": None}
    index = {"version": 1, "assistant_header_ids": list(HDR), "entries": [entry]}
    if variant == "all_turns":
        index["assistant_turn_end_ids"] = list(TURN_END)
    (reg / ls.REGISTRY_INDEX).write_text(json.dumps(index))
    return reg


@pytest.mark.parametrize("variant,served", [("all_positions", la.POSITIONS_ALL),
                                            ("all_turns", la.POSITIONS_RESPONSE_ALL_TURNS)])
def test_worker_registry_load_honours_the_named_carve_out(tmp_path, variant, served):
    """The load path: the same guard (it rides get_adapter), then the ops are
    bound to the declaration and the declaration is what the registry reports."""
    loaded = ls.WorkerObjectRegistry.load(_registry_with_named_entry(tmp_path, variant))
    (obj,) = loaded.objects()
    assert {o.positions for o in obj.ops} == {served}
    assert loaded.serve_declarations_by_cell() == {
        f"arm__c1@{variant}": sd.variant_declarations(variant)}
