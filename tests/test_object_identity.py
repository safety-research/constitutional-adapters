"""``sisteer.capeval.tinycap.object_identity`` — the HF POST-LOAD intervention
hash of a fitted object, derived on CPU from a weightless meta shell.

CPU-only and weightless by design, so this file monkeypatches the two heavy
seams (the tokenizer/shell construction and the per-object hydration) and pins
what the module OWNS: the config the derivation runs against, the request →
report shape, the two verdicts (`match`, `header_guard_will_pass`), the
one-bad-object-does-not-hide-the-others rule, and the CLI's exit code.

The end-to-end proof needs the real objects, which live on the Modal volume
and — for the two §PW21-stage staged — under an off-repo local root. The
real-bytes tests at the end run when `SISTEER_TINYCAP_STAGED_OBJECTS` names
that root (`<root>/<bank name>/<cell>/object.pt`) and the pinned tokenizers
are cached, and skip otherwise: llama3b `safety_refusal__psr_kl_all` →
`93b6466dd791cc7e` (H=4) and the no-refusals gemma cell → `35bc0822bfdc7dba`
(H=3), each equal to §PW21-stage's record (§PW20b-L-4). The "lens static"
values those records carried beside them (`2f22ff02fb018478` /
`b5f0de15b01b6e3e`) were `runtime.cell_intervention_hash`, which nothing on
the lens ROUTE stamps (E40 M4); the route's own hash is pinned here against
`RouteEntry.intervention_hash` enumerated through the standup's chain.
"""

from __future__ import annotations

import dataclasses
import io
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from sisteer.capeval.serving import serve_declaration as sd
from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import object_identity as oi

REPO = Path(__file__).resolve().parents[1]
#: the off-repo root the two §PW21-stage objects are staged under (a copy of the
#: volume's `/vol/tinycap/objects/`), for the real-bytes tests; unset -> skip
STAGED_ENV = "SISTEER_TINYCAP_STAGED_OBJECTS"


def test_load_config_forces_the_served_pins_and_applies_the_overlay(tmp_path):
    src = tmp_path / "cfg.yaml"
    src.write_text(yaml.safe_dump({"model": "m", "model_revision": None, "layer": None,
                                   "scale": 3.0, "si_ids": ["a"]}))
    cfg = oi.load_config(src, {"layer": 15, "model_revision": "r"})
    assert cfg["layer"] == 15 and cfg["model_revision"] == "r"
    # the pins every published number was produced at are FORCED, not defaulted:
    # a config that later grew a different scale would otherwise derive the hash
    # of a different intervention than the one being reproduced
    assert cfg["steer_positions"] == oi.SERVED_POSITIONS == "response"
    assert cfg["scale"] == oi.SERVED_SCALE == 1.0
    assert cfg["si_ids"] == ["a"]                      # nothing else moved
    # the checked-in file is never touched
    assert yaml.safe_load(src.read_text())["layer"] is None


def test_artifact_sha256_streams_the_file(tmp_path):
    import hashlib
    blob = b"x" * (3 << 20) + b"tail"
    f = tmp_path / "object.pt"
    f.write_bytes(blob)
    assert oi.artifact_sha256(f) == hashlib.sha256(blob).hexdigest()


def _patched(monkeypatch, *, header_len=3, derive):
    """The module with its heavy seams replaced: no tokenizer, no shell, no
    torch, and no registry over the (fake) bank for the route hash."""
    tok = SimpleNamespace(name="tok")
    monkeypatch.setattr(oi, "build_meta_shell",
                        lambda model_id, revision: (SimpleNamespace(), "Fake", "fake"))
    monkeypatch.setattr(oi, "derive_one", derive)
    monkeypatch.setattr(oi, "route_intervention_hash", lambda **kw: {
        "harness_cell_id": f"h:{kw['cell_id']}", "lens_route_intervention_hash": "r" * 16,
        "lens_route_derivation": oi.ROUTE_DERIVATION})
    import sisteer.chat_markers as cm_mod
    import sisteer.models as models_mod
    monkeypatch.setattr(models_mod, "load_tokenizer", lambda *a, **k: tok)
    monkeypatch.setattr(models_mod, "pin_reasoning_template_off", lambda t: True)
    monkeypatch.setattr(models_mod, "serving_identity",
                        lambda t, cfg, model=None: {"chat_template_hash": "cafecafecafecafe"})
    monkeypatch.setattr(cm_mod, "generation_header_ids", lambda t: list(range(header_len)))
    return tok


def _request(tmp_path, targets):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"model": "m", "model_revision": None, "layer": None}))
    return {"model": "org/model", "revision": "r" * 40, "config": str(cfg),
            "config_overrides": {"layer": 15, "model_revision": "r" * 40},
            "objects_dir": str(tmp_path / "bank"), "targets": targets}


def test_derive_request_reports_the_match_and_the_header_guard(tmp_path, monkeypatch):
    def derive(*, artifact, cell_id, si_id, arm, cfg, shell, tok, coeff=1.0):
        # the config the derivation runs against carries the overlay AND the pins
        assert cfg["layer"] == 15 and cfg["model_revision"] == "r" * 40
        assert cfg["steer_positions"] == "response" and cfg["scale"] == 1.0
        assert cfg["model"] == "org/model"
        return {"cell": cell_id, "intervention_hash": {"a": "1111111111111111",
                                                       "b": "9999999999999999"}[cell_id],
                "resolved_spec": {"positions": "response"},
                "response_header_len": 3 if cell_id == "a" else 7,
                "positions": "response", "artifact": str(artifact),
                "artifact_sha256": "s" * 64}
    _patched(monkeypatch, header_len=3, derive=derive)
    rep = oi.derive_request(_request(tmp_path, [
        {"cell": "a", "si_id": "si", "arm": "psr_kl_all", "expected": "1111111111111111"},
        {"cell": "b", "si_id": "si", "arm": "psr_kl_all", "expected": "1111111111111111"},
    ]))
    assert rep["generation_header_len"] == 3
    assert rep["chat_template_hash"] == "cafecafecafecafe"
    assert rep["derivation"] == oi.DERIVATION
    a, b = rep["results"]
    assert a["match"] is True and a["header_guard_will_pass"] is True
    # a recorded header that AGREES is checked, and says so
    assert a["header_guard"] == oi.HEADER_GUARD_CHECKED
    # the route hash rides beside the HF one, with the harness id it is filed under
    assert a["lens_route_intervention_hash"] == "r" * 16 and a["harness_cell_id"] == "h:a"
    assert a["lens_route_derivation"] == oi.ROUTE_DERIVATION and "lens_static_intervention_hash" not in a
    # b is BOTH a different object and a header the guard would reject
    assert b["match"] is False and b["header_guard_will_pass"] is False
    assert b["header_guard"] == oi.HEADER_GUARD_MISMATCH
    assert b["expected"] == "1111111111111111" != b["intervention_hash"]
    assert rep["ok"] is False


def test_a_header_less_object_predicts_pass_with_the_reason(tmp_path, monkeypatch):
    """O22: every LoRA blob records NO ``response_header_len`` — it is a native
    adapter, not a header-masked lens object, and it grows neither deployment
    attribute (``jailbreak._set_response_positions`` takes its
    ``set_fit_positions`` branch, which stores the width on
    ``fit_response_header_len`` instead). The three real guards all key on
    whether a header IS recorded, so they do not refuse it; predicting a
    refusal failed every ``*__lora_r4`` cell of the 10-cell Gemma waves at
    preflight. It passes, and the report says WHY rather than passing
    silently."""
    def derive(*, artifact, cell_id, si_id, arm, cfg, shell, tok, coeff=1.0):
        return {"cell": cell_id, "intervention_hash": "1111111111111111",
                "resolved_spec": {"positions": "response"},
                # the LoRA shape: neither attribute exists on the loaded object
                "response_header_len": None, "positions": None,
                "object_kind": "lora_all",
                "artifact": str(artifact), "artifact_sha256": "s" * 64}
    _patched(monkeypatch, header_len=3, derive=derive)
    rep = oi.derive_request(_request(tmp_path, [
        {"cell": "safety_refusal__lora_r4", "si_id": "safety_refusal", "arm": "lora_r4",
         "expected": "1111111111111111"}]))
    (rec,) = rep["results"]
    assert rec["header_guard_will_pass"] is True
    assert rec["header_guard"] == oi.HEADER_GUARD_NOT_APPLICABLE == \
        "not_applicable_no_recorded_header"
    assert rec["match"] is True and rep["ok"] is True       # the wave is not refused
    # a derivation that RAISED never claims the guard was evaluated
    _patched(monkeypatch, header_len=3,
             derive=lambda **kw: (_ for _ in ()).throw(RuntimeError("blob is truncated")))
    (bad,) = oi.derive_request(_request(tmp_path, [
        {"cell": "safety_refusal__lora_r4", "si_id": "safety_refusal",
         "arm": "lora_r4", "expected": "1111111111111111"}]))["results"]
    assert bad["header_guard_will_pass"] is False
    assert bad["header_guard"] == oi.HEADER_GUARD_NOT_EVALUATED


#: an invented generation-prompt header and turn-end scaffold for the guard
#: matrix below — three and two token ids, the widths the synthetic fit legs
#: are written against (gemma-4 derives 3; Llama-3.1, 4)
_GUARD_HDR = (7, 8, 9)
_GUARD_TURN_END = (99, 11)
#: hidden width of the synthetic psr tensors; nothing here runs a model
_GUARD_D = 8


def _psr_bytes(**fit_leg):
    """A synthetic psr blob — the kind a tinycap standup serves through the
    lens — carrying the fit leg under test: ``positions`` and, when the fit
    recorded one, ``response_header_len``. Invented tensors, seeded."""
    import torch

    g = torch.Generator().manual_seed(9)
    blob = {"kind": "psr", "layers": [0, 2],
            "directions": {str(l): torch.randn(_GUARD_D, generator=g) * 0.05
                           for l in (0, 2)},
            "gate_w": {str(l): torch.randn(_GUARD_D, generator=g) * 0.3
                       for l in (0, 2)},
            "gate_b": {str(l): torch.randn(1, generator=g) * 0.5
                       for l in (0, 2)}}
    blob.update(fit_leg)
    buf = io.BytesIO()
    torch.save(blob, buf)
    return buf.getvalue()


def _lora_bytes(**extra):
    """A synthetic all-position rank-1 lora blob, the shape the PEFT export
    leg takes. ``RankOneLoRAObject.save`` in this tree writes no
    ``response_header_len`` at all — which is the O22 fact below."""
    import torch

    g = torch.Generator().manual_seed(4)
    blob = {"kind": "lora", "layer": 0, "positions": "all",
            "A": torch.randn(_GUARD_D, generator=g), "B": torch.randn(_GUARD_D, generator=g)}
    blob.update(extra)
    buf = io.BytesIO()
    torch.save(blob, buf)
    return buf.getvalue()


def test_header_guard_prediction_mirrors_the_real_guards(tmp_path):
    """The prediction's rule, and the guards it mirrors — checked by RUNNING
    them on synthetic bytes, never by grepping their source for a line.

    The rule: a recorded header is compared EXACTLY (a mismatch refuses); no
    record means there is nothing to compare and the guard does not fire, so
    the verdict is keyed on whether the object RECORDS a header, never on its
    kind.

    Why this test executes rather than pins. It used to assert the lens leg's
    condition as a source literal (``mode == POSITIONS_RESPONSE and fit_h !=
    len(header)``), which pins PRESENCE, not BEHAVIOUR. The all-turns landing
    widened that condition to ``mode in RESPONSE_MASK_MODES`` — the same
    comparison over one more served mode, no verdict changed anywhere — and
    the grep went red on a guard that still mirrors the prediction exactly.
    A pin that cannot tell a rewrite from a regression is not evidence; every
    claim below is raise-vs-return from the real guard.

    The case set, after the lens half's named carve-outs landed (section 8):
    UNDECLARED cells keep every verdict above (sections 2-7, byte for byte);
    a cell that NAMES ``serve_experiment: fit_response_serve_all_turns``
    serves the response mask over every assistant turn and the width compare
    runs there too (verdicts as at ``response``); a cell that names
    ``fit_response_serve_all_positions`` serves at ``all``, where there is no
    header mask to compare, so the lens leg never fires and a recorded header
    passes BY NAME (``HEADER_GUARD_NOT_COMPARED_ALL_POSITIONS``); a name whose
    (fit, served) pair the deployment does not match, or a name outside the
    closed vocabulary, is refused by the fit/serve REGIME guard whatever the
    header says. Prediction == guard, case by case, on the real adapt path.
    """
    pytest.importorskip("torch")
    import torch

    from sisteer.capeval.serving import lens_adapters as la
    from sisteer.capeval.serving import lora_export as lx
    from sisteer.eval import jailbreak as jb

    # ---- 1. the prediction's own rule -------------------------------------
    assert oi.header_guard_prediction(None, 3) == (True, oi.HEADER_GUARD_NOT_APPLICABLE)
    assert oi.header_guard_prediction(3, 3) == (True, oi.HEADER_GUARD_CHECKED)
    assert oi.header_guard_prediction(7, 3) == (False, oi.HEADER_GUARD_MISMATCH)
    assert oi.header_guard_prediction(4, 4) == (True, oi.HEADER_GUARD_CHECKED)

    def _adapt(kind, blob, serve):
        """One cell through the REAL serving path: ``get_adapter`` is what a
        worker calls and ``register_adapter`` wrapped it in the fit/serve
        regime guard, so this runs both legs exactly as serving runs them."""
        try:
            ops = la.get_adapter(kind)({"kind": kind, "positions": serve}, blob,
                                       b"", header_ids=_GUARD_HDR,
                                       turn_end_ids=_GUARD_TURN_END)
            return True, ops, ""
        except la.AdapterError as e:
            return False, (), str(e)

    # ---- 2. the lens serving leg, at the pin the prediction is written for
    # The prediction takes no served position mode because load_config FORCES
    # one: every target on this path resolves to a response-masked spec.
    cfg_src = tmp_path / "served_pin.yaml"
    cfg_src.write_text(yaml.safe_dump({"model": "m", "steer_positions": "all"}))
    served = oi.load_config(cfg_src, {})["steer_positions"]
    assert served == oi.SERVED_POSITIONS == la.POSITIONS_RESPONSE
    # the modes under which the lens leg compares the recorded width — the
    # prediction hand-pins them (object_identity is torch-free); they are
    # lens_adapters' own, and the forced pin is one of them
    assert oi.HEADER_MASKED_SERVES == la.RESPONSE_MASK_MODES
    assert oi.SERVED_POSITIONS in oi.HEADER_MASKED_SERVES
    # FIT_POSITION_MODES is the whole set of fit legs, so this is exhaustive
    assert sorted(la.FIT_POSITION_MODES) == ["all", "response"]
    for fit in la.FIT_POSITION_MODES:
        for recorded, reason in ((None, oi.HEADER_GUARD_NOT_APPLICABLE),
                                 (len(_GUARD_HDR), oi.HEADER_GUARD_CHECKED),
                                 (len(_GUARD_HDR) + 4, oi.HEADER_GUARD_MISMATCH)):
            leg = {"positions": fit}
            if recorded is not None:
                leg["response_header_len"] = recorded
            will_pass, predicted = oi.header_guard_prediction(recorded, len(_GUARD_HDR))
            assert predicted == reason
            ok, ops, msg = _adapt("psr", _psr_bytes(**leg), served)
            assert ok is will_pass, (fit, recorded, msg)
            if ok:
                assert {o.positions for o in ops} == {la.POSITIONS_RESPONSE}
                assert all(o.header_ids == _GUARD_HDR for o in ops)
            else:
                # and it refuses for the HEADER's reason — a cell that
                # refused for some other guard's reason would agree on the
                # verdict while mirroring nothing
                assert f"the fit recorded a {recorded}-token" in msg
                assert "DIFFERENT regions" in msg

    # ---- 3. the all-turns carve-out extends the mirror ---------------------
    # What the deleted grep could not see: a "response" fit is also servable
    # at "response_all_turns", and the width compare runs there too, so the
    # prediction is right for that serve as well.
    assert la.RESPONSE_MASK_MODES == (la.POSITIONS_RESPONSE,
                                      la.POSITIONS_RESPONSE_ALL_TURNS)
    for recorded in (None, len(_GUARD_HDR), len(_GUARD_HDR) + 4):
        leg = {"positions": la.POSITIONS_RESPONSE}
        if recorded is not None:
            leg["response_header_len"] = recorded
        will_pass, _ = oi.header_guard_prediction(recorded, len(_GUARD_HDR))
        ok, ops, msg = _adapt("psr", _psr_bytes(**leg),
                              la.POSITIONS_RESPONSE_ALL_TURNS)
        assert ok is will_pass, (recorded, msg)
        if ok:
            assert {o.positions for o in ops} == {la.POSITIONS_RESPONSE_ALL_TURNS}
        else:
            assert "DIFFERENT regions" in msg

    # ---- 4. the fit/serve position-regime guard, called directly -----------
    # It is what makes section 3 reachable and section 5 unreachable, so it
    # is executed here rather than assumed: raise vs return on every cell.
    def _op(mode):
        return la.HookOp(
            op=la.OP_ADD_VECTOR, layer=0, vector=torch.ones(_GUARD_D), positions=mode,
            header_ids=_GUARD_HDR if mode in la.RESPONSE_MASK_MODES else None,
            turn_end_ids=(_GUARD_TURN_END if mode == la.POSITIONS_RESPONSE_ALL_TURNS
                          else None))

    def _regime(fit, serve):
        try:
            return la._check_fit_serve_positions(
                "psr", {"kind": "psr", "positions": serve},
                _psr_bytes(positions=fit), b"", (_op(serve),))
        except la.AdapterError as e:
            return e

    for fit in la.FIT_POSITION_MODES:                 # both fits serve the mask
        assert _regime(fit, la.POSITIONS_RESPONSE) == fit
    assert _regime(la.POSITIONS_ALL, la.POSITIONS_ALL) == la.POSITIONS_ALL
    assert _regime(la.POSITIONS_RESPONSE,             # THE carve-out
                   la.POSITIONS_RESPONSE_ALL_TURNS) == la.POSITIONS_RESPONSE
    all_at_turns = _regime(la.POSITIONS_ALL, la.POSITIONS_RESPONSE_ALL_TURNS)
    assert isinstance(all_at_turns, la.AdapterError)
    assert "KNOWN 'response' fit only" in str(all_at_turns)
    resp_at_all = _regime(la.POSITIONS_RESPONSE, la.POSITIONS_ALL)
    assert isinstance(resp_at_all, la.AdapterError)
    assert "DECODE-ONLY" in str(resp_at_all)
    # ...and the refusal names the ONE door: the all-positions experiment
    assert sd.EXPERIMENT_FIT_RESPONSE_SERVE_ALL_POSITIONS in str(resp_at_all)

    # the NAMED branch: a spec carrying serve_experiment describes exactly one
    # (recorded fit, served positions) pair and is admitted iff the deployment
    # IS that pair — raise vs return, every cell of it
    def _named_regime(fit, name, serve):
        spec = {"kind": "psr", "positions": serve, sd.SERVE_EXPERIMENT_KEY: name}
        leg = {} if fit is None else {"positions": fit}
        try:
            return la._check_fit_serve_positions("psr", spec, _psr_bytes(**leg),
                                                 b"", (_op(serve),))
        except la.AdapterError as e:
            return e

    ALL_POS, ALL_TURNS = (sd.EXPERIMENT_FIT_RESPONSE_SERVE_ALL_POSITIONS,
                          sd.EXPERIMENT_FIT_RESPONSE_SERVE_ALL_TURNS)
    assert sd.EXPERIMENT_FIT_SERVE_PAIRS == {
        ALL_TURNS: (la.POSITIONS_RESPONSE, la.POSITIONS_RESPONSE_ALL_TURNS),
        ALL_POS: (la.POSITIONS_RESPONSE, la.POSITIONS_ALL)}
    # the two admits: the collapse corner BY NAME, and the all-turns pair by name
    assert _named_regime(la.POSITIONS_RESPONSE, ALL_POS, la.POSITIONS_ALL) == la.POSITIONS_RESPONSE
    assert _named_regime(la.POSITIONS_RESPONSE, ALL_TURNS, la.POSITIONS_RESPONSE_ALL_TURNS) \
        == la.POSITIONS_RESPONSE
    # a name whose SERVED positions the ops do not express refuses, both ways round
    for name, serve in ((ALL_TURNS, la.POSITIONS_ALL), (ALL_POS, la.POSITIONS_RESPONSE),
                        (ALL_POS, la.POSITIONS_RESPONSE_ALL_TURNS)):
        err = _named_regime(la.POSITIONS_RESPONSE, name, serve)
        assert isinstance(err, la.AdapterError), (name, serve)
        assert "must describe the deployment" in str(err), (name, serve)
    # a name whose FIT the artifact does not record refuses: an all-fit, and an
    # ABSENT record (the class-documented all-position fit) — a positive
    # "response" record is required, exactly as for the unnamed all-turns serve
    for fit in (la.POSITIONS_ALL, None):
        for name, serve in ((ALL_POS, la.POSITIONS_ALL),
                            (ALL_TURNS, la.POSITIONS_RESPONSE_ALL_TURNS)):
            err = _named_regime(fit, name, serve)
            assert isinstance(err, la.AdapterError), (fit, name, serve)
    # a name outside the closed vocabulary is a malformed declaration
    bogus = _named_regime(la.POSITIONS_RESPONSE, "fit_all_serve_everything", la.POSITIONS_ALL)
    assert isinstance(bogus, la.AdapterError) and "malformed serve declaration" in str(bogus)

    # ---- 5. the two divergences, both DOCUMENTED and both accounted for ----
    # (a) served all-position there is no id-located mask to compare against,
    # so the header leg never fires and a stale record PASSES the real guard
    # while the PINNED prediction refuses. On the pinned path the column is
    # foreclosed (section 2: load_config forces the served pin to "response");
    # the one way onto it is a cell that NAMES the all-positions experiment,
    # and there the prediction takes the name and agrees — section 8.
    stale = len(_GUARD_HDR) + 4
    assert oi.header_guard_prediction(stale, len(_GUARD_HDR)) == \
        (False, oi.HEADER_GUARD_MISMATCH)
    ok, _, msg = _adapt("psr", _psr_bytes(positions=la.POSITIONS_ALL,
                                          response_header_len=stale),
                        la.POSITIONS_ALL)
    assert ok is True, msg
    # (b) PW-oos: --allow-out-of-support-thinking turns the PREDICTED refusal
    # into a report, because the mismatch is then the finding the run exists
    # to record. It moves the plan, never the guard — the serve still refuses.
    assert oi.header_guard_prediction(stale, len(_GUARD_HDR), allow_out_of_support=True) \
        == (True, oi.HEADER_GUARD_OVERRIDDEN)
    assert oi.header_guard_prediction(len(_GUARD_HDR), len(_GUARD_HDR),
                                      allow_out_of_support=True) \
        == (True, oi.HEADER_GUARD_CHECKED)            # never turns a pass over
    refused, _, _ = _adapt("psr", _psr_bytes(positions=la.POSITIONS_RESPONSE,
                                             response_header_len=stale), served)
    assert refused is False

    # ---- 6. the fit-mask leg (eval.jailbreak._check_recorded_header) -------
    # Runs INSIDE derive_one, off the loaded object's attributes rather than a
    # blob, and keys on the same record: None is not a record and passes.
    for attr in ("fit_response_header_len", "response_header_len"):
        for recorded in (None, len(_GUARD_HDR), len(_GUARD_HDR) + 4):
            will_pass, _ = oi.header_guard_prediction(recorded, len(_GUARD_HDR))
            try:
                jb._check_recorded_header(SimpleNamespace(**{attr: recorded}),
                                          width=len(_GUARD_HDR))
                ok, msg = True, ""
            except ValueError as e:
                ok, msg = False, str(e)
            assert ok is will_pass, (attr, recorded, msg)
            if not ok:
                assert f"{attr}={recorded}" in msg

    # ---- 7. the PEFT export leg -------------------------------------------
    # Validated when present, absent is not an error, and never compared
    # against a served width — there is no mask on a native adapter to
    # compare one to. That is why predicting a refusal on a LoRA blob (they
    # record no header at all) failed every ``*__lora_r4`` cell of the
    # 10-cell Gemma waves at preflight (O22, 2026-09-06).
    pytest.importorskip("safetensors")
    export = {"base_model": "meta-llama/Llama-3.1-8B-Instruct",
              "decoder_prefix": lx.LLAMA_DECODER_PREFIX}
    spec = {"kind": "lora", "layer": 0, "positions": "all",
            "intervention_hash": "ab" * 8}
    for i, blob in enumerate((_lora_bytes(),                       # the O22 shape
                              _lora_bytes(response_header_len=len(_GUARD_HDR)),
                              _lora_bytes(response_header_len=len(_GUARD_HDR) + 4))):
        assert lx.export_genuine_adapter(spec, blob, tmp_path / f"ad{i}",
                                         **export).is_dir()
    with pytest.raises(lx.LoraExportError, match="response_header_len"):
        lx.export_genuine_adapter(spec, _lora_bytes(response_header_len=-1),
                                  tmp_path / "bad", **export)

    # ---- 8. the NAMED carve-outs: prediction == guard, case by case ---------
    # The lens half's serve_declaration.apply_variant writes the cell's spec
    # (served positions + the name) — the exact spec a standup adapts — so
    # this runs the real adapt path under each name over the same three
    # records as section 2: no record, the served width, a stale width.
    def _adapt_named(variant, blob):
        spec = sd.apply_variant({"kind": "psr", "positions": la.POSITIONS_RESPONSE}, variant)
        assert spec[sd.SERVE_EXPERIMENT_KEY] in sd.SERVE_EXPERIMENTS
        try:
            ops = la.get_adapter("psr")(spec, blob, b"", header_ids=_GUARD_HDR,
                                        turn_end_ids=_GUARD_TURN_END)
            return True, ops, "", spec[sd.SERVE_EXPERIMENT_KEY]
        except la.AdapterError as e:
            return False, (), str(e), spec[sd.SERVE_EXPERIMENT_KEY]

    records = (None, len(_GUARD_HDR), stale)
    # (i) fit-response served at ALL, admitted ONLY under its name: the mask
    # is not header-located, so the lens leg never fires — every record
    # passes, and the prediction says so BY NAME rather than "checked"
    for recorded in records:
        leg = {"positions": la.POSITIONS_RESPONSE}
        if recorded is not None:
            leg["response_header_len"] = recorded
        ok, ops, msg, name = _adapt_named(sd.VARIANT_ALL_POSITIONS, _psr_bytes(**leg))
        assert name == sd.EXPERIMENT_FIT_RESPONSE_SERVE_ALL_POSITIONS
        assert ok is True, (recorded, msg)
        assert {o.positions for o in ops} == {la.POSITIONS_ALL}
        will_pass, reason = oi.header_guard_prediction(recorded, len(_GUARD_HDR),
                                                       serve_experiment=name)
        assert will_pass is ok
        assert reason == (oi.HEADER_GUARD_NOT_APPLICABLE if recorded is None
                          else oi.HEADER_GUARD_NOT_COMPARED_ALL_POSITIONS), recorded
        # the override never changes a pass, here as everywhere
        assert oi.header_guard_prediction(recorded, len(_GUARD_HDR), serve_experiment=name,
                                          allow_out_of_support=True) == (will_pass, reason)
    # (ii) fit-response served at response_all_turns under its name: the same
    # width comparison over every assistant-turn span — the pinned verdicts
    for recorded, want in ((None, oi.HEADER_GUARD_NOT_APPLICABLE),
                           (len(_GUARD_HDR), oi.HEADER_GUARD_CHECKED),
                           (stale, oi.HEADER_GUARD_MISMATCH)):
        leg = {"positions": la.POSITIONS_RESPONSE}
        if recorded is not None:
            leg["response_header_len"] = recorded
        ok, ops, msg, name = _adapt_named(sd.VARIANT_ALL_TURNS, _psr_bytes(**leg))
        assert name == sd.EXPERIMENT_FIT_RESPONSE_SERVE_ALL_TURNS
        will_pass, reason = oi.header_guard_prediction(recorded, len(_GUARD_HDR),
                                                       serve_experiment=name)
        assert ok is will_pass, (recorded, msg)
        assert reason == want
        if ok:
            assert {o.positions for o in ops} == {la.POSITIONS_RESPONSE_ALL_TURNS}
            assert all(o.header_ids == _GUARD_HDR for o in ops)
        else:
            assert f"the fit recorded a {recorded}-token" in msg and "DIFFERENT regions" in msg
        # the named verdict IS the pinned one: the name adds validation and
        # attestation to this pair, never behaviour
        assert (will_pass, reason) == oi.header_guard_prediction(recorded, len(_GUARD_HDR))
        assert oi.header_guard_prediction(stale, len(_GUARD_HDR), serve_experiment=name,
                                          allow_out_of_support=True) \
            == (True, oi.HEADER_GUARD_OVERRIDDEN)
    # (iii) a name the deployment does not match refuses at adapt WHATEVER the
    # header says — the regime guard's refusal, not the header leg's — so a
    # header-pass prediction is never a "will serve" on such a cell
    for recorded in records:
        leg = {"positions": la.POSITIONS_RESPONSE}
        if recorded is not None:
            leg["response_header_len"] = recorded
        wrong = {"kind": "psr", "positions": la.POSITIONS_ALL,
                 sd.SERVE_EXPERIMENT_KEY: sd.EXPERIMENT_FIT_RESPONSE_SERVE_ALL_TURNS}
        with pytest.raises(la.AdapterError, match="must describe the deployment") as ei:
            la.get_adapter("psr")(wrong, _psr_bytes(**leg), b"", header_ids=_GUARD_HDR,
                                  turn_end_ids=_GUARD_TURN_END)
        assert "DIFFERENT regions" not in str(ei.value)
    # a name on an all-fit or on an absent record: the pair's FIT is not met.
    # The regime guard refuses either way; at all_turns its all-turns rule
    # ("a KNOWN 'response' fit only") runs before the named branch, so the
    # reason is one of the two REGIME reasons — never the header leg's
    for leg in ({"positions": la.POSITIONS_ALL, "response_header_len": len(_GUARD_HDR)}, {}):
        for variant in (sd.VARIANT_ALL_POSITIONS, sd.VARIANT_ALL_TURNS):
            ok, _, msg, _ = _adapt_named(variant, _psr_bytes(**leg))
            assert ok is False, (leg, variant)
            assert ("must describe the deployment" in msg
                    or "KNOWN 'response' fit only" in msg), (leg, variant, msg)
            assert "DIFFERENT regions" not in msg
    # (iv) a name outside the closed vocabulary: the guard refuses it as a
    # malformed declaration and the prediction has no verdict to offer
    with pytest.raises(la.AdapterError, match="malformed serve declaration"):
        la.get_adapter("psr")({"kind": "psr", "positions": la.POSITIONS_ALL,
                               sd.SERVE_EXPERIMENT_KEY: "fit_all_serve_everything"},
                              _psr_bytes(positions=la.POSITIONS_RESPONSE), b"", header_ids=_GUARD_HDR)
    with pytest.raises(ValueError, match="not a named transfer experiment"):
        oi.header_guard_prediction(len(_GUARD_HDR), len(_GUARD_HDR),
                                   serve_experiment="fit_all_serve_everything")
    # (v) undeclared cells keep their verdicts: None is the pinned path, and
    # the pinned rule above (sections 1-2) was asserted with no name at all
    for recorded in records:
        assert oi.header_guard_prediction(recorded, len(_GUARD_HDR), serve_experiment=None) \
            == oi.header_guard_prediction(recorded, len(_GUARD_HDR))


def test_a_cell_with_no_expectation_is_derived_and_refuses(tmp_path, monkeypatch):
    """E40 H1 (inverting the test that codified the opposite): a fitted object
    with NO published expectation used to pass as "unverified" — a method
    outside EXPECTED_INTERVENTION_HASHES sailed through with expected=None. It
    is derived (so the hash can be pinned with --expect-hashes) and REFUSES."""
    _patched(monkeypatch, header_len=3, derive=lambda **kw: {
        "cell": kw["cell_id"], "intervention_hash": "abcdefabcdefabcd",
        "response_header_len": 3, "positions": "response"})
    rep = oi.derive_request(_request(tmp_path, [
        {"cell": "a", "si_id": "si", "arm": "psr_kl_all"}]))
    (rec,) = rep["results"]
    assert rec["expected"] is None and rec["match"] is False and rep["ok"] is False
    assert rec["intervention_hash"] == "abcdefabcdefabcd"   # reported, so it can be pinned
    assert "no expected HF post-load intervention hash" in rec["no_expectation"]
    assert "abcdefabcdefabcd" in rec["no_expectation"] and "--expect-hashes" in rec["no_expectation"]
    assert "error" not in rec                                # a refusal, not a crash
    # the CLI exit follows: a report that is not ok exits 1
    monkeypatch.setattr(oi, "derive_request", lambda req: rep)
    req = tmp_path / "req.json"
    req.write_text(json.dumps({"model": "org/model"}))
    assert oi.main(["--request", str(req)]) == 1


def test_one_exploding_object_does_not_hide_the_others(tmp_path, monkeypatch):
    def derive(**kw):
        if kw["cell_id"] == "boom":
            raise RuntimeError("blob is truncated")
        return {"cell": kw["cell_id"], "intervention_hash": "1111111111111111",
                "response_header_len": 3, "positions": "response"}
    _patched(monkeypatch, header_len=3, derive=derive)
    rep = oi.derive_request(_request(tmp_path, [
        {"cell": "boom", "si_id": "si", "arm": "psr_kl_all", "expected": "1111111111111111"},
        {"cell": "fine", "si_id": "si", "arm": "psr_kl_all", "expected": "1111111111111111"},
    ]))
    boom, fine = rep["results"]
    assert "RuntimeError: blob is truncated" in boom["error"] and boom["match"] is False
    assert fine["match"] is True                       # the sweep was not cut short
    assert rep["ok"] is False


def test_the_artifact_defaults_to_the_bank_layout(tmp_path, monkeypatch):
    """The artifact is resolved by the standup's OWN resolver
    (``defense_cell.object_path_for`` -> ``methods.object_artifact_name``):
    ``object.pt`` for a torch-blob arm, ``object.safetensors`` for antidote —
    E40 H2: a typed ``object.pt`` reported a valid antidote object missing. A
    bare Class B target carries no SI and is derived as class B."""
    pytest.importorskip("torch")
    seen = {}

    def derive(**kw):
        seen[kw["cell_id"]] = (str(kw["artifact"]), kw["si_id"])
        return {"cell": kw["cell_id"], "intervention_hash": "1" * 16,
                "response_header_len": 3, "positions": "response"}
    _patched(monkeypatch, header_len=3, derive=derive)
    rep = oi.derive_request(_request(tmp_path, [
        {"cell": "safety_refusal__psr_kl_all", "si_id": "safety_refusal", "arm": "psr_kl_all",
         "expected": "1" * 16},
        {"cell": "antidote", "arm": "antidote", "expected": "1" * 16},
        {"cell": "alphasteer", "si_id": None, "arm": "alphasteer", "expected": "1" * 16}]))
    bank = tmp_path / "bank"
    assert seen["safety_refusal__psr_kl_all"] == (str(bank / "safety_refusal__psr_kl_all" / "object.pt"),
                                                  "safety_refusal")
    assert seen["antidote"] == (str(bank / "antidote" / "object.safetensors"), None)
    assert seen["alphasteer"] == (str(bank / "alphasteer" / "object.pt"), None)
    assert [r["artifact"] for r in rep["results"]] == [a for a, _ in seen.values()]
    assert oi.bank_artifact_path(bank, "antidote") == bank / "antidote" / "object.safetensors"
    # the legacy lora_r1_all directory is honoured when it holds the artifact
    legacy = bank / "safety_refusal__lora_r1_all"
    legacy.mkdir(parents=True)
    (legacy / "object.pt").write_bytes(b"x")
    assert oi.bank_artifact_path(bank, "safety_refusal__lora_r4") == legacy / "object.pt"
    # a baseline has no artifact, and no bank is an error, both reported per target
    with pytest.raises(ValueError, match="baseline"):
        oi.bank_artifact_path(bank, "prompted__safety_refusal")
    with pytest.raises(ValueError, match="no objects_dir"):
        oi.bank_artifact_path("", "antidote")
    rep = oi.derive_request({**_request(tmp_path, [{"cell": "antidote", "arm": "antidote"}]),
                             "objects_dir": ""})
    (rec,) = rep["results"]
    assert "no objects_dir" in rec["error"] and rep["ok"] is False


def test_derive_one_builds_a_class_b_cell_without_an_si(monkeypatch):
    """A bare Class B target hands ``arm_spec_for_cell`` a class-B, SI-less
    cell (the shape the registry gives it), never the string 'None'."""
    import sisteer.eval.cell_controls as cc
    seen = {}

    # a baseline-shaped answer (no spec) stops the derivation before any hydration
    monkeypatch.setattr(cc, "arm_spec_for_cell", lambda cell, cfg: seen.setdefault("cell", cell) and None)
    with pytest.raises(ValueError, match="no ablation spec"):
        oi.derive_one(artifact="/x/object.safetensors", cell_id="antidote", si_id=None,
                      arm="antidote", cfg={"model": "m", "model_revision": "r"}, shell=None, tok=None)
    assert (seen["cell"].class_, seen["cell"].si_id, seen["cell"].arm) == ("B", None, "antidote")
    oi_a = {}
    monkeypatch.setattr(cc, "arm_spec_for_cell", lambda cell, cfg: oi_a.setdefault("cell", cell) and None)
    with pytest.raises(ValueError):
        oi.derive_one(artifact="/x/object.pt", cell_id="safety_refusal__psr_kl_all",
                      si_id="safety_refusal", arm="psr_kl_all",
                      cfg={"model": "m", "model_revision": "r"}, shell=None, tok=None)
    assert (oi_a["cell"].class_, oi_a["cell"].si_id) == ("A", "safety_refusal")


def test_the_cli_writes_the_report_and_exits_on_the_verdict(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(oi, "derive_request", lambda req: {"ok": True, "results": [],
                                                           "echo": req["model"]})
    req = tmp_path / "req.json"
    req.write_text(json.dumps({"model": "org/model"}))
    out = tmp_path / "report.json"
    assert oi.main(["--request", str(req), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["echo"] == "org/model"
    assert json.loads(capsys.readouterr().out)["echo"] == "org/model"
    monkeypatch.setattr(oi, "derive_request", lambda req: {"ok": False, "results": []})
    assert oi.main(["--request", str(req)]) == 1        # a mismatch is a non-zero exit
    capsys.readouterr()


def test_the_module_names_both_hashes_it_can_produce():
    """The whole point of the module is that these are different numbers; a
    reader must never have to guess which one a value is. E40 M4: the second
    one is the hash the lens ROUTE stamps (``PetriCell`` / ``RouteEntry.
    intervention_hash`` through ``petri_intervention_hash``), NOT
    ``runtime.cell_intervention_hash`` — which the module used to report as
    "the lens static hash" and now names only to say nothing stamps it."""
    assert "HFEngine._steering_arm" in oi.DERIVATION
    assert "minus obj.to_(model.device)" in oi.DERIVATION
    assert "petri_intervention_hash" in oi.ROUTE_DERIVATION
    assert "PetriCell.intervention_hash" in oi.ROUTE_DERIVATION
    assert "RouteEntry.intervention_hash" in oi.ROUTE_DERIVATION
    assert "cell_intervention_hash" not in oi.ROUTE_DERIVATION
    assert "not what the lens ROUTE stamps" in oi.RUNTIME_STATIC_HASH_NOTE
    assert not hasattr(oi, "LENS_DERIVATION")
    doc = (REPO / "src/sisteer/capeval/tinycap/object_identity.py").read_text()
    # the worked example names all THREE numbers for the one gemma object
    assert "d50f46c19733de40" in doc and "d6e1b51a96588745" in doc and "8d7131521d4e8f88" in doc
    assert "lens_static_intervention_hash" not in doc.split('"""', 2)[2]     # gone from the code
    src = (REPO / "src/sisteer/capeval/tinycap/object_identity.py").read_text()
    assert "rt.cell_intervention_hash" not in src


def test_derive_one_refuses_a_cell_with_no_object(monkeypatch):
    """A baseline resolves to no ablation spec; deriving "its" identity would
    be inventing one."""
    import sisteer.eval.cell_controls as cc
    monkeypatch.setattr(cc, "arm_spec_for_cell", lambda cell, cfg: None)
    with pytest.raises(ValueError, match="no ablation spec"):
        oi.derive_one(artifact="/x/object.pt", cell_id="none", si_id="si", arm="psr_kl_all",
                      cfg={"model": "m", "model_revision": "r"}, shell=None, tok=None)


# --------------------------------------------------------------------------- #
# real bytes: the two staged objects (skips unless SISTEER_TINYCAP_STAGED_OBJECTS) #
# --------------------------------------------------------------------------- #


def _staged_object(slug: str, cell: str):
    """The staged ``object.pt`` of ``cell`` under the registry row's bank
    name, or a skip when the root is not named / the object is not there."""
    root = os.environ.get(STAGED_ENV)
    if not root:
        pytest.skip(f"{STAGED_ENV} unset: the staged objects live off-repo")
    row = C.model_row(slug)
    path = Path(root).expanduser() / Path(row.objects_dir).name / cell / "object.pt"
    if not path.is_file():
        pytest.skip(f"{path} is not staged here")
    return row, path


def _route_through_the_standup(row, cell: str, cfg_path: Path, bank: Path):
    """``RouteEntry`` for ``cell`` the way ``capctrl_serving_up.bring_up``
    enumerates it: the row's petri manifest re-pinned to the row's target,
    ``traintest_cell_source`` over the bank, ``enumerate_cells``,
    ``routes_from_cells``."""
    from sisteer.capeval import manifest as cm
    from sisteer.capeval.petri import defense_cells as dc
    from sisteer.capeval.petri.registry_adapter import (
        harness_cell_id,
        traintest_cell_source,
    )
    from sisteer.capeval.serving.proxy import routes_from_cells
    from sisteer.capeval.tiers.petri_runner import resolve_prompt_files

    hid = harness_cell_id(cell)
    spec0 = cm.load_petri_spec(REPO / (row.petri_manifest or "configs/capeval/manifests/petri_twosi.yaml"))
    spec = dataclasses.replace(
        spec0, target_model=row.hf, target_revision=row.rev, serving_mode="interleaved",
        cells=["none", hid],
        serving_launch_flags=[f for f in spec0.serving_launch_flags if f != "--enable-lora"],
        scenario_bundle_hash="ab" * 8, serving_impl_commit="fedcba9876" * 4,
        chat_template_hash="cd" * 8)
    src = traintest_cell_source(spec, repo_root=REPO, config_path=cfg_path, defenses_dir=bank)
    prompts = (resolve_prompt_files(spec, override_root=REPO / "configs/capeval/petri/prompts")
               if spec.prompt_provenance.startswith("override@")
               else {"auditor": b"# a\n", "judge": b"# j\n"})
    cells = dc.enumerate_cells(
        spec, src, behavior_files={k: f"# {k}\n".encode() for k in cm.petri_expected_behavior_keys(spec)},
        prompt_files=prompts)
    assert not cells.absent, [a.cell_id for a in cells.absent]
    return hid, src, {r.cell_id: r for r in routes_from_cells(cells)}[hid]


@pytest.mark.parametrize("slug,cell,expected_hf", [
    ("llama3b", "safety_refusal__psr_kl_all", "93b6466dd791cc7e"),
    ("gemma4e4b-nors", "si_constitution__psr_kl_all", "35bc0822bfdc7dba"),
])
def test_the_reported_route_hash_is_what_the_route_stamps(slug, cell, expected_hf, tmp_path, monkeypatch):
    """E40 M4, on real bytes: the module's ``lens_route_intervention_hash`` for a
    staged object EQUALS ``RouteEntry.intervention_hash`` when the same bank is
    enumerated through the standup's own chain (``traintest_cell_source`` ->
    ``enumerate_cells`` -> ``routes_from_cells``); the HF hash still matches the
    published expectation; and the three numbers — HF post-load, route,
    ``runtime.cell_intervention_hash`` — are three DIFFERENT values, which is
    why the module no longer reports the third as "the lens static hash".
    ``route_cell_source`` is pinned against ``traintest_cell_source``'s entry."""
    pytest.importorskip("torch")
    row, staged = _staged_object(slug, cell)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer
    try:
        AutoTokenizer.from_pretrained(row.hf, revision=row.rev, local_files_only=True)
    except Exception as e:                                  # noqa: BLE001 — a cache probe, not a verdict
        pytest.skip(f"the pinned tokenizer is not cached: {type(e).__name__}")
    from sisteer.capeval import manifest as cm
    from sisteer.capeval import runtime as rt

    bank = tmp_path / "bank"
    (bank / cell).mkdir(parents=True)
    shutil.copy(staged, bank / cell / "object.pt")
    # the standup reads the OVERLAID config (overlaid_config writes this copy);
    # the derivation gets the checked-in config + the row's overrides, as the
    # launcher's identity_request hands them over
    doc = yaml.safe_load((REPO / row.cfg).read_text())
    doc.update(row.cfg_overrides)
    overlay = tmp_path / "config_overlay.yaml"
    overlay.write_text(yaml.safe_dump(doc, sort_keys=False))
    hid, src, route = _route_through_the_standup(row, cell, overlay, bank)
    assert route.serving_identity == "proxy_vllm+lens+interleaved"

    rep = oi.derive_request({
        "model": row.hf, "revision": row.rev, "config": str(REPO / row.cfg),
        "config_overrides": dict(row.cfg_overrides), "objects_dir": str(bank),
        "targets": [{"cell": cell, "si_id": C.parse_launcher_cell(cell).si, "arm": "psr_kl_all",
                     "expected": expected_hf}]})
    (rec,) = rep["results"]
    assert rep["ok"] is True and rec["match"] is True and "error" not in rec
    assert rec["intervention_hash"] == expected_hf                        # the published HF identity
    assert rec["harness_cell_id"] == hid
    assert rec["lens_route_intervention_hash"] == route.intervention_hash    # WHAT THE ROUTE STAMPS
    assert rec["lens_route_derivation"] == oi.ROUTE_DERIVATION
    assert "lens_static_intervention_hash" not in rec
    static = rt.cell_intervention_hash(cm.AblationSpec(**rec["ablation_spec_kwargs"]), 1.0)
    assert len({expected_hf, route.intervention_hash, static}) == 3        # three numbers, one file
    # the module's registry over the bank IS the standup's
    mine, theirs = oi.route_cell_source(oi.load_config(overlay), bank).get(hid), src.get(hid)
    assert (mine.kind, mine.resolved_spec, mine.sidecar_path) == (theirs.kind, theirs.resolved_spec,
                                                                  theirs.sidecar_path)
    assert Path(mine.object_path).resolve() == Path(theirs.object_path).resolve() == (bank / cell / "object.pt")
    # a bank without the object: the registry names the path but cannot resolve it —
    # refused loudly as the standup's ABSENT, never a silent None; and an artifact
    # the registry would not serve is refused as two resolvers disagreeing
    empty = tmp_path / "empty"
    with pytest.raises(ValueError, match="ABSENT"):
        oi.route_intervention_hash(cfg=oi.load_config(overlay), objects_dir=empty,
                                   cell_id=cell, artifact=empty / cell / "object.pt")
    with pytest.raises(ValueError, match="two resolvers disagree"):
        oi.route_intervention_hash(cfg=oi.load_config(overlay), objects_dir=bank,
                                   cell_id=cell, artifact=tmp_path / "elsewhere.pt")


# --------------------------------------------------------------------------- #
# PW-oos (U10 item 5) — the preflight half of the out-of-support override      #
# --------------------------------------------------------------------------- #

#: A fake Qwen3.5-shaped template: the assistant header the render APPENDS is
#: 7 tokens thinking-OFF (the empty <think></think> closed inside it) and 5
#: thinking-ON (left open) — the real widths, measured 2026-09-06 on
#: Qwen/Qwen3.5-4B @ 851bf6e8 (the 2026-09-09 thinking-header measurement).
_HDR = {"off": (248045, 74455, 198, 248068, 271, 248069, 271),
        "on": (248045, 74455, 198, 248068, 198)}


def _arm_tok(headers=_HDR):
    """A tokenizer whose ``apply_chat_template`` appends a DIFFERENT assistant
    header per arm, so both renders can be read off it the way the real one is
    (True minus False)."""
    body = (1, 2, 3)

    def apply(messages, add_generation_prompt=False, tokenize=True, **kw):
        arm = "on" if kw.get("enable_thinking") else "off"
        ids = list(body) + (list(headers[arm]) if add_generation_prompt else [])
        return {"input_ids": ids}
    return SimpleNamespace(apply_chat_template=apply)


def test_header_ids_under_arm_reads_each_arms_appended_header():
    tok = _arm_tok()
    assert oi.header_ids_under_arm(tok, "off") == _HDR["off"]
    assert oi.header_ids_under_arm(tok, "on") == _HDR["on"]
    # a render whose False side is not a token PREFIX of the True side cannot
    # be read at all, and says so rather than guessing a width
    bad = SimpleNamespace(apply_chat_template=lambda m, add_generation_prompt=False,
                          tokenize=True, **kw: {"input_ids": [9] if add_generation_prompt else [1]})
    with pytest.raises(ValueError, match="not a token prefix"):
        oi.header_ids_under_arm(bad, "on")


def test_the_override_predicts_against_the_served_arm_and_does_not_block(tmp_path, monkeypatch):
    """PW-oos: with ``allow_out_of_support_thinking`` in the request, BOTH arms'
    headers are measured and recorded, the object's guard is predicted against
    the arm the run will SERVE (H=5), and the mismatch against the fit's H=7 is
    reported as ``predicted_refuse_overridden`` — the preflight does not fail,
    which is the whole point of an explicit override."""
    def derive(*, artifact, cell_id, si_id, arm, cfg, shell, tok, coeff=1.0):
        return {"cell": cell_id, "intervention_hash": "1111111111111111",
                "resolved_spec": {"positions": "response"},
                "response_header_len": 7,            # fit under the pinned-OFF render
                "positions": "response", "artifact": str(artifact),
                "artifact_sha256": "s" * 64}
    tok = _patched(monkeypatch, header_len=7, derive=derive)
    tok.apply_chat_template = _arm_tok().apply_chat_template
    req = dict(_request(tmp_path, [{"cell": "safety_refusal__psr_kl_all",
                                    "si_id": "safety_refusal", "arm": "psr_kl_all",
                                    "expected": "1111111111111111"}]),
               allow_out_of_support_thinking=True)
    rep = oi.derive_request(req)
    # the measurement the override KEEPS, both arms, before any GPU minute
    assert rep[C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING] is True
    assert rep["out_of_support_thinking_note"] == C.OUT_OF_SUPPORT_THINKING_NOTE
    assert rep["generation_header_len_by_arm"] == {"off": 7, "on": 5}
    assert rep["generation_header_ids_by_arm"] == {"off": list(_HDR["off"]),
                                                   "on": list(_HDR["on"])}
    assert rep["generation_header_moves"] is True
    # the top-level pinned-OFF derivation is UNCHANGED (it is a different fact)
    assert rep["generation_header_len"] == 7
    (rec,) = rep["results"]
    assert rec["header_guard"] == oi.HEADER_GUARD_OVERRIDDEN == "predicted_refuse_overridden"
    assert rec["header_guard_will_pass"] is True
    assert rec["header_guard_compared_against"] == {
        "arm": "on", "generation_header_len": 5, "pinned_off_generation_header_len": 7}
    assert rec["generation_header_len"] == 7          # the pinned-OFF one, unmoved
    assert rep["ok"] is True                          # the plan is not blocked


def test_the_same_request_without_the_flag_still_refuses(tmp_path, monkeypatch):
    """The invariance pin: identical inputs, no override key — the report has no
    per-arm block, the guard is predicted against the pinned-OFF header (which
    AGREES with the fit's 7, so it passes) and no key the override adds
    appears anywhere."""
    def derive(*, artifact, cell_id, si_id, arm, cfg, shell, tok, coeff=1.0):
        return {"cell": cell_id, "intervention_hash": "1111111111111111",
                "resolved_spec": {"positions": "response"},
                "response_header_len": 7, "positions": "response",
                "artifact": str(artifact), "artifact_sha256": "s" * 64}
    tok = _patched(monkeypatch, header_len=7, derive=derive)
    tok.apply_chat_template = _arm_tok().apply_chat_template
    rep = oi.derive_request(_request(tmp_path, [
        {"cell": "safety_refusal__psr_kl_all", "si_id": "safety_refusal",
         "arm": "psr_kl_all", "expected": "1111111111111111"}]))
    for k in (C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING, "out_of_support_thinking_note",
              "generation_header_len_by_arm", "generation_header_ids_by_arm",
              "generation_header_moves"):
        assert k not in rep, k
    (rec,) = rep["results"]
    assert "header_guard_compared_against" not in rec
    assert rec["header_guard"] == oi.HEADER_GUARD_CHECKED
    assert rep["ok"] is True
    # and a fit/serve DISAGREEMENT without the flag is still a refusal
    assert oi.header_guard_prediction(7, 5) == (False, oi.HEADER_GUARD_MISMATCH)
    assert oi.header_guard_prediction(7, 5, allow_out_of_support=True) == (
        True, oi.HEADER_GUARD_OVERRIDDEN)


def test_an_unmeasurable_arm_is_recorded_not_guessed(tmp_path, monkeypatch):
    """A tokenizer that cannot answer one arm records ``None`` for its width and
    the exception text for its ids — never a width the derivation invented."""
    def derive(*, artifact, cell_id, si_id, arm, cfg, shell, tok, coeff=1.0):
        return {"cell": cell_id, "intervention_hash": "1111111111111111",
                "resolved_spec": {"positions": "response"},
                "response_header_len": 7, "positions": "response",
                "artifact": str(artifact), "artifact_sha256": "s" * 64}
    _patched(monkeypatch, header_len=7, derive=derive)   # the bare tok has no apply_chat_template
    rep = oi.derive_request(dict(
        _request(tmp_path, [{"cell": "safety_refusal__psr_kl_all", "si_id": "safety_refusal",
                             "arm": "psr_kl_all", "expected": "1111111111111111"}]),
        allow_out_of_support_thinking=True))
    assert rep["generation_header_len_by_arm"] == {"off": None, "on": None}
    assert all(isinstance(v, str) and "Error" in v
               for v in rep["generation_header_ids_by_arm"].values())
    assert rep["generation_header_moves"] is None
    # with no ON width measured the guard falls back to the pinned-OFF one,
    # which AGREES with the fit — reported as checked, never as overridden
    (rec,) = rep["results"]
    assert rec["header_guard"] == oi.HEADER_GUARD_CHECKED


# --------------------------------------------------------------------------- #
# the explicit-cells dispatch and the roster pin                                 #
# --------------------------------------------------------------------------- #


def test_route_cell_source_dispatches_like_the_standup(tmp_path, monkeypatch):
    """A config carrying ``explicit_cells`` builds the EXPLICIT registry over
    ``objects_dir`` handed in as the ABSOLUTE ``defenses_dir`` (never the config's
    own — one bank for both resolvers); any other config builds the Cartesian one
    exactly as before, with ``si_ids`` / class arm lists and module defaults."""
    from sisteer.eval import defense_cell as tt

    seen = {}

    def explicit(cfg, *, repo_root=None):
        seen["explicit"] = (dict(cfg), repo_root)
        return []

    def cartesian(ddir, *, si_ids, class_a_arms, class_b_arms, **kw):
        seen["cartesian"] = (Path(ddir), si_ids, class_a_arms, class_b_arms, kw)
        return []

    monkeypatch.setattr(tt, "registry_from_explicit_cells", explicit)
    monkeypatch.setattr(tt, "build_registry", cartesian)
    bank = tmp_path / "bank"
    src = oi.route_cell_source({tt.EXPLICIT_CELLS_KEY: ["none", "constv2-mix-full__psr_ce_all"],
                                "defenses_dir": "out/full22/gemma4e4b/cells_view/it",
                                "model": "m"}, bank)
    assert "cartesian" not in seen
    cfg_abs, root = seen["explicit"]
    assert cfg_abs["defenses_dir"] == str(bank) and root is None
    assert cfg_abs[tt.EXPLICIT_CELLS_KEY] == ["none", "constv2-mix-full__psr_ce_all"] and cfg_abs["model"] == "m"
    assert Path(src.defenses_dir) == bank
    seen.clear()
    oi.route_cell_source({"si_ids": ["safety_refusal"], "class_a_arms": ["psr_kl_all"]}, bank)
    assert "explicit" not in seen
    ddir, si_ids, a_arms, b_arms, kw = seen["cartesian"]
    assert ddir == bank and si_ids == ("safety_refusal",) and a_arms == ("psr_kl_all",)
    assert b_arms == tuple(tt.CLASS_B_ARMS) and kw == {}                  # module defaults; no stacked_cells
    seen.clear()
    oi.route_cell_source({}, bank)
    assert seen["cartesian"][1] == tuple(tt.DEFAULT_SI_IDS) and seen["cartesian"][2] == tuple(tt.CLASS_A_ARMS)


def test_derive_request_records_the_roster_pin_match_and_refuses_a_mismatch(tmp_path, monkeypatch):
    """With ``pins`` in the request each result carries ``pin_match`` (artifact
    sha == the roster pin); a mismatch turns ``ok`` false even when the HF hash
    matches; a cell with no pin records null and nothing else changes; a
    per-target ``pin_sha256`` (what the launcher's ``identity_request`` carries
    from the lane roster) wins over the request map."""
    def derive(*, artifact, cell_id, si_id, arm, cfg, shell, tok, coeff=1.0):
        return {"cell": cell_id, "intervention_hash": "1111111111111111",
                "resolved_spec": {"positions": "response"}, "response_header_len": None,
                "positions": "response", "artifact": str(artifact),
                "artifact_sha256": {"a": "a" * 64, "b": "b" * 64, "c": "c" * 64}[cell_id]}
    _patched(monkeypatch, header_len=3, derive=derive)
    targets = [{"cell": c, "si_id": "si", "arm": "psr_kl_all", "expected": "1111111111111111"} for c in "abc"]
    req = _request(tmp_path, targets)
    plain = oi.derive_request(req)
    assert plain["ok"] is True and [r["pin_match"] for r in plain["results"]] == [None, None, None]
    assert all(r["pin_sha256"] is None for r in plain["results"])
    rep = oi.derive_request({**req, "pins": {"a": "a" * 64, "b": "0" * 64}})
    a, b, c = rep["results"]
    assert a["pin_match"] is True and a["match"] is True
    assert b["pin_match"] is False and b["match"] is True                # the HF hash alone is not enough
    assert c["pin_match"] is None
    assert rep["ok"] is False
    # a per-target pin wins over the request map
    rep2 = oi.derive_request({**req, "pins": {"a": "0" * 64},
                              "targets": [{**targets[0], "pin_sha256": "a" * 64}]})
    assert rep2["results"][0]["pin_match"] is True and rep2["ok"] is True
    # a per-target WRONG pin refuses even with the request map agreeing
    rep3 = oi.derive_request({**req, "pins": {"a": "a" * 64},
                              "targets": [{**targets[0], "pin_sha256": "f" * 64}]})
    assert rep3["results"][0]["pin_match"] is False and rep3["ok"] is False
    assert oi.pin_match("A" * 64, "a" * 64) is True and oi.pin_match("a" * 64, "zz") is False


# --------------------------------------------------------------------------- #
# the SERVED-ONLY Class B branch                                               #
# --------------------------------------------------------------------------- #


def test_a_served_only_class_b_object_derives_from_its_kind_and_sidecar_never_from_its_name(tmp_path, monkeypatch):
    """`arm_spec_for_cell` refuses a bare id with no trainer in methods.ARMS (`arditi_allpos`: the explicit-cells
    registry admits it by name — defense_cells.SERVED_ONLY_CLASS_B_IDS — but the --pf identity derivation refused
    every such cell). The branch builds the spec from the OBJECT: the hydrated blob's stored kind must be a served-only
    Class B kind (`vector_all`), the dir must declare the `fit_provenance.json` identity contract and pass its verdict,
    and the layer is the object's OWN (a multi-layer object declares its whole set). Guarded by kind + sidecar, never
    by the cell's name; a Class A cell or a registered arm never enters the branch."""
    import sisteer.eval.train_budget as tb
    art = tmp_path / "bank" / "arditi_allpos" / "object.pt"
    art.parent.mkdir(parents=True)
    art.write_bytes(b"blob")
    monkeypatch.setattr(tb, "object_identity_kind", lambda path: "fit_provenance")
    monkeypatch.setattr(tb, "fit_provenance_problems", lambda path, cell_id: [])
    obj = SimpleNamespace(kind="vector_all", layers=[12], positions="all", skip_bos=False)
    spec = oi.served_only_class_b_spec(cell_id="arditi_allpos", artifact=art, obj=obj, cfg={"layer": 15, "scale": 1.0})
    assert spec == {"name": "arditi_allpos", "kind": "sisteer_object", "artifact": str(art), "layer": 12, "coeffs": [1.0]}
    multi = oi.served_only_class_b_spec(cell_id="arditi_allpos", artifact=art, obj=SimpleNamespace(kind="vector_all", layers=[8, 12]), cfg={})
    assert multi["layer"] == 8 and multi["layers"] == [8, 12]                      # the whole set declared, F9 holds it EQUAL
    # the guards: another kind, another identity contract, a failing sidecar verdict, no layer
    with pytest.raises(ValueError, match="not a served-only Class B kind"):
        oi.served_only_class_b_spec(cell_id="arditi_allpos", artifact=art, obj=SimpleNamespace(kind="lora_subtraction", layers=[12]), cfg={})
    monkeypatch.setattr(tb, "object_identity_kind", lambda path: "staged")
    with pytest.raises(ValueError, match="declares 'staged'"):
        oi.served_only_class_b_spec(cell_id="arditi_allpos", artifact=art, obj=obj, cfg={})
    monkeypatch.setattr(tb, "object_identity_kind", lambda path: "fit_provenance")
    monkeypatch.setattr(tb, "fit_provenance_problems", lambda path, cell_id: ["sidecar sha != bytes"])
    with pytest.raises(ValueError, match="does not certify"):
        oi.served_only_class_b_spec(cell_id="arditi_allpos", artifact=art, obj=obj, cfg={})
    monkeypatch.setattr(tb, "fit_provenance_problems", lambda path, cell_id: [])
    with pytest.raises(ValueError, match="reports no layer"):
        oi.served_only_class_b_spec(cell_id="arditi_allpos", artifact=art, obj=SimpleNamespace(kind="vector_all", layers=[]), cfg={})
    assert oi.SERVED_ONLY_CLASS_B_KINDS == ("vector_all",)
    # derive_one ROUTES a bare id outside methods.ARMS through the branch (hydrating first), and a registered arm through
    # arm_spec_for_cell as before; a Class A cell (an SI) never enters the branch whatever its arm
    import sisteer.eval.cell_controls as cc
    calls = {"spec": [], "hydrate": [], "hash": []}
    monkeypatch.setattr(oi, "methods_arms", lambda: {"psr_kl_all": object()})
    import sisteer.methods as methods_mod
    monkeypatch.setattr(methods_mod, "_hydrate_object", lambda path, shell, tok: calls["hydrate"].append(str(path)) or obj)
    monkeypatch.setattr(cc, "arm_spec_for_cell", lambda cell, cfg: calls["spec"].append(cell.arm) or {"name": cell.id, "kind": "sisteer_object", "artifact": str(art), "layer": 15, "coeffs": [1.0]})
    import sisteer.capeval.serving.resolved_spec as rs
    monkeypatch.setattr(rs, "resolve_and_hash", lambda spec, o, coeff, err, arm_response_mask=None: calls["hash"].append(spec.name) or ({"kind": spec.kind, "layers": [spec.layer]}, "0" * 16))
    import sisteer.capeval.serving.hf_engine as hf
    monkeypatch.setattr(hf, "HFEngine", lambda *a, **k: SimpleNamespace(_arm_response_mask=None))
    rec = oi.derive_one(artifact=art, cell_id="arditi_allpos", si_id=None, arm="arditi_allpos", cfg={"model": "m", "model_revision": "r"}, shell=None, tok=None)
    assert rec["intervention_hash"] == "0" * 16 and rec["ablation_spec_kwargs"]["layer"] == 12 and calls["spec"] == []   # the branch, not ARMS
    assert calls["hydrate"] == [str(art)] and calls["hash"] == ["arditi_allpos"]
    rec = oi.derive_one(artifact=art, cell_id="safety_refusal__psr_kl_all", si_id="safety_refusal", arm="psr_kl_all", cfg={"model": "m", "model_revision": "r"}, shell=None, tok=None)
    assert calls["spec"] == ["psr_kl_all"] and rec["ablation_spec_kwargs"]["layer"] == 15                              # a Class A cell: the registry path
    # a Class B id outside ARMS whose object is NOT a served-only kind refuses (no fall-through to a guessed spec)
    monkeypatch.setattr(methods_mod, "_hydrate_object", lambda path, shell, tok: SimpleNamespace(kind="lora", layers=[3]))
    with pytest.raises(ValueError, match="not a served-only Class B kind"):
        oi.derive_one(artifact=art, cell_id="mystery", si_id=None, arm="mystery",
                      cfg={"model": "m", "model_revision": "r"}, shell=None, tok=None)


ARDITI_REAL = Path.home() / "arditi-stage/objects/arditi_allpos/qwen35_4b/ladder/cells/ar_c0p3__arditi_allpos/object.pt"


@pytest.mark.skipif(not ARDITI_REAL.is_file(), reason="the grading host Arditi stage (ba2212f0) is not here")
def test_the_real_arditi_object_derives_to_its_pinned_hash_through_the_branch(tmp_path):
    """The real bytes (sha256 ba2212f0…, the pins' qwen35_4b/ar_c0p3__arditi_allpos) through `derive_request` on the
    pinned Qwen3.5-4B meta shell: kind vector_all, layer 12, positions all, skip_bos false; pin_match True; the hash is
    the one EXPECTED_INTERVENTION_HASHES pins for the three seed rows (the same bytes + spec on each — one object,
    one identity; the container preflight re-derives and refuses a disagreement)."""
    import hashlib
    pytest.importorskip("torch")
    if hashlib.sha256(ARDITI_REAL.read_bytes()).hexdigest() != "ba2212f0c2f9ffef2997bdd24831182052560cf980054bcee9cf95ea0db085b2":
        pytest.skip("the grading host Arditi stage does not carry the pinned bytes")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained("Qwen/Qwen3.5-4B", revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", local_files_only=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"the pinned Qwen3.5-4B config is not in the local HF cache ({type(e).__name__})")
    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "arditi_allpos").symlink_to(ARDITI_REAL.parent)
    cfg = tmp_path / "cfg.yaml"
    src = (REPO / C.bcbw_config_path("qwen35_4b", "s1")).read_text()
    body = src[src.index("# -- everything below:"):]
    import re as _re
    body = _re.sub(r"^defenses_dir: .*$", f"defenses_dir: {bank}", body, count=1, flags=_re.M)
    cfg.write_text("explicit_cells:\n- none\n- arditi_allpos\n\n" + body)
    rep = oi.derive_request({"model": "Qwen/Qwen3.5-4B", "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "config": str(cfg),
                             "config_overrides": {}, "objects_dir": str(bank),
                             "targets": [{"cell": "arditi_allpos", "si_id": None, "arm": "arditi_allpos",
                                          "expected": C.EXPECTED_INTERVENTION_HASHES["qwen35_4b-f22s1"]["arditi_allpos"],
                                          "pin_sha256": "ba2212f0c2f9ffef2997bdd24831182052560cf980054bcee9cf95ea0db085b2"}]})
    (t,) = rep["results"]
    assert t.get("error") is None, t.get("error")
    assert t["object_kind"] == "vector_all" and t["resolved_spec"]["layers"] == [12] and t["resolved_spec"]["positions"] == "all"
    assert t["pin_match"] is True and t["match"] is True and rep["ok"] is True
    assert t["ablation_spec_kwargs"]["layer"] == 12 and "layers" not in t["ablation_spec_kwargs"]
    for v in C.BCBW_SEED_VIEWS[:3]:
        assert C.EXPECTED_INTERVENTION_HASHES[f"qwen35_4b-f22{v}"]["arditi_allpos"] == t["intervention_hash"]

