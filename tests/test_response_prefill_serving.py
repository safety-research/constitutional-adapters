"""The prefill-seeded response mask (Phase 2, served loglikelihood): the
declaration's whole path — proxy gate -> selector -> HookContext ->
located boundary — plus the loglikelihood positive control's ability to
FAIL. All offline, all synthetic (invented prompts/tokens, placeholder
ids, the proxy tests' own canonical cell set — no real recorded data).

What is covered, end to end:

* ``lens_adapters._response_start``: an UNDECLARED seeded render still
  refuses (the exact pre-Phase-2 behavior); a DECLARED one locates the
  boundary at the LAST header occurrence (parity's ``_rfind_subseq``
  semantics), so the mask covers header + seed + the final prefill row
  (the scored distribution's predictor) BY CONSTRUCTION; a declared
  render whose header ids occur nowhere refuses; ``positions="last"``
  refuses under the declaration (derivation locality).
* ``lens_steer.parse_selector`` accepts the fact strictly (bool only).
* ``lens_worker.build_request_slots`` threads the fact from the selector
  into exactly the flagged request's HookContext.
* ``proxy.RoutingProxy``: the ``petri_response_prefill`` gate's full
  refusal matrix — cannot be set accidentally, cannot be omitted where
  the engine would otherwise refuse fatally — and the verified
  declaration travelling as the injected selector's ``response_prefill``
  while the client key never goes upstream.
* ``fidelity.check_logprob_steering_bites``: PASSES on a real divergence,
  FAILS on bit-identical logprobs (the silently-unsteered case), FAILS on
  a divergence inside the measured repeat-noise ceiling, refuses vacuous
  or malformed evidence — the control can fail, proven.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from sisteer.capeval.serving import lens_adapters as la  # noqa: E402
from sisteer.capeval.serving import lens_steer as ls  # noqa: E402
from sisteer.capeval.serving import lens_worker as lw  # noqa: E402
from sisteer.capeval.serving import proxy as px  # noqa: E402
from sisteer.capeval.serving.fidelity import (  # noqa: E402
    FidelityError,
    check_logprob_steering_bites,
)

from test_petri_proxy import _proxy, _spec  # noqa: E402

HEADER = (901, 902, 903, 904)  # synthetic assistant-header ids
D = 8


def _resp_op(**over):
    kw = dict(op=la.OP_ADD_VECTOR, layer=0, positions=la.POSITIONS_RESPONSE,
              header_ids=HEADER, vector=torch.zeros(D, dtype=torch.float32))
    kw.update(over)
    return la.HookOp(**kw)


def _prefill_ctx(ids, *, seeded=False):
    return la.HookContext(la.PHASE_PREFILL, tuple(range(len(ids))),
                          tuple(ids), response_prefill=seeded)


# --------------------------------------------------------------------------- #
# _response_start: the located boundary                                        #
# --------------------------------------------------------------------------- #


def test_undeclared_seeded_render_still_refuses():
    """The pre-Phase-2 refusal is intact: a prompt that does not END at
    the header refuses unless the request declared the seed."""
    ids = (1, 2, *HEADER, 55, 56)  # header + a 2-token seed, undeclared
    with pytest.raises(la.AdapterError, match="DECLARES"):
        la._response_start(_resp_op(), _prefill_ctx(ids), len(ids))


def test_declared_seed_locates_header_and_masks_through_tail():
    ids = (1, 2, 3, *HEADER, 55, 56)  # user turn, header, seed "Answer:"
    ctx = _prefill_ctx(ids, seeded=True)
    op = _resp_op()
    start = la._response_start(op, ctx, len(ids))
    assert start == 3
    mask = la._row_mask(op, ctx, torch.zeros(len(ids), D))
    # header + seed + (in particular) the FINAL prefill row — the forward
    # that emits token 1, i.e. the scored distribution — all inside.
    assert mask.tolist() == [False] * 3 + [True] * (len(HEADER) + 2)
    assert bool(mask[-1])


def test_declared_unseeded_render_is_the_tail_boundary():
    """The declaration on an ordinary render (header IS the tail) is
    harmless: rfind finds the tail occurrence, same boundary as the
    undeclared path."""
    ids = (1, 2, 3, *HEADER)
    op = _resp_op()
    seeded = la._response_start(op, _prefill_ctx(ids, seeded=True), len(ids))
    plain = la._response_start(op, _prefill_ctx(ids), len(ids))
    assert seeded == plain == 3


def test_declared_seed_takes_the_last_header_occurrence():
    """Multi-turn: an earlier assistant turn's header must not win —
    parity._rfind_subseq semantics, the LAST occurrence."""
    ids = (*HEADER, 7, 8, 9, *HEADER, 55)
    start = la._response_start(_resp_op(), _prefill_ctx(ids, seeded=True),
                               len(ids))
    assert start == 7


def test_declared_seed_with_no_header_refuses():
    ids = (1, 2, 3, 4, 5, 6)
    with pytest.raises(la.AdapterError, match="occur NOWHERE"):
        la._response_start(_resp_op(), _prefill_ctx(ids, seeded=True),
                           len(ids))


def test_positions_last_refuses_a_declared_seed():
    ids = (1, 2, 3, *HEADER, 55)
    op = la.HookOp(op=la.OP_ADD_VECTOR, layer=0,
                   positions=la.POSITIONS_LAST,
                   vector=torch.zeros(D, dtype=torch.float32))
    with pytest.raises(la.AdapterError, match="positions='last'"):
        la._row_mask(op, _prefill_ctx(ids, seeded=True),
                     torch.zeros(len(ids), D))


def test_hookcontext_response_prefill_is_a_strict_bool():
    with pytest.raises(la.AdapterError, match="must be a bool"):
        la.HookContext(la.PHASE_PREFILL, (0, 1), (1, 2),
                       response_prefill=1)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# selector parsing + worker threading                                          #
# --------------------------------------------------------------------------- #


def test_parse_selector_accepts_the_fact_strictly():
    # both wire forms: a real bool, and the transport's lax-coerced int
    # (vllm_xargs types values str|int|float, so a JSON true ARRIVES as 1
    # — a bool-only parse killed the engine on the first live request,
    # 2026-09-01)
    for wire in (True, 1):
        sel = ls.parse_selector({"petri_object": "c", "expect_hash": "h",
                                 "response_prefill": wire})
        assert sel["response_prefill"] is True
    for wire in (False, 0):
        sel = ls.parse_selector({"petri_object": "c", "expect_hash": "h",
                                 "response_prefill": wire})
        assert sel["response_prefill"] is False
    sel = ls.parse_selector({"petri_object": "c", "expect_hash": "h"})
    assert "response_prefill" not in sel
    for bad in ("yes", 2, 1.0):
        with pytest.raises(ls.LensSteerError, match="response_prefill"):
            ls.parse_selector({"petri_object": "c", "expect_hash": "h",
                               "response_prefill": bad})


class _FakeRegistry:
    """Just enough of WorkerObjectRegistry for build_request_slots: one
    plan-needing object per selector."""

    def __init__(self):
        self.obj = ls.LoadedObject(
            cell_id="cell", kind="obj_psr_kl_all", expect_hash="ih",
            ops=(_resp_op(),), effective_scale=1.0)

    def select(self, petri_object, expect_hash, scale=None):
        return self.obj


def test_slot_builder_threads_the_fact_per_request():
    """Two co-batched prefills, only one flagged: exactly that one's ctx
    carries response_prefill."""
    ids_a = [1, 2, 3, *HEADER, 55]           # flagged, seeded
    ids_b = [9, 8, 7, *HEADER]               # unflagged, ordinary
    q = [len(ids_a), len(ids_b)]
    qsl = [0, q[0], q[0] + q[1]]
    positions = list(range(q[0])) + list(range(q[1]))
    toks = {"a": ids_a, "b": ids_b}
    slots = lw.build_request_slots(
        query_start_loc=qsl, seq_lens=q, num_actual_tokens=sum(q),
        positions=positions, req_ids=["a", "b"],
        raw_selectors=[
            {"petri_object": "cell", "expect_hash": "ih",
             "response_prefill": 1},  # the wire form, as vllm_xargs lands
            {"petri_object": "cell", "expect_hash": "ih"},
        ],
        token_lookup=lambda rid: toks[rid],
        prompt_len_lookup=lambda rid: len(toks[rid]),
        registry=_FakeRegistry())
    by_id = {s.req_id: s for s in slots.slots}
    assert by_id["a"].ctx is not None and by_id["a"].ctx.response_prefill
    assert by_id["b"].ctx is not None and not by_id["b"].ctx.response_prefill
    # and the flagged context locates the seeded boundary
    assert la._response_start(_resp_op(), by_id["a"].ctx, q[0]) == 3


# --------------------------------------------------------------------------- #
# the proxy gate                                                               #
# --------------------------------------------------------------------------- #

LENS_CELL = "si_safety__psr_kl_all"


def _vmodel(cells, cell_id):
    return next(r.virtual_model for r in px.routes_from_cells(cells)
                if r.cell_id == cell_id)


def _seeded_chat(model, *, declare=True, continued=True, agp=False,
                 last_role="assistant"):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Which letter? Reply A-D."},
                     {"role": last_role, "content": "Answer:"}],
        "max_tokens": 1, "temperature": 0.0,
        "logprobs": True, "top_logprobs": 4,
    }
    if continued:
        body["continue_final_message"] = True
        body["add_generation_prompt"] = agp
    if declare:
        body[px.RESPONSE_PREFILL_KEY] = True
    return body


def _ask(proxy, request):
    return asyncio.run(proxy.handle_chat_completions(request))


def test_declared_seed_travels_as_a_selector_fact(tmp_path):
    proxy, backend, cells = _proxy(
        tmp_path, exact_replay_strategy=None,
        spec=_spec(serving_mode="interleaved"))
    vm = _vmodel(cells, LENS_CELL)
    _ask(proxy, _seeded_chat(vm))
    up = backend.payloads[-1]
    sel = up[px.UPSTREAM_EXTRA_ARGS_KEY]
    # the transport form: int 1, never a bool (vllm_xargs lax-coerces
    # JSON bools to ints; the proxy sends the one honest spelling)
    assert sel["response_prefill"] == 1 and not isinstance(
        sel["response_prefill"], bool)
    assert sel["petri_object"] == LENS_CELL
    # the client key never travels upstream; the render keys do
    assert px.RESPONSE_PREFILL_KEY not in up
    assert up["continue_final_message"] is True
    assert up["add_generation_prompt"] is False


def test_undeclared_continued_message_on_a_lens_route_refuses(tmp_path):
    proxy, backend, cells = _proxy(
        tmp_path, exact_replay_strategy=None,
        spec=_spec(serving_mode="interleaved"))
    vm = _vmodel(cells, LENS_CELL)
    with pytest.raises(px.ProxyRefusal,
                       match="without declaring") as e:
        _ask(proxy, _seeded_chat(vm, declare=False))
    assert e.value.code == "response_prefill_undeclared"
    assert backend.payloads == []  # refused BEFORE the engine


def test_declaration_on_a_non_lens_route_refuses(tmp_path):
    proxy, backend, cells = _proxy(
        tmp_path, exact_replay_strategy=None,
        spec=_spec(serving_mode="interleaved"))
    vm = _vmodel(cells, "none")  # passthrough: no mask to re-locate
    with pytest.raises(px.ProxyRefusal, match="not lens-served") as e:
        _ask(proxy, _seeded_chat(vm))
    assert e.value.code == "response_prefill_not_lens"
    # ... but an undeclared continued message on a MASKLESS route is fine
    resp = _ask(proxy, _seeded_chat(vm, declare=False))
    assert resp["model"] == vm
    assert backend.payloads[-1]["continue_final_message"] is True


def test_declaration_shape_mismatches_refuse(tmp_path):
    proxy, _, cells = _proxy(
        tmp_path, exact_replay_strategy=None,
        spec=_spec(serving_mode="interleaved"))
    vm = _vmodel(cells, LENS_CELL)
    with pytest.raises(px.ProxyRefusal, match="continue_final_message"):
        _ask(proxy, _seeded_chat(vm, continued=False))
    with pytest.raises(px.ProxyRefusal, match="add_generation_prompt"):
        _ask(proxy, _seeded_chat(vm, agp=True))
    with pytest.raises(px.ProxyRefusal, match="assistant"):
        _ask(proxy, _seeded_chat(vm, last_role="user"))
    with pytest.raises(px.ProxyRefusal, match="JSON boolean"):
        req = _seeded_chat(vm)
        req[px.RESPONSE_PREFILL_KEY] = "true"
        _ask(proxy, req)


def test_selector_smuggling_of_the_fact_still_refuses(tmp_path):
    """The fact rides the PROXY's selector; a client writing vllm_xargs
    itself is still selector smuggling."""
    proxy, _, cells = _proxy(
        tmp_path, exact_replay_strategy=None,
        spec=_spec(serving_mode="interleaved"))
    vm = _vmodel(cells, LENS_CELL)
    req = _seeded_chat(vm)
    req[px.UPSTREAM_EXTRA_ARGS_KEY] = {"response_prefill": True}
    with pytest.raises(px.ProxyRefusal, match="reserved selector"):
        _ask(proxy, req)


# --------------------------------------------------------------------------- #
# the loglikelihood positive control CAN FAIL                                  #
# --------------------------------------------------------------------------- #


def _lp_items(steered_shift=1.0, noise=0.0):
    base = [-1.0, -2.5, -3.0, -4.0]
    return [{
        "item_id": f"m{i:04d}-aaaaaaa{i}",
        "letter_ids": [11, 12, 13, 14],
        "steered": [v - steered_shift * (j == i % 4)
                    for j, v in enumerate(base)],
        "scale0": list(base),
        "scale0_repeat": [v + noise * (j == (i + 1) % 4)
                          for j, v in enumerate(base)],
    } for i in range(6)]


def test_logprob_control_passes_on_real_divergence():
    diag: dict = {}
    check_logprob_steering_bites({"items": _lp_items(steered_shift=1.0)},
                                 diag)
    assert diag["n_items"] == 6
    assert diag["steered_divergence_median"] == pytest.approx(1.0)
    assert diag["noise_ceiling"] == 0.0
    assert diag["bar"] == pytest.approx(1e-3)


def test_logprob_control_fails_on_identical_logprobs():
    """THE control-must-fail case: steered == twin bitwise — the silently
    unsteered scored position."""
    with pytest.raises(FidelityError, match="BIT-IDENTICAL"):
        check_logprob_steering_bites(
            {"items": _lp_items(steered_shift=0.0)}, {})


def test_logprob_control_fails_inside_the_noise_ceiling():
    """A 'difference' no bigger than the twin's own repeat noise proves
    nothing — the hypersensitive-readout trap, refused."""
    with pytest.raises(FidelityError, match="does not clear the bar"):
        check_logprob_steering_bites(
            {"items": _lp_items(steered_shift=0.02, noise=0.02)}, {})


def test_logprob_control_refuses_vacuous_or_malformed_evidence():
    with pytest.raises(FidelityError, match="zero probe items"):
        check_logprob_steering_bites({"items": []}, {})
    bad = _lp_items()
    bad[0]["scale0_repeat"] = [float("nan")] * 4
    with pytest.raises(FidelityError, match="finite"):
        check_logprob_steering_bites({"items": bad}, {})
    short = _lp_items()
    short[2]["steered"] = short[2]["steered"][:2]
    with pytest.raises(FidelityError, match="disagree on"):
        check_logprob_steering_bites({"items": short}, {})
