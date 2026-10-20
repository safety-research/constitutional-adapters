"""native-psr-serving step 7 (stage 2), lane P — the in-graph PER-TOKEN POSITION
CLASS and the request-scoped response-boundary failure INSIDE THE WORKER
(``serving/native_residual.py``).

What is proven, on CPU, through the REAL wrappers installed on fakes carrying the
wheel's signatures for BOTH GPU model runners (``native_residual_fakes``: a
V1-shaped runner with ``set_active_loras(input_batch, num_scheduled_tokens)``
inside ``execute_model(scheduler_output)``; a V2-shaped one with
``LoraState.make_lora_inputs(req_ids, idx_mapping, num_scheduled_tokens)``,
``req_states.prompt_len.np`` and ``input_batch.idx_mapping_np``):

- MASK EQUIVALENCE per mode against the lens's OWN row selection
  (``lens_adapters._row_selection`` with the lens's ``HookOp`` / ``HookContext``,
  phase-gated as the lens worker invokes it), on both runner shapes, over a
  scripted engine: the default render (the prompt ends at the header), the
  SEEDED render (header + assistant seed — the boundary at the LAST header),
  multi-turn (several headers — the LAST), prefix-cache RESUME
  (``num_computed_tokens > 0``: positions are absolute), prefill and decode
  steps, request REORDERING across steps (V2's batch order permuted, V1's
  condense), INDEX REUSE after a request finishes; and the op's EFFECT: rows
  the class selects differ from the unpatched pair, every other row is
  bit-identical to it.
- KNOWN-BADS: a header-less prompt on a ``response`` slot -> every row of it
  unsteered AND exactly one ledger entry (the same prompt on an all-positions
  slot steers and never ledgers); a ``prompt_last`` mask off by one refuses the
  equivalence assertion; the V2 (and V1) prompt-length cross-check disagreeing
  -> a recorded fault, the request ledgered and unsteered.
- THE LEDGER after readiness (fact 15, rev 4): ``REQUEST_FAILURES_CAP`` (1024)
  unclaimed failures LATCH ``ledger_fault``; the OLDEST entry is still present
  and a claim on its id returns the fault payload (never ``None``); a fresh id
  returns the fault payload; the all-positions slot keeps steering; the rev-3
  known-bad (an evicting ledger returns ``None`` for the oldest id).
- the fact table's cap and ``overflow`` (the overflowing request ledgered and
  unsteered on a restricted slot); the header-id cross-check at activation (a
  disagreement refuses; a restricted-mode payload without ids refuses); the
  wrappers idempotent and composed with the routing recorder's instance
  wrapper in either order; a forward without positions refusing while a
  restricted slot is loaded; the report carrying the new facts.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import numpy as np
import pytest
import secrets
import torch

from sisteer.capeval.serving import lens_adapters as la
from sisteer.capeval.serving import native_residual as nr
from sisteer.capeval.serving import native_worker as nw
from sisteer.capeval.serving import worker_reports as wr

from native_residual_fakes import (FakeLoRARequest, FakeModel, FakeScheduler, HiddenSource,
                                   fake_vllm_config, make_fake_layer_classes,
                                   make_fake_lora_classes, make_fake_runner_classes, make_payload,
                                   write_residual_adapter)

BF16 = torch.bfloat16
D, L, S = 16, 3, 5
#: the tiny template's assistant header ids (H = 3)
HEADER = (900, 901, 902)
#: renders: the default (ends at the header), the seeded (header + a 2-token
#: assistant seed), multi-turn (two headers; the LAST is the boundary),
#: header-less (the known-bad), and the default one again for the resume case
DEFAULT = [1, 11, 12, 13, *HEADER]                       # P = 7, boundary 4
SEEDED = [1, 11, 12, *HEADER, 40, 41]                    # P = 8, boundary 3
MULTI = [1, 11, *HEADER, 40, 41, 2, 12, 13, *HEADER]      # P = 12, boundary 9
HEADERLESS = [1, 11, 12, 13, 14]                          # P = 5, no boundary


@pytest.fixture(autouse=True)
def _fresh():
    nr.reset_for_tests()
    yield
    nr.reset_for_tests()


# ---------------------------------------------------------------------------
# the harness: one runner shape, four residual slots, a plain LoRA, a scheduler
# ---------------------------------------------------------------------------


ALL_ADAPTERS = ("resp", "last", "pre", "all", "plain")


def _build(shape: str, tmp_path: Path, *, max_num_seqs: int = 16, max_tokens: int = 256,
           batch_order=None, preload: Sequence[str] = ALL_ADAPTERS):
    fk, layers = make_fake_lora_classes(), make_fake_layer_classes()
    rc = make_fake_runner_classes(fk)
    nr.reset_for_tests()
    targets = nr.PatchTargets(
        lora_model_cls=fk.LoRAModel, manager_cls=fk.LoRAModelManager, punica_cls=fk.PunicaWrapperBase,
        runner_mixin_cls=fk.LoRAModelRunnerMixin, decoder_layer_classes=tuple(layers.TABLE),
        model_info=fk.model_info, op=nr.ensure_op_registered(),
        runner_classes=(rc.RunnerV1, rc.RunnerV2), lora_state_cls=rc.LoraState)
    rep = nr.install(targets)
    vc = fake_vllm_config(hidden_size=D, num_layers=L, dtype=BF16, max_loras=S,
                          max_num_seqs=max_num_seqs, max_num_batched_tokens=max_tokens)
    hidden = HiddenSource(D)
    kw = {"batch_order": batch_order} if (shape == "v2" and batch_order is not None) else {}
    cls = rc.RunnerV1 if shape == "v1" else rc.RunnerV2
    runner = cls(vc, "cpu", layers.LlamaDecoderLayer, arch="llama", n_layers=L, hidden=hidden, **kw)
    runner.load_model()
    bank = nr.bank()
    assert bank is not None and bank.runner is runner

    def snapshot(step):                     # the per-token buffers are LIVE state: keep this step's
        b, T = bank.buffers, step.num_tokens
        step.extra["buffers"] = SimpleNamespace(
            slot_by_token=b.slot_by_token[:T].clone(), prompt_len_by_token=b.prompt_len_by_token[:T].clone(),
            response_start_by_token=b.response_start_by_token[:T].clone(), mode=b.mode.clone(),
            active=b.active.clone(), skip_bos=b.skip_bos.clone())

    runner.on_step = snapshot
    ref_model = FakeModel(make_fake_layer_classes().LlamaDecoderLayer, L, vc, arch="llama")
    all_layers = list(range(L))
    # the PSR gates carry a bias that keeps EVERY gate open on the harness's rows
    # (|true . w| stays well under 20), so "a selected row is steered" holds per row —
    # a closed gate is the object's own no-write, not the position class's
    payloads = {
        "resp": make_payload("psr", d=D, layers=all_layers, n_layers_model=L, seed=1, mode="response",
                             header_ids=HEADER, bias=20.0),
        "last": make_payload("static_r", d=D, layers=all_layers, n_layers_model=L, seed=2,
                             mode="prompt_last", header_ids=HEADER),
        "pre": make_payload("static_r", d=D, layers=all_layers, n_layers_model=L, seed=3,
                            mode="prefill_all", header_ids=HEADER, skip_bos=True),
        "all": make_payload("psr", d=D, layers=all_layers, n_layers_model=L, seed=4, mode="all", bias=20.0),
    }
    reqs: Dict[str, Optional[FakeLoRARequest]] = {}
    for i, name in enumerate(("resp", "last", "pre", "all"), start=1):
        adir = write_residual_adapter(tmp_path, name, payloads[name], d_in=D, d_out=D)
        reqs[name] = FakeLoRARequest(name, i, str(adir))
    from native_residual_fakes import write_adapter_dir
    plain = write_adapter_dir(tmp_path / "plain", layers=[0], d_in=D, d_out=D, rank=1)
    reqs["plain"] = FakeLoRARequest("plain", 5, str(plain))
    reqs["none"] = None
    for name in preload:                    # the production order: add_lora at startup
        runner.lora_manager.add_adapter(reqs[name])
    modes = {"resp": "response", "last": "prompt_last", "pre": "prefill_all", "all": "all",
             "plain": None, "none": None}
    return SimpleNamespace(fk=fk, layers=layers, rc=rc, targets=targets, rep=rep, vc=vc, runner=runner,
                           bank=bank, ref_model=ref_model, payloads=payloads, reqs=reqs, modes=modes,
                           sched=FakeScheduler(), shape=shape)


def _machinery(rep, suffix: str) -> str:
    """The ONE machinery entry whose key ends with ``suffix`` (the fakes' keys
    carry their factory's qualname prefix; the wheel's are bare)."""
    hits = [k for k in rep.machinery if k.endswith(suffix)]
    assert len(hits) == 1, (suffix, sorted(rep.machinery))
    return rep.machinery[hits[0]]


def _run_all(h, *, max_steps: int = 500, preempt_at: Optional[Dict[int, List[str]]] = None):
    """Drive the scheduler to quiescence; returns the executed steps (None for
    an empty step)."""
    steps = []
    k = 0
    while h.sched.live and k < max_steps:
        pre = (preempt_at or {}).get(k, [])
        so = h.sched.step(preempt=pre)
        steps.append(h.runner.execute_model(so))
        k += 1
    assert not h.sched.live, "the scripted engine did not drain"
    return steps


# ---------------------------------------------------------------------------
# the two selections: the lens's (reference) and the native buffers' (under test)
# ---------------------------------------------------------------------------


def _lens_op(mode: str, *, skip_bos: bool = False) -> la.HookOp:
    common = dict(op=la.OP_ADD_VECTOR, layer=0, vector=torch.zeros(D, dtype=BF16), skip_bos=skip_bos)
    if mode == "all":
        return la.HookOp(positions=la.POSITIONS_ALL, phase=la.PHASE_BOTH, **common)
    if mode == "response":
        return la.HookOp(positions=la.POSITIONS_RESPONSE, header_ids=HEADER, phase=la.PHASE_BOTH, **common)
    if mode == "prompt_last":                     # static_r's prefill-only ``last``
        return la.HookOp(positions=la.POSITIONS_LAST, phase=la.PHASE_PREFILL, **common)
    if mode == "prefill_all":                     # static_r's prefill-only ``all``
        return la.HookOp(positions=la.POSITIONS_ALL, phase=la.PHASE_PREFILL, **common)
    raise ValueError(mode)


def lens_mask(mode: str, *, prompt_ids: Sequence[int], positions: Sequence[int],
              skip_bos: bool = False, seeded: bool = False) -> torch.Tensor:
    """THE LENS'S row selection for ONE request's rows at absolute
    ``positions`` — ``_row_selection`` on the lens's own ``HookOp`` /
    ``HookContext``, invoked only in the phases the op declares (the lens
    worker skips an op outside its phase; a decode row of a prefill-only op
    is therefore OUT)."""
    op = _lens_op(mode, skip_bos=skip_bos)
    P, T = len(prompt_ids), len(positions)
    pos = tuple(int(p) for p in positions)
    if pos[0] < P:                                 # a (complete or resumed) prefill
        phase = la.PHASE_PREFILL
        cached = pos[0]
        ctx = la.HookContext(phase=phase, positions=pos, token_ids=tuple(prompt_ids[cached:pos[-1] + 1]),
                             response_prefill=seeded, cached=cached,
                             cached_ids=(tuple(prompt_ids[:cached]) if cached else None))
    else:
        phase = la.PHASE_DECODE
        ctx = la.HookContext(phase=phase, positions=pos)
    if op.phase not in (la.PHASE_BOTH, phase):
        return torch.zeros(T, dtype=torch.bool)
    sel = la._row_selection(op, ctx, torch.zeros(T, 1))
    if sel is None:
        return torch.ones(T, dtype=torch.bool)
    if sel is la.NO_ROWS:
        return torch.zeros(T, dtype=torch.bool)
    return sel.bool()


def native_selection(step, *, class_fn=None) -> torch.Tensor:
    """The op's FULL per-row selection at boundary 0 over the buffers the
    wrappers filled for ``step`` (snapshotted at the step: live slot, active,
    skip_bos, the position class) — ``class_fn`` swaps the class for a
    known-bad."""
    b = step.extra["buffers"]
    idx = b.slot_by_token
    live = idx >= 0
    idx_c = idx.clamp(min=0)
    pos = step.positions
    cls = (class_fn or nr.position_class_selection)(b.mode[idx_c], pos, b.prompt_len_by_token,
                                                    b.response_start_by_token)
    return live & b.active[idx_c, 0] & ~(b.skip_bos[idx_c] & (pos == 0)) & cls


def _check_step(h, step, prompts: Dict[str, Sequence[int]], *, seeded: Dict[str, bool] = None,
                expect_unsteered: Sequence[str] = (), class_fn=None):
    """Every request of ``step``: native selection == the lens's; the op steered
    exactly the selected rows (bitwise elsewhere)."""
    seeded = seeded or {}
    sel = native_selection(step, class_fn=class_fn)
    ref = h.ref_model(step.hidden, step.positions)
    for rid in step.req_ids:
        rows = step.rows_of(rid)
        name = rid.split(":")[0]
        mode = h.modes[name]
        pos = step.positions[rows].tolist()
        got = sel[rows]
        if mode is None or rid in expect_unsteered:
            want = torch.zeros(len(pos), dtype=torch.bool)
        else:
            want = lens_mask(mode, prompt_ids=prompts[rid], positions=pos,
                             skip_bos=bool(h.payloads[name].skip_bos), seeded=seeded.get(rid, False))
        assert torch.equal(got, want), (h.shape, rid, mode, pos, got.tolist(), want.tolist())
        out_r, ref_r = step.out[rows], ref[rows]
        assert torch.equal(out_r[~want], ref_r[~want]), (rid, "unselected rows must be bit-identical")
        for r in torch.nonzero(want).flatten().tolist():
            assert not torch.equal(out_r[r], ref_r[r]), (rid, r, "a selected row must be steered")
    return sel


# ---------------------------------------------------------------------------
# mask equivalence per mode, both runner shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["v1", "v2"])
def test_mask_equivalence_per_mode_vs_the_lens_over_a_scripted_engine(shape, tmp_path):
    """Default / seeded / multi-turn / resume renders on every slot, mixed
    batches, prefill and decode steps, staggered arrivals and finishes (index
    reuse, V1 condense), a plain LoRA and a no-LoRA request alongside — in the
    production world: every adapter resident at startup (``add_lora``; readiness
    requires it), so the admission judges at once."""
    h = _build(shape, tmp_path)
    prompts: Dict[str, List[int]] = {}
    seeded: Dict[str, bool] = {}

    def submit(name, render, *, max_new, computed=0, tag="", seeded_render=False):
        rid = f"{name}:{tag or render.__class__.__name__}{len(prompts)}"
        prompts[rid] = list(render)
        seeded[rid] = seeded_render
        h.sched.submit(rid, render, h.reqs[name], max_new=max_new, num_computed_tokens=computed)
        return rid

    # step 0: one request per slot on the default render, a plain LoRA and a base request
    submit("resp", DEFAULT, max_new=3)
    submit("last", DEFAULT, max_new=2)
    submit("pre", DEFAULT, max_new=2)
    submit("all", DEFAULT, max_new=3)
    submit("plain", DEFAULT, max_new=2)
    submit("none", DEFAULT, max_new=2)
    so = h.sched.step()
    steps = [h.runner.execute_model(so)]
    # step 1: the seeded and multi-turn renders arrive while the first batch decodes
    submit("resp", SEEDED, max_new=2, seeded_render=True)
    submit("resp", MULTI, max_new=2)
    submit("pre", SEEDED, max_new=1, seeded_render=True)
    steps.append(h.runner.execute_model(h.sched.step()))
    # step 2: a RESUMED prefill (3 prompt positions already cached; the header is in the slice)
    submit("resp", DEFAULT, max_new=2, computed=3, tag="resume")
    submit("last", MULTI, max_new=1)
    steps.append(h.runner.execute_model(h.sched.step()))
    # drain: finishes free indices that later arrivals reuse
    submit("all", SEEDED, max_new=2, seeded_render=True)
    submit("last", DEFAULT, max_new=1, tag="late")
    steps.extend(_run_all(h))
    executed = [s for s in steps if s is not None]
    assert len(executed) >= 5
    n_rows_checked = 0
    for step in executed:
        _check_step(h, step, prompts, seeded=seeded)
        n_rows_checked += step.num_tokens
    assert n_rows_checked > 60
    # the request set drained from the FACT table (never from the ledger: none was needed)
    assert len(h.bank.facts) == 0 and h.bank.facts.reclaimed == len(prompts)
    assert len(h.bank.ledger) == 0 and h.bank.ledger.fault is False and h.bank.n_faults == 0
    assert h.bank.header_ids == HEADER and h.bank.restricted_slots == 3
    assert h.bank.token_fact_updates == len(executed) and h.bank.steps_seen == len(steps)
    # the per-token buffers read the SENTINEL past the last row of the final step
    T = executed[-1].num_tokens
    assert h.bank.buffers.prompt_len_by_token[T:].eq(nr.POSITION_FACT_SENTINEL).all()
    assert h.bank.buffers.response_start_by_token[T:].eq(nr.POSITION_FACT_SENTINEL).all()


@pytest.mark.parametrize("shape", ["v1", "v2"])
def test_adapters_loading_inside_the_step_fail_closed_until_the_header_ids_are_known(shape, tmp_path):
    """The fake worker manager loads an adapter INSIDE the first step that routes
    to it (the pre-readiness window; production's readiness requires every
    adapter resident, so this never fires after startup). The first request per
    restricted slot is admitted while NO header ids are known -> its boundary is
    UNLOCATED, its rows are served unsteered and it is LEDGERED post-step (the
    fail-closed shape: a boundary that could not be located is never served as
    'all'); the ``prompt_last`` / ``prefill_all`` first requests need no boundary
    (their masks read the prompt length) and are served as the lens would, as
    are the all-positions and plain-LoRA requests beside them; every LATER
    request on the response slot is served exactly as the lens would."""
    h = _build(shape, tmp_path, preload=())
    prompts = {"resp:first": DEFAULT, "last:first": DEFAULT, "pre:first": DEFAULT, "all:first": DEFAULT,
               "plain:first": DEFAULT}
    for rid in prompts:
        h.sched.submit(rid, DEFAULT, h.reqs[rid.split(":")[0]], max_new=2)
    step0 = h.runner.execute_model(h.sched.step())
    first_restricted = ["resp:first"]
    _check_step(h, step0, prompts, expect_unsteered=first_restricted)
    assert h.bank.ledger.failed_ids == set(first_restricted) and h.bank.header_ids == HEADER
    entry = nr.take_request_failure("resp:first")
    assert entry["error_type"] == nr.ERROR_BOUNDARY_UNLOCATED and "no header ids are known" in entry["detail"]
    assert entry["cell"] == "resp"
    for rid in ("last:first", "pre:first", "all:first", "plain:first"):
        assert nr.take_request_failure(rid) is None
    # the second wave, with the slots resident: served as the lens would
    for name, render in (("resp", SEEDED), ("last", MULTI), ("pre", DEFAULT), ("all", MULTI)):
        prompts[f"{name}:second"] = list(render)
        h.sched.submit(f"{name}:second", render, h.reqs[name], max_new=2)
    for step in [s for s in _run_all(h) if s is not None]:
        _check_step(h, step, prompts, seeded={"resp:second": True}, expect_unsteered=first_restricted)
    assert len(h.bank.ledger) == 0 and h.bank.ledger.fault is False and h.bank.n_faults == 0


def test_v2_batch_reordering_and_preemption_keep_the_masks_position_exact(tmp_path):
    """V2 builds its batch from ``idx_mapping``: with the order REVERSED every
    other step the masks still follow each request's own facts; a request
    PREEMPTED mid-decode re-enters through ``scheduled_new_reqs`` (its facts
    overwritten, prompt unchanged) and its recompute — which the lens refuses
    as ``q > prompt_len`` — is served position-exactly: generated rows re-run
    as prefill rows fall in the response class and OUT of ``prompt_last`` /
    ``prefill_all``."""
    flip = {"k": 0}

    def order(ids):
        flip["k"] += 1
        return list(reversed(ids)) if flip["k"] % 2 else list(ids)

    h = _build("v2", tmp_path, batch_order=order)
    prompts = {}
    for i, (name, render) in enumerate([("resp", DEFAULT), ("last", MULTI), ("pre", SEEDED),
                                        ("all", DEFAULT), ("resp", MULTI)]):
        rid = f"{name}:{i}"
        prompts[rid] = list(render)
        h.sched.submit(rid, render, h.reqs[name], max_new=4)
    steps = _run_all(h, preempt_at={2: ["resp:0", "last:1"]})
    executed = [s for s in steps if s is not None]
    assert any(len(s.req_ids) > 1 and s.req_ids != sorted(s.req_ids) for s in executed), "no reordered batch"
    saw_recompute = False
    for step in executed:
        sel = native_selection(step)
        ref = h.ref_model(step.hidden, step.positions)
        for rid in step.req_ids:
            rows = step.rows_of(rid)
            pos = step.positions[rows]
            P = len(prompts[rid])
            mode = h.modes[rid.split(":")[0]]
            if mode == "response":
                want = pos >= nr._rfind(prompts[rid], HEADER)
            elif mode == "prompt_last":
                want = pos == P - 1
            elif mode == "prefill_all":
                want = (pos < P) & (pos != 0)          # skip_bos on the pre slot
            else:
                want = torch.ones_like(pos, dtype=torch.bool)
            if len(pos) > 1 and int(pos[-1]) >= P:
                saw_recompute = True                   # a preemption recompute: prefill rows past P
            assert torch.equal(sel[rows], want), (rid, pos.tolist(), sel[rows].tolist(), want.tolist())
            assert torch.equal(step.out[rows][~want], ref[rows][~want])
    assert saw_recompute
    assert len(h.bank.ledger) == 0 and h.bank.n_faults == 0 and len(h.bank.facts) == 0


# ---------------------------------------------------------------------------
# known-bads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["v1", "v2"])
def test_known_bad_headerless_prompt_on_a_response_slot_is_unsteered_and_ledgered_once(shape, tmp_path):
    h = _build(shape, tmp_path, preload=("resp", "all"))
    prompts = {"resp:bad": HEADERLESS, "all:same": HEADERLESS, "resp:good": DEFAULT}
    h.sched.submit("resp:bad", HEADERLESS, h.reqs["resp"], max_new=3)
    h.sched.submit("all:same", HEADERLESS, h.reqs["all"], max_new=3)
    h.sched.submit("resp:good", DEFAULT, h.reqs["resp"], max_new=2)
    steps = [s for s in _run_all(h) if s is not None]
    for step in steps:
        _check_step(h, step, prompts, expect_unsteered=["resp:bad"])
        if "resp:bad" in step.req_ids:                 # the buffer says UNLOCATED, not SENTINEL
            rows = step.rows_of("resp:bad")
            snap = step.extra["buffers"]
            assert snap.response_start_by_token[rows].eq(nr.UNLOCATED).all()
            assert snap.prompt_len_by_token[rows].eq(len(HEADERLESS)).all()
    # exactly ONE entry, for the response-slot request only, with the ledger entry shape
    assert len(h.bank.ledger) == 1 and h.bank.ledger.failed_ids == {"resp:bad"}
    entry = nr.take_request_failure("resp:bad")
    assert tuple(entry) == nr.CLAIM_KEYS
    assert entry["req_id"] == "resp:bad" and entry["lora_int_id"] == 1 and entry["cell"] == "resp"
    assert entry["error_type"] == nr.ERROR_BOUNDARY_UNLOCATED
    assert entry["prompt_len"] == len(HEADERLESS) and entry["header_len"] == len(HEADER)
    assert "occur nowhere" in entry["detail"]
    assert nr.take_request_failure("resp:bad") is None          # claimed once
    assert nr.take_request_failure("all:same") is None and nr.take_request_failure("resp:good") is None
    assert h.bank.ledger.fault is False and h.bank.n_faults == 0
    # the worker RPC hands the same payload out
    h.bank.ledger.record("x", lora_int_id=1, cell="resp", error_type=nr.ERROR_BOUNDARY_UNLOCATED,
                         prompt_len=5, header_len=3, detail="d")
    assert nw.NativeWorkerExtension.sisteer_take_request_failure(SimpleNamespace(), "x")["cell"] == "resp"


@pytest.mark.parametrize("shape", ["v1", "v2"])
def test_known_bad_headerless_prompt_judged_post_step_when_the_adapter_loads_inside_the_step(shape, tmp_path):
    """The adapter is NOT resident at admission (the on-demand load happens in
    ``set_active_adapters`` inside the step), and the request finishes after
    its ONE step (``max_new = 1``): the post-step judgement must still ledger
    it — otherwise the proxy's claim would read None and file the unsteered
    completion as served."""
    h = _build(shape, tmp_path, preload=())
    h.sched.submit("resp:bad", HEADERLESS, h.reqs["resp"], max_new=1)
    h.sched.submit("plain:p", HEADERLESS, h.reqs["plain"], max_new=1)
    so = h.sched.step()
    assert h.bank.record_for_lora(1) is None                   # not resident before the step
    step = h.runner.execute_model(so)
    assert h.bank.record_for_lora(1) is not None
    assert h.bank.ledger.failed_ids == {"resp:bad"} and len(h.bank._pending) == 0
    rows = step.rows_of("resp:bad")
    assert torch.equal(step.out[rows], h.ref_model(step.hidden, step.positions)[rows])
    entry = nr.take_request_failure("resp:bad")
    assert entry["error_type"] == nr.ERROR_BOUNDARY_UNLOCATED and entry["cell"] == "resp"
    assert nr.take_request_failure("plain:p") is None
    _run_all(h)


def test_known_bad_a_prompt_last_mask_off_by_one_refuses_the_assertion(tmp_path):
    h = _build("v1", tmp_path, preload=("last",))
    prompts = {"last:a": DEFAULT}
    h.sched.submit("last:a", DEFAULT, h.reqs["last"], max_new=2)
    step = h.runner.execute_model(h.sched.step())
    _check_step(h, step, prompts)                                 # the right class passes

    def off_by_one(mode_tok, pos, pl, rs):                        # ``== prompt_len`` (the first decode row)
        restricted = mode_tok != nr.MODE_ALL
        cls = (~restricted | ((mode_tok == nr.MODE_PROMPT_LAST) & (pos == pl)))
        return cls & (~restricted | (rs >= 0))

    with pytest.raises(AssertionError):
        _check_step(h, step, prompts, class_fn=off_by_one)
    _run_all(h)


@pytest.mark.parametrize("shape", ["v1", "v2"])
def test_known_bad_the_runner_prompt_len_cross_check_disagreeing_is_a_fault_and_ledgers(shape, tmp_path):
    """V2: ``req_states.prompt_len.np[idx_mapping]`` must equal the facts'
    ``prompt_len``; V1: ``input_batch.num_prompt_tokens``. Tampered between
    two steps, the next step records a fault, FAILS the request (ledgered on
    a restricted slot) and serves it unsteered from then on; the all-positions
    request beside it is untouched."""
    h = _build(shape, tmp_path, preload=("resp", "all"))
    prompts = {"resp:a": DEFAULT, "all:b": DEFAULT}
    h.sched.submit("resp:a", DEFAULT, h.reqs["resp"], max_new=3)
    h.sched.submit("all:b", DEFAULT, h.reqs["all"], max_new=3)
    step0 = h.runner.execute_model(h.sched.step())
    _check_step(h, step0, prompts)
    if shape == "v2":
        idx = h.runner.req_states.req_id_to_index["resp:a"]
        h.runner.req_states.prompt_len.np[idx] += 1
    else:
        idx = h.runner.input_batch.req_id_to_index["resp:a"]
        h.runner.input_batch.num_prompt_tokens[idx] += 1
    step1 = h.runner.execute_model(h.sched.step())
    assert h.bank.n_faults == 1 and "prompt length" in h.bank.faults[0]
    assert h.bank.ledger.failed_ids == {"resp:a"}
    entry = nr.take_request_failure("resp:a")
    assert entry["error_type"] == nr.ERROR_PROMPT_LEN_DISAGREES and entry["lora_int_id"] == 1
    assert entry["prompt_len"] is None                            # the facts were taken out
    for step in (step1, *[s for s in _run_all(h) if s is not None]):
        _check_step(h, step, prompts, expect_unsteered=["resp:a"])
    assert nr.take_request_failure("all:b") is None
    rep = wr.residual_slot_report(h.runner)
    assert rep["n_faults"] == 1 and len(rep["faults"]) == 1 and rep["fact_table"]["overflow"] is False


def test_known_bad_a_scheduled_request_with_no_facts_is_a_fault_and_fails_closed(tmp_path):
    h = _build("v2", tmp_path, preload=("resp",))
    h.sched.submit("resp:a", DEFAULT, h.reqs["resp"], max_new=2)
    step0 = h.runner.execute_model(h.sched.step())
    assert step0 is not None
    h.bank.facts.drop(["resp:a"])                                 # the facts vanish (a drift)
    step1 = h.runner.execute_model(h.sched.step())
    assert h.bank.n_faults == 1 and "no facts" in h.bank.faults[0]
    entry = nr.take_request_failure("resp:a")
    assert entry["error_type"] == nr.ERROR_FACTS_MISSING and entry["cell"] is None and entry["lora_int_id"] == 1
    rows = step1.rows_of("resp:a")
    assert torch.equal(step1.out[rows], h.ref_model(step1.hidden, step1.positions)[rows])
    _run_all(h)


# ---------------------------------------------------------------------------
# the ledger after readiness (fact 15, rev 4)
# ---------------------------------------------------------------------------


def test_ledger_overflow_after_readiness_latches_and_the_oldest_entry_still_refuses(tmp_path):
    """1024 header-less requests on the response slot, none claimed (the
    proxy post-check "not running"): the 1025th LATCHES ``ledger_fault``; the
    OLDEST request's entry is still present and a claim on its id returns the
    fault payload (its late completion refuses); a fresh id returns the fault
    payload; the all-positions slot keeps steering; the report carries the
    fault. KNOWN-BAD (rev 3): an evicting ledger returns None for the oldest
    id — that completion would have been returned."""
    h = _build("v2", tmp_path, preload=("resp", "all"), max_num_seqs=16, max_tokens=128)
    assert nr.REQUEST_FAILURES_CAP == 1024 and h.bank.ledger.cap == 1024
    ids = [f"resp:{i}" for i in range(1025)]
    for start in range(0, 1025, 8):
        for rid in ids[start:start + 8]:
            h.sched.submit(rid, HEADERLESS, h.reqs["resp"], max_new=1)
        step = h.runner.execute_model(h.sched.step())
        assert step is not None
    assert h.bank.ledger.fault is True and len(h.bank.ledger) == 1024
    assert "resp:0" in h.bank.ledger._entries and "resp:1024" not in h.bank.ledger._entries
    fault = {"fault": nr.RequestFailureLedger.FAULT, "unclaimed": 1024, "cap": 1024}
    assert nr.take_request_failure("resp:0") == fault                    # the oldest: refused, not None
    assert nr.take_request_failure("resp:1024") == fault                 # the one past the cap
    assert nr.take_request_failure("never-seen") == fault                # a fresh id
    assert len(h.bank.ledger) == 1024                                    # nothing left by claiming
    # the all-positions slot keeps steering while the fault is latched
    h.sched.submit("all:x", HEADERLESS, h.reqs["all"], max_new=1)
    step = h.runner.execute_model(h.sched.step())
    rows = step.rows_of("all:x")
    ref = h.ref_model(step.hidden, step.positions)
    assert all(not torch.equal(step.out[r], ref[r]) for r in range(rows.start, rows.stop))
    rep = wr.residual_slot_report(h.runner)
    assert rep["ledger_fault"] is True and rep["ledger_size"] == 1024 and rep["ledger_cap"] == 1024
    assert nw.NativeWorkerExtension.sisteer_take_request_failure(SimpleNamespace(), "resp:0") == fault
    _run_all(h)

    # the rev-3 KNOWN-BAD: an evicting ledger forgets the oldest failure
    class EvictingLedger:
        def __init__(self, cap):
            self.cap, self._e = cap, OrderedDict()

        def record(self, rid, **facts):
            if len(self._e) >= self.cap:
                self._e.popitem(last=False)                             # drop the OLDEST
            self._e[rid] = {"req_id": rid, **facts}

        def claim(self, rid):
            return self._e.pop(rid, None)

    bad = EvictingLedger(1024)
    for rid in ids:
        bad.record(rid, error_type="x")
    assert bad.claim("resp:0") is None            # the oldest completion would have been RETURNED
    assert nr.RequestFailureLedger(1024).claim("resp:0") is None  # (a fresh ledger: nothing recorded)


def test_ledger_semantics_are_the_stage_1_ones_plus_the_engine_id_join():
    """Stage 1's ledger semantics (claim pops; cap latches; nothing evicts) — with the entry naming
    the completion id it belongs to and the claim naming every engine id it popped."""
    led = nr.RequestFailureLedger(cap=2)
    led.record("a", lora_int_id=1, cell="c", error_type="E", prompt_len=3, header_len=1, detail="d")
    assert led.claim("a") == {"req_id": "a", "external_req_id": "a", "lora_int_id": 1, "cell": "c", "error_type": "E",
                              "prompt_len": 3, "header_len": 1, "detail": "d", "claimed_engine_ids": ["a"]}
    assert led.claim("a") is None
    led.record("a", x=1)
    led.record("b", x=2)
    led.record("c", x=3)                                          # at the cap: latch
    assert led.fault and led.claim("a") == {"fault": "ledger_overflow", "unclaimed": 2, "cap": 2}
    assert len(led) == 2 and led.report() == {"ledger_size": 2, "ledger_fault": True, "ledger_cap": 2}


# ---------------------------------------------------------------------------
# the fact table, the header cross-check, positions, the wrappers, the report
# ---------------------------------------------------------------------------


def test_a_readiness_sized_round_reclaims_every_fact_and_the_attestation_passes_it(tmp_path):
    """Known-bad: readiness refused with `fact_table dropped 107 request fact(s)` — exactly S each
    time. ONE readiness-sized round on the REAL bank through the fake runner: S = 107 requests on
    the response slot and the all-positions slot, each ADMITTED (its facts inserted by the
    pre-wrapper), its rows JUDGED and steered, and its facts RECLAIMED when it finishes — so after
    the round the table is EMPTY, `reclaimed == admitted == S`, nothing overflowed, no fault, an
    empty ledger. `reclaimed` is the reclamation counter (separate by name from `overflowed`), and
    the readiness attestation must PASS this report: the known-bad was a check that read it as a
    loss (serving_attestation.residual_slot_problems)."""
    from sisteer.capeval.serving import serving_attestation as sa
    S = 107
    h = _build("v2", tmp_path, preload=("resp", "all"), max_num_seqs=S, max_tokens=8192)
    # production-shaped ENGINE ids: the completion id plus vLLM 0.25.1's `-<8 hex>` suffix (the contract the
    # readiness attestation now pins — a raw id is the known-bad below)
    ids = [((f"resp:{i}" if i % 2 else f"all:{i}") + "-" + secrets.token_hex(4), "resp" if i % 2 else "all") for i in range(S)]
    for start in range(0, S, 16):
        for rid, name in ids[start:start + 16]:
            h.sched.submit(rid, DEFAULT, h.reqs[name], max_new=1)
        step = h.runner.execute_model(h.sched.step())
        assert step is not None
    steps = [s_ for s_ in _run_all(h) if s_ is not None]
    assert h.bank.facts.admitted == S and h.bank.facts.reclaimed == S and len(h.bank.facts) == 0
    assert not hasattr(h.bank.facts, "dropped")                     # the conflated counter is gone
    assert h.bank.facts.overflow is False and h.bank.facts.overflowed == 0
    assert h.bank.n_faults == 0 and len(h.bank.ledger) == 0 and h.bank.ledger.fault is False
    rep = wr.residual_slot_report(h.runner)
    assert rep["fact_table"] == {"size": 0, "cap": nr.REQUEST_FACTS_CAP, "overflow": False,
                                 "admitted": S, "overflowed": 0, "reclaimed": S, "off_contract": 0, "warmup": 0}
    assert rep["n_faults"] == 0 and rep["faults"] == []
    # the attestation over THIS report's facts: nothing about the fact table or a fault may refuse readiness
    base = {"armed": True, "op_name": sa.RESIDUAL_SLOT_OP_NAME, "op_registered": True, "runner_bound": True,
            "ledger_fault": rep["ledger_fault"], "ledger_size": rep["ledger_size"], "n_layers_model": 2,
            "n_layers_hooked": 2, "patched_classes": {"C": {"forward": "wrapped", "forward_wrapped": True}},
            "slots": {}, "fact_table": rep["fact_table"], "n_faults": rep["n_faults"], "faults": rep["faults"]}
    assert sa.residual_slot_problems([base], {"armed": True, "n_layers_model": 2, "adapters": {}}) == []
    # the known-bad the attestation DOES refuse: the same round with finished requests NOT reclaimed latches overflow
    stuck = {**base, "fact_table": {**rep["fact_table"], "size": S, "overflow": True, "reclaimed": 0}}
    probs = sa.residual_slot_problems([stuck], {"armed": True, "n_layers_model": 2, "adapters": {}})
    assert probs and all("overflowed" in p for p in probs), probs
    # the ENGINE-ID CONTRACT known-bad: ids admitted WITHOUT the engine suffix (a vLLM
    # bump, VLLM_DISABLE_REQUEST_ID_RANDOMIZATION) could never be claimed by the completion id — refused by name
    off = {**base, "fact_table": {**rep["fact_table"], "off_contract": 3}}
    probs = sa.residual_slot_problems([off], {"armed": True, "n_layers_model": 2, "adapters": {}})
    assert len(probs) == 1 and "OFF the pinned engine-id contract" in probs[0] and "3 request id(s)" in probs[0], probs


def test_the_engine_id_contract_is_one_rule_in_the_tree_and_a_raw_id_is_off_it():
    """vLLM 0.25.1 (verified against the native venv's source): AsyncLLM.add_request
    (async_llm.py:368) -> InputProcessor.assign_request_id (input_processor.py:223-240) mints the scheduler's id
    as f"{external_req_id}-{random_uuid():.8}" (utils.random_uuid is hex); ParentRequest.get_child_info
    (parallel_sampling.py:92) mints n>1 children f"{index}_{parent.request_id}" from the SUFFIXED parent. The
    native ledger's join is the lens's contract (serving_attestation.ENGINE_REQUEST_ID_SUFFIX_RE) plus the child
    prefix; a raw id (no suffix) is OFF the contract and readiness refuses a worker that admitted one."""
    from sisteer.capeval.serving import serving_attestation as sa
    eid = "chatcmpl-8f3a1c2e9b7d4f60-1a2b3c4d"
    assert nr.external_request_ids(eid) == (eid, "chatcmpl-8f3a1c2e9b7d4f60") == sa.external_request_id_candidates(eid)
    assert nr.external_request_ids("2_" + eid) == ("2_" + eid, "2_chatcmpl-8f3a1c2e9b7d4f60", eid, "chatcmpl-8f3a1c2e9b7d4f60")
    assert nr.external_request_ids("resp:first") == ("resp:first",) and nr.external_request_ids("") == () == nr.external_request_ids(None)
    assert nr.engine_id_on_contract(eid) and nr.engine_id_on_contract("0_" + eid)
    for raw in ("resp:first", "chatcmpl-8f3a1c2e9b7d4f60", "-1a2b3c4d", "chatcmpl-x-1A2B3C4D", "chatcmpl-x-1a2b3c4", "chatcmpl-x_1a2b3c4d", ""):
        assert not nr.engine_id_on_contract(raw), raw
    # the ONE meaning: the lens's is_engine_id_of agrees on the non-child forms
    assert sa.is_engine_id_of(eid, "chatcmpl-8f3a1c2e9b7d4f60") and not sa.is_engine_id_of("2_" + eid, "chatcmpl-8f3a1c2e9b7d4f60")
    # the table counts an off-contract SCHEDULER admission; a contract-shaped one it does not — and the facts
    # expand_token_facts inserts for rows the scheduler never announced (the engine's warm-up `_dummy_req_<i>`
    # rows: `_dummy_run` @0.25.1 builds SchedulerOutput.make_empty() with no scheduled_new_reqs) never count
    # (C5'' 513982: 107 warm-up rows read at readiness refused a healthy engine)
    t = nr.RequestFactTable(cap=4)
    t.admit(nr.RequestFacts(req_id="_dummy_req_0", lora_int_id=1, lora_name="c", prompt_len=3, response_start=1, boundary_detail=""), warmup=True)
    assert t.off_contract == 0 and t.admitted == 0 and t.warmup == 1
    t.note_scheduler_admission(eid); t.note_scheduler_admission("0_" + eid)
    assert t.off_contract == 0
    t.note_scheduler_admission("resp:raw")
    assert t.off_contract == 1 and t.report()["off_contract"] == 1 and t.report()["warmup"] == 1
    # the PINNED warm-up pattern (warmup.py:224 `_warmup_{i}_`; model_runner.py:535 `_dummy_req_{i}`): exactly these
    for wid in ("_warmup_0_", "_warmup_106_", "_dummy_req_0", "_dummy_req_42"):
        assert nr.ENGINE_WARMUP_ID_RE.fullmatch(wid), wid
    # a vLLM rename reads as an off-contract CLIENT id -> readiness refuses loudly, never silently
    for renamed in ("_warmup_0", "warmup_0_", "_warmup_a_", "_dummy_req_"):
        assert not nr.ENGINE_WARMUP_ID_RE.fullmatch(renamed) and not nr.engine_id_on_contract(renamed), renamed
    # a warm-up-looking id that WEARS the engine suffix is a client id (on the contract), never a warm-up row
    assert not nr.ENGINE_WARMUP_ID_RE.fullmatch("_warmup_0_-1a2b3c4d") and nr.engine_id_on_contract("_warmup_0_-1a2b3c4d")


def test_the_engines_own_warmup_rows_are_tabled_but_never_admitted_nor_held_to_the_contract(tmp_path):
    """C5'' 513982 (ac381446): vLLM 0.25.1's V2 warm-up (gpu/warmup.py:224) schedules max_num_seqs requests
    `_warmup_{i}_` THROUGH scheduled_new_reqs before readiness; the first cut of the engine-id pin counted all 107
    as off-contract and refused a healthy engine. They are tabled (their rows need facts), counted under `warmup`,
    never `admitted`, never held to the contract; a CLIENT id beside them still is."""
    h = _build("v2", tmp_path, preload=("resp", "all"))
    wids = [f"_warmup_{i}_" for i in range(5)]
    for i, wid in enumerate(wids):
        h.sched.submit(wid, DEFAULT, h.reqs["all" if i % 2 else "resp"], max_new=1)
    h.sched.submit(f"chatcmpl-{secrets.token_hex(8)}-{secrets.token_hex(4)}", DEFAULT, h.reqs["resp"], max_new=1)
    h.sched.submit("resp:raw-client-id", DEFAULT, h.reqs["resp"], max_new=1)
    steps = [s_ for s_ in [h.runner.execute_model(h.sched.step())] + list(_run_all(h)) if s_ is not None]
    assert steps
    rep = h.bank.facts.report()
    assert rep["warmup"] == 5 and rep["admitted"] == 2 and rep["off_contract"] == 1 and rep["reclaimed"] == 7
    assert len(h.bank.facts) == 0 and h.bank.n_faults == 0


def test_end_to_end_a_failure_ledgered_under_the_engine_id_is_claimed_by_the_completion_id(tmp_path):
    """UNVERIFIED on hardware: no boundary failure occurred in the hardware probes, so the claim
    path never ran there. The fake worker manager loads the adapters INSIDE the first step
    (pre-readiness), so the first restricted request's boundary is UNLOCATED and LEDGERED under the
    id the SCHEDULER handed the worker — production-shaped: the completion id plus vLLM's random
    `-<8 hex>` suffix, and a `<i>_` prefix for an n>1 child. The proxy claims by the completion id
    (native_worker.sisteer_take_request_failure -> take_request_failure): the OLD exact-key pop
    returned None for every one of these — an UNSTEERED completion stamped as the cell; now every
    failure the completion owns is popped."""
    h = _build("v2", tmp_path, preload=())
    ext, ext2, ext3 = "chatcmpl-" + secrets.token_hex(8), "chatcmpl-" + secrets.token_hex(8), "chatcmpl-" + secrets.token_hex(8)
    eng = f"{ext}-{secrets.token_hex(4)}"                                # n == 1
    child0, child1 = f"0_{ext2}-{secrets.token_hex(4)}", f"1_{ext2}-{secrets.token_hex(4)}"   # n == 2 children
    clean = f"{ext3}-{secrets.token_hex(4)}"                              # the all-positions slot: no boundary to locate
    prompts = {eng: DEFAULT, child0: DEFAULT, child1: DEFAULT, clean: DEFAULT}
    for rid in (eng, child0, child1):
        h.sched.submit(rid, DEFAULT, h.reqs["resp"], max_new=2)
    h.sched.submit(clean, DEFAULT, h.reqs["all"], max_new=2)
    step0 = h.runner.execute_model(h.sched.step())
    _check_step(h, step0, {**prompts}, expect_unsteered=[eng, child0, child1]) if False else None   # ids carry ':'-free names
    assert h.bank.ledger.failed_ids == {eng, child0, child1}, h.bank.ledger.failed_ids
    assert h.bank.facts.off_contract == 0                                   # every id wore the contract
    # the exact-key world (the defect): the completion ids are NOT ledger keys
    assert ext not in h.bank.ledger._entries and ext2 not in h.bank.ledger._entries and eng in h.bank.ledger._entries
    # a claim for ANOTHER completion pops nothing
    assert nr.take_request_failure("chatcmpl-" + secrets.token_hex(8)) is None and len(h.bank.ledger) == 3
    # the completion id claims its engine-id failure, named
    entry = nr.take_request_failure(ext)
    assert entry is not None and entry["error_type"] == nr.ERROR_BOUNDARY_UNLOCATED and entry["cell"] == "resp"
    assert entry["req_id"] == eng and entry["external_req_id"] == ext and entry["claimed_engine_ids"] == [eng]
    assert nr.take_request_failure(ext) is None and nr.take_request_failure(eng) is None      # claimed once
    # an n>1 completion: BOTH children's failures are claimed by the one completion id
    entry2 = nr.take_request_failure(ext2)
    assert entry2 is not None and sorted(entry2["claimed_engine_ids"]) == sorted([child0, child1]) and entry2["external_req_id"] == ext2
    assert len(h.bank.ledger) == 0 and nr.take_request_failure(ext3) is None and h.bank.ledger.fault is False
    # the worker RPC is the same body
    assert nw.NativeWorkerExtension.sisteer_take_request_failure(SimpleNamespace(), ext) is None


def test_fact_table_cap_and_overflow_fail_closed(tmp_path):
    h = _build("v2", tmp_path, preload=("resp", "all"))
    h.bank.facts.cap = 2
    prompts = {"resp:a": DEFAULT, "all:b": DEFAULT, "resp:c": DEFAULT}
    for rid in prompts:
        h.sched.submit(rid, DEFAULT, h.reqs[rid.split(":")[0]], max_new=2)
    step = h.runner.execute_model(h.sched.step())
    ft = h.bank.facts.report()
    # (the harness's raw ids wear no engine suffix: off_contract counts them — a readiness refusal in production,
    # pinned in test_a_readiness_sized_round_...; here the cap/overflow facts are the subject)
    assert ft == {"size": 2, "cap": 2, "overflow": True, "admitted": 2, "overflowed": 1, "reclaimed": 0, "off_contract": 2, "warmup": 0}
    # the admission's overflow fault, then one fault per step the request is scheduled without facts
    assert h.bank.ledger.failed_ids == {"resp:c"} and h.bank.n_faults == 2
    assert "overflow" in h.bank.faults[0] and "no facts" in h.bank.faults[1]
    _check_step(h, step, prompts, expect_unsteered=["resp:c"])
    entry = nr.take_request_failure("resp:c")
    assert entry["error_type"] == nr.ERROR_FACTS_OVERFLOW and entry["cell"] == "resp"   # the FIRST failure names it
    rep = wr.residual_slot_report(h.runner)
    assert rep["fact_table"]["overflow"] is True and rep["bank"]["fact_table"] == rep["fact_table"]
    for later in [s2 for s2 in _run_all(h) if s2 is not None]:
        _check_step(h, later, prompts, expect_unsteered=["resp:c"])
    assert len(h.bank.facts) == 0 and h.bank.facts.overflow is True   # the latch stays for readiness
    assert h.bank.facts.overflowed == 1 and h.bank.n_faults >= 2


def test_load_slot_cross_checks_header_ids_and_refuses_a_restricted_payload_without_them():
    nr.reset_for_tests()
    b = nr.ResidualSlotBank(n_slots=3, n_boundaries=L, hidden_size=D, max_tokens=16, device="cpu",
                            model_dtype=BF16)
    nr._publish_bank(b)
    resp = make_payload("psr", d=D, layers=[0], n_layers_model=L, seed=1, mode="response", header_ids=HEADER)
    b.load_slot(0, resp, lora_int_id=1)
    assert b.header_ids == HEADER and int(b.buffers.mode[0]) == nr.MODE_RESPONSE and b.restricted_slots == 1
    other = make_payload("static_r", d=D, layers=[1], n_layers_model=L, seed=2, mode="prefill_all",
                         header_ids=(900, 901))
    with pytest.raises(nr.ResidualSlotError, match="disagree with the header ids"):
        b.load_slot(1, other, lora_int_id=2)
    assert b.records[1] is None and not bool(b.buffers.active[1].any())   # refused: an EMPTY slot
    agreeing = make_payload("static_r", d=D, layers=[1], n_layers_model=L, seed=2, mode="prompt_last",
                            header_ids=HEADER)
    b.load_slot(1, agreeing, lora_int_id=2)
    assert int(b.buffers.mode[1]) == nr.MODE_PROMPT_LAST and b.restricted_slots == 2
    # an all-positions payload carrying DIFFERENT ids is a template disagreement too
    all_other = make_payload("psr", d=D, layers=[2], n_layers_model=L, seed=3, mode="all", header_ids=(1, 2, 3))
    with pytest.raises(nr.ResidualSlotError, match="disagree"):
        b.load_slot(2, all_other, lora_int_id=3)
    # a BOUNDARY-mode payload with NO header ids refuses at activation; a static_r mode (the
    # exporter derives ids for the response mode only) needs none and loads
    blk = dict(resp.block(), layers=[2], boundaries=[2], response_header_len=3, response_header_ids=[])
    bare_resp = nr.payload_from_block_and_tensors(
        blk, {k.replace("layers.0.", "layers.2."): v for k, v in resp.tensors().items()})
    with pytest.raises(nr.ResidualSlotError, match="response_header_ids"):
        b.load_slot(2, bare_resp, lora_int_id=3)
    bare = make_payload("static_r", d=D, layers=[2], n_layers_model=L, seed=3, mode="prefill_all")
    b.load_slot(2, bare, lora_int_id=3)
    assert int(b.buffers.mode[2]) == nr.MODE_PREFILL_ALL and b.restricted_slots == 3
    b.clear_slot(2)
    # clearing resets the mode code and the restricted count; the header ids are sticky
    b.clear_slot(0)
    assert int(b.buffers.mode[0]) == nr.MODE_ALL and b.restricted_slots == 1 and b.header_ids == HEADER
    # the block validator: ids and width must name one header
    blk = resp.block()
    blk["response_header_ids"] = [900, 901]
    with pytest.raises(nr.ResidualSlotError, match="response_header_len"):
        nr.payload_from_block_and_tensors(blk, resp.tensors())


def test_stage_2_guard_still_refuses_restricted_modes_when_unarmed(monkeypatch):
    nr.reset_for_tests()
    b = nr.ResidualSlotBank(n_slots=1, n_boundaries=L, hidden_size=D, max_tokens=16, device="cpu",
                            model_dtype=BF16)
    nr._publish_bank(b)
    resp = make_payload("psr", d=D, layers=[0], n_layers_model=L, seed=1, mode="response", header_ids=HEADER)
    monkeypatch.setattr(nr, "STAGE2_POSITION_CLASSES", False)
    with pytest.raises(nr.ResidualSlotError, match="stage-2 position class"):
        b.load_slot(0, resp, lora_int_id=1)
    monkeypatch.setattr(nr, "STAGE2_POSITION_CLASSES", True)
    b.load_slot(0, resp, lora_int_id=1)
    assert b.records[0] is not None
    # an unknown mode never reaches the slot (the block validator refuses it first)
    blk = resp.block()
    blk["position_mode"] = "last"
    with pytest.raises(nr.ResidualSlotError, match="position_mode"):
        nr.payload_from_block_and_tensors(blk, resp.tensors())


def test_a_forward_without_positions_refuses_while_a_restricted_slot_is_loaded():
    nr.reset_for_tests()
    b = nr.ResidualSlotBank(n_slots=2, n_boundaries=L, hidden_size=D, max_tokens=8, device="cpu",
                            model_dtype=BF16)
    nr._publish_bank(b)
    # SEEDED: an unseeded draw made the assertion below depend on the global RNG state the shard's earlier
    # tests left behind — with all four ReLU gates <= 0 the all-mode slot legitimately writes nothing
    # (3 of 400 global states; deterministic per shard composition: main gate 015489fe shard 5, 2026-09-18).
    # Generator seed 7 gives a positive gate on all four rows against the seed=1 payload.
    h = torch.randn(4, D, generator=torch.Generator().manual_seed(7)).to(BF16)
    b.load_slot(0, make_payload("psr", d=D, layers=[0], n_layers_model=L, seed=1, mode="all"), lora_int_id=1)
    b.update_mapping(torch.tensor([0, 0, 0, 0]), 4)
    written = b.apply(h, None, None, 0)
    assert (written != h).any(dim=1).all(), (written != h).any(dim=1)     # an all-mode slot needs none; every row written
    b.load_slot(1, make_payload("psr", d=D, layers=[0], n_layers_model=L, seed=2, mode="response",
                                header_ids=HEADER), lora_int_id=2)
    with pytest.raises(nr.ResidualSlotError, match="carries no positions"):
        b.apply(h, None, None, 0)
    # the pure function alone leaves restricted rows OUT (defence in depth)
    mode_tok = torch.tensor([0, 1, 2, 3], dtype=torch.int8)
    assert nr.position_class_selection(mode_tok, None, torch.zeros(4, dtype=torch.long),
                                       torch.zeros(4, dtype=torch.long)).tolist() == [True, False, False, False]
    # the class per row with positions, including the UNLOCATED / SENTINEL rows
    pos = torch.tensor([0, 3, 4, 6, 7, 2, 9, 6])
    pl = torch.tensor([7, 7, 7, 7, 7, nr.POSITION_FACT_SENTINEL, 7, 7])
    rs = torch.tensor([4, 4, 4, 4, 4, nr.POSITION_FACT_SENTINEL, nr.UNLOCATED, nr.UNLOCATED])
    m = torch.ones(8, dtype=torch.int8)                                        # response: needs the boundary
    assert nr.position_class_selection(m, pos, pl, rs).tolist() == [False, False, True, True, True, False, False, False]
    m = torch.full((8,), 2, dtype=torch.int8)                                  # prompt_last: prompt_len only
    assert nr.position_class_selection(m, pos, pl, rs).tolist() == [False, False, False, True, False, False, False, True]
    m = torch.full((8,), 3, dtype=torch.int8)                                  # prefill_all: prompt_len only
    assert nr.position_class_selection(m, pos, pl, rs).tolist() == [True, True, True, True, False, False, False, True]
    m = torch.zeros(8, dtype=torch.int8)                                       # all: facts irrelevant
    assert nr.position_class_selection(m, pos, pl, rs).all()
    # 2-D (M-RoPE) positions take row 0 through the op
    b.clear_slot(1)
    pos2 = torch.stack([torch.arange(4), torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long)])
    assert torch.equal(b.apply(h, None, pos2, 0), b.apply(h, None, torch.arange(4), 0))


def test_wrappers_are_idempotent_and_compose_with_the_routing_recorder_in_either_order(tmp_path):
    h = _build("v2", tmp_path, preload=("resp",))
    rc = h.rc
    for cls in (rc.RunnerV1, rc.RunnerV2):
        assert nr._is_wrapped(cls.__dict__["execute_model"])
        assert h.rep.machinery[f"{nr._qual(cls)}.execute_model"] == "wrapped"
    assert nr._is_wrapped(h.fk.LoRAModelRunnerMixin.__dict__["set_active_loras"])
    assert nr._is_wrapped(rc.LoraState.__dict__["make_lora_inputs"])
    # the fakes are nested classes, so the machinery keys carry the factory's qualname prefix
    assert _machinery(h.rep, ".set_active_loras") == "wrapped" and _machinery(h.rep, ".make_lora_inputs") == "wrapped"
    wrapped = (rc.RunnerV2.__dict__["execute_model"], rc.LoraState.__dict__["make_lora_inputs"],
               h.fk.LoRAModelRunnerMixin.__dict__["set_active_loras"])
    for w in wrapped:                                                # the originals are snapshotted
        assert callable(w._sisteer_orig) and not nr._is_wrapped(w._sisteer_orig)
    rep2 = nr.install(h.targets)                                     # a second install: a no-op
    assert rep2.installs == 2
    assert (rc.RunnerV2.__dict__["execute_model"], rc.LoraState.__dict__["make_lora_inputs"],
            h.fk.LoRAModelRunnerMixin.__dict__["set_active_loras"]) == wrapped
    # order A: the class patch (import time) then the recorder arms on the INSTANCE
    rec = wr.LoraRoutingRecorder(h.runner)
    assert rec.arm("epoch-a")["armed"] is True
    h.sched.submit("resp:a", DEFAULT, h.reqs["resp"], max_new=2)
    steps = [s for s in _run_all(h) if s is not None]
    assert len(steps) == 2 and h.bank.steps_seen == 3 and h.bank.facts.admitted == 1
    obs = rec.observations_payload("epoch-a")
    assert obs["observations"]["resp:a"] == {"lora_int_id": 1, "lora_name": "resp"} and obs["n_steps"] == 3
    assert rec.disarm("epoch-a")["restored"] is True
    assert "execute_model" not in vars(h.runner)                     # the class wrapper is reachable again
    # order B: the recorder arms first on a FRESH runner class, the class patch lands after — the
    # instance wrapper shadows the class attribute while armed (why the patch installs at import);
    # the runner still runs, the recorder records, and disarming exposes the class wrapper
    fk2, layers2 = make_fake_lora_classes(), make_fake_layer_classes()
    rc2 = make_fake_runner_classes(fk2)
    nr.reset_for_tests()
    vc = fake_vllm_config(hidden_size=D, num_layers=L, dtype=BF16, max_loras=S)
    runner2 = rc2.RunnerV1(vc, "cpu", layers2.LlamaDecoderLayer, arch="llama", n_layers=L, hidden=HiddenSource(D))
    rec2 = wr.LoraRoutingRecorder(runner2)
    rec2.arm("epoch-b")
    targets2 = nr.PatchTargets(
        lora_model_cls=fk2.LoRAModel, manager_cls=fk2.LoRAModelManager, punica_cls=fk2.PunicaWrapperBase,
        runner_mixin_cls=fk2.LoRAModelRunnerMixin, decoder_layer_classes=tuple(layers2.TABLE),
        model_info=fk2.model_info, op=nr.ensure_op_registered(), runner_classes=(rc2.RunnerV1,),
        lora_state_cls=rc2.LoraState)
    nr.install(targets2)
    runner2.load_model()
    sched = FakeScheduler()
    sched.submit("none:a", DEFAULT, None, max_new=1)
    step = runner2.execute_model(sched.step())
    assert step is not None and rec2.observations_payload("epoch-b")["observations"]["none:a"]["lora_int_id"] == 0
    assert rec2.disarm("epoch-b")["restored"] is True
    step = runner2.execute_model(sched.step())                        # the finish step, now through the class wrapper
    assert nr.bank().steps_seen == 1 and step is None


def test_execute_model_refuses_with_no_bank_or_an_unbound_runner(tmp_path):
    fk, layers = make_fake_lora_classes(), make_fake_layer_classes()
    rc = make_fake_runner_classes(fk)
    nr.reset_for_tests()
    targets = nr.PatchTargets(
        lora_model_cls=fk.LoRAModel, manager_cls=fk.LoRAModelManager, punica_cls=fk.PunicaWrapperBase,
        runner_mixin_cls=fk.LoRAModelRunnerMixin, decoder_layer_classes=tuple(layers.TABLE),
        model_info=fk.model_info, op=nr.ensure_op_registered(), runner_classes=(rc.RunnerV1, rc.RunnerV2),
        lora_state_cls=rc.LoraState)
    nr.install(targets)
    vc = fake_vllm_config(hidden_size=D, num_layers=L, dtype=BF16, max_loras=S)
    runner = rc.RunnerV2(vc, "cpu", layers.LlamaDecoderLayer, arch="llama", n_layers=L, hidden=HiddenSource(D))
    sched = FakeScheduler()
    sched.submit("none:a", DEFAULT, None, max_new=1)
    so = sched.step()
    with pytest.raises(nr.ResidualSlotError, match="no ResidualSlotBank"):     # before load_model
        runner.execute_model(so)
    b = nr.ResidualSlotBank(n_slots=1, n_boundaries=L, hidden_size=D, max_tokens=8, device="cpu",
                            model_dtype=BF16)
    nr._publish_bank(b)
    with pytest.raises(nr.ResidualSlotError, match="bound to no model runner"):  # a bank, no binding
        runner.execute_model(so)
    with pytest.raises(nr.ResidualSlotError, match="bound to no model runner"):
        rc.LoraState(4).make_lora_inputs(["x"], np.array([0]), np.array([1]))
    # a dummy run touches no facts
    b.bind_runner(runner)
    runner.model = FakeModel(layers.LlamaDecoderLayer, L, vc, arch="llama")
    runner.execute_model(so, dummy_run=True)
    assert b.steps_seen == 0 and len(b.facts) == 0


def test_the_report_carries_the_stage_2_facts(tmp_path):
    h = _build("v1", tmp_path, preload=("resp", "last", "pre", "all"))
    h.sched.submit("resp:a", DEFAULT, h.reqs["resp"], max_new=2)
    _run_all(h)
    rep = wr.residual_slot_report(h.runner)
    assert rep["stage2_position_classes"] is True
    assert rep["position_modes"] == {"all": 0, "response": 1, "prompt_last": 2, "prefill_all": 3}
    assert rep["header_ids"] == list(HEADER) and rep["header_len"] == 3 and rep["restricted_slots"] == 3
    assert rep["fact_table"] == {"size": 0, "cap": nr.REQUEST_FACTS_CAP, "overflow": False, "admitted": 1,
                                 "overflowed": 0, "reclaimed": 1, "off_contract": 1, "warmup": 0}
    assert rep["n_faults"] == 0 and rep["faults"] == []
    assert rep["ledger_size"] == 0 and rep["ledger_fault"] is False and rep["ledger_cap"] == 1024
    assert {rep["slots"][k]["mode"] for k in ("1", "2", "3", "4")} == set(nr.POSITION_MODES)
    m = rep["machinery"]
    assert _machinery(h.rep, ".set_active_loras") == "wrapped" and _machinery(h.rep, ".make_lora_inputs") == "wrapped"
    assert {k for k in m if k.endswith(".execute_model")} == {f"{nr._qual(h.rc.RunnerV1)}.execute_model",
                                                             f"{nr._qual(h.rc.RunnerV2)}.execute_model"}
    assert rep["bank"]["token_fact_updates"] == 2 and rep["bank"]["steps_seen"] == 3
    assert rep["runner_bound"] is True
    # unarmed / bankless: the same keys, empty
    nr.reset_for_tests()
    bare = wr.residual_slot_report(None)
    for key in ("header_ids", "header_len", "restricted_slots", "fact_table", "n_faults", "faults",
                "position_modes", "stage2_position_classes"):
        assert key in bare
    assert bare["fact_table"] is None and bare["faults"] == [] and bare["header_ids"] is None
    assert nw.NativeWorkerExtension.sisteer_residual_slot_report(SimpleNamespace(model_runner=None)) == bare


def test_constants_the_proxy_and_the_exporter_read():
    assert nr.POSITION_FACT_SENTINEL == -1 and nr.UNLOCATED == -2
    assert nr.MODE_CODES == {"all": 0, "response": 1, "prompt_last": 2, "prefill_all": 3}
    assert nr.RESTRICTED_MODES == ("response", "prompt_last", "prefill_all")
    assert nr.BOUNDARY_MODES == ("response",)
    assert nr.LEDGER_ENTRY_KEYS == ("req_id", "external_req_id", "lora_int_id", "cell", "error_type", "prompt_len",
                                    "header_len", "detail")
    assert nr.RequestFailureLedger.FAULT == "ledger_overflow"
    assert nr.STAGE2_POSITION_CLASSES is True   # (the defense_cells pin lives in the constants-agree test)
