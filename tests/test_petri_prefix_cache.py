"""PREFIX CACHING ON, WITH THE ISOLATION PROOF.

The settled leakage answer (read out of the pinned vLLM v0.25.1 block
manager): KV blocks are keyed on the parent-hash chain + token ids +
extra keys (LoRA NAME every block; the request ``cache_salt`` on block
0, chained into every descendant) — the steering selector is in the key
NOWHERE, and cached tokens skip the forward pass, so the lens hooks
never fire on them. Cross-cell reuse of an identical token prefix would
silently serve the WRONG intervention's KV.

Covers: the proxy's per-request ``cache_salt`` (cell-stable for
unhooked/LoRA legs, REQUEST-UNIQUE for lens legs; distinct across
cells; a client-supplied salt refuses); the reworked Phase-0 assertion
7 (caching ON needs the live isolation proof — arm bites, twin
unchanged warm-vs-cold; OFF keeps the legacy repeat pair); the plan
lattice's RESUMED segment class (tolerated for selector-less requests
exactly when the live engine has caching on, ctx=None; a TRACKED resume
refuses unless the cell's ops pass lens_adapters.resumed_prefill_ok —
the 2026-09-06 resumed-prefill landing, tests/test_lens_resumed_prefill.py
— and then serves as a RESUMED PREFILL); the worker's row-local
salt-discipline guard (now biting only for a non-resumable cell's
scale-0 twin, the only non-resumable object that path can carry); and
the composed/attested cache state (build_server_launch default ON,
expected_scheduler_state + scheduler_config_report round trip).

CPU-only; no vLLM, no model loads, no network."""

import asyncio

import pytest
import torch

from sisteer.capeval import manifest as cm
from sisteer.capeval.petri import defense_cells as dc
from sisteer.capeval.serving import lens_adapters as la
from sisteer.capeval.serving import lens_batch as lb
from sisteer.capeval.serving import lens_steer as ls
from sisteer.capeval.serving import lens_worker as lw
from sisteer.capeval.serving import proxy as px

from test_petri_stacked import (_behavior_files, _entries, _spec,
                                CONSTITUTION_PREFIX, SAFETY_PREFIX,
                                STACKED_A, STACKED_B, BASE_MODEL)


# --- the proxy's salt ------------------------------------------------------------


class _StubBackend:
    def __init__(self):
        self.payloads = []

    async def take_request_failure(self, request_id):
        # the lens post-check surface (per-request containment); this
        # stub never poisons a request.
        return None

    async def chat_completions(self, payload):
        self.payloads.append(dict(payload))
        return {"id": "stub", "model": payload.get("model"),
                "choices": [{"index": 0, "message":
                             {"role": "assistant", "content": "ok"}}]}


def _proxy(tmp_path, entries=None):
    spec = _spec(serving_mode="interleaved")
    entries = entries if entries is not None else _entries(tmp_path)
    cs = dc.enumerate_cells(
        spec, dc.MappingCellSource(entries),
        behavior_files=_behavior_files(spec),
        prompt_files={"auditor": b"a", "judge": b"j"})
    source = dc.MappingCellSource(entries)
    backend = _StubBackend()
    proxy = px.RoutingProxy(
        px.routes_from_cells(cs), backend, base_model=BASE_MODEL,
        serving_mode="interleaved",
        prompted_prefixes=px.prompted_prefix_specs(cs, source))
    return proxy, backend


def _chat(proxy, vmodel, messages, **extra):
    return asyncio.run(proxy.handle_chat_completions(
        {"model": vmodel, "messages": messages, "max_tokens": 8, **extra}))


USER = [{"role": "user", "content": "hello"}]
PROMPTED = [{"role": "system", "content": SAFETY_PREFIX.decode()},
            {"role": "user", "content": "hello"}]


def test_every_leg_gets_a_proxy_owned_salt_in_its_own_namespace(tmp_path):
    proxy, backend = _proxy(tmp_path)
    _chat(proxy, "petri__base", USER)
    _chat(proxy, "petri__prompted__safety", PROMPTED)
    _chat(proxy, "petri__alphasteer", USER)
    salts = [p["cache_salt"] for p in backend.payloads]
    assert all(isinstance(x, str) and x for x in salts)
    assert len(set(salts)) == 3  # one namespace per cell — never shared


def test_unhooked_legs_reuse_their_cell_stable_salt(tmp_path):
    proxy, backend = _proxy(tmp_path)
    _chat(proxy, "petri__prompted__safety", PROMPTED)
    _chat(proxy, "petri__prompted__safety", PROMPTED)
    a, b = (p["cache_salt"] for p in backend.payloads)
    # the whole point of enabling: within-cell prefix reuse for the
    # prompted legs (the ~5.9k CONSTITUTION prefix was the R7 cost)
    assert a == b


def test_lens_legs_get_a_request_unique_salt(tmp_path):
    proxy, backend = _proxy(tmp_path)
    _chat(proxy, "petri__si_safety__psr_kl_all", USER)
    _chat(proxy, "petri__si_safety__psr_kl_all", USER)
    a, b = (p["cache_salt"] for p in backend.payloads)
    # a steered prefill must never resume from cache: cached rows skip
    # the forward and the hooks would silently not fire on them
    assert a != b
    stacked = [{"role": "system", "content": SAFETY_PREFIX.decode()},
               {"role": "user", "content": "hello"}]
    _chat(proxy, "petri__" + STACKED_A, stacked)
    _chat(proxy, "petri__" + STACKED_A, stacked)
    c, d = (p["cache_salt"] for p in backend.payloads[2:])
    assert c != d  # stacked cells ride the lens path: unique too


def test_lora_leg_salt_is_cell_stable(tmp_path):
    # no stacked cell rides LoRA any more (enumerate_cells refuses a
    # stacked entry marked lora), so the LoRA leg here is a PLAIN object
    # cell marked lora — the legacy lens+lora hybrid table the default
    # enumeration still admits
    ent = _entries(tmp_path)
    ent["alphasteer"] = dc.RegistryEntry(
        cell_id="alphasteer", kind=la.SYNTHETIC_KIND, si_id=None,
        object_path=ent["alphasteer"].object_path, serving_path="lora",
        resolved_spec={"kind": la.SYNTHETIC_KIND, "layer": 3})
    proxy, backend = _proxy(tmp_path, ent)
    _chat(proxy, "petri__alphasteer", USER)   # the LoRA-path object cell
    _chat(proxy, "petri__alphasteer", USER)
    a, b = (p["cache_salt"] for p in backend.payloads)
    # weight-level serving has no hooks to miss: within-cell reuse is
    # safe (and the LoRA NAME is natively part of every block's key)
    assert a == b


def test_client_supplied_cache_salt_refuses(tmp_path):
    proxy, backend = _proxy(tmp_path)
    with pytest.raises(px.ProxyRefusal, match="reserved selector key"):
        _chat(proxy, "petri__base", USER, cache_salt="mine")
    assert not backend.payloads


# --- the proxy-level salt-discipline guard -----------------------------------------
#
# The engine-side guard on the same discipline is FATAL (a steered
# resume raises inside EngineCore's generate path and the whole vLLM
# server dies with it — job 227309), so the PROXY must refuse first,
# with a per-request 409, and KEEP SERVING. Through the unpatched
# public surface the collision is impossible by construction
# (request-unique uuid4 lens salts), so these tests pin uuid4 to
# simulate exactly the construction bug / collision the guard exists
# to catch.


def test_steered_salt_reuse_refused_at_proxy_which_keeps_serving(
        tmp_path, monkeypatch):
    import uuid as _uuid

    proxy, backend = _proxy(tmp_path)
    monkeypatch.setattr(px.uuid, "uuid4", lambda: _uuid.UUID(int=7))
    _chat(proxy, "petri__si_safety__psr_kl_all", USER)  # claims the salt
    with pytest.raises(px.ProxyRefusal,
                       match="STEERED request would reuse") as ei:
        _chat(proxy, "petri__si_safety__psr_kl_all", USER)
    # a 4xx naming the cell — never the engine-fatal in-worker refusal
    assert ei.value.http_status == 409
    assert ei.value.code == "cache_salt_collision"
    assert "si_safety__psr_kl_all" in str(ei.value)
    # the refused request never went upstream: the engine never saw it
    assert len(backend.payloads) == 1
    monkeypatch.undo()  # real request-unique salts again
    # THE SURVIVAL PROPERTY: the SAME proxy serves normal traffic after
    # the guard trip — the trip cost one request, not the run.
    _chat(proxy, "petri__si_safety__psr_kl_all", USER)
    _chat(proxy, "petri__base", USER)
    assert len(backend.payloads) == 3


def test_unsteered_request_riding_a_steered_salt_refuses(tmp_path):
    # the SILENT wrong-serve direction: an unsteered request resuming a
    # steered request's blocks would inherit the steered KV under its
    # own identity, and no engine guard even fires there (hooks are not
    # active on a selector-less request). Unreachable through the
    # public surface by construction — an unhooked cell-stable salt can
    # equal neither a uuid-suffixed lens salt nor (since the
    # resumed-prefill landing) a `|lens`-marked cell-stable one — so the
    # guard is driven directly, the way a construction bug would drive it.
    proxy, _ = _proxy(tmp_path)
    route = next(iter(proxy._routes.values()))
    proxy._guard_cache_salt(route, salt="petri|x|y|abc", lens=True,
                            regime=px.SALT_REGIME_REQUEST_UNIQUE)
    with pytest.raises(px.ProxyRefusal,
                       match="UNSTEERED request would reuse") as ei:
        proxy._guard_cache_salt(route, salt="petri|x|y|abc", lens=False,
                                regime=px.SALT_REGIME_CELL_STABLE)
    assert ei.value.http_status == 409
    assert ei.value.code == "cache_salt_collision"
    # the safe direction stays open: stable-salt reuse by UNSTEERED
    # requests is the whole point of enabling caching
    proxy._guard_cache_salt(route, salt="petri|x|stable", lens=False,
                            regime=px.SALT_REGIME_CELL_STABLE)
    proxy._guard_cache_salt(route, salt="petri|x|stable", lens=False,
                            regime=px.SALT_REGIME_CELL_STABLE)


def test_cell_stable_salts_occupy_one_guard_slot(tmp_path):
    proxy, _ = _proxy(tmp_path)
    _chat(proxy, "petri__prompted__safety", PROMPTED)
    _chat(proxy, "petri__prompted__safety", PROMPTED)
    assert len(proxy._salt_guard) == 1
    assert len(proxy._salt_guard_order) == 1
    assert proxy.n_salt_guard_evicted == 0


def test_salt_guard_memory_is_bounded_and_eviction_is_counted(
        tmp_path, monkeypatch):
    proxy, _ = _proxy(tmp_path)
    monkeypatch.setattr(px, "SALT_GUARD_MAX", 3)
    route = next(iter(proxy._routes.values()))
    for i in range(4):
        proxy._guard_cache_salt(route, salt=f"petri|s|{i}", lens=True,
                                regime=px.SALT_REGIME_REQUEST_UNIQUE)
    assert len(proxy._salt_guard) == 3
    assert len(proxy._salt_guard_order) == 3
    assert proxy.n_salt_guard_evicted == 1  # never silent
    # the evicted salt is deliberately forgotten (bounded memory): its
    # blocks cannot still be live — the guard window exceeds the whole
    # KV pool's block capacity (SALT_GUARD_MAX's own argument)
    proxy._guard_cache_salt(route, salt="petri|s|0", lens=True,
                            regime=px.SALT_REGIME_REQUEST_UNIQUE)


def test_guard_maps_to_http_409_and_the_app_keeps_serving(
        tmp_path, monkeypatch):
    # the gate's step 4 over the REAL HTTP surface, CPU-sized: 409 on
    # the wire (an OpenAI-style error body the target client surfaces
    # as one failed call), then 200s from the same live app.
    pytest.importorskip("aiohttp")
    import uuid as _uuid

    from aiohttp.test_utils import TestClient, TestServer

    proxy, backend = _proxy(tmp_path)
    monkeypatch.setattr(px.uuid, "uuid4", lambda: _uuid.UUID(int=7))
    lens_req = {"model": "petri__si_safety__psr_kl_all",
                "messages": USER, "max_tokens": 8}

    async def scenario():
        client = TestClient(TestServer(px.make_app(proxy)))
        await client.start_server()
        try:
            first = await client.post("/v1/chat/completions", json=lens_req)
            assert first.status == 200
            dup = await client.post("/v1/chat/completions", json=lens_req)
            assert dup.status == 409
            body = await dup.json()
            assert body["error"]["code"] == "cache_salt_collision"
            assert "si_safety__psr_kl_all" in body["error"]["message"]
            monkeypatch.undo()  # real salts again
            after = await client.post("/v1/chat/completions", json=lens_req)
            assert after.status == 200
            base = await client.post(
                "/v1/chat/completions",
                json={"model": "petri__base", "messages": USER,
                      "max_tokens": 8})
            assert base.status == 200
        finally:
            await client.close()

    asyncio.run(scenario())
    assert len(backend.payloads) == 3  # the 409 never reached upstream


# --- Phase-0 assertion 7, reworked -------------------------------------------------


def _iso(**over):
    ev = {"prefix_caching_enabled": True,
          "lens_salts_unique": True,
          "first_run_tokens": [5, 6], "second_run_tokens": [5, 6],
          "arm_tokens": [1, 2, 3],
          "twin_cold_tokens": [7, 8, 9],
          "twin_warm_tokens": [7, 8, 9]}
    ev.update(over)
    return {"prefix_cache_isolation": ev}


def _check(ev):
    ls._PHASE0_CHECKS["prefix_cache_isolation"](
        ev["prefix_cache_isolation"])


def test_caching_on_with_full_isolation_proof_passes():
    _check(_iso())


def test_caching_off_keeps_the_legacy_repeat_pair():
    _check({"prefix_cache_isolation": {
        "prefix_caching_enabled": False,
        "first_run_tokens": [5, 6], "second_run_tokens": [5, 6]}})
    with pytest.raises(ls.Phase0Failure, match="different tokens"):
        _check({"prefix_cache_isolation": {
            "prefix_caching_enabled": False,
            "first_run_tokens": [5, 6], "second_run_tokens": [5, 7]}})


def test_caching_on_without_isolation_evidence_fails():
    ev = _iso()
    del ev["prefix_cache_isolation"]["twin_warm_tokens"]
    with pytest.raises(ls.Phase0Failure, match="twin_warm_tokens"):
        _check(ev)


def test_a_leak_fails_loudly():
    # the twin moved after the arm served the same prompt: it read the
    # arm's cached (steered) blocks — the cross-cell leak itself
    with pytest.raises(ls.Phase0Failure, match="cache_salt isolation"):
        _check(_iso(twin_warm_tokens=[1, 2, 3]))


def test_an_inert_probe_arm_is_vacuous_and_fails():
    with pytest.raises(ls.Phase0Failure, match="DISCRIMINATE"):
        _check(_iso(arm_tokens=[7, 8, 9]))  # == twin_cold: cannot see a leak


def test_unattested_salting_fails():
    with pytest.raises(ls.Phase0Failure, match="lens_salts_unique"):
        _check(_iso(lens_salts_unique=None))


def test_unreadable_cache_state_fails():
    with pytest.raises(ls.Phase0Failure, match="not a bool"):
        _check(_iso(prefix_caching_enabled="yes"))


# --- the plan lattice's RESUMED class ----------------------------------------------


def test_resumed_shapes_refuse_by_default_and_classify_when_allowed():
    # one complete prefill (rows 0..4), one resumed hit (5 computed,
    # rows covering 5..8), one decode
    qsl, seqs, nat = [0, 5, 9, 10], [5, 9, 20], 10
    with pytest.raises(lb.LensBatchError, match="prefix-cache hit"):
        lb.derive_batch_plan(qsl, seqs, nat)
    plan = lb.derive_batch_plan(qsl, seqs, nat, allow_resumed=True)
    assert plan.n_resumed == 1 and plan.n_decode == 1
    assert len(plan.full_prefills) == 1
    kinds = [(s.is_prefill, s.is_resumed) for s in plan.segments]
    assert kinds == [(True, False), (False, True), (False, False)]


def test_resumed_segment_gets_no_context_and_positions_still_checked():
    qsl, seqs, nat = [0, 4], [9], 4  # one resumed segment: rows at 5..8
    plan = lb.derive_batch_plan(qsl, seqs, nat, allow_resumed=True)
    ctxs = lb.build_segment_contexts(plan, positions=[5, 6, 7, 8])
    assert ctxs[0].ctx is None  # unrepresentable by design — and unneeded
    with pytest.raises(lb.LensBatchError, match="positions tensor"):
        lb.build_segment_contexts(plan, positions=[0, 1, 2, 3])


def test_tracked_resumed_segment_refuses_unless_the_cell_can_resume():
    seg = lb.Segment(start=0, end=4, q=4, computed=5, seq_len=9)
    # the default (resumed_ok False): the pre-landing unconditional refusal
    with pytest.raises(lb.LensBatchError, match="salt"):
        lb.check_tracked_segment(seg, prompt_len=9, req_id="r")
    # a resumable cell's slice that ends at the prompt end is a RESUMED PREFILL
    assert lb.check_tracked_segment(seg, prompt_len=9, req_id="r",
                                    resumed_ok=True) == lb.SEG_RESUMED_PREFILL


# --- the worker under a caching-on engine ------------------------------------------


from test_petri_lens_worker import (HIDDEN, FakeView, _forward, _obj,
                                    _registry, _row_local_op, _sel)


def test_worker_tolerates_selectorless_cache_hits_in_a_mixed_forward():
    """The audit's steady state under caching: a steered request's
    complete prefill co-scheduled with a prompted leg's cache hit. The
    hit's rows are tolerated (ctx=None, no ops) instead of condemning
    the worker; the steered request is served normally."""
    arm = _obj("arm-last", [la.HookOp(op=la.OP_ADD_VECTOR, layer=1,
                                      vector=torch.ones(4),
                                      positions=la.POSITIONS_LAST,
                                      phase=la.PHASE_PREFILL)])
    reg = _registry(arm)
    view = FakeView(layers=[torch.nn.Identity()] * 3, prefix_caching=True)
    # rows: steered complete prefill of 5, then a RESUMED hit (4 rows at
    # absolute 5..8 of a 9-token selector-less prompt)
    qsl, seqs, nat = [0, 5, 9], [5, 9], 9
    pos = list(range(5)) + [5, 6, 7, 8]
    fwd = lw.build_request_slots(
        query_start_loc=qsl, seq_lens=seqs, num_actual_tokens=nat,
        positions=pos, req_ids=["steered", "cached"],
        raw_selectors=[_sel(arm), None],
        token_lookup={"steered": list(range(100, 105)),
                      "cached": list(range(200, 209))}.__getitem__,
        prompt_len_lookup={"steered": 5, "cached": 9}.__getitem__,
        registry=reg, allow_resumed=True)
    assert fwd.slots[0].ctx is not None
    assert fwd.slots[1].ctx is None and fwd.slots[1].obj is None
    assert fwd.plan.n_resumed == 1


def test_worker_refuses_a_steered_cache_hit_on_the_plan_path():
    arm = _obj("arm-last", [la.HookOp(op=la.OP_ADD_VECTOR, layer=1,
                                      vector=torch.ones(4),
                                      positions=la.POSITIONS_LAST,
                                      phase=la.PHASE_PREFILL)])
    reg = _registry(arm)
    qsl, seqs, nat = [0, 4], [9], 4  # resumed: rows 5..8 of a 9-prompt
    with pytest.raises(lb.LensBatchError, match="salt"):
        lw.build_request_slots(
            query_start_loc=qsl, seq_lens=seqs, num_actual_tokens=nat,
            positions=[5, 6, 7, 8], req_ids=["steered"],
            raw_selectors=[_sel(arm)],
            token_lookup={"steered": list(range(200, 209))}.__getitem__,
            prompt_len_lookup={"steered": 9}.__getitem__,
            registry=reg, allow_resumed=True)


def test_worker_serves_a_resumable_cache_hit_on_the_row_local_path():
    """Every row-local op (positions all, both phases, per-position) is
    resumable, so a steered resume on this path is SERVED — as a recorded
    RESUMED PREFILL (2026-09-06), no longer a salt-discipline refusal."""
    arm = _obj("arm-rowlocal", [_row_local_op(layer=1)])
    reg = _registry(arm)
    qsl, seqs, nat = [0, 4], [9], 4  # resumed shape, row-local op
    fwd = lw.build_request_slots(
        query_start_loc=qsl, seq_lens=seqs, num_actual_tokens=nat,
        positions=[5, 6, 7, 8], req_ids=["steered"],
        raw_selectors=[_sel(arm)],
        token_lookup={"steered": list(range(200, 209))}.__getitem__,
        prompt_len_lookup={"steered": 9}.__getitem__,
        registry=reg, allow_resumed=True)
    assert fwd.slots[0].ctx is None  # the row-local path builds no context
    assert fwd.slots[0].resumed == lw.ResumedPrefill(cached=5, computed=4)


def test_worker_refuses_a_non_resumable_twins_cache_hit_on_the_row_local_path():
    """The row-local path's salt-discipline guard still bites for the ONE
    non-resumable object it can carry: a scale-0 twin of a kind whose
    ops cannot resume (the cache-isolation gate's engine backstop)."""
    twin = _obj("static__scale0", [la.HookOp(op=la.OP_ADD_VECTOR, layer=1,
                                              vector=torch.ones(HIDDEN),
                                              phase=la.PHASE_PREFILL)],
                scale=0.0)
    reg = _registry(twin)
    qsl, seqs, nat = [0, 4], [9], 4  # resumed shape
    with pytest.raises(lw.LensWorkerError, match="salt"):
        lw.build_request_slots(
            query_start_loc=qsl, seq_lens=seqs, num_actual_tokens=nat,
            positions=[5, 6, 7, 8], req_ids=["twin"],
            raw_selectors=[_sel(twin, scale=0.0)],
            token_lookup={"twin": list(range(200, 209))}.__getitem__,
            prompt_len_lookup={"twin": 9}.__getitem__,
            registry=reg, allow_resumed=True)


def test_preflight_reads_the_live_cache_state_and_refuses_unreadable():
    arm = _obj("arm-rowlocal", [_row_local_op(layer=1)])
    reg = _registry(arm)
    view = FakeView(layers=[torch.nn.Identity()] * 3, prefix_caching=None)
    steering = lw.LensWorkerSteering(view=view, registry=reg)
    with pytest.raises(lw.LensWorkerError, match="prefix-caching state"):
        steering.preflight(tokenizer=object(),
                           header_verifier=lambda tok, rec: None)


# --- the composed + attested cache state --------------------------------------------


def test_expected_scheduler_state_asserts_the_cache_flag(tmp_path):
    from sisteer.capeval.serving import vllm_server as vs
    launch = vs.ServerLaunch(
        cmd=("x",), env={}, host="127.0.0.1", port=1, model="m",
        max_num_seqs=8, max_num_batched_tokens=8 * 16384,
        max_model_len=16384, prefix_caching=True)
    exp = vs.expected_scheduler_state(launch)
    assert exp["enable_prefix_caching"] is True
    # the report also carries the resolved KV size (covering the
    # 131,072-token budget here) — readiness refuses under-coverage
    good = dict(exp, enable_chunked_prefill=True,
                kv_num_gpu_blocks=8192, kv_block_size=16,
                kv_num_cache_groups=1)
    vs.validate_scheduler_reports([good], exp)
    flipped = dict(good, enable_prefix_caching=False)
    with pytest.raises(vs.VllmServerError, match="enable_prefix_caching"):
        vs.validate_scheduler_reports([flipped], exp)
    bare = vs.ServerLaunch(cmd=("x",), env={}, host="127.0.0.1", port=1,
                           model="m", max_num_seqs=8,
                           max_num_batched_tokens=8 * 16384,
                           max_model_len=16384)  # prefix_caching=None
    with pytest.raises(vs.VllmServerError, match="cache state"):
        vs.expected_scheduler_state(bare)
