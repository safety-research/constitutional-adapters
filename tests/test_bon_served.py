"""BoN served concurrently against the steered vLLM server — correctness gates.

Covers the three properties the deliverable must not ship without, all
CPU-only (no vLLM, no model loads, no network — a stub backend stands in for
the server, exactly as the proxy's own tests do):

1. PER-CELL CACHE ISOLATION. Two distinct cells can never share a cached
   prefix; a lens cell's salt is request-unique, an unhooked/LoRA cell's is
   cell-stable; and the assertion FAILS LOUDLY if the salt mechanism ever
   regressed to sharing (proved by monkeypatching the salt to collide).

2. INERT-SERVE STILL CAUGHT. steering_bites refuses an arm whose serving-scale
   output equals its scale-0 output (the silently-unhooked / stock-extension
   failure mode), and a steered served run refuses to start with no probe.

3. RESUME UNDER CONCURRENCY. A torn tail line is skipped, never counted as
   covered and never duplicated; a crash mid-run leaves the completed rows
   durable and resume completes with no duplication; concurrent writers never
   interleave a line.
"""

import asyncio
import json
from dataclasses import dataclass

import pytest

from sisteer.capeval import manifest as cm
from sisteer.capeval.serving import proxy as px
from sisteer.capeval.serving.fidelity import FidelityError
from sisteer.eval.attacks import bon_served as bs

from test_petri_proxy import _cells, _source, _spec

# Prompted cells verify their SI prefix against these exact bytes (H8).
from test_petri_stacked import CONSTITUTION_PREFIX, SAFETY_PREFIX


# --------------------------------------------------------------------------- #
# scaffolding                                                                  #
# --------------------------------------------------------------------------- #


@dataclass
class _Goal:
    goal_id: str
    text: str
    category: str = "harmful"


def _goals(n=3):
    return [_Goal(goal_id=f"g{i}", text=f"do the forbidden thing number {i}")
            for i in range(n)]


class RecordingBackend:
    """A scripted upstream that records every rewritten payload (so the test
    can read the proxy-injected ``cache_salt``) and answers with a response
    whose content encodes the request — so each (goal, sample) maps to a
    distinguishable, order-independent reply. Optional hooks: a per-call delay
    (to force overlap) and a predicate that raises (to model a crash)."""

    def __init__(self, *, delay_s=0.0, raise_if=None):
        self.payloads = []
        self.delay_s = delay_s
        self.raise_if = raise_if
        self.n_calls = 0
        self.in_flight = 0
        self.max_in_flight = 0

    async def take_request_failure(self, request_id):
        # the lens post-check surface (per-request containment); this
        # stub never poisons a request.
        return None

    async def chat_completions(self, payload):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            self.payloads.append(json.loads(json.dumps(payload)))
            self.n_calls += 1
            user = payload["messages"][-1]["content"]
            if self.raise_if is not None and self.raise_if(user):
                raise RuntimeError(f"synthetic upstream crash on {user!r}")
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            return {
                "id": "cmpl", "object": "chat.completion", "created": 0,
                "model": payload["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant",
                                         "content": f"reply::{user}"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                          "total_tokens": 2},
            }
        finally:
            self.in_flight -= 1


def _proxy(tmp_path, backend):
    """An interleaved (concurrent, non-serialized) proxy over the real 24-cell
    canonical set — the mode BoN uses."""
    spec = _spec(serving_mode="interleaved")
    source = _source(tmp_path)
    cells = _cells(spec, source)
    routes = px.routes_from_cells(cells)
    proxy = px.RoutingProxy(
        routes, backend, base_model=spec.target_model,
        serving_mode="interleaved",
        prompted_prefixes=px.prompted_prefix_specs(cells, source))
    return proxy, routes


def _routes(tmp_path):
    spec = _spec(serving_mode="interleaved")
    return px.routes_from_cells(_cells(spec, _source(tmp_path)))


# --------------------------------------------------------------------------- #
# 1. per-cell cache isolation                                                  #
# --------------------------------------------------------------------------- #


def test_isolation_holds_over_canonical_set(tmp_path):
    routes = _routes(tmp_path)
    # the real mechanism must pass its own assertion.
    bs.assert_cell_salt_isolation(routes)
    rep = bs.cell_salt_isolation_report(routes)
    # every lens cell is request-unique, every unhooked/LoRA cell cell-stable.
    for e in rep["per_cell"].values():
        if e["lens"]:
            assert not e["stable"], e
        else:
            assert e["stable"], e
    assert rep["stable_collisions"] == []


def test_lens_salt_request_unique_unhooked_stable(tmp_path):
    routes = {r.cell_id: r for r in _routes(tmp_path)}
    lens = routes["si_safety__psr_kl_all"]
    unh = routes["none"]
    lora = routes["si_safety__lora_r1"]
    # lens: two draws differ (never resume a steered prefill from cache).
    assert (px.RoutingProxy.cache_salt_for(lens, lens=True)
            != px.RoutingProxy.cache_salt_for(lens, lens=True))
    # unhooked / LoRA: two draws identical (the SI prefix caches within-cell).
    assert (px.RoutingProxy.cache_salt_for(unh, lens=False)
            == px.RoutingProxy.cache_salt_for(unh, lens=False))
    assert (px.RoutingProxy.cache_salt_for(lora, lens=False)
            == px.RoutingProxy.cache_salt_for(lora, lens=False))


def test_isolation_fails_loudly_on_salt_collision(tmp_path, monkeypatch):
    routes = _routes(tmp_path)
    # simulate a regressed salt mechanism that returns a constant — distinct
    # cells would then share a cached prefix. The assertion must catch it.
    monkeypatch.setattr(px.RoutingProxy, "cache_salt_for",
                        staticmethod(lambda route, *, lens: "COLLIDE"))
    with pytest.raises(bs.BonServedError, match="SAME"):
        bs.assert_cell_salt_isolation(routes)


def test_isolation_fails_when_lens_salt_stable(tmp_path, monkeypatch):
    routes = _routes(tmp_path)
    # a lens salt that no longer varies would let a steered prefill resume.
    monkeypatch.setattr(
        px.RoutingProxy, "cache_salt_for",
        staticmethod(lambda route, *, lens: f"petri|{route.cell_id}"))
    with pytest.raises(bs.BonServedError, match="STABLE"):
        bs.assert_cell_salt_isolation(routes)


def test_isolation_rejects_empty_route_set():
    with pytest.raises(bs.BonServedError, match="no routes"):
        bs.assert_cell_salt_isolation([])


def test_served_request_carries_correct_salt_via_proxy(tmp_path):
    """End-to-end: ServedDefense -> proxy injects the salt; the client never
    sends one. Lens requests get distinct salts, unhooked identical."""
    backend = RecordingBackend()
    proxy, _ = _proxy(tmp_path, backend)
    lens = bs.ServedDefense.over_proxy(
        proxy, virtual_model="petri__si_safety__psr_kl_all",
        cell_id="si_safety__psr_kl_all")
    unh = bs.ServedDefense.over_proxy(
        proxy, virtual_model="petri__base", cell_id="none")

    async def go():
        # the sampler is the caller's to name (2026-09-17): the served model's vendor preset here
        dec = bs.bon_sampler("gemma4e4b")
        await lens.generate_one("a", sampling_seed=1, max_new_tokens=4, decode=dec)
        await lens.generate_one("b", sampling_seed=2, max_new_tokens=4, decode=dec)
        await unh.generate_one("c", sampling_seed=3, max_new_tokens=4, decode=dec)
        await unh.generate_one("d", sampling_seed=4, max_new_tokens=4, decode=dec)
    asyncio.run(go())

    salts = [p["cache_salt"] for p in backend.payloads]
    # the client never smuggled a salt (all four are proxy-injected top-level).
    assert all("cache_salt" not in m for p in backend.payloads
               for m in p.get("messages", []))
    lens_salts, unh_salts = salts[:2], salts[2:]
    assert lens_salts[0] != lens_salts[1]          # lens: request-unique
    assert unh_salts[0] == unh_salts[1]            # unhooked: cell-stable
    assert set(lens_salts).isdisjoint(unh_salts)   # cross-cell: never shared


# --------------------------------------------------------------------------- #
# 2. inert-serve (steering must bite)                                          #
# --------------------------------------------------------------------------- #


def test_steering_bites_passes_when_arm_bites():
    def probe(arm):
        return [[1, 2, 3]], [[1, 9, 3]]   # steered != scale0
    diag = bs.assert_steering_bites(["vector_all"], probe)
    assert diag["arms"]["vector_all"]["n_differing"] >= 1


def test_steering_bites_catches_inert_arm():
    def probe(arm):
        return [[1, 2, 3]], [[1, 2, 3]]   # steered == scale0 -> silently no-op
    with pytest.raises(FidelityError, match="NO"):
        bs.assert_steering_bites(["vector_all"], probe)


def test_steered_run_refuses_without_probe(tmp_path):
    backend = RecordingBackend()
    proxy, routes = _proxy(tmp_path, backend)
    defense = bs.ServedDefense.over_proxy(
        proxy, virtual_model="petri__si_safety__vector_all",
        cell_id="si_safety__vector_all")
    with pytest.raises(bs.BonServedError, match="no generation-time proof"):
        bs.bon_generate_served(
            defense, _goals(1), out_dir=tmp_path / "out", routes=routes,
            n_max=2, steering_arms=["vector_all"], steering_probe=None, model="gemma4e4b")


def test_steered_run_refuses_on_inert_probe(tmp_path):
    backend = RecordingBackend()
    proxy, routes = _proxy(tmp_path, backend)
    defense = bs.ServedDefense.over_proxy(
        proxy, virtual_model="petri__si_safety__vector_all",
        cell_id="si_safety__vector_all")
    with pytest.raises(FidelityError):
        bs.bon_generate_served(
            defense, _goals(1), out_dir=tmp_path / "out", routes=routes,
            n_max=2, steering_arms=["vector_all"],
            steering_probe=lambda arm: ([[1, 2]], [[1, 2]]), model="gemma4e4b")


def test_bare_base_url_refused():
    with pytest.raises(bs.BonServedError, match="bare base_url"):
        bs.ServedDefense("http://host:8000", virtual_model="petri__base",
                         cell_id="none")


# --------------------------------------------------------------------------- #
# 3. generation + resume under concurrency                                     #
# --------------------------------------------------------------------------- #


def _read(jsonl):
    rows = []
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def test_full_run_writes_every_unit_once(tmp_path):
    backend = RecordingBackend()
    proxy, routes = _proxy(tmp_path, backend)
    defense = bs.ServedDefense.over_proxy(
        proxy, virtual_model="petri__base", cell_id="none")
    goals, n_max = _goals(3), 5
    jsonl = bs.bon_generate_served(
        defense, goals, out_dir=tmp_path / "out", routes=routes,
        n_max=n_max, max_concurrency=8, model="gemma4e4b")
    rows = _read(jsonl)
    assert len(rows) == 3 * n_max
    keys = [(r["goal_id"], r["sample_idx"], r["trial"]) for r in rows]
    assert len(set(keys)) == len(keys)                       # no duplicates
    assert all(r["engine"] == bs.SERVED_ENGINE for r in rows)
    assert defense.n_target_forwards == 3 * n_max
    # reassembly is by sample_idx regardless of completion order.
    from sisteer.eval.attacks.bon import load_bon_responses
    by_goal = load_bon_responses(jsonl)
    assert set(by_goal) == {"g0", "g1", "g2"}
    assert all(len(v) == n_max for v in by_goal.values())


def test_resume_is_idempotent(tmp_path):
    backend = RecordingBackend()
    proxy, routes = _proxy(tmp_path, backend)
    defense = bs.ServedDefense.over_proxy(
        proxy, virtual_model="petri__base", cell_id="none")
    goals, n_max = _goals(2), 4
    kw = dict(out_dir=tmp_path / "out", routes=routes, n_max=n_max,
              max_concurrency=4, model="gemma4e4b")
    jsonl = bs.bon_generate_served(defense, goals, **kw)
    first = _read(jsonl)
    calls_after_first = backend.n_calls
    # resume: nothing left to do -> no new upstream calls, no new rows.
    bs.bon_generate_served(defense, goals, **kw)
    assert backend.n_calls == calls_after_first
    assert len(_read(jsonl)) == len(first) == 2 * n_max


def test_torn_tail_skipped_not_duplicated(tmp_path):
    backend = RecordingBackend()
    proxy, routes = _proxy(tmp_path, backend)
    defense = bs.ServedDefense.over_proxy(
        proxy, virtual_model="petri__base", cell_id="none")
    goals, n_max = _goals(2), 4
    kw = dict(out_dir=tmp_path / "out", routes=routes, n_max=n_max,
              max_concurrency=1, model="gemma4e4b")
    jsonl = bs.bon_generate_served(defense, goals, **kw)
    full = _read(jsonl)
    assert len(full) == 2 * n_max

    # model a SIGKILL mid-append: drop the newline + half the last line's bytes.
    raw = jsonl.read_text().splitlines()
    torn_key = json.loads(raw[-1])
    torn = "\n".join(raw[:-1]) + "\n" + raw[-1][: len(raw[-1]) // 2]  # no newline
    jsonl.write_text(torn)

    from sisteer.eval.attacks.bon import _load_covered
    covered = _load_covered(jsonl)
    torn_unit = (torn_key["goal_id"], torn_key["sample_idx"], torn_key["trial"])
    assert torn_unit not in covered                        # torn -> uncovered
    assert len(covered) == 2 * n_max - 1

    # resume: the torn unit regenerates, nothing duplicates, all units present.
    bs.bon_generate_served(defense, goals, **kw)
    rows = _read(jsonl)
    keys = [(r["goal_id"], r["sample_idx"], r["trial"]) for r in rows]
    assert len(set(keys)) == 2 * n_max                     # complete + unique
    assert keys.count(torn_unit) == 1                      # regenerated once


def test_crash_midrun_leaves_completed_durable_then_resumes(tmp_path):
    # crash the upstream on one specific goal's variants; the rest complete and
    # are fsync'd. Then resume with a clean backend -> full, no duplicates.
    goals, n_max = _goals(3), 4
    seeds = {bs.bon_seed(g.goal_id, s): g.goal_id
             for g in goals for s in range(n_max)}
    # variants of g1 will crash; identify them by their augmented text.
    import random as _r
    from sisteer.eval.attacks.bon import augment
    crash_variants = {augment(g.text, bs.DEFAULT_SIGMA,
                              _r.Random(bs.bon_seed(g.goal_id, s)))
                      for g in goals if g.goal_id == "g1"
                      for s in range(n_max)}
    backend = RecordingBackend(raise_if=lambda u: u in crash_variants)
    proxy, routes = _proxy(tmp_path, backend)
    defense = bs.ServedDefense.over_proxy(
        proxy, virtual_model="petri__base", cell_id="none")
    kw = dict(out_dir=tmp_path / "out", routes=routes, n_max=n_max,
              max_concurrency=4, model="gemma4e4b")
    with pytest.raises(Exception):
        bs.bon_generate_served(defense, goals, **kw)
    rows = _read(jsonl := (tmp_path / "out" / "bon_responses.jsonl"))
    # every persisted row is COMPLETE (asyncio can't tear a lock-held write)
    # and belongs to a non-crashing goal.
    keys = [(r["goal_id"], r["sample_idx"]) for r in rows]
    assert len(set(keys)) == len(keys)
    assert all(r["goal_id"] != "g1" for r in rows)

    clean = RecordingBackend()
    proxy2, routes2 = _proxy(tmp_path, clean)
    defense2 = bs.ServedDefense.over_proxy(
        proxy2, virtual_model="petri__base", cell_id="none")
    bs.bon_generate_served(defense2, goals, out_dir=tmp_path / "out",
                           routes=routes2, n_max=n_max, max_concurrency=4, model="gemma4e4b")
    rows2 = _read(jsonl)
    keys2 = [(r["goal_id"], r["sample_idx"], r["trial"]) for r in rows2]
    assert len(set(keys2)) == 3 * n_max                    # complete
    assert len(keys2) == len(set(keys2))                   # no duplicates


def test_concurrent_writes_never_interleave(tmp_path):
    # a real delay forces overlap; every persisted line must still be valid
    # JSON (one writer at a time under the lock) and each unit appears once.
    backend = RecordingBackend(delay_s=0.005)
    proxy, routes = _proxy(tmp_path, backend)
    defense = bs.ServedDefense.over_proxy(
        proxy, virtual_model="petri__base", cell_id="none")
    goals, n_max = _goals(4), 6
    jsonl = bs.bon_generate_served(
        defense, goals, out_dir=tmp_path / "out", routes=routes,
        n_max=n_max, max_concurrency=16, model="gemma4e4b")
    # overlap actually happened (else the test proves nothing).
    assert backend.max_in_flight > 1
    text = jsonl.read_text().splitlines()
    for line in text:
        if line.strip():
            json.loads(line)                                # no torn/interleaved
    rows = _read(jsonl)
    keys = [(r["goal_id"], r["sample_idx"]) for r in rows]
    assert len(keys) == len(set(keys)) == 4 * n_max


def test_engine_mixing_refused(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    jsonl = out / "bon_responses.jsonl"
    # an HF-path row (no engine tag) already present.
    jsonl.write_text(json.dumps({
        "goal_id": "g0", "sample_idx": 0, "trial": 0,
        "variant": "x", "response": "y"}) + "\n")
    backend = RecordingBackend()
    proxy, routes = _proxy(tmp_path, backend)
    defense = bs.ServedDefense.over_proxy(
        proxy, virtual_model="petri__base", cell_id="none")
    with pytest.raises(bs.BonServedError, match="mix"):
        bs.bon_generate_served(defense, _goals(1), out_dir=out, routes=routes,
                               n_max=2, model="gemma4e4b")


def test_prompted_cell_sends_verified_si_prefix(tmp_path):
    # a prompted cell carries an SI system prefix the proxy verifies (H8);
    # ServedDefense must send it or the proxy refuses.
    backend = RecordingBackend()
    proxy, routes = _proxy(tmp_path, backend)
    defense = bs.ServedDefense.over_proxy(
        proxy, virtual_model="petri__prompted__safety",
        cell_id="prompted__safety", system=SAFETY_PREFIX.decode())
    jsonl = bs.bon_generate_served(
        defense, _goals(2), out_dir=tmp_path / "out", routes=routes,
        n_max=3, max_concurrency=4, model="gemma4e4b")
    rows = _read(jsonl)
    assert len(rows) == 6
    # the system prefix rode every upstream request.
    assert all(p["messages"][0]["role"] == "system"
               and p["messages"][0]["content"] == SAFETY_PREFIX.decode()
               for p in backend.payloads)
    # prompted is an unhooked leg -> cell-stable salt (SI prefix caches).
    salts = {p["cache_salt"] for p in backend.payloads}
    assert len(salts) == 1
