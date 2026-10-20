"""The NON-BoN static/held-out families served through the routing proxy —
the correctness gates for wiring them onto the served (vLLM proxy) path.

CPU-only, no vLLM, no model loads, no network: a stub backend stands in for the
server (exactly as the proxy's own tests do), and a real, pure-Python family
(``pasttense`` — no torch) is replayed through ``run_static`` over a
``ServedStaticDefense``. What must hold (the five things the deliverable's spec
flagged as landmines):

1. WHOLE FAMILY FANS OUT concurrently through the proxy and is recorded IN
   ORDER with an ``engine`` provenance tag, and every response row's serving
   salt is the proxy's (never the client's).
2. ONE SHARED event loop across families — a fresh ServedStaticDefense per
   family reusing one loop works (an asyncio.run per family would close the
   loop and error the second family).
3. SHORT / ERRORED / EMPTY STORE FAILS LOUDLY (``assert_static_store_complete``)
   — the swallow-batch-exception landmine can never leave an empty cell that
   looks complete.
4. RESUME UNDER CONCURRENCY: a torn tail is cleaned, not duplicated; a served
   request failure leaves the family's goals UNCOVERED (regenerate on resume),
   never finalized-as-error; and served rows never mix with HF rows in one file.
5. The steered/unhooked gate wiring: ``_served_setup`` yields a bites probe +
   arm for a lens cell and NONE for an unhooked cell, and ``_effective_engines``
   resolves the --gen-engine/--bon-engine precedence.
"""

import asyncio
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from sisteer.capeval.serving import proxy as px
from sisteer.eval.attacks import bon_served as bs
from sisteer.eval.attacks import get_attack
from sisteer.eval.attacks.pasttense import pasttense_variants
from sisteer.eval.attacks.static_common import (StaticStoreIncompleteError,
                                                assert_static_store_complete,
                                                run_static)

from test_petri_proxy import _cells, _source, _spec

_DRIVER = Path(__file__).resolve().parents[1] / "scripts" / "unified_sweep_run.py"


@pytest.fixture(scope="module")
def drv():
    spec = importlib.util.spec_from_file_location("unified_sweep_run", _DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class _Goal:
    goal_id: str
    text: str
    category: str = "harmful"


def _goals(n=4):
    return [_Goal(goal_id=f"g{i}", text=f"do the forbidden thing number {i}")
            for i in range(n)]


class RecordingBackend:
    """A scripted upstream: records every rewritten payload (so a test can read
    the proxy-injected cache_salt), answers a distinguishable reply per request,
    and (optionally) delays to force overlap or raises on a predicate."""

    def __init__(self, *, delay_s=0.0, raise_if=None):
        self.payloads = []
        self.delay_s = delay_s
        self.raise_if = raise_if
        self.n_calls = 0
        self.in_flight = 0
        self.max_in_flight = 0

    async def chat_completions(self, payload):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            self.payloads.append(json.loads(json.dumps(payload)))
            self.n_calls += 1
            user = payload["messages"][-1]["content"]
            if self.raise_if is not None and self.raise_if(user):
                raise RuntimeError(f"synthetic upstream crash on {user[:24]!r}")
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            return {
                "id": "cmpl", "object": "chat.completion", "created": 0,
                "model": payload["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant",
                                         "content": f"reply::{user[:40]}"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                          "total_tokens": 2},
            }
        finally:
            self.in_flight -= 1


def _proxy(tmp_path, backend):
    spec = _spec(serving_mode="interleaved")
    source = _source(tmp_path)
    cells = _cells(spec, source)
    routes = px.routes_from_cells(cells)
    proxy = px.RoutingProxy(
        routes, backend, base_model=spec.target_model,
        serving_mode="interleaved",
        prompted_prefixes=px.prompted_prefix_specs(cells, source))
    return proxy, routes


def _served_static(proxy, *, loop, cell_id="none", virtual_model="petri__base",
                   max_concurrency=8, raise_on_error=True):
    sd = bs.ServedDefense.over_proxy(proxy, virtual_model=virtual_model,
                                     cell_id=cell_id)
    return bs.ServedStaticDefense(sd, loop=loop, max_concurrency=max_concurrency,
                                  raise_on_error=raise_on_error)


def _read(jsonl):
    rows = []
    for line in Path(jsonl).read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _expected_rows(goals):
    return sum(len(pasttense_variants(g, 0)) for g in goals)


# --------------------------------------------------------------------------- #
# 1. whole-family fan-out through the proxy, ordered + engine-tagged           #
# --------------------------------------------------------------------------- #


def test_family_fans_out_and_records_in_order(tmp_path):
    backend = RecordingBackend(delay_s=0.02)  # force overlap
    proxy, _ = _proxy(tmp_path, backend)
    loop = asyncio.new_event_loop()
    try:
        defense = _served_static(proxy, loop=loop, max_concurrency=8)
        goals = _goals(4)
        out_dir = tmp_path / "pt"
        summary = get_attack("pasttense")(defense, goals, out_dir=out_dir,
                                          max_new_tokens=16, seed=0)
    finally:
        loop.close()
    rows = _read(out_dir / "responses.jsonl")
    exp = _expected_rows(goals)
    assert exp > 0 and len(rows) == exp                 # complete
    assert all(r["engine"] == bs.SERVED_ENGINE for r in rows)   # provenance tag
    # goal-major, sample-order preserved (== the unbatched loop order)
    order = [(r["goal_id"], r["sample_idx"]) for r in rows]
    assert order == sorted(order, key=lambda t: (int(t[0][1:]), t[1]))
    assert defense.n_target_forwards == exp
    assert backend.max_in_flight > 1                    # truly concurrent
    assert defense.n_errors == 0
    # the client never smuggled a salt; the proxy injected one per request.
    assert all("cache_salt" not in m for p in backend.payloads
               for m in p.get("messages", []))
    assert all("cache_salt" in p for p in backend.payloads)


def test_batch_size_is_one_giant_chunk(tmp_path):
    """run_static reads defense.batch_size; ServedStaticDefense makes it huge so
    a whole family is ONE chunk (whole family in flight at once)."""
    backend = RecordingBackend()
    proxy, _ = _proxy(tmp_path, backend)
    loop = asyncio.new_event_loop()
    try:
        defense = _served_static(proxy, loop=loop)
        goals = _goals(5)
        summary = get_attack("pasttense")(defense, goals,
                                          out_dir=tmp_path / "pt",
                                          max_new_tokens=16, seed=0)
    finally:
        loop.close()
    assert summary["n_batches"] == 1
    assert summary["batch_size"] >= _expected_rows(goals)


# --------------------------------------------------------------------------- #
# 2. one shared loop across families                                          #
# --------------------------------------------------------------------------- #


def test_one_loop_serves_multiple_families(tmp_path):
    backend = RecordingBackend()
    proxy, _ = _proxy(tmp_path, backend)
    loop = asyncio.new_event_loop()          # ONE loop for BOTH families
    goals = _goals(3)
    try:
        for fam in ("pasttense", "codeattack"):
            defense = _served_static(proxy, loop=loop)   # fresh per family
            summary = get_attack(fam)(defense, goals, out_dir=tmp_path / fam,
                                      max_new_tokens=16, seed=0)
            assert_static_store_complete(summary, tmp_path / fam, family=fam,
                                         n_errors=defense.n_errors)
    finally:
        loop.close()
    # both families produced their full stores on the SAME loop.
    for fam in ("pasttense", "codeattack"):
        assert len(_read(tmp_path / fam / "responses.jsonl")) > 0


# --------------------------------------------------------------------------- #
# 3. loud completeness assertion (the swallow-batch-exception landmine)        #
# --------------------------------------------------------------------------- #


def test_completeness_passes_on_a_full_store(tmp_path):
    backend = RecordingBackend()
    proxy, _ = _proxy(tmp_path, backend)
    loop = asyncio.new_event_loop()
    try:
        defense = _served_static(proxy, loop=loop)
        goals = _goals(3)
        summary = get_attack("pasttense")(defense, goals, out_dir=tmp_path / "pt",
                                          max_new_tokens=16, seed=0)
    finally:
        loop.close()
    assert_static_store_complete(summary, tmp_path / "pt", family="pasttense",
                                 n_errors=0)   # no raise


def test_completeness_fails_on_nonzero_errors(tmp_path):
    with pytest.raises(StaticStoreIncompleteError, match="failed"):
        assert_static_store_complete({"per_goal": []}, tmp_path, family="x",
                                     n_errors=3)


def test_completeness_fails_on_empty_store(tmp_path):
    (tmp_path).mkdir(exist_ok=True)
    with pytest.raises(StaticStoreIncompleteError, match="generated nothing"):
        assert_static_store_complete({}, tmp_path, family="x", n_errors=0)


def test_completeness_fails_on_error_outcome(tmp_path):
    # a store whose goals were finalized as error with no rows (the landmine)
    (tmp_path / "progress.jsonl").write_text(
        json.dumps({"goal_id": "g0", "n_variants": 2,
                    "outcome": "error:RuntimeError"}) + "\n")
    (tmp_path / "responses.jsonl").write_text("")
    with pytest.raises(StaticStoreIncompleteError, match="error outcome"):
        assert_static_store_complete({}, tmp_path, family="x", n_errors=0)


def test_completeness_fails_on_short_store(tmp_path):
    (tmp_path / "progress.jsonl").write_text(
        json.dumps({"goal_id": "g0", "n_variants": 3, "outcome": "done"}) + "\n")
    (tmp_path / "responses.jsonl").write_text(
        json.dumps({"goal_id": "g0", "sample_idx": 0}) + "\n")  # 1 row, want 3
    with pytest.raises(StaticStoreIncompleteError, match="short"):
        assert_static_store_complete({}, tmp_path, family="x", n_errors=0)


def test_completeness_passes_on_a_loud_skip(tmp_path):
    # renellm-style loud skip (corpus absent): not an incompleteness.
    assert_static_store_complete({"skipped": True, "per_goal": []}, tmp_path,
                                 family="renellm", n_errors=0)


# --------------------------------------------------------------------------- #
# 4. resume under concurrency + error semantics + engine-mix guard            #
# --------------------------------------------------------------------------- #


def test_served_error_raises_and_leaves_goals_uncovered(tmp_path):
    """A served request failure (default raise_on_error) makes .generate raise,
    run_static (engine_tag) re-raises WITHOUT finalizing, so the goals stay
    UNCOVERED and a resume regenerates them — never a store that looks
    complete-but-empty, never a permanently error-poisoned goal."""
    # fail on goal g1's variants only.
    backend = RecordingBackend(raise_if=lambda u: "number 1" in u)
    proxy, _ = _proxy(tmp_path, backend)
    out_dir = tmp_path / "pt"
    goals = _goals(3)
    loop = asyncio.new_event_loop()
    try:
        defense = _served_static(proxy, loop=loop, raise_on_error=True)
        with pytest.raises(bs.BonServedError, match="request"):
            get_attack("pasttense")(defense, goals, out_dir=out_dir,
                                    max_new_tokens=16, seed=0)
    finally:
        loop.close()
    # the whole giant chunk failed all-or-nothing: NO progress records, NO rows.
    prog = out_dir / "progress.jsonl"
    assert (not prog.exists()) or prog.read_text().strip() == ""
    resp = out_dir / "responses.jsonl"
    assert (not resp.exists()) or resp.read_text().strip() == ""

    # resume with a healthy backend regenerates every goal cleanly.
    backend2 = RecordingBackend()
    proxy2, _ = _proxy(tmp_path, backend2)
    loop2 = asyncio.new_event_loop()
    try:
        defense2 = _served_static(proxy2, loop=loop2)
        summary = get_attack("pasttense")(defense2, goals, out_dir=out_dir,
                                          max_new_tokens=16, seed=0)
    finally:
        loop2.close()
    assert_static_store_complete(summary, out_dir, family="pasttense",
                                 n_errors=defense2.n_errors)
    assert len(_read(resp)) == _expected_rows(goals)


def test_torn_tail_cleaned_not_duplicated(tmp_path):
    """A SIGKILL mid-append leaves a torn tail line and the current goal's
    partial rows UNCOVERED; resume drops them and completes with no
    duplication (the per-goal ordering must hold even though generation was
    concurrent)."""
    backend = RecordingBackend()
    proxy, _ = _proxy(tmp_path, backend)
    out_dir = tmp_path / "pt"
    goals = _goals(4)
    loop = asyncio.new_event_loop()
    try:
        defense = _served_static(proxy, loop=loop)
        get_attack("pasttense")(defense, goals, out_dir=out_dir,
                                max_new_tokens=16, seed=0)
    finally:
        loop.close()
    full = _read(out_dir / "responses.jsonl")
    exp = _expected_rows(goals)
    assert len(full) == exp

    # Model a crash mid-write: drop the LAST goal's terminal progress record and
    # tear the responses tail (drop its newline + half the last line's bytes).
    prog = out_dir / "progress.jsonl"
    prog_lines = [l for l in prog.read_text().splitlines() if l.strip()]
    last_goal = json.loads(prog_lines[-1])["goal_id"]
    prog.write_text("\n".join(prog_lines[:-1]) + ("\n" if prog_lines[:-1] else ""))
    resp = out_dir / "responses.jsonl"
    raw = [l for l in resp.read_text().splitlines() if l.strip()]
    resp.write_text("\n".join(raw[:-1]) + "\n" + raw[-1][: len(raw[-1]) // 2])

    # resume: filter drops the uncovered last goal's partial rows, regenerates it.
    backend2 = RecordingBackend()
    proxy2, _ = _proxy(tmp_path, backend2)
    loop2 = asyncio.new_event_loop()
    try:
        defense2 = _served_static(proxy2, loop=loop2)
        summary = get_attack("pasttense")(defense2, goals, out_dir=out_dir,
                                          max_new_tokens=16, seed=0)
    finally:
        loop2.close()
    resumed = _read(resp)
    assert len(resumed) == exp                                # no shortfall
    keys = [(r["goal_id"], r["sample_idx"]) for r in resumed]
    assert len(set(keys)) == len(keys)                        # no duplication
    assert_static_store_complete(summary, out_dir, family="pasttense",
                                 n_errors=defense2.n_errors)


def test_served_refuses_to_mix_onto_hf_rows(tmp_path):
    """A served run refuses to append onto a responses file holding HF rows
    (rows with no engine field) — the engines are not bit-identical."""
    out_dir = tmp_path / "pt"
    out_dir.mkdir(parents=True)
    # an HF-style row (no engine field) already present, its goal COVERED.
    (out_dir / "responses.jsonl").write_text(
        json.dumps({"goal_id": "g0", "sample_idx": 0, "response": "x"}) + "\n")
    (out_dir / "progress.jsonl").write_text(
        json.dumps({"goal_id": "g0", "n_variants": 1, "outcome": "done"}) + "\n")
    backend = RecordingBackend()
    proxy, _ = _proxy(tmp_path, backend)
    loop = asyncio.new_event_loop()
    try:
        defense = _served_static(proxy, loop=loop)
        with pytest.raises(StaticStoreIncompleteError, match="mix engines"):
            get_attack("pasttense")(defense, _goals(2), out_dir=out_dir,
                                    max_new_tokens=16, seed=0)
    finally:
        loop.close()


def test_served_static_defense_refuses_bare_url():
    with pytest.raises(bs.BonServedError, match="ServedDefense"):
        bs.ServedStaticDefense("http://host:8000/v1", loop=None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 5. driver seam: engine precedence + steered/unhooked gate wiring             #
# --------------------------------------------------------------------------- #


def test_effective_engines_precedence(drv):
    import argparse
    f = drv._effective_engines

    def ns(**kw):
        base = {"gen_engine": None, "bon_engine": None}
        base.update(kw)
        return argparse.Namespace(**base)

    # default: everything HF (byte-identical unopted path).
    assert f(ns()) == ("hf", "hf")
    # --gen-engine served: whole leg served (bon defers to gen).
    assert f(ns(gen_engine="served")) == ("served", "served")
    # --bon-engine served ALONE: bon served, non-bon HF (pre-generalisation).
    assert f(ns(bon_engine="served")) == ("hf", "served")
    # explicit split is honoured.
    assert f(ns(gen_engine="served", bon_engine="hf")) == ("served", "hf")


def test_served_setup_arms_a_probe_for_lens_and_none_for_unhooked(drv, tmp_path):
    """A lens (steered) cell yields a bites arm + probe (the gate WILL run); an
    unhooked cell yields none (the gate must NOT run — no arm to probe)."""
    import argparse
    routes = px.routes_from_cells(_cells(_spec(serving_mode="interleaved"),
                                         _source(tmp_path)))
    table = tmp_path / "routing_table.json"
    # export the same routes so _served_setup can load them.
    from sisteer.capeval.serving.proxy import export_routing_table

    spec = _spec(serving_mode="interleaved")
    src = _source(tmp_path)
    cells = _cells(spec, src)
    export_routing_table(cells, src, table, base_model=spec.target_model,
                         serving_mode="interleaved")

    def args_for():
        return argparse.Namespace(bon_proxy_url="http://127.0.0.1:1/v1",
                                  bon_routing_table=str(table))

    class _Def:
        system = None

    # steered (lens) cell -> arm + probe present. The TrainTest cell id is
    # safety_refusal__psr_kl_all (harness id si_safety__psr_kl_all).
    steered = drv.sweep.Condition(label="s", cell="safety_refusal__psr_kl_all",
                                  static=("bon",), adaptive=(), benign=(),
                                  held_out=(), target_engine="vllm")
    _c, _r, route, arms, probe, _sys = drv._served_setup(steered, args_for(),
                                                         _Def())
    assert arms and probe is not None
    # unhooked baseline -> NO arm, NO probe (bites gate does not apply).
    base = drv.sweep.Condition(label="n", cell="none", static=("bon",),
                               adaptive=(), benign=(), held_out=(),
                               target_engine="vllm")
    _c, _r, _route, arms0, probe0, _sys0 = drv._served_setup(base, args_for(),
                                                            _Def())
    assert arms0 == () and probe0 is None


def test_served_only_defense_resolves_system_weightlessly(drv):
    """On a fully-served run the HF model is not loaded; _served_only_defense
    supplies .system from the SAME source build_defense_cell uses (so the proxy
    prefix-hash check passes) and .system is None for a compiled/none cell."""
    from sisteer.eval.jailbreak import resolve_instruction
    C = drv.sweep.Condition

    def cond(cell):
        return C(label="x", cell=cell, static=("bon",), adaptive=(),
                 benign=(), held_out=())

    # prompted__<si>: exactly the SI text build_defense_cell would send.
    d = drv._served_only_defense(cond("prompted__si_constitution"), {}, None)
    assert d.system == resolve_instruction("si_constitution").text
    assert d.batch_size is None
    # single-SI-lane prompted -> the safety SI.
    assert (drv._served_only_defense(cond("prompted"), {}, None).system
            == resolve_instruction("safety_refusal").text)
    # none and a compiled (lens) cell send NO system turn.
    assert drv._served_only_defense(cond("none"), {}, None).system is None
    assert drv._served_only_defense(
        cond("safety_refusal__psr_kl_all"), {}, None).system is None
    # the stub has no generate/model surface (a stray HF-forward leg fails loud).
    assert not hasattr(d, "generate") and not hasattr(d, "model")
