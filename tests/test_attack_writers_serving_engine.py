"""The ``serving_engine`` field on every persisted served ATTACK row.

The row-facing engine field (``"lens"`` | ``"lora_native"``; additive, unhashed) is derived ONCE
from the proxy's ``/v1/models`` ``petri_proxy`` identity block (``engine.kind``) and stamped by
the four real writers — ``static_common.run_static``, ``multiturn.run_actorattack``,
``pyrit_adapters.run_pyrit_attack_async`` and ``bon_served`` — beside the ``engine`` tag; a row
WITHOUT it predates the native engine and was lens BY CONSTRUCTION. Every writer runs the
pre-write compare BEFORE its resume filter rewrites a file and before its first append, so a
cross-engine attempt refuses with ``responses.jsonl`` AND ``progress.jsonl`` byte-identical; a
lens run onto a legacy lens store (the in-flight campaign) keeps working; the HF path
stamps nothing. Driven through the REAL persistence paths with a scripted ``ServedChatTarget`` /
``ServedDefense`` over a fake proxy whose identity block names each engine; synthetic goals.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from sisteer.capeval.serving import served_window as sw
from sisteer.capeval.serving.proxy import RouteEntry
from sisteer.capeval.tinycap import contracts as tc
from sisteer.eval.attacks import bon_served as bs
from sisteer.eval.attacks import served_target as st
from sisteer.eval.attacks import static_common as sc
from sisteer.eval.attacks.multiturn import run_actorattack

LENS, NATIVE = st.SERVING_ENGINE_LENS, st.SERVING_ENGINE_LORA_NATIVE
FIELD = st.SERVING_ENGINE_FIELD
PASS = "proxy_vllm+passthrough+interleaved"

#: a native standup's identity block (the proxy echoes ``engine`` only when attested)
NATIVE_IDENTITY = {"version": 1, "base_model": "org/model",
                   "engine": {"kind": "lora_native", "enforce_eager": False,
                              "cudagraph_mode": "FULL", "adapters": {}}}
#: a lens standup's: no ``engine`` block at all (byte-identical to before the native engine)
LENS_IDENTITY = {"version": 1, "base_model": "org/model", "lens_reduction_mode": "per_row"}
#: torn: a native engine block beside a lens reduction-mode attestation
TORN_IDENTITY = {**NATIVE_IDENTITY, "lens_reduction_mode": "per_row"}


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #


def _sha(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def _snapshot(d: Path) -> dict:
    """sha256 of every file under ``d`` (the byte-identity witness)."""
    return {p.name: _sha(p) for p in sorted(Path(d).iterdir())} if Path(d).exists() else {}


def _rows(p: Path) -> list:
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def _write_rows(p: Path, rows) -> None:
    Path(p).write_text("".join(json.dumps(r) + "\n" for r in rows))


@dataclass(frozen=True)
class _Goal:
    goal_id: str
    text: str
    category: str = "benign_synthetic"


def _goals(n: int) -> list:
    return [_Goal(goal_id=f"g{i:02d}", text=f"benign topic {i}") for i in range(n)]


def _variants(goal, seed=0):
    return [("v0", f"{goal.text} / a"), ("v1", f"{goal.text} / b")]


class _StubChat:
    """The proxy transport: a per-request reply, payloads recorded."""

    def __init__(self):
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(dict(payload))
        user = payload["messages"][-1]["content"]
        return {"choices": [{"message": {"content": f"reply::{user[:24]}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                "petri_intervention_hash": "cc" * 8}


def _identity(engine):
    return NATIVE_IDENTITY if engine == NATIVE else LENS_IDENTITY


def _target(engine, **kw) -> st.ServedChatTarget:
    """A scripted ServedChatTarget whose engine is DERIVED from the identity block (the way
    build_served_target hands it over in route_info["proxy_identity"])."""
    return st.ServedChatTarget(_StubChat(), virtual_model="petri__x", cell_id="x",
                               route_info={"proxy_identity": _identity(engine)}, **kw)


async def _achat(payload):
    user = payload["messages"][-1]["content"]
    return {"choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": f"reply::{user[:24]}"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


def _served_static(engine, loop) -> bs.ServedStaticDefense:
    """run_static's REAL served adapter: ServedStaticDefense over a ServedDefense whose engine
    is derived from the identity block (serving_engine_of reaches it through the wrapper)."""
    sd = bs.ServedDefense(_achat, virtual_model="petri__base", cell_id="none",
                          proxy_identity=_identity(engine))
    return bs.ServedStaticDefense(sd, loop=loop, max_concurrency=4)


class _HFLike:
    """The in-process HF target's run_static surface: no engine_tag, no serving_engine."""

    def __init__(self):
        self.n_target_forwards = 0
        self.batch_size = 8

    def generate(self, prompts, *, max_new_tokens=512):
        self.n_target_forwards += len(prompts)
        return [f"hf:{p}" for p in prompts]


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


# --------------------------------------------------------------------------- #
# the vocabulary and the target-derived accessor                               #
# --------------------------------------------------------------------------- #


def test_vocabulary_is_contracts_by_reference():
    assert (st.SERVING_ENGINE_FIELD, st.SERVING_ENGINE_LENS, st.SERVING_ENGINE_LORA_NATIVE) == \
        (tc.SERVED_ENGINE_FIELD, tc.SERVING_ENGINE_LENS, tc.SERVING_ENGINE_LORA_NATIVE) == \
        ("serving_engine", "lens", "lora_native")
    assert st.SERVING_ENGINES == tuple(tc.SERVING_ENGINES) == bs.SERVING_ENGINES
    assert (bs.SERVING_ENGINE_FIELD, bs.SERVING_ENGINE_LENS, bs.SERVING_ENGINE_LORA_NATIVE) == \
        (FIELD, LENS, NATIVE)


def test_target_reads_the_engine_once_from_the_identity_block():
    assert _target(LENS).serving_engine == LENS
    assert _target(NATIVE).serving_engine == NATIVE
    # no identity at all (a proxy predating the block, or no route_info): lens by era
    assert st.ServedChatTarget(_StubChat(), virtual_model="petri__x", cell_id="x").serving_engine == LENS
    assert st.ServedChatTarget(_StubChat(), virtual_model="petri__x", cell_id="x",
                               route_info={"proxy_identity": None}).serving_engine == LENS
    # a torn identity refuses rather than being read around
    with pytest.raises(st.ServedTargetError, match="torn|not a native standup"):
        st.ServedChatTarget(_StubChat(), virtual_model="petri__x", cell_id="x",
                            route_info={"proxy_identity": TORN_IDENTITY})
    # the identity's two spellings disagreeing is torn as well
    with pytest.raises(st.ServedTargetError, match="torn"):
        st.ServedChatTarget(_StubChat(), virtual_model="petri__x", cell_id="x",
                            route_info={"proxy_identity": {**NATIVE_IDENTITY, FIELD: "lens"}})
    # a scripted target may STATE the engine (validated); it wins over the identity
    t = st.ServedChatTarget(_StubChat(), virtual_model="petri__x", cell_id="x",
                            route_info={"proxy_identity": LENS_IDENTITY}, serving_engine=NATIVE)
    assert t.serving_engine == NATIVE
    with pytest.raises(st.ServedTargetError, match="serving_engine must be one of"):
        st.ServedChatTarget(_StubChat(), virtual_model="petri__x", cell_id="x", serving_engine="hf")
    # serving_engine_from_identity is the one rule
    assert st.serving_engine_from_identity(None) == LENS
    assert st.serving_engine_from_identity({}) == LENS
    assert st.serving_engine_from_identity(NATIVE_IDENTITY) == NATIVE


def test_serving_engine_of_reaches_every_served_defense_and_nothing_else(loop):
    assert st.serving_engine_of(_target(LENS)) == LENS
    assert st.serving_engine_of(_target(NATIVE)) == NATIVE
    assert st.serving_engine_of(_served_static(NATIVE, loop)) == NATIVE     # through the wrapper
    assert st.serving_engine_of(_served_static(LENS, loop)) == LENS
    assert st.serving_engine_of(_HFLike()) is None                          # the HF path: nothing
    assert st.serving_engine_of(object()) is None
    # a defense that states an engine outside the vocabulary is refused, never stamped
    with pytest.raises(st.ServedTargetError, match="serving_engine must be one of"):
        st.serving_engine_of(SimpleNamespace(serving_engine="vllm"))


def test_served_defense_derives_the_engine_from_client_proxy_and_states_the_default():
    # over_client: the identity block ProxyClientEngine.load captured
    class _Client:
        def __init__(self, ident):
            self.proxy_identity = ident

        def chat_completions(self, payload):
            return {}

    assert bs.ServedDefense.over_client(_Client(NATIVE_IDENTITY), virtual_model="petri__base",
                                        cell_id="none").serving_engine == NATIVE
    assert bs.ServedDefense.over_client(_Client(LENS_IDENTITY), virtual_model="petri__base",
                                        cell_id="none").serving_engine == LENS
    assert bs.ServedDefense.over_client(_Client(None), virtual_model="petri__base",
                                        cell_id="none").serving_engine == LENS
    with pytest.raises(bs.BonServedError, match="torn|not a native standup"):
        bs.ServedDefense.over_client(_Client(TORN_IDENTITY), virtual_model="petri__base", cell_id="none")

    # over_proxy: the in-process proxy's own /v1/models payload
    class _Proxy:
        def __init__(self, ident):
            self._ident = ident

        def models_payload(self):
            return {"object": "list", "data": [], "petri_proxy": self._ident}

        async def handle_chat_completions(self, payload):
            return {}

    assert bs.ServedDefense.over_proxy(_Proxy(NATIVE_IDENTITY), virtual_model="petri__base",
                                       cell_id="none").serving_engine == NATIVE
    assert bs.ServedDefense.over_proxy(_Proxy(LENS_IDENTITY), virtual_model="petri__base",
                                       cell_id="none").serving_engine == LENS
    # an explicit statement wins over the identity; a bad statement refuses
    assert bs.ServedDefense.over_client(_Client(NATIVE_IDENTITY), virtual_model="petri__base",
                                        cell_id="none", serving_engine=LENS).serving_engine == LENS
    with pytest.raises(bs.BonServedError, match="serving_engine must be one of"):
        bs.ServedDefense(_achat, virtual_model="petri__base", cell_id="none", serving_engine="hf")
    # a bare transport with no identity: lens by the era rule (every such caller drives a lens standup)
    assert bs.ServedDefense(_achat, virtual_model="petri__base", cell_id="none").serving_engine == LENS


# --------------------------------------------------------------------------- #
# the manifest and the driver-level compare                                    #
# --------------------------------------------------------------------------- #


def test_write_target_manifest_carries_the_field(tmp_path):
    for engine in (LENS, NATIVE):
        doc = json.loads(st.write_target_manifest(tmp_path / engine, _target(engine)).read_text())
        assert doc[FIELD] == engine and doc["target_engine"] == "served"
    # a duck-typed stand-in without the attribute writes no key (its bytes are unchanged)
    fake = SimpleNamespace(route_info={}, evidence={}, timing_log=None, max_model_len=None,
                           n_clamped=0, n_prompt_over_window=0, n_window_mismatch=0,
                           window_mismatches=[], n_tokenize_resolved=0, n_floor_only_unresolvable=0,
                           n_target_forwards=0, n_all_turns_refused={})
    assert FIELD not in json.loads(st.write_target_manifest(tmp_path / "fake", fake).read_text())


def _covered(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "progress.jsonl").write_text(json.dumps({"goal_id": "g0", "outcome": "done"}) + "\n")


def test_assert_target_decode_compatible_engine_compare(tmp_path):
    # absent field in the manifest = lens by era: lens accepts, native refuses (covered goals)
    legacy = tmp_path / "legacy"; _covered(legacy)
    (legacy / "served_target.json").write_text(json.dumps({"decode": None, "manifest_stage": "final"}))
    st.assert_target_decode_compatible(legacy, None, serving_engine=LENS)
    st.assert_target_decode_compatible(legacy, None)                       # the pre-landing call: no compare
    before = _snapshot(legacy)
    with pytest.raises(st.ServedTargetError, match="lens by construction") as ei:
        st.assert_target_decode_compatible(legacy, None, serving_engine=NATIVE)
    assert "'lora_native'" in str(ei.value) and _snapshot(legacy) == before
    # present and different = refusal, either way round
    native = tmp_path / "native"; _covered(native)
    (native / "served_target.json").write_text(json.dumps({"decode": None, FIELD: NATIVE}))
    st.assert_target_decode_compatible(native, None, serving_engine=NATIVE)
    with pytest.raises(st.ServedTargetError, match="serving engine"):
        st.assert_target_decode_compatible(native, None, serving_engine=LENS)
    lens = tmp_path / "lens"; _covered(lens)
    (lens / "served_target.json").write_text(json.dumps({"decode": None, FIELD: LENS}))
    st.assert_target_decode_compatible(lens, None, serving_engine=LENS)
    with pytest.raises(st.ServedTargetError, match="serving engine"):
        st.assert_target_decode_compatible(lens, None, serving_engine=NATIVE)
    # the engine compare comes before the decode compare (a cross-engine store is refused as such)
    with pytest.raises(st.ServedTargetError, match="serving engine"):
        st.assert_target_decode_compatible(lens, {"do_sample": True, "temperature": 0.7}, serving_engine=NATIVE)
    # no covered goals: accepts any engine (nothing drawn to protect)
    fresh = tmp_path / "fresh"; fresh.mkdir()
    (fresh / "served_target.json").write_text(json.dumps({"decode": None, FIELD: LENS}))
    st.assert_target_decode_compatible(fresh, None, serving_engine=NATIVE)
    # the static leg's sidecar-less legacy greedy store is lens by era: lens resumes, native refuses
    static = tmp_path / "static"; _covered(static)
    _write_rows(static / "responses.jsonl", [{"goal_id": "g0", "sample_idx": 0, "response": "r",
                                              "engine": bs.SERVED_ENGINE}])
    st.assert_target_decode_compatible(static, None, legacy_leg_without_sidecar=True, serving_engine=LENS)
    with pytest.raises(st.ServedTargetError, match="LENS serving path"):
        st.assert_target_decode_compatible(static, None, legacy_leg_without_sidecar=True, serving_engine=NATIVE)
    assert not (static / "served_target.json").exists()
    with pytest.raises(st.ServedTargetError, match="serving_engine must be one of"):
        st.assert_target_decode_compatible(lens, None, serving_engine="vllm")


def test_assert_store_serving_engine_compatible_rules(tmp_path):
    out = tmp_path / "s"; out.mkdir()
    # an empty store accepts either engine; None (the HF path) checks nothing, even on a foreign store
    st.assert_store_serving_engine_compatible(out, LENS)
    st.assert_store_serving_engine_compatible(out, NATIVE)
    _write_rows(out / "responses.jsonl", [{"goal_id": "g0", "order": 0, "response": "r"}])     # no field: lens by era
    st.assert_store_serving_engine_compatible(out, None)
    st.assert_store_serving_engine_compatible(out, LENS)
    with pytest.raises(st.ServedTargetError, match="lens by construction"):
        st.assert_store_serving_engine_compatible(out, NATIVE)
    # a torn tail line is skipped exactly as resume does
    (out / "responses.jsonl").write_text((out / "responses.jsonl").read_text() + '{"goal_id": "g1", "ord')
    st.assert_store_serving_engine_compatible(out, LENS)
    # rows naming the other engine refuse; a mixed store is caught past its first row
    _write_rows(out / "responses.jsonl", [{"goal_id": "g0", "order": 0, "response": "r", FIELD: NATIVE}])
    st.assert_store_serving_engine_compatible(out, NATIVE)
    with pytest.raises(st.ServedTargetError, match="'lora_native' serving engine"):
        st.assert_store_serving_engine_compatible(out, LENS)
    _write_rows(out / "responses.jsonl", [{"goal_id": "g0", "order": 0, "response": "r", FIELD: NATIVE},
                                          {"goal_id": "g1", "order": 0, "response": "r", FIELD: LENS}])
    with pytest.raises(st.ServedTargetError, match="goal 'g1'"):
        st.assert_store_serving_engine_compatible(out, NATIVE)
    # the run manifest beside the rows: absent field = lens; present and different = refusal;
    # compared whenever it exists, rows or no rows (the owning run's declaration)
    man = tmp_path / "m"; man.mkdir()
    (man / "served_target.json").write_text(json.dumps({"decode": None}))
    st.assert_store_serving_engine_compatible(man, LENS)
    with pytest.raises(st.ServedTargetError, match="lens by construction"):
        st.assert_store_serving_engine_compatible(man, NATIVE)
    (man / "served_target.json").write_text(json.dumps({"decode": None, FIELD: NATIVE}))
    st.assert_store_serving_engine_compatible(man, NATIVE)
    with pytest.raises(st.ServedTargetError, match="served_target.json records"):
        st.assert_store_serving_engine_compatible(man, LENS)
    (man / "served_target.json").write_text("{not json")
    with pytest.raises(st.ServedTargetError, match="unreadable"):
        st.assert_store_serving_engine_compatible(man, LENS)
    # the BoN file name is a parameter; an engine outside the vocabulary is a caller bug
    bon = tmp_path / "b"; bon.mkdir()
    _write_rows(bon / "bon_responses.jsonl", [{"goal_id": "g0", "sample_idx": 0, "trial": 0, FIELD: LENS}])
    st.assert_store_serving_engine_compatible(bon, LENS, responses_name="bon_responses.jsonl")
    with pytest.raises(st.ServedTargetError):
        st.assert_store_serving_engine_compatible(bon, NATIVE, responses_name="bon_responses.jsonl")
    with pytest.raises(ValueError, match="serving_engine must be one of"):
        st.assert_store_serving_engine_compatible(bon, "vllm")


# --------------------------------------------------------------------------- #
# static families: run_static                                                  #
# --------------------------------------------------------------------------- #


def test_run_static_stamps_every_served_row_and_no_hf_row(tmp_path, loop):
    for engine in (LENS, NATIVE):
        res = sc.run_static(_served_static(engine, loop), _goals(2), out_dir=tmp_path / engine,
                            family="fam", variants_fn=_variants)
        rows = _rows(res["responses_path"])
        assert len(rows) == 4 and all(r["engine"] == bs.SERVED_ENGINE for r in rows)
        assert all(r[FIELD] == engine for r in rows)
        # beside the engine tag, before the geometry stamp
        keys = list(rows[0])
        assert keys.index("engine") + 1 == keys.index(FIELD)
    res = sc.run_static(_HFLike(), _goals(2), out_dir=tmp_path / "hf", family="fam", variants_fn=_variants)
    rows = _rows(res["responses_path"])
    assert len(rows) == 4 and all("engine" not in r and FIELD not in r for r in rows)


def test_run_static_driven_by_a_served_chat_target_stamps_every_row(tmp_path):
    """THE CALLER'S HALF (2026-09-17). The test above proves the writer stamps GIVEN a tag; the
    static / sampled dispatch in jailbreak_run hands ``run_static`` the ``ServedChatTarget``
    ``build_served_target`` returns, not ``ServedStaticDefense`` — and ``run_static`` reads
    ``engine_tag`` OFF THE DEFENSE. A served target class without the attribute wrote rows with
    neither ``engine`` nor ``serving_engine`` (every served sampled-static store, until this), while
    the multi-turn writers, keyed on ``serving_engine`` alone, stamped. So: the real class, both
    engines, every row."""
    for engine in (LENS, NATIVE):
        res = sc.run_static(_target(engine), _goals(2), out_dir=tmp_path / engine,
                            family="fam", variants_fn=_variants)
        rows = _rows(res["responses_path"])
        assert len(rows) == 4 and all(r["engine"] == bs.SERVED_ENGINE for r in rows), [sorted(r) for r in rows]
        assert all(r[FIELD] == engine for r in rows)
        keys = list(rows[0])
        assert keys.index("engine") + 1 == keys.index(FIELD)
    # the class declares it (the same shape as ServedStaticDefense), so no call site has to remember
    assert "engine_tag" in vars(st.ServedChatTarget) and st.ServedChatTarget.engine_tag == bs.SERVED_ENGINE


class _ServedLookingNoTag:
    """A defense that STATES a serving engine but carries no ``engine_tag`` — a served target
    masquerading as HF to ``run_static``'s tag read."""

    serving_engine = LENS

    def __init__(self):
        self.n_target_forwards = 0
        self.batch_size = 8

    def generate(self, prompts, *, max_new_tokens=512):
        self.n_target_forwards += len(prompts)
        return [f"r::{p}" for p in prompts]


def test_run_static_refuses_a_defense_that_states_an_engine_but_no_tag(tmp_path):
    """The writer's backstop for the hole above: a defense with ``serving_engine`` and no
    ``engine_tag`` is a served target that would write HF-shaped rows — refused before a byte is
    written, never stamped as HF. The HF target has neither attribute and is untouched."""
    out = tmp_path / "x"
    with pytest.raises(TypeError, match="engine_tag"):
        sc.run_static(_ServedLookingNoTag(), _goals(1), out_dir=out, family="fam", variants_fn=_variants)
    assert not (out / "responses.jsonl").exists() and not (out / "progress.jsonl").exists()


def _static_lens_store(out: Path, *, sidecar: bool, stamped: bool) -> None:
    """A lens static store: goal g00 covered (two rows + a terminal record) and goal g01 with ONE
    partial row and no record — so ``filter_responses_to_covered`` WOULD rewrite responses.jsonl on
    a resume. ``stamped`` writes ``serving_engine: lens`` on the rows / sidecar; unstamped rows
    predate the field (lens by construction)."""
    out.mkdir(parents=True, exist_ok=True)
    tag = {FIELD: LENS} if stamped else {}

    def row(gid, i):
        return {"goal_id": gid, "sample_idx": i, "trial": 0, "variant_key": f"{i}:0", "variant_name": f"v{i}",
                "family": "fam", "variant": f"topic {gid} / {i}", "attack_prompt": f"topic {gid} / {i}",
                "response": "an earlier reply", "engine": bs.SERVED_ENGINE, **tag}
    _write_rows(out / "responses.jsonl", [row("g00", 0), row("g00", 1), row("g01", 0)])
    _write_rows(out / "progress.jsonl", [{"goal_id": "g00", "category": "benign_synthetic", "family": "fam",
                                          "n_variants": 2, "n_target_forwards": 2, "outcome": "done",
                                          "capped": False}])
    if sidecar:
        (out / "served_target.json").write_text(json.dumps(
            {"target_engine": "served", "manifest_stage": "final", "decode": None, **tag}, indent=2))


@pytest.mark.parametrize("sidecar", [True, False], ids=["with_sidecar", "no_sidecar"])
@pytest.mark.parametrize("stamped", [False, True], ids=["legacy_rows", "lens_rows"])
def test_run_static_native_onto_a_lens_store_refuses_before_the_filter_rewrites(tmp_path, loop, sidecar, stamped):
    out = tmp_path / "store"
    _static_lens_store(out, sidecar=sidecar, stamped=stamped)
    before = _snapshot(out)
    with pytest.raises(sc.StaticStoreIncompleteError, match="'lora_native'") as ei:
        sc.run_static(_served_static(NATIVE, loop), _goals(2), out_dir=out, family="fam", variants_fn=_variants)
    assert ("lens by construction" in str(ei.value)) == (not stamped)
    # nothing written, nothing rewritten: responses.jsonl (the partial g01 row still there),
    # progress.jsonl and the sidecar are byte-identical
    assert _snapshot(out) == before
    assert len(_rows(out / "responses.jsonl")) == 3


@pytest.mark.parametrize("sidecar", [True, False], ids=["with_sidecar", "no_sidecar"])
def test_run_static_lens_onto_a_legacy_lens_store_resumes_as_before(tmp_path, loop, sidecar):
    out = tmp_path / "store"
    _static_lens_store(out, sidecar=sidecar, stamped=False)
    covered_lines = (out / "responses.jsonl").read_text().splitlines()[:2]
    progress_before = (out / "progress.jsonl").read_text()
    res = sc.run_static(_served_static(LENS, loop), _goals(2), out_dir=out, family="fam", variants_fn=_variants)
    assert res["n_goals_run"] == 1                                    # only the uncovered g01
    lines = (out / "responses.jsonl").read_text().splitlines()
    assert lines[:2] == covered_lines                                 # the covered legacy rows: verbatim
    new = [json.loads(l) for l in lines[2:]]
    assert [(r["goal_id"], r["sample_idx"]) for r in new] == [("g01", 0), ("g01", 1)]
    assert all(r[FIELD] == LENS for r in new)                         # the new rows carry the field
    assert (out / "progress.jsonl").read_text().startswith(progress_before)
    if sidecar:
        assert FIELD not in json.loads((out / "served_target.json").read_text())   # run_static never writes it


def test_run_static_lens_onto_a_complete_legacy_lens_store_is_byte_identical(tmp_path, loop):
    out = tmp_path / "store"
    _static_lens_store(out, sidecar=True, stamped=False)
    # complete the store: g01 covered too
    with (out / "responses.jsonl").open("a") as f:
        f.write(json.dumps({"goal_id": "g01", "sample_idx": 1, "trial": 0, "variant_key": "1:0",
                            "variant_name": "v1", "family": "fam", "variant": "x", "attack_prompt": "x",
                            "response": "an earlier reply", "engine": bs.SERVED_ENGINE}) + "\n")
    with (out / "progress.jsonl").open("a") as f:
        f.write(json.dumps({"goal_id": "g01", "category": "benign_synthetic", "family": "fam",
                            "n_variants": 2, "n_target_forwards": 2, "outcome": "done", "capped": False}) + "\n")
    before = _snapshot(out)
    d = _served_static(LENS, loop)
    res = sc.run_static(d, _goals(2), out_dir=out, family="fam", variants_fn=_variants)
    assert res["n_goals_run"] == 0 and d.n_target_forwards == 0
    assert _snapshot(out) == before                                   # today's resume, byte for byte


def test_run_static_native_onto_a_native_store_resumes(tmp_path, loop):
    out = tmp_path / "store"
    res = sc.run_static(_served_static(NATIVE, loop), _goals(1), out_dir=out, family="fam", variants_fn=_variants)
    assert res["n_goals_run"] == 1
    res2 = sc.run_static(_served_static(NATIVE, loop), _goals(2), out_dir=out, family="fam", variants_fn=_variants)
    assert res2["n_goals_run"] == 1 and all(r[FIELD] == NATIVE for r in _rows(out / "responses.jsonl"))
    # ... and a lens run onto it refuses, byte-preserving
    before = _snapshot(out)
    with pytest.raises(sc.StaticStoreIncompleteError, match="'lora_native' serving engine"):
        sc.run_static(_served_static(LENS, loop), _goals(2), out_dir=out, family="fam", variants_fn=_variants)
    assert _snapshot(out) == before


# --------------------------------------------------------------------------- #
# static families: a SILENT row (no ``engine``) is described by the run manifest #
# --------------------------------------------------------------------------- #

#: what ``served_target.write_target_manifest`` writes for every served store (+ ``serving_engine``)
SERVED_MANIFEST = {"target_engine": "served", "manifest_stage": "final"}


def _silent_static_store(out: Path, *, manifest, row_extra=None) -> Path:
    """A static store whose rows carry NO ``engine`` and no ``serving_engine`` — the shape every served
    static store had while ``ServedChatTarget`` carried no ``engine_tag`` (2026-09-10 -> 2026-09-17):
    goal g00 covered (two rows + a terminal record), goal g01 with ONE partial row and no record, so a
    resume would rewrite responses.jsonl. ``manifest`` None = no ``served_target.json``; a dict is
    written as the run manifest verbatim. ``row_extra`` is merged into every row. Returns the
    responses path."""
    out.mkdir(parents=True, exist_ok=True)

    def row(gid, i):
        return {"goal_id": gid, "sample_idx": i, "trial": 0, "variant_key": f"{i}:0", "variant_name": f"v{i}",
                "family": "fam", "variant": f"topic {gid} / {i}", "attack_prompt": f"topic {gid} / {i}",
                "response": "an earlier reply", **(row_extra or {})}
    _write_rows(out / "responses.jsonl", [row("g00", 0), row("g00", 1), row("g01", 0)])
    _write_rows(out / "progress.jsonl", [{"goal_id": "g00", "category": "benign_synthetic", "family": "fam",
                                          "n_variants": 2, "n_target_forwards": 2, "outcome": "done",
                                          "capped": False}])
    if manifest is not None:
        (out / st.TARGET_MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
    return out / "responses.jsonl"


def test_silent_rows_under_a_served_manifest_naming_their_engine_are_admitted(tmp_path, loop):
    """THE POSITIVE CASE (2026-09-17). A served static store written while ``ServedChatTarget`` had no
    ``engine_tag`` has SILENT rows (no ``engine``) beside a run manifest that names the engine
    (``target_engine: served`` + ``serving_engine``). On resume the manifest describes the silent rows,
    so a served run of the SAME engine is admitted and the covered rows stay verbatim (never
    relabelled). Before this rule the engine-tag guard read every silent row as HF by era and refused
    every such store with StaticStoreIncompleteError."""
    for engine in (LENS, NATIVE):                       # the guard itself: both engines, described
        p = _silent_static_store(tmp_path / engine, manifest={**SERVED_MANIFEST, FIELD: engine})
        sc._assert_static_engine_consistent(p, bs.SERVED_ENGINE, serving_engine=engine)
    # end to end through run_static (a lens store: the silent rows are lens under BOTH guards' rules)
    out = tmp_path / "e2e"
    _silent_static_store(out, manifest={**SERVED_MANIFEST, FIELD: LENS})
    covered = (out / "responses.jsonl").read_text().splitlines()[:2]
    res = sc.run_static(_served_static(LENS, loop), _goals(2), out_dir=out, family="fam", variants_fn=_variants)
    assert res["n_goals_run"] == 1                                    # only the uncovered g01
    lines = (out / "responses.jsonl").read_text().splitlines()
    assert lines[:2] == covered                                       # the covered silent rows: verbatim
    new = [json.loads(l) for l in lines[2:]]
    assert [(r["goal_id"], r["sample_idx"]) for r in new] == [("g01", 0), ("g01", 1)]
    assert all(r["engine"] == bs.SERVED_ENGINE and r[FIELD] == LENS for r in new)


def test_silent_rows_under_a_native_manifest_resume_natively_end_to_end(tmp_path, loop):
    out = tmp_path / "e2e"
    _silent_static_store(out, manifest={**SERVED_MANIFEST, FIELD: NATIVE})
    res = sc.run_static(_served_static(NATIVE, loop), _goals(2), out_dir=out, family="fam", variants_fn=_variants)
    assert res["n_goals_run"] == 1


@pytest.mark.parametrize("manifest, why", [
    ({"target_engine": "hf", "manifest_stage": "final"}, "target_engine='hf'"),   # an HF store that says so
    (None, "absent"),                                                             # era inference survives
    ({"manifest_stage": "final"}, "names no engine"),                             # a manifest naming nothing
], ids=["hf_manifest", "no_manifest", "manifest_names_nothing"])
def test_silent_rows_not_described_as_served_refuse_a_served_resume(tmp_path, loop, manifest, why):
    """KNOWN-BADS: a silent row is admitted ONLY when the run manifest describes it as served. An HF
    manifest, no manifest (the era inference: silent = HF) and a manifest naming nothing all refuse —
    a naive 'admit every silent row' rule fails each of these."""
    out = tmp_path / "s"
    p = _silent_static_store(out, manifest=manifest)
    with pytest.raises(sc.StaticStoreIncompleteError, match="mix engines") as ei:
        sc._assert_static_engine_consistent(p, bs.SERVED_ENGINE, serving_engine=LENS)
    assert why in str(ei.value), str(ei.value)
    before = _snapshot(out)                                          # end to end: refused, byte-identical
    with pytest.raises(sc.StaticStoreIncompleteError, match="mix engines"):
        sc.run_static(_served_static(LENS, loop), _goals(2), out_dir=out, family="fam", variants_fn=_variants)
    assert _snapshot(out) == before and len(_rows(p)) == 3


def test_a_row_stating_an_engine_is_compared_literally_whatever_the_manifest_says(tmp_path, loop):
    """LITERAL WINS: the manifest describes SILENT rows only. A row that states ``engine: hf`` under a
    served manifest refuses on its own word, and a torn store (silent described rows beside one row
    stating another engine) refuses too."""
    out = tmp_path / "stating"
    p = _silent_static_store(out, manifest={**SERVED_MANIFEST, FIELD: NATIVE}, row_extra={"engine": "hf"})
    with pytest.raises(sc.StaticStoreIncompleteError, match="engine='hf' \\(not 'vllm_served'\\)"):
        sc._assert_static_engine_consistent(p, bs.SERVED_ENGINE, serving_engine=NATIVE)
    torn = tmp_path / "torn"
    p2 = _silent_static_store(torn, manifest={**SERVED_MANIFEST, FIELD: LENS})
    rows = _rows(p2)
    rows[2]["engine"] = "hf"
    _write_rows(p2, rows)
    with pytest.raises(sc.StaticStoreIncompleteError, match="engine='hf'"):
        sc._assert_static_engine_consistent(p2, bs.SERVED_ENGINE, serving_engine=LENS)
    before = _snapshot(torn)
    with pytest.raises(sc.StaticStoreIncompleteError, match="engine='hf'"):
        sc.run_static(_served_static(LENS, loop), _goals(2), out_dir=torn, family="fam", variants_fn=_variants)
    assert _snapshot(torn) == before


def test_silent_rows_under_a_native_manifest_refuse_a_lens_resume(tmp_path, loop):
    """The manifest-described engine must match the RESUMING defense on both dimensions: silent rows a
    served manifest describes as lora_native refuse a lens served resume (the tag agrees, the serving
    engine does not)."""
    out = tmp_path / "s"
    p = _silent_static_store(out, manifest={**SERVED_MANIFEST, FIELD: NATIVE})
    with pytest.raises(sc.StaticStoreIncompleteError, match="'lora_native'") as ei:
        sc._assert_static_engine_consistent(p, bs.SERVED_ENGINE, serving_engine=LENS)
    assert "'lens'" in str(ei.value)
    before = _snapshot(out)
    with pytest.raises(sc.StaticStoreIncompleteError, match="'lora_native'"):
        sc.run_static(_served_static(LENS, loop), _goals(2), out_dir=out, family="fam", variants_fn=_variants)
    assert _snapshot(out) == before


def test_an_unreadable_manifest_refuses_a_silent_store_and_both_guards_share_the_lookup(tmp_path):
    out = tmp_path / "s"
    p = _silent_static_store(out, manifest=None)
    (out / st.TARGET_MANIFEST_FILE).write_text("{not json")
    with pytest.raises(sc.StaticStoreIncompleteError, match="unreadable"):
        sc._assert_static_engine_consistent(p, bs.SERVED_ENGINE, serving_engine=LENS)
    (out / st.TARGET_MANIFEST_FILE).write_text("[]")
    with pytest.raises(sc.StaticStoreIncompleteError, match="not an object"):
        sc._assert_static_engine_consistent(p, bs.SERVED_ENGINE, serving_engine=LENS)
    # ONE lookup: the serving-engine manifest compare and the static engine-tag guard both reach the
    # store's manifest through served_target.read_target_manifest (one file name, one parse, one set of
    # refusals), so the two guards cannot disagree about which manifest a store has
    assert "read_target_manifest(" in inspect.getsource(st.assert_manifest_serving_engine_compatible)
    assert "read_target_manifest(" in inspect.getsource(sc._store_manifest_engine)
    assert "read_target_manifest(" in inspect.getsource(st._manifest_serving_engine)      # the silent-row describer of the row-loop compare
    assert "_manifest_serving_engine(" in inspect.getsource(st.assert_store_serving_engine_compatible)
    assert st.read_target_manifest(tmp_path / "nowhere") is None


def test_static_guards_run_before_the_filter_in_source():
    src = Path(sc.__file__).read_text()
    body = src[src.index("def run_static("):]
    i_engine = body.index("_assert_static_engine_consistent(responses_path, engine_tag, serving_engine=serving_engine)")
    i_serving = body.index("_assert_static_serving_engine_consistent(responses_path, serving_engine)")
    i_filter = body.index("filter_responses_to_covered(responses_path, covered)")
    assert i_engine < i_filter and i_serving < i_filter


# --------------------------------------------------------------------------- #
# multi-turn: run_actorattack                                                  #
# --------------------------------------------------------------------------- #

_ACTOR_JSON = '{"actors": [{"name": "a1", "queries": ["q0", "q1"]}]}'


def _actor(defense, goals, out, **kw):
    return run_actorattack(defense, goals, attacker_fn=lambda chat: _ACTOR_JSON,
                           scorer_fn=lambda p: ("no" if "Refused" in p else "1"),
                           out_dir=out, actor_num=1, discovery_retries=0, max_retries=1, **kw)


def _mt_lens_store(out: Path, *, sidecar: bool, stamped: bool, transcripts: bool) -> None:
    """A lens multi-turn store: g00 covered (two replies + a terminal record), g01 with ONE partial
    reply and no record — so the resume filter WOULD rewrite responses.jsonl (and transcripts)."""
    out.mkdir(parents=True, exist_ok=True)
    tag = {FIELD: LENS} if stamped else {}
    _write_rows(out / "responses.jsonl", [{"goal_id": "g00", "order": 0, "response": "r0", **tag},
                                          {"goal_id": "g00", "order": 1, "response": "r1", **tag},
                                          {"goal_id": "g01", "order": 0, "response": "partial", **tag}])
    if transcripts:
        _write_rows(out / "transcripts.jsonl", [
            {"goal_id": "g00", "order": 0, "messages": [{"role": "user", "content": "q0"}], "response": "r0", **tag},
            {"goal_id": "g01", "order": 0, "messages": [{"role": "user", "content": "q0"}], "response": "partial",
             **tag}])
    _write_rows(out / "progress.jsonl", [{"goal_id": "g00", "category": "benign_synthetic", "n_target_forwards": 3,
                                          "outcome": "failure", "actors_used": 1, "capped": False}])
    if sidecar:
        (out / "served_target.json").write_text(json.dumps(
            {"target_engine": "served", "manifest_stage": "final", "decode": None, **tag}, indent=2))


def test_actorattack_stamps_every_persisted_reply_and_transcript_row(tmp_path):
    for engine in (LENS, NATIVE):
        out = tmp_path / engine
        res = _actor(_target(engine), _goals(1), out)
        assert res["per_goal"][0]["outcome"] in ("success", "failure")
        rows = _rows(out / "responses.jsonl")
        assert len(rows) == 3 and all(r[FIELD] == engine for r in rows)          # 2 turns + the summary
        trows = _rows(out / "transcripts.jsonl")
        assert len(trows) == 3 and all(r[FIELD] == engine for r in trows)
        assert FIELD not in _rows(out / "progress.jsonl")[0]                      # the per-goal record: unchanged
    # the HF-shaped target (no serving_engine) stamps nothing: rows byte-identical to before
    hf = SimpleNamespace(n_target_forwards=0, mode="none")

    def chat(messages, *, max_new_tokens=512, **kw):
        hf.n_target_forwards += 1
        return "a benign answer"
    hf.chat = chat
    _actor(hf, _goals(1), tmp_path / "hf")
    assert all(set(r) == {"goal_id", "order", "response"} for r in _rows(tmp_path / "hf" / "responses.jsonl"))
    assert all(set(r) == {"goal_id", "order", "messages", "response"}
               for r in _rows(tmp_path / "hf" / "transcripts.jsonl"))


@pytest.mark.parametrize("sidecar", [True, False], ids=["with_sidecar", "no_sidecar"])
@pytest.mark.parametrize("stamped", [False, True], ids=["legacy_rows", "lens_rows"])
@pytest.mark.parametrize("resume", [True, False], ids=["resume", "no_resume"])
def test_actorattack_native_onto_a_lens_store_refuses_with_nothing_written(tmp_path, sidecar, stamped, resume):
    out = tmp_path / "store"
    _mt_lens_store(out, sidecar=sidecar, stamped=stamped, transcripts=True)
    before = _snapshot(out)
    t = _target(NATIVE)
    with pytest.raises(st.ServedTargetError, match="'lora_native'") as ei:
        _actor(t, _goals(2), out, resume=resume)
    assert ("lens by construction" in str(ei.value)) == (not stamped)
    assert _snapshot(out) == before and t.n_target_forwards == 0            # no filter rewrite, no append, no call


@pytest.mark.parametrize("sidecar", [True, False], ids=["with_sidecar", "no_sidecar"])
def test_actorattack_lens_onto_a_legacy_lens_store_resumes_as_before(tmp_path, sidecar):
    out = tmp_path / "store"
    _mt_lens_store(out, sidecar=sidecar, stamped=False, transcripts=True)
    covered_lines = (out / "responses.jsonl").read_text().splitlines()[:2]
    res = _actor(_target(LENS), _goals(2), out)
    assert res["n_goals_run"] == 1
    lines = (out / "responses.jsonl").read_text().splitlines()
    assert lines[:2] == covered_lines                                        # legacy covered rows: verbatim
    new = [json.loads(l) for l in lines[2:]]
    assert new and all(r["goal_id"] == "g01" and r[FIELD] == LENS for r in new)
    assert [r["goal_id"] for r in _rows(out / "progress.jsonl")] == ["g00", "g01"]


def test_actorattack_lens_onto_a_native_store_refuses_too(tmp_path):
    out = tmp_path / "store"
    _actor(_target(NATIVE), _goals(1), out)
    before = _snapshot(out)
    with pytest.raises(st.ServedTargetError, match="'lora_native' serving engine"):
        _actor(_target(LENS), _goals(2), out)
    assert _snapshot(out) == before
    assert _actor(_target(NATIVE), _goals(2), out)["n_goals_run"] == 1     # native resumes native


# --------------------------------------------------------------------------- #
# multi-turn: the PyRIT runner (PAIR / TAP / Crescendo share it)               #
# --------------------------------------------------------------------------- #


def _run_pyrit(defense, goals, out, db, *, resume=True):
    from pyrit.models import Message
    from sisteer.eval.attacks.pyrit_adapters import AttackerChatTarget, run_pyrit_attack_async

    class _OneTurn:
        """A one-send executor over the REAL SteeredPromptTarget (its on_response persists)."""

        def __init__(self, target):
            self.target = target

        async def execute_async(self, *, objective):
            await self.target._send_prompt_to_target_async(
                normalized_conversation=[Message.from_prompt(prompt=objective, role="user")])
            return SimpleNamespace(outcome="failure", executed_turns=1)

    return asyncio.run(run_pyrit_attack_async(
        defense=defense, goals=goals,
        attacker_factory=lambda lock: AttackerChatTarget(gpu_lock=lock, generate_fn=lambda chat: "x"),
        scorer_fn=lambda p: "1",
        attack_factory=lambda *, target, adv_cfg, scorers: _OneTurn(target),
        out_dir=out, memory_db_path=db, resume=resume))


def test_pyrit_runner_stamps_every_persisted_reply(tmp_path):
    for engine in (LENS, NATIVE):
        out = tmp_path / engine
        res = _run_pyrit(_target(engine), _goals(2), out, tmp_path / f"{engine}.db")
        assert res["n_goals_run"] == 2
        rows = _rows(out / "responses.jsonl")
        assert len(rows) == 2 and all(r[FIELD] == engine for r in rows)
        assert FIELD not in _rows(out / "progress.jsonl")[0]
    hf = SimpleNamespace(n_target_forwards=0, mode="none")

    def chat(messages, *, max_new_tokens=512, **kw):
        hf.n_target_forwards += 1
        return "a benign answer"
    hf.chat = chat
    _run_pyrit(hf, _goals(1), tmp_path / "hf", tmp_path / "hf.db")
    assert all(set(r) == {"goal_id", "order", "response"} for r in _rows(tmp_path / "hf" / "responses.jsonl"))


@pytest.mark.parametrize("sidecar", [True, False], ids=["with_sidecar", "no_sidecar"])
@pytest.mark.parametrize("stamped", [False, True], ids=["legacy_rows", "lens_rows"])
def test_pyrit_runner_native_onto_a_lens_store_refuses_with_nothing_written(tmp_path, sidecar, stamped):
    out = tmp_path / "store"
    _mt_lens_store(out, sidecar=sidecar, stamped=stamped, transcripts=False)
    before = _snapshot(out)
    t = _target(NATIVE)
    with pytest.raises(st.ServedTargetError, match="'lora_native'") as ei:
        _run_pyrit(t, _goals(2), out, tmp_path / "m.db")
    assert ("lens by construction" in str(ei.value)) == (not stamped)
    assert _snapshot(out) == before and t.n_target_forwards == 0
    assert not (tmp_path / "m.db").exists()                                  # refused before memory init


@pytest.mark.parametrize("sidecar", [True, False], ids=["with_sidecar", "no_sidecar"])
def test_pyrit_runner_lens_onto_a_legacy_lens_store_resumes_as_before(tmp_path, sidecar):
    out = tmp_path / "store"
    _mt_lens_store(out, sidecar=sidecar, stamped=False, transcripts=False)
    covered_lines = (out / "responses.jsonl").read_text().splitlines()[:2]
    res = _run_pyrit(_target(LENS), _goals(2), out, tmp_path / "m.db")
    assert res["n_goals_run"] == 1
    lines = (out / "responses.jsonl").read_text().splitlines()
    assert lines[:2] == covered_lines
    new = [json.loads(l) for l in lines[2:]]
    # the REAL SteeredPromptTarget always sends an explicit sampler, so the served target
    # records a per-reply ``decode`` block beside the engine field — compare minus it
    assert all("decode" in r for r in new)
    assert [{k: v for k, v in r.items() if k != "decode"} for r in new] == [
        {"goal_id": "g01", "order": 0, "response": new[0]["response"], FIELD: LENS}]
    # and the native run onto the now-mixed-era lens store refuses byte-preserving
    before = _snapshot(out)
    with pytest.raises(st.ServedTargetError):
        _run_pyrit(_target(NATIVE), _goals(2), out, tmp_path / "n.db")
    assert _snapshot(out) == before


# --------------------------------------------------------------------------- #
# BoN                                                                          #
# --------------------------------------------------------------------------- #

ROUTES = [RouteEntry(virtual_model="petri__base", cell_id="none", cell_class="base",
                     serving_identity=PASS, intervention_hash="aa" * 8)]


def _bon(engine) -> bs.ServedDefense:
    return bs.ServedDefense(_achat, virtual_model="petri__base", cell_id="none",
                            proxy_identity=_identity(engine))


def _bon_run(defense, out, n_goals=2):
    return bs.bon_generate_served(defense, _goals(n_goals), out_dir=out, routes=ROUTES,
                                  n_max=2, max_concurrency=2, model="gemma4e4b")


def test_bon_rows_errors_and_manifest_carry_the_field(tmp_path):
    for engine in (LENS, NATIVE):
        out = tmp_path / engine
        jsonl = _bon_run(_bon(engine), out)
        rows = _rows(jsonl)
        assert len(rows) == 4 and all(r["engine"] == bs.SERVED_ENGINE and r[FIELD] == engine for r in rows)
        keys = list(rows[0])
        assert keys.index("engine") + 1 == keys.index(FIELD)
        man = json.loads((out / bs.BON_SERVED_MANIFEST).read_text())
        assert man[FIELD] == engine and man["engine"] == bs.SERVED_ENGINE

    # the typed-error ledger rows carry it as well
    class _OverWindow(bs.ServedDefense):
        async def generate_one_detailed(self, variant, *, sampling_seed, max_new_tokens,
                                        temperature=1.0, decode=None):
            self.n_target_forwards += 1
            raise sw.PromptOverWindow(prompt_tokens=8690, requested_max_tokens=max_new_tokens, window=8704,
                                      server_window=8704, effective_window=8704,
                                      reason="prompt 8690 leaves 14 of window 8704 (< 64 minimum completion)")

    d = _OverWindow(_achat, virtual_model="petri__base", cell_id="none", proxy_identity=NATIVE_IDENTITY)
    out = tmp_path / "errors"
    jsonl = _bon_run(d, out, n_goals=1)
    # the writer opens the responses file for append before the first unit (it has always
    # done so), so an all-errors run leaves it present and EMPTY — no row of any engine
    assert jsonl.exists() and jsonl.read_text() == ""
    erows = _rows(out / bs.BON_ERRORS_FILE)
    assert len(erows) == 2 and all(r[FIELD] == NATIVE and r["engine"] == bs.SERVED_ENGINE for r in erows)


def _strip_field(jsonl: Path) -> None:
    """Make a store's rows PREDATE the field (a pre-landing lens store)."""
    _write_rows(jsonl, [{k: v for k, v in r.items() if k != FIELD} for r in _rows(jsonl)])


def test_bon_cross_engine_refuses_byte_preserving_and_legacy_lens_resumes(tmp_path):
    # a legacy lens store (rows without the field) with a stale manifest beside it
    out = tmp_path / "legacy"
    jsonl = _bon_run(_bon(LENS), out)
    _strip_field(jsonl)
    man = json.loads((out / bs.BON_SERVED_MANIFEST).read_text()); man.pop(FIELD)
    (out / bs.BON_SERVED_MANIFEST).write_text(json.dumps(man, indent=2))
    before = _snapshot(out)
    with pytest.raises(bs.BonServedError, match="lens by construction") as ei:
        _bon_run(_bon(NATIVE), out)
    assert "'lora_native'" in str(ei.value)
    assert _snapshot(out) == before                       # rows AND manifest untouched (before the try/finally)
    # lens onto legacy lens: passes, nothing regenerated (every unit covered), rows stay legacy
    d = _bon(LENS)
    _bon_run(d, out)
    assert d.n_target_forwards == 0 and all(FIELD not in r for r in _rows(jsonl))
    # ... and a further goal's rows carry the field beside the legacy ones
    _bon_run(_bon(LENS), out, n_goals=3)
    rows = _rows(jsonl)
    assert [r.get(FIELD) for r in rows] == [None] * 4 + [LENS] * 2

    # explicit lens rows: native refuses; explicit native rows: lens refuses; each byte-preserving
    for store_engine, run_engine in ((LENS, NATIVE), (NATIVE, LENS)):
        out = tmp_path / f"{store_engine}_store"
        _bon_run(_bon(store_engine), out)
        before = _snapshot(out)
        with pytest.raises(bs.BonServedError, match=f"'{store_engine}' serving engine"):
            _bon_run(_bon(run_engine), out)
        assert _snapshot(out) == before
        d = _bon(store_engine)
        _bon_run(d, out)                                  # the same engine resumes (nothing left to do)
        assert d.n_target_forwards == 0


def test_bon_assert_engine_consistent_holds_the_serving_engine(tmp_path):
    p = tmp_path / "bon_responses.jsonl"
    _write_rows(p, [{"goal_id": "g0", "sample_idx": 0, "trial": 0, "engine": bs.SERVED_ENGINE}])
    bs._assert_engine_consistent(p)                                        # the pre-landing call: no compare
    bs._assert_engine_consistent(p, serving_engine=LENS)                   # absent = lens
    with pytest.raises(bs.BonServedError, match="lens by construction"):
        bs._assert_engine_consistent(p, serving_engine=NATIVE)
    _write_rows(p, [{"goal_id": "g0", "sample_idx": 0, "trial": 0, "engine": bs.SERVED_ENGINE, FIELD: NATIVE}])
    bs._assert_engine_consistent(p, serving_engine=NATIVE)
    with pytest.raises(bs.BonServedError, match="'lora_native' serving engine"):
        bs._assert_engine_consistent(p, serving_engine=LENS)
    # the engine-tag refusal still comes first for an HF row
    _write_rows(p, [{"goal_id": "g0", "sample_idx": 0, "trial": 0}])
    with pytest.raises(bs.BonServedError, match="mix HF and vLLM"):
        bs._assert_engine_consistent(p, serving_engine=LENS)
    with pytest.raises(bs.BonServedError, match="serving_engine must be one of"):
        bs._assert_engine_consistent(p, serving_engine="vllm")


def test_bon_manifest_omits_the_key_for_a_stand_in_without_an_engine(tmp_path):
    fake = SimpleNamespace(cell_id="none", virtual_model="petri__base", clamp_report=lambda: {})
    man = json.loads(bs.write_bon_served_manifest(tmp_path, fake, counts={}, max_concurrency=1, model="gemma4e4b").read_text())
    assert FIELD not in man and man["engine"] == bs.SERVED_ENGINE


def test_write_target_manifest_holds_the_engine_boundary(tmp_path):
    """Review r1 (e): the ONE manifest writer refuses to replace a manifest another engine wrote — ``started`` or
    ``final``, covered goals or none — with the store byte-identical (absent field = lens by era; an unreadable manifest
    cannot be established as either engine and is refused, never overwritten). The same engine rewrites it (started ->
    final, the lens resume of a lens-era store); a stand-in with no engine stamps nothing and compares nothing."""
    out = tmp_path / "lens_era"; out.mkdir()
    (out / "served_target.json").write_text(json.dumps({"target_engine": "served", "manifest_stage": "final", "decode": None}))
    (out / "responses.jsonl").write_text("")
    before = _snapshot(out)
    for stage in ("started", "final"):
        with pytest.raises(st.ServedTargetError, match="lens by construction"):
            st.write_target_manifest(out, _target(NATIVE), stage=stage)
    assert _snapshot(out) == before
    # the lens target rewrites it (a lens run onto a lens-era store) and stamps its engine, started then final
    st.write_target_manifest(out, _target(LENS), stage="started")
    doc = json.loads(st.write_target_manifest(out, _target(LENS), stage="final").read_text())
    assert doc[FIELD] == LENS and doc["manifest_stage"] == "final"
    # a stamped manifest refuses the other engine by its recorded value, either way round
    with pytest.raises(st.ServedTargetError, match="records the store's serving engine as 'lens'"):
        st.write_target_manifest(out, _target(NATIVE))
    nat = tmp_path / "native"
    st.write_target_manifest(nat, _target(NATIVE))
    with pytest.raises(st.ServedTargetError, match="records the store's serving engine as 'lora_native'"):
        st.write_target_manifest(nat, _target(LENS), stage="final")
    assert json.loads(st.write_target_manifest(nat, _target(NATIVE), stage="final").read_text())["manifest_stage"] == "final"
    # a stand-in with no engine (the pre-landing duck type) still writes and compares nothing
    fake = SimpleNamespace(route_info={}, evidence={}, timing_log=None, max_model_len=None, n_clamped=0,
                           n_prompt_over_window=0, n_window_mismatch=0, window_mismatches=[], n_tokenize_resolved=0,
                           n_floor_only_unresolvable=0, n_target_forwards=0, n_all_turns_refused={})
    assert FIELD not in json.loads(st.write_target_manifest(tmp_path / "standin", fake).read_text())
    # unreadable: refused, untouched
    (out / "served_target.json").write_text("{not json")
    with pytest.raises(st.ServedTargetError, match="unreadable"):
        st.write_target_manifest(out, _target(LENS))
    assert (out / "served_target.json").read_text() == "{not json"
    with pytest.raises(ValueError, match="serving_engine must be one of"):
        st.assert_manifest_serving_engine_compatible(out, "vllm")


def test_a_silent_row_is_described_by_the_run_manifest_not_by_the_era_inference(tmp_path):
    """A NATIVE static store whose rows carry no engine key at all (its producing
    run's call path stamped none, so neither `engine` nor `serving_engine` was written) was read as
    LENS by the absence rule, so the store's own engine could not append to it and every re-run
    refused — 68 of 69 driver failures on one job, the passthrough `none` cells included. A silent
    row is now described by the run MANIFEST when one names an engine; the era inference survives
    only where nothing else speaks, and a row that STATES an engine is still compared literally."""
    def store(name, rows, manifest_engine):
        out = tmp_path / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        if manifest_engine is not None:
            (out / st.TARGET_MANIFEST_FILE).write_text(
                json.dumps({"target_engine": "served", FIELD: manifest_engine}))
        return out

    silent = [{"goal_id": "g0", "response": "x"}, {"goal_id": "g1", "response": "y"}]
    # THE BUG: silent rows + a manifest naming the native engine -> a native append is admitted
    native_store = store("native", silent, NATIVE)
    st.assert_store_serving_engine_compatible(native_store, NATIVE)
    # ... and a LENS append onto that same store still refuses, naming what the manifest says
    with pytest.raises(st.ServedTargetError, match="lora_native"):
        st.assert_store_serving_engine_compatible(native_store, LENS)
    # a manifest naming lens describes its silent rows as lens (the ordinary in-flight case)
    lens_store = store("lens", silent, LENS)
    st.assert_store_serving_engine_compatible(lens_store, LENS)
    with pytest.raises(st.ServedTargetError, match="lens"):
        st.assert_store_serving_engine_compatible(lens_store, NATIVE)
    # NO manifest: the era inference stands unchanged — silent means lens, native refuses
    legacy = store("legacy", silent, None)
    st.assert_store_serving_engine_compatible(legacy, LENS)
    with pytest.raises(st.ServedTargetError, match="lens by construction"):
        st.assert_store_serving_engine_compatible(legacy, NATIVE)
    # a row that STATES an engine is never overridden by the manifest: a torn store still refuses
    torn = store("torn", [{"goal_id": "g0", FIELD: LENS}], NATIVE)
    with pytest.raises(st.ServedTargetError, match="lens"):
        st.assert_store_serving_engine_compatible(torn, NATIVE)
