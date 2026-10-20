"""Four review findings on commit f4f3d465 (2 P1 + 2 P2), each RE-VALIDATED and fixed with a KNOWN-BAD case beside the
good one. Everything synthetic: a placeholder wave sha and uuid, invented goal ids, no host
names.

The request-evidence tests run the REAL audit over rung attempt roots whose evidence files THIS consumer emitted;
the client and the audit are loaded by path from this tree or the sibling worktrees (``_seedrep_testlib.peer_file``)
and the tests SKIP by name when neither is present."""
from __future__ import annotations

import copy
import json
import sys
import types
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _seedrep_testlib as T  # noqa: E402

L = T.load_ladder()
RR = T.load_rung_redraw()


# =========================================================================== #
# SECTION P2-4 — RemoteVolume: an absent path is empty; auth / transport / API failures propagate
# =========================================================================== #


class _ListdirClient:
    """A modal.Volume's `listdir` as the adapter calls it: `tree` = {dir path: [file names]}; `fail` raised on every call."""

    def __init__(self, tree, fail=None):
        self.tree, self.fail, self.calls = dict(tree), fail, []

    def listdir(self, path, *, recursive=False):
        self.calls.append(path)
        if self.fail is not None:
            raise self.fail
        if path not in self.tree:
            raise FileNotFoundError(path)              # the SDK's reading of an absent path (volume.py maps NotFoundError onto it)
        return [types.SimpleNamespace(path=f"/{path}/{n}", type=types.SimpleNamespace(name="FILE")) for n in self.tree[path]]


def test_sol_r2_p2_4_remote_volume_reads_an_absent_path_as_empty_and_propagates_every_other_failure():
    rv = L.RemoteVolume(_ListdirClient({"seedrep/x/queue": ["a.json"]}))
    assert rv.listdir("seedrep/x/queue") == ["a.json"] and rv.exists("seedrep/x/queue/a.json") is True
    assert rv.listdir("seedrep/x/nothing") == [] and rv.walk("seedrep/x/nothing") == [] and rv.exists("seedrep/x/nothing/CLOSED.json") is False

    class AuthError(Exception):
        pass

    class ConnectionError_(Exception):
        pass

    # KNOWN-BAD: an expired token, a dead connection, a service error — each surfaces from every read, never as "no files"
    for exc in (AuthError("token expired"), ConnectionError_("connection reset"), RuntimeError("grpc UNAVAILABLE")):
        rv = L.RemoteVolume(_ListdirClient({}, fail=exc))
        for call in (lambda: rv.listdir("seedrep/x/queue"), lambda: rv.walk("seedrep/x/queue"), lambda: rv.exists("seedrep/x/CLOSED.json")):
            with pytest.raises(type(exc), match=str(exc)):
                call()
    # the reconciler over such a volume propagates too (its first read is the queue listing): a queue never reads as empty
    spec, kv = T.static_spec(L, shards=("0",)), T.kvs(L)
    with pytest.raises(AuthError):
        L.reconcile_queue(L.RemoteVolume(_ListdirClient({}, fail=AuthError("token expired"))), kv["accepted"], spec)


def test_sol_r2_p2_4_modals_own_not_found_error_is_admitted_as_absent_when_a_client_is_present(monkeypatch):
    class NotFoundError(Exception):
        pass

    class AuthError(Exception):
        pass

    mod = types.ModuleType("modal")
    mod.exception = types.SimpleNamespace(NotFoundError=NotFoundError, AuthError=AuthError)
    monkeypatch.setitem(sys.modules, "modal", mod)
    assert set(L._absent_path_errors()) == {FileNotFoundError, NotFoundError}
    assert L.RemoteVolume(_ListdirClient({}, fail=NotFoundError("no such path"))).walk("seedrep/x/queue") == []
    with pytest.raises(AuthError):                         # KNOWN-BAD: the client's auth error is not "absent"
        L.RemoteVolume(_ListdirClient({}, fail=AuthError("bad token"))).walk("seedrep/x/queue")
    monkeypatch.delitem(sys.modules, "modal")
    assert L._absent_path_errors() == (FileNotFoundError,)


# =========================================================================== #
# SECTION P2-3 — redraw.sampling: the complete frozen sampler key set, at enqueue, before the send, at the assembly
# =========================================================================== #


def _record_with_sampling(good, sampling, goal="g1"):
    rec = copy.deepcopy(good)
    rec["item_key"] = L.item_key("static", "pasttense", goal, 1)
    rec["identity"] = {**good["identity"], "goal_id": goal}
    if sampling is None:
        rec["redraw"].pop("sampling", None)
    else:
        rec["redraw"]["sampling"] = sampling
    return rec


def test_sol_r2_p2_3_a_drawable_record_needs_the_complete_frozen_sampler_row_at_enqueue_and_before_the_send():
    shared, kv = L.SharedVolume(), T.kvs(L)
    q8 = T.static_spec(L, shards=("0",))
    c = T.Container(L, shared, q8, "0", "a1")
    good = c.enqueue_row(T.static_row("g0", 1, seed=7))
    assert tuple(L.sampler_keys()) == ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")
    assert set(good["redraw"]["sampling"]) == set(L.sampler_keys()) and RR.request_body(good)["top_k"] == 20
    # KNOWN-BAD 1: the block ABSENT — the wire would run under the server's defaults
    absent = _record_with_sampling(good, None, "g1")
    with pytest.raises(L.LadderRefusal, match="redraw.sampling is None, not the sampler row"):
        c.producer.enqueue(absent)
    with pytest.raises(L.LadderRefusal, match="redraw.sampling is None, not the sampler row"):
        RR.request_body(absent)
    # KNOWN-BAD 2: top_k OMITTED — that one term would default server-side
    no_top_k = _record_with_sampling(good, {k: v for k, v in T.PRESET.items() if k != "top_k"}, "g2")
    with pytest.raises(L.LadderRefusal, match=r"missing \['top_k'\], extra \[\]"):
        c.producer.enqueue(no_top_k)
    with pytest.raises(L.LadderRefusal, match=r"missing \['top_k'\]"):
        RR.request_body(no_top_k)
    # KNOWN-BAD 3: a term beyond the frozen set (no preset froze it)
    with pytest.raises(L.LadderRefusal, match=r"missing \[\], extra \['n'\]"):
        c.producer.enqueue(_record_with_sampling(good, {**T.PRESET, "n": 2}, "g3"))
    # KNOWN-BAD 4: not a mapping
    with pytest.raises(L.LadderRefusal, match="not the sampler row"):
        RR.request_body(_record_with_sampling(good, [0.7, 0.8], "g4"))
    # a record read back off the volume is validated the same way (a forged file cannot smuggle a bare block past the projection)
    c.complete()
    assert c.finish(kv["accepted"])["state"] == "accepted"
    rel = L.queue_entry_rel(q8.ladder_root, "none__s1", good["item_key"], "a1")
    shared.files[rel] = (L.canonical_json(absent | {"item_key": good["item_key"], "identity": good["identity"]}) + "\n").encode()
    with pytest.raises(L.LadderRefusal, match="redraw.sampling is None"):
        L.accepted_queue_entries(shared.read_only_view(), kv["accepted"], q8)
    # a GSM8K record has no redraw block by design (Inspect draws) and stays valid
    gspec = L.QueueSpec.build(T.W7, T.WAVE, "gsm8k", "s1", "gsm8k", 16384, ["0"])
    grec = L.gsm8k_queue_record(gspec, arm="none", s=1, item={"item_id": "q1", "sample_idx": 0, "at_cap": True, "stop_reason": "max_tokens", "seed": 5},
                                base_store="b", producer=gspec.producers[0], producer_attempt_id="a1")
    assert "redraw" not in grec and L.check_redraw_sampling(grec, where="x") == {}
    L._validate_record(grec, gspec, where="x")


def test_sol_r2_p2_3_the_assembly_treats_an_absent_sampling_block_as_a_preset_mismatch():
    shared, kv = L.SharedVolume(), T.kvs(L)
    q8 = T.static_spec(L, shards=("0",))
    c = T.Container(L, shared, q8, "0", "a1")
    good = c.enqueue_row(T.static_row("g0", 1, seed=7))
    record = {"text": "t", "finish_reason": "stop", "completion_tokens": 3, "prompt_tokens": 1, "max_tokens": 8192, "sampling_seed": 7}
    rv = shared.snapshot("rung-1")
    path, doc = L.write_materialised(rv, q8, queue_record=good, attempt_id="rung-1", call_id="fc", record=record, request_id="rq", completion_id="c")
    el = L.Elected(good["item_key"], "none__s1", {"attempt_id": "rung-1", "path": path, "text_sha256": doc["text_sha256"]}, doc, path)
    L.verify_elected(el, base_row_sha256=good["base_row_sha256"], sampling_seed=7, cap=8192, preset=T.PRESET, accepted_producers=["a1"])
    # KNOWN-BAD: a materialised document (an older / foreign writer's) whose queue record carries NO sampling block
    bare = copy.deepcopy(doc)
    bare["queue_record"]["redraw"].pop("sampling")
    el_bare = L.Elected(good["item_key"], "none__s1", el.value, bare, path)
    with pytest.raises(L.ElectedMismatch, match="drawn under sampling None \\(absent = server defaults\\)"):
        L.verify_elected(el_bare, base_row_sha256=good["base_row_sha256"], sampling_seed=7, cap=8192, preset=T.PRESET, accepted_producers=["a1"])
    # ... and one whose block omits top_k
    partial = copy.deepcopy(doc)
    partial["queue_record"]["redraw"]["sampling"].pop("top_k")
    with pytest.raises(L.ElectedMismatch, match="drawn under sampling"):
        L.verify_elected(L.Elected(good["item_key"], "none__s1", el.value, partial, path), base_row_sha256=good["base_row_sha256"],
                         sampling_seed=7, cap=8192, preset=T.PRESET, accepted_producers=["a1"])
    # with NO preset supplied (a caller that has none) the block is not judged — the seams always pass this rung's preset
    L.verify_elected(el_bare, base_row_sha256=good["base_row_sha256"], sampling_seed=7, cap=8192, preset=None, accepted_producers=["a1"])


# =========================================================================== #
# SECTION P1-1 — a lost container re-executed under the SAME Modal FunctionCall resumes its claims
# =========================================================================== #


def _draw_stop(rec):
    text = f"redraw {rec['identity']['goal_id']}"
    return RR.Draw(record={"text": text, "finish_reason": "stop", "completion_tokens": 3, "prompt_tokens": 1, "max_tokens": 8192,
                           "sampling_seed": rec["sampling_seed"]}, text=text, response=None, request_id=f"rq-{rec['item_key']}", completion_id="cmpl-x")


def _closed_static_world():
    shared, kv = L.SharedVolume(), T.kvs(L)
    q8 = T.static_spec(L, shards=("0",))
    c = T.Container(L, shared, q8, "0", "a1")
    for row in T.rows3():
        c.enqueue_row(row)
    c.complete()
    assert c.finish(kv["accepted"])["state"] == "accepted"
    assert L.reconcile_queue(shared.read_only_view(), kv["accepted"], q8)["queue_closed"]
    return shared, kv, q8


def test_sol_r2_p1_1_a_re_execution_of_the_same_call_under_a_new_attempt_id_draws_and_elects_what_its_predecessor_claimed():
    shared, kv, q8 = _closed_static_world()
    g0 = L.item_key("static", "pasttense", "g0", 1)
    # execution A (attempt rung-A, FunctionCall fc-C) claims g0 and loses its container before writing / electing
    assert L.claim_item(kv["claims"], q8, "none__s1", g0, attempt_id="rung-A", call_id="fc-C")
    # the reconciler sees the call RUNNING (Modal is re-executing it) and, correctly, leaves the claim alone
    st = L.reconcile_queue(shared.read_only_view(), kv["accepted"], q8, claims_kv=kv["claims"], results_kv=kv["results"],
                           lease_state=lambda cid: "running")
    assert st["reclaimed"] == [] and len(st["claims_left"]) == 1
    # execution B: a NEW attempt id is minted under the SAME call id (seedrep_served.mint_attempt_id + current_function_call_id)
    t = RR.consume(shared.snapshot("rung-B"), accepted_kv=kv["accepted"], claims_kv=kv["claims"], results_kv=kv["results"], spec=q8,
                   attempt_id="rung-B", call_id="fc-C", draw=_draw_stop, sleep=lambda s: None, log=lambda m: None)
    assert (t["drawn"], t["won"], t["claims_resumed"], t["skipped_held"], t["closed"], t["covered"]) == (3, 3, 1, 0, True, True)
    el = L.load_elected_one(shared.read_only_view(), kv["results"], q8, "none__s1", g0)
    assert el.value["attempt_id"] == "rung-B" and el.doc["claimant"] == {"attempt_id": "rung-B", "call_id": "fc-C"}
    assert el.path == L.materialised_rel(q8.ladder_root, "none__s1", g0, "rung-B")
    # the claim record itself is untouched (no takeover write): the CALL is what bound the two executions
    assert kv["claims"].get(L.item_scoped_key(q8.queue_key, "none__s1", g0)) | {"ts": None} == {"attempt_id": "rung-A", "call_id": "fc-C", "ts": None}


def test_sol_r2_p1_1_a_claim_of_a_different_live_call_is_still_held_and_the_predicate_never_matches_an_empty_call():
    shared, kv, q8 = _closed_static_world()
    g0 = L.item_key("static", "pasttense", "g0", 1)
    assert L.claim_item(kv["claims"], q8, "none__s1", g0, attempt_id="rung-X", call_id="fc-D")   # another LIVE container's claim
    t = RR.consume(shared.snapshot("rung-B"), accepted_kv=kv["accepted"], claims_kv=kv["claims"], results_kv=kv["results"], spec=q8,
                   attempt_id="rung-B", call_id="fc-C", draw=_draw_stop, sleep=lambda s: None, log=lambda m: None, max_rounds=2)
    assert (t["drawn"], t["skipped_held"], t["claims_resumed"], t["closed"], t["covered"]) == (2, 2, 0, True, False)   # KNOWN-BAD: held both rounds
    assert L.load_elected_one(shared.read_only_view(), kv["results"], q8, "none__s1", g0) is None
    # the predicate, pinned: same attempt -> ours; same call, other attempt -> ours; other call -> held; an empty call id never binds
    assert L.claim_is_this_calls({"attempt_id": "rung-A", "call_id": "fc-C"}, attempt_id="rung-A", call_id="fc-Z") is True
    assert L.claim_is_this_calls({"attempt_id": "rung-A", "call_id": "fc-C"}, attempt_id="rung-B", call_id="fc-C") is True
    assert L.claim_is_this_calls({"attempt_id": "rung-A", "call_id": "fc-C"}, attempt_id="rung-B", call_id="fc-D") is False
    assert L.claim_is_this_calls({"attempt_id": "rung-A", "call_id": ""}, attempt_id="rung-B", call_id="") is False
    assert L.claim_is_this_calls(None, attempt_id="rung-B", call_id="fc-C") is False


# =========================================================================== #
# SECTION request evidence — every rung send through the client; the REAL audit over the emitted files
# =========================================================================== #


def _engine_proxy():
    """The rung container's proxy + engine as the loopback stub plays them: every request answered `stop`, engine-shaped
    completion ids (`chatcmpl-<24 hex>`), the exact request bytes + headers + the completion id kept per request."""
    proxy = T.StubProxy(decide=lambda b: (f"redrawn {b['messages'][-1]['content']}", "stop", 23))
    proxy.completion_id = lambda n: f"chatcmpl-{uuid.uuid4().hex[:24]}"
    return proxy


def _ledger_and_engine_log(proxy, C):
    """What the PROXY and the ENGINE of the rung container write for the requests the consumer sent: one usage-ledger row per
    attempt (`contracts.LEDGER_ROW_FIELDS`, the proxy's own reading of the same wire — request id, labels, body digest,
    completion id) and one engine `Received request <completion id>-<8 hex>` line (the audit fixtures' shapes)."""
    rows, lines = [], []
    for seen in proxy.bodies:
        h = {k.lower(): v for k, v in seen["headers"].items()}
        body, cid = seen["body"], seen["completion_id"]
        row = dict.fromkeys(C.LEDGER_ROW_FIELDS)
        row.update(ts="2026-01-01T00:00:00Z", task=h["x-tinycap-task"], sample=h["x-tinycap-sample"], vmodel=body["model"], thinking=None,
                   prompt_tokens_pre=11, max_tokens_requested=body["max_tokens"], has_tools=False, n_tools=0, status=200, error_code=None,
                   rejected=False, usage_prompt_tokens=11, usage_completion_tokens=23, finish_reason="stop", latency_s=0.5,
                   tokenize_mismatch=False, error_class="none", request_id=h["x-tinycap-request-id"], body_sha256=C.wire_body_sha256(seen["raw"]))
        row["completion_id"] = cid
        rows.append(row)
        lines.append(f"INFO [request_logger.py:64] Received request {cid}-{uuid.uuid4().hex[:8]}: params: SamplingParams(n=1, "
                     f"seed={body.get('seed')}, max_tokens={body.get('max_tokens')}), lora_request: None.")
    return rows, lines


def _rung_root_from_consumer(tmp_path, monkeypatch, *, view, arm, seeds):
    """A base root (a synthetic audit fixture, one static store per seed slot, g0 at cap) + a RUNG root whose request pipeline
    THIS consumer emitted: the CLOSED 8192 queue of every at-cap base row, `consume` through `http_draw_fn` with the request
    evidence pointed at the rung attempt root (the env contract for the rung child: ENV_DIR only), the proxy's ledger and
    the engine's log for what was sent, and the rung store per seed slot in escalate_static's emission — copied rows verbatim
    with their provenance, the at-cap rows from the ELECTED materialised records — every row held to the writer's own
    `row_source_cap` rule. Returns (base, rung, tally, evidence, (ac, rea, fx, C))."""
    REC = T.load_request_evidence_client()
    ac, rea, fx = T.load_ws3g_audit()
    from sisteer.capeval.tinycap import contracts as C

    stores = tmp_path / "stores"
    pin = ("a" * 64) if view == "arditi" else None
    base = fx.make_root(stores, leg="pasttense", view=view, arms=(arm,), arditi_pin=pin)
    base_shards = {s: fx.write_static_store(base, family="pasttense", arm=arm, s=s, goals=["g0", "g1"], at_cap_goals=("g0",)) for s in seeds}
    base.flush_pipeline()
    rung = fx.make_root(stores, leg="pasttense", view=view, arms=(arm,), stage="8192", arditi_pin=pin)
    spec = L.QueueSpec.build(T.W7, T.WAVE, "pasttense", view, "static", 8192, ["0"])
    vol, kv = L.LocalVolume(tmp_path / "vol"), T.kvs(L)
    records = [L.static_queue_record(spec, arm=arm, s=s, family="pasttense", row=row, base_store=str(shard / "responses.jsonl"),
                                     producer=spec.producers[0], producer_attempt_id="a1", vmodel=f"petri__{arm}", sampling=T.PRESET)
               for s, shard in base_shards.items() for row in ac.read_jsonl(shard / "responses.jsonl") if row["finish_reason"] == "length"]
    T.close_queue(L, vol, spec, records, kv["accepted"])
    monkeypatch.setenv(REC.ENV_DIR, str(rung.path))
    monkeypatch.delenv(REC.ENV_EXTRA, raising=False)
    monkeypatch.delenv(REC.ENV_SEED_BASE, raising=False)
    evidence = RR.RungEvidence.from_env(leg="pasttense", attempt_id=rung.attempt_id)
    proxy = _engine_proxy()
    try:
        tally = RR.consume(vol, accepted_kv=kv["accepted"], claims_kv=kv["claims"], results_kv=kv["results"], spec=spec, attempt_id=rung.attempt_id,
                           call_id="fc-rung", draw=RR.http_draw_fn(proxy.url, evidence=evidence), width=2, sleep=lambda s: None, log=lambda m: None)
    finally:
        proxy.close()
    assert tally["closed"] and tally["covered"] and tally["drawn"] == tally["won"] == len(records) == 2 * len(seeds)
    rows, lines = _ledger_and_engine_log(proxy, C)
    fx._jsonl(rung.path / ac.LEDGER_FILE, rows)
    (rung.serve / "vllm_server.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    esc = ac.escalate_static_module()
    for s, shard in base_shards.items():
        elected = L.load_elected(vol, kv["results"], spec, L.arm_s(arm, s))
        base_rows, base_prog = ac.read_jsonl(shard / "responses.jsonl"), ac.read_jsonl(shard / "progress.jsonl")
        out_rows, regen_per_goal, n_after = [], {}, 0
        for b in base_rows:
            row = dict(b)
            row["escalated_from_cap"] = 4096
            src_cap = int(esc.row_source_cap(b, store_cap=4096, where=str(shard)))
            el = elected.get(L.item_key("static", "pasttense", b["goal_id"], int(b["sample_idx"])))
            if el is None:
                row["regenerated"], row["source_cap"] = False, src_cap
                n_after += int(bool(b.get("at_cap")))
            else:
                r = el.doc["record"]
                row.update(response=r["text"], decode={**b["decode"], "cap": 8192}, finish_reason=r["finish_reason"], completion_tokens=r["completion_tokens"],
                           prompt_tokens=r["prompt_tokens"], max_tokens=8192, at_cap=r["finish_reason"] == "length", regenerated=True, source_cap=8192,
                           source_completion_tokens=b.get("completion_tokens"), source_finish_reason=b.get("finish_reason"), source_draw_cap=src_cap,
                           boundary_landing=int(r["completion_tokens"]) < src_cap, prefix_extension=True, identical=False, first_divergence_char=None,
                           first_divergence_token=None, request_id=r[RR.ROW_REQUEST_ID_FIELD], completion_id=r[RR.ROW_COMPLETION_ID_FIELD])
                regen_per_goal[b["goal_id"]] = regen_per_goal.get(b["goal_id"], 0) + 1
                n_after += int(row["at_cap"])
            esc.row_source_cap(row, store_cap=8192, where=str(shard))
            out_rows.append(row)
        rdir = rung.path / "exp" / fx._private_run(view, 8192).format(s=s) / "pasttense" / arm / "shard00"
        progress = [{**p, "n_target_forwards": regen_per_goal.get(p["goal_id"], 0), "n_regenerated": regen_per_goal.get(p["goal_id"], 0),
                     "n_copied": int(p.get("n_variants", 0)) - regen_per_goal.get(p["goal_id"], 0)} for p in base_prog]
        block_to = {**base_rows[0]["decode"], "cap": 8192}
        fx._jsonl(rdir / "responses.jsonl", out_rows)
        fx._jsonl(rdir / "progress.jsonl", progress)
        fx._json(rdir / "served_target.json", {"target_engine": "served", "virtual_model": f"petri__{arm}", "cell_id": arm, "manifest_stage": "final",
                                                "decode": block_to})
        n_regen = sum(regen_per_goal.values())
        fx._json(rdir / "escalation.json", {"attack": "pasttense", "defense": arm, "cap_from": 4096, "cap_to": 8192, "n_rows": len(out_rows),
                                             "n_at_cap_before": sum(1 for r in base_rows if r.get("at_cap")), "n_regenerated": n_regen,
                                             "n_at_cap_after": n_after, "n_copied": len(out_rows) - n_regen,
                                             "n_rows_by_source_cap": {str(c): sum(1 for r in out_rows if r["source_cap"] == c)
                                                                      for c in sorted({r["source_cap"] for r in out_rows})},
                                             "seed_base": s, "trial": int(block_to["trial"]), "arm": block_to["arm"], "decode": block_to,
                                             "re_draw_not_continuation": True})
    fx._json(rung.path / ac.COMPLETE_FILE, {"written_utc": "2026-01-01T02:00:00Z", "attempt_id": rung.attempt_id})
    return base, rung, tally, evidence, (ac, rea, fx, C)


def _real_audit(rung, base, peers, out_dir):
    ac, rea, fx, _ = peers
    return rea.audit_root(ac.read_attempt_root(rung.path), out_dir=out_dir, legs=["pasttense"], policy_file=None,
                          thinking_expect=dict(ac.THINKING_EXPECT_DEFAULT), petri_behaviours=list(fx.PETRI_BEHAVIOURS), model_key=ac.MODEL_KEY,
                          client_status="absent", siblings=[ac.read_attempt_root(base.path)])


def test_sol_r2_p1_2_an_ordinary_rung_roots_emitted_evidence_passes_ws3gs_real_audit_and_unbound_rows_refuse(tmp_path, monkeypatch):
    base, rung, tally, evidence, peers = _rung_root_from_consumer(tmp_path, monkeypatch, view="s1", arm="none", seeds=(1,))
    ac, rea, fx, C = peers
    from sisteer.eval.attacks.static_common import static_sampling_seed
    recs, resp = ac.read_jsonl(rung.path / ac.EVIDENCE_FILE), ac.read_jsonl(rung.path / ac.RESPONSE_EVIDENCE_FILE)
    assert len(recs) == len(resp) == tally["drawn"] == 2 and evidence.summary() | {"dir": None} == {"dir": None, "n_requests": 2, "n_responses": 2, "arm_s": ["none__s1"]}
    for r in recs:                                                       # the version-3 record from the EXACT wire bytes, labelled by the client's API
        assert r["request_evidence_version"] == C.REQUEST_EVIDENCE_VERSION and r["kind"] == "task" and r["task"] == "static/pasttense"
        assert r["vmodel"] == "petri__none" and r["item_id"] == "g0" and r["sample"] == f"g0/{r['sample_idx']}" and r["max_tokens"] == 8192
        assert (r["temperature"], r["top_p"], r["top_k"], r["min_p"], r["repetition_penalty"]) == (0.7, 0.8, 20, 0.0, 1.0)
        assert r["presence_penalty_present"] is True and r["presence_penalty_value"] == 1.5 and r["enable_thinking_present"] is False
        assert r["seed"] == static_sampling_seed(1, 0, "pasttense", "g0", r["sample_idx"])
        assert {k: r["extra"][k] for k in ("leg", "arm", "s", "attempt_id")} == {"leg": "pasttense", "arm": "none", "s": 1, "attempt_id": rung.attempt_id}
    assert {r["request_id"] for r in recs} == {r["request_id"] for r in resp} and all(r["status"] == 200 and r["completion_id"] for r in resp)
    # the REAL audit: the store's verdict over THIS root's records / ledger / engine log, the copied rows bound on the base root
    (v,) = _real_audit(rung, base, peers, tmp_path / "audits")
    assert v["verdict"] == "AGREES", v["problem"]
    assert (v["stage"], v["s"], v["arm"], v["n_draws"], v["n_draws_bound"], v["n_records_task"]) == ("8192", 1, "none", 4, 4, 2)
    rb = v["checks"]["row_binding"]
    assert rb["n_copied_forward"] == 2 and rb["ancestor_attempt_ids"] == {"4096": [base.attempt_id]} and rb["n_problems"] == 0
    assert Path(v["path"]).name == "none__s1__cap8192.json"
    # KNOWN-BAD 1: the pre-fix materialised record carried no binding -> rows without request_id / completion_id do not bind
    rdir = Path(v["store_dir"]) / "shard00"
    fx.rewrite_jsonl(rdir / "responses.jsonl", lambda r: {k: val for k, val in r.items() if not (r.get("regenerated") and k in ("request_id", "completion_id"))})
    (v2,) = _real_audit(rung, base, peers, tmp_path / "audits2")
    assert v2["verdict"] == "REFUSED" and "carries no request_id / completion_id" in v2["problem"]
    # KNOWN-BAD 2: a rung root with NO request_evidence.jsonl (the pre-fix consumer) cannot be audited at all
    (rung.path / ac.EVIDENCE_FILE).unlink()
    with pytest.raises(ac.AuditRefusal):
        _real_audit(rung, base, peers, tmp_path / "audits3")


def test_sol_r2_p1_2_the_arditi_rung_root_stamps_extra_s_per_request_and_the_audit_attributes_three_stores(tmp_path, monkeypatch):
    ac0, _, _ = T.load_ws3g_audit()
    base, rung, tally, evidence, peers = _rung_root_from_consumer(tmp_path, monkeypatch, view="arditi", arm=ac0.ARDITI_CELL, seeds=(1, 2, 3))
    ac, rea, fx, C = peers
    from sisteer.eval.attacks.static_common import static_sampling_seed
    recs = ac.read_jsonl(rung.path / ac.EVIDENCE_FILE)
    assert len(recs) == tally["drawn"] == 6 and evidence.summary()["arm_s"] == ["arditi_allpos__s1", "arditi_allpos__s2", "arditi_allpos__s3"]
    # extra.s is the SEED SLOT of the record each request re-drew — the body's seed is that slot's rule, not a constant of the container
    assert sorted({r["extra"]["s"] for r in recs}) == [1, 2, 3]
    for r in recs:
        assert r["seed"] == static_sampling_seed(r["extra"]["s"], 0, "pasttense", r["item_id"], r["sample_idx"]) and r["extra"]["arm"] == "arditi_allpos"
    verdicts = _real_audit(rung, base, peers, tmp_path / "audits")
    assert sorted(v["s"] for v in verdicts) == [1, 2, 3]
    for v in verdicts:
        assert v["verdict"] == "AGREES", (v["s"], v["problem"])
        assert v["n_draws"] == 4 and v["n_draws_bound"] == 4 and v["n_records_task"] == 2 and v["view"] == "arditi" and v["stage"] == "8192"
    assert sorted(Path(v["path"]).name for v in verdicts) == ["arditi_allpos__s1__cap8192.json", "arditi_allpos__s2__cap8192.json",
                                                             "arditi_allpos__s3__cap8192.json"]
    # KNOWN-BAD: ONE seed base stamped on every record (a static extra — the pre-fix shape) cannot be attributed: the audit refuses
    fx.rewrite_jsonl(rung.path / ac.EVIDENCE_FILE, lambda r: {**r, "extra": {**r["extra"], "s": 1}})
    bad = _real_audit(rung, base, peers, tmp_path / "audits2")
    assert {v["s"] for v in bad if v["verdict"] == "REFUSED"} >= {2, 3}, [(v["s"], v["verdict"]) for v in bad]


def test_sol_r2_p1_2_labels_env_contract_and_failure_records():
    REC = T.load_request_evidence_client()
    spec = T.static_spec(L, shards=("0",))
    static = L.static_queue_record(spec, arm="none", s=1, family="pasttense", row=T.static_row("g0", 1, seed=7), base_store="b",
                                   producer=spec.producers[0], producer_attempt_id="a1", vmodel="petri__none", sampling=T.PRESET)
    assert RR.evidence_label(static) == ("static/pasttense", "g0", 1)
    cspec = L.QueueSpec.build(T.W7, T.WAVE, "controls", "s1", "controls", 2048, ["0"])
    ev = {"item_id": "xst0001-bbbbbbbb", "at_cap": True, "finish_reason": "length", "sampling_seed": 5, "text_sha256": "a" * 64}
    ctl = L.controls_queue_record(cspec, arm="none", s=1, control="xstest", event=ev, prompt="p", system=None, base_store="s",
                                  producer=cspec.producers[0], producer_attempt_id="a", vmodel="petri__none", sampling=T.PRESET)
    from sisteer.capeval.tiers import judged
    assert RR.evidence_label(ctl) == (judged.TASK_XSTEST, "xst0001-bbbbbbbb", 0)         # the judged control's OWN task; one draw per item
    mspec = L.QueueSpec.build(T.W7, T.WAVE, "misalign", "s1", "misalign", 16384, ["0"])
    mrow = {"family": "agentic", "behavior": "leaking", "item_id": "sc-7", "sample_idx": 1, "cell_id": "none", "virtual_model": "petri__none",
            "prompt_sha": "p" * 16, "response": "r", "finish_reason": "length", "seed": 9}
    mis = L.misalign_queue_record(mspec, arm="none", s=1, row=mrow, messages=[{"role": "user", "content": "hi"}], base_store="b",
                                  producer=mspec.producers[0], producer_attempt_id="a1", sampling=T.PRESET)
    assert RR.evidence_label(mis) == ("agentic/leaking", "sc-7", 1)
    gspec = L.QueueSpec.build(T.W7, T.WAVE, "gsm8k", "s1", "gsm8k", 16384, ["0"])
    grec = L.gsm8k_queue_record(gspec, arm="none", s=1, item={"item_id": "q1", "sample_idx": 0, "at_cap": True, "stop_reason": "max_tokens"},
                                base_store="b", producer=gspec.producers[0], producer_attempt_id="a1")
    with pytest.raises(L.LadderRefusal, match="no rung request to label"):
        RR.evidence_label(grec)
    # the env contract: ENV_DIR alone is enough; a pinned seed base / arm that disagrees with a record refuses, never mis-stamps
    assert RR.RungEvidence.from_env(leg="pasttense", attempt_id="r1", env={}) is None
    e = RR.RungEvidence.from_env(leg="pasttense", attempt_id="r1", env={REC.ENV_DIR: "/tmp/x", REC.ENV_SEED_BASE: "2"})
    with pytest.raises(L.LadderRefusal, match="pins seed base 2 but this record is s=1"):
        e.client_for(static)
    e = RR.RungEvidence.from_env(leg="pasttense", attempt_id="r1", env={REC.ENV_DIR: "/tmp/x", REC.ENV_EXTRA: json.dumps({"arm": "lora_x"})})
    with pytest.raises(L.LadderRefusal, match="pins arm 'lora_x'"):
        e.client_for(static)
    with pytest.raises(L.LadderRefusal, match="names attempt_id='other'"):
        RR.RungEvidence.from_env(leg="pasttense", attempt_id="r1", env={REC.ENV_DIR: "/tmp/x", REC.ENV_EXTRA: json.dumps({"attempt_id": "other"})})
    with pytest.raises(L.LadderRefusal, match="given twice and disagreeing"):
        RR.RungEvidence.from_env(leg="pasttense", attempt_id="r1", env={REC.ENV_DIR: "/tmp/x", REC.ENV_EXTRA: json.dumps({"s": 1}), REC.ENV_SEED_BASE: "2"})


def test_sol_r2_p1_2_a_proxy_refusal_and_a_dead_transport_are_recorded_as_failed_attempts_then_raised(tmp_path):
    REC = T.load_request_evidence_client()
    spec = T.static_spec(L, shards=("0",))
    rec = L.static_queue_record(spec, arm="none", s=1, family="pasttense", row=T.static_row("g0", 1, seed=7), base_store="b",
                                producer=spec.producers[0], producer_attempt_id="a1", vmodel="petri__none", sampling=T.PRESET)
    ev = RR.RungEvidence(tmp_path / "root", leg="pasttense", attempt_id="rung-1")

    class Refusal(Exception):
        code, http_status = "si_prefix_missing", 400

    proxy = T.StubProxy(verify=lambda body: (_ for _ in ()).throw(Refusal("no prefix")))
    try:
        with pytest.raises(L.LadderRefusal, match="proxy returned 400"):
            RR.http_draw_fn(proxy.url, evidence=ev)(rec)
    finally:
        proxy.close()
    with pytest.raises(L.LadderRefusal, match="died before a response"):
        RR.http_draw_fn("http://127.0.0.1:1/v1", evidence=ev)(rec)
    reqs = [json.loads(l) for l in (tmp_path / "root" / REC.RESPONSE_EVIDENCE_FILENAME).read_text().splitlines()]
    assert [r["status"] for r in reqs] == [400, None] and all(r["completion_id"] is None for r in reqs)
    assert "si_prefix_missing" in reqs[0]["error"] and reqs[1]["error"].startswith("URLError")
    recs = [json.loads(l) for l in (tmp_path / "root" / "request_evidence.jsonl").read_text().splitlines()]
    assert [r["request_id"] for r in recs] == [r["request_id"] for r in reqs] and len({r["request_id"] for r in recs}) == 2   # a retry is a NEW attempt
    assert ev.summary()["n_requests"] == 2 and ev.summary()["n_responses"] == 2


def test_sol_r2_p1_2_the_production_cli_refuses_to_start_without_the_evidence_environment(monkeypatch, capsys):
    REC = T.load_request_evidence_client()
    assert REC.ENV_DIR == RR.EVIDENCE_ENV_DIR == "SISTEER_REQUEST_EVIDENCE_DIR"      # the one spelling, drift-checked by _evidence_module
    monkeypatch.delenv(REC.ENV_DIR, raising=False)
    mod = types.ModuleType("modal")                                                # the Volume argument checks pass (a fake mount handle)...
    mod.Volume = types.SimpleNamespace(from_name=lambda name: types.SimpleNamespace(commit=lambda: None, reload=lambda: None))
    monkeypatch.setitem(sys.modules, "modal", mod)
    spec = T.static_spec(L, shards=("0",))
    axes = ["--W7", spec.W7, "--wave-uuid", spec.wave_uuid, "--leg", spec.leg, "--view", spec.view, "--kind", spec.kind, "--cap-next", str(spec.cap_next),
            "--producers", "0"]
    argv = ["consume", "--vol-root", "/vol", "--volume", "twosi-fits", *axes, "--attempt-id", "r1", "--call-id", "fc", "--proxy-url", "http://127.0.0.1:1/v1",
            "--width", "3"]
    with pytest.raises(SystemExit) as ei:                 # ... KNOWN-BAD: a production rung without SISTEER_REQUEST_EVIDENCE_DIR
        RR.main(argv)
    assert ei.value.code == 2 and "SISTEER_REQUEST_EVIDENCE_DIR is not set" in capsys.readouterr().err
    # the Volume argument refusal still comes FIRST (the P1-1 parser pins): no --volume and no smoke dir names --volume, not the env
    with pytest.raises(SystemExit):
        RR.main([a_ for a_ in argv if a_ not in ("--volume", "twosi-fits")])
    assert "needs --volume" in capsys.readouterr().err
    # an unset environment is decided without the client: a smoke (--memory-kv-dir) on a tree without the client's module never imports it
    assert RR.RungEvidence.from_env(leg="pasttense", attempt_id="r1", env={}) is None
    assert RR.build_parser().parse_args([*argv, "--no-request-evidence"]).no_request_evidence is True   # the recorded opt-out parses
    assert RR.build_parser().parse_args([*argv, "--memory-kv-dir", "/tmp/kv"]).memory_kv_dir == "/tmp/kv"   # a smoke needs neither
