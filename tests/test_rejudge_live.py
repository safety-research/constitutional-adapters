"""The live slot of a re-judge (scripts/rejudge/common/live.py, scripts/rejudge/strongreject/live.py) and the StrongREJECT filing that
merges it with the batch run (scripts/rejudge/strongreject/file_records.py): the key set is exactly the source run's never-submitted
keys, the body sent is the batch manifest's byte for byte (plus the model and the routing), the batch track's parser and re-ask policy
decide every outcome, a restart never re-sends a terminal key, 429s halve the concurrency without spending an attempt, the spend
ceiling and the global check stop the run before they are passed, a flood is held, and the filing combines the full run, a reuse
run directory and the live run (a duplicate refused unless it is a handed-over key, where the first fileable outcome stands). Offline: synthetic
transcripts, a local mock of OpenRouter's chat completions (aiohttp on 127.0.0.1), no registry."""
from __future__ import annotations

import asyncio
import collections
import gzip
import hashlib
import json
import os
import pathlib
import re
import sqlite3
import sys
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from aiohttp import web  # noqa: E402

from rejudge.common import batchfile as BF  # noqa: E402
from rejudge.common import live as LV  # noqa: E402
from rejudge.common.harvester import Harvester  # noqa: E402
from rejudge.common.state import Ledger, readiness  # noqa: E402
from rejudge.strongreject import file_records as FR  # noqa: E402
from rejudge.strongreject import judges as J  # noqa: E402
from rejudge.strongreject import live as SRL  # noqa: E402
from rejudge.strongreject import request as R  # noqa: E402

MM = J.get("minimax-m3")
TEST_KEY = "test-key-not-real"
RUN = "sr:synthetic-run-1"
FAST = {"concurrency": 4, "min_concurrency": 1, "flush_n": 50, "flush_s": 0.05, "control_s": 0.05, "progress_s": 0.1,
        "global_check_s": 3600.0, "transport_retries": 1, "transport_backoff_s": 0.001, "transport_backoff_max_s": 0.01,
        "rate_limit_min_s": 0.0, "rate_limit_backoff_s": 0.001, "rate_limit_backoff_max_s": 0.01, "rl_window_s": 30.0, "rl_halve_after": 3,
        "rl_halve_spacing_s": 0.0, "rl_recover_s": 3600.0, "request_timeout_s": 20.0, "cooldown_s": 0.05, "cooldown_max_s": 0.2,
        "price_tolerance": 0}                     # the mock's flat cost is not a list price; the price check has its own tests
ARGS = {"reasoning": "synthetic", "refused": 0, "convincing": 4, "specific": 5}
#: the synthetic global ceiling every test live directory states (init has no default)
CEILING = 1000.0


def key(i: int) -> str:
    return hashlib.sha256(f"synthetic-key-{i}".encode()).hexdigest()


def body(i: int, behavior: str, max_tokens: int = None) -> dict:
    return R.request_body(MM["request"], "sr", f"Synthetic goal {i % 3}", f"MOCK:{behavior} #{i} synthetic response text", max_tokens=max_tokens)


def zero_walker(root, exclude=()):
    return {"usd": 0.0, "per_dir": {}, "errors": []}


# ---------------------------------------------------------------------------------------------------------------------------------------
# a synthetic source run directory shaped like the full run: harvested, in-flight, a 413-split parent with pending children, a pending
# re-ask batch, a plain pending batch
# ---------------------------------------------------------------------------------------------------------------------------------------
BEHAVIORS = {6: "verdict", 7: "text_then_verdict", 8: "cut_then_verdict", 9: "refusal", 10: "text_always", 11: "cut_always",
             12: "filter", 13: "mod403", 14: "e500_then_verdict", 15: "text_always"}


def _track(rd, reuse=None):
    spec = {"track": MM["track"], "model": MM["model"], "endpoint": MM["endpoint"], "provider": None,
            "parser": "rejudge.strongreject.request:parse_result", "retry": MM["retry"], "request": MM["request"],
            "truncated_retry_max_tokens": MM["truncated_retry_max_tokens"], "health": dict(MM["health"], pause_on_batch_end=True),
            "judge": "minimax-m3", "max_spend_usd": 100.0, "tag": "synthetic_full",
            "submit_gate": {"requires_files": [str(pathlib.Path(rd).parent / "GO")]}}
    if reuse:
        spec["reuse_run_dirs"] = [reuse]
    BF.write_track(str(rd), spec)


def _manifest(rd, bk, items, attempt=0, parent=None):
    hdr = BF.batch_header(MM["model"], MM["endpoint"])
    reqs = [BF.PackedRequest(cid, BF.serialize_request(cid, b), 1000.0, 300.0) for cid, b in items]
    return BF.write_manifest(str(rd), bk, hdr, reqs, MM["track"], attempt=attempt, parent=parent)


def make_source(tmp: pathlib.Path, reuse=None, behaviors=None) -> str:
    beh = dict(BEHAVIORS)
    beh.update(behaviors or {})
    rd = tmp / "full"
    _track(rd, reuse)
    led = Ledger(str(rd)).writer()
    m0 = _manifest(rd, "b0", [(key(i), body(i, "verdict")) for i in range(4)])
    led.register_manifest(m0)
    led.update_batch("b0", state="harvested", batch_id="batch-0", cost=0.01)
    v = {"refused": 0, "convincing": 4, "specific": 5, "score": 0.875}
    led.record_outcome(key(0), "b0", "batch-0", "verdict", True, v, None, 200, "tool_calls", None, None, "DeepInfra")
    led.record_outcome(key(1), "b0", "batch-0", "verdict", True, v, None, 200, "tool_calls", None, None, "DeepInfra")
    led.record_outcome(key(2), "b0", "batch-0", "parse_error", False, None, {"kind": "no_tool_call"}, 200, "stop", None, None, "DeepInfra")
    led.record_outcome(key(3), "b0", "batch-0", "truncated", False, None, {"kind": "truncated"}, 200, "length", None, None, "DeepInfra")
    m1 = _manifest(rd, "b1", [(key(i), body(i, "verdict")) for i in (4, 5)])
    led.register_manifest(m1)
    led.update_batch("b1", state="submitted", batch_id="batch-1", est_cost=0.002)
    mp = _manifest(rd, "b2", [(key(i), body(i, beh[i])) for i in (6, 7, 8, 9)])
    led.register_manifest(mp)
    led.update_batch("b2", state="split")
    for bk, ks in (("b2.s0", (6, 7)), ("b2.s1", (8, 9))):
        led.register_manifest(_manifest(rd, bk, [(key(i), body(i, beh[i])) for i in ks], parent="b2"))
    m3 = _manifest(rd, "b3", [(key(i), body(i, beh[i])) for i in (10, 11)])
    led.register_manifest(m3)
    led.update_batch("b3", state="harvested", batch_id="batch-3", cost=0.01)
    led.record_outcome(key(10), "b3", "batch-3", "parse_error", False, None, {"kind": "no_tool_call"}, 200, "stop", None, None, "DeepInfra")
    led.record_outcome(key(11), "b3", "batch-3", "truncated", False, None, {"kind": "truncated"}, 200, "length", None, None, "DeepInfra")
    # the re-ask batch the harvester built (attempt 1): the cut reply already at the raised cap
    led.register_manifest(_manifest(rd, "retry1_20261111T000000Z_000", [(key(10), body(10, beh[10])), (key(11), body(11, beh[11], 4000))],
                                    attempt=1))
    led.register_manifest(_manifest(rd, "b4", [(key(i), body(i, beh[i])) for i in (12, 13, 14, 15)]))
    led.close()
    (rd / "PAUSED").write_text("2026-11-11T17:54:23Z submit gate: the go-ahead file is missing\n")
    return str(rd)


def init_live(tmp, src, max_spend=50.0, **live):
    d = str(pathlib.Path(tmp) / "live")
    LV.init_live(d, src, SRL.routing(), SRL.PROVIDERS, max_spend, live=dict(dict(FAST, global_root=str(tmp), global_ceiling_usd=CEILING), **live),
                 log=lambda m: None)
    return d


# ---------------------------------------------------------------------------------------------------------------------------------------
# a mock of OpenRouter's chat completions
# ---------------------------------------------------------------------------------------------------------------------------------------
def completion(msg, finish, provider="GMICloud", cost=0.0006, reasoning_tokens=0, model="minimax/minimax-m3"):
    return {"id": f"gen-{time.time_ns()}", "model": model, "provider": provider, "object": "chat.completion",
            "choices": [{"index": 0, "message": msg, "finish_reason": finish, "native_finish_reason": finish}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 300, "total_tokens": 1300, "cost": cost,
                      "prompt_tokens_details": {"cached_tokens": 640, "cache_write_tokens": 0},
                      "completion_tokens_details": {"reasoning_tokens": reasoning_tokens}}}


TOOL = {"role": "assistant", "content": None, "refusal": None,
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "StrongREJECTRating", "arguments": json.dumps(ARGS)}}]}
TEXT = {"role": "assistant", "content": "I would rate this as follows", "refusal": None, "tool_calls": None}
CUT = {"role": "assistant", "content": "partial reasoning", "refusal": None, "tool_calls": None}


class Mock:
    def __init__(self, cost=0.0006, provider="GMICloud", global_429=0, global_500=0, latency=0.0, reasoning_tokens=0, status_override=None,
                 model="minimax/minimax-m3", reasoning_first_call=False, retry_after=None, on_request=None, provider_fn=None):
        self.cost, self.provider, self.global_429, self.global_500 = cost, provider, global_429, global_500
        self.latency, self.reasoning_tokens, self.status_override = latency, reasoning_tokens, status_override
        self.model, self.reasoning_first_call, self.retry_after, self.on_request = model, reasoning_first_call, retry_after, on_request
        self.provider_fn = provider_fn
        self.calls = collections.Counter()
        self.total = 0
        self.payloads = []
        self.active = collections.Counter()
        self.max_active = collections.Counter()
        self.sent = {}
        self.auth = set()

    def respond(self, beh, n, payload):
        if self.status_override:
            return self.status_override
        ok = completion(TOOL, "tool_calls", self.provider, self.cost)
        if beh == "verdict":
            return 200, ok
        if beh == "text_then_verdict":
            return (200, completion(TEXT, "stop", self.provider, self.cost)) if n == 1 else (200, ok)
        if beh == "text_always":
            return 200, completion(TEXT, "stop", self.provider, self.cost)
        if beh == "cut_then_verdict":                                            # cut at the base cap, answered at the raised one
            return (200, ok) if payload["max_tokens"] == 4000 else (200, completion(CUT, "length", self.provider, self.cost))
        if beh == "cut_always":
            return 200, completion(CUT, "length", self.provider, self.cost)
        if beh == "refusal":
            return 200, completion({"role": "assistant", "content": None, "refusal": "I can't help", "tool_calls": None}, "content_filter",
                                   self.provider, self.cost)
        if beh == "filter":
            return 200, completion({"role": "assistant", "content": None, "refusal": None, "tool_calls": None}, "content_filter", self.provider,
                                   self.cost)
        if beh == "mod403":
            return 403, {"error": {"code": 403, "message": "Your input was flagged for moderation",
                                   "metadata": {"reasons": ["harassment"], "flagged_input": "x" * 20}}}
        if beh == "e500_then_verdict":
            return (500, {"error": {"code": 500, "message": "Internal Server Error"}}) if n <= 2 else (200, ok)
        if beh == "auth401":
            return 401, {"error": {"code": 401, "message": "No auth credentials found"}}
        if beh == "bad400":
            return 400, {"error": {"code": 400, "message": "Provider returned error"}}
        raise AssertionError(beh)

    async def handle(self, request):
        payload = await request.json()
        self.auth.add(request.headers.get("Authorization"))
        m = re.search(r"MOCK:(\w+) #(\d+)", payload["messages"][0]["content"])
        beh, k = m.group(1), int(m.group(2))
        self.total += 1
        self.calls[k] += 1
        n = self.calls[k]
        if self.on_request is not None:
            self.on_request(self, k, n)
        if self.provider_fn is not None:
            self.provider = self.provider_fn(k, n, payload)                      # set and used with no await between
        self.active[k] += 1
        self.max_active[k] = max(self.max_active[k], self.active[k])
        try:
            if self.latency:
                await asyncio.sleep(self.latency)
            if self.total <= self.global_429:
                status, out = 429, {"error": {"code": 429, "message": "Rate limit exceeded"}}
            elif self.total <= self.global_429 + self.global_500:
                status, out = 500, {"error": {"code": 500, "message": "Internal Server Error"}}
            else:
                status, out = self.respond(beh, n, payload)
                if status == 200 and "choices" in out:                           # the serving facts of every completion
                    rt = self.reasoning_tokens if not self.reasoning_first_call else (400 if n == 1 else 0)
                    out = dict(out, model=self.model, usage=dict(out["usage"], completion_tokens_details={"reasoning_tokens": rt}))
        finally:
            self.active[k] -= 1
        self.payloads.append(payload)
        self.sent[(k, n)] = (status, out)
        hdrs = {"Retry-After": str(self.retry_after)} if status == 429 and self.retry_after is not None else None
        return web.json_response(out, status=status, headers=hdrs)


def run_engine(live_dir, mock, limit=None, walker=None, max_spend=None, **ov):
    async def go():
        app = web.Application()
        app.router.add_post("/api/v1/chat/completions", mock.handle)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            r = LV.LiveRunner(live_dir, overrides=dict(FAST, base_url=f"http://127.0.0.1:{port}/api/v1", **ov), limit=limit,
                              log=lambda m: None, spend_walker=walker or zero_walker, max_spend_usd=max_spend)
            return await r.run_async(), r
        finally:
            await runner.cleanup()
    return asyncio.run(go())


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", TEST_KEY)


def ledger_rows(d):
    con = sqlite3.connect(os.path.join(d, "state.sqlite"))
    con.row_factory = sqlite3.Row
    try:
        req = {r["custom_id"]: dict(r) for r in con.execute("SELECT * FROM requests")}
        att = collections.defaultdict(list)
        for r in con.execute("SELECT * FROM live_attempts ORDER BY custom_id, attempt"):
            att[r["custom_id"]].append(dict(r))
        return req, att
    finally:
        con.close()


# ---------------------------------------------------------------------------------------------------------------------------------------
# key selection and body identity
# ---------------------------------------------------------------------------------------------------------------------------------------
def test_init_takes_only_never_submitted_keys_with_their_prior_outcomes(tmp_path):
    src = make_source(tmp_path)
    d = init_live(tmp_path, src)
    con = sqlite3.connect(os.path.join(d, "state.sqlite"))
    rows = {r[0]: (r[1], r[2], json.loads(r[3])) for r in con.execute("SELECT custom_id, source_batch_key, origin, prior_outcomes_json FROM live_keys")}
    con.close()
    assert set(rows) == {key(i) for i in range(6, 16)}          # pending children, the pending re-ask batch, the plain pending batch
    assert not {key(i) for i in range(6)} & set(rows)            # never a harvested or in-flight key; the split parent holds none
    assert rows[key(10)] == ("retry1_20261111T000000Z_000", "pending", ["parse_error"])
    assert rows[key(11)][2] == ["truncated"] and rows[key(6)][2] == []
    tr = BF.read_track(d)
    assert tr["backend"] == "live" and tr["source_run_dir"] == src and tr["model"] == "minimax/minimax-m3"
    assert tr["request"] == MM["request"] and tr["retry"] == MM["retry"] and tr["parser"] == "rejudge.strongreject.request:parse_result"
    assert tr["live"]["routing"]["quantizations"] == ["fp8"] and "parasail/fast" in tr["live"]["routing"]["ignore"]
    assert "coreweave" in tr["live"]["routing"]["ignore"] and all(p["quantization"] == "fp8" for p in tr["live"]["providers"].values())
    assert oct(os.stat(d).st_mode & 0o777) == "0o700"
    r = readiness(d)
    assert r["requests"] == 10 and r["terminal"] == 0
    with pytest.raises(SystemExit, match="already holds"):
        init_live(tmp_path, src)


def test_init_refuses_a_claude_model_and_a_tampered_body(tmp_path):
    src = make_source(tmp_path)
    with pytest.raises(SystemExit, match="Claude"):
        LV.init_live(str(tmp_path / "x"), src, dict(SRL.routing(), only=["anthropic/claude"]), SRL.PROVIDERS, 10.0, log=lambda m: None)
    p = pathlib.Path(src) / "bodies" / "b4.json.gz"
    raw = gzip.decompress(p.read_bytes()).replace(b"Synthetic goal", b"Tampered! goal")
    p.write_bytes(gzip.compress(raw))
    with pytest.raises(ValueError, match="does not hash"):
        LV.init_live(str(tmp_path / "y"), src, SRL.routing(), SRL.PROVIDERS, 10.0, live={"global_root": str(tmp_path), "global_ceiling_usd": CEILING},
                     log=lambda m: None)
    # no global root or ceiling: refused before anything is read (neither has a default)
    for live in ({}, {"global_root": str(tmp_path)}, {"global_ceiling_usd": CEILING}, {"global_root": "relative", "global_ceiling_usd": CEILING}):
        with pytest.raises(SystemExit, match="REFUSED"):
            LV.init_live(str(tmp_path / "z"), src, SRL.routing(), SRL.PROVIDERS, 10.0, live=live, log=lambda m: None)
    assert not os.path.exists(tmp_path / "z")


def test_the_post_body_is_the_manifest_body_plus_model_and_routing(tmp_path):
    src = make_source(tmp_path)
    d = init_live(tmp_path, src)
    mock = Mock()
    code, _ = run_engine(d, mock)
    assert code == 0 and mock.auth == {f"Bearer {TEST_KEY}"}
    manifest_bodies = {}
    for bk in BF.list_manifests(src):
        man = BF.read_manifest(src, bk)
        for r in BF.body_requests(BF.read_body(src, man)):
            manifest_bodies.setdefault(r["custom_id"], []).append(r["body"])
    _, att = ledger_rows(d)
    by_content = {b["messages"][0]["content"]: cid for cid, bs in manifest_bodies.items() for b in bs}
    for p in mock.payloads:
        assert p["model"] == "minimax/minimax-m3" and p["provider"] == SRL.routing()
        sent = {k: v for k, v in p.items() if k not in ("model", "provider")}
        assert list(p)[0] == "model" and list(p)[-1] == "provider"
        cid = by_content[sent["messages"][0]["content"]]
        if sent["max_tokens"] == 4000 and cid in (key(8),):
            assert sent == dict(manifest_bodies[cid][0], max_tokens=4000)          # the raised-cap re-ask of a cut reply
        else:
            assert sent in manifest_bodies[cid]                                     # byte for byte the batch request body
            assert LV.compact(sent) in [LV.compact(b) for b in manifest_bodies[cid]]
    con = sqlite3.connect(os.path.join(d, "state.sqlite"))
    shas = dict(con.execute("SELECT custom_id, body_sha256 FROM live_keys").fetchall())
    con.close()
    assert att[key(6)][0]["body_sha256"] == shas[key(6)] == LV.body_sha256(manifest_bodies[key(6)][0])


# ---------------------------------------------------------------------------------------------------------------------------------------
# outcomes and re-asks: the batch track's parser and policy
# ---------------------------------------------------------------------------------------------------------------------------------------
def test_outcomes_and_reasks_follow_the_batch_tracks_parser_and_policy(tmp_path):
    src = make_source(tmp_path)
    d = init_live(tmp_path, src)
    mock = Mock()
    code, runner = run_engine(d, mock)
    assert code == 0
    req, att = ledger_rows(d)
    want = {6: ("verdict", 1), 7: ("verdict", 2), 8: ("verdict", 2), 9: ("judge_refusal", 1), 10: ("parse_error", 2),
            11: ("truncated", 1), 12: ("moderation", 1), 13: ("moderation", 1), 14: ("verdict", 2), 15: ("parse_error", 3)}
    for i, (outcome, n) in want.items():
        r = req[key(i)]
        assert (r["outcome"], r["terminal"], len(att[key(i)])) == (outcome, 1, n), (i, r["outcome"], len(att[key(i)]))
    assert [a["outcome"] for a in att[key(7)]] == ["parse_error", "verdict"]
    assert [a["max_tokens"] for a in att[key(8)]] == [1500, 4000]                     # the cut reply re-asked at the raised cap
    assert att[key(11)][0]["max_tokens"] == 4000                                       # the pending re-ask batch's body as it was
    assert [a["outcome"] for a in att[key(14)]] == ["provider_error", "verdict"] and mock.calls[14] == 3   # 1 + 1 transport retry, then the re-ask
    assert att[key(14)][0]["transport_retries"] == 1
    # parity: each recorded outcome is what the batch parser makes of the same reply wrapped as a batch result entry
    for (k, n), (status, out) in mock.sent.items():
        if status == 500 and n == 1:
            continue                                                                   # retried in place, not an attempt
        a = att[key(k)][n - 1 if k != 14 else (0 if n == 2 else 1)]
        assert a["outcome"] == R.parse_result({"custom_id": key(k), "response": {"status_code": status, "body": out}})["outcome"]
    v = json.loads(req[key(6)]["verdict_json"])
    assert v["score"] == R.strongreject_score(0, 4, 5) and v["refused"] == 0
    a6 = att[key(6)][0]
    assert (a6["provider"], a6["quantization"], a6["cost_source"], a6["prompt_tokens"], a6["cached_tokens"]) == ("GMICloud", "fp8", "reported", 1000, 640)
    assert a6["cost_usd"] == pytest.approx(0.0006) and a6["reasoning_tokens"] == 0 and a6["generation_id"].startswith("gen-")
    prog = json.load(open(os.path.join(d, "progress.json")))
    assert prog["state"] == "done" and prog["keys"] == {"total": 10, "terminal": 10, "remaining": 0, "sent": 10}
    assert prog["provider_mix"]["GMICloud"]["quantization"] == "fp8"
    n_200 = sum(1 for v in att.values() for a in v if a["http_status"] == 200)
    assert n_200 == 14 and runner.spent_with_fee() == pytest.approx(14 * 0.0006 * 1.055, rel=1e-9)
    assert prog["spend_usd"]["spent_with_fee"] == pytest.approx(14 * 0.0006 * 1.055, abs=1e-4)
    assert readiness(d)["ready"]


@pytest.mark.parametrize("seq", [["parse_error", "parse_error", "parse_error"], ["truncated", "parse_error"], ["provider_error"] * 4,
                                 ["held", "parse_error", "balance_error", "parse_error", "verdict"], ["moderation"], ["batch_expired", "parse_error"]])
def test_the_terminal_rule_is_the_harvesters(tmp_path, seq):
    rd = str(tmp_path / "h")
    BF.write_track(rd, {"track": "t", "model": "minimax/minimax-m3", "endpoint": "/v1/chat/completions", "parser": "rejudge.strongreject.request:parse_result",
                        "retry": MM["retry"]})
    man = BF.write_manifest(rd, "b", BF.batch_header("minimax/minimax-m3", "/v1/chat/completions"), [BF.PackedRequest("k", BF.serialize_request("k", {}))], "t")
    h = Harvester(rd, None, log=lambda m: None, dry_run=True)
    h.ledger.register_manifest(man)
    prior = []
    for o in seq:
        h._record("k", "b", "id", o, {"outcome": o}, None, "minimax/minimax-m3")
        h.ledger.conn.commit()
        assert bool(h.ledger.get("k")["terminal"]) == LV.is_terminal(o, prior, h.retry), (seq, o)
        prior.append(o)
    h.ledger.close()


# ---------------------------------------------------------------------------------------------------------------------------------------
# resume, one attempt in flight per key
# ---------------------------------------------------------------------------------------------------------------------------------------
def test_a_restart_never_resends_a_terminal_key_and_one_key_is_never_in_flight_twice(tmp_path):
    src = make_source(tmp_path)
    d = init_live(tmp_path, src)
    mock = Mock(latency=0.02)
    assert run_engine(d, mock, concurrency=8)[0] == 0
    assert max(mock.max_active.values()) == 1
    n1 = mock.total
    mock2 = Mock()
    code, _ = run_engine(d, mock2)
    assert code == 0 and mock2.total == 0 and n1 > 10


def test_an_interrupted_run_resumes_where_it_stopped(tmp_path):
    src = make_source(tmp_path)
    d = init_live(tmp_path, src)
    mock = Mock()
    code, _ = run_engine(d, mock, limit=4)                                       # a pilot stops after 4 keys
    req, att = ledger_rows(d)
    sent = {c for c, a in att.items() if a}
    assert code == 0 and len(sent) == 4 and all(req[c]["terminal"] for c in sent)
    mock2 = Mock()
    assert run_engine(d, mock2)[0] == 0
    sent2 = {key(k) for k in mock2.calls}
    assert not sent & sent2 and len(sent | sent2) == 10


def test_the_pilot_limit_spreads_over_batches_and_is_cumulative(tmp_path):
    src = make_source(tmp_path, behaviors={i: "verdict" for i in range(6, 16)})
    d = init_live(tmp_path, src)
    m1 = Mock()
    assert run_engine(d, m1, limit=3)[0] == 0
    con = sqlite3.connect(os.path.join(d, "state.sqlite"))
    batches = {r[0] for r in con.execute("SELECT DISTINCT k.source_batch_key FROM live_keys k JOIN live_attempts a USING (custom_id)")}
    con.close()
    assert m1.total == 3 and len(batches) == 3
    m2 = Mock()
    assert run_engine(d, m2, limit=3)[0] == 0 and m2.total == 0                  # the limit counts keys already sent


# ---------------------------------------------------------------------------------------------------------------------------------------
# rate limits, spend, the global check, health
# ---------------------------------------------------------------------------------------------------------------------------------------
def test_persistent_429s_halve_the_concurrency_and_spend_no_attempt(tmp_path):
    src = make_source(tmp_path, behaviors={i: "verdict" for i in range(6, 16)})
    d = init_live(tmp_path, src)
    mock = Mock(global_429=12)
    code, runner = run_engine(d, mock, concurrency=8)
    req, att = ledger_rows(d)
    assert code == 0 and all(r["outcome"] == "verdict" for r in req.values())
    assert all(len(a) == 1 for a in att.values())                               # a 429 is never an attempt
    assert runner.limiter.halvings >= 1 and runner.limiter.limit < 8 and runner.limiter.n_rate_limited[429] == 12
    assert sum(a[0]["rate_limited"] for a in att.values()) == 12
    con = sqlite3.connect(os.path.join(d, "state.sqlite"))
    assert con.execute("SELECT COUNT(*) FROM events WHERE kind = 'CONCURRENCY'").fetchone()[0] >= 1
    con.close()


def test_the_spend_ceiling_stops_before_it_is_passed(tmp_path):
    src = make_source(tmp_path, behaviors={i: "verdict" for i in range(6, 16)})
    d = init_live(tmp_path, src)
    mock = Mock(cost=1.0)
    code, runner = run_engine(d, mock, concurrency=1, max_spend=3.0)
    assert code == 4 and mock.total == 2                                         # 2 x $1.055; a third would pass $3
    paused = open(os.path.join(d, "PAUSED")).read()
    assert "live spend ceiling" in paused and "$3.00" in paused and "$2.11" in paused
    assert runner.spent_with_fee() <= 3.0
    # resuming without raising the ceiling sends nothing
    os.remove(os.path.join(d, "PAUSED"))
    m2 = Mock(cost=1.0)
    assert run_engine(d, m2, concurrency=1, max_spend=3.0)[0] == 4 and m2.total == 0


def test_the_global_check_pauses_at_start_and_mid_run(tmp_path):
    src = make_source(tmp_path, behaviors={i: "verdict" for i in range(6, 16)})
    d = init_live(tmp_path, src)
    m = Mock()
    code, _ = run_engine(d, m, walker=lambda root, exclude=(): {"usd": CEILING, "per_dir": {"other": CEILING}, "errors": []})
    assert code == 4 and m.total == 0 and "GLOBAL ceiling" in open(os.path.join(d, "PAUSED")).read()
    os.remove(os.path.join(d, "PAUSED"))
    calls = {"n": 0}

    def rising(root, exclude=()):
        calls["n"] += 1
        assert os.path.realpath(exclude[0]) == os.path.realpath(d)               # the live run's own spend is counted separately
        return {"usd": 100.0 if calls["n"] == 1 else CEILING + 10.0, "per_dir": {}, "errors": []}

    m2 = Mock(latency=0.05)
    code, _ = run_engine(d, m2, walker=rising, global_check_s=0.1, concurrency=1)
    assert code == 4 and 0 < m2.total < 10 and calls["n"] >= 2
    assert "GLOBAL ceiling" in open(os.path.join(d, "PAUSED")).read()


def test_an_account_error_pauses_and_makes_nothing_terminal(tmp_path):
    src = make_source(tmp_path, behaviors={i: "auth401" for i in range(6, 16)})
    d = init_live(tmp_path, src)
    code, _ = run_engine(d, Mock(), concurrency=1, flush_n=1)
    req, att = ledger_rows(d)
    assert code == 4 and "account / credential" in open(os.path.join(d, "PAUSED")).read()
    assert not any(r["terminal"] for r in req.values())


def test_an_identical_error_flood_is_held_and_pauses(tmp_path):
    beh = {i: "bad400" for i in range(6, 16)}
    src = make_source(tmp_path, behaviors=beh)
    d = init_live(tmp_path, src)
    code, _ = run_engine(d, Mock(), concurrency=10, flush_n=10, flush_s=2.0)
    req, att = ledger_rows(d)
    assert code == 4 and "identical-error flood" in open(os.path.join(d, "PAUSED")).read()
    assert all(r["outcome"] == "held" and not r["terminal"] for r in req.values())


def test_a_transient_flood_cools_down_and_resends_the_held_keys(tmp_path):
    src = make_source(tmp_path, behaviors={i: "verdict" for i in range(6, 16)})
    d = init_live(tmp_path, src)
    mock = Mock(global_500=20)                                                   # every key's first two calls fail
    code, _ = run_engine(d, mock, concurrency=10, flush_n=10, flush_s=2.0)
    req, att = ledger_rows(d)
    assert code == 0 and all(r["outcome"] == "verdict" for r in req.values())
    assert all([a["outcome"] for a in att[k]] == ["held", "verdict"] for k in att)    # held is uncounted, then re-sent
    con = sqlite3.connect(os.path.join(d, "state.sqlite"))
    assert con.execute("SELECT COUNT(*) FROM events WHERE kind = 'COOLDOWN'").fetchone()[0] == 1
    con.close()


# ---------------------------------------------------------------------------------------------------------------------------------------
# the protocol of a served reply: off-protocol replies are held (never terminal, never filed), the pause can be cleared, reopen
# ---------------------------------------------------------------------------------------------------------------------------------------
def edit_live(d, fn):
    tr = BF.read_track(d)
    fn(tr["live"])
    BF.write_track(d, tr)


@pytest.mark.parametrize("provider,model,kind", [("Venice", "minimax/minimax-m3", "off_map_provider"),
                                                  ("Minimax", "minimax/minimax-m3", "off_map_provider"),
                                                  ("GMICloud", "minimax/minimax-m2.5", "wrong_model"),
                                                  ("GMICloud", "minimax/minimax-m3.5", "wrong_model"),        # a prefix is another model
                                                  ("GMICloud", None, "wrong_model"),
                                                  (None, "minimax/minimax-m3", "unknown_provider")])
def test_an_off_protocol_reply_is_held_not_terminal_and_resent_after_the_fix(tmp_path, provider, model, kind):
    src = make_source(tmp_path, behaviors={i: "verdict" for i in range(6, 16)})
    d = init_live(tmp_path, src)
    code, runner = run_engine(d, Mock(provider=provider, model=model), concurrency=4, flush_n=2)
    req, att = ledger_rows(d)
    assert code == 4 and "off-protocol replies" in open(os.path.join(d, "PAUSED")).read()
    sent = [a for v in att.values() for a in v]
    assert sent and all(a["outcome"] == "held" and not a["terminal"] and a["violation"] == kind for a in sent)
    assert not any(r["terminal"] for r in req.values()) and runner.violations[kind] == len(sent)
    e = json.loads(req[next(iter(att))]["error_json"])
    assert e["kind"] == "off_protocol" and e["violation"] == kind and e["parsed_outcome"] == "verdict" and e["parsed_verdict"]["convincing"] == 4
    assert readiness(d)["terminal"] == 0
    # the routing is fixed (here: the mock serves from an allowed provider again), PAUSED removed: the held keys are judged once each
    os.remove(os.path.join(d, "PAUSED"))
    m2 = Mock()
    assert run_engine(d, m2)[0] == 0
    req, att = ledger_rows(d)
    assert all(r["outcome"] == "verdict" and r["terminal"] for r in req.values())
    for cid, v in att.items():
        assert [a["outcome"] for a in v[:-1]] == ["held"] * (len(v) - 1) and v[-1]["provider"] == "GMICloud" and v[-1]["violation"] is None
    assert all(c == 1 for c in m2.calls.values())


def test_a_held_raised_cap_reask_goes_out_again_at_the_raised_cap_after_a_restart(tmp_path):
    src = make_source(tmp_path, behaviors={i: "verdict" for i in range(6, 16) if i != 8})        # key 8: cut at 1500, answered at 4000
    d = init_live(tmp_path, src)
    m = Mock(provider_fn=lambda k, n, p: "Venice" if (k == 8 and p["max_tokens"] == 4000) else "GMICloud")
    assert run_engine(d, m, concurrency=1, flush_n=1)[0] == 4
    _, att = ledger_rows(d)
    assert [(a["outcome"], a["max_tokens"]) for a in att[key(8)]] == [("truncated", 1500), ("held", 4000)]
    os.remove(os.path.join(d, "PAUSED"))
    assert run_engine(d, Mock())[0] == 0
    req, att = ledger_rows(d)
    # the held attempt is re-sent as it went out (at 4000), so the cut reply's one re-ask is still answered: a verdict, not a second cut
    assert [(a["outcome"], a["max_tokens"]) for a in att[key(8)]] == [("truncated", 1500), ("held", 4000), ("verdict", 4000)]
    assert req[key(8)]["outcome"] == "verdict"


def test_a_reasoning_provider_pauses_and_dropping_it_from_the_routing_clears_the_guard(tmp_path):
    src = make_source(tmp_path, behaviors={i: "verdict" for i in range(6, 16)})
    d = init_live(tmp_path, src, reasoning_min_replies=3, max_reasoning_resends=1)
    code, _ = run_engine(d, Mock(provider="StreamLake", reasoning_tokens=400), concurrency=1, flush_n=1)
    req, att = ledger_rows(d)
    paused = open(os.path.join(d, "PAUSED")).read()
    assert code == 4 and "StreamLake billed reasoning tokens" in paused and "streamlake/fp8" in paused
    assert all(a["outcome"] == "held" and a["violation"] == "reasoning_billed" for v in att.values() for a in v)
    assert not any(r["terminal"] for r in req.values())
    # removing PAUSED alone: the guard still sees StreamLake as allowed and pauses at once
    os.remove(os.path.join(d, "PAUSED"))
    m = Mock()
    code, _ = run_engine(d, m, concurrency=1, flush_n=1)
    assert code == 4 and m.total <= 2 and "StreamLake" in open(os.path.join(d, "PAUSED")).read()
    # dropped as the PAUSED text says (its slug in live.routing.ignore): the guard no longer counts it, every key is judged
    os.remove(os.path.join(d, "PAUSED"))
    edit_live(d, lambda lv: lv["routing"]["ignore"].append("streamlake/fp8"))
    assert run_engine(d, Mock())[0] == 0
    req, att = ledger_rows(d)
    assert all(r["outcome"] == "verdict" for r in req.values())
    assert not [a for v in att.values() for a in v if a["terminal"] and (a["reasoning_tokens"] or 0) > 0]


def test_sporadic_reasoning_replies_are_resent_never_filed_and_spend_no_reask(tmp_path):
    src = make_source(tmp_path)
    d = init_live(tmp_path, src)
    mock = Mock(reasoning_first_call=True)                                      # every key's first reply bills reasoning tokens
    code, runner = run_engine(d, mock)
    req, att = ledger_rows(d)
    # every key whose first call was served (all but the 403 moderation and the 500s of key 14) drew reasoning once
    assert code == 0 and runner.violations["reasoning_billed"] == 8
    for cid, v in att.items():
        for a in v:
            if (a["reasoning_tokens"] or 0) > 0:
                assert a["outcome"] == "held" and a["violation"] == "reasoning_billed" and not a["terminal"]
        assert (v[-1]["reasoning_tokens"] or 0) == 0
    want = {6: "verdict", 7: "verdict", 8: "verdict", 9: "judge_refusal", 10: "parse_error", 11: "truncated", 12: "moderation",
            13: "moderation", 14: "verdict", 15: "parse_error"}
    assert {i: req[key(i)]["outcome"] for i in want} == want and all(r["terminal"] for r in req.values())
    # the held reply spent none of the re-asks: key 15 still gets its three parse-error attempts, key 8 its raised-cap re-ask
    assert [a["outcome"] for a in att[key(15)]] == ["held", "parse_error", "parse_error", "parse_error"]
    assert [a["max_tokens"] for a in att[key(8)]] == [1500, 1500, 4000] and req[key(8)]["outcome"] == "verdict"


def test_a_key_that_keeps_drawing_reasoning_is_parked(tmp_path):
    src = make_source(tmp_path, behaviors={i: "verdict" for i in range(6, 16)})
    d = init_live(tmp_path, src, max_reasoning_resends=2)
    code, runner = run_engine(d, Mock(reasoning_tokens=50))                     # reasoning_min_replies 50 is never reached
    req, att = ledger_rows(d)
    assert code == 5 and len(runner.parked) == 10
    assert all(len(v) == 3 and all(a["violation"] == "reasoning_billed" for a in v) for v in att.values())
    assert not any(r["terminal"] for r in req.values())
    assert json.load(open(os.path.join(d, "progress.json")))["parked_keys"] == 10


def test_a_dearer_tier_under_an_allowed_providers_name_is_held_by_the_price_check(tmp_path):
    # Parasail's fp8 endpoint is allowed and its fast tier is not; a reply names "Parasail" either way, so only the price tells them apart
    list_cost = (360 * 0.30 + 640 * 0.06 + 300 * 1.20) / 1e6                   # the mock's usage at parasail/fp8's list price
    src = make_source(tmp_path, behaviors={i: "verdict" for i in range(6, 16)})
    d = init_live(tmp_path, src, price_tolerance=1.2)
    code, _ = run_engine(d, Mock(provider="Parasail", cost=list_cost * 1.5), flush_n=2, price_tolerance=1.2)
    req, att = ledger_rows(d)
    assert code == 4 and "price_above_map" in open(os.path.join(d, "PAUSED")).read()
    assert all(a["violation"] == "price_above_map" and a["outcome"] == "held" for v in att.values() for a in v)
    os.remove(os.path.join(d, "PAUSED"))
    assert run_engine(d, Mock(provider="Parasail", cost=list_cost), price_tolerance=1.2)[0] == 0       # at the list price: fine
    assert run_engine(init_live(tmp_path / "g", make_source(tmp_path / "g", behaviors={i: "verdict" for i in range(6, 16)})),
                      Mock(provider="GMICloud", cost=0.8 * (360 * 0.24 + 640 * 0.048 + 300 * 0.96) / 1e6), price_tolerance=1.2)[0] == 0
    # (a discount, as GMICloud lists: fine) and the default tolerance is on outside the tests
    assert LV.LIVE_DEFAULTS["price_tolerance"] == 1.2


def test_allowed_providers_and_served_models():
    r = SRL.routing()
    assert LV.allowed_providers(SRL.PROVIDERS, r) == {"DeepInfra", "GMICloud", "StreamLake", "Parasail"}
    assert LV.allowed_providers(SRL.PROVIDERS, dict(r, ignore=r["ignore"] + ["streamlake"])) == {"DeepInfra", "GMICloud", "Parasail"}
    assert LV.allowed_providers(SRL.PROVIDERS, dict(r, only=["deepinfra/fp8"])) == {"DeepInfra"}
    assert LV.allowed_providers(SRL.PROVIDERS, dict(r, quantizations=["fp4"])) == set()
    for ok in ("minimax/minimax-m3", "minimax/minimax-m3-20260531"):
        assert LV.served_model_ok("minimax/minimax-m3", ok)
    for bad in ("minimax/minimax-m3.5", "minimax/minimax-m3-fast", "minimax/minimax-m3:batch", "MiniMax-M3", None, ""):
        assert not LV.served_model_ok("minimax/minimax-m3", bad)
    assert LV.served_model_ok("minimax/minimax-m3", "MiniMax-M3", accepted=["MiniMax-M3"])


def test_a_malformed_completion_is_read_without_raising():
    m = LV.reply_meta({"provider": 5, "model": ["x"], "choices": ["not a dict"], "usage": {"prompt_tokens": "many", "cost": True,
                       "prompt_tokens_details": "x", "completion_tokens_details": {"reasoning_tokens": "lots"}, "completion_tokens": 3}})
    assert m["provider"] is None and m["model"] is None and m["served"] and m["reasoning_tokens"] == 0
    assert m["usage"] == {"completion_tokens": 3.0, "prompt_tokens_details": {}, "completion_tokens_details": {}}


def test_the_filing_refuses_an_off_protocol_live_outcome_and_reopen_clears_it(tmp_path):
    arm, src, d, jobs = filing_world(tmp_path)
    LV.build_handover(d, ["in_flight", "reasks"], "fallback", log=lambda m: None)
    assert run_engine(d, Mock())[0] == 0
    assert file_dry(tmp_path, jobs, src, [arm, src, d], "s0")[1]["gate"]["problems"] == []
    # a ledger whose routing changed under it: GMICloud, which served every live reply, is dropped
    edit_live(d, lambda lv: lv["routing"]["ignore"].append("gmicloud"))
    with pytest.raises(SystemExit, match="off_map_provider"):
        file_dry(tmp_path, jobs, src, [arm, src, d], "s1")
    dry = LV.reopen_live(d, off_protocol=True, log=lambda m: None)
    # 14 terminal live keys; key 13's outcome is OpenRouter's 403 moderation, which no provider served
    assert not dry["applied"] and dry["keys"] == 13 and readiness(d)["ready"]                  # counting changes nothing
    led = Ledger(d).writer()                                                                    # an engine holding the lock
    with pytest.raises(SystemExit, match="REFUSED"):
        LV.reopen_live(d, off_protocol=True, apply=True, log=lambda m: None)
    led.close()
    out = LV.reopen_live(d, off_protocol=True, reason="GMICloud dropped after the pilot", apply=True, log=lambda m: None)
    assert out["applied"] and out["keys"] == 13 and not readiness(d)["ready"]
    req, att = ledger_rows(d)
    assert all(not r["terminal"] and r["outcome"] == "held" and r["verdict_json"] is None for c, r in req.items() if c != key(13))
    assert req[key(13)]["terminal"] and req[key(13)]["outcome"] == "moderation"
    assert all(json.loads(a["reopened_json"])["outcome"] in ("verdict", "parse_error", "truncated", "judge_refusal", "moderation")
               for v in att.values() for a in v if a["provider"] == "GMICloud" and a["http_status"] == 200)
    m = Mock(provider="DeepInfra")
    assert run_engine(d, m)[0] == 0
    code, rep, rows = file_dry(tmp_path, jobs, src, [arm, src, d], "s2")
    assert code == 0 and rep["gate"]["problems"] == []
    items = _items(rows)
    assert {it["extra"]["provider"] for it in items.values() if it["extra"]["transport"] == "openrouter_live"} == {"DeepInfra"}
    # the re-opened attempts spent no re-ask: key 15 (text every time) gets its full parse-error budget again from DeepInfra
    req, att = ledger_rows(d)
    assert [a["outcome"] for a in att[key(15)]] == ["held"] * 3 + ["parse_error"] * 3
    # a tampered served model or billed reasoning on a terminal live row is refused the same way
    con = sqlite3.connect(os.path.join(d, "state.sqlite"))
    con.execute("UPDATE live_attempts SET reasoning_tokens = 7 WHERE custom_id = ? AND terminal = 1", (key(6),))
    con.commit()
    con.close()
    with pytest.raises(SystemExit, match="reasoning_billed"):
        file_dry(tmp_path, jobs, src, [arm, src, d], "s3")


def test_reopen_by_provider_re_judges_only_that_providers_keys(tmp_path):
    src = make_source(tmp_path, behaviors={i: "verdict" for i in range(6, 16)})
    d = init_live(tmp_path, src)
    assert run_engine(d, Mock(provider="StreamLake"), limit=4)[0] == 0
    assert run_engine(d, Mock(provider="DeepInfra"))[0] == 0
    out = LV.reopen_live(d, providers=["StreamLake"], reason="after the pilot", apply=True, log=lambda m: None)
    assert out["keys"] == 4 and out["by_match"] == {"provider:StreamLake": 4}
    m = Mock(provider="GMICloud")
    assert run_engine(d, m)[0] == 0 and m.total == 4
    req, att = ledger_rows(d)
    assert all(r["terminal"] and r["outcome"] == "verdict" for r in req.values())
    assert {v[-1]["provider"] for v in att.values()} == {"DeepInfra", "GMICloud"}
    con = sqlite3.connect(os.path.join(d, "state.sqlite"))
    assert con.execute("SELECT COUNT(*) FROM live_reopens").fetchone()[0] == 4
    assert con.execute("SELECT COUNT(*) FROM attempts WHERE error_kind = 'reopened' AND outcome = 'held'").fetchone()[0] == 4
    con.close()


def test_the_engine_pauses_when_the_batch_path_reopens_a_pending_batch(tmp_path):
    src = make_source(tmp_path)
    d = init_live(tmp_path, src)
    led = Ledger(src).writer()
    led.update_batch("b4", state="submitted", batch_id="batch-4")
    led.close()
    m = Mock()
    code, _ = run_engine(d, m)
    assert code == 4 and m.total == 0 and "re-opened" in open(os.path.join(d, "PAUSED")).read()


def test_stop_file_and_concurrency_file(tmp_path):
    src = make_source(tmp_path, behaviors={i: "verdict" for i in range(6, 16)})
    d = init_live(tmp_path, src)
    pathlib.Path(d, "CONCURRENCY").write_text("2\n")
    code, runner = run_engine(d, Mock(latency=0.01), concurrency=6)
    assert code == 0 and runner.limiter.target == 2
    pathlib.Path(d, "STOP").write_text("")
    with pytest.raises(SystemExit, match="STOP exists"):
        run_engine(d, Mock())


# ---------------------------------------------------------------------------------------------------------------------------------------
# handover: the fallback for in-flight batches, and the re-asks a gated batch harvester will not build
# ---------------------------------------------------------------------------------------------------------------------------------------
def test_handover_in_flight_after_a_deadline_and_reasks_with_their_prior_outcomes(tmp_path):
    src = make_source(tmp_path)
    d = init_live(tmp_path, src)
    with pytest.raises(SystemExit, match="not before"):
        LV.build_handover(d, ["in_flight"], "fallback", not_before="2099-01-01T00:00:00Z", log=lambda m: None)
    path = LV.build_handover(d, ["in_flight", "reasks"], "fallback", not_before="2020-01-01T00:00:00Z", log=lambda m: None)
    doc = json.loads(gzip.open(path).read())
    by = {k["custom_id"]: k for k in doc["keys"]}
    assert set(by) == {key(2), key(3), key(4), key(5)}
    assert by[key(4)]["origin"] == "in_flight" and by[key(4)]["source_batch_key"] == "b1"
    assert by[key(3)]["origin"] == "reask" and by[key(3)]["prior_outcomes"] == ["truncated"]
    led = Ledger(d).writer()
    assert LV.ingest_handovers(led, d, log=lambda m: None) == 4
    assert LV.ingest_handovers(led, d, log=lambda m: None) == 0                  # once per file
    led.close()
    assert LV.build_handover(d, ["in_flight", "reasks"], "again", log=lambda m: None) is None   # nothing left that is not live
    mock = Mock()
    assert run_engine(d, mock)[0] == 0
    req, att = ledger_rows(d)
    assert att[key(3)][0]["max_tokens"] == 4000                                  # a cut reply in the source is re-asked at the raised cap
    assert att[key(2)][0]["max_tokens"] == 1500 and req[key(4)]["outcome"] == "verdict"
    assert len(req) == 14


def _set_run_state(d, **kw):
    led = Ledger(d).writer()
    LV.ensure_schema(led.conn)
    rs = LV.meta_get(led.conn, "run_state") or {}
    rs.update(kw)
    LV.meta_set(led.conn, "run_state", rs)
    led.close()


def test_auto_start_only_after_a_full_run_that_ended_done(tmp_path):
    src = make_source(tmp_path, behaviors={i: "verdict" for i in range(6, 16)})
    d = init_live(tmp_path, src)
    assert "no engine run" in LV.auto_start_refusal(d)
    assert run_engine(d, Mock(), limit=2)[0] == 0
    assert "pilot" in LV.auto_start_refusal(d) and LV.last_run(d)["end_state"] == "pilot_done"
    pathlib.Path(d, "STOP").write_text("")                                       # a deliberate stop of the full run
    with pytest.raises(SystemExit):
        run_engine(d, Mock())
    os.remove(os.path.join(d, "STOP"))
    mock = Mock(latency=0.05, on_request=lambda m, k, n: pathlib.Path(d, "STOP").touch() if m.total == 3 else None)
    assert run_engine(d, mock, concurrency=1)[0] == 3
    assert LV.last_run(d)["end_state"] == "stopped" and "ended stopped" in LV.auto_start_refusal(d)
    assert run_engine(d, Mock())[0] == 0
    assert LV.last_run(d)["end_state"] == "done" and LV.auto_start_refusal(d) is None
    for kw, why in (({"end_state": "paused", "exit_code": 4}, "ended paused"), ({"end_state": "unfinished", "exit_code": 5}, "ended unfinished"),
                    ({"end_state": "crashed", "clean_exit": False}, "ended crashed"), ({"end_state": None}, "has not recorded an end"),
                    ({"end_state": "done", "clean_exit": False}, "clean_exit False")):
        _set_run_state(d, **dict({"end_state": "done", "exit_code": 0, "clean_exit": True, "limit": None}, **kw))
        assert why in LV.auto_start_refusal(d), kw
    _set_run_state(d, end_state="done", exit_code=0, clean_exit=True, limit=None)
    pathlib.Path(d, "PAUSED").write_text("x\n")
    assert "PAUSED" in LV.auto_start_refusal(d)


def test_a_handover_after_a_pilot_is_registered_and_an_unattended_restart_is_refused_until_a_full_run_ends_done(tmp_path):
    src = make_source(tmp_path, behaviors={i: "verdict" for i in range(6, 16)})
    d = init_live(tmp_path, src)
    assert run_engine(d, Mock(), limit=2)[0] == 0
    assert SRL.main(["handover", "--live-dir", d, "--reasks", "--ingest"]) == 0
    assert len(ledger_rows(d)[0]) == 12                                          # the re-asks are registered
    assert "pilot" in LV.auto_start_refusal(d)                                   # after a pilot, a restart is a person's decision
    assert run_engine(d, Mock())[0] == 0
    assert LV.auto_start_refusal(d) is None                                      # a full run that ended done: a scheduled restart may go
    assert SRL.main(["handover", "--live-dir", d, "--in-flight", "--not-before", "2020-01-01T00:00:00Z", "--ingest"]) == 0
    assert run_engine(d, Mock())[0] == 0                                         # it ingested the in-flight handover and finished it


def test_a_429_with_retry_after_zero_waits_at_least_the_floor_and_halves_at_once(tmp_path):
    src = make_source(tmp_path, behaviors={i: "verdict" for i in range(6, 16)})
    d = init_live(tmp_path, src)
    mock = Mock(global_429=6, retry_after=0)
    code, runner = run_engine(d, mock, concurrency=4, rate_limit_min_s=0.02, rl_halve_after=3, rl_halve_spacing_s=60.0)
    assert code == 0 and runner.rl_sleep_n == 6 and runner.rl_sleep_min_s >= 0.02
    assert runner.limiter.halvings == 1 and runner.limiter.limit == 2           # the first halving needs no wait; the next waits 60 s
    r = LV.LiveRunner(d, overrides={"rate_limit_min_s": 1.0, "rate_limit_backoff_s": 5.0, "rate_limit_backoff_max_s": 120.0}, log=lambda m: None)
    for n, ra, lo, hi in ((1, 0.0, 3.75, 6.25), (1, None, 3.75, 6.25), (3, 0.0, 15.0, 25.0), (1, 30.0, 22.5, 37.5), (9, 0.0, 90.0, 120.0),
                          (1, 1e6, 90.0, 120.0)):
        for _ in range(50):
            assert lo <= r._rl_sleep(n, ra) <= hi, (n, ra)
    r2 = LV.LiveRunner(d, overrides={"rate_limit_min_s": 1.0, "rate_limit_backoff_s": 0.001}, log=lambda m: None)
    assert min(r2._rl_sleep(1, 0.0) for _ in range(200)) >= 1.0


def test_an_engine_exception_is_a_counted_attempt_not_a_silent_requeue(tmp_path, monkeypatch):
    src = make_source(tmp_path, behaviors={i: "verdict" for i in range(6, 16)})
    d = init_live(tmp_path, src)
    orig = LV.LiveRunner._send

    async def crashing(self, session, item, headers):
        res = await orig(self, session, item, headers)
        if item.custom_id == key(6):
            raise RuntimeError("synthetic crash after the reply arrived")
        return res

    monkeypatch.setattr(LV.LiveRunner, "_send", crashing)
    mock = Mock()
    code, runner = run_engine(d, mock)
    req, att = ledger_rows(d)
    n = int(MM["retry"]["provider_error"]) + 1
    assert code == 0 and mock.calls[6] == n and len(att[key(6)]) == n               # bounded by the provider_error budget
    assert req[key(6)]["outcome"] == "provider_error" and req[key(6)]["terminal"]
    assert json.loads(req[key(6)]["error_json"])["kind"] == "engine_exception"
    assert all(a["cost_source"] == "unknown_estimated" for a in att[key(6)])
    assert LV.live_spend(sqlite3.connect(os.path.join(d, "state.sqlite")))["unknown_estimated"] > 0


def test_a_refused_handover_mid_run_pauses_cleanly_and_registers_nothing(tmp_path):
    src = make_source(tmp_path, behaviors={i: "verdict" for i in range(6, 16)})
    d = init_live(tmp_path, src)

    def drop_bad_handover(m, k, n):
        if m.total == 1:
            hdir = pathlib.Path(d, "handovers")
            hdir.mkdir(exist_ok=True)
            doc = {"schema": LV.HANDOVER_SCHEMA, "handover_id": "20990101T000000Z-reasks", "source_run_dir": src, "modes": ["reasks"],
                   "keys": [{"custom_id": key(2), "source_batch_key": "b0", "origin": "reask", "prior_outcomes": ["parse_error"]},
                            {"custom_id": key(0), "source_batch_key": "b1", "origin": "reask", "prior_outcomes": []}]}   # key 0 is not in b1
            (hdir / "20990101T000000Z-reasks.json.gz").write_bytes(gzip.compress(json.dumps(doc).encode()))

    code, _ = run_engine(d, Mock(latency=0.05, on_request=drop_bad_handover), concurrency=1, global_check_s=0.05)
    assert code == 4 and "handover file could not be ingested" in open(os.path.join(d, "PAUSED")).read()
    rs = LV.last_run(d)
    assert rs["clean_exit"] is True and rs["end_state"] == "paused" and rs["inflight_est_no_fee"] == 0.0
    con = sqlite3.connect(os.path.join(d, "state.sqlite"))
    assert con.execute("SELECT COUNT(*) FROM live_keys WHERE custom_id = ?", (key(2),)).fetchone()[0] == 0      # rolled back
    assert con.execute("SELECT COUNT(*) FROM live_handovers").fetchone()[0] == 0
    con.close()


def test_an_exception_past_the_monitor_keeps_the_unrecorded_spend(tmp_path, monkeypatch):
    src = make_source(tmp_path, behaviors={i: "verdict" for i in range(6, 16)})
    d = init_live(tmp_path, src)
    orig = LV.LiveRunner._flush
    calls = {"n": 0}

    def failing(self):
        calls["n"] += 1
        if calls["n"] >= 2 and self.window:
            self.window, w = [], self.window
            self._flushing = w
            raise sqlite3.OperationalError("synthetic: disk I/O error")
        return orig(self)

    monkeypatch.setattr(LV.LiveRunner, "_flush", failing)
    with pytest.raises(sqlite3.OperationalError):
        run_engine(d, Mock(latency=0.02), concurrency=2, flush_n=1)
    rs = LV.last_run(d)
    assert rs["clean_exit"] is False and rs["end_state"] == "crashed" and rs["inflight_est_no_fee"] > 0
    monkeypatch.setattr(LV.LiveRunner, "_flush", orig)
    assert run_engine(d, Mock())[0] == 0
    con = sqlite3.connect(os.path.join(d, "state.sqlite"))
    adj = con.execute("SELECT usd_no_fee FROM live_spend_adjustments").fetchall()
    con.close()
    assert [round(a[0], 9) for a in adj] == [round(rs["inflight_est_no_fee"], 9)]
    assert run_engine(d, Mock())[0] == 0                                         # counted once, never again
    con = sqlite3.connect(os.path.join(d, "state.sqlite"))
    assert con.execute("SELECT COUNT(*) FROM live_spend_adjustments").fetchone()[0] == 1
    con.close()


# ---------------------------------------------------------------------------------------------------------------------------------------
# the filing: full run + bake-off arm + live run
# ---------------------------------------------------------------------------------------------------------------------------------------
def filing_world(tmp_path):
    arm = tmp_path / "arm"
    BF.write_track(str(arm), {"track": "strongreject_bakeoff__minimax-m3", "model": MM["model"], "endpoint": MM["endpoint"], "provider": None,
                              "parser": "rejudge.strongreject.request:parse_result", "retry": dict(MM["retry"], parse_error=1),
                              "request": MM["request"], "truncated_retry_max_tokens": 4000})
    led = Ledger(str(arm)).writer()
    man = _manifest(arm, "b00000", [(key(16), body(16, "verdict")), (key(7), body(7, "text_then_verdict"))])
    led.register_manifest(man)
    led.update_batch("b00000", state="harvested", batch_id="batch-arm", cost=0.001)
    led.record_outcome(key(16), "b00000", "batch-arm", "verdict", True, {"refused": 1, "convincing": 1, "specific": 1, "score": 0.0}, None, 200,
                       "tool_calls", None, None, "DeepInfra")
    led.record_outcome(key(7), "b00000", "batch-arm", "parse_error", True, None, {"kind": "no_tool_call"}, 200, "stop", None, None, "DeepInfra")
    led.close()
    src = make_source(tmp_path, reuse=str(arm))
    json.dump({"keys": [key(16)]}, open(os.path.join(src, "reuse.json"), "w"))   # the arm's parse error (key 7) was re-sent, not carried
    with gzip.open(os.path.join(src, "units.tsv.gz"), "wt") as fh:
        fh.write("run_id\titem_id\ttrial\tsample\tturn\tcustom_id\tkind\tresponse_chars\n")
        for i in range(17):
            fh.write(f"{RUN}\tgoal_{i % 3}\t0\t{i}\t-1\t{key(i)}\tsr\t10\n")
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text(json.dumps({"run_id": RUN, "kind": "sr", "eval_id": "pasttense", "haiku_judgment_id": "h" * 36,
                                "haiku_judge_id": FR.HAIKU_JUDGES["sr"]}) + "\n")
    d = init_live(tmp_path, src)
    return str(arm), src, d, str(jobs)


def file_dry(tmp_path, jobs, src, run_dirs, name="staging"):
    staging = tmp_path / name
    argv = ["--jobs", jobs, "--units", os.path.join(src, "units.tsv.gz"), "--staging", str(staging), "--code-sha", "0" * 40, "--dry-run",
            "--judge", "minimax-m3"]
    for r in run_dirs:
        argv += ["--run-dir", r]
    code = FR.main(argv)
    rep = json.load(open(staging / "filing_report.json"))
    rows = []
    for p in staging.rglob("*.jsonl*"):
        op = gzip.open if str(p).endswith(".gz") else open
        with op(p, "rt") as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    return code, rep, rows


def _items(rows, metric="strongreject"):
    out = {}
    for r in rows:
        p = r.get("payload") or r
        if isinstance(p, dict) and p.get("metric_id") == metric:
            out[p["extra"]["request_key"]] = p
        for it in (p.get("rows") or []) if isinstance(p, dict) else []:
            if it.get("metric_id") == metric:
                out[it["extra"]["request_key"]] = it
    return out


def test_filing_merges_full_arm_and_live_with_the_gate_reading_live_coverage(tmp_path):
    arm, src, d, jobs = filing_world(tmp_path)
    with pytest.raises(SystemExit, match="pass that run directory"):
        file_dry(tmp_path, jobs, src, [arm, d], "s0")                            # the live run files with its source
    assert run_engine(d, Mock())[0] == 0
    code, rep, _ = file_dry(tmp_path, jobs, src, [arm, src, d], "s1")
    probs = " ".join(rep["gate"]["problems"])
    assert code == 0 and "waiting on a re-ask" in probs and "live batches" in probs    # keys 2-5 are neither terminal nor live yet
    # the in-flight batch has not come back: hand it over with the stranded re-asks
    LV.build_handover(d, ["in_flight", "reasks"], "fallback", log=lambda m: None)
    assert run_engine(d, Mock())[0] == 0
    code, rep, rows = file_dry(tmp_path, jobs, src, [arm, src, d], "s2")
    assert code == 0 and rep["gate"]["problems"] == [], rep["gate"]
    assert rep["runs"][RUN]["status"] == "complete"
    assert rep["judge_backend"] == "batch_or_live" and rep["judges"]["sr"] != "e0cff0ab5c6ad6225e8149a20022ed63afc3cd04cc5ed22628a368810f8cb9cc"
    assert rep["merge"]["reuse_not_carried_ignored"] == 1                        # the arm's parse error of key 7 yields to its re-send
    assert rep["backend_counts"]["openrouter_batch"] == 3 and rep["backend_counts"]["openrouter_live"] == 14
    assert rep["live_coverage"][src]["uncovered_non_terminal"] == 0 and rep["live_coverage"][src]["raw_paused"]
    items = _items(rows)
    assert items[key(0)]["extra"]["transport"] == "openrouter_batch" and items[key(16)]["extra"]["transport"] == "openrouter_batch"
    assert items[key(7)]["extra"]["transport"] == "openrouter_live" and items[key(7)]["extra"]["quantization"] == "fp8"
    assert items[key(7)]["extra"]["provider"] == "GMICloud" and items[key(7)]["extra"]["generation_id"].startswith("gen-")
    judges = [(r.get("payload") or r) for r in rows if (r.get("payload") or r).get("judge_backend")]
    assert judges and all(j["judge_backend"] == "batch_or_live" and j["extra"]["protocol"]["live_route"]["provider_routing"] == SRL.routing()
                          for j in judges)


def test_filing_refuses_a_duplicate_and_resolves_a_handed_over_key_by_time(tmp_path):
    arm, src, d, jobs = filing_world(tmp_path)
    LV.build_handover(d, ["in_flight", "reasks"], "fallback", log=lambda m: None)
    assert run_engine(d, Mock())[0] == 0
    # the late batch comes back after the live verdicts: key 4 (handed over) is a duplicate resolved by time, live first
    led = Ledger(src).writer()
    led.record_outcome(key(4), "b1", "batch-1", "verdict", True, {"refused": 1, "convincing": 1, "specific": 1, "score": 0.0}, None, 200,
                       "tool_calls", None, None, "DeepInfra")
    led.conn.execute("UPDATE requests SET harvested_utc = '2099-01-01T00:00:00Z' WHERE custom_id = ?", (key(4),))
    led.conn.commit()
    led.close()
    code, rep, rows = file_dry(tmp_path, jobs, src, [arm, src, d], "s1")
    assert code == 0 and rep["merge"]["handover_duplicates_first_kept"] == 1
    it = _items(rows)[key(4)]
    assert it["extra"]["transport"] == "openrouter_live" and it["extra"]["duplicate_outcomes"][0]["backend"] == "batch"
    # ... and a batch result that came FIRST stands over the live one
    led = Ledger(src).writer()
    led.conn.execute("UPDATE requests SET harvested_utc = '2000-01-01T00:00:00Z' WHERE custom_id = ?", (key(4),))
    led.conn.commit()
    led.close()
    code, rep, rows = file_dry(tmp_path, jobs, src, [arm, src, d], "s2")
    it = _items(rows)[key(4)]
    assert it["extra"]["transport"] == "openrouter_batch" and it["extra"]["duplicate_outcomes"][0]["backend"] == "live"
    # a key the live run took at its init (never submitted) judged in the batch run as well is refused
    led = Ledger(src).writer()
    led.record_outcome(key(6), "b2.s0", "batch-x", "verdict", True, {"refused": 1, "convincing": 1, "specific": 1, "score": 0.0}, None, 200,
                       "tool_calls", None, None, "DeepInfra")
    led.close()
    with pytest.raises(SystemExit, match="more than one ledger"):
        file_dry(tmp_path, jobs, src, [arm, src, d], "s3")


def test_read_outcomes_lets_an_infrastructure_outcome_give_way(tmp_path):
    arm, src, d, jobs = filing_world(tmp_path)
    led = Ledger(d).writer()
    led.record_outcome(key(6), "src:b2.s0", None, "provider_error", True, None, {"kind": "transport_timeout"}, None, None, None, None, None)
    led.record_outcome(key(12), "src:b4", None, "verdict", True, {"refused": 0, "convincing": 2, "specific": 2, "score": 0.25}, None, 200,
                       "tool_calls", None, None, "StreamLake")
    led.close()
    led = Ledger(src).writer()
    led.record_outcome(key(6), "b2.s0", "batch-y", "verdict", True, {"refused": 0, "convincing": 3, "specific": 3, "score": 0.5}, None, 200,
                       "tool_calls", None, None, "DeepInfra")
    led.record_outcome(key(12), "b4", "batch-z", "provider_error", True, None, {"kind": "provider_error"}, 502, None, None, None, None)
    led.close()
    st = collections.Counter()
    out = FR.read_outcomes([arm, src, d], stats=st)
    assert out[key(6)]["backend"] == "batch" and out[key(6)]["settled_infra"] == [d]
    assert out[key(12)]["backend"] == "live" and out[key(12)]["settled_infra"] == [src]
    assert st["infra_settled_by_another_ledger"] == 2


def test_the_batch_only_judge_rows_are_unchanged_and_the_live_ones_are_pinned(tmp_path):
    for kind, jid in (("sr", "e0cff0ab5c6ad6225e8149a20022ed63afc3cd04cc5ed22628a368810f8cb9cc"),
                      ("sr2", "0c6a124c2bcb3c5bffd0b8917594f9d7df46ed884fb5a60e0b22b0ae1f17f1cc")):
        assert FR.judge_spec(kind, judge="minimax-m3")["judge_id"] == jid
    src = make_source(tmp_path)
    d = init_live(tmp_path, src)
    live = FR.live_protocol([src, d])
    assert live["with_batch"] and live["providers"]["DeepInfra"] == {"slug": "deepinfra/fp8", "quantization": "fp8"}
    j = {k: FR.judge_spec(k, judge="minimax-m3", live=live) for k in ("sr", "sr2")}
    assert {k: v["judge_id"] for k, v in j.items()} == LIVE_JUDGE_IDS
    assert all(v["judge_backend"] == "batch_or_live" for v in j.values())
    assert FR.judge_spec("sr", 4000, judge="minimax-m3", live=live)["judge_id"] == j["sr"]["judge_id"]   # the cap stays out of the identity


#: the MiniMax-M3 batch-and-live judge rows (the routing of scripts/rejudge/strongreject/live.py): a change here is a new judge
LIVE_JUDGE_IDS = {"sr": "192df392a676e6b046e86c08ebb96484d31cad7573736d10e7c5351a5128605c",
                  "sr2": "d6c07d238e6ae370846ed272524201b423bc9885f3d1b66933af47bfc7eb6be7"}


def test_the_global_walker_counts_live_ledgers(tmp_path):
    src = make_source(tmp_path, behaviors={i: "verdict" for i in range(6, 16)})
    d = init_live(tmp_path, src)
    assert run_engine(d, Mock(cost=0.01))[0] == 0
    out = LV.openrouter_spend(str(tmp_path))
    assert out["live_usd"] == pytest.approx(10 * 0.01 * 1.055)
    assert out["usd"] == pytest.approx(10 * 0.01 * 1.055 + 0.02 * 1.055 + 0.002)  # + the source's batches (two harvested, one in flight)
    assert LV.openrouter_spend(str(tmp_path), exclude=[d])["live_usd"] == 0


def test_the_dry_run_prices_the_keys_left(tmp_path):
    src = make_source(tmp_path)
    d = init_live(tmp_path, src)
    out = LV.dry_run(d, assume_latency_s=10.0)
    assert out["keys"]["remaining"] == 10 and out["keys"]["remaining_with_prior_attempts"] == 2
    c = out["cost_usd_with_fee"]
    assert 0 < c["low"] <= c["central"] <= c["high"]
    assert out["duration"]["latency_source"].startswith("ASSUMED") and out["prices_per_mtok"]["high"][0] == 0.30


def test_the_live_cli_states_every_ceiling_and_starts_nothing_by_itself(tmp_path, capsys):
    """``init`` takes no default for the run's ceiling, the global ceiling or the root it counts; there is no unit or start command."""
    src = make_source(tmp_path)
    base = ["init", "--live-dir", str(tmp_path / "live"), "--source-run-dir", src]
    full = base + ["--max-spend-usd", "5", "--global-root", str(tmp_path), "--global-ceiling-usd", "40"]
    for drop in ("--max-spend-usd", "--global-root", "--global-ceiling-usd"):
        i = full.index(drop)
        with pytest.raises(SystemExit) as e:
            SRL.main(full[:i] + full[i + 2:])
        assert e.value.code == 2 and drop in capsys.readouterr().err
    with pytest.raises(SystemExit, match="not an absolute path"):
        SRL.main(base + ["--max-spend-usd", "5", "--global-root", "relative", "--global-ceiling-usd", "40"])
    assert SRL.main(full) == 0
    tr = BF.read_track(str(tmp_path / "live"))
    assert tr["max_spend_usd"] == 5.0 and tr["live"]["global_ceiling_usd"] == 40.0 and tr["live"]["global_root"] == str(tmp_path)
    for cmd in ("unit-cmd", "start"):
        with pytest.raises(SystemExit):
            SRL.main([cmd, "--live-dir", str(tmp_path / "live")])
    capsys.readouterr()
