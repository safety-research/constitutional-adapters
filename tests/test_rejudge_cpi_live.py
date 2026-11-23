"""The C/P/I track's live slot (scripts/rejudge/terra/live.py on the shared engine scripts/rejudge/common/live.py) and the track's filing
that merges it with the batch runs (scripts/rejudge/terra/file_records.py over scripts/rejudge/common/merge.py): the key set is exactly
each batch run's never-submitted keys, a key set registered while the batch gate is open is only STAGED, the body sent is the batch
manifest's byte for byte (plus the model and the routing), the routing keeps only endpoints that declare every parameter the bodies
carry (the agentic json_schema never goes to Parasail), the track's parsers and re-ask policy decide every outcome, the moderation-share
hold and the gate guard pause, the spend ceilings hold (none has a default), one process runs all three engines, and the filing files each key once
(duplicates refused, handed-over keys by time, infrastructure outcomes giving way) under batch-and-live judges. Offline: synthetic
texts, a local mock of OpenRouter's chat completions (aiohttp on 127.0.0.1), no registry."""
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
import threading
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from aiohttp import web  # noqa: E402

import judge_rubrics as JR  # noqa: E402
from rejudge.common import batchfile as BF  # noqa: E402
from rejudge.common import live as LV  # noqa: E402
from rejudge.common.state import Ledger, readiness  # noqa: E402
from rejudge.strongreject import live as SRL  # noqa: E402
from rejudge.terra import agentic as A  # noqa: E402
from rejudge.terra import cpi as C  # noqa: E402
from rejudge.terra import file_records as FR  # noqa: E402
from rejudge.terra import harvest as TH  # noqa: E402
from rejudge.terra import judges as J  # noqa: E402
from rejudge.terra import live as TL  # noqa: E402
from rejudge.terra import track as T  # noqa: E402

MM = "minimax-m3"
TEST_KEY = "test-key-not-real"
FAST = {"concurrency": 4, "min_concurrency": 1, "flush_n": 50, "flush_s": 0.05, "control_s": 0.05, "progress_s": 0.1,
        "global_check_s": 3600.0, "transport_retries": 1, "transport_backoff_s": 0.001, "transport_backoff_max_s": 0.01,
        "rate_limit_min_s": 0.0, "rate_limit_backoff_s": 0.001, "rate_limit_backoff_max_s": 0.01, "rl_window_s": 30.0, "rl_halve_after": 3,
        "rl_halve_spacing_s": 0.0, "rl_recover_s": 3600.0, "request_timeout_s": 20.0, "cooldown_s": 0.05, "cooldown_max_s": 0.2,
        "price_tolerance": 0}                     # the mock's flat cost is not a list price (the engine's own tests check the price rule)
FAMILY = {"cpi": "cpi", "agentic": "agentic", "option_xstest_contrast": "cpi"}
#: the synthetic ceilings every test live directory states (init has no default): per run, and the global one
SPEND = {"cpi": 30.0, "agentic": 24.0, "option_xstest_contrast": 6.0}
CEILING = 1000.0
#: the layout every synthetic source shares (the real full runs': harvested, in flight, a 413-split parent with pending children, a
#: pending re-ask batch, a plain pending batch)
LIVE_KEYS = tuple(range(6, 16))


def zero_walker(root, exclude=()):
    return {"usd": 0.0, "per_dir": {}, "errors": []}


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", TEST_KEY)


# ---------------------------------------------------------------------------------------------------------------------------------------
# synthetic sources, built with the track's own builders (track.init_run, judges.build_entry)
# ---------------------------------------------------------------------------------------------------------------------------------------
def text(run: str, i: int, beh: str) -> dict:
    if FAMILY[run] == "cpi":
        rubric = "xstest_cpi" if run == "option_xstest_contrast" else "or_bench_cpi"
        p, r = f"Synthetic {run} request {i % 3}", f"MOCK:{beh} #{i} synthetic {run} response"
        return {"k": C.request_key(rubric, p, r), "rubric": rubric, "item_id": f"{run}-item-{i}", "prompt": p, "response": r}
    resp = f"MOCK:{beh} #{i} synthetic agentic response"
    return {"k": hashlib.sha256(f"synthetic agentic {i} {beh}".encode()).hexdigest(), "tag": A.TAGS["leaking"], "behavior": "leaking",
            "item_id": f"leaking_synthetic-{i}", "prompt": JR.PROMPTS[JR.AM_LEAK_TAG].format(response=resp, email_context="")}


def make_source(tmp: pathlib.Path, run: str, behaviors=None, extra_param=None, n_extra=0):
    """``<tmp>/full/<run>`` shaped like the real full run, its scope (``<tmp>/scope_<run>``), and the GO gate's path in track.json; keys
    0-15 in the layout above, ``n_extra`` more in one more pending batch."""
    fam = FAMILY[run]
    n = 16 + int(n_extra)
    beh = {i: "verdict" for i in range(n)}
    beh.update(behaviors or {})
    texts = {i: text(run, i, beh[i]) for i in range(n)}
    rd = tmp / "full" / run
    scope = tmp / f"scope_{run}"
    T.init_run(str(rd), fam, judge=MM, extra={"submit_gate": {"requires_files": [str(tmp / "GO")]}, "scope": str(scope),
                                              "health": dict(T.HEALTH, pause_on_batch_end=True), "max_spend_usd": 50.0})
    spec = BF.read_track(str(rd))
    hdr = BF.batch_header(spec["model"], spec["endpoint"], spec.get("provider"))

    def man(bk, idxs, attempt=0, parent=None, cap=None):
        reqs = []
        for i in idxs:
            e = J.build_entry(fam, texts[i], spec["request"])
            if cap:
                e["body"] = dict(e["body"], max_tokens=cap)
            if extra_param:
                e["body"] = dict(e["body"], **extra_param)
            reqs.append(BF.PackedRequest(e["custom_id"], BF.serialize_request(e["custom_id"], e["body"]), 1000.0, 200.0))
        return BF.write_manifest(str(rd), bk, hdr, reqs, spec["track"], attempt=attempt, parent=parent)

    k = {i: texts[i]["k"] for i in texts}
    led = Ledger(str(rd)).writer()
    v = {"label": "full_refusal", "reasoning_chars": 9, "n_tool_calls": 1} if fam == "cpi" else {"harmful": False, "reasoning_chars": 9}
    second = "truncated" if fam == "agentic" else "provider_error"
    led.register_manifest(man("b00000", [0, 1, 2, 3]))
    led.update_batch("b00000", state="harvested", batch_id="batch-0", cost=0.01)
    led.record_outcome(k[0], "b00000", "batch-0", "verdict", True, v, None, 200, "tool_calls", None, None, "DeepInfra")
    led.record_outcome(k[1], "b00000", "batch-0", "verdict", True, v, None, 200, "tool_calls", None, None, "DeepInfra")
    led.record_outcome(k[2], "b00000", "batch-0", "parse_error", False, None, {"kind": "no_tool_call"}, 200, "stop", None, None, "DeepInfra")
    led.record_outcome(k[3], "b00000", "batch-0", second, False, None, {"kind": second}, 200, "length", None, None, "DeepInfra")
    led.register_manifest(man("b00001", [4, 5]))
    led.update_batch("b00001", state="submitted", batch_id="batch-1", est_cost=0.002)
    led.register_manifest(man("b00002", [6, 7, 8, 9]))
    led.update_batch("b00002", state="split")
    for bk, ks in (("b00002.s0", (6, 7)), ("b00002.s1", (8, 9))):
        led.register_manifest(man(bk, ks, parent="b00002"))
    led.register_manifest(man("b00003", [10, 11]))
    led.update_batch("b00003", state="harvested", batch_id="batch-3", cost=0.01)
    led.record_outcome(k[10], "b00003", "batch-3", "parse_error", False, None, {"kind": "no_tool_call"}, 200, "stop", None, None, "DeepInfra")
    led.record_outcome(k[11], "b00003", "batch-3", second, False, None, {"kind": second}, 200, "length", None, None, "DeepInfra")
    # the re-ask batch the harvester built (attempt 1): an agentic cut verdict already at the raised cap
    led.conn.commit()
    retry = [(10, None)] + [(11, 8000 if fam == "agentic" else None)]
    reqs_hdr = []
    for i, cap in retry:
        e = J.build_entry(fam, texts[i], spec["request"])
        if cap:
            e["body"] = dict(e["body"], max_tokens=cap)
        if extra_param:
            e["body"] = dict(e["body"], **extra_param)
        reqs_hdr.append(BF.PackedRequest(e["custom_id"], BF.serialize_request(e["custom_id"], e["body"]), 1000.0, 200.0))
    led.register_manifest(BF.write_manifest(str(rd), "retry1_20261111T000000Z_000", hdr, reqs_hdr, spec["track"], attempt=1))
    led.register_manifest(man("b00004", [12, 13, 14, 15]))
    if n_extra:
        led.register_manifest(man("b00005", list(range(16, n))))
    led.close()
    # its scope: one run per family holding every key (the filing's FULL unit map)
    scope.mkdir(parents=True, exist_ok=True)
    run_id = f"{run}:synthetic-run"
    rubric = texts[0].get("rubric") or A.TAGS["leaking"]
    ev = {"cpi": "or_bench_hard", "option_xstest_contrast": "xstest", "agentic": "agentic_leaking"}[run]
    with gzip.open(scope / "units.tsv.gz", "wt") as fh:
        fh.write("run_id\titem_id\ttrial\tsample\tturn\tcustom_id\trubric\teval_id\torig_label\torig_status\tnote\n")
        for i in range(n):
            fh.write("\t".join([run_id, texts[i]["item_id"], "0", "0", "-1", k[i], rubric, ev, "0", "judged", json.dumps({"gate": 1})]) + "\n")
    with open(scope / "runs.jsonl", "w") as fh:
        rr = {"run_id": run_id, "eval_id": ev, "lane": "synthetic", "sonnet_judgment_id": "j-" + run, "sonnet_judge_id": "sonnet"}
        if FAMILY[run] == "cpi":
            rr["rubric"] = rubric
        fh.write(json.dumps(rr) + "\n")
    return str(rd), k


def req(run, tmp, **kw):
    """The settings every init states for ``run`` (none has a default)."""
    return dict({"max_spend_usd": SPEND[run], "global_ceiling_usd": CEILING, "global_root": str(tmp)}, **kw)


def init(tmp, run, gate_open=False, **kw):
    paths = TL.Paths(str(tmp))
    if gate_open:
        pathlib.Path(paths.gate_file).write_text("go\n")
    kw.setdefault("max_spend_usd", SPEND[run])
    kw.setdefault("global_ceiling_usd", CEILING)
    kw.setdefault("global_root", str(tmp))
    out = TL.init_run(run, paths, live_overrides=dict(FAST, **kw.pop("live", {})), log=lambda m: None, **kw)
    return paths, out


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
# a mock of OpenRouter's chat completions (C/P/I tool calls and agentic json_schema replies)
# ---------------------------------------------------------------------------------------------------------------------------------------
def completion(msg, finish, provider="GMICloud", cost=0.0004):
    return {"id": f"gen-{time.time_ns()}", "model": "minimax/minimax-m3", "provider": provider, "object": "chat.completion",
            "choices": [{"index": 0, "message": msg, "finish_reason": finish, "native_finish_reason": finish}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 150, "total_tokens": 1150, "cost": cost,
                      "prompt_tokens_details": {"cached_tokens": 200, "cache_write_tokens": 0}, "completion_tokens_details": {"reasoning_tokens": 0}}}


CPI_OK = {"role": "assistant", "content": None, "refusal": None, "tool_calls": [
    {"id": "c1", "type": "function", "function": {"name": C.TOOL_NAME, "arguments": json.dumps({"reasoning": "synthetic", "label": "full_compliance"})}}]}
AG_OK = {"role": "assistant", "content": json.dumps({"reasoning": "synthetic", "harmful": True}), "refusal": None}
TEXT = {"role": "assistant", "content": "I would classify this as follows", "refusal": None, "tool_calls": None}
CUT = {"role": "assistant", "content": "{\"reasoning\": \"partial", "refusal": None}


class Mock:
    def __init__(self, provider="GMICloud", cost=0.0004, provider_fn=None, on_request=None, latency=0.0, reasoning_fn=None):
        self.provider, self.cost, self.provider_fn, self.on_request, self.latency = provider, cost, provider_fn, on_request, latency
        self.reasoning_fn = reasoning_fn                     # (key index, provider, payload) -> reasoning tokens the reply bills
        self.calls = collections.Counter()
        self.total = 0
        self.payloads = []
        self.sent = {}

    def respond(self, beh, n, payload):
        cpi = "tools" in payload
        ok = completion(CPI_OK if cpi else AG_OK, "tool_calls" if cpi else "stop", self.provider, self.cost)
        if beh == "verdict":
            return 200, ok
        if beh == "text_then_verdict":
            return (200, completion(TEXT, "stop", self.provider, self.cost)) if n == 1 else (200, ok)
        if beh == "text_always":
            return 200, completion(TEXT if cpi else {"role": "assistant", "content": "not json", "refusal": None}, "stop", self.provider, self.cost)
        if beh == "cut_then_verdict":
            return (200, ok) if payload["max_tokens"] == 8000 else (200, completion(CUT, "length", self.provider, self.cost))
        if beh == "cut_always":
            return 200, completion(CUT, "length", self.provider, self.cost)
        if beh == "refusal":
            return 200, completion({"role": "assistant", "content": None, "refusal": "I can't help with that", "tool_calls": None},
                                   "content_filter", self.provider, self.cost)
        if beh == "filter":
            return 200, completion({"role": "assistant", "content": None, "refusal": None, "tool_calls": None}, "content_filter",
                                   self.provider, self.cost)
        if beh == "e500_then_verdict":
            return (500, {"error": {"code": 500, "message": "Internal Server Error"}}) if n <= 2 else (200, ok)
        if beh == "auth403":
            return 403, {"error": {"code": 403, "message": "Forbidden"}}
        raise AssertionError(beh)

    async def handle(self, request):
        payload = await request.json()
        content = " ".join(str(m.get("content")) for m in payload["messages"])
        m = re.search(r"MOCK:(\w+) #(\d+)", content)
        beh, k = m.group(1), int(m.group(2))
        if self.provider_fn is not None:
            self.provider = self.provider_fn(k, payload)
        self.total += 1
        self.calls[k] += 1
        n = self.calls[k]
        if self.on_request is not None:
            self.on_request(self, k, n)
        if self.latency:
            await asyncio.sleep(self.latency)
        status, out = self.respond(beh, n, payload)
        if self.reasoning_fn is not None and status == 200 and isinstance(out.get("usage"), dict):
            out["usage"]["completion_tokens_details"]["reasoning_tokens"] = int(self.reasoning_fn(k, self.provider, payload))
        self.payloads.append(payload)
        self.sent[(k, n)] = (status, out)
        return web.json_response(out, status=status)


def run_engine(live_dir, mock, limit=None, walker=None, max_spend=None, gate_file=None, **ov):
    async def go():
        app = web.Application()
        app.router.add_post("/api/v1/chat/completions", mock.handle)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            r = TL.TerraLiveRunner(live_dir, gate_file=gate_file, overrides=dict(FAST, base_url=f"http://127.0.0.1:{port}/api/v1", **ov),
                                   limit=limit, log=lambda m: None, spend_walker=walker or zero_walker, max_spend_usd=max_spend)
            return await r.run_async(), r
        finally:
            await runner.cleanup()
    return asyncio.run(go())


class ThreadedMock:
    """The mock served from a thread (for the run-all engines, which are child processes)."""

    def __init__(self, mock):
        self.mock = mock
        self.loop = asyncio.new_event_loop()
        self.port = None
        ready = threading.Event()

        def serve():
            asyncio.set_event_loop(self.loop)
            app = web.Application()
            app.router.add_post("/api/v1/chat/completions", mock.handle)
            self.runner = web.AppRunner(app)
            self.loop.run_until_complete(self.runner.setup())
            site = web.TCPSite(self.runner, "127.0.0.1", 0)
            self.loop.run_until_complete(site.start())
            self.port = site._server.sockets[0].getsockname()[1]
            ready.set()
            self.loop.run_forever()

        self.thread = threading.Thread(target=serve, daemon=True)
        self.thread.start()
        ready.wait(10)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/api/v1"

    def close(self):
        asyncio.run_coroutine_threadsafe(self.runner.cleanup(), self.loop).result(10)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(10)


# ---------------------------------------------------------------------------------------------------------------------------------------
# key selection, the batch gate, STAGED and the refresh
# ---------------------------------------------------------------------------------------------------------------------------------------
def test_init_takes_only_never_submitted_keys_and_is_staged_while_the_batch_gate_is_open(tmp_path):
    src, k = make_source(tmp_path, "cpi")
    paths, out = init(tmp_path, "cpi", gate_open=True)
    d = paths.live("cpi")
    con = sqlite3.connect(os.path.join(d, "state.sqlite"))
    rows = {r[0]: (r[1], r[2], json.loads(r[3])) for r in con.execute("SELECT custom_id, source_batch_key, origin, prior_outcomes_json FROM live_keys")}
    con.close()
    assert set(rows) == {k[i] for i in LIVE_KEYS} and not {k[i] for i in range(6)} & set(rows)
    assert rows[k[10]] == ("retry1_20261111T000000Z_000", "pending", ["parse_error"]) and rows[k[11]][2] == ["provider_error"]
    assert out["staged"] and TL.staged(d) and "gate" in TL.staged(d)
    tr = BF.read_track(d)
    assert tr["backend"] == "live" and tr["family"] == "cpi" and tr["terra_run"] == "cpi" and tr["batch_gate_file"] == paths.gate_file
    assert tr["max_spend_usd"] == SPEND["cpi"] and tr["live"]["global_ceiling_usd"] == CEILING and tr["live"]["global_root"] == str(tmp_path)
    assert tr["live"]["concurrency"] == FAST["concurrency"]                    # the tests' override of TL.CONCURRENCY["cpi"] (140)
    assert tr["parser"] == "rejudge.terra.cpi:parse_result" and tr["retry"] == T.RETRY and tr["request"] == BF.read_track(src)["request"]
    assert oct(os.stat(d).st_mode & 0o777) == "0o700"
    v = TL.verify_run("cpi", paths)
    assert v["ok"] and v["equal_to_source_pending"] and v["body_sha256_mismatches"] == 0 and v["keys_not_in_scope_units"] == 0
    # a staged key set never runs, and a refresh needs the gate closed
    with pytest.raises(SystemExit, match="STAGED"):
        TL.run_one("cpi", paths, log=lambda m: None)
    with pytest.raises(SystemExit, match="submit gate"):
        TL.init_run("cpi", paths, refresh=True, log=lambda m: None, **req("cpi", tmp_path))
    with pytest.raises(SystemExit, match="already holds"):
        TL.init_run("cpi", paths, log=lambda m: None, **req("cpi", tmp_path))
    # the batch harvester POSTed one more pending batch before the gate closed: the refresh takes only what is still pending
    led = Ledger(src).writer()
    led.update_batch("b00004", state="submitted", batch_id="batch-4")
    led.close()
    os.remove(paths.gate_file)
    out2 = TL.init_run("cpi", paths, refresh=True, live_overrides=FAST, log=lambda m: None, **req("cpi", tmp_path))
    con = sqlite3.connect(os.path.join(d, "state.sqlite"))
    keys2 = {r[0] for r in con.execute("SELECT custom_id FROM live_keys")}
    con.close()
    assert keys2 == {k[i] for i in range(6, 12)} and not out2["staged"] and TL.staged(d) is None
    assert out2["superseded"] and os.path.exists(os.path.join(out2["superseded"], "state.sqlite"))     # moved aside, never deleted
    assert TL.verify_run("cpi", paths)["ok"]
    # the gate reopened: the engine refuses to start; closed again, it runs; after a run the key set is fixed
    pathlib.Path(paths.gate_file).write_text("go\n")
    with pytest.raises(SystemExit, match="submit gate"):
        TL.run_one("cpi", paths, log=lambda m: None)
    os.remove(paths.gate_file)
    assert run_engine(d, Mock(), gate_file=paths.gate_file)[0] == 0
    with pytest.raises(SystemExit, match="already run"):
        TL.init_run("cpi", paths, refresh=True, log=lambda m: None, **req("cpi", tmp_path))


def test_a_refresh_refuses_while_a_source_batch_is_mid_post(tmp_path):
    src, _ = make_source(tmp_path, "agentic")
    paths, _ = init(tmp_path, "agentic", gate_open=True)
    os.remove(paths.gate_file)
    led = Ledger(src).writer()
    led.update_batch("b00004", state="submit_unknown")
    led.close()
    with pytest.raises(SystemExit, match="mid-POST"):
        TL.init_run("agentic", paths, refresh=True, log=lambda m: None, **req("agentic", tmp_path))


# ---------------------------------------------------------------------------------------------------------------------------------------
# the routing: every endpoint left declares every parameter the bodies carry
# ---------------------------------------------------------------------------------------------------------------------------------------
def test_the_routing_keeps_only_endpoints_that_declare_every_body_parameter(tmp_path):
    make_source(tmp_path, "cpi")
    make_source(tmp_path, "agentic")
    paths, oc = init(tmp_path, "cpi")
    _, oa = init(tmp_path, "agentic")
    tc, ta = BF.read_track(paths.live("cpi")), BF.read_track(paths.live("agentic"))
    # C/P/I: tools + tool_choice + reasoning + max_tokens, which all four fp8 endpoints declare: the StrongREJECT live routing
    assert oc["body_parameters"] == ["max_tokens", "reasoning", "tool_choice", "tools"]
    assert tc["live"]["routing"] == SRL.routing() and set(tc["live"]["providers"]) == {"DeepInfra", "GMICloud", "StreamLake", "Parasail"}
    # agentic: response_format, which Parasail fp8 does not declare (OpenRouter would drop it there): Parasail excluded and ignored
    assert oa["body_parameters"] == ["max_tokens", "reasoning", "response_format"]
    r = ta["live"]["routing"]
    assert r["only"] == ["deepinfra/fp8", "gmicloud/fp8", "streamlake/fp8"] and "parasail/fp8" in r["ignore"] and "parasail/fast" in r["ignore"]
    assert r["require_parameters"] is False and r["quantizations"] == ["fp8"] and r["max_price"] == {"prompt": 0.3, "completion": 1.2}
    assert ta["live"]["routing_basis"]["excluded"] == {"Parasail": ["response_format"]} and ta["live"]["routing_basis"]["batch_endpoint_missing"] == []
    assert LV.allowed_providers(ta["live"]["providers"], r) == {"DeepInfra", "GMICloud", "StreamLake"}
    # every endpoint left declares exactly what the batch endpoint declares for these bodies
    for name in ("DeepInfra", "GMICloud", "StreamLake"):
        for p in oa["body_parameters"] + oc["body_parameters"]:
            assert p in TL.ENDPOINTS[name]["parameters"] and p in TL.BATCH_ENDPOINT["parameters"]
    # a reply Parasail served on the agentic run is off the routing: held, never terminal, and the run pauses
    code, _ = run_engine(paths.live("agentic"), Mock(provider="Parasail"), flush_n=2)
    req, att = ledger_rows(paths.live("agentic"))
    assert code == 4 and "off-protocol" in open(os.path.join(paths.live("agentic"), "PAUSED")).read()
    assert all(a["violation"] == "off_map_provider" and a["outcome"] == "held" for v in att.values() for a in v)


def test_init_refuses_bodies_whose_parameters_are_not_the_judge_entrys(tmp_path):
    make_source(tmp_path, "cpi", extra_param={"temperature": 0.0})
    with pytest.raises(SystemExit, match="carry the parameters"):
        init(tmp_path, "cpi")
    with pytest.raises(SystemExit, match="no fp8 endpoint"):
        TL.routing_for(["max_tokens", "structured_outputs"])


# ---------------------------------------------------------------------------------------------------------------------------------------
# body identity, the track's parsers and re-ask policy
# ---------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("run", ["cpi", "agentic"])
def test_the_post_body_is_the_manifest_body_plus_model_and_routing(tmp_path, run):
    src, k = make_source(tmp_path, run, behaviors={8: "cut_then_verdict"} if run == "agentic" else None)
    paths, _ = init(tmp_path, run)
    d = paths.live(run)
    mock = Mock()
    assert run_engine(d, mock, gate_file=paths.gate_file)[0] == 0
    bodies = collections.defaultdict(list)
    for bk in BF.list_manifests(src):
        for r in BF.body_requests(BF.read_body(src, BF.read_manifest(src, bk))):
            bodies[r["custom_id"]].append(r["body"])
    routing = BF.read_track(d)["live"]["routing"]
    by_content = {json.dumps(b["messages"]): cid for cid, bs in bodies.items() for b in bs}
    for p in mock.payloads:
        assert list(p)[0] == "model" and list(p)[-1] == "provider" and p["model"] == "minimax/minimax-m3" and p["provider"] == routing
        sent = {x: v for x, v in p.items() if x not in ("model", "provider")}
        cid = by_content[json.dumps(sent["messages"])]
        if run == "agentic" and cid == k[8] and sent["max_tokens"] == 8000:
            assert sent == dict(bodies[cid][0], max_tokens=8000)                     # the cut verdict's one re-ask at the raised cap
        else:
            assert LV.compact(sent) in [LV.compact(b) for b in bodies[cid]]           # byte for byte a batch request body
    con = sqlite3.connect(os.path.join(d, "state.sqlite"))
    shas = dict(con.execute("SELECT custom_id, body_sha256 FROM live_keys").fetchall())
    con.close()
    _, att = ledger_rows(d)
    assert att[k[6]][0]["body_sha256"] == shas[k[6]] == LV.body_sha256(bodies[k[6]][0])


@pytest.mark.parametrize("run", ["cpi", "agentic"])
def test_outcomes_and_reasks_follow_the_terra_parser_and_policy(tmp_path, run):
    beh = {6: "verdict", 7: "text_then_verdict", 8: "cut_then_verdict", 9: "refusal", 10: "text_always", 11: "cut_always",
           12: "filter", 13: "e500_then_verdict", 14: "text_always", 15: "verdict"}
    src, k = make_source(tmp_path, run, behaviors=beh)
    paths, _ = init(tmp_path, run)
    d = paths.live(run)
    mock = Mock()
    code, runner = run_engine(d, mock, gate_file=paths.gate_file)
    assert code == 0
    req, att = ledger_rows(d)
    parse = C.parse_result if run == "cpi" else A.parse_result
    # parity: every recorded attempt is what the track's parser makes of the same reply wrapped as a batch result entry
    for (i, n), (status, out) in mock.sent.items():
        if status == 500 and n == 1:
            continue                                                          # retried in place, not an attempt
        a = att[k[i]][n - 1 if i != 13 else (0 if n == 2 else 1)]
        assert a["outcome"] == parse({"custom_id": k[i], "response": {"status_code": status, "body": out}})["outcome"], (i, n)
    cut = "truncated"
    want = {6: ("verdict", 1), 7: ("verdict", 2), 9: ("judge_refusal", 1), 12: ("moderation", 1), 13: ("verdict", 2), 15: ("verdict", 1),
            14: ("parse_error", 2)}                                           # parse_error: one re-ask (track.json retry)
    if run == "agentic":
        want.update({8: ("verdict", 2), 10: ("parse_error", 1), 11: (cut, 1)})   # 10: its source parse error spent the re-ask; 11: cut at 8000
    else:
        want.update({8: (cut, 1), 10: ("parse_error", 1), 11: (cut, 1)})   # no raised cap for C/P/I: a cut call is terminal at once
    for i, (outcome, n) in want.items():
        assert (req[k[i]]["outcome"], req[k[i]]["terminal"], len(att[k[i]])) == (outcome, 1, n), (i, req[k[i]]["outcome"], len(att[k[i]]))
    if run == "agentic":
        assert [a["max_tokens"] for a in att[k[8]]] == [1500, 8000] and att[k[11]][0]["max_tokens"] == 8000
    a6 = att[k[6]][0]
    assert (a6["provider"], a6["quantization"], a6["cost_source"]) == ("GMICloud", "fp8", "reported") and a6["generation_id"].startswith("gen-")
    assert readiness(d)["ready"]


@pytest.mark.parametrize("seq", [["parse_error", "parse_error"], ["truncated", "parse_error"], ["provider_error"] * 4, ["moderation"],
                                 ["account_error", "held", "parse_error", "balance_error", "parse_error"], ["batch_expired", "parse_error"],
                                 ["batch_failed_other", "provider_error", "provider_error", "provider_error", "provider_error"]])
def test_the_terminal_rule_is_the_terra_harvesters(tmp_path, seq):
    rd = str(tmp_path / "h")
    T.init_run(rd, "cpi", judge=MM)
    spec = BF.read_track(rd)
    man = BF.write_manifest(rd, "b", BF.batch_header(spec["model"], spec["endpoint"]), [BF.PackedRequest("k", BF.serialize_request("k", {}))], spec["track"])
    h = TH.TerraHarvester(rd, None, log=lambda m: None, dry_run=True)
    h.ledger.register_manifest(man)
    prior = []
    for o in seq:
        h._record("k", "b", "id", o, {"outcome": o}, None, spec["model"])
        h.ledger.conn.commit()
        assert bool(h.ledger.get("k")["terminal"]) == LV.is_terminal(o, prior, dict(LV.DEFAULT_RETRY, **spec["retry"])), (seq, o)
        prior.append(o)
    h.ledger.close()


def test_an_account_refusal_pauses_and_makes_nothing_terminal(tmp_path):
    make_source(tmp_path, "cpi", behaviors={i: "auth403" for i in range(16)})
    paths, _ = init(tmp_path, "cpi")
    d = paths.live("cpi")
    code, _ = run_engine(d, Mock(), concurrency=1, flush_n=1, gate_file=paths.gate_file)
    req, _ = ledger_rows(d)
    assert code == 4 and "account / credential" in open(os.path.join(d, "PAUSED")).read()
    assert not any(r["terminal"] for r in req.values())


# ---------------------------------------------------------------------------------------------------------------------------------------
# the track's rules on the engine: the moderation-share hold and the batch-gate guard
# ---------------------------------------------------------------------------------------------------------------------------------------
def test_a_window_above_the_moderation_share_is_held_and_pauses_until_acknowledged(tmp_path):
    # 10 keys' replies are content filters (moderation) and 16 are verdicts, all in one window: 38%, under the identical-error flood's
    # share (50%), far above the track hold's 5% (and at its min_count 10)
    beh = {i: ("filter" if 6 <= i <= 15 else "verdict") for i in range(32)}
    make_source(tmp_path, "cpi", behaviors=beh, n_extra=16)
    paths, _ = init(tmp_path, "cpi")
    d = paths.live("cpi")
    code, _ = run_engine(d, Mock(), concurrency=32, flush_n=500, flush_s=3.0, gate_file=paths.gate_file)
    req, att = ledger_rows(d)
    paused = open(os.path.join(d, "PAUSED")).read()
    assert code == 4 and "moderation share 10 of 26" in paused and TH.ACK_MODERATION in paused
    held = [c for c, r in req.items() if r["outcome"] == "held"]
    assert len(held) == 10 and all(json.loads(req[c]["error_json"])["kind"] == "held_moderation_share" and not req[c]["terminal"] for c in held)
    assert not any(r["outcome"] == "moderation" for r in req.values()) and sum(r["outcome"] == "verdict" for r in req.values()) == 16
    # a person read them: acknowledged, PAUSED removed, the held keys are re-sent. Alone in their window they are an identical-error
    # flood of the shared rule (as the batch harvester's re-ask batch of them would be): held again until that signature is acknowledged
    pathlib.Path(d, TH.ACK_FILE).write_text(TH.ACK_MODERATION + "\n")
    os.remove(os.path.join(d, "PAUSED"))
    m2 = Mock()
    assert run_engine(d, m2, concurrency=32, flush_n=500, flush_s=3.0, gate_file=paths.gate_file)[0] == 4 and m2.total == 10
    paused = open(os.path.join(d, "PAUSED")).read()
    assert "identical-error flood" in paused and "moderation share" not in paused
    sig = re.search(r"add the line (\w+) to", paused).group(1)
    pathlib.Path(d, TH.ACK_FILE).write_text(TH.ACK_MODERATION + "\n" + sig + "\n")
    os.remove(os.path.join(d, "PAUSED"))
    m3 = Mock()
    assert run_engine(d, m3, concurrency=32, flush_n=500, flush_s=3.0, gate_file=paths.gate_file)[0] == 0 and m3.total == 10
    req, att = ledger_rows(d)
    assert all(req[c]["outcome"] == "moderation" and req[c]["terminal"] for c in held)
    assert all([a["outcome"] for a in att[c]] == ["held", "held", "moderation"] for c in held)      # held attempts spend no re-ask
    # below the share it never holds: the same 10 among many more verdicts is recorded as moderation at once (the batch rule's parity)
    assert TH.MODERATION_SHARE == 0.05 and BF.read_track(d)["health"]["moderation_share"] == 0.05


def test_the_batch_gate_reappearing_mid_run_pauses_the_engine(tmp_path):
    make_source(tmp_path, "cpi")
    paths, _ = init(tmp_path, "cpi")
    d = paths.live("cpi")
    mock = Mock(latency=0.05, on_request=lambda m, k, n: pathlib.Path(paths.gate_file).write_text("go\n") if m.total == 2 else None)
    code, _ = run_engine(d, mock, concurrency=1, gate_file=paths.gate_file)
    assert code == 4 and "batch submit gate" in open(os.path.join(d, "PAUSED")).read() and mock.total < 10


# ---------------------------------------------------------------------------------------------------------------------------------------
# spend
# ---------------------------------------------------------------------------------------------------------------------------------------
def test_the_per_run_and_global_ceilings_stop_before_they_are_passed(tmp_path):
    make_source(tmp_path, "agentic")
    paths, _ = init(tmp_path, "agentic", max_spend_usd=3.0)
    d = paths.live("agentic")
    assert BF.read_track(d)["max_spend_usd"] == 3.0
    code, runner = run_engine(d, Mock(cost=1.0), concurrency=1, gate_file=paths.gate_file)
    assert code == 4 and "live spend ceiling" in open(os.path.join(d, "PAUSED")).read() and runner.spent_with_fee() <= 3.0
    os.remove(os.path.join(d, "PAUSED"))
    m = Mock()
    code, _ = run_engine(d, m, walker=lambda root, exclude=(): {"usd": CEILING, "per_dir": {"strongreject/live": CEILING}, "errors": []},
                         max_spend=100.0, gate_file=paths.gate_file)
    assert code == 4 and m.total == 0 and "GLOBAL ceiling" in open(os.path.join(d, "PAUSED")).read()
    # no ceiling has a default (every init states them); the concurrency split does
    assert not hasattr(TL, "MAX_SPEND_USD") and not hasattr(TL, "GLOBAL_CEILING_USD") and "global_ceiling_usd" not in LV.LIVE_DEFAULTS
    assert sum(TL.CONCURRENCY.values()) == 300


def test_the_global_walker_counts_every_live_ledger_of_the_track(tmp_path):
    for r in ("cpi", "agentic"):
        make_source(tmp_path, r)
    paths = TL.Paths(str(tmp_path))
    for r in ("cpi", "agentic"):
        init(tmp_path, r)
        assert run_engine(paths.live(r), Mock(cost=0.01), gate_file=paths.gate_file)[0] == 0
    out = LV.openrouter_spend(str(tmp_path))
    assert out["live_usd"] == pytest.approx(2 * 10 * 0.01 * 1.055)
    assert LV.openrouter_spend(str(tmp_path), exclude=[paths.live("cpi")])["live_usd"] == pytest.approx(10 * 0.01 * 1.055)


# ---------------------------------------------------------------------------------------------------------------------------------------
# one process for all three runs
# ---------------------------------------------------------------------------------------------------------------------------------------
def test_run_all_runs_every_engine_combines_progress_and_exit_codes(tmp_path):
    paths = TL.Paths(str(tmp_path))
    for r in TL.RUNS:
        make_source(tmp_path, r, behaviors={14: "auth403"} if r == "agentic" else None)
        init(tmp_path, r, live={"flush_s": 0.2, "control_s": 0.2, "progress_s": 0.2})
    tm = ThreadedMock(Mock())
    try:
        extra = ["--base-url", tm.url, "--global-check-s", "3600"]
        assert TL.split_limit(6, TL.RUNS) == {"cpi": 3, "agentic": 2, "option_xstest_contrast": 1}
        code = TL.run_all(paths, TL.RUNS, limit=6, extra=extra, progress_s=0.2, log=lambda m: None)
        prog = json.load(open(os.path.join(paths.live_root, "progress.json")))
        assert code == 0 and prog["state"] == "done" and prog["keys"]["sent"] == 6 and prog["keys"]["total"] == 30
        assert {r: prog["runs"][r]["keys"]["sent"] for r in TL.RUNS} == {"cpi": 3, "agentic": 2, "option_xstest_contrast": 1}
        assert all(prog["runs"][r]["exit_code"] == 0 for r in TL.RUNS)
        for r in TL.RUNS:
            assert "[run-all]" in open(os.path.join(paths.live(r), "unit.log")).read()
        # the full run: agentic meets an account refusal (key 14) and pauses; the others finish; the unit's exit is PAUSED
        code = TL.run_all(paths, TL.RUNS, extra=extra, progress_s=0.2, log=lambda m: None)
        prog = json.load(open(os.path.join(paths.live_root, "progress.json")))
        assert code == 4 and prog["runs"]["agentic"]["exit_code"] == 4 and prog["runs"]["cpi"]["exit_code"] == 0
        assert readiness(paths.live("cpi"))["ready"] and readiness(paths.live("option_xstest_contrast"))["ready"]
        assert LV.last_run(paths.live("cpi"))["end_state"] == "done"
    finally:
        tm.close()
    # refused before anything starts: a staged run, or the gate open
    pathlib.Path(paths.gate_file).write_text("go\n")
    assert TL.run_all(paths, TL.RUNS, log=lambda m: None) == TL.EXIT_REFUSED
    assert TL.combined_code([0, 3, 0]) == 3 and TL.combined_code([3, 5, 0]) == 5 and TL.combined_code([5, 4]) == 4 and TL.combined_code([0, 1, 4]) == 1


def test_the_init_cli_states_every_ceiling_and_starts_nothing_by_itself(tmp_path, capsys):
    """``init`` takes no default for a run's ceiling, the global ceiling or the root it counts; there is no unit or start command."""
    make_source(tmp_path, "cpi")
    make_source(tmp_path, "option_xstest_contrast")
    base = ["init", "--terra-dir", str(tmp_path), "--runs", "cpi,option_xstest_contrast"]
    full = base + ["--max-spend-usd", "cpi=20,option_xstest_contrast=5", "--global-root", str(tmp_path), "--global-ceiling-usd", "400"]
    for drop in ("--max-spend-usd", "--global-root", "--global-ceiling-usd"):
        i = full.index(drop)
        with pytest.raises(SystemExit) as e:
            TL.main(full[:i] + full[i + 2:])
        assert e.value.code == 2 and drop in capsys.readouterr().err
    with pytest.raises(SystemExit, match="states no ceiling"):
        TL.main(base + ["--max-spend-usd", "cpi=20", "--global-root", str(tmp_path), "--global-ceiling-usd", "400"])
    with pytest.raises(SystemExit, match="not an absolute path"):
        TL.main(base + ["--max-spend-usd", "cpi=20,option_xstest_contrast=5", "--global-root", "relative", "--global-ceiling-usd", "400"])
    assert TL.main(full) == 0
    paths = TL.Paths(str(tmp_path))
    for run, usd in (("cpi", 20.0), ("option_xstest_contrast", 5.0)):
        tr = BF.read_track(paths.live(run))
        assert tr["max_spend_usd"] == usd and tr["live"]["global_ceiling_usd"] == 400.0 and tr["live"]["global_root"] == str(tmp_path)
    for cmd in ("unit-cmd", "start"):
        with pytest.raises(SystemExit):
            TL.main([cmd, "--terra-dir", str(tmp_path)])
    capsys.readouterr()


def test_handovers_are_registered_and_an_unattended_restart_waits_for_full_runs_that_ended_done(tmp_path):
    paths = TL.Paths(str(tmp_path))
    for r in TL.RUNS:
        make_source(tmp_path, r)
        init(tmp_path, r)
    base = ["--terra-dir", str(tmp_path)]
    for r in TL.RUNS:
        assert run_engine(paths.live(r), Mock(), limit=2, gate_file=paths.gate_file)[0] == 0
    assert TL.main(["handover", *base, "--reasks", "--ingest"]) == 0
    assert all("pilot" in LV.auto_start_refusal(paths.live(r), allow_parked=True) for r in TL.RUNS)
    for r in TL.RUNS:
        assert run_engine(paths.live(r), Mock(), gate_file=paths.gate_file)[0] == 0
        assert LV.auto_start_refusal(paths.live(r), allow_parked=True) is None
    assert TL.main(["handover", *base, "--in-flight", "--not-before", "2020-01-01T00:00:00Z", "--ingest"]) == 0
    for r in TL.RUNS:
        assert run_engine(paths.live(r), Mock(), gate_file=paths.gate_file)[0] == 0
    # a staged run takes no handover
    pathlib.Path(paths.live("cpi"), TL.STAGED_FILE).write_text("staged\n")
    assert TL.main(["handover", *base, "--run", "cpi", "--reasks"]) == 0


# ---------------------------------------------------------------------------------------------------------------------------------------
# the filing: each batch run with its live run
# ---------------------------------------------------------------------------------------------------------------------------------------
def filing_world(tmp_path, gate_closed=True):
    for r in ("cpi", "agentic"):
        make_source(tmp_path, r, behaviors={9: "text_then_verdict"})
    paths = TL.Paths(str(tmp_path))
    for r in ("cpi", "agentic"):
        init(tmp_path, r)
        # the gated batch harvester's pause
        pathlib.Path(paths.source(r), "PAUSED").write_text("2026-11-11T22:00:00Z submit gate: the go-ahead file is missing\n")
    return paths


def file_dry(tmp_path, paths, run_dirs, name, *extra):
    staging = tmp_path / name
    argv = ["--judge", MM, "--scope-cpi", str(tmp_path / "scope_cpi"), "--scope-agentic", str(tmp_path / "scope_agentic"),
            "--staging", str(staging), "--code-sha", "0" * 40, "--dry-run", *extra]
    for r in run_dirs:
        argv += ["--run-dir", r]
    code = FR.main(argv)
    rep = json.load(open(staging / "filing_report.json")) if (staging / "filing_report.json").exists() else {}
    rows = []
    for p in staging.rglob("*.jsonl*"):
        with (gzip.open(p, "rt") if str(p).endswith(".gz") else open(p)) as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    return code, rep, rows


def _items(rows):
    out = {}
    for r in rows:
        if r.get("record_type") not in (None, "item_result"):
            continue
        p = r.get("payload") or r
        for it in (p.get("rows") or [p]) if isinstance(p, dict) else []:
            if it.get("metric_id") in ("orb_label", "classifier_harmful", "orb_unscored", "agentic_unscored"):
                out[it["extra"]["request_key"]] = it
    return out


def test_the_filing_merges_each_batch_run_with_its_live_run(tmp_path):
    paths = filing_world(tmp_path)
    rds = [paths.source("cpi"), paths.live("cpi"), paths.source("agentic"), paths.live("agentic")]
    with pytest.raises(SystemExit, match="pass that run directory"):
        file_dry(tmp_path, paths, [paths.live("cpi"), paths.source("agentic")], "s0")          # a live run files with its source
    # the live runs have not run, and the in-flight batch has not come back: the sources are still moving where live does not cover
    code, rep, _ = file_dry(tmp_path, paths, rds, "s1")
    assert code == 2
    for r in ("cpi", "agentic"):
        assert run_engine(paths.live(r), Mock(), gate_file=paths.gate_file)[0] == 0
        LV.build_handover(paths.live(r), ["in_flight", "reasks"], "fallback", log=lambda m: None)
        assert run_engine(paths.live(r), Mock(), gate_file=paths.gate_file)[0] == 0
    code, rep, rows = file_dry(tmp_path, paths, rds, "s2")
    assert code == 0, rep
    assert rep["paused_run_dirs"] == {} and rep["unfinished_run_dirs"] == {}           # the sources' gate pause and pending batches are covered
    assert rep["live_coverage"][paths.source("cpi")]["uncovered_non_terminal"] == 0 and rep["live_coverage"][paths.source("cpi")]["raw_paused"]
    assert rep["judge_backend"] == {"or_bench_cpi": "batch_or_live", A.RUBRIC_NAME: "batch_or_live"}
    assert rep["totals"]["runs_complete"] == 2 and rep["backend_counts"] == {"openrouter_batch": 4, "openrouter_live": 28}
    items = _items(rows)
    k_cpi = {i: text("cpi", i, "verdict")["k"] for i in range(16)}
    assert items[k_cpi[0]]["extra"]["transport"] == "openrouter_batch" and "quantization" not in items[k_cpi[0]]["extra"]
    it = items[k_cpi[6]]["extra"]
    assert it["transport"] == "openrouter_live" and it["provider"] == "GMICloud" and it["quantization"] == "fp8" and it["generation_id"].startswith("gen-")
    # each family's judge rows carry that family's live routing
    judges = {(r.get("payload") or r)["rubric_name"]: (r.get("payload") or r) for r in rows if (r.get("payload") or r).get("judge_backend")}
    assert judges["or_bench_cpi"]["extra"]["protocol"]["live_route"]["provider_routing"] == BF.read_track(paths.live("cpi"))["live"]["routing"]
    assert judges[A.RUBRIC_NAME]["extra"]["protocol"]["live_route"]["provider_routing"] == BF.read_track(paths.live("agentic"))["live"]["routing"]
    assert "parasail/fp8" not in judges[A.RUBRIC_NAME]["extra"]["protocol"]["live_route"]["provider_routing"]["only"]
    batch_only = {r: FR.judge_spec(r, MM)["judge_id"] for r in ("or_bench_cpi", A.RUBRIC_NAME)}
    assert all(rep["judges"][r] != batch_only[r] for r in batch_only)
    # --deliver-only refuses a ledger that changed since the staging
    led = Ledger(paths.live("cpi")).writer()
    led.conn.execute("UPDATE requests SET harvested_utc = '2099-01-01T00:00:00Z' WHERE rowid = 1")
    led.record_outcome("not-a-key", "src:x", None, "verdict", True, {"label": "full_compliance"}, None, 200, None, None, None, "GMICloud")
    led.close()
    with pytest.raises(SystemExit, match="changed since"):
        FR.main(["--staging", str(tmp_path / "s2"), "--registry-root", str(tmp_path / "reg"), "--deliver", "runs", "--deliver-only",
                 "--accept-run-repoint", *[x for rd in rds for x in ("--run-dir", rd)]])
    assert not (tmp_path / "reg").exists()


def test_the_filing_refuses_duplicates_and_resolves_handed_over_keys_by_time(tmp_path):
    paths = filing_world(tmp_path)
    rds = [paths.source("cpi"), paths.live("cpi"), paths.source("agentic"), paths.live("agentic")]
    for r in ("cpi", "agentic"):
        LV.build_handover(paths.live(r), ["in_flight", "reasks"], "fallback", log=lambda m: None)
        assert run_engine(paths.live(r), Mock(), gate_file=paths.gate_file)[0] == 0
    k = {i: text("cpi", i, "verdict")["k"] for i in range(16)}
    src = paths.source("cpi")
    # a late batch result of a handed-over key (key 4, in flight at the handover): the first fileable outcome stands
    led = Ledger(src).writer()
    led.record_outcome(k[4], "b00001", "batch-1", "verdict", True, {"label": "full_refusal"}, None, 200, "tool_calls", None, None, "DeepInfra")
    led.conn.execute("UPDATE requests SET harvested_utc = '2099-01-01T00:00:00Z' WHERE custom_id = ?", (k[4],))
    led.conn.commit()
    led.close()
    code, rep, rows = file_dry(tmp_path, paths, rds, "s1")
    assert code == 0 and rep["merge"]["handover_duplicates_first_kept"] == 1
    it = _items(rows)[k[4]]
    assert it["extra"]["transport"] == "openrouter_live" and it["extra"]["duplicate_outcomes"][0]["backend"] == "batch"
    led = Ledger(src).writer()
    led.conn.execute("UPDATE requests SET harvested_utc = '2000-01-01T00:00:00Z' WHERE custom_id = ?", (k[4],))
    led.conn.commit()
    led.close()
    it = _items(file_dry(tmp_path, paths, rds, "s2")[2])[k[4]]
    assert it["extra"]["transport"] == "openrouter_batch" and it["value"] == FR.LABEL_VALUE["full_refusal"]
    # an infrastructure outcome gives way to a fileable one from the other ledger
    led = Ledger(paths.live("cpi")).writer()
    led.record_outcome(k[0], "src:b00000", None, "provider_error", True, None, {"kind": "transport_timeout"}, None, None, None, None, None)
    led.close()
    st = collections.Counter()
    out = FR.read_outcomes(rds, stats=st)
    assert out[k[0]]["backend"] == "batch" and out[k[0]]["settled_infra"] == [paths.live("cpi")] and st["infra_settled_by_another_ledger"] == 1
    # a key the live run took at its init, judged by the batch run as well: refused
    led = Ledger(src).writer()
    led.record_outcome(k[12], "b00004", "batch-x", "verdict", True, {"label": "full_refusal"}, None, 200, "tool_calls", None, None, "DeepInfra")
    led.close()
    with pytest.raises(SystemExit, match="more than one ledger"):
        file_dry(tmp_path, paths, rds, "s4")


def test_the_filing_refuses_an_off_protocol_live_outcome_until_reopened(tmp_path):
    paths = filing_world(tmp_path)
    rds = [paths.source("cpi"), paths.live("cpi"), paths.source("agentic"), paths.live("agentic")]
    for r in ("cpi", "agentic"):
        LV.build_handover(paths.live(r), ["in_flight", "reasks"], "fallback", log=lambda m: None)
        assert run_engine(paths.live(r), Mock(), gate_file=paths.gate_file)[0] == 0
    tr = BF.read_track(paths.live("agentic"))
    tr["live"]["routing"]["ignore"].append("gmicloud")                         # GMICloud dropped after it served
    BF.write_track(paths.live("agentic"), tr)
    with pytest.raises(SystemExit, match="off_map_provider"):
        file_dry(tmp_path, paths, rds, "s1")
    assert LV.reopen_live(paths.live("agentic"), off_protocol=True, apply=True, log=lambda m: None)["keys"] > 0
    assert run_engine(paths.live("agentic"), Mock(provider="StreamLake"), gate_file=paths.gate_file)[0] == 0
    code, rep, rows = file_dry(tmp_path, paths, rds, "s2")
    assert code == 0 and rep["judge_backend"][A.RUBRIC_NAME] == "batch_or_live"
    assert rep["judges"][A.RUBRIC_NAME] not in (TERRA_LIVE_DEFAULT_AGENTIC_JUDGE_ID, TERRA_LIVE_JUDGE_IDS[A.RUBRIC_NAME])   # another judge


#: the batch-only judge ids of the three rubrics (a filing with no live run directory): never filed; the paper's rows are batch_or_live
BATCH_ONLY_JUDGE_IDS = ("d99d27f5aac5d4f40185ac2eee4625628dfbce8f92908cc380695d500aabc892",
                        "17c2c3df760670ff15377d37e58ed7fdef4681b271c37c5b4e4f2136ad2d3324",
                        "1e33e24c7e629344de7be84516ed35e37c93c7d1575d2f7887c908ff23e83096")


def test_a_batch_only_filing_is_unchanged_by_the_merge(tmp_path):
    """With no live run directory the judge rows are the batch-only ones (pinned here), and the rows say openrouter_batch."""
    assert tuple(FR.judge_spec(r, MM)["judge_id"] for r in ("or_bench_cpi", "xstest_cpi", A.RUBRIC_NAME)) == BATCH_ONLY_JUDGE_IDS
    for r in ("or_bench_cpi", "xstest_cpi", A.RUBRIC_NAME):
        assert FR.judge_spec(r, MM)["judge_backend"] == "batch" and FR.judge_spec(r, MM, live=None) == FR.judge_spec(r, MM)


def test_the_batch_and_live_judge_ids_are_pinned(tmp_path):
    for r in ("cpi", "agentic", "option_xstest_contrast"):
        make_source(tmp_path, r)
        init(tmp_path, r)
    paths = TL.Paths(str(tmp_path))
    lives = FR.live_protocols([paths.source(r) for r in TL.RUNS] + [paths.live(r) for r in TL.RUNS])
    assert lives["cpi"]["with_batch"] and lives["cpi"]["routing"] == SRL.routing()
    ids = {r: FR.judge_spec(r, MM, live=lives[FR.FAMILY_OF_RUBRIC[r]])["judge_id"] for r in ("or_bench_cpi", "xstest_cpi", A.RUBRIC_NAME)}
    # the C/P/I rows were filed under the routing init writes; the agentic routing init writes was never filed
    assert ids == dict(TERRA_LIVE_JUDGE_IDS, **{A.RUBRIC_NAME: TERRA_LIVE_DEFAULT_AGENTIC_JUDGE_ID})
    assert FR.judge_spec("or_bench_cpi", MM, live=lives["cpi"])["judge_backend"] == "batch_or_live"
    # the agentic rows were filed under the routing a person set in the agentic live run's track.json: only DeepInfra allowed
    tr = BF.read_track(paths.live("agentic"))
    tr["live"]["routing"] = filed_agentic_routing(tr["live"]["routing"])
    BF.write_track(paths.live("agentic"), tr)
    assert LV.allowed_providers(tr["live"]["providers"], tr["live"]["routing"]) == {"DeepInfra"}
    lives = FR.live_protocols([paths.source(r) for r in TL.RUNS] + [paths.live(r) for r in TL.RUNS])
    assert {r: FR.judge_spec(r, MM, live=lives[FR.FAMILY_OF_RUBRIC[r]])["judge_id"]
            for r in ("or_bench_cpi", "xstest_cpi", A.RUBRIC_NAME)} == TERRA_LIVE_JUDGE_IDS
    # two C/P/I live runs with different routings would be two protocols under one judge: refused
    tr = BF.read_track(paths.live("option_xstest_contrast"))
    tr["live"]["routing"]["ignore"].append("streamlake/fp8")
    BF.write_track(paths.live("option_xstest_contrast"), tr)
    with pytest.raises(SystemExit, match="different routings"):
        FR.live_protocols([paths.source("cpi"), paths.live("cpi"), paths.live("option_xstest_contrast")])


#: the MiniMax-M3 batch-and-live judge rows of the C/P/I track AS FILED in the registry: the C/P/I rows under the routing of
#: scripts/rejudge/terra/live.py, the agentic row under that routing as :func:`filed_agentic_routing` edits it. A change here is a new judge
TERRA_LIVE_JUDGE_IDS = {"or_bench_cpi": "0a4bfa498117f48293bb1636c37b74dc4fca2595032b2b32941cb83d71689907",
                        "xstest_cpi": "5e79a37817fef85e8b4d0d8b0043a762ecb8b53870354737ed98bee6e675a2c0",
                        A.RUBRIC_NAME: "9fb4893f62c34b15299b2fc60472c9562b54acc81c84da969dd9ee89b79a7f38"}
#: the agentic batch-and-live judge row of the routing ``init`` writes (DeepInfra, GMICloud and StreamLake allowed), never filed: the run
#: paused on it, as GMICloud and then StreamLake answered every agentic reply with a json_schema validation failure
TERRA_LIVE_DEFAULT_AGENTIC_JUDGE_ID = "20760c6cc311bfff715f07773cbb912c5ecfff7ab70a8a59fc3574becec90886"


def filed_agentic_routing(routing):
    """The agentic live run's routing as a person edited its track.json before the run that was filed (2026-11-11): ``only`` narrowed to
    DeepInfra, the GMICloud and StreamLake endpoints appended to ``ignore`` in the order they were dropped. The endpoint map is unchanged
    (a provider dropped from ``only`` stops being allowed without leaving it), so the protocol is the defaults' with this routing."""
    dropped = ["gmicloud/fp8", "streamlake/fp8"]
    assert all(s in routing["only"] for s in dropped + ["deepinfra/fp8"]), routing
    return dict(routing, only=["deepinfra/fp8"], ignore=list(routing["ignore"]) + dropped)


# ---------------------------------------------------------------------------------------------------------------------------------------
# billed reasoning: a key that keeps drawing it is re-routed to the endpoints that have not billed it, parked only when
# none is left; a full run whose only unfinished keys are parked may be started by a handover; a person settles what stays parked
# ---------------------------------------------------------------------------------------------------------------------------------------
def _key_of(payload):
    return int(re.search(r"MOCK:\w+ #(\d+)", " ".join(str(m.get("content")) for m in payload["messages"])).group(1))


def _onlys(mock, i):
    return [p["provider"]["only"] for p in mock.payloads if _key_of(p) == i]


def _first_allowed(order=("GMICloud", "StreamLake", "DeepInfra", "Parasail")):
    """A provider_fn that serves from the first provider of ``order`` the request's routing allows (OpenRouter's choice, made stable)."""
    slugs = {n: TL.ENDPOINTS[n]["slug"] for n in order}
    return lambda k, payload: next(n for n in order if slugs[n] in payload["provider"]["only"])


def test_a_key_that_keeps_drawing_reasoning_is_rerouted_and_parked_only_when_no_endpoint_is_left(tmp_path):
    make_source(tmp_path, "agentic")
    paths, _ = init(tmp_path, "agentic")
    d = paths.live("agentic")
    tr = BF.read_track(d)
    assert tr["live"]["reasoning_reroute"] is True and LV.LIVE_DEFAULTS["reasoning_reroute"] is False     # on for this track only
    full = tr["live"]["routing"]["only"]
    assert full == ["deepinfra/fp8", "gmicloud/fp8", "streamlake/fp8"] and tr["live"]["max_reasoning_resends"] == 3
    # key 6 draws reasoning on GMICloud only; key 7 on every endpoint; the rest never
    reason = lambda k, prov, payload: 50 if (k == 6 and prov == "GMICloud") or k == 7 else 0  # noqa: E731
    mock = Mock(provider_fn=_first_allowed(), reasoning_fn=reason)
    code, runner = run_engine(d, mock, gate_file=paths.gate_file)
    req, att = ledger_rows(d)
    k = {i: text("agentic", i, "verdict")["k"] for i in range(16)}
    assert code == 5 and runner.parked == {k[7]} and set(runner.rerouted) == {k[6]}
    # key 6: four GMICloud replies held for reasoning (the first and its three re-sends), then StreamLake on the narrowed routing
    assert [(a["provider"], a["outcome"], a["violation"]) for a in att[k[6]]] == [("GMICloud", "held", "reasoning_billed")] * 4 + [
        ("StreamLake", "verdict", None)]
    assert _onlys(mock, 6) == [full] * 4 + [["deepinfra/fp8", "streamlake/fp8"]] and req[k[6]]["terminal"] == 1
    # every narrowed request is the run's routing with only ``only`` cut; the body is the batch body
    narrowed = [p for p in mock.payloads if p["provider"]["only"] != full]
    assert narrowed and all({x: v for x, v in p["provider"].items() if x != "only"} == {x: v for x, v in tr["live"]["routing"].items() if x != "only"}
                            for p in narrowed)
    # key 7: GMICloud x4, then StreamLake, then DeepInfra, each billing reasoning: no endpoint left, parked, not terminal
    assert [a["provider"] for a in att[k[7]]] == ["GMICloud"] * 4 + ["StreamLake", "DeepInfra"]
    assert _onlys(mock, 7) == [full] * 4 + [["deepinfra/fp8", "streamlake/fp8"], ["deepinfra/fp8"]] and not req[k[7]]["terminal"]
    con = sqlite3.connect(os.path.join(d, "state.sqlite"))
    ev = collections.Counter(r[0] for r in con.execute("SELECT kind FROM events"))
    con.close()
    assert ev["REROUTED"] == 3 and ev["PARKED"] == 1
    rs = LV.last_run(d)
    assert (rs["end_state"], rs["exit_code"], rs["parked_keys"], rs["non_terminal_keys"], rs["rerouted_keys"]) == ("unfinished", 5, 1, 1, 1)
    prog = json.load(open(os.path.join(d, "progress.json")))
    assert prog["parked_keys"] == 1 and prog["rerouted_keys"] == 1
    # a full run that ended with nothing but parked keys: a person-less start is allowed where the caller allows parked keys
    assert "ended unfinished" in LV.auto_start_refusal(d) and LV.auto_start_refusal(d, allow_parked=True) is None
    # a restart re-sends the parked key once (the run's routing), finds no endpoint left, and parks it again
    m2 = Mock(provider_fn=_first_allowed(), reasoning_fn=reason)
    assert run_engine(d, m2, gate_file=paths.gate_file)[0] == 5 and m2.total == 1 and m2.calls[7] == 1
    # settle-parked: counted read only, then made terminal unjudged by a person's decision
    out = LV.settle_parked(d, reason="synthetic", log=lambda m: None)
    assert (out["keys"], out["reroutable"], out["applied"]) == (1, 0, False) and not readiness(d)["ready"]
    assert LV.settle_parked(d, reason="synthetic", apply=True, log=lambda m: None)["applied"]
    req, _ = ledger_rows(d)
    assert req[k[7]]["terminal"] == 1 and req[k[7]]["outcome"] == "held" and "settled" in json.loads(req[k[7]]["error_json"])["finalized"]
    _spec, cfg, allowed = LV.live_settings(d)
    con = sqlite3.connect(os.path.join(d, "state.sqlite"))
    assert readiness(d)["ready"] and LV.parked(con, cfg, allowed) == {}
    con.close()
    m3 = Mock()
    assert run_engine(d, m3, gate_file=paths.gate_file)[0] == 0 and m3.total == 0 and LV.auto_start_refusal(d) is None
    # the settled key files no row (an infrastructure outcome: the unit stays unjudged)
    assert FR.unit_rows(A.RUBRIC_NAME, "r", "j", {"item_id": "i", "trial": 0, "sample": 0, "turn": -1, "custom_id": k[7], "note": {"gate": 1}},
                        dict(req[k[7]], backend="live"), "t") == []


def test_the_engine_without_reasoning_reroute_parks_at_once_as_before(tmp_path):
    """The StrongREJECT live run's behaviour (reasoning_reroute off, the engine's default) is unchanged: parked after its re-sends."""
    make_source(tmp_path, "agentic")
    paths, _ = init(tmp_path, "agentic", live={"reasoning_reroute": False})
    d = paths.live("agentic")
    mock = Mock(provider_fn=_first_allowed(), reasoning_fn=lambda k, prov, payload: 50 if k == 6 else 0)
    code, runner = run_engine(d, mock, gate_file=paths.gate_file)
    full = BF.read_track(d)["live"]["routing"]["only"]
    assert code == 5 and len(runner.parked) == 1 and not runner.rerouted and _onlys(mock, 6) == [full] * 4
    assert all(p["provider"] == BF.read_track(d)["live"]["routing"] for p in mock.payloads)


def test_an_unattended_restart_is_allowed_after_full_runs_whose_only_unfinished_keys_are_parked(tmp_path):
    paths = TL.Paths(str(tmp_path))
    for r in TL.RUNS:
        make_source(tmp_path, r)
        init(tmp_path, r)
    every = lambda k, prov, payload: 50 if k == 8 else 0  # noqa: E731     # key 8 bills reasoning on every endpoint of every run
    for r in TL.RUNS:
        assert run_engine(paths.live(r), Mock(provider_fn=_first_allowed(), reasoning_fn=every), gate_file=paths.gate_file)[0] == 5
        assert LV.last_run(paths.live(r))["parked_keys"] == 1
    assert all(LV.auto_start_refusal(paths.live(r), allow_parked=True) is None for r in TL.RUNS)
    assert all(LV.auto_start_refusal(paths.live(r)) is not None for r in TL.RUNS)       # without allow_parked: a person's decision
    # an unfinished end with a key that is NOT parked (here: as recorded) is still a person's decision
    led = Ledger(paths.live("cpi")).writer()
    rs = LV.meta_get(led.conn, "run_state")
    rs["non_terminal_keys"] = 2
    LV.meta_set(led.conn, "run_state", rs)
    led.close()
    assert "ended unfinished" in LV.auto_start_refusal(paths.live("cpi"), allow_parked=True)


# ---------------------------------------------------------------------------------------------------------------------------------------
# the refresh keeps a person's edits
# ---------------------------------------------------------------------------------------------------------------------------------------
def _edit_track(d, fn):
    tr = BF.read_track(d)
    fn(tr)
    BF.write_track(d, tr)


def test_a_refresh_carries_a_persons_edits_over_and_restates_the_ceilings(tmp_path):
    make_source(tmp_path, "cpi")
    paths, out = init(tmp_path, "cpi", gate_open=True)
    d = paths.live("cpi")
    dflt = json.load(open(os.path.join(d, TL.DEFAULTS_FILE)))
    assert dflt == BF.read_track(d) and dflt["max_spend_usd"] == SPEND["cpi"] and out["carried_over"] == {}
    _edit_track(d, lambda tr: (tr.update(max_spend_usd=90.0), tr["live"].update(concurrency=7, flush_n=77)))
    # still gated: refused, unless --stage (which keeps the result STAGED and the edits)
    with pytest.raises(SystemExit, match="submit gate"):
        TL.init_run("cpi", paths, refresh=True, live_overrides=FAST, log=lambda m: None, **req("cpi", tmp_path))
    logs = []
    o = TL.init_run("cpi", paths, refresh=True, stage=True, live_overrides=FAST, log=logs.append, **req("cpi", tmp_path))
    tr = BF.read_track(d)
    assert o["staged"] and TL.staged(d) and (tr["live"]["concurrency"], tr["live"]["flush_n"]) == (7, 77)
    assert tr["max_spend_usd"] == SPEND["cpi"]                                  # the ceiling is the one the init states, never an old edit
    assert o["carried_over"] == {"live": {"concurrency": 7, "flush_n": 77}}
    assert any("carried over the edit live.concurrency = 7" in m for m in logs)
    assert json.load(open(os.path.join(d, "init.json")))["carried_over"] == o["carried_over"]
    os.remove(paths.gate_file)
    # --stage with the gate closed: still STAGED, owning nothing; the edits carried again
    o = TL.init_run("cpi", paths, refresh=True, stage=True, live_overrides=FAST, log=lambda m: None, **req("cpi", tmp_path))
    assert o["staged"] and "--stage" in TL.staged(d) and BF.read_track(d)["live"]["concurrency"] == 7
    with pytest.raises(SystemExit, match="STAGED"):
        TL.run_one("cpi", paths, log=lambda m: None)
    # the edits survive a second refresh too (they are measured against the defaults); the stated ceiling is the run's
    o = TL.init_run("cpi", paths, refresh=True, live_overrides=FAST, log=lambda m: None, **req("cpi", tmp_path, max_spend_usd=80.0))
    tr = BF.read_track(d)
    assert not o["staged"] and TL.staged(d) is None and TL._live_activity(d)["keys"] == 10
    assert (tr["max_spend_usd"], tr["live"]["concurrency"]) == (80.0, 7) and o["explicit"] == {"max_spend_usd": 80.0}
    # --discard-edits: the defaults
    o = TL.init_run("cpi", paths, refresh=True, discard_edits=True, live_overrides=FAST, log=lambda m: None, **req("cpi", tmp_path))
    tr = BF.read_track(d)
    assert (tr["max_spend_usd"], tr["live"]["concurrency"], tr["live"]["flush_n"]) == (SPEND["cpi"], FAST["concurrency"], FAST["flush_n"])
    assert o["discarded_edits"]["live"]["concurrency"] == 7 and o["carried_over"] == {}
    assert len(os.listdir(os.path.join(d, "superseded"))) == 4 and TL.verify_run("cpi", paths)["ok"]


def test_a_refresh_refuses_an_edited_routing_that_allows_an_undeclaring_endpoint_and_a_dir_without_defaults(tmp_path):
    make_source(tmp_path, "agentic")
    paths, _ = init(tmp_path, "agentic", gate_open=True)
    d = paths.live("agentic")
    os.remove(paths.gate_file)
    # Parasail fp8 does not declare response_format: an edit that lets it serve the agentic bodies is refused, nothing moved
    _edit_track(d, lambda tr: (tr["live"]["routing"]["only"].append("parasail/fp8"), tr["live"]["routing"]["ignore"].remove("parasail/fp8"),
                               tr["live"]["providers"].update(Parasail={"slug": "parasail/fp8", "quantization": "fp8", "price": [0.3, 1.2, 0.06]})))
    with pytest.raises(SystemExit, match="Parasail"):
        TL.init_run("agentic", paths, refresh=True, live_overrides=FAST, log=lambda m: None, **req("agentic", tmp_path))
    assert os.path.exists(os.path.join(d, "state.sqlite")) and TL.staged(d)
    # a narrowing edit (StreamLake dropped) is carried over
    _edit_track(d, lambda tr: tr.update(live=json.load(open(os.path.join(d, TL.DEFAULTS_FILE)))["live"]))
    _edit_track(d, lambda tr: (tr["live"]["routing"]["only"].remove("streamlake/fp8"), tr["live"]["routing"]["ignore"].append("streamlake/fp8")))
    TL.init_run("agentic", paths, refresh=True, live_overrides=FAST, log=lambda m: None, **req("agentic", tmp_path))
    assert BF.read_track(d)["live"]["routing"]["only"] == ["deepinfra/fp8", "gmicloud/fp8"]
    # a directory an older init wrote (no defaults file): refused, unless --discard-edits
    make_source(tmp_path, "cpi")
    init(tmp_path, "cpi", gate_open=True)
    os.remove(os.path.join(paths.live("cpi"), TL.DEFAULTS_FILE))
    with pytest.raises(SystemExit, match="no track.defaults.json"):
        TL.init_run("cpi", paths, refresh=True, stage=True, live_overrides=FAST, log=lambda m: None, **req("cpi", tmp_path))
    o = TL.init_run("cpi", paths, refresh=True, stage=True, discard_edits=True, live_overrides=FAST, log=lambda m: None, **req("cpi", tmp_path))
    assert "unknown" in o["discarded_edits"] and os.path.exists(os.path.join(paths.live("cpi"), TL.DEFAULTS_FILE))
    # the CLI's per-run values
    assert TL.parse_per_run("cpi=90,agentic=70", list(TL.RUNS)) == {"cpi": 90.0, "agentic": 70.0} and TL.parse_per_run("90", ["cpi"]) == {"cpi": 90.0}
    with pytest.raises(SystemExit, match="names no run"):
        TL.parse_per_run("90", list(TL.RUNS))
    with pytest.raises(SystemExit, match="not one of"):
        TL.parse_per_run("cpi=90", ["agentic"])


def test_a_handed_over_cut_batch_reply_files_as_the_batch_reply_it_is(tmp_path):
    paths = filing_world(tmp_path)
    rds = [paths.source("cpi"), paths.live("cpi"), paths.source("agentic"), paths.live("agentic")]
    k = {i: text("cpi", i, "verdict")["k"] for i in range(16)}
    src = paths.source("cpi")
    # an in-flight cpi batch came back after the gate closed with a reply cut at the cap: the gated harvester never finalizes it
    led = Ledger(src).writer()
    led.record_outcome(k[3], "b00000", "batch-0", "truncated", False, None, {"kind": "truncated"}, 200, "length",
                       {"prompt_tokens": 900, "completion_tokens": 1024}, None, "DeepInfra")
    led.conn.execute("UPDATE requests SET harvested_utc = '2026-11-11T23:00:00Z' WHERE custom_id = ?", (k[3],))
    led.conn.commit()
    led.close()
    m = {}
    for r in ("cpi", "agentic"):
        assert run_engine(paths.live(r), Mock(), gate_file=paths.gate_file)[0] == 0
        LV.build_handover(paths.live(r), ["in_flight", "reasks"], "fallback", log=lambda m_: None)
        m[r] = Mock()
        assert run_engine(paths.live(r), m[r], gate_file=paths.gate_file)[0] == 0
    assert m["cpi"].calls[3] == 0                                                                # never sent: no higher cap for C/P/I
    req, att = ledger_rows(paths.live("cpi"))
    r3, e3 = req[k[3]], json.loads(req[k[3]]["error_json"])
    assert (r3["outcome"], r3["terminal"], r3["batch_id"], r3["provider"], r3["harvested_utc"]) == ("truncated", 1, "batch-0", "DeepInfra",
                                                                                                    "2026-11-11T23:00:00Z")
    assert e3["from_source_run"] and e3["source_attempts"] == 2 and e3["source_batch_id"] == "batch-0" and k[3] not in att
    out = FR.read_outcomes(rds)
    assert (out[k[3]]["backend"], out[k[3]]["attempts"], out[k[3]]["batch_id"]) == ("batch", 2, "batch-0")
    code, rep, rows = file_dry(tmp_path, paths, rds, "s1")
    assert code == 0, rep
    it = _items(rows)[k[3]]
    assert it["metric_id"] == "orb_unscored" and it["label"] == "truncated"
    ex = it["extra"]
    assert (ex["transport"], ex["provider"], ex["batch_id"], ex["attempts"]) == ("openrouter_batch", "DeepInfra", "batch-0", 2)
    assert "quantization" not in ex and "generation_id" not in ex
    # a live outcome of the same run still files as live
    assert _items(rows)[k[6]]["extra"]["transport"] == "openrouter_live"
