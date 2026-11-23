"""The request bytes of the paper's judges, rebuilt from scripts/rejudge/ and compared with fixed goldens
(tests/fixtures/rejudge/request_goldens.json): for one synthetic benign input per rubric, the whole batch body (max_tokens included), the
custom id, the serialized batch line, the live payload (``{"model", **body, "provider": routing}``, as ``common.live`` sends it) and
each judge's entry and routing. A change to any of them names another request, and for the paper's rows another judge id. Offline."""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (ROOT / "scripts", ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from rejudge.common import batchfile as BF  # noqa: E402
from rejudge.common import live as LV  # noqa: E402
from rejudge.petri import file_records as PFR  # noqa: E402
from rejudge.petri import judges as PJ  # noqa: E402
from rejudge.petri import request as PR  # noqa: E402
from rejudge.strongreject import judges as SJ  # noqa: E402
from rejudge.strongreject import live as SRL  # noqa: E402
from rejudge.strongreject import request as SR  # noqa: E402
from rejudge.terra import cpi as C  # noqa: E402
from rejudge.terra import judges as TJ  # noqa: E402
from rejudge.terra import live as TL  # noqa: E402

G = json.loads((ROOT / "tests" / "fixtures" / "rejudge" / "request_goldens.json").read_text(encoding="utf-8"))


def _live_payload(model, body, routing):
    """What the live engine POSTs (``common.live.LiveRunner._send``)."""
    payload = dict({"model": model}, **body)
    payload["provider"] = routing
    return LV.compact(payload).decode("ascii")


def test_the_strongreject_entry_and_route_are_the_goldens():
    e = SJ.get("minimax-m3")
    want = G["strongreject"]["entry"]
    assert {k: e[k] for k in want} == want
    assert SJ.get() == e
    live = G["strongreject"]["live"]
    assert SRL.PROVIDERS == live["providers"] and SRL.IGNORE == live["ignore"] and SRL.routing() == live["routing"]


@pytest.mark.parametrize("kind", ["sr", "sr2"])
def test_strongreject_requests_are_the_goldens(kind):
    g = G["strongreject"][kind]
    e = SJ.get("minimax-m3")
    body = SR.request_body(e["request"], kind, g["goal"], g["response"])
    assert body == g["body"]
    cid = SR.cache_key(kind, g["dataset"], g["revision"], g["goal"], g["response"])
    assert cid == g["custom_id"]
    assert BF.serialize_request(cid, body).decode("ascii") == g["line"]
    assert _live_payload(e["model"], body, SRL.routing()) == g["live_payload"]
    entry = SR.build_entry(kind, g["dataset"], g["revision"], g["goal"], g["response"], e["request"])
    assert entry["custom_id"] == cid and entry["body"] == body


def test_the_cpi_settings_and_route_are_the_goldens():
    assert TJ.request_settings("minimax-m3", "cpi") == G["cpi"]["request_settings"]
    providers, routing, _ = TL.routing_for(TL.expected_parameters({"judge": "minimax-m3", "family": "cpi"}))
    assert {"providers": providers, "routing": routing} == G["cpi"]["live"]


@pytest.mark.parametrize("rubric", ["or_bench_cpi", "xstest_cpi"])
def test_cpi_requests_are_the_goldens(rubric):
    g = G["cpi"][rubric]
    key, _ = C.original_request(rubric, g["prompt"], g["response"])
    assert key == g["custom_id"]
    ent = TJ.build_entry("cpi", {"rubric": rubric, "prompt": g["prompt"], "response": g["response"], "k": key},
                         TJ.request_settings("minimax-m3", "cpi"))
    assert ent["body"] == g["body"]
    assert BF.serialize_request(key, ent["body"]).decode("ascii") == g["line"]
    _, routing, _ = TL.routing_for(TL.expected_parameters({"judge": "minimax-m3", "family": "cpi"}))
    assert _live_payload(TJ.get("minimax-m3")["openrouter_model"], ent["body"], routing) == g["live_payload"]


def test_the_agentic_request_is_the_golden():
    """Kept for the shared C/P/I code paths (the paper reports no agentic eval)."""
    g = G["agentic"]
    rq = TJ.request_settings("minimax-m3", "agentic")
    assert rq == g["request_settings"]
    assert TJ.build_entry("agentic", {"k": g["custom_id"], "prompt": g["prompt"]}, rq)["body"] == g["body"]
    providers, routing, _ = TL.routing_for(TL.expected_parameters({"judge": "minimax-m3", "family": "agentic"}))
    assert {"providers": providers, "routing": routing} == g["live"]


def test_the_petri_request_is_the_golden():
    g = G["petri"]
    prof = PJ.profile("glm-5.3")
    body = PJ.arm_body(prof, g["prompt"], g["tool"])
    assert body == g["body"]
    assert PR.judge_input_key(g["prompt"], g["tool"]) == g["custom_id"] and PR.tool_sha(g["tool"]) == g["tool_sha"]
    assert BF.serialize_request(g["custom_id"], body).decode("ascii") == g["line"]
    assert PJ.arm_request(prof) == g["request"] and PJ.arm_shape(prof) == g["shape"] and prof["provider"] == g["provider"]
    assert PJ.candidate(prof)["truncated_retry_max_tokens"] == g["truncated_retry_max_tokens"]
    proto = json.dumps(PFR.arm_protocol(prof), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    assert BF.sha256_bytes(proto) == g["protocol_sha256"] == "b6ee14ea0a88884f17c50ff4ea0847a15a6cc48c9c1b3ed88a01321c367dd65c"
