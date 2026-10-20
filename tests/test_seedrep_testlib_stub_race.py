"""The shared StubProxy stamps each request's completion id on the HANDLER-LOCAL dict, under a lock (2026-09-20).

The landing gate hit `KeyError: 'completion_id'` in `test_seedrep_sol_r2_fixes.py` on a Modal shard under CPU
contention: `StubProxy.do_POST` appended to the shared `bodies` list, ran `decide()`, THEN stamped `bodies[-1]` -- with two
handlers in flight (the consumer sends at width 2) the later one stamped the other request's dict and its own never got an
id. This test drives two concurrent POSTs through a slow `decide` and requires both recorded bodies to carry distinct ids."""
from __future__ import annotations

import json
import threading
import time
import urllib.request

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _seedrep_testlib as T  # noqa: E402


def _post(url: str, i: int) -> dict:
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": f"q{i}"}]}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())


def test_two_concurrent_posts_each_get_their_own_completion_id():
    proxy = T.StubProxy()
    slow = proxy.decide

    def decide(body):
        time.sleep(0.2)
        return slow(body)
    proxy.decide = decide
    try:
        out: dict = {}
        ts = [threading.Thread(target=lambda i=i: out.__setitem__(i, _post(f"{proxy.url}/chat/completions", i))) for i in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=15)
    finally:
        proxy.close()
    assert len(proxy.bodies) == 2
    ids = [b.get("completion_id") for b in proxy.bodies]
    assert all(ids), f"a recorded body has no completion_id: {ids}"
    assert len(set(ids)) == 2
    assert {out[0]["id"], out[1]["id"]} == set(ids)
