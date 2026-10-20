"""The attacker client's FAILOVER (31d): when a request exhausts its transport retries against the current pool member (connection
refused, timeout, 5xx — a member that retired or died), the client rotates to the next member that answers /health and retries there,
recording the rotation and the member each request hit; a 401/403 or a prompt-over-window never rotates; with no survivor it raises by
name, never falls back to HF. The driver hands the member list in through --attacker-endpoints (the bound member first)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from sisteer.eval.attacks.vllm_attacker import VLLMAttackerClient, VLLMEndpointUnauthorized, VLLMEndpointUnreachable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402


class Resp:
    def __init__(self, status, body=None, text=""):
        self.status_code = status; self._body = body; self.text = text
    def json(self):
        return self._body
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """Per-endpoint behaviour: `down` endpoints raise on every call; `unhealthy` answer 503 to /health; others answer."""
    def __init__(self, *, down=(), unhealthy=(), unauthorized=()):
        self.down = {d.rstrip("/") for d in down}; self.unhealthy = {u.rstrip("/") for u in unhealthy}; self.unauthorized = {u.rstrip("/") for u in unauthorized}
        self.posts: list = []; self.gets: list = []; self.headers = {}
    def _root(self, url):
        return url.rsplit("/v1/", 1)[0] if "/v1/" in url else url.rsplit("/health", 1)[0]
    def get(self, url, timeout=None):
        self.gets.append(url); root = self._root(url)
        if root in self.down:
            raise ConnectionError(f"refused {root}")
        return Resp(503 if root in self.unhealthy else 200)
    def post(self, url, json=None, timeout=None):
        self.posts.append(url); root = self._root(url)
        if root in self.down:
            raise ConnectionError(f"refused {root}")
        if root in self.unauthorized:
            return Resp(401, text="unauthorized")
        return Resp(200, {"choices": [{"message": {"content": f"reply from {root}"}}]})


A, B, C = "http://gpu-0:8707", "http://gpu-13:8709", "http://gpu-4:8710"


def test_a_dead_member_rotates_to_the_next_healthy_one_and_the_request_records_the_member_it_hit():
    fs = FakeSession(down=[A], unhealthy=[B])
    c = VLLMAttackerClient(A, "m", session=fs, max_retries=1, backoff=0, failover=[A, B, C])
    assert c.failover == [B, C]                                                                 # the bound member is not its own failover
    out = c.chat([{"role": "user", "content": "x"}], temperature=1.0, top_p=0.9, max_new_tokens=8)
    assert out == f"reply from {C}" and c.endpoint == C and c.n_failovers == 1
    assert sum(1 for u in fs.posts if u.startswith(A)) == 2 and not any(u.startswith(B) for u in fs.posts)   # retries on A, B skipped (unhealthy)
    assert c.last_meta["endpoint"] == C and c.last_meta["failover"]["from"] == A and c.last_meta["failover"]["to"] == C and "ConnectionError" in c.last_meta["failover"]["reason"]
    # the next request goes straight to C, no further rotation
    c.chat([{"role": "user", "content": "y"}], temperature=1.0, top_p=0.9, max_new_tokens=8)
    assert c.n_failovers == 1 and fs.posts[-1].startswith(C)


def test_no_surviving_member_raises_by_name_and_a_401_never_rotates():
    fs = FakeSession(down=[A, C], unhealthy=[B])
    c = VLLMAttackerClient(A, "m", session=fs, max_retries=0, backoff=0, failover=[B, C])
    with pytest.raises(VLLMEndpointUnreachable, match="No surviving pool member answered either"):
        c.chat([{"role": "user", "content": "x"}], temperature=1.0, top_p=0.9, max_new_tokens=8)
    fs2 = FakeSession(unauthorized=[A])
    c2 = VLLMAttackerClient(A, "m", session=fs2, max_retries=0, backoff=0, failover=[B])
    with pytest.raises(VLLMEndpointUnauthorized):
        c2.chat([{"role": "user", "content": "x"}], temperature=1.0, top_p=0.9, max_new_tokens=8)
    assert c2.endpoint == A and c2.n_failovers == 0
    # a client without a failover list behaves as before: the plain refusal
    c3 = VLLMAttackerClient(A, "m", session=FakeSession(down=[A]), max_retries=0, backoff=0)
    with pytest.raises(VLLMEndpointUnreachable, match="NOT falling back to HF") as ei:
        c3.chat([{"role": "user", "content": "x"}], temperature=1.0, top_p=0.9, max_new_tokens=8)
    assert "surviving" not in str(ei.value)


def test_the_startup_health_check_rotates_too_when_the_bound_member_is_already_gone():
    fs = FakeSession(down=[A])
    c = VLLMAttackerClient(A, "m", session=fs, max_retries=0, backoff=0, failover=[B])
    c.check_health()
    assert c.endpoint == B and c.failover_log[0]["from"] == A
    with pytest.raises(VLLMEndpointUnreachable):
        VLLMAttackerClient(A, "m", session=FakeSession(down=[A, B]), max_retries=0, backoff=0, failover=[B]).check_health()


def test_the_driver_takes_the_member_list_and_the_target_job_hands_it_over():
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "jailbreak_run.py"), "generate", "--help"], capture_output=True, text=True, timeout=300, env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0 and "--attacker-endpoints" in r.stdout and "FAILOVER" in r.stdout, r.stdout[-800:] + r.stderr[-800:]
    sb = (REPO / "scripts" / "sdfbase" / "full22" / "target_attacks.sbatch").read_text()
    assert '--attacker-endpoints "$url$(' in sb and 'grep -vxF "$url"' in sb                      # the bound member first, the others behind it
    assert subprocess.run(["bash", "-n", str(REPO / "scripts" / "sdfbase" / "full22" / "target_attacks.sbatch")], capture_output=True).returncode == 0
