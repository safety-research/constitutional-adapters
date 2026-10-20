import pathlib
"""The opt-in vLLM attacker adapter (scripts/jailbreak_run.py --attacker-engine).

CPU only, NO server: a fake ``session`` seam stands in for the vLLM HTTP
endpoint. Covers the invariants the task fixed:

- a selected-but-unreachable vLLM endpoint raises LOUDLY and never silently
  falls back to HF (a silent fallback would mislabel an HF run as a vLLM run);
- the attack's pinned sampling params (temperature/top_p/max_new_tokens) are
  threaded into every request, and the scorer is greedy + short;
- the callable signatures match the HF seams (attacker_fn / scorer_fn /
  attacker_factory), so PyRIT and the hand-rolled multiturn loop are unchanged;
- ``--attacker-engine`` DEFAULTS to ``hf`` -- nothing changes unless passed;
- the engine is recorded in a per-run manifest so transcripts self-identify.
"""

import argparse
import json
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import jailbreak_run  # noqa: E402
from sisteer.eval.attacks.vllm_attacker import (  # noqa: E402
    VLLMAttackerClient,
    VLLMEndpointUnreachable,
    make_attacker_factory_vllm,
    make_attacker_fn_vllm,
    make_scorer_fn_vllm,
)


# --------------------------------------------------------------------------- #
# Fake HTTP transport (a requests-like Session)                                 #
# --------------------------------------------------------------------------- #


class _FakeResp:
    def __init__(self, status=200, json_data=None):
        self.status_code = status
        self._json = json_data or {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


class _RecordingSession:
    """Answers /health with ``health_status`` and every chat with ``content``,
    recording each POST body so a test can assert the params sent."""

    def __init__(self, *, health_status=200, content="reply"):
        self.health_status = health_status
        self.content = content
        self.posts = []
        self.gets = []

    def get(self, url, timeout=None):
        self.gets.append({"url": url, "timeout": timeout})
        return _FakeResp(status=self.health_status)

    def post(self, url, json=None, timeout=None):
        self.posts.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResp(
            status=200,
            json_data={"choices": [{"message": {"content": self.content}}]},
        )


class _DeadSession:
    """Every call raises ConnectionError, like a server that isn't up."""

    def get(self, url, timeout=None):
        raise requests.exceptions.ConnectionError("connection refused")

    def post(self, url, json=None, timeout=None):
        raise requests.exceptions.ConnectionError("connection refused")


# --------------------------------------------------------------------------- #
# Fail loud, never fall back to HF                                              #
# --------------------------------------------------------------------------- #


def test_health_unreachable_raises_loudly():
    client = VLLMAttackerClient("http://127.0.0.1:8000", "m", session=_DeadSession())
    with pytest.raises(VLLMEndpointUnreachable) as ei:
        client.check_health()
    msg = str(ei.value)
    assert "http://127.0.0.1:8000" in msg
    assert "vllm_attacker_serve.sh" in msg  # actionable remediation
    assert "fall" in msg.lower()  # says it is NOT falling back


def test_health_non_2xx_raises():
    client = VLLMAttackerClient("http://x", "m", session=_RecordingSession(health_status=503))
    with pytest.raises(VLLMEndpointUnreachable):
        client.check_health()


def test_health_2xx_passes():
    client = VLLMAttackerClient("http://x", "m", session=_RecordingSession(health_status=200))
    client.check_health()  # no raise


def test_chat_server_error_raises_no_fallback():
    class _Sess500:
        def get(self, url, timeout=None):
            return _FakeResp(200)

        def post(self, url, json=None, timeout=None):
            return _FakeResp(status=503)

    client = VLLMAttackerClient("http://x", "m", session=_Sess500(), max_retries=0)
    with pytest.raises(VLLMEndpointUnreachable):
        client.chat([{"role": "user", "content": "x"}],
                    temperature=1.0, top_p=0.9, max_new_tokens=16)


def test_build_seams_vllm_unreachable_raises():
    """The whole --attacker-engine vllm wiring fails at seam-build time when the
    endpoint is down (real connection-refused, no server)."""
    cfg = {"attacker": {"model": "the-attacker"}, "attacks": {}}
    args = argparse.Namespace(
        attack="actorattack", attacker_engine="vllm",
        attacker_endpoint="http://127.0.0.1:1",  # port 1 -> refused instantly
    )
    with pytest.raises(VLLMEndpointUnreachable):
        jailbreak_run._build_attacker_seams(cfg, args)


# --------------------------------------------------------------------------- #
# Sampling params threaded through; signatures match the HF seams               #
# --------------------------------------------------------------------------- #


def test_attacker_fn_threads_sampling_and_signature():
    sess = _RecordingSession(content="ADVERSARIAL")
    client = VLLMAttackerClient("http://host:8000", "the-model", session=sess)
    fn = make_attacker_fn_vllm(client, temperature=0.7, top_p=0.55, max_new_tokens=321)
    chat = [{"role": "user", "content": "produce actors"}]
    out = fn(chat)  # same call shape as _make_attacker_fn's attacker_fn(chat)
    assert out == "ADVERSARIAL"
    body = sess.posts[-1]["json"]
    assert sess.posts[-1]["url"].endswith("/v1/chat/completions")
    assert body["model"] == "the-model"
    assert body["messages"] == chat
    assert body["temperature"] == 0.7
    assert body["top_p"] == 0.55
    assert body["max_tokens"] == 321
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_scorer_fn_is_greedy_short_and_signature():
    sess = _RecordingSession(content="7")
    client = VLLMAttackerClient("http://x", "m", session=sess)
    fn = make_scorer_fn_vllm(client, max_new_tokens=8)
    out = fn("rate this response 1-10")  # scorer_fn(prompt:str) -> str
    assert out == "7"
    body = sess.posts[-1]["json"]
    assert body["temperature"] == 0.0  # greedy, matches HF do_sample=False
    assert body["max_tokens"] == 8
    assert body["messages"] == [{"role": "user", "content": "rate this response 1-10"}]


def test_factory_builds_seamed_attacker_target(tmp_path):
    import asyncio

    from sisteer.eval.attacks.pyrit_adapters import init_pyrit_memory_async

    sess = _RecordingSession(content="ATTACKER TURN")

    async def go():
        # AttackerChatTarget is a PyRIT PromptTarget; construction needs central
        # memory set (in production the factory runs AFTER PyRIT memory init).
        await init_pyrit_memory_async(tmp_path / "m.db")
        client = VLLMAttackerClient("http://x", "m", session=sess)
        factory = make_attacker_factory_vllm(
            client, temperature=1.0, top_p=0.9, max_new_tokens=1024)
        target = factory(None)  # attacker_factory(gpu_lock) -> AttackerChatTarget
        # PyRIT target routes through the generate_fn seam -> the vLLM client.
        return target._generate([{"role": "user", "content": "q"}])

    assert asyncio.run(go()) == "ATTACKER TURN"
    body = sess.posts[-1]["json"]
    assert body["temperature"] == 1.0
    assert body["top_p"] == 0.9
    assert body["max_tokens"] == 1024


def test_chat_retries_then_succeeds():
    """A transient failure is retried on the SAME endpoint (not a fallback)."""

    class _FlakySession:
        def __init__(self):
            self.calls = 0

        def get(self, url, timeout=None):
            return _FakeResp(200)

        def post(self, url, json=None, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise requests.exceptions.ConnectionError("blip")
            return _FakeResp(
                status=200,
                json_data={"choices": [{"message": {"content": "ok"}}]},
            )

    sess = _FlakySession()
    client = VLLMAttackerClient("http://x", "m", session=sess, max_retries=2, backoff=0.0)
    out = client.chat([{"role": "user", "content": "x"}],
                      temperature=1.0, top_p=0.9, max_new_tokens=16)
    assert out == "ok"
    assert sess.calls == 2


# --------------------------------------------------------------------------- #
# Default stays hf; engine recorded in the manifest                             #
# --------------------------------------------------------------------------- #


def test_attacker_engine_flag_defaults_to_vllm():
    """The default is vLLM. Two consequences pinned here:

    1. The attacker needs no steering hooks, so vanilla vLLM remains the
       serving default; the retired cluster's old speed/cost estimates are not
       evidence for this test.
    2. vLLM and HF cannot produce identical attacker text, so the DEFAULT run is
       not comparable to any pre-cutover HF run. Accepted going forward; the
       engine is recorded per run so no transcript is ambiguous.

    Asserted through argparse rather than by grepping the source, so a
    reformat cannot silently break the check.
    """
    p = argparse.ArgumentParser()
    jailbreak_run._add_attacker_engine_args(p)
    ns = p.parse_args([])
    assert ns.attacker_engine == "vllm", "default engine must be vllm"
    assert ns.attacker_endpoint  # has a sane default
    ns2 = p.parse_args(["--attacker-engine", "hf"])
    assert ns2.attacker_engine == "hf", "hf must remain selectable"

    # With vLLM as the default, "no server running" is the common first
    # failure, so the error must name both remedies and say why it will not
    # fall back on its own.
    from sisteer.eval.attacks.vllm_attacker import VLLMEndpointUnreachable
    msg = str(VLLMEndpointUnreachable.for_endpoint("http://x:8000"))
    assert "vllm_attacker_serve.sh" in msg
    assert "--attacker-engine hf" in msg
    assert "DIFFERENT attack prompts" in msg

def test_engine_manifest_records_engine(tmp_path):
    cfg = {"attacker": {"model": "the-attacker", "snapshot": "deadbeef"}}
    # hf (default) manifest
    args_hf = argparse.Namespace(attacker_engine="hf", attacker_endpoint="http://x")
    jailbreak_run._write_engine_manifest(tmp_path / "hf", cfg, args_hf, "actorattack")
    m_hf = json.loads((tmp_path / "hf" / "attacker_engine.json").read_text())
    assert m_hf["attacker_engine"] == "hf"
    assert m_hf["attacker_endpoint"] is None
    assert m_hf["attacker_model"] == "the-attacker"
    assert m_hf["comparable_across_engines"] is False
    # vllm manifest carries the endpoint
    args_vllm = argparse.Namespace(attacker_engine="vllm",
                                   attacker_endpoint="http://gpu:8000")
    jailbreak_run._write_engine_manifest(tmp_path / "v", cfg, args_vllm, "pair")
    m_v = json.loads((tmp_path / "v" / "attacker_engine.json").read_text())
    assert m_v["attacker_engine"] == "vllm"
    assert m_v["attacker_endpoint"] == "http://gpu:8000"
    assert m_v["attack"] == "pair"


def test_build_seams_hf_does_not_touch_endpoint(monkeypatch):
    """engine=hf builds the HF seams via _load_attacker and never constructs a
    vLLM client / hits an endpoint."""
    sentinel = object()

    def fake_load_attacker(cfg):
        return sentinel, sentinel

    monkeypatch.setattr(jailbreak_run, "_load_attacker", fake_load_attacker)
    made = {}
    monkeypatch.setattr(jailbreak_run, "_make_scorer_fn",
                        lambda *a, **k: made.setdefault("scorer", object()))
    monkeypatch.setattr(jailbreak_run, "_make_attacker_factory",
                        lambda *a, **k: made.setdefault("factory", object()))
    monkeypatch.setattr(jailbreak_run, "_make_attacker_fn",
                        lambda *a, **k: made.setdefault("fn", object()))

    cfg = {"attacker": {"model": "m"}, "attacks": {}}
    args = argparse.Namespace(attack="actorattack", attacker_engine="hf",
                              attacker_endpoint="http://should-not-be-touched:9999")
    scorer_fn, factory, fn = jailbreak_run._build_attacker_seams(cfg, args)
    assert scorer_fn is made["scorer"]
    assert factory is made["factory"]
    assert fn is made["fn"]
