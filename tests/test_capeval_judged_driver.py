"""capeval_judged_run.py driver wiring tests (capeval-controls §3.4).

Stubbed engine + judge — no GPU, no network, no real Claude, synthetic
rows only. Covers the generate→judge handoff through the cell manifest
(spool binding, base-cache provenance, judge-phase spec rebuilt from the
manifest), the coordination post printing BEFORE judging, and
results.json as the cell-done marker.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from sisteer.capeval import judging
from sisteer.capeval.serving import GenOut

ROOT = Path(__file__).parents[1]
# bound at import, BEFORE the driver fixture swaps judging.ClaudeJudge for
# StubJudge — the base-url tests need the real constructor as a reference
_RealClaudeJudge = judging.ClaudeJudge
REV = "0123456789abcdef0123456789abcdef01234567"


def _driver():
    spec = importlib.util.spec_from_file_location(
        "capeval_judged_run", ROOT / "scripts" / "capeval_judged_run.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["capeval_judged_run"] = mod
    spec.loader.exec_module(mod)
    return mod


class _Tok:
    chat_template = None


class StubEngine:
    name = "hf"
    tokenizer = _Tok()

    def __init__(self, *a, **k):
        pass

    def load(self):
        pass

    def generate(self, prompts, arm, gen_cfg):
        return [GenOut(text=f"stub generation for::{p.strip()}",
                       gen_token_logprobs=[], ended_with_eos=True)
                for p in prompts]


class StubJudge:
    temperature = judging.TEMPERATURE
    max_tokens = judging.MAX_TOKENS
    model = judging.JUDGE_MODEL

    def __init__(self, *, cache=None, concurrency=200, **kw):
        self.cache = cache
        self.concurrency = concurrency
        self.n_cache_hits = 0
        self.n_api_calls = 0
        self.n_calls_by_org = {"lowprio": 0}

    def judge_all(self, pairs, *, allow_refusal_abstention=False):
        out = {}
        for req, rubric in pairs:
            self.n_api_calls += 1
            if rubric.paired:
                out[req.key()] = {"verdict": "tie"}
            else:
                out[req.key()] = {"label": "full_compliance"}
        return out


@pytest.fixture()
def driver(monkeypatch, tmp_path):
    mod = _driver()
    monkeypatch.setattr(mod, "_model_pin",
                        lambda: ("test-org/tiny", REV))
    import sisteer.capeval.serving.hf_engine as he

    monkeypatch.setattr(he, "HFEngine", StubEngine)
    monkeypatch.setattr(mod.judging, "ClaudeJudge", StubJudge)

    rows = [{"item_id": f"orb{i:04d}-{'ab' * 4}",
             "prompt": f"synthetic driver prompt {i}", "category": "c"}
            for i in range(4)]
    monkeypatch.setitem(
        mod.TASKS, "or_bench",
        (mod.judged.TASK_ORBENCH, "judgedorb", lambda: list(rows),
         judging.ORBENCH_RUBRIC_V1))
    return mod


def _common(tmp_path, task):
    return ["--task", task, "--sweep", "t", "--out-root", str(tmp_path)]


def test_generate_then_judge_or_bench(driver, tmp_path, capsys,
                                      monkeypatch):
    monkeypatch.setenv(judging.CACHE_ENV_VAR,
                       str(tmp_path / "cache.sqlite"))
    assert driver.main(["generate", *_common(tmp_path, "or_bench"),
                        "--base"]) == 0
    cells = list((tmp_path / "t" / "cells").iterdir())
    assert len(cells) == 1
    manifest = json.loads((cells[0] / "manifest.json").read_text())
    assert manifest["task"] == driver.judged.TASK_ORBENCH
    assert manifest["rubric"]["sha"] == judging.ORBENCH_RUBRIC_V1.sha
    spool_path = Path(manifest["spool"]["path"])
    assert not spool_path.is_absolute()  # judge runs on ANOTHER machine
    assert (tmp_path / "t" / spool_path).exists()
    assert not (cells[0] / "results.json").exists()  # not done yet

    monkeypatch.setenv("SISTEER_COORDINATION_CHANNEL", "#example-api-coordination")
    assert driver.main(["judge", *_common(tmp_path, "or_bench"),
                        "--offline-bounds", "--wandb-mode", "disabled"]) == 0
    out = capsys.readouterr().out
    # the coordination post prints BEFORE judging, to the site's channel, channel format first
    assert "[capeval-judged] #example-api-coordination post" in out
    assert "200 sonnet-5 lowprio" in out
    assert "cache_hits/api_calls" in out
    results = json.loads((cells[0] / "results.json").read_text())
    assert results["n"] == 4
    assert results["fracs"]["full_compliance"] == 1.0
    manifest2 = json.loads((cells[0] / "manifest.json").read_text())
    assert manifest2["judge_phase"]["bounds"]["n_unique_requests"] == 4
    assert "coordination_post" in manifest2["judge_phase"]


def test_judge_refuses_item_drift(driver, tmp_path, monkeypatch):
    monkeypatch.setenv(judging.CACHE_ENV_VAR,
                       str(tmp_path / "cache.sqlite"))
    assert driver.main(["generate", *_common(tmp_path, "or_bench"),
                        "--base"]) == 0
    # judging under a different --limit resolves different items -> refusal
    with pytest.raises(SystemExit, match="items_sha"):
        driver.main(["judge", *_common(tmp_path, "or_bench"), "--limit", "2",
                     "--offline-bounds", "--wandb-mode", "disabled"])


def test_judge_model_cross_check_refuses_mismatch(driver, tmp_path,
                                                  monkeypatch):
    monkeypatch.setenv(judging.CACHE_ENV_VAR,
                       str(tmp_path / "cache.sqlite"))
    assert driver.main(["generate", *_common(tmp_path, "or_bench"),
                        "--base"]) == 0  # archives claude-sonnet-5
    with pytest.raises(SystemExit, match="judge model"):
        driver.main(["judge", *_common(tmp_path, "or_bench"),
                     "--judge-model", "claude-haiku-4-5",
                     "--offline-bounds", "--wandb-mode", "disabled"])
    # naming the archived model is fine (a pure cross-check)
    assert driver.main(["judge", *_common(tmp_path, "or_bench"),
                        "--judge-model", judging.JUDGE_MODEL,
                        "--offline-bounds", "--wandb-mode", "disabled"]) == 0


# --------------------------------------------------------------------------- #
# The count-tokens client's per-request timeout                                 #
# --------------------------------------------------------------------------- #


def _stub_anthropic(monkeypatch):
    """Replace the anthropic module the driver imports lazily with a fake whose
    client records its constructor kwargs and answers count_tokens, and stub
    the pool-key lookup so no credential is resolved."""
    import types as _types

    seen = []

    class _Counts:
        def count_tokens(self, **kw):
            return _types.SimpleNamespace(input_tokens=7)

    class _FakeClient:
        def __init__(self, **kw):
            seen.append(dict(kw))
            self.messages = _Counts()

    fake = _types.ModuleType("anthropic")
    fake.Anthropic = _FakeClient
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    # the count-tokens client is the vault's (op mode here: the stubbed key, never the secret-manager CLI); _pool_key stays stubbed
    # for the openrouter pool
    monkeypatch.setattr(judging, "_pool_key", lambda org: "k")
    monkeypatch.delenv(judging.vault.SECRETS_MODE_ENV, raising=False)
    monkeypatch.setattr(judging.vault, "anthropic_api_key", lambda org, **kw: "k")
    return seen


def test_count_tokens_client_helper_sets_an_explicit_request_timeout(
        driver, monkeypatch):
    """The judge phase's count-tokens client was the one client in this
    driver built outside ``capeval.judging`` — ``max_retries=1`` and nothing
    else, so it inherited the SDK's 600 s default. A count-tokens call is
    cheaper than a judge call (no generation), and the judge call answers in
    ~1.8 s; MEASURED 2026-09-11, two sockets ESTAB with empty send/recv queues
    stalled a 5,953-item run behind that default and the watchdog throttled a
    healthy pool. The bound must reach the constructor, and the retry count
    must be untouched — bounds are a courtesy estimate, not a retry loop."""
    seen = _stub_anthropic(monkeypatch)
    driver._count_tokens_client()
    assert len(seen) == 1
    kw = seen[0]
    assert "timeout" in kw, (
        "count-tokens client built with no explicit timeout — it inherits the "
        "SDK's 600 s default and a hung socket stalls the bounds step")
    assert kw["timeout"] == driver.COUNT_TOKENS_TIMEOUT_S
    assert 0 < float(kw["timeout"]) <= 300
    assert kw["max_retries"] == 1


def test_judge_phase_count_tokens_client_is_bounded_end_to_end(
        driver, tmp_path, monkeypatch):
    """Through ``main()``: the judge phase WITHOUT ``--offline-bounds`` builds
    the count-tokens client, and that client carries the timeout. The
    manifest's recorded bound method proves the stubbed client was the one
    actually used (an unusable client silently falls back to the byte bound,
    which would let this test pass on a client that was never exercised)."""
    monkeypatch.setenv(judging.CACHE_ENV_VAR,
                       str(tmp_path / "cache.sqlite"))
    seen = _stub_anthropic(monkeypatch)
    assert driver.main(["generate", *_common(tmp_path, "or_bench"),
                        "--base"]) == 0
    assert driver.main(["judge", *_common(tmp_path, "or_bench"),
                        "--wandb-mode", "disabled"]) == 0
    assert len(seen) == 1
    assert "timeout" in seen[0]
    assert 0 < float(seen[0]["timeout"]) <= 300
    cells = list((tmp_path / "t" / "cells").iterdir())
    manifest = json.loads((cells[0] / "manifest.json").read_text())
    method = manifest["judge_phase"]["bounds"]["input_bound_method"]
    assert method.startswith("anthropic count-tokens endpoint"), method


def _judge_client_kwargs(monkeypatch, tmp_path, seen):
    """Build the judge module's own client through the same stubbed anthropic
    module and return the kwargs it passed — the reference the count-tokens
    client must match."""
    judge = _RealClaudeJudge(cache=judging.JudgeCache(tmp_path / "c.sqlite"))
    before = len(seen)
    judge._client("lowprio")
    assert len(seen) == before + 1
    return seen[-1]


def test_count_tokens_client_pins_the_base_url_over_the_environment(
        driver, tmp_path, monkeypatch):
    """A session shell exports ANTHROPIC_BASE_URL pointing at a local gateway.
    The judge's own constructor passes ``base_url`` explicitly so the SDK
    never reads that variable; the count-tokens client did not, so the two
    clients the judge phase builds could go to DIFFERENT hosts — the count
    silently tunnelled to the gateway and hung (now: timed out to the byte
    bound) while the judge reached the API. Both must resolve the base URL
    the same way: the judge override env, else the real API."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8787/t/deadbeef")
    monkeypatch.delenv(judging.JUDGE_BASE_URL_ENV, raising=False)
    seen = _stub_anthropic(monkeypatch)
    driver._count_tokens_client()
    kw = seen[0]
    assert "base_url" in kw, (
        "count-tokens client built with no explicit base_url — the SDK reads "
        "ANTHROPIC_BASE_URL and a hijacked shell sends it somewhere else")
    assert kw["base_url"] == judging.DEFAULT_JUDGE_BASE_URL
    assert "127.0.0.1" not in kw["base_url"]
    assert kw["base_url"] == _judge_client_kwargs(monkeypatch, tmp_path, seen)["base_url"]


def test_count_tokens_client_follows_the_judge_base_url_override(
        driver, tmp_path, monkeypatch):
    """The judge's override env (a mock endpoint, a test harness) must move
    both clients together — same resolution, same host."""
    monkeypatch.setenv(judging.JUDGE_BASE_URL_ENV, "http://localhost:9/judge-mock")
    seen = _stub_anthropic(monkeypatch)
    driver._count_tokens_client()
    assert seen[0]["base_url"] == "http://localhost:9/judge-mock"
    assert seen[0]["base_url"] == _judge_client_kwargs(monkeypatch, tmp_path, seen)["base_url"]
