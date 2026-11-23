"""Credentials in scripts/rejudge/: the only one is the OpenRouter key, read from ``$OPENROUTER_API_KEY`` in the environment of the process
that sends requests (the batch client, the live engine), never from a flag or a file, and written nowhere. The code never reads or sets an
``ANTHROPIC_*`` variable and never builds a Claude client: the Claude judges' modules it imports for their prompt text and schemas
(``scripts/judge_batch.py``, ``judge_sonnet.py``) import the Anthropic SDK, which builds no client on import and reads only its own
logging switch. Offline: a local mock of OpenRouter's chat completions, temp directories."""
from __future__ import annotations

import gzip
import json
import os
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (ROOT / "scripts", ROOT / "src", ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from rejudge.common import openrouter as OR  # noqa: E402
from test_rejudge_live import Mock, init_live, make_source, run_engine  # noqa: E402

CODE = sorted((ROOT / "scripts" / "rejudge").rglob("*.py"))
#: a sentinel standing in for the key (low entropy, assembled at run time: no credential-shaped literal sits in the repository)
SENTINEL = "sentinel-" + "q" * 24


def test_the_code_names_no_anthropic_variable_and_builds_no_claude_client():
    pat = re.compile(r"ANTHROPIC_|anthropic_client|anthropic_async_client|wif_credentials|wif_ids|WorkloadIdentity|sisteer\.vault|"
                     r"from sisteer import vault|^\s*(?:import|from) anthropic\b", re.M)
    hits = [f"{p.relative_to(ROOT)}: {m.group(0).strip()}" for p in CODE for m in pat.finditer(p.read_text(encoding="utf-8"))]
    assert CODE and hits == []


def test_no_flag_or_file_carries_a_key():
    flag = re.compile(r"""add_argument\(\s*["']--[a-z0-9-]*(?:api-key|key-file|token\b|tokens-file|secret|credential)""", re.I)
    hits = [f"{p.relative_to(ROOT)}: {m.group(0)}" for p in CODE for m in flag.finditer(p.read_text(encoding="utf-8"))]
    assert hits == []
    reads = [p.relative_to(ROOT) for p in CODE if "OPENROUTER_API_KEY" in p.read_text(encoding="utf-8")]
    assert {str(p) for p in reads} <= {"scripts/rejudge/common/openrouter.py", "scripts/rejudge/common/live.py", "scripts/rejudge/common/harvester.py",
                                       "scripts/rejudge/strongreject/live.py", "scripts/rejudge/terra/live.py", "scripts/rejudge/terra/harvest_all.py",
                                       "scripts/rejudge/terra/build_manifests.py"}


def test_the_request_paths_read_no_anthropic_variable(monkeypatch):
    """Building every track's request and judge row, the Claude judges' modules imported on the way, reads no ``ANTHROPIC_*`` variable
    beyond the SDK's logging switch, and constructs no Anthropic client."""
    seen = []
    orig = os._Environ.__getitem__

    def recording(self, k):
        if str(k).upper().startswith("ANTHROPIC"):
            seen.append(str(k))
        return orig(self, k)

    monkeypatch.setattr(os._Environ, "__getitem__", recording)
    import anthropic

    def refuse(*a, **k):
        raise AssertionError("an Anthropic client was constructed")

    monkeypatch.setattr(anthropic.Anthropic, "__init__", refuse)
    monkeypatch.setattr(anthropic.AsyncAnthropic, "__init__", refuse)
    from rejudge.petri import file_records as PFR
    from rejudge.petri import judges as PJ
    from rejudge.strongreject import file_records as SFR
    from rejudge.strongreject import judges as SJ
    from rejudge.strongreject import request as SR
    from rejudge.terra import file_records as TFR
    from rejudge.terra import judges as TJ
    g = json.loads((ROOT / "tests" / "fixtures" / "rejudge" / "request_goldens.json").read_text(encoding="utf-8"))
    for kind in ("sr", "sr2"):
        SR.request_body(SJ.get()["request"], kind, "goal", "response")
        SFR.judge_spec(kind, judge="minimax-m3")
    c = g["cpi"]["or_bench_cpi"]
    TJ.build_entry("cpi", {"rubric": "or_bench_cpi", "prompt": c["prompt"], "response": c["response"], "k": c["custom_id"]},
                   TJ.request_settings("minimax-m3", "cpi"))
    TJ.build_entry("agentic", {"k": g["agentic"]["custom_id"], "prompt": g["agentic"]["prompt"]}, TJ.request_settings("minimax-m3", "agentic"))
    for rubric in ("or_bench_cpi", "xstest_cpi"):
        TFR.judge_spec(rubric, judge="minimax-m3")
    PJ.arm_body(PJ.profile(), g["petri"]["prompt"], g["petri"]["tool"])
    PFR.arm_protocol(PJ.profile())
    assert set(seen) <= {"ANTHROPIC_LOG"}


def test_the_batch_client_reads_the_key_from_the_environment_and_never_shows_it(monkeypatch):
    monkeypatch.delenv(OR.KEY_ENV, raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OR.BatchClient()
    monkeypatch.setenv(OR.KEY_ENV, SENTINEL)
    client = OR.BatchClient()
    assert SENTINEL not in repr(client) and SENTINEL not in str(client)
    assert SENTINEL not in OR.redact(f"upstream echoed Authorization: Bearer {SENTINEL}", SENTINEL)


def test_the_live_engine_reads_the_key_from_its_environment_and_writes_it_nowhere(tmp_path, monkeypatch):
    src = make_source(tmp_path)
    d = init_live(tmp_path, src)
    monkeypatch.delenv(OR.KEY_ENV, raising=False)
    with pytest.raises(SystemExit, match="OPENROUTER_API_KEY"):
        run_engine(d, Mock())
    monkeypatch.setenv(OR.KEY_ENV, SENTINEL)
    mock = Mock()
    code, _ = run_engine(d, mock)
    assert code == 0 and mock.total > 0
    leaked = []
    for p in tmp_path.rglob("*"):
        if p.is_file():
            data = p.read_bytes()
            if p.suffix == ".gz":
                data = gzip.decompress(data)
            if SENTINEL.encode() in data:
                leaked.append(str(p.relative_to(tmp_path)))
    assert leaked == []
