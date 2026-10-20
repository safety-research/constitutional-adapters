"""``grokmt_analyze``'s attacker-refusal classifier judges on the Low-Prio org
through ``sisteer.vault`` — never on whatever ``ANTHROPIC_API_KEY`` holds.

WHAT THIS REPLACES. ``_judge_items`` built its own client from
``os.environ.get("ANTHROPIC_API_KEY")`` and printed a coordination line that
said ``lowprio``. That variable IS the High Prio key (``sisteer.vault``: each
org has its own variable, and ``ANTHROPIC_API_KEY`` names High Prio and
nothing else), so every classification run went to High Prio while announcing
Low Prio — the identical silent misroute ``eval.api_keys.make_client("lowprio")``
had before the vault landing, as a READ rather than a file. It also demanded
``ANTHROPIC_BASE_URL`` be set by hand to keep an inherited shell's gateway out;
``vault.anthropic_client`` pins the base URL itself.

Now the client is ``vault.anthropic_client(JUDGE_ORG, max_retries=3)``: the
org's own key, the pinned base URL, the module's explicit request timeout,
and this script's retry count unchanged. Resolution happens only when a
client is actually built, never at import.

CPU-only, no network, no vault: ``vault.anthropic_client`` is faked at the
module ``grokmt_analyze`` calls it on; the import-time check runs a fresh
interpreter with no org key in its environment and a fake secret-manager CLI that fails.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

import grokmt_analyze as ga  # noqa: E402

from sisteer import vault  # noqa: E402


class _Messages:
    def __init__(self, text):
        self.text, self.calls = text, 0

    def create(self, **kwargs):
        self.calls += 1
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text=self.text)])


def _patch_client(monkeypatch, text="REFUSE"):
    captured, clients = [], []

    def fake_anthropic_client(org="lowprio", *, max_retries=5, **kw):
        captured.append({"org": org, "max_retries": max_retries, **kw})
        # a vault-built client carries its pool stamp; the placeholder is not a key
        clients.append(types.SimpleNamespace(
            messages=_Messages(text),
            sisteer_pool_stamp=vault.pool_stamp(org, "ORG-KEY-PLACEHOLDER")))
        return clients[-1]

    monkeypatch.setattr(vault, "anthropic_client", fake_anthropic_client)
    return captured, clients


def test_classifier_client_is_built_on_lowprio_with_the_scripts_retries(
        monkeypatch, tmp_path):
    captured, clients = _patch_client(monkeypatch)
    labels = ga._judge_items([("k1", "prompt one")], model="m", concurrency=2,
                             cache_path=tmp_path / "judge_cache.jsonl")
    assert ga.JUDGE_ORG == "lowprio"
    assert captured == [{"org": "lowprio", "max_retries": 3}]   # one client
    assert labels == {"k1": "REFUSE"} and clients[0].messages.calls == 1
    # the label survives for the resume path
    assert '"label": "REFUSE"' in (tmp_path / "judge_cache.jsonl").read_text()


def test_coordination_line_names_the_org_the_client_uses(monkeypatch, tmp_path,
                                                        capsys):
    _patch_client(monkeypatch)
    monkeypatch.setenv("SISTEER_COORDINATION_CHANNEL", "#example-api-coordination")
    ga._judge_items([("k1", "p")], model="m", concurrency=7,
                    cache_path=tmp_path / "judge_cache.jsonl")
    out = capsys.readouterr().out
    assert f"7 m {ga.JUDGE_ORG} " in out and "ready-to-post for #example-api-coordination" in out
    assert "highprio" not in out
    # the site names no channel: no ready-to-post line (the judging itself is unchanged)
    monkeypatch.delenv("SISTEER_COORDINATION_CHANNEL")
    (tmp_path / "judge_cache.jsonl").unlink(missing_ok=True)
    ga._judge_items([("k1", "p")], model="m", concurrency=7,
                    cache_path=tmp_path / "judge_cache.jsonl")
    assert "ready-to-post" not in capsys.readouterr().out


def test_run_facts_record_the_resolved_pool_beside_the_concurrency(monkeypatch,
                                                                   tmp_path):
    """``_judge_items(..., run_facts=d)`` fills the routing record ``main``
    writes under ``judge_run``: the pool the client actually resolved (never
    the key), the org asked for, the model, the concurrency, SDK retries."""
    _patch_client(monkeypatch)
    rf: dict = {}
    ga._judge_items([("k1", "p")], model="m", concurrency=4,
                    cache_path=tmp_path / "judge_cache.jsonl", run_facts=rf)
    assert rf == {**vault.pool_stamp("lowprio", "ORG-KEY-PLACEHOLDER"),
                  "org": "lowprio", "model": "m", "concurrency": 4,
                  "sdk_max_retries": 3}
    assert "ORG-KEY-PLACEHOLDER" not in str(rf)


def test_main_threads_the_judge_record_into_the_analysis():
    """The record reaches the written analysis: ``refusal_read`` forwards the
    dict to ``_judge_items`` and ``main`` files it as ``judge_run`` (None when
    no judging happened)."""
    src = Path(ga.__file__).read_text()
    assert "run_facts=judge_run" in src
    assert "judge_run=judge_run" in src
    assert 'res["judge_run"] = judge_run or None' in src


def test_missing_key_is_a_named_refusal_before_any_work(monkeypatch, tmp_path):
    def no_key(org="lowprio", **_kw):
        env, item = vault.ANTHROPIC_ORGS[org]
        raise vault.VaultSecretError(
            f"{env} is unset and the vault item {item!r} could not be read: "
            "the secret-manager CLI that SISTEER_SECRET_CMD names is not an executable (not found, or not on PATH)")

    monkeypatch.setattr(vault, "anthropic_client", no_key)
    cache = tmp_path / "judge_cache.jsonl"
    with pytest.raises(SystemExit) as ei:
        ga._judge_items([("k1", "p")], model="m", concurrency=1, cache_path=cache)
    msg = str(ei.value)
    assert "'lowprio'" in msg and "ANTHROPIC_API_KEY_LOWPRIO" in msg
    assert "never a file" in msg
    assert ei.value.__cause__ is None                   # nothing chained
    assert not cache.exists()                           # refused before any work


def test_no_direct_environment_read_remains():
    """The residue was a READ of High Prio's variable; the module must not
    touch ``os.environ`` for a credential at all now."""
    src = Path(ga.__file__).read_text()
    assert 'environ.get("ANTHROPIC_API_KEY' not in src
    assert 'environ["ANTHROPIC_API_KEY' not in src
    assert "anthropic.Anthropic(" not in src           # only vault builds clients


def test_import_resolves_no_credential():
    """A fresh interpreter with no org key and a failing secret-manager CLI
    imports the module with ZERO vault subprocesses; the first (and only)
    attempt happens when ``_judge_items`` actually builds a client."""
    code = textwrap.dedent(f"""
        import sys, pathlib, tempfile
        sys.path.insert(0, {str(SCRIPTS)!r})
        import sisteer.vault as v
        attempts = []
        def record(*a, **k):
            attempts.append(a[0][0])
            raise FileNotFoundError("fake-secret-cmd")
        v.subprocess.check_output = record
        import grokmt_analyze
        assert attempts == [], attempts
        print("IMPORT-OK")
        try:
            grokmt_analyze._judge_items([("k", "p")], model="m", concurrency=1,
                cache_path=pathlib.Path(tempfile.mkdtemp()) / "c.jsonl")
        except SystemExit as e:
            assert "'lowprio'" in str(e), str(e)
            print("ENTRY-REFUSED", attempts)
    """)
    env = {k: val for k, val in os.environ.items()
           if not k.startswith("ANTHROPIC_API_KEY") and k != "OPENROUTER_API_KEY"}
    env["PATH"] = "/usr/bin:/bin"
    env["SISTEER_SECRET_CMD"] = "fake-secret-cmd"                 # the site's CLI, named; the stub above makes it fail
    env["PYTHONPATH"] = f"{REPO / 'src'}{os.pathsep}{SCRIPTS}"
    r = subprocess.run([sys.executable, "-c", code], env=env,
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "IMPORT-OK" in r.stdout
    assert "ENTRY-REFUSED ['fake-secret-cmd']" in r.stdout, r.stdout
