"""Every live-Anthropic path this repo drives RECORDS the pool it resolved,
beside the concurrency it used, and builds its client ONLY through the shared
vault helpers. Static: source text and AST, no network, no vault.

WHY (2026-09-14). The shared helper ``eval/api_keys.make_client("lowprio")``
once fell back to a bare SDK client when its key file was absent, and the SDK
then read ``ANTHROPIC_API_KEY`` — the HIGH PRIO key — so callers asking for
the shared Low Prio pool got the priority pool and were told nothing. The
routing rule (on 429s throttle down, never hop pools) was never in force on
those paths: the helper hopped by construction. The vault closed the hop
(each pool has its own variable; a missing pool raises ``PoolUnavailable``).
What was still missing was the RECORD: nothing a run wrote said which pool it
had actually drawn on, so the quota accounting rested on inference. Now every
vault-built client carries a pool stamp and each recorder below writes it.

Two private copies of ``make_client`` (``grade_bon_live``, ``build_probe_set``)
are the per-lane-copy pattern that propagated the defect; both now build
through ``sisteer.vault`` (the second through ``anthropic_async_client``).
"""

from __future__ import annotations

import ast
import sys
import warnings
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: recorder -> (the read of the stamp, a fragment of the write). Exact text:
#: an entry whose fragments are gone FAILS, so a refactor that drops the
#: record cannot pass quietly.
RECORDERS = {
    "scripts/judge_overrefusal.py":
        ("vault.client_pool_stamp(client)", 'd / "overrefusal_judge_run.json"'),
    "scripts/judge_sonnet.py":
        ("judge.run_stats()", 'out / "judge_sonnet.run.json"'),
    "scripts/renellm_intent_judge.py":
        ("vault.client_pool_stamp(client)", '"judge_run": judge_run'),
    "scripts/grade_bon_live.py":
        ("vault.client_pool_stamp(client)", '"api_pool": api_pool'),
    "scripts/sdfbase/build_probe_set.py":
        ("client_pool_stamp(client)", '"api_pool": api_pool'),
    "scripts/grokmt_analyze.py":
        ("vault.client_pool_stamp(client)", 'res["judge_run"] = judge_run or None'),
    "scripts/alphasteer_f22/grade_alphasteer_f22.py":
        ("grader.run_stats()", '"api_pool":'),
}

#: The modules the pool fix covers: each builds its Anthropic client ONLY
#: through ``sisteer.eval.api_keys.make_client`` / ``sisteer.vault`` — never a
#: direct ``Anthropic(`` / ``AsyncAnthropic(`` of its own (a private copy is
#: where the last misroute lived). ``served_capability_run.py`` is listed in
#: the audit but has NO Anthropic client at all (its ``_make_client`` is the
#: served-proxy engine), which the last test pins.
SHARED_HELPER_ONLY = (
    "scripts/judge_overrefusal.py",
    "scripts/judge_sonnet.py",
    "scripts/renellm_intent_judge.py",
    "scripts/grade_bon_live.py",
    "scripts/sdfbase/build_probe_set.py",
    "src/sisteer/eval/attacks/llm_rewriter.py",
    "src/sisteer/eval/attacks/grading_live.py",
    "scripts/alphasteer_f22/grade_alphasteer_f22.py",
    "scripts/grokmt_analyze.py",
    "src/sisteer/eval/api_keys.py",
)


def _src(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _sdk_client_calls(rel: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(_src(rel), filename=rel)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else (
                f.attr if isinstance(f, ast.Attribute) else None)
            if name in ("Anthropic", "AsyncAnthropic"):
                yield node.lineno, name


@pytest.mark.parametrize("rel", sorted(RECORDERS))
def test_recorder_reads_the_stamp_and_writes_it(rel):
    read, write = RECORDERS[rel]
    s = _src(rel)
    assert read in s, f"{rel} no longer reads the pool stamp ({read!r})"
    assert write in s, f"{rel} no longer writes the record ({write!r})"


def test_no_recorder_carries_a_hardcoded_pool_label():
    """``grade_bon_live`` used to write ``"org": "low-prio"`` as a literal
    whatever ``--org`` was. A label is not a record."""
    for rel in RECORDERS:
        s = _src(rel)
        assert '"org": "low-prio"' not in s, rel
        assert "live Low-Prio" not in s, rel


@pytest.mark.parametrize("rel", SHARED_HELPER_ONLY)
def test_covered_module_never_constructs_an_sdk_client_itself(rel):
    calls = list(_sdk_client_calls(rel))
    assert not calls, (f"{rel} constructs an SDK client directly at "
                       f"{calls}: build it through sisteer.vault / "
                       "sisteer.eval.api_keys.make_client so the pool is "
                       "resolved once, refused when missing, and stamped")


def test_the_two_private_copies_build_through_the_vault():
    gbl = _src("scripts/grade_bon_live.py")
    assert "vault.anthropic_client(org" in gbl
    bps = _src("scripts/sdfbase/build_probe_set.py")
    assert "anthropic_async_client(org" in bps
    assert "read_text" not in bps.split("def make_client", 1)[1].split("\n\n\n", 1)[0]


def test_build_probe_set_offers_no_batch_pool_for_live_calls():
    """Live concurrents never run on the Batch pool (the pool rule); the
    file never calls the Batch API, so ``batch`` is not a live choice."""
    bps = _src("scripts/sdfbase/build_probe_set.py")
    assert 'choices=("lowprio", "highprio")' in bps
    assert '"batch")' not in bps and ".batches" not in bps


def test_served_capability_run_builds_no_anthropic_client():
    """Listed in the audit; NOT a pool path: its ``_make_client`` is the
    served-proxy engine. Pinned so the audit's 'out of scope' stays true."""
    s = _src("scripts/served_capability_run.py")
    assert "import anthropic" not in s and "Anthropic(" not in s
    assert "def _make_client(proxy_url: str, route) -> ProxyClientEngine" in s


def test_pool_unavailable_is_the_typed_named_refusal():
    sys.path.insert(0, str(REPO / "src"))
    from sisteer import vault
    assert issubclass(vault.PoolUnavailable, vault.VaultSecretError)
    e = vault.PoolUnavailable("lowprio", "pool 'lowprio' unavailable: no key")
    assert e.pool == "lowprio" and "lowprio" in str(e)
