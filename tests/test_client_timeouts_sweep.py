"""Every Anthropic / OpenAI SDK client this repo constructs passes an explicit
per-request ``timeout=``, an explicit ``base_url=``, AND an explicit
``api_key=``.

The SDK default (anthropic 1.0.0 -- 0.120.2 before it -- and openai 2.52.0 alike) is
``Timeout(connect=5, read=600, write=600, pool=600)``. Against a judge call
that answers in ~1.8 s that is a 330x margin, and it cannot tell a slow call
from a dead socket. This class of defect has now bitten twice, each time
fixed at ONE site while the same shape stood elsewhere:

* 2026-08-28 — the live grader wedged with 100 sockets open and no progress
  for 40+ minutes (``tests/test_grader_has_timeout.py`` pins that one site).
* 2026-09-11 — two sockets ESTAB with empty send/recv queues stalled a
  5,953-item judged run behind ``capeval.judging``'s clients; legs run
  serially, so the whole run waited, and its watchdog read the silence as
  pool degradation and throttled 200 -> 50 — the wrong remedy for a
  misidentified cause (both pools answered in 1.8 s at ``max_retries=0``).
  ``sisteer.vault`` had set the timeout and documented why; the judge module
  one file away had not.

A per-site test cannot stop the third occurrence. This one walks ``src/`` and
``scripts/`` (tests excluded: their fakes are not clients) and refuses ANY
``Anthropic(`` / ``AsyncAnthropic(`` / ``OpenAI(`` / ``AsyncOpenAI(`` call
without a literal ``timeout=`` keyword. Sites that build through
``sisteer.vault.anthropic_client`` or ``sisteer.eval.api_keys.make_client``
are covered by construction (both always pass it — see ``test_api_keys.py``
and ``test_capeval_judging.py``).

The second keyword is the same lesson for a different class. A client that
omits ``base_url`` lets the SDK read ``ANTHROPIC_BASE_URL`` from the shell;
a session that points that at a local gateway makes the client tunnel there
and look healthy — silent misrouting. ``sisteer.vault`` pinned it and said
why (2026-09-07); the count-tokens client and the persona anchor did
not. Omitted host bit twice, so it gets a gate, not a third per-site test.
The rule is "a ``base_url`` is PASSED", never "it equals the Anthropic
default": the proxy target's URL is a legitimately different value, and a
gate that could not express that would be wrong about the one case it most
needs to be right about.

The third keyword is the pool hop (2026-09-14). A client built with no
``api_key=`` makes the SDK read ``ANTHROPIC_API_KEY`` from the shell, and
that variable is the HIGH PRIO key — so an unkeyed client asked for on behalf
of the shared Low Prio pool IS a pool hop, silent, before any 429 could be
throttled. ``eval/api_keys.make_client`` did exactly that when its key file
was absent (the file was deleted on purpose on 2026-09-04; the vault landed
2026-09-07). The keyword is necessary, not sufficient — ``api_key=None`` also
falls back to the shell — which is why every site is expected to pass a
value the vault resolved (``sisteer.vault`` never returns an empty one). Its
inventory (``KNOWN_UNKEYED``) is empty and has no imaginable entry.

Each inventory (``KNOWN_UNBOUNDED``, ``KNOWN_UNPINNED_HOST``,
``KNOWN_UNKEYED``) is a record of
sites that miss the keyword ON PURPOSE, not an allowlist. The timeout one
names the frozen serving tree's client (byte-identical to its pinned commit
for live campaigns; its fix is a ruling, not a patch). The base_url one is
EMPTY: that same frozen client passes ``base_url=config.base_url`` — the
proxy — so it complies, and an entry for it would be false. Both inventories
are asserted EXACT in both directions, so fixing an inventoried site fails
the test until its entry is removed — the debt cannot silently outlive the
freeze, and a stale entry cannot hide a fixed site.
"""

from __future__ import annotations

import ast
import warnings
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCANNED_ROOTS = ("src", "scripts")
CLIENT_CLASSES = frozenset({"Anthropic", "AsyncAnthropic", "OpenAI", "AsyncOpenAI"})

#: (relative path, callee) -> reason. Exact: an entry whose site is gone or
#: bounded FAILS the test as stale.
KNOWN_UNBOUNDED = {
    ("src/sisteer/capeval/serving/petri_target.py", "AsyncOpenAI"):
        "FROZEN serving tree (byte-identical to its pinned commit for live "
        "campaigns) — threading a timeout is a decision for the campaign "
        "owner, not a sweep fix. Dormant on main: make_async_client has no "
        "production caller; the target role's live traffic is inspect_ai's "
        "own client via petri_runner.get_model(...). Ruled 2026-09-11: no "
        "bound there OR in the runner until the decode-time distribution at "
        "the response cap is measured — 120 s would censor the slow leg.",
}
#: Sites that omit ``base_url`` on purpose. None: the frozen client above
#: passes base_url=config.base_url (the proxy), so it complies with THIS rule.
KNOWN_UNPINNED_HOST: dict = {}
#: Sites that omit ``api_key`` on purpose. None, and none can be: an unkeyed
#: client resolves to whatever pool the shell holds (High Prio here), which
#: is the silent hop this whole rule exists to make impossible.
KNOWN_UNKEYED: dict = {}

#: keyword every construction must pass -> its inventory of deliberate misses
REQUIRED_KEYWORDS = {"timeout": KNOWN_UNBOUNDED, "base_url": KNOWN_UNPINNED_HOST,
                     "api_key": KNOWN_UNKEYED}


def _callee_name(node: ast.Call) -> str | None:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _client_constructions():
    """Yield (relpath, lineno, callee, passed_keywords) for every SDK client
    call — ``passed_keywords`` is the frozenset of keyword names literally
    present at the call site."""
    for root in SCANNED_ROOTS:
        for path in sorted((REPO / root).rglob("*.py")):
            rel = path.relative_to(REPO).as_posix()
            try:
                with warnings.catch_warnings():
                    # an unrelated file's invalid escape sequence is not
                    # this test's finding
                    warnings.simplefilter("ignore", SyntaxWarning)
                    tree = ast.parse(path.read_text(encoding="utf-8"),
                                     filename=rel)
            except SyntaxError as e:  # a broken file is its own failure elsewhere
                pytest.fail(f"{rel}: {e}")
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = _callee_name(node)
                if name not in CLIENT_CLASSES:
                    continue
                passed = frozenset(kw.arg for kw in node.keywords if kw.arg)
                yield rel, node.lineno, name, passed


def test_the_sweep_sees_the_known_bounded_sites():
    """Guard against the scanner silently scanning nothing (a moved root, a
    renamed class): the two DIRECT constructors by design — the vault helper
    everything else builds through, and the judge module the reference fix
    bounded — must be visible. (grade_bon_live builds through the vault, so
    it is covered by construction and correctly absent here.)"""
    seen = {(rel, name) for rel, _ln, name, _kw in _client_constructions()}
    for expected in [
        ("src/sisteer/vault.py", "Anthropic"),
        ("src/sisteer/vault.py", "AsyncAnthropic"),
        ("src/sisteer/capeval/judging.py", "Anthropic"),
    ]:
        assert expected in seen, f"scanner no longer sees {expected}"


@pytest.mark.parametrize("keyword", sorted(REQUIRED_KEYWORDS))
def test_every_sdk_client_construction_passes(keyword):
    inventory = REQUIRED_KEYWORDS[keyword]
    missing = {(rel, name): ln
               for rel, ln, name, kws in _client_constructions() if keyword not in kws}
    unexpected = {k: ln for k, ln in missing.items() if k not in inventory}
    why = {
        "timeout": ("each inherits the SDK's 600 s default and a hung socket "
                    "reads as pool degradation"),
        "base_url": ("each lets the SDK read ANTHROPIC_BASE_URL from the shell, "
                     "and a hijacked shell tunnels it elsewhere while it looks "
                     "healthy"),
        "api_key": ("each lets the SDK read ANTHROPIC_API_KEY from the shell — "
                    "the High Prio key — so a client asked for on behalf of "
                    "another pool silently hops to High Prio; resolve the pool "
                    "through sisteer.vault and pass its key"),
    }[keyword]
    assert not unexpected, (
        f"SDK client(s) built with no explicit {keyword}= — {why}:\n" +
        "\n".join(f"  {rel}:{ln} {name}(" for (rel, name), ln in sorted(unexpected.items()))
        + "\nPlumb a module constant (WHY + the measurement) through a "
          "parameter, the way capeval/judging.py does; do not hardcode a "
          "value at the call site.")


@pytest.mark.parametrize("keyword", sorted(REQUIRED_KEYWORDS))
def test_the_deliberate_miss_inventory_is_exact(keyword):
    """Both directions: every inventoried site still misses the keyword (else
    the entry is stale and must go), and nothing is inventoried under a path
    the scanner does not reach."""
    inventory = REQUIRED_KEYWORDS[keyword]
    missing = {(rel, name) for rel, _ln, name, kws in _client_constructions()
               if keyword not in kws}
    stale = set(inventory) - missing
    assert not stale, (
        f"{keyword} inventory entries no longer missing the keyword (fixed or "
        f"removed) — delete them: {sorted(stale)}")
