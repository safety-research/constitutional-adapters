"""THE SERVED-TARGET <-> PROXY CONTRACT: the experiment-2 client
(``served_target.py`` on ``fst-exp2``) PINS the literals the proxy/lens
code (``proxy.py`` on ``fst-lens``) defines — the attestation keys, the
four all-turns refusal codes, the salt policy vocabulary and labels, the
thinking-ON declaration key, the standup's per-cell digest surface —
because the client cannot import that code. This test imports those
constants and asserts equality.

EXPECTED RED ON ``fst-exp2`` ALONE (that code is not merged: ``proxy.py`` has
no ``SERVE_DECLARATIONS_KEY`` yet) and GREEN ON THE INTEGRATION BRANCH
(``fst-integration``, which merges both branches). The client's targeted runs
exclude it by name; the integration CPU suite runs it. A red here on the
integration branch is a drift between the two lanes' literals — fix the pin
or the definition, never the test.
"""
from __future__ import annotations

import dataclasses
import inspect
import sys
from pathlib import Path

import pytest

from sisteer.capeval.serving import proxy
from sisteer.eval.attacks import served_target as st

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def _need(module, name):
    if not hasattr(module, name):
        pytest.fail(f"{module.__name__} defines no {name!r} — WS-A's landing (plan §2.4) is "
                    "not on this branch, or renamed the constant; the client pins its VALUE")
    return getattr(module, name)


def test_attestation_map_key_matches_proxy():
    assert _need(proxy, "SERVE_DECLARATIONS_KEY") == st.ATTEST_SERVE_DECLARATIONS_KEY \
        == "lens_serve_declarations"


def test_all_turns_refusal_codes_match_proxy():
    """The proxy's exported ``ALL_TURNS_REFUSAL_CODES`` tuple is exactly the client's
    four, and every ``ALL_TURNS_*`` str constant on proxy.py is one of them — so a
    changed/added/dropped refusal literal fails here, while audit-row KEYS that merely
    start with ``all_turns_`` (``LEDGER_KEY_ALL_TURNS_CHECK = "all_turns_check"``)
    are not refusal codes and are excluded by name."""
    codes = set(_need(proxy, "ALL_TURNS_REFUSAL_CODES"))
    assert codes == set(st.ALL_TURNS_REFUSAL_CODES)
    consts = {v for k, v in vars(proxy).items()
              if k.startswith("ALL_TURNS_") and isinstance(v, str) and v.startswith("all_turns_")}
    assert consts == codes


def test_salt_policy_vocabulary_and_labels_match_proxy():
    assert tuple(_need(proxy, "LENS_SALT_POLICIES")) == st.LENS_SALT_POLICIES
    assert _need(proxy, "POLICY_FORCED_REQUEST_UNIQUE") == st.SALT_POLICY_LABEL_FORCED_REQUEST_UNIQUE
    assert _need(proxy, "POLICY_DECLARED_SHARES_NOTHING") == st.SALT_POLICY_LABEL_DECLARED_REQUEST_UNIQUE


def test_thinking_on_declaration_key_matches_proxy():
    assert _need(proxy, "THINKING_ON_KEY") == st.THINKING_ON_DECLARATION_KEY == "petri_thinking_on"


def test_models_payload_echoes_the_keys_the_client_reads():
    """``/v1/models.petri_proxy`` must carry the map, the salt policy and the
    per-cell digests under the keys the client reads."""
    src = inspect.getsource(proxy.RoutingProxy.models_payload)
    for key in (st.ATTEST_SERVE_DECLARATIONS_KEY, st.ATTEST_LENS_SALT_POLICY_KEY,
                st.ATTEST_OBJECT_SHA256_BY_CELL_KEY, st.ATTEST_THINKING_HEADER_KEY):
        assert (key in src or key.upper() in src
                or any(key in str(v) for k, v in vars(proxy).items() if k.isupper())), key


def test_capctrl_standup_surface():
    """``capctrl_serving_up.py``: repeatable ``--cell``, ``--lens-salt-policy``,
    ``Standup.object_sha256_by_cell`` and the two endpoint keys."""
    import capctrl_serving_up as cs
    ap = cs.build_parser()
    ns = ap.parse_args(["--repo-dir", "/r", "--config", "c.yaml", "--cell", "none",
                        "--cell", "safety_refusal__psr_kl_all@all_turns",
                        "--defenses-dir", "/d", "--out", "/o", "--stop-file", "/s",
                        "--lens-salt-policy", st.LENS_SALT_POLICY_REQUEST_UNIQUE_ALL,
                        "--expect-object-sha256", f"safety_refusal__psr_kl_all@all_turns={'ab' * 32}"])
    assert list(ns.cell) == ["none", "safety_refusal__psr_kl_all@all_turns"]
    assert ns.lens_salt_policy == st.LENS_SALT_POLICY_REQUEST_UNIQUE_ALL
    fields = {f.name for f in dataclasses.fields(cs.Standup)}
    assert "object_sha256_by_cell" in fields, sorted(fields)
    src = inspect.getsource(cs.Standup.endpoint)
    for key in (st.ATTEST_OBJECT_SHA256_BY_CELL_KEY, st.ATTEST_LENS_SALT_POLICY_KEY,
                st.ATTEST_SERVE_DECLARATIONS_KEY, st.ATTEST_SERVE_DECLARATIONS_ERROR_KEY):
        assert key in src, key
