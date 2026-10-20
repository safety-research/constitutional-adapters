"""THE PROXY HALF of resumed steered prefills: which KV namespace a steered
leg is minted into, and on whose say-so.

The lens half (``lens_adapters.resumed_prefill_policy``) made a steered prefill
RESUMABLE for per-position ops and left the decision to the proxy, because
the proxy is what mints every ``cache_salt``: while every lens leg carried a
request-unique salt, both new worker paths — the ``always`` serve and the
``never`` engine-fatal refusal — were unreachable. This module pins the
decision the proxy now makes:

- the VOCABULARY is the adapters' (``lens_adapters.RESUMED_POLICIES`` /
  ``RESUMED_ALWAYS``), mirrored as literals in ``proxy`` for the same reason
  ``LENS_REDUCTION_MODE_KEY`` is a literal there — ``lens_adapters`` imports
  torch and the proxy must stay importable wherever it runs. The three WORDS
  are mirrored; the per-cell TABLE never is;
- the PARITY: over the adapters' whole op partition, the proxy hands out a
  cell-stable lens salt exactly when ``lens_adapters.resumed_prefill_ok``
  (i.e. policy ``always``) says the ops can carry a resume — so the two can
  never drift apart in the direction that shares a prefill the ops cannot
  carry;
- the REGIME per policy value, and for each of the three ways a policy can be
  MISSING (nothing attested; a map that does not name the cell; a leg that
  carries no selector at all): everything but an attested ``always`` keeps the
  request-unique salt, so a broadcast-kind cell (alphasteer,
  ``prefill_last_*``, ``antidote:last`` — whatever the adapters call ``never``)
  is never routed onto a shared prefix BY THIS DECISION, rather than by
  relying on the worker's engine-fatal refusal to catch it;
- the GUARD: a repeat on a cell-stable lens salt is the feature and is
  permitted; a repeat on any other lens salt, a steered request on another
  cell's namespace, a steered request on an unsteered salt and an unsteered
  request on a steered one are all still 409s; a client-supplied ``cache_salt``
  is still forbidden;
- the LEDGER: every forwarded request's audit row carries the regime it was
  minted under and the policy the proxy read, and so does the response's
  ``petri`` context block;
- the PLUMBING: both standups read the per-cell policy off the launch's
  steering attestation with the ONE strict reader and hand it to the proxy;
  neither derives it from kind names.

CPU-only; no vLLM, no GPU, no network.
"""

import inspect
import json
import re
import sys
from pathlib import Path

import pytest

from sisteer.capeval.petri import defense_cells as dc
from sisteer.capeval.serving import lens_adapters as la
from sisteer.capeval.serving import lens_worker as lw
from sisteer.capeval.serving import proxy as px

from test_petri_prefix_cache import PROMPTED, USER, _chat, _proxy
from test_petri_stacked import _entries
from test_lens_resumed_prefill import (HEADER, _affine, _alphasteer, _cosine,
                                       _gated, _lora, _lowrank, _vec)

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import capctrl_serving_up as csu   # noqa: E402
import lens_gate_standup as lgs    # noqa: E402

#: the lens cell the harness's routing table carries, and its unhooked and
#: LoRA siblings — real cells, so the policy map is keyed the way a worker
#: keys its install payload (by ``route.cell_id``).
LENS_CELL = "si_safety__psr_kl_all"
LENS_VMODEL = "petri__si_safety__psr_kl_all"
BROADCAST_CELL = "alphasteer"
BROADCAST_VMODEL = "petri__alphasteer"
PROMPTED_VMODEL = "petri__prompted__safety"


def _policy_proxy(tmp_path, policies, entries=None):
    """The harness proxy, handed ``policies`` as its ATTESTED per-cell map."""
    proxy, backend = _proxy(tmp_path, entries)
    return px.RoutingProxy(
        tuple(proxy._routes.values()), backend, base_model=proxy.base_model,
        serving_mode="interleaved",
        prompted_prefixes=tuple(proxy._prefixes.values()),
        resumed_prefill_policy_cells=policies), backend


def _row(proxy):
    return dict(proxy.request_log[-1])


# --- the vocabulary is the adapters' ---------------------------------------------


def test_the_policy_vocabulary_is_the_adapters_own():
    """proxy's literals ARE lens_adapters' — pinned so the mirror cannot
    drift (the same pin ``LENS_REDUCTION_MODE_KEY`` carries)."""
    assert px.RESUMED_PREFILL_ALWAYS == la.RESUMED_ALWAYS
    assert px.RESUMED_PREFILL_CONDITIONAL == la.RESUMED_CONDITIONAL
    assert px.RESUMED_PREFILL_NEVER == la.RESUMED_NEVER
    assert set(px.RESUMED_PREFILL_POLICIES) == set(la.RESUMED_POLICIES)
    # the reader the launcher uses accepts exactly this vocabulary
    assert all(p in la.RESUMED_POLICIES for p in px.RESUMED_PREFILL_POLICIES)
    # the three non-policies are disjoint from the three policies: a reader of
    # an audit row can always tell an attested verdict from a missing one
    assert not set(px.POLICY_NON_VALUES) & set(px.RESUMED_PREFILL_POLICIES)


def test_the_proxy_never_re_derives_the_table():
    """E62: the per-cell table arrives ATTESTED. The proxy module must not
    import the adapters (torch) nor carry a kind/op -> policy table of its
    own — the only thing mirrored is the three words above."""
    src = (REPO / "src/sisteer/capeval/serving/proxy.py").read_text()
    assert "import lens_adapters" not in src and "import lens_worker" not in src
    for op_name in sorted(la.PER_POSITION_OPS | la.BROADCAST_OPS):
        assert op_name not in src, (
            f"proxy.py names the op {op_name!r} — the resumed-prefill policy "
            "is read from the attested per-cell map, never derived from ops")


@pytest.mark.parametrize("ops,label", [
    ((_vec(),), "add_vector all/both"),
    ((_gated(),), "relu gated all/both"),
    ((_gated(op=la.OP_LINEAR_GATED_VECTOR),), "linear gated all/both"),
    ((_affine(),), "affine subspace all/both"),
    ((_lowrank(),), "lowrank relu all/both"),
    ((_lowrank(op=la.OP_LOWRANK_LINEAR_GATED),), "lowrank linear all/both"),
    ((_lora(),), "lora oproj all/both"),
    ((_vec(positions=la.POSITIONS_RESPONSE, header_ids=HEADER),), "add_vector response/both"),
    ((_gated(positions=la.POSITIONS_RESPONSE, header_ids=HEADER),), "gated response/both"),
    ((_vec(positions=la.POSITIONS_LAST),), "add_vector last/both"),
    ((_vec(phase=la.PHASE_PREFILL),), "add_vector all/prefill"),
    ((_vec(phase=la.PHASE_DECODE),), "add_vector all/decode"),
    ((_alphasteer(),), "alphasteer (broadcast)"),
    ((_cosine(),), "cosine gated (broadcast)"),
    ((_vec(), _alphasteer()), "mixed: one broadcast op poisons the cell"),
    ((_vec(), _vec(positions=la.POSITIONS_RESPONSE, header_ids=HEADER)), "mixed: all + response"),
    ((), "no ops at all"),
])
def test_the_regime_agrees_with_the_adapters_over_the_op_partition(ops, label):
    """PARITY. For every shape the adapters classify, the proxy's regime is
    CELL-STABLE iff ``lens_adapters.resumed_prefill_ok`` — i.e. iff the policy
    is ``always``. The two are wired through the attested map, so this is the
    property that keeps them from drifting into sharing a prefill the ops
    cannot carry."""
    policy = la.resumed_prefill_policy(ops)
    regime = px.RoutingProxy.salt_regime_for(lens=True,
                                             resumed_prefill_policy=policy)
    assert (regime == px.SALT_REGIME_CELL_STABLE) is la.resumed_prefill_ok(ops), label
    # ...and the broadcast half of the partition is never shared, whatever
    # else its positions/phase say
    if any(op.op in la.BROADCAST_OPS for op in ops):
        assert regime == px.SALT_REGIME_REQUEST_UNIQUE, label


# --- the regime, per policy and per way of having none ----------------------------


def test_an_attested_always_cell_gets_one_cell_stable_namespace(tmp_path):
    proxy, backend = _policy_proxy(tmp_path, {LENS_CELL: la.RESUMED_ALWAYS})
    _chat(proxy, LENS_VMODEL, USER)
    _chat(proxy, LENS_VMODEL, [{"role": "user", "content": "a different tail"}])
    a, b = (p["cache_salt"] for p in backend.payloads)
    assert a == b, "two requests of one always cell must share a namespace"
    assert a == f"petri|{LENS_CELL}|{proxy._routes[LENS_VMODEL].intervention_hash}|lens"
    row = _row(proxy)
    assert row[px.LEDGER_KEY_SALT_REGIME] == px.SALT_REGIME_CELL_STABLE
    assert row[px.LEDGER_KEY_RESUMED_POLICY] == la.RESUMED_ALWAYS


@pytest.mark.parametrize("policy", [la.RESUMED_CONDITIONAL, la.RESUMED_NEVER])
def test_conditional_and_never_keep_the_request_unique_salt(tmp_path, policy):
    """E62 constraint 1 + 2. ``conditional`` is not shared by THIS landing
    (the worker's per-request header decision stays unreachable through the
    proxy), and ``never`` is not shared BY THE PROXY'S DECISION — never by
    handing the worker its engine-fatal refusal and hoping."""
    proxy, backend = _policy_proxy(tmp_path, {LENS_CELL: policy})
    _chat(proxy, LENS_VMODEL, USER)
    _chat(proxy, LENS_VMODEL, USER)
    a, b = (p["cache_salt"] for p in backend.payloads)
    assert a != b
    stable = f"petri|{LENS_CELL}|{proxy._routes[LENS_VMODEL].intervention_hash}"
    assert not a.endswith("|lens") and a.startswith(stable + "|")
    row = _row(proxy)
    assert row[px.LEDGER_KEY_SALT_REGIME] == px.SALT_REGIME_REQUEST_UNIQUE
    assert row[px.LEDGER_KEY_RESUMED_POLICY] == policy


def test_a_broadcast_cell_is_never_routed_onto_a_shared_prefix(tmp_path):
    """The alphasteer route with its REAL attested policy: the edit is
    computed from the last prompt token and written across every prompt
    position, so a shared prefix is unsound; the proxy decides that, and the
    worker's fatal path is never reached."""
    proxy, backend = _policy_proxy(
        tmp_path, {BROADCAST_CELL: la.RESUMED_NEVER,
                   LENS_CELL: la.RESUMED_ALWAYS})
    _chat(proxy, BROADCAST_VMODEL, USER)
    _chat(proxy, BROADCAST_VMODEL, USER)
    a, b = (p["cache_salt"] for p in backend.payloads)
    assert a != b
    assert _row(proxy)[px.LEDGER_KEY_SALT_REGIME] == px.SALT_REGIME_REQUEST_UNIQUE
    # and the always cell beside it in the SAME proxy does share: the decision
    # is per cell, not per proxy
    _chat(proxy, LENS_VMODEL, USER)
    _chat(proxy, LENS_VMODEL, USER)
    c, d = (p["cache_salt"] for p in backend.payloads[2:])
    assert c == d


def test_nothing_attested_shares_nothing(tmp_path):
    """``None`` (an old worker, an unreadable attestation, workers that
    disagreed) reads as "nothing is resumable" — the pre-landing behaviour
    exactly, and the row says WHY it was not shared."""
    proxy, backend = _policy_proxy(tmp_path, None)
    assert proxy.resumed_prefill_policy_cells is None
    _chat(proxy, LENS_VMODEL, USER)
    _chat(proxy, LENS_VMODEL, USER)
    a, b = (p["cache_salt"] for p in backend.payloads)
    assert a != b
    row = _row(proxy)
    assert row[px.LEDGER_KEY_SALT_REGIME] == px.SALT_REGIME_REQUEST_UNIQUE
    assert row[px.LEDGER_KEY_RESUMED_POLICY] == px.POLICY_UNATTESTED


def test_a_cell_absent_from_the_attested_map_shares_nothing(tmp_path):
    """A map that names OTHER cells says nothing about this one; the row
    distinguishes that from "nobody told this proxy anything"."""
    proxy, backend = _policy_proxy(
        tmp_path, {"some__other__cell": la.RESUMED_ALWAYS})
    _chat(proxy, LENS_VMODEL, USER)
    _chat(proxy, LENS_VMODEL, USER)
    a, b = (p["cache_salt"] for p in backend.payloads)
    assert a != b
    row = _row(proxy)
    assert row[px.LEDGER_KEY_SALT_REGIME] == px.SALT_REGIME_REQUEST_UNIQUE
    assert row[px.LEDGER_KEY_RESUMED_POLICY] == px.POLICY_CELL_ABSENT


def test_an_unhooked_leg_is_cell_stable_and_consults_no_policy(tmp_path):
    proxy, backend = _policy_proxy(tmp_path, {LENS_CELL: la.RESUMED_ALWAYS})
    _chat(proxy, PROMPTED_VMODEL, PROMPTED)
    _chat(proxy, PROMPTED_VMODEL, PROMPTED)
    a, b = (p["cache_salt"] for p in backend.payloads)
    assert a == b and not a.endswith("|lens")
    row = _row(proxy)
    assert row[px.LEDGER_KEY_SALT_REGIME] == px.SALT_REGIME_CELL_STABLE
    assert row[px.LEDGER_KEY_RESUMED_POLICY] == px.POLICY_NOT_CONSULTED


def test_the_lens_stable_salt_can_never_equal_the_unhooked_one():
    """The ``|lens`` marker, and why it is there: a steered and an unhooked
    leg of one (cell, intervention) must not be mintable into ONE namespace,
    since that collision is the silent wrong-serve the whole isolation
    exists to prevent — impossible by construction beats caught by the
    guard."""
    r = px.RouteEntry(virtual_model="petri__x", cell_id="c", cell_class="k",
                      serving_identity="proxy_vllm+lens+interleaved",
                      intervention_hash="ab" * 8)
    unhooked = px.RoutingProxy.cache_salt_for(r, lens=False)
    stable = px.RoutingProxy.cache_salt_for(
        r, lens=True, resumed_prefill_policy=la.RESUMED_ALWAYS)
    unique = px.RoutingProxy.cache_salt_for(
        r, lens=True, resumed_prefill_policy=la.RESUMED_NEVER)
    assert len({unhooked, stable, unique}) == 3
    assert stable == unhooked + "|lens"
    # a uuid4 hex can never spell the marker, so the three forms are readable
    # off a salt in a log without ambiguity
    assert not re.fullmatch(r"[0-9a-f]{32}", px.LENS_STABLE_SALT_MARKER)
    assert re.fullmatch(r"[0-9a-f]{32}", unique.rsplit("|", 1)[1])


def test_the_default_policy_argument_is_the_pre_landing_behaviour():
    """Every caller that has not been taught about policies — bon_served, the
    cache-isolation gate's forced collision — keeps the salt it had."""
    r = px.RouteEntry(virtual_model="petri__x", cell_id="c", cell_class="k",
                      serving_identity="proxy_vllm+lens+interleaved",
                      intervention_hash="ab" * 8)
    assert (px.RoutingProxy.cache_salt_for(r, lens=True)
            != px.RoutingProxy.cache_salt_for(r, lens=True))
    assert (px.RoutingProxy.cache_salt_for(r, lens=False)
            == px.RoutingProxy.cache_salt_for(r, lens=False))
    assert px.RoutingProxy.salt_regime_for(lens=True) == px.SALT_REGIME_REQUEST_UNIQUE


# --- the salt-discipline guard ----------------------------------------------------


def test_a_repeat_on_an_always_cell_is_permitted_and_serves(tmp_path):
    """THE relaxation, through the public surface: the avg@k shape (the same
    prompt again on the same cell) no longer 409s — it is the resume."""
    proxy, backend = _policy_proxy(tmp_path, {LENS_CELL: la.RESUMED_ALWAYS})
    for _ in range(4):
        _chat(proxy, LENS_VMODEL, USER)
    assert len(backend.payloads) == 4
    assert len({p["cache_salt"] for p in backend.payloads}) == 1
    # one namespace = one guard slot, whatever the traffic
    assert len(proxy._salt_guard) == 1 and proxy.n_salt_guard_evicted == 0


@pytest.mark.parametrize("policies,why", [
    ({LENS_CELL: la.RESUMED_CONDITIONAL}, "conditional"),
    ({LENS_CELL: la.RESUMED_NEVER}, "never"),
    (None, "unattested"),
    ({"other": la.RESUMED_ALWAYS}, "cell absent"),
])
def test_a_repeat_on_any_other_cell_still_refuses(tmp_path, monkeypatch,
                                                  policies, why):
    """A request-unique salt is a namespace of one by construction, so a
    repeat is a minting bug — and the resume it would allow is the
    engine-fatal case. uuid4 is pinned to simulate exactly that bug."""
    import uuid as _uuid
    proxy, backend = _policy_proxy(tmp_path, policies)
    monkeypatch.setattr(px.uuid, "uuid4", lambda: _uuid.UUID(int=11))
    _chat(proxy, LENS_VMODEL, USER)
    with pytest.raises(px.ProxyRefusal, match="STEERED request would reuse") as ei:
        _chat(proxy, LENS_VMODEL, USER)
    assert ei.value.http_status == 409 and ei.value.code == "cache_salt_collision"
    assert px.SALT_REGIME_REQUEST_UNIQUE in str(ei.value), why
    assert len(backend.payloads) == 1     # the engine never saw the second


def test_a_steered_request_on_another_cells_namespace_refuses(tmp_path):
    """docs §4: for a resumable cell the worker accepts any resume its policy
    admits, so the salt namespace is the WHOLE isolation — a steered request
    that somehow reached another cell's namespace is refused here."""
    proxy, _ = _policy_proxy(tmp_path, {LENS_CELL: la.RESUMED_ALWAYS})
    mine = proxy._routes[LENS_VMODEL]
    theirs = proxy._routes[BROADCAST_VMODEL]
    salt = px.RoutingProxy.cache_salt_for(
        mine, lens=True, resumed_prefill_policy=la.RESUMED_ALWAYS)
    proxy._guard_cache_salt(mine, salt=salt, lens=True,
                            regime=px.SALT_REGIME_CELL_STABLE)
    proxy._guard_cache_salt(mine, salt=salt, lens=True,       # the repeat: fine
                            regime=px.SALT_REGIME_CELL_STABLE)
    with pytest.raises(px.ProxyRefusal, match="cache_salt of cell") as ei:
        proxy._guard_cache_salt(theirs, salt=salt, lens=True,
                                regime=px.SALT_REGIME_CELL_STABLE)
    assert ei.value.http_status == 409 and ei.value.code == "cache_salt_collision"


def test_an_unsteered_request_on_a_stable_steered_salt_refuses(tmp_path):
    """The SILENT wrong-serve direction, now reachable in principle because a
    steered salt can be stable: still a 409, before the engine sees it."""
    proxy, _ = _policy_proxy(tmp_path, {LENS_CELL: la.RESUMED_ALWAYS})
    route = proxy._routes[LENS_VMODEL]
    salt = px.RoutingProxy.cache_salt_for(
        route, lens=True, resumed_prefill_policy=la.RESUMED_ALWAYS)
    proxy._guard_cache_salt(route, salt=salt, lens=True,
                            regime=px.SALT_REGIME_CELL_STABLE)
    with pytest.raises(px.ProxyRefusal, match="UNSTEERED request would reuse") as ei:
        proxy._guard_cache_salt(route, salt=salt, lens=False,
                                regime=px.SALT_REGIME_CELL_STABLE)
    assert ei.value.http_status == 409


def test_a_steered_request_on_an_unsteered_salt_refuses(tmp_path):
    proxy, _ = _policy_proxy(tmp_path, {LENS_CELL: la.RESUMED_ALWAYS})
    route = proxy._routes[LENS_VMODEL]
    unhooked = px.RoutingProxy.cache_salt_for(route, lens=False)
    proxy._guard_cache_salt(route, salt=unhooked, lens=False,
                            regime=px.SALT_REGIME_CELL_STABLE)
    with pytest.raises(px.ProxyRefusal,
                       match="an UNSTEERED request already sent upstream") as ei:
        proxy._guard_cache_salt(route, salt=unhooked, lens=True,
                                regime=px.SALT_REGIME_CELL_STABLE)
    assert ei.value.http_status == 409


def test_an_unknown_regime_is_a_startup_class_error(tmp_path):
    proxy, _ = _policy_proxy(tmp_path, {LENS_CELL: la.RESUMED_ALWAYS})
    route = proxy._routes[LENS_VMODEL]
    with pytest.raises(px.ProxyError, match="unknown cache-salt regime"):
        proxy._guard_cache_salt(route, salt="s", lens=True, regime="shared-ish")


def test_a_client_supplied_cache_salt_is_still_forbidden(tmp_path):
    proxy, backend = _policy_proxy(tmp_path, {LENS_CELL: la.RESUMED_ALWAYS})
    assert "cache_salt" in px.FORBIDDEN_CLIENT_KEYS
    with pytest.raises(px.ProxyRefusal, match="reserved selector key"):
        _chat(proxy, LENS_VMODEL, USER, cache_salt="mine")
    assert not backend.payloads


# --- what the run can read back ---------------------------------------------------


def test_the_response_petri_block_echoes_the_decision(tmp_path):
    """CONTEXT, never the contract: ``proxy_engine.STAMP_FIELDS`` names four
    fields and reads only those, so the echo is additive to every stamp
    check — and it is the only way an out-of-process observer (the gate) can
    read the decision made for ITS request."""
    from sisteer.capeval.serving import proxy_engine as pe
    proxy, _ = _policy_proxy(tmp_path, {LENS_CELL: la.RESUMED_ALWAYS,
                                        BROADCAST_CELL: la.RESUMED_NEVER})
    out = _chat(proxy, LENS_VMODEL, USER)
    assert out["petri"][px.LEDGER_KEY_SALT_REGIME] == px.SALT_REGIME_CELL_STABLE
    assert out["petri"][px.LEDGER_KEY_RESUMED_POLICY] == la.RESUMED_ALWAYS
    other = _chat(proxy, BROADCAST_VMODEL, USER)
    assert other["petri"][px.LEDGER_KEY_SALT_REGIME] == px.SALT_REGIME_REQUEST_UNIQUE
    assert other["petri"][px.LEDGER_KEY_RESUMED_POLICY] == la.RESUMED_NEVER
    # the four stamped fields are untouched by the two additions: the stamp
    # checker names them and reads only those, so a strict consumer sees no
    # change at all
    assert px.LEDGER_KEY_SALT_REGIME not in pe.STAMP_FIELDS
    assert f"petri.{px.LEDGER_KEY_SALT_REGIME}" not in pe.STAMP_FIELDS
    assert f"petri.{px.LEDGER_KEY_RESUMED_POLICY}" not in pe.STAMP_FIELDS
    assert set(pe.response_stamps(out)) == set(pe.STAMP_FIELDS)
    pe.assert_response_stamps(out, proxy._routes[LENS_VMODEL])


def test_every_forwarded_request_carries_the_two_fields(tmp_path):
    # the harness fixture's stacked cells all ride the lens now
    # (enumerate_cells refuses a stacked entry marked lora), so the table has
    # a `none__lora0` leg only if some PLAIN object cell is marked lora — the
    # broadcast cell is, for this test alone (the legacy lens+lora hybrid
    # table the default enumeration still admits); its regime is
    # path-agnostic here, only the two fields are read
    ent = _entries(tmp_path)
    ent[BROADCAST_CELL] = dc.RegistryEntry(
        cell_id=BROADCAST_CELL, kind=la.SYNTHETIC_KIND, si_id=None,
        object_path=ent[BROADCAST_CELL].object_path, serving_path="lora",
        resolved_spec={"kind": la.SYNTHETIC_KIND, "layer": 3})
    proxy, _ = _policy_proxy(tmp_path, {LENS_CELL: la.RESUMED_ALWAYS}, ent)
    for vm, msgs in ((LENS_VMODEL, USER), (PROMPTED_VMODEL, PROMPTED),
                     (BROADCAST_VMODEL, USER), ("petri__base", USER),
                     (f"petri__{dc.NOOP_LORA_CELL_ID}", USER)):
        _chat(proxy, vm, msgs)
    assert len(proxy.request_log) == 5
    for row in proxy.request_log:
        assert row[px.LEDGER_KEY_SALT_REGIME] in px.SALT_REGIMES
        assert (row[px.LEDGER_KEY_RESUMED_POLICY]
                in tuple(px.RESUMED_PREFILL_POLICIES) + px.POLICY_NON_VALUES)


def test_the_identity_block_carries_the_map_only_when_attested(tmp_path):
    """Same discipline as ``lens_reduction_mode``: echoed when the workers
    attested it, ABSENT (never defaulted) when they did not — a zero-traffic
    read of what every steered request of this proxy will be minted under."""
    proxy, _ = _policy_proxy(tmp_path, {LENS_CELL: la.RESUMED_ALWAYS})
    block = proxy.models_payload()["petri_proxy"]
    assert block["resumed_prefill_policy_cells"] == {LENS_CELL: la.RESUMED_ALWAYS}
    bare, _ = _policy_proxy(tmp_path, None)
    assert "resumed_prefill_policy_cells" not in bare.models_payload()["petri_proxy"]


def test_a_malformed_map_refuses_to_start(tmp_path):
    """The attested reader hands over a clean map or ``None``; junk here is a
    wiring defect and must be loud BEFORE any traffic, not fall back."""
    with pytest.raises(px.ProxyError, match="outside"):
        _policy_proxy(tmp_path, {LENS_CELL: "sometimes"})
    with pytest.raises(px.ProxyError, match="outside"):
        _policy_proxy(tmp_path, {LENS_CELL: True})
    with pytest.raises(px.ProxyError, match="mapping or None"):
        _policy_proxy(tmp_path, [LENS_CELL])


# --- the standups read it with the ONE strict reader ------------------------------


def _attestation(policies, n_workers=1):
    return {"worker_reports": [
        {lw.RESUMED_PREFILL_POLICY_CELLS_KEY: dict(policies)}
        for _ in range(n_workers)]}


def test_the_gate_standup_reader_is_the_workers_own(tmp_path):
    att = _attestation({LENS_CELL: la.RESUMED_ALWAYS,
                        BROADCAST_CELL: la.RESUMED_NEVER}, n_workers=2)
    cells, err = lgs.resumed_prefill_policy_cells_attested(att)
    assert err is None and cells == {LENS_CELL: la.RESUMED_ALWAYS,
                                     BROADCAST_CELL: la.RESUMED_NEVER}
    # it IS lens_worker's reader, not a second implementation
    assert cells == lw.resumed_prefill_policy_cells_attested(att)[0]
    # disagreement, an unknown vocabulary and a pre-landing tree are all
    # (None, why) — the proxy then shares nothing
    bad = {"worker_reports": [{lw.RESUMED_PREFILL_POLICY_CELLS_KEY: {LENS_CELL: "always"}},
                              {lw.RESUMED_PREFILL_POLICY_CELLS_KEY: {LENS_CELL: "never"}}]}
    assert (lgs.resumed_prefill_policy_cells_attested(bad)
            == lw.resumed_prefill_policy_cells_attested(bad))
    assert lgs.resumed_prefill_policy_cells_attested(bad)[0] is None
    junk = {"worker_reports": [{lw.RESUMED_PREFILL_POLICY_CELLS_KEY: {LENS_CELL: "sometimes"}}]}
    assert lgs.resumed_prefill_policy_cells_attested(junk)[0] is None

    class _OldTree:
        pass
    cells, err = lgs.resumed_prefill_policy_cells_attested(att, lw=_OldTree())
    assert cells is None and "predates the resumed-prefill landing" in err


def test_the_tinycap_standup_reader_reads_the_attestation_file(tmp_path):
    path = tmp_path / "steering_attestation.json"
    path.write_text(json.dumps(_attestation({LENS_CELL: la.RESUMED_CONDITIONAL})))
    cells, err = csu.attested_resumed_prefill_policy_cells_at(path)
    assert err is None and cells == {LENS_CELL: la.RESUMED_CONDITIONAL}
    # a missing file is (None, why) — recorded, never guessed
    cells, err = csu.attested_resumed_prefill_policy_cells_at(tmp_path / "nope.json")
    assert cells is None and err
    # so is a worker report that carries no such key at all
    cells, err = csu.attested_resumed_prefill_policy_cells_at(
        _write(tmp_path / "old.json", {"worker_reports": [{}]}))
    assert cells is None and "resumed_prefill_policy_cells" in err


def _write(path, obj):
    path.write_text(json.dumps(obj))
    return path


class TestCrossTreeConstructorGuard:
    """THE HAZARD THAT COST A GPU GATE RUN (2026-09-07). The vectorize gate
    launches the REFERENCE standup with the CANDIDATE tree's
    ``lens_gate_standup.py`` but the REFERENCE tree's ``sisteer`` on the path
    (``standup_env``; the endpoint's ``served_from`` records the split). An
    unconditional ``resumed_prefill_policy_cells=`` kwarg is therefore a
    ``TypeError`` before readiness on EVERY reference standup — which the gate
    reports as a per-case FAIL with a traceback its own line truncation cuts
    off. ``reduction_mode_attested``'s ``getattr(lw, ...)`` guard is the same
    hazard on the reader side; this is its constructor twin."""

    def test_a_proxy_without_the_parameter_is_not_given_it(self):
        class OldProxy:
            def __init__(self, routes, backend, *, base_model, serving_mode,
                         prompted_prefixes=(), exact_replay_strategy=None,
                         launch_attestation=None, api_token=None,
                         agent_loop_render=None, petri_agent_loop_entry=True,
                         base_revision=None):
                pass

        kw, err = lgs.resumed_prefill_proxy_kwargs({LENS_CELL: la.RESUMED_ALWAYS}, OldProxy)
        assert kw == {} and err and "before the resumed-prefill landing" in err
        # and the kwargs it DOES return are accepted by the real constructor
        kw2, err2 = lgs.resumed_prefill_proxy_kwargs({LENS_CELL: la.RESUMED_ALWAYS},
                                                     px.RoutingProxy)
        assert err2 is None
        assert kw2 == {lgs.RESUMED_PREFILL_POLICY_KWARG: {LENS_CELL: la.RESUMED_ALWAYS}}
        assert lgs.RESUMED_PREFILL_POLICY_KWARG in inspect.signature(
            px.RoutingProxy.__init__).parameters

    def test_an_unintrospectable_constructor_falls_closed(self):
        kw, err = lgs.resumed_prefill_proxy_kwargs({LENS_CELL: la.RESUMED_ALWAYS}, object())
        assert kw == {} and err

    def test_the_call_site_splats_and_never_names_the_kwarg(self):
        """A literal ``resumed_prefill_policy_cells=`` in the constructor call
        is the bug; the guard is only a guard if the call goes through it."""
        src = (REPO / "scripts/lens_gate_standup.py").read_text()
        ctor = src.index("proxy = RoutingProxy(")
        end = src.index("proxy_port = free_port()", ctor)
        call = src[ctor:end]
        assert f"{lgs.RESUMED_PREFILL_POLICY_KWARG}=" not in call
        assert "**resumed_policy_kwargs" in call
        assert "resumed_prefill_proxy_kwargs(resumed_policy_cells, RoutingProxy)" in src

    def test_a_blocked_handover_clears_the_published_map(self):
        """The served proxy honours no policy, so the endpoint must publish
        none — a reader deriving a salt regime from a map the proxy never
        received would be reading a fiction."""
        src = (REPO / "scripts/lens_gate_standup.py").read_text()
        blk = src[src.index("if resumed_policy_kwarg_error:"):src.index("proxy = RoutingProxy(")]
        assert "resumed_policy_cells = None" in blk
        assert "resumed_policy_error" in blk

    def test_the_tinycap_standup_is_deliberately_unguarded(self):
        """`capctrl_serving_up` serves the tree it lives in — there is no
        cross-tree split there, so a constructor that cannot take the kwarg is
        a genuine wiring defect and must be LOUD, not recorded and continued."""
        src = (REPO / "scripts/capctrl_serving_up.py").read_text()
        assert "resumed_prefill_policy_cells=rp_cells" in src
        assert "resumed_prefill_proxy_kwargs" not in src


@pytest.mark.parametrize("script", ["capctrl_serving_up.py", "lens_gate_standup.py"])
def test_both_standups_hand_the_attested_map_to_the_proxy(script):
    """The read happens once, off the launch's steering attestation, and its
    RESULT is what the proxy is constructed with — so a standup whose workers
    attest nothing builds a proxy that shares nothing."""
    src = (REPO / "scripts" / script).read_text()
    assert "resumed_prefill_policy_cells" in src
    # the READ is the workers' own strict reader, off the launch attestation
    assert "resumed_prefill_policy_cells_attested" in src
    # ...its RESULT reaches the constructor (directly, or through the
    # cross-tree guard for the standup the gate runs against an older sisteer)
    ctor = src.index("RoutingProxy(")
    end = src.index("proxy_port", ctor)
    call = src[ctor:end]
    assert ("resumed_prefill_policy_cells=rp_cells" in call
            or "**resumed_policy_kwargs" in call), call
    # and it is published in the standup's own endpoint record, so a client
    # can derive the regime without asking the proxy
    assert '"resumed_prefill_policy_cells"' in src


# --- DECLARED CELLS SHARE NOTHING + the lens salt policy ---------------------

from sisteer.capeval.petri import defense_cells as dc  # noqa: E402
from sisteer.capeval.serving import serve_declaration as sd  # noqa: E402
from test_petri_proxy import CANONICAL_22, StubBackend, _cells, _source, _spec  # noqa: E402

BASE = "si_safety__psr_kl_all"
TURNS = f"{BASE}@all_turns"
ALLP = f"{BASE}@all_positions"
TWIN = dc.NOOP_LENS_CELL_ID
#: the experiment-2 standup's four lens routes and their ATTESTED policies
#: (a response object is conditional; an all-positions object is always;
#: the twin carries its donor's — the base's — ops)
EXP2_POLICIES = {BASE: la.RESUMED_CONDITIONAL, TURNS: la.RESUMED_CONDITIONAL,
                 ALLP: la.RESUMED_ALWAYS, TWIN: la.RESUMED_CONDITIONAL}
EXP2_DECLARATIONS = {TURNS: sd.variant_declarations("all_turns"),
                     ALLP: sd.variant_declarations("all_positions")}


class _AgreeingProbe:
    """A synthetic render probe whose located and message-derived all-turns
    spans AGREE on the single-turn USER conversation these tests send, so the
    @all_turns route's per-request render check passes and the SALT decision
    is what is under test here (the check itself is
    tests/test_all_turns_render_check.py's)."""

    header_ids = (900, 901, 902)
    turn_end_ids = (990, 77)

    def render(self, messages, tools):
        return (1, 2, *self.header_ids)

    def assistant_content_spans(self, messages):
        return (1, 2), ()


def _exp2_proxy(tmp_path, *, lens_salt_policy="attested", policies=EXP2_POLICIES,
                declarations=EXP2_DECLARATIONS):
    spec = _spec(serving_mode="interleaved", cells=list(CANONICAL_22) + [TURNS, ALLP])
    src = _source(tmp_path)
    entries = dict(src.entries)
    base = entries[BASE]
    for variant, cid in (("all_turns", TURNS), ("all_positions", ALLP)):
        entries[cid] = dc.RegistryEntry(cell_id=cid, kind=base.kind, si_id=base.si_id,
                                        object_path=base.object_path, serving_path="lens",
                                        resolved_spec=sd.apply_variant(base.resolved_spec, variant))
    source = dc.MappingCellSource(entries)
    cells = _cells(spec, source)
    backend = StubBackend()
    proxy = px.RoutingProxy(
        px.routes_from_cells(cells), backend, base_model=spec.target_model,
        serving_mode="interleaved", prompted_prefixes=px.prompted_prefix_specs(cells, source),
        exact_replay_strategy=None, agent_loop_render=_AgreeingProbe(),
        resumed_prefill_policy_cells=policies, lens_serve_declarations=declarations,
        lens_salt_policy=lens_salt_policy)
    return proxy, backend


def _vm(cell):
    return "petri__" + cell


def test_a_declared_always_cell_mints_request_unique_labelled_declared(tmp_path):
    """P2: without this rule setting (3) — an all-positions object, attested
    `always` — would run WARM beside its cold siblings."""
    proxy, backend = _exp2_proxy(tmp_path)
    _chat(proxy, _vm(ALLP), USER)
    _chat(proxy, _vm(ALLP), USER)
    a, b = (p["cache_salt"] for p in backend.payloads)
    assert a != b and not a.endswith("|lens")
    row = _row(proxy)
    assert row[px.LEDGER_KEY_SALT_REGIME] == px.SALT_REGIME_REQUEST_UNIQUE == "request-unique"
    assert row[px.LEDGER_KEY_RESUMED_POLICY] == px.POLICY_DECLARED_SHARES_NOTHING == "declared-request-unique"
    # the ATTESTATION map still says `always` — the label names WHY the salt
    # was request-unique; the map is the workers' fact, untouched
    assert proxy.resumed_prefill_policy_cells[ALLP] == la.RESUMED_ALWAYS
    assert proxy.models_payload()["petri_proxy"]["resumed_prefill_policy_cells"][ALLP] == "always"


def test_the_experiment_2_four_route_table_under_the_attested_policy(tmp_path):
    """All four lens routes request-unique — the base and twin because
    `conditional` maps to request-unique TODAY (pinned so a future
    single-turn stable-salt refinement fails here instead of silently
    confounding the run), the variants because they are declared."""
    proxy, backend = _exp2_proxy(tmp_path)
    want_label = {BASE: "conditional", TWIN: "conditional",
                  TURNS: "declared-request-unique", ALLP: "declared-request-unique"}
    for cell in (BASE, TURNS, ALLP, TWIN):
        out = _chat(proxy, _vm(cell), USER)
        _chat(proxy, _vm(cell), USER)
        a, b = (p["cache_salt"] for p in backend.payloads[-2:])
        assert a != b, cell
        assert out["petri"][px.LEDGER_KEY_SALT_REGIME] == "request-unique", cell
        assert out["petri"][px.LEDGER_KEY_RESUMED_POLICY] == want_label[cell], cell
    assert proxy.models_payload()["petri_proxy"][px.LENS_SALT_POLICY_KEY] == "attested"
    # an undeclared-cell absent from the policy map: `absent`, request-unique
    proxy2, _ = _exp2_proxy(tmp_path, policies={ALLP: la.RESUMED_ALWAYS})
    out = _chat(proxy2, _vm(BASE), USER)
    assert out["petri"][px.LEDGER_KEY_RESUMED_POLICY] == px.POLICY_CELL_ABSENT


def test_the_experiment_2_four_route_table_under_request_unique_all(tmp_path):
    proxy, backend = _exp2_proxy(tmp_path, lens_salt_policy="request_unique_all")
    for cell in (BASE, TURNS, ALLP, TWIN):
        out = _chat(proxy, _vm(cell), USER)
        _chat(proxy, _vm(cell), USER)
        a, b = (p["cache_salt"] for p in backend.payloads[-2:])
        assert a != b and not a.endswith("|lens"), cell
        assert out["petri"][px.LEDGER_KEY_SALT_REGIME] == "request-unique", cell
        assert out["petri"][px.LEDGER_KEY_RESUMED_POLICY] == px.POLICY_FORCED_REQUEST_UNIQUE == "forced-request-unique", cell
    # a non-lens leg is untouched: cell-stable, policy not consulted
    out = _chat(proxy, "petri__base", USER)
    _chat(proxy, "petri__base", USER)
    a, b = (p["cache_salt"] for p in backend.payloads[-2:])
    assert a == b
    assert out["petri"][px.LEDGER_KEY_SALT_REGIME] == "cell-stable"
    assert out["petri"][px.LEDGER_KEY_RESUMED_POLICY] == px.POLICY_NOT_CONSULTED
    # the proxy's own record
    assert proxy.models_payload()["petri_proxy"][px.LENS_SALT_POLICY_KEY] == "request_unique_all"
    assert proxy.lens_salt_policy == "request_unique_all"
    # the base cell is NOT declared out of support by the knob
    assert BASE not in proxy.lens_serve_declarations


def test_an_unknown_salt_policy_refuses_at_construction(tmp_path):
    with pytest.raises(px.ProxyError, match="lens_salt_policy"):
        _exp2_proxy(tmp_path, lens_salt_policy="mostly_unique")
    assert px.LENS_SALT_POLICIES == ("attested", "request_unique_all")
    assert px.LENS_SALT_POLICY_KEY == "lens_salt_policy"


def test_the_two_new_labels_are_non_values_and_the_regimes_are_unchanged():
    assert px.POLICY_DECLARED_SHARES_NOTHING in px.POLICY_NON_VALUES
    assert px.POLICY_FORCED_REQUEST_UNIQUE in px.POLICY_NON_VALUES
    assert not set(px.POLICY_NON_VALUES) & set(px.RESUMED_PREFILL_POLICIES)
    assert px.SALT_REGIMES == ("cell-stable", "request-unique")


def test_the_proxy_cli_takes_the_salt_policy():
    src = (REPO / "src/sisteer/capeval/serving/proxy.py").read_text()
    assert '"--lens-salt-policy"' in src and "lens_salt_policy=args.lens_salt_policy" in src
    csrc = (REPO / "scripts/capctrl_serving_up.py").read_text()
    assert '"--lens-salt-policy"' in csrc and "lens_salt_policy=args.lens_salt_policy" in csrc
    ap = csu.build_parser()
    ns = ap.parse_args(["--repo-dir", "r", "--config", "c", "--cell", "none", "--defenses-dir", "d",
                        "--out", "o", "--stop-file", "s"])
    assert ns.lens_salt_policy == "attested"
    with pytest.raises(SystemExit):
        ap.parse_args(["--repo-dir", "r", "--config", "c", "--cell", "none", "--defenses-dir", "d",
                       "--out", "o", "--stop-file", "s", "--lens-salt-policy", "warm"])
