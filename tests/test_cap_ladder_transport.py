"""The transport guard on the cap ladder (tiers/cap_ladder.py).

A rung is climbed only if the serving transport can carry a generation that
long. The failure it guards against is silent: a hop that times out surfaces
as a SHORT completion, the store says zero rejections, and the ladder reads
the truncated share as falling. So the check runs BEFORE an escalation is
issued, refuses rather than warns, and treats absent evidence as a refusal.
"""
import pytest

from sisteer.capeval.tiers import cap_ladder as cl

OK_HOPS = {cl.HOP_DRIVER: 7200.0, cl.HOP_PROXY_UPSTREAM: 7200.0}
INCIDENT_HOPS = {cl.HOP_DRIVER: 7200.0, cl.HOP_PROXY_UPSTREAM: 600.0}  # the GLM incident


def _verdict(cap=8192, share=0.05, n_trunc=5, **kw):
    return cl.verdict_for(share_truncated=share, n_truncated=n_trunc, cap=cap, **kw)


def test_a_rung_that_fits_the_configured_timeout_escalates():
    v, esc, nxt, _, tv = _verdict(hop_timeouts_s=OK_HOPS)
    assert (v, esc, nxt) == (cl.ESCALATE, True, 16384)
    assert tv is not None and tv.ok
    assert tv.required_s == pytest.approx(16384 / 4.0 * 1.5)
    assert tv.binding_s == 7200.0


def test_a_rung_that_does_not_fit_is_refused_with_the_required_value_in_the_message():
    v, esc, nxt, _, tv = _verdict(hop_timeouts_s=INCIDENT_HOPS)
    assert (v, esc) == (cl.EXIT_TRANSPORT_REFUSED, False)
    assert nxt == 16384                      # the rung it WOULD have climbed to is named
    assert tv is not None and not tv.ok
    assert tv.binding_hop == cl.HOP_PROXY_UPSTREAM and tv.binding_s == 600.0
    req = 16384 / 4.0 * 1.5
    assert tv.required_s == pytest.approx(req)
    assert f"{req:.0f} s" in tv.reason and "raise" in tv.reason.lower()
    assert "upstream" in tv.reason           # tells the operator WHERE to change it


def test_the_smallest_of_several_hops_binds():
    hops = {"a": 9000.0, "b": 650.0, "c": 7200.0}
    tv = cl.transport_check(2048, hops)
    assert tv.binding_hop == "b" and tv.binding_s == 650.0
    # 2048 tokens at 4 tok/s x 1.5 = 768 s > 650 s -> refused on the smallest hop
    assert not tv.ok


def test_no_hop_timeouts_at_all_refuses_as_undeterminable():
    v, esc, nxt, _, tv = _verdict(hop_timeouts_s=None)
    assert (v, esc) == (cl.EXIT_TRANSPORT_REFUSED, False)
    assert tv is not None and not tv.ok and "undeterminable" in tv.reason
    # and the DEFAULT is None: a caller that says nothing gets a refusal, never a climb
    v2, esc2, *_ = _verdict()
    assert (v2, esc2) == (cl.EXIT_TRANSPORT_REFUSED, False)


def test_a_hop_with_an_unrecorded_timeout_refuses_naming_that_hop():
    tv = cl.transport_check(1024, {cl.HOP_DRIVER: 7200.0, cl.HOP_PROXY_UPSTREAM: None})
    assert not tv.ok and tv.binding_hop == cl.HOP_PROXY_UPSTREAM
    assert "proxy_upstream" in tv.reason and "undeterminable" in tv.reason


def test_an_in_process_leg_has_no_hop_and_passes():
    tv = cl.transport_check(16384, {})
    assert tv.ok and tv.binding_hop is None and "in-process" in tv.reason
    v, esc, *_ = _verdict(hop_timeouts_s={})
    assert (v, esc) == (cl.ESCALATE, True)


def test_the_floor_rate_and_margin_are_overridable_and_flip_the_verdict():
    # 600 s cannot carry 8192 tokens at the default floor ...
    assert not cl.transport_check(8192, {"h": 600.0}).ok
    # ... but does at an optimistic 40 tok/s with no margin (8192/40 = 205 s)
    assert cl.transport_check(8192, {"h": 600.0}, floor_tok_per_s=40.0, margin=1.0).ok
    # and the override threads through verdict_for
    v, esc, *_ = _verdict(hop_timeouts_s={"h": 600.0}, floor_tok_per_s=40.0, margin=1.0)
    assert (v, esc) == (cl.ESCALATE, True)
    with pytest.raises(cl.CapLadderError):
        cl.transport_check(8192, {"h": 600.0}, floor_tok_per_s=0)


def test_the_guard_runs_before_any_escalation_is_issued_never_after():
    """The regression that matters: ``escalate`` is True ONLY when the
    transport check passed. Every other exit never reaches the check."""
    for hops in (None, {}, OK_HOPS, INCIDENT_HOPS, {"x": None}):
        for share, n_trunc, cap in ((0.0, 0, 512), (0.005, 1, 512), (0.5, 50, 16384),
                                    (0.05, 5, 512), (0.05, 5, 8192)):
            v, esc, nxt, _, tv = cl.verdict_for(share_truncated=share, n_truncated=n_trunc,
                                                cap=cap, hop_timeouts_s=hops)
            if esc:
                assert v == cl.ESCALATE and tv is not None and tv.ok
            if v in (cl.EXIT_CLEAN, cl.EXIT_BELOW_THRESHOLD, cl.EXIT_CEILING,
                     cl.EXIT_CEILING_AT_OUTSET, cl.EXIT_NO_RUNG):
                assert tv is None            # never reached the check
    # no_rung wins over the transport check: an Inspect leg is never asked
    v, esc, nxt, _, tv = cl.verdict_for(share_truncated=0.05, n_truncated=5, cap=512,
                                        ladder_available=False, hop_timeouts_s=INCIDENT_HOPS)
    assert (v, esc, tv) == (cl.EXIT_NO_RUNG, False, None)


def test_the_new_exit_is_in_the_vocabularies_and_the_rule_is_stamped():
    assert cl.EXIT_TRANSPORT_REFUSED in cl.EXITS
    assert cl.EXIT_TRANSPORT_REFUSED in cl.UNRESOLVED_EXITS
    assert "transport" in cl.RungStatus.__dataclass_fields__
    st = cl.rung_status({}, 512, hop_timeouts_s=OK_HOPS)
    assert st.rules["transport"] == cl.TRANSPORT_RULE
    assert st.transport is None              # clean: never reached the check


def test_served_hop_timeouts_read_the_records_and_refuse_an_unrecorded_proxy_hop():
    # a standup that recorded no upstream timeout (this branch's, today)
    hops = cl.served_hop_timeouts({"proxy_url": "http://x"}, 7200.0)
    assert hops == {cl.HOP_DRIVER: 7200.0, cl.HOP_PROXY_UPSTREAM: None}
    assert not cl.transport_check(16384, hops).ok
    # one that did
    hops = cl.served_hop_timeouts({cl.ENDPOINT_UPSTREAM_TIMEOUT_KEY: 7200}, 7200.0)
    assert hops == {cl.HOP_DRIVER: 7200.0, cl.HOP_PROXY_UPSTREAM: 7200.0}
    assert cl.transport_check(16384, hops).ok
    # the recorded value is what binds, not a default: 600 recorded -> refused
    hops = cl.served_hop_timeouts({cl.ENDPOINT_UPSTREAM_TIMEOUT_KEY: 600}, 7200.0)
    tv = cl.transport_check(16384, hops)
    assert not tv.ok and tv.binding_hop == cl.HOP_PROXY_UPSTREAM


def test_served_hop_timeouts_accepts_both_standups_key_names_and_refuses_a_torn_record():
    """Two standups record the same fact under two names: the sdfbase one as
    upstream_timeout_s, the tinycap one (oos-thinking-run) as
    proxy_upstream_timeout_s. The guard must read either, or the tinycap
    branch — the one that carries the --upstream-timeout-s flag — records a
    value the guard cannot see and refuses every climb."""
    a = cl.served_hop_timeouts({"upstream_timeout_s": 7200}, 7200.0)
    b = cl.served_hop_timeouts({"proxy_upstream_timeout_s": 7200}, 7200.0)
    assert a == b == {cl.HOP_DRIVER: 7200.0, cl.HOP_PROXY_UPSTREAM: 7200.0}
    assert cl.transport_check(16384, b).ok
    # both present and equal: fine
    both = cl.served_hop_timeouts({"upstream_timeout_s": 7200, "proxy_upstream_timeout_s": 7200.0},
                                  7200.0)
    assert both[cl.HOP_PROXY_UPSTREAM] == 7200.0
    # both present and DIFFERENT: a torn record, refused rather than resolved by preference
    with pytest.raises(cl.CapLadderError, match="DISAGREE"):
        cl.served_hop_timeouts({"upstream_timeout_s": 7200, "proxy_upstream_timeout_s": 600}, 7200.0)


def test_parse_hop_timeouts_cli_forms():
    assert cl.parse_hop_timeouts(None) is None
    assert cl.parse_hop_timeouts([], in_process=True) == {}
    assert cl.parse_hop_timeouts(["driver=7200", "proxy_upstream=600.5"]) == {
        "driver": 7200.0, "proxy_upstream": 600.5}
    assert cl.parse_hop_timeouts(["proxy_upstream=none"]) == {"proxy_upstream": None}
    for bad in (["driver"], ["=7200"], ["driver=abc"], ["driver=0"]):
        with pytest.raises(cl.CapLadderError):
            cl.parse_hop_timeouts(bad)
    with pytest.raises(cl.CapLadderError):
        cl.parse_hop_timeouts(["driver=7200"], in_process=True)


def test_measure_rung_threads_the_hops_through_to_the_status(monkeypatch):
    class Tok:
        def __call__(self, t, add_special_tokens=False):
            return {"input_ids": list(range(len(t.split())))}
    # 100 items, 5 truncated at cap 512 (>1%), nothing degenerate: the long
    # text must NOT repeat, or the loop rule (correctly) files it as
    # non-terminating and excludes it from the trigger
    long = " ".join(f"w{k}" for k in range(5000))
    entries = {f"i{k}": (long if k < 5 else "short answer.") for k in range(100)}
    # at cap 512 the next rung is 1024, which a 600 s hop CAN carry (384 s needed):
    # the guard correctly lets that climb proceed even on the incident's hops
    st_low, _ = cl.measure_rung(entries, Tok(), 512, hop_timeouts_s=INCIDENT_HOPS)
    assert st_low.exit == cl.ESCALATE and st_low.next_cap == 1024 and st_low.transport["ok"]
    # the incident geometry: at cap 4096 the next rung is 8192, which needs
    # 3072 s and a 600 s proxy hop cannot carry -> refused, not climbed
    st_ok, _ = cl.measure_rung(entries, Tok(), 4096, hop_timeouts_s=OK_HOPS)
    st_no, _ = cl.measure_rung(entries, Tok(), 4096, hop_timeouts_s=INCIDENT_HOPS)
    assert st_ok.exit == cl.ESCALATE and st_ok.escalate and st_ok.transport["ok"]
    assert st_no.exit == cl.EXIT_TRANSPORT_REFUSED and not st_no.escalate
    assert st_no.transport["binding_hop"] == cl.HOP_PROXY_UPSTREAM
    assert st_no.next_cap == 8192            # the rung it would have climbed to
