"""fz5 (the gap fz3 disclosed): the at-cap table reads its served cap and truncation share off the
ATTEMPT INVENTORY on the row's request path — the usage ledger where the bound plan declares one (GSM8K) or one exists
undeclared, else (`RoutingProxy` writes no ledger — the ordinary completed Modal root of BoN, PAIR, Crescendo,
ActorAttack, Petri, XSTest and OR-Bench-Hard-1k) the response hook's completions bound to their REQUEST RECORDS.

Known-bad at 59c02703b: `at_cap_table.ledger_task_rows` unconditionally called `RootReads.ledger_rows()`, which refuses a
missing `usage_ledger.jsonl`; on every one of the seven ledger-less legs `build()` aborted before `main()` wrote `at_cap.csv`,
so the WHOLE table was missing and `battery_tables` marked every otherwise-valid ladder `ladder: MISSING (no at_cap.csv)`.

Green: the seven legs through the real `row_for` on Modal-shaped roots (request + response evidence + store, NO ledger) produce
rows with the caps the records asked and the shares the response evidence implies; the identity of the partition (record ↔
attempt, the fz3/fz4 topology: a failed retry attempt is an attempt, never a completion; an orphan on either side refuses; a
label disagreement refuses), the Arditi seed attribution (`extra.s`), the cap-consistency and completeness checks are retained;
a served completion with no `finish_reason` and a record with no `max_tokens` refuse — never a default; a DECLARED ledger that
is missing (GSM8K) still refuses; every row carries the qualification keys the verdicts carry (`ledger`, `attempt_source`,
`reconciliation_contract`), resolved over every contributing accepted attempt root (a disagreement refuses). Synthetic roots.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "scripts" / "sdfbase" / "seedrep", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import at_cap_table as act  # noqa: E402
import audit_common as ac  # noqa: E402
import request_evidence_audit as rea  # noqa: E402
import seedrep_audit_fixtures as fx  # noqa: E402

GOALS = ["g0", "g1", "g2"]
CONTRACT_LEDGERLESS = "seedrep_request_evidence_reconciliation/2-D106"
CONTRACT_LEDGER = "seedrep_request_evidence_reconciliation/1"
LEDGERLESS = {"ledger": "not_on_path", "attempt_source": "response_evidence", "reconciliation_contract": CONTRACT_LEDGERLESS}
FAILED_RID = "f" * 32


def _row(stores: Path, leg: str, arm="none", s=1, view="s1", reads=None, plan=None):
    return act.row_for(leg, arm, s, view, ac.discover_attempt_roots(stores), petri=list(fx.PETRI_BEHAVIOURS),
                       trig_static=act.trigger_share_static(), trig_gsm8k=act.trigger_share_gsm8k(), reads=reads, plan=plan)


def _modal(root: fx.Root):
    """The Modal path's shape on a non-GSM8K row: the fixture's ledger process never ran."""
    (root.path / ac.LEDGER_FILE).unlink()
    return ac.read_attempt_root(root.path)


def _pin_known_bad(ar):
    """The 59c02703b refusal: the single-ledger rule on a root that never had one; the topology says not_on_path."""
    with pytest.raises(ac.AuditRefusal, match=r"no 'usage_ledger.jsonl'"):
        ac.RootReads().ledger_rows(ar)
    assert ac.ledger_status(ar)[0] == ac.LEDGER_NOT_ON_PATH


def _qual(r):
    return {k: r[k] for k in act.QUALIFICATION_KEYS}


# --------------------------------------------------------------------------- #
# the seven ledger-less legs through the real row_for                                                               #
# --------------------------------------------------------------------------- #

def test_known_bad_seven_ledgerless_legs_now_read_caps_and_shares_off_the_bound_evidence(tmp_path: Path):
    stores = tmp_path / "stores"
    bon = fx.make_root(stores, leg="bon", view="s1", arms=("none",))
    fx.write_bon_store(bon, arm="none", s=1, goals=GOALS, at_cap_goals=("g0",))
    bon.flush_pipeline()
    pair = fx.make_root(stores, leg="pair", view="s1", arms=("none",))
    fx.write_multiturn_store(pair, attack="pair", arm="none", s=1, goals=GOALS, at_cap_goals=("g0",))
    pair.flush_pipeline()
    cres = fx.make_root(stores, leg="crescendo", view="s1", arms=("none",))
    fx.write_multiturn_store(cres, attack="crescendo", arm="none", s=1, goals=GOALS)
    cres.flush_pipeline()
    actor = fx.make_root(stores, leg="actorattack", view="s1", arms=("none",))
    fx.write_multiturn_store(actor, attack="actorattack", arm="none", s=1, goals=GOALS, at_cap_goals=("g0", "g1"))
    actor.flush_pipeline()
    petri = fx.make_root(stores, leg="petri", view="s1", arms=("none",))
    fx.write_petri_store(petri, arm="none", s=1, scenarios={"sc0": "beh_a", "sc1": "beh_b"}, at_cap_scenarios=("sc0",))
    petri.flush_pipeline()
    ctl = fx.make_root(stores, leg="controls", view="s1", arms=("none",))
    fx.write_controls_store(ctl, arm="none", s=1, items_by_control={"xstest": ["item0", "item1", "item2"], "or_bench_hard_1k": ["ob0", "ob1", "ob2"]},
                            mmlu_items=["mm0"], at_cap={"xstest": ["item0"]})
    ctl.flush_pipeline()
    for root in (bon, pair, cres, actor, petri, ctl):
        _pin_known_bad(_modal(root))
    reads = ac.RootReads()  # one build's memo across the seven rows, as build() runs them
    r = _row(stores, "bon", reads=reads)
    assert (r["rung_read"], r["n"], r["n_at_cap_final"], r["share_at_cap_final"], r["over_1pct"], r["ladder_state"]) == ("fixed:4096", 6, 2, 0.3333, True, "fixed_cap")
    assert r["source"] == "bon rows finish_reason + request-record cap (D106: ledger not on path)" and _qual(r) == LEDGERLESS
    r = _row(stores, "pair", reads=reads)
    assert (r["rung_read"], r["n"], r["n_at_cap_final"], r["share_at_cap_final"], r["ladder_state"]) == ("fixed:4096", 6, 2, 0.3333, "fixed_cap")
    assert r["source"] == "response evidence finish_reason (target task completions; D106: ledger not on path)" and _qual(r) == LEDGERLESS
    r = _row(stores, "crescendo", reads=reads)
    assert (r["rung_read"], r["n"], r["n_at_cap_final"], r["share_at_cap_final"], r["over_1pct"]) == ("fixed:4096", 6, 0, 0.0, False) and _qual(r) == LEDGERLESS
    r = _row(stores, "actorattack", reads=reads)
    assert (r["rung_read"], r["n"], r["n_at_cap_final"], r["share_at_cap_final"]) == ("fixed:4096", 6, 4, 0.6667) and _qual(r) == LEDGERLESS
    r = _row(stores, "petri", reads=reads)  # the Petri store rows carry no finish reason: the truncation is read off the response evidence
    assert (r["rung_read"], r["n"], r["n_at_cap_final"], r["share_at_cap_final"], r["ladder_state"]) == ("fixed:1024", 4, 2, 0.5, "fixed_cap")
    assert _qual(r) == LEDGERLESS
    r = _row(stores, "xstest", reads=reads)  # the controls cap is held to the request records' max_tokens (the ledger's counterpart)
    assert (r["base_cap"], r["final_cap"], r["n"], r["n_at_cap_final"], r["share_at_cap_final"], r["ladder_state"]) == (512, 512, 3, 1, 0.3333, "escalate_pending(2048)")
    assert r["source"] == "gen_stats.json n_at_cap/n_total" and _qual(r) == LEDGERLESS
    r = _row(stores, "or_bench_hard_1k", reads=reads)
    assert (r["base_cap"], r["n"], r["share_at_cap_final"], r["ladder_state"]) == (512, 3, 0.0, "exit_le_1pct") and _qual(r) == LEDGERLESS
    m = _row(stores, "mmlu", reads=reads)  # the MMLU row rides the controls root: its qualification is that root's
    assert m["applies"] is False and m["ladder_state"] == "not_applicable" and _qual(m) == LEDGERLESS
    assert set(act.COLUMNS) >= set(act.QUALIFICATION_KEYS) and all(k in r for k in act.QUALIFICATION_KEYS)


def test_no_store_row_carries_the_qualification_keys_never_blank(tmp_path: Path):
    stores = tmp_path / "stores"
    pair = fx.make_root(stores, leg="pair", view="s1", arms=("none",))
    fx.write_multiturn_store(pair, attack="pair", arm="none", s=1, goals=GOALS)
    pair.flush_pipeline()
    _modal(pair)
    r = _row(stores, "pair", arm="prompted__si_constitution")
    assert r["ladder_state"] == "no_store" and _qual(r) == {k: "no_store" for k in act.QUALIFICATION_KEYS}


# --------------------------------------------------------------------------- #
# identity over the response inventory: retries, orphans, label disagreement                                          #
# --------------------------------------------------------------------------- #

def _ledgerless_pair(tmp_path: Path, **kw):
    stores = tmp_path / "stores"
    root = fx.make_root(stores, leg="pair", view="s1", arms=("none",))
    fx.write_multiturn_store(root, attack="pair", arm="none", s=1, goals=GOALS, **kw)
    root.flush_pipeline()
    return stores, root, _modal(root)


def _write(p: Path, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_failed_attempt_then_retry_is_an_attempt_not_a_completion_and_orphans_refuse(tmp_path: Path):
    stores, root, ar = _ledgerless_pair(tmp_path, at_cap_goals=("g0",))
    ep, rp = root.path / ac.EVIDENCE_FILE, root.path / ac.RESPONSE_EVIDENCE_FILE
    recs, resp = ac.read_jsonl(ep), ac.read_jsonl(rp)
    success = next(r for r in resp if r["completion_id"])
    rec0 = next(r for r in recs if r["request_id"] == success["request_id"])
    # a 503 on a FRESH request id (its own request record — the client records every attempt it sends), then the recorded success
    failed_rec = {**rec0, "request_id": FAILED_RID}
    failed_resp = {**success, "request_id": FAILED_RID, "completion_id": None, "status": 503, "vmodel": None, "finish_reason": None,
                   ac.RESPONSE_TEXT_SHA_FIELD: None, ac.RESPONSE_TEXT_SHA_OF_FIELD: None, "error": "HTTP 503: upstream busy"}
    _write(ep, [failed_rec] + recs)
    _write(rp, [failed_resp] + resp)
    inv = ac.RootReads().attempt_inventory(ar)
    assert inv.attempt_source == "response_evidence" and len(inv.rows) == 7 and FAILED_RID in inv.by_id
    r = _row(stores, "pair")
    assert (r["n"], r["n_at_cap_final"], r["share_at_cap_final"]) == (6, 2, 0.3333) and _qual(r) == LEDGERLESS  # 7 attempts, 6 completions
    # (a) a response row naming NO record: refused as an orphan of the partition (its own labels put it there)
    _write(rp, [failed_resp, {**success, "request_id": "0" * 32, "completion_id": "chatcmpl-orphan"}] + resp)
    with pytest.raises(ac.AuditRefusal, match=r"1 response-evidence attempt\(s\) of \('multi_turn/pair', 'petri__none'\) have NO request-evidence record"):
        _row(stores, "pair")
    _write(rp, [failed_resp] + resp)
    # (b) a record the response evidence never recorded: a record without an attempt
    _write(ep, [failed_rec, {**rec0, "request_id": "e" * 32}] + recs)
    with pytest.raises(ac.AuditRefusal, match=r"1 request-evidence record\(s\) of \('multi_turn/pair', 'petri__none'\) have NO response-evidence attempt"):
        _row(stores, "pair")
    _write(ep, [failed_rec] + recs)
    # (c) the failed attempt's record and its response row disagree on the sample: the label clause, inside the record's partition
    _write(ep, [{**failed_rec, "sample": "g9/9"}] + recs)
    with pytest.raises(ac.AuditRefusal, match=f"request_id {FAILED_RID!r}: the response evidence labels the attempt task 'multi_turn/pair' sample 'g0/0' but the record says task 'multi_turn/pair' sample 'g9/9'"):
        _row(stores, "pair")


def test_missing_finish_reason_or_max_tokens_refuses_never_a_default(tmp_path: Path):
    stores, root, ar = _ledgerless_pair(tmp_path)
    ep, rp = root.path / ac.EVIDENCE_FILE, root.path / ac.RESPONSE_EVIDENCE_FILE
    recs, resp = ac.read_jsonl(ep), ac.read_jsonl(rp)
    victim = resp[0]["request_id"]
    _write(rp, [{**r, "finish_reason": None} if r["request_id"] == victim else r for r in resp])
    with pytest.raises(ac.AuditRefusal, match=r"1 of 6 served 'multi_turn/pair' completion\(s\) carry no finish_reason .* never read as 'stop'"):
        _row(stores, "pair")
    _write(rp, resp)
    assert _row(stores, "pair")["share_at_cap_final"] == 0.0
    _write(ep, [{**r, "max_tokens": None} if r["request_id"] == victim else r for r in recs])
    with pytest.raises(ac.AuditRefusal, match=r"the request records' max_tokens is not one value on the 'multi_turn/pair' rows \(\['4096', 'None'\]\)"):
        _row(stores, "pair")
    _write(ep, [{**r, "max_tokens": 8192} if r["request_id"] == victim else r for r in recs])  # two caps on one partition: not one cap
    with pytest.raises(ac.AuditRefusal, match=r"max_tokens is not one value on the 'multi_turn/pair' rows \(\['4096', '8192'\]\)"):
        _row(stores, "pair")


def test_controls_ledgerless_cap_is_held_to_the_request_records(tmp_path: Path):
    stores = tmp_path / "stores"
    ctl = fx.make_root(stores, leg="controls", view="s1", arms=("none",))
    fx.write_controls_store(ctl, arm="none", s=1, items_by_control={"xstest": GOALS}, at_cap={"xstest": ["g0"]})
    ctl.flush_pipeline()
    _modal(ctl)
    assert _row(stores, "xstest")["base_cap"] == 512
    task = ac.CONTROLS_TASK_OF["xstest"]
    ep = ctl.path / ac.EVIDENCE_FILE
    fx.rewrite_jsonl(ep, lambda x: {**x, "max_tokens": 1024} if x.get("task") == task else x)
    with pytest.raises(ac.AuditRefusal, match=r"the request records' max_tokens \['1024'\] on the xstest requests drawn here != the store's cap 512"):
        _row(stores, "xstest")


# --------------------------------------------------------------------------- #
# the ledger path is unchanged: declared (GSM8K) and present_undeclared                                                #
# --------------------------------------------------------------------------- #

def test_gsm8k_declared_ledger_unchanged_and_a_missing_declared_ledger_still_refuses(tmp_path: Path):
    stores = tmp_path / "stores"
    g = fx.make_root(stores, leg="gsm8k", view="s1", arms=("none",))
    fx.write_gsm8k_store(g, arm="none", s=1, items=["item0", "item1", "item2"], at_cap_items=("item1",))
    g.flush_pipeline()   # the REAL layout: one child ledger per child store, nothing at the root
    ar = ac.read_attempt_root(g.path)
    assert ac.ledger_status(ar)[0] == ac.LEDGER_DECLARED
    r = _row(stores, "gsm8k")
    assert (r["base_cap"], r["n"], r["n_at_cap_final"], r["ladder_state"]) == (8192, 3, 1, "escalate_pending(16384)")
    assert _qual(r) == {"ledger": "declared", "attempt_source": "ledger", "reconciliation_contract": CONTRACT_LEDGER}
    for p in g.pipeline_files(ac.LEDGER_FILE):
        p.unlink()
    # a declared ledger that is absent refuses by name — it never makes the row ledger-less (the row read only the outcomes header before fz5)
    with pytest.raises(ac.AuditRefusal, match=r"no child 'usage_ledger.jsonl' under exp/ — a GSM8K row starts one per child store"):
        _row(stores, "gsm8k")


def test_present_undeclared_keeps_the_ledger_path_and_its_source_strings(tmp_path: Path):
    stores = tmp_path / "stores"
    pair = fx.make_root(stores, leg="pair", view="s1", arms=("none",))
    fx.write_multiturn_store(pair, attack="pair", arm="none", s=1, goals=GOALS, at_cap_goals=("g0",))
    pair.flush_pipeline()
    bon = fx.make_root(stores, leg="bon", view="s1", arms=("none",))
    fx.write_bon_store(bon, arm="none", s=1, goals=GOALS)
    bon.flush_pipeline()
    undeclared = {"ledger": "present_undeclared", "attempt_source": "ledger", "reconciliation_contract": CONTRACT_LEDGER}
    r = _row(stores, "pair")
    assert r["source"] == "usage ledger finish_reason (target task rows)" and r["share_at_cap_final"] == 0.3333 and _qual(r) == undeclared
    r = _row(stores, "bon")
    assert r["source"] == "bon rows finish_reason + ledger cap" and _qual(r) == undeclared
    # the ledger path still refuses a ledger row with no record, in the ledger's words (the pin the r2 P2-7 test holds)
    lp = pair.path / ac.LEDGER_FILE
    rows = ac.read_jsonl(lp)
    _write(lp, rows + [{**rows[0], "request_id": "n0rec0rd" * 4}])
    with pytest.raises(ac.AuditRefusal, match=r"1 ledger attempt\(s\) of \('multi_turn/pair', 'petri__none'\) have NO request-evidence record"):
        _row(stores, "pair")
    # ...and a ledger row whose sample disagrees with its record: the label clause (new here, the reconciliation's own)
    _write(lp, [{**rows[0], "sample": "g9/9"}] + rows[1:])
    with pytest.raises(ac.AuditRefusal, match=r"the ledger labels the attempt task 'multi_turn/pair' sample 'g9/9' but the record says"):
        _row(stores, "pair")


def test_attempt_inventory_is_the_audits_topology(tmp_path: Path):
    """The shared reader and `request_evidence_audit.load_evidence` agree on every qualification key and on the attempt ids —
    a ledger-less static root and a GSM8K root (declared)."""
    stores = tmp_path / "stores"
    st = fx.make_root(stores, leg="pasttense", view="s1", arms=("none",))
    fx.populate_root(st, s=1, arms=("none",), n_items=3)
    g = fx.make_root(stores, leg="gsm8k", view="s1", arms=("none",))
    fx.populate_root(g, s=1, arms=("none",), n_items=2)
    for root in (st, g):
        ar = _modal(root) if root is st else ac.read_attempt_root(root.path)
        inv, ev = ac.RootReads().attempt_inventory(ar), rea.load_evidence(ar)
        assert (inv.ledger, inv.ledger_source, inv.attempt_source, inv.contract) == (ev.ledger, ev.ledger_source, ev.attempt_source, ev.contract)
        assert set(inv.by_id) == set(ev.attempts_by_id) and len(inv.rows) == len(ev.attempt_rows)
    assert ac.RootReads().attempt_inventory(ac.read_attempt_root(st.path)).qualification == LEDGERLESS


def test_contributing_roots_must_reconcile_on_one_inventory(tmp_path: Path):
    """A static base root that HAS a ledger (present_undeclared) and its rung root without one (not_on_path): one arm-run, one
    contract — refused by name rather than stamped with either."""
    stores = tmp_path / "stores"
    base = fx.make_root(stores, leg="pasttense", view="s1", arms=("none",))
    base_shard = fx.write_static_store(base, family="pasttense", arm="none", s=1, goals=GOALS, at_cap_goals=("g0",))
    base.flush_pipeline()
    rung = fx.make_root(stores, leg="pasttense", view="s1", arms=("none",), stage="8192")
    fx.write_static_rung(rung, base_shard=base_shard, family="pasttense", arm="none", s=1, cap_from=4096, cap_to=8192, still_at_cap=())
    rung.flush_pipeline()
    r = _row(stores, "pasttense")
    assert r["ladder_state"] == "exit_le_1pct" and _qual(r) == {"ledger": "present_undeclared", "attempt_source": "ledger", "reconciliation_contract": CONTRACT_LEDGER}
    _modal(rung)
    with pytest.raises(ac.AuditRefusal, match=r"pasttense/none/s1: the contributing attempt roots reconcile on different attempt inventories .* one arm-run, one contract"):
        _row(stores, "pasttense")
    _modal(base)  # both ledger-less: one contract again, the ladder read as before
    r = _row(stores, "pasttense")
    assert r["ladder_state"] == "exit_le_1pct" and r["attempt_ids"] == [base.attempt_id, rung.attempt_id] and _qual(r) == LEDGERLESS


# --------------------------------------------------------------------------- #
# the Arditi engine's per-seed partition on the response inventory                                                   #
# --------------------------------------------------------------------------- #

def test_arditi_ledgerless_per_seed_shares_and_a_record_without_extra_s_refuses(tmp_path: Path):
    stores = tmp_path / "stores"
    root = fx.make_root(stores, leg="pair", view="arditi", arms=(ac.ARDITI_CELL,), arditi_pin="a" * 64)
    for s in (1, 2, 3):
        fx.write_multiturn_store(root, attack="pair", arm=ac.ARDITI_CELL, s=s, goals=GOALS, at_cap_goals=GOALS if s == 1 else ())
    root.flush_pipeline()
    _modal(root)
    rows = {s: _row(stores, "pair", arm=ac.ARDITI_CELL, s=s, view="arditi") for s in (1, 2, 3)}
    assert [rows[s]["share_at_cap_final"] for s in (1, 2, 3)] == [1.0, 0.0, 0.0] and [rows[s]["n"] for s in (1, 2, 3)] == [6, 6, 6]
    assert all(r["rung_read"] == "fixed:4096" and _qual(r) == LEDGERLESS for r in rows.values())
    ev = root.path / ac.EVIDENCE_FILE
    victim = next(r for r in ac.read_jsonl(ev) if r.get("extra", {}).get("s") == 1)["request_id"]
    fx.rewrite_jsonl(ev, lambda r: {**r, "extra": {k: v for k, v in r["extra"].items() if k != "s"}} if r.get("request_id") == victim else r)
    with pytest.raises(ac.AuditRefusal, match=r"1 served 'multi_turn/pair' response-evidence completion\(s\) on the Arditi engine without a valid seed attribution"):
        _row(stores, "pair", arm=ac.ARDITI_CELL, s=1, view="arditi")
    with pytest.raises(ac.AuditRefusal, match="without a valid seed attribution"):  # every seed's partition refuses, none hides the row
        _row(stores, "pair", arm=ac.ARDITI_CELL, s=3, view="arditi")
    bon = fx.make_root(stores, leg="bon", view="arditi", arms=(ac.ARDITI_CELL,), arditi_pin="a" * 64)
    for s in (1, 2, 3):
        fx.write_bon_store(bon, arm=ac.ARDITI_CELL, s=s, goals=GOALS, at_cap_goals=GOALS if s == 2 else ())
    bon.flush_pipeline()
    _modal(bon)
    assert [_row(stores, "bon", arm=ac.ARDITI_CELL, s=s, view="arditi")["share_at_cap_final"] for s in (1, 2, 3)] == [0.0, 1.0, 0.0]


# --------------------------------------------------------------------------- #
# the whole table: build() + main() over a mixed wave without a synthetic ledger anywhere but GSM8K's declared children   #
# --------------------------------------------------------------------------- #

def test_full_table_build_and_main_write_at_cap_csv_over_a_mixed_ledgerless_wave(tmp_path: Path):
    stores = tmp_path / "stores"
    roots = fx.build_wave(stores, legs=ac.CONTAINER_LEGS, views=("s1",), arms=("none",), n_items=2)
    for (leg, _view), root in roots.items():
        if leg != "gsm8k":
            _modal(root)   # every non-GSM8K root the Modal way; the GSM8K root keeps its DECLARED child ledgers
    ledgers = list(stores.rglob(ac.LEDGER_FILE))
    assert ledgers and all("gsm8k" in str(p) for p in ledgers)   # the only ledgers on disk are GSM8K's declared children
    expected = fx.expected_file(tmp_path / "expected.json", arms=("none",), views=("s1",))
    rows = act.build(stores, expected=expected, petri=list(fx.PETRI_BEHAVIOURS))
    assert len(rows) == len(ac.LEGS) and {r["leg"] for r in rows} == set(ac.LEGS)
    for r in rows:
        assert r["ladder_state"] in ("fixed_cap", "exit_le_1pct", "not_applicable"), (r["leg"], r["ladder_state"])
        assert _qual(r) == ({"ledger": "declared", "attempt_source": "ledger", "reconciliation_contract": CONTRACT_LEDGER} if r["leg"] == "gsm8k" else LEDGERLESS), r["leg"]
    out = tmp_path / "tables" / "at_cap.csv"
    rc = ac.run_main(act.main, ["--stores", str(stores), "--expected", str(expected), "--out", str(out), "--petri-behaviours", ",".join(fx.PETRI_BEHAVIOURS)])
    assert rc == ac.EXIT_OK and out.is_file()
    with open(out, newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == len(ac.LEGS) and set(csv_rows[0]) == {c.rstrip("?") for c in act.COLUMNS}
    by_leg = {r["leg"]: r for r in csv_rows}
    assert by_leg["pair"]["ledger"] == "not_on_path" and by_leg["pair"]["attempt_source"] == "response_evidence" and by_leg["pair"]["reconciliation_contract"] == CONTRACT_LEDGERLESS
    assert by_leg["gsm8k"]["ledger"] == "declared" and by_leg["gsm8k"]["attempt_source"] == "ledger" and by_leg["gsm8k"]["reconciliation_contract"] == CONTRACT_LEDGER
    assert all(r["ledger"] and r["attempt_source"] and r["reconciliation_contract"] for r in csv_rows)  # never blank
    assert by_leg["petri"]["rung_read"] == "fixed:1024" and by_leg["bon"]["rung_read"] == "fixed:4096" and by_leg["xstest"]["final_cap"] == "512"


def test_plan_bound_topology_declared_ledger_missing_refuses_and_children_less_plan_reads_not_on_path(tmp_path: Path):
    stores, root, ar = _ledgerless_pair(tmp_path)

    def plan_with(children):
        row = {"standup_id": ar.production_standup_id, "leg": ar.leg, "view": ar.view, "stage": ar.stage, "shard": ar.shard, "children": children}
        return ac.LaunchPlan(path=root.path / "launch_plan.json", W=fx.W_FIXTURE, wave_uuid=fx.WAVE_UUID, rows={ar.production_standup_id: row})
    r = _row(stores, "pair", plan=plan_with([{"eval_leg": "pair", "arm": "none", "s": 1}]))
    assert r["ladder_state"] == "fixed_cap" and _qual(r) == LEDGERLESS
    with pytest.raises(ac.AuditRefusal, match=r"1 declared child ledger\(s\) missing .* a declared ledger that is absent refuses"):
        _row(stores, "pair", plan=plan_with([{"eval_leg": "pair", "arm": "none", "s": 1, "ledger": {"ledger_path_rel": "exp/x/usage_ledger.jsonl"}}]))
    # the CLI: --plan binds the same topology (a plan file whose row carries a children list, as the generator's does)
    plan_p = fx.launch_plan_file(tmp_path / "launch_plan.json", [root])
    d = json.loads(plan_p.read_text())
    for row in d["standups"]:
        row["children"] = [{"eval_leg": "pair", "arm": "none", "s": 1}]
    plan_p.write_text(json.dumps(d))
    expected = fx.expected_file(tmp_path / "expected.json", arms=("none",), views=("s1",), legs=("pair",))
    out = tmp_path / "at_cap.csv"
    assert ac.run_main(act.main, ["--stores", str(stores), "--expected", str(expected), "--out", str(out), "--plan", str(plan_p)]) == ac.EXIT_OK
    with open(out, newline="") as f:
        (row,) = list(csv.DictReader(f))
    assert row["ledger"] == "not_on_path" and row["reconciliation_contract"] == CONTRACT_LEDGERLESS
    for prow in d["standups"]:
        prow["children"] = [{"eval_leg": "pair", "arm": "none", "s": 1, "ledger": {"ledger_path_rel": "exp/x/usage_ledger.jsonl"}}]
    plan_p.write_text(json.dumps(d))
    assert ac.run_main(act.main, ["--stores", str(stores), "--expected", str(expected), "--out", str(out), "--plan", str(plan_p)]) == ac.EXIT_REFUSED
