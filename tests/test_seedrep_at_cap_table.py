"""The at-cap table on synthetic base + rung roots: the final-rung at-cap share and the rung the table reads per leg
family (static escalation.json, controls gen_stats, GSM8K outcomes header, misalign rows with the zlib loop rule,
fixed-cap legs from the ledger, MMLU not applicable), the 1 % rule imported from `ladder_decide` / `inspect_rung`, and
the pending / ceiling / no-store states named.
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
import seedrep_audit_fixtures as fx  # noqa: E402

GOALS = ["g0", "g1", "g2"]


def _roots(stores: Path):
    return ac.discover_attempt_roots(stores)


def _row(stores: Path, leg: str, arm="none", s=1, view="s1"):
    return act.row_for(leg, arm, s, view, _roots(stores), petri=list(fx.PETRI_BEHAVIOURS), trig_static=act.trigger_share_static(),
                       trig_gsm8k=act.trigger_share_gsm8k())


def test_trigger_shares_are_the_harness_rules():
    assert act.trigger_share_static() == 0.01 and act.trigger_share_gsm8k() == 0.01


def test_static_base_pending_then_rung_exit_then_ceiling(tmp_path: Path):
    stores = tmp_path / "stores"
    base = fx.make_root(stores, leg="pasttense", view="s1", arms=("none",))
    base_shard = fx.write_static_store(base, family="pasttense", arm="none", s=1, goals=GOALS, at_cap_goals=("g0",))
    base.flush_pipeline()
    r = _row(stores, "pasttense")
    assert r["applies"] is True and r["base_cap"] == 4096 and r["final_cap"] == 4096 and r["rung_read"] == "cap4096"
    assert r["n"] == 6 and r["n_at_cap_final"] == 2 and r["share_at_cap_final"] == 0.3333 and r["over_1pct"] is True
    assert r["ladder_state"] == "escalate_pending(8192)" and r["rungs_present"] == [4096] and r["source"] == "base rows at_cap"
    rung = fx.make_root(stores, leg="pasttense", view="s1", arms=("none",), stage="8192")
    rung_shard = fx.write_static_rung(rung, base_shard=base_shard, family="pasttense", arm="none", s=1, cap_from=4096, cap_to=8192,
                                      still_at_cap=())
    rung.flush_pipeline()
    r = _row(stores, "pasttense")
    assert r["final_cap"] == 8192 and r["rung_read"] == "cap8192" and r["rungs_present"] == [4096, 8192]
    assert r["n"] == 6 and r["n_at_cap_final"] == 0 and r["share_at_cap_final"] == 0.0 and r["over_1pct"] is False
    assert r["ladder_state"] == "exit_le_1pct" and r["source"] == "rung escalation.json n_at_cap_after/n_rows"
    assert r["attempt_ids"] == [base.attempt_id, rung.attempt_id]
    ceil = fx.make_root(stores, leg="pasttense", view="s1", arms=("none",), stage="16384")
    # a rung-shaped source (the 8192 store): g0's rows re-drawn again at the ceiling and still at cap (the synthetic scenario)
    fx.write_static_rung(ceil, base_shard=rung_shard, family="pasttense", arm="none", s=1, cap_from=8192, cap_to=16384,
                         regenerate=[("g0", 0), ("g0", 1)], still_at_cap=("g0",))
    ceil.flush_pipeline()
    r = _row(stores, "pasttense")
    assert r["final_cap"] == 16384 and r["share_at_cap_final"] == 0.3333 and r["ladder_state"] == "exit_ceiling"


def test_static_rung_without_escalation_json_refuses(tmp_path: Path):
    stores = tmp_path / "stores"
    base = fx.make_root(stores, leg="pasttense", view="s1", arms=("none",))
    fx.write_static_store(base, family="pasttense", arm="none", s=1, goals=GOALS)
    base.flush_pipeline()
    rung = fx.make_root(stores, leg="pasttense", view="s1", arms=("none",), stage="8192")
    fx.write_static_store(rung, family="pasttense", arm="none", s=1, goals=GOALS, cap=8192, stage_cap=8192)
    rung.flush_pipeline()
    with pytest.raises(ac.AuditRefusal, match="rung root without escalation.json"):
        _row(stores, "pasttense")


def test_controls_gen_stats_ladder(tmp_path: Path):
    stores = tmp_path / "stores"
    base = fx.make_root(stores, leg="controls", view="s1", arms=("none",))
    items = ["item0", "item1", "item2"]
    bcell = fx.write_controls_store(base, arm="none", s=1, items_by_control={"xstest": items, "or_bench_hard_1k": ["ob0", "ob1"]},
                                    mmlu_items=["mm0"], at_cap={"xstest": ["item0"]})
    base.flush_pipeline()
    r = _row(stores, "xstest")
    assert r["base_cap"] == 512 and r["share_at_cap_final"] == 0.3333 and r["ladder_state"] == "escalate_pending(2048)"
    assert r["source"] == "gen_stats.json n_at_cap/n_total"
    r2 = _row(stores, "or_bench_hard_1k")
    assert r2["share_at_cap_final"] == 0.0 and r2["ladder_state"] == "exit_le_1pct" and r2["n"] == 2
    rung = fx.make_root(stores, leg="controls", view="s1", arms=("none",), stage="2048")
    fx.write_controls_store(rung, arm="none", s=1, items_by_control={"xstest": items}, stage_cap=2048, source=bcell, regenerate={"xstest": ["item0"]})
    rung.flush_pipeline()
    r = _row(stores, "xstest")
    assert r["final_cap"] == 2048 and r["rungs_present"] == [512, 2048] and r["share_at_cap_final"] == 0.0 and r["ladder_state"] == "exit_le_1pct"
    m = _row(stores, "mmlu")
    assert m["applies"] is False and m["ladder_state"] == "not_applicable" and m["rung_read"] == "fixed:1" and m["n"] == 1


def test_gsm8k_outcomes_ladder(tmp_path: Path):
    stores = tmp_path / "stores"
    base = fx.make_root(stores, leg="gsm8k", view="s1", arms=("none",))
    fx.write_gsm8k_store(base, arm="none", s=1, items=["item0", "item1", "item2"], at_cap_items=("item1",))
    base.flush_pipeline()
    r = _row(stores, "gsm8k")
    assert r["base_cap"] == 8192 and r["n"] == 3 and r["n_at_cap_final"] == 1 and r["ladder_state"] == "escalate_pending(16384)"
    assert r["source"] == "outcomes header n_at_cap/n_attempted_samples"
    rung = fx.make_root(stores, leg="gsm8k", view="s1", arms=("none",), stage="16384")
    fx.write_gsm8k_store(rung, arm="none", s=1, items=["item0", "item1", "item2"], cap=16384, stage_cap=16384)
    rung.flush_pipeline()
    r = _row(stores, "gsm8k")
    assert r["final_cap"] == 16384 and r["share_at_cap_final"] == 0.0 and r["ladder_state"] == "exit_le_1pct" and r["rungs_present"] == [8192, 16384]


def test_misalign_loop_rule_and_rung_replacement(tmp_path: Path):
    stores = tmp_path / "stores"
    base = fx.make_root(stores, leg="misalign", view="s1", arms=("none",))
    items = {"sc0": "leaking", "sc1": "leaking", "sc2": "murder", "sc3": "leaking"}
    fx.write_misalign_store(base, arm="none", s=1, items=items, n_samples=1, at_cap_items=("sc0",), loop_items=("sc1",))
    base.flush_pipeline()
    r = _row(stores, "misalign_leaking")
    assert r["n"] == 3 and r["n_at_cap_final"] == 2 and r["share_at_cap_final"] == 0.6667
    assert r["share_truncated_eligible"] == 0.3333 and r["ladder_state"] == "escalate_pending(16384)"  # sc1 is a loop: never eligible
    assert _row(stores, "misalign_murder")["share_at_cap_final"] == 0.0 and _row(stores, "misalign_murder")["ladder_state"] == "exit_le_1pct"
    rung = fx.make_root(stores, leg="misalign", view="s1", arms=("none",), stage="16384")
    fx.write_misalign_store(rung, arm="none", s=1, items=items, n_samples=1, cap=16384, stage_cap=16384, only_pairs=[("sc0", 0)])
    rung.flush_pipeline()
    r = _row(stores, "misalign_leaking")
    assert r["final_cap"] == 16384 and r["n_at_cap_final"] == 1 and r["share_at_cap_final"] == 0.3333  # the loop stays at cap
    assert r["share_truncated_eligible"] == 0.0 and r["ladder_state"] == "exit_le_1pct"
    assert _row(stores, "misalign_murder")["rungs_present"] == [3000]  # the rung re-drew no murder pair


def test_fixed_cap_legs_read_the_ledger(tmp_path: Path):
    stores = tmp_path / "stores"
    bon = fx.make_root(stores, leg="bon", view="s1", arms=("none",))
    fx.write_bon_store(bon, arm="none", s=1, goals=GOALS, at_cap_goals=("g0",))
    bon.flush_pipeline()
    r = _row(stores, "bon")
    assert r["rung_read"] == "fixed:4096" and r["n"] == 6 and r["n_at_cap_final"] == 2 and r["share_at_cap_final"] == 0.3333
    assert r["ladder_state"] == "fixed_cap" and r["over_1pct"] is True and r["source"] == "bon rows finish_reason + ledger cap"
    mt = fx.make_root(stores, leg="pair", view="s1", arms=("none",))
    fx.write_multiturn_store(mt, attack="pair", arm="none", s=1, goals=GOALS)
    mt.flush_pipeline()
    r = _row(stores, "pair")
    assert r["rung_read"] == "fixed:4096" and r["n"] == 6 and r["share_at_cap_final"] == 0.0 and r["ladder_state"] == "fixed_cap"
    assert r["source"] == "usage ledger finish_reason (target task rows)"
    pe = fx.make_root(stores, leg="petri", view="s1", arms=("none",))
    fx.write_petri_store(pe, arm="none", s=1, scenarios={"sc0": "beh_a", "sc1": "beh_b"})
    pe.flush_pipeline()
    r = _row(stores, "petri")
    assert r["rung_read"] == "fixed:1024" and r["n"] == 4 and r["ladder_state"] == "fixed_cap"


def test_multiturn_shards_are_summed(tmp_path: Path):
    stores = tmp_path / "stores"
    for shard, goals in (("shard00", ["g0", "g1"]), ("shard01", ["g2"])):
        r = fx.make_root(stores, leg="crescendo", view="s1", arms=("none",), shard=shard)
        fx.write_multiturn_store(r, attack="crescendo", arm="none", s=1, goals=goals)
        r.flush_pipeline()
    row = _row(stores, "crescendo")
    assert row["n"] == 6 and len(row["attempt_ids"]) == 2 and row["rung_read"] == "fixed:4096"


def test_no_store_row_and_cli(tmp_path: Path):
    stores = tmp_path / "stores"
    base = fx.make_root(stores, leg="pasttense", view="s1", arms=("none",))
    base_shard = fx.write_static_store(base, family="pasttense", arm="none", s=1, goals=GOALS, at_cap_goals=("g0",))
    base.flush_pipeline()
    r = _row(stores, "pasttense", arm="prompted__si_constitution")
    assert r["ladder_state"] == "no_store" and r["share_at_cap_final"] == "no_store" and r["rung_read"] == "none" and r["attempt_ids"] == []
    expected = fx.expected_file(tmp_path / "expected.json", arms=("none", "prompted__si_constitution"), views=("s1",), legs=("pasttense",))
    out = tmp_path / "tables" / "at_cap.csv"
    rc = ac.run_main(act.main, ["--stores", str(stores), "--expected", str(expected), "--out", str(out)])
    assert rc == ac.EXIT_FAILED  # escalate_pending + no_store
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and set(rows[0]) == {c.rstrip("?") for c in act.COLUMNS}
    by_arm = {r["arm"]: r for r in rows}
    assert by_arm["none"]["ladder_state"] == "escalate_pending(8192)" and by_arm["prompted__si_constitution"]["ladder_state"] == "no_store"
    assert by_arm["none"]["share_truncated_eligible"] == ""  # the optional column is blank ONLY where the leg has no such notion
    rung = fx.make_root(stores, leg="pasttense", view="s1", arms=("none",), stage="8192")
    fx.write_static_rung(rung, base_shard=base_shard, family="pasttense", arm="none", s=1, cap_from=4096, cap_to=8192, still_at_cap=())
    rung.flush_pipeline()
    expected = fx.expected_file(tmp_path / "expected.json", arms=("none",), views=("s1",), legs=("pasttense",))
    assert ac.run_main(act.main, ["--stores", str(stores), "--expected", str(expected), "--out", str(out)]) == 0
    assert ac.run_main(act.main, ["--stores", str(tmp_path / "nope"), "--expected", str(expected), "--out", str(out)]) == ac.EXIT_REFUSED


# --------------------------------------------------------------------------- #
# P2-8: the Arditi engine's per-seed at-cap share partitions the ledger by request-evidence seed attribution           #
# --------------------------------------------------------------------------- #

def test_arditi_per_seed_at_cap_is_100_0_0_not_a_pooled_third(tmp_path: Path):
    """ONE Arditi engine, one task, one vmodel, three client-side seed bases: s1's goals all at cap, s2 / s3 none. The
    ledger filter by (task, vmodel) alone pooled the three seeds into 33.3 % each; partitioned by the request-evidence
    records' `extra.s` the shares read 1.0 / 0.0 / 0.0 over each seed's OWN rows."""
    stores = tmp_path / "stores"
    root = fx.make_root(stores, leg="pair", view="arditi", arms=(ac.ARDITI_CELL,), arditi_pin="a" * 64)
    for s in (1, 2, 3):
        fx.write_multiturn_store(root, attack="pair", arm=ac.ARDITI_CELL, s=s, goals=GOALS, at_cap_goals=GOALS if s == 1 else ())
    root.flush_pipeline()
    rows = {s: _row(stores, "pair", arm=ac.ARDITI_CELL, s=s, view="arditi") for s in (1, 2, 3)}
    assert [rows[s]["share_at_cap_final"] for s in (1, 2, 3)] == [1.0, 0.0, 0.0]
    assert [rows[s]["n"] for s in (1, 2, 3)] == [6, 6, 6] and [rows[s]["n_at_cap_final"] for s in (1, 2, 3)] == [6, 0, 0]
    assert rows[1]["over_1pct"] is True and rows[2]["over_1pct"] is False and all(r["rung_read"] == "fixed:4096" for r in rows.values())
    assert all(r["ladder_state"] == "fixed_cap" and r["attempt_ids"] == [root.attempt_id] for r in rows.values())
    # KNOWN-BAD: one s1 task record loses its extra.s → its ledger row has no seed attribution → refused, never pooled
    ev = root.path / ac.EVIDENCE_FILE
    victim = next(r for r in ac.read_jsonl(ev) if r.get("extra", {}).get("s") == 1)["request_id"]
    fx.rewrite_jsonl(ev, lambda r: {**r, "extra": {k: v for k, v in r["extra"].items() if k != "s"}} if r.get("request_id") == victim else r)
    with pytest.raises(ac.AuditRefusal, match="1 served 'multi_turn/pair' ledger row\\(s\\) on the Arditi engine without a valid seed attribution"):
        _row(stores, "pair", arm=ac.ARDITI_CELL, s=1, view="arditi")
    with pytest.raises(ac.AuditRefusal, match="without a valid seed attribution"):  # every seed's partition refuses, none hides the row
        _row(stores, "pair", arm=ac.ARDITI_CELL, s=2, view="arditi")


def test_arditi_bon_cap_reads_per_seed(tmp_path: Path):
    stores = tmp_path / "stores"
    root = fx.make_root(stores, leg="bon", view="arditi", arms=(ac.ARDITI_CELL,), arditi_pin="a" * 64)
    for s in (1, 2, 3):
        fx.write_bon_store(root, arm=ac.ARDITI_CELL, s=s, goals=GOALS, at_cap_goals=GOALS if s == 2 else ())
    root.flush_pipeline()
    rows = {s: _row(stores, "bon", arm=ac.ARDITI_CELL, s=s, view="arditi") for s in (1, 2, 3)}
    assert [rows[s]["share_at_cap_final"] for s in (1, 2, 3)] == [0.0, 1.0, 0.0] and all(r["n"] == 6 for r in rows.values())
    assert all(r["rung_read"] == "fixed:4096" for r in rows.values())


# --------------------------------------------------------------------------- #
# The record/ledger identity of a partition is a PREREQUISITE of any share read from it                               #
# --------------------------------------------------------------------------- #

def test_at_cap_refuses_a_partition_the_reconciliation_would_refuse(tmp_path: Path):
    """A request-evidence TASK record with no ledger attempt (and, separately, a ledger attempt with no record): the
    canonical reconciliation refuses both; the at-cap read used to ignore the record side (it only filtered ledger rows) and
    read a share from the torn partition. Now `audit_common.partition_identity_problems` — one routine for both tables —
    refuses the partition by name before any row is counted."""
    stores = tmp_path / "stores"
    root = fx.make_root(stores, leg="pair", view="s1", arms=("none",))
    fx.write_multiturn_store(root, attack="pair", arm="none", s=1, goals=GOALS)
    root.flush_pipeline()
    assert _row(stores, "pair")["ladder_state"] == "fixed_cap"
    ev = root.path / ac.EVIDENCE_FILE
    recs = ac.read_jsonl(ev)
    orphan = {**next(r for r in recs if r["kind"] == "task"), "request_id": "0rphan0rphan0rphan0rphan0rphan00"}
    with open(ev, "a", encoding="utf-8") as f:
        f.write(json.dumps(orphan, sort_keys=True) + "\n")
    with pytest.raises(ac.AuditRefusal, match="1 request-evidence record\\(s\\) of \\('multi_turn/pair', 'petri__none'\\) have NO ledger attempt"):
        _row(stores, "pair")
    # the reverse orphan: a ledger attempt with no record
    fx.rewrite_jsonl(ev, lambda r: None if r.get("request_id") == orphan["request_id"] else r)
    lp = root.path / ac.LEDGER_FILE
    row = {**ac.read_jsonl(lp)[0], "request_id": "n0rec0rdn0rec0rdn0rec0rdn0rec0rd"}
    with open(lp, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    with pytest.raises(ac.AuditRefusal, match="1 ledger attempt\\(s\\) of \\('multi_turn/pair', 'petri__none'\\) have NO request-evidence record"):
        _row(stores, "pair")


# --------------------------------------------------------------------------- #
# The rungs present must be the ladder's PREFIX — a gap is never final                                               #
# --------------------------------------------------------------------------- #

def test_static_ladder_gap_is_not_final(tmp_path: Path):
    """Base 4096 (g0 at cap) + a 16384 rung written by the writer's own rule from the base (cap_from 4096 → cap_to 16384)
    and NO 8192 root: the old table read the 16384 store as the final rung and, with its share below the trigger, called
    the ladder `exit_le_1pct`. The rungs present [4096, 16384] are not the ladder's prefix [4096, 8192]: `ladder_gap`, the
    CLI exits 1, and the same chain through 8192 stays final."""
    stores = tmp_path / "stores"
    base = fx.make_root(stores, leg="pasttense", view="s1", arms=("none",))
    base_shard = fx.write_static_store(base, family="pasttense", arm="none", s=1, goals=GOALS, at_cap_goals=("g0",))
    base.flush_pipeline()
    ceil = fx.make_root(stores, leg="pasttense", view="s1", arms=("none",), stage="16384")
    fx.write_static_rung(ceil, base_shard=base_shard, family="pasttense", arm="none", s=1, cap_from=4096, cap_to=16384, still_at_cap=())
    ceil.flush_pipeline()
    r = _row(stores, "pasttense")
    assert r["ladder_state"] == "ladder_gap(present [4096, 16384], ladder prefix [4096, 8192], missing [8192])"
    assert r["rungs_present"] == [4096, 16384] and r["final_cap"] == 16384 and r["share_at_cap_final"] == 0.0 and r["over_1pct"] is False
    exp = fx.expected_file(tmp_path / "expected.json", arms=("none",), views=("s1",), legs=("pasttense",))
    assert ac.run_main(act.main, ["--stores", str(stores), "--expected", str(exp), "--out", str(tmp_path / "at_cap.csv")]) == ac.EXIT_FAILED
    # the predecessor materialised → the prefix holds and the ladder is final at 16384
    mid = fx.make_root(stores, leg="pasttense", view="s1", arms=("none",), stage="8192")
    fx.write_static_rung(mid, base_shard=base_shard, family="pasttense", arm="none", s=1, cap_from=4096, cap_to=8192, still_at_cap=("g0",))
    mid.flush_pipeline()
    r = _row(stores, "pasttense")
    assert r["rungs_present"] == [4096, 8192, 16384] and r["ladder_state"] == "exit_le_1pct"


# --------------------------------------------------------------------------- #
# the controls cap is REQUIRED and held to every declaration — never a default                                         #
# --------------------------------------------------------------------------- #

def test_controls_cap_is_required_and_must_agree(tmp_path: Path):
    """`int(gs.get("max_new_tokens") or gs.get("cap") or 512)` certified 512 for a block that declared neither. Now the
    writer's `gen_stats[control].cap` is required and must equal the spool header's `gen_cfg.max_new_tokens`, the ledger's
    `max_tokens_requested` on the requests drawn on that root, and a rung root's stage — each disagreement refused by name."""
    stores = tmp_path / "stores"
    base = fx.make_root(stores, leg="controls", view="s1", arms=("none",))
    bcell = fx.write_controls_store(base, arm="none", s=1, items_by_control={"xstest": GOALS}, at_cap={"xstest": ["g0"]})
    base.flush_pipeline()
    rung = fx.make_root(stores, leg="controls", view="s1", arms=("none",), stage="2048")
    fx.write_controls_store(rung, arm="none", s=1, items_by_control={"xstest": GOALS}, stage_cap=2048, source=bcell, regenerate={"xstest": ["g0"]})
    rung.flush_pipeline()
    r = _row(stores, "xstest")
    assert r["base_cap"] == 512 and r["final_cap"] == 2048 and r["rungs_present"] == [512, 2048] and r["ladder_state"] == "exit_le_1pct"
    gsp = bcell / "gen_stats.json"
    gs = ac.read_json(gsp)
    del gs["xstest"]["cap"]                     # (a) no cap declared → refused, never 512
    gsp.write_text(json.dumps(gs))
    with pytest.raises(ac.AuditRefusal, match="the xstest block carries no cap"):
        _row(stores, "xstest")
    gs["xstest"]["cap"] = 2048                  # (b) gen_stats disagrees with the spool header
    gsp.write_text(json.dumps(gs))
    with pytest.raises(ac.AuditRefusal, match="xstest cap 2048 != the spool header's gen_cfg.max_new_tokens 512"):
        _row(stores, "xstest")
    gs["xstest"]["cap"] = 512
    gsp.write_text(json.dumps(gs))
    task = ac.CONTROLS_TASK_OF["xstest"]
    lp = base.path / ac.LEDGER_FILE             # (c) the ledger's max_tokens_requested disagrees with the store's cap
    fx.rewrite_jsonl(lp, lambda x: {**x, "max_tokens_requested": 1024} if x["task"] == task else x)
    with pytest.raises(ac.AuditRefusal, match="max_tokens_requested \\['1024'\\] on the xstest requests drawn here != the store's cap 512"):
        _row(stores, "xstest")
    fx.rewrite_jsonl(lp, lambda x: {**x, "max_tokens_requested": 512} if x["task"] == task else x)
    assert _row(stores, "xstest")["final_cap"] == 2048
    r8 = fx.make_root(stores, leg="controls", view="s1", arms=("none",), stage="8192")   # (d) a root whose stage is not its store's cap
    fx.write_controls_store(r8, arm="none", s=1, items_by_control={"xstest": GOALS}, stage_cap=2048, source=bcell, regenerate={"xstest": ["g0"]})
    r8.flush_pipeline()
    with pytest.raises(ac.AuditRefusal, match="rung stage 8192 != the store's cap 2048"):
        _row(stores, "xstest")

