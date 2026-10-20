"""The best-of-N decode precedence guard (ruling ``r2v4-reference-bon-climb-decode-precedence-vendor-preset-over-t1``;
``sisteer.registry.decode_precedence``).

The rule: within one best-of-N key -- (model, object, standup) on ``strongreject_bon`` at the cap-4,096 climb and the N = 75 headline --
the vendor-preset decode row is the row a figure reads when one exists; the T = 1 climb row (``grandfathered_T1``) is the fallback
only where no vendor-preset row exists; an unsteered anchor follows the steered row it pairs with. Pinned three ways:

* the vendor-preset rule itself, in Python and in SQL, over every ``DECODE_SOURCES`` value and the policy-id spellings of
  decode_policy.json (the derived match first, the producer's source where nothing matched, the match winning a disagreement);
* a synthetic corpus of six keys: a steered key with both decodes selected IS listed as the violation; a T = 1-only key, an unsteered
  ``none`` key with both decodes, a key whose vendor row is at another N and one whose vendor row is at another cap are NOT; a key whose
  vendor row exists but is not in the roster is listed as pending, not as a violation -- then the remedy the module generates (a
  key-scoped supersede of the violating key's T = 1 rows) is filed into the same corpus and moves exactly that one run to
  ``superseded``, after which the guard lists nothing;
* the status page carries the query, the count and the section; and the LIVE registry (read-only, skipped where absent) lists no
  violation -- the invariant the ruling states, not today's census.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3

import pytest

from sisteer.registry import db as DB
from sisteer.registry import decode_precedence as DP
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry import status
from sisteer.registry import synth
from sisteer.registry import vocab as V
from sisteer.registry.adapters import roster as RO
from sisteer.registry import reads_protocol as RP

pytestmark = pytest.mark.live_db   # this module opens the live registry (one v_paper_current read): land.sh's local step runs it, the cluster gate never does

LIVE_DB = os.path.expanduser(os.environ.get("REGISTRY_LIVE_DB", "~/registry/registry.sqlite"))   # the replica's node-local copy on Slurm
BON_ITEMS = "strongreject_bon_synth_8"
STANDUP = "synth_it"
N_ITEMS = 8


def _sha(name: str) -> str:
    return hashlib.sha256(f"synthetic object {name}".encode("utf-8")).hexdigest()


def _bon_instrument(**over):
    base = dict(eval_id=DP.EVAL_ID, item_set_id=BON_ITEMS, token_cap_tokens=DP.CLIMB_CAP_TOKENS, token_cap_ladder_json="[512, 4096]",
                bon_n=DP.HEADLINE_BON_N, decode_seed_rule="per_draw", context_window_tokens=16384)
    base.update(over)
    return synth.instrument_fields(**base)


T1_INSTRUMENT = _bon_instrument(decode_source=DP.T1_DECODE_SOURCE, decode_temperature=1.0)
VENDOR_INSTRUMENT = _bon_instrument(decode_source="model_card", decode_temperature=0.6, decode_top_p=0.95, decode_top_k=20)
VENDOR_N64_INSTRUMENT = _bon_instrument(decode_source="model_card", decode_temperature=0.6, decode_top_p=0.95, decode_top_k=20, bon_n=64)
VENDOR_CAP512_INSTRUMENT = _bon_instrument(decode_source="model_card", decode_temperature=0.6, decode_top_p=0.95, decode_top_k=20,
                                           token_cap_tokens=512, token_cap_ladder_json="[512]")
INSTRUMENTS = (T1_INSTRUMENT, VENDOR_INSTRUMENT, VENDOR_N64_INSTRUMENT, VENDOR_CAP512_INSTRUMENT)

# (paper_label, object_id, role): the steered objects of the corpus beside synth's own constitution_lora and none
OBJECTS = (("control_lora", _sha("control"), "control"), ("safety_lora", _sha("safety"), "safety"),
           ("difference_lora", _sha("difference"), "difference"), ("other", _sha("other"), "irrelevant"))


# --------------------------------------------------------------------------------------------
# the vendor-preset rule, in Python and in SQL
# --------------------------------------------------------------------------------------------

CASES = [
    # (decode_source, decode_policy_id, kind)
    ("grandfathered_T1", None, "t1"),
    ("grandfathered_T1", "", "t1"),
    ("grandfathered_T1", "qwen35_4b_it|bon|off|grandfathered_T1@v1", "t1"),
    ("model_card", "qwen35_4b_it|bon|off|vendor_preset@v1", "vendor"),
    ("generation_config", "llama8b_it|bon|not_available|vendor_preset@v1", "vendor"),
    ("generation_config", "gemma4e4b_it|bon|off|vendor_preset@v2", "vendor"),
    ("model_card", None, "vendor"),
    ("generation_config", "", "vendor"),
    ("vendor_recommendation", None, "vendor"),
    ("unverified", None, "other"),
    ("vllm_default", None, "other"),
    ("greedy", None, "other"),
    ("runner_forced_T1", None, "other"),
    ("legacy_contracts_row", None, "other"),
    # the derived match wins a disagreement with the producer's word, both ways
    ("generation_config", "gemma4e4b_it|bon|off|grandfathered_T1@v1", "other"),
    ("grandfathered_T1", "gemma4e4b_it|bon|off|vendor_preset@v1", "vendor"),
    # a vendor source under a non-best-of-N policy is not a vendor preset either: the id decides
    ("generation_config", "gemma4e4b_it|agentic|off|runner_T1@v1", "other"),
]


def test_vendor_preset_rule_in_python_and_sql_agree_on_every_case():
    assert set(DP.VENDOR_DECODE_SOURCES) < set(V.DECODE_SOURCES) and DP.T1_DECODE_SOURCE in V.DECODE_SOURCES
    conn = DB.connect_memory()
    conn.execute("CREATE TABLE r (decode_source TEXT, decode_policy_id TEXT)")
    for src, pid, _ in CASES:
        conn.execute("INSERT INTO r VALUES (?, ?)", (src, pid))
    got = [row[0] for row in conn.execute(f"SELECT {DP.decode_kind_sql('r')} FROM r").fetchall()]
    assert got == [kind for _, _, kind in CASES], list(zip(CASES, got))
    for src, pid, kind in CASES:
        assert DP.decode_kind(src, pid) == kind, (src, pid)
        assert DP.is_vendor_preset(src, pid) == (kind == "vendor"), (src, pid)
    # every controlled source is classified, and only the three vendor ones read vendor when nothing matched
    for src in V.DECODE_SOURCES:
        assert DP.decode_kind(src, None) == ("vendor" if src in DP.VENDOR_DECODE_SOURCES else "t1" if src == DP.T1_DECODE_SOURCE else "other")


def test_every_best_of_n_policy_of_the_shipped_table_is_classified_as_its_source_says():
    """decode_policy.json: the ``|bon|`` vendor_preset rows carry a vendor source and read vendor; the grandfathered rows read t1."""
    pkg = os.path.dirname(DP.__file__)
    with open(os.path.join(pkg, "decode_policy.json"), "r", encoding="utf-8") as fh:
        pols = [p for p in json.load(fh)["policies"] if "|bon|" in str(p.get("policy_ref", ""))]
    assert len(pols) >= 10
    for p in pols:
        kind = DP.decode_kind(p["decode_source"], p["policy_id"])
        if DP.VENDOR_POLICY_MARK in p["policy_id"]:
            assert kind == "vendor" and p["decode_source"] in DP.VENDOR_DECODE_SOURCES, p["policy_id"]
        else:
            assert kind == "t1" and p["decode_source"] == DP.T1_DECODE_SOURCE, p["policy_id"]


# --------------------------------------------------------------------------------------------
# the synthetic corpus
# --------------------------------------------------------------------------------------------

def _write(root, sub, name, recs):
    return R.write_records(os.path.join(root, "records", sub, name), recs)


def _dimensions(i0=30):
    recs = [synth.rec("eval", {"eval_id": DP.EVAL_ID, "eval_category": "best_of_n", "headline_metric_id": "asr",
                               "metric_direction": "lower_is_better", "attacker_required": 0, "paper_status": "in_paper"}, synth.t(i0)),
            synth.rec("item_set", {"item_set_id": BON_ITEMS, "eval_id": DP.EVAL_ID, "dataset_version": "synthetic/strongreject_bon@v1",
                                   "n_items": N_ITEMS, "items_sha256": synth.SHA256_B, "source_uri": "synthetic://items/strongreject_bon_8.jsonl",
                                   "item_id_scheme": "q<index>", "subsample_rule": "full"}, synth.t(i0 + 1))]
    for k, inst in enumerate(INSTRUMENTS):
        recs.append(synth.rec("instrument", inst, synth.t(i0 + 2 + k)))
    for k, (label, oid, role) in enumerate(OBJECTS):
        recs.append(synth.rec("object", synth.object_lora(sha=oid, paper_label=label, role=role), synth.t(i0 + 10 + k)))
    return recs


def _run(root, key, i0, object_id, instrument, **over):
    """One judged run on the best-of-N eval under ``instrument``, at the standup, its decode source the instrument's."""
    fields = dict(eval_id=DP.EVAL_ID, item_set_id=BON_ITEMS, instrument_id=instrument["instrument_id"], decode_source=instrument["decode_source"],
                  token_cap_tokens=instrument["token_cap_tokens"], view_legacy=STANDUP, n_items_planned=N_ITEMS)
    fields.update(over)
    obj_key = "none" if object_id == "none" else f"{object_id[:8]}@synthetic"
    rid, recs = synth.run_records(key, i0, obj_key, object_id, n_items=N_ITEMS, **fields)
    synth.write_run_dir(root, synth.CAMPAIGN, rid, [(str(r["record_type"]), [r]) for r in recs])
    return rid


def _ruling(rid, action, scope, i, target=None, ruled_utc=None):
    raw = {"ruling_id": rid, "round_id": "r_synth", "action": action, "scope_kind": "key", "scope_json": scope, "target_json": target or {},
           "reason": "synthetic selection for the decode precedence guard", "ruled_utc": ruled_utc or synth.t(i), "ruled_by": "tests",
           "source_ref": "tests/test_registry_bon_decode_precedence.py",
           "provenance": {"kind": "user_direct", "message_utc": "2026-09-28T17:25:00Z"}}
    return synth.rec("ruling", RO.normalise_ruling(raw), synth.t(i))


def _ingest(root):
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, (rep.refused, rep.blocked)


def _selection(conn):
    return {r[0]: r[1] for r in conn.execute("SELECT run_id, selection_state FROM v_paper_current").fetchall()}


@pytest.fixture
def world(tmp_path):
    """Six keys at the standup, every run judged; an include ruling on run ids selects what each case needs:
    A constitution_lora: T = 1 + vendor, both selected (THE violation);  B control_lora: T = 1 only;  C none: T = 1 + vendor (an anchor);
    D safety_lora: T = 1 selected, vendor not covered (not_in_roster: pending);  E difference_lora: T = 1 + vendor at N 64;
    F other: T = 1 + vendor at cap 512."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    dims = synth.seed_dimensions(0) + _dimensions(30)
    R.write_records(os.path.join(root, "records", "dimensions", "seed", "000001-dimensions.jsonl"), dims)
    ids = {}
    i = 100
    ids["A_t1"] = _run(root, "A_t1", i, synth.OBJECT_SHA, T1_INSTRUMENT); i += 20
    ids["A_vendor"] = _run(root, "A_vendor", i, synth.OBJECT_SHA, VENDOR_INSTRUMENT); i += 20
    ids["B_t1"] = _run(root, "B_t1", i, _sha("control"), T1_INSTRUMENT); i += 20
    ids["C_t1"] = _run(root, "C_t1", i, "none", T1_INSTRUMENT); i += 20
    ids["C_vendor"] = _run(root, "C_vendor", i, "none", VENDOR_INSTRUMENT); i += 20
    ids["D_t1"] = _run(root, "D_t1", i, _sha("safety"), T1_INSTRUMENT); i += 20
    ids["D_vendor"] = _run(root, "D_vendor", i, _sha("safety"), VENDOR_INSTRUMENT); i += 20
    ids["E_t1"] = _run(root, "E_t1", i, _sha("difference"), T1_INSTRUMENT); i += 20
    ids["E_vendor_n64"] = _run(root, "E_vendor_n64", i, _sha("difference"), VENDOR_N64_INSTRUMENT); i += 20
    ids["F_t1"] = _run(root, "F_t1", i, _sha("other"), T1_INSTRUMENT); i += 20
    ids["F_vendor_cap512"] = _run(root, "F_vendor_cap512", i, _sha("other"), VENDOR_CAP512_INSTRUMENT); i += 20
    included = [v for k, v in ids.items() if k != "D_vendor"]
    _write(root, "rulings_synth", "000001-include.jsonl",
           [_ruling("synth-include-bon-keys", "include", {"match": {"run_ids": sorted(included)}}, 9000)])
    _ingest(root)
    return root, ids


def test_synthetic_keys_list_the_violation_and_the_pending_key_and_nothing_else(world):
    root, ids = world
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        sel = _selection(conn)
        assert sel[ids["A_t1"]] == sel[ids["A_vendor"]] == "selected" and sel[ids["D_vendor"]] == "not_in_roster", sel
        assert all(sel[v] == "selected" for k, v in ids.items() if k != "D_vendor"), sel
        rows = DP.precedence_rows(conn)
        assert [(r["state"], r["paper_label"]) for r in rows] == [(DP.VIOLATION_STATE, "constitution_lora"), (DP.PENDING_STATE, "safety_lora")], rows
        a, d = rows
        assert (a["model_id"], a["object_id"], a["view_legacy"]) == (synth.MODEL_ID, synth.OBJECT_SHA, STANDUP)
        assert json.loads(a["t1_run_ids_json"]) == [ids["A_t1"]] and json.loads(a["vendor_run_ids_json"]) == [ids["A_vendor"]]
        assert json.loads(a["t1_instrument_ids_json"]) == [T1_INSTRUMENT["instrument_id"]]
        assert json.loads(a["vendor_decode_sources_json"]) == ["model_card"] and json.loads(a["vendor_decode_policy_ids_json"]) == [""]
        assert (a["n_t1_selected"], a["n_vendor_selected"], a["n_vendor_not_in_roster"]) == (1, 1, 0)
        assert (d["n_t1_selected"], d["n_vendor_selected"], d["n_vendor_not_in_roster"]) == (1, 0, 1)
        assert json.loads(d["vendor_run_ids_json"]) == [ids["D_vendor"]]
        assert [r["object_id"] for r in DP.violations(conn)] == [synth.OBJECT_SHA]
        # the census sees every row the guard reads, the none anchors included, and nothing at another N or cap
        cen = DP.census(conn)
        assert sum(int(c["n_runs"]) for c in cen) == 9  # A x2, B, C x2, D x2, E t1, F t1
        assert sum(int(c["n_runs"]) for c in cen if not c["steered"]) == 2
        kinds = {(c["decode_source"], c["decode_kind"]) for c in cen}
        assert kinds == {(DP.T1_DECODE_SOURCE, "t1"), ("model_card", "vendor")}, kinds
    finally:
        conn.close()


def test_the_generated_supersede_moves_exactly_the_violating_t1_row_and_the_guard_then_lists_nothing(world):
    root, ids = world
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        before = _selection(conn)
        [row] = DP.violations(conn)
        ruling = DP.supersede_ruling(row, ruled_utc=synth.t(9500), ruled_by="tests", round_id="r_synth")
        scope = ruling["scope_json"]
        assert ruling["action"] == "supersede" and ruling["scope_kind"] == "key" and ruling["round_id"] == "r_synth"
        assert ruling["ruling_id"].startswith("r2v4-supersede-bon-t1-") and ruling["ruling_id"].endswith("-by-vendor-preset")
        assert re.fullmatch(r"[a-z0-9_-]+", ruling["ruling_id"]), ruling["ruling_id"]
        assert scope["match"] == {"model_ids": [synth.MODEL_ID], "object_ids": [synth.OBJECT_SHA], "eval_ids": [DP.EVAL_ID],
                                  "decode_sources": [DP.T1_DECODE_SOURCE], "token_caps": [DP.CLIMB_CAP_TOKENS],
                                  "instrument_ids": [T1_INSTRUMENT["instrument_id"]], "views_legacy": [STANDUP]}
        assert ruling["target_json"]["superseded_by_run_ids"] == [ids["A_vendor"]] and ruling["target_json"]["rule"] == DP.RULING_ID
        assert DP.RULING_ID in ruling["reason"]
        # the key's fields reach exactly the one run the guard listed (a direct read of `run`, selected or not)
        assert DP.runs_in_scope(conn, scope) == [ids["A_t1"]]
        normalized = RO.normalise_ruling(dict(ruling, provenance={"kind": "user_direct", "message_utc": "2026-09-28T17:25:00Z"}))
    finally:
        conn.close()
    _write(root, "rulings_synth", "000002-supersede.jsonl", [synth.rec("ruling", normalized, synth.t(9500))])
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        after = _selection(conn)
        # the supersede is now the LATEST ruling covering the T = 1 run, so the include no longer selects it and it leaves the paper; the
        # vendor row stands. The label a roster-less corpus gives the run is `not_in_roster` (v_paper_current tests roster membership
        # before it spells `superseded`); a live T = 1 row is roster-matched and reads `superseded`. What the remedy must do is asserted
        # on the flag and the decisive ruling, not on that label.
        t1 = conn.execute("SELECT selection_state, in_current_paper, decisive_ruling_id, decisive_action FROM v_paper_current WHERE run_id = ?",
                          (ids["A_t1"],)).fetchone()
        assert tuple(t1) == (after[ids["A_t1"]], 0, ruling["ruling_id"], "supersede") and after[ids["A_t1"]] != "selected", tuple(t1)
        vendor = conn.execute("SELECT selection_state, in_current_paper FROM v_paper_current WHERE run_id = ?", (ids["A_vendor"],)).fetchone()
        assert tuple(vendor) == ("selected", 1)
        changed = {k for k in before if before[k] != after[k]}
        assert changed == {ids["A_t1"]}, changed
        assert DP.violations(conn) == []
        # the pending key is untouched: still listed as pending, never as a violation
        assert [(r["state"], r["paper_label"]) for r in DP.precedence_rows(conn)] == [(DP.PENDING_STATE, "safety_lora")]
    finally:
        conn.close()


def test_status_page_carries_the_query_the_count_and_the_section(world, tmp_path, monkeypatch):
    monkeypatch.setenv(RP.STATE_ENV, str(tmp_path / "landing-state"))      # the page reads the landing state: an empty per-test one, never the host's
    root, _ids = world
    assert status.QUERIES["bon_decode_precedence"] == DP.PRECEDENCE_SQL
    ctx = status.status_json(root)
    assert len(ctx["bon_decode_precedence_violations"]) == 1 and len(ctx["bon_decode_precedence_pending"]) == 1
    assert ctx["bon_decode_precedence_ruling_id"] == DP.RULING_ID and "bon_decode_precedence" in ctx["timings_ms"]
    out = str(tmp_path / "index.html")
    status.render(root, out_path=out, register=False)
    html = open(out, encoding="utf-8").read()
    assert "Best-of-N decode precedence" in html and DP.RULING_ID in html and "both_selected" in html and "constitution_lora" in html
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\b\w*_legacy\b", " ", text)
    for word in ("arm", "leg", "lane", "cell", "wave", "grader"):
        assert not re.search(r"(?<![A-Za-z0-9_])" + word + r"(?![A-Za-z0-9_])", text, re.IGNORECASE), word


# --------------------------------------------------------------------------------------------
# the live registry: the invariant the ruling states
# --------------------------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(LIVE_DB), reason="no live registry on this host")
def test_live_registry_no_steered_key_reads_t1_beside_a_selected_vendor_preset():
    """Read-only against the live registry (~1 min: the query reads v_paper_current). A listed violation is a key whose T = 1 climb
    row is still read although a vendor-preset row is selected beside it -- the remedy is a key-scoped supersede of that key's T = 1
    rows (decode_precedence.supersede_ruling; dry run first with the supersede generator, which prints the scope and the run ids it
    covers), never a change to this test."""
    with DB.reading(LIVE_DB) as conn:
        conn.execute("PRAGMA busy_timeout=120000")
        bad = DP.violations(conn)
    assert bad == [], (f"ruling {DP.RULING_ID} (D581): {len(bad)} best-of-N key(s) carry a selected T = 1 row beside a selected vendor-preset "
                       f"row; file a key-scoped supersede of each key's T = 1 rows (decode_precedence.supersede_ruling(row), after a dry run "
                       f"that prints the scope and the run ids it covers)", bad)
