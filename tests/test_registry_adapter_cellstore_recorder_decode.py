"""The cell-store reader's decode columns from the store-level recorder (`<store root>/resolved_sampler.json`; commit 27, corrected by 27b).

A thinking_card / greedy-gen cell applies its sampler SERVER-side: the manifest's `resolved_config.gen` carries no sampler numbers, so the
reader takes the recorder's evidence. 27b: SENT = the client's actual sent keys from the client REQUEST evidence
only — the client rows' recorded sampler when the driver recorded it, else the producer's `--expect` reference (the row the client builds
its requests from) corroborated by the AGREES verdict — never the engine's resolved defaults; RESOLVED = the client class's one
server-resolved sampler, in `decode_resolved_json`; with no sent provenance, sent stays {} and the reason is said. The recorder is a
per-store revision INPUT (aliased `<recorder>#<store dir>`), so a store harvested before its recorder exists re-reads, with no force, the
pass the recorder lands."""
from __future__ import annotations

import json
import os
import sys

import pytest

from sisteer.registry import db as DB
from sisteer.registry import decode_evidence as DE
from sisteer.registry import harvest as H
from sisteer.registry.adapters import cellstore as CS
from sisteer.registry.adapters import runner as RUN

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registry_items_fixtures as F  # noqa: E402
from test_registry_decode_match import _ingest as dm_ingest, _policy as dm_policy, _row as dm_row, _run as dm_run  # noqa: E402
from sisteer.registry import synth  # noqa: E402

NOW = "2026-09-30T22:00:00Z"
RESOLVED = {"min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0, "temperature": 1.0, "top_k": 64, "top_p": 0.95}
CARD = {"temperature": 1.0, "top_k": 64.0, "top_p": 0.95}            # the producer's --expect reference: the thinking card row the client sends
SENT_CARD = {"temperature": 1.0, "top_k": 64, "top_p": 0.95}
#: the served driver's tree as a store's `manifest.json:producer.tree_sha` records it -- the run's `code_sha` under a root that reads the block
#: (`StoreRoot.producer_tree_sha`), and the pin a DECLARED recorder reference derives the sent set against (synthetic, forty hex)
TREE = "a1b2c3d4e" + "f" * 31


def _pin(store: str, tree_sha) -> None:
    """Give the fixture store the served driver's producer block (`manifest.json:producer.tree_sha`), the shape every store since the block
    was added carries; None leaves the LEGACY shape (no block: `code_sha` reads `unverified`)."""
    if tree_sha is None:
        return
    mp = os.path.join(store, "manifest.json")
    with open(mp) as fh:
        man = json.load(fh)
    man["producer"] = {"driver": "served_capability_run", "tree_sha": tree_sha}
    with open(mp, "w") as fh:
        json.dump(man, fh)


def _recorder(view_dir: str, verdict: str, samplers=(RESOLVED,), *, reference=CARD, reference_source="--expect", rows_sent=None, n_rows=0) -> str:
    """The recorder document in the live shape: `classes.client` (the one resolved sampler, the reference it was compared against), the
    top-level `sent` block (the client rows' recorded sampler, empty when the driver records none)."""
    client = {"n": 200, "verdict": verdict, "distinct_resolved": [{"sampler": dict(s), "n": 200 // len(samplers)} for s in samplers],
              "reference_kind": "expect" if reference_source == "--expect" else "rows", "reference": dict(reference) if reference else None,
              "reference_source": reference_source}
    sent = {"n_rows": n_rows, "distinct": [{"sampler": dict(s), "n": n_rows} for s in (rows_sent or [])], "distinct_max_tokens": []}
    doc = {"verdict": verdict, "why": "synthetic", "reference": dict(reference) if reference else None, "reference_source": reference_source, "sent": sent,
           "classes": {"client": client}}
    p = os.path.join(view_dir, "resolved_sampler.json")
    with open(p, "w") as fh:
        json.dump(doc, fh)
    return p


def _read(store, *, pinned: bool = True):
    """The reader over one store; ``pinned`` = a root that reads `manifest.json:producer.tree_sha` as the run's `code_sha` (the default here:
    the derived recorder claims below need the pin), False = every production root today (`code_sha` reads `unverified`)."""
    reader = CS.StoreReader(F.dims_map(), "synthcampaign", NOW, producer_tree_sha=pinned)
    return reader.read(store, CS.store_inputs(store), view=os.path.basename(os.path.dirname(os.path.dirname(store))))


def _runs(sr):
    return {p["eval_id"]: p for rt, p in sr.records if rt == "run"}


def _store(tmp_path, *, tree_sha=TREE):
    view = str(tmp_path / "root" / "synthmodel_u0"); os.makedirs(view)
    store = F.write_controls_store(view, "none", n=3, sampled=False)              # greedy gen block: no sampler numbers, like a thinking_card cell
    _pin(store, tree_sha)
    return view, store


def test_a_partial_key_agreeing_recorder_files_the_reference_keys_as_sent_and_the_six_as_resolved(tmp_path, capsys):
    view, store = _store(tmp_path); rec = _recorder(view, "AGREES")
    runs = _runs(_read(store))
    for e, r in runs.items():
        assert r["code_sha"] == TREE and r["field_confidence_json"]["code_sha"]["confidence"] == "recorded"      # the pin: the row's own forty-hex code sha
        assert r["decode_sent_json"] == SENT_CARD and r["decode_resolved_json"] == RESOLVED, (e, r["decode_sent_json"], r["decode_resolved_json"])
        assert r["decode_temperature_sent"] == 1.0 and r["decode_top_p_sent"] == 0.95 and r["decode_top_k_sent"] == 64
        fc = r["field_confidence_json"]
        # the reference is a DECLARATION (sent_rows 0): `derived` against the pinned request builder at this run's code_sha, never `recorded`
        # (the registry owner's Change 1, 2026-10-13 18:57 PT; the shared reader decode_evidence applies the same rule)
        assert fc["decode_sent_json"]["confidence"] == "derived" and "--expect reference" in fc["decode_sent_json"]["source"] and "AGREES" in fc["decode_sent_json"]["source"]
        assert DE.RECORDER_REQUEST_BUILDER in fc["decode_sent_json"]["source"] and f"code_sha {TREE}" in fc["decode_sent_json"]["source"] and "sent_rows 0" in fc["decode_sent_json"]["source"]
        assert fc["decode_resolved_json"]["confidence"] == "recorded" and rec in fc["decode_resolved_json"]["source"]
        assert "unfiled_declared_reference" not in r["extra"]["decode"]
    assert "decode_sent_json" not in capsys.readouterr().err


def test_a_declared_reference_without_a_forty_hex_run_sha_files_no_sent_column_and_says_why(tmp_path, capsys):
    """Review round 3 (Major 1): a DECLARED reference derives the sent set against the pinned request builder AT THE RUN'S CODE SHA -- the
    column the row carries. A legacy store without `manifest.producer.tree_sha` files `code_sha` `unverified`, and so does any store under a
    root that does not read the block (every production root today): there the sent columns stay EMPTY with the reason, the server-resolved
    sampler stays `recorded`, and nothing reads `derived ... at this run's code_sha unverified` (the shared reader decode_evidence applies the
    same rule: `_derive_from_reference` files nothing without a forty-hex sha). Before the fix the adapter filed the five typed / JSON sent
    columns `derived` at `code_sha unverified` -- a derivation naming no builder commit."""
    # (a) the legacy shape: no producer block -> code_sha unverified -> no sent column, the reason, the resolved sampler recorded
    view, store = _store(tmp_path / "legacy", tree_sha=None); rec = _recorder(view, "AGREES")
    sr = _read(store); runs = _runs(sr)
    for e, r in runs.items():
        assert r["code_sha"] == "unverified", e
        assert r["decode_sent_json"] == {} and r["decode_temperature_sent"] is None and r["decode_top_p_sent"] is None and r["decode_top_k_sent"] is None, (e, r["decode_sent_json"])
        assert r["decode_resolved_json"] == RESOLVED
        fc = r["field_confidence_json"]
        assert fc["decode_sent_json"]["confidence"] == "unverified", fc["decode_sent_json"]
        why = fc["decode_sent_json"]["source"]
        assert "DECLARATION" in why and "sent_rows 0" in why and DE.RECORDER_REQUEST_BUILDER in why and "code_sha is 'unverified'" in why and "nothing filed for the sent columns" in why, why
        assert fc["decode_resolved_json"]["confidence"] == "recorded" and rec in fc["decode_resolved_json"]["source"]
        # the declaration the recorder carries rides the extra, unfiled -- the columns say nothing
        unfiled = r["extra"]["decode"]["unfiled_declared_reference"]
        assert unfiled["sent"] == SENT_CARD and unfiled["reference_source"] == "--expect" and unfiled["sent_rows"] == 0 and unfiled["request_builder"] == DE.RECORDER_REQUEST_BUILDER
        assert r["extra"]["decode"]["sent"] == {} and r["extra"]["decode"]["sent_confidence"] is None
    assert sr.unverified.get("decode_sent_json", 0) >= len(runs)
    err = capsys.readouterr().err
    assert "left empty" in err and "DECLARATION" in err and "resolved_sampler.json" in err, err
    # (b) a PINNED manifest under a root that does NOT read producer.tree_sha: the row's own code_sha column decides (unverified) -- never a
    # sha read behind the row's back
    view_b, store_b = _store(tmp_path / "pinned_unread"); _recorder(view_b, "AGREES")
    r = next(iter(_runs(_read(store_b, pinned=False)).values()))
    assert r["code_sha"] == "unverified" and r["decode_sent_json"] == {} and r["decode_temperature_sent"] is None and r["decode_resolved_json"] == RESOLVED
    assert r["field_confidence_json"]["decode_sent_json"]["confidence"] == "unverified" and "code_sha is 'unverified'" in r["field_confidence_json"]["decode_sent_json"]["source"]
    # (c) the client rows' RECORDED sampler is an OBSERVATION: filed `recorded` with no pin at all, legacy shape and unread root alike
    view_c, store_c = _store(tmp_path / "rows", tree_sha=None)
    _recorder(view_c, "AGREES", rows_sent=[dict(RESOLVED)], n_rows=200)
    r = next(iter(_runs(_read(store_c, pinned=False)).values()))
    assert r["code_sha"] == "unverified" and r["decode_sent_json"] == RESOLVED and r["decode_temperature_sent"] == 1.0
    assert r["field_confidence_json"]["decode_sent_json"]["confidence"] == "recorded" and "client rows' recorded sampler" in r["field_confidence_json"]["decode_sent_json"]["source"]
    # (d) the unit behind it: a derived set with no forty-hex sha is emptied in place with the reason; a forty-hex sha, or a recorded set, is left alone
    d = {"sent": dict(SENT_CARD), "sent_source": "x", "sent_confidence": DE.SENT_DERIVED, "recorder_note": None, "reference_source": "--expect", "sent_rows": 0}
    for bad in ("unverified", None, "", "abc", "A" * 40, TREE[:39], TREE + "0"):
        dd = dict(d); CS.StoreReader._pin_derived_sent(dd, bad)
        assert dd["sent"] == {} and dd["sent_source"] is None and dd["sent_confidence"] is None and "DECLARATION" in dd["recorder_note"] and repr(bad) in dd["recorder_note"], (bad, dd)
        assert dd["unfiled_declared_reference"]["sent"] == SENT_CARD
    dd = dict(d); CS.StoreReader._pin_derived_sent(dd, TREE)
    assert dd["sent"] == SENT_CARD and dd["sent_confidence"] == DE.SENT_DERIVED and dd["recorder_note"] is None and "unfiled_declared_reference" not in dd
    dr = dict(d, sent_confidence=DE.SENT_RECORDED); CS.StoreReader._pin_derived_sent(dr, "unverified")
    assert dr["sent"] == SENT_CARD and dr["sent_confidence"] == DE.SENT_RECORDED and dr["recorder_note"] is None


def test_the_partial_key_sent_set_matches_its_real_policy_exactly_and_a_policy_pinning_the_defaults_as_sent_does_not(tmp_path):
    view, store = _store(tmp_path); _recorder(view, "AGREES")
    sent = next(iter(_runs(_read(store)).values()))["decode_sent_json"]
    assert sent == SENT_CARD
    root = str(tmp_path / "reg")
    # the thinking card's real policy: six values pinned, min_p and the penalties listed NOT SENT (the engine resolves them)
    card = dm_policy("p_card", "synthmodel/gsm8k/thinking_card", eval_id=synth.EVAL_ID, temperature=1.0, top_p=0.95, top_k=64, min_p=0.0, repetition_penalty=1.0,
                     presence_penalty=0.0, not_sent={"min_p": "0.0 (engine default)", "repetition_penalty": "1.0 (engine default)", "presence_penalty": "0.0 (engine default)"})
    # a policy that pins the same six as SENT: the run did not send three of them -> not this policy
    strict = dm_policy("p_strict", "synthmodel/gsm8k/strict", temperature=1.0, top_p=0.95, top_k=64, min_p=0.0, repetition_penalty=1.0, presence_penalty=0.0)
    run = dm_run("card", sent, i0=100)
    dm_ingest(root, [card, strict], [run])
    conn = DB.connect(DB.db_path(root), readonly=True)
    row, view_row = dm_row(conn, run[0])
    assert (row["decode_policy_id"], row["decode_policy_match"]) == ("p_card", "exact"), (row, view_row)
    assert (view_row["match"], json.loads(view_row["candidates_json"])) == ("exact", ["p_card"])      # the strict policy pins keys the run did not send
    # the OLD reading — the six resolved keys filed as sent — read as a run that SENT the defaults: the strict policy became a candidate
    # too and the view read AMBIGUOUS (a tie broken by scope, the loser recorded beside the stamp)
    old = dm_run("old", RESOLVED, i0=200)
    dm_ingest(str(tmp_path / "reg_old"), [card, strict], [old])
    conn2 = DB.connect(DB.db_path(str(tmp_path / "reg_old")), readonly=True)
    _, view2 = dm_row(conn2, old[0])
    assert (view2["match"], json.loads(view2["candidates_json"])) == ("ambiguous", ["p_card", "p_strict"]), view2
    conn.close(); conn2.close()


def test_client_rows_with_one_recorded_sampler_are_the_sent_evidence_over_the_reference(tmp_path):
    view, store = _store(tmp_path)
    _recorder(view, "AGREES", rows_sent=[{"temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0}], n_rows=200)
    r = next(iter(_runs(_read(store)).values()))
    assert r["decode_sent_json"] == RESOLVED and r["decode_resolved_json"] == RESOLVED and "client rows' recorded sampler" in r["field_confidence_json"]["decode_sent_json"]["source"]
    assert r["field_confidence_json"]["decode_sent_json"]["confidence"] == "recorded"       # an observation per request, not a declaration
    # two distinct recorded samplers: not one sent set — sent stays empty, the resolved sampler is still recorded
    _recorder(view, "AGREES", rows_sent=[dict(RESOLVED), dict(RESOLVED, temperature=0.7)], n_rows=200)
    sr = _read(store); r = next(iter(_runs(sr).values()))
    assert r["decode_sent_json"] == {} and r["decode_resolved_json"] == RESOLVED and "2 distinct sent samplers" in r["field_confidence_json"]["decode_sent_json"]["source"]
    assert sr.unverified.get("decode_sent_json", 0) >= 1


def test_without_rows_or_a_reference_sent_stays_empty_while_the_resolved_sampler_is_recorded(tmp_path, capsys):
    view, store = _store(tmp_path); _recorder(view, "AGREES", reference=None, reference_source=None)
    sr = _read(store); r = next(iter(_runs(sr).values()))
    assert r["decode_sent_json"] == {} and r["decode_temperature_sent"] is None and r["decode_resolved_json"] == RESOLVED
    assert r["field_confidence_json"]["decode_sent_json"]["confidence"] == "unverified" and "neither the client rows nor a reference" in r["field_confidence_json"]["decode_sent_json"]["source"]
    assert r["field_confidence_json"]["decode_resolved_json"]["confidence"] == "recorded"
    assert "left empty" in capsys.readouterr().err
    # a reference key that did NOT resolve to its value despite AGREES: refused as sent evidence, by name
    _recorder(view, "AGREES", reference={"temperature": 0.7, "top_p": 0.95, "top_k": 64})
    r = next(iter(_runs(_read(store)).values()))
    assert r["decode_sent_json"] == {} and "['temperature']" in r["field_confidence_json"]["decode_sent_json"]["source"]


def test_no_recorder_leaves_both_columns_empty_and_unrecorded(tmp_path):
    view, store = _store(tmp_path)
    sr = _read(store); runs = _runs(sr)
    for r in runs.values():
        assert r["decode_sent_json"] == {} and r["decode_resolved_json"] == {} and r["decode_temperature_sent"] is None
        assert r["field_confidence_json"]["decode_sent_json"]["confidence"] == "unverified" and "decode_resolved_json" not in r["field_confidence_json"]
    assert sr.unverified.get("decode_sent_json", 0) >= len(runs)


@pytest.mark.parametrize("verdict", ["DISAGREES", "UNCLASSIFIED_REQUESTS"])
def test_a_non_agreeing_recorder_leaves_both_columns_empty_and_says_so(tmp_path, capsys, verdict):
    view, store = _store(tmp_path); _recorder(view, verdict)
    sr = _read(store); r = next(iter(_runs(sr).values()))
    assert r["decode_sent_json"] == {} and r["decode_resolved_json"] == {} and verdict in r["field_confidence_json"]["decode_sent_json"]["source"]
    err = capsys.readouterr().err
    assert "resolved_sampler.json" in err and verdict in err and "left empty" in err, err


def test_a_sampled_store_whose_gen_carries_the_numbers_keeps_them_and_the_recorder_only_resolves(tmp_path):
    view = str(tmp_path / "root" / "synthmodel_u0"); os.makedirs(view)
    store = F.write_controls_store(view, "none", n=3, sampled=True)
    _recorder(view, "AGREES", samplers=(dict(RESOLVED, top_k=20),), reference={"temperature": 0.7})
    r = next(iter(_runs(_read(store)).values()))
    assert r["decode_sent_json"] and r["field_confidence_json"]["decode_sent_json"]["source"] == "manifest.json:resolved_config.gen"      # the manifest's own numbers stand
    assert r["decode_resolved_json"] == {}                                                                                               # the recorder is not consulted for a sampled cell


def test_the_recorder_is_a_per_store_input_aliased_to_its_store(tmp_path):
    view = str(tmp_path / "root" / "synthmodel_u0"); os.makedirs(view)
    s1 = F.write_controls_store(view, "none", n=2, sampled=False)
    s2 = F.write_controls_store(view, F.LORA_CELL, n=2, sampled=False)
    assert not any(CS.is_recorder_input(i) for i in CS.store_inputs(s1))                         # no recorder yet: no input
    rec = _recorder(view, "AGREES")
    i1 = [i for i in CS.store_inputs(s1) if CS.is_recorder_input(i)]; i2 = [i for i in CS.store_inputs(s2) if CS.is_recorder_input(i)]
    assert len(i1) == len(i2) == 1 and i1[0].path == i2[0].path == os.path.realpath(rec) and i1[0].sha256 == i2[0].sha256 and i1[0].role == "report"
    assert i1[0].uri == f"{rec}#{os.path.abspath(s1)}" and i1[0].identity_uri != i2[0].identity_uri                                      # one file, two per-store identities
    assert CS.group_of_input(i1[0]) == os.path.abspath(s1) and CS.group_of_input(i2[0]) == os.path.abspath(s2)
    ad = CS.CellStoreAdapter(roots=[CS.StoreRoot("synthcampaign", str(tmp_path / "root"), "targets")])
    groups = {}
    for inp in ad.discover():
        groups.setdefault(ad.comparison_group(inp), []).append(inp)
    assert set(groups) == {s1, s2} and all(sum(1 for i in inps if CS.is_recorder_input(i)) == 1 and any(not CS.is_recorder_input(i) and i.role == "report" for i in inps) for inps in groups.values()), groups
    # the per-store alias is the reader's recorder evidence (no second lookup by path)
    r = CS.StoreReader(F.dims_map(), "synthcampaign", NOW).recorder_decode(s1, CS.store_inputs(s1))
    assert r["sent"] == SENT_CARD and r["path"] == os.path.realpath(rec)


def test_a_store_harvested_before_its_recorder_re_reads_without_force_when_the_recorder_lands(tmp_path):
    root = str(tmp_path / "reg"); F.ingest_dimensions(root)
    view = str(tmp_path / "stores" / "synthmodel_u0"); os.makedirs(view)
    _pin(F.write_controls_store(view, "none", ihash8="2ff9cdff", object_sha16=None, n=3, sampled=False), TREE)
    # a root that reads producer.tree_sha: the recorder's declared reference then has the run's forty-hex code sha to derive against
    ad = CS.CellStoreAdapter(roots=[CS.StoreRoot("synthcampaign", str(tmp_path / "stores"), "targets", producer_tree_sha=True)])
    conn = DB.connect(DB.db_path(root), readonly=True); rep = RUN.run_pass(root, [ad], conn=conn, now="2026-01-05T00:00:00.000Z"); conn.close()
    assert rep.groups_refused == 0 and rep.groups_refreshed == 1, [r.error for r in rep.results if r.error]
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    conn = DB.connect(DB.db_path(root), readonly=True)
    before = conn.execute("SELECT run_id, decode_sent_json, decode_resolved_json FROM run WHERE campaign = 'synthcampaign'").fetchall()
    assert before and all(json.loads(r[1]) == {} and json.loads(r[2] or "{}") == {} for r in before)
    n_rev_before = conn.execute("SELECT COUNT(*) FROM source_revision WHERE adapter = 'cellstore'").fetchone()[0]
    conn.close()
    # nothing changed: a second pass refreshes nothing
    conn = DB.connect(DB.db_path(root), readonly=True); rep = RUN.run_pass(root, [ad], conn=conn, now="2026-01-05T01:00:00.000Z"); conn.close()
    assert rep.groups_refreshed == 0, [r.error for r in rep.results if r.error]
    # the producer writes the recorder AFTER the driver: the next pass sees a new input of the store's group and re-reads, no force
    _recorder(view, "AGREES")
    conn = DB.connect(DB.db_path(root), readonly=True); rep = RUN.run_pass(root, [ad], conn=conn, now="2026-01-05T02:00:00.000Z"); conn.close()
    assert rep.groups_refused == 0 and rep.groups_refreshed == 1, [r.error for r in rep.results if r.error]
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    conn = DB.connect(DB.db_path(root), readonly=True)
    after = conn.execute("SELECT run_id, decode_sent_json, decode_resolved_json, field_confidence_json, code_sha FROM run WHERE campaign = 'synthcampaign'").fetchall()
    assert {r[0] for r in after} == {r[0] for r in before}
    for r in after:
        assert json.loads(r[1]) == SENT_CARD and json.loads(r[2]) == RESOLVED and json.loads(r[3])["decode_sent_json"]["confidence"] == "derived", r
        assert r[4] == TREE and f"code_sha {TREE}" in json.loads(r[3])["decode_sent_json"]["source"], r
    assert conn.execute("SELECT COUNT(*) FROM source_revision WHERE adapter = 'cellstore'").fetchone()[0] == n_rev_before + 1
    conn.close()
