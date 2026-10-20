"""THE STATE-ROW RULE of the adapter pass (`runner` module docstring; adapter re-anchoring plan commit 7):
the `source_state` rows a revision publishes are the source's WHOLE state, and the `$.row_refusals` ledger is per group and per revision.

* a re-activation of a committed revision (a reverted input) replays THAT revision's own state rows -- re-stamped, never asked of the
  adapter, whose per-read ledger is the LAST read's (a later revision of this very group) -- and clears (`found = 0`) every row of the
  source the outgoing head wrote under a key the re-activated revision never did;
* a fresh read clears the rows it does not restate: the rows keyed on an input that left discovery while the group's other inputs stayed
  (the runner retired watermarks for those, never state rows, so a row keyed on the gone file stood `found = 1` with its last list);
* a group whose every input left discovery clears its group-keyed ledger row with the per-input rows;
* the ledger row is keyed on the group (`base.row_refusals_uri`), its sha the inputs' fingerprint, and a path-valued group's ledger URI is
  canonicalized from the canonical source id under a relocated read root;
* THE STATE-KEY OWNERSHIP RULE (the review's third finding): a re-activation replays and clears GROUP-OWNED rows only -- those keyed on the
  revision's inputs, the group's ledger, or a path the adapter declared (`state_owned_paths`) -- and never re-stamps a committed snapshot of a
  SHARED or AUXILIARY key (a root-wide directory listing every group restates, a group's own directory listing): the latest fresh read's row
  stands; the module docstring's table classifies every key the adapters emit, and a test here pins that it names every emission site.

Synthetic adapters over invented stores (tests/test_registry_adapter_pass.py's `SyntheticStoreAdapter`); nothing here reads a real store.
"""
from __future__ import annotations

import ast
import glob
import json
import os
import re
import shutil
from typing import Iterator, List, Sequence, Tuple

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import synth
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import lineage_common as LC
from sisteer.registry.adapters import runner as RUN

from test_registry_adapter_pass import SyntheticStoreAdapter, _pass_and_ingest, _setup, _store  # noqa: E402


def _q(root, sql, *params):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        return [tuple(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _states(root, source_id):
    """{(input_uri, json_path): (found, members, observed_utc, input_sha256)} of one source's state rows."""
    return {(r[0], r[1]): (r[2], json.loads(r[3]), r[4], r[5]) for r in
            _q(root, "SELECT input_uri, json_path, found, current_members_json, observed_utc, input_sha256 FROM source_state WHERE source_id = ? ORDER BY 1, 2", source_id)}


class LedgerStoreAdapter(SyntheticStoreAdapter):
    """The synthetic store reader with a per-read refusal ledger naming every item of the store it read (so the ledger's content follows the
    store's bytes), the ledger row keyed on the group the way `_bcbw_common` / `sampled_static` key theirs, and -- for a store of three or
    more items -- one more state row under the group's source id that a two-item revision never writes. Every `source_states` call is logged
    with the pass stamp it was asked under."""
    name = "ledger_store"

    def __init__(self, store_root: str):
        super().__init__(store_root)
        self.ledger = LC.RowRefusals(self.name)
        self.state_calls: List[Tuple[str, str]] = []

    def read(self, ctx: AB.AdapterContext, group: str, inputs: Sequence[AB.Input]) -> Iterator[AB.RecordBatch]:
        self.ledger = LC.RowRefusals(self.name, read_of=(group, ctx.now))
        for inp in inputs:
            with open(inp.path) as fh:
                for it in json.load(fh)["items"]:
                    self.ledger.add(f"{group}|{it['id']}", "synthetic: named by the read", inp.uri)
        yield from super().read(ctx, group, inputs)

    def source_states(self, ctx, inputs):
        group = self.comparison_group(inputs[0])
        self.state_calls.append((group, ctx.now))
        sid = RUN.source_id_for(self.name, group)
        rows = [AB.SourceStateRow(sid, i.uri, "$.object", 1, 1, [json.load(open(i.path))["object"]], i.sha256) for i in inputs]
        with open(inputs[0].path) as fh:
            items = [it["id"] for it in json.load(fh)["items"]]
        if len(items) >= 3:
            rows.append(AB.SourceStateRow(sid, inputs[0].uri, "$.third", 1, 1, items[2:], inputs[0].sha256))
        return rows + self.ledger.ledger_rows(ctx, group, inputs, sid)


def test_a_reactivation_replays_the_committed_revisions_state_rows_and_clears_the_outgoing_heads_keys_without_asking_the_adapter(tmp_path):
    """s1 with two items (rev1) -> three items (rev2: a longer ledger, a `$.third` row rev1 never wrote) -> the two-item bytes again, the SAME
    adapter instance: rev1 is re-activated. Before the fix the adapter's `source_states` was asked and its ledger -- the rev2 read's, this very
    group -- was written as rev1's state; now the bundle carries rev1's own rows re-stamped and `$.third` goes `found = 0`."""
    root, store_root = _setup(tmp_path)
    two = _store(store_root, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8])
    ad = LedgerStoreAdapter(store_root)
    sid = RUN.source_id_for(ad.name, "s1")
    ledger_key = (AB.row_refusals_uri(sid), AB.ROW_REFUSALS_PATH)
    rep1, ing1 = _pass_and_ingest(root, ad, synth.t(2000))
    (res1,) = rep1.results
    assert res1.accepted and ing1.refused == 0 and ing1.blocked == 0, (res1.error, ing1.refusals)
    rev1 = res1.revision_id
    st1 = _states(root, sid)
    inputs1 = [i for i in ad.discover() if ad.comparison_group(i) == "s1"]
    assert set(st1) == {(two, "$.object"), ledger_key}
    assert st1[ledger_key] == (1, ["s1|q0 -- synthetic: named by the read", "s1|q1 -- synthetic: named by the read"], synth.t(2000), AB.inputs_fingerprint(inputs1))
    assert ad.state_calls == [("s1", synth.t(2000))]
    # three items: a new revision with a longer ledger and a row rev1 never wrote
    _store(store_root, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])
    rep2, ing2 = _pass_and_ingest(root, ad, synth.t(3000))
    (res2,) = rep2.results
    assert res2.accepted and res2.revision_id != rev1 and ing2.blocked == 0, (res2.error, ing2.refusals)
    st2 = _states(root, sid)
    assert set(st2) == {(two, "$.object"), (two, "$.third"), ledger_key}
    assert st2[ledger_key][0] == 1 and len(st2[ledger_key][1]) == 3 and st2[ledger_key][2] == synth.t(3000) and st2[(two, "$.third")] == (1, ["q2"], synth.t(3000), st2[(two, "$.object")][3])
    assert ad.state_calls == [("s1", synth.t(2000)), ("s1", synth.t(3000))]
    # the two-item bytes again (byte for byte: the same sha): rev1's id, committed and accepted -> re-activated, its state replayed
    _store(store_root, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8])
    rep3, ing3 = _pass_and_ingest(root, ad, synth.t(4000))
    (res3,) = rep3.results
    assert res3.accepted and res3.reactivated and res3.revision_id == rev1 and f"{os.sep}_activations{os.sep}" in res3.files[0], res3
    assert ing3.refused == 0 and ing3.blocked == 0, ing3.refusals
    assert ad.state_calls == [("s1", synth.t(2000)), ("s1", synth.t(3000))], "a re-activation never asks the adapter for state rows"
    st3 = _states(root, sid)
    assert set(st3) == {(two, "$.object"), (two, "$.third"), ledger_key}
    assert st3[ledger_key] == (1, st1[ledger_key][1], synth.t(4000), st1[ledger_key][3]), "rev1's own ledger, re-stamped -- never the rev2 read's three lines"
    assert st3[(two, "$.object")] == (1, ["constitution_lora@aaaa"], synth.t(4000), st1[(two, "$.object")][3])
    assert st3[(two, "$.third")] == (0, [], synth.t(4000), None), "the outgoing head's key the re-activated revision never wrote is cleared"
    # the bundle's records say the same: rev1's rows under fresh stamps, the cleared key found = 0, and the adapter's (stale) ledger nowhere
    from sisteer.registry import records as R
    recs = []
    for f in res3.files:
        if "source_revision" in os.path.basename(f):
            recs += [rec["payload"] for _, rec, _ in R.read_record_file(f) if rec["record_type"] == "source_state"]
    by_key = {(p["input_uri"], p["json_path"]): p for p in recs}
    assert set(by_key) == {(two, "$.object"), (two, "$.third"), ledger_key} and all(p["observed_utc"] == synth.t(4000) for p in recs)
    assert by_key[ledger_key]["current_members_json"] == st1[ledger_key][1] and by_key[(two, "$.third")]["found"] == 0
    # quiet after
    rep4, _ = _pass_and_ingest(root, ad, synth.t(5000))
    assert rep4.results == [] and _states(root, sid) == st3


class TwoFileGroupAdapter(SyntheticStoreAdapter):
    """One group per directory, TWO inputs per group (`store.json`, read as the parent does, and `note.json`, a report the read never opens
    but whose `$.note` members are state), a group-keyed ledger row, and a drop threshold of 1.0 so a partial loss is read rather than
    refused (the framework default would refuse a group that lost half its inputs before any store is read)."""
    name = "two_file_store"
    inputs_dropped_threshold = 1.0

    def __init__(self, store_root: str):
        super().__init__(store_root)
        self.ledger = LC.RowRefusals(self.name)

    def discover(self) -> List[AB.Input]:
        out = []
        for d in sorted(os.listdir(self.roots[0])):
            for name in ("note.json", "store.json"):
                p = os.path.join(self.roots[0], d, name)
                if os.path.lexists(p):
                    out.append(AB.check_input(p, role="report", allow_symlink=self.allow_symlinks))
        return out

    def read(self, ctx: AB.AdapterContext, group: str, inputs: Sequence[AB.Input]) -> Iterator[AB.RecordBatch]:
        self.ledger = LC.RowRefusals(self.name, read_of=(group, ctx.now))
        stores = [i for i in inputs if os.path.basename(i.uri) == "store.json"]
        self.ledger.add(f"{group}|note", "synthetic: the note is " + ("present" if len(stores) < len(inputs) else "absent"), group)
        yield from super().read(ctx, group, stores)

    def source_states(self, ctx, inputs):
        group = self.comparison_group(inputs[0])
        sid = RUN.source_id_for(self.name, group)
        rows = []
        for i in inputs:
            with open(i.path) as fh:
                d = json.load(fh)
            if os.path.basename(i.uri) == "store.json":
                rows.append(AB.SourceStateRow(sid, i.uri, "$.object", 1, 1, [d["object"]], i.sha256))
            else:
                rows.append(AB.SourceStateRow(sid, i.uri, "$.note", 1, 1, [str(d["note"])], i.sha256))
        return rows + self.ledger.ledger_rows(ctx, group, inputs, sid)


def test_a_read_clears_the_rows_of_an_input_that_left_while_the_group_stayed_and_a_vanished_group_clears_its_group_keyed_ledger(tmp_path):
    """s1 = store.json + note.json: both rows `found = 1` and the ledger under the group key. note.json leaves (one input of two, admitted
    by the adapter's own threshold): the group re-reads, `$.note` goes `found = 0` at the pass stamp, `$.object` and the ledger are restated at
    the SAME keys (the ledger's sha moves with the inputs). Then the whole directory leaves: the retirement clears the ledger row -- keyed on
    no input URI -- beside the per-input rows."""
    root, store_root = _setup(tmp_path)
    store = os.path.join(store_root, "s1", "store.json")
    note = os.path.join(store_root, "s1", "note.json")
    with open(note, "w") as fh:
        json.dump({"note": "hello"}, fh)
    ad = TwoFileGroupAdapter(store_root)
    sid = RUN.source_id_for(ad.name, "s1")
    ledger_key = (AB.row_refusals_uri(sid), AB.ROW_REFUSALS_PATH)
    rep1, ing1 = _pass_and_ingest(root, ad, synth.t(2000))
    assert rep1.groups_refreshed == 1 and ing1.refused == 0 and ing1.blocked == 0, ([r.error for r in rep1.results], ing1.refusals)
    st1 = _states(root, sid)
    assert set(st1) == {(store, "$.object"), (note, "$.note"), ledger_key} and all(v[0] == 1 for v in st1.values())
    assert st1[ledger_key][1] == ["s1|note -- synthetic: the note is present"] and st1[ledger_key][3] == AB.inputs_fingerprint(ad.discover())
    # a row ANOTHER source id files on the note (the way sampled_static files its final_rung rows): cleared with the input when it leaves
    aside = RUN.source_id_for(ad.name, "aside")
    assert _states(root, aside) == {}
    from sisteer.registry import records as R
    R.write_records(os.path.join(root, "records", "adapters", ad.name, "aside", "000001-source_state.jsonl"),
                    [RUN._vanished_state_record(AB.AdapterContext(root=root, now=synth.t(2500), producer=synth.producer()), aside, note, "$.aside")])
    H.ingest_once(root, workers=0, min_free=0)
    with DB.connect(DB.db_path(root)) as w:
        w.execute("UPDATE source_state SET found = 1, current_members_json = '[\"x\"]' WHERE source_id = ? AND input_uri = ?", (aside, note))
        w.commit()
    assert _states(root, aside) == {(note, "$.aside"): (1, ["x"], synth.t(2500), None)}
    os.remove(note)
    rep2, ing2 = _pass_and_ingest(root, ad, synth.t(3000))
    (res2,) = rep2.results
    assert res2.accepted and not res2.reactivated and ing2.refused == 0 and ing2.blocked == 0, (res2.error, ing2.refusals)
    st2 = _states(root, sid)
    assert set(st2) == set(st1)
    assert st2[(note, "$.note")] == (0, [], synth.t(3000), None), "the row keyed on the input that left is cleared, not left standing with its last members"
    assert _states(root, aside) == {(note, "$.aside"): (0, [], synth.t(3000), None)}, "the other source's row on the gone input is cleared with it"
    assert st2[(store, "$.object")] == (1, ["constitution_lora@aaaa"], synth.t(3000), st1[(store, "$.object")][3])
    assert st2[ledger_key] == (1, ["s1|note -- synthetic: the note is absent"], synth.t(3000), AB.inputs_fingerprint(ad.discover())), "the same group key, the new list, the new fingerprint"
    assert _q(root, "SELECT last_error FROM source_watermark WHERE input_uri = ?", note) == [(RUN.VANISHED_PREFIX + note,)]
    # the whole group leaves
    shutil.rmtree(os.path.join(store_root, "s1"))
    rep3, ing3 = _pass_and_ingest(root, ad, synth.t(4000))
    assert rep3.withdrawals >= 1 and ing3.refused == 0 and ing3.blocked == 0, ([r.error for r in rep3.results], ing3.refusals)
    st3 = _states(root, sid)
    assert set(st3) == set(st1) and all(v[0] == 0 and v[1] == [] and v[3] is None for v in st3.values()), st3
    assert st3[ledger_key][2] == st3[(store, "$.object")][2] == synth.t(4000) and st3[(note, "$.note")][2] == synth.t(3000)   # already cleared, not written twice
    # idempotent: a further pass writes nothing
    rep4, _ = _pass_and_ingest(root, ad, synth.t(5000))
    assert rep4.withdrawals == 0 and _states(root, sid) == st3


def test_the_ledger_uri_is_canonicalized_from_the_canonical_source_id_under_a_relocated_read_root(tmp_path):
    """A path-valued group read under a relocated root (a cluster shard): the adapter spells the ledger's source id locally, and the runner's
    record carries `registry://row-refusals/<canonical source id>` -- the grading host's key -- never the local spelling."""
    local = str(tmp_path / "local_home")
    os.makedirs(local)
    canonical = "/canonical/home"
    group_local, group_canonical = os.path.join(local, "grades", "g1"), os.path.join(canonical, "grades", "g1")

    class PathGroupAdapter:
        name = "path_group"

        def source_states(self, ctx, inputs):
            sid = f"{self.name}:{group_local}"
            ledger = LC.RowRefusals(self.name, read_of=(group_local, ctx.now))
            ledger.add("k", "why", group_local)
            return ledger.ledger_rows(ctx, group_local, inputs, sid)

    ctx = AB.AdapterContext(root=str(tmp_path), now=synth.t(1), producer=synth.producer())
    with AB.using_identity_map(AB.IdentityMap.build({local: canonical})):
        inp = AB.Input(uri=os.path.join(group_local, "a.json"), path=os.path.join(group_local, "a.json"), sha256="a" * 64, size=1, mtime_utc=synth.t(0), role="report",
                       identity_uri=os.path.join(group_canonical, "a.json"))
        (rec,) = RUN._source_state_records(ctx, PathGroupAdapter(), [inp])
    p = rec["payload"]
    assert p["source_id"] == f"path_group:{group_canonical}" and p["input_uri"] == AB.row_refusals_uri(f"path_group:{group_canonical}")
    assert p["json_path"] == AB.ROW_REFUSALS_PATH and p["current_members_json"] == ["k -- why"] and p["input_sha256"] == AB.inputs_fingerprint([inp])
    assert local not in json.dumps(p)


class SharedKeyStoreAdapter(SyntheticStoreAdapter):
    """The synthetic store reader with the two kinds of state row the ownership rule tells apart beside its group-owned `$.object` row, in
    the PRE-RULE shape no adapter uses any more (the listings-are-inputs rule; `ListingStoreAdapter` below is the shape they use): a
    ROOT-WIDE `$.dirs` row -- the root's directory listing under one root-wide source id, written identically by every group's read -- and
    a per-group AUXILIARY `$.listing` row -- the group directory's entries under the group's own source id. The runner's contract for such
    keys (off the inputs: never replayed, never cleared by a re-activation, the latest fresh read's row stands) is what this pins."""
    name = "shared_key_store"

    def source_states(self, ctx, inputs):
        root = self.roots[0]
        group = self.comparison_group(inputs[0])
        sid = RUN.source_id_for(self.name, group)
        rows = [AB.SourceStateRow(sid, i.uri, "$.object", 1, 1, [json.load(open(i.path))["object"]], i.sha256) for i in inputs]
        dirs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        rows.append(AB.SourceStateRow(f"{self.name}:{os.path.basename(root)}", root, "$.dirs", 1 if dirs else 0, 1, dirs, None))
        gdir = os.path.dirname(inputs[0].path)
        rows.append(AB.SourceStateRow(sid, gdir, "$.listing", 1, 1, sorted(os.listdir(gdir)), None))
        return rows


def test_a_reactivation_of_one_group_leaves_a_shared_key_another_groups_read_refreshed_and_the_groups_auxiliary_row_at_the_fresher_value(tmp_path):
    """The synthetic twin of gsm8k_p200's `$.dirs`. Commit A (root dirs A, B); change A and commit again (rev A2); add an
    undeclared directory C and refresh B (a touched store.json with new bytes): `$.dirs` correctly names A, B, C; restore A's original bytes:
    A re-activates its first revision -- its `$.object` row replayed, its stale `$.dirs` snapshot (A, B) NOT replayed, the root-wide row
    standing at B's fresher A, B, C -- and A's own `$.listing` (a note file added to A's directory meanwhile, named by A2's read) stands at
    A2's value too: a re-activation observes nothing and never clears an auxiliary row either."""
    root, store_root = _setup(tmp_path)
    a_two = _store(store_root, "A", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8])
    b = _store(store_root, "B", "legacy_store_2", "none", [0.1, 0.9])
    ad = SharedKeyStoreAdapter(store_root)
    shutil.rmtree(os.path.join(store_root, "s1"))              # the default store the world ships with: only A and B here
    sid_a, sid_b = RUN.source_id_for(ad.name, "A"), RUN.source_id_for(ad.name, "B")
    shared = f"{ad.name}:{os.path.basename(store_root)}"
    dirs_key = (store_root, "$.dirs")
    a_dir = os.path.join(store_root, "A")
    rep1, ing1 = _pass_and_ingest(root, ad, synth.t(2000))
    assert rep1.groups_refreshed == 2 and ing1.refused == 0 and ing1.blocked == 0, ([r.error for r in rep1.results], ing1.refusals)
    rev_a1 = next(r.revision_id for r in rep1.results if r.group == "A")
    assert _states(root, shared) == {dirs_key: (1, ["A", "B"], synth.t(2000), None)}
    assert _states(root, sid_a)[(a_dir, "$.listing")] == (1, ["store.json"], synth.t(2000), None)
    # A changes (a note file lands in its directory too) and commits again
    with open(os.path.join(a_dir, "note.txt"), "w") as fh:
        fh.write("aside\n")
    _store(store_root, "A", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])
    rep2, ing2 = _pass_and_ingest(root, ad, synth.t(3000))
    assert [r.group for r in rep2.results] == ["A"] and rep2.results[0].revision_id != rev_a1 and ing2.blocked == 0, ([r.error for r in rep2.results], ing2.refusals)
    assert _states(root, sid_a)[(a_dir, "$.listing")] == (1, ["note.txt", "store.json"], synth.t(3000), None)
    assert _states(root, shared) == {dirs_key: (1, ["A", "B"], synth.t(3000), None)}
    # an undeclared directory C appears (no store.json: nothing to read, nothing discovered) and B's store is touched: B's read names A, B, C
    os.makedirs(os.path.join(store_root, "C"))
    _store(store_root, "B", "legacy_store_2", "none", [0.1, 0.9, 0.3])
    rep3, ing3 = _pass_and_ingest(root, ad, synth.t(4000))
    assert [r.group for r in rep3.results] == ["B"] and ing3.blocked == 0, ([r.error for r in rep3.results], ing3.refusals)
    assert _states(root, shared) == {dirs_key: (1, ["A", "B", "C"], synth.t(4000), None)}
    # A's original bytes again: A's first revision re-activates
    _store(store_root, "A", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8])
    rep4, ing4 = _pass_and_ingest(root, ad, synth.t(5000))
    (res4,) = rep4.results
    assert res4.group == "A" and res4.accepted and res4.reactivated and res4.revision_id == rev_a1 and ing4.blocked == 0, (res4, ing4.refusals)
    assert _states(root, shared) == {dirs_key: (1, ["A", "B", "C"], synth.t(4000), None)}, "the shared key stands at the fresher read's inventory, never A1's snapshot"
    st_a = _states(root, sid_a)
    assert st_a[(a_two, "$.object")] == (1, ["constitution_lora@aaaa"], synth.t(5000), st_a[(a_two, "$.object")][3]), "A's own row replayed"
    assert st_a[(a_dir, "$.listing")] == (1, ["note.txt", "store.json"], synth.t(3000), None), "A's auxiliary listing: the latest fresh read's, neither replayed nor cleared"
    # the bundle's own records carry no `$.dirs` and no `$.listing`
    from sisteer.registry import records as R
    recs = []
    for f in res4.files:
        if "source_revision" in os.path.basename(f):
            recs += [rec["payload"] for _, rec, _ in R.read_record_file(f) if rec["record_type"] == "source_state"]
    assert {(p["source_id"], p["input_uri"], p["json_path"]) for p in recs} == {(sid_a, a_two, "$.object")}, recs
    # a FRESH read of any group refreshes the shared key (C is removed; B gets new bytes -- its first bytes again would re-activate B's first
    # revision, which observes nothing): one authoritative publisher, the latest read
    os.rmdir(os.path.join(store_root, "C"))
    _store(store_root, "B", "legacy_store_2", "none", [0.1, 0.9, 0.3, 0.5])
    rep5, _ = _pass_and_ingest(root, ad, synth.t(6000))
    assert [r.group for r in rep5.results] == ["B"] and not rep5.results[0].reactivated
    assert _states(root, shared) == {dirs_key: (1, ["A", "B"], synth.t(6000), None)}


class DeclaredPathStoreAdapter(SyntheticStoreAdapter):
    """A reader whose owned row rides a key that is not its input's identity URI (the `~`-spelled path the rulings cite, the way `r2_export`
    keys `$.arm_operating_point_id`): without a declaration the rule reads it as auxiliary and a re-activation leaves the later read's row
    standing; declared under `state_owned_paths`, the row is the group's and replays."""
    name = "declared_path_store"
    state_owned_paths = ("$.declared",)

    def source_states(self, ctx, inputs):
        group = self.comparison_group(inputs[0])
        sid = RUN.source_id_for(self.name, group)
        rows = []
        for i in inputs:
            with open(i.path) as fh:
                d = json.load(fh)
            spelled = "~/" + os.path.relpath(i.uri, os.path.expanduser("~")) if i.uri.startswith(os.path.expanduser("~")) else "spelled://" + i.uri
            rows.append(AB.SourceStateRow(sid, spelled, "$.declared", 1, 1, [str(len(d["items"]))], i.sha256))
            rows.append(AB.SourceStateRow(sid, spelled, "$.undeclared", 1, 1, [str(len(d["items"]))], i.sha256))
        return rows


def test_a_declared_owned_path_replays_on_reactivation_where_an_undeclared_key_off_the_inputs_stands_at_the_later_reads_value(tmp_path):
    root, store_root = _setup(tmp_path)
    store = _store(store_root, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8])
    ad = DeclaredPathStoreAdapter(store_root)
    sid = RUN.source_id_for(ad.name, "s1")
    assert AB.state_owned_paths_of(ad) == ("$.declared",) and AB.state_owned_paths_of(SyntheticStoreAdapter(store_root)) == ()
    rep1, _ = _pass_and_ingest(root, ad, synth.t(2000))
    rev1 = rep1.results[0].revision_id
    st1 = _states(root, sid)
    (spelled,) = {k[0] for k in st1}
    assert spelled != store and set(st1) == {(spelled, "$.declared"), (spelled, "$.undeclared")} and st1[(spelled, "$.declared")][1] == ["2"]
    _store(store_root, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])
    rep2, _ = _pass_and_ingest(root, ad, synth.t(3000))
    assert rep2.results[0].revision_id != rev1 and _states(root, sid)[(spelled, "$.declared")][1] == ["3"]
    _store(store_root, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8])
    rep3, ing3 = _pass_and_ingest(root, ad, synth.t(4000))
    (res3,) = rep3.results
    assert res3.reactivated and res3.revision_id == rev1 and ing3.blocked == 0, (res3, ing3.refusals)
    st3 = _states(root, sid)
    assert st3[(spelled, "$.declared")] == (1, ["2"], synth.t(4000), st1[(spelled, "$.declared")][3]), "declared group-owned: replayed from rev1"
    assert st3[(spelled, "$.undeclared")][:3] == (1, ["3"], synth.t(3000)), "undeclared and keyed off the inputs: the later read's row stands"
    # a malformed declaration refuses by name
    class Bad(SyntheticStoreAdapter):
        name = "bad_declaration"
        state_owned_paths = "$.declared"
    import pytest
    with pytest.raises(AB.AdapterRefusal, match="state_owned_paths must be a sequence"):
        AB.state_owned_paths_of(Bad(store_root))


def _emitted_state_keys():
    """(module, input_uri expression, json_path expression) of every state-row emission site under src/sisteer/registry/adapters/, by AST:
    `SourceStateRow(...)` calls, `<ledger>.source_state_rows(uri, ...)` calls (the ledger on the given key) and `.ledger_rows(...)` calls (the
    ledger on the group key); a `source_state_rows` of something that is not a refusal ledger is a helper delegation, named as such."""
    here = os.path.dirname(os.path.abspath(RUN.__file__))
    out = set()
    for path in sorted(glob.glob(os.path.join(here, "*.py"))):
        mod = os.path.basename(path)[:-3]
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
            if name == "SourceStateRow" and len(node.args) >= 3:
                out.add((mod, ast.unparse(node.args[1]), ast.unparse(node.args[2])))
            elif name == "source_state_rows" and isinstance(f, ast.Attribute) and len(node.args) >= 2:
                if ast.unparse(f.value).endswith("refusals"):
                    out.add((mod, ast.unparse(node.args[1]), "$.row_refusals"))
                else:
                    out.add((mod, ast.unparse(f), "(its rows)"))
            elif name == "ledger_rows" and isinstance(f, ast.Attribute):
                out.add((mod, "row_refusals_uri(source_id)", "$.row_refusals"))
    return out


def test_the_ownership_table_in_the_runner_docstring_classifies_every_emitted_state_key_and_nothing_else():
    """The runner module docstring's table (the state-key ownership rule) names every state-row emission site of every adapter -- by module,
    key expression and path expression, exactly as the source spells them -- with one of the three classes, and no row of it is stale."""
    rows = re.findall(r"^\| (\S+) \| `(.+?)` \| `(.+?)` \| (owned|owned \(declared\)|auxiliary \(self-correcting\)|auxiliary|shared|retired \(stated absent\)) \|",
                      RUN.__doc__, flags=re.MULTILINE)
    table = {(m, u, p): c for m, u, p, c in rows}
    emitted = _emitted_state_keys()
    assert len(table) == len(rows) and len(table) > 90, len(table)
    assert set(table) == emitted, {"not in the table": sorted(emitted - set(table)), "stale in the table": sorted(set(table) - emitted)}
    # the structural rule and the table agree on the known shapes: every row keyed on an input expression is owned, the ledgers are owned, the
    # listings are inputs and owned (em_misalign's entries, degeneracy_union's cap-4,096 presence probe, the root inventory of gsm8k_p200's root
    # pseudo-group), NO key is shared, the auxiliary class holds exactly the one self-correcting absence row (cap4096's gone guard, whose return
    # re-reads its own group), and the one RETIRED key (gsm8k_p200's pre-rule root-path `$.dirs`) is stated absent by the root group's read
    assert table[("gsm8k_p200", "inp.uri", "'$.dirs'")] == "owned" and table[("gsm8k_p200", "self.root", "'$.dirs'")] == "retired (stated absent)"
    assert {k for k in table if k[2] == "'$.dirs'"} == {("gsm8k_p200", "inp.uri", "'$.dirs'"), ("gsm8k_p200", "self.root", "'$.dirs'")}
    assert {k for k in table if k[2] == "'$.entries'"} == {("em_misalign", "i.uri", "'$.entries'"), ("degeneracy_union", "inp.uri", "'$.entries'")}
    assert all(table[k] == "owned" for k in table if k[2] == "'$.entries'")
    assert {k for k, c in table.items() if c == "shared"} == set()
    assert {k for k, c in table.items() if c.startswith("auxiliary")} == {("cap4096_declarations", "ruling_uri(os.path.join(self.run_state, rel), self.home)", "'$'")}
    assert {c for c in table.values() if c.startswith("auxiliary")} == {"auxiliary (self-correcting)"}
    assert ("degeneracy_union", "mt", "'$.cells'") not in table, "the in-flight absence is no longer a found = 0 row of another group"
    assert table[("r2_export", "ruling_uri(pq.uri if pq else self._p(PARQUET))", "'$.arm_operating_point_id'")] == "owned (declared)"
    counts = {}
    for c in table.values():
        counts[c] = counts.get(c, 0) + 1
    assert counts == {"owned": len(table) - 3, "owned (declared)": 1, "auxiliary (self-correcting)": 1, "retired (stated absent)": 1}, counts
    from sisteer.registry.adapters import r2_export as R2
    assert AB.state_owned_paths_of(R2.R2ExportAdapter if hasattr(R2, "R2ExportAdapter") else next(c for c in vars(R2).values() if getattr(c, "name", None) == "r2_export")) == ("$.arm_operating_point_id",)
    assert all(c == "owned" for k, c in table.items() if k[2] == "$.row_refusals")
    assert all(c == "owned" for k, c in table.items() if re.fullmatch(r"(inp|i|op|agg|anc|curves|man|sel|ex|lane_legacy|summary_in|first)\.uri|inputs\[0\]\.uri", k[1]))


class ListingStoreAdapter(SyntheticStoreAdapter):
    """The synthetic store reader stating what its store directories CONTAIN the way the adapters now do (the runner's listings-are-inputs
    rule, `base.ListingInput`): each group's directory listing is a discovered input of the group, and `$.entries` is keyed on it."""
    name = "listing_store"

    def discover(self) -> List[AB.Input]:
        out = super().discover()
        for d in sorted({os.path.dirname(i.path) for i in out}):
            out.append(AB.ListingInput.of_directory(RUN.source_id_for(self.name, os.path.basename(d)), os.path.basename(d), d))
        return out

    def comparison_group(self, inp: AB.Input) -> str:
        return AB.listing_group(inp.uri, self.name) if AB.is_listing_input(inp) else super().comparison_group(inp)

    def read(self, ctx: AB.AdapterContext, group: str, inputs: Sequence[AB.Input]) -> Iterator[AB.RecordBatch]:
        yield from super().read(ctx, group, [i for i in inputs if not AB.is_listing_input(i)])

    def source_states(self, ctx, inputs):
        group = self.comparison_group(inputs[0])
        sid = RUN.source_id_for(self.name, group)
        rows = []
        for i in inputs:
            if AB.is_listing_input(i):
                rows.append(AB.SourceStateRow(sid, i.uri, "$.entries", 1, 1, list(i.entries), i.sha256))
            else:
                rows.append(AB.SourceStateRow(sid, i.uri, "$.object", 1, 1, [json.load(open(i.path))["object"]], i.sha256))
        return rows


def test_a_listing_is_an_input_an_ignored_entry_added_then_removed_is_a_revision_and_back_and_its_entries_row_reverts_with_it(tmp_path):
    """The synthetic shape of em_misalign's `$.entries`. s1 committed (entries: store.json); an ignored entry lands while the
    store's bytes change (rev2: the listing input's sha moved with the file's); the entry is removed and the bytes restored: rev1 RE-ACTIVATES
    (the listing is part of the identity) and its own `$.entries` row -- keyed on the listing input, so group-owned -- replays, naming the
    store.json alone. The listing input is watermarked, registers no artifact, and a listing change ALONE is a revision change too."""
    root, store_root = _setup(tmp_path)
    store = os.path.join(store_root, "s1", "store.json")
    ad = ListingStoreAdapter(store_root)
    sid = RUN.source_id_for(ad.name, "s1")
    (listing,) = [i for i in ad.discover() if AB.is_listing_input(i)]
    assert listing.uri == AB.listing_uri(sid, "s1") and listing.entries == ("store.json",) and listing.role == AB.LISTING_ROLE and listing.path == os.path.join(store_root, "s1")   # the listed directory: the shard compares it with the manifest attestation
    key = (listing.uri, "$.entries")
    rep1, ing1 = _pass_and_ingest(root, ad, synth.t(2000))
    (res1,) = rep1.results
    assert res1.accepted and ing1.refused == 0 and ing1.blocked == 0, (res1.error, ing1.refusals)
    rev1 = res1.revision_id
    st1 = _states(root, sid)
    assert st1[key] == (1, ["store.json"], synth.t(2000), listing.sha256) and st1[(store, "$.object")][0] == 1
    assert _q(root, "SELECT sha256, size, last_error FROM source_watermark WHERE input_uri = ?", listing.uri) == [(listing.sha256, 1, None)]
    assert _q(root, "SELECT COUNT(*) FROM artifact WHERE uri_primary LIKE 'registry://%'") == [(0,)], "a listing registers no artifact"
    # an ignored entry lands while the store's bytes change: a later revision whose entries name it
    with open(os.path.join(store_root, "s1", "killed_entry"), "w") as fh:
        fh.write("partial rows that are not an arm\n")
    _store(store_root, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4, 0.6])
    rep2, ing2 = _pass_and_ingest(root, ad, synth.t(3000))
    (res2,) = rep2.results
    assert res2.accepted and not res2.reactivated and res2.revision_id != rev1 and ing2.blocked == 0, (res2, ing2.refusals)
    st2 = _states(root, sid)
    assert st2[key][:3] == (1, ["killed_entry", "store.json"], synth.t(3000)) and st2[key][3] != listing.sha256
    # the entry is removed and the bytes restored: rev1 re-activates and its entries row reverts with it
    os.remove(os.path.join(store_root, "s1", "killed_entry"))
    _store(store_root, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])
    rep3, ing3 = _pass_and_ingest(root, ad, synth.t(4000))
    (res3,) = rep3.results
    assert res3.accepted and res3.reactivated and res3.revision_id == rev1 and ing3.blocked == 0, (res3, ing3.refusals)
    st3 = _states(root, sid)
    assert st3[key] == (1, ["store.json"], synth.t(4000), listing.sha256), "the entries row is the re-activated revision's own, keyed on its listing input"
    # a listing change ALONE (an entry lands, the bytes stand) is a revision change: the group re-reads and the row names the entry
    with open(os.path.join(store_root, "s1", "killed_entry"), "w") as fh:
        fh.write("again\n")
    rep4, ing4 = _pass_and_ingest(root, ad, synth.t(5000))
    (res4,) = rep4.results
    assert res4.accepted and not res4.reactivated and res4.revision_id not in (rev1, res2.revision_id) and ing4.blocked == 0
    assert _states(root, sid)[key][:3] == (1, ["killed_entry", "store.json"], synth.t(5000))


def _em_world(tmp_path):
    """The real em_misalign reader's vendor-leak world (tests/test_registry_declared_em_d60.py) with a registry whose dimension rows resolve
    its records -- the model, the agentic eval and its item set, the two objects -- so the records FOLD. Returns (registry root, adapter, the
    store directory, the panel root)."""
    import test_registry_declared_em_d60 as EMT
    import test_registry_declared_runner as DR
    import registry_declared_fixtures as F
    from sisteer.registry import records as R
    root, grade, _fx = EMT._vendor_leak_world(tmp_path)
    reg = str(tmp_path / "reg")
    H.ensure_layout(reg)
    dims = DR._declared_dims()
    gemma = next(r for r in dims if r["record_type"] == "model" and r["payload"]["model_id"] == "gemma4e4b_it")
    dims += [synth.rec("model", dict(gemma["payload"], model_id="qwen35_4b_it", family="qwen35", size_label="4b", hf_repo="synthetic/qwen35-4b-it"), synth.t(30)),
             synth.rec("eval", {"eval_id": "agentic_leaking", "eval_category": "agentic", "headline_metric_id": "misaligned_rate", "metric_direction": "lower_is_better",
                                "attacker_required": 0, "paper_status": "in_paper"}, synth.t(31)),
             synth.rec("item_set", {"item_set_id": "agentic_leaking_scenarios", "eval_id": "agentic_leaking", "dataset_version": "synthetic/agentic@v1", "n_items": 5,
                                    "items_sha256": "unverified", "source_uri": "synthetic://items/agentic.jsonl", "item_id_scheme": "q<index>", "subsample_rule": "full",
                                    "field_confidence_json": {"items_sha256": {"confidence": "unverified", "reason": "synthetic fixture: never read"}}}, synth.t(32)),
             synth.rec("object", synth.object_lora(F.SHA_B, "control_lora", "constitution"), synth.t(33))]
    R.write_records(os.path.join(reg, "records", "dims", "000001-dims.jsonl"), dims)
    ing = H.ingest_once(reg, workers=0, min_free=0)
    assert ing.refused == 0, ing.refusals
    return reg, EMT._adapter(root, grade), os.path.join(grade, "misalign_vendor_leak", "qwen35_4b"), root


def test_em_misalign_an_ignored_killed_directory_added_then_removed_reverts_the_entries_row_with_the_reactivated_revision(tmp_path):
    """The same shape on the real reader. The judged store's listing is an INPUT of the `qwen35_4b` group (`registry://listing/em_misalign:
    qwen35_4b/misalign_vendor_leak/qwen35_4b`). Pass 1 parks, pass 2 files (rev2: `$.entries` = cells_full.json). A `killed_alphasteer`
    directory (skipped by name, never read) lands while the panel's CSVs change: rev3, whose `$.entries` names it. The directory is removed and
    the CSVs restored: rev2 RE-ACTIVATES and its `$.entries` -- its own row, keyed on its listing input -- replays without the entry. Before the
    rule the row rode the directory path, was neither replayed nor cleared, and kept asserting the removed entry indefinitely."""
    import json as _json
    import test_registry_declared_em_d60 as EMT
    import registry_declared_fixtures as F
    from sisteer.registry.adapters import em_misalign as EM
    reg, ad, store, panels = _em_world(tmp_path)
    sid = RUN.source_id_for(ad.name, "qwen35_4b")
    (listing,) = [i for i in ad.discover() if AB.is_listing_input(i)]
    assert listing.uri == AB.listing_uri(sid, os.path.join("misalign_vendor_leak", "qwen35_4b")) and listing.entries == ("cells_full.json",)
    key = (listing.uri, "$.entries")

    def fold(now):
        with DB.reading(DB.db_path(reg)) as conn:
            rep = RUN.run_pass(reg, [ad], conn=conn, now=now)
        ing = H.ingest_once(reg, workers=0, min_free=0)
        assert rep.groups_refused == 0 and ing.refused == 0 and ing.blocked == 0, ([r.error for r in rep.results], ing.refusals)
        return rep

    def panel(lora_harm: float):
        fx = F.build_em_panel(panels, stem="qwen35_4b_vendor_leak", arms=(("none", 0.36, 510), (EMT.LORA_CELL, lora_harm, 510), (EMT.PSR_CELL, 0.1, 510)))
        EMT._add_engine_columns(fx["dir"], {"none": EMT.PASS, EMT.LORA_CELL: EMT.LENS, EMT.PSR_CELL: EMT.LORA}, cross={EMT.PSR_CELL: True})

    fold("2026-10-04T10:00:00.000Z")                                  # parks (the none is not folded yet)
    rep2 = fold("2026-10-04T10:30:00.000Z")                           # the fold made it current: files
    (res2,) = rep2.results
    rev2 = res2.revision_id
    assert fold("2026-10-04T11:00:00.000Z").results == []             # quiet
    st2 = _states(root := reg, sid)
    assert st2[key] == (1, ["cells_full.json"], "2026-10-04T10:30:00.000Z", listing.sha256)
    # the killed directory lands (never read: skipped by name) and the CSVs change: a later revision whose entries name it
    killed = os.path.join(store, EM.KILLED_DIR, "agentic_alphasteer")
    os.makedirs(killed)
    with open(os.path.join(killed, "meta.json"), "w") as fh:
        _json.dump({"leg": "alphasteer", "status": "killed", "n_rows": 7}, fh)
    panel(0.22)
    rep3 = fold("2026-10-04T12:00:00.000Z")
    (res3,) = rep3.results
    assert res3.accepted and not res3.reactivated and res3.revision_id != rev2, res3
    st3 = _states(root, sid)
    assert st3[key][:3] == (1, ["cells_full.json", EM.KILLED_DIR], "2026-10-04T12:00:00.000Z") and st3[key][3] != listing.sha256
    # the directory is removed and the CSVs restored: rev2 re-activates, its entries row reverts with it
    shutil.rmtree(os.path.join(store, EM.KILLED_DIR))
    panel(0.2)
    rep4 = fold("2026-10-04T13:00:00.000Z")
    (res4,) = rep4.results
    assert res4.accepted and res4.reactivated and res4.revision_id == rev2 and f"{os.sep}_activations{os.sep}" in res4.files[0], res4
    assert _states(root, sid)[key] == (1, ["cells_full.json"], "2026-10-04T13:00:00.000Z", listing.sha256), "rev2's own entries row, never rev3's"
    assert fold("2026-10-04T14:00:00.000Z").results == []
