"""The emitted-nothing guard of the legacy-store reader framework (adapters/runner.py): a group that read at least one STORE
input, emitted NO record of any type and recorded no non-empty `refuse` / `refused` / `refusals` / `row_refusals` / `missing` /
`torn` / `held` entry is a SILENT read — its `source_status` row is filed `ok = 0` with `extra_json.emitted_nothing_unexplained`,
its `detail` says so, the pass logs one line and counts it under `groups_emitted_nothing` beside the refusals — so a consumer can
tell it from a group that was never read. The live case (2026-09-21): a graded store read by `strength_mtdiff` produced
"0 record(s) ... no row refused" with `missing [] refuse [] row_refusals []`, and read downstream as "never run".

What must NOT fire: a recorded reason (any explanation key non-empty), an alias-only or pointer-only input set, a `_`-prefixed
bookkeeping group, a reader-declared `emits_no_runs` marker (bare or per group), the framework's known facts-only shapes, a vanished
group's empty revision (its inputs are under `missing`), and every normal read — whose status record is byte-identical to before.

Fixtures are synthetic (the synthetic store adapter of test_registry_adapter_pass.py); nothing here reads a real store.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
from types import SimpleNamespace
from typing import Iterator, List, Sequence

import pytest

from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import synth
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import runner as RUN

from test_registry_adapter_pass import SyntheticStoreAdapter, _pass_and_ingest, _setup, _store  # noqa: E402

LOGGER = "sisteer.registry.adapters.runner"


def _q(root, sql, *params):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        return [tuple(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _status_rows(root):
    """(source_id, ok, cells_measured, detail, extra) per `source_status` row, by source id then read time (a dict over it keeps the latest read)."""
    out = []
    for sid, ok, cells, detail, extra in _q(root, "SELECT source_id, ok, cells_measured, detail, extra_json FROM source_status ORDER BY source_id, observed_utc"):
        out.append((sid, ok, cells, detail, json.loads(extra)))
    return out


class SilentStoreAdapter(SyntheticStoreAdapter):
    """Reads its store inputs and emits NOTHING, recording whatever `facts` say — the 2026-09-21 defect shape when `facts` is empty.
    `alias_groups` declare their store under a per-group `#<group>` alias (a shared file's convention); `pointer_groups` declare
    it as a completion marker; `shared_index` adds `<root>/index.json` to every group under an alias (a mixed input set)."""
    name = "silent_store"
    covers_sources = ("synthetic",)

    def __init__(self, store_root: str, facts=None, alias_groups=(), pointer_groups=(), shared_index: bool = False):
        super().__init__(store_root)
        self.facts = dict(facts or {})
        self.alias_groups = set(alias_groups)
        self.pointer_groups = set(pointer_groups)
        self.shared_index = shared_index

    def discover(self) -> List[AB.Input]:
        out: List[AB.Input] = []
        index = None
        if self.shared_index:
            p = os.path.join(self.roots[0], "index.json")
            if not os.path.exists(p):
                with open(p, "w") as fh:
                    json.dump({"stores": []}, fh)
            index = AB.check_input(p, role="report")
        for inp in super().discover():
            g = os.path.basename(os.path.dirname(inp.uri))
            if g in self.alias_groups:
                inp = dataclasses.replace(inp, uri=f"{inp.uri}#{g}")
            if g in self.pointer_groups:
                inp = dataclasses.replace(inp, role="completion")
            out.append(inp)
            if index is not None:
                out.append(dataclasses.replace(index, uri=f"{index.uri}#{g}"))
        return out

    def comparison_group(self, inp: AB.Input) -> str:
        if "#" in inp.uri:
            return inp.uri.rsplit("#", 1)[1]
        return os.path.basename(os.path.dirname(inp.uri))

    def read(self, ctx: AB.AdapterContext, group: str, inputs: Sequence[AB.Input]) -> Iterator[AB.RecordBatch]:
        yield AB.RecordBatch(RUN.source_id_for(self.name, group), group)

    def source_states(self, ctx, inputs):
        return []

    def status_extra(self, group: str):
        return dict(self.facts)


# ============================================================================================
# the guard fires
# ============================================================================================

def test_a_store_read_that_emits_nothing_and_records_no_reason_is_flagged_filed_not_ok_counted_and_logged(tmp_path, caplog):
    root, store_root = _setup(tmp_path)
    ad = SilentStoreAdapter(store_root)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        rep, ing = _pass_and_ingest(root, ad, synth.t(2000))
    assert ing.refused == 0 and ing.blocked == 0
    # the read was ACCEPTED (nothing refused, the revision is current) and is counted under its own label beside the refusals
    assert rep.groups_refreshed == 1 and rep.groups_refused == 0 and rep.groups_held == 0
    assert rep.groups_emitted_nothing == 1
    (res,) = rep.results
    assert res.accepted and res.emitted_nothing and res.error is None
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", res.revision_id) == [(1,)]
    # the status row: ok = 0, the flag with what it judged, the detail names it
    ((sid, ok, cells, detail, extra),) = _status_rows(root)
    assert sid == "silent_store:s1" and ok == 0 and cells == 0
    assert extra[RUN.EMITTED_NOTHING_KEY] == {"inputs": 1, "input_kinds": ["report"], "store_inputs": 1, "rule": RUN.EMITTED_NOTHING_RULE}
    assert extra["record_counts"] == {} and extra["refuse"] == [] and extra["missing"] == [] and extra["torn"] == [] and extra["row_refusals"] == []
    assert "0 record(s)" in detail and "EMITTED NOTHING (unexplained): 1 store input(s) read" in detail and "filed ok = 0" in detail
    # the paper's status row counts it like a refusal
    assert _q(root, "SELECT ok, cells_measured FROM v_source_status WHERE adapter = 'silent_store'") == [(0, 0)]
    view_extra = json.loads(_q(root, "SELECT extra_json FROM v_source_status WHERE adapter = 'silent_store'")[0][0])
    assert view_extra[RUN.EMITTED_NOTHING_KEY]["store_inputs"] == 1
    # measurements untouched: nothing was emitted and nothing is invented; the input's artifact is registered as always
    assert _q(root, "SELECT COUNT(*) FROM run") == [(0,)] and _q(root, "SELECT COUNT(*) FROM observation") == [(0,)]
    store_sha = ad.discover()[0].sha256
    assert _q(root, "SELECT COUNT(*) FROM artifact WHERE artifact_id = ?", store_sha) == [(1,)]
    # one log line per silent group, naming the source and the revision
    lines = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert len(lines) == 1 and lines[0].startswith("emitted nothing (unexplained): silent_store read 1 input(s) of silent_store:s1 (revision ")
    assert res.revision_id in lines[0] and "ok = 0" in lines[0] and "input kinds: report" in lines[0]


def test_empty_explanation_values_explain_nothing(tmp_path):
    """`held: []`, `missing: []`, `refuse: []` (the live shape of most facts) are not a reason: the flag still fires."""
    root, store_root = _setup(tmp_path)
    ad = SilentStoreAdapter(store_root, facts={"held": [], "missing": [], "refuse": [], "refused": "", "refusals": {}, "torn": None, "row_refusals": []})
    rep, _ = _pass_and_ingest(root, ad, synth.t(2000))
    assert rep.groups_emitted_nothing == 1
    ((_, ok, _, _, extra),) = _status_rows(root)
    assert ok == 0 and RUN.EMITTED_NOTHING_KEY in extra


def test_the_pass_report_counts_silent_groups_beside_refusals_and_the_normal_reads(tmp_path):
    root, store_root = _setup(tmp_path)
    p2 = _store(store_root, "s2", "legacy_store_2", "none", [0.1, 0.9])
    _store(store_root, "s3", "legacy_store_3", "none", [0.3])
    ad = SilentStoreAdapter(store_root, alias_groups=("s3",))
    rep, ing = _pass_and_ingest(root, ad, synth.t(2000))
    # s1 and s2 silent (flagged), s3 alias-only (accepted, not flagged): the count is per group, beside the other counters
    assert ing.refused == 0 and rep.groups_refreshed == 3 and rep.groups_refused == 0 and rep.groups_held == 0
    by = {r.group: r for r in rep.results}
    assert by["s1"].emitted_nothing and by["s2"].emitted_nothing and not by["s3"].emitted_nothing
    assert rep.groups_emitted_nothing == 2
    rows = {sid: (ok, extra) for sid, ok, _, _, extra in _status_rows(root)}
    assert rows["silent_store:s1"][0] == 0 and RUN.EMITTED_NOTHING_KEY in rows["silent_store:s1"][1]
    assert rows["silent_store:s2"][0] == 0 and RUN.EMITTED_NOTHING_KEY in rows["silent_store:s2"][1]
    assert rows["silent_store:s3"][0] == 1 and RUN.EMITTED_NOTHING_KEY not in rows["silent_store:s3"][1]
    assert p2  # the second store exists; nothing about it was refused


def test_a_refusal_raised_by_the_reader_is_a_refusal_not_a_silent_read(tmp_path):
    root, store_root = _setup(tmp_path)
    p2 = _store(store_root, "s2", "legacy_store_2", "none", [0.1, 0.9])

    class Refusing(SilentStoreAdapter):
        def read(self, ctx, group, inputs):
            for inp in inputs:
                if inp.uri in self.broken:
                    raise AB.AdapterRefusal(inp.uri, "synthetic read failure")
            yield AB.RecordBatch(RUN.source_id_for(self.name, group), group)

    ad = Refusing(store_root)
    ad.broken.add(p2)
    rep, _ = _pass_and_ingest(root, ad, synth.t(2000))
    assert rep.groups_refused == 1 and rep.groups_emitted_nothing == 1 and rep.groups_refreshed == 1
    by = {r.group: r for r in rep.results}
    assert by["s1"].emitted_nothing and not by["s2"].emitted_nothing and by["s2"].error and not by["s2"].accepted


# ============================================================================================
# the guard stays off
# ============================================================================================

@pytest.mark.parametrize("facts, ok_expected", [
    ({"refuse": ["index moved under the read"]}, 0),             # a recorded refusal flips ok on its own -- no flag on top
    ({"refused": "the standup names no readable aggregate"}, 1),
    ({"refusals": ["row 3: no run of the read matches"]}, 1),
    ({"row_refusals": ["s1|rows[0]: names no readable store"]}, 1),
    ({"missing": ["synth lam3: still running"]}, 1),
    ({"torn": ["s1: judged.jsonl shorter than responses.jsonl"]}, 1),
    ({"held": {"s1": "the ceiling pass has not landed"}}, 1),
    ({"held": "the bank holds no staged directory"}, 1),
    ({"held": True}, 1),
])
def test_a_recorded_reason_explains_the_empty_read_and_the_flag_stays_off(tmp_path, facts, ok_expected, caplog):
    root, store_root = _setup(tmp_path)
    ad = SilentStoreAdapter(store_root, facts=facts)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        rep, ing = _pass_and_ingest(root, ad, synth.t(2000))
    assert ing.refused == 0 and rep.groups_refreshed == 1 and rep.groups_emitted_nothing == 0 and not rep.results[0].emitted_nothing
    ((_, ok, _, detail, extra),) = _status_rows(root)
    assert ok == ok_expected and RUN.EMITTED_NOTHING_KEY not in extra and "EMITTED NOTHING" not in detail
    assert not [r for r in caplog.records if r.name == LOGGER]


def test_row_refusals_from_the_emitters_report_explain_the_empty_read(tmp_path):
    """The `row_refusals` list the runner lifts off `adapter.last_report` (the declared readers' EmitReport) is an explanation too."""
    root, store_root = _setup(tmp_path)
    ad = SilentStoreAdapter(store_root)
    ad.last_report = SimpleNamespace(refusals=["s1|rows[0]: the row names no readable store"])
    rep, _ = _pass_and_ingest(root, ad, synth.t(2000))
    assert rep.groups_emitted_nothing == 0
    ((_, ok, _, _, extra),) = _status_rows(root)
    assert ok == 1 and RUN.EMITTED_NOTHING_KEY not in extra and extra["row_refusals"] == ["s1|rows[0]: the row names no readable store"]


def test_an_alias_only_group_and_a_pointer_only_group_are_not_silent_but_a_mixed_group_is(tmp_path):
    root, store_root = _setup(tmp_path)
    _store(store_root, "s2", "legacy_store_2", "none", [0.1, 0.9])
    _store(store_root, "s3", "legacy_store_3", "none", [0.3])
    # s1: its store under a `#s1` alias (plus the shared index's alias); s2: a completion marker plus the index's alias; s3: a store
    # beside the index's alias. Only s3 has a store to emit from.
    ad = SilentStoreAdapter(store_root, alias_groups=("s1",), pointer_groups=("s2",), shared_index=True)
    rep, ing = _pass_and_ingest(root, ad, synth.t(2000))
    assert ing.refused == 0 and rep.groups_refreshed == 3 and rep.groups_emitted_nothing == 1
    rows = {sid: (ok, extra) for sid, ok, _, _, extra in _status_rows(root)}
    assert rows["silent_store:s1"][0] == 1 and RUN.EMITTED_NOTHING_KEY not in rows["silent_store:s1"][1]
    assert rows["silent_store:s2"][0] == 1 and RUN.EMITTED_NOTHING_KEY not in rows["silent_store:s2"][1]
    flag = rows["silent_store:s3"][1][RUN.EMITTED_NOTHING_KEY]
    assert rows["silent_store:s3"][0] == 0 and flag["inputs"] == 2 and flag["input_kinds"] == ["alias", "report"] and flag["store_inputs"] == 1


def test_input_is_alias_tells_a_per_group_alias_from_a_file_whose_name_carries_a_hash(tmp_path):
    p = tmp_path / "shared.json"
    p.write_text("{}")
    inp = AB.check_input(str(p), role="report")
    assert not RUN.input_is_alias(inp)
    assert RUN.input_is_alias(dataclasses.replace(inp, uri=f"{inp.uri}#qwen35_4b_d1p25_loradiff64s1"))
    literal = tmp_path / "rows#2.jsonl"
    literal.write_text("{}\n")
    assert not RUN.input_is_alias(AB.check_input(str(literal), role="rows"))
    # an alias of the file whose name carries a `#`: the uri does not resolve to the hashed path
    assert RUN.input_is_alias(dataclasses.replace(AB.check_input(str(literal), role="rows"), uri=f"{literal}#g"))


def test_a_bookkeeping_group_by_the_underscore_convention_never_fires(tmp_path):
    root, store_root = _setup(tmp_path)
    _store(store_root, "_root_files", "legacy_root", "none", [0.1])
    ad = SilentStoreAdapter(store_root)
    rep, ing = _pass_and_ingest(root, ad, synth.t(2000))
    assert ing.refused == 0 and rep.groups_refreshed == 2 and rep.groups_emitted_nothing == 1
    rows = {sid: (ok, extra) for sid, ok, _, _, extra in _status_rows(root)}
    assert rows["silent_store:_root_files"][0] == 1 and RUN.EMITTED_NOTHING_KEY not in rows["silent_store:_root_files"][1]
    assert rows["silent_store:s1"][0] == 0 and RUN.EMITTED_NOTHING_KEY in rows["silent_store:s1"][1]


def test_a_reader_declared_emits_no_runs_marker_bare_or_per_group_never_fires(tmp_path):
    root, store_root = _setup(tmp_path)
    _store(store_root, "s2", "legacy_store_2", "none", [0.1, 0.9])

    class DeclaredEverywhere(SilentStoreAdapter):
        emits_no_runs = True

    rep, ing = _pass_and_ingest(root, DeclaredEverywhere(store_root), synth.t(2000))
    assert ing.refused == 0 and rep.groups_refreshed == 2 and rep.groups_emitted_nothing == 0
    assert all(ok == 1 and RUN.EMITTED_NOTHING_KEY not in extra for _, ok, _, _, extra in _status_rows(root))

    class DeclaredPerGroup(SilentStoreAdapter):
        name = "silent_store_per_group"

        def emits_no_runs(self, group: str) -> bool:
            return group == "s2"

    rep2, ing2 = _pass_and_ingest(root, DeclaredPerGroup(store_root), synth.t(3000))
    assert ing2.refused == 0 and rep2.groups_refreshed == 2 and rep2.groups_emitted_nothing == 1
    rows = {sid: (ok, extra) for sid, ok, _, _, extra in _status_rows(root) if sid.startswith("silent_store_per_group:")}
    assert rows["silent_store_per_group:s2"][0] == 1 and RUN.EMITTED_NOTHING_KEY not in rows["silent_store_per_group:s2"][1]
    assert rows["silent_store_per_group:s1"][0] == 0 and RUN.EMITTED_NOTHING_KEY in rows["silent_store_per_group:s1"][1]
    # the marker helper itself
    assert AB.emits_no_runs_of(SimpleNamespace(), "g") is False
    assert AB.emits_no_runs_of(SimpleNamespace(emits_no_runs=True), "g") is True
    assert AB.emits_no_runs_of(SimpleNamespace(emits_no_runs=lambda g: g == "x"), "x") is True
    assert AB.emits_no_runs_of(SimpleNamespace(emits_no_runs=lambda g: g == "x"), "y") is False


def test_the_frameworks_known_facts_only_shapes_match_the_readers_contracts():
    """Each FACTS_ONLY_GROUPS entry restates a reader's own read(): the group it names is exempt, every other group of that reader
    (and of any other reader) is judged."""
    fake = lambda name: SimpleNamespace(name=name)  # noqa: E731
    assert RUN.facts_only_group(fake("cap4096_lane"), "declarations")
    assert not RUN.facts_only_group(fake("cap4096_lane"), "grades_cap4096_qwen35_4b")
    assert RUN.facts_only_group(fake("regrade_twin"), "manifests")
    assert not RUN.facts_only_group(fake("regrade_twin"), "/twin/regrade/swap_a/store_x")
    root = "/state/full22/sampled"
    for g in (f"{root}/ladders/gemma4e4b_it_d0p1_t0", f"{root}/standups/qwen35_4b", f"{root}/serving_records", f"{root}/final_rung"):
        assert RUN.facts_only_group(fake("sampled_static"), g), g
    assert not RUN.facts_only_group(fake("sampled_static"), f"{root}/grades_sampled/qwen35_4b_it_d0p5_t0")
    # the `_` convention holds for every reader; a name that merely CONTAINS an underscore does not
    assert RUN.facts_only_group(fake("strength_mtdiff"), "_root_files") and RUN.facts_only_group(fake("multiturn_grades"), "_lane_files")
    assert not RUN.facts_only_group(fake("strength_mtdiff"), "qwen35_4b_d1p25_loradiff64s1")
    # every entry is keyed by a reader name the framework does not otherwise know: the table is the stopgap, the marker the contract
    assert set(RUN.FACTS_ONLY_GROUPS) == {"cap4096_lane", "regrade_twin", "sampled_static"}


def test_a_vanished_groups_empty_revision_is_explained_by_missing_not_flagged(tmp_path):
    root, store_root = _setup(tmp_path)
    p2 = _store(store_root, "s2", "legacy_store_2", "none", [0.1, 0.9])
    ad = SyntheticStoreAdapter(store_root)
    _pass_and_ingest(root, ad, synth.t(2000))
    os.remove(p2)
    os.rmdir(os.path.dirname(p2))
    rep, ing = _pass_and_ingest(root, ad, synth.t(3000))
    assert ing.refused == 0 and rep.groups_emitted_nothing == 0
    rows = {sid: (ok, extra) for sid, ok, _, _, extra in _status_rows(root)}
    ok, extra = rows["synthetic_store:s2"]
    assert ok == 0 and extra["missing"] == [p2] and RUN.EMITTED_NOTHING_KEY not in extra and "record_counts" not in extra


# ============================================================================================
# a normal read is byte-identical to before
# ============================================================================================

def test_a_normal_read_carries_no_flag_and_its_status_record_is_byte_identical_with_or_without_the_guards_inputs(tmp_path):
    root, store_root = _setup(tmp_path)
    ad = SyntheticStoreAdapter(store_root)
    rep, ing = _pass_and_ingest(root, ad, synth.t(2000))
    assert ing.refused == 0 and rep.groups_refreshed == 1 and rep.groups_emitted_nothing == 0 and not rep.results[0].emitted_nothing
    ((_, ok, cells, detail, extra),) = _status_rows(root)
    assert ok == 1 and cells == 1 and "EMITTED NOTHING" not in detail
    # the exact key set of before: the generic lists, the emitter's row refusals, the record counts -- nothing else
    assert extra == {"refuse": [], "torn": [], "missing": [], "row_refusals": [],
                     "record_counts": {"item_result": 3, "judgment": 1, "run": 1, "run_metric": 1, "run_state": 1}}
    # the runner's status writer produces the same bytes whether or not it is handed the inputs (the guard's new parameter)
    inputs = ad.discover()
    ctx = AB.AdapterContext(root=root, now=synth.t(2000), producer={"script": "t", "code_sha": synth.code_sha("a"), "session": None, "host": "TESTHOST", "substrate": "grading_host"})
    tally = RUN._ReadTally()
    for batch in ad.read(ctx, "s1", inputs):
        for rec in batch.records:
            tally.note(rec)
    args = (ctx, ad, "s1", "synthetic_store:s1", "1" * 16, [i.uri for i in inputs], tally)
    assert RUN._source_status_records(*args) == RUN._source_status_records(*args, inputs=inputs)
    # an EMPTY tally with no inputs (the direct-call shape older tests use) is not a read of a store: no flag
    (rec,) = RUN._source_status_records(ctx, ad, "s1", "synthetic_store:s1", "1" * 16, [], RUN._ReadTally())
    assert RUN.EMITTED_NOTHING_KEY not in rec["payload"]["extra"] and rec["payload"]["ok"] == 1


def test_a_reread_of_a_normal_head_carries_no_flag_and_moves_no_pin(tmp_path):
    root, store_root = _setup(tmp_path)
    ad = SyntheticStoreAdapter(store_root)
    rep, _ = _pass_and_ingest(root, ad, synth.t(2000))
    head = rep.results[0].revision_id
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        rep2 = RUN.run_pass(root, [ad], conn=conn, now=synth.t(3000), reread=True)
    finally:
        conn.close()
    assert rep2.groups_reread == 1 and rep2.groups_emitted_nothing == 0 and rep2.results[0].revision_id == head
    # a silent reader's re-read of ITS head is flagged like a fresh read (the guard judges the read, not the revision)
    silent = SilentStoreAdapter(store_root)
    rep3, _ = _pass_and_ingest(root, silent, synth.t(4000))
    assert rep3.groups_emitted_nothing == 1
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        rep4 = RUN.run_pass(root, [silent], conn=conn, now=synth.t(5000), reread=True)
    finally:
        conn.close()
    assert rep4.groups_reread == 1 and rep4.groups_emitted_nothing == 1 and rep4.results[0].emitted_nothing


# ============================================================================================
# the CLI summary
# ============================================================================================

def test_adapters_run_reports_the_count_and_prints_one_line_per_silent_group(tmp_path, monkeypatch, capsys):
    root, store_root = _setup(tmp_path)
    _store(store_root, "s2", "legacy_store_2", "none", [0.1, 0.9])
    from sisteer.registry import adapters as ADP
    from sisteer.registry.adapters import dimensions as DIM
    monkeypatch.setattr(ADP, "load_all", lambda: {"silent_store": lambda: SilentStoreAdapter(store_root, alias_groups=("s2",))})
    monkeypatch.setattr(DIM, "adapter_legacy_spellings", lambda seeds: {})
    args = cli.build_parser().parse_args(["--root", root, "adapters", "run", "--workers", "0"])
    assert args.fn(args) == 0
    out = capsys.readouterr()
    summary, _ = json.JSONDecoder().raw_decode(out.out)              # the pass summary; the ingest summary follows it
    assert summary["groups_refreshed"] == 2 and summary["groups_refused"] == 0 and summary["groups_emitted_nothing"] == 1
    lines = [l for l in out.err.splitlines() if l.startswith("emitted nothing (unexplained): ")]
    assert len(lines) == 1 and "silent_store s1:" in lines[0] and "filed ok = 0" in lines[0]
