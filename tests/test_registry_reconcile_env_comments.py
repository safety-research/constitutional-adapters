"""`registry reconcile env-comments` (reconcile.py): the run-overrides stamp restated from its overrides FILE of record, where the old
reader kept inline comment text. One `run_patch` per `run` row whose `extra_json.run_overrides` differs from the reading the fixed reader
makes of the file whose sha256 the stamp carries (`--source`: the files; the repository's own `reproduce/overrides` is always read), the
WHOLE stamp under `fields.extra_json.run_overrides` (path and sha256 as stored, keys the file's reading, id the file's OVERRIDE_ID), a
marker naming every replaced key with the value it replaced and the file, and a confidence entry on the one column the record restates.
The evidence is the file, never the stored text: a stored `note #42` read from `'note #42'` is preserved, not stripped, and every value
the pass replaces is established PER KEY as what the OLD reader made of the file's line (`historical_env_value`, the reader before the fix
read over the same file). A stamp whose file is not given and whose values may carry a comment is AMBIGUOUS (listed, never corrected --
and NOT a clean class: the apply refuses it by name and the verify lists it until its file is given or `--leave-unsourced` leaves it as
stored by name; `--check-sources` is that coverage alone, what the landing window runs before anything moves); a stamp whose file is given
but whose key set is not the file's, or whose value under some key is neither the file's reading nor the old reader's (a later statement
restating one key while keeping the file as provenance), is UNCLASSIFIED (refused by name, nothing of it restated). run-patch-extra compares
a legacy stamp to the live one by the same evidence. Only `run` is corrected: the other tables' rows are full statements, counted by the
dry run and never corrected. Folded and rebuilt through the real harvester on a temp root; the CLI's exit codes pinned; the hold behind an
unfolded statement of the stamp; the writer stamps nothing of its own.
"""
from __future__ import annotations

import hashlib
import json
import os

import pytest

from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import emit
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import reconcile as RC
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry import validate as VAL

#: the overrides file as the runs saw it: three commented lines the old reader kept the comments of (an unquoted value, a quoted empty
#: value, an EMPTY value before a comment), two values the shell reads with a literal '#' (quoted; no whitespace before it), one clean
SOURCE_NAME = "synthetic_pipeline.env"
SOURCE_TEXT = ("# the synthetic pipeline's overrides\n"
               "OVERRIDE_ID=synthetic_pipeline_v1\n"
               "OVERRIDE_EFFECTIVE_BATCH=64                  # unchanged: the paper's\n"
               "OVERRIDE_LORA_RANK=''                        # the pick from the published table\n"
               "OVERRIDE_HPO_WARMUP_FRAC=   # empty means the default\n"
               "OVERRIDE_HOLD_OUT_EVALS=\"codeattack,agentic_murder\"\n"
               "OVERRIDE_NOTE='note #42'\n"
               "OVERRIDE_TAG=v#x\n"
               "   # an indented comment line\n")
SOURCE_SHA = hashlib.sha256(SOURCE_TEXT.encode()).hexdigest()


def _old_reading(text: str):
    """The reader BEFORE the fix, replicated: the whole right-hand side kept, surrounding quotes stripped only when they wrap it whole."""
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        if k.startswith("OVERRIDE_"):
            out[k] = v
    return out


#: the stamp as the old reader stamped the file: three values with their comment text; `note #42` and `v#x` as the shell reads them
POLLUTED = {"id": "synthetic_pipeline_v1", "path": SOURCE_NAME, "sha256": SOURCE_SHA, "keys": _old_reading(SOURCE_TEXT)}
assert POLLUTED["keys"]["OVERRIDE_EFFECTIVE_BATCH"] == "64                  # unchanged: the paper's"
assert POLLUTED["keys"]["OVERRIDE_LORA_RANK"] == "''                        # the pick from the published table"
assert POLLUTED["keys"]["OVERRIDE_HPO_WARMUP_FRAC"] == "# empty means the default"
assert POLLUTED["keys"]["OVERRIDE_NOTE"] == "note #42" and POLLUTED["keys"]["OVERRIDE_TAG"] == "v#x"
#: the same file read by the fixed reader
CLEAN = {**POLLUTED, "keys": {"OVERRIDE_ID": "synthetic_pipeline_v1", "OVERRIDE_EFFECTIVE_BATCH": "64", "OVERRIDE_LORA_RANK": "",
                              "OVERRIDE_HPO_WARMUP_FRAC": "", "OVERRIDE_HOLD_OUT_EVALS": "codeattack,agentic_murder",
                              "OVERRIDE_NOTE": "note #42", "OVERRIDE_TAG": "v#x"}}
REPLACED = {k: POLLUTED["keys"][k] for k in ("OVERRIDE_EFFECTIVE_BATCH", "OVERRIDE_LORA_RANK", "OVERRIDE_HPO_WARMUP_FRAC")}
#: a stamp of a file NOBODY has (its digest matches no source) whose values may carry a comment: ambiguous, listed, never corrected --
#: `note #42` here is undecidable without the file
UNSOURCED_POLLUTED = {"id": "lost_file_v2", "path": "lost_file.env", "sha256": "9" * 64,
                      "keys": {"OVERRIDE_ID": "lost_file_v2", "OVERRIDE_X": "1   # c", "OVERRIDE_NOTE": "note #42", "OVERRIDE_Y": "2"}}
#: a stamp of a file nobody has whose values look clean: nothing to list
UNSOURCED_CLEAN = {"id": "other_v1", "path": "other.env", "sha256": "a" * 64, "keys": {"OVERRIDE_ID": "other_v1", "OVERRIDE_TAG": "t", "OVERRIDE_Z": "3"}}
#: the source file's digest, but a key the file does not have: no reading of that file -- unclassified, refused by name
MISMATCH = {**CLEAN, "keys": {**CLEAN["keys"], "OVERRIDE_MODEL": "m"}}
#: the source file's digest and key set, every value clean but ONE restated to a value the file never carried (a later accepted patch that
#: restated the key for one run and kept the file as provenance): a later statement, not the reader's artifact -- unclassified, refused by
#: name, never restated to the file's 64
DIVERGENT = {**CLEAN, "keys": {**CLEAN["keys"], "OVERRIDE_EFFECTIVE_BATCH": "32"}}
#: the same divergent value beside the old reader's comment text on the other keys: refused whole, nothing of it restated
DIVERGENT_POLLUTED = {**POLLUTED, "keys": {**POLLUTED["keys"], "OVERRIDE_EFFECTIVE_BATCH": "32"}}
#: a file whose OVERRIDE_ID line carries a comment: the old reader polluted the stamp's `id` too
ID_COMMENTED_TEXT = "OVERRIDE_ID=second_v1   # the id\nOVERRIDE_K=1\n"
ID_COMMENTED_SHA = hashlib.sha256(ID_COMMENTED_TEXT.encode()).hexdigest()
OVERRIDES_TEXT = "OVERRIDE_ID=operator_shell_v9\nexport OVERRIDE_INIT_SCHEME=random_lecun_1e-3   # an operator's shell\n"


def _root(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    return root


def _source(tmp_path):
    d = tmp_path / "sources"
    d.mkdir(exist_ok=True)
    f = d / SOURCE_NAME
    f.write_text(SOURCE_TEXT)
    return str(f)


def _write(root, sub, name, recs, gz=False):
    return R.write_records(os.path.join(root, "records", sub, name), recs, gz=gz)


def _run(root, key, i0, stamp, **over):
    """A run through `running` whose registration carries ``stamp`` under `extra.run_overrides` (and another extra key beside it)."""
    rid, recs = synth.run_records(key, i0, through="running", extra={"run_overrides": stamp, "cluster_label": "TESTSITE"}, **over)
    sub = os.path.join(synth.CAMPAIGN, I.run_dir_name(rid))
    _write(root, sub, "000001-run.jsonl", recs)
    return rid, sub


def _corpus(tmp_path, with_unclassified=False, with_divergent=False, with_unsourced=False):
    """Runs A (the polluted stamp of the source file: SELECTED), B (the clean stamp of the same file: a fixed point, not listed), C2 (a
    clean-looking stamp of a file nobody has: nothing), D (polluted, through `judged`, so the other tables carry the stamp too: SELECTED),
    with ``with_unsourced`` C (the polluted stamp of a file nobody has: AMBIGUOUS, listed, never corrected, and refused by the apply and
    listed by the verify until its file is given or it is left by name), with ``with_unclassified`` F (the source's digest with a key the
    file does not have: UNCLASSIFIED, refused by name) and, with ``with_divergent``, G (registered with the polluted stamp, then a FOLDED
    run_patch restating the stamp with one clean value the file never carried: UNCLASSIFIED, refused by name, the 32 kept). Returns the
    root, the source file's path and the runs."""
    root = _root(tmp_path)
    src = _source(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    a, sub_a = _run(root, "ovA", 100, POLLUTED)
    b, _ = _run(root, "ovB", 110, CLEAN)
    c = None
    if with_unsourced:
        c, _ = _run(root, "ovC", 120, UNSOURCED_POLLUTED)
    c2, _ = _run(root, "ovC2", 125, UNSOURCED_CLEAN)
    d_id, d_recs = synth.run_records("ovD", 130, through="judged", extra={"run_overrides": POLLUTED})
    # the writer stamps EVERY record of a run (the run, its state events, the judgment, the metric): the synthetic lifecycle carries the
    # stamp on the run alone, so the other records are re-made with it, as the producer's writer would have stamped them
    d_recs = [r if r["record_type"] in ("run", "run_patch") else synth.rec(r["record_type"], {**r["payload"], "extra": {"run_overrides": POLLUTED}}, r["written_utc"])
              for r in d_recs]
    sub_d = os.path.join(synth.CAMPAIGN, I.run_dir_name(d_id))
    _write(root, sub_d, "000001-run.jsonl", d_recs)
    runs = {"a": (a, sub_a), "b": b, "c": c, "c2": c2, "d": (d_id, sub_d)}
    if with_unclassified:
        f, _ = _run(root, "ovF", 140, MISMATCH)
        runs["f"] = f
    if with_divergent:
        g, sub_g = _run(root, "ovG", 150, POLLUTED)
        _write(root, sub_g, "000002-run_patch.jsonl", [synth.rec("run_patch", {"run_id": g, "fields": {"extra_json": {"run_overrides": DIVERGENT}}}, synth.t(155))])
        runs["g"] = (g, sub_g)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    return root, src, runs


def _live(root, rid):
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = conn.execute("SELECT extra_json, field_confidence_json FROM run WHERE run_id = ?", (rid,)).fetchone()
    conn.close()
    assert row is not None, rid
    return json.loads(row[0] or "{}"), json.loads(row[1] or "{}")


def _read_one(path):
    (_i, rec, _s), = list(R.read_record_file(path, compute_sha=False))
    return rec


def _run_files(root, sub):
    return sorted(os.listdir(os.path.join(root, "records", sub)))


def _rebuild_equal(root, tmp_path):
    live = DB.connect(DB.db_path(root), readonly=True)
    out = str(tmp_path / "rebuilt.sqlite")
    H.rebuild(root, out, workers=0)
    rebuilt = DB.connect(out, readonly=True)
    diff = {k: v for k, v in H.verify_equal(live, rebuilt).items() if v}
    assert not diff, diff
    live.close()
    rebuilt.close()


def _tree(root):
    out = {}
    for dp, _dns, fns in os.walk(root):
        for fn in fns:
            if fn.endswith(("-wal", "-shm")):
                continue
            p = os.path.join(dp, fn)
            out[os.path.relpath(p, root)] = (os.path.getsize(p), os.path.getmtime(p))
    return out


# --------------------------------------------------------------------------------------------
# the pure rules: the sources by digest, the restatement from a file, the flags, the one-file comparison
# --------------------------------------------------------------------------------------------


def test_the_sources_are_read_by_digest_from_files_and_directories_and_a_missing_path_is_refused(tmp_path):
    src = _source(tmp_path)
    other = tmp_path / "elsewhere" / "renamed_copy.env"
    other.parent.mkdir()
    other.write_text(SOURCE_TEXT)                                           # the same bytes under another name: one source
    second = b"OVERRIDE_ID=second_v1\nOVERRIDE_K=1\n"
    (tmp_path / "elsewhere" / "second.env").write_bytes(second)
    (tmp_path / "elsewhere" / "notes.txt").write_text("OVERRIDE_ID=not_an_env_file\n")      # a directory contributes its .env files only
    srcs = RC.overrides_sources([src, str(tmp_path / "elsewhere")], include_defaults=False)
    assert set(srcs) == {SOURCE_SHA, hashlib.sha256(second).hexdigest()}
    s = srcs[SOURCE_SHA]
    assert (s.path, s.sha256, s.id, s.name) == (src, SOURCE_SHA, "synthetic_pipeline_v1", SOURCE_NAME) and s.keys == CLEAN["keys"]
    assert srcs[hashlib.sha256(second).hexdigest()].keys == {"OVERRIDE_ID": "second_v1", "OVERRIDE_K": "1"}
    with pytest.raises(FileNotFoundError):
        RC.overrides_sources([str(tmp_path / "nowhere.env")], include_defaults=False)
    # the repository's own overrides directory is read without being named (digest-matched: it matches no synthetic stamp)
    assert os.path.isdir(RC.REPO_OVERRIDES_DIR) and RC.default_source_dirs() == [RC.REPO_OVERRIDES_DIR]
    defaults = RC.overrides_sources([], include_defaults=True)
    assert defaults and all(s_.path.startswith(RC.REPO_OVERRIDES_DIR) for s_ in defaults.values()) and SOURCE_SHA not in defaults
    assert SOURCE_SHA in RC.overrides_sources([src])


def test_a_stamp_is_restated_from_its_file_keeping_a_literal_hash_and_the_stored_text_alone_is_only_a_flag(tmp_path):
    src_path = _source(tmp_path)
    (tmp_path / "sources" / "second.env").write_text(ID_COMMENTED_TEXT)
    srcs = RC.overrides_sources([src_path, str(tmp_path / "sources" / "second.env")], include_defaults=False)
    src, second = srcs[SOURCE_SHA], srcs[ID_COMMENTED_SHA]
    # the source carries BOTH readings of its file: the fixed reader's (`keys`) and the old reader's (`historical_keys`, the reader before
    # the fix replicated in the module, read over the same lines) -- the test's own replica of the old reader agrees on every key
    assert src.keys == CLEAN["keys"] and src.historical_keys == POLLUTED["keys"] == _old_reading(SOURCE_TEXT)
    assert emit._parse_env_file(src_path, RC.historical_env_value) == {**POLLUTED["keys"]}
    assert RC.historical_env_value("64   # unchanged") == "64   # unchanged" and RC.historical_env_value("''   # c") == "''   # c"
    assert RC.historical_env_value("   # c") == "# c" and RC.historical_env_value("'note #42'") == "note #42" and RC.historical_env_value('"a # b"') == "a # b"
    assert src.readings("OVERRIDE_EFFECTIVE_BATCH") == ("64", "64                  # unchanged: the paper's") and src.readings("OVERRIDE_NOTE") == ("note #42",)
    assert second.keys == {"OVERRIDE_ID": "second_v1", "OVERRIDE_K": "1"} and second.historical_keys == {"OVERRIDE_ID": "second_v1   # the id", "OVERRIDE_K": "1"}
    # the polluted stamp: the three commented values restated, `note #42` and `v#x` untouched (the file says they are literal); every
    # replaced value IS the old reader's reading of its line
    restated, replaced, refusal = RC.source_restatement(POLLUTED, src)
    assert (restated, replaced, refusal) == (CLEAN, REPLACED, None)
    assert all(v == src.historical_keys[k] for k, v in replaced.items())
    # the clean stamp is a fixed point; the restated stamp too
    assert RC.source_restatement(CLEAN, src) == (CLEAN, {}, None) and RC.source_restatement(restated, src) == (CLEAN, {}, None)
    # an `id` the old reader polluted (the OVERRIDE_ID line commented) is restated from the file and named in the replaced map
    polluted_id = {"id": "second_v1   # the id", "path": "second.env", "sha256": ID_COMMENTED_SHA, "keys": dict(second.historical_keys)}
    clean_id = {**polluted_id, "id": "second_v1", "keys": dict(second.keys)}
    r2, rep2, why2 = RC.source_restatement(polluted_id, second)
    assert (r2, rep2, why2) == (clean_id, {"OVERRIDE_ID": "second_v1   # the id", "id": "second_v1   # the id"}, None)
    assert second.ids_for(polluted_id) == ("second_v1", "second_v1   # the id") and src.ids_for(POLLUTED) == ("synthetic_pipeline_v1", "synthetic_pipeline_v1")
    # a stamp with the file's digest but not the file's key set is no reading of that file: refused by name, nothing restated
    r3, rep3, why = RC.source_restatement(MISMATCH, src)
    assert (r3, rep3) == (MISMATCH, {}) and "not a reading of synthetic_pipeline.env" in why and "['OVERRIDE_MODEL']" in why
    assert RC.source_restatement({"id": "x", "keys": "not an object"}, src)[2] == "the stamp's keys is not an object"
    # A VALUE THAT IS NEITHER READING is a later statement, not the reader's artifact: refused by name with the key, the stored value and
    # both readings, and NOTHING of the stamp is restated -- not the divergent key (never overwritten with the file's 64), and not the
    # polluted keys beside it either (the stamp as a whole is no reading of the file)
    for odd in (DIVERGENT, DIVERGENT_POLLUTED):
        r4, rep4, why4 = RC.source_restatement(odd, src)
        assert (r4, rep4) == (odd, {}), odd
        assert why4.startswith("OVERRIDE_EFFECTIVE_BATCH is '32', neither the file's reading ('64') nor what the old reader made of its line "
                               "(\"64                  # unchanged: the paper's\"): a later statement, not the reader's artifact; not a reading of synthetic_pipeline.env"), why4
        assert why4.endswith("refused by name, nothing restated")
    # ... and an `id` that is neither the file's nor the old reader's, the same
    r5, rep5, why5 = RC.source_restatement({**polluted_id, "id": "someone_else"}, second)
    assert rep5 == {} and r5["id"] == "someone_else" and "id is 'someone_else', neither the file's ('second_v1') nor the old reader's ('second_v1   # the id')" in why5
    # the flag for rows whose file is not given: the stored text alone -- `note #42` reads as a comment, `#tag` as what `KEY=   # tag`
    # left, `v#x` and `64` as fixed points; never a correction, only a listing
    assert RC.looks_polluted("note #42") and RC.looks_polluted("1   # c") and RC.looks_polluted("#tag") and RC.looks_polluted("''   # c")
    assert not RC.looks_polluted("v#x") and not RC.looks_polluted("64") and not RC.looks_polluted("") and not RC.looks_polluted(7)
    # the comparison run-patch-extra uses, by the same evidence: two differing stamps are one file only when the file is given and each is a
    # reading of it (old or fixed) -- the polluted and the clean stamp are; a divergent value is not, whichever side carries it; a different
    # digest, path or key set is not; a stamp without a digest never is; WITHOUT the file nothing is
    assert RC.same_overrides_stamp(POLLUTED, CLEAN, srcs) and RC.same_overrides_stamp(CLEAN, POLLUTED, srcs) and RC.same_overrides_stamp(CLEAN, CLEAN, srcs)
    assert not RC.same_overrides_stamp(CLEAN, DIVERGENT, srcs) and not RC.same_overrides_stamp(POLLUTED, DIVERGENT, srcs) and not RC.same_overrides_stamp(DIVERGENT, POLLUTED, srcs)
    assert not RC.same_overrides_stamp(POLLUTED, DIVERGENT_POLLUTED, srcs)
    assert not RC.same_overrides_stamp(CLEAN, MISMATCH, srcs) and not RC.same_overrides_stamp(CLEAN, {**CLEAN, "sha256": "f" * 64}, srcs)
    assert not RC.same_overrides_stamp(CLEAN, {**CLEAN, "path": "renamed.env"}, srcs) and not RC.same_overrides_stamp({**CLEAN, "sha256": ""}, {**CLEAN, "sha256": ""}, srcs)
    assert not RC.same_overrides_stamp(CLEAN, "x", srcs) and not RC.same_overrides_stamp(None, None, srcs)
    assert not RC.same_overrides_stamp(POLLUTED, CLEAN) and not RC.same_overrides_stamp(POLLUTED, CLEAN, {}) and not RC.same_overrides_stamp(POLLUTED, CLEAN, {ID_COMMENTED_SHA: second})
    # the classification of one stamp: selected / ambiguous / unclassified / nothing
    assert RC.classify_stamp("r", "c", POLLUTED, srcs).selected and RC.classify_stamp("r", "c", CLEAN, srcs) is None
    amb = RC.classify_stamp("r", "c", UNSOURCED_POLLUTED, srcs)
    assert amb.ambiguous and not amb.selected and amb.suspicious == ("OVERRIDE_NOTE", "OVERRIDE_X") and amb.source is None
    assert RC.classify_stamp("r", "c", UNSOURCED_CLEAN, srcs) is None
    assert RC.classify_stamp("r", "c", MISMATCH, srcs).unclassified and RC.classify_stamp("r", "c", POLLUTED, {}).ambiguous
    odd_row = RC.classify_stamp("r", "c", DIVERGENT, srcs)
    assert odd_row.unclassified and not odd_row.selected and not odd_row.ambiguous and odd_row.replaced == {} and odd_row.restated == DIVERGENT
    assert RC.classify_stamp("r", "c", DIVERGENT_POLLUTED, srcs).unclassified


# --------------------------------------------------------------------------------------------
# the pass on a temp root: dry run, apply, fold, verify, idempotence, rebuild
# --------------------------------------------------------------------------------------------


def test_dry_run_selects_the_polluted_runs_of_a_given_file_lists_the_unsourced_ones_counts_the_other_tables_and_writes_nothing(tmp_path, capsys):
    root, src, runs = _corpus(tmp_path, with_unclassified=True, with_unsourced=True)
    a, _ = runs["a"]
    d, _ = runs["d"]
    assert _live(root, a)[0]["run_overrides"] == POLLUTED                   # folded as the producer stated it
    rep = RC.reconcile_env_comments(root, apply=False, sources=[src])
    assert [r.run_id for r in rep.selected] == sorted([a, d]) and not rep.applied and rep.written == [] and rep.held == []
    assert [r.run_id for r in rep.ambiguous] == [runs["c"]] and [r.run_id for r in rep.unclassified] == [runs["f"]]
    assert (rep.stamped, rep.sourced) == (6, 4) and [s.sha256 for s in rep.sources if s.sha256 == SOURCE_SHA] == [SOURCE_SHA]
    run_a = next(r for r in rep.selected if r.run_id == a)
    assert run_a.replaced == REPLACED and run_a.restated == CLEAN and run_a.pending is None and run_a.group == ("synthetic_pipeline_v1", SOURCE_SHA[:8])
    assert run_a.source.path == src
    assert rep.by_group() == {("synthetic_pipeline_v1", SOURCE_SHA[:8]): 2}
    assert rep.by_key() == {"OVERRIDE_EFFECTIVE_BATCH": 2, "OVERRIDE_HPO_WARMUP_FRAC": 2, "OVERRIDE_LORA_RANK": 2}
    assert rep.ambiguous_by_group() == {("lost_file_v2", "99999999"): (1, ("OVERRIDE_NOTE", "OVERRIDE_X"))}
    # the other tables: D's state events (queued, running, generated, judged), its judgment and its metric carry the polluted stamp as full
    # statements -- counted, never corrected; the item-scale table is not scanned
    assert rep.row_tables == {"run_state_event": 4, "judgment": 1, "run_metric": 1, "run_degeneration": 0, "object": 0, "fit_search": 0, "fit_trial": 0,
                              "item_result": None}
    d_ = rep.as_dict()
    assert d_["runs"] == 2 and d_["written"] == 0 and d_["ambiguous_runs"] == 1 and d_["unclassified"] == 1 and d_["sourced_runs"] == 4
    assert d_["row_tables_not_corrected"]["run_state_event"] == 4 and d_["by_key"]["OVERRIDE_LORA_RANK"] == 2
    assert d_["ambiguous"] == [{"id": "lost_file_v2", "sha256_prefix": "99999999", "runs": 1, "suspicious_keys": ["OVERRIDE_NOTE", "OVERRIDE_X"]}]
    # WITHOUT the file every stamp whose values MAY carry a comment is ambiguous -- the clean B and the mismatched F included, since their
    # `note #42` is undecidable from the stored text alone -- nothing is selected, nothing would be written: the stored text is never evidence
    blind = RC.reconcile_env_comments(root, apply=False, sources=[])
    assert blind.selected == [] and sorted(r.run_id for r in blind.ambiguous) == sorted([a, runs["b"], runs["c"], d, runs["f"]]) and blind.unclassified == []
    assert next(r for r in blind.ambiguous if r.run_id == runs["b"]).suspicious == ("OVERRIDE_NOTE",)
    # the CLI's dry run: the sources, the table by file, the keys, the ambiguous and unclassified lines, the other tables, exit 0, nothing
    # written under the root (no last_run stamp); --source on another action is refused; a source that is not there is refused
    before = _tree(root)
    assert cli.main(["--root", root, "reconcile", "env-comments", "--source", src]) == 0
    assert cli.main(["--root", root, "reconcile", "env-comments", "--dry-run", "--json", "--source", src]) == 0
    out = capsys.readouterr().out
    assert f"  {SOURCE_SHA[:8]} {'synthetic_pipeline_v1':<36} {src}" in out and "stamped run rows 6; with a source 4" in out
    assert f"  {'synthetic_pipeline_v1':<36} {SOURCE_SHA[:8]:<8} {2:>6}" in out and "by key: OVERRIDE_EFFECTIVE_BATCH 2, OVERRIDE_HPO_WARMUP_FRAC 2, OVERRIDE_LORA_RANK 2" in out
    assert ("ambiguous (no file given for the stamp and values that may carry a comment; listed, never corrected; --apply refuses them and --verify "
            "lists them until the file is given or --leave-unsourced leaves them by name): 1 run(s)\n"
            f"  {'lost_file_v2':<36} {'99999999':<8} {1:>6}  keys: OVERRIDE_NOTE, OVERRIDE_X") in out
    assert f"unclassified (a file given, but the stamp is no reading of it; refused by name, never corrected): 1 run(s)\n  {runs['f']} the stored key set is not the file's" in out
    assert "other tables (full statements, not corrected by this action): run_state_event 4, judgment 1, run_metric 1, run_degeneration 0, object 0, fit_search 0, fit_trial 0, item_result not counted" in out
    assert "runs 2 in 1 (overrides id, file sha) group(s); nothing written (dry run)" in out and '"row_tables_not_corrected"' in out and '"sourced_runs": 4' in out
    assert _tree(root) == before and not os.path.exists(os.path.join(root, "state", "last_run"))
    assert cli.main(["--root", root, "reconcile", "multi-turn-scope", "--source", src]) == 1
    assert "--source belongs to env-comments" in capsys.readouterr().err
    assert cli.main(["--root", root, "reconcile", "env-comments", "--source", str(tmp_path / "nowhere.env")]) == 1
    assert "refused: no overrides file or directory at" in capsys.readouterr().err


def test_apply_writes_one_patch_per_run_that_folds_to_the_files_reading_preserving_the_literal_hash_verify_reads_zero_and_a_second_pass_selects_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(emit.socket, "gethostname", lambda: "TESTHOST")
    root, src, runs = _corpus(tmp_path, with_unsourced=True)
    a, sub_a = runs["a"]
    d, sub_d = runs["d"]
    c = runs["c"]
    assert _run_files(root, sub_a) == ["000001-run.jsonl"]
    app = RC.reconcile_env_comments(root, apply=True, sources=[src])
    # the two selected runs are written; the unsourced C is listed, never written, and REFUSED by name -- the apply does not complete past a
    # stamp it could not decide (exit 2 below), the way out named: its file, or the operator's word
    assert app.applied and len(app.written) == 2 and [r.run_id for r in app.held] == [c] and [r.run_id for r in app.ambiguous] == [c]
    assert app.held[0].refused == app.held[0].unsourced_reason and app.held[0].written is None
    assert app.held[0].refused.startswith("no overrides file among the sources has this stamp's sha256 (999999999999; stored path 'lost_file.env') and its "
                                          "values may carry a comment (OVERRIDE_NOTE, OVERRIDE_X): the stored text is never evidence, so nothing is restated; "
                                          "name the file with --source, or leave the stamp as stored by name with --leave-unsourced")
    run_a = next(r for r in app.selected if r.run_id == a)
    assert run_a.written == os.path.join(root, "records", sub_a, "000002-run_patch.jsonl")      # the run's OWN directory, next in sequence
    rec = _read_one(run_a.written)
    assert rec["record_type"] == "run_patch" and rec["producer"]["script"] == RC.ENV_COMMENTS_PRODUCER_SCRIPT and "extra" not in rec["payload"]
    conf_src = rec["payload"]["fields"]["field_confidence_json"]["extra_json"]["source"]
    assert rec["payload"] == {"run_id": a, "fields": {
        "extra_json": {"run_overrides": CLEAN,
                       RC.ENV_COMMENTS_RECONCILED_KEY: {"replaced_keys": REPLACED, "source_file": SOURCE_NAME, "source_sha256": SOURCE_SHA, "pass": RC.ENV_COMMENTS_PRODUCER_SCRIPT}},
        "field_confidence_json": {"extra_json": {"confidence": "recorded", "source": conf_src}}}}
    assert RC.ENV_COMMENTS_PRODUCER_SCRIPT in conf_src and SOURCE_NAME in conf_src and SOURCE_SHA[:12] in conf_src
    assert isinstance(VAL.validate_record(rec, run_a.written, R.record_sha(rec)), VAL.Validated)
    # before the ingest a second apply holds both runs (their records are in the log, not folded) and the CLI says so with exit 2
    again = RC.reconcile_env_comments(root, apply=True, sources=[src])
    assert again.written == [] and all(r.pending == R.record_sha(_read_one(r_.written)) and r.pending_own for r in again.selected for r_ in app.selected if r_.run_id == r.run_id)
    assert all("this pass's own correction record" in (r.refused or "") for r in again.selected)
    assert cli.main(["--root", root, "reconcile", "env-comments", "--apply", "--source", src]) == 2
    err = capsys.readouterr().err
    assert "3 row(s) held" in err and "unsourced: name the file with --source, or leave the stamp as stored by name with --leave-unsourced" in err
    # the verify before the ingest still lists both (the predicate reads the folded row), the unsourced C among the listed rows (its own
    # line, the id first, as every listed row's), the ambiguous group beside them with the consequence, exit 3
    assert cli.main(["--root", root, "reconcile", "env-comments", "--verify", "--source", src]) == 3
    out = capsys.readouterr().out
    assert f"  {a} OVERRIDE_EFFECTIVE_BATCH=" in out and "-> '64'" in out and out.rstrip().endswith("polluted 3 run(s) (1 unsourced)")
    assert f"  {c} unsourced: no file given for this stamp (lost_file_v2 99999999); values that may carry a comment: OVERRIDE_NOTE, OVERRIDE_X" in out
    assert ("ambiguous (no file given for this stamp; values that may carry a comment: OVERRIDE_NOTE, OVERRIDE_X): lost_file_v2 99999999 1 run(s) -- listed "
            "above: not a clean class until the file is given or --leave-unsourced") in out
    # the fold: the stamp is the file's reading whole (path, sha256 unchanged; `note #42` and `v#x` as the shell reads them), the marker
    # beside it, the other extra key stands, the confidence entry lands; B (clean), C (unsourced) and C2 are untouched
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    extra, conf = _live(root, a)
    assert extra["run_overrides"] == CLEAN and extra["run_overrides"]["keys"]["OVERRIDE_NOTE"] == "note #42" and extra["run_overrides"]["keys"]["OVERRIDE_HPO_WARMUP_FRAC"] == ""
    assert extra["cluster_label"] == "TESTSITE" and extra["verified_fields"] == {"serving_path": "lora_native"}
    assert extra[RC.ENV_COMMENTS_RECONCILED_KEY] == {"replaced_keys": REPLACED, "source_file": SOURCE_NAME, "source_sha256": SOURCE_SHA, "pass": RC.ENV_COMMENTS_PRODUCER_SCRIPT}
    assert conf["extra_json"]["confidence"] == "recorded" and RC.ENV_COMMENTS_PRODUCER_SCRIPT in conf["extra_json"]["source"]
    assert _live(root, d)[0]["run_overrides"] == CLEAN
    assert _live(root, runs["b"])[0]["run_overrides"] == CLEAN and _live(root, runs["c"])[0]["run_overrides"] == UNSOURCED_POLLUTED
    assert _live(root, runs["c2"])[0]["run_overrides"] == UNSOURCED_CLEAN
    # the other tables' rows keep the producer's stamp: full statements, outside this action
    conn = DB.connect(DB.db_path(root), readonly=True)
    (ev_extra,) = conn.execute("SELECT extra_json FROM run_state_event WHERE run_id = ? AND state = 'judged'", (d,)).fetchone()
    conn.close()
    assert json.loads(ev_extra)["run_overrides"] == POLLUTED
    # the verify: the selected rows left the class, but the unsourced C still FAILS it (exit 3) -- a class the pass could not decide is not a
    # clean class -- until the operator leaves it by name, which reads zero with the clause and the group marked left; the dry run's
    # other-table counts stand (not corrected)
    ver = RC.verify_env_comments(root, sources=[src])
    assert [r.run_id for r in ver.listed] == [c] and ver.verify and ver.selected == [] and ver.unclassified == [] and [r.run_id for r in ver.unsourced_failing] == [c]
    assert RC.format_overrides_report(ver).endswith(f"  {c} unsourced: no file given for this stamp (lost_file_v2 99999999); values that may carry a comment: OVERRIDE_NOTE, OVERRIDE_X\n"
                                                    "ambiguous (no file given for this stamp; values that may carry a comment: OVERRIDE_NOTE, OVERRIDE_X): lost_file_v2 99999999 1 run(s) "
                                                    "-- listed above: not a clean class until the file is given or --leave-unsourced\npolluted 1 run(s) (1 unsourced)")
    assert cli.main(["--root", root, "reconcile", "env-comments", "--verify", "--source", src]) == 3
    left = RC.verify_env_comments(root, sources=[src], leave_unsourced=True)
    assert left.listed == [] and left.unsourced_failing == [] and [r.run_id for r in left.ambiguous] == [c] and left.leave_unsourced
    assert RC.format_overrides_report(left).endswith("ambiguous (no file given for this stamp; values that may carry a comment: OVERRIDE_NOTE, OVERRIDE_X): lost_file_v2 99999999 1 run(s) "
                                                     "-- left as stored by name\npolluted 0 run(s); the class is consistent; 1 unsourced run(s) left as stored by name (--leave-unsourced)")
    assert cli.main(["--root", root, "reconcile", "env-comments", "--verify", "--source", src, "--leave-unsourced"]) == 0
    assert RC.reconcile_env_comments(root, sources=[src]).row_tables["run_state_event"] == 4
    # idempotent: a second apply after the fold selects nothing and writes nothing; C is still refused by name (exit 2) unless left by name
    # (exit 0, nothing written either way); the folded stamp IS the file's reading
    third = RC.reconcile_env_comments(root, apply=True, sources=[src])
    assert third.selected == [] and third.written == [] and [r.run_id for r in third.ambiguous] == [c] and [r.run_id for r in third.held] == [c]
    assert cli.main(["--root", root, "reconcile", "env-comments", "--apply", "--source", src]) == 2
    fourth = RC.reconcile_env_comments(root, apply=True, sources=[src], leave_unsourced=True)
    assert fourth.selected == [] and fourth.written == [] and fourth.held == [] and [r.run_id for r in fourth.ambiguous] == [c] and fourth.leave_unsourced
    assert cli.main(["--root", root, "reconcile", "env-comments", "--apply", "--source", src, "--leave-unsourced"]) == 0
    assert _run_files(root, sub_a) == ["000001-run.jsonl", "000002-run_patch.jsonl"]
    assert emit.run_overrides_stamp(src)["keys"] == _live(root, a)[0]["run_overrides"]["keys"]
    _rebuild_equal(root, tmp_path)


def test_an_unclassified_stamp_is_refused_by_name_never_written_and_listed_by_the_verify(tmp_path, capsys):
    root, src, runs = _corpus(tmp_path, with_unclassified=True)
    f = runs["f"]
    app = RC.reconcile_env_comments(root, apply=True, sources=[src])
    odd = next(r for r in app.runs if r.run_id == f)
    assert odd.written is None and odd.unclassified and "not a reading of synthetic_pipeline.env" in odd.refused
    assert len(app.written) == 2 and _run_files(root, os.path.join(synth.CAMPAIGN, I.run_dir_name(f))) == ["000001-run.jsonl"]
    assert cli.main(["--root", root, "reconcile", "env-comments", "--apply", "--source", src]) == 2
    err = capsys.readouterr().err
    assert "held" in err and "unclassified" in err
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    # after the fold the selected runs left the class; the unclassified stamp still stands and the verify names it, exit 3
    assert _live(root, f)[0]["run_overrides"] == MISMATCH and _live(root, runs["a"][0])[0]["run_overrides"] == CLEAN
    assert cli.main(["--root", root, "reconcile", "env-comments", "--verify", "--source", src]) == 3
    out = capsys.readouterr().out
    assert f"  {f} unclassified: the stored key set is not the file's" in out and "polluted 1 run(s)" in out


def test_a_later_statement_that_is_neither_reading_of_the_file_is_refused_by_name_and_never_restated_to_the_files_value(tmp_path, capsys):
    """Run G registered with the polluted stamp; a later, FOLDED run_patch restated its stamp with OVERRIDE_EFFECTIVE_BATCH = 32 (every
    other key clean) and kept the file's digest as provenance -- the file reads 64. The digest and the key set match, so a pass that
    trusted them would write 64 over the clean, deliberate 32 as a durable correction. Instead: the dry run lists G as unclassified, naming
    the key, the stored value and both readings of the file; the apply writes nothing for it (exit 2); after the ingest the 32 stands; the
    verify lists G (exit 3) so a landing window fails by name rather than read a clean class; a rebuild agrees. The polluted runs beside
    it are corrected as before."""
    root, src, runs = _corpus(tmp_path, with_divergent=True)
    a, _ = runs["a"]
    g, sub_g = runs["g"]
    assert _live(root, g)[0]["run_overrides"] == DIVERGENT                                # the later statement folded, as the producer stated it
    rep = RC.reconcile_env_comments(root, apply=False, sources=[src])
    assert [r.run_id for r in rep.unclassified] == [g] and g not in [r.run_id for r in rep.selected] and g not in [r.run_id for r in rep.ambiguous]
    odd = rep.unclassified[0]
    assert odd.replaced == {} and odd.restated == DIVERGENT and odd.source.path == src
    assert odd.unclassified.startswith("OVERRIDE_EFFECTIVE_BATCH is '32', neither the file's reading ('64') nor what the old reader made of its line")
    assert "a later statement, not the reader's artifact" in odd.unclassified and odd.unclassified.endswith("refused by name, nothing restated")
    assert sorted(r.run_id for r in rep.selected) == sorted([a, runs["d"][0]]) and (rep.stamped, rep.sourced) == (5, 4)
    assert rep.as_dict()["unclassified"] == 1 and next(c for c in rep.as_dict()["corrections"] if c["run_id"] == g)["replaced"] == {}
    assert cli.main(["--root", root, "reconcile", "env-comments", "--source", src]) == 0
    out = capsys.readouterr().out
    assert f"unclassified (a file given, but the stamp is no reading of it; refused by name, never corrected): 1 run(s)\n  {g} OVERRIDE_EFFECTIVE_BATCH is '32'" in out
    # the apply: G refused by name and NOT written (its directory keeps the registration and the later statement alone); A and D written; exit 2
    app = RC.reconcile_env_comments(root, apply=True, sources=[src])
    by = {r.run_id: r for r in app.runs}
    assert by[g].written is None and by[g].refused == by[g].unclassified and len(app.written) == 2
    assert _run_files(root, sub_g) == ["000001-run.jsonl", "000002-run_patch.jsonl"]
    assert cli.main(["--root", root, "reconcile", "env-comments", "--apply", "--source", src]) == 2
    assert "unclassified" in capsys.readouterr().err
    # the fold: G's 32 stands (never overwritten with the file's 64); A is the file's reading
    rep2 = H.ingest_once(root, workers=0, min_free=0)
    assert rep2.refused == 0 and rep2.blocked == 0, rep2.refusals
    assert _live(root, g)[0]["run_overrides"] == DIVERGENT and _live(root, g)[0]["run_overrides"]["keys"]["OVERRIDE_EFFECTIVE_BATCH"] == "32"
    assert RC.ENV_COMMENTS_RECONCILED_KEY not in _live(root, g)[0] and _live(root, a)[0]["run_overrides"] == CLEAN
    # the verify lists G by name (exit 3): the row is in the folded table and the pass refuses it, so the window fails by name
    ver = RC.verify_env_comments(root, sources=[src])
    assert [r.run_id for r in ver.listed] == [g] and ver.listed[0].unclassified and ver.selected == []
    assert cli.main(["--root", root, "reconcile", "env-comments", "--verify", "--source", src]) == 3
    out = capsys.readouterr().out
    assert f"  {g} unclassified: OVERRIDE_EFFECTIVE_BATCH is '32', neither the file's reading ('64')" in out and "polluted 1 run(s)" in out
    # idempotent on the rest: a second apply selects nothing, writes nothing, and still refuses G (exit 2)
    third = RC.reconcile_env_comments(root, apply=True, sources=[src])
    assert third.selected == [] and third.written == [] and [r.run_id for r in third.unclassified] == [g]
    assert cli.main(["--root", root, "reconcile", "env-comments", "--apply", "--source", src]) == 2
    _rebuild_equal(root, tmp_path)


def test_the_writer_adds_no_central_stamp_even_with_an_overrides_file_in_the_environment(tmp_path, monkeypatch):
    """The rule run-patch-extra pinned: a writer that stamped the operator's own `$SISTEER_RUN_OVERRIDES` would restate the very value this
    record corrects with the operator's shell -- so the correction record carries the restated stamp and nothing of the writer's."""
    root, src, runs = _corpus(tmp_path)
    a, _ = runs["a"]
    f = tmp_path / "operator_shell.env"
    f.write_text(OVERRIDES_TEXT)
    monkeypatch.setenv(emit.RUN_OVERRIDES_ENV, str(f))
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    assert emit.run_overrides_stamp()["id"] == "operator_shell_v9"          # an ordinary producer writer WOULD stamp this
    app = RC.reconcile_env_comments(root, apply=True, sources=[src])
    rec = _read_one(next(r for r in app.selected if r.run_id == a).written)
    assert rec["payload"]["fields"]["extra_json"]["run_overrides"] == CLEAN and "extra" not in rec["payload"]
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    assert _live(root, a)[0]["run_overrides"] == CLEAN


def test_a_run_is_held_behind_an_unfolded_record_stating_its_stamp_wherever_in_the_log_it_lies(tmp_path, capsys):
    """A full `run` record carrying the stamp under its extra (a re-registration) and a `run_patch` carrying it under fields.extra_json (a
    writer on the fixed reader), each in ANOTHER directory and not yet folded, hold the run: the dry run names the record, the apply writes
    nothing for it (exit 2); a legacy patch whose stamp rides a top-level extra never folds the key and holds nothing; after the ingest
    the predicate decides again from the folded row."""
    root, src, runs = _corpus(tmp_path)
    a, _ = runs["a"]
    d, _ = runs["d"]
    # a full run record of A in a foreign directory, the clean stamp under its extra: unfolded -> held
    payload = synth.run_payload("ovA", i=200, extra={"run_overrides": CLEAN})
    assert payload["run_id"] == a
    _write(root, "elsewhere", "000001-run.jsonl", [synth.rec("run", payload, synth.t(200))])
    # a legacy patch of D with the stamp under a TOP-LEVEL extra (run-patch-extra's business): never folds the key, holds nothing
    _write(root, "legacy", "000001-run_patch.jsonl", [synth.rec("run_patch", {"run_id": d, "fields": {"job_name": "legacy"}, "extra": {"run_overrides": CLEAN}}, synth.t(201))])
    rep = RC.reconcile_env_comments(root, apply=False, sources=[src])
    held = {r.run_id: r for r in rep.selected if r.pending}
    assert set(held) == {a} and held[a].pending_type == "run" and not held[a].pending_own
    assert cli.main(["--root", root, "reconcile", "env-comments", "--source", src]) == 0
    out = capsys.readouterr().out
    assert "pending (a record stating the run's run_overrides is in the log, not folded yet): 1 run(s)" in out and f"  {a} record " in out and "(a full run record)" in out
    app = RC.reconcile_env_comments(root, apply=True, sources=[src])
    by = {r.run_id: r for r in app.selected}
    assert by[a].written is None and "a full run record of this run stating its run_overrides is in the log and not folded yet" in by[a].refused
    assert by[d].written is not None
    assert cli.main(["--root", root, "reconcile", "env-comments", "--apply", "--source", src]) == 2     # D now held behind its own record, A behind the full record
    # the fold: the foreign full record's clean stamp settles A; D's correction folds; the predicate selects neither
    rep2 = H.ingest_once(root, workers=0, min_free=0)
    assert rep2.refused == 0 and rep2.blocked == 0, rep2.refusals
    assert _live(root, a)[0]["run_overrides"] == CLEAN and _live(root, d)[0]["run_overrides"] == CLEAN
    assert RC.reconcile_env_comments(root, sources=[src]).selected == [] and RC.verify_env_comments(root, sources=[src]).listed == []
    # a run_patch stating the stamp under fields.extra_json, unfolded, holds the same way
    e, sub_e = _run(root, "ovE", 300, POLLUTED)
    rep3 = H.ingest_once(root, workers=0, min_free=0)
    assert rep3.refused == 0, rep3.refusals
    _write(root, "elsewhere", "000002-run_patch.jsonl", [synth.rec("run_patch", {"run_id": e, "fields": {"extra_json": {"run_overrides": CLEAN}}}, synth.t(310))])
    rep4 = RC.reconcile_env_comments(root, apply=True, sources=[src])
    (run_e,) = rep4.selected
    assert run_e.run_id == e and run_e.written is None and run_e.pending_type == "run_patch" and "a record of this run stating its run_overrides" in run_e.refused


def test_the_runs_own_directory_is_read_again_at_the_write(tmp_path, monkeypatch):
    """The second guard (the multi-turn-scope precedent): a statement that lands in the run's own directory AFTER the whole-log walk and
    before the write (here: the walk stubbed to have seen nothing) holds the run at the write -- nothing is written, the hold names the
    record; once it folds, the predicate decides again from the folded row (the late statement was the clean stamp: deselected)."""
    root, src, runs = _corpus(tmp_path)
    a, sub_a = runs["a"]
    d, _ = runs["d"]
    late = synth.rec("run_patch", {"run_id": a, "fields": {"extra_json": {"run_overrides": CLEAN}}}, synth.t(400))
    _write(root, sub_a, "000002-run_patch.jsonl", [late])
    monkeypatch.setattr(RC, "overrides_statements", lambda records_root, run_ids=None: {})
    app = RC.reconcile_env_comments(root, apply=True, sources=[src])
    by = {r.run_id: r for r in app.selected}
    assert by[a].written is None and by[a].pending == R.record_sha(late) and not by[a].pending_own and by[a].pending_type == "run_patch"
    assert by[a].refused == f"a record of this run stating its run_overrides is in the log and not folded yet ({R.record_sha(late)[:12]}); ingest first, then re-run"
    assert by[d].written is not None and _run_files(root, sub_a) == ["000001-run.jsonl", "000002-run_patch.jsonl"]
    monkeypatch.undo()
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    assert _live(root, a)[0]["run_overrides"] == CLEAN and RC.reconcile_env_comments(root, apply=True, sources=[src]).selected == []


# --------------------------------------------------------------------------------------------
# an unsourced stamp is not a clean class: the apply and the verify fail by name until the file is given or the stamp is left by name
# --------------------------------------------------------------------------------------------


def test_an_unsourced_stamp_whose_values_may_carry_a_comment_fails_the_apply_and_the_verify_by_name_until_its_file_is_given_or_it_is_left_by_name(tmp_path, capsys):
    """The sources unset, or an existing directory holding none of the files the stamps were read from, makes EVERY polluted stamp
    ambiguous. A pass that counted only what it could decide would write nothing, exit 0 and verify 'consistent' over a table it had not
    touched -- a landing window completing with every stored value as it was. Instead, with no file: the apply writes nothing and refuses
    every such row by name (exit 2), the verify lists them all (exit 3), and the coverage check (`--check-sources`, what the window runs
    before anything moves) lists them by stamp (exit 3), nothing under the root touched. With the file: the check still lists the one
    stamp nobody has, and the apply and the verify fail on it alone until `--leave-unsourced` leaves it by name -- the two sourced rows
    corrected, the verify reading zero with the clause, a rebuild agreeing. The flags belong to env-comments: refused by name on another
    action."""
    root, src, runs = _corpus(tmp_path, with_unsourced=True)
    a, sub_a = runs["a"]
    d, _sub_d = runs["d"]
    b, c, c2 = runs["b"], runs["c"], runs["c2"]
    empty = tmp_path / "empty_overrides"
    empty.mkdir()
    before = _tree(root)
    may_carry = sorted([a, b, c, d])          # every stamp whose stored text MAY carry a comment: A and D (polluted), B (`note #42`), C; not C2
    groups = {("lost_file_v2", "99999999"): (1, ("OVERRIDE_NOTE", "OVERRIDE_X")),
              ("synthetic_pipeline_v1", SOURCE_SHA[:8]): (3, ("OVERRIDE_EFFECTIVE_BATCH", "OVERRIDE_HPO_WARMUP_FRAC", "OVERRIDE_LORA_RANK", "OVERRIDE_NOTE"))}
    for sources in ([], [str(empty)]):
        argv = [x for s in sources for x in ("--source", s)]
        blind = RC.reconcile_env_comments(root, apply=True, sources=sources)
        assert blind.selected == [] and blind.written == [] and sorted(r.run_id for r in blind.ambiguous) == may_carry, sources
        assert sorted(r.run_id for r in blind.held) == may_carry and all(r.refused == r.unsourced_reason and r.written is None for r in blind.held)
        assert (blind.stamped, blind.sourced) == (5, 0) and blind.ambiguous_by_group() == groups and _tree(root) == before
        ver = RC.verify_env_comments(root, sources=sources)
        assert sorted(r.run_id for r in ver.listed) == may_carry and ver.selected == [] and ver.unclassified == [] and len(ver.unsourced_failing) == 4
        chk = RC.check_env_comments_sources(root, sources=sources)
        assert chk.check_sources and not chk.verify and sorted(r.run_id for r in chk.listed) == may_carry and chk.runs == chk.ambiguous
        assert (chk.stamped, chk.sourced) == (5, 0) and chk.ambiguous_by_group() == groups and chk.selected == [] and chk.by_group() == {}
        # the CLI: the apply exits 2 naming the way out, the verify 3 with every unsourced row on its own line, the check 3 with the groups;
        # nothing written under the root by any of them
        assert cli.main(["--root", root, "reconcile", "env-comments", "--apply", *argv]) == 2
        err = capsys.readouterr().err
        assert "4 row(s) held" in err and "unsourced: name the file with --source, or leave the stamp as stored by name with --leave-unsourced" in err
        assert cli.main(["--root", root, "reconcile", "env-comments", "--verify", *argv]) == 3
        out = capsys.readouterr().out
        assert out.rstrip().endswith("polluted 4 run(s) (4 unsourced)")
        assert f"  {c} unsourced: no file given for this stamp (lost_file_v2 99999999); values that may carry a comment: OVERRIDE_NOTE, OVERRIDE_X" in out
        assert f"  {b} unsourced: no file given for this stamp (synthetic_pipeline_v1 {SOURCE_SHA[:8]}); values that may carry a comment: OVERRIDE_NOTE" in out
        assert cli.main(["--root", root, "reconcile", "env-comments", "--check-sources", *argv]) == 3
        out = capsys.readouterr().out
        assert out.startswith("env-comments sources (read-only, the folded run table): every stamped run row whose values may carry a comment has its overrides file among the sources\n")
        assert "stamped run rows 5; with a source 0" in out
        assert (f"uncovered (no file given for this stamp; values that may carry a comment: OVERRIDE_EFFECTIVE_BATCH, OVERRIDE_HPO_WARMUP_FRAC, OVERRIDE_LORA_RANK, "
                f"OVERRIDE_NOTE): synthetic_pipeline_v1 {SOURCE_SHA[:8]} 3 run(s)") in out
        assert "uncovered (no file given for this stamp; values that may carry a comment: OVERRIDE_NOTE, OVERRIDE_X): lost_file_v2 99999999 1 run(s)" in out
        assert f"  {a} unsourced: no file given for this stamp (synthetic_pipeline_v1 {SOURCE_SHA[:8]})" in out
        assert out.rstrip().endswith("uncovered 4 run(s); name their files with --source, or leave them as stored by name with --leave-unsourced")
        assert _tree(root) == before and not os.path.exists(os.path.join(root, "state", "last_run"))
    # with the file the check still lists the one stamp nobody has (exit 3); left by name it completes (exit 0) with the clause; --json carries it
    chk = RC.check_env_comments_sources(root, sources=[src])
    assert [r.run_id for r in chk.listed] == [c] and (chk.stamped, chk.sourced) == (5, 3) and chk.ambiguous_by_group() == {("lost_file_v2", "99999999"): (1, ("OVERRIDE_NOTE", "OVERRIDE_X"))}
    assert cli.main(["--root", root, "reconcile", "env-comments", "--check-sources", "--source", src]) == 3
    out = capsys.readouterr().out
    assert f"  {c} unsourced:" in out and "stamped run rows 5; with a source 3" in out and out.rstrip().endswith("uncovered 1 run(s); name their files with --source, or leave them as stored by name with --leave-unsourced")
    left = RC.check_env_comments_sources(root, sources=[src], leave_unsourced=True)
    assert left.listed == [] and left.unsourced_failing == [] and [r.run_id for r in left.ambiguous] == [c] and left.leave_unsourced
    assert cli.main(["--root", root, "reconcile", "env-comments", "--check-sources", "--source", src, "--leave-unsourced"]) == 0
    out = capsys.readouterr().out
    assert out.rstrip().endswith("uncovered 0 run(s); every stamp whose values may carry a comment has its file; 1 unsourced run(s) left as stored by name (--leave-unsourced)")
    assert "uncovered (no file given for this stamp; values that may carry a comment: OVERRIDE_NOTE, OVERRIDE_X): lost_file_v2 99999999 1 run(s)" in out
    assert cli.main(["--root", root, "reconcile", "env-comments", "--check-sources", "--source", src, "--json"]) == 3
    d_ = json.loads(capsys.readouterr().out)
    assert d_["check_sources"] is True and d_["verify"] is False and d_["leave_unsourced"] is False and d_["listed"] == 1 and d_["unsourced_failing"] == 1
    assert d_["ambiguous_runs"] == 1 and d_["runs"] == 0 and d_["sourced_runs"] == 3 and d_["stamped_runs"] == 5 and [x["run_id"] for x in d_["corrections"]] == [c]
    # the flags belong to env-comments
    for flag in ("--check-sources", "--leave-unsourced"):
        assert cli.main(["--root", root, "reconcile", "fit-effective-batch", flag]) == 1
        assert "refused: --check-sources and --leave-unsourced belong to env-comments" in capsys.readouterr().err
        assert cli.main(["--root", root, "reconcile", "run-patch-extra", flag]) == 1
        capsys.readouterr()
    # the dry run with the operator's word says so in its ambiguous line; the apply with it completes (exit 0): the two sourced rows
    # written, C listed and left, never written, not held
    rep = RC.reconcile_env_comments(root, apply=False, sources=[src], leave_unsourced=True)
    assert rep.leave_unsourced and [r.run_id for r in rep.ambiguous] == [c] and rep.unsourced_failing == [] and sorted(r.run_id for r in rep.selected) == sorted([a, d])
    assert rep.as_dict()["leave_unsourced"] is True and rep.as_dict()["unsourced_failing"] == 0 and rep.as_dict()["listed"] == 2
    assert cli.main(["--root", root, "reconcile", "env-comments", "--source", src, "--leave-unsourced"]) == 0
    out = capsys.readouterr().out
    assert "ambiguous (no file given for the stamp and values that may carry a comment; listed, never corrected; left as stored by name, --leave-unsourced): 1 run(s)" in out
    app = RC.reconcile_env_comments(root, apply=True, sources=[src], leave_unsourced=True)
    assert len(app.written) == 2 and app.held == [] and [r.run_id for r in app.ambiguous] == [c] and next(r for r in app.runs if r.run_id == c).written is None
    assert _run_files(root, sub_a) == ["000001-run.jsonl", "000002-run_patch.jsonl"] and _run_files(root, os.path.join(synth.CAMPAIGN, I.run_dir_name(c))) == ["000001-run.jsonl"]
    rep2 = H.ingest_once(root, workers=0, min_free=0)
    assert rep2.refused == 0 and rep2.blocked == 0, rep2.refusals
    assert _live(root, a)[0]["run_overrides"] == CLEAN and _live(root, d)[0]["run_overrides"] == CLEAN
    assert _live(root, c)[0]["run_overrides"] == UNSOURCED_POLLUTED and RC.ENV_COMMENTS_RECONCILED_KEY not in _live(root, c)[0] and _live(root, c2)[0]["run_overrides"] == UNSOURCED_CLEAN
    # the verify: C fails it (exit 3) until left by name (exit 0, the clause); the check likewise; a second apply with the word writes nothing
    assert cli.main(["--root", root, "reconcile", "env-comments", "--verify", "--source", src]) == 3
    assert capsys.readouterr().out.rstrip().endswith("polluted 1 run(s) (1 unsourced)")
    assert cli.main(["--root", root, "reconcile", "env-comments", "--verify", "--source", src, "--leave-unsourced"]) == 0
    assert capsys.readouterr().out.rstrip().endswith("polluted 0 run(s); the class is consistent; 1 unsourced run(s) left as stored by name (--leave-unsourced)")
    assert cli.main(["--root", root, "reconcile", "env-comments", "--check-sources", "--source", src]) == 3
    capsys.readouterr()
    again = RC.reconcile_env_comments(root, apply=True, sources=[src], leave_unsourced=True)
    assert again.selected == [] and again.written == [] and again.held == [] and [r.run_id for r in again.ambiguous] == [c]
    _rebuild_equal(root, tmp_path)
