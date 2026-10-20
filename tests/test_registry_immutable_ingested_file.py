"""An ingested record file is immutable: other bytes at its uri are refused by name.

The probe (2026-10-06): cycle 1 ingested `X.jsonl` and the gzip pass renamed it `X.jsonl.gz`; a plain `X.jsonl`
with DIFFERENT bytes then arrived beside it, and cycle 2 folded it as a CHANGED FILE -- the same `record_ingest` row with its sha replaced
(`reranks` 1), its records re-folded, 0 refusals: a silent replacement of an ingested record. The rule now: a file the ledger has RANKED
(`record_ingest`) is immutable. A second physical file at its logical uri (the `.gz` stripped) with other bytes is refused by name -- one
`record_refusal` row keyed by the arriving file's logical sha, `record_type` NULL, its reason naming the uri, the ingested sha and the
arriving sha -- the ledger row, the delivery audit and every folded row untouched, both files left in place; with the SAME bytes it is a
no-op the listing never raises. The one legitimate change left: a file never ranked (wholly refused, unreadable, empty) is not ingested and
still folds when its bytes change before its first fold. The gzip pass never compresses a plain twin over the ingested `.gz` nor bytes the
fold did not deliver; a rebuild ranks the twin the `file_arrival` fact recorded and refuses the other, as live did.
"""
from __future__ import annotations

import json
import os

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import synth

NOW1, NOW2, NOW3, NOW4 = synth.t(9000), synth.t(9001), synth.t(9002), synth.t(9003)


def _root(tmp_path, name="reg"):
    root = str(tmp_path / name)
    H.ensure_layout(root)
    return root


def _ingest(root, now, **kw):
    return H.ingest_once(root, workers=0, min_free=0, now=now, gzip_min_age_s=0, **kw)


def _q(root, sql, *args):
    with DB.reading(DB.db_path(root)) as conn:
        return [tuple(r) for r in conn.execute(sql, args).fetchall()]


def _ledger(root, uri):
    rows = _q(root, "SELECT file_sha256, first_written_utc, arrival_utc, n_records, ingest_seq FROM record_ingest WHERE file_uri = ?", uri)
    return rows[0] if rows else None


def _two_cycle_twin(tmp_path):
    """Cycle 1: a dims file ingested and gz-renamed. Then a plain twin with OTHER bytes (one record fewer, one record changed) is written at
    the plain name. Returns (root, uri, plain path, the ingested sha, the twin's sha, the dims, the ledger row after cycle 1)."""
    root = _root(tmp_path)
    dims = synth.seed_dimensions()
    path = R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), dims)
    uri = H.logical_uri(root, path)
    rep1 = _ingest(root, NOW1)
    assert rep1.ok >= len(dims) and rep1.refused == 0 and rep1.gzipped >= 1           # the dims file (and the cycle's own fact file)
    assert not os.path.exists(path) and os.path.exists(path + ".gz")
    row1 = _ledger(root, uri)
    ingested_sha = R.file_logical_sha(path + ".gz")
    assert row1[0] == ingested_sha
    R.write_records(path, dims[:-1])                        # other bytes at the plain name: one record fewer, every record well-formed
    twin_sha = R.file_logical_sha(path)
    assert twin_sha != ingested_sha
    return root, uri, path, ingested_sha, twin_sha, dims, row1


def test_a_plain_twin_with_other_bytes_after_the_gz_was_ingested_is_refused_by_name_and_nothing_moves(tmp_path):
    root, uri, path, ingested_sha, twin_sha, dims, row1 = _two_cycle_twin(tmp_path)
    before = {t: _q(root, f"SELECT * FROM {t} ORDER BY 1") for t in ("model", "eval", "instrument", "object", "decode_policy")}
    before["row_history"] = _q(root, "SELECT * FROM row_history WHERE table_name NOT IN ('file_arrival', 'record_refusal_fact') ORDER BY 1, 2, 3")
    delivery1 = _q(root, "SELECT file_sha256, status, n_ok, n_records FROM record_delivery WHERE file_uri = ?", uri)
    rep2 = _ingest(root, NOW2)
    # the twin is listed (its sha is not the delivery ledger's), parsed once, and refused by name; nothing folds and nothing re-ranks
    assert rep2.ok == 0 and rep2.duplicates == 0 and rep2.refused == 1 and rep2.files_replaced_refused == 1 and rep2.reranks == 0
    assert rep2.files == 2 and rep2.refusal_facts == 1                  # the twin and the cycle's own refusal-fact file, nothing else
    assert len(rep2.refusals) == 1 and rep2.refusals[0].startswith(f"{uri}: replaced: ") and ingested_sha in rep2.refusals[0] and twin_sha in rep2.refusals[0]
    assert _ledger(root, uri) == row1                                                   # the ledger row stands whole
    assert _q(root, "SELECT file_sha256, status, n_ok, n_records FROM record_delivery WHERE file_uri = ?", uri) == delivery1   # the audit too
    for t, rows in before.items():
        if t == "row_history":
            assert _q(root, "SELECT * FROM row_history WHERE table_name NOT IN ('file_arrival', 'record_refusal_fact') ORDER BY 1, 2, 3") == rows
        else:
            assert _q(root, f"SELECT * FROM {t} ORDER BY 1") == rows, t
    ref = _q(root, "SELECT record_uri, record_type, field, reason FROM record_refusal WHERE record_sha256 = ?", twin_sha)
    assert ref and ref[0][:3] == (uri, None, None) and f"ingested under sha {ingested_sha}" in ref[0][3] and f"hashes to {twin_sha}" in ref[0][3] and "D968" in ref[0][3]
    assert os.path.exists(path) and os.path.exists(path + ".gz") and R.file_logical_sha(path) == twin_sha and R.file_logical_sha(path + ".gz") == ingested_sha
    # the judgment is a fact in the log like every refusal: one record_refusal fact for the twin's sha, folded by the cycle's end pass
    assert _q(root, "SELECT COUNT(*) FROM record_refusal_fact WHERE record_sha256 = ? AND record_uri = ?", twin_sha, uri) == [(1,)]
    # the sidecar line names it
    lines = [json.loads(l) for l in open(os.path.join(H.state_dir(root), "refused.jsonl"), encoding="utf-8") if l.strip()]
    assert [(l["uri"], l["sha"]) for l in lines] == [(uri, twin_sha)]
    # the next cycle does not raise it again: already judged under this code, not new work
    rep3 = _ingest(root, NOW3)
    assert rep3.files == 0 and rep3.refused == 0 and rep3.files_replaced_refused == 0 and _ledger(root, uri) == row1
    with DB.reading(DB.db_path(root)) as conn:
        assert H.new_or_changed_files(root, conn) == []
        assert H.refused_replacement(conn, root, path, twin_sha) is True
        assert H.refused_replacement(conn, root, path + ".gz", ingested_sha) is False       # the ingested bytes are never a replacement
        assert H.refused_replacement(conn, root, path, None) is False


def test_the_same_bytes_at_the_plain_name_are_a_no_op_the_listing_never_raises(tmp_path):
    root = _root(tmp_path)
    dims = synth.seed_dimensions()
    path = R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), dims)
    uri = H.logical_uri(root, path)
    _ingest(root, NOW1)
    row1 = _ledger(root, uri)
    R.write_records(path, dims)                             # the mirror delivered the plain file again after the harvester's rename
    assert R.file_logical_sha(path) == R.file_logical_sha(path + ".gz") == row1[0]
    rep2 = _ingest(root, NOW2)
    assert rep2.files == 0 and rep2.refused == 0 and rep2.files_replaced_refused == 0 and _ledger(root, uri) == row1
    assert _q(root, "SELECT COUNT(*) FROM record_refusal") == [(0,)]
    assert os.path.exists(path) and os.path.exists(path + ".gz")       # both left in place; the gzip pass does not touch a twin pair


def test_a_file_never_ranked_still_changes_before_its_first_fold(tmp_path):
    """The legitimate change that remains: a file wholly refused (torn, malformed) has no ledger row -- it was never ingested -- so the
    corrected bytes at the same name fold as new work; a changed file is refused only once its uri has been RANKED."""
    root = _root(tmp_path)
    dims = synth.seed_dimensions()
    path = os.path.join(root, "records", "dims", "000001-dims.jsonl")
    os.makedirs(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{not a record\n")
    rep1 = _ingest(root, NOW1)
    uri = H.logical_uri(root, path)
    assert rep1.files == 1 and rep1.refused == 1 and rep1.files_replaced_refused == 0 and _ledger(root, uri) is None
    assert _q(root, "SELECT status FROM record_delivery WHERE file_uri = ?", uri) == [("refused",)]
    os.remove(path)
    R.write_records(path, dims)                             # the producer re-wrote the torn file whole
    rep2 = _ingest(root, NOW2)
    assert rep2.refused == 0 and rep2.ok >= len(dims) and rep2.files_replaced_refused == 0 and rep2.files == 2     # the file and its arrival fact
    assert _ledger(root, uri) is not None and _ledger(root, uri)[0] == R.file_logical_sha(path + ".gz")
    # and from then on it is immutable: the next edit is refused
    R.write_records(path, dims[:3])
    rep3 = _ingest(root, NOW3)
    assert rep3.refused == 1 and rep3.files_replaced_refused == 1 and rep3.ok == 0


def test_the_gzip_pass_never_compresses_a_twin_over_the_ingested_gz_nor_bytes_the_fold_did_not_deliver(tmp_path):
    root, uri, path, ingested_sha, twin_sha, dims, row1 = _two_cycle_twin(tmp_path)
    # the twin beside the .gz, BEFORE the fold has judged it: both stay, the .gz keeps the ingested bytes
    conn = DB.connect(DB.db_path(root))
    try:
        assert H.gzip_after_ingest(root, conn, min_age_s=0) == 0
        assert R.file_logical_sha(path) == twin_sha and R.file_logical_sha(path + ".gz") == ingested_sha
        # the .gz gone (an operator moved it): a plain file whose bytes are not the delivered ones is left for the fold to judge, not compressed
        os.rename(path + ".gz", path + ".gz.aside")
        assert H.gzip_after_ingest(root, conn, min_age_s=0) == 0
    finally:
        conn.close()
    assert os.path.exists(path) and not os.path.exists(path + ".gz")
    os.rename(path + ".gz.aside", path + ".gz")
    # after the judgment the pair still stands as it was
    _ingest(root, NOW2)
    assert os.path.exists(path) and os.path.exists(path + ".gz") and R.file_logical_sha(path + ".gz") == ingested_sha


def test_a_rebuild_ranks_the_twin_the_arrival_fact_recorded_and_refuses_the_other_as_live_did(tmp_path):
    root, uri, path, ingested_sha, twin_sha, dims, row1 = _two_cycle_twin(tmp_path)
    _ingest(root, NOW2)
    _ingest(root, NOW3)                                     # the end-of-cycle fact pass folded; the log carries the refusal fact
    out = str(tmp_path / "rebuilt.sqlite")
    H.rebuild(root, out, workers=0)
    live, rebuilt = DB.connect(DB.db_path(root), readonly=True), DB.connect(out, readonly=True)
    try:
        diff = {k: v for k, v in H.verify_equal(live, rebuilt).items() if v}
        assert not diff, diff
        assert tuple(rebuilt.execute("SELECT file_sha256 FROM record_ingest WHERE file_uri = ?", (uri,)).fetchone()) == (ingested_sha,)
        assert tuple(rebuilt.execute("SELECT record_uri, record_type FROM record_refusal WHERE record_sha256 = ?", (twin_sha,)).fetchone()) == (uri, None)
    finally:
        live.close()
        rebuilt.close()
    # the ordering rule on its own: the recorded twin first, everything else in place; no fact (or a fact naming neither file's bytes)
    # -> the listing's order, the plain name before its .gz; a file that is nobody's twin is never hashed
    other = (os.path.join(root, "records", "b", "000001-y.jsonl"), None)
    listing = [(path, None), (path + ".gz", None), other]
    assert H.prefer_recorded_twins(root, listing, {uri: ingested_sha}) == [(path + ".gz", None), (path, None), other]
    assert H.prefer_recorded_twins(root, listing, {}) == listing
    assert H.prefer_recorded_twins(root, listing, {uri: twin_sha}) == listing
    assert H.prefer_recorded_twins(root, listing, {uri: "0" * 64}) == listing
    assert H.prefer_recorded_twins(root, [(path, ingested_sha), (path + ".gz", None)], {uri: ingested_sha}) == [(path, ingested_sha), (path + ".gz", None)]   # a listed sha is trusted
    # read_arrival_facts fills the recorded shas of the winning facts
    recorded = {}
    arrivals = H.read_arrival_facts(root, recorded)
    assert uri in arrivals and recorded[uri] == ingested_sha


def test_a_replacement_is_re_judged_once_when_the_judging_stamp_moves(tmp_path, monkeypatch):
    root, uri, path, ingested_sha, twin_sha, dims, row1 = _two_cycle_twin(tmp_path)
    _ingest(root, NOW2)
    stamp0 = _q(root, "SELECT judged_with FROM record_refusal WHERE record_sha256 = ?", twin_sha)[0][0]
    assert stamp0 == H.validator_stamp()
    monkeypatch.setattr(H, "validator_stamp", lambda: "f" * 16)
    rep = _ingest(root, NOW3)
    assert rep.refused == 1 and rep.files_replaced_refused == 1 and _ledger(root, uri) == row1
    assert _q(root, "SELECT judged_with FROM record_refusal WHERE record_sha256 = ?", twin_sha) == [("f" * 16,)]
    rep = _ingest(root, NOW4)
    assert rep.files == 0 and rep.refused == 0


def test_a_dimension_seed_revision_re_published_with_force_still_re_applies_the_appended_file(tmp_path):
    """The one changed-file case kept: `dimensions load --force` APPENDS to a seed revision's files by design (tests/test_registry_dimensions_atomic.py
    pins the append) and the cycle re-applies the directory whole in one transaction -- the immutability rule never looks at a file applied
    through `apply_dimension_bundle`."""
    root = _root(tmp_path)
    prod = synth.producer("tests.immutable_ingested_file")
    by = {}
    for r in synth.seed_dimensions(100):
        by.setdefault(str(r["record_type"]), []).append(synth.rec(str(r["record_type"]), dict(r["payload"]), str(r["written_utc"]), prod))
    d = os.path.join(root, "records", "dimensions", "c" * 16)
    files = [R.write_records(os.path.join(d, R.record_file_name(seq, rt)), recs) for seq, (rt, recs) in enumerate(by.items(), start=1)]
    rep1 = _ingest(root, NOW1)
    assert rep1.dimension_transactions == 1 and rep1.refused == 0
    model_file = next(f for f in files if f.endswith("-model.jsonl"))
    model_payload = dict(next(r for r in synth.seed_dimensions(100) if r["record_type"] == "model")["payload"], notes="forced re-publication")
    R.write_records(model_file, [synth.rec("model", model_payload, synth.t(500), prod)])          # appended: the file's bytes changed
    rep2 = _ingest(root, NOW2)
    assert rep2.dimension_transactions == 1 and rep2.files == 1 and rep2.ok == 1 and rep2.refused == 0 and rep2.files_replaced_refused == 0
    assert _q(root, "SELECT notes FROM model WHERE model_id = ?", synth.FIT_MODEL_ID) == [("forced re-publication",)]
    assert _q(root, "SELECT COUNT(*) FROM record_refusal") == [(0,)]
