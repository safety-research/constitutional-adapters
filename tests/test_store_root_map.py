"""The STORE-ROOT MAP: a producer records the COMPUTE NODE's absolute store path
(`run.store_uri`); the grading host holds a mirror of each cluster's stores under its own root. `pending.store_path_of(row, map)`
relocates the recorded path for the row's substrate (longest matching remote root; unchanged when no root matches) and the worklist
carries the mapped `store_path` on every row -- the recorded URI stays the provenance and the live database is never rewritten. Two
clusters whose producer roots differ from their mirrors, and a third `slurm` cluster, are mapped side by side; graded bytes come from the
mirror."""
from __future__ import annotations

import os

import pytest

from sisteer.registry import db as DB
from sisteer.registry import emit, synth
from sisteer.registry import harvest as H
from sisteer.registry import pending as P
from sisteer.registry import records as R
from sisteer.registry import worklist as W

MAP = ["cluster_a=/cluster_a/user/exp=/mirror/cluster_a", "cluster_b=/cluster_b/user/exp=/mirror/cluster_b",
       "slurm=/scratch/user/exp=/mirror/cluster_c", "cluster_a=/cluster_a/user/exp/deep=/mirror/deep"]


def test_map_store_path_relocates_per_substrate_longest_root_wins_and_leaves_unmatched_paths():
    m = P.parse_store_root_map(MAP)
    assert P.map_store_path("/cluster_a/user/exp/sdfbase/x/shard00", "cluster_a", m) == "/mirror/cluster_a/sdfbase/x/shard00"
    assert P.map_store_path("/cluster_a/user/exp/deep/x", "cluster_a", m) == "/mirror/deep/x"                  # longest remote root
    assert P.map_store_path("/cluster_b/user/exp/y", "cluster_b", m) == "/mirror/cluster_b/y"
    assert P.map_store_path("/scratch/user/exp/z", "slurm", m) == "/mirror/cluster_c/z"
    assert P.map_store_path("/cluster_a/user/exp/x", "cluster_b", m) == "/cluster_a/user/exp/x"                 # another substrate's root: untouched
    assert P.map_store_path("/elsewhere/x", "cluster_a", m) == "/elsewhere/x"
    assert P.map_store_path("/cluster_a/user/exp/x", None, m) == "/mirror/cluster_a/x"                         # no substrate: every root is a candidate
    assert P.map_store_path("/cluster_a/user/expansion/x", "cluster_a", m) == "/cluster_a/user/expansion/x"     # a prefix match is on a path boundary
    assert P.map_store_path("", "cluster_a", m) == "" and P.map_store_path("/a/b", "cluster_a", None) == "/a/b"
    assert P.store_path_of({"store_uri": "file:///cluster_b/user/exp/y", "substrate": "cluster_b"}, m) == "/mirror/cluster_b/y"
    assert P.store_path_of({"store_uri": "file:///cluster_b/user/exp/y"}) == "/cluster_b/user/exp/y"           # no map: the recorded path
    with pytest.raises(ValueError):
        P.parse_store_root_map(["cluster_a=/only-two-parts"])


def _root(tmp_path, monkeypatch):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    rids = {}
    for k, (key, sub, uri) in enumerate((("k_a", "cluster_a", "/cluster_a/user/exp/stores/k_a"), ("k_b", "cluster_b", "file:///cluster_b/user/exp/stores/k_b"))):
        rid, recs = synth.run_records(key, 100 + 20 * k, through="generated", substrate=sub)
        recs[0]["payload"]["store_uri"] = uri
        R.write_records(os.path.join(root, "records", "c", f"00000{k + 1}-run.jsonl"), recs)
        rids[key] = rid
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.as_dict()
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.setenv("REGISTRY_ROOT", root)
    return root, rids


def test_worklist_rows_carry_the_mirror_path_and_grading_reads_the_mirrored_bytes(tmp_path, monkeypatch):
    root, rids = _root(tmp_path, monkeypatch)
    # the grading host's mirrors of the two clusters' stores (producer paths differ from the mirror roots on both)
    for key, mirror in (("k_a", tmp_path / "mirror" / "cluster_a" / "stores" / "k_a"), ("k_b", tmp_path / "mirror" / "cluster_b" / "stores" / "k_b")):
        mirror.mkdir(parents=True)
        (mirror / "rows.jsonl").write_text(f'{{"item_id": "q1", "score": 1.0, "origin": "{key}"}}\n')
    m = P.parse_store_root_map([f"cluster_a=/cluster_a/user/exp={tmp_path / 'mirror' / 'cluster_a'}",
                                f"cluster_b=/cluster_b/user/exp={tmp_path / 'mirror' / 'cluster_b'}"])
    wl = W.RegistryWorklist(root, [synth.JUDGE_ID], "s", driver="t", store_root_map=m)
    rows = {r["run_id"]: r for r in wl.pending()}
    assert rows[rids["k_a"]]["substrate"] == "cluster_a" and rows[rids["k_a"]]["store_path"] == str(tmp_path / "mirror" / "cluster_a" / "stores" / "k_a")
    assert rows[rids["k_b"]]["substrate"] == "cluster_b" and rows[rids["k_b"]]["store_path"] == str(tmp_path / "mirror" / "cluster_b" / "stores" / "k_b")
    assert wl.store_path(rows[rids["k_b"]]) == rows[rids["k_b"]]["store_path"]
    # the recorded URI is untouched in the row and in the database (provenance)
    assert rows[rids["k_a"]]["store_uri"] == "/cluster_a/user/exp/stores/k_a"
    read = {}

    def grade(row, scope):
        with open(os.path.join(row["store_path"], "rows.jsonl"), "r", encoding="utf-8") as fh:
            import json
            r = json.loads(fh.readline())
        read[row["run_id"]] = r["origin"]
        scope.items([{"item_id": r["item_id"], "metric_id": "asr", "value": r["score"]}])
        scope.metrics([{"metric_id": "asr", "threshold": 0.5, "value": r["score"], "k": 1, "n": 1}])

    wl.grade(sorted(rows.values(), key=lambda r: r["run_id"]), grade)
    assert read == {rids["k_a"]: "k_a", rids["k_b"]: "k_b"}
    H.ingest_once(root, workers=0, min_free=0, derive_facts=False)   # the harvester's fold of the spooled completion records (finish() folds nothing by default, 2026-09-30)
    with DB.reading(DB.db_path(root)) as conn:
        assert conn.execute("SELECT store_uri FROM run WHERE run_id = ?", (rids["k_a"],)).fetchone()[0] == "/cluster_a/user/exp/stores/k_a"
        assert conn.execute("SELECT store_uri FROM run WHERE run_id = ?", (rids["k_b"],)).fetchone()[0] == "file:///cluster_b/user/exp/stores/k_b"
        assert {r[0] for r in conn.execute("SELECT lifecycle_state FROM run").fetchall()} == {"judged"}
    # without a map the rows carry the recorded path (one item of 200 planned was judged, so the runs stay pending on coverage), so a
    # driver that compares that path to its local roots refuses by name
    wl0 = W.RegistryWorklist(root, [synth.JUDGE_ID], "s2", driver="t")
    rows0 = {r["run_id"]: r for r in wl0.pending()}
    assert rows0[rids["k_a"]]["store_path"] == "/cluster_a/user/exp/stores/k_a" and rows0[rids["k_b"]]["store_path"] == "/cluster_b/user/exp/stores/k_b"
    assert P.store_path_of({"store_uri": "/cluster_a/user/exp/stores/k_a", "substrate": "cluster_a"}) == "/cluster_a/user/exp/stores/k_a"
    with pytest.raises(W.RegistryUnavailable):
        W.refuse_unlocated("t", [{"run_id": "r", "store_uri": "/cluster_a/user/exp/stores/k_a"}], {})
