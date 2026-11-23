"""`scripts/registry/bcbw_dry_read.py`, the dry read behind every two-tree diff of a bcbw reader, on the SYNTHETIC full-suite mirror of `_bcbw_full_fixtures`
(the gemma4e4b `s1` / `s1p` rows, which file contrasts, and a three-seed-family `qwen35_4b_s2` store, a group of one `none` that files none):

* WITHOUT `--db` the context carries no registry connection, as before: a group that files a contrast refuses at the anchor currency by name
  (`AnchorCurrencyUnavailable`), never reads a frame from anywhere else;
* WITH `--db` the read gets a READ-ONLY connection (`db.connect(path, readonly=True)`: a write through it raises `sqlite3.OperationalError`), a contrast-filing
  group reads through the pass's anchor currency over that database, and every group's read starts with `ctx.begin_group(<group>)`, so each group's
  `<group>.anchor_decisions.json` holds its own decisions alone (the one-`none` group's is empty, never the previous group's);
* the output directory under the same tree, root and database is byte-identical across two runs (what the two-tree diff relies on).

The adapter the script constructs is the fixture's: `adapters.load_all` is patched to register a subclass of the full reader bound to the fixture's harness roster /
pins and to the seeds copy that declares the fixture's 5-id item set (the script's own construction, `cls(root=..., seeds=...)`, then reads the fixture)."""
from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
from pathlib import Path

import pytest

import _b1_fixtures as F
import _bcbw_full_fixtures as Z
import _bcbw_seedrep_fixtures as X
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry.adapters import anchor_current as AC
from sisteer.registry.adapters import bigcodebench_full as BCF
from sisteer.registry.adapters import lineage_common as LC

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "registry" / "bcbw_dry_read.py"
spec = importlib.util.spec_from_file_location("bcbw_dry_read_under_test", SCRIPT)
DRY = importlib.util.module_from_spec(spec)
spec.loader.exec_module(DRY)


@pytest.fixture(scope="module")
def seeds():
    return LC.Seeds.locate()


def _mirror(tmp_path, seeds):
    return Z.mirror(tmp_path, seeds, eval_id="bigcodebench", root=str(tmp_path / "mirror"), qwen_store=True)


def _register_fixture_adapter(monkeypatch, fx, seeds):
    """``adapters.load_all`` returns ONE adapter, the full reader bound to the fixture's harness roster / pins and seeds, whatever the script passes."""
    fseeds = Z.seeds_for(seeds)

    class FixtureFull(BCF.BigcodebenchFullAdapter):
        def __init__(self, root=None, seeds=None, **_kw):
            super().__init__(root=root or fx["root"], seeds=fseeds, roster_path=fx["roster"], pins_path=fx["pins"], arditi_roster_path=fx["arditi_roster"],
                             arditi_pins_path=fx["arditi_pins"])

    from sisteer.registry import adapters as ADP
    monkeypatch.setattr(ADP, "load_all", lambda names=None: {BCF.NAME: FixtureFull})
    return FixtureFull


def _registry_root(tmp_path, seeds, ad) -> str:
    """A scratch registry holding the dimensions and the fixture's fitted objects, with ONE pass of the fixture's reader folded: its `none` runs are current,
    so a later read's contrasts are decided against them."""
    from sisteer.registry.adapters import dimensions as DIM
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    now = R.utc_now()
    recs = [r for r in LC.standin_dimension_records(seeds, F.producer(), now) if r["record_type"] == "paper_round"]
    recs += DIM.build_records(DIM.load_seeds(seeds.source_dir), F.producer(), now=now, verify_si=False).all_records()
    for o in (ad.resolve_object(Z.RELU, Z.SEL, pin_key=Z.STAGED, model_key=Z.MK)[0], ad.resolve_object(Z.COMPOSED, Z.SEL_P, pin_key=Z.STAGED_P, model_key=Z.MK)[0]):
        r = LC.standin_object_record(o, seeds, F.producer(), now, "gemma4e4b_base")
        if r:
            recs.append(r)
    R.write_records(os.path.join(root, "records", "standins", "000001-dims.jsonl"), recs)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    rep0, ing0, ing0b = F.run_and_ingest(root, ad, now=X.NOW)
    assert rep0.groups_refused == 0 and ing0.refused + ing0b.refused == 0, ([r.error for r in rep0.results], ing0.refusals, ing0b.refusals)
    return root


def _outputs(out: Path) -> dict:
    return {p.name: p.read_bytes() for p in sorted(out.iterdir())}


def test_without_db_the_read_has_no_connection_and_a_contrast_refuses_at_the_currency(tmp_path, seeds, monkeypatch):
    fx = _mirror(tmp_path, seeds)
    _register_fixture_adapter(monkeypatch, fx, seeds)
    with pytest.raises(AC.AnchorCurrencyUnavailable, match="no registry connection"):
        DRY.main(["--adapter", BCF.NAME, "--root", fx["root"], "--out", str(tmp_path / "out"), "--no-record-id"])


def test_with_db_the_connection_is_read_only_and_each_group_logs_its_own_anchor_decisions(tmp_path, seeds, monkeypatch, capsys):
    fx = _mirror(tmp_path, seeds)
    cls = _register_fixture_adapter(monkeypatch, fx, seeds)
    root = _registry_root(tmp_path, seeds, cls())
    db_file = DB.db_path(root)
    opened = []
    real_connect = DB.connect

    class Held:
        """The connection the script opens, kept open past its `close` so the test can try a write through it."""
        def __init__(self, conn):
            self.conn = conn

        def execute(self, *a, **kw):
            return self.conn.execute(*a, **kw)

        def close(self):
            pass

    def spy(path, readonly=False, **kw):
        h = Held(real_connect(path, readonly=readonly, **kw))
        opened.append((path, readonly, h))
        return h

    monkeypatch.setattr(DB, "connect", spy)
    out = tmp_path / "out"
    assert DRY.main(["--adapter", BCF.NAME, "--root", fx["root"], "--db", db_file, "--out", str(out), "--no-record-id"]) == 0
    capsys.readouterr()
    assert [(p, ro) for p, ro, _h in opened] == [(db_file, True)], "one connection, read-only"
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        opened[0][2].conn.execute("CREATE TABLE dry_read_probe (x INTEGER)")
    opened[0][2].conn.close()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["db"] == db_file and sorted(summary["groups"]) == [Z.TAG, Z.QWEN_TAG]
    gemma, qwen = summary["groups"][Z.TAG], summary["groups"][Z.QWEN_TAG]
    assert gemma["by_type"].get("run_contrast", 0) == 3 and gemma["malformed"] == [] and qwen["malformed"] == [], summary
    assert "run_contrast" not in qwen["by_type"] and qwen["runs"] == 1
    gemma_runs = {json.loads(ln)["payload"]["run_id"] for ln in (out / f"{Z.TAG}.records.jsonl").read_text().splitlines() if json.loads(ln)["record_type"] == "run"}
    decisions = json.loads((out / f"{Z.TAG}.anchor_decisions.json").read_text())
    assert decisions and gemma["n_anchor_decisions"] == len(decisions) and {d["comparison"]["run_id"] for d in decisions} <= gemma_runs, decisions
    assert json.loads((out / f"{Z.QWEN_TAG}.anchor_decisions.json").read_text()) == [] and qwen["n_anchor_decisions"] == 0, \
        "begin_group clears the log: the one-none group carries none of the gemma group's decisions"


def test_the_output_is_byte_identical_across_two_runs_on_one_tree_root_and_database(tmp_path, seeds, monkeypatch, capsys):
    fx = _mirror(tmp_path, seeds)
    cls = _register_fixture_adapter(monkeypatch, fx, seeds)
    db_file = DB.db_path(_registry_root(tmp_path, seeds, cls()))
    outs = []
    for k in (1, 2):
        out = tmp_path / f"out{k}"
        assert DRY.main(["--adapter", BCF.NAME, "--root", fx["root"], "--db", db_file, "--out", str(out), "--no-record-id"]) == 0
        outs.append(_outputs(out))
    capsys.readouterr()
    assert outs[0] == outs[1] and f"{Z.TAG}.anchor_decisions.json" in outs[0] and "source_states.json" in outs[0], sorted(outs[0])
