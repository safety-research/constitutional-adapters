"""base_mmlu retains per-question correctness and publishes it
NAMESPACED, PROVENANCED and ATOMICALLY.

Pins: score_cell returns per-question rows (qidx, qid, subject, picked,
answer, correct) whose counts MATCH the aggregate; a two-cell fixture
yields a derivable paired table (join on qid, discordant counts consistent
with the two aggregates); qids are content-derived and identical across
cells. Publication: ``--run-label`` is required and rides in dir AND
filename so an ``r1`` (snapshot bank) and an ``r2`` (refit bank) run
collide on no path; the per-question rows land FIRST, then the aggregate,
each via tmp+rename, both carrying one shared run identity (label, object
sha, dataset revision, seed, ordered-qid hash); a torn or headerless rows
file refuses to pair; the summary exists only after the FULL requested set
scored.

Model-side tests run on the CPU tiny model with a synthetic 8-question
fixture — the directive's contract is retention + pairability, not MMLU
accuracy. The publication tests are pure filesystem.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts" / "sdfbase") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))

import base_mmlu as bm  # noqa: E402


def _rows():
    qs = []
    for i in range(8):
        subject = "logic" if i % 2 == 0 else "history"
        qs.append({
            "question": f"Synthetic question number {i} about {subject}?",
            "choices": [f"option {c}{i}" for c in "wxyz"],
            "answer": i % 4,
            "subject": subject,
        })
    return qs


def _dev():
    return {
        s: [{"question": f"{s} exemplar {j}",
             "choices": [f"ex {c}{j}" for c in "wxyz"],
             "answer": j % 4} for j in range(5)]
        for s in ("logic", "history")
    }


def _loud_object(model):
    import torch

    from sisteer.methods import VectorObject
    from sisteer.vectors import SteeringVector

    hidden = model.config.hidden_size
    g = torch.Generator().manual_seed(3)
    return VectorObject(SteeringVector(
        vectors={0: 30.0 * torch.randn(hidden, generator=g)},
        method="test", model_name="tiny", instruction_id="t"))


def _identity(label="r2", cell="none", **over) -> dict:
    ident = {"run_label": label, "model": "tiny", "model_revision": "r",
             "config": "cfg.yaml", "config_sha256": "c" * 64,
             "dataset": bm.DATASET, "dataset_split": bm.DATASET_SPLIT,
             "dataset_revision": "d" * 40, "n": 8, "requested_n": 8,
             "seed": 0, "shots": 2, "scale": 1.0,
             "ordered_qid_hash": bm.ordered_qid_hash(_rows()),
             "render": "plain_text_5shot", "chat_template_bypassed": True,
             "dtype": "bfloat16", "attn_implementation": "sdpa",
             "defenses_root": None, "cell": cell, "object_path": None,
             "object_sha256": None}
    ident.update(over)
    return ident


# --------------------------------------------------------------------------- #
# retention + pairability (tiny model)                                          #
# --------------------------------------------------------------------------- #


def test_per_question_rows_match_aggregate_and_pair(tiny_llama, tmp_path):
    model, tok = tiny_llama
    rows, dev = _rows(), _dev()
    agg_a, per_a = bm.score_cell(model, tok, None, rows, dev, shots=2)
    agg_b, per_b = bm.score_cell(model, tok, _loud_object(model), rows, dev,
                                 shots=2)

    for agg, per in ((agg_a, per_a), (agg_b, per_b)):
        assert len(per) == agg["n"] == len(rows)
        assert sum(r["correct"] for r in per) == agg["correct"]
        for r in per:
            assert r["picked"] in bm.LETTERS and r["answer"] in bm.LETTERS
            assert isinstance(r["correct"], bool)
            assert r["correct"] == (r["picked"] == r["answer"])
        assert [r["qidx"] for r in per] == list(range(len(rows)))

    # qids: content-derived, unique, identical across cells -> pairable
    ids_a = [r["qid"] for r in per_a]
    ids_b = [r["qid"] for r in per_b]
    assert ids_a == ids_b and len(set(ids_a)) == len(ids_a)

    # publish through the same writer main uses, re-read through the one
    # reader, assert pairability, derive the paired McNemar table
    run_dir = bm.run_dir_for(tmp_path, "r2")
    run_dir.mkdir()
    back = {}
    for cell, agg, per in (("none", agg_a, per_a),
                           ("loud__cell", agg_b, per_b)):
        agg_path, pq_path = bm.write_cell_results(
            run_dir, "r2", cell, agg, per, _identity(cell=cell))
        assert (agg_path, pq_path) == bm.cell_paths(run_dir, "r2", cell)
        back[cell] = bm.read_per_question(pq_path)
    bm.assert_pairable(back["none"][0], back["loud__cell"][0])
    back_a, back_b = back["none"][1], back["loud__cell"][1]
    b_by_id = {r["qid"]: r for r in back_b}
    table = {"both": 0, "a_only": 0, "b_only": 0, "neither": 0}
    for ra in back_a:
        rb = b_by_id[ra["qid"]]
        key = {(True, True): "both", (True, False): "a_only",
               (False, True): "b_only", (False, False): "neither"}[
            (ra["correct"], rb["correct"])]
        table[key] += 1
    assert sum(table.values()) == len(rows)
    # the paired table's margins reproduce both aggregates exactly
    assert table["both"] + table["a_only"] == agg_a["correct"]
    assert table["both"] + table["b_only"] == agg_b["correct"]


def test_deterministic_per_question_rows(tiny_llama):
    model, tok = tiny_llama
    rows, dev = _rows(), _dev()
    _, p1 = bm.score_cell(model, tok, None, rows, dev, shots=2)
    _, p2 = bm.score_cell(model, tok, None, rows, dev, shots=2)
    assert p1 == p2


# --------------------------------------------------------------------------- #
# namespaced, provenanced, atomic publication (pure filesystem)               #
# --------------------------------------------------------------------------- #


def test_run_label_is_a_plain_token():
    for ok in ("r1", "r2", "r2.cap24", "r1_snapshot-bank"):
        assert bm.validate_run_label(ok) == ok
    for bad in ("", "../r1", "r 1", "/abs", ".hidden"):
        with pytest.raises(SystemExit, match="run-label"):
            bm.validate_run_label(bad)


def test_snapshot_and_refit_runs_collide_on_no_path(tmp_path):
    cell = "constv2-mix-1m__psr_ce_all"
    paths = {}
    for label in ("r1", "r2"):
        run_dir = bm.run_dir_for(tmp_path, label)
        paths[label] = (*bm.cell_paths(run_dir, label, cell),
                        bm.summary_path(run_dir, label))
    assert set(paths["r1"]).isdisjoint(paths["r2"])
    # different DIRECTORIES and different FILENAMES (a flattening consumer
    # still cannot confuse them)
    for a, b in zip(paths["r1"], paths["r2"]):
        assert a.parent != b.parent and a.name != b.name
        assert "r1" in a.name and "r2" in b.name
    assert paths["r1"][0].parent == tmp_path / "r1"


def _per_q(n=8):
    return [{"qidx": i, "qid": bm.question_id(r), "subject": r["subject"],
             "picked": "A", "answer": bm.LETTERS[r["answer"]],
             "correct": r["answer"] == 0} for i, r in enumerate(_rows()[:n])]


def _agg(per):
    c = sum(r["correct"] for r in per)
    return {"acc": c / len(per), "correct": c, "n": len(per),
            "ci95": [0.0, 1.0]}


def test_rows_land_first_then_aggregate_each_atomically(tmp_path, monkeypatch):
    order = []
    real = bm._atomic_write

    def spy(path, text):
        order.append(Path(path).name)
        real(path, text)

    monkeypatch.setattr(bm, "_atomic_write", spy)
    run_dir = bm.run_dir_for(tmp_path, "r2")
    run_dir.mkdir()
    per = _per_q()
    ident = _identity(cell="none")
    agg_path, pq_path = bm.write_cell_results(run_dir, "r2", "none",
                                              _agg(per), per, ident)
    assert order == [pq_path.name, agg_path.name]  # rows FIRST
    assert not list(run_dir.glob("*.tmp"))          # rename left no temp
    record = json.loads(agg_path.read_text())
    assert record["identity"] == ident
    assert record["per_question_file"] == pq_path.name
    assert record["correct"] == _agg(per)["correct"]
    header_ident, rows = bm.read_per_question(pq_path)
    assert header_ident == ident and rows == per


def test_atomic_write_is_tmp_plus_rename(tmp_path):
    p = tmp_path / "x.json"
    bm._atomic_write(p, "payload")
    assert p.read_text() == "payload"
    assert not (tmp_path / "x.json.tmp").exists()


def test_disagreeing_aggregate_refuses_to_publish_either_file(tmp_path):
    run_dir = bm.run_dir_for(tmp_path, "r2")
    run_dir.mkdir()
    per = _per_q()
    bad = {**_agg(per), "correct": _agg(per)["correct"] + 1}
    with pytest.raises(SystemExit, match="disagree"):
        bm.write_cell_results(run_dir, "r2", "none", bad, per, _identity())
    assert not list(run_dir.iterdir())


def test_torn_or_headerless_rows_files_refuse_to_pair(tmp_path):
    run_dir = bm.run_dir_for(tmp_path, "r2")
    run_dir.mkdir()
    per = _per_q()
    _, pq_path = bm.write_cell_results(run_dir, "r2", "none", _agg(per), per,
                                       _identity())
    lines = pq_path.read_text().splitlines()
    torn = run_dir / "torn.jsonl"
    torn.write_text("\n".join(lines[:-2]) + "\n")
    with pytest.raises(SystemExit, match="torn write"):
        bm.read_per_question(torn)
    legacy = run_dir / "legacy.jsonl"  # the round-1 shape: rows only
    legacy.write_text("\n".join(lines[1:]) + "\n")
    with pytest.raises(SystemExit, match="no run-identity header"):
        bm.read_per_question(legacy)


def test_assert_pairable_requires_same_draw_of_same_data():
    a = _identity(label="r1")
    bm.assert_pairable(a, _identity(label="r2"))  # labels may differ
    for k, v in (("dataset_revision", "e" * 40),
                 ("ordered_qid_hash", "0" * 64), ("n", 7), ("shots", 5)):
        with pytest.raises(SystemExit, match=f"not pairable: {k}"):
            bm.assert_pairable(a, _identity(**{k: v}))


def test_ordered_qid_hash_is_order_sensitive_and_content_derived():
    rows = _rows()
    h = bm.ordered_qid_hash(rows)
    assert h == bm.ordered_qid_hash([dict(r) for r in rows])
    assert h != bm.ordered_qid_hash(list(reversed(rows)))
    changed = [dict(r) for r in rows]
    changed[0]["question"] += " (edited)"
    assert h != bm.ordered_qid_hash(changed)


def test_dataset_revision_read_off_the_loaded_split():
    sha = "f" * 40
    cached = types.SimpleNamespace(
        cache_files=[{"filename": f"/hf/datasets/cais___mmlu/all/0.0.0/{sha}/"
                                  "mmlu-test.arrow"}], info=None)
    assert bm.dataset_revision_of(cached) == sha
    checks = types.SimpleNamespace(
        cache_files=[], info=types.SimpleNamespace(
            download_checksums={f"hf://datasets/cais/mmlu@{sha}/all/test.parquet":
                                {"num_bytes": 1}}))
    assert bm.dataset_revision_of(checks) == sha
    assert bm.dataset_revision_of(types.SimpleNamespace(
        cache_files=[], info=None)) is None


class _FakeSplit:
    def __init__(self, rows, revision):
        self._rows = rows
        self.cache_files = ([{"filename": f"/c/cais___mmlu/all/0.0.0/"
                                          f"{revision}/mmlu-test.arrow"}]
                            if revision else [])
        self.info = None

    def shuffle(self, seed):
        return self

    def select(self, idx):
        return _FakeSplit([self._rows[i] for i in idx], None)

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)


def _install_fake_datasets(monkeypatch, revision):
    rows = _rows()
    dev = [{"question": f"{s} exemplar {j}", "choices": ["a", "b", "c", "d"],
            "answer": j % 4, "subject": s}
           for s in ("logic", "history") for j in range(5)]
    seen = {}

    def load_dataset(name, cfg, split, **kw):
        seen[split] = kw
        return _FakeSplit(rows if split == "test" else dev, revision)

    monkeypatch.setitem(sys.modules, "datasets",
                        types.SimpleNamespace(load_dataset=load_dataset))
    return seen


def test_load_rows_pins_or_refuses_the_dataset_revision(monkeypatch):
    sha = "1" * 40
    seen = _install_fake_datasets(monkeypatch, sha)
    rows, dev, resolved = bm.load_rows(8, 0)
    assert resolved == sha and len(rows) == 8 and set(dev) == {"logic",
                                                                "history"}
    assert seen["test"] == {}  # no pin passed -> none forwarded
    # an explicit pin is forwarded and must agree with what loaded
    _, _, resolved = bm.load_rows(8, 0, revision=sha)
    assert resolved == sha and seen["test"] == {"revision": sha}
    with pytest.raises(SystemExit, match="refusing to stamp"):
        bm.load_rows(8, 0, revision="2" * 40)
    # a load that cannot establish its revision refuses unless pinned
    _install_fake_datasets(monkeypatch, None)
    with pytest.raises(SystemExit, match="cannot establish"):
        bm.load_rows(8, 0)
    _, _, resolved = bm.load_rows(8, 0, revision="3" * 40)
    assert resolved == "3" * 40  # pinned: the pin IS the identity


def test_main_requires_run_label_and_out_dir(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("model: tiny\n")
    with pytest.raises(SystemExit):
        bm.main(["--config", str(cfg), "--cells", "none",
                 "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        bm.main(["--config", str(cfg), "--cells", "none", "--run-label", "r2"])


def _wire_main(monkeypatch, tmp_path, *, fail_cell=None):
    """main() with every GPU/network seam faked: load_rows, the model
    loader, resolve_object_path/load_object, score_cell (deterministic per
    cell; ``fail_cell`` raises mid-run)."""
    import sisteer.extraction.fitmon as fitmon
    import sisteer.methods as methods
    import sisteer.models as models

    rows, dev = _rows(), _dev()
    monkeypatch.setattr(bm, "load_rows",
                        lambda n, seed, revision=None: (rows, dev, "d" * 40))
    monkeypatch.setattr(fitmon, "pin_attention_backends", lambda: None)
    monkeypatch.setattr(models, "load_model_and_tokenizer",
                        lambda *a, **k: (types.SimpleNamespace(
                            eval=lambda: None), None))
    obj_path = tmp_path / "bank" / "cells_all" / "c__lora_r4_ce" / "object.pt"
    obj_path.parent.mkdir(parents=True)
    obj_path.write_bytes(b"object bytes")
    seen_resolve = []

    def fake_resolve(root, cell, *, parents_root=None, explicit_path=None):
        seen_resolve.append((root, cell, parents_root, explicit_path))
        return obj_path

    monkeypatch.setattr(bm, "resolve_object_path", fake_resolve)
    bm._seen_resolve = seen_resolve
    monkeypatch.setattr(methods, "load_object", lambda p, model, tok: "OBJ")

    def score_cell(model, tok, obj, rows_, dev_, shots, scale=1.0):
        cell_tag = "none" if obj is None else "fit"
        if fail_cell == cell_tag:
            raise RuntimeError("planted mid-run failure")
        per = [{"qidx": i, "qid": bm.question_id(r), "subject": r["subject"],
                "picked": "A" if obj is None else "B",
                "answer": bm.LETTERS[r["answer"]],
                "correct": (r["answer"] == 0) if obj is None
                else (r["answer"] == 1)} for i, r in enumerate(rows_)]
        return _agg(per), per

    monkeypatch.setattr(bm, "score_cell", score_cell)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("model: tiny\nmodel_revision: r\n")
    return cfg, obj_path


def test_main_publishes_pairs_with_shared_identity_then_summary(
        tmp_path, monkeypatch):
    from sisteer.eval.defense_cell import sha256_file

    cfg, obj_path = _wire_main(monkeypatch, tmp_path)
    out = tmp_path / "mmlu"
    rc = bm.main(["--config", str(cfg), "--cells", "c__lora_r4_ce,none",
                 "--defenses-root", str(tmp_path / "bank"),
                 "--out-dir", str(out), "--run-label", "r2", "--shots", "2"])
    assert rc == 0
    run_dir = out / "r2"
    agg_none = json.loads((run_dir / "mmlu_r2_none.json").read_text())
    agg_fit = json.loads((run_dir / "mmlu_r2_c__lora_r4_ce.json").read_text())
    id_none, rows_none = bm.read_per_question(
        run_dir / "mmlu_r2_none.per_question.jsonl")
    id_fit, rows_fit = bm.read_per_question(
        run_dir / "mmlu_r2_c__lora_r4_ce.per_question.jsonl")
    # one shared run identity, per-cell object binding
    shared = ("run_label", "dataset_revision", "ordered_qid_hash", "seed",
              "n", "shots", "config_sha256", "model_revision")
    for k in shared:
        assert id_none[k] == id_fit[k] == agg_none["identity"][k]
    assert id_none["object_sha256"] is None
    assert id_fit["object_sha256"] == sha256_file(obj_path)
    assert id_fit["ordered_qid_hash"] == bm.ordered_qid_hash(_rows())
    bm.assert_pairable(id_none, id_fit)
    # bare scored FIRST so the fitted aggregate carries the within-run delta
    assert agg_none["delta_vs_bare"] is None
    assert agg_fit["delta_vs_bare"] == pytest.approx(
        agg_fit["acc"] - agg_none["acc"])
    # the summary exists, names the full set, and points at every file
    summary = json.loads((run_dir / "mmlu_r2_summary.json").read_text())
    assert summary["requested_cells"] == ["none", "c__lora_r4_ce"]
    assert set(summary["cells"]) == {"none", "c__lora_r4_ce"}
    assert summary["run"]["dataset_revision"] == "d" * 40
    assert summary["cells"]["c__lora_r4_ce"]["per_question_file"] == \
        "mmlu_r2_c__lora_r4_ce.per_question.jsonl"
    assert not list(run_dir.glob("*.tmp"))
    # a second label on the same out-dir shares no path
    rc = bm.main(["--config", str(cfg), "--cells", "none",
                 "--out-dir", str(out), "--run-label", "r1", "--shots", "2"])
    assert rc == 0
    assert (out / "r1" / "mmlu_r1_none.json").exists()
    assert not (out / "r1" / "mmlu_r2_none.json").exists()


def test_main_partial_run_leaves_pairs_and_no_summary(tmp_path, monkeypatch):
    cfg, _ = _wire_main(monkeypatch, tmp_path, fail_cell="fit")
    out = tmp_path / "mmlu"
    with pytest.raises(RuntimeError, match="planted"):
        bm.main(["--config", str(cfg), "--cells", "none,c__lora_r4_ce",
                 "--defenses-root", str(tmp_path / "bank"),
                 "--out-dir", str(out), "--run-label", "r2", "--shots", "2"])
    run_dir = out / "r2"
    assert (run_dir / "mmlu_r2_none.json").exists()
    assert (run_dir / "mmlu_r2_none.per_question.jsonl").exists()
    assert not (run_dir / "mmlu_r2_summary.json").exists()
    assert not (run_dir / "mmlu_r2_c__lora_r4_ce.json").exists()


def test_main_refuses_duplicate_cells(tmp_path, monkeypatch):
    cfg, _ = _wire_main(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="duplicate"):
        bm.main(["--config", str(cfg), "--cells", "none,none",
                 "--out-dir", str(tmp_path / "o"), "--run-label", "r2"])


def test_main_explicit_objects_and_parents_root(tmp_path, monkeypatch):
    """The derived app's path: a cell named by --object CELL=PATH needs no
    --defenses-root, the resolver gets the explicit path + --parents-root,
    and the run identity records both; malformed --object entries refuse."""
    cfg, obj_path = _wire_main(monkeypatch, tmp_path)
    out = tmp_path / "mmlu"
    rc = bm.main(["--config", str(cfg), "--cells", "none,c__lora_r4_ce",
                 "--object", f"c__lora_r4_ce={obj_path}",
                 "--parents-root", str(tmp_path / "bank"),
                 "--out-dir", str(out), "--run-label", "r2d", "--shots", "2"])
    assert rc == 0
    assert bm._seen_resolve == [(None, "c__lora_r4_ce", str(tmp_path / "bank"),
                                 str(obj_path))]
    ident = json.loads((out / "r2d" / "mmlu_r2d_c__lora_r4_ce.json").read_text()
                       )["identity"]
    assert ident["parents_root"] == str(tmp_path / "bank")
    assert ident["explicit_objects"] == {"c__lora_r4_ce": str(obj_path)}
    assert ident["defenses_root"] is None
    # a fitted cell with neither a bank nor an explicit path refuses
    with pytest.raises(SystemExit, match="defenses-root"):
        bm.main(["--config", str(cfg), "--cells", "none,c__lora_r4_ce",
                 "--out-dir", str(out), "--run-label", "r3"])
    # --object entries: CELL=PATH, once, for a cell in --cells
    with pytest.raises(SystemExit, match="CELL=PATH"):
        bm.parse_explicit_objects(["nope"], ["none"])
    with pytest.raises(SystemExit, match="twice"):
        bm.parse_explicit_objects(["a=1", "a=2"], ["a"])
    with pytest.raises(SystemExit, match="not in --cells"):
        bm.parse_explicit_objects(["b=1"], ["a"])
    assert bm.parse_explicit_objects(["a=/x/object.pt"], ["a", "none"]) == \
        {"a": "/x/object.pt"}
    assert bm.parse_explicit_objects([], ["a"]) == {}


def test_resolve_object_path_ignores_provenance_sidecar(tmp_path, monkeypatch):
    """A PSR cell dir with object.pt + object.pt.provenance.json must resolve
    unambiguously to object.pt (the sidecar is provenance, not an artifact)."""
    import importlib, sys
    bm = sys.modules.get("base_mmlu") or importlib.import_module("base_mmlu")
    cell = "constv2-mix-1m__psr_ce_all"
    d = tmp_path / "cells_all" / cell
    d.mkdir(parents=True)
    (d / "object.pt").write_bytes(b"x")
    (d / "object.pt.provenance.json").write_text("{}")
    # bypass the identity gate at its inner call: only discovery is under test
    import sisteer.eval.train_budget as tb
    monkeypatch.setattr(tb, "assert_staged_cell", lambda *a, **k: None)
    monkeypatch.setattr(tb, "assert_derived_object", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(bm, "object_artifact_name", lambda arm: "object.does-not-exist", raising=False)
    got = bm.resolve_object_path(tmp_path, cell)
    assert str(got).endswith("/object.pt"), got


def test_summary_merge_keeps_a_pre_merge_summary_and_refuses_identity_drift(
        tmp_path):
    """write_summary (2026-09-03): topping a label up with more cells keeps
    the earlier run's record (a legacy summary without ``invocations`` is
    recorded as the first invocation); a summary of the same label under a
    different identity (here: scale) is refused untouched."""
    spath = tmp_path / "mmlu_r2_summary.json"
    legacy = {"run": _identity(), "requested_cells": ["none", "a__b"],
              "cells": {"none": {"acc": 0.7}, "a__b": {"acc": 0.6}}}
    spath.write_text(json.dumps(legacy))
    bm.write_summary(spath, _identity(), ["c__d"], {"c__d": {"acc": 0.5}})
    s = json.loads(spath.read_text())
    assert s["requested_cells"] == ["none", "a__b", "c__d"]
    assert s["cells"]["none"] == {"acc": 0.7} and "c__d" in s["cells"]
    assert [i["requested_cells"] for i in s["invocations"]] == \
        [["none", "a__b"], ["c__d"]]
    assert s["invocations"][0]["written_at"] is None
    assert s["invocations"][1]["written_at"]
    with pytest.raises(SystemExit, match="scale differs"):
        bm.write_summary(spath, _identity(scale=2.0), ["none"], {"none": {}})
    assert json.loads(spath.read_text()) == s
    # a fresh label: one invocation, no synthetic legacy entry
    fresh = tmp_path / "mmlu_r2_x2_summary.json"
    bm.write_summary(fresh, _identity("r2_x2", scale=2.0), ["none"],
                     {"none": {"acc": 0.1}})
    assert len(json.loads(fresh.read_text())["invocations"]) == 1


def test_preflight_runs_before_the_model_loads_and_refuses_collisions(
        tmp_path, monkeypatch):
    """The label identity check must fire BEFORE any per-cell file is
    written (and before the model loads) — a label collision used to be
    caught only at the summary merge, after every cell file had landed.
    Pins the source order in main and the refusals of preflight_label on
    the MMLU file layout."""
    src = Path(bm.__file__).read_text()
    body = src.split("def main(", 1)[1]
    assert body.index("preflight_label(") < body.index("load_model_and_tokenizer(")
    assert body.index("resolve_object_path(") < body.index("load_rows(")
    run_dir = bm.run_dir_for(tmp_path, "r2")
    run_dir.mkdir()
    rows, dev = _rows(), _dev()
    per = [{"qidx": i, "qid": bm.question_id(r), "subject": r["subject"],
            "picked": "A", "answer": bm.LETTERS[r["answer"]],
            "correct": r["answer"] == 0} for i, r in enumerate(rows)]
    agg = {"acc": sum(p["correct"] for p in per) / 8,
           "correct": sum(p["correct"] for p in per), "n": 8, "ci95": [0, 1]}
    bm.write_cell_results(run_dir, "r2", "x__psr", agg, per,
                          _identity(cell="x__psr", object_sha256="a" * 64))
    kw = dict(cell_paths_fn=bm.cell_paths, summary_path_fn=bm.summary_path)
    with pytest.raises(SystemExit, match="ONE object per cell"):
        bm.preflight_label(run_dir, "r2", _identity(), {"x__psr": "b" * 64},
                           **kw)
    with pytest.raises(SystemExit, match="already holds 'x__psr'.*shots"):
        bm.preflight_label(run_dir, "r2", _identity(shots=5),
                           {"x__psr": "a" * 64}, **kw)
    assert bm.preflight_label(run_dir, "r2", _identity(),
                              {"x__psr": "a" * 64, "none": None},
                              **kw) == ["x__psr"]
    bm.write_summary(bm.summary_path(run_dir, "r2"), _identity(), ["x__psr"],
                     {"x__psr": agg})
    with pytest.raises(SystemExit, match="already holds a summary.*scale"):
        bm.preflight_label(run_dir, "r2", _identity(scale=2.0), {"none": None},
                           **kw)


def test_runtime_identity_and_legacy_summaries(tmp_path):
    """On the MMLU side, dtype / attn_implementation are merge keys; a
    summary that predates their stamping is accepted iff its
    config_sha256 equals the new run's, else refused by name."""
    spath = tmp_path / "mmlu_r2_summary.json"
    legacy = _identity()
    legacy.pop("dtype"); legacy.pop("attn_implementation")
    spath.write_text(json.dumps({"run": legacy, "requested_cells": ["none"],
                                 "cells": {"none": {"acc": 0.7}}}))
    kw = dict(cell_paths_fn=bm.cell_paths, summary_path_fn=bm.summary_path)
    assert bm.preflight_label(tmp_path, "r2", _identity(), {"none": None},
                              **kw) == []
    with pytest.raises(SystemExit, match="attn_implementation is absent|dtype is absent"):
        bm.preflight_label(tmp_path, "r2", _identity(config_sha256="d" * 64),
                           {"none": None}, **kw)
    bm.write_summary(spath, _identity(), ["x__psr"], {"x__psr": {"acc": 0.5}})
    with pytest.raises(SystemExit, match="dtype differs"):
        bm.write_summary(spath, _identity(dtype="float32"), ["y__psr"],
                         {"y__psr": {}})
    with pytest.raises(SystemExit, match="attn_implementation differs"):
        bm.write_summary(spath, _identity(attn_implementation="eager"),
                         ["y__psr"], {"y__psr": {}})
    src = Path(bm.__file__).read_text().split("def main(", 1)[1]
    assert '"dtype": cfg.get("dtype", "bfloat16")' in src
    assert '"attn_implementation": cfg.get("attn_implementation", "sdpa")' in src
