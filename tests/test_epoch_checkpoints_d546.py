"""Every full-scale fit keeps a DURABLE object checkpoint at the end of epoch 1, epoch 2 and the final epoch — named
artefacts with provenance, never rotated by the step-interval resume checkpointing — persisted under a STABLE
fingerprint-bound root (so a restarted fit keeps the epochs an earlier attempt saved), merged into the publish dir with
a VALID identity sidecar per epoch, and registered as per-epoch `object` rows (servable by pinned sha) plus artifacts.

Pinned here: the saver (root + key; files, index, named roles, `final` re-pointing, no rotation, a foreign key refused);
finalize/merge (identity sidecars re-keyed from the cell's; the production consumer `staged_cell_problems` accepts each
epoch object; key mismatch refused by name); a REAL two-process resume (a fresh interpreter writes a later epoch under
the same root, this process finalizes and publishes — all epochs present); the hook's per-epoch object + artifact rows;
the lockstep and solo (train_object_ce) seams by signature.
"""
import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts", REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import train_two_si_objects as t2  # noqa: E402
from sisteer.extraction import lockstep, objects  # noqa: E402
from sisteer.registry import hooks  # noqa: E402

CELL = "constv2-mix-full__psr_ce_all"
KEY = "fp-" + "a" * 60


class _FakeObj:
    def __init__(self, payload: bytes):
        self.payload = payload

    def save(self, path):
        Path(path).write_bytes(self.payload)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _save(root, payload, epoch, **kw):
    return t2.save_epoch_checkpoint(root, _FakeObj(payload), arm="psr_ce_all", si_id="constv2-mix-full", epoch=epoch, step=267 * epoch,
                                    history=[{"loss": 2.0 - 0.1 * epoch}], heldout_curve=[2.04] + [2.0 - 0.1 * e for e in range(1, epoch + 1)],
                                    stopped=False, epochs_cap=16, checkpoint_key=KEY, **kw)


def _cell_meta(tmp: Path, final_obj: bytes) -> dict:
    (tmp / "object.pt").write_bytes(final_obj)
    # a document-fit sidecar as the trainer writes it: completion_tokens is NULL by schema, sequence_max_len bounds the docs
    meta = {"cell_id": CELL, "si_id": "constv2-mix-full", "arm": "psr_ce_all", "object_sha256": hashlib.sha256(final_obj).hexdigest(),
            "completion_tokens": None, "training_budget_kind": "document_tokens", "sequence_max_len": 8192, "sample_manifest_sha": None,
            "model_id": "Qwen/Qwen3.5-4B-Base", "model_revision": "0" * 40, "max_len": 8192, "positions_at_fit": "all", "positions": "all",
            "seed": 0, "lr": 6.4565e-3, "epochs": 16, "template_hash": "e3b0c44298fc1c14", "trainer": "sdfbase.fit", "target_layers": [4, 8]}
    (tmp / "train_budget_meta.json").write_text(json.dumps(meta))
    (tmp / "fit_selection.json").write_text(json.dumps({"fit_search_id": "hpo-native:feedface", "fit_lr_selected_by": "search",
                                                         "init_scheme": "random_lecun_1e-3"}))
    (tmp / "metrics.jsonl").write_text("")
    return meta


def test_seams_exist_on_both_fit_paths():
    assert "epoch_cb" in inspect.signature(lockstep.train_lockstep).parameters
    assert "epoch_cb" in inspect.signature(objects.train_object_ce).parameters, "the ReLU/CE arms train SOLO through train_object_ce"


def test_epoch_root_is_namespaced_by_the_fingerprint_or_none():
    import hashlib as _h
    r = t2.epoch_checkpoint_root({"checkpoint_dir": "/ckpt/cell", "checkpoint_key": KEY}, "psr_ce_all")
    assert r == Path("/ckpt/cell/epoch_ckpts/psr_ce_all") / _h.sha256(KEY.encode()).hexdigest()[:16]
    r2 = t2.epoch_checkpoint_root({"checkpoint_dir": "/ckpt/cell", "checkpoint_key": "fp-other"}, "psr_ce_all")
    assert r2 != r and r2.parent == r.parent, "two fingerprints never share a directory (a HIGH review finding, round 2)"
    assert t2.epoch_checkpoint_root({"checkpoint_dir": "/ckpt/cell"}, "psr_ce_all") is None
    assert t2.epoch_checkpoint_root({}, "psr_ce_all") is None and t2.epoch_checkpoint_root(None, "psr_ce_all") is None


def test_saver_writes_named_durable_artefacts_repoints_final_and_refuses_a_foreign_key(tmp_path):
    root = tmp_path / "ckpt" / "epoch_ckpts" / "psr_ce_all"
    e1 = _save(root, b"epoch-one", 1)
    art1 = root / "epoch_01" / "object.pt"
    assert art1.read_bytes() == b"epoch-one" and e1["artifact"] == "epoch_ckpts/epoch_01/object.pt" and e1["artifact_sha256"] == _sha(art1)
    assert e1["epoch_id"] == f"{CELL}@epoch_01" and e1["checkpoint_key"] == KEY and e1["heldout_loss"] == pytest.approx(1.9)
    idx = json.loads((root / "epoch_checkpoints.json").read_text())
    assert idx["checkpoint_key"] == KEY and idx["named"] == {"epoch_1": "epoch_01", "final": "epoch_01"}
    _save(root, b"epoch-two", 2); _save(root, b"epoch-three", 3)
    idx = json.loads((root / "epoch_checkpoints.json").read_text())
    assert [c["epoch"] for c in idx["checkpoints"]] == [1, 2, 3]
    assert idx["named"] == {"epoch_1": "epoch_01", "epoch_2": "epoch_02", "final": "epoch_03"}, "final names the LAST epoch run"
    assert art1.read_bytes() == b"epoch-one", "earlier epochs are never rotated"
    _save(root, b"epoch-three-again", 3)   # a resumed fit replaying the epoch replaces the entry, never duplicates it
    idx = json.loads((root / "epoch_checkpoints.json").read_text())
    assert [c["epoch"] for c in idx["checkpoints"]] == [1, 2, 3]
    with pytest.raises(RuntimeError, match="another identity"):
        t2.save_epoch_checkpoint(root, _FakeObj(b"x"), arm="psr_ce_all", si_id="constv2-mix-full", epoch=4, step=1, history=[], heldout_curve=[],
                                 stopped=False, epochs_cap=16, checkpoint_key="fp-other")


def test_finalize_merges_into_the_publish_dir_with_valid_identity_sidecars_and_the_consumer_accepts_them(tmp_path):
    from sisteer.eval.train_budget import staged_cell_problems
    root = tmp_path / "ckpt" / "epoch_ckpts" / "psr_ce_all"
    for ep, payload in ((1, b"e1"), (2, b"e2"), (5, b"e5")):
        _save(root, payload, ep)
    tmp = tmp_path / f"{CELL}.tmp-123"; tmp.mkdir()
    meta = _cell_meta(tmp, b"final-object")
    idx = t2.finalize_epoch_checkpoints(root, tmp, checkpoint_key=KEY, cell_id=CELL)
    assert [c["epoch"] for c in idx["checkpoints"]] == [1, 2, 5] and idx["named"]["final"] == "epoch_05"
    for ep in (1, 2, 5):
        d = tmp / "epoch_ckpts" / f"epoch_{ep:02d}"
        side = json.loads((d / "train_budget_meta.json").read_text())
        assert side["cell_id"] == CELL and side["object_sha256"] == _sha(d / "object.pt") != meta["object_sha256"]
        assert side["epoch_checkpoint"]["epoch"] == ep and side["epoch_checkpoint"]["parent_object_sha256"] == meta["object_sha256"]
        assert side["sequence_max_len"] == meta["sequence_max_len"] and side["completion_tokens"] is None
        assert (d / "fit_selection.json").exists() and (d / "provenance.json").exists()
        # the production consumers' identity check (base_mmlu / ce_score go through staged_cell_problems) accepts the epoch object
        assert staged_cell_problems(d / "object.pt", CELL, require_sidecar=True) == []
    # publish = the dir rename: everything travels with the cell
    final = tmp_path / CELL
    tmp.rename(final)
    assert (final / "epoch_ckpts" / "epoch_02" / "object.pt").read_bytes() == b"e2" and (final / "epoch_checkpoints.json").exists()
    # a key mismatch refuses by name — another identity's snapshots never publish under this cell
    tmp2 = tmp_path / f"{CELL}.tmp-124"; tmp2.mkdir(); _cell_meta(tmp2, b"final2")
    with pytest.raises(RuntimeError, match="another identity"):
        t2.finalize_epoch_checkpoints(root, tmp2, checkpoint_key="fp-other", cell_id=CELL)
    assert not (tmp2 / "epoch_ckpts").exists()
    # no index -> nothing to merge, no error
    assert t2.finalize_epoch_checkpoints(tmp_path / "nowhere", tmp2, checkpoint_key=KEY, cell_id=CELL) is None


def test_cross_process_resume_keeps_the_earlier_epochs(tmp_path):
    """Process A (this one) saves epochs 1-2 under the stable root and dies (its temp cell dir is gone); process B (a fresh
    interpreter) resumes under the SAME fingerprint, saves epoch 3, finalizes into ITS new temp cell dir and publishes.
    All three epochs are in the published cell, indexed and named."""
    root = tmp_path / "ckpt" / "epoch_ckpts" / "psr_ce_all"
    _save(root, b"e1", 1); _save(root, b"e2", 2)
    code = f"""
import json, sys, hashlib
from pathlib import Path
sys.path[:0] = [{str(REPO / 'scripts')!r}, {str(REPO / 'src')!r}]
import train_two_si_objects as t2
class O:
    def __init__(s, b): s.b = b
    def save(s, p): Path(p).write_bytes(s.b)
root = Path({str(root)!r}); tmp = Path({str(tmp_path / (CELL + '.tmp-999'))!r}); tmp.mkdir()
t2.save_epoch_checkpoint(root, O(b'e3'), arm='psr_ce_all', si_id='constv2-mix-full', epoch=3, step=801, history=[], heldout_curve=[2.0, 1.9, 1.8, 1.7],
                         stopped=True, epochs_cap=16, checkpoint_key={KEY!r})
(tmp / 'object.pt').write_bytes(b'final'); (tmp / 'fit_selection.json').write_text('{{}}')
(tmp / 'train_budget_meta.json').write_text(json.dumps({{'cell_id': {CELL!r}, 'object_sha256': hashlib.sha256(b'final').hexdigest(), 'completion_tokens': 1}}))
idx = t2.finalize_epoch_checkpoints(root, tmp, checkpoint_key={KEY!r}, cell_id={CELL!r})
tmp.rename(Path({str(tmp_path / CELL)!r}))
print(json.dumps([c['epoch'] for c in idx['checkpoints']]), json.dumps(idx['named']))
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-1500:]
    final = tmp_path / CELL
    assert sorted(p.name for p in (final / "epoch_ckpts").iterdir()) == ["epoch_01", "epoch_02", "epoch_03"]
    idx = json.loads((final / "epoch_checkpoints.json").read_text())
    assert [c["epoch"] for c in idx["checkpoints"]] == [1, 2, 3] and idx["named"] == {"epoch_1": "epoch_01", "epoch_2": "epoch_02", "final": "epoch_03"}
    assert (final / "epoch_ckpts" / "epoch_01" / "object.pt").read_bytes() == b"e1", "process A's epoch survived process B's publish"
    assert json.loads((final / "epoch_ckpts" / "epoch_03" / "train_budget_meta.json").read_text())["epoch_checkpoint"]["stopped"] is True


def test_hook_emits_per_epoch_object_rows_and_artifacts(tmp_path, monkeypatch):
    root = tmp_path / "ckpt" / "epoch_ckpts" / "psr_ce_all"
    for ep, payload in ((1, b"e1"), (2, b"e2"), (6, b"e6")):
        _save(root, payload, ep)
    cell = tmp_path / CELL; cell.mkdir()
    _cell_meta(cell, b"final-object")
    t2.finalize_epoch_checkpoints(root, cell, checkpoint_key=KEY, cell_id=CELL)
    written = []

    class FakeWriter:
        def __init__(self, *a, **k): pass
        def emit_object(self, payload): written.append(("object", payload)); return f"obj:{payload['object_id'][:8]}"
        def emit_artifact(self, payload): written.append(("artifact", payload)); return f"art:{payload['artifact_id'][:8]}"
        def push(self, final=False): return 0

    monkeypatch.setattr(hooks.emit, "RecordWriter", FakeWriter)
    monkeypatch.setattr(hooks.emit, "producer_block", lambda script, session=None: {"script": script, "code_sha": "unverified"})
    monkeypatch.setenv(hooks.emit.SPOOL_ENV, str(tmp_path / "spool"))      # the real gates: a spool resolves, emission explicitly on
    monkeypatch.delenv(hooks.emit.DISABLED_ENV, raising=False)
    out = hooks.emit_object_from_sidecar(str(cell), campaign="two_si", object_kind="relu_steer", paper_label="other", object_role="control",
                                         fit_target_kind="corpus_document_fit", fit_objective="ce", script="scripts/train_two_si_objects.py",
                                         registry_emit=True, spool=str(tmp_path / "spool"))
    assert written, "the hook's emission gates must be open in this test (spool set, registry_emit=True)"
    objs = [p for k, p in written if k == "object"]; arts = [p for k, p in written if k == "artifact"]
    assert len(objs) == 4 and len(arts) == 4, (len(objs), len(arts))                      # the cell + 3 epochs
    final_sha = hashlib.sha256(b"final-object").hexdigest()
    epoch_objs = [p for p in objs if p["object_id"] != final_sha]
    assert sorted(p["extra"]["epoch"] for p in epoch_objs) == [1, 2, 6]
    assert all(p["extra"]["parent_object_id"] == final_sha and p["extra"]["role"] == "epoch_checkpoint" for p in epoch_objs)
    assert {p["extra"]["named_role"] for p in epoch_objs} == {"epoch_1", "epoch_2", "final"}
    assert all(p["extra"]["epoch_id"].startswith(CELL + "@epoch_") for p in epoch_objs)
    by_epoch = {p["extra"]["epoch"]: p for p in epoch_objs}
    assert by_epoch[2]["object_id"] == hashlib.sha256(b"e2").hexdigest(), "an epoch object is pinned by ITS artefact sha"
    assert len(out) == 8


def test_identity_is_checked_before_any_byte_is_written(tmp_path):
    root = tmp_path / "root"
    _save(root, b"k1-epoch-one", 1)
    before = (root / "epoch_01" / "object.pt").read_bytes()
    with pytest.raises(RuntimeError, match="another identity"):
        t2.save_epoch_checkpoint(root, _FakeObj(b"k2-bytes"), arm="psr_ce_all", si_id="constv2-mix-full", epoch=1, step=1, history=[],
                                 heldout_curve=[], stopped=False, epochs_cap=16, checkpoint_key="fp-other")
    assert (root / "epoch_01" / "object.pt").read_bytes() == before, "the foreign identity's artefact was never touched"
    with pytest.raises(RuntimeError, match="refusing to touch"):
        t2.save_epoch_checkpoint(root, _FakeObj(b"x"), arm="psr_ce_all", si_id="OTHER-si", epoch=1, step=1, history=[],
                                 heldout_curve=[], stopped=False, epochs_cap=16, checkpoint_key=KEY)


def test_a_retried_cell_under_a_new_fingerprint_publishes_beside_an_interrupted_old_root(tmp_path):
    """An interrupted fit left epoch 1 under key K1; the cell is retried under K2 (changed config).
    K2's epochs live in their own namespace, K1's bytes stay, and K2 finalizes + publishes without touching K1."""
    ck = {"checkpoint_dir": str(tmp_path / "ckpt" / CELL)}
    root1 = t2.epoch_checkpoint_root({**ck, "checkpoint_key": "K1" + "1" * 62}, "psr_ce_all")
    root2 = t2.epoch_checkpoint_root({**ck, "checkpoint_key": KEY}, "psr_ce_all")
    assert root1 != root2
    t2.save_epoch_checkpoint(root1, _FakeObj(b"k1"), arm="psr_ce_all", si_id="constv2-mix-full", epoch=1, step=1, history=[], heldout_curve=[],
                             stopped=False, epochs_cap=16, checkpoint_key="K1" + "1" * 62)
    _save(root2, b"k2-e1", 1); _save(root2, b"k2-e2", 2)
    tmp = tmp_path / f"{CELL}.tmp-7"; tmp.mkdir(); _cell_meta(tmp, b"k2-final")
    idx = t2.finalize_epoch_checkpoints(root2, tmp, checkpoint_key=KEY, cell_id=CELL)
    assert [c["epoch"] for c in idx["checkpoints"]] == [1, 2]
    assert (root1 / "epoch_01" / "object.pt").read_bytes() == b"k1", "the interrupted K1 root is untouched"
    assert (tmp / "epoch_ckpts" / "epoch_01" / "object.pt").read_bytes() == b"k2-e1"


def test_manifest_covers_the_epoch_subtree_and_loss_after_publish_is_incomplete(tmp_path):
    """The COMPLETE manifest lists the index, every epoch artefact and its sidecars, so a lost or altered epoch artefact
    after publication makes validated_complete return None (resume re-runs) instead of failing only when the battery
    serves the pinned epoch object."""
    import json as _json
    root = tmp_path / "ckpt" / "epoch_ckpts" / "psr_ce_all" / "k"
    _save(root, b"e1", 1); _save(root, b"e2", 2); _save(root, b"e3", 3)
    save_dir = tmp_path / "store"; save_dir.mkdir()
    tmp = t2._cell_tmp_dir(save_dir, CELL)
    _cell_meta(tmp, b"final-object")
    t2.finalize_epoch_checkpoints(root, tmp, checkpoint_key=KEY, cell_id=CELL)
    names = t2._required_publication_files(tmp, "object.pt")
    assert "epoch_checkpoints.json" in names and "epoch_ckpts/epoch_02/object.pt" in names
    assert "epoch_ckpts/epoch_02/train_budget_meta.json" in names and "epoch_ckpts/epoch_02/provenance.json" in names
    final = t2.publish_cell(save_dir, CELL, tmp, si_id="constv2-mix-full", arm="psr_ce_all", positions="all",
                            audit={"metrics_rows": 0, "object_file": "object.pt", "monitors_on": False}, fingerprint=KEY)
    marker = _json.loads((final / "COMPLETE").read_text())
    assert set(n for n in marker["files"] if n.startswith("epoch_ckpts/")) >= {"epoch_ckpts/epoch_01/object.pt", "epoch_ckpts/epoch_02/object.pt", "epoch_ckpts/epoch_03/object.pt"}
    assert t2.validated_complete(save_dir, CELL, KEY) == final / "object.pt"
    (final / "epoch_ckpts" / "epoch_02" / "object.pt").write_bytes(b"corrupted")
    assert t2.validated_complete(save_dir, CELL, KEY) is None, "an altered epoch artefact makes the cell INCOMPLETE"
    (final / "epoch_ckpts" / "epoch_02" / "object.pt").write_bytes(b"e2")
    assert t2.validated_complete(save_dir, CELL, KEY) == final / "object.pt"
    (final / "epoch_checkpoints.json").unlink()
    assert t2.validated_complete(save_dir, CELL, KEY) is None, "a lost index makes the cell INCOMPLETE"


def test_publish_refuses_without_the_named_epochs(tmp_path):
    root = tmp_path / "root"
    _save(root, b"e1", 1); _save(root, b"e2", 2); _save(root, b"e4", 4)
    (root / "epoch_02" / "object.pt").unlink()            # epoch 2 was reached but its artefact is gone
    tmp = tmp_path / f"{CELL}.tmp-9"; tmp.mkdir(); _cell_meta(tmp, b"final")
    with pytest.raises(RuntimeError, match=r"named epoch checkpoint\(s\) \['epoch_2'\]"):
        t2.finalize_epoch_checkpoints(root, tmp, checkpoint_key=KEY, cell_id=CELL)


def test_a_hooked_fit_that_saved_nothing_does_not_publish(tmp_path):
    """Every epoch save failed (logged, not fatal) -> no index. A fit whose hook WAS wired must not publish as an
    epoch-checkpointed fit; a legacy/no-hook fit (required=False) still publishes without epoch artefacts."""
    tmp = tmp_path / f"{CELL}.tmp-1"; tmp.mkdir(); _cell_meta(tmp, b"final")
    with pytest.raises(RuntimeError, match="hook was wired but no index exists"):
        t2.finalize_epoch_checkpoints(tmp_path / "ckpt" / "nowhere", tmp, checkpoint_key=KEY, cell_id=CELL, required=True)
    assert t2.finalize_epoch_checkpoints(tmp_path / "ckpt" / "nowhere", tmp, checkpoint_key=KEY, cell_id=CELL, required=False) is None


def test_a_malformed_published_index_reads_incomplete_at_the_resume_gate(tmp_path, capsys):
    """Publication corruption of the epoch index is INCOMPLETE to validated_complete (resume re-runs /
    re-publishes), never an uncaught crash at the gate; publish-time derivation still fails closed."""
    root = tmp_path / "ckpt" / "k"
    _save(root, b"e1", 1); _save(root, b"e2", 2)
    save_dir = tmp_path / "store"; save_dir.mkdir()
    tmp = t2._cell_tmp_dir(save_dir, CELL); _cell_meta(tmp, b"final")
    t2.finalize_epoch_checkpoints(root, tmp, checkpoint_key=KEY, cell_id=CELL, required=True)
    final = t2.publish_cell(save_dir, CELL, tmp, si_id="constv2-mix-full", arm="psr_ce_all", positions="all",
                            audit={"metrics_rows": 0, "object_file": "object.pt", "monitors_on": False}, fingerprint=KEY)
    assert t2.validated_complete(save_dir, CELL, KEY) == final / "object.pt"
    (final / "epoch_checkpoints.json").write_text("{not json")
    assert t2.validated_complete(save_dir, CELL, KEY) is None
    assert "INCOMPLETE" in capsys.readouterr().err
    with pytest.raises(RuntimeError, match="unreadable epoch-checkpoint index"):
        t2._required_publication_files(final, "object.pt")


KEY2 = "fp-" + "b" * 60


def _published_predecessor(tmp_path, *, with_index=True):
    """A predecessor cell as `--continue-from <cell>/final_ckpt/ckpt.pt` names it: epochs 1-2 saved under its root,
    finalized + published (index + epoch_ckpts/ in the cell dir), a step checkpoint under final_ckpt/."""
    pred_root = tmp_path / "pred_root"
    _save(pred_root, b"p-e1", 1); _save(pred_root, b"p-e2", 2)
    save_dir = tmp_path / "pred_store"; save_dir.mkdir()
    tmp = t2._cell_tmp_dir(save_dir, CELL); _cell_meta(tmp, b"p-final")
    if with_index:
        t2.finalize_epoch_checkpoints(pred_root, tmp, checkpoint_key=KEY, cell_id=CELL, required=True)
    cell = t2.publish_cell(save_dir, CELL, tmp, si_id="constv2-mix-full", arm="psr_ce_all", positions="all",
                           audit={"metrics_rows": 0, "object_file": "object.pt", "monitors_on": False}, fingerprint=KEY)
    (cell / "final_ckpt").mkdir(); (cell / "final_ckpt" / "ckpt.pt").write_bytes(b"step-ckpt")
    cont = {"arm": "psr_ce_all", "checkpoint": str(cell / "final_ckpt" / "ckpt.pt"),
            "marker": {"fit_fingerprint": KEY, "epoch": 2, "step": 534, "batches_done": 534, "epochs_cap": 2},   # as continuation_marker writes it
            "record": {"checkpoint": str(cell / "final_ckpt" / "ckpt.pt")}}
    return cell, cont


def _save2(root, payload, epoch):
    return t2.save_epoch_checkpoint(root, _FakeObj(payload), arm="psr_ce_all", si_id="constv2-mix-full", epoch=epoch, step=267 * epoch,
                                    history=[{"loss": 1.5}], heldout_curve=[2.04] + [2.0 - 0.1 * e for e in range(1, epoch + 1)],
                                    stopped=False, epochs_cap=24, checkpoint_key=KEY2)


def test_continuation_inherits_the_predecessors_published_epochs_and_publishes_all_named(tmp_path):
    """The `lora_r4_ce` cap-raise continuation API stays; a continued generation IMPORTS the predecessor's verified
    epoch_1/epoch_2 (its published epoch_ckpts/) into its own fingerprint root with `imported_from` provenance, then
    saves its own later epochs; the publication names epoch_1, epoch_2 and final."""
    pred_cell, cont = _published_predecessor(tmp_path)
    root2 = tmp_path / "ckpt" / "k2"
    idx = t2.import_predecessor_epoch_checkpoints(cont, root2, checkpoint_key=KEY2, cell_id=CELL)
    assert [c["epoch"] for c in idx["checkpoints"]] == [1, 2] and idx["checkpoint_key"] == KEY2
    assert all(c["imported_from"]["cell_dir"] == str(pred_cell) and c["imported_from"]["fit_fingerprint"] == KEY for c in idx["checkpoints"])
    assert idx["lineage"]["inherited_epochs"] == [1, 2] and idx["named"] == {"epoch_1": "epoch_01", "epoch_2": "epoch_02", "final": "epoch_02"}
    assert (root2 / "epoch_01" / "object.pt").read_bytes() == b"p-e1" and not (root2 / "epoch_01" / "train_budget_meta.json").exists(), \
        "the predecessor's identity sidecar is NOT copied: finalize re-keys one for THIS cell"
    # the continued run trains epochs 3 and 4 under its own key, in the same root
    _save2(root2, b"c-e3", 3); _save2(root2, b"c-e4", 4)
    idx = json.loads((root2 / t2.EPOCH_CKPT_INDEX).read_text())
    assert idx["lineage"]["predecessor_has_epoch_index"] is True and idx["lineage"]["predecessor_epochs_done"] == 2
    assert idx["named"]["final"] == "epoch_04" and idx["named"]["epoch_1"] == "epoch_01", "the saver keeps the lineage and re-points final"
    save_dir = tmp_path / "store2"; save_dir.mkdir()
    tmp = t2._cell_tmp_dir(save_dir, CELL); _cell_meta(tmp, b"c-final")
    out = t2.finalize_epoch_checkpoints(root2, tmp, checkpoint_key=KEY2, cell_id=CELL, required=True)
    assert out["not_reached_in_lineage"] == [] and {c["epoch"] for c in out["checkpoints"]} == {1, 2, 3, 4}
    from sisteer.eval.train_budget import staged_cell_problems
    for ep in ("epoch_01", "epoch_02", "epoch_03", "epoch_04"):
        assert staged_cell_problems(tmp / "epoch_ckpts" / ep / "object.pt", CELL, require_sidecar=True) == [], ep
    side = json.loads((tmp / "epoch_ckpts" / "epoch_01" / "train_budget_meta.json").read_text())
    assert side["object_sha256"] == hashlib.sha256(b"p-e1").hexdigest() and side["epoch_checkpoint"]["parent_object_sha256"] == hashlib.sha256(b"c-final").hexdigest()


def test_import_skips_an_altered_predecessor_artefact_and_refuses_a_foreign_root(tmp_path, capsys):
    pred_cell, cont = _published_predecessor(tmp_path)
    (pred_cell / "epoch_ckpts" / "epoch_02" / "object.pt").write_bytes(b"tampered")
    root2 = tmp_path / "ckpt" / "k2"
    idx = t2.import_predecessor_epoch_checkpoints(cont, root2, checkpoint_key=KEY2, cell_id=CELL)
    assert [c["epoch"] for c in idx["checkpoints"]] == [1] and "altered" in capsys.readouterr().err
    # a root already owned by another identity is not imported into
    _save(tmp_path / "ckpt" / "other", b"x", 1)
    with pytest.raises(RuntimeError, match="another identity's checkpoint_key"):
        t2.import_predecessor_epoch_checkpoints(cont, tmp_path / "ckpt" / "other", checkpoint_key=KEY2, cell_id=CELL)


def test_continuation_of_a_pre_d546_predecessor_publishes_and_records_the_lineage_gap(tmp_path, capsys):
    """A predecessor fitted before epoch checkpointing has no epoch index: nothing to inherit; the continued run saves
    ITS epochs (first_epoch_run > 1), publishes with `final` named, and records epochs 1-2 as not_reached_in_lineage
    instead of refusing — the named-epoch requirement is lineage-aware, so an unreachable epoch is never demanded."""
    pred_cell, cont = _published_predecessor(tmp_path, with_index=False)
    assert not (pred_cell / t2.EPOCH_CKPT_INDEX).exists()
    root2 = tmp_path / "ckpt" / "k2"
    idx = t2.import_predecessor_epoch_checkpoints(cont, root2, checkpoint_key=KEY2, cell_id=CELL)
    assert idx["checkpoints"] == [] and idx["lineage"] == {"continued_from": str(pred_cell), "predecessor_fit_fingerprint": KEY,
                                                           "predecessor_has_epoch_index": False, "predecessor_epochs_done": 2,
                                                           "inherited_epochs": []}
    assert "pre-D546" in capsys.readouterr().err
    _save2(root2, b"c-e3", 3); _save2(root2, b"c-e4", 4)
    save_dir = tmp_path / "store2"; save_dir.mkdir()
    tmp = t2._cell_tmp_dir(save_dir, CELL); _cell_meta(tmp, b"c-final")
    out = t2.finalize_epoch_checkpoints(root2, tmp, checkpoint_key=KEY2, cell_id=CELL, required=True)
    assert out["not_reached_in_lineage"] == [1, 2] and out["named"] == {"final": "epoch_04"}
    # ... but an epoch this lineage DID reach is still required
    _save2(root2, b"c-e5", 5)
    (root2 / "epoch_04" / "object.pt").unlink()
    tmp2 = t2._cell_tmp_dir(save_dir, CELL); _cell_meta(tmp2, b"c-final2")
    assert t2.finalize_epoch_checkpoints(root2, tmp2, checkpoint_key=KEY2, cell_id=CELL, required=True)["named"]["final"] == "epoch_05"
    # a FRESH fit whose epoch-1 save failed transiently has no lineage block -> epoch_1 is still required; the
    # exemption is never inferred from which save happened to succeed first
    root3 = tmp_path / "ckpt" / "k3"
    _save2(root3, b"d-e2", 2); _save2(root3, b"d-e3", 3)
    tmp3 = t2._cell_tmp_dir(save_dir, CELL); _cell_meta(tmp3, b"d-final")
    with pytest.raises(RuntimeError, match=r"\['epoch_1'\]"):
        t2.finalize_epoch_checkpoints(root3, tmp3, checkpoint_key=KEY2, cell_id=CELL, required=True)
    assert "first_epoch_run" not in json.loads((root3 / t2.EPOCH_CKPT_INDEX).read_text())


def test_a_d546_predecessor_whose_epoch_was_not_inherited_gives_no_exemption(tmp_path, capsys):
    """Only a VERIFIED predecessor fitted before epoch checkpointing exempts early epochs. A predecessor WITH an index
    whose epoch_2 artefact was altered (so the import skipped it) leaves epoch_2 required -> the continued cell refuses."""
    pred_cell, cont = _published_predecessor(tmp_path)
    (pred_cell / "epoch_ckpts" / "epoch_02" / "object.pt").write_bytes(b"tampered")
    root2 = tmp_path / "ckpt" / "k2"
    idx = t2.import_predecessor_epoch_checkpoints(cont, root2, checkpoint_key=KEY2, cell_id=CELL)
    assert idx["lineage"]["predecessor_has_epoch_index"] is True and idx["lineage"]["inherited_epochs"] == [1]
    _save2(root2, b"c-e3", 3)
    save_dir = tmp_path / "store2"; save_dir.mkdir()
    tmp = t2._cell_tmp_dir(save_dir, CELL); _cell_meta(tmp, b"c-final")
    with pytest.raises(RuntimeError, match=r"\['epoch_2'\]"):
        t2.finalize_epoch_checkpoints(root2, tmp, checkpoint_key=KEY2, cell_id=CELL, required=True)

def test_publish_refuses_when_the_last_epochs_save_failed(tmp_path):
    """Epochs 1-2 saved, the final epoch-3 save failed (the callbacks log and continue) -> the index's `final` still
    names epoch_02. Finalize binds `final` to the epoch the trainer actually ran (the driver's callback count, recorded
    before any save) and refuses; with the matching count it publishes."""
    root = tmp_path / "root"
    _save(root, b"e1", 1); _save(root, b"e2", 2)
    tmp = tmp_path / f"{CELL}.tmp-5"; tmp.mkdir(); _cell_meta(tmp, b"final")
    with pytest.raises(RuntimeError, match=r"ran 3 epoch\(s\) but the index's final checkpoint is 'epoch_02'"):
        t2.finalize_epoch_checkpoints(root, tmp, checkpoint_key=KEY, cell_id=CELL, required=True, expected_final_epoch=3)
    out = t2.finalize_epoch_checkpoints(root, tmp, checkpoint_key=KEY, cell_id=CELL, required=True, expected_final_epoch=2)
    assert out["named"]["final"] == "epoch_02"
    # a final artefact LOST after a successful save is refused too (the named-epoch check catches it first: `final`
    # names epoch_03, which is no longer kept); the trainer binding stands behind it
    _save(root, b"e3", 3); (root / "epoch_03" / "object.pt").unlink()
    tmp2 = tmp_path / f"{CELL}.tmp-6"; tmp2.mkdir(); _cell_meta(tmp2, b"final")
    with pytest.raises(RuntimeError, match=r"\['final'\] missing|refusing to certify an earlier epoch"):
        t2.finalize_epoch_checkpoints(root, tmp2, checkpoint_key=KEY, cell_id=CELL, required=True, expected_final_epoch=3)


def test_continuation_import_is_wired_for_ce_arms_only_at_adoption():
    src = inspect.getsource(t2.train_si_objects)
    assert "import_predecessor_epoch_checkpoints(continuation" in src and 'continuation["arm"] in CE_ARMS' in src
    assert not hasattr(t2, "refuse_continuation_under_d546"), "the round-2 unconditional refusal is gone (review round 3)"
    # BOTH callbacks record the epoch they were invoked for BEFORE the save attempt, and BOTH finalize calls bind
    # `final` to that count
    assert src.count("_epoch_seen[a.name] = int(epoch) + 1") == 1 and src.count("_epoch_seen[_arm] = int(epoch) + 1") == 1
    assert src.count("expected_final_epoch=_epoch_seen.get(") == 2
    for cb in ("_epoch_seen[a.name] = int(epoch) + 1", "_epoch_seen[_arm] = int(epoch) + 1"):
        assert src.index(cb) < src.index("save_epoch_checkpoint(", src.index(cb)), "the count is recorded before the save is attempted"


def _open_hook_gates(monkeypatch, tmp_path, written, *, fake_producer=True):
    class FakeWriter:
        def __init__(self, *a, **k): pass
        def emit_object(self, payload): written.append(("object", json.loads(json.dumps(payload, default=str)))); return f"obj:{payload['object_id'][:8]}"
        def emit_artifact(self, payload): written.append(("artifact", json.loads(json.dumps(payload, default=str)))); return f"art:{payload['artifact_id'][:8]}"
        def push(self, final=False): return 0
    monkeypatch.setattr(hooks.emit, "RecordWriter", FakeWriter)
    if fake_producer:
        monkeypatch.setattr(hooks.emit, "producer_block", lambda script, session=None: {"script": script, "code_sha": "unverified"})
    monkeypatch.setenv(hooks.emit.SPOOL_ENV, str(tmp_path / "spool"))
    monkeypatch.delenv(hooks.emit.DISABLED_ENV, raising=False)
    monkeypatch.setenv("REGISTRY_CAMPAIGN", "two_si")      # the gates include the campaign: the emitter refuses without one since 2026-10-02


def _as_full_event(i, payload, record_type="object"):
    from sisteer.registry.reducer import HistoryEvent
    sha = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    body = {"_kind": "full", **{k: v for k, v in payload.items() if k != "extra"}, "_extra": payload.get("extra") or {}}
    return HistoryEvent(f"2026-10-01T00:00:{i:02d}.000Z", sha, record_type, f"c/{record_type}", body)


def test_hook_fix2_an_epoch_with_the_published_objects_bytes_is_a_role_not_a_second_record(tmp_path, monkeypatch):
    """The CE trainer restores the BEST epoch after the callbacks, so the published object's bytes ARE an epoch's. The
    registry folds full records by natural key, later wins, so a second `object` record under the published object_id
    (epoch extra, self-parent, epoch-only locations) would overwrite its provenance. Now: one object row per distinct
    sha; same-bytes epochs are ROLE MEMBERSHIPS on the canonical object and extra LOCATIONS of its artifact; distinct
    epochs are parented to the published object, never to themselves. Shown through the real reducer, and the defect
    shape is refused by the validator by name."""
    root = tmp_path / "ckpt" / "k"
    _save(root, b"best", 1); _save(root, b"e2", 2); _save(root, b"e2", 3); _save(root, b"e4", 4)   # epoch 3 repeats epoch 2's bytes (plateau)
    cell = tmp_path / CELL; cell.mkdir(); _cell_meta(cell, b"best")                                  # best-epoch restore: published == epoch 1
    t2.finalize_epoch_checkpoints(root, cell, checkpoint_key=KEY, cell_id=CELL, required=True, expected_final_epoch=4)
    real_producer = hooks.emit.producer_block("tests")          # the REAL producer block: the records must validate end to end
    written = []; _open_hook_gates(monkeypatch, tmp_path, written, fake_producer=False)
    hooks.emit_object_from_sidecar(str(cell), campaign="two_si", object_kind="relu_steer", paper_label="other", object_role="control",
                                   fit_target_kind="corpus_document_fit", fit_objective="ce", script="scripts/train_two_si_objects.py",
                                   registry_emit=True, spool=str(tmp_path / "spool"))
    objs = [p for k, p in written if k == "object"]; arts = [p for k, p in written if k == "artifact"]
    best, e2, e4 = (hashlib.sha256(b).hexdigest() for b in (b"best", b"e2", b"e4"))
    assert [p["object_id"] for p in objs] == [best, e2, e4], "ONE object row per distinct sha, the published object first"
    assert [a["artifact_id"] for a in arts] == [best, e2, e4]
    assert all(str((p.get("extra") or {}).get("parent_object_id")) != p["object_id"] for p in objs), "no object is its own parent"
    main = objs[0]; ec = main["extra"]["epoch_checkpoints"]; ep = ec["epochs"]
    assert list(ep) == ["epoch_01", "epoch_02", "epoch_03", "epoch_04"], "one membership entry per epoch, keyed by the epoch's unique key"
    assert ec["named"] == {"epoch_1": "epoch_01", "epoch_2": "epoch_02", "final": "epoch_04"}
    assert ep["epoch_01"]["object_id"] == best and ep["epoch_01"]["same_bytes_as"] == "published_object"
    assert ep["epoch_02"]["object_id"] == e2 and "same_bytes_as" not in ep["epoch_02"]
    assert ep["epoch_03"]["object_id"] == e2 and ep["epoch_03"]["same_bytes_as"] == f"epoch_object:{e2[:16]}" and ep["epoch_03"]["step"] == 267 * 3
    assert ep["epoch_04"]["object_id"] == e4 and ep["epoch_04"]["epoch"] == 4
    assert ec["same_bytes_as_published"] == [f"{CELL}@epoch_01"]
    assert str(cell / "object.pt") in arts[0]["locations_json"] and str(cell / "epoch_ckpts" / "epoch_01" / "object.pt") in arts[0]["locations_json"]
    assert arts[0]["uri_primary"] == str(cell / "object.pt"), "the published artifact keeps its primary location"
    assert str(cell / "epoch_ckpts" / "epoch_03" / "object.pt") in arts[1]["locations_json"], "a repeated epoch is a location of the earlier epoch's artifact"
    assert objs[1]["extra"]["parent_object_id"] == best and objs[1]["extra"]["epoch"] == 2
    # through the real reducer: every object_id folds from exactly one full event; the published object's columns are its own
    from sisteer.registry.reducer import fold
    by_id = {}
    for i, p in enumerate(objs):
        by_id.setdefault(p["object_id"], []).append(_as_full_event(i, p))
    assert all(len(evs) == 1 for evs in by_id.values())
    state, _ = fold("object", by_id[best])[-1]
    assert state["artifact_locations_json"][0] == str(cell / "object.pt") and json.loads(state["extra_json"]).get("parent_object_id") is None
    # the DEFECT, through the same reducer: a pre-fix second full record under the published object_id overwrites it
    bad = {**main, "artifact_locations_json": [str(cell / "epoch_ckpts" / "epoch_01" / "object.pt")],
           "extra": {**{k: v for k, v in main["extra"].items() if k != "epoch_checkpoints"}, "role": "epoch_checkpoint", "epoch": 1,
                     "parent_object_id": best}}
    overwritten, _ = fold("object", [_as_full_event(0, main), _as_full_event(1, bad)])[-1]
    assert overwritten["artifact_locations_json"] == bad["artifact_locations_json"] and json.loads(overwritten["extra_json"])["parent_object_id"] == best
    # ... and the validator refuses exactly that shape by name, while the fixed shapes validate
    from sisteer.registry import validate as VAL, records as R
    producer = real_producer
    counter = iter(range(1000))
    def _rec(rt, payload):
        return R.make_record(rt, payload, producer, record_id=f"test-{rt}-{next(counter)}")
    v = VAL.validate_record(_rec("object", bad)); assert type(v).__name__ == "Malformed" and v.field == "parent_object_id", v   # parent_object_id is a column now; the lineage refusal names the column
    # the pre-fix hook emitted the SIDECAR-built epoch records — build the exact old shapes with the real
    # object_from_sidecar over the epoch_01 dir (its bytes ARE the published object's): the old epoch artifact carries
    # extra {object_id: <its own sha>, sidecars} and NO role, under the epoch path, sharing the published artifact's key
    ep_dir = cell / "epoch_ckpts" / "epoch_01"
    old_obj, old_art = hooks.object_from_sidecar(str(ep_dir), campaign="two_si", object_kind="relu_steer", paper_label="other",
                                                 object_role="control", fit_target_kind="corpus_document_fit", fit_objective="ce",
                                                 producer=real_producer, extra={"role": "epoch_checkpoint", "epoch": 1, "parent_object_id": best})
    assert old_art["artifact_id"] == best and old_art["uri_primary"] == str(ep_dir / "object.pt") and set(old_art["extra"]) == {"object_id", "sidecars"}
    # through the real reducer: the old artifact record overwrites the published artifact's primary location (the locations list is the
    # fold's union on every artifact record since the analysis filing's fourth closing pass, so the published path stays a location --
    # the moved primary alone is the clobber the validator refuses below)
    clob, _ = fold("artifact", [_as_full_event(0, arts[0], "artifact"), _as_full_event(1, old_art, "artifact")])[-1]
    assert clob["uri_primary"] == str(ep_dir / "object.pt")
    assert json.loads(clob["locations_json"]) == [str(cell / "object.pt"), str(ep_dir / "object.pt")]
    # ... so the validator refuses exactly that record (and the old self-parent object) by name
    v = VAL.validate_record(_rec("artifact", old_art)); assert type(v).__name__ == "Malformed" and v.field == "uri_primary", v
    v = VAL.validate_record(_rec("object", old_obj)); assert type(v).__name__ == "Malformed" and v.field == "parent_object_id", v   # parent_object_id is a column now; the lineage refusal names the column
    # the harvester classifies every record through validate_record before any fold (the refusal is real at harvest)
    import inspect as _inspect
    from sisteer.registry import harvest as H
    assert "validate_record(" in _inspect.getsource(H._classify) if hasattr(H, "_classify") else "validate_record(" in _inspect.getsource(H)
    # the fixed shapes validate: the published artifact (cell root + alias location), the distinct epoch artifact (provenance marker)
    assert arts[1]["extra"]["parent_object_id"] == best and arts[1]["extra"]["role"] == "epoch_checkpoint"
    for rt, payload in (("object", main), ("object", objs[1]), ("artifact", arts[0]), ("artifact", arts[1])):
        v = VAL.validate_record(_rec(rt, payload)); assert type(v).__name__ == "Validated", (rt, getattr(v, "reason", v))


def test_validator_keeps_the_run_lifecycle_guard_beside_the_hook_fix2_rules():
    """The first cut of the artifact rule had displaced the pre-existing `run` guard — a full run record may only claim
    queued | running | generated; terminal states move through run_state / run_ingest. Pinned here so the two
    epoch-checkpoint refusals and the run guard coexist."""
    from sisteer.registry import validate as VAL, records as R, synth
    producer = hooks.emit.producer_block("tests")
    ok = synth.run_payload("d546-guard-key")                                   # a VALID full run record (the synth fixture's)
    v = VAL.validate_record(R.make_record("run", ok, producer, record_id="t-run-0"))
    assert type(v).__name__ == "Validated", getattr(v, "reason", v)
    for forbidden in ("failed", "withdrawn", "ingested"):
        bad = {**ok, "lifecycle_state": forbidden}
        v = VAL.validate_record(R.make_record("run", bad, producer, record_id=f"t-run-{forbidden}"))
        assert type(v).__name__ == "Malformed" and v.field == "lifecycle_state" and "run_state" in v.reason, (forbidden, v)


def test_hook_fix3_every_unnamed_epoch_keeps_its_own_membership_entry(tmp_path, monkeypatch):
    """A >4-epoch CE fit whose epochs 3, 4, 5 produce identical bytes used to collapse onto one `intermediate` role slot,
    losing epoch 3/4's identity (step, loss, epoch_id). Now every epoch keeps its own entry under its unique key, the
    named roles alias onto those keys, and same_bytes_as_published lists EVERY epoch that aliases the published object."""
    root = tmp_path / "ckpt" / "k"
    for ep, payload in ((1, b"a"), (2, b"b"), (3, b"plateau"), (4, b"plateau"), (5, b"plateau"), (6, b"plateau")):
        t2.save_epoch_checkpoint(root, _FakeObj(payload), arm="psr_ce_all", si_id="constv2-mix-full", epoch=ep, step=267 * ep,
                                 history=[{"loss": 2.0 - 0.1 * ep}], heldout_curve=[2.04] + [2.0 - 0.1 * e for e in range(1, ep + 1)],
                                 stopped=(ep == 6), epochs_cap=6, checkpoint_key=KEY)
    cell = tmp_path / CELL; cell.mkdir(); _cell_meta(cell, b"plateau")       # the published object IS the plateau bytes (epochs 3-6)
    t2.finalize_epoch_checkpoints(root, cell, checkpoint_key=KEY, cell_id=CELL, required=True, expected_final_epoch=6)
    written = []; _open_hook_gates(monkeypatch, tmp_path, written)
    hooks.emit_object_from_sidecar(str(cell), campaign="two_si", object_kind="relu_steer", paper_label="other", object_role="control",
                                   fit_target_kind="corpus_document_fit", fit_objective="ce", script="scripts/train_two_si_objects.py",
                                   registry_emit=True, spool=str(tmp_path / "spool"))
    objs = [p for k, p in written if k == "object"]; arts = [p for k, p in written if k == "artifact"]
    plateau, a, b = (hashlib.sha256(x).hexdigest() for x in (b"plateau", b"a", b"b"))
    assert [p["object_id"] for p in objs] == [plateau, a, b] and [x["artifact_id"] for x in arts] == [plateau, a, b]
    ec = objs[0]["extra"]["epoch_checkpoints"]; ep = ec["epochs"]
    assert list(ep) == [f"epoch_{i:02d}" for i in range(1, 7)]
    for i in (3, 4, 5, 6):
        e = ep[f"epoch_{i:02d}"]
        assert e["object_id"] == plateau and e["same_bytes_as"] == "published_object" and e["step"] == 267 * i, (i, e)
        assert e["heldout_loss"] == pytest.approx(2.0 - 0.1 * i) and e["epoch_id"] == f"{CELL}@epoch_{i:02d}"
    assert ec["named"] == {"epoch_1": "epoch_01", "epoch_2": "epoch_02", "final": "epoch_06"}
    assert ec["same_bytes_as_published"] == [f"{CELL}@epoch_{i:02d}" for i in (3, 4, 5, 6)]
    assert [str(cell / "epoch_ckpts" / f"epoch_{i:02d}" / "object.pt") in arts[0]["locations_json"] for i in (3, 4, 5, 6)] == [True] * 4
    # through the real reducer: the single canonical record folds with every epoch's identity intact
    from sisteer.registry.reducer import fold
    state, _ = fold("object", [_as_full_event(0, objs[0])])[-1]
    folded = json.loads(state["extra_json"])["epoch_checkpoints"]["epochs"]
    assert {k: v["step"] for k, v in folded.items()} == {f"epoch_{i:02d}": 267 * i for i in range(1, 7)}


def test_hook_fix5_weight_content_hash_sees_through_serialization(tmp_path):
    """hook-fix-5: the trainer's best-epoch restore gives the published object an epoch's EXACT weights but a
    different file sha (control cell 821352: bitwise-equal tensors, shas ea01c6c4… vs c8ffe0cc…). The canonical weight
    hash is the same across serializations and differs on any value, dtype or shape change."""
    torch = pytest.importorskip("torch")
    w = {"kind": "psr", "layers": [0, 1], "gate_w": {"0": torch.arange(6.).reshape(2, 3), "1": torch.ones(3)}, "directions": [torch.zeros(4), torch.full((2,), 2.0)]}
    a = tmp_path / "a.pt"; b = tmp_path / "b.pt"
    torch.save(w, a)
    # a different serialization of the SAME weights: reordered keys, a non-tensor field changed, storages re-materialised
    w2 = {"directions": [t.clone() for t in w["directions"]], "gate_w": {"1": w["gate_w"]["1"].clone(), "0": w["gate_w"]["0"].clone()},
          "layers": [0, 1], "kind": "psr", "note": "saved by another path"}
    torch.save(w2, b)
    assert hashlib.sha256(a.read_bytes()).hexdigest() != hashlib.sha256(b.read_bytes()).hexdigest(), "byte shas differ by construction"
    ha, hb = t2.weights_content_sha256(a), t2.weights_content_sha256(b)
    assert ha and ha == hb, "same tensors -> same weight hash regardless of serialization / non-tensor fields"
    for mut in ({"gate_w": {"0": w["gate_w"]["0"] + 1e-6}}, {"gate_w": {"0": w["gate_w"]["0"].to(torch.float64)}}, {"gate_w": {"0": w["gate_w"]["0"].reshape(3, 2)}}):
        c = tmp_path / "c.pt"; torch.save({**w, **{"gate_w": {**w["gate_w"], **mut["gate_w"]}}}, c)
        assert t2.weights_content_sha256(c) != ha, mut
    (tmp_path / "raw.bin").write_bytes(b"not a torch blob")
    assert t2.weights_content_sha256(tmp_path / "raw.bin") is None, "a non-torch file is a stated absence"
    # bf16 tensors hash too (numpy has no bf16: reinterpreted at the element width)
    d = tmp_path / "d.pt"; torch.save({"x": torch.ones(5, dtype=torch.bfloat16)}, d); assert t2.weights_content_sha256(d)


def test_hook_fix5_saver_records_the_weight_hash_and_the_hook_records_weight_equivalence(tmp_path, monkeypatch):
    """The saver stamps `weights_sha256` per epoch entry (a real torch blob); at emission the trainer hands the hook the
    weight identity of the published object + every epoch (`weights_identity`), and the canonical object's membership map
    gains `same_weights_as_published` beside `same_bytes_as_published` — the row count is unchanged (dedup stays by byte sha)."""
    torch = pytest.importorskip("torch")

    class _TorchObj:
        def __init__(self, seed): self.w = {"kind": "psr", "gate_w": {"0": torch.full((3,), float(seed))}}
        def save(self, path): torch.save(self.w, path)

    root = tmp_path / "ckpt" / "k"
    for ep, seed in ((1, 1.0), (2, 2.0), (3, 3.0)):
        t2.save_epoch_checkpoint(root, _TorchObj(seed), arm="psr_ce_all", si_id="constv2-mix-full", epoch=ep, step=267 * ep, history=[],
                                 heldout_curve=[2.0] * (ep + 1), stopped=(ep == 3), epochs_cap=3, checkpoint_key=KEY)
    idx = json.loads((root / t2.EPOCH_CKPT_INDEX).read_text())
    assert all(len(c["weights_sha256"]) == 64 for c in idx["checkpoints"]) and len({c["weights_sha256"] for c in idx["checkpoints"]}) == 3
    # the published object = the best epoch (2) RESTORED and re-serialized through another path (extra metadata -> different bytes)
    cell = tmp_path / CELL; cell.mkdir()
    torch.save({"kind": "psr", "gate_w": {"0": torch.full((3,), 2.0)}, "restored_from_epoch": 2}, cell / "object.pt")
    pub_bytes = (cell / "object.pt").read_bytes()
    assert hashlib.sha256(pub_bytes).hexdigest() != idx["checkpoints"][1]["artifact_sha256"], "different bytes by construction"
    meta = _cell_meta(cell, pub_bytes)
    t2.finalize_epoch_checkpoints(root, cell, checkpoint_key=KEY, cell_id=CELL, required=True, expected_final_epoch=3)
    # the identity map the trainer builds at publish: the published object + each epoch artefact (saver-recorded hash)
    widentity = {str(cell / "object.pt"): t2.weights_content_sha256(cell / "object.pt")}
    for c in json.loads((cell / t2.EPOCH_CKPT_INDEX).read_text())["checkpoints"]:
        widentity[str(cell / c["artifact"])] = c["weights_sha256"]
    written = []; _open_hook_gates(monkeypatch, tmp_path, written)
    hooks.emit_object_from_sidecar(str(cell), campaign="two_si", object_kind="relu_steer", paper_label="other", object_role="control",
                                   fit_target_kind="corpus_document_fit", fit_objective="ce", script="scripts/train_two_si_objects.py",
                                   registry_emit=True, spool=str(tmp_path / "spool"), weights_identity=widentity)
    objs = [p for k, p in written if k == "object"]
    assert len(objs) == 4, "row count unchanged: the published object + 3 distinct-byte epochs"
    ec = objs[0]["extra"]["epoch_checkpoints"]
    assert ec["same_bytes_as_published"] == [] and ec["same_weights_as_published"] == [f"{CELL}@epoch_02"]
    assert ec["epochs"]["epoch_02"]["same_weights_as"] == "published_object" and "same_weights_as" not in ec["epochs"]["epoch_01"]
    assert ec["published_weights_sha256"] == widentity[str(cell / "object.pt")] == objs[0]["extra"]["weights_sha256"]
    assert all(len(ec["epochs"][k]["weights_sha256"]) == 64 for k in ("epoch_01", "epoch_02", "epoch_03"))
    # without the identity map (an older trainer) the shape is additive-absent, never wrong
    written.clear()
    hooks.emit_object_from_sidecar(str(cell), campaign="two_si", object_kind="relu_steer", paper_label="other", object_role="control",
                                   fit_target_kind="corpus_document_fit", fit_objective="ce", script="scripts/train_two_si_objects.py",
                                   registry_emit=True, spool=str(tmp_path / "spool"))
    ec2 = [p for k, p in written if k == "object"][0]["extra"]["epoch_checkpoints"]
    assert ec2["same_weights_as_published"] == [] and ec2["published_weights_sha256"] is None and "weights_sha256" not in ec2["epochs"]["epoch_02"]
    # the trainer's publish path hands the identity ONLY to a hook that accepts it (an older hook never sees the kwarg)
    src = inspect.getsource(t2._registry_publish)
    assert '"weights_identity" in hook_params' in src and 'globals().get("weights_content_sha256")' in src, \
        "the publish path resolves the hash helper through globals() (the writers test execs the function in a tiny namespace)"


def test_hook_fix5_the_publish_path_hands_the_weight_identity_to_the_REAL_decorated_hook(tmp_path, monkeypatch):
    """`_registry_publish` feature-detects `weights_identity` via inspect.signature(hook), but the production hook is
    `@guarded` and the wrapper hid the signature as (*a, **kw) — the map was never passed and same_weights_as_published
    stayed [] in production. Now the decorator keeps the signature (functools.wraps) and this test drives
    `_registry_publish` through the REAL decorated hook with a captured writer: a best-epoch-restored object with
    different bytes gets its equivalence recorded."""
    torch = pytest.importorskip("torch")
    import inspect as _inspect
    assert "weights_identity" in _inspect.signature(hooks.emit_object_from_sidecar).parameters, "the guarded hook exposes its real signature"

    class _TorchObj:
        def __init__(self, seed): self.w = {"kind": "psr", "layers": [4, 8], "gate_w": {"4": torch.full((3,), float(seed))}}
        def save(self, path): torch.save(self.w, path)

    root = tmp_path / "ckpt" / "k"
    for ep, seed in ((1, 1.0), (2, 2.0), (3, 3.0)):
        t2.save_epoch_checkpoint(root, _TorchObj(seed), arm="psr_ce_all", si_id="constv2-mix-full", epoch=ep, step=267 * ep, history=[],
                                 heldout_curve=[2.0] * (ep + 1), stopped=(ep == 3), epochs_cap=3, checkpoint_key=KEY)
    cell = tmp_path / CELL; cell.mkdir()
    torch.save({"kind": "psr", "layers": [4, 8], "gate_w": {"4": torch.full((3,), 2.0)}, "restored_from_epoch": 2}, cell / "object.pt")
    _cell_meta(cell, (cell / "object.pt").read_bytes())
    t2.finalize_epoch_checkpoints(root, cell, checkpoint_key=KEY, cell_id=CELL, required=True, expected_final_epoch=3)
    written = []; _open_hook_gates(monkeypatch, tmp_path, written, fake_producer=False)
    t2._registry_publish(cell, si_id="constv2-mix-full", arm="psr_ce_all")          # the real hook, no emitter override
    ev = json.loads((cell / "registry_emission.json").read_text())
    assert ev["status"] == "emitted", ev
    objs = [p for k, p in written if k == "object"]
    assert len(objs) == 4 and objs[0]["object_id"] == hashlib.sha256((cell / "object.pt").read_bytes()).hexdigest()
    ec = objs[0]["extra"]["epoch_checkpoints"]
    assert ec["same_weights_as_published"] == [f"{CELL}@epoch_02"] and ec["same_bytes_as_published"] == []
    assert ec["epochs"]["epoch_02"]["same_weights_as"] == "published_object"
    assert ec["published_weights_sha256"] == objs[0]["extra"]["weights_sha256"] == t2.weights_content_sha256(cell / "object.pt")


def test_fix4_epoch_rows_and_artifacts_state_the_fits_attested_provenance_and_their_own_created_utc(tmp_path, monkeypatch):
    """A corrective re-emission of a cell with epoch checkpoints rebuilds every epoch object row and artifact too; the
    cell's fit_job_provenance.json reached the published object only, so the epoch rows would have restated the
    re-emission's job/node/sha/substrate and the epoch artifacts a fresh created_utc. Now: when the cell attests, every
    epoch dir must carry its OWN sidecar (the fit's attested fit_* plus THAT epoch's original created_utc, from its
    fit-job record) — refused BEFORE any write otherwise, in the dry run and the live call alike — and each epoch
    row/artifact states them."""
    from sisteer.registry import emit, hooks

    root = tmp_path / "ckpt" / "k"
    _save(root, b"e1", 1); _save(root, b"e2", 2); _save(root, b"e3", 3)
    cell = tmp_path / CELL; cell.mkdir()
    _cell_meta(cell, b"final-bytes")
    t2.finalize_epoch_checkpoints(root, cell, checkpoint_key=KEY, cell_id=CELL, required=True, expected_final_epoch=3)
    index = json.loads((cell / t2.EPOCH_CKPT_INDEX).read_text())
    epoch_dirs = [cell / os.path.dirname(c["artifact"]) for c in index["checkpoints"]]
    assert len(epoch_dirs) == 3 and all((d / "train_budget_meta.json").is_file() for d in epoch_dirs)
    monkeypatch.setenv("SLURM_JOB_ID", "828839")                                   # the RE-EMISSION's context
    monkeypatch.setattr(emit, "producer_block", lambda script, code_sha=None, session=None: {"script": script, "code_sha": "d" * 40, "host": "gpu-15", "substrate": "slurm", "session": None})
    kw = dict(campaign="sdfbase_full22", object_kind="relu_steer", paper_label="constitution_relu", object_role="constitution",
              fit_target_kind="corpus_document_fit", fit_objective="ce", fit_thinking_render="off", fit_corpus_id="constv2-mix-full", script="w")
    # no attestation anywhere: context everywhere (a first emission), epoch rows included
    recs = hooks.emit_object_from_sidecar(str(cell), dry_run=True, **kw)
    objs = [p for r, p in recs if r == "object"]
    assert len(objs) == 4 and all(o["fit_job_id"] == "828839" and o["fit_node_hostname"] == "gpu-15" for o in objs)
    # the cell attests the FIT's provenance but the epoch dirs carry nothing: refused before any record, dry run and live alike
    attested = {"fit_code_sha": "a" * 40, "fit_job_id": "821364", "fit_node_hostname": "gpu-23", "fit_substrate": "cluster_a",
                "artifact_created_utc": "2026-10-01T19:32:21.754Z", "sources": {"result": "r", "attempt_record": "a", "fit_job_record": "000001"},
                "note": "corrected", "written_by": "tests", "written_utc": "t"}
    (cell / hooks.FIT_PROVENANCE_SIDECAR).write_text(json.dumps(attested))
    assert hooks.emit_object_from_sidecar(str(cell), dry_run=True, **kw) is None          # the guard caught the ValueError: nothing planned
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    spool = tmp_path / "spool"
    monkeypatch.setenv(emit.SPOOL_ENV, str(spool))
    monkeypatch.setenv("SISTEER_REGISTRY_CLUSTER_LABEL", "cluster_a")
    assert hooks.emit_object_from_sidecar(str(cell), **kw) is None and not spool.exists()  # live: nothing written either
    # every epoch dir attests too — the fit's job/node/sha/substrate and ITS OWN original created_utc
    for i, d in enumerate(epoch_dirs, start=1):
        (d / hooks.FIT_PROVENANCE_SIDECAR).write_text(json.dumps(dict(attested, artifact_created_utc=f"2026-10-01T1{i}:00:00.000Z",
                                                                       sources={"fit_job_record": f"00000{2 * i + 1}-object.jsonl", "parent_attestation": "cell"})))
    recs = hooks.emit_object_from_sidecar(str(cell), dry_run=True, **kw)
    objs = {p["object_id"]: p for r, p in recs if r == "object"}
    arts = {p["artifact_id"]: p for r, p in recs if r == "artifact"}
    assert len(objs) == 4 and len(arts) >= 4
    for o in objs.values():
        assert (o["fit_code_sha"], o["fit_job_id"], o["fit_node_hostname"], o["fit_substrate"]) == ("a" * 40, "821364", "gpu-23", "cluster_a")
        assert "fit_provenance" in o["extra"]
    main_sha = hashlib.sha256(b"final-bytes").hexdigest()
    assert arts[main_sha]["created_utc"] == "2026-10-01T19:32:21.754Z"
    epoch_created = sorted(a["created_utc"] for aid, a in arts.items() if aid != main_sha and a["extra"].get("role") == "epoch_checkpoint")
    assert epoch_created == ["2026-10-01T11:00:00.000Z", "2026-10-01T12:00:00.000Z", "2026-10-01T13:00:00.000Z"]
    # the live call writes exactly those records
    paths = hooks.emit_object_from_sidecar(str(cell), **kw)
    assert len(paths) == len(recs)
