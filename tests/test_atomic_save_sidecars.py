"""An atomic save must publish its SIDECARS, not just the data file.

The defect this pins (2026-08-24): _save_object writes to
``object.safetensors.tmp`` and renames. SteeringVector.save puts its
sidecar at ``<given path>.with_suffix(".json")``, and with_suffix strips
only the LAST suffix — so the sidecar landed at
"object.safetensors.json" while the rename published only the data file,
leaving three wave artifacts that raise FileNotFoundError on load. The
predicate is "any save routed through a temp filename", not "vector
family": fit_class_b saves directly and its sidecar name is correct.

Tests assert the hazard's PRESENCE in the naive form as well as its
absence in the fixed one — a test that only checks the good path would
not notice a regression to a bare rename.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_two_si_objects as tt  # noqa: E402


class FakeSidecarObject:
    """Mimics SteeringVector.save: a data file plus a with_suffix('.json')
    sidecar derived from whatever path it is handed."""

    kind = "vector"
    fit_site = "o_proj_in"   # the saver refuses an unrecorded hook site

    def save(self, path):
        path = Path(path)
        path.write_bytes(b"DATA")
        path.with_suffix(".json").write_text(json.dumps({"method": "fake"}))


class FakePlainObject:
    kind = "lora"
    fit_site = "o_proj_in"

    def save(self, path):
        Path(path).write_bytes(b"BLOB")


def test_naive_atomic_rename_strands_the_sidecar(tmp_path):
    """The bug, reproduced directly — if this stops failing, the hazard is
    gone from the platform and this test should be revisited."""
    path = tmp_path / "object.safetensors"
    tmp = path.with_suffix(path.suffix + ".tmp")
    FakeSidecarObject().save(tmp)
    tmp.replace(path)                      # the OLD publish step
    assert (tmp_path / "object.safetensors.json").exists()
    assert not (tmp_path / "object.json").exists(), (
        "the naive rename must strand the sidecar, or this test is moot")


def _save(tmp_path, obj, name="object.safetensors"):
    """Drive the real _save_object publish step via a minimal cfg."""
    save_dir = tmp_path / "objects"
    save_dir.mkdir()
    tt._save_object(obj, save_dir, "si_x", "vector", {"completion_tokens": 8},
                    128, {"batch_size": 4}, model=None, tok=None)
    return save_dir


def test_save_object_publishes_the_sidecar_under_the_canonical_name(tmp_path):
    save_dir = _save(tmp_path, FakeSidecarObject())
    cell = save_dir / "si_x__vector"
    assert (cell / "object.safetensors").read_bytes() == b"DATA"
    assert (cell / "object.json").exists(), (
        "sidecar must be published as object.json — the name "
        "SteeringVector.load actually reads")
    assert json.loads((cell / "object.json").read_text())["method"] == "fake"
    assert not list(cell.glob("*.tmp")), "no temp files may survive"
    assert not (cell / "object.safetensors.json").exists(), (
        "the tmp-derived sidecar name must not survive publication")


def test_plain_object_save_unaffected(tmp_path):
    save_dir = _save(tmp_path, FakePlainObject())
    cell = save_dir / "si_x__vector"
    data = [f for f in cell.glob("object.*") if f.suffix != ".json"]
    assert len(data) == 1 and data[0].read_bytes() == b"BLOB", (
        "a single-file save must publish exactly its one data file")
    assert not list(cell.glob("*.tmp"))
    assert not (cell / "object.json").exists(), (
        "a save that writes no sidecar must not gain one")
