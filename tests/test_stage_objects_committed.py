"""scripts/sdfbase/full22/stage_objects.py's COMMITTED source and identity-sidecar readiness (demo-import M1). A `committed:<repo-relative dir>`
row is one of the paper's committed objects, copied from THIS checkout (no ssh pull) and held to its registry id; the records and the manifest
rows carry it under `source_path` like every other kind. Readiness: `object.pt` plus exactly ONE identity sidecar (train_budget_meta.json /
derived_meta.json / fit_provenance.json) that names the cell; a `kind: fit` cell still needs its COMPLETE marker; a PRESENT marker must be
coherent — which admits the committed refusal direction (fit_provenance.json + object.pt) and keeps every fit's COMPLETE check. The nine
committed rows reproduce/tools/import_objects.py writes stage through a one-cluster `local:` site; the refusals use synthetic directories."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402
from _rsync_fallback import rsync_on_path  # noqa: E402,F401  (autouse: the fallback rsync on PATH where the host has none; a no-op where it has one)
from test_stage_objects_registry_id import _site  # noqa: E402

sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))
sys.path.insert(0, str(REPO / "src"))
import stage_objects as SO  # noqa: E402

KEY = "qwen35_4b_demo"
TOOL = REPO / "reproduce" / "tools" / "import_objects.py"


def _stage(tmp_path: Path, cells: list, site: Path, name: str = "bank") -> tuple:
    bank = tmp_path / name
    cf = tmp_path / f"{name}_cells.json"; cf.write_text(json.dumps(cells))
    r = subprocess.run([sys.executable, str(REPO / "scripts/sdfbase/full22/stage_objects.py"), "--site-dir", str(site), "--bank", str(bank), "--target", KEY,
                        "--cells", str(cf), "--clusters", "cluster_a", "--no-manifest"], capture_output=True, text=True, timeout=600)
    return r.returncode, r.stdout + r.stderr, bank


def test_the_nine_committed_rows_stage_from_the_checkout_and_the_records_carry_the_source_under_source_path(tmp_path):
    site, shared = _site(tmp_path)
    cells = json.loads(subprocess.run([sys.executable, str(TOOL), "cells", "--model", KEY], capture_output=True, text=True, check=True).stdout)
    assert len(cells) == 9 and all(c["source"].startswith("committed:objects/") for c in cells)
    rc, out, bank = _stage(tmp_path, cells, site)
    assert rc == 0, out
    assert "NOT READY" not in out and "REFUSED" not in out and "PULL FAILED" not in out and "ssh" not in out.lower().replace("local:", "")
    recs = SO.object_records(bank / "_records")
    assert len(recs) == 9 and all(r["status"] == "staged" for r in recs), [(r["staged_id"], r["status"]) for r in recs]
    for c, r in zip(sorted(cells, key=lambda c: f"{c['lane']}__{c['cell_id']}"), sorted(recs, key=lambda r: r["staged_id"])):
        assert r["staged_id"] == f"{c['lane']}__{c['cell_id']}" and r["source_path"] == c["source"] and "source" not in r
        assert r["object_sha256"] == c["object_sha256_expected"] and r["object_sha256_expected_match"] is True and r["cluster_a_verified"] is True
        assert (shared / "exp/objects/sdfbase/full22" / KEY / "cells" / r["staged_id"] / "object.pt").is_file()
        assert (shared / "exp/objects/sdfbase/full22" / KEY / "cells" / r["staged_id"] / c["identity_sidecar"]).is_file()
    # the refusal direction (fit_provenance.json + object.pt, no COMPLETE) is ready as a kind: derived cell; the fits carried their COMPLETE
    ar = next(r for r in recs if r["lane"] == "arditi")
    assert (Path(ar["local_path"]) / "fit_provenance.json").is_file() and not (Path(ar["local_path"]) / "COMPLETE").exists()
    assert all((Path(r["local_path"]) / "COMPLETE").is_file() for r in recs if r["lane"] == "base")
    # idempotent: a second run finds every remote copy already staged and moves nothing
    rc, out, _bank = _stage(tmp_path, cells, site)
    assert rc == 0 and out.count("already_staged verified=True") == 9, out


def _synthetic(tmp_path: Path, name: str, files: dict) -> Path:
    d = tmp_path / "src" / name; d.mkdir(parents=True, exist_ok=True)
    for fname, content in files.items():
        if isinstance(content, bytes):
            (d / fname).write_bytes(content)
        else:
            (d / fname).write_text(json.dumps(content))
    return d


def test_readiness_needs_exactly_one_identity_sidecar_that_names_the_cell_and_a_coherent_marker(tmp_path):
    site, _shared = _site(tmp_path)
    payload = b"synthetic bytes " * 32
    sha = hashlib.sha256(payload).hexdigest()
    cases = {
        "two_sidecars": ({"object.pt": payload, "derived_meta.json": {"derived_id": "two_sidecars", "object_sha256": sha}, "fit_provenance.json": {"staged_id": "two_sidecars"}},
                         "derived", "exactly one identity sidecar"),
        "no_sidecar": ({"object.pt": payload}, "derived", "found none"),
        "names_another_cell": ({"object.pt": payload, "derived_meta.json": {"derived_id": "someone_else", "object_sha256": sha}}, "derived", "derived_meta.json does not name the cell 'names_another_cell'"),
        "fit_without_complete": ({"object.pt": payload, "train_budget_meta.json": {"cell_id": "fit_without_complete", "object_sha256": sha}}, "fit", "needs its COMPLETE marker"),
        "torn_marker": ({"object.pt": payload, "train_budget_meta.json": {"cell_id": "torn_marker", "object_sha256": sha},
                         "COMPLETE": {"cell_id": "torn_marker", "files": {"object.pt": sha, "metrics.jsonl": "0" * 64}}}, "fit", "COMPLETE names metrics.jsonl, which is absent"),
        "wrong_sha_marker": ({"object.pt": payload, "train_budget_meta.json": {"cell_id": "wrong_sha_marker", "object_sha256": sha},
                              "COMPLETE": {"cell_id": "wrong_sha_marker", "files": {"object.pt": "0" * 64}}}, "fit", f"COMPLETE records object.pt as {'0' * 12} but it hashes to {sha[:12]}"),
    }
    cells = []
    for name, (files, kind, _why) in cases.items():
        d = _synthetic(tmp_path, name, files)
        cells.append({"cell_id": name, "lane": "derived" if kind == "derived" else "base", "kind": kind, "source": f"local:{d}"})
    # ... and the shapes that ARE ready: a fit with a coherent marker, a derived dir, a fit_provenance dir, a fit_provenance dir whose cell_id is the harness id
    good = {
        "good_fit": ({"object.pt": payload, "train_budget_meta.json": {"cell_id": "good_fit", "object_sha256": sha},
                      "COMPLETE": {"cell_id": "good_fit", "files": {"object.pt": sha, "train_budget_meta.json": hashlib.sha256(json.dumps({"cell_id": "good_fit", "object_sha256": sha}).encode()).hexdigest()}}}, "fit"),
        "good_derived": ({"object.pt": payload, "derived_meta.json": {"derived_id": "good_derived", "object_sha256": sha}}, "derived"),
        "ar_c0p3__good_prov": ({"object.pt": payload, "fit_provenance.json": {"cell_id": "good_prov", "staged_id": "ar_c0p3__good_prov", "object_sha256": sha}}, "derived"),
        "note_marker": ({"object.pt": payload, "derived_meta.json": {"derived_id": "note_marker", "object_sha256": sha}, "COMPLETE": b"promoted 2026-10-10\n"}, "derived"),
    }
    for name, (files, kind) in good.items():
        d = _synthetic(tmp_path, name, files)
        cells.append({"cell_id": name, "lane": "derived" if kind == "derived" else "base", "kind": kind, "source": f"local:{d}"})
    rc, out, bank = _stage(tmp_path, cells, site)
    assert rc == 2, out
    recs = {r["cell_id"]: r for r in SO.local_records(bank / "_records")}
    for name, (_files, _kind, why) in cases.items():
        assert recs[name]["status"] == "not_ready" and why in recs[name]["reason"], (name, recs[name])
        assert f"NOT READY ({recs[name]['reason']})" in out
    for name in good:
        assert recs[name]["status"] == "staged", (name, recs[name])
    # the helper's own answers, and the sidecar names this script and the registry's reader hold equal
    assert SO.readiness_problem(tmp_path / "src" / "good_derived", "good_derived", "derived") is None
    assert SO.readiness_problem(tmp_path / "src" / "good_derived", "good_derived", "fit") == "a kind: fit cell needs its COMPLETE marker (found derived_meta.json, no COMPLETE)"
    assert SO.readiness_problem(tmp_path / "src" / "nowhere", "x", "fit") == "no object.pt"
    from sisteer.eval.train_budget import IDENTITY_SIDECARS
    assert SO.IDENTITY_SIDECARS == tuple(IDENTITY_SIDECARS)


def test_a_committed_source_outside_the_checkouts_objects_refuses_by_name_before_any_copy(tmp_path):
    site, shared = _site(tmp_path)
    for src in ("committed:/etc", "committed:objects/../reproduce", "committed:objects/nope/" + "0" * 64):
        rc, out, bank = _stage(tmp_path, [{"cell_id": "x", "lane": "derived", "kind": "derived", "source": src}], site, "bank_" + hashlib.sha256(src.encode()).hexdigest()[:6])
        assert rc != 0 and "REFUSED: source" in out and "a committed source is a repo-relative object directory of this checkout" in out, (src, out)
        assert not (shared / "exp/objects/sdfbase/full22" / KEY / "cells" / "derived__x").exists()
