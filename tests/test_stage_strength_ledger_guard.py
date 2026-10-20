"""stage_strength.py refuses a ledger row whose `out` dir is gone instead of crashing the run.

A build-ledger row survives its cell dir: the dir gets moved aside (a superseded copy) or deleted, and the row
stays. The stager hashes every row's `out/object.pt` before anything else; an unguarded hash raised
FileNotFoundError on the first such row and killed the whole staging pass mid-write (found 2026-09-13 while
retiring the in-place-overwritten qwen d*it__ copies). The guard names the row and continues, exactly like the
existing sha-mismatch REFUSE, so one stale row costs one refusal, not the pass.

Synthetic ledger only (tmp_path; invented ids under the shared TESTHOST convention). No cluster is contacted:
both rows are refused before the remote probe, so the test needs no --dry-run stub of ssh."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STRENGTH = REPO / "scripts" / "sdfbase" / "full22" / "strength"


def _load():
    spec = importlib.util.spec_from_file_location("stage_strength", STRENGTH / "stage_strength.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(target: str, sid: str, out: Path, sha: str) -> dict:
    return {"target": target, "staged_id": sid, "cell_id": sid.split("__", 1)[1], "out": str(out), "object_sha256": sha,
            "parent_staged_id": "parent__" + sid.split("__", 1)[1], "parent_object_sha256": "a" * 64, "scale": 0.25,
            "producer": "scale_lora_dose.py", "object_type": "constitution LoRA r4", "scaling_method": "B x scale"}


def test_missing_out_path_is_a_named_refusal_not_an_exception(tmp_path, monkeypatch, capsys):
    ss = _load()
    monkeypatch.setattr(ss, "RECORDS", tmp_path / "_records")      # never touch the real records dir
    gone = tmp_path / "bank" / "web" / "cells" / "d0p25it64__cell_a"  # moved aside: no dir at all
    present = tmp_path / "bank" / "web" / "cells" / "d0p25it64__cell_b"
    present.mkdir(parents=True)
    (present / "object.pt").write_bytes(b"not the ledger's bytes")
    ledger = tmp_path / "build_ledger.json"
    ledger.write_text(json.dumps({"cells": [_row("web", "d0p25it64__cell_a", gone, "b" * 64),
                                            _row("web", "d0p25it64__cell_b", present, "c" * 64)]}), encoding="utf-8")
    rc = ss.main(["--ledger", str(ledger), "--dry-run", "--no-manifest", "--clusters", "cluster_a"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "REFUSE web/d0p25it64__cell_a: ledger row d0p25it64__cell_a: out path missing" in out
    assert "retire the row or restore the object" in out
    # the run went on to the next row and gave it the pre-existing sha-mismatch refusal
    assert "REFUSE web/d0p25it64__cell_b: local object.pt" in out
    assert not (tmp_path / "_records").exists() or not any((tmp_path / "_records").iterdir())


def test_a_committed_ledger_row_passes_the_stager(tmp_path, monkeypatch, capsys):
    """demo-import M7: 09_dose_ladder.sh --step build writes a COMMITTED dose copy's ledger row in the builder's own schema (source committed,
    built false, object_sha256 = the registry id, the sidecar's dose_scaled_from block) with `out` = the bank dir --step stage fills through
    stage_objects.py (object.pt + the identity sidecar + staged_from.json). The strength stager over that ledger reads the row like any built
    one — hashes `out/object.pt`, reads its staged_from.json, builds the record from the row's own fields — so a committed row passes and a
    built row beside it is still judged on its own. Synthetic bytes; the remote leg is stubbed (nothing is contacted)."""
    ss = _load()
    monkeypatch.setattr(ss, "RECORDS", tmp_path / "_records")
    payload = b"committed copy bytes " * 16
    oid = __import__("hashlib").sha256(payload).hexdigest()
    sid = "d0p75__demo-mix__lora_r4_minus_ctl"
    out = tmp_path / "bank" / "web" / "cells" / sid; out.mkdir(parents=True)
    (out / "object.pt").write_bytes(payload)
    (out / "derived_meta.json").write_text(json.dumps({"derived_id": sid.split("__", 1)[1], "object_sha256": oid, "dose_scaled_from": {"scale": 0.75}}))
    (out / "staged_from.json").write_text(json.dumps({"source": "committed:objects/web_base/" + oid, "pulled_at": "2026-01-01T00:00:00Z",
                                                      "recorded_hash_check": {"status": "match", "object_sha256_recorded_match": True}}))
    stamp = {"parent_staged_id": "derived__demo-mix__lora_r4_minus_ctl", "parent_object_sha256": "a" * 64, "scale": 0.75, "anchor": False,
             "producer": "scripts/sdfbase/scale_lora_dose.py", "object_type": "lora_subtraction", "scaling_method": "B x scale", "tensors_scaled": ["B_pos", "B_neg"],
             "written_at": "2026-09-17T13:54:59+00:00", "verification": {"checks": []}, "producer_commit": "b" * 40}
    committed = {"target": "web", "family": "committed", "cell_id": sid.split("__", 1)[1], "parent_lane": "derived", "parent_staged_id": stamp["parent_staged_id"],
                 "staged_id": sid, "view": "d0p75", "scale": 0.75, "anchor": False, "producer": stamp["producer"], "object_type": stamp["object_type"],
                 "parent_dir": None, "parent_origin": None, "parent_object_sha256": stamp["parent_object_sha256"], "object_sha256": oid,
                 "scaling_method": stamp["scaling_method"], "tensors_scaled": stamp["tensors_scaled"], "identity_sidecar": "derived_meta.json",
                 "verification": stamp["verification"], "out": str(out), "parent_consumed_state": None, "producer_commit": stamp["producer_commit"],
                 "built_at": stamp["written_at"], "source": "committed", "built": False, "committed_dir": "objects/web_base/" + oid, "dose_scaled_from": stamp}
    gone = tmp_path / "bank" / "web" / "cells" / "d0p75__cell_a"                               # a built row whose dir is gone: still its own named refusal
    ledger = tmp_path / "build_ledger.json"
    ledger.write_text(json.dumps({"cells": [committed, _row("web", "d0p75__cell_a", gone, "c" * 64)]}), encoding="utf-8")
    probed = []

    def fake_stage_one(cluster, local_dir, remote_dir, sha, n, dry):
        probed.append((cluster, str(local_dir), remote_dir, sha, n, dry))
        return {"action": "already_staged", "verified": True, "remote_sha256": sha, "remote_n_files": n, "bytes_moved": 0}
    monkeypatch.setattr(ss, "stage_one", fake_stage_one)
    # the cluster's block as the site config gives it: the registry label names it, the site's own label is the key
    monkeypatch.setattr(ss, "_SITE_BLOCKS", {"site_one": {"CLUSTER_SSH": "login-a", "CLUSTER_SHARED_ROOT": "/shared/a", "CLUSTER_SUBSTRATE": "cluster_a"}})
    monkeypatch.setattr(ss, "load_site", lambda site_dir: ss._SITE_BLOCKS)
    rc = ss.main(["--ledger", str(ledger), "--dry-run", "--no-manifest", "--clusters", "cluster_a"])
    text = capsys.readouterr().out
    assert rc == 2                                                                              # the built row's missing dir, as before
    assert "REFUSE web/d0p75__cell_a: ledger row d0p75__cell_a: out path missing" in text
    assert f"REFUSE web/{sid}" not in text and f"[cluster_a] web/{sid}: already_staged verified=True remote_sha={oid[:12]} n=3" in text
    assert probed == [("cluster_a", str(out), f"/shared/a/exp/objects/sdfbase/full22/web/cells/{sid}", oid, 3, True)]
    # the committed row ALONE: a clean pass
    ledger.write_text(json.dumps({"cells": [committed]}), encoding="utf-8")
    probed.clear()
    assert ss.main(["--ledger", str(ledger), "--dry-run", "--no-manifest", "--clusters", "cluster_a"]) == 0
    assert "REFUSE" not in capsys.readouterr().out and len(probed) == 1
