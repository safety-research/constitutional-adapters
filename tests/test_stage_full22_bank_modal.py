"""scripts/modal/stage_full22_bank_modal.py — the generalised-by-model-key full22 bank stager, over a SYNTHETIC tree.

A tiny roster (one target, one view: two fitted cells + their LoRA-subtraction difference + a prompted baseline), a pins file, and
a mirror of small fake ``object.pt`` bytes with sidecars carrying the matching shas (plus the extra files a real mirror dir holds —
epoch checkpoints, COMPLETE, RESULT — which are never staged). The Modal CLI is doubled by a runner that keeps an in-memory volume and
records every call; the registry by a throwaway ``object`` table. Synthetic names only (``TESTORG/model``, placeholder uuids).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "modal" / "stage_full22_bank_modal.py"

MK = "testmodel_7b"
FIT_MK = "testmodel_7b_base"
VIEW = "s1"
CTL = "ctlsi-full__lora_r4_ce"
POS = "possi-full__lora_r4_ce"
DIFF = "possi-full__lora_r4_ce_minus_ctl"
TREE_SHA = "1111111122222222333333334444444455555555"


@pytest.fixture(scope="module")
def M():
    name = "_stage_full22_bank_under_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod          # dataclasses under `from __future__ import annotations` resolve through sys.modules
    spec.loader.exec_module(mod)
    return mod


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# --------------------------------------------------------------------------------------------------------------------------- #
# the synthetic world                                                                                                           #
# --------------------------------------------------------------------------------------------------------------------------- #

def _write_json(p: Path, doc) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def make_world(root: Path, *, pin_override: dict | None = None, sidecar_sha_override: dict | None = None,
               parent_sha_override: str | None = None, model_key: str = MK) -> dict:
    """``{tree, source, objects: {staged_id: bytes}}``. Objects are distinct fake bytes; every sidecar records its object's sha."""
    tree, source = root / "tree", root / "mirror" / "cells"
    objects = {f"base__{CTL}": b"CTL-OBJECT-BYTES-" + b"\x00" * 40, f"base__{POS}": b"POS-OBJECT-BYTES-" + b"\x01" * 48,
               f"derived__{DIFF}": b"DIFF-OBJECT-BYTES-" + b"\x02" * 64}
    shas = {sid: sha(b) for sid, b in objects.items()}
    recorded = dict(shas)
    recorded.update(sidecar_sha_override or {})
    # the mirror: object.pt + sidecar + the clutter a real fit dir carries (never staged)
    tbm_sha = {}
    for cell, sid in ((CTL, f"base__{CTL}"), (POS, f"base__{POS}")):
        d = source / sid
        d.mkdir(parents=True)
        (d / "object.pt").write_bytes(objects[sid])
        _write_json(d / "train_budget_meta.json", {"cell_id": cell, "object_sha256": recorded[sid], "arm": "lora_r4_ce", "si_id": cell.split("__")[0],
                                                   "model_id": "TESTORG/model-base", "training_budget_kind": "document_tokens"})
        tbm_sha[cell] = sha((d / "train_budget_meta.json").read_bytes())
        (d / "metrics.jsonl").write_text('{"step": 1, "loss": 1.0}\n', encoding="utf-8")
        (d / "RESULT").write_text("COMPLETED 0 00:10:00\n")
        (d / "fit_selection.json").write_text("{}\n")
        (d / "epoch_ckpts" / "epoch_01").mkdir(parents=True)
        (d / "epoch_ckpts" / "epoch_01" / "object.pt").write_bytes(b"epoch1")
        _write_json(d / "COMPLETE", {"cell_id": cell, "object_sha256": recorded[sid],
                                     "files": {"object.pt": shas[sid], "train_budget_meta.json": tbm_sha[cell], "epoch_ckpts/epoch_01/object.pt": sha(b"epoch1")}})
    d = source / f"derived__{DIFF}"
    d.mkdir(parents=True)
    (d / "object.pt").write_bytes(objects[f"derived__{DIFF}"])
    pos_sha = parent_sha_override or shas[f"base__{POS}"]
    _write_json(d / "derived_meta.json", {"kind": "lora_subtraction", "derived_id": DIFF, "model_key": FIT_MK, "object_sha256": recorded[f"derived__{DIFF}"],
                                          "parent_pos": {"cell_id": POS, "model_key": FIT_MK, "object_sha256": pos_sha, "budget_meta_sha256": tbm_sha[POS],
                                                         "model_id": "TESTORG/model-base", "model_revision": "0" * 40},
                                          "parent_neg": {"cell_id": CTL, "model_key": FIT_MK, "object_sha256": shas[f"base__{CTL}"], "budget_meta_sha256": tbm_sha[CTL],
                                                         "model_id": "TESTORG/model-base", "model_revision": "0" * 40}})
    (d / "ADMISSION.json").write_text("{}\n")
    # the tree: roster + pins (+ no f22_roster.py, so the stager falls back to this checkout's reader)
    roster = f"""wave: full22
targets:
  {model_key}:
    model: TESTORG/model
    model_revision: {"a" * 40}
    views:
      {VIEW}:
        config: configs/sdfbase_full22_{model_key}_{VIEW}.yaml
        cells:
          {CTL}: base__{CTL}
          {POS}: base__{POS}
          {DIFF}: derived__{DIFF}
          prompted__safety_refusal: null
"""
    (tree / "configs").mkdir(parents=True)
    (tree / "configs" / "sdfbase_full22_roster.yaml").write_text(roster, encoding="utf-8")
    pins = {sid: {"object_sha256": s, "source": "synthetic"} for sid, s in shas.items()}
    for sid, s in (pin_override or {}).items():
        pins[sid]["object_sha256"] = s
    _write_json(tree / "configs" / "sdfbase_full22_pins.json", {model_key: pins})
    return {"tree": tree, "source": source, "objects": objects, "shas": shas, "tbm_sha": tbm_sha}


class FakeModal:
    """A ``subprocess.run``-shaped double of the Modal CLI over an in-memory volume ``{remote path (no leading slash): bytes}``."""

    def __init__(self, remote: dict[str, bytes] | None = None, *, probe_pct: float = 42.0, probe_floor_hit: bool = False):
        self.remote = dict(remote or {})
        self.calls: list[list[str]] = []
        self.probe_pct, self.probe_floor_hit = probe_pct, probe_floor_hit

    def __call__(self, argv, **kw):
        argv = list(argv)
        self.calls.append(argv)
        sub = argv[1:]
        if sub[:2] == ["volume", "ls"]:
            js = "--json" in sub
            rel = sub[-1]
            hits = {k: v for k, v in self.remote.items() if k == rel or k.startswith(rel.rstrip("/") + "/")}
            if not hits:
                return subprocess.CompletedProcess(argv, 1, "", "No such file or directory")
            if not js:
                return subprocess.CompletedProcess(argv, 0, "\n".join(hits), "")
            entries, seen = [], set()
            for k, v in hits.items():
                rest = k[len(rel.rstrip("/")) + 1:] if k != rel else ""
                if "/" in rest:
                    top = rest.split("/", 1)[0]
                    if top not in seen:
                        seen.add(top)
                        entries.append({"filename": f"{rel.rstrip('/')}/{top}", "type": "dir", "size": "", "created_modified": "x"})
                else:
                    entries.append({"filename": k, "type": "file", "size": f"{len(v)} B", "created_modified": "x"})
            return subprocess.CompletedProcess(argv, 0, json.dumps(entries), "")
        if sub[:2] == ["volume", "get"]:
            remote_abs, local_dir = sub[-2], Path(sub[-1])
            rel = remote_abs.lstrip("/")
            if rel not in self.remote:
                return subprocess.CompletedProcess(argv, 1, "", "No such file or directory")
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / Path(rel).name).write_bytes(self.remote[rel])
            return subprocess.CompletedProcess(argv, 0, "", "")
        if sub[:2] == ["volume", "put"]:
            local, remote_abs = Path(sub[-2]), sub[-1]
            rel = remote_abs.lstrip("/")
            if rel in self.remote and "--force" not in sub:
                return subprocess.CompletedProcess(argv, 1, "", "already exists")
            self.remote[rel] = local.read_bytes()
            return subprocess.CompletedProcess(argv, 0, "", "")
        if sub[:1] == ["run"]:
            rec = {"volume": "twosi-fits", "inodes_total": 500000, "inodes_used": int(5000 * self.probe_pct), "pct_used": self.probe_pct}
            return subprocess.CompletedProcess(argv, 3 if self.probe_floor_hit else 0, json.dumps(rec) + "\n", "")
        raise AssertionError(f"unexpected modal invocation {argv}")

    def puts(self) -> list[str]:
        return [c[-1] for c in self.calls if c[1:3] == ["volume", "put"]]


def make_registry(root: Path, rows: list[dict]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(root / "registry.sqlite")
    conn.execute("CREATE TABLE object (object_id TEXT PRIMARY KEY, object_kind TEXT, campaign TEXT, campaign_cell_id_legacy TEXT, artifact_sha256 TEXT, "
                 "artifact_bytes INTEGER, status TEXT, derivation TEXT, parent_object_ids_json TEXT, fit_model_id TEXT)")
    for r in rows:
        conn.execute("INSERT INTO object VALUES (:object_id, :object_kind, :campaign, :campaign_cell_id_legacy, :artifact_sha256, :artifact_bytes, "
                     ":status, :derivation, :parent_object_ids_json, :fit_model_id)",
                     {"object_kind": "lora", "campaign": "sdfbase_full22", "status": "reference", "derivation": "none",
                      "parent_object_ids_json": None, "fit_model_id": FIT_MK, **r})
    conn.commit()
    conn.close()
    return root


def registry_rows(world: dict) -> list[dict]:
    s, o = world["shas"], world["objects"]
    return [{"object_id": s[f"base__{CTL}"], "campaign_cell_id_legacy": f"base__{CTL}", "artifact_sha256": s[f"base__{CTL}"], "artifact_bytes": len(o[f"base__{CTL}"])},
            {"object_id": s[f"base__{POS}"], "campaign_cell_id_legacy": f"base__{POS}", "artifact_sha256": s[f"base__{POS}"], "artifact_bytes": len(o[f"base__{POS}"])},
            {"object_id": s[f"derived__{DIFF}"], "campaign_cell_id_legacy": f"derived__{DIFF}", "artifact_sha256": s[f"derived__{DIFF}"],
             "artifact_bytes": len(o[f"derived__{DIFF}"]), "object_kind": "lora_subtraction", "derivation": "lora_subtraction",
             "parent_object_ids_json": json.dumps([s[f"base__{POS}"], s[f"base__{CTL}"]])}]


def run_kw(M, world, tmp_path, **over):
    kw = dict(tree=world["tree"], model_key=MK, view=VIEW, source=world["source"], receipts_dir=tmp_path / "receipts", go=False, parents=False,
              tree_sha=TREE_SHA, tree_info={"config_dirty": [], "tree_on_remote": True}, modal_bin=Path("/nonexistent/modal"),
              stage_dir=tmp_path / "stage", registry_root=str(tmp_path / "no-registry"), log=lambda s: None)
    kw.update(over)
    return kw


# --------------------------------------------------------------------------------------------------------------------------- #
# the plan                                                                                                                      #
# --------------------------------------------------------------------------------------------------------------------------- #

def test_plan_cells_staged_ids_sidecars_destinations_and_shas(M, tmp_path):
    w = make_world(tmp_path)
    plan = M.plan_bank(w["tree"], MK, VIEW, w["source"], tree_sha=TREE_SHA)
    assert [c.cell_id for c in plan.cells] == [CTL, POS, DIFF]
    assert [c.staged_id for c in plan.cells] == [f"base__{CTL}", f"base__{POS}", f"derived__{DIFF}"]
    assert [c.sidecar for c in plan.cells] == ["train_budget_meta.json", "train_budget_meta.json", "derived_meta.json"]
    assert [c.lane_kind for c in plan.cells] == ["base", "base", "derived"]
    assert plan.remote_view_root == f"/objects/sdfbase/full22/{MK}/{VIEW}"
    for c in plan.cells:
        assert c.remote_dir == f"{plan.remote_view_root}/{c.cell_id}"          # the BARE cell id, never the staged id
        assert c.object_sha256 == w["shas"][c.staged_id]
        assert [f.remote for f in c.files] == [f"{c.remote_dir}/object.pt", f"{c.remote_dir}/{c.sidecar}"]
    assert plan.prompted_cells_not_staged == ["prompted__safety_refusal"]
    assert plan.pins_compared == w["shas"]
    diff = plan.cells[2]
    assert {p.role: (p.cell_id, p.object_sha256, p.staged_id) for p in diff.parents} == {
        "parent_pos": (POS, w["shas"][f"base__{POS}"], f"base__{POS}"), "parent_neg": (CTL, w["shas"][f"base__{CTL}"], f"base__{CTL}")}


def test_exactly_two_files_per_object_despite_the_mirror_clutter(M, tmp_path):
    w = make_world(tmp_path)
    plan = M.plan_bank(w["tree"], MK, VIEW, w["source"], tree_sha=TREE_SHA)
    for c in plan.cells:
        assert len(c.files) == 2 and [f.name for f in c.files] == ["object.pt", c.sidecar]
        assert len(list(c.source_dir.iterdir())) > 2       # the mirror holds more; none of it is planned
    assert len(plan.files) == 6


def test_refuses_a_pin_mismatch(M, tmp_path):
    w = make_world(tmp_path, pin_override={f"base__{CTL}": "f" * 64})
    with pytest.raises(M.Refusal, match="source object sha .* != pin"):
        M.plan_bank(w["tree"], MK, VIEW, w["source"], tree_sha=TREE_SHA)


def test_refuses_a_sidecar_that_disagrees_with_the_object(M, tmp_path):
    w = make_world(tmp_path, sidecar_sha_override={f"base__{POS}": "e" * 64})
    with pytest.raises(M.Refusal, match="records object_sha256 .* != pin .* stale"):
        M.plan_bank(w["tree"], MK, VIEW, w["source"], tree_sha=TREE_SHA)


def test_refuses_a_derived_parent_sha_mismatch(M, tmp_path):
    w = make_world(tmp_path, parent_sha_override="d" * 64)
    with pytest.raises(M.Refusal, match=f"parent {POS} names sha .* but .*base__{POS}.* pins"):
        M.plan_bank(w["tree"], MK, VIEW, w["source"], tree_sha=TREE_SHA)


def test_refuses_an_ambiguous_identity_a_missing_source_and_an_unknown_lane(M, tmp_path):
    w = make_world(tmp_path)
    (w["source"] / f"base__{CTL}" / "derived_meta.json").write_text("{}")
    with pytest.raises(M.Refusal, match="identity sidecars present .* ambiguous"):
        M.plan_bank(w["tree"], MK, VIEW, w["source"], tree_sha=TREE_SHA)
    (w["source"] / f"base__{CTL}" / "derived_meta.json").unlink()
    with pytest.raises(M.Refusal, match="source dir .* is missing"):
        M.plan_bank(w["tree"], MK, VIEW, tmp_path / "elsewhere", tree_sha=TREE_SHA)
    assert M.sidecar_for("base64s2__x__lora_r4") == ("base", "train_budget_meta.json")
    assert M.sidecar_for("derived_base64s2__x_minus_ctl") == ("derived", "derived_meta.json")
    assert M.sidecar_for("paired_eb64s1__x_minus_ctl") == ("paired", "derived_meta.json")
    assert M.sidecar_for("ar_c0p3__arditi_allpos") == ("arditi", "fit_provenance.json")
    # paper-companion-repo dose-1.0 random-init ReLU pair: the LeCun lane of the s2rl view stages as-is —
    # `lecun__<cell>` is a fit (train_budget_meta.json), `lecun_derived__<cell>` the composition of two fits (derived_meta.json)
    assert M.sidecar_for("lecun__x__psr_ce_all") == ("base", "train_budget_meta.json")
    assert M.sidecar_for("lecun_derived__x_minus_ctl") == ("derived", "derived_meta.json")
    assert M._derived_lane("lecun_derived") and not M._derived_lane("lecun")
    with pytest.raises(M.Refusal, match="maps to no identity sidecar"):
        M.sidecar_for("si12__safety_refusal__psr_kl_all")
    with pytest.raises(M.Refusal, match="maps to no identity sidecar"):
        M.sidecar_for("d0p75__x_minus_ctl")                       # a dose copy is only knowable from its source dir
    with pytest.raises(M.Refusal, match="no 'dose_scaled_from' block"):
        M.sidecar_for("d0p75__x_minus_ctl", w["source"] / f"derived__{DIFF}")


def test_refuses_an_unknown_view_or_model_key_by_name(M, tmp_path):
    w = make_world(tmp_path)
    with pytest.raises(M.Refusal, match="no \\(target, view\\)"):
        M.plan_bank(w["tree"], MK, "s9", w["source"], tree_sha=TREE_SHA)
    with pytest.raises(M.Refusal, match="no \\(target, view\\)"):
        M.plan_bank(w["tree"], "othermodel", VIEW, w["source"], tree_sha=TREE_SHA)


# --------------------------------------------------------------------------------------------------------------------------- #
# the manifest and the parents                                                                                                  #
# --------------------------------------------------------------------------------------------------------------------------- #

def test_manifest_content_matches_the_staged_schema(M, tmp_path):
    w = make_world(tmp_path)
    plan = M.plan_bank(w["tree"], MK, VIEW, w["source"], tree_sha=TREE_SHA)
    verified = {f.remote: {"verified_by": "get+sha256 and ls names"} for f in plan.files}
    man = M.build_manifest(plan, verified=verified, dry=False, written_at="2026-01-01T00:00:00Z")
    assert man["schema"] == "tinycap-full22-staged-objects/1" and man["wave"] == "full22"
    assert (man["model_key"], man["view"], man["tree_sha"], man["staged_by"]) == (MK, VIEW, TREE_SHA, M.STAGED_BY)
    assert man["volume"] == "twosi-fits" and man["volume_path"] == f"/objects/sdfbase/full22/{MK}/{VIEW}"
    assert man["objects_dir"] == f"/vol/objects/sdfbase/full22/{MK}/{VIEW}" and man["manifest_name"] == "staged_manifest.json"
    assert man["roster"] == "configs/sdfbase_full22_roster.yaml" and len(man["roster_sha256"]) == 64 and len(man["pins_sha256"]) == 64
    assert man["remote_verified"] == {"method": "modal volume ls --json names per cell dir; modal volume get + sha256 per file", "n_files": 6}
    assert man["prompted_cells_not_staged"] == ["prompted__safety_refusal"] and man["written_at"] == "2026-01-01T00:00:00Z"
    assert list(man["cells"]) == [CTL, POS, DIFF]
    c = man["cells"][DIFF]
    assert c["cell_id"] == c["cell"] == DIFF and c["staged_id"] == f"derived__{DIFF}" and c["lane_kind"] == "derived"
    assert c["object_sha256"] == w["shas"][f"derived__{DIFF}"] and c["bytes"] == len(w["objects"][f"derived__{DIFF}"])
    assert c["sidecar"] == c["identity_sidecar"] == "derived_meta.json" and len(c["sidecar_sha256"]) == 64 and c["pin_match"] is True
    assert c["n_files"] == 2 and [f["path"] for f in c["files"]] == [f"{DIFF}/object.pt", f"{DIFF}/derived_meta.json"]
    assert c["volume_path"] == f"twosi-fits:/objects/sdfbase/full22/{MK}/{VIEW}/{DIFF}" and c["objects_dir_entry"] == f"/vol/objects/sdfbase/full22/{MK}/{VIEW}/{DIFF}"
    assert set(c["parents"]) == {"parent_pos", "parent_neg"} and c["parents"]["parent_neg"]["staged_id"] == f"base__{CTL}"
    assert "parents" not in man["cells"][CTL]
    # the content comparison ignores only the timestamp
    assert M.manifest_content(man) == M.manifest_content({**man, "written_at": "later"})


def test_parents_are_staged_without_a_complete_marker(M, tmp_path):
    w = make_world(tmp_path)
    plan = M.plan_bank(w["tree"], MK, VIEW, w["source"], tree_sha=TREE_SHA)
    pins = M.load_pins(w["tree"], MK)
    pp = M.plan_parents(plan, pins=pins, stage_dir=tmp_path / "stage", written_at="2026-01-01T00:00:00Z")
    assert sorted(p.cell_id for p in pp.parents) == sorted([CTL, POS])
    for p in pp.parents:
        assert p.fit_model_key == FIT_MK and p.remote_dir == f"/out/{FIT_MK}/cells_all/{p.cell_id}"
        names = [f.name for f in p.files]
        assert names == ["object.pt", "train_budget_meta.json", "metrics.jsonl", "staged_from.json"]
        assert "COMPLETE" not in names and (p.source_dir / "COMPLETE").is_file()      # the mirror HAS one; it is not staged
        assert p.consumers == [DIFF] and p.object_sha256 == w["shas"][p.staged_id] and p.budget_meta_sha256 == w["tbm_sha"][p.cell_id]
        sf = json.loads([f for f in p.files if f.name == "staged_from.json"][0].local.read_text())
        assert sf == p.staged_from and sf["object_sha256"] == p.object_sha256 and sf["tree_sha"] == TREE_SHA
        assert sf["source"].startswith("grading_host:") and "complete_marker" in sf and set(sf["files"]) == {"object.pt", "train_budget_meta.json", "metrics.jsonl"}


def test_parents_refuse_a_budget_sidecar_that_changed_under_the_derivation(M, tmp_path):
    w = make_world(tmp_path)
    tbm = w["source"] / f"base__{CTL}" / "train_budget_meta.json"
    doc = json.loads(tbm.read_text())
    doc["lr"] = 1e-4                                   # same object sha, different sidecar bytes
    tbm.write_text(json.dumps(doc, sort_keys=True))
    plan = M.plan_bank(w["tree"], MK, VIEW, w["source"], tree_sha=TREE_SHA)
    with pytest.raises(M.Refusal, match="budget_meta_sha256"):
        M.plan_parents(plan, pins=M.load_pins(w["tree"], MK), stage_dir=tmp_path / "stage", written_at="x")


def add_dose_copies(w: dict) -> dict:
    """A dose view ``d0p5_relu`` with two dose-scaled copies (scripts/sdfbase/scale_psr_dose.py's output shape): ``d0p5__<DIFF>`` inherits
    the derived parent's derived_meta.json, ``half__<CTL>`` the fit parent's train_budget_meta.json — each with a ``dose_scaled_from``
    block naming its parent and ``object_sha256`` re-bound to the scaled bytes."""
    tree, source, shas = w["tree"], w["source"], w["shas"]
    copies = {}
    for sid, parent_sid, sidecar in ((f"d0p5__{DIFF}", f"derived__{DIFF}", "derived_meta.json"), (f"half__{CTL}", f"base__{CTL}", "train_budget_meta.json")):
        d = source / sid
        d.mkdir(parents=True)
        blob = b"DOSE-" + w["objects"][parent_sid]
        (d / "object.pt").write_bytes(blob)
        meta = json.loads((source / parent_sid / sidecar).read_text())
        meta["object_sha256"] = sha(blob)
        meta["dose_scaled_from"] = {"parent_object_sha256": shas[parent_sid], "parent_staged_id": parent_sid, "scale": 0.5, "producer": "scripts/sdfbase/scale_psr_dose.py"}
        _write_json(d / sidecar, meta)
        (d / "staged_from.json").write_text("{}\n")
        copies[sid] = {"sha": sha(blob), "parent": shas[parent_sid], "sidecar": sidecar}
    roster = (tree / "configs" / "sdfbase_full22_roster.yaml").read_text()
    roster += f"""      d0p5_relu:
        dose: 0.5
        config: configs/sdfbase_full22_{MK}_d0p5_relu.yaml
        cells:
          {DIFF}: d0p5__{DIFF}
          {CTL}: half__{CTL}
"""
    (tree / "configs" / "sdfbase_full22_roster.yaml").write_text(roster)
    pins = json.loads((tree / "configs" / "sdfbase_full22_pins.json").read_text())
    for sid, c in copies.items():
        pins[MK][sid] = {"object_sha256": c["sha"], "source": "synthetic dose copy"}
    _write_json(tree / "configs" / "sdfbase_full22_pins.json", pins)
    return copies


def test_dose_copies_resolve_their_identity_from_the_source_dir_and_bind_the_dose_source(M, tmp_path):
    w = make_world(tmp_path)
    copies = add_dose_copies(w)
    plan = M.plan_bank(w["tree"], MK, "d0p5_relu", w["source"], tree_sha=TREE_SHA)
    by = {c.staged_id: c for c in plan.cells}
    diff, half = by[f"d0p5__{DIFF}"], by[f"half__{CTL}"]
    assert (diff.lane_kind, diff.sidecar) == ("dose", "derived_meta.json") and (half.lane_kind, half.sidecar) == ("dose", "train_budget_meta.json")
    assert diff.dose_source["parent_object_sha256"] == w["shas"][f"derived__{DIFF}"] and diff.dose_source["parent_staged_id"] == f"derived__{DIFF}"
    assert half.dose_source["parent_object_sha256"] == w["shas"][f"base__{CTL}"] and half.parents == []
    assert {p.cell_id for p in diff.parents} == {CTL, POS}            # the inherited fit lineage, still bound to the pins
    assert diff.remote_dir == f"/objects/sdfbase/full22/{MK}/d0p5_relu/{DIFF}" and [f.name for f in diff.files] == ["object.pt", "derived_meta.json"]
    man = M.build_manifest(plan, verified={}, dry=True, written_at=None)
    assert man["cells"][DIFF]["dose_scaled_from"] == {"parent_object_sha256": w["shas"][f"derived__{DIFF}"], "parent_staged_id": f"derived__{DIFF}", "scale": 0.5}
    # the registry models a dose copy as derived from its ONE dose source (derivation dose_scale), not from the fit lineage
    objs = M.registry_objects_of(plan, None)
    assert objs[copies[f"d0p5__{DIFF}"]["sha"]]["parents"] == [w["shas"][f"derived__{DIFF}"]]
    parent_of = {f"d0p5__{DIFF}": f"derived__{DIFF}", f"half__{CTL}": f"base__{CTL}"}
    rows = [{"object_id": c["sha"], "campaign_cell_id_legacy": sid, "artifact_sha256": c["sha"], "artifact_bytes": len(b"DOSE-") + len(w["objects"][parent_of[sid]]),
             "object_kind": "composed", "derivation": "dose_scale", "parent_object_ids_json": json.dumps([c["parent"]])} for sid, c in copies.items()]
    assert M.registry_check(objs, registry_root=str(make_registry(tmp_path / "reg", rows)))["status"] == "match"
    # a dose source that is not pinned refuses (the d0p5 copy's source is the derived object)
    pins = json.loads((w["tree"] / "configs" / "sdfbase_full22_pins.json").read_text())
    del pins[MK][f"derived__{DIFF}"]
    _write_json(w["tree"] / "configs" / "sdfbase_full22_pins.json", pins)
    with pytest.raises(M.Refusal, match="pinned under no staged id"):
        M.plan_bank(w["tree"], MK, "d0p5_relu", w["source"], tree_sha=TREE_SHA)


# --------------------------------------------------------------------------------------------------------------------------- #
# the registry cross-check                                                                                                      #
# --------------------------------------------------------------------------------------------------------------------------- #

def test_registry_check_match_unverified_and_mismatch(M, tmp_path):
    w = make_world(tmp_path)
    plan = M.plan_bank(w["tree"], MK, VIEW, w["source"], tree_sha=TREE_SHA)
    objs = M.registry_objects_of(plan, None)
    rows = registry_rows(w)
    assert M.registry_check(objs, registry_root=str(make_registry(tmp_path / "reg-ok", rows)))["status"] == "match"
    absent = M.registry_check(objs, registry_root=str(tmp_path / "reg-none"))
    assert absent["status"] == "unverified" and "no registry database" in absent["detail"]
    partial = M.registry_check(objs, registry_root=str(make_registry(tmp_path / "reg-partial", rows[:2])))
    assert partial["status"] == "unverified" and "no object row" in partial["detail"]
    bad = [dict(r) for r in rows]
    bad[0]["artifact_bytes"] = 999999
    assert M.registry_check(objs, registry_root=str(make_registry(tmp_path / "reg-bad", bad)))["status"] == "mismatch"
    badp = [dict(r) for r in rows]
    badp[2]["parent_object_ids_json"] = json.dumps(["0" * 64, "1" * 64])
    assert M.registry_check(objs, registry_root=str(make_registry(tmp_path / "reg-badp", badp)))["status"] == "mismatch"


# --------------------------------------------------------------------------------------------------------------------------- #
# dry run, --go, refusals, receipts                                                                                             #
# --------------------------------------------------------------------------------------------------------------------------- #

def test_dry_run_makes_no_modal_call_and_writes_dry_receipts(M, tmp_path):
    w = make_world(tmp_path)
    fake = FakeModal()
    lines = []
    out = M.run(**run_kw(M, w, tmp_path, parents=True, runner=fake, log=lines.append))
    assert fake.calls == [] and out["dry"] is True
    bank = json.loads(Path(out["bank"]).read_text())
    parents = json.loads(Path(out["parents"]).read_text())
    assert bank["step"] == "bank" and bank["dry"] is True and bank["verified_utc"] is None
    assert parents["step"] == "parents" and parents["dry"] is True and parents["verified_utc"] is None
    assert bank["manifest_outcome"] == "dry" and "written_at" not in bank["manifest"]
    assert all(f["outcome"] == "dry" for c in bank["cells"].values() for f in c["files"])
    assert any("DRY " in l and "volume put" in l for l in lines) and bank["inode_probe"]["status"] == "not_run"
    # the dry receipts land under <receipts-dir>/dry/, never at the live paths
    assert Path(out["bank"]) == tmp_path / "receipts" / "dry" / "bank.json" and Path(out["parents"]) == tmp_path / "receipts" / "dry" / "parents.json"
    assert not (tmp_path / "receipts" / "bank.json").exists() and not (tmp_path / "receipts" / "parents.json").exists()
    assert any("DRY  receipts ->" in l and "/dry" in l for l in lines)


def test_a_dry_run_after_a_live_stage_leaves_the_live_receipts_byte_identical(M, tmp_path):
    """The lane ran a DRY preview against the landed tree AFTER the live bank stage and the
    dry receipt briefly replaced the live one — the launch-set prerequisite check would have read `dry: true`. The dry run's receipts go
    under <receipts-dir>/dry/; the live receipts are untouched."""
    w = make_world(tmp_path)
    fake = FakeModal()
    live = M.run(**run_kw(M, w, tmp_path, go=True, parents=True, runner=fake))
    live_bank, live_parents = Path(live["bank"]).read_bytes(), Path(live["parents"]).read_bytes()
    assert Path(live["bank"]) == tmp_path / "receipts" / "bank.json" and json.loads(live_bank)["dry"] is False
    dry = M.run(**run_kw(M, w, tmp_path, parents=True, runner=FakeModal()))
    assert dry["dry"] is True and Path(dry["bank"]).parent == tmp_path / "receipts" / "dry"
    assert Path(live["bank"]).read_bytes() == live_bank and Path(live["parents"]).read_bytes() == live_parents
    assert json.loads(Path(dry["bank"]).read_text())["dry"] is True and json.loads(Path(dry["bank"]).read_text())["verified_utc"] is None


def test_go_puts_verifies_and_publishes_the_manifest_last(M, tmp_path):
    w = make_world(tmp_path)
    fake = FakeModal()
    reg = make_registry(tmp_path / "reg", registry_rows(w))
    out = M.run(**run_kw(M, w, tmp_path, go=True, parents=True, runner=fake, registry_root=str(reg), inode_probe=Path("/x/probe.py")))
    root = f"objects/sdfbase/full22/{MK}/{VIEW}"
    expected_bank = {f"{root}/{c}/object.pt" for c in (CTL, POS, DIFF)} | {f"{root}/{CTL}/train_budget_meta.json", f"{root}/{POS}/train_budget_meta.json",
                                                                          f"{root}/{DIFF}/derived_meta.json", f"{root}/staged_manifest.json"}
    expected_parents = {f"out/{FIT_MK}/cells_all/{c}/{n}" for c in (CTL, POS) for n in ("object.pt", "train_budget_meta.json", "metrics.jsonl", "staged_from.json")}
    assert set(fake.remote) == expected_bank | expected_parents
    assert not any(k.endswith("/COMPLETE") for k in fake.remote)
    # bytes on the volume are the mirror's bytes
    assert fake.remote[f"{root}/{DIFF}/object.pt"] == w["objects"][f"derived__{DIFF}"]
    # the manifest is the LAST put of the view, after every object/sidecar put and after the view's verification gets
    puts = fake.puts()
    i_man = puts.index(f"/{root}/staged_manifest.json")
    assert all(p.startswith(f"/{root}/") for p in puts[:i_man]) and len(puts[:i_man]) == 6
    first_get = next(i for i, c in enumerate(fake.calls) if c[1:3] == ["volume", "get"])
    i_man_call = next(i for i, c in enumerate(fake.calls) if c[1:3] == ["volume", "put"] and c[-1] == f"/{root}/staged_manifest.json")
    assert first_get < i_man_call
    man = json.loads(fake.remote[f"{root}/staged_manifest.json"])
    assert man["schema"] == "tinycap-full22-staged-objects/1" and man["remote_verified"]["n_files"] == 6 and man["written_at"]
    assert fake.calls[0][1:2] == ["run"] and fake.calls[0][-2:] == ["--floor-pct", "90.0"]   # the probe runs FIRST
    bank = json.loads(Path(out["bank"]).read_text())
    assert bank["dry"] is False and bank["verified_utc"] and bank["manifest_outcome"] == "put" and bank["inode_probe"]["status"] == "ok"
    assert bank["registry_check"]["status"] == "match"
    assert all(f["outcome"] == "put" and f["verification"].startswith("get+sha256") for c in bank["cells"].values() for f in c["files"])
    parents = json.loads(Path(out["parents"]).read_text())
    assert parents["root"] == f"/out/{FIT_MK}/cells_all" and parents["verified_utc"]
    assert [f["outcome"] for f in parents["parents"][CTL]["files"]] == ["put"] * 4


def test_go_is_idempotent_identical_remote_files_are_kept(M, tmp_path):
    w = make_world(tmp_path)
    fake = FakeModal()
    M.run(**run_kw(M, w, tmp_path, go=True, parents=True, runner=fake))
    n_remote = len(fake.remote)
    fake.calls.clear()
    out = M.run(**run_kw(M, w, tmp_path / "second", go=True, parents=True, runner=fake))
    assert fake.puts() == [] and len(fake.remote) == n_remote
    bank = json.loads(Path(out["bank"]).read_text())
    assert bank["manifest_outcome"] == "already_published"
    assert all(f["outcome"] == "kept_identical" for c in bank["cells"].values() for f in c["files"])
    parents = json.loads(Path(out["parents"]).read_text())
    outcomes = {f["name"]: f["outcome"] for f in parents["parents"][POS]["files"]}
    assert outcomes == {"object.pt": "kept_identical", "train_budget_meta.json": "kept_identical", "metrics.jsonl": "kept_identical",
                        "staged_from.json": "kept_same_object"}
    # the kept staged_from.json on the volume is the FIRST run's record, and the second receipt carries exactly that one
    first = json.loads(fake.remote[f"out/{FIT_MK}/cells_all/{POS}/staged_from.json"])
    assert parents["parents"][POS]["staged_from"] == first and first["object_sha256"] == w["shas"][f"base__{POS}"]
    sf = [f for f in parents["parents"][POS]["files"] if f["name"] == "staged_from.json"][0]
    assert sf["sha256"] == hashlib.sha256(fake.remote[f"out/{FIT_MK}/cells_all/{POS}/staged_from.json"]).hexdigest()


def test_go_refuses_a_differing_remote_file_without_manifest_or_receipt(M, tmp_path):
    w = make_world(tmp_path)
    root = f"objects/sdfbase/full22/{MK}/{VIEW}"
    fake = FakeModal({f"{root}/{POS}/object.pt": b"SOMEONE ELSE'S BYTES"})
    with pytest.raises(M.Refusal, match=f"{POS}/object.pt differs on the volume .* refusing to overwrite"):
        M.run(**run_kw(M, w, tmp_path, go=True, parents=True, runner=fake))
    assert fake.remote[f"{root}/{POS}/object.pt"] == b"SOMEONE ELSE'S BYTES"          # never overwritten, never deleted
    assert f"{root}/staged_manifest.json" not in fake.remote
    assert not any("cells_all" in k for k in fake.remote)                               # the parents never started
    assert not (tmp_path / "receipts").exists()
    # the files put BEFORE the refusal stay (the CTL cell precedes POS in the roster) — idempotent on the rerun, never deleted
    assert f"{root}/{CTL}/object.pt" in fake.remote


def test_go_refuses_a_different_published_manifest_and_an_extra_remote_file(M, tmp_path):
    w = make_world(tmp_path)
    root = f"objects/sdfbase/full22/{MK}/{VIEW}"
    foreign = {"schema": "tinycap-full22-staged-objects/1", "model_key": MK, "view": VIEW, "staged_by": "another lane", "cells": {}, "written_at": "x"}
    fake = FakeModal({f"{root}/staged_manifest.json": json.dumps(foreign).encode()})
    with pytest.raises(M.Refusal, match="DIFFERENT staged_manifest.json is already published .*another lane"):
        M.run(**run_kw(M, w, tmp_path, go=True, runner=fake))
    assert json.loads(fake.remote[f"{root}/staged_manifest.json"]) == foreign and not (tmp_path / "receipts").exists()
    fake2 = FakeModal({f"{root}/{CTL}/COMPLETE": b"{}"})
    with pytest.raises(M.Refusal, match="remote verification FAILED .* remote files .* != planned"):
        M.run(**run_kw(M, w, tmp_path / "b", go=True, runner=fake2))
    assert f"{root}/staged_manifest.json" not in fake2.remote and not (tmp_path / "b" / "receipts").exists()


def test_go_refuses_the_inode_floor_a_registry_mismatch_and_a_dirty_config_before_any_put(M, tmp_path):
    w = make_world(tmp_path)
    fake = FakeModal(probe_floor_hit=True, probe_pct=93.2)
    with pytest.raises(M.Refusal, match="inode floor"):
        M.run(**run_kw(M, w, tmp_path, go=True, runner=fake, inode_probe=Path("/x/probe.py")))
    assert fake.puts() == [] and fake.remote == {}
    bad = [dict(r) for r in registry_rows(w)]
    bad[1]["artifact_sha256"] = "9" * 64
    fake = FakeModal()
    with pytest.raises(M.Refusal, match="registry's object rows disagree"):
        M.run(**run_kw(M, w, tmp_path, go=True, runner=fake, registry_root=str(make_registry(tmp_path / "reg-bad", bad))))
    assert fake.calls == []
    fake = FakeModal()
    with pytest.raises(M.Refusal, match="roster/pins are modified"):
        M.run(**run_kw(M, w, tmp_path, go=True, runner=fake, tree_info={"config_dirty": [" M configs/sdfbase_full22_pins.json"], "tree_on_remote": True}))
    assert fake.calls == [] and not (tmp_path / "receipts").exists()


def test_cli_dry_run_through_main_with_a_tree_sha_override(M, tmp_path, capsys):
    w = make_world(tmp_path)
    rc = M.main(["--model-key", MK, "--view", VIEW, "--tree", str(w["tree"]), "--source", str(w["source"]), "--receipts-dir", str(tmp_path / "r"),
                 "--parents", "--tree-sha", TREE_SHA, "--stage-dir", str(tmp_path / "stage"), "--registry-root", str(tmp_path / "none")])
    out = capsys.readouterr().out
    assert rc == 0 and "DONE (DRY" in out and "volume put" in out and "3 cell(s) x 2 files" in out
    assert not (tmp_path / "r" / "bank.json").exists()                     # A dry run never writes the live receipt path
    bank = json.loads((tmp_path / "r" / "dry" / "bank.json").read_text())
    assert bank["tree_sha"] == TREE_SHA and bank["tree_sha_source"] == "--tree-sha argument" and bank["dry"] is True
    assert bank["registry_check"]["status"] == "unverified"
    rc2 = M.main(["--model-key", MK, "--view", "nope", "--tree", str(w["tree"]), "--source", str(w["source"]), "--receipts-dir", str(tmp_path / "r2"),
                  "--tree-sha", TREE_SHA])
    assert rc2 == M.EXIT_REFUSED and "REFUSED" in capsys.readouterr().out and not (tmp_path / "r2").exists()


def test_receipt_shape(M, tmp_path):
    w = make_world(tmp_path)
    fake = FakeModal()
    out = M.run(**run_kw(M, w, tmp_path, go=True, parents=True, runner=fake, registry_root=str(make_registry(tmp_path / "reg", registry_rows(w)))))
    bank = json.loads(Path(out["bank"]).read_text())
    for k in ("schema", "step", "dry", "model_key", "view", "volume", "view_root", "objects_dir", "tree", "tree_sha", "source", "staged_by", "started_utc",
              "written_utc", "verified_utc", "verified_by", "roster_sha256", "pins_sha256", "pins_compared", "cells", "prompted_cells_not_staged",
              "manifest_remote_path", "manifest_outcome", "manifest", "registry_check", "inode_probe", "config_dirty", "tree_on_remote"):
        assert k in bank, k
    assert bank["schema"] == M.RECEIPT_SCHEMA and bank["pins_compared"] == w["shas"] and bank["manifest"]["written_at"]
    f = bank["cells"][CTL]["files"][0]
    assert set(f) == {"name", "remote_path", "sha256", "bytes", "outcome", "verification"} and f["remote_path"].startswith("/objects/")
    assert bank["cells"][DIFF]["parents"][0]["staged_id"].startswith("base__")
    parents = json.loads(Path(out["parents"]).read_text())
    for k in ("schema", "step", "dry", "root", "parents_root", "tree_sha", "verified_utc", "complete_marker", "parents", "registry_check"):
        assert k in parents, k
    p = parents["parents"][CTL]
    assert set(p) == {"fit_model_key", "staged_id", "source", "object_sha256", "budget_meta_sha256", "consumers", "volume_path", "staged_from", "files"}
    assert [x["remote_path"] for x in p["files"]] == [f"/out/{FIT_MK}/cells_all/{CTL}/{n}" for n in ("object.pt", "train_budget_meta.json", "metrics.jsonl", "staged_from.json")]
    # no personal identifiers in what the repo file prints or writes by default (paths come from the caller)
    src = SCRIPT.read_text()
    assert "/home/" not in src and "Path.home()" in src


# --------------------------------------------------------------------------------------------------------------------------- #
# the registry row must be THIS object's; the tree_sha-only manifest replacement                                                 #
# --------------------------------------------------------------------------------------------------------------------------- #

def test_registry_check_refuses_a_row_of_another_object_id_that_shares_the_artifact_sha(M, tmp_path):
    w = make_world(tmp_path)
    plan = M.plan_bank(w["tree"], MK, VIEW, w["source"], tree_sha=TREE_SHA)
    objs = M.registry_objects_of(plan, None)
    rows = registry_rows(w)
    other = [dict(r) for r in rows]
    other[0]["object_id"] = "f" * 64                      # another object's row carrying the same artifact sha
    res = M.registry_check(objs, registry_root=str(make_registry(tmp_path / "reg-otherid", other)))
    assert res["status"] == "mismatch" and "not this object's" in res["detail"]
    # the honest row (object_id == sha == artifact sha) still matches
    assert M.registry_check(objs, registry_root=str(make_registry(tmp_path / "reg-ok", rows)))["status"] == "match"


def test_the_manifest_replacement_admits_only_a_tree_sha_change_on_this_stagers_own_manifest(M, tmp_path):
    w = make_world(tmp_path)
    plan = M.plan_bank(w["tree"], MK, VIEW, w["source"], tree_sha=TREE_SHA)
    fresh = M.build_manifest(plan, verified={}, dry=False, written_at="2026-10-03T22:00:00Z")
    published = dict(fresh, tree_sha="a" * 40, written_at="2026-10-03T21:00:00Z")
    assert M.manifest_replace_problem(published, fresh) is None
    assert "identical" in M.manifest_replace_problem(dict(fresh, written_at="x"), fresh)
    assert "another lane" in M.manifest_replace_problem(dict(published, staged_by="someone-else"), fresh)
    assert "only ['tree_sha']" in M.manifest_replace_problem(dict(published, model_key="other"), fresh)
    # through publish_manifest over the in-memory volume
    remote_dir = plan.remote_view_root
    logs = []
    fake = FakeModal({f"{remote_dir.lstrip('/')}/{M.MANIFEST_NAME}": (json.dumps(published) + "\n").encode()})
    vol = M.VolumeCLI(Path("/fake/modal"), runner=fake, log=logs.append)
    with pytest.raises(M.Refusal, match="DIFFERENT staged_manifest.json"):
        M.publish_manifest(fresh, vol, remote_dir=remote_dir, stage_dir=tmp_path / "st1", dry=False, log=logs.append)
    res = M.publish_manifest(fresh, vol, remote_dir=remote_dir, stage_dir=tmp_path / "st2", dry=False, log=logs.append, replace_manifest=True)
    assert res["outcome"] == "replaced_tree_sha" and res["replaced"]["tree_sha"] == "a" * 40
    assert json.loads(fake.remote[f"{remote_dir.lstrip('/')}/{M.MANIFEST_NAME}"].decode())["tree_sha"] == TREE_SHA
    assert any("--force" in c for c in fake.calls if c[1:3] == ["volume", "put"])
    # any other difference, even with the flag, refuses — and so does another lane's manifest
    fake2 = FakeModal({f"{remote_dir.lstrip('/')}/{M.MANIFEST_NAME}": (json.dumps(dict(published, model_key="other")) + "\n").encode()})
    vol2 = M.VolumeCLI(Path("/fake/modal"), runner=fake2, log=logs.append)
    with pytest.raises(M.Refusal, match="--replace-manifest refused"):
        M.publish_manifest(fresh, vol2, remote_dir=remote_dir, stage_dir=tmp_path / "st3", dry=False, log=logs.append, replace_manifest=True)
    fake3 = FakeModal({f"{remote_dir.lstrip('/')}/{M.MANIFEST_NAME}": (json.dumps(dict(published, staged_by="someone-else")) + "\n").encode()})
    vol3 = M.VolumeCLI(Path("/fake/modal"), runner=fake3, log=logs.append)
    with pytest.raises(M.Refusal, match="another lane"):
        M.publish_manifest(fresh, vol3, remote_dir=remote_dir, stage_dir=tmp_path / "st4", dry=False, log=logs.append, replace_manifest=True)
