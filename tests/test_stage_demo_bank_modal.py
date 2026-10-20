"""reproduce/tools/stage_demo_bank_modal.py — the worked example's Modal bank from the paper's COMMITTED objects.

The plan is derived from objects/DEMO.json + reproduce/models.yaml (never hand-listed): the seven launcher tokens of the wave, each `<bank
root>/<token>/{object.pt, <one identity sidecar>}` (exactly two files, never COMPLETE), plus the two derived cells' FOUR parents (since the
one-seed DEMO.json, the same bytes as four of the cells) under `/out/<parent model_key>/cells_all/<cell>/{object.pt, train_budget_meta.json}`;
`--bank-root` defaults to contracts.MODELS[<key>].objects_dir
minus the volume mount (never a literal) and refuses the full22 root; every object is held to its id, its sidecar's name for the token and the
paper key's pins; a DOSE view plans its committed dose copy into the view's own row's bank (`launcher_slug`), pinned
under the key's block; a dry run writes its receipts under `<receipts-dir>/dry/` and makes no Modal call; the Modal CLI is a subprocess.run-shaped
double over an in-memory volume that REJECTS an unconverted `/vol/...` argument, and the live protocol is exercised over
it: put-if-absent, get+sha verify, listings exact for the cells, the manifest LAST. The refusals run over a tmp copy of the committed inputs
with ONE fact tampered. Synthetic names only; nothing here names a host or a credential."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, REPRO  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPRO / "lib"))
import models as MT  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402

TOOL = REPO / "reproduce" / "tools" / "stage_demo_bank_modal.py"
KEY = "qwen35_4b_demo"
PAPER_KEY = "qwen35_4b"
#: the seven launcher tokens the wave serves with an object (lib/models.py launcher_cells over the models table), pinned literally
TOKENS = ("arditi_allpos", "constv2ctl2-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl",
          "constv2ctl2-full__psr_ce_all", "constv2-mix-full__psr_ce_all", "constv2-mix-full__psr_ce_all_minus_ctl")
PARENT_MK = "qwen35_4b_base_eb64_s3"
PARENTS = ("constv2-mix-full__lora_r4_ce", "constv2ctl2-full__lora_r4_ce", "constv2-mix-full__psr_ce_all", "constv2ctl2-full__psr_ce_all")
#: the BATTERY manifest's key sets, FROZEN: the wave's staged_manifest.json is already published on the volume
#: from earlier smokes and publish_manifest refuses a different manifest at the same path, so any key added to build_manifest would make the
#: battery's stage phase refuse its own bank. `written_at` rides on a live manifest only; `parents` on a derived cell only; `dry` in
#: remote_verified on a dry manifest only. A dose view's manifest (its own path) adds `dose` PER CELL and nothing at the top level.
BATTERY_MANIFEST_KEYS = ("schema", "model_key", "view", "paper_key", "bank_source", "table", "table_sha256", "demo", "demo_sha256", "manifest",
                         "manifest_sha256", "pins", "pins_sha256", "volume_path", "objects_dir", "manifest_name", "staged_by", "remote_verified",
                         "parents_root", "parents", "prompted_cells_not_staged", "cells")
BATTERY_MANIFEST_CELL_KEYS = ("cell", "cell_id", "view_cell", "paper_label", "staged_id", "kind", "object_id", "object_sha256", "bytes", "identity_sidecar",
                              "sidecar_sha256", "source", "pinned_as", "volume_path", "objects_dir_entry", "n_files", "files")
BATTERY_MANIFEST_FILE_KEYS = ("path", "sha256", "bytes", "verified_by")
BATTERY_MANIFEST_PARENT_KEYS = ("object_sha256", "budget_meta_sha256", "consumers", "volume_path", "pinned_as")


@pytest.fixture(scope="module")
def S():
    spec = importlib.util.spec_from_file_location("stage_demo_bank_under_test", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeModal:
    """A subprocess.run-shaped double of the Modal CLI over an in-memory volume {rel path (no leading slash): bytes}. An
    argument that still carries the CONTAINER mount (`/vol/...`) is a bug in the caller's path conversion and is REJECTED, never served."""

    def __init__(self, remote=None):
        self.remote = dict(remote or {})
        self.calls: list = []

    def __call__(self, argv, **kw):
        argv = list(argv)
        self.calls.append(argv)
        sub = argv[1:]
        for a in sub:
            assert not (a == "/vol" or a.startswith("/vol/")), f"an unconverted container path reached the modal CLI: {a!r} in {argv}"
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
        raise AssertionError(f"unexpected modal invocation {argv}")

    def puts(self):
        return [c[-1] for c in self.calls if c[1:3] == ["volume", "put"]]


def _run_kw(tmp_path, **over):
    kw = dict(key=KEY, view=None, receipts_dir=tmp_path / "07_bank", go=False, volume="twosi-fits", modal_bin=Path("/nonexistent/modal"), repo=REPO,
              table=None, bank_root=None, stage_dir=tmp_path / "stage", registry_root=str(tmp_path / "no-registry"), runner=FakeModal(), log=lambda s: None)
    kw.update(over)
    return kw


def test_the_plan_is_the_seven_tokens_and_the_four_parents_at_the_rows_own_root(S):
    plan = S.plan_demo_bank(KEY, None, repo=REPO)
    view = MT.baseline_view(KEY)
    assert plan.view == view and plan.paper_key == PAPER_KEY
    assert plan.bank_root == C.model_row(KEY).objects_dir[len(S.M.VOL_MOUNT):] == f"/objects/sdfbase/{KEY}" and plan.objects_dir == C.model_row(KEY).objects_dir
    assert not plan.bank_root.startswith(C.VOL_OBJECTS_FULL22[len(S.M.VOL_MOUNT):])                       # outside the full22 root
    assert sorted(c.token for c in plan.cells) == sorted(TOKENS)
    assert [c.token for c in plan.cells] == [t for _c, t in MT.launcher_cells(KEY, view) if t != "none" and not t.startswith("prompted__")]
    assert plan.prompted_cells_not_staged == ["prompted__safety_refusal", "prompted__si_constitution"]
    demo = {e["paper_label"]: e for e in json.loads((REPO / "objects" / "DEMO.json").read_text())["objects"]}
    for c in plan.cells:
        assert c.remote_dir == f"{plan.bank_root}/{c.token}" and [f.name for f in c.files] == ["object.pt", c.sidecar] and len(c.files) == 2
        assert all(f.remote.startswith(plan.bank_root + "/") and not f.remote.startswith("/vol/") for f in c.files)
        assert c.object_id == demo[c.paper_label]["object_id"] == c.files[0].sha256 == hashlib.sha256(c.files[0].local.read_bytes()).hexdigest()
        assert "COMPLETE" not in [f.name for f in c.files] and c.pinned_as and all(c.object_id == json.loads((REPO / "configs/sdfbase_full22_pins.json").read_text())[PAPER_KEY][sid]["object_sha256"] for sid in c.pinned_as)
    by_tok = {c.token: c for c in plan.cells}
    assert by_tok["arditi_allpos"].sidecar == "fit_provenance.json" and by_tok["constv2ctl2-full__lora_r4_ce"].sidecar == "train_budget_meta.json"
    assert by_tok["constv2-mix-full__lora_r4_ce_minus_ctl"].sidecar == by_tok["constv2-mix-full__psr_ce_all_minus_ctl"].sidecar == "derived_meta.json"
    # the four s3 parents, from the derived sidecars' blocks: object.pt + train_budget_meta.json each, never COMPLETE / metrics / staged_from
    assert sorted((p.model_key, p.cell_id) for p in plan.parents) == sorted((PARENT_MK, c) for c in PARENTS)
    for p in plan.parents:
        assert p.remote_dir == f"/out/{PARENT_MK}/cells_all/{p.cell_id}" and [f.name for f in p.files] == ["object.pt", "train_budget_meta.json"]
        assert p.budget_meta_sha256 == hashlib.sha256((p.source_dir / "train_budget_meta.json").read_bytes()).hexdigest() and p.pinned_as
        assert (p.source_dir / "COMPLETE").is_file() and (p.source_dir / "metrics.jsonl").is_file()       # the committed dir has them; none is planned
    assert {tuple(sorted(p.consumers)) for p in plan.parents} == {("constv2-mix-full__lora_r4_ce_minus_ctl",), ("constv2-mix-full__psr_ce_all_minus_ctl",)}
    assert len(plan.files) == 14 and len(plan.parent_files) == 8


def test_a_dry_run_makes_no_modal_call_and_writes_its_receipts_under_dry(S, tmp_path):
    fake = FakeModal()
    out = S.run(**_run_kw(tmp_path, runner=fake))
    assert fake.calls == [] and out["dry"] is True and out["receipts_dir"] == str(tmp_path / "07_bank" / "dry")
    bank = json.loads((tmp_path / "07_bank" / "dry" / "bank.json").read_text())
    parents = json.loads((tmp_path / "07_bank" / "dry" / "parents.json").read_text())
    assert bank["dry"] is True and bank["verified_utc"] is None and sorted(bank["cells"]) == sorted(TOKENS) and bank["bank_root"] == f"/objects/sdfbase/{KEY}"
    assert bank["objects_dir"] == C.model_row(KEY).objects_dir and bank["manifest"]["schema"] == S.MANIFEST_SCHEMA and bank["manifest"]["staged_by"] == S.STAGED_BY
    assert "written_at" not in bank["manifest"] and bank["manifest_outcome"] == "dry"
    assert sorted(parents["parents"]) == sorted(f"{PARENT_MK}/{c}" for c in PARENTS) and parents["dry"] is True
    assert not (tmp_path / "07_bank" / "bank.json").exists()
    for c in bank["cells"].values():
        assert [f["name"] for f in c["files"]] == ["object.pt", c["identity_sidecar"]] and all(f["outcome"] == "dry" for f in c["files"])


def test_the_live_protocol_puts_verifies_and_publishes_the_manifest_last_over_the_fake_volume(S, tmp_path):
    fake = FakeModal()
    out = S.run(**_run_kw(tmp_path, runner=fake, go=True))
    assert out["dry"] is False and (tmp_path / "07_bank" / "bank.json").is_file() and not (tmp_path / "07_bank" / "dry").exists()
    puts = fake.puts()
    assert len(puts) == 14 + 8 + 1 and puts[-1] == f"/objects/sdfbase/{KEY}/staged_manifest.json"            # the manifest LAST
    assert all(not p.startswith("/vol/") for p in puts)
    for tok in TOKENS:
        names = sorted(k.rsplit("/", 1)[1] for k in fake.remote if k.startswith(f"objects/sdfbase/{KEY}/{tok}/"))
        assert len(names) == 2 and "object.pt" in names and "COMPLETE" not in names, (tok, names)
    for cell in PARENTS:
        names = sorted(k.rsplit("/", 1)[1] for k in fake.remote if k.startswith(f"out/{PARENT_MK}/cells_all/{cell}/"))
        assert names == ["object.pt", "train_budget_meta.json"], (cell, names)
    man = json.loads(fake.remote[f"objects/sdfbase/{KEY}/staged_manifest.json"])
    assert man["schema"] == S.MANIFEST_SCHEMA and sorted(man["cells"]) == sorted(TOKENS) and man["written_at"] and man["objects_dir"] == C.model_row(KEY).objects_dir
    assert sorted(man["parents"]) == sorted(f"{PARENT_MK}/{c}" for c in PARENTS) and man["remote_verified"]["n_files"] == 14
    bank = json.loads((tmp_path / "07_bank" / "bank.json").read_text())
    assert bank["dry"] is False and bank["verified_utc"] and all(f["outcome"] == "put" and f["verification"].startswith("get+sha256") for c in bank["cells"].values() for f in c["files"])
    # a second live run: identical files kept, the identical manifest already published, nothing put again
    fake2 = FakeModal(fake.remote)
    S.run(**_run_kw(tmp_path / "again", runner=fake2, go=True))
    assert fake2.puts() == []
    bank2 = json.loads((tmp_path / "again" / "07_bank" / "bank.json").read_text())
    assert bank2["manifest_outcome"] == "already_published" and all(f["outcome"] == "kept_identical" for c in bank2["cells"].values() for f in c["files"])
    # a DIFFERENT object already on the volume under a cell refuses AT that file (the per-file protocol: absent -> put, present -> get + sha,
    # different -> refuse by name): the foreign bytes are never overwritten or deleted, nothing after it is put, no receipt is written
    foreign = f"objects/sdfbase/{KEY}/{TOKENS[1]}/object.pt"
    fake3 = FakeModal({foreign: b"someone else's bytes"})
    with pytest.raises(S.Refusal, match="differs on the volume"):
        S.run(**_run_kw(tmp_path / "diff", runner=fake3, go=True))
    assert fake3.remote[foreign] == b"someone else's bytes" and "/" + foreign not in fake3.puts() and not (tmp_path / "diff" / "07_bank").exists()
    assert all(p.startswith(f"/objects/sdfbase/{KEY}/{TOKENS[0]}/") for p in fake3.puts())      # only the cell before it was reached


def test_the_fake_modal_rejects_an_unconverted_container_path_and_the_conversion_is_checked(S, tmp_path):
    fake = FakeModal()
    with pytest.raises(AssertionError, match="unconverted container path"):
        fake(["modal", "volume", "ls", "twosi-fits", "/vol/objects/sdfbase/x"])
    assert S.volume_rel("/vol/objects/sdfbase/qwen35_4b_demo") == "/objects/sdfbase/qwen35_4b_demo"
    assert S.volume_rel("/objects/sdfbase/qwen35_4b_demo") == "/objects/sdfbase/qwen35_4b_demo"
    with pytest.raises(S.Refusal, match="neither a container path"):
        S.volume_rel("objects/relative")


def test_bank_root_is_the_rows_objects_dir_minus_the_mount_and_the_full22_root_refuses(S, tmp_path):
    assert S.default_bank_root(KEY) == f"/objects/sdfbase/{KEY}"
    with pytest.raises(S.Refusal, match="sits under the full22 root"):
        S.plan_demo_bank(KEY, None, repo=REPO, bank_root=C.full22_objects_dir(PAPER_KEY, "s1")[len(S.M.VOL_MOUNT):])
    with pytest.raises(S.Refusal, match="volume-root path"):
        S.plan_demo_bank(KEY, None, repo=REPO, bank_root="/vol/objects/sdfbase/other")
    with pytest.raises(S.Refusal, match="volume-root path"):
        S.plan_demo_bank(KEY, None, repo=REPO, bank_root="objects/sdfbase/other")
    with pytest.raises(S.Refusal, match="contracts.MODELS has no row"):
        S.default_bank_root("no_such_slug")


def test_a_dose_view_plans_its_committed_copy_into_the_views_own_rows_bank(S, tmp_path):
    """demo-import (2026-10-12; the Known gap "the dose views stay on the cluster" closed): a DOSE view's plan is the view's committed dose-scaled
    copy (import_objects.Plan: the manifest's dose_scale row whose stamp names the DEMO parent at the view's dose; the same bytes 09 imports
    as `d0p75__<cell_id>`) under the wave's parent's TOKEN in the dose view's OWN registry row's bank (`launcher_slug` <key>-<view>, objects_dir
    `<the key's bank>/<view>`), with the copy's inherited sidecar — so its parents are the wave's — pinned under the KEY's block by its staged id
    while its parent is pinned under the paper key; the manifest records the dose per cell and the receipt (bank.json) the slug — the manifest
    gains NO top-level key (the ruling below); a tampered stamp refuses by name."""
    m = MT.model(KEY)
    plan_wave = S.plan_demo_bank(KEY, None, repo=REPO)
    for view in [v for v, vd in m["views"].items() if "dose" in (vd or {})]:
        slug = MT.launcher_slug(KEY, view)
        plan = S.plan_demo_bank(KEY, view, repo=REPO)
        assert (plan.slug, plan.view, plan.bank_root, plan.objects_dir) == (slug, view, f"/objects/sdfbase/{KEY}/{view}", C.model_row(slug).objects_dir)
        assert S.default_bank_root(slug) == plan.bank_root and plan.bank_root.startswith(plan_wave.bank_root + "/")
        [cell] = plan.cells
        wave_cell = next(c for c in plan_wave.cells if c.token == cell.token)
        vc = m["views"][view]["cells"][1]
        assert cell.view_cell == vc and cell.token == MT.roster_cell_id(m, vc) == wave_cell.token and cell.staged_id == MT.staged_id(m, vc, view)
        assert cell.object_id != wave_cell.object_id and cell.dose == {**cell.dose, "scale": 0.75, "parent_object_sha256": wave_cell.object_id}
        assert cell.dose["parent_pinned_as"] == {PAPER_KEY: wave_cell.pinned_as} and cell.kind == "derived" and cell.sidecar == "derived_meta.json"
        pins = json.loads((REPO / "configs" / "sdfbase_full22_pins.json").read_text())
        assert cell.pinned_as == [cell.staged_id] and pins[KEY][cell.staged_id]["object_sha256"] == cell.object_id and cell.staged_id not in pins[PAPER_KEY]
        assert cell.remote_dir == f"{plan.bank_root}/{cell.token}" and [f.remote for f in cell.files] == [f"{cell.remote_dir}/object.pt", f"{cell.remote_dir}/derived_meta.json"]
        assert hashlib.sha256(cell.files[0].local.read_bytes()).hexdigest() == cell.object_id == cell.files[0].sha256
        # the copy inherits its parent's sidecar: the same two parents as the wave's derived cell, pinned under the paper key
        assert sorted(p.cell_id for p in plan.parents) == sorted(p["cell_id"] for p in wave_cell.parents) and all(p.pinned_as for p in plan.parents)
        assert plan.prompted_cells_not_staged == []
        man = S.build_manifest(plan, verified={}, dry=True, written_at=None)
        assert "launcher_slug" not in man and man["view"] == view and man["objects_dir"] == C.model_row(slug).objects_dir      # the slug: bank.json only
        assert man["cells"][cell.token]["dose"] == cell.dose and man["cells"][cell.token]["pinned_as"] == {KEY: [cell.staged_id]} and man["cells"][cell.token]["object_id"] == cell.object_id
        assert set(man) == set(S.build_manifest(plan_wave, verified={}, dry=True, written_at=None)) == set(BATTERY_MANIFEST_KEYS)   # the same top-level keys as the wave's
        # the dry run of the tool writes per-view receipts naming the slug and the dose, no Modal call
        out = S.run(**_run_kw(tmp_path / view, view=view))
        bank = json.loads((tmp_path / view / "07_bank" / "dry" / "bank.json").read_text())
        assert bank["launcher_slug"] == slug and bank["bank_root"] == plan.bank_root and bank["cells"][cell.token]["dose"]["scale"] == 0.75
        assert bank["cells"][cell.token]["object_id"] == cell.object_id and bank["cells"][cell.token]["pinned_as"] == {KEY: [cell.staged_id]} and out["dry"] is True
    # the wave's plan is unchanged by the dose rows: the key's own row, its root, no dose on any cell
    assert plan_wave.slug == KEY and plan_wave.bank_root == f"/objects/sdfbase/{KEY}" and all(c.dose is None for c in plan_wave.cells)
    assert S.build_manifest(plan_wave, verified={}, dry=True, written_at=None)["cells"][TOKENS[0]]["pinned_as"] == {PAPER_KEY: plan_wave.cells[0].pinned_as}
    # a copy whose stamp names another parent or another dose refuses by name (a tmp copy of the committed inputs, one fact tampered)
    tree = tmp_path / "tamper"
    shutil.copytree(REPO / "objects", tree / "objects", symlinks=True)
    for d in ("configs", "reproduce", "src", "scripts"):
        (tree / d).symlink_to(REPO / d)
    cell = S.plan_demo_bank(KEY, "d0p75", repo=REPO).cells[0]
    side = tree / "objects" / cell.source_dir.relative_to(REPO / "objects") / "derived_meta.json"
    original = side.read_text()                                   # copytree copied the sidecar as a regular file: keep its bytes to restore below
    doc = json.loads(original); doc["dose_scaled_from"]["scale"] = 0.5
    side.write_text(json.dumps(doc))
    # the importer's Plan reads the same stamp and refuses first, in its words (no committed copy at the view's dose); the stager's own
    # re-assertion of the stamp stands behind it for a plan that passed — either refusal is by name, nothing is planned
    with pytest.raises(S.Refusal, match=r"0 committed dose_scale copies of objects/DEMO.json:difference_lora .* at scale 0.75|the view's copy is of"):
        S.plan_demo_bank(KEY, "d0p75", repo=tree)
    pins = json.loads((REPO / "configs" / "sdfbase_full22_pins.json").read_text()); pins[KEY].pop(cell.staged_id)
    (tree / "configs").unlink(); (tree / "configs").mkdir()
    for f in (REPO / "configs").iterdir():
        (tree / "configs" / f.name).symlink_to(f)
    (tree / "configs" / "sdfbase_full22_pins.json").unlink(); (tree / "configs" / "sdfbase_full22_pins.json").write_text(json.dumps(pins))
    side.write_text(original)
    with pytest.raises(S.Refusal, match=rf"pins no '{cell.staged_id}'"):
        S.plan_demo_bank(KEY, "d0p75", repo=tree)


def test_the_battery_manifests_key_sets_are_unchanged_so_the_published_one_is_kept(S, tmp_path):
    """On the dose-view work: the battery's staged_manifest.json already sits on the volume from earlier smokes,
    and publish_manifest keeps an identical manifest (minus written_at) but REFUSES a different one at the same path — so the manifest must gain
    no key (`launcher_slug` stays in the bank RECEIPT alone). Pinned: the wave manifest's top-level, per-cell, per-file and per-parent key sets
    are the ones above, dry and live alike (live adds `written_at` only); no cell carries `dose`; the dose view's manifest adds `dose` per cell
    and nothing at the top level; the receipt carries the slug; and a wave manifest built twice is byte-identical, which is what a re-stage over
    the published one relies on (`already_published`)."""
    plan = S.plan_demo_bank(KEY, None, repo=REPO)
    dry = S.build_manifest(plan, verified={}, dry=True, written_at=None)
    assert tuple(dry) == BATTERY_MANIFEST_KEYS and "launcher_slug" not in dry and "written_at" not in dry
    assert set(dry["remote_verified"]) == {"method", "n_files", "dry"} and dry["remote_verified"]["n_files"] == 0
    derived = {t for t in TOKENS if t.endswith("_minus_ctl")}
    for tok, cell in dry["cells"].items():
        want = BATTERY_MANIFEST_CELL_KEYS + (("parents",) if tok in derived else ())
        assert tuple(cell) == want, (tok, tuple(cell))
        assert "dose" not in cell and cell["pinned_as"] == {PAPER_KEY: next(c for c in plan.cells if c.token == tok).pinned_as}
        assert all(tuple(f) == BATTERY_MANIFEST_FILE_KEYS for f in cell["files"]) and cell["n_files"] == len(cell["files"]) == 2
    assert all(tuple(p) == BATTERY_MANIFEST_PARENT_KEYS for p in dry["parents"].values()) and len(dry["parents"]) == 4
    assert json.dumps(dry, sort_keys=True) == json.dumps(S.build_manifest(plan, verified={}, dry=True, written_at=None), sort_keys=True)
    # live: the same keys plus written_at, remote_verified without `dry`; the manifest the fake volume holds is exactly the built one
    fake = FakeModal()
    S.run(**_run_kw(tmp_path, runner=fake, go=True))
    live = json.loads(fake.remote[f"objects/sdfbase/{KEY}/staged_manifest.json"])             # published with sort_keys: compare as sets
    assert set(live) == set(BATTERY_MANIFEST_KEYS) | {"written_at"} and set(live["remote_verified"]) == {"method", "n_files"}
    assert {t: set(c) for t, c in live["cells"].items()} == {t: set(c) for t, c in dry["cells"].items()} and all("dose" not in c for c in live["cells"].values())
    bank = json.loads((tmp_path / "07_bank" / "bank.json").read_text())
    assert bank["launcher_slug"] == KEY and bank["manifest"] == live                       # the slug in the receipt; the manifest as published
    # a dose view's manifest: its own path, the same top-level keys, `dose` on its one cell and nowhere else
    m = MT.model(KEY)
    for view in [v for v, vd in m["views"].items() if "dose" in (vd or {})]:
        dman = S.build_manifest(S.plan_demo_bank(KEY, view, repo=REPO), verified={}, dry=True, written_at=None)
        assert tuple(dman) == BATTERY_MANIFEST_KEYS and dman["volume_path"] == f"/objects/sdfbase/{KEY}/{view}" != dry["volume_path"]
        [(tok, cell)] = dman["cells"].items()
        assert tuple(cell) == BATTERY_MANIFEST_CELL_KEYS + ("parents", "dose") and tok in dry["cells"]


def test_every_volume_command_is_scoped_to_the_configured_modal_environment(S, tmp_path, monkeypatch):
    """The launcher runs under `modal run -e $SISTEER_MODAL_ENVIRONMENT`, and a volume NAME is per environment — so a stager
    that let the CLI resolve its own environment (MODAL_ENVIRONMENT, the profile, the workspace default) could stage into a volume the launcher
    never reads. VolumeCLI.argv puts `--env <name>` right after the subcommand verb (`modal volume get --env X …`; `modal run --env X <script>`,
    before the script path), every ls/get/put of a live stage carries it, the receipts record it, and no environment means no `--env` (the
    full22 stager's own argv is byte-identical). The CLI takes `--environment`, else $SISTEER_MODAL_ENVIRONMENT, else nothing."""
    V = S.M.VolumeCLI
    assert V(Path("/m"), "vol", environment="demo-env").argv("volume", "get", "--force", "vol", "/a/b", "/tmp/x") == ["/m", "volume", "get", "--env", "demo-env", "--force", "vol", "/a/b", "/tmp/x"]
    assert V(Path("/m"), "vol", environment="demo-env").argv("volume", "ls", "--json", "vol", "a/b") == ["/m", "volume", "ls", "--env", "demo-env", "--json", "vol", "a/b"]
    assert V(Path("/m"), "vol", environment="demo-env").argv("run", "/s.py", "--floor-pct", "90") == ["/m", "run", "--env", "demo-env", "/s.py", "--floor-pct", "90"]
    assert V(Path("/m"), "vol").argv("volume", "put", "vol", "/l", "/r") == ["/m", "volume", "put", "vol", "/l", "/r"] and V(Path("/m"), "vol", environment="").environment is None
    fake = FakeModal()
    out = S.run(**_run_kw(tmp_path, runner=fake, go=True, environment="demo-env"))
    assert out["environment"] == "demo-env" and fake.calls and all(c[1] == "volume" and c[3:5] == ["--env", "demo-env"] for c in fake.calls), fake.calls[:3]
    assert len(fake.puts()) == 14 + 8 + 1
    for name in ("bank.json", "parents.json"):
        assert json.loads((tmp_path / "07_bank" / name).read_text())["environment"] == "demo-env"
    fake2 = FakeModal()
    S.run(**_run_kw(tmp_path / "noenv", runner=fake2, go=True))
    assert fake2.calls and all("--env" not in c for c in fake2.calls)
    assert json.loads((tmp_path / "noenv" / "07_bank" / "bank.json").read_text())["environment"] is None
    # the CLI: the flag, else the site's variable, else the CLI's own resolution (named in the mode line)
    env = {"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"}
    base = [sys.executable, str(TOOL), "--model", KEY, "--receipts-dir", str(tmp_path / "cli" / "07_bank"), "--modal", "/nonexistent/modal", "--registry-root", str(tmp_path / "noreg")]
    r = subprocess.run([*base, "--environment", "flag-env"], capture_output=True, text=True, timeout=600, env={**env, "SISTEER_MODAL_ENVIRONMENT": "site-env"})
    assert r.returncode == 0 and "(Modal environment flag-env)" in r.stdout, r.stdout[-1500:] + r.stderr[-1500:]
    r = subprocess.run(base, capture_output=True, text=True, timeout=600, env={**env, "SISTEER_MODAL_ENVIRONMENT": "site-env"})
    assert r.returncode == 0 and "(Modal environment site-env)" in r.stdout and json.loads((tmp_path / "cli" / "07_bank" / "dry" / "bank.json").read_text())["environment"] == "site-env"
    r = subprocess.run(base, capture_output=True, text=True, timeout=600, env=env)
    assert r.returncode == 0 and "(Modal environment the CLI default)" in r.stdout


def _tmp_repo(tmp_path: Path, plan) -> Path:
    """A copy of the committed inputs the plan reads (DEMO.json, MANIFEST.json, models.yaml + its reader, the pins, the seven object dirs — the four
    parents are four of the cells) under tmp, so ONE fact can be tampered; the dirs are copied (not linked) only where a test edits them."""
    r = tmp_path / "repo"
    (r / "objects").mkdir(parents=True); (r / "reproduce" / "lib").mkdir(parents=True); (r / "configs").mkdir()
    for rel in ("objects/DEMO.json", "objects/MANIFEST.json", "reproduce/models.yaml", "reproduce/lib/models.py", "configs/sdfbase_full22_pins.json"):
        shutil.copy(REPO / rel, r / rel)
    (r / "src").symlink_to(REPO / "src")
    dirs = {c.source_dir for c in plan.cells} | {p.source_dir for p in plan.parents}
    for d in dirs:
        rel = d.relative_to(REPO)
        (r / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(d, r / rel)
    return r


def test_each_tampered_fact_refuses_by_name_before_any_put(S, tmp_path):
    plan = S.plan_demo_bank(KEY, None, repo=REPO)
    diff = next(c for c in plan.cells if c.token == "constv2-mix-full__lora_r4_ce_minus_ctl")
    fit = next(c for c in plan.cells if c.token == "constv2ctl2-full__lora_r4_ce")
    cases = []
    # a derived sidecar naming another derived_id
    r = _tmp_repo(tmp_path / "a", plan); p = r / diff.source_dir.relative_to(REPO) / "derived_meta.json"
    d = json.loads(p.read_text()); d["derived_id"] = "someone_else"; p.write_text(json.dumps(d))
    cases.append((r, "names derived_id='someone_else'"))
    # a fit sidecar naming another cell
    r = _tmp_repo(tmp_path / "b", plan); p = r / fit.source_dir.relative_to(REPO) / "train_budget_meta.json"
    d = json.loads(p.read_text()); d["cell_id"] = "other__cell"; p.write_text(json.dumps(d))
    cases.append((r, "names cell_id='other__cell'"))
    # the object bytes not the id
    r = _tmp_repo(tmp_path / "c", plan); (r / fit.source_dir.relative_to(REPO) / "object.pt").write_bytes(b"not the committed bytes")
    cases.append((r, "not the committed object id"))
    # a second identity sidecar beside the one
    r = _tmp_repo(tmp_path / "d", plan); (r / fit.source_dir.relative_to(REPO) / "derived_meta.json").write_text("{}")
    cases.append((r, "exactly ONE identity sidecar|identity sidecars present"))
    # the paper key's pins lacking the object
    r = _tmp_repo(tmp_path / "e", plan); pins_p = r / "configs/sdfbase_full22_pins.json"; pins = json.loads(pins_p.read_text())
    pins[PAPER_KEY] = {sid: e for sid, e in pins[PAPER_KEY].items() if e.get("object_sha256") != fit.object_id}; pins_p.write_text(json.dumps(pins))
    cases.append((r, "pins no entry with object_sha256"))
    # a parent's train_budget_meta.json changed under the derivation (its sha no longer the derived block's)
    r = _tmp_repo(tmp_path / "f", plan); par = next(p for p in plan.parents if p.cell_id == "constv2-mix-full__lora_r4_ce")
    tbm = r / par.source_dir.relative_to(REPO) / "train_budget_meta.json"; tbm.write_text(tbm.read_text() + "\n")
    cases.append((r, "budget_meta_sha256"))
    # the parent's pin gone — since the one-seed DEMO.json this parent is also the constitution_lora CELL (the same bytes), whose own pin check
    # would refuse first ("pins no entry with object_sha256"); drop that cell from the tmp table's battery view so the PARENT refusal is the one reached
    r = _tmp_repo(tmp_path / "g", plan); pins_p = r / "configs/sdfbase_full22_pins.json"; pins = json.loads(pins_p.read_text())
    tbl = r / "reproduce/models.yaml"; tbl.write_text(tbl.read_text().replace("control_lora, constitution_lora, difference_lora", "control_lora, difference_lora", 1))
    pins[PAPER_KEY] = {sid: e for sid, e in pins[PAPER_KEY].items() if e.get("object_sha256") != par.object_sha256}; pins_p.write_text(json.dumps(pins))
    cases.append((r, "pinned under no staged id"))
    for repo, needle in cases:
        fake = FakeModal()
        with pytest.raises(S.Refusal, match=needle):
            S.run(**_run_kw(tmp_path, runner=fake, go=True, repo=repo, table=str(repo / "reproduce/models.yaml")))
        assert fake.calls == [], needle
        assert not (tmp_path / "07_bank").exists()


def test_the_cli_dry_run_prints_the_plan_and_the_commands_and_exits_zero(tmp_path):
    r = subprocess.run([sys.executable, str(TOOL), "--model", KEY, "--receipts-dir", str(tmp_path / "07_bank"), "--modal", "/nonexistent/modal",
                        "--registry-root", str(tmp_path / "noreg")], capture_output=True, text=True, timeout=600, env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"})
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    assert f"plan {KEY}/{MT.baseline_view(KEY)}: 7 committed object(s) x 2 files -> /objects/sdfbase/{KEY}" in out and "parents: 4 dir(s)" in out
    assert out.count("DRY  /nonexistent/modal volume put") == 14 + 8 and "staged_manifest.json   (LAST" in out and "/vol/" not in " ".join(ln for ln in out.splitlines() if "volume put" in ln or "volume ls" in ln)
    assert (tmp_path / "07_bank" / "dry" / "bank.json").is_file() and not (tmp_path / "07_bank" / "bank.json").exists()
    assert "DONE (DRY" in out and "registry check: unverified" in out
