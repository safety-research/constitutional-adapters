"""05 stages a derived object by its REGISTRY id, never by directory discovery — and the pin
it stages by is written by the generator FROM THE REGISTRY OBJECT ROW, never from the staging manifest (which 05 itself produces, so a new
derived object could never have been staged the first time). Order: 04 builds + registers (+ the fold) -> `make_full22_configs.py
--pins-from <manifest> --registry-root <root>` writes the front-door derived pin from the row -> 05's cells_json carries it as
`object_sha256_expected` and stage_objects.py compares the pulled object.pt's digest to it BEFORE any cluster copy (a mismatch: refused by
name, status digest_mismatch with both digests, nothing rsynced, rc 2); pins_from_staging_manifest refuses a mismatching manifest row.
Before: the stager hashed whatever object.pt it pulled and verified the copies against THAT, and the pin rewrite promoted it — a stale but
self-consistent object.pt under derived/psr_paired_write/…/<cell> would have become the pin and served as the paper object under a
different registry id. A local host (`local:`) stands in for the cluster; a SYNTHETIC registry (the real schema, one object row) stands in
for the grading host's; synthetic bytes, placeholder ids. One test reads the LIVE registry read-only (skipped where none exists)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402
from _rsync_fallback import rsync_on_path  # noqa: E402,F401  (autouse: the fallback rsync on PATH where the host has none; a no-op where it has one)

sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))
sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "reproduce" / "lib"))
import stage_objects as SO  # noqa: E402
import make_full22_configs as GEN  # noqa: E402
import models as MT  # noqa: E402

KEY = "qwen35_4b_demo"; FIT = "qwen35_4b_demo_base"                                            # the front-door model (models.front_door_keys()) and its fit id
CELL = "constv2-mix-full__lora_r4_ce_minus_ctl"                                         # its declared derived object with a registry identity: difference_lora (object_kind lora_subtraction)
STAGED = f"derived__{CELL}"
KIND, LABEL = "lora_subtraction", "difference_lora"
RELU_CELL = "constv2-mix-full__psr_ce_all_minus_ctl"; RELU_STAGED = f"derived__{RELU_CELL}"   # its difference_relu: the composed paired write (object_kind composed; demo-import M2)
RELU_KIND = "composed"
OBJ_SHA = "0123456789abcdef" * 4                                                        # a synthetic registry id


def _manifest(bank: Path) -> Path:
    """The staging manifest rebuilt from the bank's records (what 05 hands to `--pins-from`)."""
    m = bank / "staging_manifest.json"
    SO.rebuild_manifest(bank / "_records", m)
    return m


def _roster(views=("battery", "d0p75")) -> dict:
    """A synthetic wave roster carrying the front-door model's block for `views` (the committed roster carries no block for a
    configured-not-run model, and the dose-pin writer reads dose views off the roster)."""
    import f22_roster as fr
    return {"targets": yaml.load("targets:\n" + MT.roster_block(KEY, list(views)), Loader=fr._StrictRosterLoader)["targets"]}


def _site(tmp_path: Path) -> tuple[Path, Path]:
    """A one-cluster site whose host is local: the stager's own Host over a tmp shared root."""
    site = tmp_path / "site"; (site / "clusters").mkdir(parents=True)
    shared = tmp_path / "shared"; shared.mkdir()
    (site / "config.env").write_text('SISTEER_CLUSTERS="cluster_a"\n')
    (site / "clusters" / "cluster_a.env").write_text(f'CLUSTER_SSH="local:{shared}"\nCLUSTER_SHARED_ROOT="{shared}"\nCLUSTER_PARTITION=general\n'
                                                     'CLUSTER_QOS=high\nCLUSTER_SUBSTRATE=slurm\nCLUSTER_LABEL=cluster_a\n')
    return site, shared


def _derived_source(tmp_path: Path, payload: bytes) -> tuple[Path, str]:
    d = tmp_path / "src" / CELL; d.mkdir(parents=True)
    (d / "object.pt").write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    (d / "derived_meta.json").write_text(json.dumps({"kind": KIND, "derived_id": CELL, "object_sha256": sha, "parents": ["a" * 64, "b" * 64]}))   # derived_id: the sidecar names its cell (stage_objects readiness)
    return d, sha


def _registry(tmp_path: Path, rows, name: str = "reg") -> Path:
    """A synthetic registry at the REAL schema holding exactly `rows` = [(object_id, status, object_kind, paper_label, fit_model_id[, dose[, parent ids]])] —
    the identity columns the generator reads, the strength lane and the parents a dose copy names (a real copy's
    parent_object_ids_json is [its parent]); the other NOT NULL columns are filled with placeholders (CHECKs and FKs off: this is a
    reader's fixture, not a producer's)."""
    from sisteer.registry import db as DB, schema_def as S
    root = tmp_path / name; root.mkdir(exist_ok=True)
    conn = DB.connect(DB.db_path(str(root)))
    conn.execute("PRAGMA foreign_keys=OFF"); conn.execute("PRAGMA ignore_check_constraints=ON")
    conn.executescript(S.render_schema_sql())
    for row in rows:
        oid, status, kind, label, fit = row[:5]
        dose = row[5] if len(row) > 5 else None          # A sixth element = a DOSE-SCALED COPY of the identity (derivation dose_scale, base_scale = dose)
        parents = row[6] if len(row) > 6 else None       # dose-pin-fix: a seventh = the row's parent_object_ids (a real dose copy names its parent)
        conn.execute("INSERT INTO object (object_id, object_kind, paper_label, campaign, object_role, fit_target_kind, fit_model_id, fit_thinking_render, "
                     "fit_objective, fit_positions, fit_seed_confidence, derivation, base_scale, artifact_sha256, status, record_uri, ingest_seq, parent_object_ids_json) "
                     "VALUES (?, ?, ?, 'sdfbase_full22', 'difference', 'derived', ?, 'not_applicable', 'not_applicable', 'all', 'not_applicable', "
                     "?, ?, ?, ?, 'synthetic', 1, ?)", (oid, kind, label, fit, "dose_scale" if dose is not None else "composition", dose, oid, status,
                                                        json.dumps(parents) if parents is not None else None))
    conn.commit(); conn.close()
    return root


def _run(tmp_path: Path, cells: list, site: Path) -> tuple[int, str, Path]:
    bank = tmp_path / "bank"
    cf = tmp_path / "cells.json"; cf.write_text(json.dumps(cells))
    r = subprocess.run([sys.executable, str(REPO / "scripts/sdfbase/full22/stage_objects.py"), "--site-dir", str(site), "--bank", str(bank), "--target", KEY,
                        "--cells", str(cf), "--clusters", "cluster_a", "--no-manifest"], capture_output=True, text=True, timeout=300)
    return r.returncode, r.stdout + r.stderr, bank


def test_a_derived_object_whose_bytes_are_the_registered_id_stages_and_the_record_says_the_id_matched(tmp_path):
    site, shared = _site(tmp_path)
    d, sha = _derived_source(tmp_path, b"the composed difference bytes " * 64)
    rc, out, bank = _run(tmp_path, [{"cell_id": CELL, "lane": "derived", "kind": "derived", "source": f"local:{d}", "object_sha256_expected": sha,
                                     "object_sha256_expected_source": "pins.json:test"}], site)
    assert rc == 0, out
    rec = json.loads((bank / "_records" / f"{KEY}__{STAGED}.json").read_text())
    assert rec["status"] == "staged" and rec["object_sha256"] == sha and rec["object_sha256_expected"] == sha and rec["object_sha256_expected_match"] is True
    assert rec["object_sha256_expected_source"] == "pins.json:test" and rec["cluster_a_verified"] is True
    assert (shared / "exp/objects/sdfbase/full22" / KEY / "cells" / STAGED / "object.pt").read_bytes() == d.joinpath("object.pt").read_bytes()
    # the pin the next 05 stages by is the REGISTRY row's id; the staged copy only contributes its verification facts
    pins = GEN.pins_from_staging_manifest(_manifest(bank), registry_root=_registry(tmp_path, [(sha, "reference", KIND, LABEL, FIT)]))
    assert pins[KEY][STAGED]["object_sha256"] == sha
    assert pins[KEY][STAGED]["source"].startswith(f"registry:sdfbase_full22/{FIT}/{KIND}/{LABEL}:reference (staged from local:") and pins[KEY][STAGED]["sidecar_object_sha256_recorded"] == "match"


def test_a_stale_object_pt_is_refused_by_name_before_any_copy_and_never_reaches_the_pins(tmp_path):
    site, shared = _site(tmp_path)
    d, sha = _derived_source(tmp_path, b"a stale copy that is self-consistent " * 64)       # derived_meta records ITS OWN digest: self-consistent
    rc, out, bank = _run(tmp_path, [{"cell_id": CELL, "lane": "derived", "kind": "derived", "source": f"local:{d}", "object_sha256_expected": OBJ_SHA,
                                     "object_sha256_expected_source": f"configs/sdfbase_full22_pins.json:{KEY}.{STAGED}"}], site)
    assert rc == 2 and f"REFUSED: {KEY}__{STAGED}: the pulled object.pt digest {sha[:12]} != the registered id {OBJ_SHA[:12]}" in out, out
    assert "a stale copy under local:" in out and "not staged, not pinned" in out
    rec = json.loads((bank / "_records" / f"{KEY}__{STAGED}.json").read_text())
    assert rec["status"] == "digest_mismatch" and rec["object_sha256_found"] == sha and rec["object_sha256_expected"] == OBJ_SHA and "object_sha256" not in rec
    assert not (shared / "exp/objects/sdfbase/full22" / KEY / "cells" / STAGED).exists()               # nothing rsynced
    assert not (bank / KEY / "cells" / STAGED / "staged_from.json").exists()                            # not stamped as a staged copy
    reg = _registry(tmp_path, [(OBJ_SHA, "reference", KIND, LABEL, FIT)])
    with pytest.raises(SystemExit, match=f"does not match the registered id {OBJ_SHA[:12]}"):
        GEN.pins_from_staging_manifest(_manifest(bank), registry_root=reg)                              # the pin rewrite is the last door: shut
    # a self-consistent row whose recorded digest contradicts the id it was held to refuses the same way
    bad = tmp_path / "bad_manifest.json"
    bad.write_text(json.dumps([{"target": KEY, "lane": "derived", "cell_id": CELL, "staged_id": STAGED, "object_sha256": sha, "object_sha256_expected": OBJ_SHA, "status": "staged"}]))
    with pytest.raises(SystemExit, match="refusing to rewrite the pins from it"):
        GEN.pins_from_staging_manifest(bad, registry_root=reg)
    # a row without an expected id (a first-time fit) is read as before — and a manifest with no front-door derived row needs no registry
    plain = tmp_path / "plain_manifest.json"
    plain.write_text(json.dumps([{"target": KEY, "lane": "base", "cell_id": "x__psr_ce_all", "staged_id": "base__x__psr_ce_all", "object_sha256": sha, "status": "staged"}]))
    assert GEN.pins_from_staging_manifest(plain)[KEY]["base__x__psr_ce_all"]["object_sha256"] == sha


def test_a_derived_pin_is_the_registry_rows_id_never_the_staging_manifests(tmp_path, capsys):
    """The front door's derived pins are written from the registry object row — one row pins (staged or not), zero rows pin nothing
    and say so, two rows refuse by name, a manifest copy the registry does not know or does not match refuses by name, and a read given
    no registry refuses a front-door derived manifest row rather than copying it."""
    nothing = tmp_path / "empty.json"; nothing.write_text("[]")
    # zero rows: no pin, the note names the identity and the order
    assert GEN.pins_from_staging_manifest(nothing, registry_root=_registry(tmp_path, [], "r0")) == {}
    out = capsys.readouterr().out
    assert f"{KEY}/{STAGED} (difference_lora; campaign sdfbase_full22, fit {FIT}, lora_subtraction): no reference/approved registry row yet — no pin; 04 builds + registers it" in out
    # the composed difference_relu declares its object_kind (composed): a registry identity with no row yet — no pin, said by name (demo-import M2:
    # the distilled placeholder that declared none is gone; its "object_kind UNDECLARED" note with it)
    assert f"{KEY}/{RELU_STAGED} (difference_relu; campaign sdfbase_full22, fit {FIT}, {RELU_KIND}): no reference/approved registry row yet — no pin; 04 builds + registers it" in out
    assert "object_kind UNDECLARED" not in out
    # one row, nothing staged yet: the pin is written — the first-time case 05 then stages by
    lora_id = "c" * 64
    pins = GEN.pins_from_staging_manifest(nothing, registry_root=_registry(tmp_path, [(lora_id, "approved", KIND, LABEL, FIT)], "r1"))
    assert pins[KEY][STAGED] == {"object_sha256": lora_id, "source": f"registry:sdfbase_full22/{FIT}/{KIND}/{LABEL}:approved (not staged yet)",
                                 "verified_cluster_a": False, "verified_cluster_b": False, "sidecar_object_sha256_recorded": None}
    assert RELU_STAGED not in pins[KEY]                                                              # no registry row for the composed identity: no pin
    # the identity is exact: another model's row, another kind's row, a withdrawn row do not pin this cell
    others = _registry(tmp_path, [(lora_id, "approved", KIND, LABEL, "qwen35_4b_base"), ("d" * 64, "approved", "composed", LABEL, FIT),
                                  ("e" * 64, "withdrawn", KIND, LABEL, FIT)], "r1b")
    assert GEN.pins_from_staging_manifest(nothing, registry_root=others) == {}
    # two reference/approved rows: refused by name, both ids listed
    two = _registry(tmp_path, [(OBJ_SHA, "reference", KIND, LABEL, FIT), ("a" * 64, "approved", KIND, LABEL, FIT)], "r2")
    with pytest.raises(SystemExit, match=f"2 reference/approved registry rows — {'a' * 12} \\(approved\\), {OBJ_SHA[:12]} \\(reference\\) — one object must be THE reference"):
        GEN.pins_from_staging_manifest(nothing, registry_root=two)
    # a staged copy the registry does not know: refused by name — the manifest is never a derived pin's source
    staged = tmp_path / "staged.json"
    staged.write_text(json.dumps([{"target": KEY, "lane": "derived", "cell_id": CELL, "staged_id": STAGED, "object_sha256": "b" * 64, "status": "staged",
                                   "cluster_a_verified": True, "recorded_hash_check": {"status": "match"}, "volume_path": "cluster_a:/x/lora_subtraction/cell"}]))
    with pytest.raises(SystemExit, match=f"staged \\(manifest digest {'b' * 12}\\) but the registry holds no reference/approved row — the staging manifest is never a derived pin's source"):
        GEN.pins_from_staging_manifest(staged, registry_root=_registry(tmp_path, [], "r3"))
    # a staged copy whose digest is not the registry id: refused by name (a stale copy)
    with pytest.raises(SystemExit, match=f"the staging manifest's digest {'b' * 12} is not the registry id {OBJ_SHA[:12]} \\(reference\\) — a stale staged copy"):
        GEN.pins_from_staging_manifest(staged, registry_root=_registry(tmp_path, [(OBJ_SHA, "reference", KIND, LABEL, FIT)], "r4"))
    # a staged copy that IS the registry id: pinned from the row, the copy's verification facts ride along
    good = tmp_path / "good.json"
    good.write_text(json.dumps([{"target": KEY, "lane": "derived", "cell_id": CELL, "staged_id": STAGED, "object_sha256": OBJ_SHA, "status": "staged",
                                 "cluster_a_verified": True, "cluster_b_verified": False, "recorded_hash_check": {"status": "match"}, "volume_path": "cluster_a:/x/lora_subtraction/cell"}]))
    pins = GEN.pins_from_staging_manifest(good, registry_root=_registry(tmp_path, [(OBJ_SHA, "reference", KIND, LABEL, FIT)], "r5"))
    assert pins[KEY][STAGED] == {"object_sha256": OBJ_SHA, "source": f"registry:sdfbase_full22/{FIT}/{KIND}/{LABEL}:reference (staged from cluster_a:/x/lora_subtraction/cell)",
                                 "verified_cluster_a": True, "verified_cluster_b": False, "sidecar_object_sha256_recorded": "match"}
    # a read given no registry refuses the front-door derived row rather than copying it; a registry that is not on disk refuses by name
    with pytest.raises(SystemExit, match="pinned by its REGISTRY id, never from the staging manifest .* this read was given no registry"):
        GEN.pins_from_staging_manifest(good)
    with pytest.raises(SystemExit, match="is not on disk — a derived pin is written from the REGISTRY object row"):
        GEN.pins_from_staging_manifest(nothing, registry_root=tmp_path / "absent")
    # the models table declares the registry kind of both derived cells (04's registry_done reads the same kinds): the LoRA subtraction and the
    # composed paired write (demo-import M2)
    m = MT.model(KEY, str(REPO / "reproduce" / "models.yaml"))
    assert m["objects"]["difference_lora"]["object_kind"] == KIND and m["objects"]["difference_relu"]["object_kind"] == RELU_KIND
    assert m["objects"]["difference_lora"]["cell_id"] == CELL and m["objects"]["difference_relu"]["cell_id"] == RELU_CELL
    assert m["fit"]["checkpoint"]["model_id"] == FIT
    src = (REPO / "scripts" / "sdfbase" / "make_full22_configs.py").read_text()
    assert '"--registry-root"' in src and "registry_root=args.registry_root if args.registry_root is not None else registry_root_default()" in src
    assert "status IN ('reference', 'approved')" in src                                               # the set 04_fit_objects.sh registry_object_ids reads


def _cells_json(table: Path, views: str, pins: Path) -> tuple[int, str, str]:
    src = (REPO / "reproduce" / "05_stage_roster.sh").read_text()
    body = src[src.index("<<'PYEOF'\n", src.index("cells_json()")) + len("<<'PYEOF'\n"):]
    body = body[:body.index("\nPYEOF\n")]
    r = subprocess.run([sys.executable, "-", str(REPO / "reproduce" / "lib"), str(table), KEY, views, "/shared", FIT, "cluster_a", str(pins)],
                       input=body, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def test_05_reads_the_pin_as_the_registry_id_and_refuses_a_derived_object_without_one(tmp_path):
    """05's cells_json (the heredoc 05_stage_roster.sh runs) reads a derived cell's pin as its REGISTRY id from the pins file it is
    handed, and refuses a derived object without one by name; a fit without a pin (a first staging) is still staged by discovery."""
    table = REPO / "reproduce" / "models.yaml"
    pins = tmp_path / "pins.json"; pins.write_text(json.dumps({KEY: {STAGED: {"object_sha256": OBJ_SHA, "source": f"registry:sdfbase_full22/{FIT}/{KIND}/{LABEL}:reference"}}}))
    rc, out, err = _cells_json(table, "d0p75", pins)                                                  # the dose view: difference_lora alone (none is no object)
    assert rc == 0, err
    rows = json.loads(out)
    assert len(rows) == 1 and rows[0]["object_sha256_expected"] == OBJ_SHA and rows[0]["object_sha256_expected_source"].endswith(f"pins.json:{KEY}.{STAGED}")
    # the one wave battery (the single-phase recipe) carries every arm: its three DERIVED objects (the LoRA difference, the composed ReLU difference,
    # the Arditi direction) each need their pin; the parents (fit objects) carry no pin here — a fit without a pin (a first
    # staging) is still staged by discovery, a derived one is not
    arditi_staged = "arditi__ar_c0p3__arditi_allpos"
    derived = {STAGED: OBJ_SHA, RELU_STAGED: "1" * 64, arditi_staged: "2" * 64}
    pins_s1 = tmp_path / "pins_s1.json"; pins_s1.write_text(json.dumps({KEY: {sid: {"object_sha256": sha, "source": "synthetic"} for sid, sha in derived.items()}}))
    rc, out, err = _cells_json(table, "battery", pins_s1)
    assert rc == 0, err
    by_cell = {r["cell_id"]: r for r in json.loads(out)}
    assert by_cell[CELL]["object_sha256_expected"] == OBJ_SHA and by_cell[RELU_CELL]["object_sha256_expected"] == "1" * 64 and by_cell["ar_c0p3__arditi_allpos"]["object_sha256_expected"] == "2" * 64
    assert all("object_sha256_expected" not in r for c, r in by_cell.items() if r["kind"] == "fit"), by_cell
    assert sorted(r["view_cell"] for r in by_cell.values() if r["kind"] == "fit") == ["constitution_lora", "constitution_relu", "control_lora", "control_relu"]
    rc, out, err = _cells_json(table, "battery", pins)                                                     # the wave with its Arditi / ReLU difference unpinned: refused by name
    assert rc != 0 and "has no pin" in err and "staged by its REGISTRY id" in err, err
    nopins = tmp_path / "nopins.json"; nopins.write_text(json.dumps({KEY: {}}))
    rc, out, err = _cells_json(table, "d0p75", nopins)
    assert rc != 0 and f"derived object {LABEL} ({STAGED}) has no pin" in err and "staged by its REGISTRY id" in err, err
    # the shell passes the pins path as the heredoc's argv 8 (SISTEER_PINS_FILE overrides for a staging rehearsal)
    src = (REPO / "reproduce" / "05_stage_roster.sh").read_text()
    assert '"${SISTEER_PINS_FILE:-$REPRO_REPO/configs/sdfbase_full22_pins.json}" <<\'PYEOF\'' in src
    assert "lib, table, key, views, shared, fit_id, fit_cluster, pins_path = sys.argv[1:]" in src


def test_a_dose_scaled_copy_never_competes_with_its_parent_for_the_derived_pin_and_a_dose_view_resolves_its_own(tmp_path):
    """Once a dose-scaled copy of a derived object is registered, the registry holds TWO reference/approved rows under
    (sdfbase_full22, the fit checkpoint, kind, label): the parent (reference, base_scale NULL) and its copy (approved, dose_scale,
    base_scale = the dose). The derived pin's resolver used to see both and refuse "2 rows"; now the base label resolves among base_scale
    NULL rows, a dose view (here d0p75: difference_lora at 0.75) asks for its own copy, and two rows in ONE lane are still refused by name."""
    copy = "7" * 64
    reg = _registry(tmp_path, [(OBJ_SHA, "reference", KIND, LABEL, FIT), (copy, "approved", KIND, LABEL, FIT, 0.75, [OBJ_SHA])])
    rows = GEN.derived_registry_rows(reg, fit_model_id=FIT, object_kind=KIND, paper_label=LABEL)
    assert [(r["object_id"], r["status"], r["base_scale"]) for r in rows] == [(OBJ_SHA, "reference", None)]
    rows = GEN.derived_registry_rows(reg, fit_model_id=FIT, object_kind=KIND, paper_label=LABEL, base_scale=0.75)
    assert [(r["object_id"], r["status"], r["base_scale"], r["derivation"]) for r in rows] == [(copy, "approved", 0.75, "dose_scale")]
    assert GEN.derived_registry_rows(reg, fit_model_id=FIT, object_kind=KIND, paper_label=LABEL, base_scale=0.5) == []
    nothing = tmp_path / "nothing.json"; nothing.write_text("[]")
    roster = _roster()                                                                                                     # the d0p75 view lives on the roster block (committed roster: no gemma block)
    pins = GEN.pins_from_staging_manifest(nothing, registry_root=reg, roster=roster)                                      # the pins rewrite no longer refuses
    assert pins[KEY][STAGED]["object_sha256"] == OBJ_SHA and pins[KEY][STAGED]["source"].startswith(f"registry:sdfbase_full22/{FIT}/{KIND}/{LABEL}:reference")
    assert pins[KEY][f"d0p75__{CELL}"]["object_sha256"] == copy                                                          # dose-pin-fix: the d0p75 view's copy, by the parent's identity at base_scale 0.75
    # two rows in ONE lane (a second full-strength subtraction) stay a refusal by name
    two = _registry(tmp_path, [(OBJ_SHA, "reference", KIND, LABEL, FIT), ("a" * 64, "approved", KIND, LABEL, FIT), (copy, "approved", KIND, LABEL, FIT, 0.75, [OBJ_SHA])], "two")
    with pytest.raises(SystemExit, match="2 reference/approved registry rows"):
        GEN.pins_from_staging_manifest(nothing, registry_root=two, roster=roster)


def test_the_stager_resolves_by_the_committed_pins_sha_never_by_label():
    """The stager compares the pulled object.pt to the pin's `object_sha256_expected` and never looks a registry row up by label — a dose
    copy or a second row of the identity cannot change what it stages (the one-line source invariant)."""
    src = (REPO / "scripts" / "sdfbase" / "full22" / "stage_objects.py").read_text()
    assert "object_sha256_expected" in src and "derived_registry_rows" not in src and "paper_label" not in src


def test_the_declared_parents_pick_the_lanes_row_among_same_identity_rows(capsys):
    """S2RL: two reference/approved rows share one derived identity (two compositions of the same cell); the row whose
    parent_object_ids_json is exactly the pinned ids of the object's DECLARED parents is the lane's — a lone other-lane row is never
    adopted (a note names what was seen), an entry without declared parents leaves the rows untouched, and an unpinned declared parent
    refuses by name. Synthetic rows and pins."""
    objs = {"control_relu": {"lane": "base", "cell_id": "ctl__psr"}, "constitution_relu": {"lane": "base", "cell_id": "mix__psr"},
            "difference_relu": {"lane": "derived", "cell_id": "mix__psr_minus_ctl", "parents": ["constitution_relu", "control_relu"]}}
    pins = {KEY: {"base__ctl__psr": {"object_sha256": "a" * 64}, "base__mix__psr": {"object_sha256": "b" * 64}}}
    ours = {"object_id": "1" * 64, "status": "reference", "parent_object_ids_json": json.dumps(["b" * 64, "a" * 64])}
    other = {"object_id": "2" * 64, "status": "approved", "parent_object_ids_json": json.dumps(["c" * 64, "d" * 64])}
    o = objs["difference_relu"]
    assert GEN._narrow_by_declared_parents([ours, other], o, KEY, objs, pins) == [ours]
    assert GEN._narrow_by_declared_parents([other], o, KEY, objs, pins) == []
    assert "none is this object's" in capsys.readouterr().out
    assert GEN._narrow_by_declared_parents([ours, other], {"lane": "derived", "cell_id": "x"}, KEY, objs, pins) == [ours, other]
    with pytest.raises(SystemExit, match="carries no pin"):
        GEN._narrow_by_declared_parents([ours], o, KEY, objs, {KEY: {}})
