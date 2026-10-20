"""The pins writer's COMMITTED-source path (demo-import M1; Q12 ruled NARROW; plan test j), driven through the REAL stager's output — never a
hand-built manifest row. scripts/sdfbase/full22/stage_objects.py stages a `committed:<repo-relative dir>` row (one of the paper's committed
objects, from this checkout) through a one-cluster `local:` site; the produced staging_manifest.json row carries `source_path ==
"committed:…"` and NO `source` key (the stager's output schema — a schema change fails this pin before it can hide a missed bypass); then
make_full22_configs.pins_from_staging_manifest, over a tmp registry seeded with the committed rows at their LIVE campaigns, bypasses ONE
lookup for that row — `derived_registry_rows`, the campaign='sdfbase_full22' / demo-fit-id query — so the LoRA difference's parent, whose
live campaign is `reseed`, PASSES; while a NON-committed `derived_bank` row of the same target (a `local:` source through the same stager)
still goes through that lookup and refuses when it finds nothing; a hand-written row carrying `source: committed:…` beside `source_path:
local:…` does NOT take the bypass; the digest-keyed status check and both digest refusals fire on committed rows too; and the parent-identity
refusals are KEPT, re-keyed on the digest's own live row (a copy of ANOTHER parent, at another dose, a non-dose_scale row, a difference whose
live parents are not its derived_meta.json's, the parent's pin absent from the block). Synthetic registry rows (the real schema, CHECKs and
FKs off), the committed bytes themselves (objects/qwen35_4b_base/<id>/, ~1.7 MB each)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402
from _rsync_fallback import rsync_on_path  # noqa: E402,F401  (autouse: the fallback rsync on PATH where the host has none; a no-op where it has one)
from test_stage_objects_registry_id import _site  # noqa: E402

sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))
sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "reproduce" / "lib"))
import stage_objects as SO  # noqa: E402
import make_full22_configs as GEN  # noqa: E402
import models as MT  # noqa: E402

KEY, FIT = "qwen35_4b_demo", "qwen35_4b_demo_base"
CELL = "constv2-mix-full__lora_r4_ce_minus_ctl"
STAGED, DOSE_STAGED = f"derived__{CELL}", f"d0p75__{CELL}"
KIND, LABEL = "lora_subtraction", "difference_lora"
DEMO = json.loads((REPO / "objects" / "DEMO.json").read_text())
PARENT_ID = next(e["object_id"] for e in DEMO["objects"] if e["paper_label"] == LABEL)             # the committed LoRA difference (live campaign reseed)
PARENT_DIR = REPO / "objects" / "qwen35_4b_base" / PARENT_ID
PARENT_META = json.loads((PARENT_DIR / "derived_meta.json").read_text())
GRANDPARENTS = [PARENT_META["parent_pos"]["object_sha256"], PARENT_META["parent_neg"]["object_sha256"]]
MANIFEST_ROWS = {r["object_id"]: r for r in json.loads((REPO / "objects" / "MANIFEST.json").read_text())["objects"]}
COPY_ID = next(oid for oid, r in MANIFEST_ROWS.items() if r["paper_label"] == LABEL and r["derivation"] == "dose_scale"
               and json.loads((REPO / "objects" / r["dir"] / "derived_meta.json").read_text())["dose_scaled_from"]["parent_object_sha256"] == PARENT_ID)
COPY_DIR = REPO / "objects" / MANIFEST_ROWS[COPY_ID]["dir"]


def _registry(tmp_path: Path, rows, name: str = "reg") -> Path:
    """A synthetic registry at the REAL schema holding exactly `rows`: dicts with object_id, status, object_kind, paper_label, fit_model_id,
    campaign, derivation, base_scale, parents (CHECKs and FKs off: a reader's fixture)."""
    from sisteer.registry import db as DB, schema_def as S
    root = tmp_path / name; root.mkdir(exist_ok=True)
    conn = DB.connect(DB.db_path(str(root)))
    conn.execute("PRAGMA foreign_keys=OFF"); conn.execute("PRAGMA ignore_check_constraints=ON")
    conn.executescript(S.render_schema_sql())
    for r in rows:
        conn.execute("INSERT INTO object (object_id, object_kind, paper_label, campaign, object_role, fit_target_kind, fit_model_id, fit_thinking_render, "
                     "fit_objective, fit_positions, fit_seed_confidence, derivation, base_scale, artifact_sha256, status, record_uri, ingest_seq, parent_object_ids_json) "
                     "VALUES (?, ?, ?, ?, 'difference', 'derived', ?, 'not_applicable', 'not_applicable', 'all', 'not_applicable', ?, ?, ?, ?, 'synthetic', 1, ?)",
                     (r["object_id"], r.get("object_kind", KIND), r.get("paper_label", LABEL), r.get("campaign", "reseed"), r.get("fit_model_id", "qwen35_4b_base"),
                      r.get("derivation", "lora_subtraction"), r.get("base_scale"), r["object_id"], r.get("status", "approved"),
                      json.dumps(r["parents"]) if r.get("parents") is not None else None))
    conn.commit(); conn.close()
    return root


def _parent_row(**over):
    return dict({"object_id": PARENT_ID, "campaign": "reseed", "derivation": "lora_subtraction", "parents": GRANDPARENTS}, **over)


def _copy_row(**over):
    return dict({"object_id": COPY_ID, "campaign": "sdfbase_full22", "derivation": "dose_scale", "base_scale": 0.75, "parents": [PARENT_ID]}, **over)


def _stage(tmp_path: Path, cells: list, site: Path, name: str = "bank") -> tuple:
    bank = tmp_path / name
    cf = tmp_path / f"{name}_cells.json"; cf.write_text(json.dumps(cells))
    r = subprocess.run([sys.executable, str(REPO / "scripts/sdfbase/full22/stage_objects.py"), "--site-dir", str(site), "--bank", str(bank), "--target", KEY,
                        "--cells", str(cf), "--clusters", "cluster_a", "--no-manifest"], capture_output=True, text=True, timeout=600)
    m = bank / "staging_manifest.json"
    SO.rebuild_manifest(bank / "_records", m)
    return r.returncode, r.stdout + r.stderr, bank, m


def _roster(views=("battery", "d0p75")) -> dict:
    import f22_roster as fr
    return {"targets": yaml.load("targets:\n" + MT.roster_block(KEY, list(views)), Loader=fr._StrictRosterLoader)["targets"]}


def _committed_rows(with_copy: bool = True) -> list:
    rows = [{"cell_id": CELL, "lane": "derived", "kind": "derived", "source": f"committed:objects/qwen35_4b_base/{PARENT_ID}", "object_sha256_expected": PARENT_ID,
             "object_sha256_expected_source": f"objects/DEMO.json:{LABEL}"}]
    if with_copy:
        rows.append({"cell_id": CELL, "lane": "d0p75", "kind": "derived", "source": f"committed:objects/{MANIFEST_ROWS[COPY_ID]['dir']}", "object_sha256_expected": COPY_ID,
                     "object_sha256_expected_source": f"objects/MANIFEST.json:{COPY_ID[:12]} (dose 0.75 copy of objects/DEMO.json:{LABEL})"})
    return rows


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    """The REAL stager over the committed parent and its 0.75 copy through a local site: (manifest path, its rows)."""
    tmp = tmp_path_factory.mktemp("committed")
    site, shared = _site(tmp)
    rc, out, bank, m = _stage(tmp, _committed_rows(), site)
    assert rc == 0, out
    rows = json.loads(m.read_text())
    return m, rows, bank, shared


def test_the_stager_stages_a_committed_row_from_the_checkout_and_records_it_under_source_path(staged):
    m, rows, bank, shared = staged
    by = {r["staged_id"]: r for r in rows}
    assert set(by) == {STAGED, DOSE_STAGED}
    for sid, oid in ((STAGED, PARENT_ID), (DOSE_STAGED, COPY_ID)):
        r = by[sid]
        assert r["status"] == "staged" and r["object_sha256"] == oid and r["object_sha256_expected_match"] is True
        assert r["source_path"].startswith("committed:objects/qwen35_4b_base/") and "source" not in r          # the stager's output schema
        assert r["cluster_a_verified"] is True and Path(r["local_path"]).joinpath("derived_meta.json").is_file()
        assert (shared / "exp/objects/sdfbase/full22" / KEY / "cells" / sid / "object.pt").read_bytes() == (REPO / "objects" / "qwen35_4b_base" / oid / "object.pt").read_bytes()
        stamp = json.loads((bank / KEY / "cells" / sid / "staged_from.json").read_text())
        assert stamp["source"] == r["source_path"] and stamp["object_sha256"] == oid           # the per-directory stamp spells `source`; no pins path reads it


def test_a_committed_difference_whose_live_campaign_is_reseed_passes_without_the_sdfbase_full22_lookup(staged, tmp_path, monkeypatch, capsys):
    m, rows, _bank, _shared = staged
    reg = _registry(tmp_path, [_parent_row(), _copy_row()])
    calls = []
    real = GEN.derived_registry_rows
    monkeypatch.setattr(GEN, "derived_registry_rows", lambda *a, **k: calls.append((k.get("object_kind"), k.get("paper_label"), k.get("base_scale"))) or real(*a, **k))
    pins = GEN.pins_from_staging_manifest(m, registry_root=reg, roster=_roster())
    assert pins[KEY][STAGED]["object_sha256"] == PARENT_ID and pins[KEY][DOSE_STAGED]["object_sha256"] == COPY_ID
    assert pins[KEY][STAGED]["source"].startswith(f"committed:objects/qwen35_4b_base/{PARENT_ID}") and "(registry reseed:approved)" in pins[KEY][STAGED]["source"]
    assert pins[KEY][DOSE_STAGED]["source"].startswith(f"committed:objects/qwen35_4b_base/{COPY_ID}") and f"dose-scaled copy of {PARENT_ID[:12]}" in pins[KEY][DOSE_STAGED]["source"]
    # the example site labels its cluster cluster_a, the registry label too, so the stager's cluster_a_verified reaches the pin
    assert pins[KEY][STAGED]["verified_cluster_a"] is True and pins[KEY][STAGED]["sidecar_object_sha256_recorded"] == "match"
    out = capsys.readouterr().out
    assert f"pins: {KEY}/{STAGED} ({LABEL}; campaign sdfbase_full22, fit {FIT}, {KIND}): committed {PARENT_ID[:12]}" in out and "ADDED" in out
    # the ONE bypassed lookup: never called for the committed LoRA difference or its copy; still called for the other derived cell (difference_relu, no row)
    assert all(label != LABEL for _k, label, _bs in calls), calls
    assert ("composed", "difference_relu", None) in calls
    # a committed entry with the same id is kept VERBATIM (import_objects.py pins' spelling survives a --pins-from replay)
    committed = {KEY: {STAGED: {"object_sha256": PARENT_ID, "source": f"objects/DEMO.json:{LABEL}", "verified_cluster_a": False, "verified_cluster_b": False, "sidecar_object_sha256_recorded": None},
                       DOSE_STAGED: {"object_sha256": COPY_ID, "source": "objects/MANIFEST.json: the copy", "verified_cluster_a": False, "verified_cluster_b": False, "sidecar_object_sha256_recorded": None}}}
    pins = GEN.pins_from_staging_manifest(m, registry_root=reg, roster=_roster(), committed=committed)
    assert pins[KEY][STAGED] == committed[KEY][STAGED] and pins[KEY][DOSE_STAGED] == committed[KEY][DOSE_STAGED]
    # the LoRA difference's REAL live campaign is reseed (objects/MANIFEST.json), the copy's sdfbase_full22: the demo-keyed query could find neither
    assert MANIFEST_ROWS[PARENT_ID]["campaign"] == "reseed" and MANIFEST_ROWS[COPY_ID]["campaign"] == "sdfbase_full22"
    assert real(reg, fit_model_id=FIT, object_kind=KIND, paper_label=LABEL) == []


def test_a_non_committed_row_still_needs_the_lookup_and_a_hand_written_source_key_takes_no_bypass(staged, tmp_path):
    m, rows, _bank, _shared = staged
    # the same bytes staged from a `local:` directory: a non-committed derived_bank row -> the campaign='sdfbase_full22' / demo-fit-id lookup, which finds nothing
    site, _shared2 = _site(tmp_path)
    local_src = tmp_path / "local_src" / CELL; local_src.mkdir(parents=True)
    for f in ("object.pt", "derived_meta.json"):
        (local_src / f).write_bytes((PARENT_DIR / f).read_bytes())
    rc, out, bank, lm = _stage(tmp_path, [{"cell_id": CELL, "lane": "derived", "kind": "derived", "source": f"local:{local_src}", "object_sha256_expected": PARENT_ID}], site, "local_bank")
    assert rc == 0, out
    lrow = json.loads(lm.read_text())[0]
    assert lrow["source_path"].startswith("local:") and lrow["object_sha256"] == PARENT_ID
    reg = _registry(tmp_path, [_parent_row()])
    with pytest.raises(SystemExit, match=f"staged \\(manifest digest {PARENT_ID[:12]}\\) but the registry holds no reference/approved row — the staging manifest is never a derived pin's source"):
        GEN.pins_from_staging_manifest(lm, registry_root=reg, roster=_roster(("battery",)))
    # ... and passes once a demo-keyed row exists (the fit-your-own path, unchanged)
    demo_keyed = _registry(tmp_path, [_parent_row(fit_model_id=FIT, campaign="sdfbase_full22")], "demo_keyed")
    assert GEN.pins_from_staging_manifest(lm, registry_root=demo_keyed, roster=_roster(("battery",)))[KEY][STAGED]["object_sha256"] == PARENT_ID
    # a HAND-WRITTEN row with `source: committed:` beside `source_path: local:` is NOT a committed row: the writer reads source_path alone
    hand = tmp_path / "hand.json"
    hand.write_text(json.dumps([dict(lrow, source=f"committed:objects/qwen35_4b_base/{PARENT_ID}")]))
    with pytest.raises(SystemExit, match="the staging manifest is never a derived pin's source"):
        GEN.pins_from_staging_manifest(hand, registry_root=reg, roster=_roster(("battery",)))
    assert GEN.is_committed_row({"source": "committed:x", "source_path": "local:/y"}) is False and GEN.is_committed_row({"source_path": "committed:objects/x"}) is True


def test_the_digest_keyed_checks_fire_on_committed_rows_too(staged, tmp_path):
    m, rows, _bank, _shared = staged
    # no reference/approved row for the committed digest: refused by name (kept)
    with pytest.raises(SystemExit, match=f"committed object {PARENT_ID[:12]} has no reference/approved registry row"):
        GEN.pins_from_staging_manifest(m, registry_root=_registry(tmp_path, [], "empty"), roster=_roster(("battery",)))
    with pytest.raises(SystemExit, match=f"committed object {PARENT_ID[:12]} has no reference/approved registry row \\(status 'withdrawn'\\)"):
        GEN.pins_from_staging_manifest(m, registry_root=_registry(tmp_path, [_parent_row(status="withdrawn")], "withdrawn"), roster=_roster(("battery",)))
    # a digest_mismatch row refuses before any branch (the stager's own refusal, carried)
    bad = tmp_path / "mismatch.json"
    bad.write_text(json.dumps([dict(rows[0], status="digest_mismatch", object_sha256=None, object_sha256_found="b" * 64)]))
    with pytest.raises(SystemExit, match="refusing to rewrite the pins from it"):
        GEN.pins_from_staging_manifest(bad, registry_root=_registry(tmp_path, [_parent_row()], "mm"), roster=_roster(("battery",)))
    # a committed BASE row (a plain stager row) without a live row is NOT pinned and named (S2RLFIX3, unchanged); with one, it is pinned from the manifest
    base = tmp_path / "base.json"
    base.write_text(json.dumps([{"target": KEY, "lane": "base", "cell_id": "constv2ctl2-full__psr_ce_all", "staged_id": "base__constv2ctl2-full__psr_ce_all", "object_sha256": "c" * 64,
                                 "source_path": "committed:objects/qwen35_4b_base/" + "c" * 64, "status": "staged", "recorded_hash_check": {"status": "match"}}]))
    assert GEN.pins_from_staging_manifest(base, registry_root=_registry(tmp_path, [], "nobase"), roster=_roster(("battery",))) == {}
    pins = GEN.pins_from_staging_manifest(base, registry_root=_registry(tmp_path, [{"object_id": "c" * 64, "object_kind": "relu_steer", "paper_label": "control_relu"}], "base"), roster=_roster(("battery",)))
    assert pins[KEY]["base__constv2ctl2-full__psr_ce_all"]["object_sha256"] == "c" * 64


def test_the_parent_identity_refusals_are_kept_re_keyed_on_the_digests_own_row(staged, tmp_path):
    m, rows, _bank, _shared = staged
    roster = _roster()
    # a committed difference whose live parents are not its derived_meta.json's
    with pytest.raises(SystemExit, match="names parents .*, the committed derived_meta.json's are"):
        GEN.pins_from_staging_manifest(m, registry_root=_registry(tmp_path, [_parent_row(parents=["f" * 64, "e" * 64]), _copy_row()], "wrongparents"), roster=roster)
    # a committed difference whose live row is not a difference's derivation
    with pytest.raises(SystemExit, match="has derivation 'composition'.*not a difference's|has derivation 'dose_scale', not a difference's"):
        GEN.pins_from_staging_manifest(m, registry_root=_registry(tmp_path, [_parent_row(derivation="dose_scale"), _copy_row()], "notdiff"), roster=roster)
    # a registered copy that names ANOTHER parent than the block's derived__ pin
    with pytest.raises(SystemExit, match=f"names parents \\['{'f' * 12}'\\], the pinned parent is {PARENT_ID[:12]}"):
        GEN.pins_from_staging_manifest(m, registry_root=_registry(tmp_path, [_parent_row(), _copy_row(parents=["f" * 64])], "otherparent"), roster=roster)
    # a registered copy at base_scale 0.5 under a dose: 0.75 view
    with pytest.raises(SystemExit, match="is a copy at base_scale 0.5, the view's dose is 0.75"):
        GEN.pins_from_staging_manifest(m, registry_root=_registry(tmp_path, [_parent_row(), _copy_row(base_scale=0.5)], "halfdose"), roster=roster)
    # a registered row with derivation != dose_scale under a dose staged id
    with pytest.raises(SystemExit, match="has derivation 'composition', not 'dose_scale'"):
        GEN.pins_from_staging_manifest(m, registry_root=_registry(tmp_path, [_parent_row(), _copy_row(derivation="composition")], "notdose"), roster=roster)
    # the parent's pin absent from the block: the copy staged alone, no committed entry for the parent
    site, _sh = _site(tmp_path)
    rc, out, _bank2, lonely = _stage(tmp_path, _committed_rows()[1:], site, "lonely")
    assert rc == 0, out
    with pytest.raises(SystemExit, match=f"the parent {STAGED} carries no pin in this block"):
        GEN.pins_from_staging_manifest(lonely, registry_root=_registry(tmp_path, [_parent_row(), _copy_row()], "lonelyreg"), roster=roster)
    # ... and present through the committed file (a committed pin the manifest does not carry: kept verbatim once its OWN row backs it), the copy is pinned beside it
    committed = {KEY: {STAGED: {"object_sha256": PARENT_ID, "source": f"objects/DEMO.json:{LABEL}", "verified_cluster_a": False, "verified_cluster_b": False, "sidecar_object_sha256_recorded": None}}}
    pins = GEN.pins_from_staging_manifest(lonely, registry_root=_registry(tmp_path, [_parent_row(), _copy_row()], "lonelyreg2"), roster=roster, committed=committed)
    assert pins[KEY][DOSE_STAGED]["object_sha256"] == COPY_ID and pins[KEY][STAGED] == committed[KEY][STAGED]
    # a committed pin entry whose own row is gone (withdrawn) or absent is NOT carried back as if the registry stood behind it (by the digest)
    with pytest.raises(SystemExit, match=f"the committed pin {PARENT_ID[:12]} \\(objects/DEMO.json:{LABEL}\\) has NO reference/approved registry row \\(status 'withdrawn'\\)"):
        GEN.pins_from_staging_manifest(lonely, registry_root=_registry(tmp_path, [_parent_row(status="withdrawn"), _copy_row()], "lonelyreg3"), roster=roster, committed=committed)
    with pytest.raises(SystemExit, match=f"the committed pin {PARENT_ID[:12]} \\(objects/DEMO.json:{LABEL}\\) has NO reference/approved registry row —"):
        GEN.pins_from_staging_manifest(lonely, registry_root=_registry(tmp_path, [_copy_row()], "lonelyreg4"), roster=roster, committed=committed)
    # the same for a committed COPY entry the manifest lacks: kept on its own row, refused without it
    cpin = {KEY: dict(committed[KEY], **{DOSE_STAGED: {"object_sha256": COPY_ID, "source": f"objects/MANIFEST.json:{COPY_ID[:12]} (dose 0.75 copy of objects/DEMO.json:{LABEL})",
                                                       "verified_cluster_a": False, "verified_cluster_b": False, "sidecar_object_sha256_recorded": None}})}
    nothing = tmp_path / "nothing.json"; nothing.write_text("[]")
    pins = GEN.pins_from_staging_manifest(nothing, registry_root=_registry(tmp_path, [_parent_row(), _copy_row()], "nothingreg"), roster=roster, committed=cpin)
    assert pins[KEY] == cpin[KEY]
    with pytest.raises(SystemExit, match=f"the committed pin {COPY_ID[:12]} \\(objects/MANIFEST.json:{COPY_ID[:12]}.*has NO reference/approved registry row"):
        GEN.pins_from_staging_manifest(nothing, registry_root=_registry(tmp_path, [_parent_row()], "nothingreg2"), roster=roster, committed=cpin)
    # the success case writes the copy beside its parent with every other key of a committed file byte-identical
    full = {"gemma4e4b": {"base__x": {"object_sha256": "9" * 64, "source": "twosi-fits:x", "verified_cluster_a": True, "verified_cluster_b": True, "sidecar_object_sha256_recorded": "match"}}}
    pins = GEN.pins_from_staging_manifest(m, registry_root=_registry(tmp_path, [_parent_row(), _copy_row()], "full"), roster=roster, committed=full)
    assert pins["gemma4e4b"] == full["gemma4e4b"] and set(pins[KEY]) == {STAGED, DOSE_STAGED}
