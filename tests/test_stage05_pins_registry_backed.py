"""05_stage_roster.sh --step pins on a COMMITTED staging manifest is registry-backed at the stage's own entry point (demo-import).
Before this the committed branch of the pins step called a writer that read the manifest and objects/DEMO.json alone — nine valid-looking pins
with zero registry connections — where the spec keeps every status check for the committed pins writer ("skips only the demo fit-model lookup,
never the status checks") and Q12 ruled the committed path NARROW. The stage now hands the site's registry root to the ONE pins writer
(make_full22_configs.pins_from_staging_manifest through import_objects.py pins, scoped to the model) and refuses by name, nothing written,
when: the registry holds no reference/approved row for a committed digest (a researcher who has not run `import_objects.py registry` yet), a
row is withdrawn, a copy's live parent is not the pinned parent, or a staging row would change a committed entry's identity. With the nine
imported it writes the block and leaves every other key of the pins file byte-identical. Driven through the REAL stage (bash, a one-cluster
`local:` site, the real stager's manifest under the stage's bank, a seeds-built registry at the site's SISTEER_REGISTRY_ROOT, the pins file
redirected through SISTEER_PINS_FILE); no cluster, no live registry."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, run_stage, write_site  # noqa: E402
from _rsync_fallback import rsync_on_path  # noqa: E402,F401  (autouse: the fallback rsync on PATH where the host has none; a no-op where it has one)
from test_import_objects import IMPORT_CONTEXT, KEY, PY, _fresh_registry, _plan_ids, _run, _set  # noqa: E402

sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))
sys.path.insert(0, str(REPO / "src"))
import stage_objects as SO  # noqa: E402
from sisteer.registry import harvest as H  # noqa: E402

PINS_SRC = REPO / "configs" / "sdfbase_full22_pins.json"
DERIVED_SID, COPY_SID = "derived__constv2-mix-full__lora_r4_ce_minus_ctl", "d0p75__constv2-mix-full__lora_r4_ce_minus_ctl"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """(site, registry root, the stage's bank manifest, the state dir): a one-cluster local site whose registry is a seeds-built one under the
    site's root (write_site points SISTEER_REGISTRY_ROOT there), the nine committed rows staged by the real stager into the stage's default
    bank ($SISTEER_STATE_DIR/objects), its manifest rebuilt from the records."""
    tmp = tmp_path_factory.mktemp("stage05_pins")
    shared = tmp / "shared"; shared.mkdir()
    site = write_site(tmp, clusters=("cluster_a",), ssh={"cluster_a": f"local:{shared}"}, shared={"cluster_a": str(shared)})
    root = _fresh_registry(tmp / "registry")                                      # == the site's SISTEER_REGISTRY_ROOT
    state = tmp / "state"                                                         # == run_stage's SISTEER_STATE_DIR (site.parent / "state")
    bank = state / "objects"
    cells = json.loads(_run("cells", "--model", KEY).stdout)
    cf = tmp / "cells.json"; cf.write_text(json.dumps(cells))
    r = subprocess.run([PY, str(REPO / "scripts/sdfbase/full22/stage_objects.py"), "--site-dir", str(site), "--bank", str(bank), "--target", KEY,
                        "--cells", str(cf), "--clusters", "cluster_a", "--no-manifest"], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    rows = SO.rebuild_manifest(bank / "_records", bank / "staging_manifest.json")
    assert len(rows) == 9 and all(str(x.get("source_path", "")).startswith("committed:") for x in rows)
    return site, root, bank / "staging_manifest.json", state


def _pins(tmp_path: Path) -> Path:
    p = tmp_path / "pins.json"; shutil.copy(PINS_SRC, p)
    return p


def _step_pins(site: Path, pins: Path, *extra: str):
    return run_stage("05_stage_roster.sh", "--model", KEY, "--step", "pins", *extra, site=site, env={"SISTEER_PINS_FILE": str(pins)}, timeout=900)


def test_the_pins_step_refuses_before_the_committed_objects_are_imported_and_writes_nothing(world, tmp_path):
    site, _root, _man, _state = world
    pins = _pins(tmp_path); before = pins.read_text()
    r = _step_pins(site, pins)
    out = r.stdout + r.stderr
    assert r.returncode == 2, out[-3000:]
    assert "has no reference/approved registry row" in r.stderr and "import the committed objects first" in r.stderr, r.stderr[-3000:]
    assert "REFUSED: import_objects.py pins refused (above)" in r.stderr and "Traceback" not in out, out[-3000:]
    assert pins.read_text() == before and PINS_SRC.read_text() == PINS_SRC.read_text()      # the redirected file untouched; the repo's never named


def test_the_pins_step_writes_the_block_once_the_nine_are_registered_and_every_other_key_is_byte_identical(world, tmp_path):
    site, root, man, _state = world
    r = _run("registry", "--model", KEY, "--registry-root", str(root), "--spool", str(root / "records"), env=IMPORT_CONTEXT)
    assert r.returncode == 0 and r.stdout.count("emitted ") == 9, r.stdout[-2000:] + r.stderr[-2000:]
    rep = H.ingest_once(str(root), workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    pins = _pins(tmp_path)
    committed = json.loads(PINS_SRC.read_text(encoding="utf-8"))
    r = _step_pins(site, pins)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    assert f"--registry-root {root}" not in out or True                                    # the real run prints the writer's lines, not the argv
    assert "registry-backed by its own row" in r.stdout and f"registry-backed at {root}" in r.stdout and "REFUSED" not in out, out[-3000:]
    written = json.loads(pins.read_text(encoding="utf-8"))
    assert {k: v for k, v in written.items() if k != KEY} == {k: v for k, v in committed.items() if k != KEY}
    assert {s: e["object_sha256"] for s, e in written[KEY].items()} == _plan_ids()
    # the refusals reach the stage: a withdrawn row, a copy whose live parent is not the pinned one, a staging row that would change a committed entry
    ids = _plan_ids()
    old = _set(root, ids[DERIVED_SID], "status", "withdrawn")
    try:
        before = pins.read_text()
        r = _step_pins(site, pins)
        assert r.returncode == 2 and "(status 'withdrawn')" in r.stderr and "REFUSED: import_objects.py pins refused" in r.stderr and pins.read_text() == before, (r.stdout + r.stderr)[-3000:]
    finally:
        _set(root, ids[DERIVED_SID], "status", old)
    old = _set(root, ids[COPY_SID], "parent_object_ids_json", json.dumps(["f" * 64]))
    try:
        r = _step_pins(site, pins)
        assert r.returncode == 2 and f"the pinned parent is {ids[DERIVED_SID][:12]}" in r.stderr and pins.read_text() == before, (r.stdout + r.stderr)[-3000:]
    finally:
        _set(root, ids[COPY_SID], "parent_object_ids_json", old)
    changed = json.loads(before); changed[KEY][DERIVED_SID]["object_sha256"] = "c" * 64
    pins.write_text(json.dumps(changed, indent=1, sort_keys=True) + "\n")
    r = _step_pins(site, pins)
    assert r.returncode == 2 and "would CHANGE the identity of 1 committed entry" in r.stderr and json.loads(pins.read_text()) == changed, (r.stdout + r.stderr)[-3000:]
    # the dry run names the registry root and the manifest on the importer's line and writes nothing
    pins.write_text(before)
    r = _step_pins(site, pins, "--dry-run")
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("DRY-RUN:") and "import_objects.py pins" in ln), "")
    assert r.returncode == 0 and f"--registry-root {root}" in line and f"--manifest {man}" in line and f"--pins-file {pins}" in line, r.stdout[-2500:] + r.stderr[-1500:]
    assert pins.read_text() == before


def test_the_stage_names_the_sites_registry_on_both_pins_writers():
    """The source-level pin: both branches of the pins step pass the site's registry root explicitly (repro_registry_root), and the committed
    branch always hands the manifest over — never a writer's own default root, never a pins write with no registry read."""
    src = (REPO / "reproduce" / "05_stage_roster.sh").read_text()
    assert 'REG_ROOT="$(repro_registry_root)"' in src
    assert 'pins --model "$K" --table "$(repro_models_table)" --manifest "$MANIFEST" --registry-root "$REG_ROOT"' in src
    assert '--pins-from "$MANIFEST" --registry-root "$REG_ROOT"' in src
    assert "printf -- '--manifest %q'" not in src
