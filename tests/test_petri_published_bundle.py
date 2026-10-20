"""The published Petri scenario bundle (the frozen bundle the paper's Petri runs audited ships with the repository, byte for
byte, at data/petri/scenarios/<content hash>/, for exact reproduction).

Covers: the committed copy IS the frozen bundle (its tree digest and manifest sha256 as pinned at publication, its content hash and
per-behaviour hashes the lock's); a checkout's copy verifies by content and never as a contained store; installing it lays a mode-0700,
read-only, byte-identical store and never rewrites one; the loader's command line; the front door finding the published copy when no
store is there, and skipping the Petri family BY NAME when no copy exists anywhere (never refusing the whole inventory), while a copy
that is there and wrong still refuses; 06 dropping a skipped family from `--family all` and refusing it by name; and the grading daemon
installing the store from the wave checkout.

Every behaviour test runs on SYNTHETIC scenario files. The one test of the real copy reads hashes and counts only, never file text.
"""

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from sisteer.capeval import manifest as cm
from sisteer.capeval.petri import scenarios as psc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPRO, run_stage  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LOCK = REPO / "configs" / "capeval" / "petri-scenarios.lock.yaml"
#: pinned when the bundle was published (2026-10-19), from the cluster store and the grading host's store alike:
#: the sha256 of `find . -type f | LC_ALL=C sort | xargs sha256sum` run inside the bundle dir, and the manifest's sha256
PUBLISHED_TREE_SHA256 = "4facf82395bf464b8f7c9b87b5763dba462daced9f301542e54ee94f7b96d303"
PUBLISHED_MANIFEST_SHA256 = "0fc18cde968d74986cbe243a448319debd40b0a3ce75489a799971995333d342"
PUBLISHED_FILES = 85
BEHAVIORS = ("self_preservation", "defer_to_users")
SIDS = ("scenario_alpha", "scenario_beta")


# --- helpers ------------------------------------------------------------------------------------------------------------------


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _tree_sha256(root: Path) -> str:
    rels = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    return hashlib.sha256("".join(f"{_sha(root / r)}  ./{r}\n" for r in rels).encode()).hexdigest()


def _synthetic_frozen(tmp_path: Path) -> psc.FrozenBundle:
    src = tmp_path / "generated"
    for b in BEHAVIORS:
        sc = src / b / "scenarios"
        (sc / "seeds").mkdir(parents=True)
        (sc / "dimensions").mkdir()
        for sid in SIDS:
            (sc / "seeds" / f"{sid}.md").write_text(f"synthetic seed for {b}/{sid}\n")
        (sc / "dimensions" / f"{b}_propensity.md").write_text(f"synthetic rubric for {b}\n")
        (sc / "understanding.md").write_text(f"synthetic understanding for {b}\n")
    return psc.freeze_scenarios(src, tmp_path / "frozen", behaviors=BEHAVIORS)


def _checkout_copy(bundle_dir: Path, dest: Path) -> Path:
    """The bundle as a git checkout lays it down: the same bytes, files 0644 and dirs 0755 (git keeps no other mode)."""
    for p in sorted(bundle_dir.rglob("*")):
        t = dest / p.relative_to(bundle_dir)
        if p.is_dir():
            t.mkdir(parents=True, exist_ok=True)
        else:
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_bytes(p.read_bytes())
            os.chmod(t, 0o644)
    for p in [dest, *dest.rglob("*")]:
        if p.is_dir():
            os.chmod(p, 0o755)
    return dest


def _frozen_lock(path: Path, bundle: psc.FrozenBundle) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({
        "version": 1, "status": "frozen", "scenarios_model": "anthropic/claude-sonnet-5", "num_scenarios_per_behavior": len(SIDS),
        "generation": {"max_connections_cap": 4, "actual_max_connections": 4, "ran_at": "2026-08-18T00:00:00Z"},
        "bundle": {"location": "$SHARED/petri/scenarios/" + bundle.content_hash, "content_hash": bundle.content_hash},
        "behaviors": dict(bundle.behavior_hashes)}), encoding="utf-8")
    return path


def _modes(root: Path):
    return {p.relative_to(root).as_posix(): stat.S_IMODE(os.stat(p).st_mode) for p in [root, *root.rglob("*")]}


def _make_writable(root: Path) -> None:
    for p in [root, *root.rglob("*")]:
        os.chmod(p, 0o700 if p.is_dir() else 0o600)


# --- the committed copy -------------------------------------------------------------------------------------------------------


def test_the_published_bundle_is_the_frozen_one():
    lock = psc.read_lock(LOCK)
    h = psc.frozen_lock_hash(LOCK)
    d = psc.repo_bundle_dir(h)
    assert d == REPO / "data" / "petri" / "scenarios" / h and d.is_dir()
    assert sum(1 for p in d.rglob("*") if p.is_file()) == PUBLISHED_FILES
    assert _tree_sha256(d) == PUBLISHED_TREE_SHA256
    assert _sha(d / psc.BUNDLE_MANIFEST_NAME) == PUBLISHED_MANIFEST_SHA256
    b = psc.verify_bundle_content(d, expected_hash=h)
    assert b.content_hash == h == "8812ba083bb5473c"
    assert b.behavior_hashes == lock["behaviors"] and set(b.behavior_hashes) == set(cm.PETRI_BEHAVIORS)
    assert all(len(psc.scenario_ids(d / beh / "scenarios")) == lock["num_scenarios_per_behavior"] for beh in cm.PETRI_BEHAVIORS)
    assert not any(p.is_symlink() for p in d.rglob("*"))


# --- verify by content, install into a contained store --------------------------------------------------------------------------


def test_a_checkout_copy_verifies_by_content_but_never_as_a_contained_store(tmp_path):
    frozen = _synthetic_frozen(tmp_path)
    copy = _checkout_copy(frozen.bundle_dir, tmp_path / "checkout" / frozen.content_hash)
    with pytest.raises(psc.ScenarioError, match="0700"):
        psc.verify_bundle(copy, expected_hash=frozen.content_hash)            # a checkout is not a containment store
    got = psc.verify_bundle_content(copy, expected_hash=frozen.content_hash)
    assert got.content_hash == frozen.content_hash and got.full_file_maps == frozen.full_file_maps
    with pytest.raises(psc.ScenarioError, match="required"):
        psc.verify_bundle_content(copy, expected_hash="")
    with pytest.raises(psc.ScenarioError, match="pinned to a different"):
        psc.verify_bundle_content(copy, expected_hash="0" * 16)
    seed = copy / BEHAVIORS[0] / "scenarios" / "seeds" / f"{SIDS[0]}.md"
    seed.write_bytes(seed.read_bytes() + b"x")
    with pytest.raises(psc.ScenarioError, match="bundle drift"):
        psc.verify_bundle_content(copy, expected_hash=frozen.content_hash)


def test_install_lays_a_contained_byte_identical_store_and_never_rewrites_one(tmp_path):
    frozen = _synthetic_frozen(tmp_path)
    copy = _checkout_copy(frozen.bundle_dir, tmp_path / "checkout" / frozen.content_hash)
    root = tmp_path / "home" / ".local" / "state" / "petri" / "scenarios"
    bundle, installed = psc.install_bundle(copy, root, expected_hash=frozen.content_hash)
    store = root / frozen.content_hash
    assert installed and bundle.bundle_dir == store and bundle.content_hash == frozen.content_hash
    # the bytes of the frozen bundle, its manifest included, and nothing else
    assert _tree_sha256(store) == _tree_sha256(frozen.bundle_dir) == _tree_sha256(copy)
    modes = _modes(store)
    assert modes.pop(".") == 0o700 and all(m in (0o400, 0o500) for m in modes.values()), modes
    assert [p.name for p in root.iterdir()] == [frozen.content_hash]          # no temporary left behind
    psc.verify_bundle(store, expected_hash=frozen.content_hash)
    # a second install verifies the store and writes nothing
    before = {p: os.stat(p).st_mtime_ns for p in store.rglob("*")}
    _, installed = psc.install_bundle(copy, root, expected_hash=frozen.content_hash)
    assert not installed and before == {p: os.stat(p).st_mtime_ns for p in store.rglob("*")}
    # a store that drifted is a refusal, never a repair in place
    os.chmod(store / BEHAVIORS[0] / "scenarios" / "understanding.md", 0o644)
    with pytest.raises(psc.ScenarioError, match="permission drift"):
        psc.install_bundle(copy, root, expected_hash=frozen.content_hash)
    _make_writable(store)


def test_install_refuses_a_tampered_copy_a_moosefs_root_and_leaves_no_store(tmp_path):
    frozen = _synthetic_frozen(tmp_path)
    copy = _checkout_copy(frozen.bundle_dir, tmp_path / "checkout" / frozen.content_hash)
    with pytest.raises(psc.ScenarioError, match="MooseFS"):
        psc.install_bundle(copy, "/workspace/someone/petri/scenarios", expected_hash=frozen.content_hash)
    (copy / BEHAVIORS[1] / "scenarios" / "dimensions" / "extra.md").write_text("an added file\n")
    root = tmp_path / "store"
    with pytest.raises(psc.ScenarioError, match="bundle drift"):
        psc.install_bundle(copy, root, expected_hash=frozen.content_hash)
    assert not root.exists()


def test_a_concurrent_install_that_lands_first_is_verified_and_kept(tmp_path, monkeypatch):
    frozen = _synthetic_frozen(tmp_path)
    copy = _checkout_copy(frozen.bundle_dir, tmp_path / "checkout" / frozen.content_hash)
    root = tmp_path / "store"
    real_replace = os.replace
    raced = []

    def racing_replace(src, dst):
        if str(dst).endswith(frozen.content_hash) and not Path(dst).exists() and not raced:
            raced.append(dst)
            real_replace(src, tmp_path / "parked")                                  # this install's copy, set aside
            psc.install_bundle(copy, root, expected_hash=frozen.content_hash)       # the other install lands first
            real_replace(tmp_path / "parked", src)
            raise OSError(39, "Directory not empty")
        return real_replace(src, dst)

    monkeypatch.setattr(psc.os, "replace", racing_replace)
    bundle, installed = psc.install_bundle(copy, root, expected_hash=frozen.content_hash)
    assert raced and not installed and bundle.bundle_dir == root / frozen.content_hash
    assert [p.name for p in root.iterdir()] == [frozen.content_hash]
    _make_writable(root / frozen.content_hash)


def test_the_command_line_installs_by_the_locks_hash_and_refuses_a_misnamed_dest(tmp_path):
    frozen = _synthetic_frozen(tmp_path)
    copy = _checkout_copy(frozen.bundle_dir, tmp_path / "checkout" / frozen.content_hash)
    lock = _frozen_lock(tmp_path / "lock.yaml", frozen)
    env = dict(os.environ, PYTHONPATH=str(REPO / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    cli = [sys.executable, "-m", "sisteer.capeval.petri.scenarios"]

    def run(*args):
        return subprocess.run([*cli, *args, "--lock", str(lock)], capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=120)

    r = run("verify", "--dir", str(copy))
    assert r.returncode == 0 and f"bundle {frozen.content_hash}" in r.stdout and "2 behaviours, 8 scenario files" in r.stdout, r.stderr
    dest = tmp_path / "store" / frozen.content_hash
    r = run("install", "--dest", str(dest), "--from", str(copy))
    assert r.returncode == 0 and r.stdout.startswith("installed ") and "mode 0700" in r.stdout, r.stderr
    r = run("install", "--dest", str(dest), "--from", str(copy))
    assert r.returncode == 0 and r.stdout.startswith("verified the store already at "), r.stderr
    r = run("install", "--dest", str(tmp_path / "store" / "elsewhere"), "--from", str(copy))
    assert r.returncode == 2 and "content-addressed" in r.stderr
    # the seed text never reaches the output
    assert "synthetic seed" not in r.stdout + r.stderr
    _make_writable(dest)


# --- the front door: find the published copy, or skip Petri by name ---------------------------------------------------------------


@pytest.fixture()
def ST(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(REPRO / "lib"))
    import stage
    monkeypatch.delenv(stage.PETRI_BUNDLE_DIR_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))                           # no grading-host store
    (tmp_path / "shared").mkdir()
    monkeypatch.setenv("SISTEER_SHARED_ROOT", str(tmp_path / "shared"))          # the lock's location resolves, and holds nothing
    monkeypatch.setenv("SISTEER_SRC_DIR", str(REPO / "src"))
    return stage


def test_a_fresh_clone_freezes_the_petri_ids_from_the_published_copy(ST):
    lock = ST.petri_lock()
    h = lock["bundle"]["content_hash"]
    assert ST.petri_bundle_dir(lock) == str(REPO / "data" / "petri" / "scenarios" / h)
    ids, src = ST.petri_scenario_ids()
    assert sorted(ids) == sorted(ST.PETRI_BEHAVIORS) and all(len(v) == lock["num_scenarios_per_behavior"] for v in ids.values())
    assert f"data/petri/scenarios/{h}" in src
    assert ST.petri_skip(None, None) == (None, {})


def test_with_no_copy_anywhere_the_freeze_skips_petri_by_name(ST, tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    (checkout / "configs" / "capeval").mkdir(parents=True)
    shutil.copy(LOCK, checkout / "configs" / "capeval")
    monkeypatch.setattr(ST, "REPO", str(checkout))                                 # a checkout without data/petri
    with pytest.raises(ST.PetriBundleAbsent) as ei:
        ST.petri_bundle_dir(ST.petri_lock())
    assert str(checkout / "data" / "petri" / "scenarios") in str(ei.value)
    fams, skipped = ST.petri_skip(None, None)
    assert fams == [f for f in ST.FAMILIES if f != "petri"] and list(skipped) == ["petri"] and "not materialized here" in skipped["petri"]
    assert ST.petri_skip(["static", "mmlu"], None) == (["static", "mmlu"], {})
    with pytest.raises(RuntimeError, match="--families names petri"):
        ST.petri_skip(["static", "petri"], None)
    # a NAMED dir that is absent is not a skip: expected_units refuses it by name (tests/test_admission.py)
    assert ST.petri_skip(None, str(tmp_path / "absent")) == (None, {})


def test_a_published_copy_that_does_not_verify_refuses_and_is_never_skipped(ST, tmp_path, monkeypatch):
    frozen = _synthetic_frozen(tmp_path)
    checkout = tmp_path / "checkout"
    _frozen_lock(checkout / "configs" / "capeval" / "petri-scenarios.lock.yaml", frozen)
    copy = _checkout_copy(frozen.bundle_dir, checkout / "data" / "petri" / "scenarios" / frozen.content_hash)
    monkeypatch.setattr(ST, "REPO", str(checkout))
    assert ST.petri_bundle_dir(ST.petri_lock()) == str(copy)
    (copy / BEHAVIORS[0] / "scenarios" / "seeds" / f"{SIDS[1]}.md").write_text("changed\n")
    with pytest.raises(RuntimeError, match="does not verify") as ei:
        ST.petri_bundle_dir(ST.petri_lock())
    assert not isinstance(ei.value, ST.PetriBundleAbsent)
    assert ST.petri_skip(None, None) == (None, {})


def _freeze_args(tmp_path: Path) -> list:
    goals = tmp_path / "goals.jsonl"
    goals.write_text("".join(json.dumps({"goal_id": f"goal-{i:03d}", "dataset": "jbb" if i < 100 else "harmbench"}) + "\n" for i in range(259)))
    wave = tmp_path / "wave.yaml"
    wave.write_text("private_run: sdfbase_full22_qwen35_4b_demo_battery\nbon:\n  n_max: 75\n  goal_subset: jbb\n")
    return ["freeze-inventory", "--model", "qwen35_4b_demo", "--models-table", str(REPRO / "models.yaml"), "--code-sha", "a" * 40,
            "--out", str(tmp_path / "state" / "gate" / "expected_units.json"), "--goals-jsonl", str(goals), "--wave-config", str(wave)]


def test_the_freeze_records_the_skip_in_the_inventory_and_otherwise_freezes_the_published_ids(ST, tmp_path, monkeypatch, capsys):
    (tmp_path / "state" / "gate").mkdir(parents=True)
    out = tmp_path / "state" / "gate" / "expected_units.json"
    assert ST.main(_freeze_args(tmp_path)) == 0, capsys.readouterr().err[-2000:]
    doc = json.loads(out.read_text())
    petri = [u for u in doc["units"] if u["family"] == "petri"]
    assert petri and "skipped_families" not in doc
    n = sum(len(ids) for u in petri for ev in u["evals"].values() if ev["coverage"]["layout"] == "petri_cellstore"
            and ev["coverage"].get("behavior") for ids in (ev["coverage"].get("expected") or {}).values())
    assert n == 60, n                                                              # 6 behaviours x 10 scenarios, from the published copy
    pooled = [ev["coverage"]["expected"]["shard00"] for u in petri for e, ev in u["evals"].items() if e == "petri_pooled"]
    assert len(pooled) == 1 and len(set(pooled[0])) == 60
    # the same freeze where no copy exists: the Petri family skipped by name, recorded, every other unit frozen
    monkeypatch.setattr(ST, "petri_repo_bundle_dir", lambda h: str(tmp_path / "no-copy" / h))
    out.unlink()
    capsys.readouterr()
    assert ST.main(_freeze_args(tmp_path)) == 0
    err = capsys.readouterr().err
    assert "SKIPPED: freeze-inventory: the petri family is left out of this inventory" in err, err[-2000:]
    skipped_doc = json.loads(out.read_text())
    assert list(skipped_doc["skipped_families"]) == ["petri"] and "not materialized here" in skipped_doc["skipped_families"]["petri"]
    assert not any(u["family"] == "petri" for u in skipped_doc["units"])
    assert {u["logical_work_id"] for u in skipped_doc["units"]} == {u["logical_work_id"] for u in doc["units"] if u["family"] != "petri"}
    # named explicitly, the family refuses instead
    out.unlink()
    assert ST.main(_freeze_args(tmp_path) + ["--families", "petri"]) == 2
    assert "--families names petri" in capsys.readouterr().err


# --- 06: a skipped family leaves `--family all` and refuses by name -----------------------------------------------------------------


@pytest.fixture()
def site(tmp_path):
    from front_door_testlib import VENV_PY
    site = tmp_path / "site"
    shutil.copytree(REPRO / "site.example", site)
    with open(site / "config.env", "a") as fh:
        fh.write(f"\nSISTEER_STATE_DIR={tmp_path / 'state'}\nSISTEER_PYTHON={VENV_PY}\nSISTEER_REGISTRY_ROOT={tmp_path / 'registry'}\n")
    return site


def _inventory(site: Path, skipped: dict) -> None:
    inv = site.parent / "state" / "qwen35_4b_demo" / "gate" / "expected_units.json"
    inv.parent.mkdir(parents=True, exist_ok=True)
    doc = {"schema": 2, "model": "qwen35_4b_demo", "view": "battery", "code_sha": "a" * 40, "units": []}
    if skipped:
        doc["skipped_families"] = skipped
    inv.write_text(json.dumps(doc))


def test_06_drops_a_family_the_freeze_skipped_and_refuses_it_by_name(site):
    why = "the frozen Petri scenario bundle 8812ba083bb5473c is not materialized here (synthetic)"
    # an EMPTY inventory makes the plan comparison list every unit the plan holds as 'planned but not frozen'
    _inventory(site, {})
    r = run_stage("06_eval_waves.sh", "--dry-run", "--model", "qwen35_4b_demo", "--view", "battery", "--family", "all", site=site, timeout=600)
    control = r.stdout + r.stderr
    assert "/petri/" in control and "planned but not frozen" in control, control[-3000:]
    _inventory(site, {"petri": why})
    r = run_stage("06_eval_waves.sh", "--dry-run", "--model", "qwen35_4b_demo", "--view", "battery", "--family", "all", site=site, timeout=600)
    out = r.stdout + r.stderr
    assert f"family petri was skipped at the freeze ({why}) — dropped from --family all" in out, out[-3000:]
    assert "planned but not frozen" in out and "/petri/" not in out, out[-3000:]
    r = run_stage("06_eval_waves.sh", "--dry-run", "--model", "qwen35_4b_demo", "--view", "battery", "--family", "petri", site=site, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 2 and f"family petri was skipped at the freeze ({why}) — it has no frozen unit to submit" in out, out[-3000:]


# --- 08: the grading daemon installs the store from the wave checkout -------------------------------------------------------------


def test_the_daemon_installs_the_store_from_the_checkouts_copy(tmp_path):
    sys.path.insert(0, str(REPO / "scripts" / "registry"))
    import grade_daemon as GD
    frozen = _synthetic_frozen(tmp_path)
    checkout = tmp_path / "checkout"
    _frozen_lock(checkout / "configs" / "capeval" / "petri-scenarios.lock.yaml", frozen)
    _checkout_copy(frozen.bundle_dir, checkout / "data" / "petri" / "scenarios" / frozen.content_hash)
    (checkout / "src").symlink_to(REPO / "src")
    store = tmp_path / "home" / ".local" / "state" / "petri" / "scenarios" / frozen.content_hash
    assert GD.petri_install_bundle(checkout=str(checkout), python=sys.executable, bundle_dir=str(store)) is None
    psc.verify_bundle(store, expected_hash=frozen.content_hash)
    assert _tree_sha256(store) == _tree_sha256(frozen.bundle_dir)
    # a store named for another hash, or a checkout that publishes no copy, is a reason, never an attempt
    other = tmp_path / "home" / "other" / ("0" * 16)
    why = GD.petri_install_bundle(checkout=str(checkout), python=sys.executable, bundle_dir=str(other))
    assert why.startswith(f"no Petri scenario bundle at {other}") and "publishes none" in why
    shutil.rmtree(checkout / "data")
    store2 = tmp_path / "home2" / frozen.content_hash
    why = GD.petri_install_bundle(checkout=str(checkout), python=sys.executable, bundle_dir=str(store2))
    assert why.startswith(f"no Petri scenario bundle at {store2}") and not store2.exists()
    _make_writable(store)
