"""The grade daemon's Petri bank-view root comes from the site's staging directory ($STAGE), never a literal home path.

  * the default is $STAGE/sdfbase/full22_objects expanded through the site (SISTEER_STAGE_DIR, from the environment or the site
    config); where no staging directory is named it stays as written
  * petri_prepare refuses a bank root that still holds a placeholder, by name and before anything is built under it
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402

sys.path.insert(0, str(REPO / "scripts" / "registry"))
import grade_daemon as GD  # noqa: E402


@pytest.fixture
def no_site(tmp_path, monkeypatch):
    monkeypatch.delenv("SISTEER_CONFIG", raising=False)
    monkeypatch.setenv("SISTEER_SITE_DIR", str(tmp_path / "no-site"))
    monkeypatch.delenv("SISTEER_STAGE_DIR", raising=False)
    return tmp_path


def test_the_default_is_named_by_the_staging_placeholder():
    assert GD.PETRI_BANK_ROOT_PLACEHOLDER == "$STAGE/sdfbase/full22_objects"
    assert "modal" + "-stage" not in Path(GD.__file__).read_text()     # assembled, so this line names no site path itself


def test_the_staging_directory_resolves_the_default(no_site, monkeypatch):
    monkeypatch.setenv("SISTEER_STAGE_DIR", str(no_site / "stage"))
    assert GD._stage_bank_root() == str(no_site / "stage" / "sdfbase" / "full22_objects")


def test_the_site_config_resolves_the_default(no_site, monkeypatch):
    cfg = no_site / "site" / "config.env"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(f"SISTEER_STAGE_DIR={no_site / 'staged'}\n")
    monkeypatch.setenv("SISTEER_CONFIG", str(cfg))
    assert GD._stage_bank_root() == str(no_site / "staged" / "sdfbase" / "full22_objects")


def test_no_staging_directory_leaves_the_placeholder(no_site):
    assert GD._stage_bank_root() == GD.PETRI_BANK_ROOT_PLACEHOLDER


def test_prepare_refuses_an_unresolved_bank_root_before_building(no_site, monkeypatch):
    bundle = no_site / "bundle"
    bundle.mkdir()
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.chdir(no_site)
    why, note = GD.petri_prepare({"model": "m", "view": "v"}, checkout=str(no_site), python="/synthetic/python-site-bank",
                                 bank_root=GD.PETRI_BANK_ROOT_PLACEHOLDER, bundle_dir=str(bundle), roster=GD.PETRI_ROSTER_DEFAULT,
                                 run=run)
    assert why and "SISTEER_STAGE_DIR" in why and "--petri-bank-root" in why and note == ""
    assert all("make_bank_view.py" not in " ".join(map(str, a)) for a in calls)      # nothing was built
    assert not (no_site / "$STAGE").exists()
