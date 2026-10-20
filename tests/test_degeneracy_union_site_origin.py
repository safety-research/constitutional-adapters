"""The rep_ratio definition origin is named through the site's staging directory (``$STAGE``), not a literal home path.

  * with SISTEER_STAGE_DIR set, the origin resolves under it: a file there is read and its sha256 checked against the pin
  * with no site value for ``$STAGE``, the origin stays as written and counts as a source outside the repository that is
    not on this host: present False, ok None, and the producers' hard stop does not refuse it
  * an in-repo pin is still judged in-repo (a placeholder path is never mistaken for a committed file)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from sisteer.eval import degeneracy_union as du

ORIGIN_KEY = "rep_ratio_loop.definition_origin"
REL = "sdfbase/fullscale/full22_join_scores.py"


@pytest.fixture
def no_site(tmp_path, monkeypatch):
    """No site config on this host and no staging directory in the environment."""
    monkeypatch.delenv("SISTEER_CONFIG", raising=False)
    monkeypatch.setenv("SISTEER_SITE_DIR", str(tmp_path / "no-site"))
    monkeypatch.delenv("SISTEER_STAGE_DIR", raising=False)
    return tmp_path


def test_origin_is_named_by_the_staging_placeholder():
    path = du.DEFINITION_SOURCES["rep_ratio_loop"]["definition_origin"]["path"]
    assert path == f"$STAGE/{REL}"


def test_origin_resolves_under_the_site_staging_directory(no_site, monkeypatch):
    stage = no_site / "stage"
    origin = stage / REL
    origin.parent.mkdir(parents=True)
    origin.write_text("def rep_ratio(t):\n    return 0.0\n")
    monkeypatch.setenv("SISTEER_STAGE_DIR", str(stage))
    assert du._resolve(f"$STAGE/{REL}") == origin
    rec = du.verify_definition_sources()[ORIGIN_KEY]
    assert rec["path"] == str(origin) and rec["present"] is True
    assert rec["actual"] == hashlib.sha256(origin.read_bytes()).hexdigest()
    assert rec["ok"] is False          # a present origin off its pin is drift, as before


def test_unresolved_origin_is_unverified_and_never_refused(no_site):
    assert du._resolve(f"$STAGE/{REL}") == Path(f"$STAGE/{REL}")
    record = du.verify_definition_sources()
    rec = record[ORIGIN_KEY]
    assert rec["present"] is False and rec["ok"] is None and rec["path"] == f"$STAGE/{REL}"
    # the hard stop judges only the other entries: whatever it decides, this one is not among the refused
    try:
        du.definition_sources_or_refuse(record)
    except du.DefinitionSourcesUnverified as exc:
        assert ORIGIN_KEY not in exc.failed


def test_in_repo_pins_stay_in_repo(no_site):
    rec = du.verify_definition_sources()["rep_ratio_loop"]
    assert rec["path"] == str(du.REPO / "src/sisteer/eval/rep_ratio.py")
