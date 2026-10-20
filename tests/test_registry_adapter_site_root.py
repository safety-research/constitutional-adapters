"""Adapter store roots written with a site placeholder (sisteer.registry.adapters.base.site_root): resolved when a module loads,
never raising at import; an unresolved one keeps its placeholder text and every guarded discover()/read() in the process refuses
by name instead of reading a literal `$GRADE` directory or reporting a store as absent."""
from __future__ import annotations

import os

import pytest

from sisteer import site as S
from sisteer.registry.adapters import base as AB


@pytest.fixture
def no_unresolved(monkeypatch):
    monkeypatch.setattr(AB, "UNRESOLVED_SITE_ROOTS", {})
    S._SITE_CACHE.clear()
    yield
    S._SITE_CACHE.clear()


class _Adapter:
    name = "site_root_probe"

    def __init__(self, roots):
        self.roots = tuple(roots)
        self.read_calls = 0

    def discover(self):
        self.read_calls += 1
        return []

    def read(self, ctx, group, inputs):
        self.read_calls += 1
        return iter(())


def test_a_root_resolves_through_the_site_values(monkeypatch, tmp_path, no_unresolved):
    monkeypatch.setenv("SISTEER_GRADE_DIR", str(tmp_path / "grade"))
    assert AB.site_root("$GRADE/objects") == str(tmp_path / "grade" / "objects")
    monkeypatch.setenv("SISTEER_STAGE_DIR", "~/stage")
    assert AB.site_root("$STAGE/sdfbase") == os.path.expanduser("~/stage/sdfbase")
    assert AB.UNRESOLVED_SITE_ROOTS == {}


def test_an_unresolved_root_keeps_its_spelling_and_every_guarded_read_refuses_by_name(monkeypatch, tmp_path, no_unresolved):
    monkeypatch.delenv("SISTEER_WORK_DIR", raising=False)
    monkeypatch.setenv("SISTEER_SITE_DIR", str(tmp_path / "no-site"))
    monkeypatch.delenv("SISTEER_CONFIG", raising=False)
    assert AB.site_root("$WORK/sweep") == "$WORK/sweep"
    assert "$WORK/sweep" in AB.UNRESOLVED_SITE_ROOTS and "SISTEER_WORK_DIR" in AB.UNRESOLVED_SITE_ROOTS["$WORK/sweep"]
    # an adapter whose own roots resolved still refuses: another path an adapter derives from the site did not
    ad = AB.guard_adapter(_Adapter([str(tmp_path / "resolved")]))
    with pytest.raises(AB.AdapterRefusal, match=r"unresolved site placeholder.*SISTEER_WORK_DIR"):
        ad.discover()
    with pytest.raises(AB.AdapterRefusal, match="unresolved site placeholder"):
        ad.read(None, "g", [])
    assert ad.read_calls == 0


def test_a_declared_root_with_a_placeholder_refuses_even_when_nothing_was_recorded(no_unresolved):
    ad = AB.guard_adapter(_Adapter(["$GRADE/x"]))
    with pytest.raises(AB.AdapterRefusal, match="declared root holds an unresolved site placeholder"):
        ad.discover()
    assert ad.read_calls == 0
