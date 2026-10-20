"""Recorded values that name site storage are committed in site-placeholder form, and the readers that match them still match.

A model spelling such as ``$SHARED/psr-transfer/models/<org>--<name>`` is the portable form of the path a site's runs recorded;
the seeds' model map (``adapters.dimensions.legacy_model_map``), the decode-policy export (``PolicyFile.model_id``) and the
lineage helper accept the placeholder form and, on a host whose site resolves the placeholder, its expansion there. The
candidate-SI snapshot path expands its placeholder the same way, and its override key is read from the environment and then
from the site config. The site module is stubbed here (``site_forms._site``), so the tests hold with or without it; the last
test uses the real one when the checkout carries it. Every path below is invented.
"""
from __future__ import annotations

import json
import os
import types

import pytest
import yaml

from sisteer.registry import decode_policy as DP
from sisteer.registry import site_forms as SF
from sisteer.registry.adapters import dimensions as D
from sisteer.registry.adapters import lineage_common as LC
from sisteer.registry.adapters import si_text as SI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = os.path.join(ROOT, "configs", "registry")
EXPORTS = (os.path.join(SEEDS, "decode_policy.json"), os.path.join(ROOT, "src", "sisteer", "registry", "decode_policy.json"))


class _PlaceholderError(ValueError):
    pass


class _SiteConfigError(ValueError):
    pass


def _stub_site(values, config=None):
    """A stand-in for sisteer.site that expands the placeholders in ``values`` and refuses any other."""
    def expand(value, *a, **k):
        out = value
        for name in [n for n in ("SHARED", "STATE", "STAGE", "WORK") if "$" + n in value]:
            if name not in values:
                raise _PlaceholderError(f"${name} is not set on this host")
            out = out.replace("$" + name, values[name])
        return out
    return types.SimpleNamespace(expand=expand, PlaceholderError=_PlaceholderError, SiteConfigError=_SiteConfigError,
                                 load_site=lambda *a, **k: types.SimpleNamespace(config=dict(config or {})))


@pytest.fixture
def site(monkeypatch):
    def install(values, config=None):
        monkeypatch.setattr(SF, "_site", lambda: _stub_site(values, config))
    return install


def _seeds_with(models, model_keys=()):
    return D.Seeds(seeds_dir="<memory>", models=list(models), evals=[], item_sets=[], judges=[], instruments=[], si_texts=[],
                   decode={"policies": [], "sampling_mode_policy": [], "rulings": {}, "citations": {}},
                   legacy={"model_keys": list(model_keys)}, base_commit="unverified")


MODEL = {"model_id": "m_it", "hf_repo": "org/m-it", "paper_status": "reference",
         "producer_spellings": ["m_it_key", "$SHARED/staging/models/org--m-it"]}


def test_a_placeholder_model_spelling_resolves_in_both_forms_where_the_site_expands_it(site):
    site({"SHARED": "/data/site-a"})
    mmap = D.legacy_model_map(_seeds_with([MODEL], [{"legacy": "org/m-it | $SHARED/staging/models/org--m-it", "model_id": "m_it"}]))
    assert mmap["$SHARED/staging/models/org--m-it"] == "m_it"
    assert mmap["/data/site-a/staging/models/org--m-it"] == "m_it"      # the literal a run on this site recorded
    assert mmap["org/m-it"] == mmap["m_it_key"] == "m_it"


def test_a_placeholder_this_host_does_not_resolve_matches_only_itself(site):
    site({})
    mmap = D.legacy_model_map(_seeds_with([MODEL]))
    assert mmap["$SHARED/staging/models/org--m-it"] == "m_it"
    assert not [k for k in mmap if k.endswith("/staging/models/org--m-it") and not k.startswith("$")]


def test_an_expansion_never_takes_a_literal_another_row_spells(site):
    """The expansion is an alias of the placeholder spelling: a literal some row spells itself keeps that row's model."""
    site({"SHARED": "/data/site-a"})
    paper = {"model_id": "p_it", "hf_repo": "org/p", "paper_status": "in_paper", "producer_spellings": ["/data/site-a/staging/models/org--m-it"]}
    mmap = D.legacy_model_map(_seeds_with([MODEL, paper]))
    assert mmap["/data/site-a/staging/models/org--m-it"] == "p_it" and mmap["$SHARED/staging/models/org--m-it"] == "m_it"


def _export(spellings):
    return {"export_version": DP.EXPORT_VERSION, "policies": [], "sampling_mode_policy": [], "models": {"m_it": {}},
            "model_spellings": spellings, "evals": {}, "eval_spellings": {}, "header": {}}


def test_policy_file_resolves_the_expansion_of_a_placeholder_spelling(site):
    site({"SHARED": "/data/site-a"})
    pf = DP.PolicyFile(_export({"$SHARED/staging/models/org--m-it": "m_it", "org/m-it": "m_it"}))
    assert pf.model_id("$SHARED/staging/models/org--m-it") == "m_it"
    assert pf.model_id("/data/site-a/staging/models/org--m-it") == "m_it"
    assert pf.model_id("/elsewhere/staging/models/org--m-it") == "/elsewhere/staging/models/org--m-it"   # no guess
    site({})
    pf = DP.PolicyFile(_export({"$SHARED/staging/models/org--m-it": "m_it"}))
    assert pf.model_id("/data/site-a/staging/models/org--m-it") == "/data/site-a/staging/models/org--m-it"


def test_an_explicit_spelling_wins_over_an_expansion_that_collides_with_it(site):
    site({"SHARED": "/data/site-a"})
    pf = DP.PolicyFile(_export({"$SHARED/x": "m_it", "/data/site-a/x": "other"}))
    assert pf.model_id("/data/site-a/x") == "other"


def test_the_lineage_helper_matches_a_placeholder_spelling_by_its_expansion(site):
    site({"SHARED": "/data/site-a"})
    s = LC.Seeds(models={"m_it": dict(MODEL)})
    assert s.model_for_hf_id("/data/site-a/staging/models/org--m-it") == "m_it"
    assert s.model_for_hf_id("$SHARED/staging/models/org--m-it") == "m_it"
    assert s.model_for_hf_id("/elsewhere/staging/models/org--m-it") is None


def test_the_snapshot_path_expands_its_placeholder_and_reads_its_override_from_the_site_config(site, monkeypatch):
    ts = {"kind": "candidate_sis_snapshot", "path": "$STATE/run-state/snap.json", "env": "SISTEER_TEST_SNAP_OVERRIDE"}
    monkeypatch.delenv("SISTEER_TEST_SNAP_OVERRIDE", raising=False)
    site({"STATE": "/data/state"})
    assert SI.external_snapshot_path(ts) == "/data/state/run-state/snap.json"
    site({"STATE": "/data/state"}, config={"SISTEER_TEST_SNAP_OVERRIDE": "/data/elsewhere/snap.json"})
    assert SI.external_snapshot_path(ts) == "/data/elsewhere/snap.json"
    monkeypatch.setenv("SISTEER_TEST_SNAP_OVERRIDE", "/data/env/snap.json")
    assert SI.external_snapshot_path(ts) == "/data/env/snap.json"                 # the environment first


def test_an_unresolved_snapshot_placeholder_is_reported_as_absent_by_name(site, monkeypatch):
    ts = {"kind": "candidate_sis_snapshot", "path": "$STATE/run-state/snap.json", "env": "SISTEER_TEST_SNAP_OVERRIDE"}
    monkeypatch.delenv("SISTEER_TEST_SNAP_OVERRIDE", raising=False)
    site({})
    with pytest.raises(SI.SnapshotUnavailable, match=r"\$STATE/run-state/snap\.json is not present on this host"):
        SI.external_text_bytes("A10", ts)


def test_the_committed_seeds_and_exports_spell_site_paths_in_placeholder_form():
    """No model spelling, legacy spelling, export key or snapshot path is an absolute or home-relative path: each is in
    placeholder form, so the shipped data names no site's storage."""
    def path_like(s):
        return isinstance(s, str) and (s.startswith("/") or s.startswith("~"))
    with open(os.path.join(SEEDS, "models.yaml"), encoding="utf-8") as fh:
        models = yaml.safe_load(fh)["models"]
    bad = [sp for m in models for sp in (m.get("producer_spellings") or []) if path_like(sp)]
    with open(os.path.join(SEEDS, "legacy_spellings.yaml"), encoding="utf-8") as fh:
        legacy = yaml.safe_load(fh)
    bad += [p.strip() for r in legacy.get("model_keys") or [] for p in str(r.get("legacy", "")).split(" | ") if path_like(p.strip())]
    for path in EXPORTS:
        with open(path, encoding="utf-8") as fh:
            bad += [k for k in json.load(fh)["model_spellings"] if path_like(k)]
    with open(os.path.join(SEEDS, "si_texts.yaml"), encoding="utf-8") as fh:
        si = yaml.safe_load(fh)
    bad += [r["text_source"]["path"] for r in si.get("candidate_texts") or [] if path_like(r["text_source"]["path"])]
    assert not bad, bad
    placeholder = [sp for m in models for sp in (m.get("producer_spellings") or []) if str(sp).startswith("$SHARED/")]
    assert placeholder, "the staged-copy spelling is committed in placeholder form"
    for path in EXPORTS:
        pf = DP.PolicyFile.load(path)
        assert all(pf.model_id(sp) == next(m["model_id"] for m in models if sp in (m.get("producer_spellings") or []))
                   for sp in placeholder)


def test_the_real_site_module_expands_a_placeholder_spelling(monkeypatch, tmp_path):
    site = pytest.importorskip("sisteer.site")
    monkeypatch.delenv("SISTEER_CONFIG", raising=False)
    monkeypatch.setenv("SISTEER_SITE_DIR", str(tmp_path / "no-site"))
    monkeypatch.delenv("SISTEER_CLUSTER_LABEL", raising=False)
    monkeypatch.setenv("SISTEER_SHARED_ROOT", "/data/site-a")
    assert site is not None and SF.expansion("$SHARED/staging/models/org--m-it") == "/data/site-a/staging/models/org--m-it"
    monkeypatch.delenv("SISTEER_SHARED_ROOT")
    monkeypatch.delenv("CLUSTER_SHARED_ROOT", raising=False)
    assert SF.expansion("$SHARED/staging/models/org--m-it") is None
