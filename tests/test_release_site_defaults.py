"""Defaults that name a site's directories are read from the site, with neutral fallbacks.

Each moved default is exercised through the REAL reader the code uses (never the helper alone), under an isolated site
(``SISTEER_CONFIG`` names a temp file, every key it reads is cleared from the environment, ``HOME`` is a temp directory):

- an empty site gives the neutral fallback (``$STATE/...`` when the site sets SISTEER_STATE_ROOT, else ``~/.local/state/...``)
- the site key wins over the fallback, and the environment wins over the site
- a site value is used exactly as written (after ``~`` and placeholder expansion), so a site that names today's directories
  gets today's paths; the live site's own values are checked against the live file by the release's live-equivalence step
- no resolved path holds a ``$``

Tinycap's local wave root is ``$GRADE/tinycap``: the dispatcher expands it through ``sisteer.site`` and refuses by name
where ``$GRADE`` is unset, never creating a literal ``$GRADE`` directory.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from sisteer import site as S

REPO = Path(__file__).resolve().parents[1]
SHA = "04d5886" + "0" * 33

#: every key a moved default reads, cleared so neither the host's environment nor its site answers for a test
_KEYS = ("SISTEER_CONFIG", "SISTEER_SITE_DIR", "SISTEER_STATE_ROOT", "SISTEER_GRADE_DIR", "SISTEER_JAILBREAK_PRIVATE_ROOT",
         "SISTEER_ATTACKER_POOL_STATE_DIR", "SISTEER_PYTEST_LOCK", "SISTEER_HELDOUT_CORPUS_DIR", "TT_TWO_SI_STORE_DIR")


@pytest.fixture
def home(monkeypatch, tmp_path):
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    _site(monkeypatch, tmp_path, "")
    yield h
    S._SITE_CACHE.clear()


def _site(monkeypatch, tmp_path, text: str) -> Path:
    cfg = tmp_path / "site" / "config.env"
    cfg.parent.mkdir(exist_ok=True)
    cfg.write_text(text)
    monkeypatch.setenv("SISTEER_CONFIG", str(cfg))
    S._SITE_CACHE.clear()
    return cfg


def _load_script(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _no_dollar(*paths):
    for p in paths:
        assert "$" not in str(p), p


# --- tinycap: the dispatcher's local wave root -------------------------------------------------------------------------

@pytest.fixture
def dispatch():
    return _load_script("scripts/tinycap_ifeval_dispatch.py", "_release_tinycap_ifeval_dispatch")


def test_tinycap_local_out_root_is_the_grade_placeholder():
    from sisteer.capeval.tinycap import contracts as C
    assert C.LOCAL_OUT_ROOT == "$GRADE/tinycap"


def test_tinycap_dispatch_default_root_reads_the_site(dispatch, home, monkeypatch, tmp_path):
    _site(monkeypatch, tmp_path, "SISTEER_GRADE_DIR='~/grade-example'\n")
    got = dispatch.local_root_for(SHA)
    assert got == home / "grade-example" / "tinycap" / "ifeval_canonical" / SHA[:7]
    _no_dollar(got)
    monkeypatch.setenv("SISTEER_GRADE_DIR", str(tmp_path / "env-grade"))      # the environment wins over the site
    assert dispatch.local_root_for(SHA) == tmp_path / "env-grade" / "tinycap" / "ifeval_canonical" / SHA[:7]


def test_tinycap_dispatch_without_a_grade_root_refuses_and_creates_nothing(dispatch, home, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(S.PlaceholderError, match=r"\$GRADE.*SISTEER_GRADE_DIR"):
        dispatch.local_root_for(SHA)
    assert not any("$" in p.name for p in tmp_path.rglob("*")), "no literal $GRADE directory"


def test_tinycap_reader_is_neutral_on_a_tilde_value(dispatch, home, monkeypatch):
    """The reader moved from os.path.expanduser to sisteer.site.expand_path: on a value with no placeholder it is the same."""
    monkeypatch.setattr(dispatch.C, "LOCAL_OUT_ROOT", "~/grade-root/tinycap")
    assert dispatch.local_root_for(SHA) == Path(os.path.expanduser("~/grade-root/tinycap")) / "ifeval_canonical" / SHA[:7]


def test_tinycap_dispatcher_default_local_root_is_the_site_root(home, monkeypatch, tmp_path):
    import test_ifeval_dispatch as T
    _site(monkeypatch, tmp_path, "SISTEER_GRADE_DIR='~/grade-example'\n")
    d = T.D.Dispatcher(T.make_launcher(T.FakeRegistry()), T.MATRIX, sha=SHA, app_name=T.APP, wave_root=T.WAVE_ROOT,
                       store_complete=lambda cfg: False, log=lambda s: None, max_ticks=1)
    want = home / "grade-example" / "tinycap" / "ifeval_canonical" / SHA[:7]
    assert d.local_root == want and d.state_path.parent == want
    _no_dollar(d.local_root, d.state_path, d.spawned_log)


# --- the jailbreak private root, the held-out corpora, the labeled store -----------------------------------------------

def test_private_root_neutral_fallback_then_state_then_the_site_key_then_the_environment(home, monkeypatch, tmp_path):
    from sisteer.eval import jailbreak_artifacts as J
    assert J.grading_host_private_root() == home / ".local/state/sisteer-jailbreak"
    _site(monkeypatch, tmp_path, "SISTEER_STATE_ROOT='~/st'\n")
    assert J.grading_host_private_root() == home / "st/sisteer-jailbreak"
    _site(monkeypatch, tmp_path, "SISTEER_STATE_ROOT='~/st'\nSISTEER_JAILBREAK_PRIVATE_ROOT='~/.local/state/legacy-jailbreak'\n")
    assert J.grading_host_private_root() == home / ".local/state/legacy-jailbreak"          # the site's value, exactly
    monkeypatch.setenv("SISTEER_JAILBREAK_PRIVATE_ROOT", str(tmp_path / "env-root"))
    assert J.grading_host_private_root() == tmp_path / "env-root"
    _no_dollar(J.grading_host_private_root())


def test_private_run_dir_default_root_is_the_site_root(home, monkeypatch, tmp_path):
    from sisteer.eval import jailbreak_artifacts as J
    _site(monkeypatch, tmp_path, "SISTEER_JAILBREAK_PRIVATE_ROOT='~/jb'\n")
    d = J.private_run_dir("run-x")
    assert d == home / "jb" / "run-x" and d.is_dir() and (d.stat().st_mode & 0o777) == 0o700


def test_the_scripts_private_root_falls_to_the_site_root_off_cluster(home, monkeypatch, tmp_path):
    _site(monkeypatch, tmp_path, "SISTEER_JAILBREAK_PRIVATE_ROOT='~/jb'\n")
    for v in ("SISTEER_SHARED_ROOT", "CLUSTER_SHARED_ROOT", "SISTEER_CLUSTER_LABEL"):
        monkeypatch.delenv(v, raising=False)
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        jr = importlib.import_module("jailbreak_run")
        bsd = _load_script("scripts/build_safety_defense.py", "_release_build_safety_defense")
        ov = _load_script("scripts/two_si_overnight.py", "_release_two_si_overnight")
    finally:
        sys.path.remove(str(REPO / "scripts"))
    assert jr._private_root({}, None) == home / "jb"
    assert bsd._private_root({}, None) == home / "jb"
    assert ov.OvernightChain._default_private_root() == home / "jb"
    assert jr._private_root({}, str(tmp_path / "explicit")) == tmp_path / "explicit"           # an override still wins


def test_heldout_corpus_readers_follow_the_environment_then_the_private_root(home, monkeypatch, tmp_path):
    from sisteer.capeval.serving import run_identity
    from sisteer.eval import jailbreak_artifacts as J
    from sisteer.eval.attacks import multilingual, renellm
    base = home / ".local/state/sisteer-jailbreak/heldout_corpora"
    assert J.heldout_corpus_dir() == base and run_identity.default_corpus_dir() == base
    assert multilingual._corpus_path() == base / "multilingual.jsonl" and renellm._corpus_path() == base / "renellm.jsonl"
    _site(monkeypatch, tmp_path, "SISTEER_JAILBREAK_PRIVATE_ROOT='~/jb'\n")
    assert run_identity.default_corpus_dir() == home / "jb/heldout_corpora"
    assert renellm._corpus_path() == home / "jb/heldout_corpora/renellm.jsonl"
    monkeypatch.setenv("SISTEER_HELDOUT_CORPUS_DIR", str(tmp_path / "corpora"))            # every wrapper exports this
    assert run_identity.default_corpus_dir() == tmp_path / "corpora"
    assert multilingual._corpus_path() == tmp_path / "corpora" / "multilingual.jsonl"
    _no_dollar(renellm._corpus_path(), multilingual._corpus_path())


def test_the_heldout_scripts_read_the_same_directory(home, monkeypatch, tmp_path):
    _site(monkeypatch, tmp_path, "SISTEER_JAILBREAK_PRIVATE_ROOT='~/jb'\n")
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        hc = _load_script("scripts/heldout_compliance_check.py", "_release_heldout_compliance_check")
    finally:
        sys.path.remove(str(REPO / "scripts"))
    assert hc._heldout() == home / "jb/heldout_corpora"
    src = (REPO / "scripts/renellm_intent_judge.py").read_text()
    assert "def _heldout() -> Path:" in src and "heldout_corpus_dir()" in src
    assert "heldout_corpus_dir() / f\"{args.family}.rewrite_cache.jsonl\"" in (REPO / "scripts/build_heldout_corpus.py").read_text()


def test_the_labeled_store_default_is_under_the_private_root(home, monkeypatch, tmp_path):
    from sisteer.eval.attacks import heldout_leakage as HL
    assert HL._default_store() == home / ".local/state/sisteer-jailbreak/labeled" == HL._store_dir()
    _site(monkeypatch, tmp_path, "SISTEER_JAILBREAK_PRIVATE_ROOT='~/jb'\n")
    assert HL._store_dir() == home / "jb/labeled"
    monkeypatch.setenv("TT_TWO_SI_STORE_DIR", str(tmp_path / "store"))
    assert HL._store_dir() == tmp_path / "store"


# --- the attacker pool's grading-host records; the S0 runner's pytest lock ---------------------------------------------

def test_attacker_pool_state_dir_follows_the_site(home, monkeypatch, tmp_path):
    from test_attacker_pool import _modal_app_module
    mod = _modal_app_module()
    assert mod._local_state_dir() == home / ".local/state/sisteer/attacker-pool-modal"
    assert mod._local_endpoint_path() == home / ".local/state/sisteer/attacker-pool-modal/endpoint.json"
    _site(monkeypatch, tmp_path, "SISTEER_ATTACKER_POOL_STATE_DIR='~/.local/state/legacy/attacker-pool-modal'\n")
    assert mod._local_key_path() == home / ".local/state/legacy/attacker-pool-modal/api_key"
    monkeypatch.setenv("SISTEER_ATTACKER_POOL_STATE_DIR", str(tmp_path / "pool"))
    assert mod._local_state_dir() == tmp_path / "pool"


def test_s0_gates_parser_default_lock_follows_the_site(home, monkeypatch, tmp_path):
    S0 = _load_script("scripts/modal/seedrep_s0_gates.py", "_release_seedrep_s0_gates")
    seen = {}

    def fake_run_gates(**kw):
        seen["lock"] = kw["lock"]
        return {"gates": {}, "failed": [], "ok": True}

    monkeypatch.setattr(S0, "run_gates", fake_run_gates)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({}))
    argv = ["--tree", str(tmp_path), "--plan", str(plan), "--rs", str(tmp_path / "rs"), "--out", str(tmp_path / "s0.json")]
    S0.main(argv)
    assert seen["lock"] == home / ".local/state/sisteer-pytest/lock"
    _site(monkeypatch, tmp_path, "SISTEER_STATE_ROOT='~/st'\n")
    S0.main(argv)
    assert seen["lock"] == home / "st/sisteer-pytest/lock"
    _site(monkeypatch, tmp_path, "SISTEER_PYTEST_LOCK='~/.local/state/legacy-pytest/lock'\n")
    S0.main(argv)
    assert seen["lock"] == home / ".local/state/legacy-pytest/lock"
    S0.main(argv + ["--lock", str(tmp_path / "explicit.lock")])
    assert seen["lock"] == tmp_path / "explicit.lock"
    _no_dollar(seen["lock"])
