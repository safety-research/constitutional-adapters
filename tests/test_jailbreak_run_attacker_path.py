"""``jailbreak_run._load_attacker``: where the attacker's weights load from.

The configs write the attacker snapshot as ``$HF_HOME/hub/...``. ``_load_attacker`` expands it through
``sisteer.site.expand_path``. An expanded path that is a directory loads from that directory. A placeholder with no value
on this host means there is no local cache here, the same as an absent path or one that is not a directory, so the pinned
repo id and snapshot load instead, with a notice on stderr. A placeholder is never probed as a literal path.

Most cases use a small stand-in for ``sisteer.site`` (its ``expand_path`` and ``PlaceholderError``), so they pin
``_load_attacker``'s own decisions whichever tree they run on. One case uses the real module and is skipped where it is
absent. No case loads a model: ``sisteer.models`` is replaced by a recorder.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import jailbreak_run  # noqa: E402
import sisteer  # noqa: E402

MODEL, SNAPSHOT = "org/attacker-model", "0123456789abcdef0123456789abcdef01234567"


def _cfg(path):
    return {"dtype": "bfloat16",
            "attacker": {"model": MODEL, "snapshot": SNAPSHOT, "path": path, "attn_implementation": "sdpa"}}


@pytest.fixture
def loads(monkeypatch):
    """Every load_model_and_tokenizer call, as (args, kwargs); returns a sentinel pair so the caller sees a result."""
    calls = []
    fake = types.ModuleType("sisteer.models")

    def load_model_and_tokenizer(*args, **kwargs):
        calls.append((args, kwargs))
        return "MODEL", "TOKENIZER"

    fake.load_model_and_tokenizer = load_model_and_tokenizer
    monkeypatch.setitem(sys.modules, "sisteer.models", fake)
    return calls


def _install_site(monkeypatch, values: dict):
    """A stand-in sisteer.site: ``$NAME`` expands from ``values``; a name with no value raises PlaceholderError."""
    site = types.ModuleType("sisteer.site")

    class PlaceholderError(ValueError):
        pass

    def expand_path(value, cluster=None, *, site=None, env=None, key=None):
        if value is None:
            return None
        text = os.fspath(value)
        if text.startswith("$"):
            name, _, rest = text[1:].partition("/")
            if name not in values:
                raise PlaceholderError(f"${name} in {text!r} ({key}) is not set on this host: set {name}")
            text = os.path.join(values[name], rest) if rest else values[name]
        return Path(os.path.expanduser(text))

    site.PlaceholderError = PlaceholderError
    site.expand_path = expand_path
    monkeypatch.setitem(sys.modules, "sisteer.site", site)
    monkeypatch.setattr(sisteer, "site", site, raising=False)
    return site


def _by_repo_id(call):
    args, kwargs = call
    return args == (MODEL,) and kwargs.get("revision") == SNAPSHOT


def test_an_expanded_snapshot_directory_loads_from_that_directory(monkeypatch, tmp_path, loads):
    snap = tmp_path / "hf" / "hub" / "snap"
    snap.mkdir(parents=True)
    _install_site(monkeypatch, {"HF_HOME": str(tmp_path / "hf")})
    assert jailbreak_run._load_attacker(_cfg("$HF_HOME/hub/snap")) == ("MODEL", "TOKENIZER")
    (args, kwargs), = loads
    assert args == (str(snap),) and "revision" not in kwargs
    assert kwargs["dtype"] == "bfloat16" and kwargs["attn_implementation"] == "sdpa"


def test_a_placeholder_with_no_value_here_loads_the_pinned_repo_id_with_a_notice(monkeypatch, tmp_path, loads, capsys):
    _install_site(monkeypatch, {})
    jailbreak_run._load_attacker(_cfg("$HF_HOME/hub/snap"))
    assert len(loads) == 1 and _by_repo_id(loads[0])
    err = capsys.readouterr().err
    assert err.startswith("[attacker] $HF_HOME in '$HF_HOME/hub/snap' (attacker.path) is not set on this host")
    assert f"loading {MODEL}@{SNAPSHOT} by repo id" in err


def test_a_placeholder_is_never_probed_as_a_literal_path(monkeypatch, tmp_path, loads):
    # a directory literally named "$HF_HOME/hub/snap" under the working directory must not be loaded
    monkeypatch.chdir(tmp_path)
    (tmp_path / "$HF_HOME" / "hub" / "snap").mkdir(parents=True)
    _install_site(monkeypatch, {})
    jailbreak_run._load_attacker(_cfg("$HF_HOME/hub/snap"))
    assert len(loads) == 1 and _by_repo_id(loads[0])


@pytest.mark.parametrize("path", [None, "/nonexistent/attacker/snapshot", "$HF_HOME/hub/absent"])
def test_no_local_directory_loads_the_pinned_repo_id(monkeypatch, tmp_path, loads, path):
    _install_site(monkeypatch, {"HF_HOME": str(tmp_path / "hf")})
    jailbreak_run._load_attacker(_cfg(path))
    assert len(loads) == 1 and _by_repo_id(loads[0])


def test_the_real_site_module_expands_hf_home_or_falls_back_to_the_repo_id(monkeypatch, tmp_path, loads, capsys):
    """With the real sisteer.site, no site config and no cluster label: $HF_HOME from the environment loads the
    directory; with $HF_HOME unset the pinned repo id loads."""
    real = pytest.importorskip("sisteer.site")
    monkeypatch.setattr(sisteer, "site", real, raising=False)
    monkeypatch.setenv("SISTEER_SITE_DIR", str(tmp_path / "no-site"))
    for var in ("SISTEER_CONFIG", "SISTEER_CLUSTER_LABEL", "HF_HOME", "SISTEER_HF_HOME"):
        monkeypatch.delenv(var, raising=False)
    jailbreak_run._load_attacker(_cfg("$HF_HOME/hub/snap"))
    assert len(loads) == 1 and _by_repo_id(loads[0])
    assert "[attacker] " in capsys.readouterr().err
    snap = tmp_path / "hf" / "hub" / "snap"
    snap.mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    jailbreak_run._load_attacker(_cfg("$HF_HOME/hub/snap"))
    assert loads[1][0] == (str(snap),)
