"""The G0DM0D3 templates, fetched at run time: a real goal slot, the single-turn fold that defines the deployed attack,
and the pointer that replaced the vendored text (pinned file, hash-verified fetch and cache, the TypeScript reader,
refusals by name, nothing at import time). Every test renders the invented stand-in
(``tests/fixtures/godmode/libertas_synthetic.ts``) with the network replaced by a failure; the one test that reads the
real upstream file runs only with ``SISTEER_GODMODE_LIVE=1``."""
from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
from pathlib import Path
from types import SimpleNamespace

import pytest

from godmode_standin import (STANDIN_ENTRIES, STANDIN_PATH, STANDIN_TEXT, godmode_standin,  # noqa: F401
                             install_standin_pins, refuse_network, standin_set_sha256)
from sisteer.eval import attacks_godmode as agm
from sisteer.eval.attacks_godmode import GODMODE_TEMPLATES, grok420, hermes_fast, render

REPO = Path(__file__).resolve().parents[1]
MODULE = REPO / "src" / "sisteer" / "eval" / "attacks_godmode.py"


def _sha(b: bytes | str) -> str:
    return hashlib.sha256(b.encode("utf-8") if isinstance(b, str) else b).hexdigest()


class _Resp(io.BytesIO):
    """A urlopen-like response: bytes, a final URL, a context manager."""

    def __init__(self, data: bytes, url: str):
        super().__init__(data)
        self._url = url

    def geturl(self):
        return self._url


def _online(monkeypatch, tmp_path, serve):
    """An EMPTY third-party root with fetching ON and ``serve(url)`` as the network (a response or an exception)."""
    root = tmp_path / "tp"
    monkeypatch.setenv(agm.THIRD_PARTY_DIR_ENV, str(root))
    monkeypatch.delenv(agm.OFFLINE_ENV, raising=False)
    calls = []

    def fake_open(url):
        calls.append(url)
        out = serve(url)
        if isinstance(out, BaseException):
            raise out
        return out

    monkeypatch.setattr(agm, "_open_url", fake_open)
    return root, root.joinpath(*agm.CACHE_RELPATH.parts), calls


def _leftovers(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()) if root.exists() else []


# --------------------------------------------------------------------------- the attack's contract (stand-in text)

def test_goal_is_spliced_and_no_slot_survives(godmode_standin):
    """An un-substituted scaffold would train against the attack's
    boilerplate with no behaviour in it -- which looks like a successful
    defense and is not one. grok420 carries {Z} twice and {QUERY} once,
    hermes_fast {QUERY} twice, as upstream does."""
    for fn, n in ((grok420, 3), (hermes_fast, 2)):
        out = fn("SENTINEL_BEHAVIOUR")
        assert out.count("SENTINEL_BEHAVIOUR") == n, fn.__name__
        assert "{QUERY}" not in out and "{Z}" not in out


def test_rendering_is_a_single_folded_turn(godmode_standin):
    """Upstream ships {system, user}; the deployed attack is the two folded
    with a blank line. An unfolded rendering measures a different attack."""
    for name in GODMODE_TEMPLATES:
        spec = STANDIN_TEXT[name]
        expected = (spec["system"] + "\n\n" + spec["user"]).replace(
            "{QUERY}", "G").replace("{Z}", "G")
        assert render(name, "G") == expected, name


def test_a_slotless_template_is_refused_and_an_unknown_key_refuses_before_any_read(monkeypatch):
    monkeypatch.setitem(GODMODE_TEMPLATES, "_fixture_noslot", {"ts_id": "x", "codename": "x"})
    monkeypatch.setattr(agm, "load_templates", lambda *a, **k: {
        "_fixture_noslot": {"ts_id": "x", "codename": "x", "system": "no slot here", "user": ""}})
    with pytest.raises(ValueError, match="no goal slot"):
        render("_fixture_noslot", "G")

    def no_read(*a, **k):
        raise AssertionError("an unknown key must refuse before the templates are read")

    monkeypatch.setattr(agm, "load_templates", no_read)
    with pytest.raises(KeyError, match="unknown G0DM0D3 template"):
        render("not_a_template", "G")


def test_load_templates_selects_the_two_entries_by_id(godmode_standin):
    """The decoy between them is never selected; codenames and ids ride along with the text."""
    assert agm.load_templates() == STANDIN_TEXT
    # a second load is the cached copy, not a second read of the file
    godmode_standin.path.write_bytes(b"replaced after the first load")
    assert agm.load_templates() == STANDIN_TEXT


# --------------------------------------------------------------------------- the TypeScript reader

def test_the_reader_cooks_every_escape_the_standin_uses():
    """\\u, \\u{}, \\x, \\\\, \\`, \\$, a line continuation, \\' and \\", a surrogate pair, a raw newline, comments, a
    boolean, and a ${} interpolation AFTER the array that the reader never reaches."""
    assert agm.parse_hall_of_fame(STANDIN_PATH.read_text(encoding="utf-8"), "standin") == STANDIN_ENTRIES


@pytest.mark.parametrize("src,why", [
    ("const HALL_OF_FAME = [ { id: 'a', system: `x ${y}` } ]", "interpolation"),
    ("const HALL_OF_FAME = [ { id: 'a', system: `x \\1` } ]", "octal"),
    ("const HALL_OF_FAME = [ { id: 'a', system: `x \\uZZZZ` } ]", "malformed \\\\u escape"),
    ("const HALL_OF_FAME = [ { id: 'a', system: `x \\uD800` } ]", "lone UTF-16 surrogate"),
    ("const HALL_OF_FAME = [ { id: 'a', system: `never closed ]", "unterminated template"),
    ("const HALL_OF_FAME = [ { id: 'a', id: 'b' } ]", "duplicate property"),
    ("const HALL_OF_FAME = [ { id: 'a', system: someIdentifier } ]", "not a string"),
    ("const HALL_OF_FAME = [ { id: 'a\nb' } ]", "line break inside a quoted string"),
    ("const OTHER = [ { id: 'a' } ]", "found 0"),
    ("const HALL_OF_FAME = []\nconst HALL_OF_FAME = []", "found 2"),
])
def test_the_reader_refuses_what_it_does_not_understand(src, why):
    with pytest.raises(agm.GodmodeTemplateError, match=why):
        agm.parse_hall_of_fame(src, "synthetic")


def test_a_missing_entry_or_field_is_refused_by_name(godmode_standin, monkeypatch):
    monkeypatch.setitem(agm.TEMPLATES, "grok420", {**agm.TEMPLATES["grok420"], "ts_id": "not-there"})
    with pytest.raises(agm.GodmodeTemplateError, match=r"'grok420' expects exactly one HALL_OF_FAME entry with id 'not-there', found 0"):
        agm.load_templates()


def test_a_field_sha_mismatch_refuses_by_template_and_field(godmode_standin, monkeypatch):
    real = agm.TEMPLATES["hermes_fast"]["user_sha256"]
    monkeypatch.setitem(agm.TEMPLATES, "hermes_fast", {**agm.TEMPLATES["hermes_fast"], "user_sha256": "0" * 64})
    with pytest.raises(agm.GodmodeTemplateError) as e:
        agm.load_templates()
    msg = str(e.value)
    assert "'hermes_fast'" in msg and "'user'" in msg and real in msg and "0" * 64 in msg


def test_a_codename_or_set_mismatch_refuses(godmode_standin, monkeypatch):
    monkeypatch.setattr(agm, "SET_SHA256", "f" * 64)
    with pytest.raises(agm.GodmodeTemplateError, match=f"template set has sha256 {standin_set_sha256()}, expected {'f' * 64}"):
        agm.load_templates()
    monkeypatch.setattr(agm, "SET_SHA256", standin_set_sha256())
    monkeypatch.setitem(agm.TEMPLATES, "grok420", {**agm.TEMPLATES["grok420"], "codename": "SOMETHING ELSE"})
    with pytest.raises(agm.GodmodeTemplateError, match="codename 'GROK 4.20', expected 'SOMETHING ELSE'"):
        agm.load_templates()


def test_the_set_hash_is_the_canonical_json_of_the_folded_templates():
    doc = {k: t["system"] + "\n\n" + t["user"] for k, t in STANDIN_TEXT.items()}
    want = _sha(json.dumps(doc, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
    assert agm.template_set_sha256(STANDIN_TEXT) == want == standin_set_sha256()


# --------------------------------------------------------------------------- the file, the cache, the fetch

def test_offline_with_no_cache_refuses_with_the_fetch_command(tmp_path, monkeypatch):
    root = tmp_path / "empty"
    monkeypatch.setenv(agm.THIRD_PARTY_DIR_ENV, str(root))
    monkeypatch.setenv(agm.OFFLINE_ENV, "1")
    monkeypatch.setattr(agm, "_open_url", refuse_network)
    with pytest.raises(agm.GodmodeTemplateError) as e:
        agm.load_templates()
    msg = str(e.value)
    for want in ("python -m sisteer.eval.attacks_godmode fetch", "reproduce/00_setup.sh --what third_party",
                 str(root.joinpath(*agm.CACHE_RELPATH.parts)), agm.THIRD_PARTY_DIR_ENV):
        assert want in msg, want
    assert _leftovers(root) == []


def test_the_cache_root_resolution_order(tmp_path, monkeypatch):
    monkeypatch.delenv(agm.THIRD_PARTY_DIR_ENV, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert agm.third_party_root() == tmp_path / "xdg" / "sisteer" / "third_party"
    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert agm.third_party_root() == tmp_path / "home" / ".cache" / "sisteer" / "third_party"
    monkeypatch.setenv(agm.THIRD_PARTY_DIR_ENV, str(tmp_path / "shared" / "third_party"))
    assert agm.cache_path() == tmp_path.joinpath("shared", "third_party", "G0DM0D3", agm.UPSTREAM_COMMIT, "src", "lib", "libertas.ts")
    assert agm.third_party_root(tmp_path / "explicit") == tmp_path / "explicit"
    for value, want in (("1", True), ("yes", True), ("", False), ("0", False), ("false", False)):
        monkeypatch.setenv(agm.OFFLINE_ENV, value)
        assert agm.offline_mode() is want, value
    assert agm.offline_mode(False) is False


def test_a_fetch_verifies_before_it_writes_and_falls_back_to_the_second_source(tmp_path, monkeypatch):
    data = STANDIN_PATH.read_bytes()
    install_standin_pins(monkeypatch, agm, data)
    root, path, calls = _online(monkeypatch, tmp_path, lambda url: (
        urllib.error.URLError("no route") if url == agm.SOURCES[0] else _Resp(data, url)))
    status, got_path, got, source = agm.ensure_file()
    assert (status, got_path, got, source) == ("FETCHED", path, data, agm.SOURCES[1])
    assert calls == list(agm.SOURCES) and path.read_bytes() == data
    assert _leftovers(root) == [str(agm.CACHE_RELPATH)]          # no temp file left beside it
    # the next call is the verified cached copy: no network
    monkeypatch.setattr(agm, "_open_url", refuse_network)
    assert agm.ensure_file()[0] == "PRESENT"
    assert agm.load_templates() == STANDIN_TEXT


def test_a_file_sha_mismatch_refuses_names_both_hashes_and_writes_nothing(tmp_path, monkeypatch):
    data = STANDIN_PATH.read_bytes()
    install_standin_pins(monkeypatch, agm, data)
    tampered = data + b"\n// one more line"
    root, path, calls = _online(monkeypatch, tmp_path, lambda url: _Resp(tampered, url))
    with pytest.raises(agm.GodmodeTemplateError) as e:
        agm.ensure_file()
    msg = str(e.value)
    assert _sha(tampered) in msg and _sha(data) in msg
    assert all(url in msg for url in agm.SOURCES) and calls == list(agm.SOURCES)
    assert not path.exists() and _leftovers(root) == []


def test_a_cached_copy_that_is_not_the_pinned_file_refuses_and_is_left_for_inspection(godmode_standin):
    tampered = godmode_standin.data + b" "
    godmode_standin.path.write_bytes(tampered)
    with pytest.raises(agm.GodmodeTemplateError) as e:
        agm.load_templates()
    msg = str(e.value)
    assert str(godmode_standin.path) in msg and _sha(tampered) in msg and _sha(godmode_standin.data) in msg
    assert "Remove it and run" in msg
    assert godmode_standin.path.read_bytes() == tampered


def test_an_interrupted_fetch_leaves_nothing(tmp_path, monkeypatch):
    data = STANDIN_PATH.read_bytes()
    install_standin_pins(monkeypatch, agm, data)

    class Cut(_Resp):
        def read(self, *a):
            raise KeyboardInterrupt

    root, path, _ = _online(monkeypatch, tmp_path, lambda url: Cut(data, url))
    with pytest.raises(KeyboardInterrupt):
        agm.ensure_file()
    assert _leftovers(root) == []
    # cut between the verified write and the rename: the temp file is removed, nothing lands
    _online(monkeypatch, tmp_path, lambda url: _Resp(data, url))

    def cut_replace(*a):
        raise KeyboardInterrupt

    monkeypatch.setattr(agm.os, "replace", cut_replace)
    with pytest.raises(KeyboardInterrupt):
        agm.ensure_file()
    assert _leftovers(root) == []


def test_sources_redirects_and_oversize_responses_are_refused(tmp_path, monkeypatch):
    for bad in ("http://raw.githubusercontent.com/x", "https://example.com/x", "file:///etc/hosts"):
        with pytest.raises(agm.GodmodeTemplateError, match="is not an https URL on"):
            agm._require_allowed(bad, why="test")
    req = SimpleNamespace(full_url=agm.SOURCES[0])
    with pytest.raises(agm.GodmodeTemplateError, match="redirect 302"):
        agm._RedirectGuard().redirect_request(req, None, 302, "Found", {}, "https://elsewhere.example/libertas.ts")
    monkeypatch.setattr(agm, "_open_url", lambda url: _Resp(b"x", "https://elsewhere.example/libertas.ts"))
    with pytest.raises(agm.GodmodeTemplateError, match="final URL"):
        agm._download(agm.SOURCES[0])
    monkeypatch.setattr(agm, "_open_url", lambda url: _Resp(b"x" * (agm._MAX_DOWNLOAD_BYTES + 1), url))
    with pytest.raises(agm.GodmodeTemplateError, match="more than"):
        agm._download(agm.SOURCES[0])
    assert {urllib.parse.urlsplit(u).hostname for u in agm.SOURCES} <= agm.ALLOWED_HOSTS


def test_importing_the_attack_reads_nothing_and_touches_no_network(tmp_path):
    """Import is offline and cheap: the registry entry, the attack module and the template module load without a
    socket, a file under the third-party root, or a template load. Run in a fresh interpreter so nothing earlier in
    the session has imported them."""
    root = tmp_path / "tp"
    root.mkdir()
    code = (
        "import socket\n"
        "def _no(*a, **k): raise AssertionError('network at import')\n"
        "socket.socket.connect = _no\n"
        "socket.create_connection = _no\n"
        "import sisteer.eval.attacks as A\n"
        "import sisteer.eval.attacks.grok420 as G\n"
        "from sisteer.eval import attacks_godmode as M\n"
        "assert callable(A.get_attack('grok420'))\n"
        "assert M._LOADED == {}, M._LOADED\n"
        "print('ok')\n")
    env = {k: v for k, v in os.environ.items() if k != agm.OFFLINE_ENV}
    env.update(PYTHONPATH=str(REPO / "src"), **{agm.THIRD_PARTY_DIR_ENV: str(root)})
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr
    assert _leftovers(root) == []


def test_the_cli_reports_verifies_and_refuses(godmode_standin, capsys, tmp_path, monkeypatch):
    assert agm.main(["fetch", "--check", "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["status"] == "PRESENT" and rep["path"] == str(godmode_standin.path)
    assert rep["relpath"] == str(agm.CACHE_RELPATH) and rep["sha256"] == _sha(godmode_standin.data)
    assert rep["set_sha256"] == standin_set_sha256()
    assert rep["upstream"] == {"repo": "elder-plinius/G0DM0D3", "commit": agm.UPSTREAM_COMMIT,
                               "path": "src/lib/libertas.ts", "license": "AGPL-3.0", "permalink": agm.PERMALINK}
    assert {k: v["folded_sha256"] for k, v in rep["templates"].items()} == {
        k: _sha(t["system"] + "\n\n" + t["user"]) for k, t in STANDIN_TEXT.items()}
    assert agm.main(["fetch"]) == 0
    assert capsys.readouterr().out.startswith("PRESENT G0DM0D3 src/lib/libertas.ts @ f6301765fb90 (AGPL-3.0)")
    assert agm.main(["fetch", "--offline", "--cache-dir", str(tmp_path / "nothing-here")]) == 2
    assert capsys.readouterr().err.startswith("REFUSED: G0DM0D3 src/lib/libertas.ts is not in the third-party cache")


# --------------------------------------------------------------------------- the pointer replaced the text

def test_attribution_and_the_not_distributed_notice_are_recorded_and_no_template_text_remains():
    """The module names its upstream and licence, says the text is not distributed with this repository, and holds
    no template text: no dict literal with system/user text, and no string constant whose sha256 is a pinned field's
    (the identity hashes are the only trace of the text)."""
    text = MODULE.read_text(encoding="utf-8")
    for needle in ("elder-plinius", "G0DM0D3", "AGPL-3.0", "libertas.ts", "not distributed with this repository",
                   "not covered by this repository's MIT licence"):
        assert needle in text, needle
    assert "PRIVATE repository" not in text
    tree = ast.parse(text)
    pinned = {v for t in agm.TEMPLATES.values() for k, v in t.items() if k.endswith("_sha256")}
    consts = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    # hermes_fast's user turn IS the bare {QUERY} slot, a placeholder spelling the module must name (_SLOTS)
    assert not [c for c in consts if _sha(c) in pinned and c not in agm._SLOTS]
    for node in ast.walk(tree):          # a template-shaped literal: a "system" or "user" key holding string text
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value in ("system", "user"):
                    assert not (isinstance(v, ast.Constant) and isinstance(v.value, str)), f"{k.value!r} text literal in the module"


def test_the_pins_are_the_upstream_commit_and_file():
    assert agm.UPSTREAM_COMMIT == "f6301765fb90eb7b336bdf365319cd2fe44b1187"
    assert agm.FILE_SHA256 == "53600aa3d83012759d186b048155f57b96d022e3f835c02a499c9205615792a5"
    assert agm.FILE_BYTES == 9281
    assert agm.SOURCES[0] == ("https://raw.githubusercontent.com/elder-plinius/G0DM0D3/"
                              "f6301765fb90eb7b336bdf365319cd2fe44b1187/src/lib/libertas.ts")
    assert agm.SOURCES[1] == ("https://archive.softwareheritage.org/api/1/content/"
                              "sha1_git:fa2ab8105db91fe2bd012ffedec6ff2326ffab50/raw/")
    assert sorted(GODMODE_TEMPLATES) == ["grok420", "hermes_fast"]
    assert {k: v["folded_sha256"] for k, v in GODMODE_TEMPLATES.items()} == {
        "grok420": "8e1a7bacec31724bbd945032da3c728c75152ff11fed59a068d172227f5fbd05",
        "hermes_fast": "bdfc1b2583618a709d3a4bfd2f985ad9ffcb677c73e8e0d73182d19e63bd3046"}


# --------------------------------------------------------------------------- the driver's preflight

def _jailbreak_run():
    if str(REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts"))
    import jailbreak_run
    return jailbreak_run


def test_the_driver_refuses_a_missing_copy_before_anything_else(tmp_path, monkeypatch):
    """``jailbreak_run generate --attack grok420`` loads the templates as its FIRST step, so a missing cache dies in
    seconds -- before the defense id check, the attacker pool, the served target or a model load."""
    jr = _jailbreak_run()
    monkeypatch.setenv(agm.THIRD_PARTY_DIR_ENV, str(tmp_path / "empty"))
    monkeypatch.setenv(agm.OFFLINE_ENV, "1")
    monkeypatch.setattr(agm, "_open_url", refuse_network)
    with pytest.raises(SystemExit) as e:
        jr.cmd_generate(SimpleNamespace(attack="grok420"), {})      # an empty cfg: nothing past the preflight could run
    assert str(e.value).startswith("[grok420] G0DM0D3 src/lib/libertas.ts is not in the third-party cache")
    assert "python -m sisteer.eval.attacks_godmode fetch" in str(e.value)


def test_the_driver_preflight_passes_with_the_templates_and_ignores_other_attacks(godmode_standin, monkeypatch):
    jr = _jailbreak_run()
    assert jr._third_party_preflight(SimpleNamespace(attack="grok420")) is None

    def no_load(*a, **k):
        raise AssertionError("only grok420 reads the G0DM0D3 templates")

    monkeypatch.setattr(agm, "load_templates", no_load)
    for attack in ("bon", "pasttense", "pair"):
        assert jr._third_party_preflight(SimpleNamespace(attack=attack)) is None


# --------------------------------------------------------------------------- the real upstream (opt-in)

@pytest.mark.skipif(os.environ.get("SISTEER_GODMODE_LIVE") != "1",
                    reason="reads the real upstream file over the network; set SISTEER_GODMODE_LIVE=1")
def test_live_the_pinned_upstream_file_and_templates_verify(tmp_path, monkeypatch):
    """Every source serves the pinned bytes, and a fetch into an empty root verifies the file and both templates."""
    monkeypatch.delenv(agm.OFFLINE_ENV, raising=False)
    for url in agm.SOURCES:
        assert _sha(agm._download(url)) == agm.FILE_SHA256, url
    status, path, data, _ = agm.ensure_file(tmp_path / "tp")
    assert status == "FETCHED" and _sha(data) == agm.FILE_SHA256 and len(data) == agm.FILE_BYTES
    templates = agm.load_templates(tmp_path / "tp")
    assert agm.template_set_sha256(templates) == agm.SET_SHA256
    for key, ident in agm.TEMPLATES.items():
        t = templates[key]
        assert (_sha(t["system"]), _sha(t["user"]), _sha(t["system"] + "\n\n" + t["user"])) == (
            ident["system_sha256"], ident["user_sha256"], ident["folded_sha256"]), key
