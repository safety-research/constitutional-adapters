"""config.remote_src: a [[clusters]] entry's clone of this repository. An explicit ``remote_src`` is taken as written; without one it is
``<shared>/repos/<checkout name>/src``, the checkout name being the directory name of ``[repo] checkout`` -- the name the repo-mirror
tool keys the clone by, which the landing scripts derive from the same checkout. A checkout named ``foo`` gives repos/foo."""
from __future__ import annotations

from sisteer.registry import config as C


def _write(root, text):
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.toml").write_text(text)


def test_a_checkout_named_foo_gives_repos_foo_and_an_explicit_remote_src_wins(tmp_path):
    _write(tmp_path / "reg", '[repo]\ncheckout = "/checkouts/foo"\n\n[[clusters]]\nname = "one"\nhost = "login-one"\nshared = "/shared/one"\n\n'
                             '[[clusters]]\nname = "two"\nhost = "login-two"\nshared = "/shared/two"\nremote_src = "/elsewhere/clone/src"\n')
    cfg = C.load(str(tmp_path / "reg"))
    assert C.checkout_name(cfg) == "foo"
    one, two = cfg["clusters"]
    assert C.remote_src(one, cfg) == "/shared/one/repos/foo/src"
    assert C.remote_src(two, cfg) == "/elsewhere/clone/src"
    mirrors = {d["name"]: d["remote_src"] for d in C.mirror_destinations(cfg)}
    snaps = {d["name"]: d["remote_src"] for d in C.snapshot_destinations(cfg)}
    assert mirrors == snaps == {"one": "/shared/one/repos/foo/src", "two": "/elsewhere/clone/src"}


def test_a_trailing_slash_or_a_home_relative_checkout_names_the_same_directory(tmp_path):
    for checkout in ("/checkouts/foo/", "~/foo"):
        _write(tmp_path / checkout.strip("/~").replace("/", "_"), f'[repo]\ncheckout = "{checkout}"\n')
        cfg = C.load(str(tmp_path / checkout.strip("/~").replace("/", "_")))
        assert C.checkout_name(cfg) == "foo", checkout
