"""qwen-seedrep-native-battery — F-Y: the compile-cache SEED (`seed_compile_caches` in scripts/modal/seedrep_served.py,
`shutil.copytree(/vol/seedrep/cache/<name> -> /root/.cache/<name>)`) runs inside the attempt-evidence literal BEFORE the venv check,
object verification and standup. It was unbounded and unconditional: once 14 rows had persisted 17,515 files / 5,994 MB the next
containers each spent ~44 min copying (398 files/min, 86 MB/min measured in-container) before standing anything up, and the cache
grows with every geometry persisted. Known-bad at 366b8e162: with a populated volume cache the default call COPIES.

Fixed: seeding is OFF by default (`SEEDREP_SEED_COMPILE_CACHES`, default "0"), read on the grading host and carried to the worker through
spawn_ctx exactly like the persist switch; when "1" the source is counted once (files / bytes) and refused over `SEED_MAX_FILES` /
`SEED_MAX_BYTES`; the `compile_cache_seed` evidence records the decision per cache (never silent) and keeps the keys it had
(`volume`, `local`, `seeded`, `seed_s` when it copied). The persist path and its default are untouched."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DRIVER = REPO / "scripts" / "modal" / "seedrep_served.py"
ENV = "SEEDREP_SEED_COMPILE_CACHES"


class _Chain:
    def __getattr__(self, name):
        return lambda *a, **k: self


class _App:
    def __init__(self, *a, **k):
        pass

    def function(self, *a, **k):
        return lambda f: f

    def local_entrypoint(self, **kw):
        return lambda f: f


def _stub_modal():
    m = types.ModuleType("modal")
    m.App = _App
    m.Volume = types.SimpleNamespace(from_name=lambda *a, **k: types.SimpleNamespace(commit=lambda: None, reload=lambda: None))
    m.Image = types.SimpleNamespace(from_registry=lambda *a, **k: _Chain(), debian_slim=lambda *a, **k: _Chain())
    m.Dict = types.SimpleNamespace(from_name=lambda *a, **k: {})
    m.is_local = lambda: False
    m.current_function_call_id = lambda: "fc-test"
    return m


_ABSENT = object()


@pytest.fixture(scope="module")
def srv():
    spec = importlib.util.spec_from_file_location("seedrep_served_fx_under_test", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    prior = sys.modules.get("modal", _ABSENT)
    sys.modules["modal"] = _stub_modal()
    try:
        spec.loader.exec_module(mod)
    finally:
        if prior is _ABSENT:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = prior
    return mod


def _populate(root: Path, n: int, size: int = 0) -> int:
    """`n` files under `root` (some nested, one symlink-free tree), each `size` bytes; returns the byte total."""
    root.mkdir(parents=True, exist_ok=True)
    total = 0
    for i in range(n):
        p = root / ("sub" if i % 2 else ".") / f"f{i}.bin"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * size)
        total += size
    return total


@pytest.fixture
def caches(srv, tmp_path, monkeypatch):
    """A fake populated volume cache (3 files, 30 bytes) for ONE cache name, with the module's roots repointed at tmp."""
    vol, local = tmp_path / "vol", tmp_path / "local" / "vllm"
    monkeypatch.setattr(srv, "CACHE_VOL", str(vol))
    monkeypatch.setattr(srv, "COMPILE_CACHES", {"vllm": str(local)})
    monkeypatch.delenv(ENV, raising=False)
    total = _populate(vol / "vllm", 3, size=10)
    return types.SimpleNamespace(vol=vol, src=vol / "vllm", dst=local, files=3, bytes=total)


# ------------------------------------------------------------------ (a) default: nothing copied, the decision recorded

def test_default_setting_seeds_nothing_and_records_by_default(srv, caches, capsys):
    """Known-bad at 366b8e162: `seed_compile_caches()` copies the populated source (the ~44 min per container of F-Y)."""
    assert any(caches.src.iterdir())                                        # the source IS populated -- the old code would copy it
    for out in (srv.seed_compile_caches(env={}), srv.seed_compile_caches(), srv.seed_compile_caches(env={ENV: ""})):
        rec = out["vllm"]
        assert not caches.dst.exists(), "the default must copy NOTHING"
        assert rec["volume"] == str(caches.src) and rec["local"] == str(caches.dst)   # the keys every reader had
        assert rec["seeded"] is False and "seed_s" not in rec
        assert rec["skipped"] == "by_default"
        assert rec["setting"] == {ENV: "0", "source": "default"}
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("[compile-cache] seed vllm")]
    assert len(lines) == 3 and all("by_default" in l for l in lines)        # one line per cache per call, naming the decision


def test_an_explicit_off_is_recorded_as_by_env(srv, caches, capsys):
    for raw in ("0", " 0 ", "no"):
        rec = srv.seed_compile_caches(env={ENV: raw})["vllm"]
        assert not caches.dst.exists()
        assert rec["seeded"] is False and rec["skipped"] == "by_env"
        assert rec["setting"] == {ENV: raw.strip(), "source": "env"}
    assert "[compile-cache] seed vllm: skipped (by_env" in capsys.readouterr().out


# ------------------------------------------------------------------ (b) on + small source: copied, timed, counted

def test_setting_one_copies_a_small_source_and_records_the_counts(srv, caches, capsys):
    rec = srv.seed_compile_caches(env={ENV: "1"})["vllm"]
    assert caches.dst.is_dir()
    assert sorted(p.name for p in caches.dst.rglob("*") if p.is_file()) == sorted(p.name for p in caches.src.rglob("*") if p.is_file())
    assert (caches.dst / "f0.bin").read_bytes() == b"x" * 10
    assert rec["seeded"] is True and isinstance(rec["seed_s"], float) and rec["seed_s"] >= 0
    assert rec["source_files"] == caches.files and rec["source_bytes"] == caches.bytes
    assert isinstance(rec["count_s"], float) and rec["count_s"] >= 0
    assert rec["setting"] == {ENV: "1", "source": "env"} and "skipped" not in rec
    assert rec["volume"] == str(caches.src) and rec["local"] == str(caches.dst)
    out = capsys.readouterr().out
    assert "[compile-cache] seed vllm: copied 3 files / 30 B" in out
    # a stale local tree is replaced, as before
    (caches.dst / "stale.bin").write_bytes(b"s")
    srv.seed_compile_caches(env={ENV: "1"})
    assert not (caches.dst / "stale.bin").exists() and (caches.dst / "f0.bin").exists()


def test_on_with_an_empty_or_missing_source_copies_nothing_and_says_so(srv, caches):
    for p in list(caches.src.rglob("*")):
        if p.is_file():
            p.unlink()
    rec = srv.seed_compile_caches(env={ENV: "1"})["vllm"]
    assert rec["seeded"] is False and rec["skipped"] == "no_source" and not caches.dst.exists()
    import shutil
    shutil.rmtree(caches.src)
    rec = srv.seed_compile_caches(env={ENV: "1"})["vllm"]
    assert rec["seeded"] is False and rec["skipped"] == "no_source" and not caches.dst.exists()


# ------------------------------------------------------------------ (c) on + over the bound: refused by name, counts recorded

def test_setting_one_refuses_a_source_over_the_file_bound(srv, caches, capsys):
    assert srv.SEED_MAX_FILES == 2000 and srv.SEED_MAX_BYTES == 512 * 1024 * 1024
    _populate(caches.src, srv.SEED_MAX_FILES + 1)                              # 2001 real empty files against the REAL constant
    rec = srv.seed_compile_caches(env={ENV: "1"})["vllm"]
    assert not caches.dst.exists(), "over the bound nothing is copied"
    assert rec["seeded"] is False and "seed_s" not in rec
    assert rec["skipped"] == "over_bound"
    assert rec["source_files"] == srv.SEED_MAX_FILES + 1 and rec["source_bytes"] == 0 and rec["count_s"] >= 0
    assert rec["bound"] == {"max_files": 2000, "max_bytes": 512 * 1024 * 1024}
    assert rec["setting"] == {ENV: "1", "source": "env"}
    out = capsys.readouterr().out
    assert "[compile-cache] seed vllm: skipped (over_bound" in out and "2001 files" in out


def test_setting_one_refuses_a_source_over_the_byte_bound(srv, caches, monkeypatch):
    monkeypatch.setattr(srv, "SEED_MAX_BYTES", caches.bytes - 1)
    rec = srv.seed_compile_caches(env={ENV: "1"})["vllm"]
    assert not caches.dst.exists() and rec["skipped"] == "over_bound"
    assert rec["source_files"] == caches.files and rec["source_bytes"] == caches.bytes
    assert rec["bound"]["max_bytes"] == caches.bytes - 1
    # exactly AT the bound is under it
    monkeypatch.setattr(srv, "SEED_MAX_BYTES", caches.bytes)
    monkeypatch.setattr(srv, "SEED_MAX_FILES", caches.files)
    assert srv.seed_compile_caches(env={ENV: "1"})["vllm"]["seeded"] is True and caches.dst.is_dir()


def test_the_count_is_one_walk_that_does_not_follow_symlinks(srv, caches, tmp_path):
    """copytree(symlinks=True) copies a link AS a link, so the count treats one as one file of its own size and never descends through it."""
    outside = tmp_path / "outside"
    _populate(outside, 50, size=100)
    (caches.src / "link_dir").symlink_to(outside, target_is_directory=True)
    (caches.src / "link_file").symlink_to(outside / "f0.bin")
    files, nbytes = srv._count_tree(caches.src)
    assert files == caches.files + 2
    assert nbytes < caches.bytes + 2 * 4096                                 # link sizes, not 5,000 bytes of the pointed-at tree


# ------------------------------------------------------------------ (d) the plumbing: the grading host environ -> spawn_ctx -> the worker

PLAN_STUB = {"W": "f" * 40, "wave_uuid": "11111111-2222-4333-8444-555555555555", "dicts": {"accepted": "a"}, "plans_root": "/vol/p", "ladder_root": "/vol/l",
             "results_root": "/vol/r", "launch_plan_sha": "0123456789abcdef"}


def test_the_setting_rides_spawn_ctx_from_the_launcher_environ_to_the_worker(srv, caches, monkeypatch):
    """Mirrors the persist switch's test: the REMOTE worker's environ is the image's, so the launcher reads $SEEDREP_SEED_COMPILE_CACHES
    in ITS process and transports the value through spawn_ctx; the worker decides on the ctx value, never on its own environ."""
    assert srv.SEED_COMPILE_CACHES_ENV == ENV and srv.SEED_COMPILE_CACHES_CTX == "seed_compile_caches"
    assert srv.SEED_COMPILE_CACHES_CTX != srv.PERSIST_COMPILE_CACHES_CTX
    ctx1 = srv.spawn_ctx(PLAN_STUB, "app", None, env={ENV: " 1 "})
    ctx0 = srv.spawn_ctx(PLAN_STUB, "app", None, env={ENV: "0"})
    ctxu = srv.spawn_ctx(PLAN_STUB, "app", None, env={})
    assert ctx1[srv.SEED_COMPILE_CACHES_CTX] == "1" and ctx0[srv.SEED_COMPILE_CACHES_CTX] == "0" and ctxu[srv.SEED_COMPILE_CACHES_CTX] == ""
    assert ctxu[srv.PERSIST_COMPILE_CACHES_CTX] == ""                           # the persist switch rides verbatim too: "" = unset = OFF (fx1)
    remote1, remoteu = json.loads(json.dumps(ctx1)), json.loads(json.dumps(ctxu))   # the local -> remote boundary
    # the worker's decision is made on the ctx value with its OWN environ saying nothing (or the opposite)
    monkeypatch.setenv(ENV, "1")
    rec = srv.seed_compile_caches(env={ENV: str(remoteu.get(srv.SEED_COMPILE_CACHES_CTX, ""))})["vllm"]
    assert rec["skipped"] == "by_default" and not caches.dst.exists()
    monkeypatch.setenv(ENV, "0")
    rec = srv.seed_compile_caches(env={ENV: str(remote1.get(srv.SEED_COMPILE_CACHES_CTX, ""))})["vllm"]
    assert rec["seeded"] is True and caches.dst.is_dir()
    # a launcher that predates the key carries nothing -> the worker's default is OFF
    assert srv.seed_compile_caches(env={ENV: str({}.get(srv.SEED_COMPILE_CACHES_CTX, ""))})["vllm"]["skipped"] == "by_default"
    # the default reads the LAUNCHER's process environment
    monkeypatch.setenv(ENV, "1")
    assert srv.spawn_ctx(PLAN_STUB, "app", None)[srv.SEED_COMPILE_CACHES_CTX] == "1"
    monkeypatch.delenv(ENV)
    assert srv.spawn_ctx(PLAN_STUB, "app", None)[srv.SEED_COMPILE_CACHES_CTX] == ""


def test_the_row_function_seeds_on_the_ctx_value_not_a_bare_call(srv):
    src = DRIVER.read_text()
    assert '"compile_cache_seed": seed_compile_caches()' not in src           # the known-bad literal
    assert '"compile_cache_seed": seed_compile_caches(env={SEED_COMPILE_CACHES_ENV: str(ctx.get(SEED_COMPILE_CACHES_CTX, ""))})' in src
    assert src.count("SEED_COMPILE_CACHES_CTX:") == 1                          # spawn_ctx writes the key once
    # the persist switch's plumbing is the same channel, now OFF by default and decided with the row's status (fx1; tests/test_seedrep_f5_*)
    assert 'PERSIST_COMPILE_CACHES_CTX: str(e.get(PERSIST_COMPILE_CACHES_ENV, "") or "").strip(),' in src
    assert 'compile_cache_persist=compile_cache_persist_evidence(env={PERSIST_COMPILE_CACHES_ENV: str(ctx.get(PERSIST_COMPILE_CACHES_CTX, ""))}, status=status)' in src
