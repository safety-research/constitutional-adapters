"""No grading host store read at any entry point.

The chokepoints in ``sisteer.registry.adapters.base`` REQUIRE a process-level store access scope: the cluster shard installs one
(``cluster_reads.run_shard``, before the adapters import), the operator's ``REGISTRY_STORE_READ_OVERRIDE=<reason>`` is the one other way,
and nothing else does -- so on the grading host every store access refuses by name (``GradingHostStoreReadRefused``) BEFORE any byte is read or any
directory listed. Because the adapter contract is a Protocol no concrete adapter subclasses, the guard sits where adapters are constructed
(``registered()`` / ``load_all()`` hand out guarded factories whose instance refuses its first ``discover()`` / ``read()``) and dispatched
(``runner.run_pass``, ``objects_bank.scan`` / ``hpo_logs.scan``, the two smoke scripts), in ``check_input``, and in the routed directory
walkers (``store_listdir`` / ``store_scandir`` / ``store_walk`` / ``store_glob`` / ``store_open``).

Instrumented, as the plan asks: (a) the REAL entry points as subprocesses on a synthetic home with declared roots -- ``registry adapters
run``, ``registry objects scan``, ``registry cluster-reads plan``, ``smoke_items_adapters.py``, ``smoke_b1_adapters.py``, ``registry
backfill`` -- and (b) DIRECT discovery in-process (``load_all()[name]().discover()``) for representative adapters of every construction
family (``objects_bank``, ``hpo_logs``, ``sampled_static``, ``relu_paired``, a ``BcbwSeedrepReader`` subclass, a ``CellStoreAdapter``
subclass, a declared reader), on an UNSCOPED host under the audit hook (``REGISTRY_STORE_AUDIT=<log>``: ``sys.addaudithook`` on ``open``,
``os.listdir``, ``os.scandir``, ``glob.glob`` under the declared roots): the named refusal fires and the audit log is EMPTY; with the override
set the same runs succeed and log it; a shard-shaped run (``run_shard``) opens freely. The suite itself runs under the override
(``tests/conftest.py``), so every test here clears it explicitly -- in-process with ``base.without_store_access_scope()``, in a child by
popping the variable from its environment.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

import sisteer
from sisteer.registry import cluster_reads as CR
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import stores as ST
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import load_all
from sisteer.registry.adapters import runner as RUN

from test_registry_adapter_pass import SyntheticStoreAdapter, _setup, _store
from test_registry_cluster_read import ADAPTER, _node, _plan_and_stage, _world_with_generation

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(sisteer.__file__)))
REPO_ROOT = os.path.dirname(SRC_DIR)
SCRIPTS = os.path.join(REPO_ROOT, "scripts", "registry")
PY = sys.executable
#: one registered adapter per construction family: the bank enumerator, the sweep-log reader, the sampled statics, relu_paired,
#: a BcbwSeedrepReader subclass, the CellStoreAdapter family (a subclass of it, and the base), a declared reader
FAMILIES = {"objects_bank": "ObjectsBankAdapter", "hpo_logs": "HpoLogsAdapter", "sampled_static": "SampledStaticAdapter",
            "relu_paired": "ReluPairedAdapter", "bigcodebench_seedrep": "BcbwSeedrepReader", "orxs_seedrep": "CellStoreAdapter",
            "cellstore": "CellStoreAdapter", "arditi_allpos": "ArditiAllposAdapter"}


def _audit_lines(path) -> list:
    return [json.loads(l) for l in open(path, encoding="utf-8").read().splitlines() if l.strip()] if os.path.exists(path) else []


#: The classifier behind every real-home audit here (``_store_reads``). PROTECTED roots win: a read under the configured registry root, a
#: root an adapter declares, the synthetic stores or the synthetic home is a store read whatever else its location matches. Only then does
#: the set-aside rule apply, by LOCATION: the one kind of read the operations under test make themselves and no store audit should count,
#: the checkout's own code, configuration and bytecode -- ``src/``, ``scripts/``, ``configs/``, ``tests/``, any ``__pycache__``, a top-level
#: file such as pyproject.toml -- and the interpreter's own installation (``sys.prefix``: the venv's packages and their bytecode). Measured
#: over both audited operations: the coverage over the real configuration and the ``cluster-reads plan`` child open the checkout's
#: ``configs/registry/*.yaml``, list ``src/sisteer`` and read ``__pycache__/*.pyc`` there and under the venv, and nothing else under the
#: home. Everything else under an audited root counts, so a declaration the audit polices is never what sets the audit's scope: the whole
#: home stays audited, with the registry root under it named explicitly (``REAL_HOME_ROOTS``, pinned by ``_assert_real_home_audited``).
_CHECKOUT_CODE_DIRS = ("src", "scripts", "configs", "tests")


def _under(path: str, root: str) -> bool:
    """``path`` (absolute) is ``root`` or beneath it -- a PATH prefix at a separator, never a string prefix; ``~`` in ``root`` is expanded."""
    root = os.path.abspath(os.path.expanduser(str(root))).rstrip(os.sep) or os.sep
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _is_code_or_config_read(path: str) -> bool:
    """The set-aside rule alone (the module note above): the checkout's code, configuration and bytecode, and the interpreter's prefix."""
    ap = os.path.abspath(path)
    if any(_under(ap, prefix) for prefix in {sys.prefix, sys.base_prefix}):
        return True
    rel = os.path.relpath(ap, REPO_ROOT)
    if rel.startswith(os.pardir) or os.path.isabs(rel):
        return False
    parts = rel.split(os.sep)
    return "__pycache__" in parts or parts[0] in _CHECKOUT_CODE_DIRS or (len(parts) == 1 and not os.path.isdir(ap))


def _is_store_read(path: str, protected) -> bool:
    """True when a recorded read counts: under any PROTECTED root first (precedence over every set-aside location -- a declared root nested
    under the checkout's ``src/`` or ``tests/``, a store under the interpreter's prefix, a set-aside prefix that happens to contain the home),
    else whatever the set-aside rule does not claim."""
    ap = os.path.abspath(path)
    if any(_under(ap, r) for r in protected if str(r).strip()):
        return True
    return not _is_code_or_config_read(ap)


def _store_reads(path, protected) -> list:
    """The audit's lines that are store reads under ``_is_store_read`` with these PROTECTED roots."""
    return [e for e in _audit_lines(path) if _is_store_read(e["path"], protected)]


def _declared_roots(rep) -> list:
    """Every root the adapters of a coverage report declare (``CoverageReport.roots``, one row per declared root; a broken declaration's None
    dropped), read off the AUDITED construction and never off one made before the audit."""
    return sorted({os.path.expanduser(str(r.root)) for r in rep.roots if getattr(r, "root", None)})


#: the roots every real-home audit here covers besides the synthetic world: the running user's whole home, and the configured registry root
#: under it by name (the one path the coverage docstring names, kept audited even if the home audit were ever narrowed)
REAL_HOME_ROOTS = [str(Path.home()), DB.DEFAULT_ROOT]


def _assert_real_home_audited(roots) -> None:
    """The structural check of an audit's scope: its roots name the running user's home (``Path.home()``) and the configured registry root
    (``db.DEFAULT_ROOT``), each as itself after ``~`` expansion and separator normalization. Fails naming what is missing."""
    audited = {os.path.abspath(os.path.expanduser(str(r))).rstrip(os.sep) for r in roots if str(r).strip()}
    required = (("the home", str(Path.home()).rstrip(os.sep)), ("the registry root", os.path.abspath(DB.DEFAULT_ROOT).rstrip(os.sep)))
    missing = [f"{label} ({p})" for label, p in required if p not in audited]
    assert not missing, f"audit scope incomplete: missing {', '.join(missing)}; audited {sorted(audited)}"


def _child_env(home: str, audit: str, roots, override=None) -> dict:
    """A child's environment: an UNSCOPED host (the suite's override popped) with the audit hook on ``roots``; ``override`` re-admits."""
    env = {k: v for k, v in os.environ.items() if k != AB.STORE_OVERRIDE_ENV}
    env.update({"HOME": home, "PYTHONPATH": SRC_DIR + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
                AB.STORE_AUDIT_ENV: audit, AB.STORE_AUDIT_ROOTS_ENV: ":".join(roots), "SISTEER_REGISTRY_DISABLED": "1", "WANDB_MODE": "disabled",
                "PYTHONUNBUFFERED": "1"})
    if override is not None:
        env[AB.STORE_OVERRIDE_ENV] = override
    return env


def _registry(argv, env, cwd=REPO_ROOT, timeout=600):
    return subprocess.run([PY, "-m", "sisteer.registry.cli", *argv], env=env, cwd=cwd, capture_output=True, text=True, timeout=timeout)


# --------------------------------------------------------------------------------------------
# the scope itself
# --------------------------------------------------------------------------------------------


def test_the_scope_is_explicit_or_the_environment_override_and_the_override_is_logged_once(monkeypatch, caplog):
    assert AB.environment_override({}) is None and AB.environment_override({AB.STORE_OVERRIDE_ENV: "  "}) is None
    ov = AB.environment_override({AB.STORE_OVERRIDE_ENV: "a drill"})
    assert ov == AB.StoreAccessScope("override", "a drill") and ov.describe() == "override: a drill"
    with pytest.raises(ValueError):
        AB.StoreAccessScope("grading_host", "x")
    with pytest.raises(ValueError):
        AB.StoreAccessScope("shard", "   ")
    # an explicit block hides the environment: None inside = an unscoped host whatever the suite set
    with AB.without_store_access_scope():
        assert AB.store_access_scope() is None
        with pytest.raises(AB.GradingHostStoreReadRefused) as ei:
            AB.require_store_access("a test read", "/some/store/file")
        assert "/some/store/file" in str(ei.value) and AB.STORE_OVERRIDE_ENV in str(ei.value) and "registry cluster-reads shard" in str(ei.value)
        with AB.using_store_access_scope(AB.StoreAccessScope("shard", "task 0", window="w")):
            assert AB.require_store_access("inside").describe() == "shard: task 0 (operation w)"
        assert AB.store_access_scope() is None
    # the environment override, logged once per process and reason
    monkeypatch.setenv(AB.STORE_OVERRIDE_ENV, "logged-once-reason")
    AB._OVERRIDE_LOGGED.discard("logged-once-reason")
    with caplog.at_level(logging.WARNING, logger="sisteer.registry.adapters.base"):
        assert AB.store_access_scope().kind == "override" and AB.store_access_scope().reason == "logged-once-reason"
    hits = [r for r in caplog.records if AB.STORE_OVERRIDE_ENV in r.getMessage() and "logged-once-reason" in r.getMessage()]
    assert len(hits) == 1 and "operator override" in hits[0].getMessage()


def test_the_walkers_check_input_and_run_pass_refuse_before_any_access_and_admit_under_a_scope(tmp_path):
    root, store_root = _setup(tmp_path)
    s1 = os.path.join(store_root, "s1", "store.json")
    audit = tmp_path / "audit.jsonl"
    AB.install_store_audit(str(audit), [store_root])
    try:
        with AB.without_store_access_scope():
            for fn, arg in ((AB.store_listdir, store_root), (AB.store_scandir, store_root), (AB.store_walk, store_root),
                            (AB.store_glob, os.path.join(store_root, "*")), (AB.store_iglob, os.path.join(store_root, "*")), (AB.store_open, s1)):
                with pytest.raises(AB.GradingHostStoreReadRefused):
                    fn(arg)
            with pytest.raises(AB.GradingHostStoreReadRefused):
                AB.check_input(s1, role="report")
            # the pass refuses BEFORE discovery: an adapter whose discover() would fail the test is never asked
            class Loud(SyntheticStoreAdapter):
                def discover(self):
                    pytest.fail("run_pass must refuse before any discovery on an unscoped host")
            with pytest.raises(AB.GradingHostStoreReadRefused) as ei:
                RUN.run_pass(root, [Loud(store_root)])
            assert "adapters run: the pass over synthetic_store" in str(ei.value)
            # the bounded import-time listing is skipped, never refused
            from sisteer.registry.adapters import _items_common as IC
            assert IC.listdir_if_scoped(store_root) == [] and AB.store_glob_if_scoped(os.path.join(store_root, "*")) == []
        assert _audit_lines(audit) == [], _audit_lines(audit)
        # under a scope every one of them works and the audit records the accesses
        with AB.using_store_access_scope(AB.StoreAccessScope("operator", "the walkers test")):
            assert AB.store_listdir(store_root) == ["s1"] and sorted(AB.store_glob(os.path.join(store_root, "*"))) == [os.path.join(store_root, "s1")]
            assert [e.name for e in AB.store_scandir(store_root)] == ["s1"] and next(AB.store_walk(store_root))[0] == store_root
            with AB.store_open(s1) as fh:
                assert json.load(fh)["key"] == "legacy_store_1"
            assert AB.check_input(s1, role="report").uri == s1
            assert IC.listdir_if_scoped(store_root) == ["s1"]
        events = {(e["event"], e["path"]) for e in _audit_lines(audit)}
        assert ("os.listdir", store_root) in events and ("open", s1) in events and ("glob.glob", os.path.join(store_root, "*")) in events
    finally:
        AB.disable_store_audit()


def test_registered_factories_are_guarded_the_instance_is_the_adapter_and_the_guard_is_idempotent(tmp_path):
    root, store_root = _setup(tmp_path)
    with AB.without_store_access_scope():
        reg = load_all()
        assert reg and all(getattr(f, "_store_access_guarded_factory", False) and callable(getattr(f, "__wrapped__", None)) for f in reg.values())
        raw = AB.registered_raw()
        assert set(raw) == set(reg) and not any(getattr(f, "_store_access_guarded_factory", False) for f in raw.values())
        inst = AB.guard_adapter(SyntheticStoreAdapter(store_root))
        assert isinstance(inst, SyntheticStoreAdapter) and AB.guard_adapter(inst) is inst
        with pytest.raises(AB.GradingHostStoreReadRefused) as ei:
            inst.discover()
        assert "synthetic_store.discover()" in str(ei.value)
        with pytest.raises(AB.GradingHostStoreReadRefused):
            list(inst.read(None, "s1", []))          # read() is guarded too: refused before the adapter's body runs
    with AB.using_store_access_scope(AB.StoreAccessScope("operator", "guard test")):
        assert [i.uri for i in inst.discover()] == [os.path.join(store_root, "s1", "store.json")]


# --------------------------------------------------------------------------------------------
# (b) direct discovery per construction family, in-process, under the audit hook
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_direct_discovery_of_every_construction_family_refuses_by_name_with_an_empty_audit_log_and_runs_under_a_scope(name, tmp_path):
    """`load_all()[name]().discover()` on an unscoped host: the guarded instance refuses BEFORE any listing (the audit hook over the
    adapter's declared roots and this home records nothing); under the override the same discovery runs -- it may refuse on an absent root
    (an `AdapterRefusal`), never a store-access refusal."""
    audit = tmp_path / f"audit-{name}.jsonl"
    with AB.without_store_access_scope():
        reg = load_all()
        assert name in reg, f"{name} is not registered"
        inst = reg[name]()
        assert FAMILIES[name] in {c.__name__ for c in type(inst).__mro__}, [c.__name__ for c in type(inst).__mro__]
        roots = [str(r) for r in (getattr(inst, "roots", ()) or ()) if r] + [os.path.expanduser("~")]
        AB.install_store_audit(str(audit), roots)
        try:
            with pytest.raises(AB.GradingHostStoreReadRefused) as ei:
                inst.discover()
            assert f"{name}.discover()" in str(ei.value)
            assert _audit_lines(audit) == [], _audit_lines(audit)
        finally:
            AB.disable_store_audit()
    with AB.using_store_access_scope(AB.StoreAccessScope("override", f"direct discovery test of {name}")):
        try:
            inst2 = reg[name]()
            inst2.discover()
        except AB.GradingHostStoreReadRefused:
            pytest.fail("a scoped discovery must never be a store-access refusal")
        except AB.AdapterRefusal:
            pass                                         # an absent root on this host: the adapter's own refusal, by name


def test_a_shard_shaped_run_opens_freely_on_an_unscoped_host_because_run_shard_installs_its_own_scope(tmp_path, monkeypatch):
    """`cluster_reads.run_shard` is the one caller that installs a scope (`kind = shard`): under an explicitly unscoped process it lays the home
    tree, reads through the pinned generation and writes its summary; the audit hook over the generation records the real reads."""
    w = _world_with_generation(tmp_path, monkeypatch)
    plan, _plan_file, _info = _plan_and_stage(w, tmp_path)
    _node(w)
    audit = tmp_path / "audit-shard.jsonl"
    AB.install_store_audit(str(audit), [w["gen_dir"]])
    try:
        with AB.without_store_access_scope():
            assert CR.run_shard(f"{plan['window_root']}/plan.json", 0, home=w["home"]) == 0
            assert AB.store_access_scope() is None           # the shard's scope was its own; nothing leaks out of it
    finally:
        AB.disable_store_audit()
    doc = json.load(open(CR.summary_path(plan["summaries_dir"], ADAPTER, 0, 2)))
    assert doc["ok"] is True and not any("store access refused" in str(r.get("error")) for r in doc["refused"])
    assert any(e["event"] == "open" and e["path"].startswith(w["gen_dir"]) for e in _audit_lines(audit)), _audit_lines(audit)[:3]


def test_the_store_read_classifier_counts_a_protected_root_before_any_set_aside_rule(tmp_path):
    """``_is_store_read``: a read under a PROTECTED root counts whatever set-aside location its path also matches; the set-aside rule applies
    only to what no protected root claims; a protected root is a path prefix, not a string prefix. And ``_assert_real_home_audited`` holds
    exactly when the roots name both the home and the registry root."""
    home = str(Path.home())
    pyc = os.path.join(REPO_ROOT, "src", "sisteer", "registry", "__pycache__", "stores.cpython-312.pyc")
    cfg = os.path.join(REPO_ROOT, "configs", "registry", "cluster_reads.yaml")
    venv = os.path.join(sys.prefix, "lib", "site-packages", "yaml", "__init__.py")
    home_store = os.path.join(home, "hpo-r1", "sweep", "trial.json")
    registry_db = os.path.join(DB.DEFAULT_ROOT, "registry.sqlite")
    # nothing protected: the set-aside rule alone -- code, configuration, bytecode and the venv are set aside; the home and the registry count
    assert not _is_store_read(pyc, []) and not _is_store_read(cfg, []) and not _is_store_read(venv, [])
    assert _is_store_read(home_store, []) and _is_store_read(registry_db, [])
    # protected roots win over every set-aside location: a declared root under the checkout's src/, the whole configs/ directory, the
    # checkout itself, a store under the interpreter's prefix, a registry root under the checkout's tests/
    nested_src = os.path.join(REPO_ROOT, "src", "fixture-stores")
    assert not _is_store_read(os.path.join(nested_src, "s1", "store.json"), []) and _is_store_read(os.path.join(nested_src, "s1", "store.json"), [nested_src])
    assert _is_store_read(cfg, [os.path.join(REPO_ROOT, "configs")]) and _is_store_read(pyc, [REPO_ROOT])
    nested_venv = os.path.join(sys.prefix, "stores")
    assert not _is_store_read(os.path.join(nested_venv, "x.json"), []) and _is_store_read(os.path.join(nested_venv, "x.json"), [nested_venv])
    assert _is_store_read(venv, [sys.prefix])
    reg_under_tests = os.path.join(REPO_ROOT, "tests", "registry")
    assert not _is_store_read(os.path.join(reg_under_tests, "registry.sqlite"), []) and _is_store_read(os.path.join(reg_under_tests, "registry.sqlite"), [reg_under_tests])
    if _under(os.path.abspath(REPO_ROOT), home):
        # the checkout under the home: protecting the home (as the audits here do) makes the checkout's own bytecode count, and `~` expands
        assert _is_store_read(pyc, [home]) and _is_store_read(pyc, ["~"])
    # a path prefix, not a string prefix: protecting `.../stores` leaves `.../stores-aside/x` to the set-aside rule
    p = os.path.join(REPO_ROOT, "src", "stores")
    assert not _is_store_read(os.path.join(REPO_ROOT, "src", "stores-aside", "x.json"), [p]) and _is_store_read(os.path.join(p, "x.json"), [p])
    # the structural check: both the home and the registry root, each spelled plainly or through `~`; either alone fails, naming the missing one
    _assert_real_home_audited([str(tmp_path), *REAL_HOME_ROOTS])
    _assert_real_home_audited(["~", os.path.join("~", os.path.relpath(DB.DEFAULT_ROOT, home))])
    with pytest.raises(AssertionError, match="missing the registry root") as ei:
        _assert_real_home_audited([str(tmp_path), home])
    assert "the home (" not in str(ei.value)
    with pytest.raises(AssertionError, match="missing the home") as ei:
        _assert_real_home_audited([str(tmp_path), DB.DEFAULT_ROOT])
    assert "the registry root (" not in str(ei.value)
    with pytest.raises(AssertionError, match="missing the home .*, the registry root"):
        _assert_real_home_audited([str(tmp_path)])


def test_stores_coverage_constructs_every_adapter_and_opens_nothing_on_an_unscoped_host(tmp_path, monkeypatch):
    """`registry stores coverage` reads the adapters' DECLARED roots: it constructs the adapters (the guard never fires on construction) and
    lists no directory, under the real registry of this tree. The audit covers the running user's whole home (the live registry root under it
    named explicitly) and the synthetic world, and the real adapters are constructed only inside the unscoped block with the audit on --
    so a constructor that discovered, or a hard-coded store path outside the roots an adapter declares, is recorded, and the declarations
    the audit polices never set its scope: the protected roots (the registry root, the roots declared under the audit, the synthetic world)
    count first, and the checkout's own code, configuration and bytecode reads are the one kind set aside (``_store_reads``). A negative
    control closes the test: an adapter whose construction reads an undeclared store under the audited synthetic home, and attempts an
    undeclared path under the running user's real home, is caught by the same audit on both; and the scope is pinned structurally
    (``_assert_real_home_audited``)."""
    w = _world_with_generation(tmp_path, monkeypatch)
    audit = tmp_path / "audit-coverage.jsonl"
    roots = [w["stores"], w["home"], *REAL_HOME_ROOTS]                   # the synthetic world, the running user's home, the registry root
    AB.install_store_audit(str(audit), roots)
    try:
        with AB.without_store_access_scope():
            rep = ST.coverage(w["cfg"], registry={ADAPTER: lambda: SyntheticStoreAdapter(w["stores"])}, home=w["home"])
            assert rep.ok, rep.as_dict()
            full = ST.coverage(ST.load_config(), registry=load_all())          # the real configuration over the real registry: constructs only
            assert full.adapters
    finally:
        AB.disable_store_audit()
    # the protected roots: the registry root, every root the real adapters declared under the audit, the synthetic stores and home
    protected = [DB.DEFAULT_ROOT, w["stores"], w["home"], *_declared_roots(full)]
    assert _declared_roots(full), full.as_dict()["adapters"]
    assert _store_reads(audit, protected) == [], _store_reads(audit, protected)[:5]

    # the negative control: the same coverage call over an adapter whose construction reads where nobody declared a store, through the plain
    # builtin (the shape of a hard-coded path, which the routed walkers never see and so never refuse) -- construction succeeds, and the
    # audit is what catches it. Two reads: a store UNDER the audited synthetic home (no audited root of its own), and an attempt on a path
    # under the running user's real home that does not exist (the ``open`` audit event fires before the syscall, so the attempt is recorded
    # while nothing is created or read there) -- the second is caught only while the real home is in the audit's roots
    undeclared = Path(w["home"]) / "undeclared-store" / "store.json"
    undeclared.parent.mkdir(parents=True)
    undeclared.write_text("{}")                                          # written with the audit off
    probe = Path.home() / f".undeclared-store-probe-{os.getpid()}" / "store.json"
    assert not probe.parent.exists(), probe

    class OpensAnUndeclaredStore(SyntheticStoreAdapter):
        def __init__(self, stores):
            super().__init__(stores)
            with open(str(undeclared), encoding="utf-8") as fh:
                fh.read()
            try:
                open(str(probe), encoding="utf-8")
            except FileNotFoundError:
                pass

    audit2 = tmp_path / "audit-coverage-negative.jsonl"
    AB.install_store_audit(str(audit2), roots)
    try:
        with AB.without_store_access_scope():
            rep2 = ST.coverage(w["cfg"], registry={ADAPTER: lambda: OpensAnUndeclaredStore(w["stores"])}, home=w["home"])
    finally:
        AB.disable_store_audit()
    assert rep2.ok, rep2.as_dict()
    assert not probe.parent.exists(), probe                              # the attempt created nothing under the real home
    caught = {(e["event"], e["path"]) for e in _store_reads(audit2, protected)}
    assert caught == {("open", str(undeclared)), ("open", str(probe))}, sorted(caught)[:5]
    _assert_real_home_audited(roots)                                      # and structurally: the home and the registry root are in the scope


# --------------------------------------------------------------------------------------------
# (a) the real entry points as subprocesses on a synthetic home, unscoped: refused by name, audit empty; under the override: admitted, logged
# --------------------------------------------------------------------------------------------


@pytest.fixture
def world(tmp_path):
    """A synthetic home with the roots two real adapters declare (hpo_logs' `~/hpo-r1`, objects_bank's `$GRADE/objects` as an
    explicit bank), a registry root initialized by the CLI, an audit log path."""
    home = tmp_path / "home"
    hpo = home / "hpo-r1"
    hpo.mkdir(parents=True)
    bank = home / "grade" / "objects"
    bank.mkdir(parents=True)
    root = tmp_path / "reg"
    env = _child_env(str(home), str(tmp_path / "audit.jsonl"), [str(home)])
    r = _registry(["--root", str(root), "init"], env)
    assert r.returncode == 0, r.stderr
    return {"home": str(home), "root": str(root), "hpo": str(hpo), "bank": str(bank), "audit": str(tmp_path / "audit.jsonl"), "tmp": tmp_path}


def test_adapters_run_refuses_by_name_with_an_empty_audit_log_and_runs_under_the_override_which_is_logged(world):
    env = _child_env(world["home"], world["audit"], [world["home"]])
    r = _registry(["--root", world["root"], "adapters", "run", "--adapter", "hpo_logs", "--no-ingest"], env)
    assert r.returncode == 5, (r.returncode, r.stdout, r.stderr)
    assert "refused:" in r.stderr and "store access refused" in r.stderr and "adapters run: the pass over hpo_logs" in r.stderr and AB.STORE_OVERRIDE_ENV in r.stderr
    assert _audit_lines(world["audit"]) == [], _audit_lines(world["audit"])
    # the override: the same run succeeds, the override is logged, the adapter lists its root (the audit shows the real access)
    env2 = _child_env(world["home"], world["audit"], [world["home"]], override="refusal test: the admitted run")
    r2 = _registry(["--root", world["root"], "adapters", "run", "--adapter", "hpo_logs", "--no-ingest"], env2)
    assert r2.returncode == 0, (r2.returncode, r2.stdout, r2.stderr)
    assert "store access refused" not in r2.stderr and "operator override" in r2.stderr and "refusal test: the admitted run" in r2.stderr
    events = _audit_lines(world["audit"])
    assert any(e["event"] == "os.listdir" and e["path"].startswith(world["hpo"]) for e in events), events[:5]


def test_objects_scan_refuses_by_name_before_any_bank_is_listed_and_scans_under_the_override(world):
    env = _child_env(world["home"], world["audit"], [world["home"]])
    argv = ["--root", world["root"], "objects", "scan", "--bank-root", f"lab={world['bank']}:model/obj", "--no-hpo", "--no-blobs", "--json"]
    r = _registry(argv, env)
    assert r.returncode == 5, (r.returncode, r.stdout, r.stderr)
    assert "refused:" in r.stderr and "objects_bank.scan" in r.stderr and _audit_lines(world["audit"]) == []
    env2 = _child_env(world["home"], world["audit"], [world["home"]], override="refusal test: objects scan")
    r2 = _registry(argv, env2)
    assert r2.returncode in (0, 1), (r2.returncode, r2.stdout, r2.stderr)
    assert "store access refused" not in r2.stderr and json.loads(r2.stdout)["objects"] is not None
    assert any(e["event"] == "os.listdir" and e["path"] == world["bank"] for e in _audit_lines(world["audit"])), _audit_lines(world["audit"])[:5]


def test_cluster_reads_plan_discovers_nothing_as_a_subprocess_the_audit_log_stays_empty(world):
    """The plan's coverage discovery was a grading-host store read; the plan now discovers nothing: on an unscoped host the
    real `cluster-reads plan` over the repo's configuration exits 0 (every adapter recorded refused -- no published generation on this fresh
    root), the plan file carries no `inputs` key, and the audit log holds no store read. The child's HOME is the synthetic home, so every
    root the real adapters declare under `~` expands beneath it and is audited; the invoking account's own home and the live registry root
    under it are audited as well, with the checkout's own code, configuration and bytecode reads (what the plan reads under `--repo-root`)
    the one kind set aside (``_store_reads``)."""
    roots = [world["home"], *REAL_HOME_ROOTS]
    _assert_real_home_audited(roots)
    env = _child_env(world["home"], world["audit"], roots)
    # the repo's cluster_reads.yaml names its destination by the cluster's registry label; this root's configuration names that cluster
    with open(os.path.join(world["root"], "config.toml"), "w", encoding="utf-8") as fh:
        fh.write(f'[[clusters]]\nname = "cluster_a"\nhost = "login-test"\nshared = "{world["tmp"]}/shared"\n')
    r = _registry(["--root", world["root"], "cluster-reads", "plan", "--window", "wtest", "--adapter", "relu_paired", "sampled_static", "--tree-sha", "a" * 40,
                   "--dest", "cluster_a", "--repo-root", REPO_ROOT, "--json"], env)
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    out = json.loads(r.stdout)
    assert out["routes"] == {"relu_paired": "grading_host", "sampled_static": "grading_host"} and set(out["refusals"]) == {"relu_paired", "sampled_static"}
    plan = json.load(open(CR.plan_path(world["root"], "wtest")))
    assert all("inputs" not in e for e in plan["adapters"].values()) and "refusals" in plan and "fallback" not in plan
    # protected: the registry roots (the configured one and this world's), the synthetic home -- under which every root the real adapters
    # declare under `~` expands in the child -- and the two declared roots the world lays out there
    protected = [DB.DEFAULT_ROOT, world["root"], world["home"], world["hpo"], world["bank"]]
    assert _store_reads(world["audit"], protected) == [], _store_reads(world["audit"], protected)[:5]


def test_the_smoke_scripts_refuse_by_name_before_any_discovery_and_the_backfill_stub_reads_nothing(world):
    env = _child_env(world["home"], world["audit"], [world["home"]])
    # the smoke's scratch root must hold a database (it counts the tables before the first adapter): the initialized registry root
    r = subprocess.run([PY, os.path.join(SCRIPTS, "smoke_items_adapters.py"), "--root", world["root"], "--skip-dimensions",
                        "--adapters", "cellstore", "--objects-worktree", ""], env=env, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 5, (r.returncode, r.stdout[-500:], r.stderr[-1500:])
    assert "refused:" in r.stderr and "smoke_items_adapters: direct discovery of cellstore" in r.stderr
    r2 = subprocess.run([PY, os.path.join(SCRIPTS, "smoke_b1_adapters.py"), "--adapter", "r2_export", "--no-standins", "--keep", str(world["tmp"] / "smoke-b1")],
                        env=env, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    assert r2.returncode == 5, (r2.returncode, r2.stdout[-500:], r2.stderr[-1500:])
    assert "refused:" in r2.stderr and "smoke_b1_adapters: the pass over r2_export" in r2.stderr
    r3 = _registry(["--root", world["root"], "backfill"], env)
    assert r3.returncode == 3 and "not available" in r3.stderr          # the stub: no adapter is run, nothing is read
    assert _audit_lines(world["audit"]) == [], _audit_lines(world["audit"])[:5]


def test_install_store_audit_from_env_installs_once_defaults_the_roots_to_the_home_and_is_inert_without_the_variable(tmp_path, monkeypatch):
    assert AB.install_store_audit_from_env({}) is False
    log = tmp_path / "a.jsonl"
    try:
        assert AB.install_store_audit_from_env({AB.STORE_AUDIT_ENV: str(log)}) is True
        assert AB._AUDIT["path"] == str(log) and AB._AUDIT["roots"] == (os.path.expanduser("~").rstrip(os.sep),)
        assert AB.install_store_audit_from_env({AB.STORE_AUDIT_ENV: str(log), AB.STORE_AUDIT_ROOTS_ENV: f"{tmp_path}:{tmp_path}/x"}) is True
        assert AB._AUDIT["roots"] == (str(tmp_path), f"{tmp_path}/x")
    finally:
        AB.disable_store_audit()
    assert AB._AUDIT["path"] is None and AB._AUDIT["roots"] == ()
