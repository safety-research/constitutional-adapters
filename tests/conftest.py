import atexit
import os
import shutil
import sys
import tempfile

import pytest

# tests/ is on sys.path when this conftest loads (its own directory, under pytest's default prepend import mode —
# the same way tests import ``derived_golden_replay``); stdlib + pytest only, so nothing of sisteer is imported at session
# start. Loud on purpose: a missing guard module is an ImportError here, never a run that silently goes unguarded.
from _full_suite_guard import refuse_whole_suite_run
from _host_site_isolation import CONFIG_ENV as _SITE_CONFIG_ENV, SITE_DIR_ENV as _SITE_DIR_ENV, isolate as isolate_from_the_host_site

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# The site placeholders that the registry's store configuration (configs/registry/cluster_reads.yaml) and the registry adapters' store
# roots are written with ($STATE, $STAGE, $WORK, $GRADE; src/sisteer/site.py) resolve, for the whole suite, to neutral directories
# under the home: the coverage tests compare the configuration's expanded bases with the adapters' expanded roots, so both sides must
# expand through the same values. A value the environment already sets stands.
SUITE_SITE_DEFAULTS = {"SISTEER_STATE_ROOT": "~/.local/state", "SISTEER_STAGE_DIR": "~/stage", "SISTEER_WORK_DIR": "~/work",
                       "SISTEER_GRADE_DIR": "~/grade"}


def _empty_suite_site_dir() -> str:
    """The run's site directory: empty and read-only (a write into it fails loudly), removed at exit."""
    d = tempfile.mkdtemp(prefix="sisteer-suite-no-site-")
    os.chmod(d, 0o500)

    def _remove(path=d):
        try:
            os.chmod(path, 0o700)
        except OSError:
            pass
        shutil.rmtree(path, ignore_errors=True)

    atexit.register(_remove)
    return d


# The suite never reads the HOST's site config (tests/_host_site_isolation.py holds the rule and the evidence): $SISTEER_CONFIG is
# removed, $SISTEER_SITE_DIR names an empty directory, and every suite default above is exported unless the environment sets it, so
# a test sees the fresh-clone site the Modal CPU suite and the Slurm runner see. SISTEER_TEST_HOST_SITE=1 opts a run into the host's
# site instead (a suite default then stands down for every key that file sets). Applied at import, before any test module is
# collected: an adapter module resolves its roots when it loads, and sisteer.site caches the site it read. Stdlib only (no sisteer
# import at session start); children a test spawns inherit the values.
SUITE_SITE = isolate_from_the_host_site(os.environ, SUITE_SITE_DEFAULTS, _empty_suite_site_dir)

TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"

# the live registry's one shared v_paper_long snapshot (session fixture ``live_paper_long``, tests/registry_live_fixtures.py) is
# registered ONCE here: a fixture imported into two test modules is registered twice, and a session scope alone would scan twice
pytest_plugins = ["registry_live_fixtures"]


def pytest_configure(config):
    """Refuse a whole-suite run outside the landing gate BEFORE anything is collected — ``tests/_full_suite_guard.py``
    holds the rule, the file limit, the venues that may run it, the evidence for each (2026-09-28: two whole-suite
    runs on the shared grading host took 2.5 h and 3 h 23 min and starved every other session) and the hook body
    itself, so ``tests/test_full_suite_guard.py`` can install the same hook on a synthetic tree without this file.

    Also registers the lane marks HERE rather than in ``pyproject.toml``: the CPU-suite image bakes the sha256 of
    ``pyproject.toml`` + ``uv.lock`` as its dependency-hash guard (``scripts/modal/cpu_suite.py``), so a marker line in
    pyproject staled that guard and errored every module that loads the Modal apps (landing 2's gate, 2026-09-29).
    ``inspect260``: the test needs the PINNED inspect-ai client (0.3.260 — the tinycap client lock, ``~/venvs/tinycap-laneB``
    on the grading host); the module ``importorskip``s ``inspect_ai`` / ``inspect_evals``, so the repo ``.venv`` skips it."""
    refuse_whole_suite_run(config)
    register = getattr(config, "addinivalue_line", None)    # the guard's own test calls this hook with a bare object
    if register is not None:
        register("markers", "inspect260: needs the pinned inspect-ai 0.3.260 client venv (importorskip'd; the repo .venv skips it)")


@pytest.fixture(scope="session", autouse=True)
def _registry_emission_off_for_the_suite():
    """Every test runs with results-registry emission OFF (``SISTEER_REGISTRY_DISABLED=1``) unless the
    caller set that variable itself. The switch (``sisteer.registry.hooks.emission_enabled``) is ON
    wherever a spool resolves, and the Modal CPU-suite container mounts the twosi-fits volume at
    ``/vol`` with ``registry/`` provisioned — so every fit a test ran through the fit writers' hooks
    (``scripts/fit_class_b.py``, ``scripts/train_two_si_objects.py``) registered its tiny-model object
    into the LIVE log: by 2026-09-18 the volume held 2,727 ``object`` records (855 run directories,
    3,168 files) blocked on ``hf-internal-testing/tiny-random-LlamaForCausalLM``, each stamped
    ``scripts/fit_class_b.py`` / ``host modal``. The log is append-only, so they cannot be removed;
    this is the belt and ``scripts/modal/cpu_suite.py``'s ``suite_env`` (the same variable, in the
    gate's environment) is the braces. A test that exercises emission opts in —
    ``monkeypatch.delenv(emit.DISABLED_ENV)`` plus ``SISTEER_REGISTRY_SPOOL=<tmp>`` or the
    ``unprovisioned_host`` fixture — and never names a real spool. Session-scoped so a function-scoped
    ``monkeypatch`` restores this default after each opt-in; ``os.environ`` directly, since a session
    fixture has no ``monkeypatch``. Children inherit it, so a producer a test spawns (``python3 -S
    scripts/registry/smoke_producer.py``, ``emit_cli``) is off too unless the test's child env says
    otherwise. ``tests/test_registry_suite_emission_off.py`` pins the default and the leak it closes."""
    var = "SISTEER_REGISTRY_DISABLED"  # sisteer.registry.emit.DISABLED_ENV; spelled out so no import runs at session start
    if os.environ.get(var) is not None:
        yield  # the caller's setting stands (``=0`` opts a whole run into emission, deliberately)
        return
    os.environ[var] = "1"
    try:
        yield
    finally:
        os.environ.pop(var, None)


@pytest.fixture(scope="session", autouse=True)
def _store_reads_admitted_for_the_suite():
    """Every test runs with the registry's store-access override SET (``REGISTRY_STORE_READ_OVERRIDE``) unless the caller set the
    variable itself (spec S2). The chokepoints in ``sisteer.registry.adapters.base`` refuse any store access --
    an adapter's ``discover()`` / ``read()``, ``check_input``, the routed directory walkers, the runner's pass -- on a host with no
    store access scope: the cluster shard installs one, and on the grading host nothing does. The suite's fixtures ARE synthetic stores read
    in-process and through CLI subprocesses (children inherit the variable), so the suite runs as the operator's override with this
    stated reason; the tests that prove the refusal (``tests/test_registry_grading_host_read_refusal.py``) install
    ``base.without_store_access_scope()`` in-process and pop the variable from a child's environment. Session-scoped, the same shape
    as the emission switch above (``os.environ`` directly; a function-scoped ``monkeypatch`` restores this default after an opt-out)."""
    var = "REGISTRY_STORE_READ_OVERRIDE"  # sisteer.registry.adapters.base.STORE_OVERRIDE_ENV; spelled out so no import runs at session start
    if os.environ.get(var) is not None:
        yield
        return
    os.environ[var] = "pytest suite: synthetic store fixtures (tests/conftest.py)"
    try:
        yield
    finally:
        os.environ.pop(var, None)


@pytest.fixture(scope="session", autouse=True)
def _no_live_registry_spool_for_the_suite(tmp_path_factory):
    """No test resolves a LIVE registry spool by default (since 2026-10-02). ``sisteer.registry.emit.spool_root``
    probes fixed host paths — the per-user shared roots, the Modal volume, ``~/registry`` — and on the grading host ``~/registry`` IS the live
    registry root (the harvester folds its ``records/`` on a timer): a test that opted emission back on (``monkeypatch.delenv(emit.DISABLED_ENV)``)
    and ran ``emit_cli register --from-config`` WITHOUT ``--spool`` wrote two synthetic ``run`` records straight into it, and they were in the
    database within the minute. This fixture points every probe at an EMPTY directory for the whole session through the resolver's own knob
    (``SISTEER_REGISTRY_PROBE_ROOT``, ``emit.PROBE_ROOT_ENV``), in-process and in every child a test spawns, so with no explicit spool NOTHING
    resolves on its own. Two belts, in-process and in every child a test spawns: (1) that probe root, so no fixed host path — the live
    ``~/registry`` included — can ever resolve; (2) ``SISTEER_REGISTRY_SPOOL`` naming a hermetic spool UNDER the empty host, so "a spool
    resolves" stays true for the tests that opt emission on and hand their hooks an explicit spool while ``hooks.emission_enabled`` still asks
    the resolver (the store recorders, the object emitters — on the grading host the live root used to answer that question for them; a trial
    showed 2 of 45 files going quiet with the first belt alone). A spool a test or a harness names (``--spool``, ``monkeypatch.setenv``, its
    child env) wins, as it always did; a test that deletes the spool variable gets "no spool resolves" (the first belt); the caller's own
    setting of either variable stands (a run deliberately pointed at a provisioned host); ``unprovisioned_host`` deletes both for its test and
    redirects the constants itself, as before. ``tests/test_registry_suite_no_live_spool.py`` pins it."""
    spool_var, probe_var = "SISTEER_REGISTRY_SPOOL", "SISTEER_REGISTRY_PROBE_ROOT"   # emit.SPOOL_ENV / emit.PROBE_ROOT_ENV; spelled out: no import at session start
    if os.environ.get(spool_var) or os.environ.get(probe_var):
        yield  # the caller's spool / probe host stands
        return
    host = tmp_path_factory.mktemp("no-live-spool-host")
    suite_spool = host / "suite-spool" / "records"
    suite_spool.mkdir(parents=True)
    os.environ[probe_var] = str(host)
    os.environ[spool_var] = str(suite_spool)
    try:
        yield
    finally:
        os.environ.pop(probe_var, None)
        os.environ.pop(spool_var, None)


@pytest.fixture(scope="session", autouse=True)
def _no_live_landing_state_for_the_suite(tmp_path_factory):
    """No test reads the host's LIVE landing state by default. ``sisteer.registry.reads_protocol.reads_dir`` resolves the cluster-read
    namespace -- the reservation, the op records and the receipts the status page's "Cluster reads, store authority and the replica"
    section lists -- from ``$REGISTRY_LANDING_STATE``, default ``~/.local/state/registry-landing``, and on the grading host that default IS
    the landing chain's live state: every tick leaves a receipt there naming the store paths it read. A status-page test that
    rendered without naming a state directory (``status.status_json(root)`` / ``status.render(root, ...)`` with no ``state_dir``)
    read those receipts into its page, and a word ban over the rendered text failed on a store path a live receipt carried -- on the
    host that runs the landing chain and nowhere else (``tests/test_registry_bon_decode_precedence.py``, the presquash-fu8 review).
    This fixture points the variable at an EMPTY directory for the whole session, in-process and in every child a test spawns
    (``os.environ`` directly, the shape of the two belts above), so with no explicit state directory the namespace is empty and
    every section of the page that reads it is the empty, green shape. A state directory a test or a harness names
    (``monkeypatch.setenv(RP.STATE_ENV, ...)``, a child's env, a CLI's ``--state``) wins, as it always did, and the caller's own
    setting of the variable stands (a run deliberately pointed at a state root; an EMPTY value is the default to ``reads_dir`` and is
    isolated like an unset one). ``tests/test_registry_suite_no_live_landing_state.py`` pins it."""
    var = "REGISTRY_LANDING_STATE"   # sisteer.registry.reads_protocol.STATE_ENV; spelled out so no import runs at session start
    if os.environ.get(var):
        yield  # the caller's state root stands
        return
    state = tmp_path_factory.mktemp("no-live-landing-state")
    os.environ[var] = str(state)
    try:
        yield
    finally:
        os.environ.pop(var, None)


@pytest.fixture(scope="session", autouse=True)
def _no_installed_units_for_the_suite():
    """No test sees the host's INSTALLED registry units by default. ``scripts/registry/install_grading_host.sh`` renders the units
    into ``$REGISTRY_UNIT_DIR``, default ``~/.config/systemd/user``, and refuses before writing anything when an installed unit
    evaluates an environment hook that the site it renders from does not name. ``reproduce/00_setup.sh --what registry`` runs it (in
    its own ``--dry-run`` too) without naming a unit directory, so on a host that had run stage 00 for its real site (the grading
    host's units evaluate the site's hook) every test that dry-ran stage 00 against a synthetic site was refused by the host's units
    (``test_front_door_dry_run``'s and ``test_reproduce_readme_committed_sequence``'s ``00_setup.sh`` cases, red there and nowhere
    else). This belt points the variable at an EMPTY, read-only directory for the whole session, in-process and in every child a
    test spawns (``os.environ`` directly, the shape of the belts above): a dry run compares against no installed unit, and a real
    install into it fails loudly. ``--unit-dir`` (every direct installer test names one) wins as it always did, and the caller's own
    setting stands. A test of the ``~/.config/systemd/user`` default under a moved ``HOME`` requests ``home_site_default``.
    The directory is made in the system temp dir, not pytest's base temp: the installer prints the unit directory, and pytest's
    base temp is named for the login (``pytest-of-<user>``), a personal identifier the stage-output token scans refuse.
    ``tests/test_suite_host_site_isolation.py`` pins it."""
    var = "REGISTRY_UNIT_DIR"   # install_grading_host.sh's unit-directory variable; spelled out, a shell script has nothing to import
    if os.environ.get(var):
        yield  # the caller's unit directory stands
        return
    units = tempfile.mkdtemp(prefix="sisteer-suite-no-units-")
    os.chmod(units, 0o500)
    os.environ[var] = units
    try:
        yield
    finally:
        os.environ.pop(var, None)
        os.chmod(units, 0o700)
        shutil.rmtree(units, ignore_errors=True)


@pytest.fixture
def home_site_default(monkeypatch):
    """For a test of the defaults under ``~`` with a MOVED ``HOME``: the site config at ``$HOME/.config/sisteer`` and the unit
    directory ``$HOME/.config/systemd/user``. Removes ``$SISTEER_CONFIG``, ``$SISTEER_SITE_DIR`` and ``$REGISTRY_UNIT_DIR`` for this
    test alone (the suite sets the last two at an empty directory; tests/_host_site_isolation.py), in-process and in every child it
    spawns from ``os.environ``. The test moves ``HOME`` itself, BEFORE it reads anything: with the real ``HOME`` the default IS the
    host's own site, which is exactly what the suite keeps out."""
    for var in (_SITE_CONFIG_ENV, _SITE_DIR_ENV, "REGISTRY_UNIT_DIR"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(scope="session", autouse=True)
def _no_third_party_fetch_for_the_suite(tmp_path_factory):
    """No test fetches or reads the real G0DM0D3 templates (``sisteer.eval.attacks_godmode``: AGPL-3.0, fetched at run time,
    never in this repository). For the whole session, in-process and in every child a test spawns, the module's cache is an
    EMPTY directory (``SISTEER_THIRD_PARTY_DIR``) and fetching is off (``SISTEER_THIRD_PARTY_OFFLINE=1``), so a test that renders
    grok420 without the invented stand-in refuses by name instead of downloading upstream's text onto the host or reading a
    copy a job environment installed. Unlike the belts above, a caller's setting does NOT stand: a cluster job's environment
    exports both variables at the real shared copy. A test that renders requests ``godmode_standin``
    (``tests/godmode_standin.py``), which points both at the stand-in; the opt-in live test (``SISTEER_GODMODE_LIVE=1``)
    names its own cache and deletes the offline variable itself."""
    tp_dir, offline = "SISTEER_THIRD_PARTY_DIR", "SISTEER_THIRD_PARTY_OFFLINE"   # attacks_godmode.THIRD_PARTY_DIR_ENV / OFFLINE_ENV; no import at session start
    saved = {k: os.environ.get(k) for k in (tp_dir, offline)}
    os.environ[tp_dir] = str(tmp_path_factory.mktemp("no-third-party"))
    os.environ[offline] = "1"
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(scope="session")
def tiny_llama():
    import torch

    from sisteer.models import load_model_and_tokenizer

    model, tok = load_model_and_tokenizer(
        TINY_LLAMA, device="cpu", dtype=torch.float32
    )
    return model, tok


@pytest.fixture(autouse=True)
def _stop_leaked_heartbeats(request):
    """Stop the ``fitmon.Heartbeat`` printer threads a test left running. A fit script under test
    that refuses (SystemExit) after building its heartbeat never reaches its ``hb.close()``; the
    daemon thread then prints ``[hb] pre-training: init (...)`` every 30 s into whichever LATER
    test has stdout captured, for the rest of the suite. The cc1f0e7d joint suite (2026-09-12)
    failed a byte-identity test over a captured dry run on one such line, with at least six
    leaked printers running by then. Acts only when the module is already imported (no
    heartbeat can exist otherwise), so it changes no test's import footprint. Set
    ``SISTEER_HB_LEAK_REPORT=<file>`` to append the node id of every test that leaked one."""
    import os
    import sys

    yield
    mod = sys.modules.get("sisteer.extraction.fitmon")
    if mod is None:
        return
    n = mod.Heartbeat.close_all()
    report = os.environ.get("SISTEER_HB_LEAK_REPORT")
    if n and report:
        with open(report, "a", encoding="utf-8") as fh:
            fh.write(f"{request.node.nodeid}\t{n}\n")


@pytest.fixture(autouse=True)
def _rsync_fallback_off_path_after_each_test():
    """The pure-Python ``rsync`` fallback (``tests/_rsync_fallback.py``) that a registry-store, cluster-read, landing or
    object-staging world builder puts on PATH for a host without the binary leaves PATH when its test ends. Without this,
    on such a host the first builder's ``ensure_rsync_on_path()`` stayed in ``os.environ["PATH"]`` for the rest of the
    session, and every later test that asks ``shutil.which("rsync")`` for its own skip (the colocate, mirror and
    grading-copy tests) found the shim instead of the host's absence and ran on a copy it never asked for, by test order.
    The wrapper directory itself (under the system temp dir) is removed with it, so a run that ends without its ``atexit``
    handlers (``os._exit``, a signal) leaves nothing behind. Acts only when the module is already imported (nothing can be
    installed otherwise), so it changes no test's import footprint; ``tests/test_rsync_fallback_scope.py`` pins it."""
    yield
    mod = sys.modules.get("_rsync_fallback")
    if mod is not None:
        mod.teardown_fallback()


@pytest.fixture
def unprovisioned_host(tmp_path, monkeypatch):
    """A host on which NO registry spool resolves: every fixed path ``sisteer.registry.emit`` probes (the per-user
    shared roots, ``~/registry``) is redirected under ``tmp_path`` and left absent, the spool env var is unset, and
    the login-name, HOME, scheduler and platform variables are synthetic. Hermetic on the grading host and a cluster
    node alike (the lesson of 2026-09-17: a gate container with a provisioned spool of its own made a test that
    isolated only the env vars and HOME find the live spool; the Modal volume probe itself left with the Modal
    transport). Returns the redirected paths; a test provisions a probe by creating the directory it names.
    In-process only: a producer spawned as a SUBPROCESS never sees these monkeypatches and is pointed at a tmp host
    through ``emit.PROBE_ROOT_ENV`` / ``emit.SUBSTRATE_ENV`` in its environment instead (the smoke-producer tests).
    This fixture leaves both variables unset — a probe root on top of the redirected constants would prefix them twice."""
    import types
    from pathlib import Path

    from sisteer.registry import emit

    root = tmp_path / "host"
    user = "nobody-synthetic"
    monkeypatch.delenv(emit.SPOOL_ENV, raising=False)
    for var in ("SLURM_CLUSTER_NAME", "RUNPOD_POD_ID", emit.SUBSTRATE_ENV, emit.PROBE_ROOT_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("USER", user)
    monkeypatch.setenv("HOME", str(root / "home"))
    monkeypatch.setattr(emit, "SHARED_BASES", tuple((name, str(root / template.lstrip("/"))) for name, template in emit.SHARED_BASES))
    return types.SimpleNamespace(
        root=root, user=user,
        shared={name: Path(template.format(user=user)) for name, template in emit.SHARED_BASES},
        home_registry=root / "home" / "registry")
