"""fits_watch.py successor spawning + the three sdfbase_fits.py seams
it rides on.

`modal` is not installed in the test env; a duck-typed stub stands in
(the test_sdfbase_fits_app.py pattern, extended with a real exception
hierarchy — modal.exception.TimeoutError is NOT the builtin — and
configurable FunctionCall / Function / Volume behaviour).

Pinned here:
- classification order: FunctionTimeoutError -> timeout;
  OutputExpiredError -> unknown; builtin TimeoutError -> running; any other
  raise -> failed; a return -> done (the pre-successor watcher filed a real
  function timeout as `failed` through its generic except).
- successor_item == the item tuple fits-spawn's source builds, from the
  ROW's own spawn pin (a merged gemma+qwen manifest
  respawns each row on ITS function with ITS config sha, never the
  manifest-level values of whichever model was spawned last; a pin-less
  row is refused).
- can_respawn: done never, COMPLETE (marker verdict) never, unverifiable
  never, cap.
- same_app_version: missing row pin / mismatch / match.
- spawn_successor mutates the cell and the manifest is rewritten
  atomically; main --once spawns exactly one successor, and a second pass
  at --max-successors 1 spawns nothing more.
- a COMPLETE cell is never respawned; a returned call is triage, never a
  successor; a redeployed app (pin mismatch) refuses loudly.
- sdfbase_fits.py: FIT_TIMEOUT_S honors SDFBASE_FIT_TIMEOUT at import, the
  two fit decorators read it, fits-spawn records function_id, watch mode
  forwards --respawn/--no-respawn and --max-successors.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
FW_PATH = REPO / "scripts" / "modal" / "fits_watch.py"
SF_PATH = REPO / "scripts" / "modal" / "sdfbase_fits.py"
SF_SRC = SF_PATH.read_text()
FW_SRC = FW_PATH.read_text()

SID = "33333333-4444-5555-6666-777777777777"  # synthetic, never real


class _Chain:
    def __getattr__(self, name):
        def _method(*a, **k):
            return self
        return _method


FP = "a" * 64          # the expected fit fingerprint the manifest rows carry
STALE = "b" * 64


def make_fake_modal() -> types.ModuleType:
    """A `modal` stub with a registry the tests mutate:
    fake.calls[call_id] = ("raise", exc) | ("return", value)
    fake.markers = {volume path: COMPLETE marker bytes} (Volume.read_file)
    fake.complete = {volume paths that list an entry} (Volume.listdir)
    fake.function_id = the deployed function's id after hydrate()
    fake.spawned = [items passed to Function.spawn], fake.cancelled = [...]
    fake.remote = handler for Function.remote(item) (fingerprint_cpu)"""
    fake = types.ModuleType("modal")

    class Error(Exception):
        pass

    class ModalTimeoutError(Error):  # modal.exception.TimeoutError
        pass

    class FunctionTimeoutError(ModalTimeoutError):
        pass

    class OutputExpiredError(ModalTimeoutError):
        pass

    fake.exception = SimpleNamespace(
        Error=Error, TimeoutError=ModalTimeoutError,
        FunctionTimeoutError=FunctionTimeoutError,
        OutputExpiredError=OutputExpiredError)
    fake.calls = {}
    fake.markers = {}
    fake.complete = set()
    fake.function_id = "fu-deployed-1"
    fake.function_ids = {}      # per deployed-function name (merged manifests)
    # the live DEPLOYMENT pin (fits_watch.deployment_pin);
    # None = the lookup fails (an unknown deployment refuses successors)
    fake.deployment = {"app_id": "ap-1", "app_version": 1,
                       "deploy_commit": "abc123def456",
                       # one record per `modal deploy`, epoch seconds
                       "history": [{"version": 1, "deployed_at": 1_700_000_000.0}]}
    fake.spawned = []
    fake.spawn_calls = []       # (app, function name, item)
    fake.cancelled = []
    fake.hydrated = 0
    fake.from_name_calls = []
    fake.remote = lambda item: {"model": item[0], "config_sha": None,
                                "fingerprints": {}, "missing_pools": []}

    class _FC:
        def __init__(self, cid):
            self.object_id = cid

        def get(self, timeout=None):
            kind, val = fake.calls.get(
                self.object_id, ("raise", TimeoutError("still running")))
            if kind == "raise":
                raise val
            return val

        def cancel(self, terminate_containers=False):
            fake.cancelled.append((self.object_id, terminate_containers))

    fake.FunctionCall = SimpleNamespace(from_id=lambda cid, **k: _FC(cid))

    class _Fn:
        def __init__(self, app, name):
            self.app, self.name = app, name
            self.object_id = None

        def hydrate(self):
            fake.hydrated += 1
            self.object_id = fake.function_ids.get(self.name, fake.function_id)
            return self

        def spawn(self, item):
            fake.spawned.append(item)
            fake.spawn_calls.append((self.app, self.name, item))
            return SimpleNamespace(object_id=f"fc-succ-{len(fake.spawned)}")

        def remote(self, item):
            return fake.remote(item)

    def _from_name(app, name, **k):
        fake.from_name_calls.append((app, name))
        return _Fn(app, name)

    fake.Function = SimpleNamespace(from_name=_from_name)

    fake.reloads = 0          # Volume.reload() calls (snapshot refreshes)
    fake.commits = 0

    class _Vol:
        def listdir(self, path, **k):
            if path in fake.complete:
                return [SimpleNamespace(path=path, mtime=1, size=1, type=1)]
            return []

        def commit(self):
            fake.commits += 1

        def reload(self):
            fake.reloads += 1

        def read_file(self, path):
            if path not in fake.markers:
                raise FileNotFoundError(path)
            yield fake.markers[path]

    fake.Volume = SimpleNamespace(from_name=lambda *a, **k: _Vol())

    class _App:
        def __init__(self, name):
            self.name = name

        def function(self, *a, **k):
            return lambda fn: fn

        def local_entrypoint(self, *a, **k):
            return lambda fn: fn

    fake.App = _App
    fake.Image = SimpleNamespace(debian_slim=lambda **k: _Chain(),
                                 from_registry=lambda *a, **k: _Chain())
    # from_dict too: a stub that outlives its module must not break a later
    # import of scripts/modal/grokmt_served.py (module-level Secret.from_dict)
    fake.Secret = SimpleNamespace(from_name=lambda n: ("secret", n),
                                  from_dict=lambda d: ("secret", tuple(sorted(d))))
    return fake


def _load(path: Path, name: str, fake) -> types.ModuleType:
    prev = sys.modules.get("modal")
    sys.modules["modal"] = fake
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if prev is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = prev
    return mod


@pytest.fixture
def fake():
    return make_fake_modal()


@pytest.fixture
def fw(fake):
    """fits_watch imports `modal` lazily inside its probes, so the stub
    must be in sys.modules for the TEST's duration, not just at import."""
    prev = sys.modules.get("modal")
    sys.modules["modal"] = fake
    try:
        yield _load(FW_PATH, "_fits_watch_under_test", fake)
    finally:
        if prev is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = prev


def _cell(cid="fc-a", model="gemma4e4b_base", si="constv2-mix-1k",
          arm="psr_ce_all", root="out/gemma4e4b_base", **extra):
    """A manifest row as fits-spawn writes it: the expected fingerprint AND
    the row's own spawn pin (app / function / function_id / config_sha /
    overwrite / checkpoint_every)."""
    c = {"model": model, "si": si, "arm": arm, "positions": "all",
         "leg": "all", "cell_id": f"{si}__{arm}", "out_root": root,
         "expect_fingerprint": FP,
         "app": "sdfbase-fits", "function": "fit_h200_std",
         "function_id": "fu-deployed-1", "config_sha": "abcdef0123456789",
         "overwrite": False, "checkpoint_every": 100,
         "continue_cap": 0, "continue_lr_mult": 1.0,
         "app_id": "ap-1", "app_version": 1, "deploy_commit": "abc123def456",
         "call_id": cid, "spawn_time": "2026-01-01T00:00:00+00:00",
         "status": "spawned (in-container skip expected)"}
    c.update(extra)
    return c


def _marker(fp=FP) -> bytes:
    return json.dumps({"cell_id": "constv2-mix-1k__psr_ce_all",
                       "fit_fingerprint": fp,
                       "files": {"object.pt": "0" * 64}}).encode()


def _doc(cells, function_id="fu-deployed-1", **extra):
    d = {"app": "sdfbase-fits", "function": "fit_h200_std",
         "model": "gemma4e4b_base", "function_id": function_id,
         "out_root": "out/gemma4e4b_base", "config_sha": "abcdef0123456789",
         "checkpoint_every": 100, "overwrite": False, "launcher": "TESTUSER",
         "written_at": None, "cells": cells}
    d.update(extra)
    return d


def _complete_path(c):
    return f"{c['out_root']}/cells_{c['leg']}/{c['cell_id']}/COMPLETE"


# --------------------------------------------------------------------------- #
# classification                                                                #
# --------------------------------------------------------------------------- #


def test_classify_call_order(fw, fake):
    ex = fake.exception
    c = _cell("fc-1")
    fake.calls["fc-1"] = ("raise", ex.FunctionTimeoutError("Function ran too long"))
    assert fw.classify_call(c) == "timeout"
    assert c["_call_error"].startswith("FunctionTimeoutError")
    fake.calls["fc-1"] = ("raise", ex.OutputExpiredError("gone"))
    assert fw.classify_call(_cell("fc-1")) == "unknown"
    fake.calls["fc-1"] = ("raise", TimeoutError("still running"))
    assert fw.classify_call(_cell("fc-1")) == "running"
    fake.calls["fc-1"] = ("raise", RuntimeError("remote boom"))
    c = _cell("fc-1")
    assert fw.classify_call(c) == "failed"
    assert c["_call_error"] == "RuntimeError: remote boom"
    fake.calls["fc-1"] = ("return", {"status": "ok"})
    assert fw.classify_call(_cell("fc-1")) == "done"
    assert fw.classify_call(_cell(None)) == "unspawned"
    # the back-compat name still points at the classifier
    assert fw._call_state is fw.classify_call


def test_modal_timeout_is_not_the_builtin(fw, fake):
    """The modal hierarchy in the stub mirrors 1.5.4: FunctionTimeoutError
    is a modal TimeoutError, which is NOT the builtin — so the builtin
    clause alone would never have caught it, and a generic except filed
    it as failed. The source must test the modal classes first."""
    assert not issubclass(fake.exception.FunctionTimeoutError, TimeoutError)
    body = FW_SRC.split("def classify_call", 1)[1].split("\ndef ", 1)[0]
    order = [body.index("except modal.exception.FunctionTimeoutError"),
             body.index("except modal.exception.OutputExpiredError"),
             body.index("except TimeoutError"),
             body.index("except Exception")]
    assert order == sorted(order), "classification order is load-bearing"


# --------------------------------------------------------------------------- #
# pure successor functions                                                      #
# --------------------------------------------------------------------------- #


def test_successor_item_equals_fits_spawn_item(fw):
    """Evaluate fits-spawn's OWN item expression (lifted from the source)
    against a manifest row and require successor_item to rebuild it from
    the ROW's spawn pin alone."""
    block = SF_SRC.split('if mode == "fits-spawn"', 1)[1].split(
        'if mode == "watch"', 1)[0]
    # one level of nesting, for the int(checkpoint_every) member
    m = re.search(r"item = (\((?:[^()]|\([^()]*\))*\))", block, re.S)
    assert m, "fits-spawn's item tuple not found"
    expr = " ".join(m.group(1).split())
    c = _cell(overwrite=True, checkpoint_every=7)
    ns = {"key": c["model"], "c": c, "expect_sha": c["config_sha"],
          "overwrite": c["overwrite"], "root": c["out_root"],
          "checkpoint_every": c["checkpoint_every"]}
    assert fw.successor_item(c) == eval(expr, {}, ns)  # noqa: S307
    assert fw.successor_item(c) == (
        "gemma4e4b_base", "constv2-mix-1k", "psr_ce_all", "abcdef0123456789",
        True, "out/gemma4e4b_base", 7)
    assert isinstance(fw.successor_item(c)[4], bool)
    assert isinstance(fw.successor_item(c)[6], int)
    # the pin is the ROW's: a manifest-level value never leaks in
    assert fw.ROW_PIN_FIELDS == ("app", "function", "function_id",
                                 "config_sha", "overwrite", "checkpoint_every",
                                 "continue_cap", "continue_lr_mult",
                                 "app_id", "app_version", "deploy_commit")
    pin, src = fw.row_pin(c)
    assert src == "row" and pin["function"] == "fit_h200_std"
    legacy = {k: v for k, v in _cell().items() if k not in fw.ROW_PIN_FIELDS}
    pin, why = fw.row_pin(legacy)
    assert pin is None and "re-run fits-spawn" in why
    with pytest.raises(ValueError, match="re-run fits-spawn"):
        fw.successor_item(legacy)


@pytest.mark.parametrize("cause", ["timeout", "failed"])
def test_can_respawn_yes_for_continuation_causes(fw, cause):
    ok, why = fw.can_respawn(_cell(), cause, max_successors=3)
    assert ok and why == "ok"


@pytest.mark.parametrize("cause", ["done", "running", "unknown", "unspawned"])
def test_can_respawn_never_for_non_continuation_states(fw, cause):
    ok, why = fw.can_respawn(_cell(), cause, max_successors=3)
    assert not ok and "not a continuation event" in why


def test_can_respawn_never_when_complete_or_unverifiable(fw):
    """The marker VERDICT (as read by the caller's pass) decides — not a
    `complete` key nobody sets."""
    ok, why = fw.can_respawn(_cell(), "timeout", max_successors=3,
                             marker="complete")
    assert not ok and "COMPLETE" in why
    ok, why = fw.can_respawn(_cell(), "timeout", max_successors=3,
                             marker="unverifiable")
    assert not ok and "re-run fits-spawn" in why
    for m in ("absent", "stale", "no-fingerprint", "unparseable"):
        assert fw.can_respawn(_cell(), "timeout", max_successors=3,
                              marker=m)[0]
    assert "complete" not in FW_SRC.split("def can_respawn", 1)[1].split(
        "\ndef ", 1)[0].replace("COMPLETE", "").replace('"complete"', "")


def test_can_respawn_refuses_an_out_of_scope_arm(fw):
    """A row whose arm the launcher no longer runs is never continued.

    2026-09-12: the PSR arms were descoped and their calls cancelled. A
    cancelled call classifies as `failed`, which IS a continuation cause,
    so a watcher pointed at a manifest that still carried those rows
    respawned all four — reviving work that had just been ruled out. The
    guard keys on the LAUNCHER'S ARMS rather than on `psr`, so the next
    descoped arm is covered without a code change. Omitting `arms` keeps
    the old unrestricted behaviour for every other caller.
    """
    live, descoped = _cell(arm="lora_r4_ce"), _cell(arm="psr_ce_all")
    # unrestricted: unchanged
    assert fw.can_respawn(descoped, "failed", max_successors=3)[0]
    # restricted to the arms the launcher actually runs
    ok, why = fw.can_respawn(descoped, "failed", max_successors=3,
                             arms=("lora_r4_ce",))
    assert not ok and "psr_ce_all" in why
    # and a timeout, the rollover cause, is refused just the same
    ok, _ = fw.can_respawn(descoped, "timeout", max_successors=3,
                           arms=("lora_r4_ce",))
    assert not ok
    # the in-scope arm still continues, or the guard would break the rollover
    assert fw.can_respawn(live, "timeout", max_successors=3,
                          arms=("lora_r4_ce",))[0]


def test_can_respawn_cap(fw):
    two = [{"call_id": "x"}, {"call_id": "y"}]
    ok, _ = fw.can_respawn(_cell(successors=two), "timeout", max_successors=3)
    assert ok
    ok, why = fw.can_respawn(_cell(successors=two), "timeout", max_successors=2)
    assert not ok and "cap" in why
    ok, why = fw.can_respawn(_cell(successors=two + [{}]), "failed",
                             max_successors=3)
    assert not ok and "3/3" in why
    ok, _ = fw.can_respawn(_cell(), "timeout", max_successors=0)
    assert not ok


def test_same_app_version(fw):
    """The version pin: the ROW's function_id (necessary, not sufficient —
    it does not change on redeploy) AND the row's app_version against the
    live production deployment."""
    fn = SimpleNamespace(object_id="fu-deployed-1")
    live = {"app_id": "ap-1", "app_version": 1, "deploy_commit": "abc123def456"}
    ok, why = fw.same_app_version(_cell(function_id=None), fn, live)
    assert not ok and "predates the version pin" in why
    c = _cell()
    c.pop("function_id")
    ok, why = fw.same_app_version(c, None)  # no pin: fn never consulted
    assert not ok and "respawn by hand" in why
    ok, why = fw.same_app_version(_cell(function_id="fu-old"), fn, live)
    assert not ok and "redeployed" in why and "fu-old" in why
    ok, why = fw.same_app_version(_cell(), fn, live)
    assert ok and why == "ok"
    # the function id is UNCHANGED across a redeploy (v10 -> v11 kept both
    # fit ids): only the deployment layer can refuse it
    moved = {"app_id": "ap-1", "app_version": 2, "deploy_commit": "fedcba"}
    ok, why = fw.same_app_version(_cell(), fn, moved)
    assert not ok and "redeployed" in why and "v1" in why and "v2" in why
    assert "fedcba" in why
    # a different app under the same name
    ok, why = fw.same_app_version(_cell(), fn, {**live, "app_id": "ap-9"})
    assert not ok and "different app" in why
    # the live deployment could not be read: refuse, never guess
    ok, why = fw.same_app_version(_cell(), fn, None)
    assert not ok and "could not be read" in why
    ok, why = fw.same_app_version(_cell(), fn, {"app_id": "ap-1"})
    assert not ok and "could not be read" in why
    # a LEGACY row (spawned before the deployment pin — the cont2 manifests):
    # the function id alone proves nothing, so it is refused unless the
    # operator opts into inferring EACH row's version from its own
    # spawn_time against the deployment history (never one number for all)
    hist = [{"version": 1, "deployed_at": 1_000.0},
            {"version": 2, "deployed_at": 2_000.0}]
    live_h = {**live, "history": hist}
    moved_h = {**moved, "history": hist}
    t_v1 = "1970-01-01T00:25:00+00:00"       # 1500 s: after v1, before v2
    t_v2 = "1970-01-01T00:41:40+00:00"       # 2500 s: after v2
    legacy = _cell(spawn_time=t_v1)
    for k in ("app_id", "app_version", "deploy_commit"):
        legacy.pop(k)
    ok, why = fw.same_app_version(legacy, fn, live_h)
    assert not ok and "no deployment pin" in why
    assert "--infer-legacy-app-version" in why
    ok, why = fw.same_app_version(legacy, fn, live_h, infer_legacy=True)
    assert ok and why == "ok"
    # the live app moved to v2: the v1-era row is refused, a v2-era row is
    # allowed — the same manifest, two verdicts
    ok, why = fw.same_app_version(legacy, fn, moved_h, infer_legacy=True)
    assert not ok and "inferred from spawn_time" in why
    assert "v1 != live production v2" in why
    later = {**legacy, "spawn_time": t_v2}
    ok, why = fw.same_app_version(later, fn, moved_h, infer_legacy=True)
    assert ok
    ok, why = fw.same_app_version(later, fn, live_h, infer_legacy=True)
    assert not ok and "v2 != live production v1" in why
    # unplaceable rows refuse: no spawn_time, a spawn before the first
    # deploy, an unparseable time, a history without dates
    for bad in ({**legacy, "spawn_time": None},
                {**legacy, "spawn_time": "1970-01-01T00:00:01+00:00"},
                {**legacy, "spawn_time": "not-a-time"}):
        ok, why = fw.same_app_version(bad, fn, live_h, infer_legacy=True)
        assert not ok and "cannot be placed" in why, why
    ok, why = fw.same_app_version(legacy, fn, {**live, "history": []},
                                  infer_legacy=True)
    assert not ok and "cannot be placed" in why
    ok, why = fw.same_app_version(legacy, fn, None, infer_legacy=True)
    assert not ok and "could not be read" in why
    # an explicit row pin is never overridden by inference
    ok, why = fw.same_app_version(_cell(app_version=1, spawn_time=t_v2), fn,
                                  live_h, infer_legacy=True)
    assert ok


def test_version_at_places_a_spawn_in_the_deployment_history(fw):
    hist = [{"version": 10, "deployed_at": 1788412084.0},   # v10
            {"version": 11, "deployed_at": 1788463423.1},   # v11
            {"version": 9, "deployed_at": 1788382029.0}]    # unsorted on purpose
    assert fw.version_at(hist, "2026-09-03T19:32:06+00:00") == 11
    assert fw.version_at(hist, "2026-09-03T06:18:00Z") == 10
    assert fw.version_at(hist, "2026-09-02T21:00:00+00:00") == 9
    assert fw.version_at(hist, "2026-09-01T00:00:00+00:00") is None
    assert fw.version_at(hist, None) is None
    assert fw.version_at(hist, "yesterday") is None
    assert fw.version_at(None, "2026-09-03T19:32:06+00:00") is None
    assert fw.version_at([{"version": 1}], "2026-09-03T19:32:06+00:00") is None
    # a naive timestamp is read as UTC
    assert fw.version_at(hist, "2026-09-03T19:32:06") == 11


def test_spawn_successor_mutates_cell_and_returns_record(fw):
    spawned = []

    class Fn:
        object_id = "fu-deployed-1"

        def spawn(self, item):
            spawned.append(item)
            return SimpleNamespace(object_id=f"fc-new-{len(spawned)}")

    c = _cell("fc-old", _call_error="FunctionTimeoutError: ran too long")
    rec = fw.spawn_successor(c, "timeout", Fn(), "2026-01-02T00:00:00+00:00")
    assert spawned == [fw.successor_item(c)]
    assert rec == {"call_id": "fc-new-1", "predecessor_call_id": "fc-old",
                   "cause": "timeout",
                   "spawn_time": "2026-01-02T00:00:00+00:00",
                   "app": "sdfbase-fits", "function": "fit_h200_std",
                   "function_id": "fu-deployed-1",
                   "app_id": None, "app_version": None, "deploy_commit": None,
                   "predecessor_error": "FunctionTimeoutError: ran too long"}
    assert c["successors"] == [rec]
    assert c["call_id"] == "fc-new-1"
    assert c["spawn_time"] == "2026-01-02T00:00:00+00:00"
    assert c["status"] == "successor #1 spawned (timeout)"
    assert "_call_error" not in c  # moved into the record, not left stale
    rec2 = fw.spawn_successor(c, "failed", Fn(), "2026-01-03T00:00:00+00:00")
    assert rec2["predecessor_call_id"] == "fc-new-1"
    assert c["call_id"] == "fc-new-2"
    assert c["status"] == "successor #2 spawned (failed)"
    assert len(c["successors"]) == 2


def test_write_manifest_is_atomic(fw, tmp_path):
    mpath = tmp_path / "fit_calls.json"
    d = _doc([_cell()])
    fw.write_manifest(d, mpath)
    on_disk = json.loads(mpath.read_text())
    assert on_disk["cells"][0]["cell_id"] == "constv2-mix-1k__psr_ce_all"
    assert on_disk["written_at"]
    assert not (tmp_path / "fit_calls.json.tmp").exists()
    src = FW_SRC.split("def write_manifest", 1)[1].split("\ndef ", 1)[0]
    assert "os.replace(" in src


# --------------------------------------------------------------------------- #
# main() --once end to end                                                      #
# --------------------------------------------------------------------------- #


def _run_once(fw, monkeypatch, mpath, *extra):
    monkeypatch.setattr(sys, "argv", ["fits_watch.py", "--manifest",
                                      str(mpath), "--once", *extra])

    def _pin(app):
        # the stub's live deployment (fits_watch.deployment_pin is the
        # modal AppDeploymentHistory call; the stub cannot serve it)
        dep = sys.modules["modal"].deployment
        if dep is None:
            raise RuntimeError("deployment lookup unavailable")
        return dict(dep)

    monkeypatch.setattr(fw, "deployment_pin", _pin)
    return fw.main()


def test_main_once_spawns_exactly_one_successor_then_honors_cap(
        fw, fake, tmp_path, monkeypatch, capsys):
    a = _cell("fc-a")
    b = _cell("fc-b", arm="lora_r4_ce")
    mpath = tmp_path / "fit_calls.json"
    mpath.write_text(json.dumps(_doc([a, b])))
    fake.calls["fc-a"] = ("raise", fake.exception.FunctionTimeoutError("24h"))
    fake.calls["fc-b"] = ("raise", TimeoutError("running"))

    rc = _run_once(fw, monkeypatch, mpath, "--max-successors", "1")
    out = capsys.readouterr().out
    assert rc == 0, out  # the successor is in flight, nothing needs attention
    assert fake.spawned == [("gemma4e4b_base", "constv2-mix-1k", "psr_ce_all",
                             "abcdef0123456789", False, "out/gemma4e4b_base",
                             100)]
    assert fake.hydrated == 1
    assert fake.from_name_calls == [("sdfbase-fits", "fit_h200_std")]
    assert re.search(r"SUCCESSOR gemma4e4b_base/constv2-mix-1k__psr_ce_all/all: "
                     r"cause=timeout #1 -> fc-succ-1", out)
    doc = json.loads(mpath.read_text())
    ca = next(c for c in doc["cells"] if c["arm"] == "psr_ce_all")
    assert ca["call_id"] == "fc-succ-1"
    assert len(ca["successors"]) == 1
    rec = ca["successors"][0]
    assert rec["predecessor_call_id"] == "fc-a"
    assert rec["cause"] == "timeout"
    assert rec["function_id"] == "fu-deployed-1"
    assert rec["predecessor_error"].startswith("FunctionTimeoutError")
    assert ca["status"] == "successor #1 spawned (timeout)"
    assert "_call_error" not in ca
    cb = next(c for c in doc["cells"] if c["arm"] == "lora_r4_ce")
    assert cb["call_id"] == "fc-b" and "successors" not in cb
    assert not (tmp_path / "fit_calls.json.tmp").exists()

    # Second pass: the successor itself times out. Cap 1 -> no new spawn,
    # the cell becomes a triage row (exit 2 under --once).
    fake.calls["fc-succ-1"] = ("raise", fake.exception.FunctionTimeoutError("again"))
    rc = _run_once(fw, monkeypatch, mpath, "--max-successors", "1")
    out = capsys.readouterr().out
    assert rc == 2, out
    assert len(fake.spawned) == 1
    assert "NO-RESPAWN" in out and "successor cap reached (1/1)" in out
    doc = json.loads(mpath.read_text())
    ca = next(c for c in doc["cells"] if c["arm"] == "psr_ce_all")
    assert ca["call_id"] == "fc-succ-1" and len(ca["successors"]) == 1

    # With a higher cap the same state DOES spawn #2, on the same item.
    rc = _run_once(fw, monkeypatch, mpath, "--max-successors", "3")
    out = capsys.readouterr().out
    assert rc == 0 and len(fake.spawned) == 2
    assert "cause=timeout #2 -> fc-succ-2" in out
    doc = json.loads(mpath.read_text())
    ca = next(c for c in doc["cells"] if c["arm"] == "psr_ce_all")
    assert [s["predecessor_call_id"] for s in ca["successors"]] == \
        ["fc-a", "fc-succ-1"]


def test_main_failed_call_is_also_a_continuation_event(
        fw, fake, tmp_path, monkeypatch, capsys):
    a = _cell("fc-a")
    mpath = tmp_path / "m.json"
    mpath.write_text(json.dumps(_doc([a])))
    fake.calls["fc-a"] = ("raise", RuntimeError("container killed"))
    rc = _run_once(fw, monkeypatch, mpath)
    out = capsys.readouterr().out
    assert rc == 0 and len(fake.spawned) == 1
    assert "cause=failed #1" in out
    rec = json.loads(mpath.read_text())["cells"][0]["successors"][0]
    assert rec["cause"] == "failed"
    assert rec["predecessor_error"] == "RuntimeError: container killed"


def test_main_no_respawn_flag_only_watches(fw, fake, tmp_path, monkeypatch,
                                           capsys):
    mpath = tmp_path / "m.json"
    before = json.dumps(_doc([_cell("fc-a")]))
    mpath.write_text(before)
    fake.calls["fc-a"] = ("raise", fake.exception.FunctionTimeoutError("24h"))
    rc = _run_once(fw, monkeypatch, mpath, "--no-respawn")
    out = capsys.readouterr().out
    assert rc == 2 and fake.spawned == []
    assert "respawning disabled" in out
    assert mpath.read_text() == before  # never rewritten


def test_complete_cell_is_never_respawned(fw, fake, tmp_path, monkeypatch,
                                          capsys):
    a = _cell("fc-a")
    mpath = tmp_path / "m.json"
    before = json.dumps(_doc([a]))
    mpath.write_text(before)
    fake.markers[_complete_path(a)] = _marker(FP)  # fingerprint-matching
    fake.calls["fc-a"] = ("raise", fake.exception.FunctionTimeoutError("24h"))
    rc = _run_once(fw, monkeypatch, mpath)
    out = capsys.readouterr().out
    assert rc == 0 and "ALL-COMPLETE" in out
    assert fake.spawned == [] and fake.hydrated == 0
    assert mpath.read_text() == before


def test_stale_marker_under_a_timed_out_call_gets_a_successor(
        fw, fake, tmp_path, monkeypatch, capsys):
    """The R2 finding-1b picture crossed with a continuation event: a
    round-1 marker (another fingerprint) sits under the refit's cell dir
    and the refit call timed out — the cell is NOT complete and it IS a
    continuation event, so exactly one successor goes out."""
    a = _cell("fc-a")
    mpath = tmp_path / "m.json"
    mpath.write_text(json.dumps(_doc([a])))
    fake.markers[_complete_path(a)] = _marker(STALE)
    fake.calls["fc-a"] = ("raise", fake.exception.FunctionTimeoutError("24h"))
    rc = _run_once(fw, monkeypatch, mpath)
    out = capsys.readouterr().out
    assert rc == 0 and len(fake.spawned) == 1
    assert "cause=timeout #1 -> fc-succ-1" in out
    assert json.loads(mpath.read_text())["cells"][0]["call_id"] == "fc-succ-1"


def test_unverifiable_row_is_never_respawned(fw, fake, tmp_path, monkeypatch,
                                             capsys):
    """A strict-app row without expect_fingerprint can never be validated
    complete, so a successor could never be harvested: triage, no spawn,
    and the call is not even probed."""
    a = _cell("fc-a")
    a.pop("expect_fingerprint")
    mpath = tmp_path / "m.json"
    before = json.dumps(_doc([a]))
    mpath.write_text(before)
    fake.calls["fc-a"] = ("raise", fake.exception.FunctionTimeoutError("24h"))
    rc = _run_once(fw, monkeypatch, mpath)
    out = capsys.readouterr().out
    assert rc == 2 and fake.spawned == [] and fake.hydrated == 0
    assert "re-run fits-spawn" in out
    assert mpath.read_text() == before


def test_returned_call_without_complete_is_triage_not_successor(
        fw, fake, tmp_path, monkeypatch, capsys):
    mpath = tmp_path / "m.json"
    before = json.dumps(_doc([_cell("fc-a")]))
    mpath.write_text(before)
    fake.calls["fc-a"] = ("return", {"status": "no-pool", "rc": 1})
    rc = _run_once(fw, monkeypatch, mpath)
    out = capsys.readouterr().out
    assert rc == 2 and fake.spawned == [] and fake.hydrated == 0
    assert "WAVE-INCOMPLETE" in out
    assert "call done but no COMPLETE marker" in out
    assert mpath.read_text() == before


def test_assess_respawn_hook_contract(fw):
    """assess() without the hook keeps its historical shape (a timed-out
    call is a triage row, like done/failed); with the hook, a spawned
    successor puts the cell back in flight and a refusal carries its
    reason into the triage line. Pure — no modal touched."""
    rows = [_cell("fc-a")]
    calls = {"constv2-mix-1k__psr_ce_all": "timeout"}
    probe = lambda c: calls[c["cell_id"]]  # noqa: E731
    res = fw.assess(rows, strict=True, read_marker=lambda c: None,
                    call_state=probe)
    assert res["all_calls_terminal"] is True
    assert res["needs_attention"] == [
        "gemma4e4b_base/constv2-mix-1k__psr_ce_all/all: call timeout but "
        "no COMPLETE marker"]
    assert res["calls"] == {"gemma4e4b_base/constv2-mix-1k__psr_ce_all/all":
                            "timeout"}
    seen = []

    def spawned(c, call, cid, marker):
        seen.append((call, cid, marker))
        c["status"] = "successor #1 spawned (timeout)"
        c["successors"] = [{"call_id": "fc-succ-1"}]
        return True, c["status"]

    res = fw.assess([_cell("fc-a")], strict=True, read_marker=lambda c: None,
                    call_state=probe, respawn=spawned)
    # the hook sees the marker verdict this pass read (absent here)
    assert seen == [("timeout", "gemma4e4b_base/constv2-mix-1k__psr_ce_all/all",
                     "absent")]
    assert res["all_calls_terminal"] is False and res["needs_attention"] == []
    assert res["lines"]["gemma4e4b_base/constv2-mix-1k__psr_ce_all/all"] \
        .endswith("successor #1 spawned (timeout)")
    res = fw.assess([_cell("fc-a")], strict=True, read_marker=lambda c: None,
                    call_state=probe,
                    respawn=lambda c, call, cid, marker: (False, "cap"))
    assert res["all_calls_terminal"] is True
    assert res["needs_attention"][0].endswith("— not respawned: cap")
    # the hook is never consulted for a returned call or a complete marker
    calls["constv2-mix-1k__psr_ce_all"] = "done"
    res = fw.assess([_cell("fc-a")], strict=True, read_marker=lambda c: None,
                    call_state=probe,
                    respawn=lambda *a: pytest.fail("hook called for done"))
    assert "call done but" in res["needs_attention"][0]
    calls["constv2-mix-1k__psr_ce_all"] = "timeout"
    res = fw.assess([_cell("fc-a")], strict=True,
                    read_marker=lambda c: _marker(FP), call_state=probe,
                    respawn=lambda *a: pytest.fail("hook called for complete"))
    assert res["n_complete"] == 1
    # a stale marker under a timed-out call reaches the hook AS "stale"
    seen.clear()
    fw.assess([_cell("fc-a")], strict=True,
              read_marker=lambda c: _marker(STALE), call_state=probe,
              respawn=spawned)
    assert seen[0][2] == "stale"


def test_redeployed_app_refuses_successor_loudly(fw, fake, tmp_path,
                                                 monkeypatch, capsys):
    mpath = tmp_path / "m.json"
    before = json.dumps(_doc([_cell("fc-a", function_id="fu-old")],
                             function_id="fu-old"))
    mpath.write_text(before)
    fake.function_id = "fu-new"
    fake.calls["fc-a"] = ("raise", fake.exception.FunctionTimeoutError("24h"))
    rc = _run_once(fw, monkeypatch, mpath)
    out = capsys.readouterr().out
    assert rc == 2 and fake.spawned == []
    assert "NO-RESPAWN" in out and "redeployed" in out
    assert "fu-old" in out and "fu-new" in out
    assert mpath.read_text() == before


def test_redeploy_with_unchanged_function_id_refuses_successor(
        fw, fake, tmp_path, monkeypatch, capsys):
    """The case the function-id pin could not see: sdfbase-
    fits v10 -> v11 kept fit_h200_std's id. The row was spawned on
    deployment v1; the app is now at v2 with the SAME function id — the
    successor must be refused, naming both versions."""
    mpath = tmp_path / "m.json"
    before = json.dumps(_doc([_cell("fc-a")]))
    mpath.write_text(before)
    fake.deployment = {"app_id": "ap-1", "app_version": 2,
                       "deploy_commit": "fedcba987654"}
    fake.calls["fc-a"] = ("raise", fake.exception.FunctionTimeoutError("24h"))
    rc = _run_once(fw, monkeypatch, mpath)
    out = capsys.readouterr().out
    assert rc == 2 and fake.spawned == []
    assert "NO-RESPAWN" in out and "redeployed" in out
    assert "v1" in out and "v2" in out and "fedcba987654" in out
    assert mpath.read_text() == before
    # the same live deployment as at spawn: the successor goes out and its
    # record carries the deployment it runs on
    fake.deployment = {"app_id": "ap-1", "app_version": 1,
                       "deploy_commit": "abc123def456",
                       # one record per `modal deploy`, epoch seconds
                       "history": [{"version": 1, "deployed_at": 1_700_000_000.0}]}
    rc = _run_once(fw, monkeypatch, mpath)
    assert len(fake.spawned) == 1
    rec = json.loads(mpath.read_text())["cells"][0]["successors"][0]
    assert (rec["app_id"], rec["app_version"], rec["deploy_commit"]) == (
        "ap-1", 1, "abc123def456")
    assert "v1 abc123def456" in capsys.readouterr().out


def test_legacy_rows_are_inferred_per_row(fw, fake, tmp_path, monkeypatch,
                                          capsys):
    """Rows spawned before the deployment pin existed (the cont2
    manifests, whose only pin is the vacuous function id): refused by
    default with the relaunch hint; --infer-legacy-app-version places EACH
    row's spawn_time in the deployment history. A merged manifest with a
    v1-era row and a v2-era row, live at v2: exactly the v2-era row gets a
    successor."""
    fake.deployment = {"app_id": "ap-1", "app_version": 2,
                       "deploy_commit": "fedcba987654",
                       "history": [{"version": 1, "deployed_at": 1_000.0},
                                   {"version": 2, "deployed_at": 2_000.0}]}
    old_row = _cell("fc-a", spawn_time="1970-01-01T00:25:00+00:00")   # v1 era
    new_row = _cell("fc-b", arm="lora_r4_ce",
                    spawn_time="1970-01-01T00:41:40+00:00")           # v2 era
    for r in (old_row, new_row):
        for k in ("app_id", "app_version", "deploy_commit"):
            r.pop(k)
    mpath = tmp_path / "m.json"
    before = json.dumps(_doc([old_row, new_row]))
    mpath.write_text(before)
    fake.calls["fc-a"] = ("raise", fake.exception.FunctionTimeoutError("24h"))
    fake.calls["fc-b"] = ("raise", fake.exception.FunctionTimeoutError("24h"))
    rc = _run_once(fw, monkeypatch, mpath)
    out = capsys.readouterr().out
    assert rc == 2 and fake.spawned == []
    assert out.count("NO-RESPAWN") == 2 and "no deployment pin" in out
    assert "--infer-legacy-app-version" in out
    assert mpath.read_text() == before
    rc = _run_once(fw, monkeypatch, mpath, "--infer-legacy-app-version")
    out = capsys.readouterr().out
    assert len(fake.spawned) == 1
    assert fake.spawned[0][2] == "lora_r4_ce"            # only the v2-era row
    assert "NO-RESPAWN gemma4e4b_base/constv2-mix-1k__psr_ce_all/all" in out
    assert "inferred from spawn_time" in out and "v1 != live production v2" in out
    saved = {c["arm"]: c for c in json.loads(mpath.read_text())["cells"]}
    assert "successors" not in saved["psr_ce_all"]
    rec = saved["lora_r4_ce"]["successors"][0]
    assert rec["app_version"] == 2 and rec["app_id"] == "ap-1"
    assert rec["row_version_source"] == "inferred_from_spawn_time"
    # a pinned row records its source as the row
    pinned = _cell("fc-c", arm="psr_ce_all", si="constv2-mix-1m",
                   app_version=2)
    mpath2 = tmp_path / "m2.json"
    mpath2.write_text(json.dumps(_doc([pinned])))
    fake.calls["fc-c"] = ("raise", fake.exception.FunctionTimeoutError("24h"))
    _run_once(fw, monkeypatch, mpath2)
    rec = json.loads(mpath2.read_text())["cells"][0]["successors"][0]
    assert rec["row_version_source"] == "row"


def test_unreadable_deployment_refuses_successor(fw, fake, tmp_path,
                                                 monkeypatch, capsys):
    mpath = tmp_path / "m.json"
    before = json.dumps(_doc([_cell("fc-a")]))
    mpath.write_text(before)
    fake.deployment = None
    fake.calls["fc-a"] = ("raise", fake.exception.FunctionTimeoutError("24h"))
    rc = _run_once(fw, monkeypatch, mpath)
    out = capsys.readouterr().out
    assert rc == 2 and fake.spawned == []
    assert "deployment lookup failed" in out and "could not be read" in out
    assert mpath.read_text() == before


def test_manifest_without_pin_refuses_without_hydrating(
        fw, fake, tmp_path, monkeypatch, capsys):
    mpath = tmp_path / "m.json"
    mpath.write_text(json.dumps(_doc([_cell("fc-a", function_id=None)],
                                     function_id=None)))
    fake.calls["fc-a"] = ("raise", fake.exception.FunctionTimeoutError("24h"))
    rc = _run_once(fw, monkeypatch, mpath)
    out = capsys.readouterr().out
    assert rc == 2 and fake.spawned == [] and fake.hydrated == 0
    assert "predates the version pin" in out


def test_merged_manifest_respawns_each_row_on_its_own_function(
        fw, fake, tmp_path, monkeypatch, capsys):
    """One manifest, gemma rows from one fits-spawn and
    qwen rows from a later one. The manifest's TOP LEVEL is the qwen spawn's
    (function fit_h200_qwen, qwen's config sha). A timed-out GEMMA cell must
    respawn on fit_h200_std with gemma's config sha and gemma's function_id
    pin — the ORIGINAL function — never the top-level qwen values."""
    g = _cell("fc-g", function="fit_h200_std", function_id="fu-std-1",
              config_sha="g" * 16)
    q = _cell("fc-q", model="qwen35_4b_base", root="out/qwen35_4b_base",
              function="fit_h200_qwen", function_id="fu-qwen-1",
              config_sha="q" * 16, checkpoint_every=250)
    doc = _doc([g, q], model="qwen35_4b_base", function="fit_h200_qwen",
               function_id="fu-qwen-1", config_sha="q" * 16,
               out_root="out/qwen35_4b_base")
    mpath = tmp_path / "merged.json"
    mpath.write_text(json.dumps(doc))
    fake.function_ids = {"fit_h200_std": "fu-std-1", "fit_h200_qwen": "fu-qwen-1"}
    fake.calls["fc-g"] = ("raise", fake.exception.FunctionTimeoutError("24h"))
    fake.calls["fc-q"] = ("raise", fake.exception.FunctionTimeoutError("24h"))
    rc = _run_once(fw, monkeypatch, mpath)
    out = capsys.readouterr().out
    assert rc == 0, out
    by_fn = {name: item for _app, name, item in fake.spawn_calls}
    assert set(by_fn) == {"fit_h200_std", "fit_h200_qwen"}
    assert by_fn["fit_h200_std"] == ("gemma4e4b_base", "constv2-mix-1k",
                                     "psr_ce_all", "g" * 16, False,
                                     "out/gemma4e4b_base", 100)
    assert by_fn["fit_h200_qwen"] == ("qwen35_4b_base", "constv2-mix-1k",
                                      "psr_ce_all", "q" * 16, False,
                                      "out/qwen35_4b_base", 250)
    assert set(fake.from_name_calls) == {("sdfbase-fits", "fit_h200_std"),
                                         ("sdfbase-fits", "fit_h200_qwen")}
    saved = {c["model"]: c for c in json.loads(mpath.read_text())["cells"]}
    rg = saved["gemma4e4b_base"]["successors"][0]
    assert (rg["function"], rg["function_id"]) == ("fit_h200_std", "fu-std-1")
    rq = saved["qwen35_4b_base"]["successors"][0]
    assert (rq["function"], rq["function_id"]) == ("fit_h200_qwen", "fu-qwen-1")
    assert "(on sdfbase-fits/fit_h200_std v1 abc123def456)" in out
    assert "(on sdfbase-fits/fit_h200_qwen v1 abc123def456)" in out


def test_row_without_spawn_pin_is_refused_not_respawned_from_the_top_level(
        fw, fake, tmp_path, monkeypatch, capsys):
    """A row written before the per-row pin carries no function: the
    watcher refuses it ("re-run fits-spawn") rather than falling back to the
    manifest's top-level function — which on a merged manifest belongs to
    another model."""
    legacy = {k: v for k, v in _cell("fc-a").items() if k not in fw.ROW_PIN_FIELDS}
    mpath = tmp_path / "m.json"
    before = json.dumps(_doc([legacy]))  # top level: fit_h200_std, pinned
    mpath.write_text(before)
    fake.calls["fc-a"] = ("raise", fake.exception.FunctionTimeoutError("24h"))
    rc = _run_once(fw, monkeypatch, mpath)
    out = capsys.readouterr().out
    assert rc == 2 and fake.spawned == [] and fake.hydrated == 0
    assert "NO-RESPAWN" in out and "records no spawn pin" in out
    assert "re-run fits-spawn" in out
    assert mpath.read_text() == before


def test_unspawned_cells_still_reported(fw, fake, tmp_path, monkeypatch,
                                        capsys):
    mpath = tmp_path / "m.json"
    mpath.write_text(json.dumps(_doc([_cell(None, status="pending-spawn")])))
    rc = _run_once(fw, monkeypatch, mpath)
    out = capsys.readouterr().out
    assert rc == 2 and "NEVER SPAWNED" in out and fake.spawned == []


def test_watch_flags_exist_with_defaults(fw, fake, tmp_path, monkeypatch):
    """--respawn defaults ON, --max-successors defaults 3 (a redeploy-pinned
    successor cap is the plan's per-item cap)."""
    src = FW_SRC.split("def main", 1)[1]
    assert '"--respawn"' in src and "BooleanOptionalAction" in src
    assert "default=True" in src.split('"--respawn"', 1)[1].split(
        '"--max-successors"', 1)[0]
    assert '"--max-successors"' in src
    assert "default=3" in src.split('"--max-successors"', 1)[1].split(
        "args = ap.parse_args", 1)[0]


# --------------------------------------------------------------------------- #
# the three sdfbase_fits.py hunks                                               #
# --------------------------------------------------------------------------- #


def test_fit_timeout_env_honored_at_import(fake, monkeypatch):
    monkeypatch.setenv("SDFBASE_FIT_TIMEOUT", "600")
    mod = _load(SF_PATH, "_sdfbase_fits_timeout_600", fake)
    assert mod.FIT_TIMEOUT_S == 600
    monkeypatch.delenv("SDFBASE_FIT_TIMEOUT", raising=False)
    mod = _load(SF_PATH, "_sdfbase_fits_timeout_default", fake)
    assert mod.FIT_TIMEOUT_S == 86400  # Modal's 24 h maximum, production


def test_fit_decorators_read_fit_timeout():
    # EVERY fit_h200_* image variant (std, qwen, nemotron today, and any
    # future one) — a variant left on the literal ignores the drill's
    # SDFBASE_FIT_TIMEOUT, never times out, and so never exercises the
    # successor path (the ctl-scale merge review found fit_h200_nemotron
    # shipped that way); enumerate from the source so a new variant
    # cannot bypass the pin by not being listed here
    fit_fns = re.findall(r"\ndef (fit_h200_\w+)\(", SF_SRC)
    assert set(fit_fns) >= {"fit_h200_std", "fit_h200_qwen",
                            "fit_h200_nemotron"}, fit_fns
    for fname in fit_fns:
        deco = SF_SRC.split(f"\ndef {fname}(", 1)[0].rsplit("@app.function", 1)[1]
        assert "timeout=FIT_TIMEOUT_S" in deco, fname
        assert "timeout=86400" not in deco, fname
    # hpo/battery functions are untouched (still the literal)
    for fname in ("hpo_h200_std", "hpo_h200_qwen"):
        deco = SF_SRC.split(f"\ndef {fname}(", 1)[0].rsplit("@app.function", 1)[1]
        assert "timeout=86400" in deco, fname
    assert re.search(r'FIT_TIMEOUT_S = int\(os\.environ\.get\("SDFBASE_FIT_TIMEOUT", '
                     r'86400\)\)', SF_SRC)
    assert "SDFBASE_FIT_TIMEOUT=600 modal deploy --name" in SF_SRC


def test_fits_spawn_records_the_function_id_pin():
    block = SF_SRC.split('if mode == "fits-spawn"', 1)[1].split(
        'if mode == "watch"', 1)[0]
    assert "fn.hydrate()" in block
    assert "function_id = fn.object_id" in block
    assert '"function_id": function_id' in block
    # the pin is best-effort at spawn (a stubbed modal has no handle) and
    # the watcher is what refuses — never a crash of the spawn itself
    assert "function_id = None" in block
    # the module docstring names the pin
    assert "function_id" in SF_SRC.split("\nimport json", 1)[0]
    # ... and every ROW gets its own copy
    assert "spawn_pin(fname, function_id, expect_sha" in block
    assert "pin=pin" in block


def test_watch_mode_forwards_successor_flags():
    block = SF_SRC.split('if mode == "watch"', 1)[1].split(
        'if mode == "probe-score"', 1)[0]
    assert '"--respawn" if respawn else "--no-respawn"' in block
    assert '"--max-successors", str(int(max_successors))' in block
    sig = SF_SRC.split("def main(", 1)[1].split("):", 1)[0]
    assert "respawn: bool = True" in sig
    assert "max_successors: int = 3" in sig


def fw_successor_item(row):
    """fits_watch.successor_item on a row, under the stub."""
    return _load(FW_PATH, "_fits_watch_for_spawn_row", make_fake_modal()
                 ).successor_item(row)


def test_fits_spawn_with_stubbed_modal_warns_and_records_none(
        fake, tmp_path, monkeypatch, capsys):
    """Run the real fits-spawn entrypoint under the stub: the deployed
    handle hydrates to the stub's id and the manifest records it next to
    the row's expect_fingerprint. Then a handle whose hydrate raises
    records None and prints the warning — the spawn itself still goes
    out (the pin is best-effort; the WATCHER is what refuses)."""
    mod = _load(SF_PATH, "_sdfbase_fits_spawn_pin", fake)
    cfg_local = mod.WORKTREE / Path(
        mod.MODELS["gemma4e4b_base"]["cfg"]).relative_to("/repo")
    sha = mod._config_sha16(cfg_local)
    # the in-container fingerprint derivation, answered by the stub with
    # the LOCAL config sha (a deployed app at the same commit)
    fake.remote = lambda item: {
        "model": item[0], "config_sha": sha, "missing_pools": [],
        "fingerprints": {f"{s}__{a}": FP for s, a in item[1]}}
    mpath = tmp_path / "fit_calls.json"
    mod.main(mode="fits-spawn", model="gemma4e4b_base", sis="constv2-mix-1k",
             arms="psr_ce_all", manifest=str(mpath), checkpoint_every=5)
    doc = json.loads(mpath.read_text())
    assert doc["function_id"] == "fu-deployed-1"
    assert doc["config_sha"] == sha
    assert fake.spawned[-1] == ("gemma4e4b_base", "constv2-mix-1k",
                                "psr_ce_all", sha, False,
                                "out/gemma4e4b_base", 5)
    row = doc["cells"][0]
    assert row["call_id"].startswith("fc-succ-")
    assert row["expect_fingerprint"] == FP
    assert row["status"] == "spawned (in-container skip expected)"
    # the ROW carries its own spawn pin — what the watcher respawns from
    # (under the stub the deployment history is unreachable: the pin is
    # recorded as None with a warning, and the watcher refuses successors
    # for such rows unless told the version — the default)
    assert {k: row[k] for k in mod.ROW_PIN_FIELDS} == {
        "app": "sdfbase-fits", "function": "fit_h200_std",
        "function_id": "fu-deployed-1", "config_sha": sha,
        "overwrite": False, "checkpoint_every": 5, "continue_cap": 0,
        "continue_lr_mult": 1.0, "app_id": None, "app_version": None,
        "deploy_commit": None}
    assert doc["deployment"] is None
    assert "WARNING: could not pin the deployment version" in \
        capsys.readouterr().out
    assert fw_successor_item(row) == fake.spawned[-1]
    # with the deployment readable, every row pins it
    mod._deployment_pin = lambda app: {"app_id": "ap-1", "app_version": 11,
                                       "deploy_commit": "ea7a058ac6e0"}
    mpath_dep = tmp_path / "fit_calls_dep.json"
    mod.main(mode="fits-spawn", model="gemma4e4b_base", sis="constv2-mix-1k",
             arms="psr_ce_all", manifest=str(mpath_dep), checkpoint_every=5)
    doc_dep = json.loads(mpath_dep.read_text())
    assert doc_dep["deployment"]["app_version"] == 11
    assert (doc_dep["cells"][0]["app_id"], doc_dep["cells"][0]["app_version"],
            doc_dep["cells"][0]["deploy_commit"]) == ("ap-1", 11,
                                                      "ea7a058ac6e0")
    # the plain item shape is untouched by the new pin fields
    assert fake.spawned[-1] == ("gemma4e4b_base", "constv2-mix-1k",
                                "psr_ce_all", sha, False,
                                "out/gemma4e4b_base", 5)

    class _Broken:
        """A fit handle whose hydrate fails but whose spawn works."""

        def hydrate(self):
            raise RuntimeError("no deployed function")

        def spawn(self, item):
            fake.spawned.append(item)
            return SimpleNamespace(object_id="fc-unpinned-1")

    good = fake.Function.from_name
    fake.Function = SimpleNamespace(
        from_name=lambda app, name, **k: (_Broken() if name.startswith("fit_")
                                          else good(app, name)))
    mpath2 = tmp_path / "fit_calls2.json"
    mod.main(mode="fits-spawn", model="gemma4e4b_base", sis="constv2-mix-1k",
             arms="psr_ce_all", manifest=str(mpath2))
    out = capsys.readouterr().out
    assert "WARNING: could not pin the deployed function id" in out
    doc2 = json.loads(mpath2.read_text())
    assert doc2["function_id"] is None
    assert doc2["cells"][0]["function_id"] is None  # the row says so too
    assert doc2["cells"][0]["function"] == "fit_h200_std"
    assert doc2["cells"][0]["call_id"] == "fc-unpinned-1"  # spawn went out
