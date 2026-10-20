"""scripts/modal/seedrep_served.py — the ATTEMPT root layout and the acceptance state machine, on a two-snapshot Modal
simulation: a fake Volume where every container has its own snapshot and commits are last-writer-wins, and a fake Dict
offering ONLY `put(skip_if_exists)` / `get` / `pop`.

Cases: two successive re-executions after partial writes -> two IMMUTABLE roots, the first untouched, neither resumed;
the completing attempt takes the claim; a later complete attempt is `superseded_by` and publishes nothing; the pull
REFUSES a shard with no accepted attempt; fault injection at every boundary — after COMPLETE, after the inventory, after
the acceptance (before CLOSED), before / after CLOSED — each ends with exactly ONE accepted attempt and a closable queue;
(rev 4) the accepted producer dies after acceptance, a superseded re-execution publishes its own queue files afterwards,
the reconciler still synthesises CLOSED from the ACCEPTED inventory (no two attempts share a queue path).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal"):
    sys.path.insert(0, str(p))
DRIVER = REPO / "scripts" / "modal" / "seedrep_served.py"


class _Chain:
    def __getattr__(self, _name):
        return lambda *a, **k: self

    def __call__(self, *a, **k):
        return self


class _App:
    def __init__(self, name=None, **k):
        self.description = None

    def function(self, **kw):
        return lambda f: f

    def local_entrypoint(self, **kw):
        return lambda f: f


class FakeDict:
    """modal.Dict's ONE atomic primitive: put(key, value, skip_if_exists=True) -> bool; get; pop."""

    store: dict = {}

    def __init__(self, name):
        self.name = name
        FakeDict.store.setdefault(name, {})

    @classmethod
    def from_name(cls, name, create_if_missing=False):
        return cls(name)

    def put(self, key, value, skip_if_exists=False):
        d = FakeDict.store[self.name]
        if skip_if_exists and key in d:
            return False
        d[key] = value
        return True

    def get(self, key, default=None):
        return FakeDict.store[self.name].get(key, default)

    def pop(self, key):
        return FakeDict.store[self.name].pop(key)

    def __setitem__(self, key, value):
        FakeDict.store[self.name][key] = value


def _stub_modal():
    m = types.ModuleType("modal")
    m.App = _App
    m.Volume = types.SimpleNamespace(from_name=lambda *a, **k: types.SimpleNamespace(commit=lambda: None, reload=lambda: None))
    m.Image = types.SimpleNamespace(from_registry=lambda *a, **k: _Chain(), debian_slim=lambda *a, **k: _Chain())
    m.Dict = FakeDict
    m.is_local = lambda: False              # no cpu_suite import, no dependency hashing of the local tree
    m.current_function_call_id = lambda: "fc-test"
    m.forward = None
    m.FunctionCall = None
    return m


_ABSENT = object()


def load_app():
    spec = importlib.util.spec_from_file_location("seedrep_served_under_test", DRIVER)
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


@pytest.fixture(scope="module")
def srv():
    return load_app()


class FakeVolume:
    """Per-container snapshots + last-writer-wins commits: a container writes its local dir; `commit(container)` copies that
    dir's files into the committed view (a later commit of the same path wins); `has(path)` reads the committed view."""

    def __init__(self, root: Path):
        self.root = root
        self.committed = root / "committed"
        self.committed.mkdir()
        self.locals: dict = {}

    def local(self, container: str) -> Path:
        p = self.root / "local" / container
        p.mkdir(parents=True, exist_ok=True)
        self.locals[container] = p
        return p

    def commit(self, container: str) -> None:
        src = self.locals[container]
        for f in src.rglob("*"):
            if f.is_file():
                dst = self.committed / f.relative_to(src)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, dst)

    def has(self, rel: str):
        p = self.committed / rel
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


ROW = {"standup_id": "pasttense__s1__base__shard00", "leg": "pasttense", "engine": "s1", "stage": "base", "shard": 0,
       "control_key": "1111111/wave/pasttense/s1/base/0", "producers": [{"queue_stage": "8192", "kind": "base"}]}


def _attempt(srv, vol: FakeVolume, container: str, attempt_id: str, *, enqueue: int = 2):
    """One container's attempt root under its snapshot: ATTEMPT.json first, then a store row + its queue files."""
    local = vol.local(container)
    root = local / srv.attempt_root_for("results", ROW, attempt_id).lstrip("/")
    root.mkdir(parents=True)
    srv.write_json(root / srv.ATTEMPT_JSON, srv.attempt_record(attempt_id=attempt_id, call_id=f"fc-{container}", task_id=f"ta-{container}",
                                                                W="1" * 40, wave_uuid="wave", row=ROW, prior_attempts=[]))
    (root / "exp").mkdir()
    (root / "exp" / "responses.jsonl").write_text(f'{{"attempt":"{attempt_id}"}}\n')
    inventory = []
    for i in range(enqueue):
        q = local / "ladder" / "pasttense" / "s1" / "8192" / "queue" / f"none__s1" / f"item{i}" / f"{attempt_id}.json"
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text(json.dumps({"item_key": f"item{i}", "producer": {"attempt_id": attempt_id}}))
        inventory.append({"item_key": f"item{i}", "attempt_id": attempt_id, "path": str(q.relative_to(local)),
                          "sha256": hashlib.sha256(q.read_bytes()).hexdigest()})
    vol.commit(container)
    return root, inventory


def _closed_marker(local: Path, attempt_id: str) -> Path:
    return local / "ladder" / "pasttense" / "s1" / "8192" / "producers" / f"shard00__{attempt_id}.CLOSED"


def test_attempt_ids_and_roots_are_distinct_and_immutable(srv):
    a = srv.mint_attempt_id("ta-abc", "0123456789abcdef")
    b = srv.mint_attempt_id("ta-abc", "fedcba9876543210")
    assert a == "abc-0123456789ab" and b == "abc-fedcba987654" and a != b
    assert srv.mint_attempt_id(None, "0" * 32).startswith("local-")
    assert srv.attempt_root_for("/vol/seedrep/1111111/exp/seedrep_battery_modal", ROW, a) == \
        "/vol/seedrep/1111111/exp/seedrep_battery_modal/pasttense/s1/shard00/attempts/abc-0123456789ab"
    rung = {**ROW, "stage": "8192"}
    assert srv.attempt_root_for("R", rung, a) == "R/pasttense/s1/rungs/8192/shard00/attempts/abc-0123456789ab"
    assert srv.results_root("1111111") == "/vol/seedrep/1111111/exp/seedrep_battery_modal"
    assert {"ATTEMPT.json", "COMPLETE.json", "queue_inventory.json", "ladder_enqueued.json", "RESULT", "modal_evidence.json", "usage_ledger.jsonl",
            "request_evidence.jsonl", "gpu_dmon.log", "vllm_metrics.jsonl", "driver_inflight.jsonl", "determinism.json",
            "exp/serve/endpoint.json", "exp/serve/vllm_server.log.gz", "exp/", "registry/records/"} <= set(srv.ATTEMPT_ROOT_LAYOUT)
    assert srv.SERVE_DIR == "exp/serve", "the serve dir lives under the attempt's exp tree (WS2-F attempt_paths maps onto <root>/exp/)"


def test_two_re_executions_two_roots_first_partial_untouched(srv, tmp_path):
    FakeDict.store.clear()
    vol = FakeVolume(tmp_path)
    acc = FakeDict("accepted")
    # container A dies after partial writes (no COMPLETE)
    root_a, _ = _attempt(srv, vol, "A", "a-000000000001", enqueue=1)
    a_bytes = (root_a / "exp" / "responses.jsonl").read_bytes()
    assert srv.attempt_state(root_a) == "partial"
    # container B (the re-execution of the SAME call) completes and takes the claim
    root_b, inv_b = _attempt(srv, vol, "B", "b-000000000002")
    fin = srv.finish_attempt(root=root_b, key=ROW["control_key"], attempt_id="b-000000000002", inventory=inv_b, accepted=acc,
                             commit=lambda: vol.commit("B"), closed_marker=_closed_marker(vol.locals["B"], "b-000000000002"), summary={"children": 8})
    assert fin["state"] == "closed" and fin["accepted_attempt"] == "b-000000000002"
    assert acc.get(ROW["control_key"]) == "b-000000000002"
    assert root_a != root_b and (root_a / "exp" / "responses.jsonl").read_bytes() == a_bytes and not (root_a / srv.COMPLETE_JSON).exists()
    assert json.loads((root_b / srv.ATTEMPT_JSON).read_text())["resumed_from"] is None
    assert vol.has("results/pasttense/s1/shard00/attempts/b-000000000002/COMPLETE.json")
    closed = json.loads(_closed_marker(vol.locals["B"], "b-000000000002").read_text())
    assert closed["n_enqueued"] == 2 and closed["attempt_id"] == "b-000000000002"
    # a LATER complete attempt (a duplicate spawn) is superseded and publishes nothing
    root_c, inv_c = _attempt(srv, vol, "C", "c-000000000003")
    fin_c = srv.finish_attempt(root=root_c, key=ROW["control_key"], attempt_id="c-000000000003", inventory=inv_c, accepted=acc,
                               commit=lambda: vol.commit("C"), closed_marker=_closed_marker(vol.locals["C"], "c-000000000003"), summary={})
    assert fin_c["state"] == "superseded" and fin_c["accepted_attempt"] == "b-000000000002"
    assert json.loads((root_c / srv.ATTEMPT_JSON).read_text())["superseded_by"] == "b-000000000002"
    assert not _closed_marker(vol.locals["C"], "c-000000000003").exists()
    assert srv.attempt_state(root_c) == "superseded"
    assert acc.get(ROW["control_key"]) == "b-000000000002"
    # exactly one accepted attempt; the pull reads it and refuses a key nobody holds
    assert srv.accepted_attempt_for(acc, ROW["control_key"]) == "b-000000000002"
    with pytest.raises(srv.NoAcceptedAttempt):
        srv.accepted_attempt_for(acc, "1111111/wave/pasttense/s2/base/0")


@pytest.mark.parametrize("crash_after", ["complete", "inventory", "acceptance"])
def test_fault_injection_ends_with_one_accepted_attempt_and_a_closable_queue(srv, tmp_path, crash_after):
    FakeDict.store.clear()
    vol = FakeVolume(tmp_path)
    acc = FakeDict("accepted")
    root_a, inv_a = _attempt(srv, vol, "A", "a-000000000001")
    key = ROW["control_key"]

    class Crash(Exception):
        pass

    steps = {"n": 0}
    boundary = {"complete": 1, "inventory": 2, "acceptance": 3}[crash_after]

    def commit():
        steps["n"] += 1
        vol.commit("A")
        if steps["n"] == boundary:
            raise Crash(crash_after)
    with pytest.raises(Crash):
        srv.finish_attempt(root=root_a, key=key, attempt_id="a-000000000001", inventory=inv_a, accepted=acc, commit=commit,
                           closed_marker=_closed_marker(vol.locals["A"], "a-000000000001"), summary={})
    if crash_after in ("complete", "inventory"):
        # no acceptance happened: the re-executed attempt completes and takes it
        assert acc.get(key) is None
        assert srv.attempt_state(root_a) == {"complete": "complete", "inventory": "inventoried"}[crash_after]
        root_b, inv_b = _attempt(srv, vol, "B", "b-000000000002")
        fin = srv.finish_attempt(root=root_b, key=key, attempt_id="b-000000000002", inventory=inv_b, accepted=acc, commit=lambda: vol.commit("B"),
                                 closed_marker=_closed_marker(vol.locals["B"], "b-000000000002"), summary={})
        assert fin["state"] == "closed" and acc.get(key) == "b-000000000002"
        ok, why = srv.reconciler_closable(vol.committed / "results/pasttense/s1/shard00/attempts/a-000000000001",
                                          accepted_attempt=acc.get(key), volume_has=vol.has)
        assert ok is False and "not the accepted attempt" in why
    else:
        # accepted, then died before its CLOSED marker: the reconciler may synthesise CLOSED from the ACCEPTED inventory
        assert acc.get(key) == "a-000000000001"
        assert not _closed_marker(vol.locals["A"], "a-000000000001").exists()
        committed_root = vol.committed / "results/pasttense/s1/shard00/attempts/a-000000000001"
        assert json.loads((committed_root / srv.ATTEMPT_JSON).read_text())["accepted"] is True
        ok, why = srv.reconciler_closable(committed_root, accepted_attempt="a-000000000001", volume_has=vol.has)
        assert ok is True, why
        # (rev 4) a superseded re-execution publishes its OWN queue files afterwards — the accepted inventory is undisturbed
        root_b, inv_b = _attempt(srv, vol, "B", "b-000000000002", enqueue=3)
        fin_b = srv.finish_attempt(root=root_b, key=key, attempt_id="b-000000000002", inventory=inv_b, accepted=acc, commit=lambda: vol.commit("B"),
                                   closed_marker=_closed_marker(vol.locals["B"], "b-000000000002"), summary={})
        assert fin_b["state"] == "superseded" and acc.get(key) == "a-000000000001"
        paths_a = {e["path"] for e in inv_a}
        paths_b = {e["path"] for e in inv_b}
        assert paths_a.isdisjoint(paths_b), "no two attempts share a queue path"
        ok, why = srv.reconciler_closable(committed_root, accepted_attempt="a-000000000001", volume_has=vol.has)
        assert ok is True, why
        # the rung reads ONLY the accepted projection: the accepted attempt's files, by its attempt id
        accepted_files = sorted(p.name for p in (vol.committed / "ladder/pasttense/s1/8192/queue/none__s1").rglob("*.json")
                                if p.name.startswith("a-"))
        assert accepted_files == ["a-000000000001.json", "a-000000000001.json"]
    assert sum(1 for v in FakeDict.store["accepted"].values()) == 1


def test_reconciler_refuses_a_tampered_or_missing_inventory_file(srv, tmp_path):
    FakeDict.store.clear()
    vol = FakeVolume(tmp_path)
    acc = FakeDict("accepted")
    root_a, inv_a = _attempt(srv, vol, "A", "a-000000000001")

    def commit_until_accept():
        vol.commit("A")
    srv.finish_attempt(root=root_a, key=ROW["control_key"], attempt_id="a-000000000001", inventory=inv_a, accepted=acc,
                       commit=commit_until_accept, closed_marker=None, summary={})
    committed_root = vol.committed / "results/pasttense/s1/shard00/attempts/a-000000000001"
    ok, _ = srv.reconciler_closable(committed_root, accepted_attempt="a-000000000001", volume_has=vol.has)
    assert ok is True
    # a queue file whose bytes moved -> sha mismatch -> not closable
    tampered = vol.committed / inv_a[0]["path"]
    tampered.write_text("{}")
    ok, why = srv.reconciler_closable(committed_root, accepted_attempt="a-000000000001", volume_has=vol.has)
    assert ok is False and "sha" in why
    tampered.unlink()
    ok, why = srv.reconciler_closable(committed_root, accepted_attempt="a-000000000001", volume_has=vol.has)
    assert ok is False and "sha None" in why


def test_pull_manifest_rows_carry_the_attempt_id(srv):
    rows = srv.pull_manifest_rows("a-1", [("serve/endpoint.json", "ab" * 32), ("exp/x.jsonl", "cd" * 32)])
    assert rows == [{"attempt_id": "a-1", "path": "serve/endpoint.json", "sha256": "ab" * 32},
                    {"attempt_id": "a-1", "path": "exp/x.jsonl", "sha256": "cd" * 32}]
