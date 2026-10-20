"""``sisteer.capeval.tiers.sandbox_owned`` — the ``modal_owned`` Inspect
sandbox provider: registered under the unqualified name the
thin task passes; creates every sandbox with the OWNER tuple in its
create-time tags (all ``str``, through Modal's real protobuf constructor),
a 3 h lifetime and NO second ``set_tags`` RPC; refuses to create without an
owner; records ``created`` / observed ``terminated`` (poll) /
``terminate_unverified`` / ``cleanup_skipped_interrupted`` durably.

Needs ``inspect_sandboxes`` + ``modal`` + ``inspect_ai`` (the client lock) —
skipped where the venv lacks them (the Modal CPU suite has all three)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("inspect_sandboxes", reason="inspect-sandboxes (client lock)")
pytest.importorskip("modal", reason="modal (client lock)")

import modal  # noqa: E402
from inspect_ai.util._sandbox.registry import registry_find_sandboxenv  # noqa: E402
from inspect_sandboxes.modal._modal import ModalSandboxEnvironment  # noqa: E402
from modal_proto import api_pb2  # noqa: E402

from sisteer.capeval.tiers import sandbox_owned as SO  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402
from sisteer.capeval.tinycap import sandbox_scope as scope  # noqa: E402

OWNER = scope.Owner("abc1234-deadbeef", "none", "fc-1111")
COMPOSE = (
    "services:\n  default:\n    image: " + C.BCB_SANDBOX_IMAGE + "\n    init: true\n"
    "    command: tail -f /dev/null\n    cpus: 1.0\n    mem_limit: 0.5gb\n    network_mode: none\n")


class _FakeSandbox:
    def __init__(self, sid, *, running_polls=0):
        self.object_id, self._running_polls = sid, running_polls
        self.tags_set = None
        self.terminated = 0

    class _A:
        def __init__(self, fn):
            self.aio = fn

    @property
    def set_tags(self):
        async def _set(tags):
            self.tags_set = tags
        return self._A(_set)

    @property
    def terminate(self):
        async def _t(**kw):
            self.terminated += 1
        return self._A(_t)

    @property
    def poll(self):
        async def _p():
            if self._running_polls > 0:
                self._running_polls -= 1
                return None
            return 0
        return self._A(_p)


def _env(monkeypatch, tmp_path, *, owner=OWNER):
    for k, v in scope.owner_env(owner, tmp_path / "watch" / "sandboxes_fc-1111.jsonl").items():
        monkeypatch.setenv(k, v)
    return tmp_path / "watch" / "sandboxes_fc-1111.jsonl"


def _patch_create(monkeypatch, created):
    """Replace the parent's app lookup and sandbox creation with recorders;
    the create kwargs are pushed through Modal's REAL SandboxTag protobuf
    constructor, so a non-str tag value fails exactly where production would."""
    async def _lookup(name):
        created.append(("lookup", name))
        return SimpleNamespace(name=name, app_id="ap-test")

    async def _create(command, kwargs):
        tags = kwargs.get("tags") or {}
        [api_pb2.SandboxTag(tag_name=k, tag_value=v) for k, v in tags.items()]   # TypeError on an int
        created.append(("create", list(command), dict(kwargs)))
        return _FakeSandbox(f"sb-{len(created)}")

    monkeypatch.setattr(SO.OwnedModalSandboxEnvironment, "_lookup_app", staticmethod(_lookup))
    monkeypatch.setattr(SO.OwnedModalSandboxEnvironment, "_create_sandbox", staticmethod(_create))


def test_the_provider_is_registered_under_the_unqualified_name_and_subclasses_the_stock_one():
    cls = registry_find_sandboxenv(C.BCB_SANDBOX_PROVIDER)
    assert cls is SO.OwnedModalSandboxEnvironment and issubclass(cls, ModalSandboxEnvironment)
    assert registry_find_sandboxenv("modal") is ModalSandboxEnvironment       # the parent is untouched
    assert SO.PROVIDER_NAME == "modal_owned" and SO.SANDBOX_TIMEOUT_S == 10_800 == C.BCB_SANDBOX_TIMEOUT_S
    assert SO.SANDBOX_APP_NAME == "inspect_modal_sandbox" == C.BCB_SANDBOX_APP_NAME
    # the three lifecycle methods and (tinycap-verify-rate-limit, 2026-09-30) the CREATE are overridden; exec / files /
    # terminate / the app lookup are the parent's. The create override wraps the PARENT's create (its transient retry kept)
    for name in ("exec", "read_file", "write_file", "_terminate_sandbox", "_lookup_app"):
        assert getattr(SO.OwnedModalSandboxEnvironment, name) is getattr(ModalSandboxEnvironment, name)
    assert SO.OwnedModalSandboxEnvironment._create_sandbox is not ModalSandboxEnvironment._create_sandbox


def test_sample_init_creates_with_owner_tags_at_create_time_all_str_and_a_3h_timeout(monkeypatch, tmp_path):
    """The owner tuple rides the CREATE RPC's tags (no second set_tags, which
    REPLACES the tag set), every value str() — the BigCodeBench int sample id
    included — and the lifetime is 3 h."""
    record = _env(monkeypatch, tmp_path)
    created = []
    _patch_create(monkeypatch, created)
    compose = tmp_path / "compose.yaml"
    compose.write_text(COMPOSE)
    from inspect_sandboxes.modal import _modal as parent
    parent.sandbox_cleanup_startup()                                    # what Inspect's task_init does
    envs = asyncio.run(SO.OwnedModalSandboxEnvironment.sample_init(
        "bigcodebench_200", str(compose), {"__sample_id__": 17, "test": "import unittest"}))
    assert set(envs) == {"default"} and isinstance(envs["default"], SO.OwnedModalSandboxEnvironment)
    kinds = [c[0] for c in created]
    assert kinds == ["lookup", "create"] and created[0][1] == "inspect_modal_sandbox"
    command, kwargs = created[1][1], created[1][2]
    assert command == ["tail", "-f", "/dev/null"]
    assert kwargs["timeout"] == 10_800 and kwargs["block_network"] is True
    assert kwargs["cpu"] == 1.0 and kwargs["memory"] == 512
    assert kwargs["tags"] == {"created_by": "inspect-ai", "tinycap.run_id": "abc1234-deadbeef",
                              "tinycap.cell": "none", "tinycap.function_call_id": "fc-1111",
                              "tinycap.task": "bigcodebench_200", "tinycap.sample_id": "17"}
    assert all(isinstance(v, str) for v in kwargs["tags"].values())
    assert envs["default"].sandbox.tags_set is None                     # never a second RPC
    assert "secrets" not in kwargs and "volumes" not in kwargs
    evs = scope.read_events(record)
    assert [e["event"] for e in evs] == ["created"]
    assert evs[0]["owner"] == OWNER.as_dict() and evs[0]["sample_id"] == "17" and evs[0]["timeout_s"] == 10_800
    # the int would have died at Modal's protobuf boundary had it not been stringified
    with pytest.raises(TypeError):
        api_pb2.SandboxTag(tag_name="tinycap.sample_id", tag_value=17)


def test_sample_init_refuses_without_an_owner_before_any_rpc(monkeypatch, tmp_path):
    for k in scope.OWNER_ENV_VARS:
        monkeypatch.delenv(k, raising=False)
    created = []
    _patch_create(monkeypatch, created)
    compose = tmp_path / "compose.yaml"
    compose.write_text(COMPOSE)
    from inspect_sandboxes.modal import _modal as parent
    parent.sandbox_cleanup_startup()
    with pytest.raises(RuntimeError, match="no sandbox owner"):
        asyncio.run(SO.OwnedModalSandboxEnvironment.sample_init("bigcodebench_200", str(compose), {}))
    assert created == []                                                 # refused BEFORE the app lookup


def test_cleanup_records_terminated_only_after_poll_observes_it(monkeypatch, tmp_path):
    """`terminated` is written only when the parent's `_terminate_sandbox` saw
    poll() != None; a sandbox still running after the retries is
    `terminate_unverified`; an interrupted sample records the skip."""
    record = _env(monkeypatch, tmp_path)
    monkeypatch.setattr(SO.OwnedModalSandboxEnvironment, "_terminate_sandbox",
                        staticmethod(ModalSandboxEnvironment._terminate_sandbox.__wrapped__))
    ok = _FakeSandbox("sb-ok")
    envs = {"default": SO.OwnedModalSandboxEnvironment(ok)}
    asyncio.run(SO.OwnedModalSandboxEnvironment.sample_cleanup("bigcodebench_200", None, envs, False))
    assert ok.terminated == 1
    stuck = _FakeSandbox("sb-stuck", running_polls=99)
    asyncio.run(SO.OwnedModalSandboxEnvironment.sample_cleanup(
        "bigcodebench_200", None, {"default": SO.OwnedModalSandboxEnvironment(stuck)}, False))
    asyncio.run(SO.OwnedModalSandboxEnvironment.sample_cleanup(
        "bigcodebench_200", None, {"default": SO.OwnedModalSandboxEnvironment(_FakeSandbox("sb-int"))}, True))
    evs = scope.read_events(record)
    assert [(e["event"], e["sandbox_id"]) for e in evs] == [
        ("terminated", "sb-ok"), ("terminate_unverified", "sb-stuck"), ("cleanup_skipped_interrupted", "sb-int")]
    assert evs[0]["verified_by"] == "poll" and "still running" in evs[1]["error"]
    # the record is one JSON object per line, owner on every line
    lines = [json.loads(line) for line in record.read_text().splitlines()]
    assert all(line["owner"] == OWNER.as_dict() for line in lines)


def test_task_cleanup_terminates_the_running_list_and_records_each(monkeypatch, tmp_path):
    record = _env(monkeypatch, tmp_path)
    from inspect_sandboxes.modal import _modal as parent
    parent.sandbox_cleanup_startup()
    parent.running_sandboxes().extend(["sb-1", "sb-2"])
    boxes = {"sb-1": _FakeSandbox("sb-1"), "sb-2": _FakeSandbox("sb-2", running_polls=99)}

    async def _from_id(sid):
        return boxes[sid]
    monkeypatch.setattr(modal.Sandbox, "from_id", SimpleNamespace(aio=_from_id))
    monkeypatch.setattr(SO.OwnedModalSandboxEnvironment, "_terminate_sandbox",
                        staticmethod(ModalSandboxEnvironment._terminate_sandbox.__wrapped__))
    asyncio.run(SO.OwnedModalSandboxEnvironment.task_cleanup("bigcodebench_200", None, True))
    evs = scope.read_events(record)
    assert [(e["event"], e["sandbox_id"], e.get("by")) for e in evs] == [
        ("terminated", "sb-1", "task_cleanup"), ("terminate_unverified", "sb-2", "task_cleanup")]
    assert parent.running_sandboxes() == []
    # cleanup=False: nothing touched, nothing recorded
    parent.running_sandboxes().append("sb-3")
    asyncio.run(SO.OwnedModalSandboxEnvironment.task_cleanup("bigcodebench_200", None, False))
    assert len(scope.read_events(record)) == 2 and parent.running_sandboxes() == ["sb-3"]


# --------------------------------------------------------------------------- #
# tinycap-verify-rate-limit (2026-09-30): the create backs off on the 5/s limit  #
# --------------------------------------------------------------------------- #


def _patch_parent_create(monkeypatch, refusals: int, created: list, *, exc=None):
    """The PARENT's create (what the override wraps) refuses ``refusals`` times with Modal's REAL
    ``ResourceExhaustedError`` (the gRPC RESOURCE_EXHAUSTED mapping — its message is the live one), then returns a
    fake sandbox; the backoff's sleep records instead of waiting."""
    from inspect_sandboxes.modal import _modal as parent
    sleeps: list = []

    async def _create(command, kwargs):
        created.append((list(command), dict(kwargs)))
        if len(created) <= refusals:
            raise (exc or modal.exception.ResourceExhaustedError)("Sandbox creation rate limit (5/s) exceeded")
        return _FakeSandbox(f"sb-{len(created)}")

    async def _sleep(s):
        sleeps.append(s)

    monkeypatch.setattr(parent.ModalSandboxEnvironment, "_create_sandbox", staticmethod(_create))
    monkeypatch.setattr(SO, "_backoff_sleep", _sleep)
    return sleeps


def test_the_create_is_retried_with_backoff_on_modals_rate_limit_and_every_retry_is_a_durable_event(monkeypatch, tmp_path, caplog):
    """The live failure (gemma (c) row, tree 082f583, 2026-09-30 07:05Z): nine verifiers x eight sandboxes ~40 ms apart,
    Modal's ``ResourceExhaustedError: Sandbox creation rate limit (5/s) exceeded`` from ``Sandbox.create``, the parent's
    five-attempt retry exhausted inside ~15 s, inspect aborted. Now the override retries the PARENT's create under the
    exponential backoff (1, 2, 4 s here; jitter off through the environment), logs each retry with the attempt and the
    wait, appends a ``create_rate_limited`` event per retry to the provider record, and hands the created sandbox back
    to ``sample_init`` exactly as before (owner tags, 3 h, the ``created`` event)."""
    record = _env(monkeypatch, tmp_path)
    monkeypatch.setenv("TINYCAP_SANDBOX_CREATE_BACKOFF_JITTER_S", "0")
    created: list = []
    sleeps = _patch_parent_create(monkeypatch, 3, created)
    lookups = []

    async def _lookup(name):
        lookups.append(name)
        return SimpleNamespace(name=name, app_id="ap-test")
    monkeypatch.setattr(SO.OwnedModalSandboxEnvironment, "_lookup_app", staticmethod(_lookup))
    compose = tmp_path / "compose.yaml"
    compose.write_text(COMPOSE)
    from inspect_sandboxes.modal import _modal as parent
    parent.sandbox_cleanup_startup()
    import logging
    with caplog.at_level(logging.WARNING, logger=SO.logger.name):
        envs = asyncio.run(SO.OwnedModalSandboxEnvironment.sample_init(
            "bigcodebench_full", str(compose), {"__sample_id__": 3, "test": "import unittest"}))
    assert envs["default"].sandbox.object_id == "sb-4" and len(created) == 4        # three refusals, the fourth create landed
    assert sleeps == [1.0, 2.0, 4.0]
    assert lookups == ["inspect_modal_sandbox"]                                     # the app looked up ONCE, before the create loop
    # the same create each time: the owner tags and the 3 h lifetime ride every attempt
    assert all(kw["tags"]["tinycap.function_call_id"] == "fc-1111" and kw["timeout"] == 10_800 for _, kw in created)
    evs = scope.read_events(record)
    assert [e["event"] for e in evs] == ["create_rate_limited"] * 3 + ["created"]
    assert [(e["attempt"], e["wait_s"]) for e in evs[:3]] == [(0, 1.0), (1, 2.0), (2, 4.0)]
    assert all(e["sandbox_id"] == "" and e["owner"] == OWNER.as_dict() and "5/s" in e["error"] for e in evs[:3])
    assert evs[3]["sandbox_id"] == "sb-4"
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 3 and "retry 1 in 1.0 s" in warnings[0] and "retry 3 in 4.0 s" in warnings[2]
    assert all("rate limit (5/s)" in w and "ResourceExhaustedError" in w for w in warnings)


def test_a_create_still_refused_at_the_bound_reraises_modals_error_itself_and_nothing_falls_back(monkeypatch, tmp_path, caplog):
    """A truly stuck limit fails LOUDLY: after the bound (5 s here through the environment; the recording sleep does not
    advance the real clock, so the waits 1, 2 and 4 s each pass the ``elapsed + wait <= total`` check and the 8 s wait is
    the one that would exceed it) the last ``ResourceExhaustedError`` is re-raised unchanged — Modal's own message, the
    same type inspect classifies today — with an ``error`` log naming the bound; no sandbox, no ``created`` event, no
    other provider."""
    record = _env(monkeypatch, tmp_path)
    monkeypatch.setenv("TINYCAP_SANDBOX_CREATE_BACKOFF_JITTER_S", "0")
    monkeypatch.setenv("TINYCAP_SANDBOX_CREATE_BACKOFF_TOTAL_S", "5")
    created: list = []
    sleeps = _patch_parent_create(monkeypatch, 10 ** 6, created)
    import logging
    with caplog.at_level(logging.WARNING, logger=SO.logger.name):
        with pytest.raises(modal.exception.ResourceExhaustedError, match=r"Sandbox creation rate limit \(5/s\) exceeded"):
            asyncio.run(SO.OwnedModalSandboxEnvironment._create_sandbox(["tail"], {"name": "n", "tags": {}}))
    assert sleeps == [1.0, 2.0, 4.0] and len(created) == 4                       # the 8 s wait would pass the bound: give up
    evs = scope.read_events(record)
    assert [e["event"] for e in evs] == ["create_rate_limited"] * 3                # the three retries; no `created`
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and "after 3 retries" in errors[0] and "bound 5 s" in errors[0] and "giving up" in errors[0]
    # a refusal that is NOT the rate limit is the parent's contract, untouched: re-raised at once, no sleep, no event
    created.clear(); sleeps.clear()
    _patch_parent_create(monkeypatch, 1, created, exc=modal.exception.RemoteError)
    with pytest.raises(modal.exception.RemoteError):
        asyncio.run(SO.OwnedModalSandboxEnvironment._create_sandbox(["tail"], {"name": "n"}))
    assert len(created) == 1 and scope.read_events(record)[3:] == []


def test_the_rate_limit_predicate_reads_modals_type_and_the_raw_grpc_status():
    import grpclib
    assert SO.is_sandbox_rate_limit(modal.exception.ResourceExhaustedError("x"))
    assert SO.is_sandbox_rate_limit(grpclib.GRPCError(grpclib.const.Status.RESOURCE_EXHAUSTED, "x"))
    assert not SO.is_sandbox_rate_limit(grpclib.GRPCError(grpclib.const.Status.UNAVAILABLE, "x"))
    assert not SO.is_sandbox_rate_limit(modal.exception.RemoteError("x"))
    assert not SO.is_sandbox_rate_limit(RuntimeError("Sandbox creation rate limit (5/s) exceeded"))
    # the parent's own retry excludes only RemoteError — so it DID retry the rate limit, five attempts inside ~15 s,
    # which is what the live failure exhausted; the override is the longer bound on top of it
    assert issubclass(modal.exception.ResourceExhaustedError, modal.exception.Error)
    assert not issubclass(modal.exception.ResourceExhaustedError, modal.exception.RemoteError)
