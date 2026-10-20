"""``sisteer.capeval.tinycap.verifier_transport`` against a FAKE provider class
carrying the pinned ``inspect-sandboxes 0.5.0`` ``exec`` / ``write_file``
shapes — no Modal, no Inspect, no network.

Covers: the byte accounting (kernel-style, never below Modal's count); under the
threshold the ORIGINAL ``exec`` sees the ORIGINAL argv object with no write and
no ledger line; over it the program rides ``write_file`` at the documented path
and ``exec`` runs ``[python, <path>, *rest]`` with the same keyword arguments,
one ledger line (bytes, sha, path) and one log line; the threshold is strict; a
command that is not ``python -c`` is never touched; a second install wraps once
and the signature Inspect's proxy reads (8 parameters) survives; installing on
the owned subclass leaves the parent untouched; the REC field reads the ledger
count; ``ledger_path_for`` lands beside the usage ledger for a real
``CellStore``; and the driver's install site sits under the sandboxed-plan
guard before the task's first ``eval_async``."""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
from pathlib import Path

import pytest

from sisteer.capeval import store as S
from sisteer.capeval.tinycap import verifier_transport as VT

REPO = Path(__file__).resolve().parents[1]
#: ~70 KB — an at-cap 16,384-token completion's size (ladder_report §4b: 74,806 / 79,063 B)
BIG = "import unittest\nx = 1\n" + ("# " + "p" * 98 + "\n") * 700
SMALL = "import unittest\nprint(1)\n"


def _fake_cls():
    class FakeSandbox:
        """The pinned provider's ``exec`` / ``write_file`` shapes, recording every call."""

        def __init__(self):
            self.execs, self.writes = [], []

        async def exec(self, cmd, input=None, cwd=None, env=None, user=None, timeout=None,
                       timeout_retry=True, concurrency=True):
            self.execs.append((cmd, dict(input=input, cwd=cwd, env=env, user=user, timeout=timeout,
                                         timeout_retry=timeout_retry, concurrency=concurrency)))
            return ("ran", list(cmd))

        async def write_file(self, file, contents):
            self.writes.append((file, contents))

    return FakeSandbox


def _proxy_call(sb, cmd, **over):
    """The call Inspect's ``SandboxEnvironmentProxy`` makes: every parameter by keyword."""
    kw = dict(cmd=cmd, input=None, cwd=None, env={}, user=None, timeout=30, timeout_retry=True, concurrency=True)
    kw.update(over)
    return asyncio.run(sb.exec(**kw))


def test_argv_bytes_is_the_kernels_accounting_and_never_below_modals():
    assert VT.argv_bytes(["python", "-c", "é"]) == (6 + 1) + (2 + 1) + (2 + 1)      # UTF-8 bytes + one NUL each
    cmd = ["python", "-c", BIG]
    assert VT.argv_bytes(cmd) >= sum(len(a) for a in cmd)                          # Modal sums code points
    assert VT.argv_bytes(cmd) > VT.MODAL_ARG_MAX_BYTES > VT.SPILL_THRESHOLD_BYTES == 60_000
    assert VT.is_python_c(["python", "-c", "x"]) and VT.is_python_c(["python3", "-c", "x", "--a"])
    assert not VT.is_python_c(["python", "-c"]) and not VT.is_python_c(["bash", "-c", "x"])
    assert not VT.is_python_c(["python", "-m", "x"]) and not VT.is_python_c(["python", "x.py"])
    sha = hashlib.sha256(BIG.encode()).hexdigest()
    assert VT.program_sha256(BIG) == sha and VT.spill_path(BIG) == f"/tmp/sisteer_verify_{sha[:16]}.py"


def test_under_the_threshold_the_original_exec_sees_the_original_argv(tmp_path):
    cls = _fake_cls()
    out = tmp_path / "out"
    ledger = out / VT.VERIFIER_SPILLS_FILENAME
    assert VT.install_exec_spill(cls, ledger_path=ledger) is True
    sb = cls()
    cmd = ["python", "-c", SMALL]
    assert _proxy_call(sb, cmd, timeout=30) == ("ran", cmd)
    assert len(sb.execs) == 1 and sb.execs[0][0] is cmd                  # the SAME list object, not a rewrite
    assert sb.execs[0][1] == dict(input=None, cwd=None, env={}, user=None, timeout=30, timeout_retry=True, concurrency=True)
    assert sb.writes == [] and not ledger.exists() and VT.n_spilled(ledger) == 0
    assert VT.rec_field(out) == {"spill_threshold_bytes": 60_000, "n_spilled": 0, "ledger": str(ledger)}


def test_over_the_threshold_the_program_rides_the_exec_channel_on_stdin_and_exec_runs_the_file(tmp_path, caplog):
    """capeval-full-tasks F-18 (2026-09-24): the spilled program is written by `sh -c 'cat > <path>'` with the program on STDIN through
    the ORIGINAL exec — never the provider's write_file (Modal's dead legacy filesystem API took a full BCB leg down on one >60 KB
    completion); then the original exec runs the file with every keyword argument as given."""
    cls = _fake_cls()
    out = tmp_path / "out"
    ledger = out / VT.VERIFIER_SPILLS_FILENAME
    VT.install_exec_spill(cls, ledger_path=ledger)
    sb = cls()
    cmd = ["python", "-c", BIG]
    with caplog.at_level(logging.WARNING, logger=VT.__name__):
        res = _proxy_call(sb, cmd, timeout=30, env={"A": "1"})
    sha = hashlib.sha256(BIG.encode()).hexdigest()
    path = f"/tmp/sisteer_verify_{sha[:16]}.py"
    assert sb.writes == []                                                # NEVER the provider's write_file (the dead legacy FS API)
    assert len(sb.execs) == 2 and sb.execs[0][0] == ["sh", "-c", f"cat > {path}"] and sb.execs[1][0] == ["python", path]
    assert sb.execs[0][1]["input"] == BIG and sb.execs[0][1]["timeout"] == VT.SPILL_WRITE_TIMEOUT_S == 60   # the whole program on stdin
    assert sb.execs[1][1] == dict(input=None, cwd=None, env={"A": "1"}, user=None, timeout=30, timeout_retry=True, concurrency=True)
    assert res == ("ran", ["python", path])
    lines = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(lines) == 1 and set(lines[0]) == set(VT.LEDGER_FIELDS) == {"ts", "argv_bytes", "program_sha256", "path"}
    assert lines[0]["argv_bytes"] == VT.argv_bytes(cmd) > 65_536 and lines[0]["program_sha256"] == sha and lines[0]["path"] == path
    assert VT.n_spilled(ledger) == 1 and VT.rec_field(out) == {"spill_threshold_bytes": 60_000, "n_spilled": 1, "ledger": str(ledger)}
    assert any(f"{VT.LOG_PREFIX} spilled {VT.argv_bytes(cmd)} bytes -> {path}" == r.getMessage() for r in caplog.records)
    # argv entries after the program survive; a positional call shape is honoured too
    cmd2 = ["python3", "-c", BIG + "\n", "--flag"]
    asyncio.run(sb.exec(cmd2, None, "/w", None, None, 30))
    assert sb.execs[3][0] == ["python3", VT.spill_path(BIG + "\n"), "--flag"] and sb.execs[3][1]["cwd"] == "/w"
    assert sb.execs[2][0] == ["sh", "-c", f"cat > {VT.spill_path(BIG + chr(10))}"] and sb.execs[2][1]["input"] == BIG + "\n"
    assert sb.writes == [] and VT.n_spilled(ledger) == 2


def test_the_threshold_is_strict(tmp_path):
    cls = _fake_cls()
    VT.install_exec_spill(cls, threshold_bytes=100, ledger_path=tmp_path / "l.jsonl")
    sb = cls()
    at = ["python", "-c", "a" * (100 - VT.argv_bytes(["python", "-c", ""]))]
    assert VT.argv_bytes(at) == 100
    _proxy_call(sb, at)
    assert sb.writes == [] and sb.execs[-1][0] is at                        # AT the threshold: untouched
    over = ["python", "-c", at[2] + "a"]
    _proxy_call(sb, over)
    assert sb.writes == [] and sb.execs[-2][0] == ["sh", "-c", f"cat > {VT.spill_path(over[2])}"] and sb.execs[-1][0] == ["python", VT.spill_path(over[2])]


def test_a_command_that_is_not_python_dash_c_is_never_touched(tmp_path):
    cls = _fake_cls()
    ledger = tmp_path / "l.jsonl"
    VT.install_exec_spill(cls, ledger_path=ledger)
    sb = cls()
    for cmd in (["bash", "-c", BIG], ["python", "-m", BIG], ["python", BIG], ["python", "-c"], ["sh", "-c", "python -c " + BIG]):
        _proxy_call(sb, cmd)
        assert sb.execs[-1][0] is cmd
    assert sb.writes == [] and not ledger.exists()


def test_install_is_idempotent_and_keeps_the_signature_inspects_proxy_reads(tmp_path):
    cls = _fake_cls()
    before = cls.exec
    params = ["cmd", "input", "cwd", "env", "user", "timeout", "timeout_retry", "concurrency"]
    assert list(inspect.signature(cls().exec).parameters) == params      # inspect_ai's proxy passes `concurrency` iff 8
    assert VT.install_exec_spill(cls, ledger_path=tmp_path / "l.jsonl") is True
    wrapped = cls.exec
    assert wrapped is not before and getattr(wrapped, VT.INSTALLED_MARKER) is True and wrapped.__wrapped__ is before
    assert VT.install_exec_spill(cls, ledger_path=tmp_path / "other.jsonl") is False and cls.exec is wrapped
    assert list(inspect.signature(cls().exec).parameters) == params
    # a subclass INHERITING the wrapped exec is already installed
    class Inherits(cls):
        pass
    assert VT.install_exec_spill(Inherits, ledger_path=tmp_path / "l.jsonl") is False and Inherits.exec is wrapped
    sb = Inherits()
    _proxy_call(sb, ["python", "-c", BIG])
    assert sb.writes == [] and len(sb.execs) == 2                       # wrapped once: one stdin write exec, one run exec


def test_installing_on_the_owned_subclass_leaves_the_parent_untouched(tmp_path):
    """The production shape: the wrapper goes on the in-repo ``modal_owned`` subclass,
    which inherits ``exec`` / ``write_file`` from inspect-sandboxes' class."""
    parent = _fake_cls()

    class Owned(parent):
        pass
    assert VT.install_exec_spill(Owned, ledger_path=tmp_path / "l.jsonl") is True
    assert Owned.exec is not parent.exec and not getattr(parent.exec, VT.INSTALLED_MARKER, False)
    assert Owned.exec.__wrapped__ is parent.exec
    o, p = Owned(), parent()
    _proxy_call(o, ["python", "-c", BIG])
    _proxy_call(p, ["python", "-c", BIG])
    assert o.writes == [] and o.execs[0][0][:2] == ["sh", "-c"] and o.execs[1][0][1].startswith("/tmp/sisteer_verify_")
    assert p.writes == [] and p.execs[0][0] == ["python", "-c", BIG]


def test_rec_field_reads_the_ledger_count_and_ledger_path_for_lands_beside_the_usage_ledger(tmp_path):
    out = tmp_path / "out"
    ledger = out / VT.VERIFIER_SPILLS_FILENAME
    assert VT.rec_field(out) == {"spill_threshold_bytes": VT.SPILL_THRESHOLD_BYTES, "n_spilled": 0, "ledger": str(ledger)}
    ledger.parent.mkdir()
    rows = [{"ts": "2026-09-12T09:00:00+00:00", "argv_bytes": 70_000 + i, "program_sha256": "a" * 64, "path": "/tmp/x.py"} for i in range(3)]
    ledger.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert VT.rec_field(out)["n_spilled"] == 3 and VT.read_spills(ledger) == rows and VT.read_spills(tmp_path / "none") == []
    # the driver's CellStore on a SOLO run: sweep_dir IS the out root (the usage ledger's directory), whatever the cell key
    st = S.CellStore(out, "controls__aaaa__bbbb")
    assert VT.ledger_path_for(st) == ledger and st.cell_dir == out / "cells" / "controls__aaaa__bbbb"
    # a cell_dir-only double (the runner's tests) ledgers beside its rows

    class Double:
        cell_dir = tmp_path / "d"
    assert VT.ledger_path_for(Double()) == tmp_path / "d" / VT.VERIFIER_SPILLS_FILENAME


def test_on_a_row_engine_the_spill_ledger_lives_in_the_row_dir_keyed_by_cell_and_store(tmp_path):
    """A spill into the STORE made the qwen s1 `constv2-mix-full__psr_ce_all` store count 41 inodes, not exactly 40.
    When the store's manifest carries the runner's `row_engine` block (naming `row_dir`), the ledger is
    `<row dir>/spills/<cell>__<store basename>__verifier_spills.jsonl` — decided HERE, where the path is decided, so
    the driver's install line and the launcher's `rec_field(out_dir)` read are unchanged; a manifest without the
    block (a solo run) keeps today's in-store path."""
    row_dir = tmp_path / "rows" / "gemma4e4b-f22s1" / "bcbw-gemma4e4b_s1-abcdef0-shared"
    out = tmp_path / "gen" / "native" / "gemma4e4b-f22s1" / "constv2-mix-full__psr_ce_all" / "thinking_off" / "bcbw-gemma4e4b_s1-abcdef0-shared"
    key = "constv2-mix-full--psr_ce_all__controls__t0__a0__2ff9cdff__a03d0582"
    cell_dir = out / "cells" / key
    cell_dir.mkdir(parents=True)
    block = {"row_engine": True, "standup_id": "1" * 32, "engine_cells": ["none", "constv2-mix-full__psr_ce_all"], "max_loras": 2,
             "residual_slot_armed": True, "compile_cache": {"hit": False, "key": "k", "vllm_cache_dir": None}, "row_dir": str(row_dir),
             "verification_pass": None}
    (cell_dir / "manifest.json").write_text(json.dumps({"cell_id": "constv2-mix-full__psr_ce_all", "row_engine": block}))
    want = row_dir / "spills" / f"constv2-mix-full__psr_ce_all__{out.name}__verifier_spills.jsonl"
    assert VT.row_spill_ledger_path(row_dir, "constv2-mix-full__psr_ce_all", out.name) == want and VT.ROW_SPILLS_SUBDIR == "spills"
    assert VT.row_engine_facts_of_manifest(cell_dir / "manifest.json") == (str(row_dir), "constv2-mix-full__psr_ce_all")
    st = S.CellStore(out, key)
    assert VT.ledger_path_for(st) == want and VT.ledger_path_for_root(out) == want
    # a spill through the installed transport lands there; the store root gains no file; the REC names the path
    cls = _fake_cls()
    VT.install_exec_spill(cls, ledger_path=VT.ledger_path_for(st))
    _proxy_call(cls(), ["python", "-c", BIG], timeout=30)
    assert want.is_file() and VT.n_spilled(want) == 1 and not (out / VT.VERIFIER_SPILLS_FILENAME).exists()
    assert VT.rec_field(out) == {"spill_threshold_bytes": VT.SPILL_THRESHOLD_BYTES, "n_spilled": 1, "ledger": str(want)}
    assert sorted(p.name for p in out.iterdir()) == ["cells"]                        # the store's inodes: untouched
    # the in-run repeat's store: its own ledger beside the base's (store-keyed)
    rep = out.parent / f"{out.name}-rep2"
    assert VT.row_spill_ledger_path(row_dir, "none", rep.name) == row_dir / "spills" / f"none__{rep.name}__verifier_spills.jsonl"
    with pytest.raises(ValueError):
        VT.row_spill_ledger_path(row_dir, "none", "a/b")
    # a solo store (no row_engine block), a torn manifest, or no manifest: today's in-store path
    solo = tmp_path / "solo"
    (solo / "cells" / key).mkdir(parents=True)
    (solo / "cells" / key / "manifest.json").write_text(json.dumps({"cell_id": "none"}))
    assert VT.ledger_path_for(S.CellStore(solo, key)) == solo / VT.VERIFIER_SPILLS_FILENAME == VT.ledger_path_for_root(solo)
    (solo / "cells" / key / "manifest.json").write_text("{torn")
    assert VT.ledger_path_for_root(solo) == solo / VT.VERIFIER_SPILLS_FILENAME and VT.row_engine_facts_of_manifest(solo / "x.json") is None
    assert VT.ledger_path_for_root(tmp_path / "absent") == tmp_path / "absent" / VT.VERIFIER_SPILLS_FILENAME
    # the CPU verifier (cpu_verify.verify_store) installs on the same row-dir path, by name (source pin)
    cv = (REPO / "src/sisteer/capeval/tinycap/cpu_verify.py").read_text(encoding="utf-8")
    assert "_VT.install_exec_spill(_SO.OwnedModalSandboxEnvironment, ledger_path=_VT.row_spill_ledger_path(row_dir, cell, store_key))" in cv


def test_the_driver_installs_on_the_owned_provider_under_the_sandboxed_guard_before_the_first_eval():
    """The install site (source pin, the launcher-test convention): inside ``run_one_task``,
    under ``if plan.sandbox is not None and not plan.generation_only`` (the BigCodeBench plans
    alone carry a sandbox; a generation-only plan — tinycap-shared-standup landing 2 — runs no
    verifier here, so it installs no transport: the CPU verifier installs it), on the
    ``modal_owned`` class the thin task names, before ``eval_async`` is called; and the launcher
    REC reads the ledger the same way it reads ``ledger_stats``."""
    src = (REPO / "src/sisteer/capeval/tiers/inspect_served.py").read_text(encoding="utf-8")
    body = src[src.index("async def run_one_task("):src.index("async def run_inspect_tasks(")]
    i_guard = body.index("if plan.sandbox is not None and not plan.generation_only:\n        # bcb-bfcl-200-widen D73")
    i_refuse = body.index("if plan.sandbox.get(\"provider\") != _SO.PROVIDER_NAME:")
    i_install = body.index("_VT.install_exec_spill(_SO.OwnedModalSandboxEnvironment, ledger_path=_VT.ledger_path_for(store))")
    i_eval = body.index("logs = await eval_async(")
    assert i_guard < i_refuse < i_install < i_eval
    launcher = (REPO / "scripts/modal/tinycap_served.py").read_text(encoding="utf-8")
    assert 'rec["verifier_transport"] = VT.rec_field(out_dir)' in launcher
    assert '"capeval/tinycap/verifier_transport.py"' in launcher            # LAUNCHER_PROCESS_MODULES: byte-identity at preflight
    assert "from sisteer.capeval.tinycap import verifier_transport as VT" in launcher
