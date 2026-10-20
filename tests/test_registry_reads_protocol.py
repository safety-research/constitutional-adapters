"""The ownership protocol of cluster read operations: ``sisteer.registry.reads_protocol``.

The reservation (O_EXCL; a second taker refused by name; released only by its holder), the Slurm reading (squeue + sacct rows folded per
array, filtered to ``reads-*`` / ``attest-*``), the window's TAKEOVER of a tick (its array cancelled by recorded id -- or recovered by name
when the record has none -- confirmed TERMINAL in sacct, ``cancelled_by`` recorded), a live window refusing a takeover and a tick yielding to
any live holder, a dead holder replaced after its jobs are confirmed terminal, the RECONCILE (an orphan job cancelled and confirmed; a
completed-unpulled operation pulled exactly once; a live operation left alone; a job that never reads terminal refuses the submission), the
probe (one batch: `current`, PUBLISHED, the pins' records, the tree), the changed set (never read / pin moved / module blob changed /
unchanged / skipped), the receipt, and THE TICK -- a real end-to-end run over the cluster-read test world (export, plan, stage, the
attestation and the shards run by the fake sbatch and watch, revalidate, pull, merge, receipt; a second tick finds nothing changed) plus
FOUR RACE REHEARSALS: the pre-submit race, a watcher timeout with a live array, a restart after submit, a cancellation after the
computation and before the pull. The CLI's ``--op`` ownership checks on ``submit`` / ``pull`` and the ``reserve`` / ``release`` / ``receipt``
actions close the module. Three further rehearsals: the holder pid is the LONG-LIVED shell (a real ``reserve --pid $$``
subprocess exits while its bash lives on; the reservation stays live, a reused pid reads as gone, a non-ancestor ``--pid`` is refused),
a takeover DURING the pull waits for the critical section (the tick receipts under its own ownership), and terminal confirmation needs an
explicit accounting row per job id (absence from both readings is not terminal). Three more: a release checks and
unlinks under the critical section (a takeover paused against it lands after and keeps its reservation; one paused inside its own section
makes the release remove nothing), the same operation id under another process is a different holder (refused by name, never held by me --
a real second bash "window" with one WINDOW_ID), and a failed ``squeue`` beside a terminal accounting row confirms nothing until both
queries answer (each query's exit status travels in the one-shell answer). Four more: a tick that dies right after
persisting ``pull`` (the next reconcile MERGES then receipts; no receipt without a recorded merge), a takeover arriving
DURING the submit section (it waits for the recorded job ids, then cancels and confirms them before the hand-over), a self-cancel that
cannot be confirmed (fail-closed: ``cancel_failed``, exit 27, the reservation kept, the window's next reconcile confirms before it submits),
and concurrent writers of one op record losing nothing (the per-record lock; ``stages`` merge). Five more: a transport
TIMEOUT on the self-cancel's scancel (one exception family; fail-closed, the reservation kept), a journal write that fails inside the
fail-closed path (the outcome stands), an exception the code did not foresee after the submit (the reservation kept, the record says the
cancel was not attempted), a watch timeout KEEPING the reservation (a window takes it over after cancelling and confirming the array; a
later tick replaces the stale holder the same way), and a marker-less or conflicting OWN revision at the merge (the adapter refused in the
receipt, ``read_whole`` false, the adapter read again). The receipt's precondition is a pure function of DURABLE state:
a conflict moved aside by an INTERRUPTED merge is rediscovered from ``records/`` by the resumed merge (refused, no whole receipt), an earlier
complete delivery REUSED by this operation is in its ``depends_dirs`` and a conflicting accepted record refuses it, and the durable-merge
check passes only when every recorded file is in ``records/`` with its sha256. Also: the harvester's ``gzip_after_ingest``
replacing an accepted ``<name>.jsonl`` by its ``.gz`` twin before a resumed merge or receipt -- the twin counts by its logical sha, the
evidence stays merged and the adapter read whole. And: two differing active forms of one record are a CONFLICT whichever one is
ours, and the inbox merge treats the logical filename as occupied by either form -- a delayed same-name conflicting delivery before or
after the fold is kept aside as ``.conflict-`` by logical sha, never installed as a second active form, and the operation depending on it
gets no whole receipt.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time

import pytest

from sisteer.registry import cli as CLI
from sisteer.registry import cluster_reads as CR
from sisteer.registry import harvest as H
from sisteer.registry import reader_state as RS
from sisteer.registry import reads_protocol as RP
from sisteer.registry import stores as ST

from test_registry_cluster_read import ADAPTER, _back_on_grading_host, _node, _store, _world_with_generation
import test_registry_stores as TS

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEAD_PID = 2 ** 22 + 7          # above pid_max on every Linux here: os.kill(pid, 0) raises ProcessLookupError
_SLURM_WAIT = RP.Slurm.wait_terminal     # bound at import: a test that monkeypatches RP.Slurm must not take the shim's wait with it
NOW = ["2026-10-04T12:00:00Z"]


def _now():
    return NOW[0]


class Died(BaseException):
    """A tick dying mid-step (a crash, a kill): nothing inside the tick handles it; only its ``finally`` runs."""


class FakeSlurm:
    """The queue and the accounting in memory: ``jobs`` = {id: {name, state}}; ``cancel`` turns a non-terminal job CANCELLED after
    ``cancel_lag`` ``states()`` calls (0 = at once; None = never: the job never reads terminal)."""

    def __init__(self, jobs=None, cancel_lag=0):
        self.jobs_ = dict(jobs or {})
        self.cancel_lag = cancel_lag
        self.cancelled = []
        self.pending_cancel = {}
        self.calls = []

    def jobs(self):
        self.calls.append("jobs")
        return [{"id": i, "name": j["name"], "state": j["state"], "where": "squeue" if not RP.TERMINAL_RE.match(j["state"]) else "sacct"}
                for i, j in sorted(self.jobs_.items()) if RP.op_of_job_name(j["name"])]

    def _lag(self):
        for i in list(self.pending_cancel):
            if self.pending_cancel[i] is None:
                continue
            if self.pending_cancel[i] <= 0:
                self.jobs_[i]["state"] = "CANCELLED"
                del self.pending_cancel[i]
            else:
                self.pending_cancel[i] -= 1

    def queue_and_accounting(self, ids):
        """The shim's two readings: a non-terminal job is the queue's, a terminal one the accounting's, an id the
        shim never heard of is in NEITHER -- so a recorded job the accounting has no row for stays unconfirmed, as on the cluster."""
        self.calls.append(("states", tuple(sorted(ids))))
        self._lag()
        queued = {i: self.jobs_[i]["state"] for i in ids if i in self.jobs_ and not RP.TERMINAL_RE.match(self.jobs_[i]["state"])}
        acct = {i: self.jobs_[i]["state"] for i in ids if i in self.jobs_ and RP.TERMINAL_RE.match(self.jobs_[i]["state"])}
        return queued, acct

    def states(self, ids):
        queued, acct = self.queue_and_accounting(ids)
        return {i: queued.get(i, acct.get(i)) for i in ids}

    def cancel(self, ids):
        self.calls.append(("cancel", tuple(sorted(ids))))
        for i in ids:
            self.cancelled.append(i)
            if i in self.jobs_ and not RP.TERMINAL_RE.match(self.jobs_[i]["state"]):
                self.pending_cancel[i] = self.cancel_lag

    def wait_terminal(self, ids, bound_s=RP.CANCEL_BOUND_S, poll_s=RP.CANCEL_POLL_S, sleep=None):
        return _SLURM_WAIT(self, ids, bound_s=bound_s, poll_s=poll_s, sleep=lambda s: None)  # type: ignore[arg-type]


def _name_the_cluster(root: str) -> None:
    """The registry configuration a `cluster-reads` CLI joins the repo's store destination with: the repo's cluster_reads.yaml names the
    destination by the cluster's registry label (cluster_a), and its login host and shared root come from this root's [[clusters]] entry."""
    with open(os.path.join(root, "config.toml"), "w", encoding="utf-8") as fh:
        fh.write('[[clusters]]\nname = "cluster_a"\nhost = "login-test"\nshared = "/shared/test"\n')


def _rd(tmp_path):
    return RP.reads_dir(str(tmp_path / "state"))


# --------------------------------------------------------------------------------------------
# the reservation, the Slurm reading
# --------------------------------------------------------------------------------------------


def test_reservation_is_o_excl_a_second_taker_is_refused_by_name_and_only_the_holder_releases(tmp_path):
    rd = _rd(tmp_path)
    res = RP.reserve(rd, "tick", "t1", "a" * 40, now=_now)
    assert res["owner"] == "tick" and res["op_id"] == "t1" and res["pid"] == os.getpid() and os.path.isfile(RP.reservation_path(rd))
    assert RP.load_op(rd, "t1")["state"] == "reserved"
    with pytest.raises(RP.ReservationHeld) as ei:
        RP.reserve(rd, "window", "W1", now=_now)
    assert ei.value.holder["op_id"] == "t1" and "held by tick t1" in str(ei.value)
    assert RP.release(rd, "W1") is False and os.path.isfile(RP.reservation_path(rd))          # a stale release never removes another's
    assert RP.release(rd, "t1") is True and not os.path.exists(RP.reservation_path(rd)) and RP.release(rd, "t1") is False
    # require_ownership names the holder and a recorded cancel
    RP.reserve(rd, "window", "W1", now=_now)
    with pytest.raises(RP.OwnershipLost) as ei2:
        RP.require_ownership(rd, "t1")
    assert ei2.value.holder["op_id"] == "W1" and "names window W1, not t1" in str(ei2.value)
    assert RP.require_ownership(rd, "W1")["op_id"] == "W1"
    assert RP.op_of_job_name("reads-W1") == ("reads", "W1") and RP.op_of_job_name("attest-x-y") == ("attest", "x-y") and RP.op_of_job_name("gate-abc") is None


def test_slurm_reads_squeue_then_sacct_folds_array_tasks_filters_by_name_and_confirms_terminal(tmp_path):
    scripts = []

    def runner(argv, timeout_s, input=None):
        scripts.append(argv[-1])
        if "scancel" in argv[-1]:
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        if "-j " in argv[-1]:
            return subprocess.CompletedProcess(argv, 0, stdout="---SQUEUE-RC=0---\n4242_0|COMPLETED\n4242_1|CANCELLED by 1000\n4241|COMPLETED\n---SACCT-RC=0---\n", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout=("4242_0|RUNNING|reads-t1\n4242_1|PENDING|reads-t1\n999|RUNNING|f22-other\n4300|PENDING|attest-t2\n"
                                                             "---SQUEUE-RC=0---\n4241|attest-t1|COMPLETED\n4242_0|reads-t1|RUNNING\n4100_0|reads-t0|COMPLETED\n4100_1|reads-t0|COMPLETED\n"
                                                             "4099|attest-t0|COMPLETED\n---SACCT-RC=0---\n"), stderr="")
    s = RP.Slurm(None, runner=runner, user="u")
    jobs = {j["id"]: j for j in s.jobs()}
    assert set(jobs) == {"4242", "4300", "4241", "4100", "4099"}, jobs               # the unrelated job filtered out; array tasks folded per base id
    assert jobs["4242"]["state"] == "RUNNING" and jobs["4242"]["where"] == "squeue"          # the queue's word over the accounting's
    assert jobs["4300"]["state"] == "PENDING" and jobs["4100"]["state"] == "COMPLETED" and jobs["4241"]["state"] == "COMPLETED"
    assert len(scripts) == 1 and "squeue -h -u u" in scripts[0] and "sacct -X -n -P -S now-24hours -u u" in scripts[0]
    # each query is followed by its OWN exit status in the one-shell answer, and neither query's stderr is suppressed
    assert 'echo "---SQUEUE-RC=$?---"' in scripts[0] and 'echo "---SACCT-RC=$?---"' in scripts[0] and "2>/dev/null" not in scripts[0], scripts[0]
    assert RP.Slurm._fold_states(["COMPLETED", "FAILED"]) == "FAILED" and RP.Slurm._fold_states(["COMPLETED", "RUNNING"]) == "RUNNING"
    assert RP.Slurm._fold_states(["COMPLETED", "COMPLETED"]) == "COMPLETED" and RP.Slurm._state("CANCELLED by 1000") == "CANCELLED"
    # states(): the queue first (4242 is not queued in this answer), then sacct -j, folded; an unknown id is None
    st = s.states(["4242", "4241", "7"])
    assert st == {"4242": "CANCELLED", "4241": "COMPLETED", "7": None}
    assert s.queue_and_accounting(["4242", "4241", "7"]) == ({}, {"4242": "CANCELLED", "4241": "COMPLETED"})
    assert s.wait_terminal(["4242", "4241"], bound_s=1, poll_s=0, sleep=lambda x: None) == {"4242": "CANCELLED", "4241": "COMPLETED"}
    # an id the accounting has NO row for is not terminal, however silent the queue -- past the bound the refusal
    # names it; the confirmed ids are not in the complaint
    with pytest.raises(RP.ReconcileRefusal) as ei:
        s.wait_terminal(["4242", "4241", "7"], bound_s=0.05, poll_s=0, sleep=lambda x: None)
    assert "7=no accounting row" in str(ei.value) and "4241" not in str(ei.value) and "4242" not in str(ei.value), str(ei.value)
    s.cancel(["4242", "4300"])
    assert scripts[-1] == "scancel 4242 4300"
    # ... and a job the accounting already calls COMPLETED is NOT terminal while the queue still lists it (COMPLETING): the queue's absence
    # is checked on its own; once the queue drops it, it is confirmed
    queue = ["4242_0|COMPLETING|reads-t1\n"]

    def runner2(argv, timeout_s, input=None):
        return subprocess.CompletedProcess(argv, 0, stdout=queue[0] + "---SQUEUE-RC=0---\n4242_0|COMPLETED\n4242_1|COMPLETED\n---SACCT-RC=0---\n", stderr="")
    s2 = RP.Slurm(None, runner=runner2, user="u")
    assert s2.queue_and_accounting(["4242"]) == ({"4242": "COMPLETING"}, {"4242": "COMPLETED"})
    with pytest.raises(RP.ReconcileRefusal) as ei2:
        s2.wait_terminal(["4242"], bound_s=0.05, poll_s=0, sleep=lambda x: None)
    assert "4242=COMPLETING (queued)" in str(ei2.value)
    queue[0] = ""
    assert s2.wait_terminal(["4242"], bound_s=1, poll_s=0, sleep=lambda x: None) == {"4242": "COMPLETED"}
    # a transport failure is a refusal, never read as "no job"
    bad = RP.Slurm(None, runner=lambda argv, t, input=None: subprocess.CompletedProcess(argv, 255, stdout="", stderr="ssh: boom"), user="u")
    with pytest.raises(RP.SlurmQueryFailed) as ei3:
        bad.jobs()
    assert isinstance(ei3.value, RP.ReconcileRefusal) and "the transport" in ei3.value.what and ei3.value.rc == 255 and "ssh: boom" in str(ei3.value)
    # the marker plumbing through a REAL shell (the local destination: `bash -c`): the first command's failure is told from the second's success
    real = RP.Slurm(None, user="u")
    assert real._two_queries("echo '1_0|RUNNING|reads-t'", "echo '1_0|reads-t|RUNNING'; echo '2|attest-t|COMPLETED'", what="x") == (["1_0|RUNNING|reads-t"], ["1_0|reads-t|RUNNING", "2|attest-t|COMPLETED"])
    with pytest.raises(RP.SlurmQueryFailed) as ei4:
        real._two_queries("(echo 'squeue: error: Socket timed out' >&2; exit 1)", "echo '1|COMPLETED'", what="x")      # a status, not a shell exit
    assert ei4.value.what == "x: squeue" and ei4.value.rc == 1 and "Socket timed out" in str(ei4.value)
    with pytest.raises(RP.SlurmQueryFailed) as ei5:
        real._two_queries("true", "(exit 3)", what="x")
    assert ei5.value.what == "x: sacct" and ei5.value.rc == 3


# --------------------------------------------------------------------------------------------
# takeover, yield, stale holder
# --------------------------------------------------------------------------------------------


def test_terminal_verdicts_is_the_one_confirmation_rule_and_an_id_in_neither_reading_has_no_state_and_is_not_confirmed():
    """Review round 1 MEDIUM: the mirror's settle of a recorded verifier (`stores.settle_awaiting_verify`) re-derived a weaker rule from
    `Slurm.states`, whose None ("no row in the queue, no row in the accounting") it took for terminal. The rule `wait_terminal` polls on is
    now one function, `terminal_verdicts(ids, queued, acct)`, and every caller that must prove a job terminal reads it: confirmed = an
    explicit TERMINAL accounting row AND no queue row; the queue's word beats a terminal accounting row (COMPLETING); a non-terminal
    accounting row with the queue silent is a known state, not confirmed; an id in NEITHER reading has state None, where None and the
    words `no accounting row` -- never confirmed. `wait_terminal`'s complaint carries the same words per id."""
    queued = {"1": "RUNNING", "2": "COMPLETING"}
    acct = {"2": "COMPLETED", "3": "RUNNING", "4": "CANCELLED", "5": "FAILED"}
    v = RP.terminal_verdicts(["1", "2", "3", "4", "5", "6", "6", " "], queued, acct)
    assert list(v) == ["1", "2", "3", "4", "5", "6"]
    assert v["1"] == {"confirmed": False, "state": "RUNNING", "where": "queue", "why": "RUNNING (queued)"}
    assert v["2"] == {"confirmed": False, "state": "COMPLETING", "where": "queue", "why": "COMPLETING (queued)"}
    assert v["3"] == {"confirmed": False, "state": "RUNNING", "where": "accounting", "why": "RUNNING (accounting)"}
    assert v["4"] == {"confirmed": True, "state": "CANCELLED", "where": "accounting", "why": "CANCELLED (accounting), out of the queue"}
    assert v["5"]["confirmed"] is True and v["5"]["state"] == "FAILED"
    assert v["6"] == {"confirmed": False, "state": None, "where": None, "why": "no accounting row"}
    assert RP.terminal_verdicts([], queued, acct) == {}
    # wait_terminal is this rule polled: past the bound its refusal names each unconfirmed id with the rule's words, and once every id is
    # confirmed it returns the accounting's terminal states
    fake = FakeSlurm({"1": {"name": "reads-t1", "state": "RUNNING"}, "4": {"name": "reads-t1", "state": "CANCELLED"}})
    with pytest.raises(RP.ReconcileRefusal) as ei:
        fake.wait_terminal(["1", "4", "6"], bound_s=0.05, poll_s=0)
    assert "1=RUNNING (queued)" in str(ei.value) and "6=no accounting row" in str(ei.value) and "4=" not in str(ei.value), str(ei.value)
    assert fake.wait_terminal(["4"], bound_s=1, poll_s=0) == {"4": "CANCELLED"}


def test_window_takeover_cancels_the_ticks_array_by_recorded_id_waits_for_terminal_and_marks_cancelled_by(tmp_path):
    rd = _rd(tmp_path)
    RP.reserve(rd, "tick", "t1", "a" * 40, now=_now)
    RP.update_op(rd, "t1", now=_now, state="submitted", jobs={"attest": "4241", "reads": "4242"})
    slurm = FakeSlurm({"4241": {"name": "attest-t1", "state": "COMPLETED"}, "4242": {"name": "reads-t1", "state": "RUNNING"}}, cancel_lag=2)
    log = []
    how, res = RP.take_reservation(rd, slurm, "window", "W1", "b" * 40, takeover=True, now=_now, log=log.append, poll_s=0, sleep=lambda s: None)
    assert how == "taken_over" and res["owner"] == "window" and res["op_id"] == "W1" and res["took_over"] == {"owner": "tick", "op_id": "t1", "how": "taken_over"}
    assert slurm.cancelled == ["4242"] and slurm.jobs_["4242"]["state"] == "CANCELLED"
    # the wait polled until sacct read CANCELLED (two lagging answers, then the terminal one)
    assert sum(1 for c in slurm.calls if isinstance(c, tuple) and c[0] == "states") >= 3
    t1 = RP.load_op(rd, "t1")
    assert t1["state"] == "cancelled" and t1["cancelled_by"] == "window W1" and t1["jobs_terminal"] == {"4241": "COMPLETED", "4242": "CANCELLED"}
    assert RP.load_op(rd, "W1")["state"] == "reserved" and RP.read_reservation(rd)["op_id"] == "W1"
    assert any("takes over the tick t1" in l for l in log)
    # the tick's own re-check reads the loss, with the recorded cancel on it
    with pytest.raises(RP.OwnershipLost) as ei:
        RP.require_ownership(rd, "t1")
    assert ei.value.cancelled_by == "window W1"
    # held by me: idempotent
    assert RP.take_reservation(rd, slurm, "window", "W1", now=_now)[0] == "held_by_me"


def test_takeover_recovers_the_array_by_name_when_the_ticks_record_has_no_job_id(tmp_path):
    """The lost write (step 1): the tick's record says `reserved` only, but the queue has reads-t1 / attest-t1 under the fixed names."""
    rd = _rd(tmp_path)
    RP.reserve(rd, "tick", "t1", now=_now)
    slurm = FakeSlurm({"5000": {"name": "attest-t1", "state": "RUNNING"}, "5001": {"name": "reads-t1", "state": "PENDING"}})
    how, _ = RP.take_reservation(rd, slurm, "window", "W1", takeover=True, now=_now, poll_s=0, sleep=lambda s: None)
    assert how == "taken_over" and sorted(slurm.cancelled) == ["5000", "5001"]
    assert RP.load_op(rd, "t1")["jobs_terminal"] == {"5000": "CANCELLED", "5001": "CANCELLED"}


def test_a_live_window_refuses_a_takeover_a_tick_yields_to_any_live_holder_and_a_dead_holder_is_replaced_after_its_jobs_end(tmp_path):
    rd = _rd(tmp_path)
    RP.reserve(rd, "window", "W1", now=_now)                      # this process: alive
    slurm = FakeSlurm()
    with pytest.raises(RP.ReservationHeld):
        RP.take_reservation(rd, slurm, "window", "W2", takeover=True, now=_now)
    with pytest.raises(RP.ReservationHeld):
        RP.take_reservation(rd, slurm, "tick", "t9", takeover=False, now=_now)
    # a tick never takes over a live tick either (takeover=False)
    RP.release(rd, "W1")
    RP.reserve(rd, "tick", "t1", now=_now)
    with pytest.raises(RP.ReservationHeld):
        RP.take_reservation(rd, slurm, "tick", "t2", takeover=False, now=_now)
    assert slurm.cancelled == []
    # the holder's process is gone: its jobs are cancelled and confirmed, the record names the replacement, the reservation is taken
    res = RP.read_reservation(rd)
    res["pid"] = DEAD_PID
    RP.replace_reservation(rd, res)
    RP.update_op(rd, "t1", now=_now, pid=DEAD_PID, state="submitted", jobs={"reads": "6000"})
    slurm2 = FakeSlurm({"6000": {"name": "reads-t1", "state": "RUNNING"}})
    how, new = RP.take_reservation(rd, slurm2, "tick", "t2", takeover=False, now=_now, poll_s=0, sleep=lambda s: None)
    assert how == "replaced_stale" and new["op_id"] == "t2" and slurm2.cancelled == ["6000"]
    t1 = RP.load_op(rd, "t1")
    assert t1["state"] == "cancelled" and t1["stale_replaced_by"] == "tick t2" and t1["cancelled_by"] == "stale-replace tick t2"
    assert RP.holder_alive({"host": "another-host", "pid": 1}) is None and RP.holder_alive({"host": os.uname().nodename, "pid": DEAD_PID}) is False


def test_the_holder_pid_is_the_long_lived_shell_a_reserve_cli_that_exits_leaves_a_live_reservation_and_a_reused_pid_is_stale(tmp_path):
    """The window reserves through a SHORT-LIVED CLI process. Its reservation must name the window shell ($$), verified,
    or the next tick -- during the window's hour-long store-mirror wait -- reads a dead holder, replaces it and cancels the window's jobs.
    A real bash "window" reserves through a child CLI with `--pid $$` and lives on: the CLI is gone, the holder reads ALIVE, a tick yields;
    the shell killed (this test's own child, by pid), the holder is gone and a tick replaces it; a pid REUSED by another process (its start
    time differs from the record's) reads as gone too; a `--pid` that is not a proper ancestor of the CLI -- an unrelated live process, a
    dead pid, the CLI itself -- is refused by name and nothing is reserved."""
    root = str(tmp_path / "root")
    H.ensure_layout(root)
    _name_the_cluster(root)
    state = str(tmp_path / "state")
    rd = RP.reads_dir(state)
    out_file = tmp_path / "reserve.out"
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO_ROOT, "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    cli = (f"{shlex.quote(sys.executable)} -m sisteer.registry.cli --root {shlex.quote(root)} cluster-reads reserve --owner window --op W1 --takeover "
           f"--no-reconcile --state {shlex.quote(state)} --dest cluster_a")
    window = subprocess.Popen(["bash", "-c", f"{cli} --pid $$ >{shlex.quote(str(out_file))} 2>&1; echo RC=$? >>{shlex.quote(str(out_file))}; exec sleep 300"],
                              env=env, cwd=REPO_ROOT)
    slurm = FakeSlurm()
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and "RC=" not in (out_file.read_text() if out_file.exists() else ""):
            time.sleep(0.05)
        out = out_file.read_text() if out_file.exists() else "(no output)"
        assert "RC=0" in out and "reserve: taken for window W1" in out, out
        res = RP.read_reservation(rd)
        assert res["owner"] == "window" and res["op_id"] == "W1" and res["pid"] == window.pid, (res, window.pid)
        assert res["pid_start"] == RP.process_start(window.pid) and res["pid_start"], res
        op = RP.load_op(rd, "W1")
        assert op["pid"] == window.pid and op["pid_start"] == res["pid_start"] and op["state"] == "reserved"
        # the CLI process is gone; the holder -- the shell -- is alive: a tick yields, nothing is cancelled, the reservation stands
        assert RP.holder_alive(res) is True and RP.op_live(op) is True
        with pytest.raises(RP.ReservationHeld) as ei:
            RP.take_reservation(rd, slurm, "tick", "t1", takeover=False, now=_now, poll_s=0, sleep=lambda s: None)
        assert ei.value.holder["op_id"] == "W1" and slurm.cancelled == [] and RP.read_reservation(rd)["op_id"] == "W1"
    finally:
        window.kill()
        window.wait()
    # the shell is gone: the holder reads dead and a tick replaces it (its record names the replacement; its own pid + start time recorded)
    assert RP.holder_alive(res) is False
    how, new = RP.take_reservation(rd, slurm, "tick", "t1", takeover=False, now=_now, poll_s=0, sleep=lambda s: None)
    assert how == "replaced_stale" and new["pid"] == os.getpid() and new["pid_start"] == RP.process_start(os.getpid())
    assert RP.load_op(rd, "W1")["stale_replaced_by"] == "tick t1" and RP.load_op(rd, "t1")["pid_start"] == new["pid_start"]
    assert RP.release(rd, "t1")
    # a REUSED pid: the record names a live pid whose process started at another time -> gone, replaced
    RP.reserve(rd, "window", "W2", now=_now, pid=os.getpid(), pid_start="1")
    assert RP.holder_alive(RP.read_reservation(rd)) is False
    assert RP.take_reservation(rd, slurm, "tick", "t2", takeover=False, now=_now, poll_s=0, sleep=lambda s: None)[0] == "replaced_stale"
    assert RP.release(rd, "t2")
    # the same pid with its true start time: alive (a record without pid_start -- an older one -- falls back to the pid alone)
    RP.reserve(rd, "window", "W3", now=_now, pid=os.getpid())
    assert RP.holder_alive(RP.read_reservation(rd)) is True and RP.holder_alive({"host": os.uname().nodename, "pid": os.getpid()}) is True
    assert RP.release(rd, "W3")
    # --pid verification: an unrelated live process, a dead pid, the CLI itself -> refused by name; the CLI reserves nothing (exit 2)
    other = subprocess.Popen(["sleep", "60"])
    try:
        for bad in (other.pid, DEAD_PID, os.getpid(), 1, 0):
            with pytest.raises(RP.HolderPidInvalid):
                RP.verify_holder_pid(bad)
        assert CLI.main(["--root", root, "cluster-reads", "reserve", "--owner", "window", "--op", "W4", "--no-reconcile", "--pid", str(other.pid),
                         "--state", state, "--dest", "cluster_a"]) == 2
        assert not os.path.exists(RP.reservation_path(rd)) and RP.load_op(rd, "W4") is None
        if os.getppid() > 1:                                                # the parent is a proper ancestor: accepted, with its start time
            assert RP.verify_holder_pid(os.getppid()) == (os.getppid(), RP.process_start(os.getppid()))
    finally:
        other.kill()
        other.wait()


# --------------------------------------------------------------------------------------------
# reconcile
# --------------------------------------------------------------------------------------------


def test_reconcile_cancels_orphan_jobs_pulls_a_completed_unpulled_operation_once_and_leaves_a_live_one_alone(tmp_path):
    rd = _rd(tmp_path)
    # t0: a dead operation whose array COMPLETED, never pulled; t1: dead, array RUNNING; t2: alive (this process), array RUNNING; t3: a job with no record at all
    for op, state, jobs in (("t0", "watched", {"attest": "100", "reads": "101"}), ("t1", "submitted", {"attest": "110", "reads": "111"}),
                            ("t2", "submitted", {"attest": "120", "reads": "121"})):
        RP.save_op(rd, dict(RP.new_op("tick", op, "a" * 40, _now), state=state, jobs=jobs, pid=(os.getpid() if op == "t2" else DEAD_PID)))
    slurm = FakeSlurm({"100": {"name": "attest-t0", "state": "COMPLETED"}, "101": {"name": "reads-t0", "state": "COMPLETED"},
                       "110": {"name": "attest-t1", "state": "COMPLETED"}, "111": {"name": "reads-t1", "state": "RUNNING"},
                       "120": {"name": "attest-t2", "state": "COMPLETED"}, "121": {"name": "reads-t2", "state": "RUNNING"},
                       "130": {"name": "reads-t3", "state": "PENDING"}})
    pulled = []

    def pull_orphan(op):
        pulled.append(op["op_id"])
        RP.write_receipt(rd, {"schema": RP.RECEIPT_SCHEMA, "op": {"op_id": op["op_id"], "owner": "tick"}, "utc": _now(), "adapters": {}})
        RP.update_op(rd, op["op_id"], now=_now, state="receipted")
        return {"pulled": 1}
    rep = RP.reconcile(rd, slurm, "W1", now=_now, poll_s=0, sleep=lambda s: None, pull_orphan=pull_orphan)
    assert rep["left_alive"] == ["t2"] and sorted(rep["cancelled"]) == ["t1", "t3"] and rep["completed_unpulled"] == ["t0"] and pulled == ["t0"]
    assert sorted(slurm.cancelled) == ["111", "130"] and slurm.jobs_["121"]["state"] == "RUNNING"           # the live operation's array untouched
    assert RP.load_op(rd, "t1")["state"] == "cancelled" and RP.load_op(rd, "t1")["cancelled_by"] == "reconcile W1"
    assert RP.load_op(rd, "t3")["state"] == "cancelled"                                                     # a record for the recordless job
    assert RP.load_op(rd, "t0")["state"] == "receipted" and os.path.isfile(RP.receipt_path(rd, "t0"))
    # a second reconcile: t0 has its receipt (never pulled again), t1 and t3 are terminal, t2 still alive
    rep2 = RP.reconcile(rd, slurm, "W1", now=_now, poll_s=0, sleep=lambda s: None, pull_orphan=pull_orphan)
    assert pulled == ["t0"] and rep2["completed_unpulled"] == [] and rep2["cancelled"] == {} and rep2["left_alive"] == ["t2"]
    # an orphan pull that raises is the orphan's failure, recorded, never the reconciler's
    RP.save_op(rd, dict(RP.new_op("tick", "t4", "a" * 40, _now), state="watched", jobs={"reads": "140"}, pid=DEAD_PID))
    slurm.jobs_["140"] = {"name": "reads-t4", "state": "COMPLETED"}

    def boom(op):
        raise RuntimeError("no plan file")
    rep3 = RP.reconcile(rd, slurm, "W1", now=_now, poll_s=0, sleep=lambda s: None, pull_orphan=boom)
    assert rep3["pulled"]["t4"]["error"].startswith("RuntimeError") and RP.load_op(rd, "t4")["state"] == "failed"


def test_reconcile_refuses_when_an_outstanding_job_never_reads_terminal_so_nothing_is_submitted(tmp_path):
    rd = _rd(tmp_path)
    RP.save_op(rd, dict(RP.new_op("tick", "t1", None, _now), state="submitted", jobs={"reads": "111"}, pid=DEAD_PID))
    slurm = FakeSlurm({"111": {"name": "reads-t1", "state": "RUNNING"}}, cancel_lag=None)       # scancel never lands
    with pytest.raises(RP.ReconcileRefusal) as ei:
        RP.reconcile(rd, slurm, "W1", now=_now, bound_s=0.01, poll_s=0, sleep=lambda s: None)
    assert "111=RUNNING" in str(ei.value) and "nothing touches the shared namespace" in str(ei.value)
    assert slurm.cancelled == ["111"]


def test_a_takeover_refuses_until_every_submitted_job_has_an_explicit_terminal_accounting_row_then_changes_hands(tmp_path):
    """`wait_terminal` used to count a job absent from BOTH the queue and the accounting as terminal (`UNKNOWN`), so an
    empty or lagging sacct right after a cancel let the reservation change hands with the old array not proven stopped. Every submitted id
    now needs an explicit terminal accounting row and the queue's silence: a recorded id the accounting has no row for refuses the takeover
    by name (the reservation unchanged, nothing of the window's recorded), and the hand-over happens once the row appears."""
    rd = _rd(tmp_path)
    RP.reserve(rd, "tick", "t1", now=_now)
    RP.update_op(rd, "t1", now=_now, state="submitted", jobs={"attest": "4241", "reads": "4242"})
    slurm = FakeSlurm({"4242": {"name": "reads-t1", "state": "RUNNING"}})          # 4241 is recorded (submitted) but the accounting has no row
    with pytest.raises(RP.ReconcileRefusal) as ei:
        RP.take_reservation(rd, slurm, "window", "W1", takeover=True, now=_now, bound_s=0.01, poll_s=0, sleep=lambda s: None)
    assert "4241=no accounting row" in str(ei.value) and "4242" not in str(ei.value), str(ei.value)
    assert RP.read_reservation(rd)["op_id"] == "t1" and slurm.cancelled == ["4242"] and slurm.jobs_["4242"]["state"] == "CANCELLED"
    assert RP.load_op(rd, "W1") is None and RP.load_op(rd, "t1")["state"] == "submitted" and not RP.load_op(rd, "t1").get("cancelled_by")
    # the row appears: both ids confirmed, the reservation changes hands, the record carries both terminal states
    slurm.jobs_["4241"] = {"name": "attest-t1", "state": "COMPLETED"}
    how, _ = RP.take_reservation(rd, slurm, "window", "W1", takeover=True, now=_now, poll_s=0, sleep=lambda s: None)
    t1 = RP.load_op(rd, "t1")
    assert how == "taken_over" and t1["jobs_terminal"] == {"4241": "COMPLETED", "4242": "CANCELLED"} and t1["cancelled_by"] == "window W1"
    assert RP.read_reservation(rd)["op_id"] == "W1"
    # a cancel by name whose second job the accounting never shows refuses the same way (the tick's `_cancel_own` records this refusal
    # as the operation's error instead of raising it; a reconcile raises it and the caller submits nothing)
    with pytest.raises(RP.ReconcileRefusal):
        RP.cancel_op(rd, FakeSlurm({"7000": {"name": "reads-t7", "state": "RUNNING"}}), "t7", by="x", now=_now, bound_s=0.01, poll_s=0, sleep=lambda s: None,
                     jobs=[{"id": "7000", "name": "reads-t7", "state": "RUNNING"}, {"id": "7001", "name": "attest-t7", "state": "RUNNING"}])


def test_a_failed_squeue_beside_a_terminal_accounting_row_confirms_nothing_until_both_queries_answer(tmp_path):
    """The one-shell command's exit status was the LAST command's (sacct's) and squeue's stderr was suppressed, so a
    squeue that failed while the accounting already read COMPLETED / CANCELLED was an EMPTY queue to `wait_terminal`, and the takeover
    transferred ownership before the job was proven out of the queue (a new array could enter the namespace beside the old one). Now each
    query's exit status travels in the answer and BOTH must be 0: a failed squeue is `SlurmQueryFailed` (a ReconcileRefusal), `wait_terminal`
    confirms no id from that batch and polls on to its bound (a transient failure is retried, never read as silence), and the takeover of a
    tick whose jobs the accounting calls terminal is refused by name while squeue fails -- and lands once both queries answer."""
    answers = []

    def runner(argv, timeout_s, input=None):
        if "scancel" in argv[-1]:
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        rc, out, err = answers.pop(0)
        return subprocess.CompletedProcess(argv, rc, stdout=out, stderr=err)
    s = RP.Slurm(None, runner=runner, user="u")
    # squeue failed (rc 1), sacct answered and terminal, the shell's overall status 0 (the last command's): a failed query, not an empty queue
    squeue_down = (0, "---SQUEUE-RC=1---\n4242_0|CANCELLED by 1000\n4241|COMPLETED\n---SACCT-RC=0---\n", "squeue: error: slurm_load_jobs error: Socket timed out on send/recv operation")
    both_ok = (0, "---SQUEUE-RC=0---\n4242_0|CANCELLED by 1000\n4241|COMPLETED\n---SACCT-RC=0---\n", "")
    answers.append(squeue_down)
    with pytest.raises(RP.SlurmQueryFailed) as ei:
        s.queue_and_accounting(["4242", "4241"])
    assert ei.value.what == "queue_and_accounting: squeue" and ei.value.rc == 1 and isinstance(ei.value, RP.ReconcileRefusal)
    assert "Socket timed out" in str(ei.value) and "proves nothing" in str(ei.value)
    # wait_terminal: the failed batch confirms nothing; the next batch, both queries answering, confirms both ids
    answers += [squeue_down, both_ok]
    assert s.wait_terminal(["4242", "4241"], bound_s=5, poll_s=0, sleep=lambda x: None) == {"4242": "CANCELLED", "4241": "COMPLETED"} and answers == []
    # past the bound with squeue still failing: the refusal names every id as unconfirmed and the query that failed; nothing is confirmed
    answers += [squeue_down] * 3
    with pytest.raises(RP.ReconcileRefusal) as ei2:
        s.wait_terminal(["4242", "4241"], bound_s=0, poll_s=0, sleep=lambda x: None)
    assert "4242=unconfirmed: queue_and_accounting: squeue failed (rc 1)" in str(ei2.value) and "4241=unconfirmed" in str(ei2.value), str(ei2.value)
    # the accounting failing is the same refusal, naming sacct -- the queue's empty answer is not read as "no accounting row"
    answers[:] = [(0, "---SQUEUE-RC=0---\n---SACCT-RC=1---\n", "sacct: error: Problem talking to the database")]
    with pytest.raises(RP.SlurmQueryFailed) as ei3:
        s.queue_and_accounting(["4242"])
    assert ei3.value.what == "queue_and_accounting: sacct" and "database" in str(ei3.value)
    # an answer cut before a marker (the shell died mid-way) proves nothing either
    answers[:] = [(0, "---SQUEUE-RC=0---\n4242_0|COMPLETED\n", "")]
    with pytest.raises(RP.SlurmQueryFailed) as ei4:
        s.queue_and_accounting(["4242"])
    assert "no exit status marker" in ei4.value.what
    # the takeover: the tick's two jobs read terminal in the accounting while squeue fails -> refused by name, the reservation unchanged,
    # nothing of the window's written; squeue back -> taken over with both terminal states on the record
    rd = _rd(tmp_path)
    RP.reserve(rd, "tick", "t1", now=_now)
    RP.update_op(rd, "t1", now=_now, state="submitted", jobs={"attest": "4241", "reads": "4242"})
    jobs_ok = (0, "---SQUEUE-RC=0---\n4241|attest-t1|COMPLETED\n4242_0|reads-t1|CANCELLED by 1000\n4242_1|reads-t1|CANCELLED by 1000\n---SACCT-RC=0---\n", "")
    answers[:] = [jobs_ok, squeue_down]
    with pytest.raises(RP.ReconcileRefusal) as ei5:
        RP.take_reservation(rd, s, "window", "W1", takeover=True, now=_now, bound_s=0, poll_s=0, sleep=lambda x: None)
    assert "squeue failed" in str(ei5.value) and answers == []
    assert RP.read_reservation(rd)["op_id"] == "t1" and RP.load_op(rd, "W1") is None and not RP.load_op(rd, "t1").get("cancelled_by")
    answers[:] = [jobs_ok, both_ok]
    how, _ = RP.take_reservation(rd, s, "window", "W1", takeover=True, now=_now, bound_s=5, poll_s=0, sleep=lambda x: None)
    assert how == "taken_over" and RP.read_reservation(rd)["op_id"] == "W1" and answers == []
    assert RP.load_op(rd, "t1")["jobs_terminal"] == {"4241": "COMPLETED", "4242": "CANCELLED"} and RP.load_op(rd, "t1")["cancelled_by"] == "window W1"


def test_the_same_operation_id_under_another_process_is_a_different_holder_refused_by_name_and_never_held_by_me(tmp_path):
    """`take_reservation` answered `held_by_me` on the op id alone, so two windows started with one WINDOW_ID (an
    operator retry while the first still runs) both proceeded as owners -- one plan file, one pair of job names, conflicting pulls and
    receipts. A reservation is one's own only when the whole holder identity matches (owner, op id, host, pid, pid start time); the same
    id under another process is `OperationIdInUse` (a ReservationHeld): refused by name whatever `takeover` says and whether the holder
    is alive or dead, nothing written, nothing cancelled -- and the real thing: a second bash "window" with the first's WINDOW_ID is refused
    by the CLI (exit 24) while the first shell's own second `reserve` is `held_by_me`."""
    rd = _rd(tmp_path)
    slurm = FakeSlurm()
    RP.reserve(rd, "window", "W1", now=_now)                                                     # this process: alive
    assert RP.take_reservation(rd, slurm, "window", "W1", now=_now)[0] == "held_by_me"           # the same identity, every field
    other = subprocess.Popen(["sleep", "60"])
    try:
        start = RP.process_start(other.pid)
        for takeover in (False, True):
            with pytest.raises(RP.OperationIdInUse) as ei:
                RP.take_reservation(rd, slurm, "window", "W1", takeover=takeover, now=_now, pid=other.pid, pid_start=start, poll_s=0, sleep=lambda x: None)
            assert isinstance(ei.value, RP.ReservationHeld) and ei.value.holder["pid"] == os.getpid() and ei.value.me["pid"] == other.pid
            assert "already held by another process" in str(ei.value) and "a retry needs a new operation id" in str(ei.value)
        with pytest.raises(RP.OperationIdInUse):                      # the same process under another owner: not the same holder either
            RP.take_reservation(rd, slurm, "tick", "W1", now=_now)
        with pytest.raises(RP.OperationIdInUse):                      # a reused pid (another start time) under the same id
            RP.take_reservation(rd, slurm, "window", "W1", now=_now, pid=os.getpid(), pid_start="1")
        res = RP.read_reservation(rd)
        assert res["pid"] == os.getpid() and res["op_id"] == "W1" and "took_over" not in res and slurm.cancelled == [] and slurm.calls == []
        assert RP.load_op(rd, "W1")["state"] == "reserved" and RP.load_op(rd, "W1")["pid"] == os.getpid()
        # a DEAD holder with the same id is not replaced as stale either (the first window's shell crashed; a retry needs a new id) ...
        RP.replace_reservation(rd, dict(res, pid=DEAD_PID))
        with pytest.raises(RP.OperationIdInUse):
            RP.take_reservation(rd, slurm, "window", "W1", takeover=True, now=_now, poll_s=0, sleep=lambda x: None)
        assert RP.read_reservation(rd)["pid"] == DEAD_PID and slurm.cancelled == []
        # ... while another operation id replaces it as stale, the ordinary rule
        assert RP.take_reservation(rd, slurm, "window", "W2", takeover=True, now=_now, poll_s=0, sleep=lambda x: None)[0] == "replaced_stale"
        assert RP.load_op(rd, "W1")["stale_replaced_by"] == "window W2"
    finally:
        other.kill()
        other.wait()
    assert RP.release(rd, "W2") and not os.path.exists(RP.reservation_path(rd))
    # two real bash "windows" with ONE WINDOW_ID: the first reserves (taken) and reserves again (held by me: the same shell, pid and start
    # time) and lives on; the second's reserve is refused by name, exit 24, the reservation and the op record still the first shell's
    root = str(tmp_path / "root")
    H.ensure_layout(root)
    _name_the_cluster(root)
    state = str(tmp_path / "state")
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO_ROOT, "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    cli = (f"{shlex.quote(sys.executable)} -m sisteer.registry.cli --root {shlex.quote(root)} cluster-reads reserve --owner window --op W9 --takeover "
           f"--no-reconcile --state {shlex.quote(state)} --dest cluster_a")
    out1, out2 = tmp_path / "first.out", tmp_path / "second.out"
    first = subprocess.Popen(["bash", "-c", f"{cli} --pid $$ >{shlex.quote(str(out1))} 2>&1; echo RC1=$? >>{shlex.quote(str(out1))}; "
                                             f"{cli} --pid $$ >>{shlex.quote(str(out1))} 2>&1; echo RC2=$? >>{shlex.quote(str(out1))}; exec sleep 300"],
                             env=env, cwd=REPO_ROOT)
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and "RC2=" not in (out1.read_text() if out1.exists() else ""):
            time.sleep(0.05)
        text1 = out1.read_text() if out1.exists() else "(no output)"
        assert "reserve: taken for window W9" in text1 and "RC1=0" in text1 and "reserve: held_by_me for window W9" in text1 and "RC2=0" in text1, text1
        res = RP.read_reservation(rd)
        assert res["pid"] == first.pid and res["pid_start"] == RP.process_start(first.pid), (res, first.pid)
        second = subprocess.run(["bash", "-c", f"{cli} --pid $$ >{shlex.quote(str(out2))} 2>&1; echo RC=$? >>{shlex.quote(str(out2))}"], env=env, cwd=REPO_ROOT, timeout=120)
        text2 = out2.read_text()
        assert "RC=24" in text2 and "REFUSED" in text2 and "already held by another process" in text2 and f"pid {first.pid}" in text2, text2
        assert "the same operation id under another process" in text2, text2
        assert RP.read_reservation(rd) == res, "the reservation is untouched by the refused second window"
        op = RP.load_op(rd, "W9")
        assert op["pid"] == first.pid and op["state"] == "reserved" and not op.get("took_over"), op
        assert RP.holder_alive(res) is True
    finally:
        first.kill()
        first.wait()


def test_a_release_checks_and_unlinks_under_the_critical_section_so_a_takeover_paused_against_it_keeps_its_reservation(tmp_path, monkeypatch):
    """`release` read the reservation and unlinked it OUTSIDE the critical section. A window's takeover that
    replaced the reservation between the two (its own hand-over runs under the lock, but the release did not) lost its reservation to the
    old holder's release, and the next tick then reserved during the window's hour-long mirror wait: two arrays in one namespace. Now the
    release's check and unlink are ONE hold of the lock. Both interleavings, with the takeover in a second thread (a second process live;
    the flock conflicts across open file descriptions): (a) the release holds the section and is PAUSED after its locked read -- the
    takeover that arrives waits, lands only after the unlink, as `taken`, and its reservation stands; (b) the takeover holds the section
    and is paused after its write -- the release waits, re-reads the window's reservation and removes nothing."""
    rd = _rd(tmp_path)
    slurm = FakeSlurm()
    # (a) the release paused between its locked read and the unlink, the takeover arriving in the pause
    RP.reserve(rd, "tick", "t1", now=_now)                                                       # this process: alive; a window may take it over
    marks = {}
    in_release, release_locked, takeover_attempted, takeover_done = threading.Event(), threading.Event(), threading.Event(), threading.Event()
    main = threading.get_ident()
    real_read = RP.read_reservation
    reads_in_release = [0]

    def read_paused(rd_):
        res = real_read(rd_)
        if threading.get_ident() == main and in_release.is_set():
            reads_in_release[0] += 1
            if reads_in_release[0] == 2:                                                          # the read UNDER the lock: pause here
                release_locked.set()                                                              # ... and only now may the takeover start
                assert takeover_attempted.wait(30), "the takeover thread never started"
                time.sleep(0.6)                                                                   # the takeover blocks on the section meanwhile
                marks["takeover_landed_in_pause"] = takeover_done.is_set()
                marks["holder_in_pause"] = (real_read(rd_) or {}).get("op_id")
        return res
    monkeypatch.setattr(RP, "read_reservation", read_paused)

    def takeover():
        assert release_locked.wait(60), "the release never reached its locked read"
        takeover_attempted.set()
        how, res = RP.take_reservation(rd, slurm, "window", "W1", takeover=True, now=_now, poll_s=0, sleep=lambda x: None, section_poll_s=0.02)
        marks["takeover"] = (how, time.monotonic(), res["op_id"])
        takeover_done.set()
    th = threading.Thread(target=takeover, daemon=True)
    th.start()
    in_release.set()
    removed = RP.release(rd, "t1", poll_s=0.02)
    marks["release_end"] = time.monotonic()
    th.join(30)
    assert not th.is_alive() and removed is True, marks
    assert marks["holder_in_pause"] == "t1" and marks["takeover_landed_in_pause"] is False, marks          # nothing replaced under the release
    assert marks["takeover"][0] == "taken" and marks["takeover"][1] >= marks["release_end"] and marks["takeover"][2] == "W1", marks
    assert RP.read_reservation(rd)["op_id"] == "W1" and os.path.isfile(RP.reservation_path(rd))            # the window's reservation stands
    assert RP.load_op(rd, "W1")["state"] == "reserved" and RP.load_op(rd, "t1")["state"] == "reserved"    # no cancel: nothing of t1's was held when it took
    monkeypatch.setattr(RP, "read_reservation", real_read)
    in_release.clear()
    assert RP.release(rd, "W1") is True
    # (b) the takeover paused inside ITS section after the write; the release arriving meanwhile waits and then removes nothing
    RP.reserve(rd, "tick", "t2", now=_now)
    marks.clear()
    in_release_b, release_preread, release_done, takeover_done = threading.Event(), threading.Event(), threading.Event(), threading.Event()
    real_replace = RP.replace_reservation

    def read_signalled(rd_):
        res = real_read(rd_)
        if threading.get_ident() == main and in_release_b.is_set():
            release_preread.set()                           # the release's lock-free pre-read saw the tick's reservation; it now waits for the lock
        return res
    monkeypatch.setattr(RP, "read_reservation", read_signalled)

    def replace_paused(rd_, res):
        assert release_preread.wait(30), "the release never did its pre-read"
        real_replace(rd_, res)
        time.sleep(0.6)                                                                           # the release blocks on the section meanwhile
        marks["release_done_in_pause"] = release_done.is_set()
        marks["file_in_pause"] = os.path.isfile(RP.reservation_path(rd_))
    monkeypatch.setattr(RP, "replace_reservation", replace_paused)

    def takeover2():
        how, _ = RP.take_reservation(rd, slurm, "window", "W2", takeover=True, now=_now, poll_s=0, sleep=lambda x: None, section_poll_s=0.02)
        marks["takeover"] = how
        takeover_done.set()
    real_section = RP.critical_section
    section_entered = threading.Event()

    @contextlib.contextmanager
    def section_observed(rd_, op_id, owner, what, **kw):
        with real_section(rd_, op_id, owner, what, **kw):
            if what == "reservation":
                section_entered.set()
            yield
    monkeypatch.setattr(RP, "critical_section", section_observed)
    th2 = threading.Thread(target=takeover2, daemon=True)
    th2.start()
    assert section_entered.wait(30), "the takeover never entered its section"
    in_release_b.set()
    removed2 = RP.release(rd, "t2", poll_s=0.02)
    release_done.set()
    in_release_b.clear()
    th2.join(30)
    assert not th2.is_alive() and removed2 is False, marks
    assert marks["release_done_in_pause"] is False and marks["file_in_pause"] is True and marks["takeover"] == "taken_over", marks
    assert RP.read_reservation(rd)["op_id"] == "W2" and RP.load_op(rd, "t2")["cancelled_by"] == "window W2"
    monkeypatch.setattr(RP, "replace_reservation", real_replace)
    monkeypatch.setattr(RP, "critical_section", real_section)
    monkeypatch.setattr(RP, "read_reservation", real_read)
    # a release that finds another holder takes no lock and removes nothing; one whose section stays busy raises, the reservation standing
    assert RP.release(rd, "t2") is False and os.path.isfile(RP.reservation_path(rd))
    fd = os.open(RP.critical_lock_path(rd), os.O_RDWR | os.O_CREAT, 0o644)
    import fcntl
    fcntl.flock(fd, fcntl.LOCK_EX)                                                                # another description holds the section
    try:
        with pytest.raises(RP.CriticalSectionBusy):
            RP.release(rd, "W2", bound_s=0.05, poll_s=0.01)
        assert os.path.isfile(RP.reservation_path(rd)) and RP.read_reservation(rd)["op_id"] == "W2"
    finally:
        os.close(fd)
    assert RP.release(rd, "W2") is True


# --------------------------------------------------------------------------------------------
# the probe, the changed set, the receipt
# --------------------------------------------------------------------------------------------


def _gen(shared, slug, sha8, published=True, sealed=True, verified=True, current=True, pub_doc=None):
    g = os.path.join(shared, "registry", "stores", "gen", slug)
    os.makedirs(os.path.join(g, sha8), exist_ok=True)
    if current:
        if os.path.islink(os.path.join(g, "current")):
            os.remove(os.path.join(g, "current"))
        os.symlink(sha8, os.path.join(g, "current"))
    for name, on in (("VERIFIED", verified), ("SEALED", sealed)):
        if on:
            open(os.path.join(g, f"{name}-{sha8}.json"), "w").write("{}\n")
    if published:
        doc = pub_doc or {"content_sha8": sha8, "published_utc": "2026-10-04T10:00:00Z", "change_utc": "2026-10-04T09:30:00Z", "mode": "push", "producer": "grading_host:x"}
        open(os.path.join(g, f"PUBLISHED-{sha8}.json"), "w").write(json.dumps(doc, indent=1) + "\n")
    return g


def _cfg_two(tmp_path, adapters):
    comps = {"c1": {"base": str(tmp_path / "home" / "c1"), "authority": "cluster_a", "producer": "frozen", "grading_host_mirror": str(tmp_path / "home" / "c1")},
             "c2": {"base": str(tmp_path / "home" / "c2"), "authority": "cluster_a", "producer": "frozen", "grading_host_mirror": str(tmp_path / "home" / "c2")}}
    return ST.load_config(TS._cfg(tmp_path, comps, adapters=adapters))


def test_probe_reads_current_published_the_pins_records_and_the_tree_in_one_batch(tmp_path):
    cfg = _cfg_two(tmp_path, {"a1": {"shards": 1, "components": ["c1"]}, "a2": {"shards": 2, "components": ["c1", "c2"]}})
    dest = cfg.destination("cluster_a")
    _gen(dest.shared, "c1", "11111111")
    _gen(dest.shared, "c2", "22222222", published=False, sealed=False)
    _gen(dest.shared, "c2", "22222221", current=False)             # an older generation of c2 with its record: a pin a receipt may read
    os.makedirs(os.path.join(dest.shared, "gates", "trees", "t" * 40, "src", "sisteer", "registry"))
    calls = []

    def runner(argv, timeout_s, input=None):
        calls.append(argv)
        return ST._default_runner(argv, timeout_s, input=input)
    p = RP.probe(cfg, dest, ["c1", "c2"], runner=runner, tree=f"{dest.shared}/gates/trees/{'t' * 40}", pins={"c2": "22222221"})
    assert len(calls) == 1, "one batch"
    assert p["components"]["c1"]["current"] == "11111111" and p["components"]["c1"]["exists"] and p["components"]["c1"]["sealed"] and p["components"]["c1"]["published_record"]
    assert p["components"]["c1"]["published"] == {"published_utc": "2026-10-04T10:00:00Z", "change_utc": "2026-10-04T09:30:00Z", "mode": "push", "producer": "grading_host:x"}
    assert p["components"]["c2"]["current"] == "22222222" and not p["components"]["c2"]["sealed"] and not p["components"]["c2"]["published_record"] and p["components"]["c2"]["published"] is None
    assert p["pins"]["c2"]["22222221"]["content_sha8"] == "22222221" and p["tree_present"] is True
    p2 = RP.probe(cfg, dest, ["c1"], runner=runner, tree=f"{dest.shared}/gates/trees/{'u' * 40}", pins={"c1": "deadbeef"})
    assert p2["tree_present"] is False and p2["pins"]["c1"]["deadbeef"] is None
    # a declared component the batch returned nothing for is a refusal by name (a transport that dropped lines), never an empty row
    with pytest.raises(CR.ClusterReadsError) as ei:
        RP.probe(cfg, dest, ["c1", "c2"], runner=lambda argv, t, input=None: subprocess.CompletedProcess(argv, 0, stdout="", stderr=""))
    assert "returned nothing for component(s) c1, c2" in str(ei.value)


def test_changed_set_rules_never_read_pin_moved_module_changed_unchanged_and_skipped(tmp_path):
    cfg = _cfg_two(tmp_path, {"a1": {"shards": 1, "components": ["c1"]}, "a2": {"shards": 2, "components": ["c1", "c2"]},
                              "a3": {"shards": 1, "components": ["c2"]}, "a4": {"cluster": "refused", "reason": "no pinned paper worktree"}})
    probed = {"components": {"c1": {"current": "11111111", "exists": True}, "c2": {"current": "22222222", "exists": True}}}
    blobs = {"a1": {"src/sisteer/registry/adapters/a1.py": "b1"}, "a2": {"src/sisteer/registry/adapters/a2.py": "b2"}, "a3": {}, "a4": {}}
    # no receipts: every readable adapter is changed (never read); the declared-refused one is skipped
    cs = RP.changed_set(cfg, ["a1", "a2", "a3", "a4", "a5"], probed, blobs, lambda a: None)
    assert set(cs["changed"]) == {"a1", "a2", "a3"} and all(r == ["never read whole (no receipt)"] for r in cs["changed"].values())
    assert cs["skipped"]["a4"].startswith("declared cluster: refused") and cs["skipped"]["a5"].startswith("not declared")
    # receipts matching the pins and blobs: unchanged; a moved pin, a changed blob: changed with the reason
    entries = {"a1": {"pins": {"c1": "11111111"}, "module_blobs": {"src/sisteer/registry/adapters/a1.py": "b1"}, "shards_total": 1, "shards_ok": 1, "refused": None},
               "a2": {"pins": {"c1": "11111111", "c2": "22222220"}, "module_blobs": {"src/sisteer/registry/adapters/a2.py": "b2"}, "shards_total": 2, "shards_ok": 2, "refused": None},
               "a3": {"pins": {"c2": "22222222"}, "module_blobs": {"src/sisteer/registry/adapters/a3.py": "old"}, "shards_total": 1, "shards_ok": 1, "refused": None}}
    last = lambda a: ({"utc": _now()}, entries[a]) if a in entries else None
    cs2 = RP.changed_set(cfg, ["a1", "a2", "a3"], probed, dict(blobs, a3={"src/sisteer/registry/adapters/a3.py": "new"}), last)
    assert cs2["unchanged"] == ["a1"] and cs2["changed"]["a2"] == ["pin moved: c2 22222220 -> 22222222"]
    assert cs2["changed"]["a3"] == ["module changed: src/sisteer/registry/adapters/a3.py old -> new"]
    # a receipt that did not read the adapter whole does not count (last_receipt_for skips it)
    rd = _rd(tmp_path)
    RP.write_receipt(rd, {"schema": RP.RECEIPT_SCHEMA, "op": {"op_id": "r1", "owner": "tick"}, "utc": "2026-10-04T11:00:00Z",
                          "adapters": {"a1": dict(entries["a1"], shards_ok=0, refused=["pull: x"])}})
    RP.write_receipt(rd, {"schema": RP.RECEIPT_SCHEMA, "op": {"op_id": "r0", "owner": "window"}, "utc": "2026-10-04T10:00:00Z", "adapters": {"a1": entries["a1"]}})
    rec, e = RP.last_receipt_for(rd, "a1")
    assert rec["op"]["op_id"] == "r0" and e["shards_ok"] == 1
    assert RP.last_receipt_for(rd, "a2") is None
    # no `current`, or a dangling one: skipped with the reason (a read would only record a refusal)
    probed2 = {"components": {"c1": {"current": None, "exists": False}, "c2": {"current": "22222222", "exists": False}}}
    cs3 = RP.changed_set(cfg, ["a1", "a3"], probed2, blobs, lambda a: None)
    assert cs3["skipped"]["a1"].startswith("component c1: nothing published") and "dangling" in cs3["skipped"]["a3"] and cs3["changed"] == {}


def test_receipt_from_plan_carries_pins_published_records_blobs_shards_pulled_dirs_and_refusals():
    plan = {"window": "W1", "tree_sha": "a" * 40, "destination": "cluster_a",
            "adapters": {"a1": {"route": "cluster", "shards": 2, "pins": {"c1": "11111111"}}, "a2": {"route": "grading_host", "shards": 1, "pins": {}}},
            "refusals": {"a2": ["plan: no published generation for component(s): c2"]},
            "stages": {"pull": {"summaries": {"a1": {"expected": 2, "present": 2, "ok": 2}}, "pulled_dirs": ["a1/rev1", "a1/rev2", "a1/_refused/rev3"],
                                "reactivations": {"a1": {"revX": "reactivated", "revY": "current"}}, "attestation": {"ok": True}, "barrier": None,
                                # a group re-read and refused again byte for byte under the same inputs and tree -- a recorded result
                                "standing_refusals": {"a1": [{"group": "g3", "revision_id": "rev3", "state": "standing_refusal", "refusal": "synthetic read failure",
                                                              "prior_refusal": {"window": "W0", "utc": "2026-10-04T08:00:00.000Z", "digest": "f" * 64,
                                                                                "attempt_revision_id": "rev3", "code_sha": "a" * 40}}]}}}}
    pub = {"c1": {"11111111": {"published_utc": "P", "change_utc": "C", "mode": "push", "producer": "grading_host:x"}}}
    rec = RP.receipt_from_plan(plan, pub, {"a1": {"src/x.py": "b1"}}, {"op_id": "W1", "owner": "window", "jobs": {"reads": "4242"}}, now=_now)
    a1, a2 = rec["adapters"]["a1"], rec["adapters"]["a2"]
    assert a1["read_whole"] and a1["shards_ok"] == 2 and a1["shards_total"] == 2 and a1["revisions_pulled"] == ["a1/rev1", "a1/rev2"] and a1["reactivated"] == ["revX"]
    # the refusal record the pull delivered is listed apart from the revisions; the standing group is a result with its state, whole all the same;
    # `refused_standing` (the list of emit-time-refused ids left unread) is untouched
    assert a1["refusal_records_pulled"] == ["a1/_refused/rev3"] and a1["refused_standing"] == []
    assert a1["standing_refusals"] == plan["stages"]["pull"]["standing_refusals"]["a1"] and a1["standing_refusals"][0]["state"] == "standing_refusal"
    assert set(a1["standing_refusals"][0]["prior_refusal"]) >= {"window", "utc", "digest"} and a1["standing_refusals"][0]["refusal"] == "synthetic read failure"
    assert a2["standing_refusals"] == [] and a2["refusal_records_pulled"] == []
    assert a1["published"] == {"c1": {"published_utc": "P", "change_utc": "C", "mode": "push", "producer": "grading_host:x"}} and a1["module_blobs"] == {"src/x.py": "b1"} and a1["refused"] is None
    assert not a2["read_whole"] and a2["shards_total"] == 0 and a2["refused"] == ["plan: no published generation for component(s): c2"]
    assert rec["n_read_whole"] == 1 and rec["n_refused"] == 1 and rec["jobs"] == {"reads": "4242"} and rec["schema"] == RP.RECEIPT_SCHEMA


def test_a_withheld_retirement_re_activation_leaves_the_receipt_incomplete_like_a_standing_id_so_the_next_tick_reads_the_adapter_again(tmp_path):
    """A retirement the pull WITHHELD (the shard's claim named a head that moved before the
    pull) is neither a refusal nor whole: the entry lists the id under `withheld`, `refused` stays None, `read_whole`
    is false, `standing_unread` returns it (so an older whole receipt with the same pins and module cannot park it) and the changed set
    reads the adapter again, where the next shard judges the group against today's head; the status page reads it as standing_unread."""
    from sisteer.registry import status as STATUS
    plan = {"window": "W1", "tree_sha": "a" * 40, "destination": "cluster_a",
            "adapters": {"a1": {"route": "cluster", "shards": 2, "pins": {"c1": "11111111"}}}, "refusals": {},
            "stages": {"pull": {"summaries": {"a1": {"expected": 2, "present": 2, "ok": 2}}, "pulled_dirs": [],
                                "reactivations": {"a1": {"revW": "withheld", "revY": "current"}}, "attestation": {"ok": True}, "barrier": None}}}
    blobs = {"a1": {"src/x.py": "b1"}}
    rec = RP.receipt_from_plan(plan, {}, blobs, {"op_id": "W1", "owner": "window"}, now=_now)
    a1 = rec["adapters"]["a1"]
    assert a1["read_whole"] is False and a1["refused"] is None and a1["withheld"] == ["revW"] and a1["refused_standing"] == [] and a1["reactivated"] == []
    assert rec["n_read_whole"] == 0 and rec["n_refused"] == 0
    assert STATUS.standing_unread_ids(a1) == ["revW"] and STATUS.newest_entry_state(a1, RP.read_whole(a1)) == ("standing_unread", "amber")
    rd = _rd(tmp_path)
    RP.write_receipt(rd, rec)
    assert RP.last_receipt_for(rd, "a1") is None and RP.standing_unread(rd, "a1") == ["revW"]
    cfg = _cfg_two(tmp_path, {"a1": {"shards": 2, "components": ["c1"]}})
    probed = {"components": {"c1": {"current": "11111111", "exists": True}}}
    cs = RP.changed_set(cfg, ["a1"], probed, blobs, lambda a: RP.last_receipt_for(rd, a), standing=lambda a: RP.standing_unread(rd, a))
    assert cs["unchanged"] == [] and "left standing unread" in cs["changed"]["a1"][-1] and "revW" in cs["changed"]["a1"][-1]
    # an older whole receipt with the same pins and module does not park it: the standing check still reads the adapter
    plan_ok = dict(plan, stages={"pull": dict(plan["stages"]["pull"], reactivations={"a1": {"revW": "reactivated", "revY": "current"}})})
    RP.write_receipt(rd, RP.receipt_from_plan(plan_ok, {}, blobs, {"op_id": "W0", "owner": "window"}, now=lambda: "2026-10-04T09:00:00Z"))
    assert RP.last_receipt_for(rd, "a1")[0]["op"]["op_id"] == "W0" and RP.standing_unread(rd, "a1") == ["revW"]
    cs2 = RP.changed_set(cfg, ["a1"], probed, blobs, lambda a: RP.last_receipt_for(rd, a), standing=lambda a: RP.standing_unread(rd, a))
    assert cs2["unchanged"] == [] and cs2["changed"]["a1"] == ["refused-known id(s) left standing unread by the last pull: revW -- read fresh on this tree"]


def test_a_retirement_the_harvester_held_for_a_moved_head_keeps_the_adapter_unread_until_a_window_attempts_it_again(tmp_path):
    """Round 2 of the review of registry-hardening 3a, as round 3 left it: a NEW retirement the harvester HOLDS because the head it was
    computed against moved before the fold (`harvest.RETIREMENT_HEAD_KIND`) is recorded at the fold, which the window that delivered it
    neither saw nor could act on, so that window's receipt reads whole and would park the adapter while its pins and module stood. The hold
    is read from the database (`harvest.retirements_held_for_moved_head`) and counts as standing unread until a window that HAD IT IN HAND
    attempted the adapter -- `held_retirements_unread` matches the hold's identity (the revision id, the held carrier's uri) against the
    holds each attempted receipt's plan recorded (`holds_seen`, `cluster_reads.plan_window` -> `receipt_from_plan`): a receipt whose plan
    saw nothing leaves it standing whatever the stamps say, a non-attempted receipt says nothing whatever its plan saw, and an attempted
    window whose plan recorded the hold ends it (its shard judged the group against today's head); `standing_ids` composes it with the
    receipt-side standing ids, and the changed set reads the adapter again."""
    plan = {"window": "W1", "tree_sha": "a" * 40, "destination": "cluster_a",
            "adapters": {"a1": {"route": "cluster", "shards": 2, "pins": {"c1": "11111111"}, "holds_seen": []}}, "refusals": {},
            "stages": {"pull": {"summaries": {"a1": {"expected": 2, "present": 2, "ok": 2}}, "pulled_dirs": ["a1/revR"],
                                "reactivations": {}, "attestation": {"ok": True}, "barrier": None}}}
    blobs = {"a1": {"src/x.py": "b1"}}
    w1 = RP.receipt_from_plan(plan, {}, blobs, {"op_id": "W1", "owner": "window"}, now=lambda: "2026-10-04T10:00:00Z")
    assert w1["adapters"]["a1"]["read_whole"] is True and w1["adapters"]["a1"]["revisions_pulled"] == ["a1/revR"] and w1["adapters"]["a1"]["holds_seen"] == []
    hold = {"revision_id": "revR", "source_id": "a1:s3", "first_seen_utc": "2026-10-04T10:30:00Z", "record_uri": "records/adapters/a1/revR/000001-source_revision.jsonl",
            "blocking_head": "revH2"}
    seen = [{"revision_id": "revR", "record_uri": hold["record_uri"], "blocking_head": "revH2"}]
    # the delivering window had no hold in hand: standing; with no receipt at all: standing; no holds: nothing
    assert RP.held_retirements_unread("a1", [w1], [hold]) == ["revR"]
    assert RP.held_retirements_unread("a1", [], [hold]) == ["revR"] and RP.held_retirements_unread("a1", [w1], []) == []
    # whatever the stamps say: a hold stamped BEFORE that receipt's utc (the harvester's cycle clock) stands just the same
    assert RP.held_retirements_unread("a1", [w1], [dict(hold, first_seen_utc="2026-10-04T09:30:00Z")]) == ["revR"]
    assert RP.held_retirements_unread("other", [w1], [hold]) == ["revR"]            # another adapter's receipts say nothing about this one
    # a later receipt whose window never attempted the adapter (routed off the cluster) does not end the hold, even with it in hand
    off = RP.receipt_from_plan(dict(plan, adapters={"a1": {"route": "grading_host", "shards": 0, "pins": {}, "holds_seen": seen}}, refusals={"a1": ["declared refused"]}),
                               {}, blobs, {"op_id": "W2", "owner": "window"}, now=lambda: "2026-10-04T11:00:00Z")
    assert not RP.attempted(off["adapters"]["a1"]) and off["adapters"]["a1"]["holds_seen"] == seen and RP.held_retirements_unread("a1", [off, w1], [hold]) == ["revR"]
    # an attempted window whose plan had the hold in hand ends it; one planned without it does not, however late its receipt
    stages3 = {"pull": dict(plan["stages"]["pull"], pulled_dirs=[], reactivations={"a1": {"revR": "reactivated"}})}
    w3 = RP.receipt_from_plan(dict(plan, adapters={"a1": dict(plan["adapters"]["a1"], holds_seen=seen)}, stages=stages3),
                              {}, blobs, {"op_id": "W3", "owner": "window"}, now=lambda: "2026-10-04T12:00:00Z")
    assert RP.attempted(w3["adapters"]["a1"]) and w3["adapters"]["a1"]["holds_seen"] == seen and RP.held_retirements_unread("a1", [w3, off, w1], [hold]) == []
    w3b = RP.receipt_from_plan(dict(plan, stages=stages3), {}, blobs, {"op_id": "W3b", "owner": "window"}, now=lambda: "2026-10-04T12:00:00Z")
    assert RP.attempted(w3b["adapters"]["a1"]) and RP.held_retirements_unread("a1", [w3b, off, w1], [hold]) == ["revR"]
    # a hold a LATER bundle of the same revision incurred (another carrier) is unread until a receipt names that one
    hold2 = dict(hold, first_seen_utc="2026-10-04T13:00:00Z", record_uri="records/adapters/a1/_activations/revR/20261004T125500Z/000999998-source_revision.jsonl")
    assert RP.held_retirements_unread("a1", [w3, off, w1], [hold, hold2]) == ["revR"]
    w4 = RP.receipt_from_plan(dict(plan, adapters={"a1": dict(plan["adapters"]["a1"], holds_seen=seen + [{"revision_id": "revR", "record_uri": hold2["record_uri"], "blocking_head": "revH2"}])},
                                   stages=stages3), {}, blobs, {"op_id": "W4", "owner": "window"}, now=lambda: "2026-10-04T14:00:00Z")
    assert RP.held_retirements_unread("a1", [w4, w3, off, w1], [hold, hold2]) == []
    # the head the hold was computed against is part of its identity (the closing pass): the SAME carrier re-held against a newer head --
    # the harvester rewrote its queue row's missing_parent_id once revH2 was superseded -- is a new hold every receipt so far missed; a
    # receipt entry naming no head (written before the head joined the identity) matches no hold at all
    hold_h3 = dict(hold, blocking_head="revH3")
    assert RP.held_retirements_unread("a1", [w4, w3, off, w1], [hold_h3]) == ["revR"] and RP.held_retirements_unread("a1", [w4, w3, off, w1], [hold, hold_h3]) == ["revR"]
    w5 = RP.receipt_from_plan(dict(plan, adapters={"a1": dict(plan["adapters"]["a1"], holds_seen=[{"revision_id": "revR", "record_uri": hold["record_uri"], "blocking_head": "revH3"}])},
                                   stages=stages3), {}, blobs, {"op_id": "W5", "owner": "window"}, now=lambda: "2026-10-04T15:00:00Z")
    assert RP.held_retirements_unread("a1", [w5, w4, w3, off, w1], [hold, hold2, hold_h3]) == []
    headless = RP.receipt_from_plan(dict(plan, adapters={"a1": dict(plan["adapters"]["a1"], holds_seen=[{"revision_id": "revR", "record_uri": hold["record_uri"]}])},
                                         stages=stages3), {}, blobs, {"op_id": "W6", "owner": "window"}, now=lambda: "2026-10-04T16:00:00Z")
    assert headless["adapters"]["a1"]["holds_seen"] == [{"revision_id": "revR", "record_uri": hold["record_uri"], "blocking_head": ""}]
    assert RP.held_retirements_unread("a1", [headless], [hold]) == ["revR"] and RP.held_retirements_unread("a1", [headless], [hold_h3]) == ["revR"]
    # standing_ids: the receipt-side ids plus the database's holds (none on a root with no database), each id once
    rd = _rd(tmp_path)
    RP.write_receipt(rd, w1)
    root = str(tmp_path / "root_without_db")
    os.makedirs(root)
    assert RP.standing_ids(root, rd, "a1") == [] and RP.last_receipt_for(rd, "a1") is not None
    cfg = _cfg_two(tmp_path, {"a1": {"shards": 2, "components": ["c1"]}})
    probed = {"components": {"c1": {"current": "11111111", "exists": True}}}
    standing = lambda a: RP.standing_unread(rd, a) + RP.held_retirements_unread(a, RP.receipts(rd), [hold])
    cs = RP.changed_set(cfg, ["a1"], probed, blobs, lambda a: RP.last_receipt_for(rd, a), standing=standing)
    assert cs["unchanged"] == [] and cs["changed"]["a1"] == ["refused-known id(s) left standing unread by the last pull: revR -- read fresh on this tree"]
    # once a window that had the hold in hand attempted the adapter, the same receipts and the same hold read the adapter as unchanged
    RP.write_receipt(rd, w3)
    cs3 = RP.changed_set(cfg, ["a1"], probed, blobs, lambda a: RP.last_receipt_for(rd, a), standing=standing)
    assert cs3["changed"] == {} and cs3["unchanged"] == ["a1"]


def test_a_hold_recorded_between_the_windows_merge_and_its_receipt_keeps_the_adapter_unread_because_the_receipt_names_the_holds_its_plan_saw(tmp_path):
    """Round 3 of the review of registry-hardening 3a (HIGH): the window merges a newly published retirement R, the independent harvester
    folds it and HOLDS it (the head moved), and the window then writes its otherwise whole receipt -- `pull_and_receipt` merges, then
    receipts, so the interleaving is an ordinary one. Round 2 ordered the hold's stamp against the receipt's, read the receipt's later
    stamp as "the shard ran after the hold", returned no standing id, and the unchanged pins and module parked the adapter for good while
    the vanished source's measurements stayed current. No stamp orders the two events -- the hold's `first_seen_utc` is the harvester's one
    clock reading per cycle, taken at the cycle's start and carried by a hold committed minutes later -- so what the receipt records is the
    fact itself: the holds its plan had in hand (`holds_seen`, built against the database after the reader-state export), and a standing
    hold is unread until an attempted receipt names it. The delivering receipt, stamped 200 ms after the hold, names nothing."""
    plan = {"window": "W1", "tree_sha": "a" * 40, "destination": "cluster_a",
            "adapters": {"a1": {"route": "cluster", "shards": 2, "pins": {"c1": "11111111"}, "holds_seen": []}}, "refusals": {},
            "stages": {"pull": {"summaries": {"a1": {"expected": 2, "present": 2, "ok": 2}}, "pulled_dirs": ["a1/revR"],
                                "reactivations": {}, "attestation": {"ok": True}, "barrier": None}}}
    blobs = {"a1": {"src/x.py": "b1"}}
    hold = {"revision_id": "revR", "source_id": "a1:s3", "first_seen_utc": "2026-10-04T10:00:00.100Z", "record_uri": "records/adapters/a1/revR/000001-source_revision.jsonl",
            "blocking_head": "revH2"}
    w1 = RP.receipt_from_plan(plan, {}, blobs, {"op_id": "W1", "owner": "window"}, now=lambda: "2026-10-04T10:00:00.300Z")
    e1 = w1["adapters"]["a1"]
    assert e1["read_whole"] is True and RP.attempted(e1) and w1["utc"] > hold["first_seen_utc"] and e1["holds_seen"] == []
    assert RP.held_retirements_unread("a1", [w1], [hold]) == ["revR"]
    # the next tick reads the adapter again, the whole receipt and the unchanged pins and module notwithstanding
    rd = _rd(tmp_path)
    RP.write_receipt(rd, w1)
    cfg = _cfg_two(tmp_path, {"a1": {"shards": 2, "components": ["c1"]}})
    probed = {"components": {"c1": {"current": "11111111", "exists": True}}}
    standing = lambda a: RP.standing_unread(rd, a) + RP.held_retirements_unread(a, RP.receipts(rd), [hold])
    cs = RP.changed_set(cfg, ["a1"], probed, blobs, lambda a: RP.last_receipt_for(rd, a), standing=standing)
    assert cs["unchanged"] == [] and cs["changed"]["a1"] == ["refused-known id(s) left standing unread by the last pull: revR -- read fresh on this tree"]
    # the window that follows is planned against a database carrying the hold: its receipt names it (the plan's `holds_seen`, copied by
    # receipt_from_plan) and, attempted, ends it -- the next tick reads the adapter as unchanged
    seen = [{"revision_id": "revR", "record_uri": hold["record_uri"], "blocking_head": "revH2"}]
    plan2 = dict(plan, window="W2", adapters={"a1": dict(plan["adapters"]["a1"], holds_seen=seen)},
                 holds_seen_from={"sidecar": "/x/rs.sqlite", "built_utc": "2026-10-04T10:20:00Z", "ledger_head": 41},
                 stages={"pull": dict(plan["stages"]["pull"], pulled_dirs=[], reactivations={"a1": {"revR": "reactivated"}})})
    w2 = RP.receipt_from_plan(plan2, {}, blobs, {"op_id": "W2", "owner": "window"}, now=lambda: "2026-10-04T10:30:00Z")
    assert w2["adapters"]["a1"]["holds_seen"] == seen and w2["adapters"]["a1"]["read_whole"] is True
    assert w2["holds_seen_from"] == plan2["holds_seen_from"] and w1["holds_seen_from"] is None            # the snapshot the holds were read from travels
    RP.write_receipt(rd, w2)
    cs2 = RP.changed_set(cfg, ["a1"], probed, blobs, lambda a: RP.last_receipt_for(rd, a), standing=standing)
    assert cs2["changed"] == {} and cs2["unchanged"] == ["a1"]
    # a receipt written before the rule (no `holds_seen` at all) saw nothing; a malformed entry in the list is skipped, not matched
    w0 = RP.receipt_from_plan(plan2, {}, blobs, {"op_id": "W0", "owner": "window"}, now=lambda: "2026-10-04T10:40:00Z")
    del w0["adapters"]["a1"]["holds_seen"]
    assert RP.held_retirements_unread("a1", [w0], [hold]) == ["revR"]
    w0["adapters"]["a1"]["holds_seen"] = ["revR", None, {"revision_id": "revR"}]
    assert RP.held_retirements_unread("a1", [w0], [hold]) == ["revR"]
    # receipt_from_plan copies the plan's list as the hold's identity (revision_id, record_uri, blocking_head) and nothing else
    plan3 = dict(plan2, adapters={"a1": dict(plan2["adapters"]["a1"], holds_seen=[{"revision_id": "revR", "record_uri": "u", "blocking_head": "revH2", "first_seen_utc": "x", "source_id": "s"}, "junk"])})
    w3 = RP.receipt_from_plan(plan3, {}, blobs, {"op_id": "W3", "owner": "window"}, now=lambda: "2026-10-04T10:50:00Z")
    assert w3["adapters"]["a1"]["holds_seen"] == [{"revision_id": "revR", "record_uri": "u", "blocking_head": "revH2"}]


def test_a_refused_standing_reactivation_leaves_the_receipt_incomplete_without_a_refusal_so_the_next_tick_reads_the_adapter_again(tmp_path):
    """An emit-time-refused id an OLDER shard returned as refused-known instead of reading it is `refused_standing`
    at the pull -- nothing written, no refusal of the adapter -- but its group was neither read nor committed, so the receipt must not
    claim `read_whole`: the entry carries the ids under `refused_standing`, `refused` stays None, `last_receipt_for` finds no whole receipt,
    and the changed set reads the adapter again (where this tree's shards read the group fresh). Before this the receipt read whole and the
    adapter was skipped while its pins and module stood, the standing refusal parked under a complete-looking receipt."""
    plan = {"window": "W1", "tree_sha": "a" * 40, "destination": "cluster_a",
            "adapters": {"a1": {"route": "cluster", "shards": 2, "pins": {"c1": "11111111"}}}, "refusals": {},
            "stages": {"pull": {"summaries": {"a1": {"expected": 2, "present": 2, "ok": 2}}, "pulled_dirs": [],
                                "reactivations": {"a1": {"revS": "refused_standing", "revY": "current"}}, "attestation": {"ok": True}, "barrier": None}}}
    blobs = {"a1": {"src/x.py": "b1"}}
    rec = RP.receipt_from_plan(plan, {}, blobs, {"op_id": "W1", "owner": "window"}, now=_now)
    a1 = rec["adapters"]["a1"]
    assert a1["read_whole"] is False and a1["shards_ok"] == 2 and a1["shards_total"] == 2 and a1["refused"] is None
    assert a1["refused_standing"] == ["revS"] and a1["reactivated"] == [] and rec["n_read_whole"] == 0 and rec["n_refused"] == 0
    rd = _rd(tmp_path)
    RP.write_receipt(rd, rec)
    assert RP.last_receipt_for(rd, "a1") is None, "a standing, unread id is no whole receipt"
    assert RP.standing_unread(rd, "a1") == ["revS"] and RP.standing_unread(rd, "a2") == []
    cfg = _cfg_two(tmp_path, {"a1": {"shards": 2, "components": ["c1"]}})
    probed = {"components": {"c1": {"current": "11111111", "exists": True}}}
    tick = lambda: RP.changed_set(cfg, ["a1"], probed, blobs, lambda a: RP.last_receipt_for(rd, a), standing=lambda a: RP.standing_unread(rd, a))
    cs = tick()
    assert cs["unchanged"] == [] and cs["changed"]["a1"][0] == "never read whole (no receipt)" and "left standing unread" in cs["changed"]["a1"][1]
    # an OLDER whole receipt with the SAME pins and module (a prior successful read, then an emit-time refusal with nothing moved):
    # last_receipt_for skips back to it, and without the standing check the adapter read as unchanged and the id stayed unread
    plan_ok = dict(plan, stages={"pull": dict(plan["stages"]["pull"], reactivations={"a1": {"revS": "reactivated", "revY": "current"}})})
    older = RP.receipt_from_plan(plan_ok, {}, blobs, {"op_id": "W0", "owner": "window"}, now=lambda: "2026-10-04T09:00:00Z")
    assert older["adapters"]["a1"]["read_whole"] is True and older["adapters"]["a1"]["refused_standing"] == [] and older["adapters"]["a1"]["reactivated"] == ["revS"]
    RP.write_receipt(rd, older)
    assert RP.last_receipt_for(rd, "a1")[0]["op"]["op_id"] == "W0" and RP.standing_unread(rd, "a1") == ["revS"]
    assert RP.changed_set(cfg, ["a1"], probed, blobs, lambda a: RP.last_receipt_for(rd, a))["unchanged"] == ["a1"]      # the gap, without it
    cs2 = tick()
    assert cs2["unchanged"] == [] and cs2["changed"]["a1"] == ["refused-known id(s) left standing unread by the last pull: revS -- read fresh on this tree"]
    # an INTERVENING window that never attempted the adapter must not erase the claim (round 3): routed off the cluster (a plan-time
    # refusal, shards_total 0) and, separately, a shard missing (shards_ok < shards_total) -- the id is still outstanding, the adapter read
    plan_off = {"window": "W2", "tree_sha": "a" * 40, "destination": "cluster_a", "adapters": {"a1": {"route": "grading_host", "shards": 0, "pins": {"c1": "11111111"}}},
                "refusals": {"a1": ["plan: no published generation for component(s): c1"]}, "stages": {"pull": {"summaries": {}, "pulled_dirs": [], "reactivations": {}}}}
    off = RP.receipt_from_plan(plan_off, {}, blobs, {"op_id": "W2", "owner": "tick"}, now=lambda: "2026-10-04T12:30:00Z")
    assert off["adapters"]["a1"]["shards_total"] == 0 and off["adapters"]["a1"]["refused_standing"] == [] and not RP.attempted(off["adapters"]["a1"])
    RP.write_receipt(rd, off)
    assert RP.standing_unread(rd, "a1") == ["revS"] and RP.last_receipt_for(rd, "a1")[0]["op"]["op_id"] == "W0"
    assert tick()["changed"]["a1"] == ["refused-known id(s) left standing unread by the last pull: revS -- read fresh on this tree"]
    short = RP.receipt_from_plan(dict(plan_ok, window="W2b", stages={"pull": dict(plan_ok["stages"]["pull"], summaries={"a1": {"expected": 2, "present": 1, "ok": 1}}, reactivations={})},
                                      refusals={"a1": ["1 of 2 shard summaries missing"]}), {}, blobs, {"op_id": "W2b", "owner": "tick"}, now=lambda: "2026-10-04T12:40:00Z")
    assert short["adapters"]["a1"]["shards_ok"] == 1 and not RP.attempted(short["adapters"]["a1"])
    RP.write_receipt(rd, short)
    assert RP.standing_unread(rd, "a1") == ["revS"] and tick()["unchanged"] == []
    # a window that ATTEMPTED the adapter (every shard ran) and REFUSED it by its reader resolves the claim: the id was read, not left standing
    tried = RP.receipt_from_plan(dict(plan_ok, window="W2c", stages={"pull": dict(plan_ok["stages"]["pull"], reactivations={})},
                                      refusals={"a1": ["pull: 1 group(s) refused by the reader on the cluster"]}), {}, blobs, {"op_id": "W2c", "owner": "tick"}, now=lambda: "2026-10-04T12:50:00Z")
    assert RP.attempted(tried["adapters"]["a1"]) and tried["adapters"]["a1"]["read_whole"] is False and tried["adapters"]["a1"]["refused_standing"] == []
    RP.write_receipt(rd, tried)
    assert RP.standing_unread(rd, "a1") == [] and tick()["unchanged"] == ["a1"]        # a refusal with nothing moved is not re-read: the standing design
    # ... and a later whole read likewise: the newest attempt carries no standing id -> unchanged
    newest = RP.receipt_from_plan(dict(plan_ok, window="W3"), {}, blobs, {"op_id": "W3", "owner": "tick"}, now=lambda: "2026-10-04T13:00:00Z")
    RP.write_receipt(rd, newest)
    assert RP.standing_unread(rd, "a1") == [] and tick()["unchanged"] == ["a1"]


# --------------------------------------------------------------------------------------------
# the tick, end to end over the cluster-read test world; then the four races
# --------------------------------------------------------------------------------------------


class FakeArray:
    """The fake sbatch (the cluster_reads runner shape) and watch: `attest.sbatch` runs `stores.attest_plan` in-process; `read_array.sbatch`
    records the plan; the WATCH is the array's execution (every shard run in-process between `_node` and `_back_on_grading_host`), then a verdict.
    ``before_watch`` / ``before_pull_verdict`` are hooks for the races."""

    def __init__(self, w, verdict="done", before_watch=None, run_shards=True, slurm=None, mark_on_watch=True):
        self.w, self.verdict, self.before_watch, self.run_shards = w, verdict, before_watch, run_shards
        self.job = 4240
        self.sbatch = []
        self.watched = []
        # the Slurm shim the fake sbatch registers its jobs in (RUNNING) and the fake watch marks terminal per its verdict (`mark_on_watch`),
        # as the real queue and accounting would: the tick's post-watch confirmation of every id reads them there
        self.slurm = slurm
        self.mark_on_watch = mark_on_watch
        self.submitted = []

    def runner(self, argv, capture_output=True, text=True, timeout=None, input=None, **kw):
        script = argv[-1] if argv and argv[0] == "bash" else None
        if script and "sbatch" in script:
            self.job += 1
            self.sbatch.append(script)
            parts = script.split()
            name = parts[parts.index("-J") + 1].strip("'") if "-J" in parts else f"job-{self.job}"
            self.submitted.append(str(self.job))
            if self.slurm is not None and hasattr(self.slurm, "jobs_"):
                self.slurm.jobs_.setdefault(str(self.job), {"name": name, "state": "RUNNING"})
            if "attest.sbatch" in script:
                i = next(k for k, a in enumerate(parts) if a.endswith("attest.sbatch"))
                window_root = parts[i + 1].strip("'")
                ST.attest_plan(f"{window_root}/plan.json", f"{window_root}/{ST.ATTESTATION_FILE}")
            else:
                i = next(k for k, a in enumerate(parts) if a.endswith("read_array.sbatch"))
                self.plan_json = f"{parts[i + 1].strip(chr(39))}/plan.json"
            return subprocess.CompletedProcess(argv, 0, stdout=f"{self.job}\n", stderr="")
        return subprocess.run(argv, capture_output=capture_output, text=text, timeout=timeout, input=input)

    def watch(self, job_id):
        self.watched.append(job_id)
        if self.run_shards:
            plan = CR._load_json(self.plan_json)
            _node(self.w)
            try:
                for t in plan["tasks"]:
                    assert CR.run_shard(self.plan_json, t["index"], home=self.w["home"]) == 0
            finally:
                _back_on_grading_host(self.w)
        if self.slurm is not None and self.mark_on_watch and hasattr(self.slurm, "jobs_"):
            final = {"done": "COMPLETED", "failed": "FAILED", "cancelled": "CANCELLED"}.get(self.verdict)
            if final:
                for jid in self.submitted:
                    j = self.slurm.jobs_.get(jid)
                    if j is not None and not RP.TERMINAL_RE.match(str(j["state"])):
                        j["state"] = final
        if self.before_watch:
            self.before_watch()
        return self.verdict, f"RUNNING {job_id} node=n1\nDONE {job_id} state=COMPLETED\n" if self.verdict == "done" else f"{self.verdict.upper()} {job_id}\n"


def _tick(w, tmp_path, fake, op_id, slurm=None, **kw):
    slurm = slurm or FakeSlurm()
    if getattr(fake, "slurm", None) is None:
        fake.slurm = slurm                                   # the fake sbatch and watch keep the shim's queue and accounting in step
    return RP.Tick(w["root"], w["cfg"], w["dest"], state_dir=str(tmp_path / "state"), adapters=[ADAPTER], repo_root=REPO_ROOT, op_id=op_id,
                   runner=fake.runner, slurm=slurm, watch=fake.watch, venv="/shared/envs/v", now=_now, log=lambda s: None,
                   sleep=lambda s: None, cancel_poll_s=0, uri_home=w["home"], blobs={ADAPTER: {"src/sisteer/registry/adapters/synthetic.py": "b1"}}, **kw)


def _probe_nothing_published(real=RP.probe):
    """A probe stub for a tick that must find nothing to read (every component unpublished) while the receipt step, which asks the probe
    for the pins' PUBLISHED records, still gets real answers."""
    def stub(cfg, dest, slugs, runner=None, tree=None, pins=None):
        out = {"components": {s: {"current": None, "exists": False} for s in slugs}, "tree_present": True, "utc": _now(), "pins": {}}
        if pins:
            out["pins"] = real(cfg, dest, [], runner=runner, pins=pins)["pins"]
        return out
    return stub


def _tree(w):
    """The base tree present on the (local) cluster, and the record-log mirror's rulings the stage links to (non-empty, as the stage requires)."""
    sha = subprocess.run(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    os.makedirs(os.path.join(w["shared"], "gates", "trees", sha, "src", "sisteer", "registry"), exist_ok=True)
    mirror = os.path.join(w["shared"], "registry", "log", "rulings")
    os.makedirs(mirror, exist_ok=True)
    open(os.path.join(mirror, "seed.jsonl"), "w").write("")
    return sha


def test_the_tick_reads_the_changed_adapter_end_to_end_writes_the_receipt_and_the_next_tick_finds_nothing_changed(tmp_path, monkeypatch, capsys):
    w = _world_with_generation(tmp_path, monkeypatch)
    s2 = _store(w["stores"], "s2", "legacy_store_2", "none", [0.1, 0.9, 0.5])
    res2 = ST.mirror_component(w["root"], w["dest"], w["cfg"].components["synthetic_stores"], runner=TS.Runner())
    assert res2.outcome == "published"
    sha = _tree(w)
    rd = _rd(tmp_path)
    fake = FakeArray(w)
    t = _tick(w, tmp_path, fake, "t1")
    rc = t.run()
    out = capsys.readouterr().out.strip().splitlines()[-1]
    summary = json.loads(out)
    assert rc == 0, summary
    assert summary["state"] == "receipted" and summary["reservation"] == "taken" and summary["changed"] == [ADAPTER] and summary["pulled"] == 1 and summary["refused"] == []
    assert summary["tree_sha"] == sha and summary["tree_present"] is True
    # the sequence: the attestation job then the array (afterok), one watch, the shards wrote their summaries, the pull delivered the revision
    assert len(fake.sbatch) == 2 and "attest.sbatch" in fake.sbatch[0] and "--dependency=afterok:4241" in fake.sbatch[1] and "-J reads-t1" in fake.sbatch[1]
    assert "-J attest-t1" in fake.sbatch[0] and fake.watched == ["4242"]
    op = RP.load_op(rd, "t1")
    assert op["state"] == "receipted" and op["jobs"] == {"attest": "4241", "reads": "4242"} and op["adapters"] == [ADAPTER] and op["routes"] == {ADAPTER: "cluster"}
    assert [h["state"] for h in op["history"]][:4] == ["reserved", "reconciled_others", "probed", "exported"]
    assert not os.path.exists(RP.reservation_path(rd)), "released"
    rec = json.load(open(RP.receipt_path(rd, "t1")))
    e = rec["adapters"][ADAPTER]
    assert e["read_whole"] and e["shards_total"] == 2 and e["shards_ok"] == 2 and e["pins"] == {"synthetic_stores": res2.content_sha8} and len(e["revisions_pulled"]) == 1
    assert e["published"]["synthetic_stores"]["published_utc"] and e["published"]["synthetic_stores"]["mode"] in ("mirror", "push")
    assert e["module_blobs"] == {"src/sisteer/registry/adapters/synthetic.py": "b1"} and rec["jobs"]["reads"] == "4242"
    # the revision reached records/ (inbox-merge) and the plan file carries the stages the window writes too
    rev_dir = e["revisions_pulled"][0]
    assert os.path.isdir(os.path.join(H.records_dir(w["root"]), "adapters", rev_dir))
    plan = CR._load_json(CR.plan_path(w["root"], "t1"))
    assert plan["op"] == {"owner": "tick", "op_id": "t1"} and set(plan["stages"]) >= {"stage", "submit", "revalidate_post", "pull", "inbox_merge"}
    assert plan["stages"]["submit"]["op"] == "t1" and plan["stages"]["pull"]["by"] == "t1" and plan["stages"]["pull"]["pulled_dirs"] == [rev_dir]
    dep = plan["stages"]["pull"]["depends_dirs"]                                  # what the receipt rests on, recorded at pull time
    assert set(dep) == {rev_dir} and dep[rev_dir]["how"] == "copied" and dep[rev_dir]["adapter"] == ADAPTER and len(dep[rev_dir]["files"]) >= 2
    assert all(len(sha) == 64 for sha in dep[rev_dir]["files"].values()) and any(H.REVISION_MARKER_TOKEN in n for n in dep[rev_dir]["files"])
    assert plan["stages"]["inbox_merge"]["evidence"] == {rev_dir: "merged"} and plan["stages"]["inbox_merge"]["problems"] == {} and "receipt_check" not in plan["stages"]
    assert RP.merge_evidence(w["root"], RP.depends_of_plan(plan))[rev_dir]["state"] == "merged"
    assert not os.path.exists(RP.sidecar_path(rd, "t1"))
    # the second tick: the pin and the blob are the receipt's -> nothing changed, nothing exported or submitted, exit 0 in one probe
    fake2 = FakeArray(w)
    exported = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(RS, "export", lambda *a, **k: exported.append(a) or pytest.fail("nothing to export"))
        t2 = _tick(w, tmp_path, fake2, "t2")
        assert t2.run() == 0
    s2_ = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert s2_["state"] == "unchanged" and s2_["unchanged"] == [ADAPTER] and fake2.sbatch == [] and exported == [] and not os.path.exists(RP.reservation_path(rd))
    assert RP.load_op(rd, "t2")["state"] == "unchanged"
    # a moved pin (a new generation after a store change) is a change: the third tick reads again
    _store(w["stores"], "s3", "legacy_store_3", "constitution_lora@aaaa", [0.7, 0.7, 0.2, 0.1])
    res3 = ST.mirror_component(w["root"], w["dest"], w["cfg"].components["synthetic_stores"], runner=TS.Runner())
    assert res3.outcome == "published" and res3.content_sha8 != res2.content_sha8
    fake3 = FakeArray(w)
    t3 = _tick(w, tmp_path, fake3, "t3")
    assert t3.run() == 0
    s3_ = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert s3_["state"] == "receipted" and s3_["changed"] == [ADAPTER] and len(fake3.sbatch) == 2
    assert RP.load_op(rd, "t3")["changed"][ADAPTER] == [f"pin moved: synthetic_stores {res2.content_sha8} -> {res3.content_sha8}"]
    rec3 = json.load(open(RP.receipt_path(rd, "t3")))
    assert rec3["adapters"][ADAPTER]["pins"] == {"synthetic_stores": res3.content_sha8} and rec3["adapters"][ADAPTER]["read_whole"]


def test_tick_dry_run_probes_and_prints_the_changed_set_without_exporting_or_submitting_and_releases(tmp_path, monkeypatch, capsys):
    w = _world_with_generation(tmp_path, monkeypatch)
    _tree(w)
    fake = FakeArray(w)
    monkeypatch.setattr(RS, "export", lambda *a, **k: pytest.fail("a dry run exports nothing"))
    t = _tick(w, tmp_path, fake, "d1", dry_run=True)
    assert t.run() == 0
    s = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert s["state"] == "dry_run" and s["changed"] == [ADAPTER] and fake.sbatch == [] and not os.path.exists(RP.reservation_path(_rd(tmp_path)))
    assert RP.load_op(_rd(tmp_path), "d1")["state"] == "dry_run"


def test_tick_yields_to_a_live_holder_without_probing_or_submitting(tmp_path, monkeypatch, capsys):
    w = _world_with_generation(tmp_path, monkeypatch)
    _tree(w)
    rd = _rd(tmp_path)
    RP.reserve(rd, "window", "W1", now=_now)
    fake = FakeArray(w)
    monkeypatch.setattr(RP, "probe", lambda *a, **k: pytest.fail("a yielding tick probes nothing"))
    t = _tick(w, tmp_path, fake, "t1")
    assert t.run() == 0
    s = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert s["state"] == "yielded" and s["held_by"]["op_id"] == "W1" and fake.sbatch == []
    assert RP.read_reservation(rd)["op_id"] == "W1" and RP.load_op(rd, "t1")["state"] == "yielded"


# -- the four races ----------------------------------------------------------------------------------------


def test_race_pre_submit_a_window_takes_over_while_the_tick_plans_the_tick_submits_nothing_and_exits_20(tmp_path, monkeypatch, capsys):
    """Race 1: the tick has reserved, probed and exported; the window takes over (cancelled_by recorded, the reservation replaced) BEFORE the
    tick's submit. The tick's pre-submit ownership check reads the loss: no sbatch, exit 20, state cancelled; the window's reservation stands."""
    w = _world_with_generation(tmp_path, monkeypatch)
    _tree(w)
    rd = _rd(tmp_path)
    fake = FakeArray(w)
    slurm = FakeSlurm()
    real_stage = CR.stage_window

    def stage_then_takeover(plan, sidecar, plan_file, **kw):
        info = real_stage(plan, sidecar, plan_file, **kw)
        how, _ = RP.take_reservation(rd, slurm, "window", "W1", takeover=True, now=_now, poll_s=0, sleep=lambda s: None)
        assert how == "taken_over"
        return info
    monkeypatch.setattr(CR, "stage_window", stage_then_takeover)
    t = _tick(w, tmp_path, fake, "t1", slurm=slurm)
    assert t.run() == RP.EXIT_CANCELLED
    s = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert s["state"] == "cancelled" and "cancelled by window W1" in s["why"] and fake.sbatch == [] and fake.watched == []
    assert RP.load_op(rd, "t1")["state"] == "cancelled" and RP.load_op(rd, "t1")["cancelled_by"] == "window W1"
    assert RP.read_reservation(rd)["op_id"] == "W1", "the tick's release never removes the window's reservation"
    assert slurm.cancelled == []                                                      # nothing of the tick's was ever queued


def test_race_a_takeover_arriving_during_the_submit_section_waits_for_the_recorded_job_ids_then_cancels_and_confirms_them(tmp_path, monkeypatch, capsys):
    """Race 1, the other interleaving: the tick's ownership check, both sbatch calls and the write of
    the job ids to the op record are ONE critical-section interval. The window arrives while the attestation's sbatch is "in flight" (a
    second thread, as it is a second process live): it waits on the section, lands only after the ids are recorded, and its hand-over cancels
    BOTH recorded jobs and confirms them terminal before the reservation changes hands. The tick then reads the loss (at its post-submit check
    or before its pull), cancels nothing new (both are terminal already) and exits 20. Before: the takeover could land between the check and
    the sbatch and finish its reconcile with no job to see; the array then ran against the window's namespace if the self-cancel failed."""
    w = _world_with_generation(tmp_path, monkeypatch)
    _tree(w)
    rd = _rd(tmp_path)
    slurm = FakeSlurm()
    marks = {}
    landed = threading.Event()

    def takeover():
        how, _ = RP.take_reservation(rd, slurm, "window", "W1", takeover=True, now=_now, poll_s=0, sleep=lambda s: None, section_poll_s=0.02)
        marks["takeover"] = (how, time.monotonic())
        landed.set()
    th = threading.Thread(target=takeover, daemon=True)
    fake = FakeArray(w, run_shards=False, before_watch=lambda: landed.wait(30), mark_on_watch=False)   # the watch returns once the takeover landed; the WINDOW's cancel ends the jobs
    real_attest, real_array = CR.submit_attest, CR.submit_array

    def attest_with_the_window_arriving(plan, **kw):
        th.start()                                                   # the window arrives now, inside the tick's submit section
        time.sleep(0.5)                                              # ... and blocks on it while the sbatch is "in flight"
        marks["landed_during_attest"] = landed.is_set()
        job = real_attest(plan, **kw)
        slurm.jobs_[job] = {"name": f"attest-{plan['window']}", "state": "RUNNING"}
        return job

    def array_recorded(plan, venv, name, **kw):
        job = real_array(plan, venv, name, **kw)
        slurm.jobs_[job] = {"name": name, "state": "PENDING"}
        marks["landed_during_array"] = landed.is_set()
        marks["submit_end"] = time.monotonic()
        return job
    monkeypatch.setattr(CR, "submit_attest", attest_with_the_window_arriving)
    monkeypatch.setattr(CR, "submit_array", array_recorded)
    monkeypatch.setattr(CR, "pull_window", lambda *a, **k: pytest.fail("nothing is pulled after the loss"))
    t = _tick(w, tmp_path, fake, "t1", slurm=slurm)
    rc = t.run()
    th.join(30)
    assert rc == RP.EXIT_CANCELLED and not th.is_alive(), (rc, marks)
    s = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert s["state"] == "cancelled" and s["submit_section_s"] >= 0.5, s                       # the hold covers the in-flight sbatch
    assert marks["landed_during_attest"] is False and marks["landed_during_array"] is False, marks   # the takeover waited the whole section out
    assert marks["takeover"][0] == "taken_over" and marks["takeover"][1] >= marks["submit_end"], marks
    op = RP.load_op(rd, "t1")
    assert op["jobs"] == {"attest": "4241", "reads": "4242"} and op["submit_section_s"] >= 0.5 and op["state"] == "cancelled"
    assert op["cancelled_by"] == "window W1" and op["jobs_terminal"] == {"4241": "CANCELLED", "4242": "CANCELLED"}
    assert slurm.cancelled[:2] == ["4241", "4242"], slurm.cancelled                                # the WINDOW's hand cancelled both recorded ids first
    assert slurm.jobs_["4241"]["state"] == "CANCELLED" and slurm.jobs_["4242"]["state"] == "CANCELLED"
    assert RP.read_reservation(rd)["op_id"] == "W1" and not os.path.exists(RP.receipt_path(rd, "t1"))
    # the window's reconcile finds nothing outstanding for t1
    rep = RP.reconcile(rd, slurm, "W1", now=_now, poll_s=0, sleep=lambda s: None)
    assert rep["cancelled"] == {} and "t1" not in rep["left_alive"]


def test_a_self_cancel_that_cannot_be_confirmed_is_fail_closed_exit_27_the_reservation_kept_and_the_next_taker_confirms_first(tmp_path, monkeypatch, capsys):
    """The fail-closed half: the array's sbatch fails after the attestation was queued (a transient ssh failure), and
    the tick's own scancel never reads terminal in the accounting. Before, `_cancel_own` swallowed the refusal into `jobs_terminal.error` and
    the tick ended as if settled. Now: the record says `cancel_failed` with the job ids and the error, the tick exits 27 (not 20, not 2: the
    unit is red), and the reservation is KEPT -- so a window's takeover (the holder alive, a tick) must cancel and confirm that job itself
    and is refused by name until the accounting agrees; a later tick finds the same holder stale (the process gone) and does the same."""
    w = _world_with_generation(tmp_path, monkeypatch)
    _tree(w)
    rd = _rd(tmp_path)
    slurm = FakeSlurm(cancel_lag=None)                                 # scancel is issued but never reads terminal
    fake = FakeArray(w)
    real_attest = CR.submit_attest

    def attest_recorded(plan, **kw):
        job = real_attest(plan, **kw)
        slurm.jobs_[job] = {"name": f"attest-{plan['window']}", "state": "RUNNING"}
        return job

    def array_sbatch_fails(*a, **k):
        raise CR.ClusterReadsError("sbatch read_array.sbatch failed (rc 255): ssh: connection reset by peer")
    monkeypatch.setattr(CR, "submit_attest", attest_recorded)
    monkeypatch.setattr(CR, "submit_array", array_sbatch_fails)
    t = _tick(w, tmp_path, fake, "t1", slurm=slurm, cancel_bound_s=0.01)
    assert t.run() == RP.EXIT_CANCEL_FAILED
    s = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert s["state"] == "cancel_failed" and s["reservation_kept"] is True and s["release"] == "kept" and "connection reset" in s["error"], s
    op = RP.load_op(rd, "t1")
    assert op["state"] == "cancel_failed" and op["jobs"] == {"attest": "4241"} and "error" in op["jobs_terminal"] and "4241" in op["cancel_error"]
    assert RP.read_reservation(rd)["op_id"] == "t1", "the reservation is KEPT over a job not proven terminal"
    assert slurm.cancelled == ["4241"] and slurm.jobs_["4241"]["state"] == "RUNNING" and fake.watched == []
    # a window's takeover: the holder is alive (this process) and a tick, so it may take over -- but only after cancelling and confirming the
    # recorded job itself; the accounting still says RUNNING -> refused by name, nothing of the window's written, the reservation unchanged
    with pytest.raises(RP.ReconcileRefusal) as ei:
        RP.take_reservation(rd, slurm, "window", "W1", takeover=True, now=_now, bound_s=0.01, poll_s=0, sleep=lambda x: None)
    assert "4241=RUNNING" in str(ei.value) and RP.read_reservation(rd)["op_id"] == "t1" and RP.load_op(rd, "W1") is None
    assert slurm.cancelled == ["4241", "4241"]                        # the window issued its own scancel
    # the accounting finally reads the cancel: the window's next attempt confirms it and takes over, the record carrying the terminal state
    slurm.pending_cancel.clear()
    slurm.jobs_["4241"]["state"] = "CANCELLED"
    how, _ = RP.take_reservation(rd, slurm, "window", "W1", takeover=True, now=_now, poll_s=0, sleep=lambda x: None)
    assert how == "taken_over" and RP.load_op(rd, "t1")["jobs_terminal"] == {"4241": "CANCELLED"} and RP.load_op(rd, "t1")["cancelled_by"] == "window W1"
    assert RP.read_reservation(rd)["op_id"] == "W1"
    # a later TICK meeting the kept reservation of a tick whose process is gone: stale -> the same cancel-and-confirm before the replacement
    RP.release(rd, "W1")
    RP.save_op(rd, dict(RP.new_op("tick", "t7", None, _now), state="cancel_failed", jobs={"attest": "7001"}, pid=DEAD_PID))
    RP.replace_reservation(rd, dict(RP._reservation("tick", "t7", None, _now), pid=DEAD_PID))
    slurm2 = FakeSlurm({"7001": {"name": "attest-t7", "state": "RUNNING"}}, cancel_lag=None)
    with pytest.raises(RP.ReconcileRefusal):
        RP.take_reservation(rd, slurm2, "tick", "t8", now=_now, bound_s=0.01, poll_s=0, sleep=lambda x: None)
    assert RP.read_reservation(rd)["op_id"] == "t7" and slurm2.cancelled == ["7001"]
    slurm2.pending_cancel.clear()
    slurm2.jobs_["7001"]["state"] = "CANCELLED"
    assert RP.take_reservation(rd, slurm2, "tick", "t8", now=_now, poll_s=0, sleep=lambda x: None)[0] == "replaced_stale"


def test_race_watcher_timeout_with_a_live_array_keeps_the_reservation_exits_21_and_the_next_operation_cancels_and_confirms_it_first(tmp_path, monkeypatch, capsys):
    """Race 2: the watcher times out while the array is still live. The tick records watch_timeout, exits
    21, pulls nothing -- and KEEPS the reservation (ownership is released only after every recorded job is confirmed terminal). A window
    arriving while the tick's process lives takes over the live tick: the array cancelled and confirmed FIRST, then the hand-over; a later
    tick meeting a kept reservation whose process is gone replaces the stale holder the same way. The reconcile's cancel of a handed-off
    record's jobs stays the belt for a reservation that is somehow gone."""
    w = _world_with_generation(tmp_path, monkeypatch)
    _tree(w)
    rd = _rd(tmp_path)
    slurm = FakeSlurm()
    fake = FakeArray(w, verdict="timeout", run_shards=False)
    real_array = CR.submit_array

    def array_recorded(plan, venv, name, **kw):
        job = real_array(plan, venv, name, **kw)
        slurm.jobs_[kw["dependency"]] = {"name": f"attest-{plan['window']}", "state": "COMPLETED"}      # the attestation ran: sacct has its row
        slurm.jobs_[job] = {"name": name, "state": "RUNNING"}
        return job
    pulled = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(CR, "submit_array", array_recorded)
        mp.setattr(CR, "pull_window", lambda *a, **k: pulled.append(1) or pytest.fail("nothing is pulled after a watcher timeout"))
        t = _tick(w, tmp_path, fake, "t1", slurm=slurm)
        assert t.run() == RP.EXIT_WATCH_TIMEOUT
    s = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert s["state"] == "watch_timeout" and s["array"] == "4242" and pulled == [] and s["reservation_kept"] is True and s["release"] == "kept", s
    op = RP.load_op(rd, "t1")
    assert op["state"] == "watch_timeout" and slurm.jobs_["4242"]["state"] == "RUNNING"
    assert RP.read_reservation(rd)["op_id"] == "t1", "the reservation is KEPT over the live array"
    assert RP.op_live(op) is False, "a handed-off operation is not live, whatever its process does"
    # a window arriving while the tick's process lives (this process): a live tick may be taken over -- the array cancelled and confirmed FIRST
    how, _ = RP.take_reservation(rd, slurm, "window", "W1", takeover=True, now=_now, poll_s=0, sleep=lambda x: None)
    assert how == "taken_over" and slurm.jobs_["4242"]["state"] == "CANCELLED" and slurm.cancelled == ["4242"]
    assert RP.load_op(rd, "t1")["state"] == "cancelled" and RP.load_op(rd, "t1")["cancelled_by"] == "window W1" and RP.load_op(rd, "t1")["jobs_terminal"]["4242"] == "CANCELLED"
    assert RP.release(rd, "W1")
    # a later tick meeting the kept reservation of a timed-out tick whose process is GONE: stale -> cancelled and confirmed, then replaced
    RP.save_op(rd, dict(RP.new_op("tick", "t5", None, _now), state="watch_timeout", jobs={"attest": "5001", "reads": "5002"}, pid=DEAD_PID))
    RP.replace_reservation(rd, dict(RP._reservation("tick", "t5", None, _now), pid=DEAD_PID))
    slurm.jobs_.update({"5001": {"name": "attest-t5", "state": "COMPLETED"}, "5002": {"name": "reads-t5", "state": "RUNNING"}})
    fake2 = FakeArray(w, run_shards=False)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(RP, "probe", _probe_nothing_published())
        t2 = _tick(w, tmp_path, fake2, "t2", slurm=slurm)
        assert t2.run() == 0
    s2 = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert s2["reservation"] == "replaced_stale" and slurm.jobs_["5002"]["state"] == "CANCELLED" and s2["reconcile"]["cancelled"] == {}
    assert RP.load_op(rd, "t5")["state"] == "cancelled" and RP.load_op(rd, "t5")["cancelled_by"] == "stale-replace tick t2"
    assert not os.path.exists(RP.reservation_path(rd)), "t2 had nothing unconfirmed: released"


def test_race_restart_after_submit_a_dead_ticks_completed_array_is_pulled_by_the_next_tick_and_a_running_one_cancelled(tmp_path, monkeypatch, capsys):
    """Race 3: the tick died right after its submit (the record has the job ids, the reservation is stale). (a) The array COMPLETED: the next
    tick replaces the stale reservation, finds the completed-unpulled operation and pulls it (receipt written under the dead op's id, `by` the
    new tick). (b) The array still RUNNING: cancelled and confirmed terminal, nothing pulled."""
    w = _world_with_generation(tmp_path, monkeypatch)
    _store(w["stores"], "s2", "legacy_store_2", "none", [0.1, 0.9, 0.5])
    assert ST.mirror_component(w["root"], w["dest"], w["cfg"].components["synthetic_stores"], runner=TS.Runner()).outcome == "published"
    _tree(w)
    rd = _rd(tmp_path)
    slurm = FakeSlurm()
    # the dead tick t1: run its steps up to and including the submit + the array's execution, then "die" (no pull, stale reservation, dead pid)
    fake = FakeArray(w)

    def die(*a, **k):
        raise Died("t1 dies before its pull")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(CR, "pull_window", die)
        t1 = _tick(w, tmp_path, fake, "t1", slurm=slurm)
        with pytest.raises(Died):
            t1.run()
    capsys.readouterr()
    assert fake.watched == ["4242"]                                                    # the array ran (the shards wrote their summaries)
    op = RP.load_op(rd, "t1")
    op.update(pid=DEAD_PID, state="watched")
    RP.save_op(rd, op)
    RP.replace_reservation(rd, dict(RP.read_reservation(rd) or RP._reservation("tick", "t1", None, _now), pid=DEAD_PID, owner="tick", op_id="t1"))
    slurm.jobs_.update({"4241": {"name": "attest-t1", "state": "COMPLETED"}, "4242": {"name": "reads-t1", "state": "COMPLETED"}})
    # (a) the next tick: the stale holder replaced, the completed array pulled now under t2, the receipt under t1
    fake2 = FakeArray(w, run_shards=False)
    monkeypatch.setattr(RP, "probe", _probe_nothing_published())
    t2 = _tick(w, tmp_path, fake2, "t2", slurm=slurm)
    assert t2.run() == 0
    s2 = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert s2["reservation"] == "replaced_stale" and s2["reconcile"]["pulled"]["t1"] == {"receipt": RP.receipt_path(rd, "t1"), "refusals": {}, "pulled": 1, "n_read_whole": 1}, s2["reconcile"]
    assert s2["reconcile"]["cancelled"] == {}
    rec = json.load(open(RP.receipt_path(rd, "t1")))
    assert rec["by"] == "t2" and rec["adapters"][ADAPTER]["read_whole"] and len(rec["adapters"][ADAPTER]["revisions_pulled"]) == 1
    assert RP.load_op(rd, "t1")["state"] == "receipted" and RP.load_op(rd, "t1")["receipted_by"] == "t2" and RP.load_op(rd, "t1")["stale_replaced_by"] == "tick t2"
    assert os.path.isdir(os.path.join(H.records_dir(w["root"]), "adapters", rec["adapters"][ADAPTER]["revisions_pulled"][0]))
    # (b) a dead tick whose array is still RUNNING: cancelled and confirmed, nothing pulled
    RP.save_op(rd, dict(RP.new_op("tick", "t5", None, _now), state="submitted", jobs={"attest": "5001", "reads": "5002"}, pid=DEAD_PID))
    RP.replace_reservation(rd, dict(RP._reservation("tick", "t5", None, _now), pid=DEAD_PID))
    slurm.jobs_.update({"5001": {"name": "attest-t5", "state": "COMPLETED"}, "5002": {"name": "reads-t5", "state": "RUNNING"}})
    t3 = _tick(w, tmp_path, FakeArray(w, run_shards=False), "t6", slurm=slurm)
    assert t3.run() == 0
    s3 = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert s3["reservation"] == "replaced_stale" and slurm.jobs_["5002"]["state"] == "CANCELLED" and RP.load_op(rd, "t5")["state"] == "cancelled"
    assert not os.path.exists(RP.receipt_path(rd, "t5"))


def test_a_transport_timeout_on_the_self_cancels_scancel_is_one_exception_family_and_fail_closed_with_the_reservation_kept(tmp_path, monkeypatch, capsys):
    """`Slurm._shell` ran `subprocess.run(..., timeout=120)`, so a stalled ssh scancel or query raised
    `subprocess.TimeoutExpired`, NOT a ReconcileRefusal -- it escaped `_cancel_own`'s handler, left `_keep_reservation` False and the finally
    unlinked the reservation over an unconfirmed job. Now `_shell` turns a transport timeout or failure into `SlurmQueryFailed` (a
    ReconcileRefusal: ONE family for every caller), `_cancel_own` sets the flag BEFORE any fallible work and catches everything, and the
    tick exits 27 with the reservation standing. The real `RP.Slurm` over a runner whose scancel stalls."""
    w = _world_with_generation(tmp_path, monkeypatch)
    _tree(w)
    rd = _rd(tmp_path)
    queue = "4241|RUNNING|attest-t1\n---SQUEUE-RC=0---\n---SACCT-RC=0---\n"

    def runner(argv, timeout_s, input=None):
        if "scancel" in argv[-1]:
            raise subprocess.TimeoutExpired(argv, timeout_s, output=b"", stderr=b"ssh: connect to host login-0: Connection timed out")
        return subprocess.CompletedProcess(argv, 0, stdout=queue, stderr="")
    slurm = RP.Slurm(None, runner=runner, user="u")
    with pytest.raises(RP.SlurmQueryFailed) as ei:                                    # the family: a timeout IS a SlurmQueryFailed
        slurm.cancel(["4241"])
    assert isinstance(ei.value, RP.ReconcileRefusal) and "timed out after 120 s" in ei.value.what and ei.value.rc == "timeout" and "Connection timed out" in str(ei.value)
    with pytest.raises(RP.SlurmQueryFailed) as ei2:                                   # ... and so is a transport that cannot start
        RP.Slurm(None, runner=lambda argv, t, input=None: (_ for _ in ()).throw(FileNotFoundError("ssh")), user="u").jobs()
    assert "the transport failed: FileNotFoundError" in ei2.value.what
    # the tick: the array's sbatch fails after the attestation was queued; its self-cancel's scancel STALLS -> fail-closed
    fake = FakeArray(w)
    real_attest = CR.submit_attest
    monkeypatch.setattr(CR, "submit_attest", lambda plan, **kw: real_attest(plan, **kw))

    def array_sbatch_fails(*a, **k):
        raise CR.ClusterReadsError("sbatch read_array.sbatch failed (rc 255): ssh: connection reset by peer")
    monkeypatch.setattr(CR, "submit_array", array_sbatch_fails)
    t = _tick(w, tmp_path, fake, "t1", slurm=slurm, cancel_bound_s=0.01)
    assert t.run() == RP.EXIT_CANCEL_FAILED
    s = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert s["state"] == "cancel_failed" and s["reservation_kept"] is True and s["release"] == "kept", s
    op = RP.load_op(rd, "t1")
    assert op["state"] == "cancel_failed" and op["jobs"] == {"attest": "4241"} and "timed out" in op["cancel_error"] and "SlurmQueryFailed" in op["jobs_terminal"]["error"]
    assert RP.read_reservation(rd)["op_id"] == "t1", "the reservation is KEPT over the unconfirmed attestation job"


def test_a_failing_journal_write_inside_the_fail_closed_path_changes_no_outcome_exit_27_and_the_reservation_kept(tmp_path, monkeypatch, capsys):
    """The second path: `_op(state="cancel_failed", ...)` could itself raise (disk full) BEFORE the flag was set, so
    the finally released the reservation. Now the flag is set before any fallible work and every record write in the fail-closed path is a
    best-effort journal: an OSError from `update_op` is logged (`journal_error` in the summary), the tick still exits 27 and the reservation
    still stands; the record keeps whatever the last successful write said."""
    w = _world_with_generation(tmp_path, monkeypatch)
    _tree(w)
    rd = _rd(tmp_path)
    slurm = FakeSlurm(cancel_lag=None)
    fake = FakeArray(w)
    real_attest = CR.submit_attest
    monkeypatch.setattr(CR, "submit_attest", lambda plan, **kw: real_attest(plan, **kw))

    def array_sbatch_fails(*a, **k):
        raise CR.ClusterReadsError("sbatch read_array.sbatch failed (rc 255): ssh: connection reset by peer")
    monkeypatch.setattr(CR, "submit_array", array_sbatch_fails)
    real_update = RP.update_op
    disk_full = {"on": False}

    def update_or_disk_full(rd_, op_id, now=RP.R.utc_now, **fields):
        if fields.get("state") == "cancel_failed":
            disk_full["on"] = True                             # from the fail-closed write on, every journal write fails
        if disk_full["on"]:
            raise OSError(28, "No space left on device")
        return real_update(rd_, op_id, now=now, **fields)
    monkeypatch.setattr(RP, "update_op", update_or_disk_full)
    t = _tick(w, tmp_path, fake, "t1", slurm=slurm, cancel_bound_s=0.01)
    assert t.run() == RP.EXIT_CANCEL_FAILED
    s = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert s["state"] == "cancel_failed" and s["reservation_kept"] is True and s["release"] == "kept" and s["journal_error"].startswith("OSError"), s
    assert RP.read_reservation(rd)["op_id"] == "t1", "the reservation is KEPT though the record could not say why"
    op = RP.load_op(rd, "t1")
    assert op["jobs"] == {"attest": "4241"} and op["state"] != "cancel_failed"       # the last successful write stands; the outcome did not depend on it
    assert slurm.cancelled == ["4241"]


def test_an_exception_the_code_did_not_foresee_after_the_submit_keeps_the_reservation_and_the_record_says_the_cancel_was_not_attempted(tmp_path, monkeypatch, capsys):
    """Whatever ends the tick after a job id exists -- here a BaseException from inside the watch -- the reservation stands
    (the flag has been True since the first recorded id) and the record says `cancel_failed` with the cancel not attempted (the process is
    ending); the exception goes on. A window's takeover then cancels and confirms both recorded jobs before it takes the reservation."""
    w = _world_with_generation(tmp_path, monkeypatch)
    _tree(w)
    rd = _rd(tmp_path)
    slurm = FakeSlurm()

    def die():
        raise Died("the process is killed while the array runs")
    fake = FakeArray(w, run_shards=False, before_watch=die, mark_on_watch=False)
    t = _tick(w, tmp_path, fake, "t1", slurm=slurm)
    with pytest.raises(Died):
        t.run()
    s = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert s["state"] == "cancel_failed" and s["reservation_kept"] is True and s["release"] == "kept" and s["why"] == "Died", s
    op = RP.load_op(rd, "t1")
    assert op["state"] == "cancel_failed" and op["jobs"] == {"attest": "4241", "reads": "4242"} and "not attempted" in op["cancel_error"]
    assert RP.read_reservation(rd)["op_id"] == "t1" and slurm.cancelled == [] and slurm.jobs_["4242"]["state"] == "RUNNING"
    how, _ = RP.take_reservation(rd, slurm, "window", "W1", takeover=True, now=_now, poll_s=0, sleep=lambda x: None)
    assert how == "taken_over" and sorted(slurm.cancelled) == ["4241", "4242"]
    assert RP.load_op(rd, "t1")["jobs_terminal"] == {"4241": "CANCELLED", "4242": "CANCELLED"} and RP.read_reservation(rd)["op_id"] == "W1"


def test_a_marker_less_or_conflicting_own_revision_at_the_merge_refuses_the_adapter_in_the_receipt_and_never_claims_read_whole(tmp_path, monkeypatch, capsys):
    """The I7 tightening: the merge recorded a stage even when `scan_inbox` reported `incomplete` or `conflict` for a revision
    THIS operation pulled, so the receipt claimed `read_whole` over records still in the inbox. Now the merge's outcomes are matched against
    the operation's `pulled_dirs`: a match refuses the adapter in the receipt (`merge_incomplete` / `merge_conflict`), `read_whole` is false,
    the tick exits 15 and the changed set reads the adapter again. A marker-less OWN revision cannot happen on the normal path (the verified
    copy is installed by one whole rename), so it is simulated: the marker removed between the pull and the merge; then a conflict."""
    w = _world_with_generation(tmp_path, monkeypatch)
    _store(w["stores"], "s2", "legacy_store_2", "none", [0.1, 0.9, 0.5])
    assert ST.mirror_component(w["root"], w["dest"], w["cfg"].components["synthetic_stores"], runner=TS.Runner()).outcome == "published"
    _tree(w)
    rd = _rd(tmp_path)
    real_scan = H.scan_inbox
    aside = tmp_path / "marker-aside"
    aside.mkdir()
    seen = {}

    def tamper_then_scan(root):
        import glob
        import shutil
        dirs = glob.glob(os.path.join(H.inbox_dir(root), "*", "adapters", ADAPTER, "*"))
        assert len(dirs) == 1, dirs
        markers = [f for f in os.listdir(dirs[0]) if H.REVISION_MARKER_TOKEN in f]
        assert markers, os.listdir(dirs[0])
        for m in markers:
            shutil.move(os.path.join(dirs[0], m), str(aside / m))          # the own revision loses its marker: a partial delivery to the merge
        seen["dir"] = dirs[0]
        return real_scan(root)
    monkeypatch.setattr(H, "scan_inbox", tamper_then_scan)
    t = _tick(w, tmp_path, FakeArray(w), "t1")
    assert t.run() == RP.EXIT_REFUSED
    s = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert s["state"] == "receipted" and s["refused"] == [ADAPTER] and s["pulled"] == 1, s
    rec = json.load(open(RP.receipt_path(rd, "t1")))
    e = rec["adapters"][ADAPTER]
    rev = e["revisions_pulled"][0]
    assert e["read_whole"] is False and e["shards_ok"] == 2 and len(e["refused"]) == 1 and e["refused"][0].startswith(f"merge_incomplete: {rev} ("), e
    assert "still in the inbox" in e["refused"][0] and "missing from records" in e["refused"][0]
    assert rec["n_read_whole"] == 0 and rec["n_refused"] == 1
    plan = CR._load_json(CR.plan_path(w["root"], "t1"))
    assert plan["stages"]["inbox_merge"]["incomplete"] == 1 and plan["stages"]["inbox_merge"]["evidence"] == {rev: "incomplete"}
    assert list(plan["stages"]["inbox_merge"]["problems"]) == [ADAPTER] and plan["refusals"][ADAPTER] == e["refused"]
    assert RP.load_op(rd, "t1")["stages"]["inbox_merge"]["evidence"] == {rev: "incomplete"}
    assert os.path.isdir(seen["dir"]) and not os.path.isdir(os.path.join(H.records_dir(w["root"]), "adapters", rev))      # still in the inbox
    assert RP.last_receipt_for(rd, ADAPTER) is None, "a refused entry is no whole receipt: the changed set reads the adapter again"
    monkeypatch.setattr(H, "scan_inbox", real_scan)
    # a CONFLICT: the marker restored (the revision complete again), a same-named record file already in records/ with other bytes ->
    # scan_inbox keeps the inbox copy as .conflict-<sha8> and reports it; the operation's adapter is refused for it
    import shutil
    for m in os.listdir(aside):
        shutil.move(str(aside / m), os.path.join(seen["dir"], m))
    files = sorted(f for f in os.listdir(seen["dir"]) if f.endswith(".jsonl") and H.REVISION_MARKER_TOKEN not in f) or sorted(f for f in os.listdir(seen["dir"]) if f.endswith(".jsonl"))
    assert files, os.listdir(seen["dir"])
    dest_dir = os.path.join(H.records_dir(w["root"]), "adapters", rev)
    os.makedirs(dest_dir, exist_ok=True)
    with open(os.path.join(dest_dir, files[0]), "w", encoding="utf-8") as fh:
        fh.write('{"other": "bytes"}\n')
    depends = {rev: {"adapter": ADAPTER, "how": "copied", "files": {f: RP.R.file_sha256(os.path.join(seen["dir"], f)) for f in os.listdir(seen["dir"])}}}
    merged = real_scan(w["root"])
    statuses = {st for _, st in merged}
    assert "conflict" in statuses, merged
    conflict_name = os.path.basename(next(p for p, st in merged if st == "conflict"))
    ev = RP.merge_evidence(w["root"], depends)                    # the evidence is read from records/, not from the scan's return
    assert ev[rev]["state"] == "conflict" and ev[rev]["problems"] == [f"{files[0]}: {conflict_name}"], ev
    problems = RP.merge_problems(ev)
    assert problems == {ADAPTER: [f"merge_conflict: {rev} ({files[0]}: {conflict_name})"]}, problems
    assert RP.merge_evidence(w["root"], {"other_adapter/deadbeefdeadbeef": {"adapter": "other_adapter", "how": "copied", "files": {}}}) == \
        {"other_adapter/deadbeefdeadbeef": {"state": "incomplete", "adapter": "other_adapter", "how": "copied", "problems": ["no records directory"]}}
    plan_doc = {"stages": {"pull": {"pulled_dirs": [rev], "depends_dirs": depends}}, "refusals": {}, "adapters": {ADAPTER: {"route": "cluster", "pins": {}}}}
    applied = RP.apply_merge_problems(plan_doc, RP.merge_evidence(w["root"], RP.depends_of_plan(plan_doc)))
    assert applied == problems and plan_doc["refusals"][ADAPTER] == problems[ADAPTER]
    rec2 = RP.receipt_from_plan(dict(plan_doc, stages={"pull": {"pulled_dirs": [rev], "depends_dirs": depends, "summaries": {ADAPTER: {"expected": 2, "ok": 2}}}}), {}, {}, {"op_id": "x", "owner": "tick"}, now=_now)
    assert rec2["adapters"][ADAPTER]["read_whole"] is False and rec2["adapters"][ADAPTER]["refused"] == problems[ADAPTER] and rec2["n_refused"] == 1


def _conflicting_record_in_records(root, inbox_rev_dir):
    """A same-named accepted record file already in records/ with OTHER bytes, for one non-marker file of the revision as it sits in the
    inbox: scan_inbox will keep the inbox copy aside as `<file>.conflict-<sha8>` and report it. Returns (rev, file name)."""
    rev = "/".join(inbox_rev_dir.rstrip("/").split("/")[-2:])
    names = sorted(f for f in os.listdir(inbox_rev_dir) if f.endswith(".jsonl") and H.REVISION_MARKER_TOKEN not in f) or \
        sorted(f for f in os.listdir(inbox_rev_dir) if f.endswith(".jsonl"))
    dest_dir = os.path.join(H.records_dir(root), "adapters", rev)
    os.makedirs(dest_dir, exist_ok=True)
    with open(os.path.join(dest_dir, names[0]), "w", encoding="utf-8") as fh:
        fh.write('{"other": "bytes"}\n')
    return rev, names[0]


def _own_inbox_dir(root):
    import glob
    dirs = glob.glob(os.path.join(H.inbox_dir(root), "*", "adapters", ADAPTER, "*"))
    assert len(dirs) == 1, dirs
    return dirs[0]


def test_a_conflict_moved_aside_by_an_interrupted_merge_is_rediscovered_from_records_by_the_resumed_merge_and_refuses_the_receipt(tmp_path, monkeypatch, capsys):
    """scan_inbox moves a conflicting file to records/.../<file>.conflict-<sha8> BEFORE returning its `conflict`; a process
    interrupted after that move and before the inbox_merge stage is persisted left no trace of the conflict for the resumed merge, whose
    scan found an empty inbox and whose receipt then read the adapter whole. Now the receipt rests on `merge_evidence`, read from records/
    against the pull stage's `depends_dirs` (name -> sha256): the resumed merge's scan merges nothing, the evidence still finds the
    `.conflict-` sibling and other bytes under the file's name, the adapter is refused (`merge_conflict`) and no whole receipt is written."""
    w = _world_with_generation(tmp_path, monkeypatch)
    _store(w["stores"], "s2", "legacy_store_2", "none", [0.1, 0.9, 0.5])
    assert ST.mirror_component(w["root"], w["dest"], w["cfg"].components["synthetic_stores"], runner=TS.Runner()).outcome == "published"
    _tree(w)
    rd = _rd(tmp_path)
    slurm = FakeSlurm()
    real_scan = H.scan_inbox
    seen = {}

    def conflict_then_scan_then_die(root):
        seen["rev"], seen["file"] = _conflicting_record_in_records(root, _own_inbox_dir(root))   # after the export (records/ had no such dir), before the merge
        out = real_scan(root)                                                                   # the conflict file moved aside, the rest merged
        seen["scan"] = {st for _, st in out}
        raise Died("interrupted after os.replace(src, conflict) and before the inbox_merge stage is written")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(H, "scan_inbox", conflict_then_scan_then_die)
        t1 = _tick(w, tmp_path, FakeArray(w), "t1", slurm=slurm)
        with pytest.raises(Died):
            t1.run()
    capsys.readouterr()
    rev, name = seen["rev"], seen["file"]
    assert "conflict" in seen["scan"] and "merged" in seen["scan"]
    plan = CR._load_json(CR.plan_path(w["root"], "t1"))
    assert "pull" in plan["stages"] and "inbox_merge" not in plan["stages"] and set(plan["stages"]["pull"]["depends_dirs"]) == {rev}
    dest_dir = os.path.join(H.records_dir(w["root"]), "adapters", rev)
    assert any(fn.startswith(name + ".conflict-") for fn in os.listdir(dest_dir)) and not os.path.isdir(os.path.join(H.inbox_dir(w["root"]), CR.INBOX_NAME, "adapters", rev))
    # the process died: a stale holder with a completed array; the next tick's reconcile RESUMES at the merge -- its scan finds nothing to merge
    op = RP.load_op(rd, "t1")
    op.update(pid=DEAD_PID)
    RP.save_op(rd, op)
    RP.replace_reservation(rd, dict(RP._reservation("tick", "t1", None, _now), pid=DEAD_PID))
    scans = []
    monkeypatch.setattr(H, "scan_inbox", lambda root: scans.append(real_scan(root)) or scans[-1])
    monkeypatch.setattr(RP, "probe", _probe_nothing_published())
    t2 = _tick(w, tmp_path, FakeArray(w, run_shards=False), "t2", slurm=slurm)
    assert t2.run() == 0
    s2 = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert s2["reconcile"]["pulled"]["t1"]["n_read_whole"] == 0 and s2["reconcile"]["pulled"]["t1"]["refusals"], s2["reconcile"]
    assert len(scans) == 1 and {st for _, st in scans[0]} == set(), "the resumed scan found nothing: the evidence did not come from it"
    plan2 = CR._load_json(CR.plan_path(w["root"], "t1"))
    im = plan2["stages"]["inbox_merge"]
    assert im["merged"] == 0 and im["conflict"] == 0 and im["evidence"] == {rev: "conflict"} and im["by"] == "t2"
    assert plan2["refusals"][ADAPTER] and plan2["refusals"][ADAPTER][0].startswith(f"merge_conflict: {rev} ({name}: {name}.conflict-"), plan2["refusals"]
    rec = json.load(open(RP.receipt_path(rd, "t1")))
    assert rec["adapters"][ADAPTER]["read_whole"] is False and rec["adapters"][ADAPTER]["shards_ok"] == 2 and rec["n_refused"] == 1 and rec["by"] == "t2"
    assert RP.last_receipt_for(rd, ADAPTER) is None, "no whole receipt: the changed set reads the adapter again"
    assert RP.load_op(rd, "t1")["state"] == "receipted" and RP.load_op(rd, "t1")["stages"]["inbox_merge"]["evidence"] == {rev: "conflict"}


def test_an_earlier_complete_delivery_reused_by_this_operation_is_in_its_depends_and_a_conflicting_accepted_record_refuses_it(tmp_path, monkeypatch, capsys):
    """pull_window reports an expected, marker-bearing directory already in the inbox as `complete (an earlier delivery
    already in the inbox)` and does not re-pull it, but `pulled_dirs` holds only state exactly `complete`, so a conflict on that reused
    revision was nobody's problem and a whole receipt followed. Now the pull records every revision the operation DEPENDS on
    (`depends_dirs`, the reused one with the inbox copy's own sha256s) and the evidence covers it. The actual path: a first tick whose merge
    never ran (the delivery stays in the inbox, marker-bearing; its receipt refused `merge_incomplete`), then a second tick that reads the
    adapter again, finds the earlier delivery, reuses it, and meets a conflicting accepted record in records/: refused `merge_conflict`."""
    w = _world_with_generation(tmp_path, monkeypatch)
    _store(w["stores"], "s2", "legacy_store_2", "none", [0.1, 0.9, 0.5])
    assert ST.mirror_component(w["root"], w["dest"], w["cfg"].components["synthetic_stores"], runner=TS.Runner()).outcome == "published"
    _tree(w)
    rd = _rd(tmp_path)
    slurm = FakeSlurm()
    real_scan = H.scan_inbox
    # tick 1: the merge is a no-op (the operator's standing inbox-merge never ran this time): the delivery stays in the inbox with its marker
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(H, "scan_inbox", lambda root: [])
        t1 = _tick(w, tmp_path, FakeArray(w), "t1", slurm=slurm)
        assert t1.run() == RP.EXIT_REFUSED
    s1 = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert s1["refused"] == [ADAPTER], s1
    plan1 = CR._load_json(CR.plan_path(w["root"], "t1"))
    rev = plan1["stages"]["pull"]["pulled_dirs"][0]
    assert plan1["stages"]["pull"]["depends_dirs"][rev]["how"] == "copied" and plan1["stages"]["inbox_merge"]["evidence"] == {rev: "incomplete"}
    inbox_dir = _own_inbox_dir(w["root"])
    assert inbox_dir.endswith(rev) and any(H.REVISION_MARKER_TOKEN in f for f in os.listdir(inbox_dir)) and RP.last_receipt_for(rd, ADAPTER) is None
    assert not os.path.isdir(os.path.join(H.records_dir(w["root"]), "adapters", rev))
    # tick 2 reads the adapter again (no whole receipt); its shards mint the same id; the pull finds the earlier delivery and REUSES it.
    # The conflicting accepted record lands AFTER the reader-state export (records/ must not hold the id when the shards run) and before
    # the pull: a `pull_window` wrapper
    real_pull = CR.pull_window
    seen = {}

    def conflict_then_pull(*a, **k):
        seen["rev"], seen["file"] = _conflicting_record_in_records(w["root"], _own_inbox_dir(w["root"]))
        rep = real_pull(*a, **k)
        seen["pulled"] = {r: v["state"] for r, v in rep["pulled"].items()}
        seen["depends"] = dict(rep["depends"])
        return rep
    monkeypatch.setattr(CR, "pull_window", conflict_then_pull)
    t2 = _tick(w, tmp_path, FakeArray(w), "t2", slurm=slurm)
    assert t2.run() == RP.EXIT_REFUSED
    s2 = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert s2["changed"] == [ADAPTER] and s2["refused"] == [ADAPTER], s2
    assert seen["rev"] == rev and seen["pulled"] == {rev: "complete (an earlier delivery already in the inbox)"}, seen
    assert set(seen["depends"]) == {rev} and seen["depends"][rev]["how"] == "reused" and seen["file"] in seen["depends"][rev]["files"]
    plan2 = CR._load_json(CR.plan_path(w["root"], "t2"))
    assert plan2["stages"]["pull"]["pulled_dirs"] == [], "the reused delivery is not in pulled_dirs (state is not exactly complete) ..."
    assert set(plan2["stages"]["pull"]["depends_dirs"]) == {rev}, "... but it IS what this operation depends on"
    assert plan2["stages"]["inbox_merge"]["conflict"] == 1 and plan2["stages"]["inbox_merge"]["evidence"] == {rev: "conflict"}
    assert plan2["refusals"][ADAPTER][0].startswith(f"merge_conflict: {rev} ({seen['file']}: {seen['file']}.conflict-"), plan2["refusals"]
    rec2 = json.load(open(RP.receipt_path(rd, "t2")))
    e = rec2["adapters"][ADAPTER]
    assert e["read_whole"] is False and e["shards_ok"] == 2 and e["revisions_pulled"] == [] and e["refused"] == plan2["refusals"][ADAPTER]
    assert RP.last_receipt_for(rd, ADAPTER) is None
    # the inbox copy is in records/ beside the conflict (the other files merged), nothing of it left in the inbox
    dest_dir = os.path.join(H.records_dir(w["root"]), "adapters", rev)
    assert any(fn.startswith(seen["file"] + ".conflict-") for fn in os.listdir(dest_dir)) and not os.path.isdir(inbox_dir)


def test_merge_evidence_reads_durable_records_only_and_passes_only_when_every_recorded_file_is_present_with_its_sha(tmp_path):
    """The durable-merge check. Over a temp root: nothing in records/ -> incomplete (every file missing); every recorded file
    present with its sha256 and no inbox directory -> merged; one file with other bytes -> conflict; a `.conflict-<sha8>` sibling -> conflict
    (named); the directory still in the inbox -> incomplete even with records/ complete; an older plan with no file list -> merged iff the
    records/ directory exists and no inbox directory does; `depends_of_plan` reads `depends_dirs`, else falls back to `pulled_dirs`."""
    root = str(tmp_path / "root")
    H.ensure_layout(root)
    rev = "syn/abcdef0123456789"
    marker = "r_source_revision_abcdef01.jsonl"
    body = {"a.jsonl": b'{"x": 1}\n', marker: b'{"source_revision": "abcdef0123456789"}\n'}
    import hashlib
    depends = {rev: {"adapter": "syn", "how": "copied", "files": {n: hashlib.sha256(b).hexdigest() for n, b in body.items()}}}
    ev = RP.merge_evidence(root, depends)
    assert ev[rev]["state"] == "incomplete" and ev[rev]["problems"] == ["a.jsonl: missing from records", f"{marker}: missing from records"], ev
    assert RP.merge_problems(ev) == {"syn": [f"merge_incomplete: {rev} (a.jsonl: missing from records; {marker}: missing from records)"]}
    dest = os.path.join(H.records_dir(root), "adapters", rev)
    os.makedirs(dest)
    for n, b in body.items():
        open(os.path.join(dest, n), "wb").write(b)
    assert RP.merge_evidence(root, depends)[rev] == {"state": "merged", "adapter": "syn", "how": "copied", "problems": []}
    assert RP.merge_problems(RP.merge_evidence(root, depends)) == {}
    # other bytes under the file's name, no sibling -> conflict; a sibling -> conflict named with the sibling
    open(os.path.join(dest, "a.jsonl"), "wb").write(b'{"x": 2}\n')
    ev = RP.merge_evidence(root, depends)
    assert ev[rev]["state"] == "conflict" and ev[rev]["problems"][0].startswith("a.jsonl: other bytes in records: a.jsonl (sha ")
    open(os.path.join(dest, "a.jsonl.conflict-0badcafe"), "wb").write(body["a.jsonl"])
    ev = RP.merge_evidence(root, depends)
    assert ev[rev]["state"] == "conflict" and ev[rev]["problems"] == ["a.jsonl: a.jsonl.conflict-0badcafe"]
    # back to merged; the directory still in the inbox -> incomplete even so
    open(os.path.join(dest, "a.jsonl"), "wb").write(body["a.jsonl"])
    os.remove(os.path.join(dest, "a.jsonl.conflict-0badcafe"))
    assert RP.merge_evidence(root, depends)[rev]["state"] == "merged"
    stray = os.path.join(H.inbox_dir(root), "cluster_a", "adapters", rev)
    os.makedirs(stray)
    ev = RP.merge_evidence(root, depends)
    assert ev[rev]["state"] == "incomplete" and ev[rev]["problems"] == ["still in the inbox"]
    os.rmdir(stray)
    # an older plan: no file list -> the directory's existence and the inbox's silence
    old = {"stages": {"pull": {"pulled_dirs": [rev, "syn/ffffffffffffffff"]}}}
    dep = RP.depends_of_plan(old)
    assert dep == {rev: {"adapter": "syn", "how": "copied", "files": {}}, "syn/ffffffffffffffff": {"adapter": "syn", "how": "copied", "files": {}}}
    ev = RP.merge_evidence(root, dep)
    assert ev[rev]["state"] == "merged" and ev["syn/ffffffffffffffff"] == {"state": "incomplete", "adapter": "syn", "how": "copied", "problems": ["no records directory"]}
    new = {"stages": {"pull": {"pulled_dirs": [], "depends_dirs": depends}}}
    assert RP.depends_of_plan(new) == depends
    # a sibling beside a CORRECT file is no conflict of ours (records holds our bytes; the sibling is another delivery's): merged
    open(os.path.join(dest, "a.jsonl.conflict-0badcafe"), "wb").write(b"someone else's bytes")
    assert RP.merge_evidence(root, depends)[rev]["state"] == "merged"
    os.remove(os.path.join(dest, "a.jsonl.conflict-0badcafe"))
    # the receipt-time re-check: a conflict appearing after a clean merge stage (records now holds OTHER bytes under our file's name and our
    # copy sits beside it as the .conflict- sibling, exactly what scan_inbox leaves) is still refused at the receipt
    plan_doc = {"schema": CR.PLAN_SCHEMA, "window": "x9", "adapters": {"syn": {"route": "cluster", "pins": {}}}, "refusals": {}, "op": {"owner": "tick"},
                "stages": {"revalidate_post": {}, "pull": {"pulled_dirs": [rev], "depends_dirs": depends, "summaries": {"syn": {"expected": 1, "ok": 1}}},
                           "inbox_merge": {"evidence": {rev: "merged"}, "problems": {}}}}
    pf = CR.plan_path(root, "x9")
    CR._write_json(pf, plan_doc)
    rd = _rd(tmp_path)
    RP.save_op(rd, dict(RP.new_op("tick", "x9", None, _now), state="merged"))
    open(os.path.join(dest, "a.jsonl"), "wb").write(b'{"x": 3}\n')
    open(os.path.join(dest, "a.jsonl.conflict-deadbeef"), "wb").write(body["a.jsonl"])
    out = RP.pull_and_receipt(root, None, None, rd, "x9", now=_now, by="x9", blobs={"syn": {}}, plan_file=pf)
    assert out["refusals"] == {"syn": [f"merge_conflict: {rev} (a.jsonl: a.jsonl.conflict-deadbeef)"]} and out["n_read_whole"] == 0
    plan_after = CR._load_json(pf)
    assert plan_after["stages"]["receipt_check"]["evidence"] == {rev: "conflict"} and plan_after["refusals"]["syn"] == out["refusals"]["syn"]


def test_accepted_records_the_harvester_compressed_stay_merged_in_the_evidence_and_the_adapter_read_whole_on_a_resumed_merge_or_receipt(tmp_path, monkeypatch, capsys):
    """The evidence looked for the exact recorded `<name>.jsonl` and hashed its raw bytes, but the harvester's
    `gzip_after_ingest` replaces an ingested record by `<name>.jsonl.gz` (the logical sha unchanged, the plain file removed; after 10 min by
    default, at once here), so a merge or receipt resumed after the harvester had folded the records read every accepted file as missing,
    refused the adapter and scheduled a needless repeat read. Now a recorded file counts under either form by its LOGICAL sha. The actual
    sequence: a whole read, the real harvester ingesting and compressing the merged records, then (a) the evidence over the recorded plan,
    (b) a receipt RESUMED (the merge stage recorded, the receipt not), (c) a merge resumed (the merge stage not recorded) -- all merged, the
    adapter read whole, no refusal. Then the both-forms rule on a temp root."""
    w = _world_with_generation(tmp_path, monkeypatch)
    _store(w["stores"], "s2", "legacy_store_2", "none", [0.1, 0.9, 0.5])
    assert ST.mirror_component(w["root"], w["dest"], w["cfg"].components["synthetic_stores"], runner=TS.Runner()).outcome == "published"
    _tree(w)
    rd = _rd(tmp_path)
    t1 = _tick(w, tmp_path, FakeArray(w), "t1")
    assert t1.run() == 0
    capsys.readouterr()
    plan = CR._load_json(CR.plan_path(w["root"], "t1"))
    rev = plan["stages"]["pull"]["pulled_dirs"][0]
    files = plan["stages"]["pull"]["depends_dirs"][rev]["files"]
    dest = os.path.join(H.records_dir(w["root"]), "adapters", rev)
    assert all(os.path.isfile(os.path.join(dest, n)) for n in files) and plan["stages"]["inbox_merge"]["evidence"] == {rev: "merged"}
    # THE HARVESTER: ingest the merged records and compress them at once (no mirrors configured in this root; min age 0)
    ing = H.ingest_once(w["root"], workers=0, min_free=0, gzip_min_age_s=0)
    assert ing.refused == 0, ing.refusals
    plain_names = sorted(n for n in files if not n.endswith(".gz"))              # a shard's own `.jsonl.gz` records stay as named (never compressed twice)
    assert plain_names and ing.gzipped >= len(plain_names), (ing.gzipped, plain_names, sorted(files))
    assert all(not os.path.exists(os.path.join(dest, n)) and os.path.isfile(os.path.join(dest, n + ".gz")) for n in plain_names), os.listdir(dest)
    assert all(RP.R.file_logical_sha(os.path.join(dest, n + ".gz")) == files[n] for n in plain_names)                 # the twin: its decompressed bytes
    assert all(os.path.isfile(os.path.join(dest, n)) and RP.R.file_sha256(os.path.join(dest, n)) == files[n] for n in files if n.endswith(".gz"))   # as named: raw bytes
    # (a) the evidence over the recorded plan: merged, no problem
    ev = RP.merge_evidence(w["root"], RP.depends_of_plan(plan))
    assert ev == {rev: {"state": "merged", "adapter": ADAPTER, "how": "copied", "problems": []}}, ev
    # (b) a RESUMED receipt (the merge stage recorded, the receipt not yet): the receipt-time re-check reads the .gz twins -> whole
    pf_b = CR.plan_path(w["root"], "t1b")
    doc_b = dict(plan, window="t1b", op={"owner": "tick", "op_id": "t1b"})
    doc_b["stages"] = {k: v for k, v in plan["stages"].items() if k not in ("receipt", "receipt_check")}
    doc_b["refusals"] = {}
    CR._write_json(pf_b, doc_b)
    RP.save_op(rd, dict(RP.new_op("tick", "t1b", plan.get("tree_sha"), _now), state="merged"))
    monkeypatch.setattr(RP, "probe", _probe_nothing_published())
    out_b = RP.pull_and_receipt(w["root"], w["cfg"], w["dest"], rd, "t1b", now=_now, by="t1b", blobs={ADAPTER: {}}, plan_file=pf_b)
    assert out_b["refusals"] == {} and out_b["n_read_whole"] == 1, out_b
    rec_b = json.load(open(RP.receipt_path(rd, "t1b")))
    assert rec_b["adapters"][ADAPTER]["read_whole"] and "receipt_check" not in CR._load_json(pf_b)["stages"]
    # (c) a RESUMED merge (the merge stage not recorded; the inbox empty, the records compressed): merged, whole
    pf_c = CR.plan_path(w["root"], "t1c")
    doc_c = dict(doc_b, window="t1c", op={"owner": "tick", "op_id": "t1c"})
    doc_c["stages"] = {k: v for k, v in doc_b["stages"].items() if k != "inbox_merge"}
    CR._write_json(pf_c, doc_c)
    RP.save_op(rd, dict(RP.new_op("tick", "t1c", plan.get("tree_sha"), _now), state="pulled"))
    out_c = RP.pull_and_receipt(w["root"], w["cfg"], w["dest"], rd, "t1c", now=_now, by="t1c", blobs={ADAPTER: {}}, plan_file=pf_c)
    assert out_c["refusals"] == {} and out_c["n_read_whole"] == 1, out_c
    plan_c = CR._load_json(pf_c)
    assert plan_c["stages"]["inbox_merge"]["evidence"] == {rev: "merged"} and plan_c["stages"]["inbox_merge"]["merged"] == 0 and plan_c["stages"]["inbox_merge"]["problems"] == {}
    assert RP.last_receipt_for(rd, ADAPTER) is not None, "a whole receipt stands: no repeat read is scheduled"
    # the both-forms rule, on a temp root: plain matching -> merged; plain other + .gz matching -> merged; both present, neither matching ->
    # conflict naming both; only a .gz with other bytes -> conflict; a .gz twin alone matching -> merged
    import gzip
    import hashlib
    root2 = str(tmp_path / "root2")
    H.ensure_layout(root2)
    rev2 = "syn/0123456789abcdef"
    ours = b'{"x": 1}\n'
    sha = hashlib.sha256(ours).hexdigest()
    dep2 = {rev2: {"adapter": "syn", "how": "copied", "files": {"a.jsonl": sha}}}
    d2 = os.path.join(H.records_dir(root2), "adapters", rev2)
    os.makedirs(d2)

    def put(name, data, gz=False):
        path = os.path.join(d2, name)
        if gz:
            with gzip.GzipFile(path, "wb", mtime=0) as fh:
                fh.write(data)
        else:
            open(path, "wb").write(data)

    def clear():
        for fn in os.listdir(d2):
            os.remove(os.path.join(d2, fn))
    put("a.jsonl.gz", ours, gz=True)
    assert RP.merge_evidence(root2, dep2)[rev2]["state"] == "merged", "a .gz twin alone, matching by its logical sha"
    put("a.jsonl", ours)
    assert RP.merge_evidence(root2, dep2)[rev2]["state"] == "merged", "both forms, both matching (the harvester between its replace and its remove)"
    put("a.jsonl", b"other")
    ev2 = RP.merge_evidence(root2, dep2)[rev2]                       # a competing plain file beside OUR twin is a conflict, never merged
    assert ev2["state"] == "conflict" and ev2["problems"][0].startswith("a.jsonl: two active forms of one record: a.jsonl (sha ") and "a.jsonl.gz (logical sha " in ev2["problems"][0], ev2
    put("a.jsonl", ours)
    put("a.jsonl.gz", b"other too", gz=True)
    ev2 = RP.merge_evidence(root2, dep2)[rev2]                       # ... and so is a competing twin beside OUR plain file
    assert ev2["state"] == "conflict" and ev2["problems"][0].startswith("a.jsonl: two active forms of one record: "), ev2
    put("a.jsonl", b"other")
    ev2 = RP.merge_evidence(root2, dep2)[rev2]
    assert ev2["state"] == "conflict" and ev2["problems"][0].startswith("a.jsonl: two active forms of one record: a.jsonl (sha ") and "a.jsonl.gz (logical sha " in ev2["problems"][0], ev2
    clear()
    put("a.jsonl.gz", b"other too", gz=True)
    ev2 = RP.merge_evidence(root2, dep2)[rev2]
    assert ev2["state"] == "conflict" and ev2["problems"] == [f"a.jsonl: other bytes in records: a.jsonl.gz (logical sha {hashlib.sha256(b'other too').hexdigest()[:8]}) != {sha[:8]}"], ev2
    clear()
    assert RP.merge_evidence(root2, dep2)[rev2] == {"state": "incomplete", "adapter": "syn", "how": "copied", "problems": ["a.jsonl: missing from records"]}
    # a recorded file the shard itself named `.jsonl.gz`: compared by its RAW bytes as named (what the summary recorded), never decompressed,
    # and no `.gz.gz` twin is looked for
    raw_gz = os.path.join(d2, "b.jsonl.gz")
    with gzip.GzipFile(raw_gz, "wb", mtime=0) as fh:
        fh.write(ours)
    dep3 = {rev2: {"adapter": "syn", "how": "copied", "files": {"b.jsonl.gz": hashlib.sha256(open(raw_gz, "rb").read()).hexdigest()}}}
    assert RP.merge_evidence(root2, dep3)[rev2]["state"] == "merged"
    dep4 = {rev2: {"adapter": "syn", "how": "copied", "files": {"b.jsonl.gz": sha}}}                  # the DECOMPRESSED sha is not what was recorded
    ev4 = RP.merge_evidence(root2, dep4)[rev2]
    assert ev4["state"] == "conflict" and ev4["problems"] == [f"b.jsonl.gz: other bytes in records: b.jsonl.gz (sha {dep3[rev2]['files']['b.jsonl.gz'][:8]}) != {sha[:8]}"], ev4


def _competing_delivery(root, rev, alter_name, other_bytes, rd, plan_a, op_id):
    """A delayed OTHER delivery of revision `rev` into the inbox -- every file of the revision as records/ holds it (the plain bytes; a `.gz`
    twin decompressed back to its plain name), with `alter_name` rewritten to `other_bytes` -- and an operation `op_id` that DEPENDS on it
    (a pull that reused the inbox copy: `depends_dirs[rev]` = the inbox copy's own raw shas). Returns the plan file."""
    import gzip as _gzip
    dest = os.path.join(H.records_dir(root), "adapters", rev)
    inbox_dir = os.path.join(H.inbox_dir(root), CR.INBOX_NAME, "adapters", rev)
    os.makedirs(inbox_dir)
    for fn in sorted(os.listdir(dest)):
        if ".conflict-" in fn:
            continue
        src = os.path.join(dest, fn)
        if fn.endswith(".gz") and not fn.endswith(".jsonl.gz"):
            continue
        name = fn[:-3] if (fn.endswith(".gz") and fn[:-3] in plan_a["stages"]["pull"]["depends_dirs"][rev]["files"]) else fn   # a harvester twin -> its plain name
        data = _gzip.open(src, "rb").read() if name != fn else open(src, "rb").read()
        open(os.path.join(inbox_dir, name), "wb").write(other_bytes if name == alter_name else data)
    files = {fn: RP.R.file_sha256(os.path.join(inbox_dir, fn)) for fn in os.listdir(inbox_dir)}
    assert alter_name in files and any(H.REVISION_MARKER_TOKEN in f for f in files), files
    pf = CR.plan_path(root, op_id)
    doc = dict(plan_a, window=op_id, op={"owner": "tick", "op_id": op_id}, refusals={})
    doc["stages"] = {"revalidate_post": plan_a["stages"]["revalidate_post"],
                     "pull": dict(plan_a["stages"]["pull"], pulled_dirs=[], depends_dirs={rev: {"adapter": ADAPTER, "how": "reused", "files": files}})}
    CR._write_json(pf, doc)
    RP.save_op(rd, dict(RP.new_op("tick", op_id, plan_a.get("tree_sha"), _now), state="pulled"))
    return pf, files


def test_a_delayed_same_name_conflicting_delivery_before_or_after_the_harvester_fold_is_kept_aside_never_a_second_active_form_and_gets_no_whole_receipt(tmp_path, monkeypatch, capsys):
    """After the harvester compressed an accepted `a.jsonl` to its `.gz` twin, `scan_inbox` checked only whether the
    PLAIN `records/.../a.jsonl` existed, so a delayed other delivery of the same revision with different bytes was installed as a new active
    plain file beside the correct twin -- two forms of one logical URI with different bytes, both eligible for ingestion -- and the evidence
    rule read the matching twin as merged. Now the inbox merge treats the logical filename as occupied by EITHER form and compares logical
    shas (duplicate dropped, other bytes kept aside as `.conflict-<sha8>`, never a second active form), and two differing active forms are a
    conflict in the evidence whichever is ours. The sequence: a whole read (A); a delayed same-name conflicting delivery for an
    operation depending on it (i) BEFORE the fold and (ii) AFTER a real fold (`ingest_once(gzip_min_age_s=0)`): the competitor aside, the
    accepted form intact and alone, the depending operation's evidence `conflict`, no whole receipt; then the belt over a hand-installed
    competing plain file beside the matching twin (what the old merger produced): A's own evidence reads a conflict."""
    w = _world_with_generation(tmp_path, monkeypatch)
    _store(w["stores"], "s2", "legacy_store_2", "none", [0.1, 0.9, 0.5])
    assert ST.mirror_component(w["root"], w["dest"], w["cfg"].components["synthetic_stores"], runner=TS.Runner()).outcome == "published"
    _tree(w)
    rd = _rd(tmp_path)
    assert _tick(w, tmp_path, FakeArray(w), "A").run() == 0
    capsys.readouterr()
    plan_a = CR._load_json(CR.plan_path(w["root"], "A"))
    rev = plan_a["stages"]["pull"]["pulled_dirs"][0]
    files_a = plan_a["stages"]["pull"]["depends_dirs"][rev]["files"]
    dest = os.path.join(H.records_dir(w["root"]), "adapters", rev)
    plain_names = sorted(n for n in files_a if not n.endswith(".gz"))
    target = next((n for n in plain_names if H.REVISION_MARKER_TOKEN not in n), plain_names[0])     # a plain-named record: the fold makes it a twin
    monkeypatch.setattr(RP, "probe", _probe_nothing_published())
    # (i) BEFORE the fold: records/ holds A's plain file; the competitor is kept aside, the plain file intact, B refused
    pf_b, files_b = _competing_delivery(w["root"], rev, target, b'{"delayed": "other bytes, before the fold"}\n', rd, plan_a, "B1")
    assert files_b[target] != files_a[target]
    out_b = RP.pull_and_receipt(w["root"], w["cfg"], w["dest"], rd, "B1", now=_now, by="B1", blobs={ADAPTER: {}}, plan_file=pf_b)
    assert out_b["n_read_whole"] == 0 and out_b["refusals"][ADAPTER][0].startswith(f"merge_conflict: {rev} ({target}: {target}.conflict-"), out_b
    plan_b = CR._load_json(pf_b)
    assert plan_b["stages"]["inbox_merge"]["conflict"] == 1 and plan_b["stages"]["inbox_merge"]["evidence"] == {rev: "conflict"}
    assert RP.R.file_sha256(os.path.join(dest, target)) == files_a[target], "A's accepted plain file is intact"
    assert len([f for f in os.listdir(dest) if f.startswith(target + ".conflict-")]) == 1 and not os.path.isdir(os.path.join(H.inbox_dir(w["root"]), CR.INBOX_NAME, "adapters", rev))
    assert RP.merge_evidence(w["root"], RP.depends_of_plan(plan_a))[rev]["state"] == "merged", "A's own evidence: its bytes stand, the competitor is aside"
    # THE FOLD: the real harvester compresses A's accepted records (the conflict file is no record delivery; it stays)
    ing = H.ingest_once(w["root"], workers=0, min_free=0, gzip_min_age_s=0)
    assert ing.refused == 0 and not os.path.exists(os.path.join(dest, target)) and os.path.isfile(os.path.join(dest, target + ".gz"))
    assert RP.R.file_logical_sha(os.path.join(dest, target + ".gz")) == files_a[target]
    # (ii) AFTER the fold: the competitor meets the twin -- kept aside by logical sha, NO plain file installed beside the twin, C refused
    pf_c, files_c = _competing_delivery(w["root"], rev, target, b'{"delayed": "other bytes, after the fold"}\n', rd, plan_a, "C1")
    assert files_c[target] not in (files_a[target], files_b[target])
    out_c = RP.pull_and_receipt(w["root"], w["cfg"], w["dest"], rd, "C1", now=_now, by="C1", blobs={ADAPTER: {}}, plan_file=pf_c)
    assert out_c["n_read_whole"] == 0 and out_c["refusals"][ADAPTER][0].startswith(f"merge_conflict: {rev} ({target}: {target}.conflict-"), out_c
    plan_c = CR._load_json(pf_c)
    assert plan_c["stages"]["inbox_merge"]["conflict"] == 1 and plan_c["stages"]["inbox_merge"]["evidence"] == {rev: "conflict"}
    assert not os.path.exists(os.path.join(dest, target)), "no second active form beside the twin"
    assert RP.R.file_logical_sha(os.path.join(dest, target + ".gz")) == files_a[target], "the twin is intact"
    assert len([f for f in os.listdir(dest) if f.startswith(target + ".conflict-")]) == 2
    assert RP.merge_evidence(w["root"], RP.depends_of_plan(plan_a))[rev]["state"] == "merged"
    assert RP.last_receipt_for(rd, ADAPTER)[0]["op"]["op_id"] == "A", "A's whole receipt stands; B1's and C1's are refused"
    # THE BELT: what the OLD merger produced -- a competing plain file installed as a second active form beside the matching twin -- is a
    # conflict in A's own evidence (never merged on the twin alone), so a resumed receipt-check of A would refuse it
    open(os.path.join(dest, target), "wb").write(b'{"installed": "by the old merger"}\n')
    ev = RP.merge_evidence(w["root"], RP.depends_of_plan(plan_a))[rev]
    assert ev["state"] == "conflict" and ev["problems"][0].startswith(f"{target}: two active forms of one record: {target} (sha ") and f"{target}.gz (logical sha " in ev["problems"][0], ev
    os.remove(os.path.join(dest, target))
    assert RP.merge_evidence(w["root"], RP.depends_of_plan(plan_a))[rev]["state"] == "merged"


def test_a_tick_that_dies_after_persisting_pull_is_merged_then_receipted_by_the_next_reconcile_and_no_receipt_is_written_without_the_merge(tmp_path, monkeypatch, capsys):
    """The pull stage was persisted and the inbox-merge ran inside the SAME `if "pull" not in stages` block, so a tick
    terminated after persisting `pull` and before the merge (the unit's bound after a long array) left its records in the inbox -- and the
    next reconcile, taking the completed-unreceipted operation, skipped the merge and wrote a whole-read receipt: the changed set then read
    the adapter as read and might never schedule that pin again. Now the merge is its own persisted stage: the resume starts at the FIRST
    missing stage (merge, then receipt), the op record mirrors each stage with who did it, and the receipt is refused outright when the plan
    records no merge (invariant I7) -- a merge that raises writes no receipt either."""
    w = _world_with_generation(tmp_path, monkeypatch)
    _store(w["stores"], "s2", "legacy_store_2", "none", [0.1, 0.9, 0.5])
    assert ST.mirror_component(w["root"], w["dest"], w["cfg"].components["synthetic_stores"], runner=TS.Runner()).outcome == "published"
    _tree(w)
    rd = _rd(tmp_path)
    slurm = FakeSlurm()
    fake = FakeArray(w)

    def die(root):
        raise Died("t1 dies after persisting pull, before the merge")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(H, "scan_inbox", die)
        t1 = _tick(w, tmp_path, fake, "t1", slurm=slurm)
        with pytest.raises(Died):
            t1.run()
    capsys.readouterr()
    plan = CR._load_json(CR.plan_path(w["root"], "t1"))
    assert "pull" in plan["stages"] and "inbox_merge" not in plan["stages"] and len(plan["stages"]["pull"]["pulled_dirs"]) == 1
    op = RP.load_op(rd, "t1")
    assert op["state"] == "pulled" and set(op["stages"]) == {"pull"} and op["stages"]["pull"]["by"] == "t1"
    rev = plan["stages"]["pull"]["pulled_dirs"][0]
    records_dir = os.path.join(H.records_dir(w["root"]), "adapters", rev)
    assert not os.path.exists(RP.receipt_path(rd, "t1")) and not os.path.isdir(records_dir)             # pulled, never merged, no receipt
    # the process died: a stale holder with a completed array; the next tick's reconcile resumes at the merge
    op.update(pid=DEAD_PID)
    RP.save_op(rd, op)
    RP.replace_reservation(rd, dict(RP._reservation("tick", "t1", None, _now), pid=DEAD_PID))
    slurm.jobs_.update({"4241": {"name": "attest-t1", "state": "COMPLETED"}, "4242": {"name": "reads-t1", "state": "COMPLETED"}})
    merges = []
    real_scan = H.scan_inbox

    def counting_scan(root):
        out = real_scan(root)
        merges.append(sum(1 for _, st in out if st == "merged"))
        return out
    monkeypatch.setattr(H, "scan_inbox", counting_scan)
    monkeypatch.setattr(CR, "pull_window", lambda *a, **k: pytest.fail("the pull stage is recorded: never pulled twice"))
    monkeypatch.setattr(RP, "probe", _probe_nothing_published())
    t2 = _tick(w, tmp_path, FakeArray(w, run_shards=False), "t2", slurm=slurm)
    assert t2.run() == 0
    s2 = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert s2["reservation"] == "replaced_stale" and s2["reconcile"]["pulled"]["t1"]["pulled"] == 1 and s2["reconcile"]["pulled"]["t1"]["n_read_whole"] == 1
    assert len(merges) == 1 and merges[0] >= 1, merges                                                     # ONE merge, of the revision's files
    plan2 = CR._load_json(CR.plan_path(w["root"], "t1"))
    assert {"pull", "inbox_merge"} <= set(plan2["stages"]) and plan2["stages"]["inbox_merge"]["by"] == "t2" and plan2["stages"]["inbox_merge"]["merged"] >= 1
    op2 = RP.load_op(rd, "t1")
    assert op2["state"] == "receipted" and op2["receipted_by"] == "t2" and [op2["stages"][k]["by"] for k in ("pull", "inbox_merge", "receipt")] == ["t1", "t2", "t2"]
    rec = json.load(open(RP.receipt_path(rd, "t1")))
    assert rec["by"] == "t2" and rec["adapters"][ADAPTER]["read_whole"] and os.path.isdir(records_dir)
    # the I7 tripwire, directly: a plan that records `pull` but whose merge is not recorded gets NO receipt
    def plan_with_pull(op_id):
        pf = CR.plan_path(w["root"], op_id)
        doc = {"schema": CR.PLAN_SCHEMA, "window": op_id, "adapters": {}, "refusals": {}, "stages": {"revalidate_post": {}, "pull": {"pulled_dirs": []}},
               "op": {"owner": "tick", "op_id": op_id}, "tree_sha": "a" * 40}
        CR._write_json(pf, doc)
        RP.save_op(rd, dict(RP.new_op("tick", op_id, "a" * 40, _now), state="pulled", stages={"pull": {"by": op_id}}))
        return pf
    pf = plan_with_pull("t3")
    real_record = CR.record_stage
    monkeypatch.setattr(CR, "record_stage", lambda plan_, name, **f: None if name == "inbox_merge" else real_record(plan_, name, **f))
    with pytest.raises(RP.ReceiptRefused) as ei:
        RP.pull_and_receipt(w["root"], w["cfg"], w["dest"], rd, "t3", now=_now, by="t9", blobs={}, plan_file=pf)
    assert "invariant I7" in str(ei.value) and "inbox_merge" in str(ei.value) and not os.path.exists(RP.receipt_path(rd, "t3"))
    monkeypatch.setattr(CR, "record_stage", real_record)
    # a merge that RAISES writes no receipt and leaves the operation `pulled` (the next reconcile resumes at the merge)
    pf4 = plan_with_pull("t4")

    def merge_fails(root):
        raise RuntimeError("inbox: disk full")
    monkeypatch.setattr(H, "scan_inbox", merge_fails)
    with pytest.raises(RuntimeError):
        RP.pull_and_receipt(w["root"], w["cfg"], w["dest"], rd, "t4", now=_now, by="t9", blobs={}, plan_file=pf4)
    assert not os.path.exists(RP.receipt_path(rd, "t4")) and RP.load_op(rd, "t4")["state"] == "pulled" and "inbox_merge" not in CR._load_json(pf4)["stages"]
    monkeypatch.setattr(H, "scan_inbox", real_scan)
    out = RP.pull_and_receipt(w["root"], w["cfg"], w["dest"], rd, "t4", now=_now, by="t9", blobs={}, plan_file=pf4)
    assert os.path.isfile(out["receipt"]) and RP.load_op(rd, "t4")["state"] == "receipted" and set(RP.load_op(rd, "t4")["stages"]) == {"pull", "inbox_merge", "receipt"}


def test_update_op_concurrent_writers_lose_no_field_and_stages_merge(tmp_path):
    """`update_op` was a whole-record load-modify-save with no lock, so the owner's own write and a foreign writer's (a
    takeover's `cancelled_by`, a reconcile's `reconciled_by`) could lose each other's fields -- the `jobs` the self-cancel reads, or a
    terminal `state` revived. Eight threads writing one record (the per-record flock conflicts across open file descriptions) lose nothing,
    `stages` merges rather than replaces, `record_op_stage` sets the stage's state and the record fields, and the lock file is not an op."""
    rd = _rd(tmp_path)
    RP.save_op(rd, RP.new_op("tick", "t1", None, _now))

    def writer(i):
        for k in range(25):
            RP.update_op(rd, "t1", now=_now, state=f"s{i}_{k}", **{f"f{i}_{k}": k})
    ths = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for th in ths:
        th.start()
    for th in ths:
        th.join(60)
    op = RP.load_op(rd, "t1")
    assert all(op.get(f"f{i}_{k}") == k for i in range(8) for k in range(25)) and len(op["history"]) == 200, (len(op), len(op["history"]))
    RP.update_op(rd, "t1", now=_now, stages={"pull": {"by": "a"}})
    RP.update_op(rd, "t1", now=_now, stages={"inbox_merge": {"by": "b"}})
    assert RP.load_op(rd, "t1")["stages"] == {"pull": {"by": "a"}, "inbox_merge": {"by": "b"}}
    RP.record_op_stage(rd, "t1", "receipt", now=_now, by="c", receipt="/r", receipted_by="c")
    op = RP.load_op(rd, "t1")
    assert op["state"] == "receipted" and op["receipt"] == "/r" and op["receipted_by"] == "c" and op["stages"]["receipt"] == {"utc": _now(), "by": "c"}
    assert set(op["stages"]) == {"pull", "inbox_merge", "receipt"}
    with pytest.raises(ValueError):
        RP.record_op_stage(rd, "t1", "watch")
    assert [o["op_id"] for o in RP.list_ops(rd)] == ["t1"] and os.path.isfile(os.path.join(rd, RP.OPS_SUBDIR, "t1.lock"))


def test_race_cancellation_after_the_computation_before_the_pull_the_tick_pulls_nothing_and_the_window_pulls_it(tmp_path, monkeypatch, capsys):
    """Race 4: the array completed; the window takes over before the tick's pull. The tick's ownership check before the pull reads the loss:
    exit 20, nothing pulled or merged. The window's reconcile then finds the completed-unpulled operation and pulls it under its own
    reservation -- every barrier of the pull applying -- and the receipt names the window as `by`."""
    w = _world_with_generation(tmp_path, monkeypatch)
    _store(w["stores"], "s2", "legacy_store_2", "none", [0.1, 0.9, 0.5])
    assert ST.mirror_component(w["root"], w["dest"], w["cfg"].components["synthetic_stores"], runner=TS.Runner()).outcome == "published"
    _tree(w)
    rd = _rd(tmp_path)
    slurm = FakeSlurm()

    def takeover():
        slurm.jobs_.update({"4241": {"name": "attest-t1", "state": "COMPLETED"}, "4242": {"name": "reads-t1", "state": "COMPLETED"}})
        how, _ = RP.take_reservation(rd, slurm, "window", "W1", takeover=True, now=_now, poll_s=0, sleep=lambda s: None)
        assert how == "taken_over"
    fake = FakeArray(w, before_watch=takeover)
    pulls = []
    real_pull = CR.pull_window

    def counting_pull(*a, **k):
        pulls.append(a[1].get("window"))
        return real_pull(*a, **k)
    monkeypatch.setattr(CR, "pull_window", counting_pull)
    t = _tick(w, tmp_path, fake, "t1", slurm=slurm)
    assert t.run() == RP.EXIT_CANCELLED
    s = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert s["state"] == "cancelled" and s["watch"] == "done" and pulls == []
    assert RP.load_op(rd, "t1")["state"] == "cancelled" and RP.load_op(rd, "t1")["cancelled_by"] == "window W1"
    assert not os.path.exists(RP.receipt_path(rd, "t1")) and RP.read_reservation(rd)["op_id"] == "W1"
    # the window's reconcile (what `cluster-reads reserve` runs after the takeover): the completed array is pulled now, once
    rep = RP.reconcile(rd, slurm, "W1", now=_now, poll_s=0, sleep=lambda s: None,
                       pull_orphan=lambda op: RP.pull_and_receipt(w["root"], w["cfg"], w["dest"], rd, op["op_id"], now=_now, by="W1",
                                                                  blobs={ADAPTER: {}}, plan_file=CR.plan_path(w["root"], op["op_id"])))
    assert rep["completed_unpulled"] == ["t1"] and rep["pulled"]["t1"]["pulled"] == 1 and pulls == ["t1"]
    rec = json.load(open(RP.receipt_path(rd, "t1")))
    assert rec["by"] == "W1" and rec["adapters"][ADAPTER]["read_whole"] and RP.load_op(rd, "t1")["state"] == "receipted"
    rep2 = RP.reconcile(rd, slurm, "W1", now=_now, poll_s=0, sleep=lambda s: None, pull_orphan=lambda op: pytest.fail("pulled twice"))
    assert rep2["completed_unpulled"] == []


def test_race_takeover_during_the_pull_waits_for_the_critical_section_and_the_tick_receipts_under_its_own_ownership(tmp_path, monkeypatch, capsys):
    """The window's takeover arrives AFTER the tick's ownership check, while the tick is pulling. Before the fix the
    array was terminal, so the takeover proceeded at once and the tick went on transferring, merging into records/adapters and writing a
    whole-read receipt under the window's ownership. Now the ownership check and the pull / merge / receipt are ONE critical section: the
    takeover -- another thread here, as it is another process live -- blocks on the lock until the receipt is written; the reservation
    names the tick throughout the pull; the receipt names the tick as `by`; only then does the reservation change hands. The tick's own
    release is held back until the takeover lands (pinning the interleaving): it finds the window's reservation and removes nothing."""
    w = _world_with_generation(tmp_path, monkeypatch)
    _store(w["stores"], "s2", "legacy_store_2", "none", [0.1, 0.9, 0.5])
    assert ST.mirror_component(w["root"], w["dest"], w["cfg"].components["synthetic_stores"], runner=TS.Runner()).outcome == "published"
    _tree(w)
    rd = _rd(tmp_path)
    slurm = FakeSlurm()
    real_array = CR.submit_array

    def array_recorded(plan, venv, name, **kw):
        job = real_array(plan, venv, name, **kw)
        slurm.jobs_[kw["dependency"]] = {"name": f"attest-{plan['window']}", "state": "COMPLETED"}
        slurm.jobs_[job] = {"name": name, "state": "COMPLETED"}            # the fake watch runs the shards: both jobs have ended by the pull
        return job
    monkeypatch.setattr(CR, "submit_array", array_recorded)
    in_pull, takeover_attempted, takeover_done = threading.Event(), threading.Event(), threading.Event()
    marks = {}
    real_pull = CR.pull_window

    def pull_observed(*a, **k):
        in_pull.set()
        assert takeover_attempted.wait(30), "the takeover thread never reached take_reservation"
        time.sleep(0.6)                                                   # the takeover is blocked on the section the whole time
        marks["holder_mid_pull"] = RP.read_reservation(rd)["op_id"]
        marks["takeover_landed_mid_pull"] = takeover_done.is_set()
        out = real_pull(*a, **k)
        marks["pull_end"] = time.monotonic()
        return out
    monkeypatch.setattr(CR, "pull_window", pull_observed)
    real_release = RP.release

    def release_after_takeover(rd_, op_id, **kw):
        marks["release_saw_takeover"] = takeover_done.wait(30)
        return real_release(rd_, op_id, **kw)
    monkeypatch.setattr(RP, "release", release_after_takeover)

    def takeover():
        in_pull.wait(60)
        takeover_attempted.set()
        how, _ = RP.take_reservation(rd, slurm, "window", "W1", takeover=True, now=_now, poll_s=0, sleep=lambda s: None, section_poll_s=0.02)
        marks["takeover"] = (how, time.monotonic())
        takeover_done.set()
    th = threading.Thread(target=takeover, daemon=True)
    th.start()
    t = _tick(w, tmp_path, FakeArray(w), "t1", slurm=slurm)
    rc = t.run()
    th.join(30)
    assert not th.is_alive() and rc == 0, (rc, marks)
    s = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert s["state"] == "receipted" and s["pulled"] == 1, s
    assert marks["holder_mid_pull"] == "t1" and marks["takeover_landed_mid_pull"] is False, marks        # the takeover waited through the pull
    assert marks["takeover"][0] == "taken_over" and marks["takeover"][1] >= marks["pull_end"], marks      # ... and landed after the receipt
    rec = json.load(open(RP.receipt_path(rd, "t1")))
    assert rec["by"] == "t1" and rec["adapters"][ADAPTER]["read_whole"] and len(rec["adapters"][ADAPTER]["revisions_pulled"]) == 1
    op = RP.load_op(rd, "t1")
    assert op["state"] == "receipted" and op["cancelled_by"] == "window W1"      # the takeover marks the record it took over, after the receipt
    assert RP.read_reservation(rd)["op_id"] == "W1" and marks["release_saw_takeover"] is True
    assert not os.path.exists(RP.critical_lock_path(rd)) or RP.read_critical_holder(rd)["op_id"] == "W1"   # the last holder of the section
    # the window's reconcile has nothing to pull: the tick's receipt stands and the revision is in records/
    rep = RP.reconcile(rd, slurm, "W1", now=_now, poll_s=0, sleep=lambda s: None, pull_orphan=lambda op: pytest.fail("pulled twice"))
    assert rep["completed_unpulled"] == [] and "t1" not in rep["left_alive"]
    assert os.path.isdir(os.path.join(H.records_dir(w["root"]), "adapters", rec["adapters"][ADAPTER]["revisions_pulled"][0]))


def test_reconcile_under_a_reservation_that_names_another_operation_is_ownership_lost_and_the_tick_exits_20(tmp_path, monkeypatch, capsys):
    """The reconcile re-reads the reservation INSIDE the critical section: one that names another operation means the caller lost the
    namespace between its reservation and this step (a takeover landed in the gap) -- it reconciles nothing (no orphan pull, no cancel under
    the window's ownership). Directly: `OwnershipLost` naming the holder and the recorded cancel; in the tick: exit 20, nothing probed."""
    rd = _rd(tmp_path)
    RP.reserve(rd, "window", "W1", now=_now)
    RP.save_op(rd, dict(RP.new_op("tick", "t0", None, _now), state="watched", jobs={"reads": "100"}, pid=DEAD_PID))
    slurm = FakeSlurm({"100": {"name": "reads-t0", "state": "COMPLETED"}})
    with pytest.raises(RP.OwnershipLost) as ei:
        RP.reconcile(rd, slurm, "t1", now=_now, poll_s=0, sleep=lambda s: None, pull_orphan=lambda op: pytest.fail("an orphan pulled under another's reservation"))
    assert ei.value.holder["op_id"] == "W1" and RP.load_op(rd, "t0")["state"] == "watched"
    RP.release(rd, "W1")
    # the tick: reserved, then the window takes over right before its reconcile -> OwnershipLost -> exit 20 (the cancel is recorded)
    w = _world_with_generation(tmp_path, monkeypatch)
    real_reconcile = RP.reconcile

    def takeover_then_reconcile(rd_, slurm_, my_op_id, **kw):
        how, _ = RP.take_reservation(rd_, slurm_, "window", "W2", takeover=True, now=_now, poll_s=0, sleep=lambda s: None)
        assert how == "taken_over"
        return real_reconcile(rd_, slurm_, my_op_id, **kw)
    monkeypatch.setattr(RP, "reconcile", takeover_then_reconcile)
    monkeypatch.setattr(RP, "probe", lambda *a, **k: pytest.fail("nothing is probed after the loss"))
    t = _tick(w, tmp_path, FakeArray(w, run_shards=False), "t1", slurm=slurm)
    assert t.run() == RP.EXIT_CANCELLED
    s = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert s["state"] == "cancelled" and RP.load_op(rd, "t1")["cancelled_by"] == "window W2" and RP.read_reservation(rd)["op_id"] == "W2"


# --------------------------------------------------------------------------------------------
# the CLI: --op on submit / pull, reserve / release / receipt
# --------------------------------------------------------------------------------------------


def test_cli_submit_and_pull_refuse_by_name_when_the_reservation_names_another_operation(tmp_path, monkeypatch, capsys):
    root = str(tmp_path / "root")
    H.ensure_layout(root)
    plan = {"schema": CR.PLAN_SCHEMA, "window": "W1", "adapters": {}, "refusals": {}, "stages": {}, "tasks": [{"index": 0}], "tree": "/t", "window_root": "/w", "host": None}
    CR._write_json(CR.plan_path(root, "W1"), plan)
    state = str(tmp_path / "state")
    rd = RP.reads_dir(state)
    RP.reserve(rd, "window", "W2", now=_now)
    monkeypatch.setattr(CR, "submit_attest", lambda *a, **k: pytest.fail("no submission under a lost reservation"))
    assert CLI.main(["--root", root, "cluster-reads", "submit", "--window", "W1", "--venv", "/v", "--op", "W1", "--state", state]) == RP.EXIT_OWNERSHIP
    assert "names window W2, not W1" in capsys.readouterr().err
    monkeypatch.setattr(CR, "pull_window", lambda *a, **k: pytest.fail("nothing pulled under a lost reservation"))
    assert CLI.main(["--root", root, "cluster-reads", "pull", "--window", "W1", "--op", "W1", "--state", state]) == RP.EXIT_OWNERSHIP
    assert "nothing pulled or merged" in capsys.readouterr().err
    # the holder itself passes the check (the submit then runs; recorded on the op)
    monkeypatch.setattr(CR, "submit_attest", lambda *a, **k: "776")
    monkeypatch.setattr(CR, "submit_array", lambda *a, **k: "777")
    assert CLI.main(["--root", root, "cluster-reads", "submit", "--window", "W1", "--venv", "/v", "--op", "W2", "--state", state]) == 0
    assert capsys.readouterr().out.strip() == "777"
    assert RP.load_op(rd, "W2")["jobs"] == {"attest": "776", "reads": "777"} and RP.load_op(rd, "W2")["state"] == "submitted"
    assert CR._load_json(CR.plan_path(root, "W1"))["stages"]["submit"]["op"] == "W2"
    # without --op the check is not made (the protocol is the window's and the tick's to invoke)
    assert CLI.main(["--root", root, "cluster-reads", "submit", "--window", "W1", "--venv", "/v"]) == 0


def test_cli_reserve_takes_over_a_tick_reconciles_and_release_is_by_name(tmp_path, monkeypatch, capsys):
    cfg = _cfg_two(tmp_path, {"a1": {"shards": 1, "components": ["c1"]}})
    root = str(tmp_path / "root")
    H.ensure_layout(root)
    state = str(tmp_path / "state")
    rd = RP.reads_dir(state)
    RP.reserve(rd, "tick", "t1", now=_now)
    RP.update_op(rd, "t1", now=_now, state="submitted", jobs={"reads": "4242"})
    slurm = FakeSlurm({"4242": {"name": "reads-t1", "state": "RUNNING"}})
    monkeypatch.setattr(RP, "Slurm", lambda host, runner=None, **k: slurm)
    monkeypatch.setattr(ST, "load_config", lambda p=None, root=None: cfg)
    assert CLI.main(["--root", root, "cluster-reads", "reserve", "--owner", "window", "--op", "W1", "--takeover", "--state", state, "--dest", "cluster_a"]) == 0
    out = capsys.readouterr().out
    assert "reserve: taken_over for window W1" in out and slurm.cancelled == ["4242"] and RP.read_reservation(rd)["op_id"] == "W1"
    assert RP.load_op(rd, "t1")["cancelled_by"] == "window W1" and RP.load_op(rd, "W1")["state"] == "reconciled_others"
    # a second window: refused by name, exit 24, nothing changed
    assert CLI.main(["--root", root, "cluster-reads", "reserve", "--owner", "window", "--op", "W2", "--takeover", "--state", state, "--dest", "cluster_a"]) == RP.EXIT_HELD
    err = capsys.readouterr()
    assert "REFUSED" in err.err and "held by window W1" in err.err and RP.read_reservation(rd)["op_id"] == "W1"
    assert CLI.main(["--root", root, "cluster-reads", "release", "--op", "W2", "--state", state]) == 0 and os.path.exists(RP.reservation_path(rd))
    assert CLI.main(["--root", root, "cluster-reads", "release", "--op", "W1", "--state", state]) == 0 and not os.path.exists(RP.reservation_path(rd))
    capsys.readouterr()
    # a tick-owner reserve yields nothing (owner tick, no takeover of a live window)
    RP.reserve(rd, "window", "W3", now=_now)
    assert CLI.main(["--root", root, "cluster-reads", "reserve", "--owner", "tick", "--op", "t2", "--state", state, "--dest", "cluster_a"]) == RP.EXIT_HELD


def test_cli_receipt_writes_the_receipt_from_the_windows_plan_and_refuses_without_ownership(tmp_path, monkeypatch, capsys):
    w = _world_with_generation(tmp_path, monkeypatch)
    _store(w["stores"], "s2", "legacy_store_2", "none", [0.1, 0.9, 0.5])
    assert ST.mirror_component(w["root"], w["dest"], w["cfg"].components["synthetic_stores"], runner=TS.Runner()).outcome == "published"
    _tree(w)
    state = str(tmp_path / "state")
    rd = RP.reads_dir(state)
    # the window's plan, staged, attested and read (the shards run in-process), then the CLI's receipt does revalidate -> pull -> merge -> receipt
    from test_registry_cluster_read import _plan_and_stage
    plan, plan_file, _ = _plan_and_stage(w, tmp_path, window="W1")
    _node(w)
    for i in range(2):
        assert CR.run_shard(f"{plan['window_root']}/plan.json", i, home=w["home"]) == 0
    _back_on_grading_host(w)
    monkeypatch.setattr(ST, "load_config", lambda p=None, root=None: w["cfg"])
    monkeypatch.setattr(RP, "module_blobs", lambda *a, **k: {ADAPTER: {}})
    assert CLI.main(["--root", w["root"], "cluster-reads", "receipt", "--window", "W1", "--op", "W1", "--state", state, "--dest", "cluster_a"]) == RP.EXIT_OWNERSHIP
    assert "nothing pulled or written" in capsys.readouterr().err and not os.path.exists(RP.receipt_path(rd, "W1"))
    RP.reserve(rd, "window", "W1", now=_now)
    # the window's sequence: `revalidate` (the pull delivers nothing without its recorded verdict), then `pull --op`: the
    # transfer AND the merge in one section, both recorded -- on the plan and on the op record
    assert CLI.main(["--root", w["root"], "cluster-reads", "revalidate", "--window", "W1"]) == 0
    capsys.readouterr()
    merges = []
    real_scan = H.scan_inbox
    monkeypatch.setattr(H, "scan_inbox", lambda root: (merges.append(1), real_scan(root))[1])
    assert CLI.main(["--root", w["root"], "cluster-reads", "pull", "--window", "W1", "--op", "W1", "--state", state]) == 0
    out = capsys.readouterr().out
    assert "1 revision(s) complete" in out and re.search(r"inbox-merge: [1-9]\d* merged", out) and "the receipt's precondition" in out, out
    plan1 = CR._load_json(plan_file)
    assert {"pull", "inbox_merge"} <= set(plan1["stages"]) and plan1["stages"]["inbox_merge"]["by"] == "W1" and plan1["stages"]["inbox_merge"]["merged"] >= 1
    op1 = RP.load_op(rd, "W1")
    assert op1["state"] == "merged" and set(op1["stages"]) == {"pull", "inbox_merge"} and op1["stages"]["pull"]["by"] == "W1" and merges == [1]
    assert CLI.main(["--root", w["root"], "cluster-reads", "receipt", "--window", "W1", "--op", "W1", "--state", state, "--dest", "cluster_a"]) == 0
    out = capsys.readouterr().out
    assert "1 adapter(s) read whole, 1 revision(s) pulled" in out
    assert merges == [1], "the receipt merged nothing again: the recorded inbox_merge stage is its precondition"
    assert RP.load_op(rd, "W1")["stages"]["receipt"]["by"] == "W1"
    rec = json.load(open(RP.receipt_path(rd, "W1")))
    assert rec["op"] == {"op_id": "W1", "owner": "window"} and rec["adapters"][ADAPTER]["read_whole"] and rec["by"] == "W1"
    plan2 = CR._load_json(plan_file)
    assert set(plan2["stages"]) >= {"revalidate_post", "pull", "inbox_merge"} and RP.load_op(rd, "W1")["state"] == "receipted"
    # a tick's changed set now sees the window's receipt: nothing changed for the adapter
    probed = {"components": {"synthetic_stores": {"current": rec["adapters"][ADAPTER]["pins"]["synthetic_stores"], "exists": True}}}
    cs = RP.changed_set(w["cfg"], [ADAPTER], probed, {ADAPTER: {}}, lambda a: RP.last_receipt_for(rd, a))
    assert cs["unchanged"] == [ADAPTER]


def test_a_receipt_with_standing_refusals_is_changed_when_the_reading_tree_moved_even_with_pins_and_module_unchanged(tmp_path):
    """A whole receipt with only standing refusals became the scheduler's baseline, and `changed_set`
    compared pins and the adapter's own module blob alone -- a fix in SHARED reader code (runner.py, a helper) never re-scheduled the read.
    Now the reading tree is part of the invalidation for such a receipt: recorded under tree A, read under tree B -> changed by name (the
    refusal cannot stand under B: its key names the tree); under A -> unchanged; a receipt WITHOUT standing refusals is unchanged under B
    (the narrow rule)."""
    cfg = _cfg_two(tmp_path, {"a1": {"shards": 1, "components": ["c1"]}, "a2": {"shards": 1, "components": ["c1"]}})
    probed = {"components": {"c1": {"current": "11111111", "exists": True}}}
    blobs = {"a1": {"src/sisteer/registry/adapters/a1.py": "b1"}, "a2": {"src/sisteer/registry/adapters/a2.py": "b2"}}
    tree_a, tree_b = "a" * 40, "b" * 40
    standing = [{"group": "g1", "revision_id": "rev1", "state": "standing_refusal", "refusal": "synthetic read failure",
                 "prior_refusal": {"window": "W0", "utc": "2026-10-04T08:00:00.000Z", "digest": "f" * 64, "attempt_revision_id": "rev1", "code_sha": tree_a}}]
    entries = {"a1": {"pins": {"c1": "11111111"}, "module_blobs": blobs["a1"], "shards_total": 1, "shards_ok": 1, "refused": None, "standing_refusals": standing},
               "a2": {"pins": {"c1": "11111111"}, "module_blobs": blobs["a2"], "shards_total": 1, "shards_ok": 1, "refused": None, "standing_refusals": []}}
    last = lambda a: ({"utc": _now(), "tree_sha": tree_a}, entries[a]) if a in entries else None
    same = RP.changed_set(cfg, ["a1", "a2"], probed, blobs, last, tree_sha=tree_a)
    assert same["unchanged"] == ["a1", "a2"] and same["changed"] == {}
    moved = RP.changed_set(cfg, ["a1", "a2"], probed, blobs, last, tree_sha=tree_b)
    assert moved["unchanged"] == ["a2"] and list(moved["changed"]) == ["a1"], moved
    assert moved["changed"]["a1"] == [f"1 standing refusal(s) recorded under tree {tree_a[:12]}; this tree is {tree_b[:12]} -- re-read: shared reader code may have "
                                      "changed what the refusal stands on"]
    # without a tree to compare with (an older caller) the rule is silent, as before
    assert RP.changed_set(cfg, ["a1", "a2"], probed, blobs, last)["unchanged"] == ["a1", "a2"]
    # the same through the receipt store: the tick's lookup is last_receipt_for over written receipts
    rd = _rd(tmp_path)
    RP.write_receipt(rd, {"schema": RP.RECEIPT_SCHEMA, "op": {"op_id": "W1", "owner": "window"}, "utc": "2026-10-04T11:00:00Z", "tree_sha": tree_a,
                          "adapters": {"a1": entries["a1"]}})
    recs = RP.receipts(rd)
    cs = RP.changed_set(cfg, ["a1"], probed, blobs, lambda a: RP.last_receipt_for(rd, a, recs), tree_sha=tree_b)
    assert "a1" in cs["changed"] and cs["changed"]["a1"][0].startswith("1 standing refusal(s) recorded under tree")


def test_cluster_reads_changed_resolves_the_landing_to_the_commit_and_lists_a_standing_refusal_receipt_from_another_tree_as_the_tick_does(tmp_path, monkeypatch, capsys):
    """Review round 1 MEDIUM: the tick handed `changed_set` its tree, but `cluster-reads changed --landing REF` passed none -- so after a
    landing that changed SHARED reader code the command (whose listing the operator passes to the window as its adapter selection) printed
    no adapter for a receipt carrying standing refusals, while the tick re-read it; the window then completed without rechecking the
    refusal. Now the command resolves `--landing` (default HEAD) to the commit through the tick's own resolver (`resolve_tree_sha`), reads
    the module blobs at it and hands it to `changed_set`: both routes, over ONE receipt store and ONE tree, list the adapter for the same
    reason; a receipt recorded under that very tree is unchanged on both (so `HEAD` is compared as the commit it names, never as the
    literal); a ref that is no commit is refused by name."""
    w = _world_with_generation(tmp_path, monkeypatch)
    sha = _tree(w)                                              # the repository's HEAD: the tree both routes read with, present on the local cluster
    rd = _rd(tmp_path)
    state = str(tmp_path / "state")
    blobs = RP.module_blobs(REPO_ROOT, [ADAPTER], ref=sha)      # what the command reads at the landing; the tick is handed the same
    standing = [{"group": "g1", "revision_id": "rev1", "state": "standing_refusal", "refusal": "synthetic read failure",
                 "prior_refusal": {"window": "W0", "utc": "2026-10-04T08:00:00.000Z", "digest": "f" * 64, "attempt_revision_id": "rev1", "code_sha": "a" * 40}}]
    entry = {"pins": {"synthetic_stores": w["sha8"]}, "module_blobs": blobs[ADAPTER], "shards_total": 2, "shards_ok": 2, "refused": None,
             "read_whole": True, "standing_refusals": standing}
    RP.write_receipt(rd, {"schema": RP.RECEIPT_SCHEMA, "op": {"op_id": "W1", "owner": "window"}, "utc": "2026-10-04T11:00:00Z", "tree_sha": "a" * 40,
                          "adapters": {ADAPTER: entry}})
    monkeypatch.setattr(ST, "load_config", lambda p=None, root=None: w["cfg"])
    reason = (f"1 standing refusal(s) recorded under tree {'a' * 12}; this tree is {sha[:12]} -- re-read: shared reader code may have changed what the "
              "refusal stands on")
    argv = ["--root", w["root"], "cluster-reads", "changed", "--state", state, "--dest", "cluster_a", "--repo-root", REPO_ROOT]
    # the command, --landing a REF: the pins and the module blob unchanged, the adapter listed for the tick's reason, the tree it resolved recorded
    assert CLI.main(argv + ["--landing", "HEAD", "--json"]) == 0
    cs = json.loads(capsys.readouterr().out)
    assert cs["tree_sha"] == sha and cs["changed"] == {ADAPTER: [reason]} and cs["skipped"] == {} and cs["unchanged"] == [], cs
    # the plain listing (the window's adapter selection reads this): the adapter, one per line
    assert CLI.main(argv + ["--landing", sha]) == 0
    assert capsys.readouterr().out.split() == [ADAPTER]
    # the tick's dry run over the same receipt store, tree and blobs: the same adapter for the same reason
    fake = FakeArray(w)

    def tick(op_id):
        return RP.Tick(w["root"], w["cfg"], w["dest"], state_dir=state, adapters=[ADAPTER], repo_root=REPO_ROOT, op_id=op_id, runner=fake.runner,
                       slurm=FakeSlurm(), watch=fake.watch, venv="/shared/envs/v", now=_now, log=lambda s: None, sleep=lambda s: None, cancel_poll_s=0,
                       uri_home=w["home"], blobs=blobs, dry_run=True)

    assert tick("d1").run() == 0
    s1 = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert s1["state"] == "dry_run" and s1["changed"] == [ADAPTER] and s1["tree_sha"] == sha, s1
    assert RP.load_op(rd, "d1")["changed"] == {ADAPTER: [reason]}
    # a newer whole receipt recorded under THIS tree: `--landing HEAD` is compared as the commit it names -> unchanged on both routes
    RP.write_receipt(rd, {"schema": RP.RECEIPT_SCHEMA, "op": {"op_id": "W2", "owner": "window"}, "utc": "2026-10-04T12:00:00Z", "tree_sha": sha,
                          "adapters": {ADAPTER: entry}})
    assert CLI.main(argv + ["--landing", "HEAD", "--json"]) == 0
    cs2 = json.loads(capsys.readouterr().out)
    assert cs2["changed"] == {} and cs2["unchanged"] == [ADAPTER] and cs2["tree_sha"] == sha, cs2
    assert CLI.main(argv) == 0 and capsys.readouterr().out.strip() == ""          # no --landing: HEAD, the same commit
    assert tick("d2").run() == 0
    s2 = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert s2["changed"] == [] and s2["tree_sha"] == sha and RP.load_op(rd, "d2")["unchanged"] == [ADAPTER], s2
    # a landing that is no commit: refused by name, nothing listed (exit 2, the CLI's error path)
    assert CLI.main(argv + ["--landing", "no-such-ref"]) == 2
    err = capsys.readouterr().err
    assert "cluster-reads changed: git rev-parse no-such-ref" in err and "failed" in err, err
