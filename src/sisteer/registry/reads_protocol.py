"""One durable ownership protocol for every cluster read operation.

A TICK (``registry cluster-reads tick``, run every 10 min by ``registry-cluster-reads.timer`` -- the unit that replaces the retired
``registry-adapters.timer``) and a landing WINDOW (``scripts/registry/landing/window.sh``) are both READ OPERATIONS over one cluster
namespace, and they share one record of who owns it: ``$STATE/reads/`` (``$STATE`` = ``REGISTRY_LANDING_STATE``, default
``~/.local/state/registry-landing``):

    reservation.json      the ONE holder: {owner: tick | window, op_id, started_utc, host, pid, tree_sha, job}; created with O_EXCL
    ops/<op_id>.json      every operation's record: its state, its Slurm job ids (reads-<op_id>, attest-<op_id>), who cancelled it
    receipts/<op_id>.json what an operation READ: per adapter the pins, each pin's PUBLISHED record (``published_utc`` / ``change_utc``,
                          the S7 clock), the adapter module's blob sha, shards ok / total, the revisions pulled, the refusal

The protocol, in the order every operation runs it:

1. RESERVATION FIRST. ``reservation.json`` is written with ``O_EXCL`` BEFORE anything is exported or submitted; the Slurm job names
   ``reads-<op_id>`` / ``attest-<op_id>`` are fixed in advance, so a job whose record write was lost is found again by name. A holder
   exists -> a tick YIELDS (exit 0, nothing read); a window TAKES OVER (4). THE HOLDER IS THE LONG-LIVED PROCESS: the tick records its
   own pid; the window, which reserves through a short-lived CLI, passes its shell's ``$$`` (``reserve --pid``), verified alive, with a
   readable ``/proc`` start time, and a proper ancestor of the CLI -- and the record carries the pid WITH its start time (``pid_start``),
   so a pid reused by another process reads as gone. A holder whose process is gone from this host (or whose pid was reused) is STALE:
   its jobs are cancelled and confirmed terminal, its record says who replaced it, and the reservation is taken. A reservation that names
   THIS operation id is "held by me" only when its whole holder identity matches (owner, op id, host, pid, pid start time); the same id
   under another process -- a second window with the same WINDOW_ID -- is refused by name (:class:`OperationIdInUse`), never adopted,
   never taken over, never replaced as stale.
2. RECONCILE BEFORE SUBMIT. Every ``reads-*`` / ``attest-*`` job of this user in ``squeue`` and 24 h of ``sacct`` is compared with the
   op records: an outstanding job with no live owner is cancelled and confirmed TERMINAL in ``sacct`` before any new array touches the
   shared node-local namespace (:class:`ReconcileRefusal` otherwise: nothing is submitted); a COMPLETED array whose operation never
   pulled is pulled now (the caller's pull sequence, every barrier of the pull applying) or left to the barrier. TERMINAL CONFIRMATION
   means, per submitted job id, an EXPLICIT terminal state in the accounting (a ``sacct`` row that matches the terminal set) AND the
   queue's silence, checked separately, from a batch in which BOTH queries succeeded (each query's exit status travels in the answer; a
   failed ``squeue`` is never an empty queue, a failed ``sacct`` never a missing row -- :class:`SlurmQueryFailed`, and the wait goes on to
   its bound): a job the accounting has no row for is not terminal, however empty the queue, and past the bound the refusal names it
   (``no accounting row``, or the query that failed).
3. SUBMIT, RECORD, WATCH. The ownership check, both ``sbatch`` calls and the durable write of the job ids to the op record are ONE
   critical-section interval (seconds: ``submit_section_s`` on the record), so a takeover that arrives during a submission waits for
   it and then always finds the recorded ids to cancel and confirm; the section is never held while the array waits in the queue or
   runs. ``watch-job.sh --poll 30 --stall-min 0`` then runs in the foreground under a bound of the array's walltime + 15 min (a
   watcher timeout leaves the array LIVE: the tick KEEPS the reservation, exit 21, and the next operation -- finding a stale holder once
   this process is gone -- cancels and confirms the array before anything new is submitted). A terminal watch verdict is followed by
   the module's OWN confirmation of every recorded id (2.), and only that clears the way to a release. THE RULE: ownership
   is released only after every recorded job of the operation is confirmed terminal -- from the first recorded id the tick keeps the
   reservation through every exit (a watch timeout, an unconfirmed self-cancel, a failed journal write, an exception the code did not
   foresee) and releases only after its own confirmation or a confirmed self-cancel. A self-cancel this operation cannot confirm (a
   refused or TIMED-OUT scancel or query -- the transport's exceptions are one family, :class:`SlurmQueryFailed` -- or no terminal
   accounting row within the bound) is FAIL-CLOSED: the record says ``cancel_failed`` with the job ids (a best-effort write), the
   reservation is KEPT, the tick exits 27 (the unit is red).
4. WINDOW TAKEOVER. The window cancels the tick's array by its recorded id (recovered by name when the record has none), waits for
   ``sacct`` TERMINAL, marks the tick's record ``cancelled_by``, then takes the reservation; the tick's own watch sees CANCELLED, pulls
   nothing and exits 20. The pre-submit race is closed by the submit section (3): the window's takeover either lands before it (the
   tick's check inside the section reads the loss and submits nothing) or after it (the ids are recorded and the takeover cancels and
   confirms them); the tick's check after the section spares its watch.
5. OWNERSHIP RE-CHECKED BEFORE PULL / MERGE, INSIDE THE CRITICAL SECTION. One ``flock`` under ``$STATE/reads/`` (``critical.lock``,
   :func:`critical_section`) is held by every hand-over of the reservation (a takeover, a stale replacement, the O_EXCL take), by every
   reconcile, and by every pull / merge / receipt from the ownership check that opens it to the receipt that closes it (the tick's
   in-process; ``cluster-reads pull --op`` and ``cluster-reads receipt`` for the window). So a takeover that arrives while a pull is in
   progress WAITS for the pull, the merge and the receipt (bound 30 min, then it refuses by name: nothing changes hands under a merge),
   and a reservation that stopped naming the operation stops it before anything is merged (``submit --op`` and ``pull --op`` refuse by
   name, exit 23; the tick exits 20 when a takeover is recorded). The RELEASE runs inside the same section: the reservation is re-read
   under the lock and unlinked only when it still names the releasing operation, so a takeover cannot slip in between the check and the
   unlink and lose its reservation to the old holder's release. The section is held in INTERVALS, never across a queue wait: in the
   tick the submit interval (3) and the pull interval (check -> pull -> merge -> receipt); in the window ``pull --op`` (check ->
   transfer -> merge, the two stages recorded) and ``receipt`` (check -> receipt; a merge the plan does not record runs first);
   plus every reconcile and every hand-over.
6. READ RECEIPTS. The grading host's half is THREE RESUMABLE STAGES, each persisted on the plan file and mirrored on the op record when it
   completes: ``pull`` (state ``pulled``) -> ``inbox_merge`` (``merged``) -> ``receipt`` (``receipted``). A reconcile that takes a
   completed-unreceipted operation resumes at the FIRST missing stage and never skips to the receipt; NO RECEIPT WITHOUT A RECORDED
   MERGE OF THIS OPERATION'S PULL (invariant I7, :class:`ReceiptRefused`), and the merge's outcome is read from DURABLE state: the pull
   records every revision directory the operation DEPENDS ON (``depends_dirs``: newly copied, with the shard's name -> sha256; an earlier
   complete delivery reused from the inbox, with its own), and :func:`merge_evidence` reads ``records/`` and the inbox against it after
   every merge attempt and again right before the receipt -- never ``scan_inbox``'s return value. A revision not durably merged
   (``conflict``: a ``<file>.conflict-<sha8>`` sibling or other bytes under its name; ``incomplete``: a file missing or the directory
   still in the inbox) REFUSES its adapter in the receipt (``merge_conflict`` / ``merge_incomplete``), so ``read_whole`` is never claimed
   over it and the changed set reads that adapter again. ``last_read`` of an adapter is DERIVED from the newest receipt that read it
   whole (every shard ok, no refusal, no refused-known id the pull left STANDING unread -- ``refused_standing``), and the receipts
   feed the changed set (below) and S7.
7. NO DOUBLE PUBLISH. The pull's barriers admit a revision the grading host does not already hold (``reader_state.published_dirs``, the
   shard's known-id refusal); a second array over the same reader state is prevented by the reservation; a completed-unpulled operation
   is pulled ONCE (its receipt is the proof: an operation with a receipt is never pulled again by a reconcile).

THE TICK: reservation -> reconcile -> ONE ssh batch (:func:`probe`: ``readlink current`` + ``PUBLISHED-<sha8>.json`` per component,
the base tree present?) -> the CHANGED SET (:func:`changed_set`): an adapter whose pin moved since its last whole receipt, or whose
module's blob sha at the base tree differs from the receipt's, or that has no receipt at all -> nothing changed -> release, exit 0
(seconds); else ``reader-state export --adapter <changed>`` -> plan (manifests and declarations only; the probe's pins) -> stage ->
``attest.sbatch`` -> the array -> watch -> revalidate (O(1) for ``cluster_a`` components) -> pull (+ the refused-known re-activations
from committed bytes) -> inbox-merge -> receipt -> release. Exit codes: 0 read (or nothing to read, or yielded), 15 the read recorded
REFUSED adapters (named; the receipt carries the reasons -- the tick refuses every 10 min with the cause until it is fixed), 20
cancelled by a window's takeover, 21 the watcher timed out (the array is live: the reservation is KEPT for the next operation to cancel
and confirm), 22 an outstanding job could not be confirmed terminal (nothing submitted), 23 the reservation stopped naming this operation
with no recorded cancel, 25 interrupted
(SIGTERM: the unit's bound; the reservation released, the record says so), 26 the critical section stayed busy past its bound (the
operation is handed to the next reconcile, like a watcher timeout), 27 FAIL-CLOSED: this operation's own jobs could not be cancelled and
confirmed terminal -- the record says ``cancel_failed`` with the ids, the reservation is KEPT for the next operation to cancel and confirm
them before it submits anything, and the unit is red (never 20).

OWNERSHIP INVARIANTS -- the list a reviewer checks line by line:

I1. ONE FILE DECIDES. ``reservation.json`` is the sole authority of who owns the namespace. The op records (``ops/``), the receipts and
    the lock file are journals: they never decide ownership, and a lost or late journal write changes no decision.
I2. WHO MAY WRITE OR REMOVE IT, UNDER WHICH LOCK. CREATED only by :func:`reserve` with ``O_EXCL`` (the create IS the check: atomic).
    REPLACED only by :func:`take_reservation`, inside :func:`critical_section`, after the holder was RE-READ under the lock, found a
    tick's (``takeover``) or stale, and its jobs were confirmed terminal (I5). UNLINKED only by :func:`release`, inside the same
    section, after the reservation was RE-READ under the lock and still names the releasing operation -- AND, for the tick, only once
    every recorded job of the operation is confirmed terminal (I5): from the first recorded job id the tick keeps the reservation
    through every exit (``_keep_reservation``, set before the id's record write and cleared only by its own confirmation of every id
    or a confirmed self-cancel), so a watch timeout, an unconfirmed cancel, a failed journal write or an exception the code did not
    foresee all leave it standing for the next operation. No other function writes or removes the file (:func:`replace_reservation` is
    the takeover's write and is called from inside the lock only).
I3. WHAT IDENTIFIES A HOLDER. Five fields, all equal or not the same holder: owner, op id, host, pid, and the pid's ``/proc`` start
    time (:func:`holder_identity`, :func:`same_holder`). The op id alone is not an identity: the same id under another process is
    :class:`OperationIdInUse` -- refused by name, never ``held_by_me``, never taken over, never replaced as stale. The holder is the
    LONG-LIVED process (the tick itself; the window SHELL through ``reserve --pid $$``, verified a live proper ancestor of the CLI).
    ALIVE = on this host a process with that pid exists AND its start time equals the record's; STALE = the pid is gone or reused; a
    holder on another host is neither (never replaced).
I4. WHO MAY TAKE IT. A tick: nobody's but a stale holder's (it yields to every live holder). A window (``takeover``): a tick's or a
    stale holder's -- never a live window's, never the same operation id's. Taking means: EVERY job of the holder -- the ids its record
    carries plus every queued or accounted job named ``reads-<op>`` / ``attest-<op>`` -- cancelled and confirmed terminal (I5), its
    record marked, then the write (I2). The taker always sees the holder's ids: they are recorded inside the holder's submit section
    (I6), so no job of the holder is in flight unrecorded when the section is the taker's. A holder that could not confirm its own
    cancel keeps the reservation (``cancel_failed``, exit 27) precisely so that the taker does this.
I5. WHAT PROVES A JOB TERMINAL. Per job id, from ONE batch in which BOTH ``squeue`` and ``sacct`` succeeded: an EXPLICIT accounting row
    in the terminal set AND no row in the queue (``COMPLETING`` is queued). Absence is never terminal; a failed query is never an
    answer; a job the accounting has no row for is not terminal however silent the queue. Nothing changes hands and nothing is
    submitted until every outstanding id is proven so (``ReconcileRefusal`` past the bound names each id and why). Every query and
    every ``scancel`` goes through ONE shell call whose transport timeout or failure is :class:`SlurmQueryFailed` (a
    ``ReconcileRefusal``) -- one exception family, so a stalled ssh lands in the same fail-closed path as a refused query.
I6. WHAT THE LOCK COVERS, IN WHICH INTERVALS. Every write or unlink of the reservation (I2); every reconcile (its first act re-reads the
    reservation under the lock); and the grading-host-side work in INTERVALS, each opened by an ownership check under the lock: in the TICK,
    the submit interval (check -> ``sbatch`` attest -> ``sbatch`` array -> the job ids written to the record; seconds, never the queue
    wait or the run) and the pull interval (check -> pull -> merge -> receipt, one interval); in the WINDOW, ``cluster-reads pull --op``
    (check -> transfer -> merge; the ``pull`` and ``inbox_merge`` stages recorded) and ``cluster-reads receipt`` (check -> receipt, a
    merge the plan does not record first) -- two intervals, so a takeover may land between them: what makes that safe is I7 (the
    receipt's precondition is recorded state, not the interval) and the takeover's own cancel-and-confirm of recorded ids. A read
    outside the lock may decide only a NON-WRITE: a tick's yield, a held-by-me answer, a release that finds another holder and removes
    nothing, the tick's post-submit check (a loss read there spares the watch; the jobs are the taker's to cancel, and were).
I7. NO RECEIPT WITHOUT A RECORDED MERGE OF THIS OPERATION'S PULL, AND THE RECEIPT'S PRECONDITION IS A PURE FUNCTION OF DURABLE STATE.
    The grading host half is three stages persisted one by one on the plan file and mirrored on the op record -- ``pull`` (``pulled``),
    ``inbox_merge`` (``merged``), ``receipt`` (``receipted``); a resume (a reconcile's orphan pull, the window's ``receipt``) starts at
    the FIRST stage the plan does not record and never skips one; the receipt is written only when the plan records both ``pull`` and
    ``inbox_merge`` (:class:`ReceiptRefused` otherwise). What a receipt PROVES: every revision this operation DEPENDS ON -- the pull
    stage's ``depends_dirs``: newly copied (the shard's name -> sha256, verified on receipt) AND an earlier complete delivery reused from
    the inbox (its own name -> sha256) -- is DURABLY MERGED, every recorded file present in ``records/adapters/<rev>/`` with its sha256
    and nothing of it left in the inbox, or its adapter is refused in the receipt (``merge_conflict``: a ``<file>.conflict-<sha8>``
    sibling or other bytes under its name; ``merge_incomplete``: a file missing or the directory still in the inbox). The evidence is
    :func:`merge_evidence`, read from ``records/`` and the inbox after every merge attempt (first or resumed) and again right before the
    receipt, never ``scan_inbox``'s return value (an interrupted scan has already moved a conflict aside and cannot report it again) --
    so ``read_whole`` is never claimed over a revision that is not in ``records/`` as the operation recorded it, and the changed set may
    rest on a whole receipt. A recorded file counts under either representation the registry keeps -- ``<name>.jsonl`` as the shard
    named it (its raw bytes) or the harvester's ``<name>.jsonl.gz`` twin (its decompressed bytes, :func:`R.file_logical_sha`) -- so the
    harvester compressing an accepted record between a merge and its resumed receipt changes nothing; but the two forms are ONE logical
    record, so both present and differing is a conflict whichever matches (the inbox merge installs no second active form: a same-name
    arrival beside either form is a duplicate or a ``.conflict-`` aside by logical sha).

Rehearsed by ``tests/test_registry_reads_protocol.py`` with a Slurm shim (squeue / sacct / scancel through the runner), the plan's four
races among them: the pre-submit race, a watcher timeout with a live array, a restart after submit, a cancellation after the computation
and before the pull; a takeover DURING the pull (it waits for the receipt), a real ``reserve --pid $$`` subprocess that exits while its
shell lives on (the reservation stays live), and a cancel whose job the accounting has no row for (refused); a takeover paused against a
release (the release's check and unlink are one locked step; the takeover lands after and keeps its reservation), the same operation id
under a second live shell (refused, nothing written; the same shell: held by me), and a failed ``squeue`` beside a terminal accounting
row (nothing confirmed until both queries answer); a tick that dies right after persisting ``pull`` (the next reconcile merges, then
receipts; no receipt without the merge), a takeover arriving during the submit section (it waits, then cancels and confirms the recorded
ids), a self-cancel that cannot be confirmed (``cancel_failed``, exit 27, the reservation kept, the window's next reconcile cancels and
confirms before it submits), and concurrent writers of one op record losing nothing; a transport timeout on the self-cancel's ``scancel``
(one exception family: fail-closed, the reservation kept), a journal write that fails inside the fail-closed path (the outcome stands),
an exception the code did not foresee after the submit (the reservation kept, the record says the cancel was not attempted), a watch
timeout keeping the reservation (the next operation replaces the stale holder after cancelling and confirming the array), and a
marker-less or conflicting OWN revision at the merge (the adapter refused in the receipt, ``read_whole`` false, the adapter read again);
a conflict moved aside by an interrupted merge (the resumed merge's evidence comes from ``records/``: refused, no whole receipt), an
earlier complete delivery reused by this operation with a conflicting accepted record (in ``depends_dirs`` though not in
``pulled_dirs``: refused), and the durable-merge check itself (merged only when every recorded file is in ``records/`` with its sha256);
the harvester compressing the merged records (``gzip_after_ingest``) before a resumed merge or receipt: the ``.gz`` twin counts by its
logical sha, the evidence stays merged and the adapter read whole; a delayed same-name conflicting delivery before and after the fold
(the inbox merge keeps it aside as ``.conflict-`` by logical sha, never a second active form; the operation depending on it gets no
whole receipt) and two differing active forms of one record read as a conflict.
"""
from __future__ import annotations

import contextlib
import fcntl
import getpass
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

from . import cluster_reads as CR
from . import db as DB
from . import harvest as H
from . import records as R
from . import stores as ST
from .mirrors import RSYNC_SSH, SSH_OPTS

RESERVATION_SCHEMA = "reads_reservation/1"
OP_SCHEMA = "reads_op/1"
RECEIPT_SCHEMA = "reads_receipt/1"
STATE_ENV = "REGISTRY_LANDING_STATE"
DEFAULT_STATE = os.path.join(os.path.expanduser("~"), ".local", "state", "registry-landing")
READS_SUBDIR = "reads"
RESERVATION_FILE = "reservation.json"
OPS_SUBDIR = "ops"
RECEIPTS_SUBDIR = "receipts"
SIDECARS_SUBDIR = "sidecars"
#: the CRITICAL SECTION of the namespace: one ``flock`` under ``$STATE/reads/`` that every hand-over of the
#: reservation (a takeover, a stale replacement, the O_EXCL take), every reconcile and every pull / merge / receipt runs inside, so a
#: takeover waits for a pull in progress instead of landing under it
CRITICAL_LOCK_FILE = "critical.lock"
CRITICAL_BOUND_S = 30 * 60.0    # how long a taker waits for the section (a pull in progress) before refusing by name
CRITICAL_POLL_S = 1.0
#: the two job kinds of one operation and their fixed names: ``reads-<op_id>`` (the array), ``attest-<op_id>`` (the attestation)
JOB_KINDS = ("reads", "attest")
#: sacct states that end a job (the same set the job-watch script and baseline_refresh.sh read)
TERMINAL_RE = re.compile(r"^(COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|BOOT_FAIL|DEADLINE)")
#: the per-query exit-status marker each cluster query is followed by in one shell: ``---SQUEUE-RC=<n>---`` closes the queue's lines,
#: ``---SACCT-RC=<n>---`` the accounting's; a missing marker or a non-zero status is :class:`SlurmQueryFailed`
_RC_MARKER = re.compile(r"^---(SQUEUE|SACCT)-RC=(\d+)---$")
#: op record states after which nothing of the operation is outstanding
OP_TERMINAL_STATES = ("receipted", "cancelled", "yielded", "unchanged", "failed", "reconciled", "ownership_lost", "interrupted", "dry_run")
#: a record in one of these states has HANDED its jobs on (the process may still be alive: it gave up watching, or the critical section
#: stayed busy past its bound when it came to pull). A ``watch_timeout`` KEEPS the reservation (ownership is released only after every
#: recorded job is confirmed terminal), so the hand-over is the next operation's stale replacement once the process is gone -- the
#: reconcile's cancel of a handed-off record's jobs is the belt for a reservation that is somehow gone
OP_HANDED_OFF_STATES = ("watch_timeout", "section_busy")
#: states under which a completed array is never pulled by a reconcile: no array was submitted, or the pull already happened / was settled
OP_NO_PULL_STATES = ("receipted", "reconciled", "failed", "unchanged", "yielded", "dry_run")
#: the adapter modules whose blob sha the changed set compares (the registry's adapters package, relative to the repository root)
ADAPTERS_REL = "src/sisteer/registry/adapters"
#: the array's walltime (read_array.sbatch --time) + 15 min: the tick's watch bound
ARRAY_TIME_S = 25 * 60
WATCH_BOUND_S = ARRAY_TIME_S + 15 * 60
CANCEL_BOUND_S = 180.0          # how long a cancel waits for sacct to read TERMINAL before refusing by name
CANCEL_POLL_S = 5.0
RECONCILE_SINCE = "now-24hours"

EXIT_OK = 0
EXIT_REFUSED = 15
EXIT_CANCELLED = 20
EXIT_WATCH_TIMEOUT = 21
EXIT_RECONCILE = 22
EXIT_OWNERSHIP = 23
EXIT_HELD = 24
EXIT_INTERRUPTED = 25
EXIT_SECTION_BUSY = 26
#: FAIL-CLOSED: this operation's own jobs could not be cancelled and confirmed terminal (scancel refused, no terminal accounting row
#: within the bound, a failed query): the record says ``cancel_failed`` with the job ids, the reservation is KEPT (never released over
#: live jobs: the next operation finds a stale holder, cancels and confirms its jobs before anything new is submitted) and the unit is red
EXIT_CANCEL_FAILED = 27
#: the three resumable stages of the grading host's half, in order, and the op-record state each one leaves: a reconcile that takes a
#: completed-unreceipted operation resumes at the FIRST missing stage and never skips to the receipt
OP_STAGES = (("pull", "pulled"), ("inbox_merge", "merged"), ("receipt", "receipted"))
OP_STAGE_STATE = dict(OP_STAGES)
#: the merge EVIDENCE states a revision this operation depends on can be in, read from DURABLE state (invariant I7: never from
#: ``scan_inbox``'s return value, which an interrupted scan loses): ``merged`` -- every recorded file is in ``records/`` with its sha256 and
#: nothing of it is left in the inbox; ``conflict`` -- a ``<file>.conflict-<sha8>`` sibling, or other bytes under the file's name, in
#: ``records/``; ``incomplete`` -- a file missing from ``records/`` or the directory still in the inbox. The two problem states refuse the
#: adapter in the receipt (``merge_conflict`` / ``merge_incomplete``) and ``read_whole`` is never claimed for it
MERGE_PROBLEM_STATES = ("incomplete", "conflict")


class HolderPidInvalid(ValueError):
    """The pid offered as the reservation's holder is not a live, proper ancestor of the process reserving on its behalf: a reservation
    taken through a short-lived CLI must name the long-lived process -- the window shell -- or nothing."""


class ReservationHeld(RuntimeError):
    """A LIVE holder owns the reservation and the caller may not take it over (a tick yields; a window refuses by name)."""

    def __init__(self, holder: Mapping[str, Any], message: Optional[str] = None):
        self.holder = dict(holder)
        super().__init__(message or f"the cluster-reads reservation is held by {holder.get('owner')} {holder.get('op_id')} "
                                    f"(since {holder.get('started_utc')}, {holder.get('host')}:{holder.get('pid')})")


class OperationIdInUse(ReservationHeld):
    """The reservation names THIS operation id but ANOTHER holder identity: a second window started with the same WINDOW_ID, an
    operator retry while the first still runs. It is never ``held_by_me``, never taken over, never replaced as stale -- the two would
    share the plan file and the job names ``reads-<op>`` / ``attest-<op>`` and pull or receipt against each other. Refused by name; a
    retry needs a NEW operation id."""

    def __init__(self, holder: Mapping[str, Any], me: Mapping[str, Any]):
        super().__init__(holder, f"the operation id {holder.get('op_id')} is already held by another process: {holder.get('owner')} on "
                                 f"{holder.get('host')} pid {holder.get('pid')} (started {holder.get('pid_start')}, since {holder.get('started_utc')}); "
                                 f"this caller is {me.get('owner')} on {me.get('host')} pid {me.get('pid')} (started {me.get('pid_start')}). The same "
                                 f"operation id under two processes would share the plan and the job names; a retry needs a new operation id")
        self.me = dict(me)


class ReconcileRefusal(RuntimeError):
    """An outstanding job could not be confirmed terminal: nothing may touch the shared namespace until it is."""


class SlurmQueryFailed(ReconcileRefusal):
    """One of the two cluster queries (``squeue`` / ``sacct``) failed or the transport did: its answer is NOT an empty answer. A failed
    queue query is never read as the queue's silence, a failed accounting query never as "no row"; nothing is confirmed terminal from a
    batch in which either query failed."""

    def __init__(self, what: str, where: str, rc: Any, stderr: str):
        self.what, self.where, self.rc = what, where, rc
        super().__init__(f"{what} on {where} failed (rc {rc}): {(stderr or '').strip()[:300] or 'no stderr'}; a failed query proves nothing "
                         f"(it is not an empty queue or a missing accounting row)")


class CriticalSectionBusy(ReconcileRefusal):
    """The namespace's critical section stayed held past the bound (a pull / merge / receipt in progress): nothing changes hands."""

    def __init__(self, holder: Optional[Mapping[str, Any]], waited_s: float, what: str):
        self.holder = dict(holder) if holder else None
        who = (f"{holder.get('owner')} {holder.get('op_id')} ({holder.get('what')}, pid {holder.get('pid')}, since {holder.get('utc')})"
               if holder else "an unrecorded holder")
        super().__init__(f"the cluster-reads critical section is held by {who} after {int(waited_s)} s; {what} waits for it to end "
                         f"(nothing changes hands under a pull, merge or receipt in progress)")


class OwnershipLost(RuntimeError):
    """The reservation stopped naming this operation."""

    def __init__(self, op_id: str, holder: Optional[Mapping[str, Any]], cancelled_by: Optional[str] = None):
        self.op_id, self.holder, self.cancelled_by = op_id, dict(holder) if holder else None, cancelled_by
        who = f"{holder.get('owner')} {holder.get('op_id')}" if holder else "nobody"
        super().__init__(f"the reservation names {who}, not {op_id}" + (f"; cancelled by {cancelled_by}" if cancelled_by else ""))


class TickInterrupted(RuntimeError):
    """SIGTERM (the unit's TimeoutStartSec, a stop): the tick releases what it holds and records why it stopped."""


class ReceiptRefused(CR.ClusterReadsError):
    """Invariant I7: no receipt without a recorded merge of this operation's pull. The plan file must record the ``pull`` AND the
    ``inbox_merge`` stage before a receipt is written -- a receipt over an unmerged pull would read the adapter as read whole while its
    records sit in the inbox, and the changed set might never schedule that pin again."""


# --------------------------------------------------------------------------------------------
# the namespace on disk
# --------------------------------------------------------------------------------------------


def reads_dir(state_dir: Optional[str] = None) -> str:
    base = state_dir or os.environ.get(STATE_ENV) or DEFAULT_STATE
    return os.path.join(os.path.abspath(os.path.expanduser(base)), READS_SUBDIR)


def reservation_path(rd: str) -> str:
    return os.path.join(rd, RESERVATION_FILE)


def op_path(rd: str, op_id: str) -> str:
    return os.path.join(rd, OPS_SUBDIR, f"{op_id}.json")


def receipt_path(rd: str, op_id: str) -> str:
    return os.path.join(rd, RECEIPTS_SUBDIR, f"{op_id}.json")


def sidecar_path(rd: str, op_id: str) -> str:
    return os.path.join(rd, SIDECARS_SUBDIR, f"reader_state-{op_id}.sqlite")


def job_name(kind: str, op_id: str) -> str:
    return f"{kind}-{op_id}"


def op_of_job_name(name: str) -> Optional[Tuple[str, str]]:
    """``(kind, op_id)`` of a job named by this protocol, else None."""
    for kind in JOB_KINDS:
        if name.startswith(kind + "-") and len(name) > len(kind) + 1:
            return kind, name[len(kind) + 1:]
    return None


def _load(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        return doc if isinstance(doc, dict) else None
    except (OSError, ValueError):
        return None


def _write(path: str, doc: Mapping[str, Any]) -> None:
    ST._write_json(path, doc)


def read_reservation(rd: str) -> Optional[Dict[str, Any]]:
    return _load(reservation_path(rd))


def _proc_stat(pid: int) -> Optional[List[str]]:
    """The fields of ``/proc/<pid>/stat`` AFTER the ``comm`` field (which may hold spaces and parentheses): index 0 = state, 1 = ppid, ...,
    19 = starttime (the man page's field 22). None when there is no such process or ``/proc`` cannot be read."""
    try:
        with open(f"/proc/{int(pid)}/stat", "rb") as fh:
            raw = fh.read()
    except (OSError, ValueError):
        return None
    tail = raw[raw.rfind(b")") + 2:].split()
    return [t.decode(errors="replace") for t in tail] if len(tail) > 19 else None


def process_start(pid: int) -> Optional[str]:
    """The process's start time in clock ticks since boot (``/proc/<pid>/stat`` starttime): with the pid, an identity a REUSED pid does not
    share. None when the process is gone or ``/proc`` is unreadable."""
    f = _proc_stat(pid)
    return f[19] if f else None


def process_parent(pid: int) -> Optional[int]:
    f = _proc_stat(pid)
    try:
        return int(f[1]) if f else None
    except ValueError:
        return None


def verify_holder_pid(pid: int, me: Optional[int] = None) -> Tuple[int, str]:
    """The pid a reservation may name as its holder when the reserving process is NOT the holder (the window: ``reserve --pid $$``): it must
    be alive on this host, its start time readable, and a PROPER ANCESTOR of this process -- the shell that invoked the CLI, never the CLI
    itself or an unrelated live process. Returns ``(pid, start)``; raises :class:`HolderPidInvalid` by name otherwise."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        raise HolderPidInvalid(f"--pid {pid!r} is not a process id")
    if pid <= 1:
        raise HolderPidInvalid(f"--pid {pid}: not a holder (init or no process)")
    me = os.getpid() if me is None else me
    if pid == me:
        raise HolderPidInvalid(f"--pid {pid} is this CLI process itself, which exits when the command returns; name the long-lived shell ($$)")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        raise HolderPidInvalid(f"--pid {pid}: no such process on {socket.gethostname()}")
    except PermissionError:
        pass
    start = process_start(pid)
    if start is None:
        raise HolderPidInvalid(f"--pid {pid}: /proc/{pid}/stat unreadable; the holder's identity cannot be recorded")
    p: Optional[int] = me
    seen = set()
    while p and p > 1 and p not in seen:
        seen.add(p)
        p = process_parent(p)
        if p == pid:
            return pid, start
    raise HolderPidInvalid(f"--pid {pid} is not an ancestor of this process (pid {me}); the holder must be the shell that runs the operation")


def holder_alive(res: Mapping[str, Any]) -> Optional[bool]:
    """True / False when the holder's host is this one; None when it cannot be told (another host, no pid). Alive = the pid exists
    (``os.kill(pid, 0)``) AND, when the record carries ``pid_start``, the process at that pid started when the record says -- a pid reused by
    another process since the holder died reads as GONE."""
    if str(res.get("host") or "") != socket.gethostname():
        return None
    try:
        pid = int(res.get("pid") or 0)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    recorded = res.get("pid_start")
    if recorded:
        now_start = process_start(pid)
        if now_start is not None and str(now_start) != str(recorded):
            return False
    return True


def holder_identity(owner: str, op_id: str, pid: Optional[int] = None, pid_start: Optional[str] = None) -> Dict[str, Any]:
    """WHAT IDENTIFIES A HOLDER (invariant I3): owner, op id, host, pid and the pid's start time -- the five fields a reservation is compared
    on. Default: this process (the tick); ``pid`` / ``pid_start`` name the long-lived holder a CLI reserves for (the window shell)."""
    pid = int(pid) if pid else os.getpid()
    return {"owner": owner, "op_id": op_id, "host": socket.gethostname(), "pid": pid,
            "pid_start": pid_start if pid_start is not None else process_start(pid)}


def same_holder(res: Mapping[str, Any], me: Mapping[str, Any]) -> bool:
    """Whether the reservation ``res`` names the holder identity ``me`` -- every one of the five fields equal, compared as strings (the
    op id alone is NOT an identity; two processes with one operation id are two holders)."""
    return all(str(res.get(k)) == str(me.get(k)) for k in ("owner", "op_id", "host", "pid", "pid_start"))


def _reservation(owner: str, op_id: str, tree_sha: Optional[str], now: Callable[[], str], pid: Optional[int] = None,
                 pid_start: Optional[str] = None) -> Dict[str, Any]:
    me = holder_identity(owner, op_id, pid=pid, pid_start=pid_start)
    return {"schema": RESERVATION_SCHEMA, "owner": owner, "op_id": op_id, "started_utc": now(), "host": me["host"], "pid": me["pid"],
            "pid_start": me["pid_start"], "tree_sha": tree_sha, "job": None}


def reserve(rd: str, owner: str, op_id: str, tree_sha: Optional[str] = None, now: Callable[[], str] = R.utc_now, pid: Optional[int] = None,
            pid_start: Optional[str] = None) -> Dict[str, Any]:
    """Take the reservation with ``O_EXCL`` -- the first step of every operation. Raises :class:`ReservationHeld` when one exists. ``pid``
    (with ``pid_start``) names the LONG-LIVED holder when the caller reserves on its behalf (the window shell through the CLI; verified by
    :func:`verify_holder_pid` there); default: this process."""
    os.makedirs(rd, exist_ok=True)
    res = _reservation(owner, op_id, tree_sha, now, pid=pid, pid_start=pid_start)
    path = reservation_path(rd)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        held = read_reservation(rd) or {"owner": "?", "op_id": "?"}
        raise ReservationHeld(held)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    update_op(rd, op_id, now=now, owner=owner, tree_sha=tree_sha, state="reserved", reserved_utc=res["started_utc"], pid=res["pid"],
              pid_start=res["pid_start"])
    return res


def critical_lock_path(rd: str) -> str:
    return os.path.join(rd, CRITICAL_LOCK_FILE)


def read_critical_holder(rd: str) -> Optional[Dict[str, Any]]:
    return _load(critical_lock_path(rd))


@contextlib.contextmanager
def critical_section(rd: str, op_id: str, owner: str, what: str, bound_s: float = CRITICAL_BOUND_S, poll_s: float = CRITICAL_POLL_S,
                     log: Callable[[str], None] = lambda s: None, now: Callable[[], str] = R.utc_now) -> Iterator[None]:
    """The namespace's critical section: an exclusive ``flock`` on ``$STATE/reads/critical.lock`` held for the
    duration of the block. Every hand-over of the reservation (:func:`take_reservation`'s write paths), every :func:`reconcile` and every
    pull / merge / receipt (the tick's, ``cluster-reads pull --op``, ``cluster-reads receipt``) runs inside it, so a takeover that arrives
    while a pull is in progress WAITS for the pull, the merge and the receipt to finish before the reservation changes hands -- and the
    ownership check at the top of each section reads a reservation nothing can replace underneath it. The holder is written into the lock
    file (owner, op, what, pid, utc) for the waiter's log line; the kernel releases the lock when the holder's process ends, so a dead holder
    never blocks anyone. Past ``bound_s`` :class:`CriticalSectionBusy` (a :class:`ReconcileRefusal`) names the holder; the wait sleeps for
    real (``time.sleep``): it is an inter-process wait, never a simulated one."""
    os.makedirs(rd, exist_ok=True)
    fd = os.open(critical_lock_path(rd), os.O_RDWR | os.O_CREAT, 0o644)
    t0 = time.monotonic()
    said = 0.0
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                waited = time.monotonic() - t0
                if waited >= bound_s:
                    raise CriticalSectionBusy(read_critical_holder(rd), waited, f"{owner} {op_id} ({what})")
                if waited - said >= 30.0 or said == 0.0:
                    h = read_critical_holder(rd) or {}
                    log(f"critical section: {owner} {op_id} ({what}) waits for {h.get('owner', '?')} {h.get('op_id', '?')} ({h.get('what', '?')}, "
                        f"pid {h.get('pid', '?')}) -- {int(waited)} s")
                    said = waited if waited > 0 else 1e-9
                time.sleep(max(0.0, float(poll_s)))
        try:
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, (json.dumps({"owner": owner, "op_id": op_id, "what": what, "pid": os.getpid(), "utc": now()}, sort_keys=True) + "\n").encode())
        except OSError:
            pass
        yield
    finally:
        try:
            os.close(fd)             # closes the open file description: the lock goes with it
        except OSError:
            pass


def replace_reservation(rd: str, res: Mapping[str, Any]) -> None:
    """The takeover's write: one rename over the holder's file (never O_EXCL: the file exists by definition)."""
    _write(reservation_path(rd), res)


def release(rd: str, op_id: str, bound_s: float = CRITICAL_BOUND_S, poll_s: float = CRITICAL_POLL_S, log: Callable[[str], None] = lambda s: None,
            now: Callable[[], str] = R.utc_now) -> bool:
    """Remove the reservation ONLY when it names ``op_id`` -- the ownership check and the unlink under ONE hold of the namespace's
    :func:`critical_section` (read outside the lock and unlinked after it, a release could delete the reservation a takeover had just
    written in between, and the next tick then reserved during the window's hour-long mirror wait). A reservation that names another
    operation is left alone without taking the lock (nothing to remove: by invariant I3 no other process can hold THIS operation's id,
    so a reservation not naming it never comes to name it). Raises :class:`CriticalSectionBusy` past ``bound_s`` (the reservation stays;
    a holder whose process is gone is replaced as stale by the next operation). True when removed."""
    res = read_reservation(rd)
    if res is None or str(res.get("op_id")) != op_id:
        return False
    with critical_section(rd, op_id, str(res.get("owner") or "?"), "release", bound_s=bound_s, poll_s=poll_s, log=log, now=now):
        res = read_reservation(rd)                       # re-read under the lock: the one the unlink acts on
        if res is None or str(res.get("op_id")) != op_id:
            log(f"release: the reservation names {res.get('owner') if res else 'nobody'} {res.get('op_id') if res else ''}, not {op_id}; nothing removed")
            return False
        try:
            os.remove(reservation_path(rd))
        except FileNotFoundError:
            return False
        return True


def require_ownership(rd: str, op_id: str) -> Dict[str, Any]:
    """The reservation must name ``op_id`` (step 5); else :class:`OwnershipLost`, naming the holder and a recorded cancel."""
    res = read_reservation(rd)
    if res is None or str(res.get("op_id")) != op_id:
        op = load_op(rd, op_id) or {}
        raise OwnershipLost(op_id, res, cancelled_by=op.get("cancelled_by"))
    return res


def new_op(owner: str, op_id: str, tree_sha: Optional[str], now: Callable[[], str] = R.utc_now, pid: Optional[int] = None) -> Dict[str, Any]:
    pid = int(pid) if pid else os.getpid()
    return {"schema": OP_SCHEMA, "op_id": op_id, "owner": owner, "started_utc": now(), "host": socket.gethostname(), "pid": pid,
            "pid_start": process_start(pid), "tree_sha": tree_sha, "state": "new", "jobs": {}, "job_names": {k: job_name(k, op_id) for k in JOB_KINDS},
            "adapters": [], "changed": {}, "skipped": {}, "cancelled_by": None, "history": []}


def load_op(rd: str, op_id: str) -> Optional[Dict[str, Any]]:
    return _load(op_path(rd, op_id))


def save_op(rd: str, op: Mapping[str, Any]) -> str:
    p = op_path(rd, str(op["op_id"]))
    _write(p, op)
    return p


@contextlib.contextmanager
def op_record_lock(rd: str, op_id: str) -> Iterator[None]:
    """One ``flock`` per op record (``ops/<op_id>.lock``) around every load-modify-save of ``ops/<op_id>.json``: the owner's own writes
    and a foreign writer's (a takeover's ``cancelled_by``, a reconcile's ``reconciled_by``, a receipt's ``receipted_by``) used to race
    whole-record, so a field -- or the ``jobs`` the self-cancel reads, or a terminal ``state`` -- could be lost or revived. Taken
    INSIDE the critical section when both are held, never the other way round (``update_op`` never takes the critical section)."""
    d = os.path.join(rd, OPS_SUBDIR)
    os.makedirs(d, exist_ok=True)
    fd = os.open(os.path.join(d, f"{op_id}.lock"), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def update_op(rd: str, op_id: str, now: Callable[[], str] = R.utc_now, **fields: Any) -> Dict[str, Any]:
    """Load-modify-save of the op record under :func:`op_record_lock`. ``stages`` (a mapping) is MERGED into the record's, never replaced,
    so two writers recording two stages keep both; every other field is set."""
    with op_record_lock(rd, op_id):
        op = load_op(rd, op_id) or new_op(str(fields.get("owner") or "?"), op_id, fields.get("tree_sha"), now)
        if "state" in fields:
            op.setdefault("history", []).append({"utc": now(), "state": fields["state"]})
        stages = fields.pop("stages", None)
        if isinstance(stages, Mapping):
            merged = dict(op.get("stages") or {})
            merged.update(stages)
            op["stages"] = merged
        op.update(fields)
        op["updated_utc"] = now()
        save_op(rd, op)
        return op


def record_op_stage(rd: str, op_id: str, stage: str, now: Callable[[], str] = R.utc_now, by: Optional[str] = None,
                    stage_fields: Optional[Mapping[str, Any]] = None, **record_fields: Any) -> Dict[str, Any]:
    """Record one of :data:`OP_STAGES` as COMPLETED on the op record -- ``stages[<stage>] = {utc, by, **stage_fields}`` and the state it
    leaves (``pulled`` / ``merged`` / ``receipted``); ``record_fields`` are set on the record itself (``receipt``, ``receipted_by``, ...).
    Called after the plan file recorded the stage (the plan is what a resume reads; the record mirrors it)."""
    if stage not in OP_STAGE_STATE:
        raise ValueError(f"not a stage of the grading host half: {stage!r} (one of {', '.join(OP_STAGE_STATE)})")
    return update_op(rd, op_id, now=now, state=OP_STAGE_STATE[stage], stages={stage: {"utc": now(), "by": by, **dict(stage_fields or {})}}, **record_fields)


def list_ops(rd: str) -> List[Dict[str, Any]]:
    d = os.path.join(rd, OPS_SUBDIR)
    if not os.path.isdir(d):
        return []
    out = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".json"):
            doc = _load(os.path.join(d, fn))
            if doc is not None:
                out.append(doc)
    return out


def op_live(op: Mapping[str, Any]) -> bool:
    """An operation whose process is alive on this host and whose record is neither terminal nor handed off (a tick whose watch timed out
    released its reservation and left its jobs to the next reconcile, however long its process lingers)."""
    if str(op.get("state")) in OP_TERMINAL_STATES or str(op.get("state")) in OP_HANDED_OFF_STATES:
        return False
    return holder_alive(op) is True


# --------------------------------------------------------------------------------------------
# the cluster's queue and accounting, through one runner (ssh, or local in tests)
# --------------------------------------------------------------------------------------------


def _default_runner(argv: Sequence[str], timeout_s: float, input: Optional[str] = None) -> subprocess.CompletedProcess:
    return subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout_s, input=input)


def terminal_verdicts(ids: Sequence[str], queued: Mapping[str, str], acct: Mapping[str, str]) -> Dict[str, Dict[str, Any]]:
    """THE terminal-confirmation rule, applied to one reading of the queue and the accounting (:meth:`Slurm.queue_and_accounting`), per
    id: ``{confirmed, state, where, why}``. Confirmed = an explicit TERMINAL accounting row (:data:`TERMINAL_RE`) AND no row in the queue,
    the two checked apart. ``state`` is the queue's while queued, else the accounting's, else None; ``where`` names
    which reading gave it (``queue`` / ``accounting`` / None). An id in NEITHER reading is NOT terminal (``state`` None, ``why`` ``no
    accounting row``): a job that ran and left the queue before the accounting caught up, or whose row the accounting purged, proves
    nothing by its absence. One rule for every caller -- :meth:`Slurm.wait_terminal` polls on it; a mirror's settle of a recorded verifier
    (``stores.settle_awaiting_verify``) reads it once -- so no caller can re-derive a weaker one from :meth:`Slurm.states` alone (review
    round 1 MEDIUM: ``states`` folds "no row anywhere" into None, which read as settled)."""
    out: Dict[str, Dict[str, Any]] = {}
    for i in sorted({str(x) for x in ids if str(x).strip()}):
        if i in queued:
            out[i] = {"confirmed": False, "state": str(queued[i]), "where": "queue", "why": f"{queued[i]} (queued)"}
        elif i not in acct:
            out[i] = {"confirmed": False, "state": None, "where": None, "why": "no accounting row"}
        elif not TERMINAL_RE.match(str(acct[i])):
            out[i] = {"confirmed": False, "state": str(acct[i]), "where": "accounting", "why": f"{acct[i]} (accounting)"}
        else:
            out[i] = {"confirmed": True, "state": str(acct[i]), "where": "accounting", "why": f"{acct[i]} (accounting), out of the queue"}
    return out


class Slurm:
    """``squeue`` / ``sacct`` / ``scancel`` on the destination, one ssh per call (or a local shell for a directory destination / a test
    shim). Every job row is ``{id, name, state, where}`` with an ARRAY's tasks folded under their base id."""

    def __init__(self, host: Optional[str], runner: Optional[Callable[..., subprocess.CompletedProcess]] = None, user: Optional[str] = None,
                 timeout_s: float = 120.0, since: str = RECONCILE_SINCE):
        self.host = host
        self.run = runner or _default_runner
        self.user = user or os.environ.get("USER") or getpass.getuser()      # the cluster account is this user's (no literal: the identifier test)
        self.timeout_s = timeout_s
        self.since = since

    def _shell(self, script: str, what: str = "slurm") -> subprocess.CompletedProcess:
        """One shell call on the destination. A transport TIMEOUT (``subprocess.TimeoutExpired`` from the runner's bound) or a transport
        failure (an ``OSError``: no ssh binary, a broken pipe) is :class:`SlurmQueryFailed` -- a :class:`ReconcileRefusal` -- never anything
        else: every caller of the queue, the accounting and ``scancel`` sees ONE exception family, so a stalled ssh lands in the same
        fail-closed path as a refused query and can never escape a handler written for refusals."""
        argv = ["bash", "-c", script] if (not self.host or self.host == "local") else ["ssh", *SSH_OPTS, self.host, script]
        try:
            return self.run(argv, self.timeout_s)
        except subprocess.TimeoutExpired as exc:
            err = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            raise SlurmQueryFailed(f"{what} (the transport timed out after {int(self.timeout_s)} s)", self.where, "timeout", str(err)) from exc
        except OSError as exc:
            raise SlurmQueryFailed(f"{what} (the transport failed: {type(exc).__name__})", self.where, type(exc).__name__, str(exc)) from exc

    @property
    def where(self) -> str:
        return self.host or "local"

    def _two_queries(self, squeue_cmd: str, sacct_cmd: str, what: str) -> Tuple[List[str], List[str]]:
        """Run ``squeue`` then ``sacct`` in ONE shell and return their output lines APART, requiring success from BOTH: each query is
        followed by a marker line carrying ITS exit status (``---SQUEUE-RC=<n>---`` / ``---SACCT-RC=<n>---``), so a queue query that failed
        is told from one that answered nothing -- a combined command whose overall status was the LAST command's let a failed ``squeue``
        beside a successful ``sacct`` read as an empty queue, and a job the accounting already called terminal was then "confirmed" out of
        a queue nobody had read. Stderr is not suppressed; the transport's own failure (ssh rc, no markers) is a failure of both. Raises
        :class:`SlurmQueryFailed` naming the query."""
        script = f"{squeue_cmd}; echo \"---SQUEUE-RC=$?---\"; {sacct_cmd}; echo \"---SACCT-RC=$?---\""
        r = self._shell(script, what=what)
        if r.returncode != 0:
            raise SlurmQueryFailed(f"{what}: squeue/sacct (the transport)", self.where, r.returncode, r.stderr or "")
        queued: List[str] = []
        acct: List[str] = []
        rcs: Dict[str, Optional[int]] = {"squeue": None, "sacct": None}
        where = "squeue"
        for line in (r.stdout or "").splitlines():
            line = line.strip()
            m = _RC_MARKER.match(line)
            if m:
                rcs[m.group(1).lower()] = int(m.group(2))
                where = "sacct" if m.group(1) == "SQUEUE" else "done"
                continue
            if where == "squeue":
                queued.append(line)
            elif where == "sacct":
                acct.append(line)
        for q in ("squeue", "sacct"):
            if rcs[q] is None:
                raise SlurmQueryFailed(f"{what}: {q} (no exit status marker in the answer)", self.where, r.returncode, r.stderr or "")
            if rcs[q] != 0:
                raise SlurmQueryFailed(f"{what}: {q}", self.where, rcs[q], r.stderr or "")
        return queued, acct

    @staticmethod
    def _base_id(job_id: str) -> str:
        return job_id.split("_", 1)[0].split(".", 1)[0]

    @staticmethod
    def _state(raw: str) -> str:
        return raw.strip().split(" ", 1)[0].split("+", 1)[0] or "UNKNOWN"

    def jobs(self) -> List[Dict[str, Any]]:
        """Every ``reads-*`` / ``attest-*`` job of this user: the queue (``%i|%T|%j``) then 24 h of accounting (``JobID|JobName|State``),
        one ssh; a job in the queue is reported from the queue (its accounting row may lag)."""
        u = shlex.quote(self.user)
        queued_lines, acct_lines = self._two_queries(f"squeue -h -u {u} -o '%i|%T|%j'",
                                                     f"sacct -X -n -P -S {shlex.quote(self.since)} -u {u} -o JobID,JobName,State", what="jobs")
        rows: Dict[str, Dict[str, Any]] = {}
        for where, line in [("squeue", l) for l in queued_lines] + [("sacct", l) for l in acct_lines]:
            parts = line.split("|")
            if len(parts) < 3:
                continue
            if where == "squeue":
                jid, state, name = parts[0], parts[1], parts[2]
            else:
                jid, name, state = parts[0], parts[1], parts[2]
            if op_of_job_name(name) is None:
                continue
            base = self._base_id(jid)
            st = self._state(state)
            prev = rows.get(base)
            if prev is None:
                rows[base] = {"id": base, "name": name, "state": st, "where": where, "states": [st]}
            elif prev["where"] == where:
                prev["states"].append(st)
                prev["state"] = self._fold_states(prev["states"])
            # a row already seen in the queue keeps the queue's word
        return sorted(rows.values(), key=lambda d: (op_of_job_name(d["name"]) or ("", ""))[1] + d["id"])

    @staticmethod
    def _fold_states(states: Sequence[str]) -> str:
        """One state for an array from its tasks': non-terminal if any is, else the first non-COMPLETED terminal, else COMPLETED."""
        for s in states:
            if not TERMINAL_RE.match(s):
                return s
        for s in states:
            if s != "COMPLETED":
                return s
        return "COMPLETED"

    def queue_and_accounting(self, ids: Sequence[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Two readings of the same ids in one ssh: ``(queued, accounted)`` -- per base id the queue's folded state (present only while
        ``squeue`` lists it) and the accounting's folded state over the array's rows (``sacct -j``; present only when it has a row).
        The two are kept apart on purpose: terminal confirmation needs BOTH an explicit terminal accounting row and the queue's silence,
        and "absent from both" is not a state. BOTH queries must have succeeded (each one's exit status travels in the answer): a failed
        ``squeue`` is :class:`SlurmQueryFailed`, never an empty queue."""
        ids = sorted({str(i) for i in ids if str(i).strip()})
        if not ids:
            return {}, {}
        u = shlex.quote(self.user)
        q = ",".join(shlex.quote(i) for i in ids)
        queued_lines, acct_lines = self._two_queries(f"squeue -h -u {u} -o '%i|%T|%j'", f"sacct -X -n -P -o JobID,State -j {q}",
                                                     what="queue_and_accounting")
        queued: Dict[str, List[str]] = {}
        acct: Dict[str, List[str]] = {}
        for where, line in [("squeue", l) for l in queued_lines] + [("sacct", l) for l in acct_lines]:
            parts = line.split("|")
            if len(parts) < 2:
                continue
            base = self._base_id(parts[0])
            if base not in ids:
                continue
            (queued if where == "squeue" else acct).setdefault(base, []).append(self._state(parts[1]))
        return {i: self._fold_states(v) for i, v in queued.items()}, {i: self._fold_states(v) for i, v in acct.items()}

    def states(self, ids: Sequence[str]) -> Dict[str, Optional[str]]:
        """Per base id its folded state: the queue's when queued, else ``sacct -j`` folded over the array's rows; None when unknown."""
        ids = sorted({str(i) for i in ids if str(i).strip()})
        if not ids:
            return {}
        queued, acct = self.queue_and_accounting(ids)
        return {i: (queued[i] if i in queued else acct.get(i)) for i in ids}

    def cancel(self, ids: Sequence[str]) -> None:
        ids = sorted({str(i) for i in ids if str(i).strip()})
        if not ids:
            return
        r = self._shell("scancel " + " ".join(shlex.quote(i) for i in ids), what="scancel")
        if r.returncode != 0:
            raise SlurmQueryFailed(f"scancel {' '.join(ids)}", self.where, r.returncode, r.stderr or "")

    def wait_terminal(self, ids: Sequence[str], bound_s: float = CANCEL_BOUND_S, poll_s: float = CANCEL_POLL_S,
                      sleep: Callable[[float], None] = time.sleep) -> Dict[str, str]:
        """Poll until EVERY id is confirmed terminal: an explicit TERMINAL state in the accounting (``sacct -j`` has a row and it matches
        :data:`TERMINAL_RE`) AND no row of it in the queue -- the two checked separately (the rule is :func:`terminal_verdicts`, shared
        with every other caller that must prove a job terminal). A job the accounting has no row for is NOT terminal, however silent the
        queue (an empty or lagging ``sacct`` right after a cancel proves nothing); a batch in which EITHER query failed
        (:class:`SlurmQueryFailed` from :meth:`queue_and_accounting`) proves nothing either -- it confirms no id and the poll goes on to
        the bound (a transient failure is retried, never read as the queue's silence); past ``bound_s`` :class:`ReconcileRefusal` names
        every id still outstanding with its reason (``<state> (queued)``, ``<state> (accounting)``, ``no accounting row``, or the failed
        query). Returns the accounting's terminal state per id."""
        ids = sorted({str(i) for i in ids if str(i).strip()})
        if not ids:
            return {}
        t0 = time.monotonic()
        while True:
            pending: Dict[str, str] = {}
            try:
                queued, acct = self.queue_and_accounting(ids)
            except SlurmQueryFailed as exc:
                queued, acct = {}, {}
                pending = {i: f"unconfirmed: {exc.what} failed (rc {exc.rc})" for i in ids}
            if not pending:
                verdicts = terminal_verdicts(ids, queued, acct)
                pending = {i: v["why"] for i, v in verdicts.items() if not v["confirmed"]}
            if not pending:
                return {i: str(verdicts[i]["state"]) for i in ids}
            if time.monotonic() - t0 >= bound_s:
                raise ReconcileRefusal("job(s) not confirmed terminal after " + f"{int(bound_s)} s: " + ", ".join(f"{i}={why}" for i, why in pending.items())
                                       + "; nothing touches the shared namespace until every one reads TERMINAL in sacct and is out of the "
                                       "queue, both queries answering (scancel by hand, wait for the accounting row, then run again)")
            sleep(poll_s)


# --------------------------------------------------------------------------------------------
# the reservation steps: take, take over, replace a stale one; cancel an operation's jobs and confirm
# --------------------------------------------------------------------------------------------


def job_ids_of(op: Optional[Mapping[str, Any]], jobs: Iterable[Mapping[str, Any]], op_id: str) -> List[str]:
    """An operation's Slurm job ids: the recorded ones plus every queued / accounted job named ``reads-<op>`` / ``attest-<op>``."""
    ids: List[str] = []
    for v in ((op or {}).get("jobs") or {}).values():
        if v:
            ids.append(str(v))
    for j in jobs:
        kind_op = op_of_job_name(str(j.get("name") or ""))
        if kind_op and kind_op[1] == op_id:
            ids.append(str(j["id"]))
    return sorted(set(ids))


def cancel_op(rd: str, slurm: Slurm, holder_op_id: str, by: str, now: Callable[[], str] = R.utc_now, log: Callable[[str], None] = lambda s: None,
              bound_s: float = CANCEL_BOUND_S, poll_s: float = CANCEL_POLL_S, sleep: Callable[[float], None] = time.sleep,
              jobs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, str]:
    """Cancel every job of ``holder_op_id`` (recorded ids and by name), wait for sacct TERMINAL (bounded), record ``cancelled_by``."""
    op = load_op(rd, holder_op_id)
    if jobs is None:
        jobs = slurm.jobs()
    ids = job_ids_of(op, jobs, holder_op_id)
    live = [j["id"] for j in jobs if j["id"] in ids and not TERMINAL_RE.match(str(j["state"]))]
    states: Dict[str, str] = {}
    if live:
        log(f"cancel: {by} cancels {holder_op_id}'s job(s) {', '.join(live)} and waits for sacct TERMINAL")
        slurm.cancel(live)
    if ids:
        states = slurm.wait_terminal(ids, bound_s=bound_s, poll_s=poll_s, sleep=sleep)
    fields: Dict[str, Any] = {"cancelled_by": by, "cancelled_utc": now(), "jobs_terminal": states, "owner": (op or {}).get("owner") or "?"}
    if live or not ids:
        # something was cancelled (or there was nothing at all): the operation is cancelled. A record whose every job had already ended
        # keeps its state -- a COMPLETED array nobody pulled is the reconcile's to pull, not to bury under `cancelled`
        fields["state"] = "cancelled"
    update_op(rd, holder_op_id, now=now, **fields)
    return states


def _may_take(res: Mapping[str, Any], takeover: bool) -> bool:
    """Whether this caller may replace the holder ``res``: a holder whose process is gone from this host (stale), or -- for a window
    (``takeover``) -- a tick."""
    return holder_alive(res) is False or (takeover and str(res.get("owner")) == "tick")


def take_reservation(rd: str, slurm: Slurm, owner: str, op_id: str, tree_sha: Optional[str] = None, takeover: bool = False,
                     now: Callable[[], str] = R.utc_now, log: Callable[[str], None] = lambda s: None, bound_s: float = CANCEL_BOUND_S,
                     poll_s: float = CANCEL_POLL_S, sleep: Callable[[float], None] = time.sleep, pid: Optional[int] = None,
                     pid_start: Optional[str] = None, section_bound_s: float = CRITICAL_BOUND_S, section_poll_s: float = CRITICAL_POLL_S) -> Tuple[str, Dict[str, Any]]:
    """Step 1 (and 4). Returns ``(how, reservation)``: ``taken`` (nobody held it), ``held_by_me`` (the reservation names THIS holder
    identity -- owner, op id, host, pid and pid start time all equal; the op id alone is not enough), ``taken_over``
    (a tick's, cancelled and confirmed; ``takeover=True`` -- the window), ``replaced_stale`` (the holder's process is gone from this host
    -- or its pid now belongs to another process -- its jobs cancelled and confirmed, the record names the replacement). Raises
    :class:`ReservationHeld` for a live holder the caller may not take over (a tick: every live holder; a window: a live window), and
    :class:`OperationIdInUse` (a ReservationHeld) when the reservation names this OPERATION ID under another holder identity -- a second
    window with the same WINDOW_ID, an operator retry while the first runs -- whatever its liveness and whatever ``takeover`` says: never
    adopted, never taken over, never replaced as stale (nothing is written). ``pid`` / ``pid_start`` name the long-lived holder when the
    caller reserves on its behalf (the window shell through the CLI). Every WRITE path runs inside the namespace's
    :func:`critical_section`: a takeover (or a stale replacement) that arrives while the holder is mid-pull waits for the pull, the merge
    and the receipt to end, and the holder re-read under the lock is the one replaced -- the yield of a tick to a live holder it may not
    take over, and the held-by-me answer, need no lock and never wait on a pull (both are reads that lead to no write)."""
    me = holder_identity(owner, op_id, pid=pid, pid_start=pid_start)
    res = read_reservation(rd)
    if res is not None:
        if str(res.get("op_id")) == op_id:
            if same_holder(res, me):
                return "held_by_me", res
            raise OperationIdInUse(res, me)
        if not _may_take(res, takeover):
            raise ReservationHeld(res)
    with critical_section(rd, op_id, owner, "reservation", bound_s=section_bound_s, poll_s=section_poll_s, log=log, now=now):
        for _attempt in range(2):
            res = read_reservation(rd)
            if res is None:
                try:
                    return "taken", reserve(rd, owner, op_id, tree_sha, now=now, pid=pid, pid_start=pid_start)
                except ReservationHeld:
                    continue          # a race: somebody took it between the read and the open; read again and decide
            holder_op = str(res.get("op_id"))
            if holder_op == op_id:
                if same_holder(res, me):
                    return "held_by_me", res
                raise OperationIdInUse(res, me)
            alive = holder_alive(res)
            if alive is False:
                why = "is gone from this host" if process_start(int(res.get("pid") or 0)) is None else "now belongs to another process (the pid was reused)"
                log(f"reservation: the holder {res.get('owner')} {holder_op} (pid {res.get('pid')}) {why}; its jobs are cancelled and confirmed")
                cancel_op(rd, slurm, holder_op, by=f"stale-replace {owner} {op_id}", now=now, log=log, bound_s=bound_s, poll_s=poll_s, sleep=sleep)
                update_op(rd, holder_op, now=now, stale_replaced_by=f"{owner} {op_id}")
                how = "replaced_stale"
            elif takeover and str(res.get("owner")) == "tick":
                log(f"reservation: {owner} {op_id} takes over the tick {holder_op}: its array is cancelled and confirmed terminal first")
                cancel_op(rd, slurm, holder_op, by=f"{owner} {op_id}", now=now, log=log, bound_s=bound_s, poll_s=poll_s, sleep=sleep)
                how = "taken_over"
            else:
                raise ReservationHeld(res)
            new = _reservation(owner, op_id, tree_sha, now, pid=pid, pid_start=pid_start)
            new["took_over"] = {"owner": res.get("owner"), "op_id": holder_op, "how": how}
            replace_reservation(rd, new)
            update_op(rd, op_id, now=now, owner=owner, tree_sha=tree_sha, state="reserved", reserved_utc=new["started_utc"], took_over=new["took_over"],
                      pid=new["pid"], pid_start=new["pid_start"])
            return how, new
        raise ReservationHeld(read_reservation(rd) or {"owner": "?", "op_id": "?"})


# --------------------------------------------------------------------------------------------
# step 2: reconcile the cluster's reads-*/attest-* jobs with the op records
# --------------------------------------------------------------------------------------------


def reconcile(rd: str, slurm: Slurm, my_op_id: str, now: Callable[[], str] = R.utc_now, log: Callable[[str], None] = lambda s: None,
              bound_s: float = CANCEL_BOUND_S, poll_s: float = CANCEL_POLL_S, sleep: Callable[[float], None] = time.sleep,
              pull_orphan: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None, section_bound_s: float = CRITICAL_BOUND_S,
              section_poll_s: float = CRITICAL_POLL_S) -> Dict[str, Any]:
    """Every ``reads-*`` / ``attest-*`` job of this user (queue + 24 h of accounting) against ``ops/``: an operation with a LIVE owner is
    left alone; any other operation's outstanding job is cancelled and confirmed terminal (:class:`ReconcileRefusal` past the bound --
    the caller submits nothing); an operation whose array COMPLETED, that has no receipt and whose plan file the caller can still read is
    handed to ``pull_orphan`` (the caller's pull sequence: revalidate -> pull -> re-activations -> inbox-merge -> receipt; every barrier
    of the pull applies) exactly once -- its receipt is what makes the next reconcile skip it. Jobs with no op record at all (a record
    write lost before it landed) are cancelled the same way. Runs INSIDE the namespace's :func:`critical_section` (an orphan's pull is a
    merge; two reconciles never interleave, and a takeover waits for one in progress), and first re-reads the reservation under the lock:
    one that names ANOTHER operation is :class:`OwnershipLost` (the caller lost the namespace between its reservation and this step and
    reconciles nothing; no reservation at all -- the unit tests' bare call -- is left to the caller). Returns the report."""
    with critical_section(rd, my_op_id, "reconcile", "reconcile", bound_s=section_bound_s, poll_s=section_poll_s, log=log, now=now):
        res = read_reservation(rd)
        if res is not None and str(res.get("op_id")) != my_op_id:
            op = load_op(rd, my_op_id) or {}
            raise OwnershipLost(my_op_id, res, cancelled_by=op.get("cancelled_by"))
        return _reconcile_locked(rd, slurm, my_op_id, now, log, bound_s, poll_s, sleep, pull_orphan)


def _reconcile_locked(rd: str, slurm: Slurm, my_op_id: str, now: Callable[[], str], log: Callable[[str], None], bound_s: float, poll_s: float,
                      sleep: Callable[[float], None], pull_orphan: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]) -> Dict[str, Any]:
    jobs = slurm.jobs()
    by_op: Dict[str, List[Dict[str, Any]]] = {}
    for j in jobs:
        kind_op = op_of_job_name(str(j["name"]))
        if kind_op:
            by_op.setdefault(kind_op[1], []).append(j)
    ops = {str(o["op_id"]): o for o in list_ops(rd)}
    candidates = set(by_op) | {o for o, rec in ops.items() if str(rec.get("state")) not in OP_TERMINAL_STATES}
    report: Dict[str, Any] = {"utc": now(), "by": my_op_id, "left_alive": [], "cancelled": {}, "pulled": {}, "completed_unpulled": [],
                              "settled": [], "jobs_seen": [{"id": j["id"], "name": j["name"], "state": j["state"]} for j in jobs]}
    for op_id in sorted(candidates):
        if op_id == my_op_id:
            continue
        op = ops.get(op_id)
        if op is not None and op_live(op):
            report["left_alive"].append(op_id)
            log(f"reconcile: {op_id} ({op.get('owner')}, pid {op.get('pid')}) is alive; left alone")
            continue
        rows = by_op.get(op_id, [])
        live = [j["id"] for j in rows if not TERMINAL_RE.match(str(j["state"]))]
        if live:
            log(f"reconcile: {op_id} has no live owner and job(s) {', '.join(live)} outstanding; cancelling and confirming terminal")
            report["cancelled"][op_id] = cancel_op(rd, slurm, op_id, by=f"reconcile {my_op_id}", now=now, log=log, bound_s=bound_s, poll_s=poll_s,
                                                   sleep=sleep, jobs=jobs)
            continue
        array_rows = [j for j in rows if op_of_job_name(str(j["name"]))[0] == "reads"]       # type: ignore[index]
        completed = bool(array_rows) and all(str(j["state"]) == "COMPLETED" for j in array_rows)
        has_receipt = os.path.isfile(receipt_path(rd, op_id))
        if op is not None and completed and not has_receipt and str(op.get("state")) not in OP_NO_PULL_STATES:
            report["completed_unpulled"].append(op_id)
            if pull_orphan is not None:
                log(f"reconcile: {op_id}'s array completed and was never pulled; pulling it now under {my_op_id}")
                try:
                    report["pulled"][op_id] = pull_orphan(op)
                except Exception as exc:  # noqa: BLE001 -- the orphan's pull must not take the current operation down with it
                    report["pulled"][op_id] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
                    update_op(rd, op_id, now=now, state="failed", reconciled_by=my_op_id, error=f"orphan pull: {type(exc).__name__}: {exc}"[:500])
                    log(f"reconcile: the orphan pull of {op_id} failed: {exc}")
            continue
        if op is not None and str(op.get("state")) not in OP_TERMINAL_STATES:
            states = {j["id"]: j["state"] for j in rows}
            update_op(rd, op_id, now=now, state="reconciled", reconciled_by=my_op_id, jobs_terminal=states,
                      note="no live owner; no outstanding job" + ("; array completed but a receipt exists" if has_receipt else ""))
            report["settled"].append(op_id)
    return report


# --------------------------------------------------------------------------------------------
# the tick's change detection: one ssh batch over every component's `current`, the module blob shas, the receipts
# --------------------------------------------------------------------------------------------


def _probe_script(dest: ST.Destination, comps: Sequence[ST.Component], pins: Mapping[str, str], tree: Optional[str]) -> str:
    q = shlex.quote
    parts = [ST._authority_batch_script(dest, comps)] if comps else ["set +e"]
    for slug, sha8 in sorted(pins.items()):
        g = dest.gen_dir(slug)
        parts.append(f"if test -f {q(g)}/PUBLISHED-{q(sha8)}.json; then printf 'Q\\t%s\\t%s\\t' {q(slug)} {q(sha8)}; "
                     f"tr -d '\\n' < {q(g)}/PUBLISHED-{q(sha8)}.json; printf '\\n'; else printf 'Q\\t%s\\t%s\\t\\n' {q(slug)} {q(sha8)}; fi")
    if tree:
        parts.append(f"if test -d {q(tree)}/src/sisteer/registry; then printf 'T\\t1\\n'; else printf 'T\\t0\\n'; fi")
    return "; ".join(parts)


def probe(cfg: ST.Config, dest: ST.Destination, slugs: Sequence[str], runner=None, tree: Optional[str] = None,
          pins: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """ONE ssh batch: per component its `current` (target, existence, VERIFIED / PUBLISHED / SEALED beside it) and the PUBLISHED record
    of that `current`; for every ``pins`` entry the PUBLISHED record of THAT generation (a receipt reads the pin's record, not
    `current`'s); whether the base tree stands at ``tree``. Returns ``{"components": {slug: {...}}, "pins": {slug: {sha8: record}},
    "tree_present": bool | None}``."""
    remote = ST.Remote(dest, runner)
    comps = [cfg.components[s] for s in slugs if s in cfg.components]
    pins = dict(pins or {})
    r = remote._cmd(_probe_script(dest, comps, pins, tree), what="cluster-reads probe batch")
    out: Dict[str, Any] = {"components": {}, "pins": {}, "tree_present": None, "utc": R.utc_now()}
    pub_current: Dict[str, Dict[str, Any]] = {}
    for line in (r.stdout or "").splitlines():
        cells = line.split("\t")
        if cells[0] == "C" and len(cells) == 8:
            out["components"][cells[1]] = {"current": cells[2] or None, "real": cells[3] or None, "exists": cells[4] == "1", "verified": cells[5] == "1",
                                           "published_record": cells[6] == "1", "sealed": cells[7] == "1", "published": None}
        elif cells[0] == "P" and len(cells) >= 3:
            try:
                doc = json.loads("\t".join(cells[2:]))
            except ValueError:
                doc = {"unreadable": True}
            pub_current[cells[1]] = doc if isinstance(doc, dict) else {}
        elif cells[0] == "Q" and len(cells) >= 3:
            text = "\t".join(cells[3:]) if len(cells) > 3 else ""
            pin_doc = None
            if text.strip():
                try:
                    pin_doc = json.loads(text)
                except ValueError:
                    pin_doc = {"unreadable": True}
            out["pins"].setdefault(cells[1], {})[cells[2]] = pin_doc if isinstance(pin_doc, dict) else None
        elif cells[0] == "T" and len(cells) == 2:
            out["tree_present"] = cells[1] == "1"
    for slug, row in out["components"].items():
        pub = pub_current.get(slug)
        if pub is not None:
            row["published"] = {"published_utc": pub.get("published_utc"), "change_utc": pub.get("change_utc"), "mode": pub.get("mode"),
                                "producer": pub.get("producer")}
    missing = [s for s in slugs if s in cfg.components and s not in out["components"]]
    if missing:
        raise CR.ClusterReadsError(f"probe: the batch returned nothing for component(s) {', '.join(missing)}")
    return out


def adapter_module_paths(adapters: Optional[Sequence[str]] = None) -> Dict[str, Optional[str]]:
    """Per registered adapter the repository-relative path of the module that registers it (``src/sisteer/registry/adapters/<m>.py``);
    imports the adapters package (every adapter imports and constructs on an unscoped host since C6; no store is read)."""
    from . import adapters as ADAPTERS
    from .adapters import base as AB
    ADAPTERS.load_all()
    out: Dict[str, Optional[str]] = {}
    for name, factory in AB.registered_raw().items():
        if adapters is not None and name not in adapters:
            continue
        mod_name = getattr(factory, "__module__", None)
        mod = sys.modules.get(str(mod_name)) if mod_name else None
        f = getattr(mod, "__file__", None) if mod else None
        rel = None
        if f:
            f = os.path.abspath(f)
            marker = os.sep + os.path.join("src", "sisteer") + os.sep
            i = f.rfind(marker)
            if i >= 0:
                rel = f[i + 1:].replace(os.sep, "/")
        out[name] = rel
    return out


def resolve_tree_sha(repo_root: str, ref: str = "HEAD") -> str:
    """``ref`` resolved to the full commit sha in ``repo_root`` (``git rev-parse <ref>^{commit}``) -- THE tree a read is compared and
    recorded with: the tick's reservation, plan and receipt carry it, and :func:`changed_set` compares a receipt's ``tree_sha`` with it
    for an adapter whose last whole receipt carries standing refusals. The ``changed`` command resolves its ``--landing`` through the same
    function (it once passed no tree, so after a landing that changed shared reader code the command listed no adapter
    while the tick, which did, re-read it -- the operator's ``ADAPTERS`` selection for the window missed the refusal). Raises
    :class:`cluster_reads.ClusterReadsError` when the ref does not resolve to a commit."""
    r = subprocess.run(["git", "-C", repo_root, "rev-parse", f"{ref}^{{commit}}"], capture_output=True, text=True)
    if r.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", r.stdout.strip()):
        raise CR.ClusterReadsError(f"git rev-parse {ref} in {repo_root} failed: {(r.stderr or r.stdout).strip()[:200]}")
    return r.stdout.strip()


def tree_blobs(repo_root: str, ref: str = "HEAD", rel_dir: str = ADAPTERS_REL) -> Dict[str, str]:
    """``{repository-relative path: blob sha}`` under ``rel_dir`` at ``ref`` (one ``git ls-tree -r``)."""
    r = subprocess.run(["git", "-C", repo_root, "ls-tree", "-r", ref, "--", rel_dir], capture_output=True, text=True)
    if r.returncode != 0:
        raise CR.ClusterReadsError(f"git ls-tree {ref} -- {rel_dir} in {repo_root} failed (rc {r.returncode}): {(r.stderr or '').strip()[:300]}")
    out: Dict[str, str] = {}
    for line in (r.stdout or "").splitlines():
        try:
            meta, path = line.split("\t", 1)
            sha = meta.split()[2]
        except (ValueError, IndexError):
            continue
        out[path] = sha
    return out


def module_blobs(repo_root: str, adapters: Sequence[str], ref: str = "HEAD", paths: Optional[Mapping[str, Optional[str]]] = None,
                 blobs: Optional[Mapping[str, str]] = None) -> Dict[str, Dict[str, Optional[str]]]:
    """Per adapter ``{module path: blob sha at ref}`` (None when the module is not in the tree at ``ref``)."""
    paths = paths if paths is not None else adapter_module_paths(adapters)
    blobs = blobs if blobs is not None else tree_blobs(repo_root, ref)
    out: Dict[str, Dict[str, Optional[str]]] = {}
    for a in adapters:
        rel = paths.get(a)
        out[a] = {rel: blobs.get(rel)} if rel else {}
    return out


def receipts(rd: str) -> List[Dict[str, Any]]:
    """Every receipt, newest first (by its ``utc``)."""
    d = os.path.join(rd, RECEIPTS_SUBDIR)
    if not os.path.isdir(d):
        return []
    out = []
    for fn in os.listdir(d):
        if fn.endswith(".json"):
            doc = _load(os.path.join(d, fn))
            if doc is not None and doc.get("schema") == RECEIPT_SCHEMA:
                out.append(doc)
    return sorted(out, key=lambda r: str(r.get("utc") or ""), reverse=True)


def read_whole(entry: Mapping[str, Any]) -> bool:
    """A receipt entry that READ the adapter whole: every shard ok, no refusal, and no refused-known id the pull left STANDING
    (``refused_standing``: an id refused at emit time on the grading host that an OLDER shard returned instead of reading, so its
    group was neither read nor committed; no refusal is recorded for it, and the entry is not whole either: the next tick reads the
    adapter again, and this tree's shards read the group fresh). Before this a receipt over such a pull read whole and the adapter was
    skipped while its pins and module stood, the standing refusal parked under a complete-looking receipt. A
    retirement the pull WITHHELD (``withheld``: the shard's claim was not the live head) is the same
    posture: its group was neither retired nor refused, so the entry is not whole and the next window's shard judges it against today's
    head."""
    total = int(entry.get("shards_total") or 0)
    return (total > 0 and int(entry.get("shards_ok") or 0) == total and not entry.get("refused") and not entry.get("refused_standing")
            and not entry.get("withheld"))


def last_receipt_for(rd: str, adapter: str, recs: Optional[Sequence[Mapping[str, Any]]] = None) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """The newest receipt whose entry for ``adapter`` read it whole, as ``(receipt, entry)``; None when it was never read whole."""
    for rec in (recs if recs is not None else receipts(rd)):
        e = (rec.get("adapters") or {}).get(adapter)
        if e and read_whole(e):
            return dict(rec), dict(e)
    return None


def attempted(entry: Mapping[str, Any]) -> bool:
    """A receipt entry whose window ATTEMPTED the adapter on the cluster: every shard ran its pass (``shards_ok == shards_total > 0``), so
    every group it owns was read fresh, refused by its reader, or returned known -- whole or refused, the outcome is in that receipt. Not
    attempted: routed off the cluster (``shards_total`` 0: declared refused, no published generation, a plan-time refusal) or a shard
    missing / not ok (its groups never ran)."""
    total = int(entry.get("shards_total") or 0)
    return total > 0 and int(entry.get("shards_ok") or 0) == total


def standing_unread(rd: str, adapter: str, recs: Optional[Sequence[Mapping[str, Any]]] = None) -> List[str]:
    """The refused-known ids a pull of ``adapter`` left standing unread and no later window has attempted: an emit-time refusal an
    older shard returned instead of reading -- and, the same way, the retirements a pull WITHHELD because the shard's claim was not the
    live head (``withheld``). Newest receipt first: an entry carrying such ids returns them; an entry whose
    window :func:`attempted` the adapter (every shard ran: the group was read fresh, refused by its reader, or returned known -- resolved
    one way or the other, in that receipt) ends the search empty; an entry whose window never attempted it (routed off the cluster, a
    shard missing) is skipped, so a transient component or admission problem between the standing pull and the retry does not erase the
    claim (an older whole receipt, then a standing one, then a plan-refused one, then the component recovers under the same pins -- the
    standing id must still be read). Empty with no receipt. Distinct from :func:`last_receipt_for` on purpose: that is the newest WHOLE
    receipt, the baseline for "what moved since", and skipping back to an older whole receipt with the same pins and module is exactly
    how a standing-unread id would be left unread for good (a prior whole read, then an emit-time refusal with nothing moved)."""
    for rec in (recs if recs is not None else receipts(rd)):
        e = (rec.get("adapters") or {}).get(adapter)
        if not e:
            continue
        unread = [str(r) for r in (e.get("refused_standing") or [])] + [str(r) for r in (e.get("withheld") or [])]
        if unread:
            return unread
        if attempted(e):
            return []
    return []


def held_retirements_unread(adapter: str, recs: Sequence[Mapping[str, Any]], holds: Sequence[Mapping[str, Any]]) -> List[str]:
    """The retirement markers of ``adapter`` the HARVESTER holds because the head they were computed against is not the live one
    (``harvest.retirements_held_for_moved_head``: ``holds``, each with the hold's identity, ``harvest.RETIREMENT_HOLD_IDENTITY`` --
    ``revision_id``; ``record_uri``, the held carrier; ``blocking_head``, the head the hold was computed against) that no attempted window
    had IN HAND when it judged the adapter -- registry-hardening 3a, review rounds 2 and 3 and the closing pass. Such a hold is new
    information the cluster has not seen: the group's watermarks stand live in the database, and the next shard that reads the adapter
    judges the group against today's head and refuses the same empty id as known with a claim the pull re-activates. A window has the
    hold in hand when its plan recorded it off the reader-state sidecar it exported for its shards (``holds_seen`` on the receipt entry:
    :func:`receipt_from_plan` from ``cluster_reads.plan_window``); so a hold stands unread until some receipt whose window
    :func:`attempted` the adapter (every shard ran) names it by its WHOLE identity -- a hold a later bundle of the same revision incurred
    (another carrier) is unread until a receipt names THAT one, and so is the SAME carrier re-held against a NEWER head (closing pass:
    once the compared head is superseded the harvester re-attempts the carrier, holds it against the live head and rewrites the same queue
    row's ``missing_parent_id`` in place; named by revision and carrier alone, that hold read as already seen by the receipt that had named
    the old head, and no later tick retried the retirement while the new head's measurements stayed current after its input vanished). A
    receipt whose window never attempted the adapter (routed off the cluster, a shard missing) says nothing, whatever its plan saw; an
    entry naming no head (written before the head joined the identity) matches nothing. No stamp orders the two events (round 3): the
    window merges the marker and the harvester can fold and hold it BEFORE the window writes its receipt (``pull_and_receipt``: pull,
    merge, then receipt), so a receipt stamped after the hold is no evidence that its shard ran after it -- round 2 read it as such and
    parked the adapter under the delivering window's whole receipt; and the hold's ``first_seen_utc`` is the harvester's one clock reading
    per cycle, taken at the cycle's start and carried by a hold committed minutes later, so no stamp of the window's orders them either.
    Whether the plan saw the hold is a recorded fact of the plan's read of its snapshot."""
    if not holds:
        return []
    key = lambda h: tuple(H.retirement_hold_identity(h)[k] for k in H.RETIREMENT_HOLD_IDENTITY)
    seen: Set[Tuple[str, ...]] = set()
    for rec in recs:
        e = (rec.get("adapters") or {}).get(adapter)
        if not e or not attempted(e):
            continue
        for h in (e.get("holds_seen") or []):
            if isinstance(h, Mapping):
                seen.add(key(h))
    out: List[str] = []
    for h in holds:
        k = key(h)
        if k[0] and k not in seen:
            out.append(k[0])
    return sorted(set(out))


def standing_ids(root: str, rd: str, adapter: str, recs: Optional[Sequence[Mapping[str, Any]]] = None) -> List[str]:
    """The ids that keep ``adapter`` UNREAD for the next tick (``changed_set``'s ``standing``): the refused-known ids a pull left standing
    (:func:`standing_unread`: ``refused_standing``, the ``withheld`` retirements) and the retirement markers the harvester holds because
    the head they were computed against moved before the fold (:func:`held_retirements_unread`: the holds read from the registry's
    database at ``root`` -- the authoritative record of a hold, the revision's own row notwithstanding -- against the holds the receipts'
    plans recorded off their sidecars, ``holds_seen``, by the hold's whole identity); nothing is read when no database exists yet (a fresh
    root). An unreadable database raises: a tick that cannot read the queue must say so, not read the adapter as whole."""
    recs = list(recs) if recs is not None else receipts(rd)
    out = standing_unread(rd, adapter, recs)
    db_path = DB.db_path(root)
    if os.path.exists(db_path):
        conn = DB.connect(db_path, readonly=True)
        try:
            holds = H.retirements_held_for_moved_head(conn, adapter)
        finally:
            conn.close()
        out = out + [r for r in held_retirements_unread(adapter, recs, holds) if r not in out]
    return out


def changed_set(cfg: ST.Config, adapters: Sequence[str], probed: Mapping[str, Any], blobs: Mapping[str, Mapping[str, Optional[str]]],
                last: Callable[[str], Optional[Tuple[Mapping[str, Any], Mapping[str, Any]]]],
                standing: Optional[Callable[[str], Sequence[str]]] = None, tree_sha: Optional[str] = None) -> Dict[str, Any]:
    """Which adapters a tick reads: an adapter whose pins moved since its last whole receipt, whose module blob changed, that has no
    whole receipt, or whose LAST pull left a refused-known id standing unread (``standing``: :func:`standing_ids` -- :func:`standing_unread`,
    where the group was neither read nor committed and this tree's shards read it fresh; and :func:`held_retirements_unread`, a
    retirement the harvester holds because its head moved before the fold -- so the adapter is read once
    more whatever an older whole receipt says) -- ``changed`` with the reasons; an undeclared, declared-refused or unpublished one is ``skipped`` with the
    reason (a read would only record a refusal; the authority problem is `check-authority`'s to show); the rest ``unchanged``.

    ``tree_sha`` (the tree THIS tick reads with): a whole receipt whose entry carries ``standing_refusals``
    was recorded under the reading tree the receipt names (``tree_sha``), and a standing refusal is byte-identical only under that tree --
    a later commit to SHARED reader code (``adapters/runner.py``, a helper) changes neither the adapter's own module blob nor a pin, so
    without this the refusal would never be re-evaluated. Such an adapter is ``changed`` when the receipt's tree is not this one: the
    shard re-reads it, and under the new tree the refusal is a fresh refusal or a read, never standing (the key names the tree)."""
    changed: Dict[str, List[str]] = {}
    skipped: Dict[str, str] = {}
    unchanged: List[str] = []
    comps = probed.get("components") or {}
    for a in adapters:
        decl = cfg.adapters.get(a)
        if decl is None:
            skipped[a] = f"not declared in {cfg.path}"
            continue
        if decl.refused:
            skipped[a] = f"declared cluster: refused ({decl.reason})"
            continue
        pins: Dict[str, str] = {}
        why_skip = None
        for slug in decl.components:
            row = comps.get(slug)
            if row is None:
                why_skip = f"component {slug}: not probed"
                break
            if not row.get("current"):
                why_skip = f"component {slug}: nothing published (no `current`)"
                break
            if not row.get("exists"):
                why_skip = f"component {slug}: `current` -> {row.get('current')} names no tree (dangling)"
                break
            pins[slug] = str(row["current"])
        if why_skip:
            skipped[a] = why_skip
            continue
        prev = last(a)
        reasons: List[str] = []
        if prev is None:
            reasons.append("never read whole (no receipt)")
        else:
            _rec, entry = prev
            old_pins = dict(entry.get("pins") or {})
            for slug, sha8 in sorted(pins.items()):
                if old_pins.get(slug) != sha8:
                    reasons.append(f"pin moved: {slug} {old_pins.get(slug) or '-'} -> {sha8}")
            for slug in sorted(set(old_pins) - set(pins)):
                reasons.append(f"component {slug} no longer declared for it")
            old_blobs = dict(entry.get("module_blobs") or {})
            for rel, sha in sorted((blobs.get(a) or {}).items()):
                if old_blobs.get(rel) != sha:
                    reasons.append(f"module changed: {rel} {str(old_blobs.get(rel) or '-')[:12]} -> {str(sha or '-')[:12]}")
            standing_groups = list(entry.get("standing_refusals") or [])
            if standing_groups and tree_sha and str(_rec.get("tree_sha") or "") != str(tree_sha):
                reasons.append(f"{len(standing_groups)} standing refusal(s) recorded under tree {str(_rec.get('tree_sha') or '-')[:12]}; this tree is "
                               f"{str(tree_sha)[:12]} -- re-read: shared reader code may have changed what the refusal stands on")
        unread = sorted(standing(a)) if standing is not None else []
        if unread:
            # the newest receipt, whole or not: an older whole receipt with the same pins and module must not park the standing id
            reasons.append(f"refused-known id(s) left standing unread by the last pull: {', '.join(unread[:4])}"
                           + (f" (+{len(unread) - 4} more)" if len(unread) > 4 else "") + " -- read fresh on this tree")      # or a retirement the harvester holds (standing_ids)
        if reasons:
            changed[a] = reasons
        else:
            unchanged.append(a)
    return {"changed": changed, "skipped": skipped, "unchanged": unchanged}


# --------------------------------------------------------------------------------------------
# receipts
# --------------------------------------------------------------------------------------------


def receipt_from_plan(plan: Mapping[str, Any], published_pins: Mapping[str, Mapping[str, Any]], blobs: Mapping[str, Mapping[str, Optional[str]]],
                      op: Mapping[str, Any], now: Callable[[], str] = R.utc_now) -> Dict[str, Any]:
    """The read receipt (step 6) from the plan file after its pull: per adapter the pins, each pin's PUBLISHED record (``published_utc``,
    ``change_utc``, ``mode``), the module blobs, shards ok / total (the pull's ``summaries``), the revisions pulled, the re-activations
    (``reactivated``; the ids left standing unread, ``refused_standing`` -- not a refusal but not whole: :func:`read_whole`; the
    retirements the pull WITHHELD because the shard's claim was not the live head, ``withheld`` -- the same
    posture: not a refusal, not whole, the next tick reads the adapter again),
    the refusal (``plan["refusals"]``) -- an adapter routed off the cluster has ``shards_total`` 0 and its refusal. The
    groups a shard re-read and refused again byte for byte under the same inputs and tree are ``standing_refusals`` (per group
    ``state = standing_refusal``, the ``refusal`` as recorded and the ``prior_refusal`` it equals -- ``window``, ``utc``, ``digest``: the
    pull's ``standing_refusals``) -- a recorded result, not a refusal, so the entry reads whole; ``refused_standing`` stays the list
    of emit-time-refused REVISION IDS left unread; the refusal records the pull delivered (``<a>/_refused/<rev>``) are listed under
    ``refusal_records_pulled``, apart from ``revisions_pulled``. ``holds_seen``
    is the plan's: the retirement markers the harvester held for a moved head that the window's reader-state sidecar -- the snapshot its
    shards judged against -- carried (``cluster_reads.holds_seen_of``, each the hold's identity, ``harvest.RETIREMENT_HOLD_IDENTITY``:
    ``{revision_id, record_uri, blocking_head}``) -- what :func:`held_retirements_unread` reads as "this window's shards judged the adapter
    with these holds in hand"; the receipt's ``holds_seen_from`` is the plan's, the snapshot they were read from (None: no sidecar, no hold
    seen)."""
    stages = plan.get("stages") or {}
    pull = stages.get("pull") or {}
    summaries = pull.get("summaries") or {}
    pulled_dirs = list(pull.get("pulled_dirs") or [])
    reactivations = pull.get("reactivations") or {}
    standing_groups = pull.get("standing_refusals") or {}
    refusals = plan.get("refusals") or {}
    adapters: Dict[str, Any] = {}
    for a, e in sorted((plan.get("adapters") or {}).items()):
        pins = dict(e.get("pins") or {})
        s = summaries.get(a) or {}
        total = int(s.get("expected") or 0) if e.get("route") == "cluster" else 0
        ok = int(s.get("ok") or 0) if e.get("route") == "cluster" else 0
        pub: Dict[str, Optional[Dict[str, Any]]] = {}
        for slug, sha8 in pins.items():
            v = (published_pins.get(slug) or {}).get(sha8)
            pub[slug] = ({"published_utc": v.get("published_utc"), "change_utc": v.get("change_utc"), "mode": v.get("mode"), "producer": v.get("producer")}
                         if isinstance(v, dict) else None)
        own = [d for d in pulled_dirs if d.startswith(a + "/")]
        adapters[a] = {"route": e.get("route"), "pins": pins, "published": pub, "module_blobs": dict(blobs.get(a) or {}),
                       "shards_total": total, "shards_ok": ok, "revisions_pulled": sorted(d for d in own if f"/{CR.REFUSED_DIR}/" not in d),
                       "refusal_records_pulled": sorted(d for d in own if f"/{CR.REFUSED_DIR}/" in d),
                       "reactivated": sorted(r for r, st in (reactivations.get(a) or {}).items() if st == "reactivated"),
                       "refused_standing": sorted(r for r, st in (reactivations.get(a) or {}).items() if st == "refused_standing"),
                       "withheld": sorted(r for r, st in (reactivations.get(a) or {}).items() if st == "withheld"),
                       "standing_refusals": [dict(x) for x in (standing_groups.get(a) or [])],
                       "holds_seen": [H.retirement_hold_identity(h) for h in (e.get("holds_seen") or []) if isinstance(h, Mapping)],
                       "refused": list(refusals.get(a) or []) or None}
        adapters[a]["read_whole"] = read_whole(adapters[a])
    return {"schema": RECEIPT_SCHEMA, "op": {"op_id": op.get("op_id"), "owner": op.get("owner")}, "utc": now(), "window": plan.get("window"),
            "tree_sha": plan.get("tree_sha"), "destination": plan.get("destination"), "jobs": dict(op.get("jobs") or {}),
            "attestation": pull.get("attestation"), "barrier": pull.get("barrier"), "holds_seen_from": plan.get("holds_seen_from"), "adapters": adapters,
            "n_read_whole": sum(1 for v in adapters.values() if v["read_whole"]), "n_refused": sum(1 for v in adapters.values() if v["refused"])}


def depends_of_plan(plan: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The revision directories whose MERGE this operation depends on: the pull stage's ``depends_dirs`` -- every revision the
    pull reported complete, newly copied (``how: copied``, the shard summary's name -> sha256, verified on receipt) or an earlier complete
    delivery found in the inbox and reused as it stands (``how: reused``, the inbox copy's own name -> sha256) -- keyed ``<adapter>/<rev16>``.
    A plan from before this record (none live yet) falls back to its ``pulled_dirs`` with no file list: the evidence then checks only that
    the directory left the inbox and exists in ``records/``."""
    pull = (plan.get("stages") or {}).get("pull") or {}
    dep = pull.get("depends_dirs")
    if isinstance(dep, Mapping) and dep:
        return {str(r): {"adapter": str(v.get("adapter") or str(r).split("/", 1)[0]), "how": str(v.get("how") or "copied"),
                         "files": {str(k): str(sv) for k, sv in dict(v.get("files") or {}).items()}} for r, v in dep.items()}
    return {str(d): {"adapter": str(d).split("/", 1)[0], "how": "copied", "files": {}} for d in (pull.get("pulled_dirs") or []) if str(d)}


def merge_evidence(root: str, depends: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """WHAT THE RECEIPT RESTS ON (I7): for every revision this operation depends on, the DURABLE state of ``records/`` and the
    inbox -- read after every merge attempt, first or resumed, and again right before the receipt -- never ``scan_inbox``'s return value
    (an interrupted scan has already moved a conflicting file to ``<file>.conflict-<sha8>`` and the resumed scan cannot see it again).
    Per revision ``{"state": merged | conflict | incomplete, "adapter", "how", "problems": [...]}``: a recorded file present in
    ``records/adapters/<rev>/`` with its sha256 is merged; a ``<file>.conflict-<sha8>`` sibling, or the file present with other bytes, is a
    CONFLICT; a file missing, or the revision directory still under ``inbox/<name>/adapters/``, is INCOMPLETE. A revision recorded with no
    file list (an older plan) is merged when its ``records/`` directory exists and no inbox directory does.

    THE RECORD'S REPRESENTATION MAY CHANGE UNDER AN ACCEPTED FILE: the harvester's ``gzip_after_ingest`` replaces an
    ingested ``<name>.jsonl`` by ``<name>.jsonl.gz`` (the LOGICAL sha -- the decompressed bytes -- unchanged, the plain file removed; 10 min
    after ingest by default, so a merge or receipt resumed later meets the ``.gz`` twin), and ``mirrors.verify_manifest(prune_plain=True)``
    removes the plain file once both forms verify. So a recorded file is looked for under BOTH names and compared by :func:`R.file_logical_sha`
    (the plain file's raw bytes; the ``.gz`` twin's decompressed bytes -- the harvester's own notion of identity): the plain file is
    authoritative when it matches; else a matching ``.gz`` twin; both present and neither matching is a conflict naming both. The other
    moves under ``records/`` were audited and change no recorded file: ``scan_inbox`` writes the merge this evidence reads (the ``.conflict-``
    case above); the state sidecar (``state/file_stat.json``) and the cycle state live under ``state/``; nothing compacts or renames an
    ``_activations`` bundle on disk."""
    records = os.path.join(H.records_dir(root), "adapters")
    ib = H.inbox_dir(root)
    inbox_hosts = [os.path.join(ib, h) for h in (sorted(os.listdir(ib)) if os.path.isdir(ib) else []) if not h.startswith(".") and os.path.isdir(os.path.join(ib, h))]
    out: Dict[str, Dict[str, Any]] = {}
    for rev, dep in sorted(depends.items()):
        dest = os.path.join(records, rev)
        problems: List[str] = []
        conflict = False
        in_inbox = [h for h in inbox_hosts if os.path.isdir(os.path.join(h, "adapters", rev))]
        if in_inbox:
            problems.append("still in the inbox")
        files = dict(dep.get("files") or {})
        if files:
            for name, sha in sorted(files.items()):
                verdict, why = _recorded_file_state(dest, name, str(sha))
                if verdict == "merged":
                    continue
                if verdict == "conflict":
                    conflict = True
                problems.append(f"{name}: {why}")
        elif not os.path.isdir(dest):
            problems.append("no records directory")
        state = "conflict" if conflict else ("incomplete" if problems else "merged")
        out[rev] = {"state": state, "adapter": str(dep.get("adapter") or rev.split("/", 1)[0]), "how": str(dep.get("how") or "copied"), "problems": problems}
    return out


def _recorded_file_state(dest: str, name: str, sha: str) -> Tuple[str, str]:
    """One recorded file of a revision against ``records/``: ``("merged", "")``, ``("conflict", why)`` or ``("missing", why)``. The recorded
    sha is the RAW sha256 of the file AS THE SHARD NAMED IT (the shard summary's ``sha256``, ``_inbox_shas``: a shard's own ``.jsonl.gz``
    included). The file may stand under that name -- compared by its raw bytes (:func:`R.file_sha256`) -- or, when the name is not itself
    ``.gz``, as the harvester's ``<name>.gz`` twin (``gzip_after_ingest``; it never compresses a ``.gz`` twice), compared by its
    DECOMPRESSED bytes (:func:`R.file_logical_sha`, the harvester's own check that the logical sha is retained). ONE form present and
    matching -> merged; BOTH forms present: merged only when both match -- two active forms of one logical URI that differ
    are a CONFLICT naming both, whichever one is ours (the harvester enumerates both names as one record, so a competing plain file
    installed beside a correct twin would be ingested as "the" record; the inbox merge no longer installs one, this is the belt); a
    ``<name>.conflict-<sha8>`` sibling with no matching form -> conflict naming it; a form present but not matching -> conflict naming it;
    nothing -> missing."""
    plain = os.path.join(dest, name)
    has_plain = os.path.isfile(plain)
    plain_sha = R.file_sha256(plain) if has_plain else None
    twin = None if name.endswith(".gz") else plain + ".gz"
    has_twin = bool(twin) and os.path.isfile(twin)
    twin_sha = R.file_logical_sha(twin) if has_twin else None          # type: ignore[arg-type]
    if has_plain and has_twin:
        if plain_sha == sha and twin_sha == sha:
            return "merged", ""
        return "conflict", (f"two active forms of one record: {name} (sha {str(plain_sha)[:8]}), {name}.gz (logical sha {str(twin_sha)[:8]}); "
                            f"recorded {sha[:8]}")
    if has_plain and plain_sha == sha:
        return "merged", ""
    if has_twin and twin_sha == sha:
        return "merged", ""
    siblings = sorted(fn for fn in (os.listdir(dest) if os.path.isdir(dest) else []) if fn.startswith(name + ".conflict-"))
    if siblings:
        return "conflict", ", ".join(siblings)
    if has_plain or has_twin:
        forms = ([f"{name} (sha {str(plain_sha)[:8]})"] if has_plain else []) + ([f"{name}.gz (logical sha {str(twin_sha)[:8]})"] if has_twin else [])
        return "conflict", f"other bytes in records: {', '.join(forms)} != {sha[:8]}"
    return "missing", "missing from records"


def merge_problems(evidence: Mapping[str, Mapping[str, Any]]) -> Dict[str, List[str]]:
    """``{adapter: ["merge_conflict: <rev> (<detail>; ...)", "merge_incomplete: <rev> (...)", ...]}`` from :func:`merge_evidence` -- every
    revision whose state is in :data:`MERGE_PROBLEM_STATES`; a merged one is no problem."""
    out: Dict[str, List[str]] = {}
    for rev, ev in sorted(evidence.items()):
        st = str(ev.get("state"))
        if st in MERGE_PROBLEM_STATES:
            detail = "; ".join(str(x) for x in (ev.get("problems") or []))
            out.setdefault(str(ev.get("adapter") or rev.split("/", 1)[0]), []).append(f"merge_{st}: {rev}" + (f" ({detail})" if detail else ""))
    return {a: sorted(set(v)) for a, v in out.items()}


def apply_merge_problems(plan: Dict[str, Any], evidence: Mapping[str, Mapping[str, Any]]) -> Dict[str, List[str]]:
    """Record the merge evidence's problems on the plan: every adapter :func:`merge_problems` names gets them appended to ``plan["refusals"]``
    (what the receipt's ``refused`` and the window's ``cluster-reads refusals`` read), so its receipt entry is refused and ``read_whole``
    false. Idempotent (a sorted set), so the receipt-time re-check may apply it again."""
    problems = merge_problems(evidence)
    if problems:
        ref = plan.setdefault("refusals", {})
        for a, whys in problems.items():
            ref[a] = sorted(set(list(ref.get(a) or []) + whys))
    return problems


def write_receipt(rd: str, rec: Mapping[str, Any]) -> str:
    p = receipt_path(rd, str(rec["op"]["op_id"]))
    _write(p, rec)
    return p


def published_for_plan(cfg: ST.Config, dest: ST.Destination, plan: Mapping[str, Any], runner=None) -> Dict[str, Dict[str, Any]]:
    """The PUBLISHED record of every pin the plan carries, one ssh batch (``probe`` with ``pins``); the probe of components is skipped."""
    pins: Dict[str, str] = {}
    for e in (plan.get("adapters") or {}).values():
        pins.update({str(s): str(g) for s, g in (e.get("pins") or {}).items()})
    if not pins:
        return {}
    return dict(probe(cfg, dest, [], runner=runner, pins=pins).get("pins") or {})


# --------------------------------------------------------------------------------------------
# the tick
# --------------------------------------------------------------------------------------------


def checkout_name(repo_root: str) -> str:
    """The checkout's directory name, which names its clone on a cluster (``<shared>/repos/<name>``, the repo-mirror tool keys each clone
    by it) and the gate venv: the main working tree of the repository ``repo_root`` belongs to (a linked worktree resolves to the checkout
    it was added from), else ``repo_root`` itself. The landing scripts derive the same name (scripts/registry/landing/site_env.sh)."""
    r = subprocess.run(["git", "-C", repo_root, "rev-parse", "--path-format=absolute", "--git-common-dir"], capture_output=True, text=True)
    common = (r.stdout or "").strip() if r.returncode == 0 else ""
    main = common[: -len("/.git")] if common.endswith("/.git") else os.path.abspath(repo_root)
    return os.path.basename(main.rstrip("/"))


def mirror_selector(dest_name: str, environ: Optional[Mapping[str, str]] = None) -> str:
    """The name the site's repo-mirror tool knows the cluster with registry label ``dest_name`` by (``mirror-repos.sh --cluster <it>``):
    ``MIRROR_REPOS_CLUSTER`` when the environment's ``CLUSTER_DEST`` is that label (a landing script exported both, site_env.sh), else
    the cluster block that carries the label -- its ``CLUSTER_MIRROR_SELECTOR``, else its site label -- among every block file in the
    site's clusters/ directory, listed in SISTEER_CLUSTERS or not (that list is the front door's default fan-out; a read destination
    names its cluster by registry label), read through reproduce/lib/clusters.py (load_addressable, resolve_label), else the label
    itself when no site config, no block or more than one carries it (a tool that does not know it reports so, and the caller notes it)."""
    env = os.environ if environ is None else environ
    if env.get("MIRROR_REPOS_CLUSTER") and env.get("CLUSTER_DEST") == dest_name:
        return str(env["MIRROR_REPOS_CLUSTER"])
    reader = os.path.join(ST.REPO_ROOT, "reproduce", "lib", "clusters.py")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_site_clusters", reader)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)          # type: ignore[union-attr]
        # the site config file: SISTEER_CONFIG, else SISTEER_SITE_DIR's config.env, else the reader's own default (the per-user site dir)
        if env.get("SISTEER_CONFIG"):
            cfg_path = os.path.abspath(env["SISTEER_CONFIG"])
        elif env.get("SISTEER_SITE_DIR"):
            cfg_path = os.path.join(env["SISTEER_SITE_DIR"], "config.env")
        else:
            cfg_path = mod.default_config_path()
        blocks = mod.load_addressable(os.path.dirname(cfg_path), mod.site_config_file(cfg_path))
        label = mod.resolve_label(blocks, dest_name)
    except (Exception, SystemExit):          # no site config, one the reader refuses, or a label no block (or more than one) carries:
        return dest_name                     # the label itself, which the tool then judges
    return str(blocks[label].get("CLUSTER_MIRROR_SELECTOR") or label)


def gate_venv_path(repo_root: str, tree_sha: str, shared: str) -> str:
    """``$SHARED/envs/<checkout name>-gate-<hash12(pyproject.toml, uv.lock at the tree)>`` -- the window's rule, from the git objects;
    the checkout name is :func:`checkout_name` of ``repo_root``."""
    import hashlib
    hs = []
    for name in ("pyproject.toml", "uv.lock"):
        r = subprocess.run(["git", "-C", repo_root, "show", f"{tree_sha}:{name}"], capture_output=True)
        if r.returncode != 0:
            raise CR.ClusterReadsError(f"git show {tree_sha}:{name} in {repo_root} failed: {r.stderr.decode(errors='replace').strip()[:200]}")
        hs.append(hashlib.sha256(r.stdout).hexdigest())
    return f"{shared}/envs/{checkout_name(repo_root)}-gate-{hashlib.sha256((hs[0] + chr(10) + hs[1] + chr(10)).encode()).hexdigest()[:12]}"


def ensure_tree(dest: ST.Destination, tree_sha: str, mirror_repos: str, tree_export: str, runner=None, timeout_s: float = 600.0,
                log: Callable[[str], None] = lambda s: None, mirror_history: Optional[str] = None) -> str:
    """The base tree on the cluster: ``mirror-repos.sh --cluster <selector> <checkout name>`` (:func:`mirror_selector` of the destination,
    :func:`checkout_name` of its source checkout; a problem is a note: tree_export verifies the sha on
    the mirror), then ``mirror_history.sh <source repo> <mirror url>`` -- the repository's archived branches and tags, which mirror-repos.sh's
    heads-and-tags push leaves behind, verified on the mirror (a problem is a note too: the reads run the tree's code only, and a gate that
    reuses the tree completes it against the mirror before the suite) -- then ``tree_export.sh SHA MIRROR TREE`` over ssh (``bash -s`` with
    the script on stdin); a line ``TREE ...`` is success. ``mirror_history`` defaults to the script beside ``tree_export``."""
    run = runner or _default_runner
    r = run([mirror_repos, "--cluster", mirror_selector(dest.name), checkout_name(dest.source_repo)], timeout_s)
    if r.returncode != 0:
        log(f"tree: mirror-repos.sh reported rc {r.returncode} (tree_export verifies the sha on the mirror)")
    history = mirror_history or os.path.join(os.path.dirname(tree_export), "mirror_history.sh")
    rh = run([history, dest.source_repo, dest.mirror_url], timeout_s)
    if rh.returncode != 0:
        tail = ((rh.stdout or "") + (rh.stderr or "")).strip().splitlines()
        log(f"tree: mirror_history.sh reported rc {rh.returncode} ({tail[-1] if tail else 'no output'}); the exported tree may lack archived branches or tags")
    mirror = f"{dest.shared}/{dest.repo_rel}"
    tree = f"{dest.shared}/{CR.TREES_REL}/{tree_sha}"
    with open(tree_export, "r", encoding="utf-8") as fh:
        script = fh.read()
    if dest.is_local:
        argv = ["bash", "-s", "--", tree_sha, mirror, tree]
    else:
        argv = ["ssh", *SSH_OPTS, dest.host or "", "bash", "-s", "--", tree_sha, mirror, tree]
    r2 = run(argv, timeout_s, input=script)
    out = (r2.stdout or "") + (r2.stderr or "")
    if r2.returncode != 0 or not any(line.startswith("TREE ") for line in out.splitlines()):
        raise CR.ClusterReadsError(f"tree export of {tree_sha} failed (rc {r2.returncode}): {out.strip()[-600:]}")
    return tree


def parse_watch(text: str, rc: int) -> str:
    """The watcher's verdict from its lines: ``done`` / ``failed`` / ``cancelled`` / ``timeout`` / ``missing``."""
    if rc == 3 or "WATCH-TIMEOUT" in text:
        return "timeout"
    for line in text.splitlines():
        if line.startswith("DONE "):
            return "done"
        if line.startswith("FAILED "):
            return "cancelled" if "CANCELLED" in line else "failed"
        if line.startswith("MISSING "):
            return "missing"
    return "failed" if rc != 0 else "done"


def _nop_log(s: str) -> None:
    print(s, flush=True)


class Tick:
    """One periodic read operation (the module docstring's THE TICK). Every external step is a hook so the rehearsal can replace it:
    ``slurm`` (queue / accounting / cancel), ``runner`` (the probe's ssh, stage, submit, tree export), ``watch`` (``job_id -> (verdict,
    text)``), ``sleep``, ``now``, ``log``; the registry steps are looked up on their modules at call time (``CR.plan_window`` ...), so
    a test that monkeypatches them is honored."""

    def __init__(self, root: str, cfg: ST.Config, dest: ST.Destination, state_dir: Optional[str] = None, adapters: Optional[Sequence[str]] = None,
                 repo_root: Optional[str] = None, ref: str = "HEAD", op_id: Optional[str] = None, runner=None, remote_runner=None, slurm: Optional[Slurm] = None,
                 watch: Optional[Callable[[str], Tuple[str, str]]] = None, watch_job: Optional[str] = None, watch_bound_s: float = WATCH_BOUND_S,
                 venv: Optional[str] = None, local_venv: bool = False, mirror_repos: Optional[str] = None, tree_export: Optional[str] = None,
                 mirror_history: Optional[str] = None, now: Callable[[], str] = R.utc_now, log: Callable[[str], None] = _nop_log, sleep: Callable[[float], None] = time.sleep,
                 cancel_bound_s: float = CANCEL_BOUND_S, cancel_poll_s: float = CANCEL_POLL_S, dry_run: bool = False, uri_home: Optional[str] = None,
                 blobs: Optional[Mapping[str, Mapping[str, Optional[str]]]] = None, section_bound_s: float = CRITICAL_BOUND_S,
                 section_poll_s: float = CRITICAL_POLL_S):
        self.root, self.cfg, self.dest = root, cfg, dest
        self.section_bound_s, self.section_poll_s = section_bound_s, section_poll_s
        self.rd = reads_dir(state_dir)
        self.adapters = list(adapters) if adapters else sorted(cfg.adapters)
        self.repo_root = os.path.abspath(repo_root or ST.REPO_ROOT)
        self.ref = ref
        self.op_id = op_id or f"tick-{R.utc_now().replace(':', '').replace('-', '')[:15]}Z-{uuid.uuid4().hex[:6]}"
        self.runner = runner or subprocess.run          # the cluster_reads steps' shape: subprocess.run(argv, capture_output=, text=, timeout=, input=)
        self.remote_runner = remote_runner                # the stores shape: runner(argv, timeout_s, input=None) -- the probe, the tree export, Slurm
        self.slurm = slurm or Slurm(dest.host, runner=remote_runner)
        self.watch_job = watch_job or dest.watch_job
        self.watch_bound_s = watch_bound_s
        self._watch = watch
        self.venv, self.local_venv = venv, local_venv
        self.mirror_repos = mirror_repos or os.path.join(os.path.expanduser("~"), "scripts", "mirror-repos.sh")
        self.tree_export = tree_export or os.path.join(ST.REPO_ROOT, "scripts", "registry", "landing", "tree_export.sh")
        self.mirror_history = mirror_history or os.path.join(os.path.dirname(self.tree_export), "mirror_history.sh")
        self.now, self.log, self.sleep = now, log, sleep
        self.cancel_bound_s, self.cancel_poll_s = cancel_bound_s, cancel_poll_s
        self.dry_run = dry_run
        self.uri_home = uri_home
        self._blobs = blobs
        self.summary: Dict[str, Any] = {"op": self.op_id, "state": None}
        self._held = False
        #: THE RULE: ownership is released only after every recorded job of this operation is confirmed terminal. True from the
        #: moment the first job id exists (inside the submit section, before its record write), False only on a confirmed-terminal outcome (the
        #: post-watch confirmation of every id, or a confirmed self-cancel); every other exit -- a watch timeout, an unconfirmed cancel, a
        #: failed journal write, an exception this code did not foresee -- leaves the reservation standing for the next operation
        self._keep_reservation = False
        self._job_ids: List[str] = []       # the ids as submitted, beside the record's: a lost record write never hides a job from the self-cancel

    # -- helpers --------------------------------------------------------------------------

    def _op(self, **fields: Any) -> Dict[str, Any]:
        return update_op(self.rd, self.op_id, now=self.now, owner="tick", **fields)

    def _tree_sha(self) -> str:
        return resolve_tree_sha(self.repo_root, self.ref)

    def _require_ownership(self, where: str) -> None:
        try:
            require_ownership(self.rd, self.op_id)
        except OwnershipLost as exc:
            self.log(f"tick: ownership lost before {where}: {exc}")
            raise

    def watch(self, job_id: str) -> Tuple[str, str]:
        """``watch-job.sh --slurm <array> --host <host> --poll 30 --stall-min 0 --max-hours 1`` in the foreground, bounded in-process at
        ``watch_bound_s`` (the array's walltime + 15 min; the shell script's bound is in whole hours): a timeout kills the watcher (this
        process's own child) and reads ``timeout``."""
        if self._watch is not None:
            return self._watch(job_id)
        argv = [self.watch_job, "--slurm", str(job_id), "--host", str(self.dest.host or "local"), "--poll", "30", "--stall-min", "0", "--max-hours", "1"]
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=self.watch_bound_s)
        except subprocess.TimeoutExpired as exc:
            text = (exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")) if exc.stdout else ""
            return "timeout", text + f"\nWATCH-TIMEOUT (in-process bound {int(self.watch_bound_s)} s)"
        return parse_watch(r.stdout or "", r.returncode), (r.stdout or "") + (r.stderr or "")

    def _journal(self, **fields: Any) -> bool:
        """A best-effort write of the op record: the record is a JOURNAL, and a write that fails (disk full, a permission) changes no outcome
        -- the reservation's fate is decided by ``_keep_reservation`` alone. Logged and noted in the summary; never raises."""
        try:
            self._op(**fields)
            return True
        except Exception as exc:  # noqa: BLE001 -- the journal must not take the outcome down with it
            self.log(f"tick: the op record could not be written ({type(exc).__name__}: {exc}); the outcome stands")
            self.summary["journal_error"] = f"{type(exc).__name__}: {exc}"[:300]
            return False

    def _own_job_ids(self) -> List[str]:
        """This operation's job ids: the ones submitted in this process (in memory) plus the record's (a record write may have been lost,
        and a record may carry ids this process never saw)."""
        ids = list(self._job_ids)
        try:
            op = load_op(self.rd, self.op_id) or {}
        except Exception:  # noqa: BLE001
            op = {}
        ids += [str(v) for v in (op.get("jobs") or {}).values() if v]
        return sorted(set(ids))

    def _cancel_own(self, why: str) -> bool:
        """Cancel this operation's own jobs and confirm them terminal. FAIL-CLOSED: ``_keep_reservation``
        is set BEFORE any fallible work, and only a confirmation of every id clears it. When the cancel or the confirmation fails -- a refused
        scancel, no terminal accounting row within the bound, a failed or TIMED-OUT query (the transport's exceptions are one family,
        :class:`SlurmQueryFailed`), anything a shim raises -- the record says ``cancel_failed`` with the job ids and the error (a best-effort
        write), the reservation is KEPT (the next operation finds a stale holder and cancels and confirms them before anything new is
        submitted) and the caller exits :data:`EXIT_CANCEL_FAILED`. True when every job is confirmed terminal (or there was nothing)."""
        ids = self._own_job_ids()
        if not ids:
            return True
        self._keep_reservation = True                      # before anything that can fail: whatever happens below, no release over these jobs
        self.log(f"tick: cancelling this operation's own job(s) {', '.join(ids)} ({why}) and confirming terminal")
        try:
            self.slurm.cancel(ids)
            states = self.slurm.wait_terminal(ids, bound_s=self.cancel_bound_s, poll_s=self.cancel_poll_s, sleep=self.sleep)
        except Exception as exc:  # noqa: BLE001 -- ReconcileRefusal / SlurmQueryFailed, or whatever the cluster layer raised: all unconfirmed
            self.log(f"tick: the self-cancel of {', '.join(ids)} could not be confirmed: {type(exc).__name__}: {exc}; the reservation is KEPT over them")
            self._journal(state="cancel_failed", jobs=dict((load_op(self.rd, self.op_id) or {}).get("jobs") or {}) or {f"job{i}": j for i, j in enumerate(ids)},
                          jobs_terminal={"error": f"{type(exc).__name__}: {exc}"}, cancel_error=f"{type(exc).__name__}: {exc}"[:800], cancel_failed_utc=self.now())
            return False
        self._keep_reservation = False                     # every recorded job confirmed terminal (I5): ownership may be released
        self._journal(jobs_terminal=states)
        return True

    def _cancel_failed(self, why: str, exc: BaseException) -> int:
        self.log(f"tick: FAIL-CLOSED -- {why}; this operation's job(s) are not proven terminal; the reservation is kept; exit {EXIT_CANCEL_FAILED}")
        self._journal(ended_utc=self.now(), error=f"{why}: {type(exc).__name__}: {exc}"[:800])
        self.summary.update(state="cancel_failed", why=why, error=str(exc)[:400], reservation_kept=True)
        return EXIT_CANCEL_FAILED

    def pull_orphan(self, op: Mapping[str, Any]) -> Dict[str, Any]:
        """The reconcile's pull of a completed-unpulled operation: its plan file, revalidate -> pull -> re-activations -> inbox-merge ->
        receipt, under THIS operation's reservation; the pull's barriers apply unchanged."""
        return pull_and_receipt(self.root, self.cfg, self.dest, self.rd, str(op["op_id"]), runner=self.runner, remote_runner=self.remote_runner,
                                now=self.now, log=self.log, blobs=self._blobs, by=self.op_id)

    # -- the run --------------------------------------------------------------------------

    def run(self) -> int:
        previous = signal.getsignal(signal.SIGTERM)

        def on_term(signum, frame):  # noqa: ARG001
            raise TickInterrupted("SIGTERM")
        try:
            signal.signal(signal.SIGTERM, on_term)
        except (ValueError, OSError):      # not the main thread (tests)
            previous = None
        try:
            return self._run()
        except TickInterrupted as exc:
            self.log("tick: interrupted (SIGTERM); releasing")
            if not self._cancel_own("interrupted"):
                return self._cancel_failed("interrupted, self-cancel unconfirmed", exc)
            self._journal(state="interrupted", ended_utc=self.now())
            self.summary["state"] = "interrupted"
            return EXIT_INTERRUPTED
        except OwnershipLost as exc:
            # a takeover between two of this operation's steps: whatever this operation submitted is cancelled by its own hand and confirmed
            # (the takeover cancelled and confirmed the recorded ids itself before it took the reservation; this is the belt)
            self.log(f"tick: {exc}; pulling nothing")
            if not self._cancel_own(f"ownership lost ({exc})"):
                return self._cancel_failed("ownership lost, self-cancel unconfirmed", exc)
            self._journal(state="cancelled" if exc.cancelled_by else "ownership_lost", ended_utc=self.now())
            self.summary.update(state="cancelled" if exc.cancelled_by else "ownership_lost", why=str(exc))
            return EXIT_CANCELLED if exc.cancelled_by else EXIT_OWNERSHIP
        except CriticalSectionBusy as exc:
            # the section stayed held past the bound (a window's pull / merge / receipt): nothing of ours is touched; a completed array of
            # ours -- confirmed terminal after the watch -- is HANDED to the next reconcile (section_busy is a handed-off state); the
            # reservation is released only if nothing of ours is unconfirmed (the flag decides, as everywhere)
            self.log(f"tick: {exc}; handing the operation to the next reconcile")
            self._journal(state="section_busy", error=str(exc), ended_utc=self.now())
            self.summary.update(state="section_busy", error=str(exc))
            return EXIT_SECTION_BUSY
        except ReconcileRefusal as exc:
            # a refused or timed-out query / cancel (one family: SlurmQueryFailed is one), or an unconfirmed holder at the reservation step
            self.log(f"tick: {exc}")
            if self._keep_reservation and not self._cancel_own(f"unconfirmed ({type(exc).__name__})"):
                return self._cancel_failed("the recorded jobs are not confirmed terminal", exc)
            self._journal(state="failed", error=str(exc), ended_utc=self.now())
            self.summary.update(state="reconcile_refused", error=str(exc))
            return EXIT_RECONCILE
        except (CR.ClusterReadsError, ST.StoresError, OSError, ValueError, subprocess.SubprocessError) as exc:
            self.log(f"tick: FAILED -- {type(exc).__name__}: {exc}")
            if not self._cancel_own(f"failed: {type(exc).__name__}"):
                return self._cancel_failed(f"failed ({type(exc).__name__}), self-cancel unconfirmed", exc)
            self._journal(state="failed", error=f"{type(exc).__name__}: {exc}"[:800], ended_utc=self.now())
            self.summary.update(state="failed", error=f"{type(exc).__name__}: {exc}"[:400])
            return 2
        except BaseException as exc:  # noqa: BLE001 -- KeyboardInterrupt, SystemExit, anything unforeseen: the flag decides the reservation
            if self._keep_reservation:
                # jobs of ours not proven terminal and the process is ending: no cancel is attempted (it could stall the exit), the record
                # says so, the reservation stands for the next operation to cancel and confirm them -- then the exception goes on
                self.log(f"tick: {type(exc).__name__} with job(s) not proven terminal; the reservation is KEPT; the next operation cancels and confirms them")
                self._journal(state="cancel_failed", cancel_error=f"{type(exc).__name__}: the process is ending; the self-cancel was not attempted"[:800],
                              cancel_failed_utc=self.now())
                self.summary.update(state="cancel_failed", why=type(exc).__name__, reservation_kept=True)
            raise
        finally:
            try:
                if self._held and self._keep_reservation:
                    # the rule: jobs of ours not proven terminal -> the reservation stands; this process ends
                    # with it, so the next operation replaces it as stale -- cancelling and confirming our jobs first -- before anything is submitted
                    self.log("tick: the reservation is KEPT over job(s) not proven terminal; the next operation cancels and confirms them first")
                    self.summary["release"] = "kept"
                elif self._held:
                    try:
                        release(self.rd, self.op_id, bound_s=self.section_bound_s, poll_s=self.section_poll_s, log=self.log, now=self.now)
                    except CriticalSectionBusy as exc:
                        # the release re-reads under the lock a takeover may be holding; past the bound the
                        # reservation is left standing: this process ends with it, so the next operation replaces it as stale
                        self.log(f"tick: the reservation could not be released ({exc}); it stands until the next operation replaces it as stale")
                        self.summary["release"] = "section_busy"
            except Exception as exc:  # noqa: BLE001 -- the release is best-effort too: a standing reservation is replaced as stale
                self.log(f"tick: the release failed ({type(exc).__name__}: {exc}); the reservation stands until the next operation replaces it as stale")
                self.summary["release"] = f"error: {type(exc).__name__}"
            if previous is not None:
                try:
                    signal.signal(signal.SIGTERM, previous)
                except (ValueError, OSError):
                    pass
            print(json.dumps(self.summary, sort_keys=True, default=str), flush=True)

    def _run(self) -> int:
        tree_sha = self._tree_sha()
        self.summary["tree_sha"] = tree_sha
        # 1. the reservation first (a live holder: yield; a dead one on this host: replaced after its jobs are confirmed terminal)
        try:
            how, _res = take_reservation(self.rd, self.slurm, "tick", self.op_id, tree_sha, takeover=False, now=self.now, log=self.log,
                                         bound_s=self.cancel_bound_s, poll_s=self.cancel_poll_s, sleep=self.sleep,
                                         section_bound_s=self.section_bound_s, section_poll_s=self.section_poll_s)
        except ReservationHeld as exc:
            self.log(f"tick: yields -- {exc}")
            self._op(state="yielded", tree_sha=tree_sha, held_by=exc.holder, ended_utc=self.now())
            self.summary.update(state="yielded", held_by=exc.holder)
            return EXIT_OK
        except ReconcileRefusal as exc:
            self.log(f"tick: the stale holder's jobs could not be confirmed terminal: {exc}")
            self._op(state="failed", tree_sha=tree_sha, error=str(exc), ended_utc=self.now())
            self.summary.update(state="reconcile_refused", error=str(exc))
            return EXIT_RECONCILE
        self._held = True
        self.summary["reservation"] = how
        self.log(f"tick {self.op_id}: reservation {how}")
        # 2. reconcile before anything is submitted
        try:
            rec = reconcile(self.rd, self.slurm, self.op_id, now=self.now, log=self.log, bound_s=self.cancel_bound_s, poll_s=self.cancel_poll_s,
                            sleep=self.sleep, pull_orphan=None if self.dry_run else self.pull_orphan, section_bound_s=self.section_bound_s,
                            section_poll_s=self.section_poll_s)
        except CriticalSectionBusy:
            raise
        except ReconcileRefusal as exc:
            self.log(f"tick: reconcile refused: {exc}")
            self._op(state="failed", error=str(exc), ended_utc=self.now())
            self.summary.update(state="reconcile_refused", error=str(exc))
            return EXIT_RECONCILE
        self._op(state="reconciled_others", reconcile=rec)
        self.summary["reconcile"] = {k: rec[k] for k in ("left_alive", "cancelled", "pulled", "settled")}
        # 3. one ssh batch + the module blobs -> the changed set
        slugs = sorted({s for a in self.adapters for s in (self.cfg.adapters[a].components if a in self.cfg.adapters else [])})
        tree = f"{self.dest.shared}/{CR.TREES_REL}/{tree_sha}"
        probed = probe(self.cfg, self.dest, slugs, runner=self.remote_runner, tree=tree)
        blobs = self._blobs if self._blobs is not None else module_blobs(self.repo_root, self.adapters, ref=self.ref)
        self._blobs = blobs
        recs = receipts(self.rd)
        cs = changed_set(self.cfg, self.adapters, probed, blobs, lambda a: last_receipt_for(self.rd, a, recs),
                         standing=lambda a: standing_ids(self.root, self.rd, a, recs), tree_sha=tree_sha)
        self._op(state="probed", changed=cs["changed"], skipped=cs["skipped"], unchanged=cs["unchanged"], tree_present=probed.get("tree_present"),
                 probe_utc=probed.get("utc"))
        self.summary.update(changed=sorted(cs["changed"]), skipped=cs["skipped"], unchanged=cs["unchanged"], tree_present=probed.get("tree_present"))
        for a, why in sorted(cs["changed"].items()):
            self.log(f"tick: {a} changed -- " + "; ".join(why))
        for a, why in sorted(cs["skipped"].items()):
            self.log(f"tick: {a} skipped -- {why}")
        if self.dry_run:
            self._op(state="dry_run", ended_utc=self.now())
            self.summary["state"] = "dry_run"
            return EXIT_OK
        if not cs["changed"]:
            self.log("tick: nothing changed; nothing to read")
            self._op(state="unchanged", ended_utc=self.now())
            self.summary["state"] = "unchanged"
            return EXIT_OK
        changed = sorted(cs["changed"])
        # 4. the base tree on the cluster
        if not probed.get("tree_present"):
            self.log(f"tick: exporting the tree {tree_sha[:9]} to the cluster")
            ensure_tree(self.dest, tree_sha, self.mirror_repos, self.tree_export, runner=self.remote_runner, log=self.log,
                        mirror_history=self.mirror_history)
        # 5. the reader state of the changed adapters (read-only on the live database)
        from . import reader_state as RS
        sidecar = sidecar_path(self.rd, self.op_id)
        os.makedirs(os.path.dirname(sidecar), exist_ok=True)
        exp = RS.export(self.root, changed, sidecar, uri_home=self.uri_home, repo_root=self.repo_root)
        self._op(state="exported", sidecar=sidecar, export=exp.as_dict() if hasattr(exp, "as_dict") else str(exp))
        # 6. the plan from the probe's pins (manifests and declarations only; no store read)
        published = {slug: row["current"] for slug, row in probed["components"].items() if row.get("current")}
        from .adapters import load_all
        plan = CR.plan_window(self.root, self.op_id, changed, self.cfg, self.dest, tree_sha, uri_home=self.uri_home, repo_root=self.repo_root, published=published,
                              registry=load_all(),        # the plan-time checks on the constructed instance, as the window's plan runs them
                              sidecar=sidecar)            # the standing retirement holds, off the snapshot just exported for these shards (closing pass)
        plan["op"] = {"owner": "tick", "op_id": self.op_id}
        plan_file = CR.plan_path(self.root, self.op_id)
        CR._write_json(plan_file, plan)
        self._op(state="planned", plan=plan_file, adapters=changed, routes={a: e["route"] for a, e in plan["adapters"].items()})
        rc = EXIT_OK
        verdict = None
        if plan.get("tasks"):
            # 7. stage (outside the lock: the sidecar's transfer), then SUBMIT INSIDE THE CRITICAL SECTION: the
            # ownership check, both sbatch calls and the durable write of the job ids to the op record are one interval nothing can
            # interleave -- a takeover that arrives waits these few seconds and then always finds the recorded ids, which it cancels and
            # confirms before the reservation changes hands. The section is held during the SUBMISSION only, never while the array waits in
            # the queue or runs (the hold is measured and recorded: `submit_section_s`)
            info = CR.stage_window(plan, sidecar, plan_file, runner=self.runner)
            CR.record_stage(plan, "stage", **info)
            CR._write_json(plan_file, plan)
            self._op(state="staged")
            venv = self.venv or gate_venv_path(self.repo_root, tree_sha, self.dest.shared)
            with critical_section(self.rd, self.op_id, "tick", "submit", bound_s=self.section_bound_s, poll_s=self.section_poll_s, log=self.log, now=self.now):
                t_submit = time.monotonic()
                self._require_ownership("submit")
                attest = CR.submit_attest(plan, runner=self.runner, job_name=job_name("attest", self.op_id))
                # THE FIRST JOB ID EXISTS: from here ownership is released only once every recorded job is confirmed terminal,
                # whatever happens -- the flag and the in-memory id are set BEFORE the record write, which may itself fail
                self._keep_reservation = True
                self._job_ids = [str(attest)]
                self._op(jobs={"attest": attest})
                # the array's sbatch failing leaves the attestation job RECORDED: the failure handler's self-cancel (or a takeover's cancel) ends it
                array = CR.submit_array(plan, venv, job_name("reads", self.op_id), runner=self.runner, local_venv=self.local_venv, dependency=attest)
                self._job_ids = [str(attest), str(array)]
                hold_s = round(time.monotonic() - t_submit, 3)
                self._op(state="submitted", jobs={"attest": attest, "reads": array}, submitted_utc=self.now(), submit_section_s=hold_s)
            CR.record_stage(plan, "submit", job_id=array, attest_job_id=attest, tasks=len(plan["tasks"]), local_venv=self.local_venv, op=self.op_id)
            CR._write_json(plan_file, plan)
            self.summary.update(array=array, attest=attest, submit_section_s=hold_s)
            self.log(f"tick: submit section held {hold_s} s (ownership check, two sbatch, the job ids recorded)")
            self._require_ownership("watch")          # no longer a race guard (the section closed it): a takeover that landed since has
                                                      # cancelled and confirmed our recorded jobs; reading the loss here spares the watch
            # 8. watch in the foreground, bounded
            self.log(f"tick: array {array} (attest {attest}) submitted; watching")
            verdict, text = self.watch(array)
            self._journal(state="watched", watch=verdict, watch_tail=text[-2000:])
            self.summary["watch"] = verdict
            if verdict == "timeout":
                # the array is LIVE: the reservation is KEPT (the rule above). This process ends; the next operation finds a stale
                # holder and cancels and confirms the array before anything new is submitted -- the hand-over the reconcile used to do
                self.log(f"tick: the watcher timed out after {int(self.watch_bound_s)} s with the array live; the reservation is KEPT for the next operation")
                self._journal(state="watch_timeout", ended_utc=self.now())
                self.summary.update(state="watch_timeout", reservation_kept=True)
                return EXIT_WATCH_TIMEOUT
            # the watcher read the array terminal; the module's OWN proof (I5) for EVERY recorded id -- the attestation included -- is what
            # clears the flag: an explicit terminal accounting row and the queue's silence, both queries answering. A refusal or a transport
            # timeout here is a ReconcileRefusal -> the self-cancel -> fail-closed
            states = self.slurm.wait_terminal(self._job_ids, bound_s=self.cancel_bound_s, poll_s=self.cancel_poll_s, sleep=self.sleep)
            self._keep_reservation = False
            self._journal(jobs_terminal=states)
            self.summary["jobs_terminal"] = states
        else:
            self._op(state="no_task", note="every adapter was recorded refused at plan time; nothing submitted")
        # 9 + 10. INSIDE the critical section: the ownership check and the whole pull / merge / receipt are one
        # section a takeover cannot land in the middle of -- a window arriving now waits for the receipt; a window that took over before the
        # lock was ours is what the check reads (the array ended: nothing of ours is left to cancel)
        with critical_section(self.rd, self.op_id, "tick", "pull", bound_s=self.section_bound_s, poll_s=self.section_poll_s, log=self.log, now=self.now):
            self._require_ownership("pull" if plan.get("tasks") else "receipt")
            if verdict == "cancelled":
                op = load_op(self.rd, self.op_id) or {}
                self.log(f"tick: the array was CANCELLED ({op.get('cancelled_by') or 'not by a recorded takeover'}); pulling nothing")
                self._op(state="cancelled", ended_utc=self.now())
                self.summary["state"] = "cancelled"
                return EXIT_CANCELLED
            out = pull_and_receipt(self.root, self.cfg, self.dest, self.rd, self.op_id, runner=self.runner, remote_runner=self.remote_runner, now=self.now,
                                   log=self.log, blobs=blobs, by=self.op_id, plan=plan, plan_file=plan_file)
        refused = sorted(out["refusals"])
        self.summary.update(state="receipted", pulled=out["pulled"], refused=refused, receipt=out["receipt"])
        try:
            os.remove(sidecar)
        except OSError:
            pass
        if refused:
            self.log("tick: REFUSED adapter(s) this read: " + "; ".join(f"{a}: {'; '.join(out['refusals'][a])}" for a in refused))
            rc = EXIT_REFUSED
        return rc


def pull_and_receipt(root: str, cfg: ST.Config, dest: ST.Destination, rd: str, op_id: str, runner=None, remote_runner=None, now: Callable[[], str] = R.utc_now,
                     log: Callable[[str], None] = lambda s: None, blobs: Optional[Mapping[str, Mapping[str, Optional[str]]]] = None, by: Optional[str] = None,
                     plan: Optional[Dict[str, Any]] = None, plan_file: Optional[str] = None, conn=None) -> Dict[str, Any]:
    """The grading-host half after an array, THREE RESUMABLE STAGES each persisted when it completes: revalidate (pass 2)
    -> PULL (the barriers, the refused-known re-activations from committed bytes, the refusal set; recorded as the plan's ``pull`` stage, then
    on the op record: state ``pulled``) -> INBOX-MERGE (``H.scan_inbox``; the plan's ``inbox_merge`` stage, then the record: ``merged``) ->
    RECEIPT (``receipts/<op_id>.json``; the record ``receipted``). A call resumes at the FIRST stage the plan does not record and never
    skips one: a reconcile that takes an operation whose process died after persisting ``pull`` merges, then receipts. Invariant I7 --
    no receipt without a recorded merge of this operation's pull -- is checked right before the receipt (:class:`ReceiptRefused`). Shared by
    the tick (one critical-section interval), the window's ``cluster-reads receipt`` (its ``pull --op`` recorded the pull and the merge in
    an earlier interval; a missing merge is run here first) and the reconcile's orphan pull (``by`` names the operation doing the work)."""
    if plan is None:
        plan_file = plan_file or CR.plan_path(root, op_id)
        plan = CR._load_json(plan_file)
    assert plan_file is not None
    stages = plan.setdefault("stages", {})
    owner = str((plan.get("op") or {}).get("owner") or (load_op(rd, op_id) or {}).get("owner") or "window")
    if "revalidate_post" not in stages:
        stale = CR.revalidate_plan(root, plan, cfg)
        CR._write_json(plan_file, plan)
        if stale:
            log(f"pull: post-run revalidation REFUSED {', '.join(stale)}")
    if "pull" not in stages:
        expect = plan_file[:-len(".json")] + ".summaries" if plan_file.endswith(".json") else plan_file + ".summaries"
        rep = CR.pull_window(root, plan, expect, runner=(runner or subprocess.run))
        reactivations = CR.reactivate_refused_known(root, plan, rep, conn=conn, now=now())
        withheld = rep.get("refused_known_withheld") or {}
        CR.record_stage(plan, "pull", pulled=len(rep["pulled"]), pulled_dirs=sorted(d for d, v in rep["pulled"].items() if v.get("state") == "complete"),
                        depends_dirs=dict(rep.get("depends") or {}),
                        quarantined=len(rep["quarantined"]), withheld=len(rep["withheld"]), unlisted=sorted(rep.get("unlisted") or {}),
                        partial=sorted(rep.get("partial") or {}), partial_aside=dict(rep.get("partial_aside") or {}), barrier=rep.get("barrier"),
                        rsync=rep["rsync"], summaries=rep["summaries"], attestation=rep.get("attestation"),
                        reactivations={a: {r: v.get("state") for r, v in per.items()} for a, per in reactivations.items()}, refused_known_withheld=withheld,
                        standing_refusals=dict(rep.get("standing_refusals") or {}), by=by)
        plan["refusals"] = CR.refused_adapters(plan, rep)
        CR._write_json(plan_file, plan)
        record_op_stage(rd, op_id, "pull", now=now, by=by, owner=owner, tree_sha=plan.get("tree_sha"))
    if "inbox_merge" not in stages:
        merged = H.scan_inbox(root)
        # I7: what the receipt rests on is DURABLE state -- every revision this operation depends on, read back from records/
        # against its recorded files -- never the scan's return (an interrupted scan has already moved a conflict aside and cannot report it again)
        evidence = merge_evidence(root, depends_of_plan(plan))
        problems = apply_merge_problems(plan, evidence)
        CR.record_stage(plan, "inbox_merge", merged=sum(1 for _, s in merged if s == "merged"), duplicate=sum(1 for _, s in merged if s == "duplicate"),
                        conflict=sum(1 for _, s in merged if s == "conflict"), incomplete=sum(1 for _, s in merged if s == "incomplete"), by=by,
                        evidence={r: v["state"] for r, v in evidence.items()}, problems=problems)
        CR._write_json(plan_file, plan)
        record_op_stage(rd, op_id, "inbox_merge", now=now, by=by, stage_fields={"problems": problems, "evidence": {r: v["state"] for r, v in evidence.items()}}, owner=owner)
        for a, whys in sorted(problems.items()):
            log(f"merge: {a} REFUSED in the receipt -- " + "; ".join(whys))
    missing = [st for st in ("pull", "inbox_merge") if st not in stages]
    if missing:
        raise ReceiptRefused(f"receipt refused for {op_id}: the plan {plan_file} records no {' / '.join(missing)} stage -- no receipt without a "
                             f"recorded merge of this operation's pull (invariant I7)")
    # the receipt-time re-check (I7): the evidence is a pure function of records/ and the inbox NOW -- a conflict or a missing file that an
    # interrupted or earlier merge left is refused here whatever the recorded stage said; idempotent on a clean merge
    evidence_now = merge_evidence(root, depends_of_plan(plan))
    late = apply_merge_problems(plan, evidence_now)
    if late:
        CR.record_stage(plan, "receipt_check", evidence={r: v["state"] for r, v in evidence_now.items()}, problems=late)
        CR._write_json(plan_file, plan)
        for a, whys in sorted(late.items()):
            log(f"receipt: {a} REFUSED -- the merge evidence at receipt time: " + "; ".join(whys))
    op = load_op(rd, op_id) or new_op(owner, op_id, plan.get("tree_sha"), now)
    published = published_for_plan(cfg, dest, plan, runner=remote_runner)
    if blobs is None:
        adapters = sorted(plan.get("adapters") or {})
        try:
            blobs = module_blobs(os.path.abspath(str(plan.get("repo_root") or ST.REPO_ROOT)), adapters, ref=str(plan.get("tree_sha") or "HEAD"))
        except CR.ClusterReadsError as exc:
            log(f"receipt: module blobs unavailable ({exc}); recorded empty")
            blobs = {a: {} for a in adapters}
    rec = receipt_from_plan(plan, published, blobs, op, now=now)
    rec["by"] = by
    path = write_receipt(rd, rec)
    record_op_stage(rd, op_id, "receipt", now=now, by=by, owner=op.get("owner") or owner, receipt=path, ended_utc=now(), receipted_by=by)
    pulled = len((stages.get("pull") or {}).get("pulled_dirs") or [])
    return {"receipt": path, "refusals": dict(plan.get("refusals") or {}), "pulled": pulled, "n_read_whole": rec["n_read_whole"]}


# --------------------------------------------------------------------------------------------
# CLI (dispatched from cluster_reads.cmd_cluster_reads)
# --------------------------------------------------------------------------------------------


def add_arguments(sub) -> None:
    s = sub.add_parser("reserve", help="the grading host: take the cluster-reads reservation for an operation (O_EXCL; the first step of every tick and window), "
                                       "then reconcile every reads-*/attest-* job of this user with the op records; --takeover (the window) cancels a "
                                       "tick's array by its recorded id, confirms it terminal in sacct and takes the reservation; a live window holding it "
                                       "is refused by name (exit 24), and so is THIS operation id held by another process -- the same WINDOW_ID twice "
                                       "(exit 24, nothing reserved: a retry needs a new id)")
    s.add_argument("--op", required=True, help="the operation id (the window's name)")
    s.add_argument("--owner", choices=["window", "tick"], default="window")
    s.add_argument("--tree-sha")
    s.add_argument("--takeover", action="store_true")
    s.add_argument("--pid", type=int, help="the LONG-LIVED holder the reservation names (the window shell's $$): this CLI process exits when the "
                                           "command returns, so without it the holder would read as dead at the next tick; verified alive, with "
                                           "a readable start time, and a proper ancestor of this process (exit 2 otherwise, nothing reserved)")
    s.add_argument("--no-reconcile", action="store_true", help="take the reservation only (tests)")
    s.add_argument("--state", help=f"the landing state directory (default ${STATE_ENV}, else ~/.local/state/registry-landing)")
    s.add_argument("--dest")
    s.add_argument("--config")
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("release", help="the grading host: release the reservation when it names this operation (a stale release never removes another's); "
                                       "the check and the unlink run under the namespace's critical section (exit 26 when it stays busy: the "
                                       "reservation stands)")
    s.add_argument("--op", required=True)
    s.add_argument("--state")
    s = sub.add_parser("receipt", help="the grading host: write receipts/<op>.json from the window's plan file after its pull (per adapter the pins, each pin's "
                                       "PUBLISHED record, the module blob shas, shards ok / total, the revisions pulled, the refusal) and mark the "
                                       "operation receipted; runs the revalidate / pull / merge itself when the plan lacks them")
    s.add_argument("--window", required=True)
    s.add_argument("--op", help="default: the window")
    s.add_argument("--state")
    s.add_argument("--dest")
    s.add_argument("--config")
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("tick", help="the grading host (registry-cluster-reads.timer, every 10 min): reservation -> reconcile -> ONE ssh batch over every "
                                    "component's `current` -> the changed set (a pin moved since the last whole receipt, a module blob changed, or "
                                    "no receipt) -> nothing changed: exit 0; else reader-state export, plan, stage, attestation + array, watch, "
                                    "revalidate, pull, inbox-merge, receipt. Exit 15 = the read recorded REFUSED adapters; 20 cancelled by a "
                                    "window's takeover; 21 watcher timeout (the array is live: the reservation is kept for the next operation to "
                                    "cancel and confirm); 22 an outstanding job could not be confirmed terminal; 26 the "
                                    "critical section stayed busy past its bound (the operation is the next reconcile's); 27 FAIL-CLOSED: this "
                                    "operation's own jobs could not be cancelled and confirmed -- the reservation is kept for the next operation "
                                    "to cancel and confirm them first")
    s.add_argument("--state")
    s.add_argument("--dest")
    s.add_argument("--config")
    s.add_argument("--adapter", nargs="*", help="narrow the tick to these adapters (default: every declared adapter)")
    s.add_argument("--repo-root", help="the base checkout whose HEAD is the tree the shards run (default this checkout)")
    s.add_argument("--ref", default="HEAD")
    s.add_argument("--venv", help="the gate venv on the cluster (default: derived from the tree's pyproject.toml + uv.lock)")
    s.add_argument("--local-venv", action="store_true")
    s.add_argument("--dry-run", action="store_true", help="reservation, reconcile (no orphan pull), probe and the changed set only; nothing exported or submitted")
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("changed", help="the grading host: the adapters whose inputs changed since their last whole receipt (one per line; --json for the reasons): "
                                       "the probe's pins against the receipts, the adapter modules' blob shas at --landing (default HEAD, resolved to "
                                       "the commit), and, for a receipt carrying standing refusals, its tree against that commit -- the tick's rule")
    s.add_argument("--state")
    s.add_argument("--dest")
    s.add_argument("--config")
    s.add_argument("--adapter", nargs="*")
    s.add_argument("--repo-root")
    s.add_argument("--landing", help="the ref whose adapter modules are compared and whose commit a standing-refusal receipt's tree is compared with (the landed sha; default HEAD)")
    s.add_argument("--from-receipts", action="store_true", help="(the default and only source; accepted for the window's spelling)")
    s.add_argument("--json", action="store_true")


def cmd_reads(args, root: str) -> int:
    """The protocol's CLI actions; returns the exit code (the module's EXIT_* for the protocol's outcomes, 2 for an error)."""
    rd = reads_dir(getattr(args, "state", None))
    if args.action == "release":
        try:
            ok = release(rd, args.op, log=_nop_log)
        except CriticalSectionBusy as exc:
            print(f"release: {exc}; the reservation stands", file=sys.stderr)
            return EXIT_SECTION_BUSY
        print(f"release: {'released' if ok else 'not held by'} {args.op}")
        return 0
    cfg = ST.load_config(getattr(args, "config", None), root=root)       # a destination joins this root's [[clusters]] entry
    dest = cfg.destination(getattr(args, "dest", None))
    if args.action == "reserve":
        slurm = Slurm(dest.host)
        pid = pid_start = None
        if getattr(args, "pid", None) is not None:
            # the holder is the caller's long-lived shell, not this CLI process: verified before anything is written
            try:
                pid, pid_start = verify_holder_pid(args.pid)
            except HolderPidInvalid as exc:
                print(f"reserve: REFUSED -- {exc}; nothing reserved", file=sys.stderr)
                return 2
        try:
            how, res = take_reservation(rd, slurm, args.owner, args.op, args.tree_sha, takeover=bool(args.takeover), log=_nop_log, pid=pid, pid_start=pid_start)
        except OperationIdInUse as exc:
            # this id is held by ANOTHER process (a second window with the same WINDOW_ID); never held_by_me
            print(f"reserve: REFUSED -- {exc}; nothing reserved", file=sys.stderr)
            print(json.dumps({"held_by": exc.holder, "me": exc.me, "operation_id_in_use": True}, indent=1) if args.json
                  else f"held_by {exc.holder.get('owner')} {exc.holder.get('op_id')} (pid {exc.holder.get('pid')}: the same operation id under another process)")
            return EXIT_HELD
        except ReservationHeld as exc:
            print(f"reserve: REFUSED -- {exc}", file=sys.stderr)
            print(json.dumps({"held_by": exc.holder}, indent=1) if args.json else f"held_by {exc.holder.get('owner')} {exc.holder.get('op_id')}")
            return EXIT_HELD
        except ReconcileRefusal as exc:
            print(f"reserve: {exc}", file=sys.stderr)
            return EXIT_RECONCILE
        rec: Dict[str, Any] = {}
        if not args.no_reconcile:
            try:
                rec = reconcile(rd, slurm, args.op, log=_nop_log,
                                pull_orphan=lambda op: pull_and_receipt(root, cfg, dest, rd, str(op["op_id"]), log=_nop_log, by=args.op,
                                                                        plan_file=CR.plan_path(root, str(op["op_id"]))))
            except OwnershipLost as exc:
                print(f"reserve: {exc}; the reservation changed hands before the reconcile -- nothing reconciled", file=sys.stderr)
                return EXIT_OWNERSHIP
            except ReconcileRefusal as exc:
                try:
                    released = release(rd, args.op, log=_nop_log)
                except CriticalSectionBusy as busy:
                    released = False
                    print(f"reserve: {busy}", file=sys.stderr)
                print(f"reserve: reconcile refused -- {exc}; the reservation is {'released' if released else 'NOT released (it stands)'}", file=sys.stderr)
                return EXIT_RECONCILE
            update_op(rd, args.op, state="reconciled_others", reconcile=rec, owner=args.owner)
        out = {"how": how, "reservation": res, "reconcile": {k: rec.get(k) for k in ("left_alive", "cancelled", "pulled", "settled")} if rec else None}
        print(json.dumps(out, indent=1, default=str) if args.json else
              f"reserve: {how} for {args.owner} {args.op} (holder pid {res.get('pid')})" + (f"; reconcile: cancelled {sorted(rec['cancelled'])}, pulled {sorted(rec['pulled'])}, "
                                                                                           f"left alive {rec['left_alive']}" if rec else ""))
        return 0
    if args.action == "receipt":
        op_id = args.op or args.window
        plan_file = CR.plan_path(root, args.window)
        try:
            # the ownership check and the pull / merge / receipt are one critical section
            with critical_section(rd, op_id, "window", "receipt", log=_nop_log):
                try:
                    require_ownership(rd, op_id)
                except OwnershipLost as exc:
                    print(f"receipt: {exc}; nothing pulled or written", file=sys.stderr)
                    return EXIT_OWNERSHIP
                if not os.path.isfile(plan_file):
                    print(f"receipt: no plan for window {args.window!r} at {plan_file}", file=sys.stderr)
                    return 2
                out = pull_and_receipt(root, cfg, dest, rd, op_id, log=_nop_log, by=op_id, plan_file=plan_file)
        except CriticalSectionBusy as exc:
            print(f"receipt: {exc}; nothing pulled or written", file=sys.stderr)
            return EXIT_SECTION_BUSY
        print(json.dumps(out, indent=1, default=str) if args.json else
              f"receipt: {out['receipt']} -- {out['n_read_whole']} adapter(s) read whole, {out['pulled']} revision(s) pulled"
              + (", REFUSED: " + ", ".join(sorted(out["refusals"])) if out["refusals"] else ""))
        return 0
    if args.action == "tick":
        t = Tick(root, cfg, dest, state_dir=getattr(args, "state", None), adapters=args.adapter or None, repo_root=args.repo_root, ref=args.ref,
                 venv=args.venv, local_venv=bool(args.local_venv), dry_run=bool(args.dry_run))
        return t.run()
    if args.action == "changed":
        adapters = list(args.adapter) if args.adapter else sorted(cfg.adapters)
        slugs = sorted({s for a in adapters for s in (cfg.adapters[a].components if a in cfg.adapters else [])})
        probed = probe(cfg, dest, slugs)
        repo_root = os.path.abspath(args.repo_root or ST.REPO_ROOT)
        # the SAME tree the tick would read with: --landing resolved to the commit, the module blobs read at it,
        # and changed_set handed it, so a receipt with standing refusals recorded under another tree lists its adapter here as well
        tree_sha = resolve_tree_sha(repo_root, args.landing or "HEAD")
        blobs = module_blobs(repo_root, adapters, ref=tree_sha)
        recs = receipts(rd)
        cs = changed_set(cfg, adapters, probed, blobs, lambda a: last_receipt_for(rd, a, recs), standing=lambda a: standing_ids(root, rd, a, recs),
                         tree_sha=tree_sha)
        cs["tree_sha"] = tree_sha
        print(json.dumps(cs, indent=1, default=str) if args.json else "\n".join(sorted(cs["changed"])))
        return 0
    return 2
