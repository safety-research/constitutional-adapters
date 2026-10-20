#!/bin/bash
# window.sh — one registry landing window on the grading host (registry-slurm-fast-windows, Phase 1; v1 = the operator's /tmp/reg-ff-chain.sh
# re-homed into the repo; since the cluster-only move: no grading host store read, the honest drain, the retired readers confirmed gone,
# the end marker written by the trap). Every landing's live effect waits for a window: the maintenance flag is written, the timers stop,
# the retired reader services are stopped and CONFIRMED gone (their cgroups empty by PID list), the writers drain, the base checkout
# fast-forwards to origin/main, the UNITS ARE INSTALLED from it (`install_grading_host.sh --units-only`), the schema migrates, the dimension seeds
# load when they changed, every named adapter reads ONCE ON THE CLUSTER (or is RECORDED REFUSED -- the grading host reads no store), ONE ingest
# folds, and the EXIT trap removes the flag, restores the timers, verifies them and writes the END MARKER. EVERY stage's exit status is
# checked: a failure aborts the window at once (nothing after it runs; the trap still restores the timers; the harvester folds whatever
# the cluster published), and the end marker `=== ff chain end` is written ONLY when every mandatory stage ran AND the restore set is
# verified `active`; otherwise `=== ff chain ABORTED (<cause>) <ISO>`, so an aborted window is never counted as completed and the
# measured window includes the restore. Log stamps keep the shape scripts/registry/measure/windows.py parses:
#
#   === ff chain start <ISO> ADAPTERS=...   -- reads reservation HH:MM:SS   timers active at stop: ...   writers quiet HH:MM:SS   -- wal checkpoint HH:MM:SS
#   -- config convert HH:MM:SS   -- store authority HH:MM:SS   base at <sha> HH:MM:SS   -- install units HH:MM:SS   -- uninstall units HH:MM:SS (...)   -- migrate (plan) HH:MM:SS
#   -- migrate (apply) HH:MM:SS   -- dimensions load [SKIPPED (...)] HH:MM:SS   -- reconcile X (dry run|apply) HH:MM:SS   -- restamp keys (plan|apply) HH:MM:SS   -- acceptance before-phase HH:MM:SS
#   -- reader-state export HH:MM:SS  -- tree export HH:MM:SS   -- cluster reads plan|stage|submit|revalidate|pull|receipt HH:MM:SS
#   -- adapter A REFUSED HH:MM:SS (reason)   -- roster load N HH:MM:SS   --- ingest HH:MM:SS   --- ingest end HH:MM:SS
#   -- reconcile multi-turn-scope (verify 1) HH:MM:SS   [-- reconcile multi-turn-scope (apply, second pass) HH:MM:SS   -- reconcile multi-turn-scope (ingest) HH:MM:SS   -- reconcile multi-turn-scope (verify 2) HH:MM:SS]
#   reads reservation released HH:MM:SS   timers restored (...) HH:MM:SS   === ff chain end <ISO>  |  === ff chain ABORTED (<cause>) <ISO>
#
# The migration plan is READ and CHECKED before anything is applied (`-- migrate (plan)`; the record-log branch's review, 2026-10-04:
# the stage used to tail six lines of the plan, ignore its exit status and apply at once, so the documented STOP on a `rebuild` line
# could not be enforced by anyone). The whole `registry migrate --plan` output goes to the log and to
# $STATE/windows/migrate-plan-<window>.txt, its PLAN TOKEN (the sha256 of that file) printed beside it; its version line must render
# (else exit 8); any `rebuild table` line ABORTS the window before the apply (exit 15; the 0020 rebuild of row_history wrote a 25 GB
# WAL) unless WINDOW_ALLOW_REBUILD=1 names a reviewed rebuild -- no approval overrides that STOP; WINDOW_EXPECT_MIGRATION=<NNNN_slug>
# pins the pending list to exactly that migration (else exit 15; the same check scripts/registry/migrate_window.sh makes with
# --expect); WINDOW_PLAN_APPROVAL=<file> PAUSES a plan that has something to apply (pending migrations, or a shape that differs) until
# the operator, having read the displayed plan, writes its token into that file with the command the log prints
# (`echo <token> > <file>`) -- bounded by WINDOW_PLAN_WAIT_S (1800) at the quiet-poll cadence, else exit 16 with nothing applied. The
# approval is BOUND to the displayed plan (the review's second round: a file created while the plan was still being generated used to
# pass): a file present when the stage begins is removed before the plan runs, a file present once the complete plan has been displayed
# is removed too (written during the plan, so nobody had seen it whole), and a file whose content is not this plan's token is ignored
# with a logged line while the wait goes on (the operator overwrites it); the accepted approval is consumed. A plan with nothing pending
# and the shape matching needs no approval.
#
# ONE WINDOW AT A TIME, enforced (the review's third round: a second window fast-forwarding the base while the first waited for its
# approval would have had the first apply the NEW migration files -- `registry migrate` enumerates the directory's CURRENT files --
# under an approval of the OLD plan): an exclusive flock on $STATE/windows/.window.lock is taken before anything else (before the start
# marker, the reads reservation, the timer stop and the fast-forward) and held until the process exits, after the trap's restore; a second window is refused
# by name at once (exit 18, the holder's line printed, no timer touched, no start marker, so the measurement counts no window). And
# the base is PINNED from the plan to the apply: the base HEAD at the fast-forward, the migration directory's file set with each file's
# sha256, and the plan file's sha256 (= the token) are recorded beside the plan file (migrate-plan-<window>.pins) and RE-CHECKED
# immediately before `registry migrate`; any difference refuses the apply (exit 19, both readings printed, nothing applied, the timers
# restored). The plan file is written once (a file already at its name is a refusal, exit 8 -- under the lock before anything moves, and
# again at the plan stage as a race guard; the window id carries the pid beside the second-resolution stamp so two windows never share a
# name), made read-only, and every check reads the text read ONCE into memory -- the token is the sha256 of those bytes -- never the file
# again.
#
# The reconciliation slot (`-- reconcile X`): WINDOW_RECONCILE names `registry reconcile` actions, in the order they run (`run-patch-extra`;
# `multi-turn-scope`, the steering scope of the runs of the multi-turn class whose scope disagrees with what their mode, their eval's
# category and their object's kind derive -- the job-start default the common case -- one run_patch per run, the dry run counting them by
# model and category and by object kind; `env-comments`, the run-overrides stamps that differ from the reading of their own overrides
# file (matched by the file's sha256; WINDOW_ENV_COMMENTS_SOURCES names the files, passed as --source to its every invocation and to
# run-patch-extra's dry run, which compares a legacy stamp to the live one as readings of one of those files), one
# run_patch per run row restating the whole stamp from the file, the dry run counting them by overrides file and by key, listing the
# stamps whose file was not given, and counting the other tables' rows it does not correct -- a stamp whose file was not given and whose
# values may carry a comment is NOT a clean class: the apply refuses it by name (exit 2) and the verify lists it (exit 3) until its file
# is named or WINDOW_ENV_COMMENTS_LEAVE_UNSOURCED=1 leaves such stamps as stored by name (`--leave-unsourced` on the action's every
# invocation); and when env-comments is listed for apply without that word, `registry reconcile env-comments --check-sources` runs under
# the worktree's CLI BEFORE the lock and the reservation (`env-comments sources:` one line; its whole text in
# $STATE/windows/env-comments-sources-<window>.txt): every stamped run row whose values may carry a comment must have its file among the
# sources, else the window is refused by name (exit 30, nothing moved) rather than applying its migration on a source list that covers
# nothing and completing with every stored value as it was; `fit-effective-batch`, the object rows whose fit_effective_batch
# is the micro-batch a sidecar hook read where the row's own recorded hyperparameters say the optimizer width, one object record per row
# restating the recorded value, the dry run listing the ambiguous rows it does not correct). Each runs its dry run between the dimensions
# load and the cluster stages, so the records it files fold in this window's ONE ingest. WINDOW_RECONCILE_APPLY is a LIST of the named
# actions to run with --apply after their dry run (e.g. "multi-turn-scope env-comments"); every other named action runs as a dry run only;
# unset, empty or `0` (the old off switch) applies nothing. A name in the list that WINDOW_RECONCILE does not name, and the retired global
# `1` (which applied EVERY named action -- a window naming run-patch-extra beside a correction aborted at run-patch-extra's release gate),
# are CONFIGURATION REFUSALS by name, exit 30, before the lock and the reservation (one line, no start marker, nothing moved). The CLI's own gate still decides an apply: a refused apply,
# exit 2 -- the release gate of run-patch-extra, or a row held behind an unfolded record that states the column the pass corrects (the
# pass's own correction, a standup's endpoint patch pushed but not yet ingested, a full run record, a native re-emission) -- aborts the
# window by name: the operator asked for a write that did not happen; ingest, then re-run the slot. A failing dry run, apply or verify
# exits 17. Order: a lifted run-patch-extra apply re-files a legacy stamp as its record stated it; env-comments corrects a re-filed stamp
# from the file once it is folded, so it reads it in the window after.
#
# THE VERIFIED FIXED POINT (after `--- ingest end`, for every correction action -- multi-turn-scope, env-comments, fit-effective-batch --
# that WINDOW_RECONCILE names AND WINDOW_RECONCILE_APPLY lists, in WINDOW_RECONCILE's order). A pass snapshots a row, walks the log and
# re-reads the row's own directory at the write, but a statement published into ANOTHER record directory after that last read is seen by
# none of it, and the pass's later-written statement, derived from the snapshot, wins the fold: for multi-turn-scope a standup's endpoint
# patch (a served mode and the scope that follows it) leaves (`response`, `every_turn_prefill_and_decode`), a rebuild agreeing. Rather than
# a lock across every producer or a conditional record, the window runs each pass to a VERIFIED fixed point: `-- reconcile <action>
# (verify 1)` reads the predicate alone over the FOLDED database (`registry reconcile <action> --verify`: exit 0 none, 3 some, read-only);
# when it lists rows, `(apply, second pass)` reconciles them ONCE more from the now-folded row (exit 2, a row held: 17), `(ingest)` folds
# them -- the one time this window ingests twice (exit 12 on a failure) -- and `(verify 2)` must read zero, else the window fails BY NAME
# with the ids (exit 29). Every verify's whole text goes to the log (the ids are the evidence) and to $STATE/windows/<file>-<window>-<n>.txt
# (`scope-verify` for multi-turn-scope, `verify-<action>` for the others). Endpoint patches state mode and scope together, so the steady
# state stays consistent and only a one-off pass can race; a second pass converges because each pass's predicate is its verify's own. An
# object has no endpoint-patch race and a run-overrides stamp is restated whole, so their verify 1 is expected to read zero; the second
# pass stands for the case it does not.
#
# THE LIFECYCLE RESTAMP SLOT (`-- restamp keys (plan|apply)`). WINDOW_RESTAMP_KEYS=<file> names a
# JSON list of run ids (the certified C1 mover class, cutover/refold27/expected_keys.json) whose lifecycle columns are RE-DERIVED on the LIVE
# database under the fast-forwarded base's reducer (C1: the deterministic same-stamp tie-break) -- `registry replica restamp-keys`, the
# registry's own refold writer (`replica.refold_apply`), never a second one. It runs after the reconciliation slot: the writers are quiet
# (timers stopped, drained, the WAL checkpointed), the maintenance flag is HELD (checked here: no flag, no apply), the base has
# fast-forwarded and the schema is current. First the PLAN -- every (run, column, old, new) the apply would write, listed in the log and
# written to $STATE/windows/restamp-<window>/restamp_plan.json; a key with no live row or no history, or ANY change outside the lifecycle
# class (lifecycle_state and the stamps a run_state event owns; the CASE columns of the paper frame read it), REFUSES by name and the window
# aborts (exit 28) with nothing written -- then the APPLY, bound to that plan (re-computed and required identical), ONE transaction, and the
# VERIFY: every changed row re-read against the plan (a mismatch is exit 14 from the CLI, 28 here). The counts (keys, rows rewritten,
# runs changed, changes verified) are the CLI's last line in the log. Unset: one line `restamp keys: skipped`, nothing else. After the
# certified movers adopt the C1 order this way, the landing chain's LAND_ACCEPT_C1_CLASS allowance is retired.
#
# THE READS RESERVATION. A window and the periodic tick (registry-cluster-reads.timer, every
# 10 min) are both READ OPERATIONS over one cluster namespace, owned through $STATE/reads/reservation.json (sisteer.registry.reads_protocol).
# The window's FIRST act after the start marker is `registry cluster-reads reserve --owner window --op <window> --takeover --pid $$` --
# under the WORKTREE's CLI ($WINDOW_NEW_PY / $WINDOW_NEW_PYPATH, like `config --check` and the store authority stage), never the base's:
# the base has not fast-forwarded yet and at the cutover its code had no `reserve` (the release in the trap runs the same way): the
# reservation names THIS SHELL ($$, verified alive and a proper ancestor of the CLI; recorded with its /proc start time) as the holder, not
# the CLI process that exits when the command returns -- else a tick during the hour-long store-mirror wait would read a dead holder,
# replace it and cancel this window's jobs. A tick that holds the reservation has its array cancelled BY ID
# (recovered by name, reads-<op>, when its record has none) and confirmed TERMINAL in sacct (an explicit terminal accounting row AND out of
# the queue, per job id) before the reservation changes hands -- and the hand-over waits for a tick mid-pull to finish its pull, merge and
# receipt (the namespace's critical section, $STATE/reads/critical.lock); then every reads-*/attest-* job of this user is reconciled with the
# op records (an outstanding job with no live owner is cancelled and confirmed; a completed array nobody pulled is pulled now, every barrier
# of the pull applying). A LIVE window holding it refuses this one by name (exit 22) -- and so does THIS WINDOW_ID held by another process
# (a second window started with the same id, an operator retry while the first runs: the two would share the plan and the job names;
# exit 22, nothing reserved -- a retry needs a new id). The cluster stages pass `--op <window>` to `submit`
# and `pull`, which re-check the ownership before a job is submitted and before anything is pulled or merged -- the pull's check, the
# transfer and the merge inside the same critical section, recorded as the operation's `pull` and `inbox_merge` stages (the CLI's exit 23 -> that
# stage fails -> every cluster adapter is RECORDED REFUSED; the receipt refuses without a recorded merge), and
# `-- cluster reads receipt` writes $STATE/reads/receipts/<window>.json (per adapter the pins, each pin's PUBLISHED
# record, the module blob shas, shards ok / total, the revisions pulled, the refusal) -- the tick's changed set and S7 read the receipts.
# The EXIT trap releases the reservation (`cluster-reads release --op <window>`: only when it still names this window, checked and unlinked
# under the critical section, so a takeover cannot lose its reservation to this release) BEFORE the timers
# restore, so the first tick after the window finds nothing held. registry-cluster-reads.timer is in the stopped set and in the default set:
# newly installed, the trap `enable --now`s it (the cutover); active at stop, it is restarted.
#
# The acceptance slot (`-- acceptance before-phase`; the adapter re-anchoring plan 7.3): ACCEPTANCE_OUT=<dir> runs the acceptance
# script's BEFORE-PHASE (ACCEPTANCE_SCRIPT, the site's acceptance script, required with ACCEPTANCE_OUT; ACCEPTANCE_LIST the materialised
# bumped-adapter list the after-phase is held to) after the migrate and the dimensions load and BEFORE the reader-state export, because
# its item-evidence snapshot and manifest must sit between the timer stop and the export (the sidecar's ledger_head is bound to the
# manifest's record_ingest head); a FAIL aborts the window (exit 25; 20 is the recorded-refusal abort). The protocol requires the sidecar, so when the window names no
# adapter with a cluster policy the export runs here anyway, over $ADAPTERS; the cluster block's own export is unchanged.
#
# NO POST-INGEST PASS (the merge of 2026-10-06). Main's POST_INGEST_ADAPTERS named
# readers whose rows depend on the FRESH frame (sampled_static's pooled contrasts over the pooled rows the first ingest folded), each read
# once more on the grading host after the first ingest and folded by a SECOND ingest. The grading host reads no store (C6, S2), so a set
# POST_INGEST_ADAPTERS is a CONFIGURATION REFUSAL, by name and before the lock and the reservation: one line in the log, exit 26, no start
# marker (no window to the measurement), nothing reserved, stopped, fast-forwarded or installed. A second read is the next window's first
# read: every window exports the frame fresh (the reader-state sidecar), so this window's ingest is already in it. A cluster-side second
# pass (export -> plan -> stage -> submit -> pull after the first ingest) is a follow-up, not this script.
#
# THE CONFIGURATION CONVERSION. `-- config convert` runs AFTER the writers are
# quiet and BEFORE the base fast-forwards -- before any new-code registry invocation, because the new loader (C10) REFUSES a
# configuration that still carries a `modal` table: this directory's convert_config.py (stdlib) backs $REGISTRY_ROOT/config.toml up to
# state/config.toml.bak-<stamp>, strips the table textually and rewrites the file atomically (a file without the table is a no-op:
# `CONVERTED=0`, nothing written). The result is validated TWICE, with the commands that exist: (a) under THIS WORKTREE's new code,
# `config --check` (C10's sub-command: exit 0 on a clean configuration, a named non-zero refusal on a Modal table or destination
# anywhere) -- the worktree's python ($WINDOW_NEW_PY, default <worktree>/.venv/bin/python when it exists, else the base's) with
# PYTHONPATH=<worktree>/src ($WINDOW_NEW_PYPATH); (b) under the BASE's OLD code, `$REG config --json` (the only validation the old
# loader offers: it must parse and print the file the old timers keep running on until the fast-forward). A failure of either
# RESTORES THE BACKUP BYTE FOR BYTE (convert_config.py --restore, verified by sha256) and aborts with exit 24 -- the trap then restores
# the OLD timers under the OLD code and the ORIGINAL configuration, and NEVER_START keeps the retired units down. Both validations'
# output goes to $STATE/windows/config-{check,json}-<window>.txt. The converter runs under $WINDOW_CONVERT_PY (default the new python;
# it needs a Python >= 3.11 for tomllib). R12 rehearses both outcomes from a copy of a real configuration.
#
# THE RETIRED UNITS UNINSTALLED (C7, extended in C10). Right after `-- install units`, `-- uninstall units` disables each unit of
# RETIRED_UNITS (registry-adapters.timer, registry-adapters.service: replaced by the cluster-reads tick; registry-modal-pull.timer,
# registry-modal-pull.service: removed with the Modal transport -- no unit file of theirs exists in the repo any
# more, so the installer cannot copy them back), removes its file from the unit directory, removes the Modal-era DROP-INS of
# RETIRED_DROPINS (the whole registry-modal-pull.service.d/ directory, and registry-mirrors.service.d/clusters-only.conf -- the hand
# pause of the Modal log mirror, redundant since the destination is gone; every other drop-in is left alone) and `daemon-reload`s
# (exit 23 on a failure). The retired SERVICES are still stopped and confirmed gone first (C6).
#
# The restore set (printed to the log) = the timers active at stop ∪ the default timers NOT installed before this window − NEVER_START
# (registry-snapshot.timer, which nothing here ever starts -- the snapshot unit stays the operator's call -- and the two RETIRED units,
# registry-adapters.timer and registry-modal-pull.timer, which no failure path, no trap and no `enable --now` ever names again; C6;
# both are uninstalled by this window, C7/C10).
# Newly installed timers are `enable --now`ed (so they survive a reboot; registry-cluster-reads.timer joins the default set in C7, so the
# cutover's trap enables the tick); previously active ones are `start`ed. After the restore,
# registry-status-heavy.service is started ONCE (--no-block) as the status page's cache initializer — only when the install stage succeeded.
#
# STORE AUTHORITY (since the merge of 2026-10-06). The store-mirror TIMER is stopped with the others: until the cutover it runs MAIN's
# `stores mirror`, which seals and records nothing, so one tick after a grading host store grows re-publishes the component's cluster `current`
# in main's format (bfcl_full and bigcodebench_full at 15:39Z on 2026-10-06, both certified hours earlier), and it ticks every 30 minutes --
# inside a window too (17:01Z and 17:31Z in W29). "Authority clean" therefore cannot be a precondition read before the window; it is
# established INSIDE it: `-- store authority`, after `-- config convert` (the new loader reads the configuration) and before the base
# moves, under the WORKTREE's new code ($WINDOW_NEW_PY / $WINDOW_NEW_PYPATH). (a) `stores check-authority --json`; every component whose
# `current` exists and reads VERIFIED but UNSEALED or UNRECORDED gets ONE `stores verify-authority --component <slug> --out
# <state>/windows/authority-<id>/` (the verify job on Slurm, the seal, the record, the certificate), bounded per component by
# WINDOW_AUTHORITY_WAIT_S (default 1800 s; ~1 min per component measured 2026-10-06 over four main-published generations); a NEVER-PUBLISHED
# grading_host_copy component (one the new configuration declares and no generation of which exists on the destination) is CARRIED FIRST under
# the worktree's code -- `stores mirror --dest <dest> --component <slug>`, bounded the same way -- and then verify-authority'd like the rest
# (five such components, published by no one before the strict check, once aborted a cutover-shaped window before the fast-forward that
# would have let the timer carry them); a carry that fails is logged by name and the strict check names the component. A dangling `current`
# or one whose grading host copy moved ahead of `current` (unverified) is NOT acted on here -- the mirror timer carries it -- and is named in the log. (b) `stores check-authority
# --require-certified --json`; the counts (certified / checked) and the problem slugs go to the log, both JSON results beside the window's
# other artifacts. (c) With WINDOW_REQUIRE_CERTIFIED=1 (the cutover window) a component still not certified, or a check that could not run,
# aborts the window (exit 27) with the base and the units untouched; a routine window logs the reading and goes on. Post-cutover the same
# timer runs the branch's `stores mirror` (seal + record), so the stage is a one-time re-certification and then a standing guard.
#
# THE GRADING HOST READS NO STORE (C6, spec S2). Every adapter the window names ($ADAPTERS) reads on the cluster through the cluster stages below,
# or is RECORDED REFUSED with its reason -- `-- adapter A REFUSED HH:MM:SS (reason)` -- and read nowhere: a failed cluster stage, a refused
# component, a missing / failed shard, a failed pull barrier, a declaration `cluster: refused`, an adapter outside the cluster set. The
# window still ingests what the other adapters published and then ends ABORTED (`=== ff chain ABORTED (refused: A B) <ISO>`, exit 20).
# The ONE grading host read left is the operator's: an adapter in $REREAD_FORCE_ADAPTERS runs `adapters run --reread --force` here only when
# REGISTRY_STORE_READ_OVERRIDE=<reason> is set in this environment (the override the registry logs); otherwise it is refused like the rest.
#
# THE HONEST DRAIN. `state/maintenance.json` is written BEFORE the timers stop: `registry ingest --honor-maintenance` (the
# harvest unit's mode) exits 0 at once while it exists, before the cluster pull and before the write lock, so the drain is at most the
# cycle in flight; the trap removes the flag. `paused.json` is the disk-pressure flag and is never touched here.
#
# THE RETIRED READERS, STOPPED AND CONFIRMED. Before the base moves, registry-adapters.service and
# registry-modal-pull.service are stopped with the timers and each is CONFIRMED terminal: `is-active` reads inactive or failed, `show -p
# MainPID` reads 0, and the unit's cgroup (`show -p ControlGroup` -> $WINDOW_CGROUP_ROOT<path>/cgroup.procs) holds NO PID -- a PID list,
# never a command-text match; a cgroup still populated after WINDOW_CONFIRM_TRIES × WINDOW_CONFIRM_SLEEP_S (60 × 1 s) fails the window
# (exit 21) with the PIDs named. Nothing is killed by text.
#
# Cluster reads (registry-slurm-fast-windows Phase 3b; sisteer.registry.cluster_reads): CLUSTER_ADAPTERS names the adapters with a
# cluster policy (default: every adapter the window names -- the configuration declares every adapter, with
# components or as `cluster: refused`; `none` = no cluster read, every named adapter refused). Stages, stamped like the rest: `-- store
# mirror` (step 0, AFTER the timers stop and BEFORE the drain: registry-store-mirror.service started and awaited -- the BASE's pass,
# quiesced -- and then THE CARRY UNDER THE WORKTREE'S CODE, `$NEW_REG stores mirror --dest <dest>` bounded by WINDOW_MIRROR_WAIT_S: a
# landing that DECLARES new components (the census landing added five) is carried in-window with the configuration that names them, which
# the base's unit cannot see; a pre-fast-forward step runs under the new tree, and a window of 2026-10-07 aborted on exactly this) · `-- reader-state export` · `-- tree export` ·
# `-- cluster reads plan` (admission pass 1 on the grading host: every component re-validated against its pin; a difference records the adapter
# refused, named in the log and in state/cluster_reads/<window>.json; NO discovery runs here) · `-- cluster reads stage` · `-- cluster reads
# submit` (watch-job.sh in the foreground) · `-- cluster reads revalidate` (pass 2; a stale copy quarantines that adapter's pulled
# directories) · `-- cluster reads pull` (marker last, behind the failure barrier; the refused-known revisions re-activated from committed
# bytes) + inbox-merge · then `-- adapter A REFUSED` for every adapter `cluster-reads refusals` names, then ONE ingest. A cluster stage that
# FAILS refuses every cluster adapter and the window goes on to the ingest, ending ABORTED (refused: ...).
#
# Knobs (all optional): ADAPTERS (default: the ctl_neg1 trio; `none` = no adapter, an adapter-less window) · KEEP_TIMERS_STOPPED=1 · FORCE_DIMS=1 · ROSTER_LOAD_VERSION=N ·
# WINDOW_ALLOW_REBUILD=1 · WINDOW_EXPECT_MIGRATION=NNNN_slug · WINDOW_PLAN_APPROVAL=/path · WINDOW_PLAN_WAIT_S (1800) ·
# WINDOW_RECONCILE="run-patch-extra multi-turn-scope env-comments fit-effective-batch" (the actions, in the order they run; each a dry run) ·
# WINDOW_RECONCILE_APPLY="multi-turn-scope env-comments fit-effective-batch" (the LIST of named actions to --apply after their dry run; every
# other named action stays a dry run; unset, empty or `0` applies nothing; a name WINDOW_RECONCILE lacks, or the retired global `1`, is
# refused by name, exit 30) · WINDOW_ENV_COMMENTS_SOURCES="/path/a.env /path/dir" (env-comments' overrides files, each a file or a
# directory of .env files, passed as --source to its dry run, apply and verify, and to run-patch-extra's every invocation, whose stamp
# comparison reads them; a path that is not there is refused by name, exit 30; no spaces in a path; when env-comments is listed for
# apply, `--check-sources` runs before the lock and a stamped run row whose values may carry a comment and whose file is not among them
# is refused by name, exit 30) · WINDOW_ENV_COMMENTS_LEAVE_UNSOURCED=1 (the operator's word: the stamps no source covers are left as
# stored -- `--leave-unsourced` on env-comments' dry run, apply and verify, and no sources check; any value but 1, 0 or empty is refused
# by name, exit 30) ·
# WINDOW_RESTAMP_KEYS=/path/to/keys.json (below) · WINDOW_REQUIRE_CERTIFIED=1 · WINDOW_AUTHORITY_WAIT_S (1800) ·
# ACCEPTANCE_OUT=/dir (+ ACCEPTANCE_LIST=/file, ACCEPTANCE_SCRIPT=/path) · (POST_INGEST_ADAPTERS set = refused by name, exit 26; above) ·
# WINDOW_STATE_DIR (default ~/.local/state/registry-landing; the log is $STATE/windows/chain.log) · WINDOW_LOG_COMPAT (default
# /tmp/reg-migrate-chain.log, the pre-repo log the measurement reads too; empty = none) · WINDOW_BASE_CHECKOUT (the site's checkout, SITE_REPO below) ·
# WINDOW_UNIT_DIR (~/.config/systemd/user) · WINDOW_INSTALLER (the base's install_grading_host.sh) · REGISTRY_ROOT (~/registry) ·
# REGISTRY_PY / REGISTRY_PYTHONPATH (the base's venv / src) · WINDOW_QUIET_TRIES (240) / WINDOW_QUIET_SLEEP_S (15) ·
# WINDOW_CONFIRM_TRIES (60) / WINDOW_CONFIRM_SLEEP_S (1) / WINDOW_CGROUP_ROOT (/sys/fs/cgroup) ·
# WINDOW_NEW_PY / WINDOW_NEW_PYPATH (the worktree's interpreter and src: the reads reservation and its release, the new loader's
# `config --check`, the store authority stage -- every registry call before `base at` that the base may not know) · WINDOW_CONVERT_PY
# (the interpreter convert_config.py runs under; Python >= 3.11) ·
# WINDOW_BASE_BUSY_PATTERN (processes that must not be running when the base moves) · CLUSTER_ADAPTERS (the adapters with a cluster
# policy, default every named adapter; `none`) · CLUSTER_HOST · CLUSTER_SHARED · CLUSTER_DEST · MIRROR_REPOS_CLUSTER · CLUSTER_REPO_MIRROR (each
# from the site config, THE CLUSTER NAMES below; an environment value wins) · CLUSTER_VENV (the gate venv
# $CLUSTER_SHARED/envs/<checkout name>-gate-<hash12(pyproject.toml, uv.lock)>) · WINDOW_WATCH_JOB
# (~/scripts/watch-job.sh) · MIRROR_REPOS_SH (~/scripts/mirror-repos.sh) · MIRROR_HISTORY_SH (this directory's mirror_history.sh: the
# archived branches and tags reach the repo mirror before the tree export; a problem is a note here, the reads need code only) ·
# WINDOW_MIRROR_WAIT_S (3600: the store mirror's bound) · WINDOW_SITE_CUTOVER (the prepared set; THE SITE CUTOVER below) ·
# WINDOW_EXPECT_SITE_ALIASES_SHA256 (the site-alias digest the cutover must install) · WINDOW_PREFLIGHT=1 (the preflight in a routine
# window) · WINDOW_PREFLIGHT_ONLY=1 · WINDOW_PREFLIGHT_SHA · WINDOW_REPLICA_ROOT ($CLUSTER_SHARED/registry/replica).
#
# THE CLUSTER NAMES (site_env.sh beside this script). Each cluster has three names: the site label (the site config's
# clusters/<label>.env), the registry label (CLUSTER_SUBSTRATE: the registry config's [[clusters]] name, every --dest) and the mirror
# selector (CLUSTER_MIRROR_SELECTOR, default the site label; only mirror-repos.sh --cluster takes it). CLUSTER_HOST, CLUSTER_SHARED,
# CLUSTER_DEST, MIRROR_REPOS_CLUSTER and the checkout come from ONE cluster block: the primary cluster's, or, with CLUSTER_DEST set,
# the block that carries that registry label (any clusters/<label>.env, listed in SISTEER_CLUSTERS or not), so the preflight and
# every later stage address the same cluster; a value the site config cannot supply, or a CLUSTER_DEST no block carries, is
# refused by name before the lock (exit 32), so no host or path is built in. The clone on the cluster (CLUSTER_REPO_MIRROR) and the
# gate venv derive from the checkout's directory name: $CLUSTER_SHARED/repos/<name> and $CLUSTER_SHARED/envs/<name>-gate-<hash12>.
#
# THE SITE CUTOVER (WINDOW_SITE_CUTOVER=<prepared set>): one window moves the live site to new cluster names. The prepared set is a
# directory holding the new registry config.toml, site_aliases.toml, the site config (config.env, clusters/*.env) and MANIFEST.tsv
# (the format is in site_cutover.py). With it set:
#   - the cluster names come from the prepared set's site config from the start (site_env.sh, SISTEER_CONFIG pointed into the set);
#     the reads reservation, taken before anything changes, runs under the base's code with the cluster's current registry label
#     (the prepared alias table read backwards): the code and the configuration the stopped timers share
#   - registry-mirrors.timer and registry-mirrors-full.timer join the stop set; their services keep destination-keyed state, so a
#     pass in flight is waited out by the drain and each service is then confirmed terminal with an empty cgroup (exit 31 otherwise)
#   - the store mirror's carry under the worktree's code and the new loader's `config --check` wait for the new configuration,
#     because the worktree's code knows only the new names
#   - `-- site cutover` runs right after `-- config convert`, inside the drain. site_cutover.py refuses by name (exit 31, nothing
#     touched) on an interrupted delivery (a .recv-* directory or a RECEIVING.json under an old inbox name), a reads operation record
#     that is not terminal, a held critical.lock or harvester.lock, a stopped unit still running (the mirrors included), a live file
#     whose sha256 is not the manifest's "before", or an inbox target that already exists. Otherwise it backs every file up under
#     $STATE/windows/site-cutover-<window>/, installs the prepared files atomically, renames the inbox directories and validates them
#     under the worktree's code (config --check, the site-alias digest); a failure there restores every backup byte for byte and moves
#     the inboxes back before exit 31. The installed site config must then read the cluster names the prepared set declares.
#   - `-- preflight` runs next (below)
#   - until the migration is applied, every abort undoes the cutover in the trap BEFORE any timer restarts: the unit directory's
#     registry entries as they were before the install (unit files, drop-ins, enablement links), the base checkout back at the commit it
#     was fast-forwarded from, and the cutover's backups, so the old timers come back on the old code and the old configuration. A
#     restore that cannot finish leaves the timers stopped and says so (exit 31). After the apply the database is at the new schema and
#     the way out is forward. A failed apply whose database's schema version moved anyway (or can no longer be read) counts as applied:
#     nothing is rolled back and every timer stays stopped, by name (exit 8).
#
# THE PREFLIGHT (`-- preflight`, with WINDOW_SITE_CUTOVER, or WINDOW_PREFLIGHT=1 in a routine window), before the store authority stage,
# the fast-forward and the migration, so nothing irreversible has run when it fails (exit 32; the trap restores the cutover):
#   - the repo-mirror tool runs with the mirror selector (a selector it does not know fails here, not in the tree export)
#   - the commit the window moves to (origin/main, fetched) is on CLUSTER_REPO_MIRROR, pushed there by object id when it is not; the
#     fast-forward then refuses an origin/main that moved since
#   - every [[clusters]] entry names its qos, and its remote_src holds sisteer/registry/__init__.py (over ssh)
#   - the registry root's site_aliases.toml and the cluster replica root's (WINDOW_REPLICA_ROOT) have the same site-alias digest
# WINDOW_PREFLIGHT_ONLY=1 (with WINDOW_SITE_CUTOVER) runs the same checks and nothing else: no lock, no reservation, no flag, no stop,
# nothing installed. It reads the prepared set itself (its config.env through SISTEER_CONFIG, its config.toml and site_aliases.toml as
# the registry root's) and checks the commit of the tree this script runs from (WINDOW_PREFLIGHT_SHA overrides); it prints to the
# terminal and to $STATE/windows/preflight-<window>.txt, and exits 0, or 32 naming the check that failed.
#
# THE UNITS (`-- install units`): the fast-forwarded base's installer renders the unit files for the site into a scratch directory
# first (--units-only --no-systemctl, the site config through SISTEER_CONFIG); their difference from the installed units goes to the
# log and to $STATE/windows/units-diff-<window>.txt; then the installer installs and reloads. With WINDOW_SITE_CUTOVER the installed
# unit files are backed up first, for the trap's restore.
#
# Exit codes: 1 no base checkout · 3 writers never quiet · 4 base dirty · 5 fast-forward failed · 6 base busy · 7 install failed ·
# 8 migrate failed, or a reused WINDOW_ID whose plan file exists (refused under the lock before anything moves) · 9 dimensions load failed · 10 adapter pass failed (the operator-override reread; `adapters run` exit 3 = no adapters
# yet is a success) · 11 roster load failed · 12 ingest failed · 13 a timer did not stop (reads other than `inactive`) · 14 a restored
# timer does not read `active` (from the trap; an earlier stage's code stands when the window had already failed) · 15 the migration plan
# has a `rebuild table` line (or is not the WINDOW_EXPECT_MIGRATION one): nothing applied · 16 the plan was not approved within
# WINDOW_PLAN_WAIT_S: nothing applied · 17 a reconcile dry run, apply or verify failed (the fixed point's second-pass apply included) · 18 another window holds $STATE/windows/.window.lock
# (refused before the start marker; no timer touched) · 19 the base HEAD, the migration files or the plan file changed between the plan
# and the apply: nothing applied · 20 an adapter was RECORDED REFUSED (never read; the ingest of the rest ran) · 21 a retired reader
# service still has a process after the stop (its cgroup PID list named) · 22 the reads reservation could not be taken (a live window
# holds it, or an outstanding job of a dead operation could not be confirmed terminal) · 23 the retired units could not be uninstalled ·
# 24 the configuration conversion failed or a converted configuration failed one of its two validations (the backup restored byte for
# byte; before the base moved). · 25 the acceptance before-phase (ACCEPTANCE_OUT) failed, or the reader-state export it requires ·
# 26 POST_INGEST_ADAPTERS is set: a grading host post-ingest adapter pass, refused by name before the lock and the reservation (one line, no
# start marker; nothing reserved, stopped, fast-forwarded or installed) · 27 the store authority stage under WINDOW_REQUIRE_CERTIFIED=1: a
# component not certified after its verify-authority, or a check-authority that could not run (before the base moved; nothing applied) ·
# 28 the lifecycle restamp slot (WINDOW_RESTAMP_KEYS): the keys file missing, the maintenance flag not held, a plan refused by name
# (an unknown key, a change outside the lifecycle class: nothing applied), a failed apply, or a verification that did not read the plan ·
# 29 a correction reconcile (multi-turn-scope, env-comments, fit-effective-batch) reached no fixed point: after the apply, the ingest, a
# second apply and a second ingest the verify still lists rows (their ids in the abort line and in the verify's text above it; the second
# ingest ran) · 30 WINDOW_RECONCILE_APPLY names an action WINDOW_RECONCILE does not, or is the retired global `1`, or
# WINDOW_ENV_COMMENTS_SOURCES names a path that is not there, or WINDOW_ENV_COMMENTS_LEAVE_UNSOURCED is neither 1, 0 nor empty, or
# env-comments is listed for apply and its sources check (`--check-sources`, under the worktree's CLI) lists a stamped run row whose
# values may carry a comment and whose file is not among the sources (or could not run): refused by name before the lock and the
# reservation (one line, no start marker; nothing reserved, stopped, fast-forwarded or installed).
# 31 the site cutover refused (an interrupted delivery, a live writer, a changed file, an existing inbox target, a mirrors service still
# running) or failed and was restored, or the trap could not restore it (the timers then stay stopped) · 32 a cluster name or path the
# site config cannot supply (refused before the lock), or a preflight check failed (nothing irreversible ran; the cutover restored).
# (The record-log branch's 15-19 landed on main first; the slurm-only codes that collided moved to 20-23 at the merge, 2026-10-05; the
# plan's step-2 abort code 17 was taken by the reconcile slot, so the conversion aborts with 24.)
# Rehearsed by tests/test_registry_window.py with systemctl / git / fuser / the registry python replaced by PATH shims, the failure
# paths included.
set -u -o pipefail

STATE="${WINDOW_STATE_DIR:-$HOME/.local/state/registry-landing}"
HERE="$(cd "$(dirname "$0")" && pwd)"          # the landing scripts ship together: tree_export.sh sits beside this script
# the cluster names and paths, read before anything else (THE CLUSTER NAMES in the header). With WINDOW_SITE_CUTOVER they come from the
# prepared set's site config; the operator's own values are kept so the installed site config can be read the same way after the cutover
SITE_CUTOVER="${WINDOW_SITE_CUTOVER:-}"
PREFLIGHT_ONLY="${WINDOW_PREFLIGHT_ONLY:-}"
SITE_KEYS="CLUSTER_HOST CLUSTER_SHARED CLUSTER_DEST MIRROR_REPOS_CLUSTER SITE_REPO MIRROR_REPOS_NAME CLUSTER_REPO_MIRROR"
ORIG_SISTEER_CONFIG="${SISTEER_CONFIG:-}"
for k in $SITE_KEYS; do printf -v "ORIG_$k" '%s' "${!k:-}"; done
if [ -n "$SITE_CUTOVER" ]; then
  SITE_CUTOVER="$(cd "$SITE_CUTOVER" 2>/dev/null && pwd || printf '%s' "$SITE_CUTOVER")"
  export SISTEER_CONFIG="$SITE_CUTOVER/config.env"
fi
# shellcheck source=site_env.sh
source "$HERE/site_env.sh"
SITE_ENV_ERRF="$(mktemp "${TMPDIR:-/tmp}/window-site-env.XXXXXX")"
site_env_load $SITE_KEYS 2>"$SITE_ENV_ERRF"; SITE_ENV_RC=$?
SITE_ENV_MSG="$(tr '\n' ' ' <"$SITE_ENV_ERRF")"; rm -f "$SITE_ENV_ERRF"
BASE="${WINDOW_BASE_CHECKOUT:-$SITE_REPO}"
REGISTRY_ROOT="${REGISTRY_ROOT:-$HOME/registry}"
UNIT_DIR="${WINDOW_UNIT_DIR:-$HOME/.config/systemd/user}"
PY="${REGISTRY_PY:-$BASE/.venv/bin/python}"
PYPATH="${REGISTRY_PYTHONPATH:-$BASE/src}"
INSTALLER="${WINDOW_INSTALLER:-$BASE/scripts/registry/install_grading_host.sh}"
LOG_COMPAT="${WINDOW_LOG_COMPAT-/tmp/reg-migrate-chain.log}"
QUIET_TRIES="${WINDOW_QUIET_TRIES:-240}"
QUIET_SLEEP="${WINDOW_QUIET_SLEEP_S:-15}"
CONFIRM_TRIES="${WINDOW_CONFIRM_TRIES:-60}"
CONFIRM_SLEEP="${WINDOW_CONFIRM_SLEEP_S:-1}"
CGROUP_ROOT="${WINDOW_CGROUP_ROOT:-/sys/fs/cgroup}"
BUSY_PATTERN="${WINDOW_BASE_BUSY_PATTERN:-run_registry_legs.sh|l1d_side|run_phase2.sh}"
ADAPTERS="${ADAPTERS-ctl_neg1_multiturn ctl_neg1_controls ctl_neg1_static}"
# the literal `none` is the EMPTY set for ADAPTERS as for CLUSTER_ADAPTERS below: an adapter-less window (a restamp, a migration) names
# no adapter, so nothing is refused and it ends `=== ff chain end`. Folded for CLUSTER_ADAPTERS alone, a window on 2026-10-09 iterated
# an adapter NAMED none, stamped it REFUSED (no cluster policy) and ended ABORTED (refused: none) exit 20 after its work had succeeded.
[ "$ADAPTERS" = none ] && ADAPTERS=""
REG="env PYTHONPATH=$PYPATH $PY -m sisteer.registry.cli"
CLUSTER_ADAPTERS="${CLUSTER_ADAPTERS-$ADAPTERS}"
[ "$CLUSTER_ADAPTERS" = none ] && CLUSTER_ADAPTERS=""
CLUSTER_VENV="${CLUSTER_VENV:-}"
CLUSTER_LOCAL_VENV="${CLUSTER_LOCAL_VENV:-}"   # 1 = every read task copies the venv node-local first (opt-in; measured ~563 s per task)
WATCH_JOB="${WINDOW_WATCH_JOB:-$HOME/scripts/watch-job.sh}"
MIRROR_REPOS="${MIRROR_REPOS_SH:-$HOME/scripts/mirror-repos.sh}"
MIRROR_HISTORY="${MIRROR_HISTORY_SH:-$HERE/mirror_history.sh}"
MIRROR_WAIT_S="${WINDOW_MIRROR_WAIT_S:-3600}"
MAINT_FLAG="$REGISTRY_ROOT/state/maintenance.json"
PLAN_WAIT_S="${WINDOW_PLAN_WAIT_S:-1800}"
# the WORKTREE this script runs from (the landing scripts ship together): its code validates the converted configuration (C9)
NEW_W="$(cd "$HERE/../../.." && pwd)"
if [ -n "${WINDOW_NEW_PY:-}" ]; then NEW_PY="$WINDOW_NEW_PY"; elif [ -x "$NEW_W/.venv/bin/python" ]; then NEW_PY="$NEW_W/.venv/bin/python"; else NEW_PY="$PY"; fi
NEW_PYPATH="${WINDOW_NEW_PYPATH:-$NEW_W/src}"
CONVERT_PY="${WINDOW_CONVERT_PY:-$NEW_PY}"
# the WORKTREE's registry CLI, for every registry call that runs BEFORE `base at` and that the BASE may not know yet: the reads reservation
# and its release. At the cutover window the base still runs the pre-merge main until its own fast-forward, and that main's `cluster-reads`
# has no `reserve` / `release` (an earlier attempt ran the reservation through $REG: "invalid choice: 'reserve'", exit 22). $REG (the
# base's code) stays where the OLD code is the point -- `config --json`, the old loader's validation -- and for every stage after `base at`,
# where the base IS the landed tree.
NEW_REG="env PYTHONPATH=$NEW_PYPATH $NEW_PY -m sisteer.registry.cli"
ACCEPTANCE_OUT="${ACCEPTANCE_OUT:-}"
ACCEPTANCE_LIST="${ACCEPTANCE_LIST:-}"
ACCEPTANCE_SCRIPT="${ACCEPTANCE_SCRIPT:-}"
POST_INGEST_ADAPTERS="${POST_INGEST_ADAPTERS:-}"      # read only to be REFUSED by name below (exit 26)
AUTHORITY_WAIT_S="${WINDOW_AUTHORITY_WAIT_S:-1800}"    # the per-component bound on `stores verify-authority` in the store authority stage
REQUIRE_CERTIFIED="${WINDOW_REQUIRE_CERTIFIED:-}"      # 1 = a component not certified after that stage aborts the window (exit 27; the cutover window)
ENV_COMMENTS_SOURCES="${WINDOW_ENV_COMMENTS_SOURCES:-}"  # env-comments' overrides files (files or directories of .env files), --source on its every invocation
ENV_COMMENTS_LEAVE_UNSOURCED="${WINDOW_ENV_COMMENTS_LEAVE_UNSOURCED:-}"  # 1 = the stamps no source covers are left as stored by name (--leave-unsourced; no sources check)
REPLICA_ROOT_REMOTE="${WINDOW_REPLICA_ROOT:-}"          # the cluster replica root whose site_aliases.toml the preflight compares (default under CLUSTER_SHARED)
CUTOVER_INSTALLED=0       # 1 once site_cutover.py installed the prepared set: the trap undoes it until the migration is applied
CUTOVER_RESTORE_BROKEN=0  # 1 when the cutover could not be put back: the trap then leaves the timers stopped
CUTOVER_BACKUP=""
UNITS_BACKED_UP=0
BASE_MOVED=0
PRE_FF_SHA=""
PREFLIGHT_SHA=""
MIGRATE_APPLIED=0
FORWARD_ONLY=""        # set when the cutover window's apply failed after the schema version moved: no rollback, timers stay stopped
# the reads reservation's code and registry label: the worktree's and the site's, except in the cutover window, whose reservation is taken
# before anything changes, under the base's code with the cluster's current registry label (the prepared alias table read backwards)
RESERVE_REG="$NEW_REG"
RESERVE_DEST="$CLUSTER_DEST"
if [ -n "$SITE_CUTOVER" ] && [ "$SITE_ENV_RC" = 0 ]; then
  RESERVE_REG="$REG"
  RESERVE_DEST=$("$CONVERT_PY" - "$SITE_CUTOVER/site_aliases.toml" "$CLUSTER_DEST" <<'PY' 2>/dev/null
import sys, tomllib
try:
    table = tomllib.load(open(sys.argv[1], "rb")).get("site_aliases") or {}
except FileNotFoundError:
    table = {}
olds = sorted(k for k, v in table.items() if v == sys.argv[2])
print(olds[0] if len(olds) == 1 else ("" if olds else sys.argv[2]))
PY
)
fi

# ---- the preflight (see the header): the mirror selector, the commit on the cluster's clone, every remote_src, the alias digests. Returns
# 1 with PF_WHY set on the first check that fails; prints one line per check that passes
PF_CLUSTERS_PY='import sys
from sisteer.registry import config as C
for d in C.mirror_destinations(C.load(sys.argv[1])):
    print("CLUSTER", d["name"], d["host"], d["remote_src"])'
# every [[clusters]] entry names its qos (stdlib only, under the window's own interpreter): without one a cluster's jobs take the code's
# default QoS, which is not the site's
PF_QOS_PY='import sys, tomllib
for c in (tomllib.load(open(sys.argv[1], "rb")).get("clusters") or []):
    if not c.get("qos"):
        print(c.get("name") or "?")'
PF_DIGEST_PY='import sys
from sisteer.registry import site_aliases as A
for p in sys.argv[1:]:
    t = A.load_table(p)
    A.validate(t)
    print("DIGEST", A.digest(t), p)'
pf_ssh() { ssh -o BatchMode=yes -o ConnectTimeout=25 "$@"; }
preflight() {
  # $1 = the registry root whose config.toml and site_aliases.toml are read, $2 = the commit, $3 = the repository it resolves in
  local root="$1" sha="$2" repo="$3" out rc n=0 tag name host rsrc local_aliases remote_aliases tmp
  local replica="${REPLICA_ROOT_REMOTE:-$CLUSTER_SHARED/registry/replica}"
  PF_WHY=""
  echo "preflight: login host $CLUSTER_HOST, registry label $CLUSTER_DEST, mirror selector $MIRROR_REPOS_CLUSTER, clone $CLUSTER_REPO_MIRROR (checkout $MIRROR_REPOS_NAME) $(date -u +%T)"
  out=$("$MIRROR_REPOS" --cluster "$MIRROR_REPOS_CLUSTER" "$MIRROR_REPOS_NAME" 2>&1); rc=$?
  if [ "$rc" != 0 ]; then
    PF_WHY="the repo-mirror tool refused '--cluster $MIRROR_REPOS_CLUSTER $MIRROR_REPOS_NAME' (rc $rc: $(tail -n 1 <<<"$out")); set CLUSTER_MIRROR_SELECTOR in that cluster's block to the name that tool knows the cluster by"
    return 1
  fi
  echo "preflight: $(basename "$MIRROR_REPOS") --cluster $MIRROR_REPOS_CLUSTER $MIRROR_REPOS_NAME: rc 0"
  [ -n "$sha" ] || { PF_WHY="no commit to look for on the clone"; return 1; }
  if ! pf_ssh "$CLUSTER_HOST" git -C "$CLUSTER_REPO_MIRROR" cat-file -e "$sha^{commit}" 2>/dev/null; then
    git -C "$repo" push -q "ssh://$CLUSTER_HOST$CLUSTER_REPO_MIRROR" "$sha:refs/heads/gate/${sha:0:7}" 2>&1 | tail -n 2
    if ! pf_ssh "$CLUSTER_HOST" git -C "$CLUSTER_REPO_MIRROR" cat-file -e "$sha^{commit}" 2>/dev/null; then
      PF_WHY="commit $sha is not on $CLUSTER_HOST:$CLUSTER_REPO_MIRROR and a direct push did not put it there (CLUSTER_REPO_MIRROR names the clone)"
      return 1
    fi
    echo "preflight: pushed ${sha:0:12} to the clone as refs/heads/gate/${sha:0:7}"
  fi
  echo "preflight: commit ${sha:0:12} is on $CLUSTER_HOST:$CLUSTER_REPO_MIRROR"
  out=$(env PYTHONPATH="$NEW_PYPATH" "$NEW_PY" -c "$PF_CLUSTERS_PY" "$root" 2>&1); rc=$?
  [ "$rc" = 0 ] || { PF_WHY="the registry config under $root could not be read under the worktree's code (rc $rc: $(tail -n 1 <<<"$out"))"; return 1; }
  noqos=$("$CONVERT_PY" -c "$PF_QOS_PY" "$root/config.toml" 2>&1); rc=$?
  [ "$rc" = 0 ] || { PF_WHY="the [[clusters]] entries of $root/config.toml could not be read for their qos (rc $rc: $(tail -n 1 <<<"$noqos"))"; return 1; }
  if [ -n "$noqos" ]; then
    PF_WHY="the [[clusters]] entr(ies) $(printf '%s\n' "$noqos" | paste -sd ' ' -) in $root/config.toml name no qos (that cluster's jobs would take the code's default QoS); set qos on every entry"
    return 1
  fi
  echo "preflight: every [[clusters]] entry of $root/config.toml names its qos"
  while read -r tag name host rsrc; do
    [ "$tag" = CLUSTER ] || continue
    n=$((n + 1))
    if ! pf_ssh "$host" test -f "$rsrc/sisteer/registry/__init__.py" 2>/dev/null; then
      PF_WHY="the [[clusters]] entry $name: $host:$rsrc/sisteer/registry/__init__.py does not exist (remote_src names the clone's src)"
      return 1
    fi
    echo "preflight: [[clusters]] $name: $host:$rsrc holds sisteer.registry"
  done <<<"$out"
  [ "$n" -gt 0 ] || { PF_WHY="the registry config under $root names no [[clusters]] entry"; return 1; }
  local_aliases="$root/site_aliases.toml"
  tmp="$(mktemp "${TMPDIR:-/tmp}/window-replica-aliases.XXXXXX")"
  if pf_ssh "$CLUSTER_HOST" test -f "$replica/site_aliases.toml" 2>/dev/null; then
    pf_ssh "$CLUSTER_HOST" cat "$replica/site_aliases.toml" >"$tmp" 2>/dev/null || { rm -f "$tmp"; PF_WHY="$CLUSTER_HOST:$replica/site_aliases.toml could not be read"; return 1; }
    remote_aliases="$tmp"
  else
    remote_aliases=""
  fi
  if [ ! -f "$local_aliases" ] && [ -z "$remote_aliases" ]; then
    rm -f "$tmp"; echo "preflight: no site_aliases.toml under $root nor $CLUSTER_HOST:$replica: both read the empty table"
    return 0
  fi
  if [ ! -f "$local_aliases" ] || [ -z "$remote_aliases" ]; then
    rm -f "$tmp"; PF_WHY="site_aliases.toml is under only one of $root and $CLUSTER_HOST:$replica (stage the same file in both)"
    return 1
  fi
  out=$(env PYTHONPATH="$NEW_PYPATH" "$NEW_PY" -c "$PF_DIGEST_PY" "$local_aliases" "$remote_aliases" 2>&1); rc=$?
  rm -f "$tmp"
  local d_local d_remote
  d_local=$(awk -v p="$local_aliases" '$1 == "DIGEST" && $3 == p {print $2}' <<<"$out")
  d_remote=$(awk -v p="$remote_aliases" '$1 == "DIGEST" && $3 == p {print $2}' <<<"$out")
  if [ "$rc" != 0 ] || [ -z "$d_local" ] || [ -z "$d_remote" ]; then
    PF_WHY="the site-alias digests could not be read under the worktree's code (rc $rc: $(tail -n 1 <<<"$out"))"
    return 1
  fi
  if [ "$d_local" != "$d_remote" ]; then
    PF_WHY="site-alias digests differ: $local_aliases $d_local, $CLUSTER_HOST:$replica/site_aliases.toml $d_remote"
    return 1
  fi
  echo "preflight: site-alias digest $d_local under $root and on $CLUSTER_HOST:$replica"
  return 0
}

# the timer set the window stops and quiesces (the DB writers/readers + the two status units + the store mirror's timer: a tick
# inside the window re-publishes a component's cluster `current` in the running code's format, main's until the cutover); registry-mirrors
# and the weekly registry-mirrors-full verify read the record log only and keep running; so does the store mirror's weekly full verify
# (registry-store-mirror-full: it reads the stores and $REGISTRY_ROOT/state only, never the database)
TIMERS="registry-harvest.timer registry-cluster-reads.timer registry-adapters.timer registry-modal-pull.timer registry-status.timer registry-status-heavy.timer registry-store-mirror.timer"
# the RETIRED reader services (C6): stopped with the timers, confirmed terminal with an empty cgroup before the base moves, never started again
RETIRED_SERVICES="registry-adapters.service registry-modal-pull.service"
# the retired UNIT FILES the window uninstalls after the install stage (C7; the Modal pull's units joined in C10): disabled, removed from
# the unit directory, daemon reloaded. Only units whose source file the repo no longer carries (else the installer would copy them back).
RETIRED_UNITS="registry-adapters.timer registry-adapters.service registry-modal-pull.timer registry-modal-pull.service"
# the Modal-era DROP-INS removed with them (C10): paths relative to the unit directory; a directory is removed whole, a file alone
RETIRED_DROPINS="registry-modal-pull.service.d registry-mirrors.service.d/clusters-only.conf"
# the mirrors timers join the stop set in the cutover window: their services keep state keyed by destination name, which the cutover renames
MIRRORS_TIMERS="registry-mirrors.timer registry-mirrors-full.timer"
[ -z "$SITE_CUTOVER" ] || TIMERS="$TIMERS $MIRRORS_TIMERS"
UNITS=${TIMERS//.timer/.service}
# the repo's default set (install_grading_host.sh TIMERS) minus NEVER_START: a member not installed before this window is started by the trap
DEFAULT_TIMERS="registry-harvest.timer registry-cluster-reads.timer registry-mirrors.timer registry-mirrors-full.timer registry-status.timer registry-status-heavy.timer registry-gate-baseline.timer registry-store-mirror.timer registry-store-mirror-full.timer"
NEVER_START="registry-snapshot.timer registry-adapters.timer registry-modal-pull.timer"
# the operation id of this window: the migration plan file and its pins, the reader-state sidecar, the reads reservation, the Slurm job
# names (reads-<id>, attest-<id>) and the receipt. The default carries the pid beside the second-resolution stamp so two windows never
# share a name (the record-log branch); an operator-supplied WINDOW_ID is honored (C7: a second window under the SAME id is refused by
# the reservation, exit 22, while the first is live; a plan file already at that id -- an earlier window ran under it -- is refused under
# the lock BEFORE anything moves, exit 8, and again at the plan stage as a race guard; a retry needs a new id)
WINDOW_ID="${WINDOW_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"

# ---- the preflight alone (WINDOW_PREFLIGHT_ONLY=1; see the header): the prepared set read in place of the installed files, the checks run,
# and nothing else -- no lock, no reservation, no flag, no stop, nothing installed. Printed to the terminal and to preflight-<window>.txt
if [ "$PREFLIGHT_ONLY" = 1 ]; then
  mkdir -p "$STATE/windows"
  {
    if [ -z "$SITE_CUTOVER" ]; then echo "preflight FAILED: WINDOW_PREFLIGHT_ONLY=1 reads a prepared set; name it with WINDOW_SITE_CUTOVER=<dir>"; exit 32; fi
    echo "preflight only: prepared set $SITE_CUTOVER (window $WINDOW_ID); no lock, no reservation, no drain, nothing installed $(date -u +%T)"
    if [ "$SITE_ENV_RC" != 0 ]; then echo "preflight FAILED: ${SITE_ENV_MSG% }"; exit 32; fi
    pf_sha="${WINDOW_PREFLIGHT_SHA:-$(git -C "$NEW_W" rev-parse HEAD 2>/dev/null)}"
    if preflight "$SITE_CUTOVER" "$pf_sha" "$NEW_W"; then echo "preflight PASSED $(date -u +%T)"; exit 0; fi
    echo "preflight FAILED: $PF_WHY $(date -u +%T)"; exit 32
  } 2>&1 | tee "$STATE/windows/preflight-$WINDOW_ID.txt"
  exit "${PIPESTATUS[0]}"
fi

mkdir -p "$STATE/windows"
LOG="$STATE/windows/chain.log"
exec >>"$LOG" 2>&1
LOG_OFFSET=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
stamp() { date -u +%T; }
in_set() { case " $2 " in *" $1 "*) return 0;; esac; return 1; }
ABORT_CAUSE=""
fail() { echo "$1; abort"; ABORT_CAUSE="$1"; exit "$2"; }   # every abort: the message, then the EXIT trap restores the timers and writes ABORTED
# the adapters that read on the cluster THIS window: the named adapters ($ADAPTERS) that have a cluster policy ($CLUSTER_ADAPTERS), in the
# policy's order. A named adapter outside the set is RECORDED REFUSED (the grading host reads no store; C6).
CLUSTER_SET=""
for a in $CLUSTER_ADAPTERS; do in_set "$a" "$ADAPTERS" && CLUSTER_SET="${CLUSTER_SET:+$CLUSTER_SET }$a"; done

# ---- the site's cluster names, refused by name before the lock and the reservation when the site config cannot supply one (exit 32)
if [ "$SITE_ENV_RC" != 0 ]; then
  echo "window $WINDOW_ID refused: ${SITE_ENV_MSG% }; nothing reserved, stopped, fast-forwarded or installed $(stamp)"
  exit 32
fi
if [ -n "$SITE_CUTOVER" ] && [ -z "$RESERVE_DEST" ]; then
  echo "window $WINDOW_ID refused: the prepared set's site_aliases.toml names more than one legacy label for $CLUSTER_DEST, so the cluster's current registry label (the reads reservation's) is ambiguous; nothing reserved, stopped, fast-forwarded or installed $(stamp)"
  exit 32
fi
if [ -n "$SITE_CUTOVER" ] && [ ! -f "$SITE_CUTOVER/MANIFEST.tsv" ]; then
  echo "window $WINDOW_ID refused: WINDOW_SITE_CUTOVER=$SITE_CUTOVER holds no MANIFEST.tsv (a prepared set); nothing reserved, stopped, fast-forwarded or installed $(stamp)"
  exit 32
fi
if [ -n "$ACCEPTANCE_OUT" ] && [ -z "$ACCEPTANCE_SCRIPT" ]; then
  echo "window $WINDOW_ID refused: ACCEPTANCE_OUT is set and ACCEPTANCE_SCRIPT names no acceptance script; nothing reserved, stopped, fast-forwarded or installed $(stamp)"
  exit 25
fi

# ---- a grading host post-ingest adapter pass is refused BY NAME, before the lock and the reservation (C6, S2): one line, exit 26,
# no start marker (no window to the measurement); nothing reserved, stopped, fast-forwarded or installed
if [ -n "$POST_INGEST_ADAPTERS" ]; then
  echo "window $WINDOW_ID refused: POST_INGEST_ADAPTERS=$POST_INGEST_ADAPTERS names a grading host post-ingest adapter pass, and the grading host reads no store (C6, S2); a second read is the next window's first read -- every window exports the frame fresh, so this window's ingest is already in it; nothing reserved, stopped, fast-forwarded or installed $(stamp)"
  exit 26
fi

# ---- the per-action apply list is checked BY NAME before the lock and the reservation (the header): every name in WINDOW_RECONCILE_APPLY
# must be an action WINDOW_RECONCILE names, and the retired global `1` is refused with the list it should be -- a window that applied every
# named action aborted at run-patch-extra's release gate after its migration had applied. One line, exit 30, no start marker, nothing moved
reconcile_applies() { in_set "$1" "${WINDOW_RECONCILE_APPLY:-}"; }
for rca in ${WINDOW_RECONCILE_APPLY:-}; do
  [ "$rca" = "0" ] && continue                                  # the old off switch still means "apply nothing": every named action a dry run
  if [ "$rca" = "1" ]; then
    echo "window $WINDOW_ID refused: WINDOW_RECONCILE_APPLY=$rca is the retired global switch (it applied EVERY action WINDOW_RECONCILE names); name the actions to apply instead, e.g. WINDOW_RECONCILE_APPLY=\"${WINDOW_RECONCILE:-multi-turn-scope}\" -- every other named action runs as a dry run; nothing reserved, stopped, fast-forwarded or installed $(stamp)"
    exit 30
  fi
  if ! in_set "$rca" "${WINDOW_RECONCILE:-}"; then
    echo "window $WINDOW_ID refused: WINDOW_RECONCILE_APPLY names '$rca', which WINDOW_RECONCILE=\"${WINDOW_RECONCILE:-}\" does not name (an apply with no dry run before it); name it in both, or drop it; nothing reserved, stopped, fast-forwarded or installed $(stamp)"
    exit 30
  fi
done
# env-comments' sources are checked the same way: a named overrides file or directory that is not there would refuse every invocation of
# the action after the migration had applied
for src in $ENV_COMMENTS_SOURCES; do
  if [ ! -e "$src" ]; then
    echo "window $WINDOW_ID refused: WINDOW_ENV_COMMENTS_SOURCES names '$src', which is not a file or a directory here (env-comments restates a stamp from its overrides file, matched by sha256); fix the path, or drop it; nothing reserved, stopped, fast-forwarded or installed $(stamp)"
    exit 30
  fi
done
# the operator's word on the stamps no source covers: 1 leaves them as stored (`--leave-unsourced` on env-comments' every invocation, and
# no sources check below); 0 or empty asks for every one to be covered; anything else is a misconfiguration, refused by name
case "$ENV_COMMENTS_LEAVE_UNSOURCED" in
  ""|0|1) : ;;
  *) echo "window $WINDOW_ID refused: WINDOW_ENV_COMMENTS_LEAVE_UNSOURCED=$ENV_COMMENTS_LEAVE_UNSOURCED is neither 1 (leave the run-overrides stamps no source covers as stored, by name) nor 0 or empty (every stamp whose values may carry a comment must have its file among WINDOW_ENV_COMMENTS_SOURCES); nothing reserved, stopped, fast-forwarded or installed $(stamp)"
     exit 30 ;;
esac
# the extra arguments of one action's every invocation (dry run, apply, verify), set in RARGS: env-comments takes its overrides files,
# and so does run-patch-extra (a legacy run_overrides stamp and the live one compare as readings of one of them; without the files a stamp
# env-comments restated would read as a conflict against its legacy record); env-comments alone takes the operator's word to leave the
# stamps no source covers
reconcile_args() {
  RARGS=()
  case "$1" in env-comments|run-patch-extra) for src in $ENV_COMMENTS_SOURCES; do RARGS+=(--source "$src"); done ;; esac
  case "$1" in env-comments) [ "$ENV_COMMENTS_LEAVE_UNSOURCED" = 1 ] && RARGS+=(--leave-unsourced) ;; esac
  return 0
}
RARGS=()
# ---- env-comments' SOURCES COVERAGE is checked before the lock and the reservation when the action is listed for apply and the operator
# has not left the uncovered stamps by name (the header): under the WORKTREE's CLI (the base may not know the action before its
# fast-forward), read-only over the folded run table, `registry reconcile env-comments --check-sources` exits 0 when every stamped run row
# whose values may carry a comment has its file among the sources, 3 when it lists the uncovered stamps. A source list that covers nothing
# -- the knob unset, a directory without the files the stamps were read from -- would otherwise reach the apply after the migration had
# applied, correct nothing, and verify a clean class over a table it had not touched. One line in the log; the check's whole text in
# $STATE/windows/env-comments-sources-<window>.txt; exit 30 and nothing moved when it lists rows or could not run.
if reconcile_applies env-comments && [ "$ENV_COMMENTS_LEAVE_UNSOURCED" != 1 ]; then
  SOURCES_CHECK="$STATE/windows/env-comments-sources-$WINDOW_ID.txt"
  reconcile_args env-comments
  $NEW_REG --root "$REGISTRY_ROOT" reconcile env-comments "${RARGS[@]}" --check-sources >"$SOURCES_CHECK" 2>&1; crc=$?
  if [ "$crc" = 0 ]; then
    echo "env-comments sources: $(tail -n 1 "$SOURCES_CHECK") (the check's text in $SOURCES_CHECK) $(stamp)"
  elif [ "$crc" = 3 ]; then
    echo "window $WINDOW_ID refused: WINDOW_ENV_COMMENTS_SOURCES=\"$ENV_COMMENTS_SOURCES\" covers no file for run-overrides stamps whose values may carry a comment ($(tail -n 1 "$SOURCES_CHECK")); env-comments is listed for apply and would restate nothing for them, then verify a class it had not touched: name their files, or WINDOW_ENV_COMMENTS_LEAVE_UNSOURCED=1 to leave them as stored by name; the check's text is in $SOURCES_CHECK; nothing reserved, stopped, fast-forwarded or installed $(stamp)"
    exit 30
  else
    echo "window $WINDOW_ID refused: env-comments' sources check could not run (registry reconcile env-comments --check-sources rc=$crc under the worktree's CLI; its output is in $SOURCES_CHECK): env-comments is listed for apply and its sources are unverified; nothing reserved, stopped, fast-forwarded or installed $(stamp)"
    exit 30
  fi
fi

# ---- one window at a time (see the header): the exclusive lock comes before the start marker, so a refused attempt is one line in the
# log and no window to the measurement; fd 9 stays open until the process exits, after the trap's restore (the id, defined above, carries
# the pid beside the second-resolution stamp so two windows never share a plan file name).
WINDOW_LOCK="$STATE/windows/.window.lock"
exec 9>>"$WINDOW_LOCK"
if ! flock -n 9; then
  holder=$(head -n 1 "$WINDOW_LOCK" 2>/dev/null)
  echo "window $WINDOW_ID refused: another window holds $WINDOW_LOCK (${holder:-no holder line}); one window at a time $(stamp)"
  exit 18
fi
printf 'window %s pid %s started %s\n' "$WINDOW_ID" "$$" "$(date -u +%FT%TZ)" >"$WINDOW_LOCK"

# ---- a reused WINDOW_ID is refused HERE, under the lock and before anything moves. The plan file is written
# once per id (the record-log branch's rule), so a file already at this id's name means an earlier window ran under it: an operator retry
# with an exported WINDOW_ID. Refusing only at the plan stage would be too late -- by then the reservation is taken, the timers are stopped,
# the base has fast-forwarded and the units are INSTALLED from it, and the trap would restore the harvester timer under new code with the
# migration unapplied (db.open_or_create refuses the old database under new code: ingestion stops until a hand repair). One line, exit 8,
# no start marker (no window to the measurement); the plan-stage check below stays as the race guard for a file that appears in between.
PLAN_FILE="$STATE/windows/migrate-plan-$WINDOW_ID.txt"
PINS_FILE="$STATE/windows/migrate-plan-$WINDOW_ID.pins"
if [ -e "$PLAN_FILE" ]; then
  echo "window $WINDOW_ID refused: plan file $PLAN_FILE already exists (a window id collision: an earlier window ran under this id); nothing reserved, stopped, fast-forwarded or installed -- a retry needs a new id $(stamp)"
  exit 8
fi

echo "=== ff chain start $(date -u +%FT%TZ) ADAPTERS=$ADAPTERS"
ACTIVE_TIMERS=$(for t in $TIMERS; do systemctl --user is-active --quiet "$t" && printf "%s " "$t"; done)
echo "timers active at stop: ${ACTIVE_TIMERS:-none}"
PRE_INSTALLED=$(for t in $DEFAULT_TIMERS; do [ -e "$UNIT_DIR/$t" ] && printf "%s " "$t"; done)
echo "default timers installed before this window: ${PRE_INSTALLED:-none}"
INSTALLED=0
INGESTED=0
REFUSED_ALL=""
RESERVED=0
STOPPED=0          # 1 once `systemctl stop` of the timers ran: the trap restores only what this window stopped

# the site cutover undone (see the header): the unit files as they were, the base checkout back where it was fast-forwarded from, the
# cutover's backups byte for byte and the inboxes under their old names -- all before any timer restarts. Returns 1 when any part failed
rollback_cutover() {
  local bad=0 out r u
  echo "site cutover: undoing it before any timer restarts (the migration was not applied) $(stamp)"
  if [ "$UNITS_BACKED_UP" = 1 ]; then
    # every registry entry of the unit directory goes, then the snapshot taken before the install comes back whole: the unit files, the
    # drop-ins and the enablement links the uninstall stage removed included
    (cd "$UNIT_DIR" && rm -rf -- registry-* && rm -f -- *.wants/registry-* *.requires/registry-*) || bad=1
    if [ -s "$CUTOVER_BACKUP/units.list" ]; then tar -xpf "$CUTOVER_BACKUP/units.tar" -C "$UNIT_DIR" || bad=1; fi
    systemctl --user daemon-reload 2>/dev/null || bad=1
    # the install is undone: no timer counts as newly installed, so the trap starts only the ones that were active at the stop
    INSTALLED=0
    echo "site cutover: unit files back as they were before the install, with their drop-ins and enablement links ($(wc -l <"$CUTOVER_BACKUP/units.list") entr(ies); daemon reloaded) $(stamp)"
  fi
  if [ "$BASE_MOVED" = 1 ] && [ -n "$PRE_FF_SHA" ]; then
    if git -C "$BASE" reset -q --keep "$PRE_FF_SHA"; then echo "site cutover: base checkout back at ${PRE_FF_SHA:0:12} $(stamp)"
    else echo "site cutover: the base checkout could not be reset to ${PRE_FF_SHA:0:12}"; bad=1; fi
  fi
  out=$("$CONVERT_PY" "$HERE/site_cutover.py" restore --backup "$CUTOVER_BACKUP" 2>&1); r=$?
  printf '%s\n' "$out" | sed 's/^/  /'
  if [ "$r" = 0 ]; then echo "site cutover restored byte for byte, inboxes under their old names (backups in $CUTOVER_BACKUP) $(stamp)"
  else echo "site cutover: the restore did not finish (rc=$r; $CUTOVER_BACKUP/cutover.json says what is where)"; bad=1; fi
  return "$bad"
}

restore() {
  local rc=$? restore_start="" restore_new="" restore_rc=0 not_active="" t cause
  if [ "$rc" != 0 ]; then echo "window ended early (rc=$rc) $(stamp)"; fi
  # the drain flag goes first: the harvester's next cycle may fold again as soon as the timers are back
  if [ -e "$MAINT_FLAG" ]; then rm -f "$MAINT_FLAG" && echo "maintenance flag removed $(stamp)" || echo "maintenance flag could not be removed: $MAINT_FLAG $(stamp)"; fi
  # the reads reservation goes before the timers come back (C7): the first tick after the window must find nothing held; released only when
  # it still names this window (a takeover by nobody is possible here, but the release is by name regardless)
  if [ "$RESERVED" = 1 ]; then
    $NEW_REG cluster-reads release --op "$WINDOW_ID" --state "$STATE" 2>&1 | tail -1
    echo "reads reservation released $(stamp)"
  fi
  # an abort with the site cutover installed and the migration not applied puts the old configuration back first
  if [ "$CUTOVER_INSTALLED" = 1 ] && [ "$MIGRATE_APPLIED" != 1 ] && [ "$rc" != 0 ]; then
    if rollback_cutover; then CUTOVER_RESTORE_BROKEN=0; else CUTOVER_RESTORE_BROKEN=1; fi
  fi
  if [ "$CUTOVER_RESTORE_BROKEN" = 1 ]; then
    # the old timers would start on a configuration that is neither the old one nor the new one: they stay stopped, by name
    echo "timers LEFT STOPPED: the site cutover could not be restored, and the timers would start on a mixed configuration; active at stop were: ${ACTIVE_TIMERS:-none} $(stamp)"
    restore_rc=31
  elif [ -n "$FORWARD_ONLY" ]; then
    # the cutover window's migration committed and its apply still failed: no code may run on this database until it is moved forward
    echo "timers LEFT STOPPED: $FORWARD_ONLY; the old code refuses this database and the new code's check did not pass, so nothing restarts until it is moved forward; active at stop were: ${ACTIVE_TIMERS:-none} $(stamp)"
    restore_rc=8
  elif [ "$STOPPED" != 1 ]; then
    # an abort before the stop (the reads reservation refused, C7): nothing was stopped or installed, so nothing is restored or started
    echo "timers never stopped (aborted before the stop): nothing to restore $(stamp)"
  elif [ "${KEEP_TIMERS_STOPPED:-0}" = "1" ]; then
    echo "timers LEFT STOPPED by request (KEEP_TIMERS_STOPPED=1); active at stop were: ${ACTIVE_TIMERS:-none} $(stamp)"
  else
    for t in ${ACTIVE_TIMERS:-}; do in_set "$t" "$NEVER_START" || restore_start="$restore_start $t"; done
    # a timer newly installed this window has a unit file to start only if the install stage ran
    if [ "$INSTALLED" = 1 ]; then
      for t in $DEFAULT_TIMERS; do
        in_set "$t" "${PRE_INSTALLED:-}" || in_set "$t" "$NEVER_START" || in_set "$t" "$restore_start" || restore_new="$restore_new $t"
      done
    fi
    local restore_all="${restore_start}${restore_new}"
    echo "restore set (active at stop ∪ newly installed − $NEVER_START):${restore_all:- none}"
    echo "  active at stop:${restore_start:- none} · newly installed:${restore_new:- none} · never started: $NEVER_START"
    if [ -n "$restore_start" ]; then systemctl --user start $restore_start 2>/dev/null || echo "start failed for:$restore_start"; fi
    if [ -n "$restore_new" ]; then systemctl --user enable --now $restore_new 2>/dev/null || echo "enable --now failed for:$restore_new"; fi
    # verified, not assumed: every timer of the restore set must read `active` now
    not_active=$(for t in $restore_all; do [ "$(systemctl --user is-active "$t" 2>/dev/null)" = active ] || printf "%s " "$t"; done)
    if [ -z "$not_active" ]; then
      echo "timers restored (trap; the set above) $(stamp)"
    else
      echo "timers NOT restored: not active after the start: $not_active(trap) $(stamp)"; restore_rc=14
    fi
    if [ "$INSTALLED" = 1 ]; then
      # the status page's cache initializer: one heavy refresh after the timers are back (registry-status-heavy.timer keeps it 6-hourly)
      systemctl --user start --no-block registry-status-heavy.service 2>/dev/null && echo "-- status heavy refresh started once (cache initializer) $(stamp)" \
        || echo "status heavy refresh could not be started $(stamp)"
    fi
  fi
  # THE END MARKER: `=== ff chain end` only when every mandatory stage ran (rc 0: the ingest included) AND the restore
  # set is verified active; anything else is `=== ff chain ABORTED (<cause>)`, which windows.py never counts as completed
  if [ "$rc" = 0 ] && [ "$restore_rc" = 0 ]; then
    echo "=== ff chain end $(date -u +%FT%TZ)"
  else
    cause="${ABORT_CAUSE:-rc=$rc}"
    if [ "$restore_rc" != 0 ]; then
      if [ "$rc" = 0 ]; then cause="restore"; else cause="$cause; restore"; fi
    fi
    echo "=== ff chain ABORTED ($cause) $(date -u +%FT%TZ)"
  fi
  if [ -n "$LOG_COMPAT" ]; then
    # the pre-repo log keeps receiving every window (the measurement parser reads both), copied as the last act
    { mkdir -p "$(dirname "$LOG_COMPAT")" && tail -c +"$((LOG_OFFSET + 1))" "$LOG" >>"$LOG_COMPAT"; } 2>/dev/null || true
  fi
  # a failed restore fails a window that had otherwise succeeded; an earlier stage's own code stands, except a cutover left unrestored (31)
  if [ "$CUTOVER_RESTORE_BROKEN" = 1 ]; then exit 31; fi
  if [ "$rc" = 0 ] && [ "$restore_rc" != 0 ]; then exit "$restore_rc"; fi
}
trap restore EXIT

# the reads reservation FIRST: a tick holding it has its array cancelled by id and confirmed terminal
# in sacct, then the reservation changes hands; every reads-*/attest-* job of this user is reconciled with the op records (an outstanding job
# with no live owner cancelled and confirmed; a completed array nobody pulled is pulled now). A live window holding it refuses this one.
echo "-- reads reservation $(stamp) (op $WINDOW_ID, holder pid $$)"
# --pid $$: the holder is this shell, which lives for the whole window -- never the CLI process in the $(...), which is gone when it returns.
# Under the WORKTREE's CLI ($NEW_REG): the base has not moved yet and may not know `reserve` (the cutover)
RES_OUT=$($RESERVE_REG cluster-reads reserve --owner window --op "$WINDOW_ID" --takeover --state "$STATE" --dest "$RESERVE_DEST" --pid $$ 2>&1); rrc=$?
[ "$RESERVE_DEST" = "$CLUSTER_DEST" ] || echo "reads reservation under the base's code with the cluster's current registry label $RESERVE_DEST (the site cutover renames it $CLUSTER_DEST)"
echo "$RES_OUT" | tail -4
[ "$rrc" = 0 ] || fail "reads reservation not taken (rc=$rrc): $(echo "$RES_OUT" | tail -1)" 22
RESERVED=1

# the honest drain (C6): the flag BEFORE the stop, so the cycle the harvest timer fires meanwhile exits at the door (`ingest --honor-maintenance`)
mkdir -p "$(dirname "$MAINT_FLAG")"
printf '{"since_utc": "%s", "reason": "landing window (window.sh pid %s)", "pid": %s, "adapters": "%s"}\n' "$(date -u +%FT%TZ)" "$$" "$$" "$ADAPTERS" >"$MAINT_FLAG.tmp.$$" \
  && mv -f "$MAINT_FLAG.tmp.$$" "$MAINT_FLAG" && echo "maintenance flag written: $MAINT_FLAG $(stamp)" || fail "maintenance flag could not be written at $MAINT_FLAG" 3
# the timers stop FIRST -- before the store mirror, not after it: a timer that fires during a 60-minute mirror step starts a
# pass the drain below then has to wait out in full (seen once: the adapters timer fired 2 s before the stop, behind a mirror step that
# ran 21:02 -> 22:02, and the drain waited on that pass); stopped here, a pass already in flight overlaps the mirror's wall instead. The
# retired services stop with the timers. The store-mirror TIMER is in $TIMERS (a tick re-publishes a component in the
# running code's format); stopping a timer never blocks a manual `start` of its service, so the mirror step below still starts it.
systemctl --user stop $TIMERS $RETIRED_SERVICES || echo "stop returned rc=$? for: $TIMERS $RETIRED_SERVICES"
STOPPED=1
# verified, not assumed: a timer still active here would fire into the migrate or an adapter pass
STILL_ACTIVE=$(for t in $TIMERS; do [ "$(systemctl --user is-active "$t" 2>/dev/null)" = inactive ] || printf "%s " "$t"; done)
[ -z "$STILL_ACTIVE" ] || fail "timers not stopped (read other than inactive): $STILL_ACTIVE" 13

# the retired readers, stopped and CONFIRMED (C6): terminal state, no main PID, an EMPTY cgroup by PID list -- never a text match
cgroup_pids() {
  # the PIDs in a unit's cgroup (empty when the unit has no cgroup any more)
  local cg; cg=$(systemctl --user show -p ControlGroup --value "$1" 2>/dev/null || true)
  [ -n "$cg" ] && [ -r "$CGROUP_ROOT$cg/cgroup.procs" ] && tr '\n' ' ' <"$CGROUP_ROOT$cg/cgroup.procs" | sed 's/ *$//'
}
confirm_retired() {
  # 0 when every retired service is terminal with no process; else prints the offenders and returns 1
  local u st pid pids bad=""
  for u in $RETIRED_SERVICES; do
    st=$(systemctl --user is-active "$u" 2>/dev/null || true)
    pid=$(systemctl --user show -p MainPID --value "$u" 2>/dev/null || true)
    pids=$(cgroup_pids "$u")
    case "$st" in inactive|failed|"") ;; *) bad="$bad $u(state=$st)";; esac
    [ -z "$pid" ] || [ "$pid" = 0 ] || bad="$bad $u(MainPID=$pid)"
    [ -z "$pids" ] || bad="$bad $u(cgroup pids: $pids)"
  done
  [ -z "$bad" ] && return 0
  CONFIRM_BAD="$bad"; return 1
}
CONFIRM_BAD=""
c=0
until confirm_retired; do
  c=$((c + 1))
  [ "$c" -ge "$CONFIRM_TRIES" ] && fail "retired reader still running after the stop:$CONFIRM_BAD" 21
  sleep "$CONFIRM_SLEEP"
done
echo "retired readers stopped and confirmed (terminal, no main PID, cgroup empty): $RETIRED_SERVICES $(stamp)"

if [ -n "$CLUSTER_SET" ]; then
  # step 0: the store copies refreshed after the timers stop and before the writers drain (the pins the plan admits are the generations
  # this run publishes); bounded, and a mirror that does not finish only costs a stale pin the admission refuses -- the adapter is then
  # recorded refused. Only when a NAMED adapter reads on the cluster: a window with none waits on no mirror (one once waited 43 min)
  echo "-- store mirror $(stamp)"
  systemctl --user start --no-block registry-store-mirror.service 2>/dev/null || echo "store mirror could not be started"
  # bounded by ITERATIONS (the sleep may be 0 in a rehearsal): at most MIRROR_WAIT_S / max(QUIET_SLEEP, 1) polls
  m=0; mirror_tries=$((MIRROR_WAIT_S / (QUIET_SLEEP > 0 ? QUIET_SLEEP : 1)))
  while [ "$(systemctl --user is-active registry-store-mirror.service 2>/dev/null)" = activating ] || [ "$(systemctl --user is-active registry-store-mirror.service 2>/dev/null)" = active ]; do
    m=$((m + 1)); [ "$m" -ge "$mirror_tries" ] && { echo "store mirror still running after ${MIRROR_WAIT_S}s; going on (admission decides) $(stamp)"; break; }
    sleep "$QUIET_SLEEP"
  done
  echo "store mirror: $(systemctl --user is-active registry-store-mirror.service 2>/dev/null || echo unknown) $(stamp)"
  # THE CARRY UNDER THE WORKTREE'S CODE (see the header): the unit above ran the BASE's `stores mirror` with the BASE's configuration,
  # which cannot see a component this landing declares; the worktree's CLI carries every component the NEW configuration names, bounded,
  # its output beside the window's artifacts. Never a reason to fail the window here: the store authority stage decides what is certified
  carry_new_tree() {
    MIRROR_OUT="$STATE/windows/store-mirror-$WINDOW_ID.jsonl"
    # the bound is the CONTROLLER's (`--bound-s`), never an outer `timeout`: a killed controller abandons its submitted Slurm verifier with the
    # MirrorLock released, and the next carry would rename and repopulate the generation that verifier still reads; under --bound-s the
    # controller cancels its own recorded job by id, confirms it terminal in sacct and out of the queue, and only then returns
    echo "store mirror: carry under the worktree's code (stores mirror --dest $CLUSTER_DEST --bound-s $MIRROR_WAIT_S under $NEW_PYPATH) $(stamp)"
    $NEW_REG --root "$REGISTRY_ROOT" stores mirror --dest "$CLUSTER_DEST" --bound-s "$MIRROR_WAIT_S" >"$MIRROR_OUT" 2>&1; mrc=$?
    echo "store mirror: new-tree carry rc=$mrc; $(grep -c '"summary": true' "$MIRROR_OUT" 2>/dev/null || echo 0) summary line(s) in $MIRROR_OUT $(stamp)"
  }
  # the cutover window carries after the site cutover: the worktree's code knows only the new names
  if [ -z "$SITE_CUTOVER" ]; then carry_new_tree; else echo "store mirror: the carry under the worktree's code runs after the site cutover $(stamp)"; fi
fi

# the drain: no $UNITS service `activating`, the harvester lock free (a pass the stop let finish, or one a timer fired before it)
quiet() {
  local a
  a=$(systemctl --user is-active $UNITS 2>/dev/null | grep -c activating)
  [ "$a" = 0 ] && ! fuser "$REGISTRY_ROOT/harvester.lock" >/dev/null 2>&1
}
i=0
until quiet; do i=$((i + 1)); [ "$i" -ge "$QUIET_TRIES" ] && fail "writers never quiet" 3; sleep "$QUIET_SLEEP"; done
echo "writers quiet $(stamp)"
# the mirrors services (cutover window): a pass in flight was waited out by the drain above; each is then confirmed terminal, no main PID and
# an empty cgroup by PID list, before anything is renamed -- never stopped mid-pass, since a pass writes its pointer when it ends
services_terminal() {
  local u st pid pids bad=""
  for u in "$@"; do
    st=$(systemctl --user is-active "$u" 2>/dev/null || true)
    pid=$(systemctl --user show -p MainPID --value "$u" 2>/dev/null || true)
    pids=$(cgroup_pids "$u")
    case "$st" in inactive|failed|"") ;; *) bad="$bad $u(state=$st)";; esac
    [ -z "$pid" ] || [ "$pid" = 0 ] || bad="$bad $u(MainPID=$pid)"
    [ -z "$pids" ] || bad="$bad $u(cgroup pids: $pids)"
  done
  TERMINAL_BAD="$bad"; [ -z "$bad" ]
}
if [ -n "$SITE_CUTOVER" ]; then
  MIRRORS_SERVICES=${MIRRORS_TIMERS//.timer/.service}
  c=0
  until services_terminal $MIRRORS_SERVICES; do
    c=$((c + 1))
    [ "$c" -ge "$CONFIRM_TRIES" ] && fail "a mirrors service is still running after the drain:$TERMINAL_BAD (its state is keyed by destination name, which the site cutover renames)" 31
    sleep "$CONFIRM_SLEEP"
  done
  echo "mirrors services confirmed terminal (no main PID, cgroup empty): $MIRRORS_SERVICES $(stamp)"
fi
# the drain is the one moment with no writer, so the WAL is checkpointed HERE -- a reader-starved file (13.7 GB on 2026-10-05; the
# harvester checkpoints only at a COMPLETED cycle) shrinks where it can. Through the registry python with a bounded busy timeout;
# readers may still hold the WAL (the three numbers say how far it got: busy, log frames, checkpointed frames); the result and the
# WAL bytes before / after go to the chain log; NEVER a reason to fail the window (a failed or empty answer is logged and the window goes on).
echo "-- wal checkpoint $(stamp)"
WAL_FILE="$REGISTRY_ROOT/registry.sqlite-wal"
wal_bytes() { stat -c %s "$WAL_FILE" 2>/dev/null || echo 0; }
WAL_BEFORE=$(wal_bytes)
CKPT_OUT=$("$PY" -c 'import sqlite3, sys; c = sqlite3.connect(sys.argv[1], timeout=120); c.execute("PRAGMA busy_timeout = 120000"); r = c.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone(); c.close(); print("busy=%s log=%s checkpointed=%s" % tuple(r))' "$REGISTRY_ROOT/registry.sqlite" 2>&1); CKPT_RC=$?
if [ "$CKPT_RC" = 0 ]; then
  echo "wal checkpoint: ${CKPT_OUT:-no result (the registry python printed nothing)}; wal bytes before=$WAL_BEFORE after=$(wal_bytes) $(stamp)"
else
  echo "wal checkpoint: rc=$CKPT_RC (${CKPT_OUT:-no output}); never fails the window; wal bytes before=$WAL_BEFORE after=$(wal_bytes) $(stamp)"
fi
if [ -n "$BUSY_PATTERN" ] && ps -eo args | grep -E "$BUSY_PATTERN" | grep -v grep | grep -q .; then fail "base checkout busy ($BUSY_PATTERN)" 6; fi

# ---- the configuration conversion (C9; see the header): BEFORE the fast-forward and before any new-code invocation; a failed
# validation restores the backup byte for byte and aborts (exit 24) with the base and the units untouched
echo "-- config convert $(stamp)"
CONV_OUT=$("$CONVERT_PY" "$HERE/convert_config.py" --root "$REGISTRY_ROOT" 2>&1); crc=$?
[ "$crc" = 0 ] || { echo "$CONV_OUT" | tail -n 3; fail "config convert FAILED rc=$crc (nothing written by a refusal; the file is as it was)" 24; }
CONVERTED=$(sed -n 's/^CONVERTED=//p' <<<"$CONV_OUT" | head -n 1); CONV_BACKUP=$(sed -n 's/^BACKUP=//p' <<<"$CONV_OUT" | head -n 1)
CONV_REASON=$(sed -n 's/^REASON=//p' <<<"$CONV_OUT" | head -n 1)
echo "config convert: ${CONV_REASON:-done}${CONV_BACKUP:+; backup $CONV_BACKUP}"
restore_config() {  # the abort path: the original bytes back, verified
  [ -n "$CONV_BACKUP" ] || { echo "config convert: nothing was converted, nothing to restore"; return 0; }
  "$CONVERT_PY" "$HERE/convert_config.py" --root "$REGISTRY_ROOT" --restore "$CONV_BACKUP" 2>&1 | tail -n 4     # RESTORED=1, CONFIG, BACKUP, SHA
  echo "config convert: backup $CONV_BACKUP restored byte for byte $(stamp)"
}
# (a) the WORKTREE's new loader: `config --check` (C10) -- exit 0 is the only pass
if [ -n "$SITE_CUTOVER" ]; then
  echo "config convert: the new loader checks the configuration the site cutover installs (config --check there), not the one it replaces"
elif ! env PYTHONPATH="$NEW_PYPATH" "$NEW_PY" -m sisteer.registry.cli --root "$REGISTRY_ROOT" config --check >"$STATE/windows/config-check-$WINDOW_ID.txt" 2>&1; then
  tail -n 3 "$STATE/windows/config-check-$WINDOW_ID.txt"; restore_config
  fail "config convert: the configuration failed the new loader's check (config --check under $NEW_PYPATH; output in $STATE/windows/config-check-$WINDOW_ID.txt)" 24
fi
# (b) the BASE's old loader must still parse what the old timers run on until the fast-forward: `config --json`
if ! $REG config --json >"$STATE/windows/config-json-$WINDOW_ID.txt" 2>&1; then
  tail -n 3 "$STATE/windows/config-json-$WINDOW_ID.txt"; restore_config
  fail "config convert: the base's loader does not parse the configuration (config --json; output in $STATE/windows/config-json-$WINDOW_ID.txt)" 24
fi
echo "config convert: validated by the new loader (config --check under $NEW_PYPATH) and the base's loader (config --json); converted=${CONVERTED:-0} $(stamp)"

# ---- the site cutover (see the header): inside the drain, after the conversion and before anything else moves. site_cutover.py refuses by
# name with nothing touched, or installs the prepared set and validates it under the worktree's code, restoring every byte itself on a failure
if [ -n "$SITE_CUTOVER" ]; then
  echo "-- site cutover $(stamp) ($SITE_CUTOVER)"
  CUTOVER_BACKUP="$STATE/windows/site-cutover-$WINDOW_ID"
  CUT_UNITS=()
  for u in $UNITS $RETIRED_SERVICES; do CUT_UNITS+=(--unit "$u"); done
  CUT_OUT=$("$CONVERT_PY" "$HERE/site_cutover.py" apply --set "$SITE_CUTOVER" --root "$REGISTRY_ROOT" --state "$STATE" --backup "$CUTOVER_BACKUP" \
    --own-op "$WINDOW_ID" "${CUT_UNITS[@]}" --new-py "$NEW_PY" --new-pypath "$NEW_PYPATH" --clusters-py "$NEW_W/reproduce/lib/clusters.py" \
    ${WINDOW_EXPECT_SITE_ALIASES_SHA256:+--expect-site-aliases-sha256 "$WINDOW_EXPECT_SITE_ALIASES_SHA256"} 2>&1); crc=$?
  printf '%s\n' "$CUT_OUT" | sed 's/^/  /'
  case "$crc" in
    0) CUTOVER_INSTALLED=1
       echo "site cutover: installed and validated (backups in $CUTOVER_BACKUP; $(sed -n 's/^DIGEST=//p' <<<"$CUT_OUT" | head -n 1 | sed 's/^$/no site-alias digest/')) $(stamp)" ;;
    31) fail "site cutover refused or failed rc=$crc (nothing installed, or every file restored and every inbox moved back: see the lines above)" 31 ;;
    *) # 3 (its own restore did not finish) or any other code (it stopped without saying): the ledger says whether anything was installed.
       # With one, the trap runs the restore again and the timers stay stopped unless that restore finishes
       if [ -f "$CUTOVER_BACKUP/cutover.json" ]; then
         CUTOVER_INSTALLED=1; CUTOVER_RESTORE_BROKEN=1
         fail "site cutover FAILED rc=$crc with files possibly installed (see the lines above and $CUTOVER_BACKUP/cutover.json; the trap restores it)" 31
       fi
       fail "site cutover FAILED rc=$crc before it installed anything (no ledger under $CUTOVER_BACKUP)" 31 ;;
  esac
  # the installed site config must read the cluster names the prepared set declares: the window reads the installed one from here on
  SET_NAMES="$(for k in $SITE_KEYS; do printf '%s=%s ' "$k" "${!k}"; done)"
  for k in $SITE_KEYS; do origk="ORIG_$k"; printf -v "$k" '%s' "${!origk}"; done
  if [ -n "$ORIG_SISTEER_CONFIG" ]; then export SISTEER_CONFIG="$ORIG_SISTEER_CONFIG"; else unset SISTEER_CONFIG; fi
  SITE_CONFIG_FILE=""
  SITE_ENV_ERRF="$(mktemp "${TMPDIR:-/tmp}/window-site-env.XXXXXX")"
  site_env_load $SITE_KEYS 2>"$SITE_ENV_ERRF"; SITE_ENV_RC=$?
  SITE_ENV_MSG="$(tr '\n' ' ' <"$SITE_ENV_ERRF")"; rm -f "$SITE_ENV_ERRF"
  [ "$SITE_ENV_RC" = 0 ] || fail "site cutover: the installed site config does not supply the cluster names: ${SITE_ENV_MSG% }" 32
  INSTALLED_NAMES="$(for k in $SITE_KEYS; do printf '%s=%s ' "$k" "${!k}"; done)"
  [ "$INSTALLED_NAMES" = "$SET_NAMES" ] || fail "site cutover: the installed site config ($SITE_CONFIG_FILE) reads ${INSTALLED_NAMES% }, the prepared set ${SET_NAMES% } (the manifest installs config.env where this window does not read it)" 32
  echo "site cutover: the installed site config ($SITE_CONFIG_FILE) reads the prepared names: ${INSTALLED_NAMES% } $(stamp)"
fi

# ---- the preflight (see the header): with the cutover installed, before the store authority, the fast-forward and the migration
if [ -n "$SITE_CUTOVER" ] || [ "${WINDOW_PREFLIGHT:-}" = 1 ]; then
  echo "-- preflight $(stamp)"
  git -C "$BASE" fetch -q origin || fail "preflight: git fetch in $BASE failed" 32
  PREFLIGHT_SHA=$(git -C "$BASE" rev-parse origin/main 2>/dev/null)
  [ -n "$PREFLIGHT_SHA" ] || fail "preflight: origin/main does not resolve in $BASE" 32
  preflight "$REGISTRY_ROOT" "$PREFLIGHT_SHA" "$BASE" || fail "preflight: $PF_WHY" 32
  echo "preflight passed: the window moves to ${PREFLIGHT_SHA:0:12} $(stamp)"
fi
# the carry under the worktree's code the store mirror step left for after the cutover
if [ -n "$SITE_CUTOVER" ] && [ -n "$CLUSTER_SET" ]; then carry_new_tree; fi

# ---- store authority (see the header): under the WORKTREE's new code, the store-mirror timer stopped above, BEFORE the base moves.
# (a) check-authority -> ONE verify-authority per component that is VERIFIED but UNSEALED or UNRECORDED (a never-published, a dangling or a
# moved-ahead one is named, not acted on); (b) the strict check; (c) WINDOW_REQUIRE_CERTIFIED=1 aborts (exit 27) on a component still not
# certified or a check that could not run; a routine window logs the reading and goes on
echo "-- store authority $(stamp)"
AUTH_DIR="$STATE/windows/authority-$WINDOW_ID"
mkdir -p "$AUTH_DIR"
NEWREG="$NEW_REG --root $REGISTRY_ROOT"
authority_fail() {   # the strict reading's abort, or the routine window's reading
  if [ "$REQUIRE_CERTIFIED" = 1 ]; then fail "store authority: $1" 27; fi
  echo "store authority: $1; WINDOW_REQUIRE_CERTIFIED is unset, the window goes on $(stamp)"
}
$NEWREG stores check-authority --dest "$CLUSTER_DEST" --json >"$AUTH_DIR/check-authority.json" 2>"$AUTH_DIR/check-authority.err"; arc=$?
if [ "$arc" = 0 ] || [ "$arc" = 1 ]; then
  # the triage over the JSON, under the shim-free interpreter convert_config.py runs under
  if ! "$CONVERT_PY" - "$AUTH_DIR/check-authority.json" "$AUTH_DIR" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
act, carry, skip = [], [], []
for slug, row in sorted((doc.get("components") or {}).items()):
    if row.get("state") == "not_applicable" or row.get("certified"):
        continue
    never = ("never_published" in (row.get("notes") or [])) or ("not_certified_never_published" in (row.get("problems") or []))
    if row.get("current") and row.get("exists") and row.get("verified") and not (row.get("sealed") and row.get("published_record")):
        act.append(slug)
    elif not row.get("current") and never and row.get("authority", "grading_host_copy") == "grading_host_copy":
        carry.append(slug)      # declared by the new configuration, published by no one: carried under the worktree's code, then verified
    else:
        skip.append(slug + "(" + ",".join(row.get("problems") or row.get("notes") or ["?"]) + ")")
open(sys.argv[2] + "/to-verify.txt", "w").write(" ".join(act) + "\n")
open(sys.argv[2] + "/to-carry.txt", "w").write(" ".join(carry) + "\n")
open(sys.argv[2] + "/not-acted-on.txt", "w").write(" ".join(skip) + "\n")
PY
  then fail "store authority: the check-authority JSON could not be read ($AUTH_DIR/check-authority.json)" 27; fi
  AUTH_ACT=$(cat "$AUTH_DIR/to-verify.txt"); AUTH_CARRY=$(cat "$AUTH_DIR/to-carry.txt"); AUTH_SKIP=$(cat "$AUTH_DIR/not-acted-on.txt")
  echo "store authority: check-authority rc=$arc; never published (carried under the worktree's code, then verify-authority): ${AUTH_CARRY:-none}; verified but unsealed/unrecorded (verify-authority runs): ${AUTH_ACT:-none}; not acted on here (the mirror timer carries them): ${AUTH_SKIP:-none} $(stamp)"
  # the carry first (see the header): the component exists on the destination only after this, so its verify-authority runs after
  AUTH_CARRIED=""
  for slug in $AUTH_CARRY; do
    # bounded by the controller itself (--bound-s: the verifier cancelled by id and confirmed terminal before it returns), never by `timeout`
    $NEWREG stores mirror --dest "$CLUSTER_DEST" --component "$slug" --bound-s "$AUTHORITY_WAIT_S" >>"$AUTH_DIR/carry.jsonl" 2>>"$AUTH_DIR/carry.err"; crc=$?
    cnote=""
    if [ "$crc" = 0 ]; then
      echo "store authority: carry $slug rc=0 (never published; carried under the worktree's code) $(stamp)"
      AUTH_CARRIED="${AUTH_CARRIED:+$AUTH_CARRIED }$slug"
    else
      echo "store authority: carry $slug FAILED rc=$crc$cnote (never published; nothing to verify -- the strict check names it) $(stamp)"
    fi
  done
  for slug in $AUTH_ACT $AUTH_CARRIED; do
    timeout "$AUTHORITY_WAIT_S" $NEWREG stores verify-authority --dest "$CLUSTER_DEST" --component "$slug" --out "$AUTH_DIR" --json >>"$AUTH_DIR/verify-authority.jsonl" 2>>"$AUTH_DIR/verify-authority.err"; vrc=$?
    vnote=""; [ "$vrc" = 124 ] && vnote=" (timed out after ${AUTHORITY_WAIT_S}s)"
    echo "store authority: verify-authority $slug rc=$vrc$vnote $(stamp)"
  done
  $NEWREG stores check-authority --dest "$CLUSTER_DEST" --require-certified --json >"$AUTH_DIR/check-authority-strict.json" 2>"$AUTH_DIR/check-authority-strict.err"; src=$?
  if [ "$src" = 0 ] || [ "$src" = 1 ]; then
    AUTH_SUMMARY=$("$CONVERT_PY" -c 'import json, sys; d = json.load(open(sys.argv[1])); print("certified %s of %s checked; %s problem(s)%s" % (d.get("n_certified"), d.get("n_checked"), d.get("n_problems"), ": " + "; ".join(d.get("problems") or []) if d.get("problems") else ""))' "$AUTH_DIR/check-authority-strict.json" 2>/dev/null || echo "summary unreadable")
    echo "store authority: strict check-authority rc=$src: $AUTH_SUMMARY $(stamp)"
    [ "$src" = 0 ] || authority_fail "not every component is certified ($AUTH_SUMMARY)"
  else
    authority_fail "the strict check-authority could not run (rc=$src: $(tail -n 1 "$AUTH_DIR/check-authority-strict.err" 2>/dev/null))"
  fi
else
  authority_fail "check-authority could not run (rc=$arc: $(tail -n 1 "$AUTH_DIR/check-authority.err" 2>/dev/null))"
fi

cd "$BASE" || fail "no base checkout at $BASE" 1
git fetch -q origin || fail "fetch failed" 5
if git status --short | grep -v "^??" | grep -q .; then fail "base checkout dirty" 4; fi
if [ -n "$PREFLIGHT_SHA" ] && [ "$(git rev-parse origin/main 2>/dev/null)" != "$PREFLIGHT_SHA" ]; then
  fail "origin/main moved after the preflight (checked ${PREFLIGHT_SHA:0:12}, now $(git rev-parse --short origin/main 2>/dev/null)); nothing fast-forwarded" 32
fi
git checkout -q main && PRE_FF_SHA=$(git rev-parse HEAD) && git merge -q --ff-only origin/main || fail "ff failed" 5
BASE_MOVED=1
BASE_SHA=$(git rev-parse HEAD)   # pinned here, re-checked before the plan and immediately before the apply
echo "base at $(git rev-parse --short HEAD) $(stamp)"

# install the units from the fast-forwarded base — nothing enabled or started here; the trap restores the timers
echo "-- install units $(stamp)"
# the installer reads the site config the window read (SISTEER_CONFIG) and the window's registry root
INSTALL_ENV=(); [ -z "${SITE_CONFIG_FILE:-}" ] || INSTALL_ENV=("SISTEER_CONFIG=$SITE_CONFIG_FILE")
RENDER_DIR="$STATE/windows/units-rendered-$WINDOW_ID"
UNITS_DIFF="$STATE/windows/units-diff-$WINDOW_ID.txt"
mkdir -p "$RENDER_DIR"
env "${INSTALL_ENV[@]}" bash "$INSTALLER" --units-only --no-systemctl --unit-dir "$RENDER_DIR" --root "$REGISTRY_ROOT" >"$RENDER_DIR.log" 2>&1; rc=$?
[ "$rc" = 0 ] || { tail -n 5 "$RENDER_DIR.log"; fail "install units FAILED rc=$rc" 7; }
n_render=0; n_diff=0
: >"$UNITS_DIFF"
for f in "$RENDER_DIR"/registry-*; do
  [ -f "$f" ] || continue
  n_render=$((n_render + 1)); u=$(basename "$f")
  diff -u "$UNIT_DIR/$u" "$f" >>"$UNITS_DIFF" 2>&1 || n_diff=$((n_diff + 1))
done
echo "units: $n_render rendered for the site, $n_diff differ from the installed ones (the diff in $UNITS_DIFF) $(stamp)"
if [ -n "$SITE_CUTOVER" ] && [ "$CUTOVER_INSTALLED" = 1 ]; then
  # the unit directory's registry entries as they are -- the unit files, their drop-in directories and the enablement links under
  # *.wants/ and *.requires/, all of which the install and the uninstall stage below change -- for the trap's restore before the migration
  (cd "$UNIT_DIR" && for e in registry-* *.wants/registry-* *.requires/registry-*; do if [ -e "$e" ] || [ -L "$e" ]; then printf '%s\n' "$e"; fi; done) \
    >"$CUTOVER_BACKUP/units.list" || fail "install units: the unit directory $UNIT_DIR could not be listed for the backup" 7
  if [ -s "$CUTOVER_BACKUP/units.list" ]; then
    tar -cf "$CUTOVER_BACKUP/units.tar" -C "$UNIT_DIR" -T "$CUTOVER_BACKUP/units.list" \
      || fail "install units: the unit directory's registry entries could not be backed up" 7
  fi
  UNITS_BACKED_UP=1
  echo "units: $(wc -l <"$CUTOVER_BACKUP/units.list") registry entr(ies) of the unit directory (unit files, drop-ins, enablement links) backed up in $CUTOVER_BACKUP/units.tar"
fi
env "${INSTALL_ENV[@]}" bash "$INSTALLER" --units-only --unit-dir "$UNIT_DIR" --root "$REGISTRY_ROOT" 2>&1 | tail -20
rc=${PIPESTATUS[0]}; [ "$rc" = 0 ] || fail "install units FAILED rc=$rc" 7
INSTALLED=1

# the retired units uninstalled (C7; C10): disabled, their files removed from the unit directory, the Modal-era drop-ins removed
# (RETIRED_DROPINS: a directory whole, a file alone; nothing else under a .d/ is touched), the daemon reloaded -- the units the
# cluster-reads tick replaces and the Modal pull's; they were stopped and confirmed gone above and no path starts them (NEVER_START)
echo "-- uninstall units $(stamp) ($RETIRED_UNITS; drop-ins $RETIRED_DROPINS)"
urc=0; removed=""
for u in $RETIRED_UNITS; do
  systemctl --user disable "$u" >/dev/null 2>&1 || true        # not loaded / not enabled is fine
  if [ -e "$UNIT_DIR/$u" ]; then rm -f "$UNIT_DIR/$u" && removed="${removed:+$removed }$u" || urc=1; fi
done
for d in $RETIRED_DROPINS; do
  case "$d" in */*.conf) [ -e "$UNIT_DIR/$d" ] && { rm -f "$UNIT_DIR/$d" && removed="${removed:+$removed }$d" || urc=1; } ;;
                *)       [ -d "$UNIT_DIR/$d" ] && { rm -rf "$UNIT_DIR/$d" && removed="${removed:+$removed }$d/" || urc=1; } ;; esac
done
systemctl --user daemon-reload 2>/dev/null || urc=1
[ "$urc" = 0 ] || fail "uninstall units FAILED ($RETIRED_UNITS; drop-ins $RETIRED_DROPINS)" 23
echo "retired units uninstalled (disabled, files removed: ${removed:-none}; daemon reloaded) $(stamp)"

# ---- the migration plan, READ and CHECKED before anything is applied (see the header): the whole plan to the log and to a file, its
# token (the file's sha256) beside it; the version line must render; a `rebuild table` line is a STOP no approval overrides;
# WINDOW_EXPECT_MIGRATION pins the pending list; WINDOW_PLAN_APPROVAL pauses a plan with something to apply until the operator, having
# read the DISPLAYED plan, writes its token into the approval file -- a file from before the plan, from during it, or carrying another
# token never counts. The base is PINNED here (HEAD, the migration files, the plan file) and re-checked before the apply.
echo "-- migrate (plan) $(stamp)"
MIG_DIR="$PYPATH/sisteer/registry/migrations"
mig_pins() {
  # the files `registry migrate` enumerates (the migration package's *.sql and *.py, one level), each with its sha256, sorted by name;
  # an absent directory reads as no lines
  [ -d "$MIG_DIR" ] || return 0
  (cd "$MIG_DIR" && find . -maxdepth 1 -type f \( -name '*.sql' -o -name '*.py' \) -print0 | sort -z | xargs -0 -r sha256sum)
}
pins_now() { printf 'base %s\nplan %s\n%s' "$(git -C "$BASE" rev-parse HEAD)" "$(sha256sum "$PLAN_FILE" 2>/dev/null | cut -d' ' -f1)" "$(mig_pins)"; }
APPROVAL="${WINDOW_PLAN_APPROVAL:-}"
if [ -n "$APPROVAL" ] && [ -e "$APPROVAL" ]; then
  rm -f -- "$APPROVAL"
  echo "a pre-existing approval at $APPROVAL was discarded: the approval must follow the plan"
fi
head_now=$(git -C "$BASE" rev-parse HEAD)
[ "$head_now" = "$BASE_SHA" ] || fail "base HEAD moved before the plan: fast-forward $BASE_SHA, now $head_now; nothing applied" 19
# the race guard: the preflight under the lock refused a reused id before anything moved; a file that appeared since (a hand-written one,
# a writer outside the lock) is still refused here, before the plan is written
[ -e "$PLAN_FILE" ] && fail "plan file $PLAN_FILE already exists (a window id collision); nothing applied" 8
MIG_PINS=$(mig_pins)
$REG migrate --plan >"$PLAN_FILE" 2>&1; prc=$?
chmod a-w "$PLAN_FILE" 2>/dev/null
[ "$(mig_pins)" = "$MIG_PINS" ] || fail "the migration files under $MIG_DIR changed while the plan was read; nothing applied" 19
# the plan is read ONCE (trailing newlines kept); the token is the sha256 of these bytes, and every check below reads them, never the file
PLAN_TEXT=$(cat "$PLAN_FILE"; printf x); PLAN_TEXT=${PLAN_TEXT%x}
printf '%s' "$PLAN_TEXT"
PLAN_TOKEN=$(printf '%s' "$PLAN_TEXT" | sha256sum | cut -d' ' -f1)
echo "plan written to $PLAN_FILE (migrate --plan rc=$prc); plan token $PLAN_TOKEN (the sha256 of that file)"
PINS=$(printf 'base %s\nplan %s\n%s' "$BASE_SHA" "$PLAN_TOKEN" "$MIG_PINS")
printf '%s\n' "$PINS" >"$PINS_FILE"
MIG_N=$(printf '%s' "$MIG_PINS" | grep -c .)
echo "pins recorded in $PINS_FILE: base $BASE_SHA, plan $PLAN_TOKEN, $MIG_N migration file(s) under $MIG_DIR"
PENDING=$(printf '%s' "$PLAN_TEXT" | sed -n 's/^schema version [0-9]*; code [0-9]*; pending: //p' | head -n 1)
[ -n "$PENDING" ] || fail "migrate --plan did not render its version line (rc=$prc; see $PLAN_FILE); nothing applied" 8
REBUILDS=$(printf '%s' "$PLAN_TEXT" | sed -n 's/^rebuild table \([A-Za-z0-9_]*\):.*/\1/p' | paste -sd, -)
if [ -n "$REBUILDS" ]; then
  if [ "${WINDOW_ALLOW_REBUILD:-0}" = "1" ]; then
    echo "plan has rebuild line(s) for: $REBUILDS -- applied because WINDOW_ALLOW_REBUILD=1 names a reviewed rebuild $(stamp)"
  else
    fail "migration plan has a rebuild line ($REBUILDS): a STOP, nothing applied, no approval overrides it -- a reviewed rebuild runs with WINDOW_ALLOW_REBUILD=1 (plan in $PLAN_FILE)" 15
  fi
fi
if [ -n "${WINDOW_EXPECT_MIGRATION:-}" ] && [ "$PENDING" != "$WINDOW_EXPECT_MIGRATION" ]; then
  fail "migration plan pending '$PENDING', expected exactly '$WINDOW_EXPECT_MIGRATION' (WINDOW_EXPECT_MIGRATION); nothing applied" 15
fi
plan_approved() {
  # the approval file counts only when its content (whitespace stripped) IS this plan's token; any other content is logged once per
  # distinct value and ignored while the wait goes on -- the operator overwrites it with the printed command
  local got
  [ -e "$APPROVAL" ] || return 1
  got=$(tr -d '[:space:]' <"$APPROVAL" 2>/dev/null)
  [ "$got" = "$PLAN_TOKEN" ] && return 0
  if [ "$got" != "$IGNORED" ]; then
    echo "approval at $APPROVAL ignored: its content '${got:0:80}' is not this plan's token $PLAN_TOKEN (written before the plan, or for another plan); still waiting $(stamp)"
    IGNORED="$got"
  fi
  return 1
}
SHAPE_NOTE=""; printf '%s' "$PLAN_TEXT" | grep -q '^shape differs' && SHAPE_NOTE="; shape differs"
if [ "$PENDING" = none ] && [ -z "$SHAPE_NOTE" ]; then
  echo "plan: nothing pending, the shape matches; no approval needed"
elif [ -n "$APPROVAL" ]; then
  # the approval boundary: the complete plan is displayed above; a file present NOW was written during the plan, not after reading it
  if [ -e "$APPROVAL" ]; then
    rm -f -- "$APPROVAL"
    echo "an approval at $APPROVAL written while the plan was being generated was discarded: nobody had seen the plan whole"
  fi
  echo "plan: pending $PENDING$SHAPE_NOTE; waiting up to ${PLAN_WAIT_S}s for the operator's approval of THIS plan -- after reading it, approve with: echo $PLAN_TOKEN > $APPROVAL $(stamp)"
  w=0; plan_tries=$((PLAN_WAIT_S / (QUIET_SLEEP > 0 ? QUIET_SLEEP : 1)))
  IGNORED="(none yet)"   # a value no whitespace-stripped file content can equal, so an empty file is logged too
  until plan_approved; do
    w=$((w + 1)); [ "$w" -ge "$plan_tries" ] && fail "migration plan not approved within ${PLAN_WAIT_S}s (no approval at $APPROVAL carrying plan token $PLAN_TOKEN); nothing applied" 16
    sleep "$QUIET_SLEEP"
  done
  rm -f -- "$APPROVAL"
  echo "plan approved ($APPROVAL carried plan token $PLAN_TOKEN; consumed) $(stamp)"
else
  echo "plan: pending $PENDING$SHAPE_NOTE; applying without a pause (WINDOW_PLAN_APPROVAL unset)"
fi
# ---- the pin re-check, immediately before the apply (see the header): the base HEAD, the migration files and the plan file must read
# exactly as the plan stage recorded them -- `registry migrate` enumerates the directory's CURRENT files, so a difference means the
# approved plan is not what the apply would do; both readings are printed
NOW=$(pins_now)
if [ "$NOW" != "$PINS" ]; then
  why=""
  b_now=$(printf '%s\n' "$NOW" | sed -n 's/^base //p'); [ "$b_now" = "$BASE_SHA" ] || why="$why; base HEAD $BASE_SHA -> ${b_now:-?}"
  p_now=$(printf '%s\n' "$NOW" | sed -n 's/^plan //p'); [ "$p_now" = "$PLAN_TOKEN" ] || why="$why; plan file sha256 $PLAN_TOKEN -> ${p_now:-?}"
  if [ "$(printf '%s\n' "$NOW" | grep -v '^base \|^plan ')" != "$(printf '%s\n' "$PINS" | grep -v '^base \|^plan ')" ]; then
    why="$why; migration files under $MIG_DIR differ (recorded < vs now >):"
    echo "migration files recorded at the plan (<) vs now (>):"
    diff <(printf '%s\n' "$PINS" | grep -v '^base \|^plan ') <(printf '%s\n' "$NOW" | grep -v '^base \|^plan ') | grep '^[<>]' | sed 's/^/  /'
  fi
  fail "pins changed between the plan and the apply${why}; the approved plan is not what the apply would do (recorded in $PINS_FILE); nothing applied" 19
fi
echo "pin check before the apply: base $BASE_SHA, $MIG_N migration file(s), plan file sha256 = token; unchanged since the plan $(stamp)"
echo "-- migrate (apply) $(stamp)"; $REG migrate 2>&1 | tail -4
rc=${PIPESTATUS[0]}
if [ "$rc" != 0 ] && [ "$CUTOVER_INSTALLED" = 1 ]; then
  # a failed apply can still have committed (`registry migrate` checks the shape after it commits), so the database's own version, read the
  # way the plan read it, decides whether the way back is open. Moved, or unreadable: the migration counts as applied, nothing is rolled
  # back and the timers stay stopped (the old code refuses this database, and the new code's check did not pass). Unchanged: nothing
  # committed, and the trap restores the cutover as for any earlier abort
  PLAN_V=$(printf '%s' "$PLAN_TEXT" | sed -n 's/^schema version \([0-9]*\); code .*/\1/p' | head -n 1)
  POST_V=$($REG migrate --plan 2>/dev/null | sed -n 's/^schema version \([0-9]*\); code .*/\1/p' | head -n 1)
  if [ -z "$POST_V" ] || [ "$POST_V" != "$PLAN_V" ]; then
    MIGRATE_APPLIED=1
    FORWARD_ONLY="the migration committed (schema version ${PLAN_V:-?} -> ${POST_V:-unreadable}) and registry migrate exited $rc"
    echo "migrate: $FORWARD_ONLY; the site cutover stays installed and the way out is forward $(stamp)"
  else
    echo "migrate: the schema version is still $POST_V after the failed apply, so nothing committed $(stamp)"
  fi
fi
[ "$rc" = 0 ] || fail "migrate FAILED rc=$rc" 8
MIGRATE_APPLIED=1

# dimensions load: skipped when NOTHING it reads changed since the last load on this database (the content sha of the SEED FILES the load
# reads + the code's schema version + the DB inode; the load cost 6–36 min per window while the seeds were unchanged).
# The key covers exactly `registry dimensions load`'s inputs (adapters/dimensions.py SEED_FILES, pinned by tests/test_registry_window.py) --
# a change to any OTHER file under configs/registry (cluster_reads.yaml, a roster, a rulings file) never forces a dims cycle.
# A seed file that is absent hashes as `missing`, so adding one later changes the key. FORCE_DIMS=1 forces.
DIMS_SEED_FILES="decode_policy_seed.yaml evals.yaml instruments.yaml item_sets.yaml judges.yaml legacy_spellings.yaml models.yaml si_texts.yaml"
SEEDS_SHA=$(cd "$BASE/configs/registry" 2>/dev/null && for f in $DIMS_SEED_FILES; do
  if [ -f "$f" ]; then printf '%s  %s\n' "$(sha256sum < "$f" | cut -d' ' -f1)" "$f"; else printf 'missing  %s\n' "$f"; fi
done | sha256sum | cut -c1-16)
SCHEMA_V=$($REG migrate --plan 2>/dev/null | grep -oE 'schema version [0-9]+' | head -n 1 | grep -oE '[0-9]+')
DIMS_KEY="seeds=${SEEDS_SHA:-?} schema=${SCHEMA_V:-?} db=$(stat -c %i "$REGISTRY_ROOT/registry.sqlite" 2>/dev/null)"
DIMS_STAMP="$REGISTRY_ROOT/.dimensions-last-load"
if [ "${FORCE_DIMS:-0}" != "1" ] && [ -f "$DIMS_STAMP" ] && [ "$(cat "$DIMS_STAMP")" = "$DIMS_KEY" ]; then
  echo "-- dimensions load SKIPPED (seeds + schema unchanged since the last load: $DIMS_KEY) $(stamp)"
else
  echo "-- dimensions load $(stamp) ($DIMS_KEY)"
  $REG dimensions load --seeds "$BASE/configs/registry" 2>&1 | tail -4
  rc=${PIPESTATUS[0]}; [ "$rc" = 0 ] || fail "dimensions load FAILED rc=$rc" 9
  echo "$DIMS_KEY" >"$DIMS_STAMP"
fi

# ---- the reconciliation slot (see the header): each named `registry reconcile` action's dry run, then --apply only for the actions
# WINDOW_RECONCILE_APPLY lists (checked by name before the lock); the records fold in this window's one ingest
for rcn in ${WINDOW_RECONCILE:-}; do
  reconcile_args "$rcn"
  echo "-- reconcile $rcn (dry run) $(stamp)"
  $REG reconcile "$rcn" "${RARGS[@]}" --dry-run 2>&1 | tail -12
  rc=${PIPESTATUS[0]}; [ "$rc" = 0 ] || fail "reconcile $rcn dry run FAILED rc=$rc" 17
  if reconcile_applies "$rcn"; then
    echo "-- reconcile $rcn (apply) $(stamp)"
    $REG reconcile "$rcn" "${RARGS[@]}" --apply 2>&1 | tail -12
    rc=${PIPESTATUS[0]}; [ "$rc" = 0 ] || fail "reconcile $rcn apply FAILED rc=$rc (2 = the release's gate refused it, or a row held behind an unfolded record; nothing written for it)" 17
  fi
done

# ---- the lifecycle restamp slot (see the header): the certified movers re-derived on the LIVE database under the fast-forwarded base's reducer,
# PLAN then APPLY + VERIFY, the maintenance flag held, the writers quiet; off unless WINDOW_RESTAMP_KEYS names the keys file (exit 28 on any refusal)
if [ -n "${WINDOW_RESTAMP_KEYS:-}" ]; then
  RESTAMP_DIR="$STATE/windows/restamp-$WINDOW_ID"
  mkdir -p "$RESTAMP_DIR"
  [ -s "$WINDOW_RESTAMP_KEYS" ] || fail "restamp keys: no keys file at $WINDOW_RESTAMP_KEYS (WINDOW_RESTAMP_KEYS); nothing applied" 28
  [ -e "$MAINT_FLAG" ] || fail "restamp keys: the maintenance flag $MAINT_FLAG is not held; the restamp runs only inside a drained window; nothing applied" 28
  echo "-- restamp keys (plan) $(stamp) ($WINDOW_RESTAMP_KEYS -> $RESTAMP_DIR)"
  $REG replica restamp-keys --db "$REGISTRY_ROOT/registry.sqlite" --keys "$WINDOW_RESTAMP_KEYS" --out "$RESTAMP_DIR" 2>&1 | tail -n 14
  rc=${PIPESTATUS[0]}; [ "$rc" = 0 ] || fail "restamp keys plan REFUSED rc=$rc (13 = an unknown key or a change outside the lifecycle class; the plan is in $RESTAMP_DIR); nothing applied" 28
  echo "-- restamp keys (apply) $(stamp)"
  $REG replica restamp-keys --db "$REGISTRY_ROOT/registry.sqlite" --keys "$WINDOW_RESTAMP_KEYS" --out "$RESTAMP_DIR" --apply 2>&1 | tail -n 6
  rc=${PIPESTATUS[0]}; [ "$rc" = 0 ] || fail "restamp keys apply FAILED rc=$rc (13 = the plan moved or was refused, nothing written; 14 = the re-read after the write does not match the plan or the refold (a planned value, an unplanned move, the whole row, a history snapshot): see $RESTAMP_DIR)" 28
  echo "restamp keys: applied and verified against the plan ($RESTAMP_DIR/restamp_apply.json) $(stamp)"
else
  echo "restamp keys: skipped (WINDOW_RESTAMP_KEYS unset) $(stamp)"
fi

# ---- the acceptance before-phase (see the header): between the timer stop and the reader-state export, a FAIL aborts the window (exit 25) ----
TREE_SHA=$(git rev-parse HEAD)
SIDECAR="$STATE/windows/reader_state-$WINDOW_ID.sqlite"
if [ -n "$ACCEPTANCE_OUT" ]; then
  echo "-- acceptance before-phase $(stamp)"
  mkdir -p "$ACCEPTANCE_OUT"
  "$ACCEPTANCE_SCRIPT" --root "$REGISTRY_ROOT" --repo "$BASE" --phase before --window "$WINDOW_ID" --out "$ACCEPTANCE_OUT" \
    ${ACCEPTANCE_LIST:+--bumped-list "$ACCEPTANCE_LIST"} > "$ACCEPTANCE_OUT/before.txt" 2>&1
  rc=$?; tail -2 "$ACCEPTANCE_OUT/before.txt" 2>/dev/null
  [ "$rc" = 0 ] || fail "acceptance before-phase FAILED rc=$rc (see $ACCEPTANCE_OUT/before.txt)" 25
  if [ -z "$CLUSTER_SET" ]; then
    # the protocol requires the sidecar the after-phase predicts from: no cluster adapter named, so the export runs here over the window's adapters
    echo "-- reader-state export $(stamp)"
    $REG reader-state export --adapter $ADAPTERS --out "$SIDECAR" --repo-root "$BASE" 2>&1 | tail -3
    rc=${PIPESTATUS[0]}; [ "$rc" = 0 ] || fail "reader-state export FAILED rc=$rc (the acceptance protocol requires the sidecar)" 25
  fi
fi

# ---- cluster reads: every named adapter reads on cluster A, or is RECORDED REFUSED; nothing reads a store on the grading host (C6, S2) ----
REFUSED_SET=""          # the adapters recorded refused this window, with the reason in REFUSED_WHY
REFUSED_WHY=""
refuse_adapter() { in_set "$1" "$REFUSED_SET" && return; REFUSED_SET="${REFUSED_SET:+$REFUSED_SET }$1"; REFUSED_WHY="$REFUSED_WHY"$'\n'"$1=$2"; }
why_of() { printf '%s\n' "$REFUSED_WHY" | grep -m1 "^$1=" | cut -d= -f2-; }
cluster_fail() {
  echo "cluster reads: $1; every cluster adapter is RECORDED REFUSED this window (the grading host reads no store) $(stamp)"
  local a; for a in $CLUSTER_SET; do refuse_adapter "$a" "cluster reads: $1"; done
}
cluster_stages() {
  # every stage in order; a failure names itself, refuses every cluster adapter and returns (the window goes on to the ingest)
  # the STANDING inbox first: a complete, marker-bearing revision an earlier window delivered and never merged (interrupted
  # before its inbox-merge) is merged into records/ BEFORE the reader-state export, so the shards' reader state already holds its id and
  # no shard reissues it (reader_state.published_dirs_of reads records/, not the inbox); a marker-less partial stays in the inbox by
  # inbox-merge's own rule (harvest.scan_inbox: "incomplete") for a window whose summaries list the id to complete. Here, after the writers
  # are quiet and the base is checked out (the merge is the same file move the post-pull inbox-merge does; nothing else writes records/ now)
  echo "-- cluster reads inbox-merge $(stamp)"
  # first the receive directories an interrupted pull may have left: a complete, verified revision under inbox/.recv-* is
  # installed from its RECEIVING manifest, anything else moved under _failed/<window>/recv-*/ with a loud UNRECOVERABLE line (never deleted)
  $REG cluster-reads recover --window "$WINDOW_ID" 2>&1 | tail -6
  rc=${PIPESTATUS[0]}; [ "$rc" = 0 ] || { cluster_fail "receive-directory recovery FAILED rc=$rc"; return; }
  $REG inbox-merge 2>&1 | tail -1
  rc=${PIPESTATUS[0]}; [ "$rc" = 0 ] || { cluster_fail "standing inbox-merge FAILED rc=$rc"; return; }
  echo "-- reader-state export $(stamp)"
  $REG reader-state export --adapter $CLUSTER_SET --out "$SIDECAR" --repo-root "$BASE" 2>&1 | tail -3
  rc=${PIPESTATUS[0]}; [ "$rc" = 0 ] || { cluster_fail "reader-state export FAILED rc=$rc"; return; }
  echo "-- tree export $(stamp)"
  "$MIRROR_REPOS" --cluster "$MIRROR_REPOS_CLUSTER" "$MIRROR_REPOS_NAME" >/dev/null 2>&1 || echo "mirror-repos.sh reported a problem (tree_export verifies the sha on the mirror)"
  # the archived branches and tags (mirror-repos.sh moves heads and tags only): a note, not a refusal -- the reads need the tree's code
  # alone, and a gate that reuses this tree completes it against the mirror before running the suite
  HOUT=$("$MIRROR_HISTORY" "$BASE" "ssh://$CLUSTER_HOST$CLUSTER_REPO_MIRROR" 2>&1); hrc=$?
  [ "$hrc" = 0 ] || echo "mirror_history.sh reported a problem (rc $hrc): $(tail -n 1 <<<"$HOUT"); the exported tree may lack archived branches or tags (the reads need neither)"
  TOUT=$(ssh -o BatchMode=yes -o ConnectTimeout=25 "$CLUSTER_HOST" bash -s -- "$TREE_SHA" "$CLUSTER_REPO_MIRROR" "$CLUSTER_SHARED/gates/trees/$TREE_SHA" < "$HERE/tree_export.sh" 2>&1); trc=$?
  echo "$TOUT" | tail -2
  if [ "$trc" != 0 ] || ! echo "$TOUT" | grep -q '^TREE '; then cluster_fail "tree export FAILED rc=$trc"; return; fi
  echo "-- cluster reads plan $(stamp)"
  # the window's per-adapter read policy travels in the plan: --full for $FULL_ADAPTERS, --reread --force (an operator override) for $REREAD_FORCE_ADAPTERS;
  # the plan reads the harvester's standing retirement holds off the sidecar exported above (--reader-state), the snapshot the shards judge
  # against, never the live database (registry-hardening 3a, closing pass)
  $REG cluster-reads plan --window "$WINDOW_ID" --adapter $CLUSTER_SET --tree-sha "$TREE_SHA" --dest "$CLUSTER_DEST" --repo-root "$BASE" \
    --reader-state "$SIDECAR" ${FULL_ADAPTERS:+--full-adapter $FULL_ADAPTERS} ${REREAD_FORCE_ADAPTERS:+--reread-force-adapter $REREAD_FORCE_ADAPTERS} 2>&1 | tail -12
  rc=${PIPESTATUS[0]}; [ "$rc" = 0 ] || { cluster_fail "plan FAILED rc=$rc"; return; }
  echo "-- cluster reads stage $(stamp)"
  $REG cluster-reads stage --window "$WINDOW_ID" --reader-state "$SIDECAR" 2>&1 | tail -3
  rc=${PIPESTATUS[0]}; [ "$rc" = 0 ] || { cluster_fail "stage FAILED rc=$rc"; return; }
  if [ -z "$CLUSTER_VENV" ]; then
    PH=$(git show "$TREE_SHA:pyproject.toml" | sha256sum | cut -d' ' -f1); LH=$(git show "$TREE_SHA:uv.lock" | sha256sum | cut -d' ' -f1)
    CLUSTER_VENV="$(site_env_gate_venv "$CLUSTER_SHARED" "$MIRROR_REPOS_NAME" "$(printf '%s\n%s\n' "$PH" "$LH" | sha256sum | cut -c1-12)")"
  fi
  echo "-- cluster reads submit $(stamp)"
  JOB=$($REG cluster-reads submit --window "$WINDOW_ID" --venv "$CLUSTER_VENV" ${CLUSTER_LOCAL_VENV:+--local-venv} --op "$WINDOW_ID" --state "$STATE" 2>&1 | tail -1); src=${PIPESTATUS[0]}
  if [ "$JOB" = none ]; then
    # a plan with no cluster task (every adapter recorded refused at plan time) is not a failed submission: the refusal set names them all
    echo "cluster reads submit: no cluster task $(stamp)"
  elif ! [[ "$JOB" =~ ^[0-9]+$ ]] || [ "$src" != 0 ]; then
    # a FAILED submission (whatever the refusal set already holds): every cluster adapter is refused
    cluster_fail "submit FAILED ($JOB)"
  else
    echo "cluster reads array $JOB submitted; watching $(stamp)"
    "$WATCH_JOB" --slurm "$JOB" --host "$CLUSTER_HOST" --poll 30 --stall-min 0 --max-hours 1 2>&1 | tail -20
    echo "-- cluster reads revalidate $(stamp)"
    $REG cluster-reads revalidate --window "$WINDOW_ID" 2>&1 | tail -3
    rc=${PIPESTATUS[0]}
    # a revalidation that did not RUN (non-zero exit) leaves no verdict: nothing is delivered, every cluster adapter is refused
    [ "$rc" = 0 ] || { cluster_fail "revalidate FAILED rc=$rc (no verdict; nothing pulled)"; return; }
    echo "-- cluster reads pull $(stamp)"
    # the ownership re-checked before anything is pulled or merged: --op refuses by name when the reservation moved. Under
    # --op the transfer AND the inbox-merge run inside ONE critical-section interval and are RECORDED as this operation's `pull` and
    # `inbox_merge` stages (plan file + op record): the receipt below requires both (no receipt without a recorded merge of
    # this operation's pull), and a window that dies between here and its receipt is resumed by the next reconcile at the first missing stage
    $REG cluster-reads pull --window "$WINDOW_ID" --op "$WINDOW_ID" --state "$STATE" 2>&1 | tail -4
    rc=${PIPESTATUS[0]}; [ "$rc" = 0 ] || cluster_fail "pull FAILED rc=$rc"
    # the standing belt: whatever else the inbox holds (an earlier window's delivery) is merged too; this operation's own merge is the one above
    $REG inbox-merge 2>&1 | tail -1
  fi
  # the read receipt (C7 step 6): what this window read, per adapter -- the tick's changed set and S7 read it; a failed receipt leaves
  # the read standing (the next tick reads these adapters again, having no whole receipt for them)
  echo "-- cluster reads receipt $(stamp)"
  $REG cluster-reads receipt --window "$WINDOW_ID" --op "$WINDOW_ID" --state "$STATE" --dest "$CLUSTER_DEST" 2>&1 | tail -3
  rc=${PIPESTATUS[0]}; [ "$rc" = 0 ] || echo "receipt FAILED rc=$rc (the read stands; without a receipt the next tick reads these adapters again)"
  # the plan's and the pull's recorded refusals (never read on the grading host), with their reasons
  local a why
  for a in $($REG cluster-reads refusals --window "$WINDOW_ID" 2>/dev/null | tr '\n' ' '); do
    why=$($REG cluster-reads refusals --window "$WINDOW_ID" --json 2>/dev/null | "$PY" -c 'import json,sys; d=json.load(sys.stdin); print("; ".join(d.get(sys.argv[1], [])))' "$a" 2>/dev/null)
    refuse_adapter "$a" "${why:-recorded refused by the plan or the pull}"
  done
}
if [ -n "$CLUSTER_SET" ]; then
  cluster_stages
elif [ -n "$CLUSTER_ADAPTERS" ] || [ -n "$ADAPTERS" ]; then
  # cluster reads are configured but this window names no adapter with a cluster policy: one stamped line; every named adapter is refused below
  echo "-- cluster reads SKIPPED (no named adapter has a cluster policy) $(stamp)"
fi
# a named adapter outside the cluster set reads nowhere (C6): the operator-override reread is the one grading host run left
for a in $ADAPTERS; do
  in_set "$a" "$CLUSTER_SET" && continue
  if in_set "$a" "${REREAD_FORCE_ADAPTERS:-}" && [ -n "${REGISTRY_STORE_READ_OVERRIDE:-}" ]; then
    echo "-- adapter $a --reread --force [operator override: ${REGISTRY_STORE_READ_OVERRIDE}] $(stamp)"
    $REG adapters run --reread --force --no-ingest --retry-refused ${ADAPTER_EXTRA_FLAGS:-} --adapter "$a" 2>&1 | grep -v "^refused:" | tail -6
    rc=${PIPESTATUS[0]}; echo "adapter $a rc=$rc"
    [ "$rc" = 0 ] || [ "$rc" = 3 ] || fail "adapter pass FAILED ($a rc=$rc)" 10
    continue
  fi
  if in_set "$a" "${REREAD_FORCE_ADAPTERS:-}"; then
    refuse_adapter "$a" "reread --force reads the store on the grading host; set REGISTRY_STORE_READ_OVERRIDE=<reason> to run it as an operator override"
  else
    refuse_adapter "$a" "no cluster policy named for it this window (CLUSTER_ADAPTERS); the grading host reads no store"
  fi
done
# every recorded refusal, stamped (the time BEFORE the reason: windows.py reads the first HH:MM:SS of a stamp line)
for a in $REFUSED_SET; do
  echo "-- adapter $a REFUSED $(stamp) ($(why_of "$a"))"
done
REFUSED_ALL="$REFUSED_SET"

if [ -n "${ROSTER_LOAD_VERSION:-}" ]; then
  echo "-- roster load ${ROSTER_LOAD_VERSION} $(stamp)"; $REG roster load --version "${ROSTER_LOAD_VERSION}" --no-ingest 2>&1 | tail -8
  rc=${PIPESTATUS[0]}; [ "$rc" = 0 ] || fail "roster load FAILED rc=$rc" 11
fi
echo "--- ingest $(stamp)"; $REG ingest --retry-held 2>&1 | tail -4   # the window's uncapped cycle re-attempts every held revision
rc=${PIPESTATUS[0]}; [ "$rc" = 0 ] || fail "ingest FAILED rc=$rc" 12
INGESTED=1
echo "--- ingest end $(stamp)"

# ---- the verified fixed point (see the header): for every correction action named AND listed for apply, the verify reads the FOLDED
# database after the one ingest; a row it lists is reconciled ONCE more (from the now-folded row), folded by a second ingest and verified
# again, which must read zero -- else the window fails by name with the ids (exit 29). Each verify's whole text goes to the log (the ids are
# the evidence) and to $STATE/windows/<file>-<window>-<n>.txt (`scope-verify` for multi-turn-scope, `verify-<action>` for the others).
CORRECTION_ACTIONS="multi-turn-scope env-comments fit-effective-batch"   # the reconcile actions that have a --verify
verify_file() { case "$1" in multi-turn-scope) echo "$STATE/windows/scope-verify-$WINDOW_ID-$2.txt" ;; *) echo "$STATE/windows/verify-$1-$WINDOW_ID-$2.txt" ;; esac; }
reconcile_verify() {   # $1 = the action, $2 = the pass number; returns the CLI's rc: 0 consistent, 3 rows listed, anything else = the verify itself failed
  reconcile_args "$1"
  echo "-- reconcile $1 (verify $2) $(stamp)"
  $REG reconcile "$1" "${RARGS[@]}" --verify 2>&1 | tee "$(verify_file "$1" "$2")"
  return "${PIPESTATUS[0]}"
}
for rcn in ${WINDOW_RECONCILE:-}; do
  in_set "$rcn" "$CORRECTION_ACTIONS" && reconcile_applies "$rcn" || continue
  reconcile_verify "$rcn" 1; rc=$?
  if [ "$rc" = 3 ]; then
    echo "-- reconcile $rcn (apply, second pass) $(stamp)"
    reconcile_args "$rcn"
    $REG reconcile "$rcn" "${RARGS[@]}" --apply 2>&1 | tail -12
    rc=${PIPESTATUS[0]}; [ "$rc" = 0 ] || fail "reconcile $rcn second-pass apply FAILED rc=$rc (2 = a run held behind a record not folded yet; nothing written for it)" 17
    echo "-- reconcile $rcn (ingest) $(stamp)"; $REG ingest --retry-held 2>&1 | tail -4
    rc=${PIPESTATUS[0]}; [ "$rc" = 0 ] || fail "reconcile $rcn second ingest FAILED rc=$rc" 12
    reconcile_verify "$rcn" 2; rc=$?
    if [ "$rc" = 3 ]; then
      ids=$(awk '/^  / {print $1}' "$(verify_file "$rcn" 2)" | tr '\n' ' ')
      fail "reconcile $rcn reached no fixed point: after the apply, the ingest, a second apply and a second ingest the verify still lists inconsistent run(s): ${ids% }" 29
    fi
  fi
  [ "$rc" = 0 ] || fail "reconcile $rcn verify FAILED rc=$rc" 17
done
# a refused adapter ends the window ABORTED after the ingest of the rest (C6): never read on the grading host, named in the end marker
[ -z "$REFUSED_ALL" ] || fail "refused: $REFUSED_ALL" 20
# the end marker is the trap's: written after the verified restore
